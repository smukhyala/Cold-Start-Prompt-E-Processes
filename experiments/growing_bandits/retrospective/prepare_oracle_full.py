"""Freeze the full-coverage positive-control studies (pilot + control check) without launching any paid episode.

    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_oracle_full.py --stage pilot
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_oracle_full.py --stage study --dry-run
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_oracle_full.py --stage study
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_oracle_full.py --stage pilot --validate

Stages (both built from ``build_full_oracle``, the mechanically derived answer-key arm):

* ``pilot`` -> ``data/oracle_full_pilot/``: arm ``GL_oracle_full_pilot`` on the 30 block-A tasks (the bank minus the
  110 Leg G tasks), replicate 0 only = 30 episodes, tasks in a seeded order (seed 20261011). Gate: the pilot passes
  iff the 30-step-rescored success rate is >= 0.80 over the 30 episodes.
* ``study`` -> ``data/control_check_gitlab/``: arms ``GL_oracle_full`` (O2) and ``GL_anchor_baseline`` (D) on the 110
  Leg G tasks x replicates {0, 1} = 440 episodes, tasks in a seeded order (seed 20261012), the 4 episodes of a task
  in a seeded order (same scheme as ``prepare_winner_check.build_queue``). The anchors pool file is a byte copy of
  ``data/winner_check_gitlab/pools/anchors_gitlab.yaml`` so D's prompt sha is identical to Leg G's.

Each data dir holds ``gitlab/queue.jsonl``, ``pools/*.yaml`` and a ``manifest.json`` whose ``files`` map (relative
path -> sha256) is what ``collect.verify_queue`` checks and whose ``pool_files`` list tells the collector which pool
files to load (``collect.pool_files``). It is idempotent: re-running reproduces the same bytes, and a differing
existing queue / pool / manifest is never overwritten. ``--dry-run`` builds and validates in a temp dir and writes
nothing. No model is called, no server is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import build_full_oracle as bfo  # noqa: E402
import collect  # noqa: E402
import make_pools  # noqa: E402
from make_pools import QueueItem  # noqa: E402

WINNER_CHECK = ROOT / "data" / "winner_check_gitlab"
QUEUE_REL = "gitlab/queue.jsonl"
ANCHORS_REL = "pools/anchors_gitlab.yaml"
D_ARM = "GL_anchor_baseline"
D_POOL = "anchor"
COST_O2_USD = 0.15  # the long answer-key prompt
COST_D_USD = 0.10
BUDGET_MULTIPLIER = 1.3
BUDGET_ROUND_USD = 5
MAX_STEPS_SCORE = 30
WORKERS = 4

STAGES = {
    "pilot": {
        "out": ROOT / "data" / "oracle_full_pilot", "log_name": "oracle_full_pilot", "seed": 20261011,
        "pool": bfo.POOL, "arm": bfo.ARM, "replicates": (0,), "arms": ("O2",), "tasks": "block_a",
        "budget_override": 8,
        "study": "Oracle-full pilot -- does the mechanically derived answer key work on the 30 block-A GitLab tasks",
    },
    "study": {
        "out": ROOT / "data" / "control_check_gitlab", "log_name": "control_check_gitlab", "seed": 20261012,
        "pool": bfo.POOL, "arm": bfo.ARM, "replicates": (0, 1), "arms": ("O2", "D"), "tasks": "study",
        "budget_override": None,
        "study": "Control check -- full-coverage oracle (O2) vs the default prompt (D) on the 110 Leg G GitLab tasks",
    },
}
PILOT_GATE = ("pilot passes iff 30-step-rescored success (success and not timed_out and steps <= 30) >= 0.80 "
              "over the 30 episodes; old GL_anchor_oracle scored 0.867 and GL_anchor_baseline 0.733 on these tasks")
PILOT_GATE_THRESHOLD = 0.80
STUDY_RULE = ("PRIMARY: mean over the 110 tasks of O2_t - Dnew_t (arm score = mean of its 2 replicates, each "
              "30-step rescored), paired t, one-sided 95% lower bound; control_holds iff the bound > 0")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---- the queue -----------------------------------------------------------------------------


def build_queue(stage: str, task_ids: list[str]) -> list[QueueItem]:
    """The stage's queue: tasks in a seeded order; within a task the episodes (arm x replicate) in a seeded order."""
    cfg = STAGES[stage]
    rng = np.random.default_rng(cfg["seed"])
    arm_of = {"O2": (cfg["pool"], cfg["arm"]), "D": (D_POOL, D_ARM)}
    task_order = rng.permutation(sorted(task_ids))
    items = []
    for task in task_order:
        episodes = []
        for role in cfg["arms"]:
            for rep in cfg["replicates"]:
                episodes.append((arm_of[role][0], arm_of[role][1], rep))
        order = rng.permutation(len(episodes))
        for k in order:
            pool, arm, rep = episodes[int(k)]
            items.append((pool, arm, str(task), rep))
    queue = []
    for i in range(len(items)):
        pool, arm, task, rep = items[i]
        queue.append(QueueItem(index=i, pool=pool, arm_id=arm, task_id=task, replicate=rep, pilot=False))
    return queue


def queue_text(queue: list[QueueItem]) -> str:
    text = ""
    for item in queue:
        text += json.dumps(asdict(item)) + "\n"  # same writer as make_pools.write_queue
    return text


def recommended_budget(queue: list[QueueItem], override: int | None = None) -> tuple[float, int]:
    """Estimate = $0.15 per O2 episode + $0.10 per D episode; budget = 1.3 x estimate rounded up to $5
    (unless the stage names its own)."""
    estimate = 0.0
    for item in queue:
        if item.arm_id == bfo.ARM:
            estimate += COST_O2_USD
        else:
            estimate += COST_D_USD
    if override is not None:
        return estimate, int(override)
    capped = BUDGET_MULTIPLIER * estimate
    return estimate, int(math.ceil(round(capped, 9) / BUDGET_ROUND_USD) * BUDGET_ROUND_USD)


# ---- the plan --------------------------------------------------------------------------------


def build_plan(stage: str, built: dict | None = None, anchors_source: Path | None = None) -> dict:
    """Everything the data dir will hold, computed without touching it."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGES)}")
    cfg = STAGES[stage]
    if built is None:
        built = bfo.build_all()
    task_ids = list(built[cfg["tasks"]])
    queue = build_queue(stage, task_ids)
    task_order = []
    for item in queue:
        if item.task_id not in task_order:
            task_order.append(item.task_id)
    estimate, budget = recommended_budget(queue, cfg["budget_override"])
    pool_texts = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = bfo.write_pools(built, Path(tmp))
        pool_texts[f"pools/{cfg['pool']}.yaml"] = paths[cfg["pool"]].read_bytes()
    pool_files = [f"pools/{cfg['pool']}.yaml"]
    anchors_bytes = None
    if "D" in cfg["arms"]:
        if anchors_source is None:
            anchors_source = WINNER_CHECK / ANCHORS_REL
        anchors_bytes = Path(anchors_source).read_bytes()
        pool_texts[ANCHORS_REL] = anchors_bytes
        pool_files.append(ANCHORS_REL)
    return {"stage": stage, "cfg": cfg, "built": built, "task_ids": task_ids, "task_order": task_order,
            "queue": queue, "queue_text": queue_text(queue), "pool_texts": pool_texts, "pool_files": pool_files,
            "estimate": estimate, "budget": budget}


def build_manifest(plan: dict, files: dict[str, str]) -> dict:
    cfg = plan["cfg"]
    stage = plan["stage"]
    info = plan["built"]["pools"][cfg["pool"]]
    arms = {"O2": {"arm_id": cfg["arm"], "pool": cfg["pool"]}}
    if "D" in cfg["arms"]:
        arms["D"] = {"arm_id": D_ARM, "pool": D_POOL}
    if stage == "pilot":
        design = (f"{cfg['arm']} (the SAME 140-task answer-key prompt the study uses) x the 30 block-A GitLab tasks "
                  "x replicate 0 only; seeded task order; isolated resets")
        decision = PILOT_GATE
    else:
        design = ("O2 (GL_oracle_full, full-coverage answer key) and D (GL_anchor_baseline) x the 110 Leg G GitLab "
                  "tasks x replicates {0, 1}; seeded task order and seeded within-task episode order; isolated resets")
        decision = STUDY_RULE
    manifest = {
        "study": cfg["study"],
        "design": design,
        "stage": stage,
        "seed": cfg["seed"],
        "decision_rule": decision,
        "arms": arms,
        "pool_files": plan["pool_files"],
        "n_tasks": len(plan["task_order"]),
        "task_ids": plan["task_order"],
        "n_episodes": len(plan["queue"]),
        "replicates": list(cfg["replicates"]),
        "max_steps": collect.MAX_STEPS,
        "timeout_s": collect.profile_spec("gitlab")["agent"]["timeout_s"],
        "cost_per_episode_assumed_usd": {"O2": COST_O2_USD, "D": COST_D_USD},
        "cost_estimate_usd": round(plan["estimate"], 2),
        "recommended_budget_usd": plan["budget"],
        "budget_rule": (f"fixed by the coordinator: ${cfg['budget_override']}" if cfg["budget_override"] is not None
                        else f"{BUDGET_MULTIPLIER} x estimate rounded up to ${BUDGET_ROUND_USD}"),
        "oracle_prompt": {"entries": len(info["entries"]), "chars": info["chars"],
                          "approx_tokens_chars_div_4": info["approx_tokens"]},
        "real_tasks_sha256": sha(bfo.bank_path()),
        "generator_sha256": {"build_full_oracle.py": sha(Path(bfo.__file__)),
                             "prepare_oracle_full.py": sha(Path(__file__))},
        "files": files,
        "provenance": {"leg_g_manifest_sha256": sha(WINNER_CHECK / "manifest.json")},
    }
    if stage == "pilot":
        manifest["gate"] = {"threshold": PILOT_GATE_THRESHOLD, "statement": PILOT_GATE,
                            "old_GL_anchor_oracle_rate": 0.867, "old_GL_anchor_baseline_rate": 0.733}
    else:
        manifest["d_arm_prompt_sha256_matches_leg_g"] = True
    return manifest


def prepare(stage: str, out: Path | None = None, built: dict | None = None,
            anchors_source: Path | None = None) -> dict:
    """Write the stage's data dir (idempotent; refuses to overwrite anything that differs)."""
    cfg = STAGES[stage]
    if out is None:
        out = cfg["out"]
    out = Path(out)
    plan = build_plan(stage, built, anchors_source)
    if "D" in cfg["arms"]:
        check_d_arm_matches_leg_g(plan["pool_texts"][ANCHORS_REL])
    files = {QUEUE_REL: text_sha(plan["queue_text"])}
    for rel in plan["pool_files"]:
        files[rel] = hashlib.sha256(plan["pool_texts"][rel]).hexdigest()
    manifest = build_manifest(plan, files)
    manifest_text = json.dumps(manifest, indent=2) + "\n"

    queue_path = out / QUEUE_REL
    manifest_path = out / "manifest.json"
    if queue_path.exists() and queue_path.read_text() != plan["queue_text"]:
        raise ValueError("existing trial queue differs; refusing to overwrite")
    for rel in plan["pool_files"]:
        dest = out / rel
        if dest.exists() and dest.read_bytes() != plan["pool_texts"][rel]:
            raise ValueError(f"existing pool file differs: {rel}; refusing to overwrite")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise ValueError("existing trial manifest differs; refusing to overwrite")
    (out / "pools").mkdir(parents=True, exist_ok=True)
    (out / "gitlab").mkdir(parents=True, exist_ok=True)
    for rel in plan["pool_files"]:
        (out / rel).write_bytes(plan["pool_texts"][rel])
    queue_path.write_text(plan["queue_text"])
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text)
    return {"plan": plan, "manifest": manifest}


def check_d_arm_matches_leg_g(anchors_bytes: bytes) -> None:
    """D must be Leg G's default prompt: the anchors file is byte-identical to its frozen sha and carries D's sha."""
    leg_g = json.loads((WINNER_CHECK / "manifest.json").read_text())
    expected = leg_g["files"][ANCHORS_REL]
    got = hashlib.sha256(anchors_bytes).hexdigest()
    if got != expected:
        raise ValueError(f"anchors pool sha256 {got} != Leg G's frozen {expected}; D would not be the same prompt")


# ---- validation with the collector's own checks (no model call, no server) --------------------------------


class _BankOnlyAdapter:
    """Stands in for the adapter in ``collect.check_bank``: only the bank's ids, no server, no model."""

    def task_ids(self):
        ids = []
        for task in bfo.load_bank():
            ids.append(str(task["id"]))
        return ids


def validate(data: Path, log_name: str = "validate") -> list[str]:
    """The collector's own start-up checks, in its order, against `data`; returns the log lines."""
    data = Path(data)
    lines = []
    manifest = json.loads((data / "manifest.json").read_text())
    collect.verify_queue(data, "gitlab")
    lines.append(f"verify_queue: queue + {len(manifest['pool_files'])} pool file(s) byte-identical to manifest.files")
    prompts = collect.load_prompts(data, "gitlab")
    lines.append(f"load_prompts: {len(prompts)} arms loaded, each rendered prompt sha256-checked by load_pool")
    queue = make_pools.read_queue(data / collect.profile_spec("gitlab")["queue"])
    collect.check_queue_arms(queue, prompts)
    lines.append(f"check_queue_arms: all {len(queue)} items name a loaded arm of the right pool")
    if len(queue) != manifest["n_episodes"]:
        raise RuntimeError(f"queue has {len(queue)} items; manifest says {manifest['n_episodes']}")
    collect.check_bank(_BankOnlyAdapter(), queue)
    lines.append("check_bank: every queued task id is in the GitLab bank")
    with tempfile.TemporaryDirectory() as tmp:
        collect.check_existing_records(Path(tmp), prompts)
        shares = collect.partition_remaining(queue, Path(tmp), WORKERS, False)
        sizes = []
        for share in shares:
            sizes.append(len(share))
        lines.append(f"check_existing_records: empty log dir ok; partition_remaining over {WORKERS} workers: {sizes}")
    args = collect.parse_args(["--profile", "gitlab", "--data", str(data), "--log-dir",
                               str(ROOT / "logs" / log_name), "--budget", "1", "--workers", str(WORKERS)])
    lines.append(f"parse_args: profile={args.profile} data={args.data} workers={args.workers}")
    cfg = collect.WorkerConfig(worker=0, n_workers=1, log_dir=Path("x"), budget_usd=1.0)
    lines.append(f"step cap: collect.MAX_STEPS={collect.MAX_STEPS}, WorkerConfig.max_steps={cfg.max_steps}")
    return lines


def launch_command(stage: str, budget: int) -> str:
    cfg = STAGES[stage]
    rel = Path(cfg["out"]).relative_to(ROOT)
    return ("./.venv/bin/python experiments/growing_bandits/empirical/collect.py --profile gitlab "
            f"--data {rel} --log-dir logs/{cfg['log_name']} --budget {budget} --workers {WORKERS}")


def summary(result: dict) -> str:
    m = result["manifest"]
    return (f"{m['stage']}: {m['n_episodes']} episodes on {m['n_tasks']} tasks; arms {m['arms']}; "
            f"prompt {m['oracle_prompt']['chars']} chars (~{m['oracle_prompt']['approx_tokens_chars_div_4']} tokens); "
            f"estimate ${m['cost_estimate_usd']}; recommended --budget {m['recommended_budget_usd']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: the stage's data dir")
    ap.add_argument("--dry-run", action="store_true", help="build + validate in a temp dir, print, write nothing")
    ap.add_argument("--validate", action="store_true", help="run the collector's own checks on the existing data dir")
    args = ap.parse_args(argv)
    cfg = STAGES[args.stage]
    out = args.out if args.out is not None else cfg["out"]
    if args.validate:
        for line in validate(out, cfg["log_name"]):
            print(line)
        return 0
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix=f"oracle_full_{args.stage}_dry_"))
        try:
            result = prepare(args.stage, tmp / "data")
            for line in validate(tmp / "data", cfg["log_name"]):
                print(line)
            print("DRY RUN (nothing written to the repo): " + summary(result))
            print(launch_command(args.stage, result["manifest"]["recommended_budget_usd"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return 0
    result = prepare(args.stage, out)
    for line in validate(out, cfg["log_name"]):
        print(line)
    print("Prepared " + summary(result) + f" under {out}")
    print(launch_command(args.stage, result["manifest"]["recommended_budget_usd"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
