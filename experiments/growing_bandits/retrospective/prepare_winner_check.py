"""Freeze Pre-registration 13 (winner check on fresh GitLab tasks) without launching any paid episode.

    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_winner_check.py            # strict rule
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_winner_check.py --eligibility het_unrun
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_winner_check.py --report    # counts only
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_winner_check.py --validate  # collector checks

Design (fixed by the PI): three arms -- W (the GLK arm with the best GLK30 block-A success rate in the OLD
data), D (``GL_anchor_baseline``) and O (``GL_anchor_oracle``, the planted positive control) -- on every
eligible GitLab task x 2 replicates, one worker, tasks in a seeded random order and the 6 episodes of a task
in a seeded random order. Writes ``data/winner_check_gitlab/`` laid out like ``data/heterogeneity/`` so that

    collect.py --profile gitlab --data data/winner_check_gitlab --log-dir logs/winner_check_gitlab --budget B --workers 1

works unchanged: ``gitlab/queue.jsonl``, ``pools/{GLG,GLK,anchors_gitlab}.yaml`` (byte copies) and a
``manifest.json`` whose ``files`` map (relative path -> sha256 of the bytes) is exactly what
``collect.verify_queue`` checks, plus the design fields.

ELIGIBILITY.  ``strict`` (the registered rule) = tasks never run by ANY prompt in ANY prior study. The
June GitLab strong-arm allocation runs cycled through all 140 bank tasks, so ``strict`` is EMPTY on this
repo and the script refuses. ``het_unrun`` (a DEVIATION that needs the PI's sign-off) = tasks never run in
the heterogeneity study's GitLab logs (``logs/heterogeneity/gitlab``): the bank minus the 30 block-A tasks.

Idempotent: re-running reproduces the same bytes; an existing data dir whose manifest (or any file) differs
is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
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
HET = ROOT / "data" / "heterogeneity"
OUT = ROOT / "data" / "winner_check_gitlab"
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import collect  # noqa: E402
import make_pools  # noqa: E402
from make_pools import QueueItem  # noqa: E402

STUDY_NAME = "Pre-registration 13 -- winner check on fresh GitLab tasks"
SEED = 20261010
REPLICATES = (0, 1)
APP = "gitlab-plan-and-track"
D_ARM = "GL_anchor_baseline"
O_ARM = "GL_anchor_oracle"
D_POOL = "anchor"
O_POOL = "anchor"
W_POOL = "GLK"
POOL_FILES = ("pools/GLG.yaml", "pools/GLK.yaml", "pools/anchors_gitlab.yaml")
QUEUE_REL = "gitlab/queue.jsonl"
EPISODE_COST_USD = 0.139  # historical GLK cost per episode (GitLab pilot); the 30-step cap should lower it
BUDGET_MULTIPLIER = 1.3
BUDGET_ROUND_USD = 5
MIN_TASKS = 20  # strict mode refuses below this many eligible tasks
ELIGIBILITIES = ("strict", "het_unrun")
SELECTION_RULE = (
    "W = the GLK arm with the highest primary GLK30 success rate (30-step rescore of every ok record: "
    "success and not timed_out and steps <= 30) over its block-A tasks, replicate 0, from the OLD "
    "heterogeneity data. Ties: higher uncapped GLK rate, then lower mean steps, then arm_id ascending."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the task bank -------------------------------------------------------------------------


def webarena_root() -> Path:
    from cold_start.tasks.webarena import _webarena_root

    return Path(_webarena_root())


def bank_path() -> Path:
    return webarena_root() / "apps" / APP / "real-tasks.json"


def load_bank() -> list[dict]:
    return json.loads(bank_path().read_text(encoding="utf-8"))


def bank_ids() -> list[str]:
    ids = []
    for task in load_bank():
        ids.append(str(task["id"]))
    return ids


def bank_instructions() -> dict[str, str]:
    out = {}
    for task in load_bank():
        out[str(task["id"])] = str(task["instruction"])
    return out


# ---- the ever-run set ------------------------------------------------------------------------

STRONG_DIR = ROOT / "results" / "gitlab_strong_arm"
HET_LOG_DIR = ROOT / "logs" / "heterogeneity" / "gitlab"
SKIP_SCAN_PARTS = ("winner_check_gitlab", "/artifacts/", "/.venv/")


def _json_lines(path: Path):
    """Parsed dict lines of a jsonl file, tolerating a torn final line and blank lines."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if hasattr(row, "get"):
                yield row


def _add(sets: dict[str, set], source: str, task_id) -> None:
    if source not in sets:
        sets[source] = set()
    sets[source].add(str(task_id))


def ever_run_by_source(bank: dict[str, str] | None = None) -> tuple[dict[str, set], list[str]]:
    """``({source: task ids ever run}, warnings)`` over every place a GitLab episode can have left a trace.

    A strong-arm record counts only if its ``task_meta.web_app_dir`` is the GitLab app AND its recorded
    instruction equals the current bank's text for that id (the June study names tasks task_e1... like
    every other app, so the id alone is not evidence). Every record counts whatever its status: an
    ``infra_error`` attempt still ran the prompt on the task.
    """
    if bank is None:
        bank = bank_instructions()
    sets: dict[str, set] = {}
    warnings: list[str] = []

    # 1. heterogeneity worker logs (+ any archive) and the per-attempt artifact dirs
    for path in sorted(HET_LOG_DIR.rglob("*.jsonl")):
        if "/artifacts/" in str(path):
            continue
        for row in _json_lines(path):
            if "task_id" in row:
                _add(sets, "het_gitlab_logs", row["task_id"])
    artifacts = HET_LOG_DIR / "artifacts"
    if artifacts.is_dir():
        for worker_dir in sorted(artifacts.iterdir()):
            for arm_dir in sorted(worker_dir.iterdir()) if worker_dir.is_dir() else []:
                for task_dir in sorted(arm_dir.iterdir()) if arm_dir.is_dir() else []:
                    _add(sets, "het_gitlab_artifacts", task_dir.name)

    # 2. queue items actually run = queue items with a record (cross-check, not an extra source)
    queue_keys = set()
    for item in make_pools.read_queue(HET / QUEUE_REL):
        queue_keys.add((item.arm_id, item.task_id, item.replicate))
    for path in sorted(HET_LOG_DIR.glob("worker_*.jsonl")):
        for row in _json_lines(path):
            key = (row.get("arm_id"), row.get("task_id"), row.get("replicate"))
            if key not in queue_keys:
                warnings.append(f"het record outside the queue: {key}")
            else:
                _add(sets, "het_queue_items_run", key[1])

    # 3. the June strong-arm study: jsonl records (app + instruction verified) and csv tables
    for path in sorted(STRONG_DIR.rglob("*.jsonl")):
        for row in _json_lines(path):
            if "task_id" not in row:
                continue
            meta = row.get("task_meta") or {}
            raw = meta.get("raw") or {}
            if APP not in str(meta.get("web_app_dir", "")):
                warnings.append(f"{path.name}: record for {row['task_id']} is not the {APP} app")
                continue
            if raw.get("instruction") != bank.get(str(row["task_id"])):
                warnings.append(f"{path.name}: instruction of {row['task_id']} differs from the bank")
                continue
            _add(sets, "strong_arm_jsonl", row["task_id"])
    for path in sorted(STRONG_DIR.rglob("*.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            names = reader.fieldnames or []
            col = "task" if "task" in names else ("task_id" if "task_id" in names else None)
            if col is None:
                continue
            for row in reader:
                if row[col]:
                    _add(sets, "strong_arm_csv", row[col])

    # 4. anything else under logs/ and results/ that names the GitLab app (never the new study's own dirs)
    handled = (str(HET_LOG_DIR), str(STRONG_DIR))
    for top in (ROOT / "logs", ROOT / "results"):
        for path in sorted(top.rglob("*.jsonl")):
            text = str(path)
            if text.startswith(handled) or any(part in text for part in SKIP_SCAN_PARTS):
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if APP not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if hasattr(row, "get") and "task_id" in row:
                        _add(sets, "other_logs_naming_gitlab_app", row["task_id"])
    return sets, warnings


def ever_run_union(sets: dict[str, set], sources: tuple[str, ...] | None = None) -> set:
    out: set = set()
    for name, ids in sets.items():
        if sources is None or name in sources:
            out |= ids
    return out


HET_SOURCES = ("het_gitlab_logs", "het_gitlab_artifacts", "het_queue_items_run")


def eligible_tasks(eligibility: str, bank: list[str], sets: dict[str, set]) -> list[str]:
    if eligibility == "strict":
        ever = ever_run_union(sets)
    elif eligibility == "het_unrun":
        ever = ever_run_union(sets, HET_SOURCES)
    else:
        raise ValueError(f"unknown eligibility {eligibility!r}; expected one of {ELIGIBILITIES}")
    out = []
    for task in sorted(bank):
        if task not in ever:
            out.append(task)
    return out


# ---- selecting W from the old data ---------------------------------------------------------


def select_winner() -> dict:
    """The selection rule on the old data; the study loader and the exported matrices must agree."""
    import pandas as pd

    from experiments.growing_bandits.empirical import study as st

    manifest = json.loads((HET / "manifest.json").read_text())
    block_a = set(manifest["gitlab_block_a"])
    capped = st.load_study_outcomes(st.STUDIES["glk30"])
    raw = st.load_study_outcomes(st.STUDIES["glk"])

    def primary(frame, pool):
        sub = frame[(frame["pool"] == pool) & (frame["replicate"] == 0) & (frame["status"] == "ok")]
        return sub[sub["task_id"].isin(block_a)]

    cap_glk = primary(capped, "GLK")
    raw_glk = primary(raw, "GLK")
    rate30 = cap_glk.groupby("arm_id")["success"].apply(lambda s: float(s.astype(float).mean()))
    rate = raw_glk.groupby("arm_id")["success"].apply(lambda s: float(s.astype(float).mean()))
    steps = raw_glk.groupby("arm_id")["steps"].apply(lambda s: float(s.astype(float).mean()))

    # agreement with the exported audit matrices
    agree = {}
    for label, loaded in (("GLK30", rate30), ("GLK", rate)):
        mat = pd.read_csv(ROOT / "results" / "audit" / "matrices" / f"{label}.csv")
        mat = mat[(mat["replicate"] == 0) & (mat["task_id"].isin(block_a))]
        mrate = mat.groupby("arm_id")["success"].mean()
        worst = 0.0
        for arm in loaded.index:
            worst = max(worst, abs(float(mrate[arm]) - float(loaded[arm])))
        agree[label] = worst
        if worst > 1e-12:
            raise ValueError(f"{label}: study loader and exported matrix disagree (max gap {worst})")

    table = []
    for arm in rate30.index:
        table.append({"arm_id": str(arm), "glk30_rate": float(rate30[arm]), "glk_rate": float(rate[arm]),
                      "mean_steps": float(steps[arm]), "n_tasks": int(len(cap_glk[cap_glk["arm_id"] == arm]))})
    table.sort(key=lambda r: (-r["glk30_rate"], -r["glk_rate"], r["mean_steps"], r["arm_id"]))

    anchors = {}
    for label, frame in (("glk30", capped), ("glk", raw)):
        sub = primary(frame, "anchor")
        by_arm = sub.groupby("arm_id")["success"].apply(lambda s: float(s.astype(float).mean()))
        anchors[label] = {str(a): float(by_arm[a]) for a in by_arm.index}
    winner = table[0]
    tied = []
    for row in table:
        if abs(row["glk30_rate"] - winner["glk30_rate"]) < 1e-12:
            tied.append(row["arm_id"])
    return {"winner": winner["arm_id"], "top5": table[:5], "n_glk_arms": len(table),
            "n_block_a_tasks": len(block_a), "tied_at_top_glk30": tied, "loader_vs_matrix_max_gap": agree,
            "anchor_block_a_rates": anchors}


# ---- the queue -----------------------------------------------------------------------------


def arm_pools(winner: str) -> dict[str, str]:
    return {winner: W_POOL, D_ARM: D_POOL, O_ARM: O_POOL}


def build_queue(tasks: list[str], winner: str, seed: int = SEED) -> list[QueueItem]:
    rng = np.random.default_rng(seed)
    pools = arm_pools(winner)
    task_order = rng.permutation(sorted(tasks))
    items = []
    for task in task_order:
        episodes = []
        for arm in (winner, D_ARM, O_ARM):
            for rep in REPLICATES:
                episodes.append((arm, rep))
        order = rng.permutation(len(episodes))
        for k in order:
            arm, rep = episodes[int(k)]
            items.append((pools[arm], arm, str(task), rep))
    queue = []
    for i in range(len(items)):
        pool, arm, task, rep = items[i]
        queue.append(QueueItem(index=i, pool=pool, arm_id=arm, task_id=task, replicate=rep, pilot=False))
    return queue


def recommended_budget(n_episodes: int) -> tuple[float, int]:
    estimate = n_episodes * EPISODE_COST_USD
    capped = BUDGET_MULTIPLIER * estimate
    return estimate, int(math.ceil(capped / BUDGET_ROUND_USD) * BUDGET_ROUND_USD)


# ---- preparation ---------------------------------------------------------------------------


def build_plan(eligibility: str) -> dict:
    """Everything the data dir will hold, computed without touching it."""
    bank = bank_ids()
    sets, warnings = ever_run_by_source()
    strict = eligible_tasks("strict", bank, sets)
    eligible = eligible_tasks(eligibility, bank, sets)
    if eligibility == "strict" and len(eligible) < MIN_TASKS:
        raise ValueError(
            f"strict eligibility leaves {len(eligible)} of {len(bank)} bank tasks never run by any prompt in "
            f"any prior study (< {MIN_TASKS}); the registered design cannot be built. Per-source ever-run "
            f"counts: { {k: len(v) for k, v in sets.items()} }. Re-run with --eligibility het_unrun only with "
            "the PI's sign-off.")
    selection = select_winner()
    winner = selection["winner"]
    queue = build_queue(eligible, winner)
    task_order = []
    for item in queue:
        if item.task_id not in task_order:
            task_order.append(item.task_id)
    estimate, budget = recommended_budget(len(queue))
    queue_text = ""
    for item in queue:
        queue_text += json.dumps(asdict(item)) + "\n"  # same writer as make_pools.write_queue
    return {"bank": bank, "sets": sets, "warnings": warnings, "strict": strict, "eligible": eligible,
            "selection": selection, "queue": queue, "queue_text": queue_text, "task_order": task_order,
            "estimate": estimate, "budget": budget, "eligibility": eligibility}


def build_manifest(plan: dict, files: dict[str, str]) -> dict:
    sel = plan["selection"]
    winner = sel["winner"]
    sets = plan["sets"]
    rates = sel["anchor_block_a_rates"]
    return {
        "study": STUDY_NAME,
        "design": "W (old-data GLK winner), D (GL_anchor_baseline), O (GL_anchor_oracle, planted positive "
                  "control) x every eligible GitLab task x 2 replicates; one worker; seeded task order and "
                  "seeded within-task episode order; isolated resets",
        "seed": SEED,
        "selection_rule": SELECTION_RULE,
        "selected_arm": winner,
        "arms": {"W": {"arm_id": winner, "pool": W_POOL}, "D": {"arm_id": D_ARM, "pool": D_POOL},
                 "O": {"arm_id": O_ARM, "pool": O_POOL}},
        "top5_table": sel["top5"],
        "tied_at_top_glk30": sel["tied_at_top_glk30"],
        "old_data_block_a": {
            "n_block_a_tasks": sel["n_block_a_tasks"],
            "W_glk30_rate": sel["top5"][0]["glk30_rate"],
            "W_glk_rate": sel["top5"][0]["glk_rate"],
            "D_glk30_rate": rates["glk30"].get(D_ARM), "D_glk_rate": rates["glk"].get(D_ARM),
            "O_glk30_rate": rates["glk30"].get(O_ARM), "O_glk_rate": rates["glk"].get(O_ARM),
        },
        "eligibility": plan["eligibility"],
        "eligibility_note": (
            "registered rule: never run by ANY prompt in ANY prior study" if plan["eligibility"] == "strict" else
            "DEVIATION (needs PI sign-off): never run in the heterogeneity study's GitLab logs; the strict "
            f"registered rule leaves {len(plan['strict'])} tasks because the June strong-arm runs covered the "
            "whole bank"),
        "bank_size": len(plan["bank"]),
        "ever_run_count_by_source": {k: len(v) for k, v in sorted(sets.items())},
        "ever_run_count_union": len(ever_run_union(sets)),
        "ever_run_count_het_only": len(ever_run_union(sets, HET_SOURCES)),
        "never_run_count_strict": len(plan["strict"]),
        "never_run_count_het_only": len(eligible_tasks("het_unrun", plan["bank"], sets)),
        "n_tasks": len(plan["task_order"]),
        "task_ids": plan["task_order"],
        "n_episodes": len(plan["queue"]),
        "replicates": list(REPLICATES),
        "max_steps": collect.MAX_STEPS,
        "timeout_s": collect.profile_spec("gitlab")["agent"]["timeout_s"],
        "cost_per_episode_assumed_usd": EPISODE_COST_USD,
        "cost_estimate_usd": round(plan["estimate"], 2),
        "recommended_budget_usd": plan["budget"],
        "budget_rule": f"{BUDGET_MULTIPLIER} x estimate rounded up to ${BUDGET_ROUND_USD}",
        "real_tasks_sha256": sha(bank_path()),
        "files": files,
        "provenance": {"source_manifest_sha256": sha(HET / "manifest.json")},
    }


def _comparable(manifest: dict) -> dict:
    out = dict(manifest)
    out.pop("provenance", None)
    return out


def prepare(eligibility: str, out: Path = OUT) -> dict:
    plan = build_plan(eligibility)
    het_manifest = json.loads((HET / "manifest.json").read_text())
    files = {}
    for rel in POOL_FILES:
        source = HET / rel
        if sha(source) != het_manifest["files"][rel]:
            raise ValueError(f"original frozen pool changed: {rel}")
        files[rel] = sha(source)
    files[QUEUE_REL] = hashlib.sha256(plan["queue_text"].encode()).hexdigest()
    manifest = build_manifest(plan, files)
    manifest_text = json.dumps(manifest, indent=2) + "\n"

    queue_path = out / QUEUE_REL
    manifest_path = out / "manifest.json"
    if queue_path.exists() and queue_path.read_text() != plan["queue_text"]:
        raise ValueError("existing trial queue differs; refusing to overwrite")
    for rel in POOL_FILES:
        dest = out / rel
        if dest.exists() and sha(dest) != files[rel]:
            raise ValueError(f"prepared pool differs from source: {rel}")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if _comparable(old) != _comparable(manifest):
            raise ValueError("existing trial manifest differs; refusing to overwrite")
    (out / "pools").mkdir(parents=True, exist_ok=True)
    (out / "gitlab").mkdir(parents=True, exist_ok=True)
    for rel in POOL_FILES:
        shutil.copy2(HET / rel, out / rel)
    queue_path.write_text(plan["queue_text"])
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text)
    return {"plan": plan, "manifest": manifest}


# ---- validation with the collector's own checks (no model call, no server) ----------------------


class _BankOnlyAdapter:
    """Stands in for the adapter in ``collect.check_bank``: only the bank's ids, no server, no model."""

    def task_ids(self):
        return bank_ids()


def validate(data: Path = OUT) -> list[str]:
    """The collector's own start-up checks, in its order, against `data`; returns the log lines."""
    lines = []
    collect.verify_queue(data, "gitlab")
    lines.append("verify_queue: queue + 3 pool files byte-identical to manifest.files")
    prompts = collect.load_prompts(data, "gitlab")
    lines.append(f"load_prompts: {len(prompts)} arms loaded, each rendered prompt sha256-checked by load_pool")
    queue = make_pools.read_queue(data / collect.profile_spec("gitlab")["queue"])
    collect.check_queue_arms(queue, prompts)
    lines.append(f"check_queue_arms: all {len(queue)} items name a loaded arm of the right pool")
    collect.check_bank(_BankOnlyAdapter(), queue)
    lines.append("check_bank: every queued task id is in the GitLab bank "
                 f"({len(bank_ids())} tasks; profile expects {collect.profile_spec('gitlab')['n_bank_tasks']})")
    with tempfile.TemporaryDirectory() as tmp:
        collect.check_existing_records(Path(tmp), prompts)
        shares = collect.partition_remaining(queue, Path(tmp), 1, False)
        lines.append(f"check_existing_records: empty log dir ok; partition_remaining: 1 worker owns "
                     f"{len(shares[0])} items, first indices {list(shares[0][:3])}")
    args = collect.parse_args(["--profile", "gitlab", "--data", str(data), "--log-dir",
                               str(ROOT / "logs" / "winner_check_gitlab"), "--budget", "1", "--workers", "1"])
    lines.append(f"parse_args: profile={args.profile} data={args.data} log_dir={args.log_dir} workers={args.workers}")
    cfg = collect.WorkerConfig(worker=0, n_workers=1, log_dir=Path("x"), budget_usd=1.0)
    lines.append(f"step cap: collect.MAX_STEPS={collect.MAX_STEPS}, WorkerConfig.max_steps={cfg.max_steps} "
                 "(passed as run_arm(..., cfg.max_steps) -> adapter.max_steps -> agent.run(max_steps=...))")
    return lines


def launch_command(budget: int) -> str:
    return ("./.venv/bin/python experiments/growing_bandits/empirical/collect.py --profile gitlab "
            f"--data data/winner_check_gitlab --log-dir logs/winner_check_gitlab --budget {budget} --workers 1")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eligibility", choices=ELIGIBILITIES, default="strict")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", action="store_true", help="print bank / ever-run / never-run counts, write nothing")
    ap.add_argument("--validate", action="store_true", help="run the collector's own checks on --out")
    args = ap.parse_args(argv)
    if args.report:
        bank = bank_ids()
        sets, warnings = ever_run_by_source()
        print(json.dumps({"bank_size": len(bank),
                          "ever_run_by_source": {k: len(v) for k, v in sorted(sets.items())},
                          "ever_run_union": len(ever_run_union(sets)),
                          "never_run_strict": len(eligible_tasks("strict", bank, sets)),
                          "ever_run_het_only": len(ever_run_union(sets, HET_SOURCES)),
                          "never_run_het_only": len(eligible_tasks("het_unrun", bank, sets)),
                          "warnings": warnings[:10], "n_warnings": len(warnings)}, indent=2))
        return
    if args.validate:
        for line in validate(args.out):
            print(line)
        return
    result = prepare(args.eligibility, args.out)
    m = result["manifest"]
    print(f"Prepared {m['n_episodes']} episodes on {m['n_tasks']} tasks under {args.out} "
          f"(eligibility {m['eligibility']}); W={m['selected_arm']}; estimate ${m['cost_estimate_usd']}; "
          f"recommended --budget {m['recommended_budget_usd']}")
    print(launch_command(m["recommended_budget_usd"]))


if __name__ == "__main__":
    main()
