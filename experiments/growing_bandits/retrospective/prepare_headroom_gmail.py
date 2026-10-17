"""Freeze the Gmail headroom study (Pre-registration 15) without launching any paid episode.

    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_headroom_gmail.py --dry-run
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_headroom_gmail.py

Design: arms ``GM_oracle_full`` (O, the mechanically derived 60-task answer key, ``build_gmail_oracle``) and
``GM_anchor_baseline`` (D, the Gmail default prompt, sha256 ae8c7c7f..., the same prompt as Pre-reg 13 Leg M's
default) x the 60 Gmail real tasks x replicates {0, 1} = 240 episodes, tasks in a seeded order (seed 20261013), the 4
episodes of a task in a seeded order (same scheme as ``prepare_oracle_full.build_queue``). Profile ``gmail``
(gpt-5.4-mini, low effort, 180 s clock), 30-step cap.

The data dir ``data/headroom_gmail/`` holds ``gmail/queue.jsonl``, ``pools/*.yaml`` and a ``manifest.json`` whose
``files`` map is what ``collect.verify_queue`` checks and whose ``pool_files`` list tells the collector which pool
files to load. The answer-key pool is rebuilt from ``build_gmail_oracle`` and must be byte-identical to
``data/headroom/pools/gmoraclefull.yaml``. Idempotent; never overwrites a differing file. No model is called, no
server is started.
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

import build_gmail_oracle as bgo  # noqa: E402
import collect  # noqa: E402
import make_pools  # noqa: E402
from make_pools import QueueItem  # noqa: E402

PROFILE = "gmail"
OUT = ROOT / "data" / "headroom_gmail"
LOG_NAME = "headroom_gmail"
SEED = 20261013
REPLICATES = (0, 1)
O_POOL, O_ARM = "gmoraclefull", "GM_oracle_full"
D_POOL, D_ARM = "anchor", "GM_anchor_baseline"
O_REL = "pools/gmoraclefull.yaml"
D_REL = "pools/anchors_gmail.yaml"
O_SOURCE = ROOT / "data" / "headroom" / "pools" / "gmoraclefull.yaml"
D_SOURCE = ROOT / "data" / "heterogeneity" / "pools" / "anchors_gmail.yaml"
D_PROMPT_SHA_PREFIX = "ae8c7c7f"  # Pre-reg 13 Leg M's default (anchor_baseline), the same rendered prompt
QUEUE_REL = "gmail/queue.jsonl"
COST_O_USD = 0.10
COST_D_USD = 0.08
BUDGET_USD = 30
WORKERS = 4
STUDY = "Pre-registration 15 -- Gmail answer-key headroom: full answer key (O) vs the default prompt (D), 60 tasks"
RULE = ("PRIMARY: H = mean over the 60 tasks of O_t - D_t (arm score = mean of its 2 replicates, each 30-step "
        "rescored), paired t, two-sided 90% CI. large iff CI lower bound >= 0.10; small iff CI upper bound < 0.10; "
        "else unresolved")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_queue(task_ids: list[str]) -> list[QueueItem]:
    """Tasks in a seeded order; within a task the 4 episodes (arm x replicate) in a seeded order."""
    rng = np.random.default_rng(SEED)
    task_order = rng.permutation(sorted(task_ids))
    items = []
    for task in task_order:
        episodes = []
        for pool, arm in ((O_POOL, O_ARM), (D_POOL, D_ARM)):
            for rep in REPLICATES:
                episodes.append((pool, arm, rep))
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
        text += json.dumps(asdict(item)) + "\n"
    return text


def estimate_cost(queue: list[QueueItem]) -> float:
    total = 0.0
    for item in queue:
        if item.arm_id == O_ARM:
            total += COST_O_USD
        else:
            total += COST_D_USD
    return total


def rebuilt_oracle_bytes() -> bytes:
    """The answer-key pool rebuilt from the generator; must equal the frozen file."""
    built = bgo.build_all()
    with tempfile.TemporaryDirectory() as tmp:
        path = bgo.pool_path(Path(tmp))
        bgo.write_oracle_pool(path, built)
        return path.read_bytes()


def check_arms(o_bytes: bytes, d_bytes: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        o_path = Path(tmp) / "o.yaml"
        d_path = Path(tmp) / "d.yaml"
        o_path.write_bytes(o_bytes)
        d_path.write_bytes(d_bytes)
        o_arms = {}
        for arm in make_pools.load_pool(o_path):
            o_arms[arm.arm.arm_id] = arm
        d_arms = {}
        for arm in make_pools.load_pool(d_path):
            d_arms[arm.arm.arm_id] = arm
    if O_ARM not in o_arms:
        raise ValueError(f"{O_ARM} not in the answer-key pool")
    if D_ARM not in d_arms:
        raise ValueError(f"{D_ARM} not in the anchors pool")
    if not d_arms[D_ARM].sha256.startswith(D_PROMPT_SHA_PREFIX):
        raise ValueError(f"D prompt sha {d_arms[D_ARM].sha256[:8]} != Leg M default {D_PROMPT_SHA_PREFIX}")


def build_manifest(queue: list[QueueItem], files: dict[str, str], o_bytes: bytes) -> dict:
    task_order = []
    for item in queue:
        if item.task_id not in task_order:
            task_order.append(item.task_id)
    estimate = estimate_cost(queue)
    return {
        "study": STUDY,
        "design": ("O (GM_oracle_full, 60-task answer key) and D (GM_anchor_baseline, the Gmail default) x the 60 "
                   "Gmail real tasks x replicates {0, 1}; seeded task order and seeded within-task episode order; "
                   "isolated resets; profile gmail (gpt-5.4-mini low, 180 s), 30-step cap"),
        "stage": "study",
        "seed": SEED,
        "decision_rule": RULE,
        "arms": {"O": {"arm_id": O_ARM, "pool": O_POOL}, "D": {"arm_id": D_ARM, "pool": D_POOL}},
        "pool_files": [O_REL, D_REL],
        "n_tasks": len(task_order),
        "task_ids": task_order,
        "n_episodes": len(queue),
        "replicates": list(REPLICATES),
        "max_steps": collect.MAX_STEPS,
        "timeout_s": collect.profile_spec(PROFILE)["agent"]["timeout_s"],
        "agent": collect.profile_spec(PROFILE)["agent"],
        "cost_per_episode_assumed_usd": {"O": COST_O_USD, "D": COST_D_USD},
        "cost_estimate_usd": round(estimate, 2),
        "budget_usd": BUDGET_USD,
        "oracle_pool_sha256": hashlib.sha256(o_bytes).hexdigest(),
        "real_tasks_sha256": sha(bgo.bank_path()),
        "generator_sha256": {"build_gmail_oracle.py": sha(Path(bgo.__file__)),
                             "prepare_headroom_gmail.py": sha(Path(__file__))},
        "files": files,
    }


def prepare(out: Path = OUT) -> dict:
    out = Path(out)
    o_bytes = O_SOURCE.read_bytes()
    if rebuilt_oracle_bytes() != o_bytes:
        raise ValueError(f"{O_SOURCE} differs from a fresh build_gmail_oracle build; refusing")
    d_bytes = D_SOURCE.read_bytes()
    check_arms(o_bytes, d_bytes)
    task_ids = []
    for task in bgo.load_bank():
        task_ids.append(str(task["id"]))
    queue = build_queue(task_ids)
    qtext = queue_text(queue)
    files = {QUEUE_REL: hashlib.sha256(qtext.encode()).hexdigest(),
             O_REL: hashlib.sha256(o_bytes).hexdigest(),
             D_REL: hashlib.sha256(d_bytes).hexdigest()}
    manifest = build_manifest(queue, files, o_bytes)
    contents = {QUEUE_REL: qtext.encode(), O_REL: o_bytes, D_REL: d_bytes,
                "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    for rel in contents:
        dest = out / rel
        if dest.exists() and dest.read_bytes() != contents[rel]:
            raise ValueError(f"existing {rel} differs; refusing to overwrite")
    for rel in contents:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(contents[rel])
    return {"queue": queue, "manifest": manifest}


class _BankOnlyAdapter:
    def task_ids(self):
        ids = []
        for task in bgo.load_bank():
            ids.append(str(task["id"]))
        return ids


def validate(data: Path) -> list[str]:
    """The collector's own start-up checks against `data` (no model call, no server)."""
    data = Path(data)
    lines = []
    manifest = json.loads((data / "manifest.json").read_text())
    collect.verify_queue(data, PROFILE)
    lines.append("verify_queue: queue + pool files byte-identical to manifest.files")
    prompts = collect.load_prompts(data, PROFILE)
    lines.append(f"load_prompts: {sorted(prompts)} (each rendered prompt sha256-checked)")
    queue = make_pools.read_queue(data / collect.profile_spec(PROFILE)["queue"])
    collect.check_queue_arms(queue, prompts)
    if len(queue) != manifest["n_episodes"]:
        raise RuntimeError(f"queue has {len(queue)} items; manifest says {manifest['n_episodes']}")
    collect.check_bank(_BankOnlyAdapter(), queue)
    lines.append(f"check_queue_arms + check_bank: {len(queue)} items ok")
    agent = collect.agent_settings(data, PROFILE)
    lines.append(f"agent: {agent['llm_provider']}/{agent['llm_model']} effort={agent['llm_reasoning_effort']} "
                 f"timeout={agent['timeout_s']}s")
    with tempfile.TemporaryDirectory() as tmp:
        shares = collect.partition_remaining(queue, Path(tmp), WORKERS, False)
        sizes = []
        for share in shares:
            sizes.append(len(share))
    lines.append(f"partition over {WORKERS} workers: {sizes}")
    return lines


def launch_command() -> str:
    rel = OUT.relative_to(ROOT)
    return (f"caffeinate -dims ./.venv/bin/python experiments/growing_bandits/empirical/collect.py --profile {PROFILE} "
            f"--data {rel} --log-dir logs/{LOG_NAME} --budget {BUDGET_USD} --workers {WORKERS}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build + validate in a temp dir; write nothing")
    args = ap.parse_args(argv)
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="headroom_gmail_dry_"))
        try:
            result = prepare(tmp / "data")
            for line in validate(tmp / "data"):
                print(line)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print("DRY RUN (nothing written)")
    else:
        result = prepare(OUT)
        for line in validate(OUT):
            print(line)
    m = result["manifest"]
    print(f"{m['n_episodes']} episodes on {m['n_tasks']} tasks; estimate ${m['cost_estimate_usd']}; "
          f"budget ${m['budget_usd']}")
    print(launch_command())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
