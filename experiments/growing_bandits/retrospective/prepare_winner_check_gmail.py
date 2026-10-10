"""Freeze Pre-registration 13, Gmail arm (the winner's-curse test): default prompt x the pilot's 30 tasks x 2.

No model calls, no servers. G_14 (September's top G prompt, 0.70 vs the default's 0.60 on Pre-reg 9's 30 tasks)
already ran twice on 30 other, never-selected Gmail tasks in the 2026-10-09 first-run pilot. This freezes the
DEFAULT arm (anchor_baseline) on those same tasks, replicates 0 and 1 (60 episodes, seeded order, one worker).
Idempotent; refuses to overwrite a differing manifest / queue / pool file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EMPIRICAL = ROOT / "data/empirical_pool"
PILOT = ROOT / "data/first_run_pilot_gmail"
OUT = ROOT / "data/winner_check_gmail"
SEED = 20261012
DEFAULT_ARM = "anchor_baseline"
DEFAULT_POOL = "anchor"
WINNER_ARM = "G_14"
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/empirical"))

from make_pools import QueueItem, load_pool  # noqa: E402

DISCLOSURE = {
    "step_cap": (
        "The pilot G_14 episodes (2026-10-09) ran effectively UNCAPPED (the 30-step cap bug was fixed on "
        "2026-10-10 in src/cold_start/tasks/webarena.py); the default episodes run NOW with the 30-step cap "
        "enforced. The PRIMARY analysis therefore scores BOTH arms at the 30-step rescore "
        "(study.rescore(outcomes, 30): success and not timed_out and steps <= 30, the glk30 rule). The "
        "as-collected comparison (uncapped G_14 vs capped default) is SECONDARY and confounded."
    ),
    "time_gap": (
        "G_14 pilot episodes ran 2026-10-09; the default episodes run after 2026-10-12 (frozen date). Provider/"
        "model drift between the two runs is not controlled; the noise recheck (2026-10-11) bounds how large "
        "rerun noise is, not drift."
    ),
    "june_baseline": (
        "These 30 tasks were also run in June 2026 as part of the gmail_paired12 baseline (12 hand-written "
        "prompts, results/audit/matrices/gmail_paired12.csv), long before G_14 was selected. They were not used "
        "to select G_14 or to define the default; disclosed for completeness."
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads((EMPIRICAL / "manifest.json").read_text())
    pilot = json.loads((PILOT / "manifest.json").read_text())
    tasks = sorted(pilot["task_ids"])
    if len(tasks) != 30 or set(tasks) & set(source["task_subset"]):
        raise ValueError("pilot tasks must be 30 and disjoint from the September task subset")
    anchors = {arm.arm.arm_id: arm for arm in load_pool(EMPIRICAL / "pool_anchor.yaml")}
    if list(anchors) != [DEFAULT_ARM] or anchors[DEFAULT_ARM].pool != DEFAULT_POOL:
        raise ValueError(f"expected exactly the default arm {DEFAULT_ARM} in pool {DEFAULT_POOL}")
    pairs = []
    for task in tasks:
        for rep in (0, 1):
            pairs.append((task, rep))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    queue = []
    for index, pos in enumerate(order):
        task, rep = pairs[int(pos)]
        queue.append(QueueItem(index=index, pool=DEFAULT_POOL, arm_id=DEFAULT_ARM, task_id=task,
                               replicate=rep, pilot=False))
    if len(queue) != 60:
        raise ValueError("queue is not 30 tasks x 2 replicates")
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("pool_G.yaml", "pool_F.yaml", "pool_anchor.yaml"):
        src = EMPIRICAL / name
        if sha(src) != source["files"][name] or sha(src) != pilot["files"][name]:
            raise ValueError(f"original frozen pool changed: {name}")
        dest = OUT / name
        if dest.exists() and sha(dest) != sha(src):
            raise ValueError(f"prepared pool differs from source: {name}")
        shutil.copy2(src, dest)
        files[name] = sha(dest)
    qpath = OUT / "queue.jsonl"
    qtext = "".join(json.dumps(asdict(q), sort_keys=True) + "\n" for q in queue)
    if qpath.exists() and qpath.read_text() != qtext:
        raise ValueError("existing queue differs; refusing to overwrite")
    qpath.write_text(qtext)
    files["queue.jsonl"] = sha(qpath)
    manifest = {
        "study": "Pre-registration 13, Gmail arm (winner's-curse test)",
        "design": "default arm x the first-run pilot's 30 held-out Gmail tasks x replicates 0 and 1; single "
        "worker, seeded random order; profile prereg9 (180 s clock); compared with the pilot's 2 G_14 runs",
        "source_manifest_sha256": sha(EMPIRICAL / "manifest.json"),
        "pilot_manifest_sha256": sha(PILOT / "manifest.json"),
        "pilot_log": "logs/first_run_pilot_gmail/worker_0.jsonl (G_14 replicates 0 = 'first', 1 = 'same')",
        "seed": SEED,
        "profile": "prereg9",
        "task_ids": tasks,
        "n_tasks": len(tasks),
        "n_episodes": len(queue),
        "default_arm": DEFAULT_ARM,
        "default_pool": DEFAULT_POOL,
        "winner_arm": WINNER_ARM,
        "september_in_sample": {"G_14_rate": 0.70, "default_rate": 0.60, "difference": 0.10,
                                "tasks": "Pre-reg 9 task_subset (30 tasks), replicate 0, uncapped"},
        "registered_reading": {
            "primary": "mean(d) with one-sided 95% bound, d_t = G14_t - default_t, both arms 30-step rescored",
            "reproduced": "95% LOWER bound >= 0.10",
            "shrank": "95% UPPER bound < 0.10",
            "otherwise": "inconclusive",
        },
        "rescore_rule": "experiments/growing_bandits/empirical/study.py::rescore(outcomes, 30)",
        "disclosure": DISCLOSURE,
        "files": files,
    }
    mpath = OUT / "manifest.json"
    mtext = json.dumps(manifest, indent=2) + "\n"
    if mpath.exists() and mpath.read_text() != mtext:
        raise ValueError("existing manifest differs; refusing to overwrite")
    mpath.write_text(mtext)
    print(f"Prepared {len(queue)} episodes ({DEFAULT_ARM} x {len(tasks)} tasks x 2) under {OUT}")


if __name__ == "__main__":
    main()
