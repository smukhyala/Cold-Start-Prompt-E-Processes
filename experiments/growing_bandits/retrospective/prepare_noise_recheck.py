"""Freeze a 30-cell (60-episode) Gmail noise recheck of September's pool-G repeat cells.

No model calls and no servers. Samples 30 of the 145 historical G (arm, task) cells that have
replicates 0 and 1, and queues each twice (replicates 0 and 1) in seeded random order.
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
OUT = ROOT / "data/noise_recheck_gmail"
SEED = 20261011
N_CELLS = 30
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/empirical"))

from make_pools import QueueItem  # noqa: E402
from replay import load_snapshot  # noqa: E402

DISCLOSURE = (
    "The 30-step cap is now enforced (bug fixed 2026-10-10 in src/cold_start/tasks/webarena.py); "
    "September (Pre-reg 9) ran effectively uncapped (4.6% of Gmail episodes exceeded 30 steps). "
    "The 2026-10-09 first-run pilot also ran uncapped."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_cells() -> list[dict]:
    """All pool-G cells with replicates 0 and 1, from the frozen snapshot, plus logged cost."""
    snap = load_snapshot(EMPIRICAL / "reservoirs")
    snap = snap[(snap["pool"] == "G") & (snap["status"] == "ok")]
    outcome = {}
    for row in snap.itertuples():
        key = (str(row.arm_id), str(row.task_id), int(row.replicate))
        if key in outcome:
            raise ValueError(f"duplicate snapshot record {key}")
        outcome[key] = int(row.success)
    cost = {}
    for path in sorted((ROOT / "logs/empirical_pool").glob("worker_*.jsonl")):
        for line in path.open():
            rec = json.loads(line)
            if rec["pool"] == "G" and rec["status"] == "ok":
                cost[(rec["arm_id"], rec["task_id"], int(rec["replicate"]))] = float(rec["cost_usd"])
    cells = []
    for arm, task, rep in sorted(outcome):
        if rep != 0 or (arm, task, 1) not in outcome:
            continue
        cells.append(
            {
                "arm_id": arm,
                "task_id": task,
                "sept_success": [outcome[arm, task, 0], outcome[arm, task, 1]],
                "sept_cost_usd": [cost.get((arm, task, 0)), cost.get((arm, task, 1))],
                "sept_disagree": int(outcome[arm, task, 0] != outcome[arm, task, 1]),
            }
        )
    return cells


def main() -> None:
    source = json.loads((EMPIRICAL / "manifest.json").read_text())
    hist = historical_cells()
    if len(hist) != 145:
        raise ValueError(f"expected 145 historical G repeat cells, found {len(hist)}")
    rng = np.random.default_rng(SEED)
    picked = sorted(rng.choice(len(hist), size=N_CELLS, replace=False).tolist())
    cells = [hist[i] for i in picked]
    order = []
    for i in range(len(cells)):
        order.extend([(i, 0), (i, 1)])
    shuffled = rng.permutation(len(order))
    queue = []
    for index, pos in enumerate(shuffled):
        ci, rep = order[int(pos)]
        queue.append(
            QueueItem(
                index=index,
                pool="G",
                arm_id=cells[ci]["arm_id"],
                task_id=cells[ci]["task_id"],
                replicate=rep,
                pilot=False,
            )
        )
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("pool_G.yaml", "pool_F.yaml", "pool_anchor.yaml"):
        src = EMPIRICAL / name
        if sha(src) != source["files"][name]:
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
    k_all = sum(c["sept_disagree"] for c in hist)
    manifest = {
        "design": "30 sampled September pool-G repeat cells, each rerun as replicates 0 and 1; "
        "single worker, seeded random order; profile prereg9 (180 s clock)",
        "hypothesis": "provider/model drift raised G self-disagreement (pilot 33% vs 10.3% in September)",
        "source_manifest_sha256": sha(EMPIRICAL / "manifest.json"),
        "seed": SEED,
        "profile": "prereg9",
        "n_cells": N_CELLS,
        "n_episodes": len(queue),
        "n_historical_repeat_cells": len(hist),
        "september_disagree_all_cells": k_all,
        "september_disagree_sampled_cells": sum(c["sept_disagree"] for c in cells),
        "disclosure": DISCLOSURE,
        "cells": cells,
        "files": files,
    }
    mpath = OUT / "manifest.json"
    mtext = json.dumps(manifest, indent=2) + "\n"
    if mpath.exists() and mpath.read_text() != mtext:
        raise ValueError("existing manifest differs; refusing to overwrite")
    mpath.write_text(mtext)
    print(
        f"Prepared {len(queue)} episodes on {N_CELLS} cells under {OUT}; "
        f"September disagreement: {manifest['september_disagree_sampled_cells']}/{N_CELLS} sampled, "
        f"{k_all}/{len(hist)} all"
    )


if __name__ == "__main__":
    main()
