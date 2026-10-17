"""Freeze the Linear fixed-pair trial without running an agent or reading task outcomes."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect  # noqa: E402
import make_pools  # noqa: E402
from cold_start.types import PromptVector  # noqa: E402

OUT = ROOT / "data" / "prospective_linear"
BANK = ROOT.parent / "webarena-infinity" / "apps" / "linear-account-settings" / "real-tasks.json"
SEED = 20261012
POOL = "fixed_pair"
ARMS = {
    "linear_baseline": make_pools.BASELINE_VECTOR,
    "linear_explorer": {"planning": 1, "verification": 0, "agency": 2,
                        "expertise": 2, "format": 0, "goal": 0},
}
QUEUE_REL = "linear/queue.jsonl"
POOL_REL = "pools/fixed_pair.yaml"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_once(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"refusing to overwrite changed frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    bank_bytes = BANK.read_bytes()
    task_ids = [str(row["id"]) for row in json.loads(bank_bytes)]
    if len(task_ids) != 120 or len(set(task_ids)) != 120:
        raise ValueError("expected 120 unique Linear task IDs")
    arm_objs = [make_pools.grid_arm(name, PromptVector(**vec)) for name, vec in ARMS.items()]
    pool_path = OUT / POOL_REL
    if not pool_path.exists():
        make_pools.write_pool(pool_path, POOL, arm_objs, ROOT / "configs/template.jinja",
                              ROOT / "configs/axes.yaml", {"study": "prospective Linear fixed pair"})
    loaded = make_pools.load_pool(pool_path)
    if {p.arm.arm_id for p in loaded} != set(ARMS):
        raise ValueError("frozen pool has the wrong arms")
    rng = np.random.default_rng(SEED)
    order = [str(x) for x in rng.permutation(sorted(task_ids))]
    queue = []
    for task in order:
        for arm in rng.permutation(list(ARMS)):
            queue.append(make_pools.QueueItem(index=len(queue), pool=POOL, arm_id=str(arm),
                                              task_id=task, replicate=0, pilot=False))
    queue_bytes = "".join(json.dumps(asdict(q)) + "\n" for q in queue).encode()
    pool_bytes = pool_path.read_bytes()
    files = {QUEUE_REL: digest(queue_bytes), POOL_REL: digest(pool_bytes)}
    manifest = {
        "study": "Prospective Linear account-settings fixed-pair trial",
        "profile": "linear", "seed": SEED, "task_ids": order, "n_tasks": 120,
        "n_episodes": 240, "arms": list(ARMS), "pool_files": [POOL_REL],
        "bank_sha256": digest(bank_bytes),
        "prompt_sha256": {p.arm.arm_id: p.sha256 for p in loaded},
        "template_sha256": digest((ROOT / "configs/template.jinja").read_bytes()),
        "axes_sha256": digest((ROOT / "configs/axes.yaml").read_bytes()),
        "files": files,
    }
    put_once(OUT / QUEUE_REL, queue_bytes)
    put_once(OUT / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    collect.verify_queue(OUT, "linear")
    prompts = collect.load_prompts(OUT, "linear")
    collect.check_queue_arms(queue, prompts)
    if set(task_ids) != {q.task_id for q in queue}:
        raise ValueError("queue does not cover the entire task bank")
    print(json.dumps({"episodes": len(queue), "tasks": len(order), "files": files,
                      "prompt_sha256": manifest["prompt_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
