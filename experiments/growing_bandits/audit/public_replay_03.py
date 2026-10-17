"""SWE-bench Verified matched-pair replay; see PUBLIC_REPLAY_PREREG_03.md."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths
from experiments.growing_bandits.audit.public_replay_01 import e1_path, first_cross

REV = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
PAIRS = (
    ("20240402_rag_gpt4", "20240402_sweagent_gpt4", "gpt-4-1106-preview"),
    ("20240402_rag_claude3opus", "20240402_sweagent_claude3opus", "claude-3-opus-20240229"),
)
OUT = Path("results/paper/public_replay_03")


def load(name: str):
    base = f"https://raw.githubusercontent.com/SWE-bench/experiments/{REV}/evaluation/verified/{name}"
    with urllib.request.urlopen(base + "/metadata.yaml", timeout=30) as f:
        meta = f.read().decode()
    with urllib.request.urlopen(base + "/results/results.json", timeout=30) as f:
        result = json.load(f)
    return meta, result


def present_ids(result):
    duplicate_records = 0
    for key, value in result.items():
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"Invalid instance list {key}")
        duplicate_records += len(value) - len(set(value))
    return set().union(*(set(v) for v in result.values())), duplicate_records


def analyze_pair(pair_index: int, reference: str, challenger: str, model: str):
    ref_meta, ref_result = load(reference)
    chal_meta, chal_result = load(challenger)
    if model not in ref_meta or model not in chal_meta:
        raise ValueError(f"Model ID {model} not found in both metadata files")
    (ref_ids, ref_duplicates), (chal_ids, chal_duplicates) = present_ids(ref_result), present_ids(chal_result)
    ambiguous = set()
    for result in (ref_result, chal_result):
        resolved = set(result["resolved"])
        bad = set().union(*(set(result.get(k, [])) for k in ("no_generation", "install_fail", "reset_failed", "test_errored")))
        ambiguous |= resolved & bad
    tasks = sorted((ref_ids & chal_ids) - ambiguous)
    if len(tasks) < 400:
        raise ValueError(f"Only {len(tasks)} unambiguous common instances")
    y = np.array([[int(task in set(result["resolved"])) for task in tasks]
                  for result in (ref_result, chal_result)], dtype=np.int8)
    n = len(tasks)
    rng = np.random.default_rng(np.random.SeedSequence([20261012, 4, pair_index]))
    order = rng.permutation(n)
    log_path = e1_path(y[:, order])
    cross = first_cross(log_path, alpha=0.025)
    boot_rng = np.random.default_rng(20261015 + pair_index)
    diff = y[1].astype(float) - y[0]
    boot = diff[boot_rng.integers(0, n, (10000, n))].mean(axis=1)
    B = 2000
    orders = [np.random.default_rng(np.random.SeedSequence([20261012, 40, pair_index, b])).permutation(n)
              for b in range(B)]
    batch = np.stack([y[:, perm] for perm in orders])
    chal_mask = np.tile(np.array([[False, True]]), (B, 1))
    paths = e1b_log_paths(batch, np.zeros(B, dtype=int), chal_mask)
    hits = paths >= np.log(40)
    crossings = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
    return {
        "pair": pair_index, "reference": reference, "challenger": challenger,
        "model": model, "common_tasks": n, "reference_only": len(ref_ids-chal_ids),
        "challenger_only": len(chal_ids-ref_ids), "ambiguous": len(ambiguous),
        "duplicate_status_records": {"reference": ref_duplicates, "challenger": chal_duplicates},
        "reference_resolved": int(y[0].sum()), "challenger_resolved": int(y[1].sum()),
        "paired_gain": float(diff.mean()),
        "gain_bootstrap_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "challenger_only_wins": int(((y[1] == 1) & (y[0] == 0)).sum()),
        "reference_only_wins": int(((y[0] == 1) & (y[1] == 0)).sum()),
        "prespecified_cross_task": cross,
        "max_log_e": float(log_path.max()), "final_log_e": float(log_path[-1]),
        "full_grid_episodes": 2*n, "prespecified_stop_episodes": 2*(cross or n),
        "replay_crossings": int((crossings > 0).sum()),
        "replay_orders": B,
        "replay_median_cross_task_among_hits": float(np.median(crossings[crossings > 0])) if np.any(crossings > 0) else None,
        "replay_min_cross_task": int(crossings[crossings > 0].min()) if np.any(crossings > 0) else None,
        "replay_max_cross_task": int(crossings.max()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [analyze_pair(i, *pair) for i, pair in enumerate(PAIRS)]
    (OUT / "summary.json").write_text(json.dumps({"revision": REV, "pairs": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
