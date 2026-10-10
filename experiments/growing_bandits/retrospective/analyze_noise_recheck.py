"""Analyze the frozen Gmail noise recheck once all 60 episodes are terminal and scored.

    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_noise_recheck.py
    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_noise_recheck.py --synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/noise_recheck_gmail"
LOGS = ROOT / "logs/noise_recheck_gmail"
OUT = ROOT / "results/noise_recheck_gmail"
HIST_RATE = 0.103


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(data: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(data / name) != expected:
            raise ValueError(f"prepared input changed: {name}")
    queue = [json.loads(line) for line in (data / "queue.jsonl").open() if line.strip()]
    return manifest, queue


def read_terminals(log_dir: Path, queue: list[dict]) -> dict:
    expected = {(q["arm_id"], q["task_id"], int(q["replicate"])) for q in queue}
    if len(expected) != 60:
        raise ValueError("queue does not hold 60 distinct episodes")
    terminals = {}
    for path in sorted(Path(log_dir).glob("worker_*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["arm_id"], row["task_id"], int(row["replicate"])
            if key not in expected:
                raise ValueError(f"record outside frozen queue: {key}")
            if row["status"] in ("ok", "missing"):
                if key in terminals:
                    raise ValueError(f"duplicate terminal record: {key}")
                terminals[key] = row
    missing = expected - set(terminals)
    if missing:
        raise ValueError(f"incomplete: {len(missing)} of 60 episodes lack terminal records")
    for key, row in terminals.items():
        if row["status"] != "ok" or row.get("success") is None:
            raise ValueError(f"episode {key} is not scored (status {row['status']})")
    return terminals


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def analyze(manifest: dict, terminals: dict) -> dict:
    cells = manifest["cells"]
    n = len(cells)
    now_dis, sept_dis = [], []
    now_succ, sept_succ, now_cost, sept_cost = [], [], [], []
    for cell in cells:
        a, t = cell["arm_id"], cell["task_id"]
        s0, s1 = (int(terminals[a, t, r]["success"]) for r in (0, 1))
        now_dis.append(int(s0 != s1))
        sept_dis.append(int(cell["sept_disagree"]))
        now_succ += [s0, s1]
        sept_succ += list(cell["sept_success"])
        now_cost += [float(terminals[a, t, r]["cost_usd"]) for r in (0, 1)]
        sept_cost += [float(c) for c in cell["sept_cost_usd"]]
    k = sum(now_dis)
    lo, hi = clopper_pearson(k, n)
    b = sum(1 for x, y in zip(now_dis, sept_dis) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(now_dis, sept_dis) if x == 0 and y == 1)
    mcnemar_p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    k_all = int(manifest["september_disagree_all_cells"])
    n_all = int(manifest["n_historical_repeat_cells"])
    mean_now, mean_sept = float(np.mean(now_cost)), float(np.mean(sept_cost))
    return {
        "n_cells": n,
        "n_episodes": len(terminals),
        "now_disagree_k": k,
        "now_disagree_rate": k / n,
        "now_disagree_ci95_clopper_pearson": [lo, hi],
        "september_disagree_same_cells_k": sum(sept_dis),
        "september_disagree_same_cells_rate": sum(sept_dis) / n,
        "september_disagree_all_k": k_all,
        "september_disagree_all_n": n_all,
        "september_disagree_all_rate": k_all / n_all,
        "binomial_test_vs_0.103": {
            "null_rate": HIST_RATE,
            "p_two_sided": float(stats.binomtest(k, n, HIST_RATE, alternative="two-sided").pvalue),
        },
        "mcnemar_now_vs_september_same_cells": {
            "now_only_disagree": b,
            "september_only_disagree": c,
            "both_disagree": sum(1 for x, y in zip(now_dis, sept_dis) if x and y),
            "neither": sum(1 for x, y in zip(now_dis, sept_dis) if not x and not y),
            "p_exact_two_sided": mcnemar_p,
        },
        "mean_success_now": float(np.mean(now_succ)),
        "mean_success_september_same_cells": float(np.mean(sept_succ)),
        "mean_cost_per_episode_now_usd": mean_now,
        "mean_cost_per_episode_september_same_cells_usd": mean_sept,
        "cost_ratio_now_over_september": mean_now / mean_sept if mean_sept > 0 else None,
        "disclosure": manifest["disclosure"],
    }


def synthetic_terminals(manifest: dict, queue: list[dict], rate: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    terminals = {}
    for cell in manifest["cells"]:
        s0 = int(rng.random() < 0.6)
        s1 = 1 - s0 if rng.random() < rate else s0
        for rep, s in ((0, s0), (1, s1)):
            terminals[cell["arm_id"], cell["task_id"], rep] = {
                "status": "ok",
                "success": s,
                "cost_usd": float(rng.uniform(0.05, 0.15)),
            }
    return terminals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--log-dir", type=Path, default=LOGS)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--synthetic", action="store_true", help="dry run on fake logs; writes summary_synthetic.json")
    args = parser.parse_args()
    manifest, queue = load_frozen(args.data)
    if args.synthetic:
        terminals = synthetic_terminals(manifest, queue, 0.3)
        name = "summary_synthetic.json"
    else:
        terminals = read_terminals(args.log_dir, queue)
        name = "summary.json"
    summary = analyze(manifest, terminals)
    summary["synthetic"] = bool(args.synthetic)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
