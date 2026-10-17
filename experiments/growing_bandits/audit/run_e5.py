#!/usr/bin/env python3
"""EXPLORATORY (data already seen): E5 pool-futility e-process (replay, validity, power).

No API calls, single process.  Outputs go to results/audit/e5/.

    ./.venv/bin/python experiments/growing_bandits/audit/run_e5.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.growing_bandits.audit import futility_eprocess as fu  # noqa: E402
from experiments.growing_bandits.audit.incumbent_gain import (  # noqa: E402
    VARIANTS,
    primary_matrix,
    split_arms_and_incumbent,
)

OUT = ROOT / "results" / "audit" / "e5"
DELTAS = (0.05, 0.10, 0.15, 0.20)
ALPHA = 0.05
FRACS = (0.25, 0.5, 0.75, 1.0)


def permuted_D(arms: np.ndarray, inc: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    K, n = arms.shape
    D = np.zeros((B, K, n), dtype=np.int8)
    diff = arms - inc[None, :]
    for b in range(B):
        D[b] = diff[:, rng.permutation(n)]
    return D


def replay(n_orders: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, name, drop in VARIANTS:
        Y = primary_matrix(name, drop)
        arms, inc = split_arms_and_incumbent(name, Y)
        K, n = arms.shape
        true_gain = arms.mean(axis=1) - inc.mean()
        D = permuted_D(arms, inc, rng, n_orders)
        for delta in DELTAS:
            logp = fu.futility_log_paths(D, delta)
            arm_cross = fu.first_crossing(logp, ALPHA)
            pool_cross = fu.first_crossing(fu.pool_log_paths(logp), ALPHA)
            # all arms ever dropped (running max), also valid at level alpha
            all_dropped = (arm_cross >= 0).all(axis=1)
            ep = fu.episodes_summary(arm_cross, pool_cross, n)
            declared = pool_cross >= 0
            if declared.any():
                med_task = float(np.median(pool_cross[declared] + 1))
            else:
                med_task = float("nan")
            row = {
                "label": "EXPLORATORY",
                "pool": label,
                "n_challengers": K,
                "n_tasks": n,
                "delta": delta,
                "alpha": ALPHA,
                "n_orders": n_orders,
                "incumbent_rate": float(inc.mean()),
                "best_arm_gain": float(true_gain.max()),
                "n_arms_truly_ge_delta": int((true_gain >= delta).sum()),
                "p_pool_futility": float(declared.mean()),
                "median_task_of_declaration": med_task,
                "p_all_arms_ever_dropped": float(all_dropped.mean()),
                "frac_arms_dropped_end": float((arm_cross >= 0).mean()),
                "episodes_full": float(ep["full"].mean()),
                "episodes_drop_loser": float(ep["drop"].mean()),
                "frac_episodes_saved_drop": float(1.0 - ep["drop"].mean() / ep["full"].mean()),
                "episodes_pool_stop": float(ep["stop"].mean()),
                "episodes_drop_and_stop": float(ep["both"].mean()),
                "frac_episodes_saved_drop_and_stop": float(1.0 - ep["both"].mean() / ep["full"].mean()),
            }
            for frac in FRACS:
                t_idx = int(math.ceil(frac * n)) - 1
                dropped_by = (arm_cross >= 0) & (arm_cross <= t_idx)
                row[f"frac_dropped_at_{int(frac * 100)}pct"] = float(dropped_by.mean())
            rows.append(row)
        print(f"replay done: {label}", flush=True)
    return pd.DataFrame(rows)


def simulate_D(p_task: np.ndarray, q_task: np.ndarray, K: int, n: int, S: int,
               rng: np.random.Generator) -> np.ndarray:
    """Tasks iid with replacement from the bank.  Arm 0 has success prob q_task, arms 1..K-1 and the
    incumbent have p_task.  Returns D (S, K, n)."""
    idx = rng.integers(0, len(p_task), size=(S, n))
    p = p_task[idx]
    q = q_task[idx]
    yb = (rng.random((S, n)) < p).astype(np.int8)
    D = np.zeros((S, K, n), dtype=np.int8)
    for k in range(K):
        if k == 0:
            prob = q
        else:
            prob = p
        yk = (rng.random((S, n)) < prob).astype(np.int8)
        D[:, k, :] = yk - yb
    return D


def validity(sims: int, seed: int, n: int, K: int, chunk: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Y = primary_matrix("G", ())
    p_task = Y.to_numpy(dtype=float).mean(axis=0)
    rows = []
    for scenario, extra in (("V1_boundary", 0.0), ("V2_boundary_plus_0.05", 0.05)):
        for delta in DELTAS:
            gain = delta + extra
            shift = fu.boundary_shift(p_task, gain)
            q_task = np.minimum(p_task + shift, 1.0)
            n_pool = 0
            n_all = 0
            n_best_arm = 0
            done = 0
            while done < sims:
                S = min(chunk, sims - done)
                D = simulate_D(p_task, q_task, K, n, S, rng)
                logp = fu.futility_log_paths(D, delta)
                arm_cross = fu.first_crossing(logp, ALPHA)
                pool_cross = fu.first_crossing(fu.pool_log_paths(logp), ALPHA)
                n_pool += int((pool_cross >= 0).sum())
                n_all += int(((arm_cross >= 0).all(axis=1)).sum())
                n_best_arm += int((arm_cross[:, 0] >= 0).sum())
                done += S
            rows.append({
                "label": "EXPLORATORY",
                "scenario": scenario,
                "delta": delta,
                "true_gain_of_best_arm": float(np.mean(q_task - p_task)),
                "K": K,
                "n_tasks": n,
                "sims": sims,
                "alpha": ALPHA,
                "p_false_pool_futility": n_pool / sims,
                "p_false_all_arms_dropped": n_all / sims,
                "p_best_arm_alone_crosses": n_best_arm / sims,
                "mc_se": math.sqrt(max(n_pool / sims, 1e-12) * (1 - n_pool / sims) / sims),
                "passes_le_alpha": bool(n_pool / sims <= ALPHA),
            })
        print(f"validity done: {scenario}", flush=True)
    return pd.DataFrame(rows)


def power(sims: int, seed: int, chunk: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Y = primary_matrix("G", ())
    p_task = Y.to_numpy(dtype=float).mean(axis=0)
    ns = (30, 60, 120, 240)
    rows = []
    for K in (12, 50):
        for delta in DELTAS:
            counts = {}
            for nn in ns:
                counts[nn] = 0
            done = 0
            while done < sims:
                S = min(chunk, sims - done)
                D = simulate_D(p_task, p_task, K, 240, S, rng)
                logp = fu.futility_log_paths(D, delta)
                pool_cross = fu.first_crossing(fu.pool_log_paths(logp), ALPHA)
                for nn in ns:
                    counts[nn] += int(((pool_cross >= 0) & (pool_cross < nn)).sum())
                done += S
            for nn in ns:
                rows.append({
                    "label": "EXPLORATORY",
                    "K": K,
                    "delta": delta,
                    "n_tasks": nn,
                    "sims": sims,
                    "alpha": ALPHA,
                    "p_futility_declared": counts[nn] / sims,
                })
        print(f"power done: K={K}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--orders", type=int, default=500)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=250)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = replay(args.orders, seed=0)
    rep.to_csv(args.out / "replay.csv", index=False)
    val = validity(args.sims, seed=1, n=240, K=12, chunk=args.chunk)
    val.to_csv(args.out / "validity.csv", index=False)
    pw = power(args.sims, seed=2, chunk=args.chunk)
    pw.to_csv(args.out / "power.csv", index=False)
    print(rep[["pool", "delta", "p_pool_futility", "median_task_of_declaration",
               "frac_arms_dropped_end", "frac_episodes_saved_drop"]].round(3).to_string(index=False))
    print(val.round(3).to_string(index=False))
    print(pw.pivot_table(index=["K", "delta"], columns="n_tasks", values="p_futility_declared").round(3))
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
