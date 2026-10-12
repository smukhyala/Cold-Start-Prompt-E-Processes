"""Retrospective live-repeat and session-drift audit; see OWN_REPEAT_DRIFT_PROTOCOL.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from experiments.growing_bandits.audit import known_baseline as kb
from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths
from experiments.growing_bandits.audit.public_replay_01 import e1_path, first_cross
from experiments.growing_bandits.retrospective.analyze_headroom_gmail import leg_m_default_scores

OUT = Path("results/paper/own_repeat_drift")
POOLS = ("G", "F", "GLK30", "GLK")


def repeat_audit():
    rows = []
    for pi, pool in enumerate(POOLS):
        df = pd.read_csv(f"results/audit/matrices/{pool}.csv")
        piv = df.pivot(index=["task_id", "arm_id"], columns="replicate", values="success").dropna(subset=[0, 1])
        if not np.isin(piv[[0, 1]].to_numpy(), [0, 1]).all():
            raise ValueError(f"Nonbinary repeat: {pool}")
        y = piv[[0, 1]].to_numpy(dtype=np.int8).T
        n = y.shape[1]
        orders = np.empty((2000, 2, n), dtype=np.int8)
        for b in range(2000):
            rng = np.random.default_rng(np.random.SeedSequence([20261012, 11, pi, b]))
            swap = rng.integers(0, 2, n).astype(bool)
            z = y.copy()
            z[:, swap] = z[::-1, swap]
            orders[b] = z[:, rng.permutation(n)]
        paths = e1b_log_paths(orders, np.zeros(2000, dtype=int),
                             np.tile(np.array([[False, True]]), (2000, 1)))
        hits = (paths >= np.log(20)).any(axis=1)
        ci = binomtest(int(hits.sum()), 2000).proportion_ci(confidence_level=0.95)
        rows.append({"pool": pool, "paired_cells": n,
                     "arm_count": int(piv.index.get_level_values("arm_id").nunique()),
                     "task_count": int(piv.index.get_level_values("task_id").nunique()),
                     "discordance": int((y[0] != y[1]).sum()),
                     "crossings": int(hits.sum()), "replays": 2000,
                     "crossing_rate": float(hits.mean()), "binomial_ci": [ci.low, ci.high],
                     "unswapped_cross": [first_cross(e1_path(y)), first_cross(e1_path(y[::-1]))]})
    return rows


def known_path(y, baseline):
    x = kb.known_baseline_x(y[None, None, :].astype(float), baseline[None, None, :])
    return kb.efficacy_arm_log_paths(x)[0, 0]


def drift_audit():
    table = pd.read_csv("results/headroom_gmail/task_table.csv").set_index("task_id")
    old = leg_m_default_scores()
    if old is None:
        raise ValueError("Incomplete Leg-M default bank")
    tasks = sorted(set(table.index) & set(old))
    if len(tasks) != 30:
        raise ValueError(f"Expected 30 matched tasks, got {len(tasks)}")
    current = table.loc[tasks]
    bank = np.array([old[t] for t in tasks], dtype=float)
    shifted = np.minimum(bank + 0.15, 1.0)
    cp = kb.cp_upper(np.rint(2*bank).astype(int), 2, 0.025/30)
    D = current[["D_r0", "D_r1"]].to_numpy(dtype=np.int8).T
    O = current[["O_r0", "O_r1"]].to_numpy(dtype=np.int8).T
    scenarios = {
        "current_D_vs_historical_plugin_r0": lambda p: known_path(D[0, p], bank[p]),
        "current_D_vs_historical_plugin_r1": lambda p: known_path(D[1, p], bank[p]),
        "concurrent_D_AA": lambda p: e1_path(D[:, p]),
        "current_O_vs_historical_plugin_r0": lambda p: known_path(O[0, p], bank[p]),
        "current_O_vs_historical_plugin_r1": lambda p: known_path(O[1, p], bank[p]),
        "concurrent_O_vs_D_r0": lambda p: e1_path(np.stack([D[0, p], O[0, p]])),
        "concurrent_O_vs_D_r1": lambda p: e1_path(np.stack([D[1, p], O[1, p]])),
        "current_D_vs_shifted_r0": lambda p: known_path(D[0, p], shifted[p]),
        "current_D_vs_shifted_r1": lambda p: known_path(D[1, p], shifted[p]),
        "current_O_vs_shifted_r0": lambda p: known_path(O[0, p], shifted[p]),
        "current_O_vs_shifted_r1": lambda p: known_path(O[1, p], shifted[p]),
        "current_D_vs_cp_r0": lambda p: known_path(D[0, p], cp[p]),
        "current_D_vs_cp_r1": lambda p: known_path(D[1, p], cp[p]),
        "current_O_vs_cp_r0": lambda p: known_path(O[0, p], cp[p]),
        "current_O_vs_cp_r1": lambda p: known_path(O[1, p], cp[p]),
    }
    main = []
    replays = []
    identity = np.arange(len(tasks))
    for name, run in scenarios.items():
        path = run(identity)
        main.append({"scenario": name, "cross_task": first_cross(path),
                     "max_e": float(np.exp(path.max())), "final_e": float(np.exp(path[-1]))})
        crosses = []
        for b in range(2000):
            order = np.random.default_rng(np.random.SeedSequence([20261012, 12, b])).permutation(len(tasks))
            crosses.append(first_cross(run(order)))
        replays.append({"scenario": name, "crossings": sum(v>0 for v in crosses),
                        "replays": len(crosses), "max_cross_task": max(crosses)})
    summary = {"matched_tasks": len(tasks), "historical_D_mean": float(bank.mean()),
               "current_D_mean": float(D.mean()), "current_O_mean": float(O.mean()),
               "default_drift": float(D.mean()-bank.mean()),
               "O_gain_vs_historical": float(O.mean()-bank.mean()),
               "O_gain_vs_concurrent": float(O.mean()-D.mean()),
               "shifted_bank_mean": float(shifted.mean()), "cp_bank_mean": float(cp.mean())}
    return summary, main, replays


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    repeats = repeat_audit()
    drift, paths, replays = drift_audit()
    data = {"repeat_audit": repeats, "drift_summary": drift,
            "drift_fixed_order": paths, "drift_order_replays": replays}
    (OUT / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
