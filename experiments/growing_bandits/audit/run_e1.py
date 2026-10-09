#!/usr/bin/env python3
"""EXPLORATORY (data already seen): run audit step E1 (replay, validity, power).

No API calls.  Outputs go to results/audit/e1/.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.growing_bandits.audit import flatness_eprocess as fe  # noqa: E402

MAT = ROOT / "results" / "audit" / "matrices"
OUT = ROOT / "results" / "audit" / "e1"
ALPHAS = (0.05, 0.10)


def load_matrix(pool: str):
    df = pd.read_csv(MAT / f"{pool}.csv")
    df = df[df["replicate"] == 0]
    wide = df.pivot(index="arm_id", columns="task_id", values="success")
    return wide


def build_pools():
    """Returns list of (pool, variant, Y_pool (K,n), arm_ids, baseline_row (n,) or None)."""
    out = []
    anchor = load_matrix("gmail_anchor")
    for pool in ("G", "F", "GLK30", "GLK", "gitlab18", "gmail_paired12", "shopping_paired12"):
        wide = load_matrix(pool)
        arm_ids = list(wide.index)
        tasks = list(wide.columns)
        if pool in ("G", "F"):
            base = anchor.iloc[0][tasks].to_numpy().astype(int)
            out.append((pool, "full", wide.to_numpy().astype(int), arm_ids, base))
        elif pool in ("GLK30", "GLK"):
            out.append((pool, "full", wide.to_numpy().astype(int), arm_ids, None))
        elif pool == "gitlab18":
            drops = {
                "all18": [],
                "no_oracle": ["gitlab_oracle_operator"],
                "no_oracle_no_explorer": ["gitlab_oracle_operator", "explorer"],
            }
            for variant, drop in drops.items():
                keep = []
                for a in arm_ids:
                    if a not in drop:
                        keep.append(a)
                sub = wide.loc[keep]
                out.append((pool, variant, sub.to_numpy().astype(int), keep, None))
        else:
            out.append((pool, "full", wide.to_numpy().astype(int), arm_ids, None))
    return out


def permute_tasks(Y: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    """Y (A,n) -> (B,A,n) with an independent random task order per replay."""
    A, n = Y.shape
    out = np.zeros((B, A, n), dtype=np.int8)
    for b in range(B):
        out[b] = Y[:, rng.permutation(n)]
    return out


def e1b_random_incumbent(Yb: np.ndarray, rng: np.random.Generator):
    B, A, n = Yb.shape
    inc = rng.integers(0, A, size=B)
    chal = np.ones((B, A), dtype=bool)
    chal[np.arange(B), inc] = False
    return fe.e1b_log_paths(Yb, inc, chal)


def row_stats(pool, variant, test, incumbent, paths, K, rows, order_rows, tag):
    n = paths.shape[1]
    for alpha in ALPHAS:
        rej, stop, fin, mx = fe.summarize_paths(paths, alpha)
        if rej.any():
            med_rej = float(np.median(stop[rej]))
        else:
            med_rej = float("nan")
        rows.append({
            "pool": pool, "variant": variant, "test": test, "incumbent": incumbent,
            "alpha": alpha, "reject_rate": float(rej.mean()),
            "median_stop": float(np.median(stop)),
            "median_stop_if_reject": med_rej,
            "median_episodes_if_reject": med_rej * K if rej.any() else float("nan"),
            "mean_final_logE": float(fin.mean()), "mean_max_logE": float(mx.mean()),
            "n_tasks": n, "K": K, "n_orders": paths.shape[0],
        })
        for i in range(paths.shape[0]):
            order_rows.append({
                "pool": pool, "variant": variant, "test": test, "incumbent": incumbent,
                "alpha": alpha, "order": i, "rejected": int(rej[i]), "stop": int(stop[i]),
                "final_logE": float(fin[i]), "max_logE": float(mx[i]),
            })


def stage_replay(n_orders: int, seed: int):
    rows = []
    order_rows = []
    for pool, variant, Y, arm_ids, base in build_pools():
        rng = np.random.default_rng(seed)
        K, n = Y.shape
        Yb = permute_tasks(Y, rng, n_orders)
        paths = fe.e1a_log_paths(Yb)
        row_stats(pool, variant, "E1a", "none", paths, K, rows, order_rows, "")
        # (ii) random incumbent
        paths = e1b_random_incumbent(Yb, rng)
        row_stats(pool, variant, "E1b", "random", paths, K, rows, order_rows, "")
        # (i) baseline incumbent
        if pool in ("G", "F"):
            Ya = np.concatenate([Y, base[None, :]], axis=0)  # baseline is last row
            Yab = permute_tasks(Ya, rng, n_orders)
            A = Ya.shape[0]
            inc = np.full(n_orders, A - 1)
            chal = np.ones((n_orders, A), dtype=bool)
            chal[:, A - 1] = False
            paths = fe.e1b_log_paths(Yab, inc, chal)
            row_stats(pool, variant, "E1b", "baseline", paths, K + 1, rows, order_rows, "")
        elif "baseline" in arm_ids:
            b = arm_ids.index("baseline")
            inc = np.full(n_orders, b)
            chal = np.ones((n_orders, K), dtype=bool)
            chal[:, b] = False
            paths = fe.e1b_log_paths(Yb, inc, chal)
            row_stats(pool, variant, "E1b", "baseline", paths, K, rows, order_rows, "")
        print("replay done", pool, variant, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "replay_summary.csv", index=False)
    pd.DataFrame(order_rows).to_csv(OUT / "replay_orders.csv", index=False)


def null_v1(Y: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    """Keep per-task success counts, redistribute successes uniformly over arms."""
    K, n = Y.shape
    S = Y.sum(axis=0)
    out = np.zeros((B, K, n), dtype=np.int8)
    for b in range(B):
        perm = rng.permutation(n)
        u = rng.random((K, n))
        ranks = u.argsort(axis=0).argsort(axis=0)
        out[b] = (ranks < S[None, :]).astype(np.int8)[:, perm]
    return out


def null_v2(Y: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    K, n = Y.shape
    p = Y.mean(axis=0)
    out = np.zeros((B, K, n), dtype=np.int8)
    for b in range(B):
        perm = rng.permutation(n)
        out[b] = (rng.random((K, n)) < p[None, :]).astype(np.int8)[:, perm]
    return out


def stage_validity(n_sims: int, seed: int):
    rows = []
    for pool, variant, Y, arm_ids, base in build_pools():
        K, n = Y.shape
        for null_name, gen in (("V1_exact_counts", null_v1), ("V2_iid_task_p", null_v2)):
            rng = np.random.default_rng(seed)
            Yb = gen(Y, rng, n_sims)
            for test, paths in (("E1a", fe.e1a_log_paths(Yb)),
                                ("E1b", e1b_random_incumbent(Yb, rng))):
                for alpha in ALPHAS:
                    rej, stop, fin, mx = fe.summarize_paths(paths, alpha)
                    rate = float(rej.mean())
                    se = math.sqrt(alpha * (1 - alpha) / n_sims)
                    rows.append({
                        "pool": pool, "variant": variant, "null": null_name,
                        "test": test, "alpha": alpha, "reject_rate": rate,
                        "mc_se": se, "n_sims": n_sims,
                        "violation_3se": int(rate > alpha + 3 * se),
                        "mean_max_logE": float(mx.mean()),
                    })
        print("validity done", pool, variant, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "validity.csv", index=False)


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------
def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_logit_sd(task_p: np.ndarray, tau: float) -> float:
    """Logit sd s so that sd_k of mean_t expit(logit p_t + delta_k) ~= tau."""
    if tau == 0.0:
        return 0.0
    nodes, wts = np.polynomial.hermite_e.hermegauss(41)
    wts = wts / wts.sum()
    tp = np.clip(task_p, 0.02, 0.98)  # same clip as gen_power
    lp = np.log(tp) - np.log1p(-tp)

    def sd_of(s):
        g = np.zeros(len(nodes))
        for i in range(len(nodes)):
            g[i] = expit(lp + s * nodes[i]).mean()
        m = (wts * g).sum()
        return math.sqrt((wts * (g - m) ** 2).sum())

    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sd_of(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gen_power(task_pool: np.ndarray, K: int, n: int, s: float, tail: bool,
              rng: np.random.Generator, B: int) -> np.ndarray:
    Y = np.zeros((B, K, n), dtype=np.int8)
    lo_clip = 0.02
    for b in range(B):
        p_t = rng.choice(task_pool, size=n, replace=True)
        p_t = np.clip(p_t, lo_clip, 1 - lo_clip)
        if tail:
            P = np.tile(p_t[None, :], (K, 1))
            P[0, :] = np.clip(p_t + 0.3, 0.01, 0.99)
        else:
            lp = np.log(p_t) - np.log1p(-p_t)
            delta = rng.normal(0.0, s, size=K)
            P = expit(lp[None, :] + delta[:, None])
        Y[b] = (rng.random((K, n)) < P).astype(np.int8)
    return Y


def stage_power(n_sims: int, seed: int):
    G = load_matrix("G").to_numpy().astype(float)
    task_means = G.mean(axis=0)
    taus = (0.0, 0.02, 0.035, 0.05, 0.08, 0.12)
    rows = []
    for K in (12, 40):
        for n in (30, 60, 120, 240):
            scen = []
            for tau in taus:
                scen.append((f"tau={tau}", tau, calibrate_logit_sd(task_means, tau), False))
            scen.append(("tail+0.3", float("nan"), 0.0, True))
            for name, tau, s, tail in scen:
                rng = np.random.default_rng(seed)
                Yb = gen_power(task_means, K, n, s, tail, rng, n_sims)
                res = (("E1a", fe.e1a_log_paths(Yb)),
                       ("E1b", e1b_random_incumbent(Yb, rng)))
                for test, paths in res:
                    for alpha in ALPHAS:
                        rej, stop, fin, mx = fe.summarize_paths(paths, alpha)
                        if rej.any():
                            med_rej = float(np.median(stop[rej]))
                        else:
                            med_rej = float("nan")
                        rows.append({
                            "K": K, "n": n, "scenario": name, "tau": tau,
                            "logit_sd": s, "test": test, "alpha": alpha,
                            "reject_rate": float(rej.mean()),
                            "median_stop": float(np.median(stop)),
                            "median_stop_if_reject": med_rej, "n_sims": n_sims,
                        })
            print("power done", K, n, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "power.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["replay", "validity", "power", "all"], default="all")
    ap.add_argument("--orders", type=int, default=500)
    ap.add_argument("--val-sims", type=int, default=1000)
    ap.add_argument("--power-sims", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("replay", "all"):
        stage_replay(args.orders, args.seed)
    if args.stage in ("validity", "all"):
        stage_validity(args.val_sims, args.seed + 1)
    if args.stage in ("power", "all"):
        stage_power(args.power_sims, args.seed + 2)


if __name__ == "__main__":
    main()
