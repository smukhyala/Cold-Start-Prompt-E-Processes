#!/usr/bin/env python3
"""EXPLORATORY (data already seen): B2 known-incumbent test and B4 finite-bank validity.

No API calls.  Outputs go to results/audit/b/ (b2_sim.csv, b2_estimated.csv, b2_variance.csv,
b4_finite.csv).

    ./.venv/bin/python experiments/growing_bandits/audit/run_b2_b4.py --which all
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

from scipy.special import logsumexp  # noqa: E402

from experiments.growing_bandits.audit import flatness_eprocess as fe  # noqa: E402
from experiments.growing_bandits.audit import futility_eprocess as fu  # noqa: E402
from experiments.growing_bandits.audit import known_baseline as kb  # noqa: E402
from experiments.growing_bandits.audit.incumbent_gain import (  # noqa: E402
    VARIANTS,
    primary_matrix,
    split_arms_and_incumbent,
)

OUT = ROOT / "results" / "audit" / "b"
MATRICES = ROOT / "results" / "audit" / "matrices"
PILOT = (ROOT / "results" / "growing_bandits" / "retrospective" / "first_run_pilot_gmail"
         / "task_outcomes.csv")
ALPHA = 0.05
CAP = 400
CHECKPOINTS = (50, 100, 200, 400)
LABEL = "EXPLORATORY"


def g_task_means() -> np.ndarray:
    Y = primary_matrix("G", ())
    return Y.to_numpy(dtype=float).mean(axis=0)


def arm_probs(q: np.ndarray, deltas, boundary_shift_arm0=None) -> np.ndarray:
    """Per-arm per-task success probabilities (K, T).  Delta == 0 gives exactly q."""
    K = len(deltas)
    P = np.zeros((K, len(q)))
    for k in range(K):
        if deltas[k] == 0.0:
            P[k, :] = q
        else:
            P[k, :] = np.clip(q + deltas[k], 0.01, 0.99)
    if boundary_shift_arm0 is not None:
        P[0, :] = np.minimum(q + boundary_shift_arm0, 1.0)
    return P


def sample_stream(q: np.ndarray, P: np.ndarray, B: int, n: int, rng: np.random.Generator):
    T = len(q)
    K = P.shape[0]
    idx = rng.integers(0, T, size=(B, n))
    yb = (rng.random((B, n)) < q[idx]).astype(np.int8)
    yk = np.zeros((B, K, n), dtype=np.int8)
    for k in range(K):
        yk[:, k, :] = (rng.random((B, n)) < P[k][idx]).astype(np.int8)
    return idx, yb, yk


def summarize_cross(cross: np.ndarray) -> dict:
    out = {}
    for cp in CHECKPOINTS:
        out[f"power_at_{cp}"] = float(np.mean((cross > 0) & (cross <= cp)))
    out["p_cap"] = float(np.mean(cross > 0))
    done = cross[cross > 0]
    if len(done) > 0:
        out["median_tasks_if_cross"] = float(np.median(done))
    else:
        out["median_tasks_if_cross"] = float("nan")
    out["tasks_q80"] = kb.tasks_quantile(cross, 0.8)
    return out


# ----------------------------------------------------------------------------
# B2 (i): simulation, known baseline vs paired
# ----------------------------------------------------------------------------
def b2_sim(sims: int, seed: int, chunk: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = g_task_means()
    rows = []
    thr = 1.0 / ALPHA
    for K in (12, 40):
        eff_scen = [("all_zero_null", [0.0] * K, None, 0.0)]
        for d in (0.10, 0.15, 0.20):
            deltas = [d] + [0.0] * (K - 1)
            eff_scen.append((f"good_arm_{d:.2f}", deltas, None, d))
        half = [-0.15] * (K // 2) + [0.0] * (K - K // 2)
        eff_scen.append(("reallike_half_-0.15", half, None, 0.0))
        for name, deltas, bshift, param in eff_scen:
            P = arm_probs(q, deltas, bshift)
            cr = {"known": [], "known_cap0.9": [], "paired_E1b": []}
            done = 0
            while done < sims:
                S = min(chunk, sims - done)
                idx, yb, yk = sample_stream(q, P, S, CAP, rng)
                X = kb.known_baseline_x(yk.astype(float), q[idx][:, None, :])
                lp = kb.efficacy_pool_log_paths(kb.efficacy_arm_log_paths(X))
                cr["known"].append(kb.first_cross_1based(lp, thr))
                lp2 = kb.efficacy_pool_log_paths(kb.efficacy_arm_log_paths(X, cap_scale=0.9))
                cr["known_cap0.9"].append(kb.first_cross_1based(lp2, thr))
                Y = np.concatenate([yb[:, None, :], yk], axis=1)
                chal = np.ones((S, K + 1), dtype=bool)
                chal[:, 0] = False
                inc = np.zeros(S, dtype=int)
                lp3 = fe.e1b_log_paths(Y, inc, chal)
                cr["paired_E1b"].append(kb.first_cross_1based(lp3, thr))
                done += S
            for design in cr:
                cross = np.concatenate(cr[design])
                summ = summarize_cross(cross)
                per_task = K if design != "paired_E1b" else K + 1
                rows.append({
                    "label": LABEL, "test": "efficacy", "design": design, "K": K, "scenario": name,
                    "param": param, "is_null": bool(name in ("all_zero_null", "reallike_half_-0.15")),
                    "sims": sims, "alpha": ALPHA, "cap_tasks": CAP, "episodes_per_task": per_task,
                    **summ, "episodes_q80": summ["tasks_q80"] * per_task,
                    "mc_se_p_cap": math.sqrt(max(summ["p_cap"] * (1 - summ["p_cap"]), 1e-12) / sims),
                })
            print(f"b2 sim efficacy K={K} {name}", flush=True)

        for delta in (0.10, 0.15):
            shift = fu.boundary_shift(q, delta)
            fut_scen = [("flat", [0.0] * K, None, True)]
            half = [-0.15] * (K // 2) + [0.0] * (K - K // 2)
            fut_scen.append(("reallike_half_-0.15", half, None, True))
            fut_scen.append(("boundary_arm0", [0.0] * K, shift, False))
            for name, deltas, bshift, _ in fut_scen:
                P = arm_probs(q, deltas, bshift)
                cr = {"known": [], "paired_E5": []}
                done = 0
                while done < sims:
                    S = min(chunk, sims - done)
                    idx, yb, yk = sample_stream(q, P, S, CAP, rng)
                    Dk = yk.astype(float) - q[idx][:, None, :]
                    lp = kb.futility_pool_log_paths(Dk, delta)
                    cr["known"].append(kb.first_cross_1based(lp, thr))
                    Dp = yk - yb[:, None, :]
                    lp2 = fu.pool_log_paths(fu.futility_log_paths(Dp, delta))
                    cr["paired_E5"].append(kb.first_cross_1based(lp2, thr))
                    done += S
                for design in cr:
                    cross = np.concatenate(cr[design])
                    summ = summarize_cross(cross)
                    per_task = K if design == "known" else K + 1
                    rows.append({
                        "label": LABEL, "test": "futility", "design": design, "K": K,
                        "scenario": name, "param": delta, "is_null": bool(name == "boundary_arm0"),
                        "sims": sims, "alpha": ALPHA, "cap_tasks": CAP, "episodes_per_task": per_task,
                        **summ, "episodes_q80": summ["tasks_q80"] * per_task,
                        "mc_se_p_cap": math.sqrt(max(summ["p_cap"] * (1 - summ["p_cap"]), 1e-12) / sims),
                    })
                print(f"b2 sim futility K={K} delta={delta} {name}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# B2 (ii): estimated baseline
# ----------------------------------------------------------------------------
def run_estimated_case(q, P, test, delta, method, m, regime, sims, rng, chunk=250):
    """Returns (cross array, offset sd).  Bank regime: b_hat fixed per bank task; fresh: new per step."""
    N = len(q)
    n = CAP
    a_level = ALPHA / (2.0 * N)
    crosses = []
    offsets = []
    done = 0
    while done < sims:
        S = min(chunk, sims - done)
        idx, yb, yk = sample_stream(q, P, S, n, rng)
        yk_f = yk.astype(float)
        if method == "known":
            b_used = q[idx]
            thr = 1.0 / ALPHA
        else:
            if regime == "bank":
                s_bank = rng.binomial(m, q[None, :], size=(S, N)).astype(float)
                offsets.append((q[None, :] - s_bank / m).mean(axis=1))
                if method == "plugin":
                    bank_b = s_bank / m
                    thr = 1.0 / ALPHA
                elif method == "conservative_bank_shift":
                    bank_b = s_bank / m
                    thr = 2.0 / ALPHA
                elif test == "efficacy":
                    bank_b = kb.cp_upper(s_bank, m, a_level)
                    thr = 2.0 / ALPHA
                else:
                    bank_b = kb.cp_lower(s_bank, m, a_level)
                    thr = 2.0 / ALPHA
                b_used = np.take_along_axis(bank_b, idx, axis=1)
            else:
                s_step = rng.binomial(m, q[idx]).astype(float)
                b_used = s_step / m
                thr = 1.0 / ALPHA
        eps_shift = 0.0
        if method == "conservative_bank_shift":
            eps_shift = math.sqrt(math.log(2.0 / ALPHA) / (2.0 * m * N))
        if test == "efficacy":
            X = kb.known_baseline_x(yk_f, b_used[:, None, :])
            lp = kb.efficacy_pool_log_paths(kb.efficacy_arm_log_paths(X, m0=(1.0 + eps_shift) / 2.0))
        else:
            Dk = yk_f - b_used[:, None, :]
            lp = kb.futility_pool_log_paths(Dk, delta - eps_shift)
        crosses.append(kb.first_cross_1based(lp, thr))
        done += S
    sd = float(np.std(np.concatenate(offsets))) if len(offsets) > 0 else float("nan")
    return np.concatenate(crosses), sd


def b2_estimated(sims: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = g_task_means()
    K = 12
    delta = 0.15
    shift = fu.boundary_shift(q, delta)
    cases = []
    cases.append(("efficacy", "all_zero_null", arm_probs(q, [0.0] * K), True, 0.0))
    cases.append(("efficacy", "good_arm_0.15", arm_probs(q, [0.15] + [0.0] * (K - 1)), False, 0.15))
    cases.append(("futility", "boundary_arm0_delta0.15",
                  arm_probs(q, [0.0] * K, shift), True, delta))
    cases.append(("futility", "flat_delta0.15", arm_probs(q, [0.0] * K), False, delta))
    rows = []
    for regime in ("bank", "fresh"):
        for test, name, P, is_null, param in cases:
            methods = [("known", 0)]
            for m in (1, 3, 10, 30, 100):
                methods.append(("plugin", m))
                if regime == "bank":
                    methods.append(("conservative_cp", m))
                    methods.append(("conservative_bank_shift", m))
            for method, m in methods:
                if regime == "fresh" and method == "known":
                    continue
                dlt = delta if test == "futility" else 0.0
                cross, sd = run_estimated_case(q, P, test, dlt, method, m, regime, sims, rng)
                summ = summarize_cross(cross)
                p = summ["p_cap"]
                rows.append({
                    "label": LABEL, "regime": regime, "test": test, "scenario": name,
                    "method": method, "m_prior_runs": m, "K": K, "bank_tasks": len(q),
                    "is_null": is_null, "sims": sims,
                    "alpha": ALPHA, "reject_threshold_alpha": ALPHA if method in ("plugin", "known") else ALPHA / 2,
                    "cap_tasks": CAP, "p_cap": p, "mc_se": math.sqrt(max(p * (1 - p), 1e-12) / sims),
                    "power_at_100": summ["power_at_100"], "power_at_200": summ["power_at_200"],
                    "median_tasks_if_cross": summ["median_tasks_if_cross"],
                    "tasks_q80": summ["tasks_q80"], "bank_offset_sd": sd,
                })
                print(f"b2 est {regime} {test} {name} {method} m={m}: {p:.3f}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# B2 (iii): variance decomposition on real data
# ----------------------------------------------------------------------------
def boot_ci(values: np.ndarray, fn, rng, reps=2000):
    n = len(values)
    out = np.zeros(reps)
    for r in range(reps):
        pick = rng.integers(0, n, size=n)
        out[r] = fn(values[pick])
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def b2_variance(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pil = pd.read_csv(PILOT)
    y1 = pil["first_success"].to_numpy(dtype=float)
    y2 = pil["same_success"].to_numpy(dtype=float)
    pair = np.stack([y1, y2], axis=1)  # (n, 2)

    def decompose(pairs: np.ndarray):
        p = float(pairs.mean())
        v_tot = p * (1 - p)
        R = float(np.mean(0.5 * (pairs[:, 0] - pairs[:, 1]) ** 2))
        return v_tot, R

    v_tot, R = decompose(pair)
    v_task = v_tot - R
    lo, hi = boot_ci(pair, lambda x: decompose(x)[1] / max(decompose(x)[0], 1e-9), rng)
    rows.append({
        "label": LABEL, "source": "gmail_pilot_G14_first_vs_rerun", "n_tasks": len(pair),
        "n_arms": 1, "n_repeat_cells": len(pair), "p_hat": float(pair.mean()),
        "var_total": v_tot, "run_noise_R": R, "var_task_difficulty": v_task,
        "var_arm_x_task": float("nan"), "share_run_noise": R / v_tot,
        "share_run_noise_ci90_lo": lo, "share_run_noise_ci90_hi": hi,
        "paired_over_known_var_ratio_no_interaction": 2.0,
        "paired_over_known_var_ratio_with_interaction": float("nan"),
        "note": "A=1: Cov(Y1,Y2)=task-difficulty variance; V_arm_x_task not identifiable",
    })

    for pool in ("G", "F", "GLK30"):
        frame = pd.read_csv(MATRICES / f"{pool}.csv")
        prim = frame[frame["replicate"] == 0]
        rep = frame[frame["replicate"] == 1]
        wide = prim.pivot(index="arm_id", columns="task_id", values="success")
        A, T = wide.shape
        arr = wide.to_numpy(dtype=float)
        p = float(arr.mean())
        v_tot = p * (1 - p)
        ybar_t = arr.mean(axis=0)
        v_task = (A * float(np.var(ybar_t, ddof=1)) - v_tot) / (A - 1.0)
        v_task = max(v_task, 0.0)
        merged = prim.merge(rep, on=["arm_id", "task_id"], suffixes=("_0", "_1"))
        pairs = merged[["success_0", "success_1"]].to_numpy(dtype=float)
        R = float(np.mean(0.5 * (pairs[:, 0] - pairs[:, 1]) ** 2))
        v_int = max(v_tot - v_task - R, 0.0)
        lo, hi = boot_ci(pairs, lambda x: float(np.mean(0.5 * (x[:, 0] - x[:, 1]) ** 2)) / v_tot, rng)
        ratio = (2 * R + 2 * v_int) / (R + 2 * v_int)
        rows.append({
            "label": LABEL, "source": f"{pool}_replicate0_vs_1", "n_tasks": T, "n_arms": A,
            "n_repeat_cells": len(pairs), "p_hat": p, "var_total": v_tot, "run_noise_R": R,
            "var_task_difficulty": v_task, "var_arm_x_task": v_int, "share_run_noise": R / v_tot,
            "share_run_noise_ci90_lo": lo, "share_run_noise_ci90_hi": hi,
            "paired_over_known_var_ratio_no_interaction": 2.0,
            "paired_over_known_var_ratio_with_interaction": ratio,
            "note": "repeat cells may not be a random sample of cells; var_arm_x_task = total - task - R",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# B4: finite bank, random-order replay (sampling without replacement)
# ----------------------------------------------------------------------------
def random_orders(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return np.argsort(rng.random((B, n)), axis=1)


def b4_finite(orders: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    thr = 1.0 / ALPHA
    for label, name, drop in VARIANTS:
        if label == "gitlab16_no_oracle_no_explorer":
            continue
        Y = primary_matrix(name, drop)
        arms, inc = split_arms_and_incumbent(name, Y)
        K, n = arms.shape
        perm = random_orders(n, orders, rng)

        # efficacy: exact-null bank per challenger (label flips, balanced discordant tasks)
        inc_k = np.zeros((K, n))
        chal_k = np.zeros((K, n))
        edited_total = 0
        for k in range(K):
            i2, c2, e = kb.balanced_null_pair(inc, arms[k], rng)
            inc_k[k] = i2
            chal_k[k] = c2
            edited_total = edited_total + e
        pair_y = np.zeros((orders * K, 2, n), dtype=np.int8)
        for k in range(K):
            pair_y[k::K, 0, :] = inc_k[k][perm].astype(np.int8)
            pair_y[k::K, 1, :] = chal_k[k][perm].astype(np.int8)
        chal_mask = np.zeros((orders * K, 2), dtype=bool)
        chal_mask[:, 1] = True
        lp = fe.e1b_log_paths(pair_y, np.zeros(orders * K, dtype=int), chal_mask)
        cross_arm = kb.first_cross_1based(lp, thr).reshape(orders, K)
        lp_arm = lp.reshape(orders, K, n)
        pool_lp = logsumexp(lp_arm, axis=1) - math.log(K)
        cross_pool = kb.first_cross_1based(pool_lp, thr)
        p_pool = float(np.mean(cross_pool > 0))
        p_arm_mean = float(np.mean(cross_arm > 0))
        p_arm_max = float(np.max(np.mean(cross_arm > 0, axis=0)))
        se_arm_max = math.sqrt(max(p_arm_max * (1 - p_arm_max), 1e-12) / orders)
        rows.append({
            "label": LABEL, "pool": label, "test": "E1b_efficacy", "construction": "label_flip_exact_null",
            "n_tasks": n, "n_challengers": K, "delta": 0.0, "delta_eff": 0.0, "n_cells_edited": edited_total,
            "orders": orders, "alpha": ALPHA,
            "p_pool": p_pool, "se_pool": math.sqrt(max(p_pool * (1 - p_pool), 1e-12) / orders),
            "p_single_arm_max": p_arm_max, "se_single_arm_max": se_arm_max,
            "p_single_arm_mean": p_arm_mean,
            "exceeds_alpha_by_3se": bool(p_pool > ALPHA + 3 * math.sqrt(ALPHA * (1 - ALPHA) / orders)
                                         or p_arm_max > ALPHA + 3 * math.sqrt(ALPHA * (1 - ALPHA) / orders)),
        })
        print(f"b4 efficacy {label}: pool {p_pool:.4f} arm_max {p_arm_max:.4f}", flush=True)

        # futility: one challenger with bank-level advantage exactly delta_eff
        gains = arms.mean(axis=1) - inc.mean()
        for delta in (0.10, 0.15):
            target = int(round(delta * n))
            delta_eff = target / n
            sums = (arms - inc[None, :]).sum(axis=1)
            kstar = int(np.argmin(np.abs(sums - target)))
            new_row, edits = kb.edit_to_target_sum(inc, arms[kstar], target, rng)
            arms2 = arms.copy()
            arms2[kstar] = new_row
            D = (arms2 - inc[None, :])
            Dp = D[:, perm]  # (K, orders, n)
            Dp = np.transpose(Dp, (1, 0, 2)).astype(np.int8)
            arm_lp = fu.futility_log_paths(Dp, delta_eff)
            arm_cross = fu.first_crossing(arm_lp, ALPHA)
            pool_cross = fu.first_crossing(fu.pool_log_paths(arm_lp), ALPHA)
            p_pool = float(np.mean(pool_cross >= 0))
            p_k = float(np.mean(arm_cross[:, kstar] >= 0))
            rows.append({
                "label": LABEL, "pool": label, "test": "E5_futility", "construction": "min_cell_edit_to_exact_delta",
                "n_tasks": n, "n_challengers": K, "delta": delta, "delta_eff": delta_eff,
                "n_cells_edited": edits, "orders": orders, "alpha": ALPHA,
                "p_pool": p_pool, "se_pool": math.sqrt(max(p_pool * (1 - p_pool), 1e-12) / orders),
                "p_single_arm_max": p_k, "se_single_arm_max": math.sqrt(max(p_k * (1 - p_k), 1e-12) / orders),
                "p_single_arm_mean": p_k,
                "exceeds_alpha_by_3se": bool(p_pool > ALPHA + 3 * math.sqrt(ALPHA * (1 - ALPHA) / orders)
                                             or p_k > ALPHA + 3 * math.sqrt(ALPHA * (1 - ALPHA) / orders)),
            })
            print(f"b4 futility {label} d={delta}: pool {p_pool:.4f} boundary-arm {p_k:.4f} "
                  f"(edits {edits}, best real gain {gains.max():.3f})", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", default="all", choices=["all", "sim", "est", "var", "b4"])
    parser.add_argument("--sims", type=int, default=500)
    parser.add_argument("--est-sims", type=int, default=1000)
    parser.add_argument("--orders", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.which in ("all", "var"):
        v = b2_variance(seed=4)
        v.to_csv(OUT / "b2_variance.csv", index=False)
        print(v.round(3).to_string(index=False))
    if args.which in ("all", "b4"):
        f = b4_finite(args.orders, seed=3)
        f.to_csv(OUT / "b4_finite.csv", index=False)
    if args.which in ("all", "sim"):
        s = b2_sim(args.sims, seed=1)
        s.to_csv(OUT / "b2_sim.csv", index=False)
    if args.which in ("all", "est"):
        e = b2_estimated(args.est_sims, seed=2)
        e.to_csv(OUT / "b2_estimated.csv", index=False)
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
