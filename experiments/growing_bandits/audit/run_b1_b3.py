#!/usr/bin/env python3
"""EXPLORATORY (data already seen): B1 adaptive efficacy, B3 adaptive futility, finite-bank replay.

No API calls.  Outputs go to results/audit/b/ (b1_sim.csv, b3_sim.csv, b13_replay.csv, b1_episodes.png).

    ./.venv/bin/python experiments/growing_bandits/audit/run_b1_b3.py [--sims-b1 500] [--sims-b3 300]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.growing_bandits.audit import adaptive_tests as ad  # noqa: E402

OUT = ROOT / "results" / "audit" / "b"
ALPHA = 0.05
KS = (12, 40)
B1_CAP = 12000  # run to 12000 episodes; the headline budget of 3000 is a column
B1_BUDGETS = (500, 1000, 2000, 3000, 6000, 12000)
B3_CAP = 24000  # headline budget 6000 is a column
B3_BUDGETS = (2000, 4000, 6000, 12000, 24000)
EXPLORATORY = "EXPLORATORY"
BOUNDARY_SIMS = 1500  # B3 validity cells get more simulations (rates near alpha need small SE)


def b1_scenarios():
    out = []
    out.append(("null", 0.0))
    for d in (0.10, 0.15, 0.20):
        out.append(("good1", d))
    out.append(("good3", 0.10))
    out.append(("below", -0.15))
    return out


def b1_cell(job):
    rule, K, scenario, delta, sims, seed = job
    t0 = time.time()
    q = ad.load_task_difficulty()
    rng = np.random.default_rng(seed)
    shifts, good = ad.build_shifts(scenario, K, sims, delta, rng)
    res = ad.simulate_b1(rule, K, shifts, q, B1_CAP, ALPHA, rng)
    ep = res["ep_rej"]
    row = {"status": EXPLORATORY, "rule": rule, "K": K, "scenario": "allnull" if scenario == "null" else scenario,
           "delta": delta, "eff_gain": ad.mean_gain(q, delta, 0.01, 0.99) if scenario.startswith("good") else
           (ad.mean_gain(q, -0.15, 0.01, 0.99) * 0.5 if scenario == "below" else 0.0),
           "sims": sims}
    for b in B1_BUDGETS:
        row[f"rej_{b}"] = ad.rate_at(ep, b)
    finite = ep[np.isfinite(ep)]
    row["median_ep_all"] = float(np.median(ep))  # inf if fewer than half rejected by the cap
    row["median_ep_rejected"] = float(np.median(finite)) if len(finite) > 0 else float("nan")
    per_step = (K + 1) if rule == "audit" else 2
    row["median_tasks_all"] = row["median_ep_all"] / per_step
    # correct arm at rejection
    n_correct = 0
    n_rej = 0
    for b in range(sims):
        if np.isfinite(ep[b]):
            n_rej += 1
            if good[b, res["best_arm"][b]]:
                n_correct += 1
    row["correct_arm_frac"] = (n_correct / n_rej) if (n_rej > 0 and scenario.startswith("good")) else float("nan")
    row["secs"] = time.time() - t0
    return row


def b3_scenarios():
    out = []
    for d in (0.10, 0.15):
        out.append(("flat", d))
        out.append(("reallike", d))
        out.append(("boundary", d))
    return out


def b3_cell(job):
    rule, K, scenario, delta, sims, seed = job
    if scenario == "boundary":
        sims = BOUNDARY_SIMS
    t0 = time.time()
    q = ad.load_task_difficulty()
    rng = np.random.default_rng(seed)
    kind = {"flat": "null", "reallike": "below", "boundary": "boundary"}[scenario]
    shifts, _ = ad.build_shifts(kind, K, sims, delta, rng, q_grid=q)
    res = ad.simulate_b3(rule, K, shifts, q, delta, B3_CAP, ALPHA, rng)
    ep = res["ep_cert"]
    row = {"status": EXPLORATORY, "rule": rule, "K": K, "scenario": scenario, "delta": delta, "sims": sims}
    for b in B3_BUDGETS:
        row[f"cert_{b}"] = ad.rate_at(ep, b)
    row["median_ep_all"] = float(np.median(ep))
    row["median_tasks_all"] = float(np.median(res["tasks_cert"]))
    row["secs"] = time.time() - t0
    return row


def replay_job(job):
    pool, rule, kind, delta, n_orders, seed = job
    rng = np.random.default_rng(seed)
    ids, Yc, y_inc = ad.load_bank(pool)
    K, n = Yc.shape
    orders = np.zeros((n_orders, n), dtype=int)
    for b in range(n_orders):
        orders[b] = rng.permutation(n)
    row = {"status": EXPLORATORY, "pool": pool, "K": K, "bank_tasks": n, "test": kind, "rule": rule,
           "delta": delta, "orders": n_orders}
    if kind == "B1":
        res = ad.replay_b1(rule, Yc, y_inc, orders, ALPHA, rng)
        ep = res["ep_rej"]
        row["reject_rate"] = ad.rate_finite(ep)
        row["median_ep_used"] = float(np.median(res["ep_used"]))
        row["median_tasks_used"] = float(np.median(res["tasks_used"]))
        fb = res["final_best"]
        oracle = ids.index("gitlab_oracle_operator") if "gitlab_oracle_operator" in ids else -1
        if oracle >= 0:
            hits = 0
            for b in range(n_orders):
                if fb[b] == oracle:
                    hits += 1
            row["oracle_top_capital_frac"] = hits / n_orders
        # max global log-e is not tracked; report final-best-arm only
    else:
        res = ad.replay_b3(rule, Yc, y_inc, orders, delta, ALPHA, rng)
        ep = res["ep_cert"]
        row["reject_rate"] = ad.rate_finite(ep)  # = certification rate
        row["median_ep_used"] = float(np.median(res["ep_used"]))
        row["median_tasks_used"] = float(np.median(res["steps_used"]))
    return row


def make_figure(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = "#0e0f12"
    fg = "#c9ccd1"
    muted = "#6b7078"
    palette = {"rr": "#7b8794", "greedy": "#8fb996", "ucb": "#d1a36b", "thompson": "#8aa9d6",
               "audit": "#c97b84"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    fig.patch.set_facecolor(bg)
    for ax, K in zip(axes, KS):
        ax.set_facecolor(bg)
        for rule in ("rr", "greedy", "ucb", "thompson", "audit"):
            xs = []
            ys = []
            cens = []
            for d in (0.10, 0.15, 0.20):
                sub = df[(df["rule"] == rule) & (df["K"] == K) & (df["scenario"] == "good1")
                         & (np.isclose(df["delta"], d))]
                v = float(sub["median_ep_all"].iloc[0])
                xs.append(d)
                if math.isfinite(v):
                    ys.append(v)
                    cens.append(False)
                else:
                    ys.append(float(B1_CAP))
                    cens.append(True)
            ax.plot(xs, ys, color=palette[rule], lw=1.0, label=rule)
            for i in range(len(xs)):
                if cens[i]:
                    ax.plot(xs[i], ys[i], marker="v", ms=4, mfc=bg, mec=palette[rule], lw=0)
                else:
                    ax.plot(xs[i], ys[i], marker="o", ms=2.5, color=palette[rule], lw=0)
        ax.set_yscale("log")
        ax.set_xticks([0.10, 0.15, 0.20])
        ax.set_title(f"K = {K}", color=fg, fontsize=9, loc="left")
        ax.set_xlabel("true gain of the good arm", color=muted, fontsize=8)
        ax.tick_params(colors=muted, labelsize=8, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#2a2d33")
        ax.grid(axis="y", color="#1c1f24", lw=0.5)
    axes[0].set_ylabel("median episodes to reject", color=muted, fontsize=8)
    axes[1].legend(frameon=False, fontsize=7.5, labelcolor=fg, loc="upper right")
    fig.suptitle("EXPLORATORY  episodes to reject H0 (one good arm); hollow triangle = not reached by 12000",
                 color=muted, fontsize=7.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "b1_episodes.png", dpi=170, facecolor=bg)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims-b1", type=int, default=500)
    ap.add_argument("--sims-b3", type=int, default=300)
    ap.add_argument("--orders", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--skip", default="", help="comma list of b1,b3,replay")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    skip = args.skip.split(",")
    t0 = time.time()

    if "b1" not in skip:
        jobs = []
        i = 0
        for K in KS:
            for scenario, delta in b1_scenarios():
                for rule in ad.B1_RULES:
                    i += 1
                    jobs.append((rule, K, scenario, delta, args.sims_b1, args.seed + i))
        with Pool(args.workers) as pool:
            rows = pool.map(b1_cell, jobs, chunksize=1)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "b1_sim.csv", index=False)
        make_figure(df)
        print("b1 done", round(time.time() - t0, 1), "s")

    if "b3" not in skip:
        jobs = []
        i = 1000
        for K in KS:
            for scenario, delta in b3_scenarios():
                for rule in ad.B3_RULES:
                    i += 1
                    jobs.append((rule, K, scenario, delta, args.sims_b3, args.seed + i))
        with Pool(args.workers) as pool:
            rows = pool.map(b3_cell, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(OUT / "b3_sim.csv", index=False)
        print("b3 done", round(time.time() - t0, 1), "s")

    if "replay" not in skip:
        jobs = []
        i = 2000
        for pool_name in ("gitlab18", "shopping_paired12", "gmail_paired12"):
            for rule in ("greedy", "audit"):
                i += 1
                jobs.append((pool_name, rule, "B1", float("nan"), args.orders, args.seed + i))
            for delta in (0.10, 0.15):
                for rule in ("audit", "drop", "focus"):
                    i += 1
                    jobs.append((pool_name, rule, "B3", delta, args.orders, args.seed + i))
        with Pool(args.workers) as pool:
            rows = pool.map(replay_job, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(OUT / "b13_replay.csv", index=False)
        print("replay done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
