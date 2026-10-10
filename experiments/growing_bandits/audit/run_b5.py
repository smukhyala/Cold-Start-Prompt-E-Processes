#!/usr/bin/env python3
"""EXPLORATORY (data already seen): B5 headline table, combined adaptive + known-baseline designs.

No API calls.  Writes results/audit/b/{b5_sim,b5_validity,b5_headline}.csv and b5_headline.md.

    ./.venv/bin/python experiments/growing_bandits/audit/run_b5.py [--sims 300] [--val-sims 1000]
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

from experiments.growing_bandits.audit import adaptive_tests as ad  # noqa: E402
from experiments.growing_bandits.audit import combined_tests as ct  # noqa: E402
from experiments.growing_bandits.audit import futility_eprocess as fu  # noqa: E402

OUT = ROOT / "results" / "audit" / "b"
ALPHA = 0.05
CAP = 24000
KS = (12, 40)
BANK_N = 200
BANK_M = 30
LABEL = "EXPLORATORY"


def cell_row(test, scenario, K, param, design, ep, tasks, sims, extra=None):
    row = {"status": LABEL, "test": test, "scenario": scenario, "K": K, "param": param,
           "design": design, "sims": sims, "median_ep": float(np.median(ep)),
           "median_tasks": float(np.median(tasks)), "decided_frac": ct.decided_frac(ep)}
    p = row["decided_frac"]
    row["mc_se"] = math.sqrt(max(p * (1 - p), 1e-12) / sims)
    if extra is not None:
        for k in extra:
            row[k] = extra[k]
    return row


def efficacy_cells(sims, val_sims, seed):
    q = ad.load_task_difficulty()
    rows = []
    scen = [("good1", 0.10), ("good1", 0.15), ("good1", 0.20), ("null", 0.0), ("below", -0.15)]
    for K in KS:
        for kind, d in scen:
            S = sims
            if kind == "null":
                S = val_sims
            name = "allnull" if kind == "null" else ("reallike_below" if kind == "below" else f"good_arm_{d:.2f}")
            t0 = time.time()
            rng = np.random.default_rng(seed + 17 * K + int(1000 * abs(d)) + len(kind))
            shifts, _ = ad.build_shifts(kind, K, S, d, rng, q_grid=q)
            r = ad.simulate_b1("audit", K, shifts, q, CAP, ALPHA, rng)
            rows.append(cell_row("efficacy", name, K, d, "audit_paired", r["ep_rej"], r["tasks_rej"], S))
            r = ad.simulate_b1("thompson", K, shifts, q, CAP, ALPHA, rng)
            rows.append(cell_row("efficacy", name, K, d, "adaptive_only", r["ep_rej"], r["tasks_rej"], S))
            r = ct.simulate_known_efficacy("all", K, shifts, q, CAP, ALPHA, rng)
            rows.append(cell_row("efficacy", name, K, d, "known_only", r["ep_rej"], r["tasks_rej"], S))
            r = ct.simulate_known_efficacy("thompson", K, shifts, q, CAP, ALPHA, rng)
            rows.append(cell_row("efficacy", name, K, d, "combined_a", r["ep_rej"], r["tasks_rej"], S))
            print(f"efficacy K={K} {name} {time.time() - t0:.0f}s", flush=True)
    return rows


def futility_cells(sims, val_sims, seed):
    q = ad.load_task_difficulty()
    rows = []
    for K in KS:
        for delta in (0.10, 0.15):
            for scen in ("flat", "reallike", "boundary"):
                S = val_sims if scen == "boundary" else sims
                kind = {"flat": "null", "reallike": "below", "boundary": "boundary"}[scen]
                t0 = time.time()
                rng = np.random.default_rng(seed + 31 * K + int(1000 * delta) + len(scen))
                bank_rng = np.random.default_rng(seed + 999)
                bank = ct.make_bank(q, BANK_N, bank_rng)
                shifts, _ = ad.build_shifts(kind, K, S, delta, rng, q_grid=q)
                if not scen == "boundary":
                    r = ad.simulate_b3("audit", K, shifts, q, delta, CAP, ALPHA, rng)
                    rows.append(cell_row("futility", scen, K, delta, "audit_paired", r["ep_cert"], r["tasks_cert"], S))
                    r = ad.simulate_b3("drop", K, shifts, q, delta, CAP, ALPHA, rng)
                    rows.append(cell_row("futility", scen, K, delta, "adaptive_only", r["ep_cert"], r["tasks_cert"], S))
                    r = ct.simulate_known_futility("all", K, shifts, q, delta, CAP, ALPHA, rng)
                    rows.append(cell_row("futility", scen, K, delta, "known_only", r["ep_cert"], r["tasks_cert"], S))
                r = ct.simulate_known_futility("drop", K, shifts, q, delta, CAP, ALPHA, rng)
                rows.append(cell_row("futility", scen, K, delta, "combined_a", r["ep_cert"], r["tasks_cert"], S))
                if scen == "boundary":
                    s_bank = fu.boundary_shift(bank, delta)
                    shifts_b = np.zeros((S, K))
                    for b in range(S):
                        shifts_b[b, rng.integers(K)] = s_bank
                else:
                    shifts_b = shifts
                r = ct.simulate_known_futility("drop", K, shifts_b, bank, delta, CAP, ALPHA, rng,
                                               baseline="bank", m=BANK_M)
                rows.append(cell_row("futility", scen, K, delta, "combined_b", r["ep_cert"], r["tasks_cert"], S,
                                     {"eps_shift": r["eps_shift"]}))
                print(f"futility K={K} d={delta} {scen} {time.time() - t0:.0f}s", flush=True)
    return rows


DESIGNS = ("audit_paired", "adaptive_only", "known_only", "combined_a", "combined_b")


def fmt(v):
    if not math.isfinite(v):
        return ">24000"
    return f"{v:.0f}"


def ratio(a, b):
    if math.isfinite(a) and math.isfinite(b) and b > 0:
        return a / b
    return float("nan")


def build_headline(sim: pd.DataFrame) -> pd.DataFrame:
    spec = []
    for d in (0.10, 0.15, 0.20):
        spec.append(("efficacy", f"good_arm_{d:.2f}", d, f"efficacy gain={d:.2f}"))
    spec.append(("futility", "flat", 0.10, "futility delta=0.10 flat"))
    spec.append(("futility", "flat", 0.15, "futility delta=0.15 flat"))
    spec.append(("futility", "reallike", 0.15, "futility delta=0.15 real-like"))
    rows = []
    for test, scen, param, label in spec:
        for K in KS:
            row = {"status": LABEL, "row": label, "K": K}
            for design in DESIGNS:
                sub = sim[(sim["test"] == test) & (sim["scenario"] == scen) & (sim["K"] == K)
                          & np.isclose(sim["param"], param) & (sim["design"] == design)]
                if len(sub) == 0:
                    row[f"{design}_ep"] = float("nan")
                    row[f"{design}_tasks"] = float("nan")
                    row[f"{design}_decided"] = float("nan")
                else:
                    row[f"{design}_ep"] = float(sub["median_ep"].iloc[0])
                    row[f"{design}_tasks"] = float(sub["median_tasks"].iloc[0])
                    row[f"{design}_decided"] = float(sub["decided_frac"].iloc[0])
            row["reduction_ep_combined_vs_audit"] = ratio(row["audit_paired_ep"], row["combined_a_ep"])
            row["reduction_tasks_combined_vs_audit"] = ratio(row["audit_paired_tasks"], row["combined_a_tasks"])
            row["feasible_80_combined_a"] = bool(row["combined_a_tasks"] <= 80)
            best = float("inf")
            for design in DESIGNS:
                v = row[f"{design}_tasks"]
                if math.isfinite(v) and v < best:
                    best = v
            row["feasible_80_any_design"] = bool(best <= 80)
            rows.append(row)
    return pd.DataFrame(rows)


def write_md(head: pd.DataFrame, val: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("# B5 headline: combined adaptive + known-baseline designs (EXPLORATORY)")
    lines.append("")
    lines.append("Simulation (G pool task difficulty, alpha=0.05, cap 24000 episodes, 300 sims/cell). "
                 "Cells are `median episodes / median distinct tasks` to decide, over ALL sims "
                 "(`>24000` = fewer than half decided within the cap). Tasks are fresh each step "
                 "(infinite supply) except B5b, where tasks are drawn from a 200-task bank "
                 "(distinct tasks counted). B5b excludes 30 prior default runs per bank task "
                 "(6000 episodes of existing data).")
    lines.append("")
    lines.append("| row | K | audit-paired | adaptive-only | known-only | combined B5a | combined B5b (est. b) | "
                 "reduction ep | reduction tasks | feasible (<=80 tasks) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in range(len(head)):
        r = head.iloc[i]
        cells = []
        for design in DESIGNS:
            e = r[f"{design}_ep"]
            t = r[f"{design}_tasks"]
            if math.isnan(e):
                cells.append("n/a")
            else:
                cells.append(f"{fmt(e)} / {fmt(t)} ({100 * r[design + '_decided']:.0f}%)")
        red_e = r["reduction_ep_combined_vs_audit"]
        red_t = r["reduction_tasks_combined_vs_audit"]
        red_e_s = "n/a" if math.isnan(red_e) else f"{red_e:.2f}x"
        red_t_s = "n/a" if math.isnan(red_t) else f"{red_t:.2f}x"
        feas = "yes" if r["feasible_80_combined_a"] else ("only via other design" if r["feasible_80_any_design"] else "no")
        lines.append(f"| {r['row']} | {int(r['K'])} | " + " | ".join(cells) + f" | {red_e_s} | {red_t_s} | {feas} |")
    lines.append("")
    lines.append("Percent in parentheses = fraction of sims that decided within the cap. "
                 "Reduction = audit-paired / combined B5a (values <1 mean combined is worse on that axis; "
                 "n/a when one side never decides at the median).")
    lines.append("")
    lines.append("## Validity (null / boundary cells)")
    lines.append("")
    lines.append("| test | scenario | K | param | design | sims | false-decision rate | mc se |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i in range(len(val)):
        r = val.iloc[i]
        lines.append(f"| {r['test']} | {r['scenario']} | {int(r['K'])} | {r['param']} | {r['design']} | "
                     f"{int(r['sims'])} | {r['decided_frac']:.3f} | {r['mc_se']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--val-sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    rows = []
    if args.which in ("all", "efficacy"):
        rows.extend(efficacy_cells(args.sims, args.val_sims, args.seed))
    if args.which in ("all", "futility"):
        rows.extend(futility_cells(args.sims, args.val_sims, args.seed))
    sim = pd.DataFrame(rows)
    sim.to_csv(OUT / f"b5_sim{'' if args.which == 'all' else '_' + args.which}.csv", index=False)
    if args.which != "all":
        return
    is_val = ((sim["scenario"].isin(["allnull", "reallike_below", "boundary"])))
    val = sim[is_val & sim["design"].isin(["combined_a", "combined_b", "known_only"])]
    val.to_csv(OUT / "b5_validity.csv", index=False)
    head = build_headline(sim)
    head.to_csv(OUT / "b5_headline.csv", index=False)
    write_md(head, val, OUT / "b5_headline.md")
    print(head.to_string())


if __name__ == "__main__":
    main()
