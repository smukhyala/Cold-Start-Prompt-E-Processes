#!/usr/bin/env python
"""M5 gate: is this dataset good enough to model, and what does it already say?

Run before any fitting. The questions it answers are the ones that decide whether the
overnight compute bought anything: how precise are the labels, is the design actually
balanced, is there signal to fit, and does the answer move when the commitment horizon
or the reservoir family changes.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load(path: Path) -> dict:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(path / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {path}")
    t = pq.read_table(files)
    print(f"{len(files)} shard files, {t.num_rows} rows, {t.num_columns} columns")
    return {n: np.asarray(t.column(n).to_pylist()) for n in t.column_names}


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/oracle_labels")
    ap.add_argument("--out", type=str, default="results/growing_bandits/tables")
    args = ap.parse_args()

    d = load(ROOT / args.data)
    A = np.asarray(d["label_A"], float)
    se = np.asarray(d["label_se"], float)
    M = np.asarray(d["label_M"], float)
    n = len(A)

    rule("LABEL QUALITY")
    decided = np.abs(A) > 2 * se
    print(f"  A_t        mean {A.mean():+.5f}   sd {A.std():.5f}")
    print("             percentiles  " + "  ".join(
        f"p{q}={np.percentile(A, q):+.5f}" for q in (1, 25, 50, 75, 99)))
    print(f"  SEARCH preferred (A_t > 0):        {100*(A > 0).mean():5.1f}%")
    print(f"  clearly decided (|A_t| > 2 SE):    {100*decided.mean():5.1f}%")
    print(f"  exactly zero (all replicates tie): {100*(A == 0).mean():5.1f}%")
    print(f"  SE         mean {se.mean():.6f}   median {np.median(se):.6f}   max {se.max():.6f}")
    print(f"  replicates mean {M.mean():.0f}   max {M.max():.0f}")
    if "label_se_unpaired" in d:
        un = np.asarray(d["label_se_unpaired"], float)
        ok = se > 1e-9
        if ok.any():
            print(f"  CRN gain   paired SE is {np.median(un[ok]/se[ok]):.0f}x smaller than unpaired")
    if "label_frac_identical" in d:
        fi = np.asarray(d["label_frac_identical"], float)
        print(f"  both branches recommend the same arm in {100*np.nanmean(fi):.1f}% of replicates")

    rule("DESIGN COVERAGE  (a correlated design cannot separate effects)")
    for key in ("meta_env", "meta_family", "meta_policy", "meta_allocation", "meta_horizon"):
        if key in d:
            c = Counter(np.asarray(d[key]).tolist())
            lo, hi = min(c.values()), max(c.values())
            print(f"  {key:18s} {len(c):3d} levels   min/max rows per level {lo}/{hi}")
    if "meta_env" in d and "meta_policy" in d:
        combos = len(set(zip(np.asarray(d["meta_env"]).tolist(),
                             np.asarray(d["meta_policy"]).tolist(), strict=False)))
        print(f"  (env x policy) combinations present: {combos}")

    rule("WHERE THE SIGNAL LIVES")
    if "f_remaining_frac" in d:
        rem = np.asarray(d["f_remaining_frac"], float)
        print(f"  {'remaining budget':>20}{'n':>8}{'mean |A|':>12}{'decided':>10}{'SEARCH':>9}")
        for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
            m = (rem >= lo) & (rem < hi)
            if m.any():
                print(f"  {f'{lo:.2f}-{hi:.2f}':>20}{m.sum():>8}{np.abs(A[m]).mean():>12.5f}"
                      f"{100*decided[m].mean():>9.1f}%{100*(A[m] > 0).mean():>8.1f}%")

    ks = sorted(int(c.split("_k")[1]) for c in d if c.startswith("label_A_k"))
    if len(ks) > 1:
        rule("COMMITMENT HORIZON  (how much can one decision be worth?)")
        print(f"  {'k':>4}{'mean |A|':>12}{'decided':>10}{'SEARCH pref':>13}{'sign agrees with k=1':>23}")
        a1 = np.asarray(d[f"label_A_k{ks[0]}"], float)
        for k in ks:
            ak = np.asarray(d[f"label_A_k{k}"], float)
            sk = np.asarray(d[f"label_se_k{k}"], float)
            agree = float(np.mean(np.sign(ak) == np.sign(a1)))
            print(f"  {k:>4}{np.abs(ak).mean():>12.5f}{100*np.mean(np.abs(ak) > 2*sk):>9.1f}%"
                  f"{100*np.mean(ak > 0):>12.1f}%{100*agree:>22.1f}%")

    if "meta_family" in d:
        rule("BY RESERVOIR FAMILY  (C is the held-out generalization set)")
        fam = np.asarray(d["meta_family"])
        print(f"  {'family':>8}{'n':>8}{'mean A':>12}{'SEARCH':>9}{'decided':>10}")
        for f in sorted(set(fam.tolist())):
            m = fam == f
            print(f"  {f:>8}{m.sum():>8}{A[m].mean():>+12.5f}"
                  f"{100*(A[m] > 0).mean():>8.1f}%{100*decided[m].mean():>9.1f}%")

    rule("VERDICT")
    issues = []
    if decided.mean() < 0.10:
        issues.append(f"only {100*decided.mean():.1f}% of labels are decided -- the corpus is "
                      f"mostly ties; fit on the sign with precision weights, or use a larger k")
    if se.mean() > 1e-3:
        issues.append(f"mean SE {se.mean():.5f} is large relative to typical |A_t| "
                      f"{np.abs(A).mean():.5f}")
    if (M >= M.max()).mean() > 0.25:
        issues.append(f"{100*(M >= M.max()).mean():.0f}% of states hit the replicate cap -- "
                      f"the precision target may be unreachable for them")
    if "meta_env" in d:
        c = Counter(np.asarray(d["meta_env"]).tolist())
        if max(c.values()) > 3 * min(c.values()):
            issues.append("environment coverage is imbalanced by more than 3x")
    if issues:
        for i in issues:
            print(f"  ! {i}")
    else:
        print("  no blocking issues found")
    print(f"\n  {n} labelled states ready for modelling")


if __name__ == "__main__":
    main()
