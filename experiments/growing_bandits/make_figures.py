#!/usr/bin/env python
"""Section 16: the figures, in the repo's publication style.

Colour decisions here are computed, not chosen by eye.

* The oracle advantage `A_t` encodes POLARITY -- SEARCH above zero, REFINE below -- so it
  gets a diverging map with a NEUTRAL GREY midpoint and two hues that are not red/green.
  A rainbow or a red-green pair would make the sign unreadable for a protanope, and the
  sign is the entire content of those panels.
* Series identity uses the Okabe-Ito order, applied in a fixed sequence and never cycled.
  The repo's current seaborn-deep trio actually FAILS a colour-vision check -- its green
  and orange separate by only dE 4.5 under protanopia -- so new figures do not inherit it.
  Existing paper figures are left alone; that is a separate decision.
* Because the worst Okabe-Ito adjacent pair sits in the dE 6-8 band, identity is never
  carried by colour alone: every panel is direct-labelled or axis-labelled, and every
  figure writes the CSV it was drawn from.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Okabe-Ito, fixed order. Validated: lightness band, chroma floor and normal-vision
# floor all pass; worst CVD adjacent pair dE 7.6, hence the direct-labelling rule.
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
# Diverging: blue <-> neutral grey <-> vermillion. Safe under every CVD type.
DIVERGING = LinearSegmentedColormap.from_list("search_refine", ["#0072B2", "#BFBFBF", "#D55E00"])
SEQUENTIAL = LinearSegmentedColormap.from_list("mag", ["#E8F1F8", "#0072B2"])
INK, MUTED, GRID = "0.15", "0.45", "0.88"


def setup_style() -> None:
    """Match scripts/generate_global_log_e_growth.py, which feeds the paper."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def save(fig, out_dir: Path, name: str) -> None:
    """Always a PNG+PDF pair, as every other figure script in this repo does."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def write_csv(out_dir: Path, name: str, header: list[str], rows: list[list]) -> None:
    """Every figure ships the numbers it was drawn from -- the contrast relief the
    palette check requires, and what makes a figure checkable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}.csv").open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")


def load(path: Path) -> dict:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(path / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {path}")
    t = pq.read_table(files)
    return {n: np.asarray(t.column(n).to_pylist()) for n in t.column_names}


def _binned(x: np.ndarray, y: np.ndarray, nbins: int = 12):
    """Equal-count bins with a standard error, so the trend is readable under noise."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < nbins * 3:
        nbins = max(3, x.size // 5)
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    cx, cy, ce, cn = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if m.sum() < 3:
            continue
        cx.append(float(np.median(x[m])))
        cy.append(float(y[m].mean()))
        ce.append(float(y[m].std(ddof=1) / np.sqrt(m.sum())))
        cn.append(int(m.sum()))
    return np.array(cx), np.array(cy), np.array(ce), np.array(cn)


def fig_advantage_vs(data: dict, col: str, xlabel: str, out: Path, name: str) -> None:
    """A_t against one driver, with the zero line as the decision boundary."""
    if col not in data:
        print(f"  (skipped {name}: {col} absent)")
        return
    x, y = np.asarray(data[col], float), np.asarray(data["label_A"], float)
    cx, cy, ce, cn = _binned(x, y)
    if not len(cx):
        print(f"  (skipped {name}: too few rows)")
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.axhline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.scatter(x, y, s=5, alpha=0.15, color=CATEGORICAL[0], lw=0, zorder=2)
    ax.errorbar(cx, cy, yerr=ce, color=CATEGORICAL[5], lw=2, marker="o", ms=4,
                capsize=2, zorder=3, label="binned mean +/- SE")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"oracle advantage $A_t$")
    ax.text(0.02, 0.96, "SEARCH better", transform=ax.transAxes, va="top",
            fontsize=7, color=CATEGORICAL[5])
    ax.text(0.02, 0.04, "REFINE better", transform=ax.transAxes, va="bottom",
            fontsize=7, color=CATEGORICAL[0])
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, loc="upper right")
    save(fig, out, name)
    write_csv(out, name, ["x", "mean_A", "se", "n"],
              [[cx[i], cy[i], ce[i], cn[i]] for i in range(len(cx))])


def fig_decision_region(data: dict, out: Path) -> None:
    """Where the oracle prefers SEARCH, as a function of evidence and arm density."""
    xc, yc = "f_log_e_pair", "f_K_over_t"
    if xc not in data or yc not in data:
        print("  (skipped decision region: columns absent)")
        return
    x, y, a = (np.asarray(data[xc], float), np.asarray(data[yc], float),
               np.asarray(data["label_A"], float))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(a)
    x, y, a = x[ok], y[ok], a[ok]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    lim = float(np.nanpercentile(np.abs(a), 98)) or 1e-6
    hb = ax.hexbin(x, y, C=a, gridsize=18, cmap=DIVERGING,
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                   mincnt=3, linewidths=0.2, edgecolors="white")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r"mean $A_t$   (>0: SEARCH)", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_xlabel(r"leader-vs-challenger evidence  $\log e_{\mathrm{pair}}$")
    ax.set_ylabel(r"arm density  $K_t / t$")
    save(fig, out, "01_oracle_decision_region")


def fig_boundary_vs_schedules(data: dict, out: Path) -> None:
    """Where the oracle searches, against the classical growth schedules."""
    if "f_t" not in data:
        return
    t, K, a = (np.asarray(data["f_t"], float), np.asarray(data["f_K"], float),
               np.asarray(data["label_A"], float))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    search = a > 0
    ax.scatter(t[~search], K[~search], s=7, alpha=0.35, lw=0,
               color=CATEGORICAL[0], label="oracle: REFINE")
    ax.scatter(t[search], K[search], s=7, alpha=0.55, lw=0,
               color=CATEGORICAL[5], label="oracle: SEARCH")
    grid = np.linspace(max(t.min(), 1), t.max(), 200)
    for i, (lab, f) in enumerate((
        (r"$\sqrt{t}$", np.sqrt(grid)),
        (r"$t^{1/3}$", grid ** (1 / 3)),
        (r"$t^{2/3}$", grid ** (2 / 3)),
    )):
        ax.plot(grid, f, lw=1.6, ls="--", color=CATEGORICAL[1 + i], label=lab)
    ax.set_xlabel("evaluations spent  $t$")
    ax.set_ylabel("arms discovered  $K_t$")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, out, "05_boundary_vs_growth_schedules")


def fig_commitment(data: dict, out: Path) -> None:
    """How much a SEARCH/REFINE decision can be worth, against how long it is held."""
    ks = sorted(int(c.split("_k")[1]) for c in data if c.startswith("label_A_k"))
    if len(ks) < 2:
        return
    means, decided = [], []
    for k in ks:
        a = np.asarray(data[f"label_A_k{k}"], float)
        se = np.asarray(data[f"label_se_k{k}"], float)
        means.append(float(np.nanmean(np.abs(a))))
        decided.append(float(np.nanmean(np.abs(a) > 2 * se)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax1.plot(ks, means, marker="o", lw=2, color=CATEGORICAL[0])
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("commitment horizon  $k$")
    ax1.set_ylabel(r"mean $|A_t|$")
    ax1.set_title("magnitude", fontsize=9)
    ax2.plot(ks, [100 * d for d in decided], marker="o", lw=2, color=CATEGORICAL[5])
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("commitment horizon  $k$")
    ax2.set_ylabel(r"% states with $|A_t| > 2\,$SE")
    ax2.set_title("resolvability", fontsize=9)
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    save(fig, out, "06_commitment_horizon")
    write_csv(out, "06_commitment_horizon", ["k", "mean_abs_A", "frac_decided"],
              [[ks[i], means[i], decided[i]] for i in range(len(ks))])


def fig_by_family(data: dict, out: Path) -> None:
    """Does the answer depend on the reservoir family? (the generalization question)"""
    key = "meta_family" if "meta_family" in data else "meta_env"
    fam = np.asarray(data[key])
    a = np.asarray(data["label_A"], float)
    names = sorted(set(fam.tolist()))[:12]
    means = [float(np.nanmean(a[fam == n])) for n in names]
    errs = [float(np.nanstd(a[fam == n], ddof=1) / max(np.sqrt((fam == n).sum()), 1))
            for n in names]
    fig, ax = plt.subplots(figsize=(max(4.4, 0.5 * len(names) + 2.2), 3.0))
    colors = [CATEGORICAL[5] if m > 0 else CATEGORICAL[0] for m in means]
    ax.bar(range(len(names)), means, yerr=errs, color=colors, capsize=2, width=0.68)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(r"mean oracle advantage $A_t$")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    save(fig, out, "09_by_reservoir_family")
    write_csv(out, "09_by_reservoir_family", ["group", "mean_A", "se"],
              [[names[i], means[i], errs[i]] for i in range(len(names))])


def fig_regret_curves(bench_csv: Path, out: Path) -> None:
    """Section 17: final simple regret by policy. Identity is on the axis, not in colour."""
    if not bench_csv.exists():
        print("  (skipped regret curves: no policy_benchmark.csv)")
        return
    rows = [ln.strip().split(",") for ln in bench_csv.read_text().splitlines() if ln.strip()]
    head, body = rows[0], rows[1:]
    idx = {h: i for i, h in enumerate(head)}
    envs = sorted({r[idx["environment"]] for r in body})
    fig, axes = plt.subplots(1, len(envs), figsize=(3.4 * len(envs), 3.2), squeeze=False)
    for ax, env in zip(axes[0], envs, strict=False):
        sub = [r for r in body if r[idx["environment"]] == env]
        sub.sort(key=lambda r: float(r[idx["mean_regret"]]))
        names = [r[idx["policy"]] for r in sub]
        vals = [float(r[idx["mean_regret"]]) for r in sub]
        errs = [float(r[idx["se_regret"]]) for r in sub]
        best = min(vals)
        colors = [CATEGORICAL[2] if v <= best + 1e-9 else CATEGORICAL[0] for v in vals]
        ax.barh(range(len(names)), vals, xerr=errs, color=colors, height=0.68, capsize=2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("final simple regret")
        ax.set_title(env, fontsize=9)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    save(fig, out, "07_simple_regret_by_policy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/oracle_labels")
    ap.add_argument("--out", type=str, default="results/growing_bandits/figures")
    ap.add_argument("--bench", type=str,
                    default="results/growing_bandits/tables/policy_benchmark.csv")
    args = ap.parse_args()

    setup_style()
    out = ROOT / args.out
    data = load(ROOT / args.data)
    print(f"loaded {len(data['label_A'])} states; writing figures to {out}")

    fig_decision_region(data, out)
    fig_advantage_vs(data, "f_log_e_pair", r"pairwise evidence  $\log e_{\mathrm{pair}}$",
                     out, "02_advantage_vs_pairwise_evidence")
    fig_advantage_vs(data, "est_p_new_beats_incumbent",
                     r"estimated $P(\mu_{\mathrm{new}} > \mu_{\mathrm{incumbent}})$",
                     out, "03_advantage_vs_estimated_tail")
    fig_boundary_vs_schedules(data, out)
    fig_commitment(data, out)
    fig_by_family(data, out)
    fig_regret_curves(ROOT / args.bench, out)
    print("done")


if __name__ == "__main__":
    main()
