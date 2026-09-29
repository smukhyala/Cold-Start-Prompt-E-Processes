"""Pre-registration 10 paper figures (spec 2026-09-28-prompt-heterogeneity, section 7), drawn only from
the CSV tables `het_verdicts.py`, `calibrate.py`, `stage0.py` and `replay.py`'s ``kgrid`` write -- this
module reads no raw episodes or reservoirs.

    .venv/bin/python -c "from figures import make_figures; make_figures(tables_dir, out_dir)"

Seven figures (spec section 7), one PNG + PDF each under `out_dir` (``fig{n}_<slug>.{png,pdf}``):

1. Variance decomposition per cell (task / prompt / prompt x task / noise), from ``het_components.csv``
   (``tau_main`` squared for the prompt component; ``task_var`` / ``interaction_var`` / ``noise_var``).
2. ``tau_main`` with its 95% MLS interval per cell, over the flat / moderate / meaningful bands. The
   flat edge is the ``tau_flat`` keyword (Pre-registration 10 fixes its *value* from the Stage-0
   calibration elsewhere -- this module never invents one; ``None`` omits the band, annotated). The
   moderate/meaningful edges come from the calibration curve's own regret-range -> true_sd mapping at
   the spec's section 6.5 rr thresholds (0.005, 0.01) when ``calibration_csv`` is given; omitted
   (annotated) otherwise. Nothing here hard-codes a tau threshold -- only the spec's own rr thresholds.
3. Value of search vs heterogeneity: the Stage-0 calibration curve (true sd on x, regret range at
   T = 200 on y; tail mixtures as a second marker set), the empirical cells as labelled points with
   their tau_main interval as horizontal error bars (y from ``het_classification.csv``'s
   ``regret_range_T200``), and the historical Stage-0 points (Pre-reg 9 Gmail G/F, the old GitLab
   paired subsets) labelled "historical". Stage 0 ran no K-grid on these, so they have no measured
   regret range: their y is the calibration curve's own value interpolated at their tau_main, which is
   an interpolated placement, never a measurement, and is only drawn when a calibration curve is given.
4. Regret-vs-K curves per empirical pool at T in {50, 100, 200}, from ``het_kgrid.csv`` (variant
   ``npmle``).
5. Policy gaps to the ceiling (``het_policy_gaps.csv``'s delta with its lo/hi interval), restricted to
   the per-cell rows of pools ``het_classification.csv`` classifies "meaningful".
6. Portability scatter: the G prompts' effects, Gmail vs GitLab (``het_portability.csv``).
7. Supplement, built from tables only (no raw episodes/reservoirs), with these documented stand-ins:
   (a) upper-tail mass per cell with its bootstrap CI, standing in for NPMLE densities -- no producer
       currently writes per-atom NPMLE columns to plot instead;
   (b) split-half reliability (``r_sb``) per cell; no producer computes a bootstrap interval for it, so
       its split-half p-value is annotated per point instead of an interval;
   (c) per-prompt timeout rates per cell, one strip per pool, from ``het_timeout_rates.csv``;
   (d) anchor recovery (``GL_anchor_*``/``GM_anchor_*`` rates), read from whichever of
       ``het_components.csv`` / ``het_timeout_rates.csv`` carries matching columns -- an annotated
       empty panel ("anchor rates: see gates output") when neither does, which is the case for every
       producer as of this task.

Every optional table that is missing or empty draws an annotated empty panel instead of raising; only a
missing ``het_components.csv`` (the one required table) raises `FileNotFoundError`. Agg backend, dark
style (surface ``#0f1115``, light text, no top/right spines), one fixed accent colour per pool family
(G, K) plus a separate historical marker; fonts/sizes match `deploy/make_deploy_figures.py`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: Default location of the heterogeneity study's K-grid: `run_deployment.DEFAULT_OUT_DIR` / "tables" /
#: `study.HETEROGENEITY.table("kgrid")`, written as a literal path here so this module needs no import
#: from the deploy tree just to name it.
DEFAULT_KGRID_CSV = ROOT / "results" / "growing_bandits" / "deploy" / "tables" / "het_kgrid.csv"

#: Spec section 6.5's regret-range thresholds -- fixed by the spec itself, not by Pre-registration 10,
#: so (unlike tau_flat) they are safe to hard-code here.
FLAT_RR = 0.005
MEANINGFUL_RR = 0.01
PRIMARY_HORIZONS: tuple[int, ...] = (50, 100, 200)

# ---- style --------------------------------------------------------------------------------------

SURFACE = "#0f1115"
INK = "#f2f1ec"
INK_2 = "#c3c2b7"
MUTED = "#87857e"
GRID = "#22252a"
AXIS = "#33363c"
#: One fixed accent colour per pool family (brief: "one accent colour per pool family (G, F, K)"). F is
#: historical-only (Pre-reg 9's F pool never re-runs in this study), so it gets its own colour even
#: though no current-study pool belongs to it; every historical point additionally uses a distinct
#: hollow marker (`HIST_MARKER`) so "historical" never rides on colour alone.
FAMILY_COLORS: dict[str, str] = {"G": "#3987e5", "K": "#c98500", "F": "#d55181"}
POOL_FAMILY: dict[str, str] = {"GMG": "G", "GLG": "G", "GMB": "G", "GMK": "K", "GLK": "K"}
HIST_COLOR = MUTED
HIST_MARKER = "D"
COMPONENT_COLORS: dict[str, str] = {
    "prompt": FAMILY_COLORS["G"], "task": "#d95926", "interaction": "#199e70", "noise": MUTED,
}
BAND_COLORS: dict[str, str] = {"flat": "#199e70", "moderate": "#c98500", "meaningful": "#d95926"}

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK_2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "font.family": "sans-serif",
    "figure.dpi": 110,
    "savefig.dpi": 200,
}

_ANCHOR_RE = re.compile(r"^(GL|GM)_anchor_")


# ---- small pure helpers (unit-tested directly) ---------------------------------------------------


def _pool_family(pool: str) -> str:
    """The pool's accent-colour family, ``"?"`` for a pool `POOL_FAMILY` does not name."""
    return POOL_FAMILY.get(str(pool), "?")


def _read_optional(path: Path) -> pd.DataFrame | None:
    """`path`'s CSV, or ``None`` if it does not exist (a present-but-empty file returns an empty frame,
    which callers treat as "nothing to plot" the same way, but distinctly from "not collected yet")."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _meaningful_pools(classification: pd.DataFrame | None) -> list[str]:
    """Every pool `het_classification.csv` classifies ``"meaningful"``; ``[]`` if the table is absent,
    empty, or lacks a ``class`` column."""
    if classification is None or classification.empty or "class" not in classification.columns:
        return []
    return sorted(
        str(p) for p, c in zip(classification["pool"], classification["class"], strict=True) if str(c) == "meaningful"
    )


def _policy_gap_rows_for_meaningful(policy_gaps: pd.DataFrame | None, classification: pd.DataFrame | None) -> pd.DataFrame:
    """`het_policy_gaps.csv` rows whose ``cells`` names one of the meaningful pools -- empty (never an
    exception) when either table is missing/empty or no cell is meaningful."""
    if policy_gaps is None or policy_gaps.empty or "cells" not in policy_gaps.columns:
        return pd.DataFrame(columns=list(policy_gaps.columns) if policy_gaps is not None else [])
    meaningful = set(_meaningful_pools(classification))
    if not meaningful:
        return policy_gaps.iloc[0:0]
    return policy_gaps[policy_gaps["cells"].isin(meaningful)].copy()


def _calibration_edge(calibration: pd.DataFrame | None, *, target: float, horizon: int = 200) -> float | None:
    """The true_sd at which the beta-family calibration curve's ``regret_range`` first crosses `target`
    at `horizon`, sorted by ``true_sd`` (mirrors `calibrate.tau_flat_from_calibration`'s crossing rule,
    but returns ``None`` instead of raising -- a figure degrades gracefully, a pre-registration script
    should not)."""
    if calibration is None or calibration.empty or not {"family", "horizon", "true_sd", "regret_range"} <= set(calibration.columns):
        return None
    beta = calibration[(calibration["family"] == "beta") & (calibration["horizon"] == int(horizon))].sort_values("true_sd")
    if len(beta) < 2:
        return None
    sds = beta["true_sd"].to_numpy(dtype=float)
    ranges = beta["regret_range"].to_numpy(dtype=float)
    for i in range(len(sds) - 1):
        lo, hi = ranges[i], ranges[i + 1]
        if (lo <= target <= hi) or (hi <= target <= lo):
            if hi == lo:
                return float(sds[i])
            frac = (target - lo) / (hi - lo)
            return float(sds[i] + frac * (sds[i + 1] - sds[i]))
    return None


def _interp_regret_range(beta_sorted: pd.DataFrame, x: float) -> float:
    """`np.interp` of ``regret_range`` on ``true_sd`` over the (already sorted, beta-family) curve;
    ``nan`` if the curve is empty."""
    if beta_sorted.empty:
        return float("nan")
    xs = beta_sorted["true_sd"].to_numpy(dtype=float)
    ys = beta_sorted["regret_range"].to_numpy(dtype=float)
    return float(np.interp(x, xs, ys))


def _anchor_columns(df: pd.DataFrame | None) -> list[str]:
    """Every column of `df` matching ``GL_anchor_*`` / ``GM_anchor_*``, in column order."""
    if df is None:
        return []
    return [c for c in df.columns if _ANCHOR_RE.match(str(c))]


def _historical_points(stage0_gmail: pd.DataFrame | None, stage0_gitlab: pd.DataFrame | None) -> list[dict]:
    """``[{label, tau_main, tau_lo, tau_hi}]`` from Stage 0's tables (figure 3's "historical" points):
    ``stage0_gmail.csv``'s ``pool`` rows (G, F) and ``stage0_gitlab_paired.csv``'s ``subset`` rows."""
    rows: list[dict] = []
    if stage0_gmail is not None and not stage0_gmail.empty and "pool" in stage0_gmail.columns:
        for _, r in stage0_gmail.iterrows():
            rows.append({"label": f"{r['pool']} (historical)", "tau_main": float(r["tau_main"]),
                        "tau_lo": float(r["tau_lo"]), "tau_hi": float(r["tau_hi"])})
    if stage0_gitlab is not None and not stage0_gitlab.empty and "subset" in stage0_gitlab.columns:
        for _, r in stage0_gitlab.iterrows():
            rows.append({"label": f"{r['subset']} (historical)", "tau_main": float(r["tau_main"]),
                        "tau_lo": float(r["tau_lo"]), "tau_hi": float(r["tau_hi"])})
    return rows


# ---- drawing helpers ------------------------------------------------------------------------------


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=9, wrap=True, transform=ax.transAxes)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return [png, pdf]


# ---- figures --------------------------------------------------------------------------------------


def _fig1_variance_decomposition(components: pd.DataFrame, out_dir: Path) -> list[Path]:
    stem = "fig1_variance_decomposition"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if components.empty or "pool" not in components.columns:
        _empty_panel(ax, "no cells in het_components.csv")
        return _save(fig, out_dir, stem)
    pools = [str(p) for p in components["pool"]]
    x = np.arange(len(pools))
    bottom = np.zeros(len(pools))
    tau_main = components.get("tau_main", pd.Series(np.nan, index=components.index)).astype(float)
    parts = (
        ("prompt (tau_main^2)", (tau_main**2).to_numpy(dtype=float)),
        ("task", components.get("task_var", pd.Series(np.nan, index=components.index)).astype(float).to_numpy(dtype=float)),
        ("prompt x task (interaction)", components.get("interaction_var", pd.Series(np.nan, index=components.index)).astype(float).to_numpy(dtype=float)),
        ("noise", components.get("noise_var", pd.Series(np.nan, index=components.index)).astype(float).to_numpy(dtype=float)),
    )
    colors = (COMPONENT_COLORS["prompt"], COMPONENT_COLORS["task"], COMPONENT_COLORS["interaction"], COMPONENT_COLORS["noise"])
    for (label, vals), color in zip(parts, colors, strict=True):
        vals = np.nan_to_num(vals, nan=0.0)
        ax.bar(x, vals, bottom=bottom, width=0.6, color=color, label=label, edgecolor=SURFACE, linewidth=1.0)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(pools)
    ax.set_ylabel("variance")
    ax.set_title("Variance decomposition per cell")
    ax.legend(loc="upper right", fontsize=7.5)
    return _save(fig, out_dir, stem)


def _fig2_tau_bands(components: pd.DataFrame, calibration: pd.DataFrame | None, tau_flat: float | None,
                    out_dir: Path) -> list[Path]:
    stem = "fig2_tau_bands"
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if components.empty or not {"pool", "tau_main"} <= set(components.columns):
        _empty_panel(ax, "no cells in het_components.csv")
        return _save(fig, out_dir, stem)
    pools = [str(p) for p in components["pool"]]
    x = np.arange(len(pools))
    tau = components["tau_main"].to_numpy(dtype=float)
    lo = components.get("tau_lo", components["tau_main"]).astype(float).to_numpy(dtype=float)
    hi = components.get("tau_hi", components["tau_main"]).astype(float).to_numpy(dtype=float)
    for xi, ti, li, hi_ in zip(x, tau, lo, hi, strict=True):
        color = FAMILY_COLORS.get(_pool_family(pools[xi]), MUTED)
        ax.errorbar([xi], [ti], yerr=[[max(ti - li, 0.0)], [max(hi_ - ti, 0.0)]], fmt="o", color=color,
                    markersize=7, markeredgecolor=SURFACE, ecolor=color, elinewidth=1.5, capsize=3)
    notes: list[str] = []
    top = float(np.nanmax(hi)) if np.isfinite(hi).any() else float(np.nanmax(tau))
    if tau_flat is not None:
        ax.axhspan(0.0, float(tau_flat), color=BAND_COLORS["flat"], alpha=0.06, lw=0, zorder=0)
        ax.axhline(float(tau_flat), color=AXIS, lw=0.9, ls=(0, (4, 3)))
        top = max(top, float(tau_flat))
    else:
        notes.append("no tau_flat given: flat band omitted")
    edge_flat = _calibration_edge(calibration, target=FLAT_RR)
    edge_meaningful = _calibration_edge(calibration, target=MEANINGFUL_RR)
    if edge_flat is not None and edge_meaningful is not None:
        ax.axhspan(edge_flat, edge_meaningful, color=BAND_COLORS["moderate"], alpha=0.05, lw=0, zorder=0)
        top = max(top, edge_meaningful)
        ax.axhspan(edge_meaningful, top * 1.15 + 1e-9, color=BAND_COLORS["meaningful"], alpha=0.08, lw=0, zorder=0)
    else:
        notes.append("no calibration curve given: moderate/meaningful bands omitted")
    ax.set_ylim(0.0, top * 1.2 + 1e-9)
    ax.set_xticks(x)
    ax.set_xticklabels(pools)
    ax.set_ylabel("tau_main (95% MLS interval)")
    ax.set_title("tau_main per cell over flat / moderate / meaningful bands")
    if notes:
        ax.text(0.02, 0.98, "; ".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=7, color=MUTED)
    return _save(fig, out_dir, stem)


def _fig3_value_of_search(components: pd.DataFrame, classification: pd.DataFrame | None,
                          calibration: pd.DataFrame | None, stage0_gmail: pd.DataFrame | None,
                          stage0_gitlab: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    stem = "fig3_value_of_search"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if calibration is None or calibration.empty or not {"family", "horizon", "true_sd", "regret_range"} <= set(calibration.columns):
        _empty_panel(ax, "no calibration.csv given: figure 3 needs the Stage-0 calibration curve")
        return _save(fig, out_dir, stem)
    cal200 = calibration[calibration["horizon"] == 200]
    beta = cal200[cal200["family"] == "beta"].sort_values("true_sd")
    tail = cal200[cal200["family"] != "beta"]
    if beta.empty:
        _empty_panel(ax, "calibration.csv has no beta-family rows at T = 200")
        return _save(fig, out_dir, stem)
    y_top = max(float(beta["regret_range"].max()), MEANINGFUL_RR) * 1.25 + 1e-9
    for y0, y1, key in ((0.0, FLAT_RR, "flat"), (FLAT_RR, MEANINGFUL_RR, "moderate"), (MEANINGFUL_RR, y_top, "meaningful")):
        ax.axhspan(y0, y1, color=BAND_COLORS[key], alpha=0.06, lw=0, zorder=0)
    ax.plot(beta["true_sd"], beta["regret_range"], color=INK_2, marker="o", markersize=4,
            markeredgecolor=SURFACE, label="calibration (beta)", zorder=2)
    if not tail.empty:
        ax.scatter(tail["true_sd"], tail["regret_range"], marker="^", s=28, color=MUTED,
                   edgecolor=SURFACE, label="calibration (tail mixture)", zorder=2)
    if (classification is not None and not classification.empty
            and {"pool", "regret_range_T200"} <= set(classification.columns) and not components.empty
            and "pool" in components.columns):
        comp_by_pool = {str(r["pool"]): r for _, r in components.iterrows()}
        for i, (_, r) in enumerate(classification.iterrows()):
            pool = str(r["pool"])
            crow = comp_by_pool.get(pool)
            if crow is None or "tau_main" not in crow.index or not np.isfinite(float(r["regret_range_T200"])):
                continue
            tau_m = float(crow["tau_main"])
            tau_lo = float(crow["tau_lo"]) if "tau_lo" in crow.index and np.isfinite(crow["tau_lo"]) else tau_m
            tau_hi = float(crow["tau_hi"]) if "tau_hi" in crow.index and np.isfinite(crow["tau_hi"]) else tau_m
            y = float(r["regret_range_T200"])
            color = FAMILY_COLORS.get(_pool_family(pool), MUTED)
            ax.errorbar([tau_m], [y], xerr=[[max(tau_m - tau_lo, 0.0)], [max(tau_hi - tau_m, 0.0)]], fmt="o",
                        color=color, markersize=7, markeredgecolor=SURFACE, ecolor=color, elinewidth=1.5,
                        capsize=3, zorder=3)
            # Points at close (tau, y) crowd their labels; stagger the vertical offset by index so
            # nearby empirical cells (a common case at small synthetic/pilot spreads) stay legible.
            ax.annotate(pool, (tau_m, y), textcoords="offset points", xytext=(6, 6 + 11 * (i % 3)),
                        fontsize=7.5, color=INK_2)
    else:
        ax.text(0.02, 0.02, "no het_classification.csv: empirical cells omitted", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7, color=MUTED)
    for i, h in enumerate(_historical_points(stage0_gmail, stage0_gitlab)):
        y = _interp_regret_range(beta, h["tau_main"])
        if not np.isfinite(y):
            continue
        ax.errorbar([h["tau_main"]], [y], xerr=[[max(h["tau_main"] - h["tau_lo"], 0.0)],
                    [max(h["tau_hi"] - h["tau_main"], 0.0)]], fmt=HIST_MARKER, color=HIST_COLOR,
                    markersize=6, markerfacecolor="none", markeredgecolor=HIST_COLOR, ecolor=HIST_COLOR,
                    elinewidth=1.2, capsize=2, zorder=1)
        ax.annotate(h["label"], (h["tau_main"], y), textcoords="offset points",
                    xytext=(6, -10 - 11 * (i % 3)), fontsize=7, color=MUTED)
    ax.text(0.98, 0.02, "historical points: y interpolated on the calibration curve (no measured "
            "regret range for these runs)", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=MUTED)
    ax.set_xlabel("true SD of the prompt pool")
    ax.set_ylabel("regret range at T = 200")
    ax.set_title("Value of search vs heterogeneity")
    ax.legend(loc="upper left", fontsize=7.5)
    return _save(fig, out_dir, stem)


def _fig4_regret_vs_k(kgrid: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    stem = "fig4_regret_vs_k"
    if kgrid is None or kgrid.empty or not {"pool", "horizon", "K", "regret"} <= set(kgrid.columns):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        _empty_panel(ax, "no het_kgrid.csv given")
        return _save(fig, out_dir, stem)
    sub = kgrid[kgrid["variant"] == "npmle"] if "variant" in kgrid.columns else kgrid
    sub = sub[sub["horizon"].isin(PRIMARY_HORIZONS)]
    present = [T for T in PRIMARY_HORIZONS if T in set(sub["horizon"])]
    if not present:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        _empty_panel(ax, "het_kgrid.csv has no rows at T in {50, 100, 200}")
        return _save(fig, out_dir, stem)
    pools = sorted(str(p) for p in sub["pool"].unique())
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.6), squeeze=False)
    for ax, T in zip(axes[0], present, strict=True):
        g = sub[sub["horizon"] == T]
        for pool in pools:
            gp = g[g["pool"] == pool].sort_values("K")
            if gp.empty:
                continue
            ax.plot(gp["K"], gp["regret"], color=FAMILY_COLORS.get(_pool_family(pool), MUTED), marker="o",
                    markersize=4, markeredgecolor=SURFACE, label=pool)
        ax.set_xscale("log")
        ax.set_xlabel("K")
        ax.set_title(f"T = {T}")
    axes[0][0].set_ylabel("regret")
    handles = [plt.Line2D([0], [0], color=FAMILY_COLORS.get(_pool_family(p), MUTED), lw=2) for p in pools]
    fig.legend(handles, pools, loc="lower center", ncol=min(len(pools), 5), bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    fig.suptitle("Regret vs K per empirical pool", color=INK, fontsize=11, y=1.02)
    fig.subplots_adjust(bottom=0.26, wspace=0.35)
    return _save(fig, out_dir, stem)


def _fig5_policy_gaps(policy_gaps: pd.DataFrame | None, classification: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    stem = "fig5_policy_gaps"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = _policy_gap_rows_for_meaningful(policy_gaps, classification)
    if rows.empty or not {"delta", "lo", "hi"} <= set(rows.columns):
        _empty_panel(ax, "no meaningful cell")
        return _save(fig, out_dir, stem)
    rows = rows.reset_index(drop=True)
    labels = [f"{r['cells']}: {r['contrast'] if 'contrast' in rows.columns else r.get('registration', '')}"
              for _, r in rows.iterrows()]
    x = np.arange(len(rows))
    delta = rows["delta"].to_numpy(dtype=float)
    lo = rows["lo"].to_numpy(dtype=float)
    hi = rows["hi"].to_numpy(dtype=float)
    for xi, d, li, hi_, cells in zip(x, delta, lo, hi, rows["cells"], strict=True):
        color = FAMILY_COLORS.get(_pool_family(cells), MUTED)
        if np.isfinite(d) and np.isfinite(li) and np.isfinite(hi_):
            ax.errorbar([xi], [d], yerr=[[max(d - li, 0.0)], [max(hi_ - d, 0.0)]], fmt="o", color=color,
                        markersize=7, markeredgecolor=SURFACE, ecolor=color, elinewidth=1.5, capsize=3)
        elif np.isfinite(d):
            ax.plot([xi], [d], marker="o", markersize=7, color=color, markeredgecolor=SURFACE)
    ax.axhline(0.0, color=AXIS, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("policy gap to the ceiling (delta)")
    ax.set_title("Policy gaps in meaningful cells")
    return _save(fig, out_dir, stem)


def _fig6_portability(portability: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    stem = "fig6_portability"
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    if portability is None or portability.empty or not {"effect_gmail", "effect_gitlab"} <= set(portability.columns):
        _empty_panel(ax, "no het_portability.csv data")
        return _save(fig, out_dir, stem)
    x = portability["effect_gmail"].to_numpy(dtype=float)
    y = portability["effect_gitlab"].to_numpy(dtype=float)
    ax.scatter(x, y, color=FAMILY_COLORS["G"], s=32, edgecolor=SURFACE, linewidth=0.8, zorder=2)
    lo = float(min(np.nanmin(x), np.nanmin(y)))
    hi = float(max(np.nanmax(x), np.nanmax(y)))
    pad = 0.08 * (hi - lo) if hi > lo else 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("G prompt effect, Gmail")
    ax.set_ylabel("G prompt effect, GitLab")
    ax.set_title("Portability of G prompt effects, Gmail vs GitLab")
    return _save(fig, out_dir, stem)


def _fig7a_tail_mass(ax: plt.Axes, components: pd.DataFrame) -> None:
    if components.empty or not {"pool", "upper_tail_mass"} <= set(components.columns):
        _empty_panel(ax, "no upper-tail mass in het_components.csv")
        return
    pools = [str(p) for p in components["pool"]]
    x = np.arange(len(pools))
    mass = components["upper_tail_mass"].to_numpy(dtype=float)
    has_ci = {"upper_tail_mass_lo", "upper_tail_mass_hi"} <= set(components.columns)
    for i in x:
        color = FAMILY_COLORS.get(_pool_family(pools[i]), MUTED)
        if has_ci:
            lo_v = float(components["upper_tail_mass_lo"].iloc[i])
            hi_v = float(components["upper_tail_mass_hi"].iloc[i])
            ax.errorbar([i], [mass[i]], yerr=[[max(mass[i] - lo_v, 0.0)], [max(hi_v - mass[i], 0.0)]], fmt="o",
                        color=color, markersize=6, markeredgecolor=SURFACE, ecolor=color, elinewidth=1.2, capsize=3)
        else:
            ax.plot([i], [mass[i]], marker="o", markersize=6, color=color, markeredgecolor=SURFACE)
    ax.set_xticks(x)
    ax.set_xticklabels(pools, fontsize=7.5)
    ax.set_ylabel("upper-tail mass")
    ax.set_title("(a) upper-tail mass (stand-in for NPMLE densities)", fontsize=9)
    if not has_ci:
        ax.text(0.02, 0.02, "no bootstrap CI given", transform=ax.transAxes, fontsize=6.5, color=MUTED)


def _fig7b_split_half(ax: plt.Axes, components: pd.DataFrame) -> None:
    if components.empty or not {"pool", "r_sb"} <= set(components.columns):
        _empty_panel(ax, "no split-half reliability (r_sb) in het_components.csv")
        return
    pools = [str(p) for p in components["pool"]]
    x = np.arange(len(pools))
    r = components["r_sb"].astype(float).to_numpy(dtype=float)
    colors = [FAMILY_COLORS.get(_pool_family(p), MUTED) for p in pools]
    ax.bar(x, np.nan_to_num(r, nan=0.0), width=0.55, color=colors, edgecolor=SURFACE, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(pools, fontsize=7.5)
    ax.set_ylabel("split-half r_sb")
    ax.set_title("(b) split-half reliability", fontsize=9)
    if "split_half_p" in components.columns:
        for xi, ri, p in zip(x, r, components["split_half_p"], strict=True):
            if np.isfinite(p) and np.isfinite(ri):
                ax.annotate(f"p={float(p):.2f}", (xi, ri), textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=6, color=MUTED)
        ax.text(0.02, 0.98, "no bootstrap interval computed for r_sb; split-half p shown per point",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=MUTED)
    else:
        ax.text(0.02, 0.98, "no bootstrap interval for r_sb", transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, color=MUTED)


def _fig7c_timeout_strip(ax: plt.Axes, timeout_rates: pd.DataFrame | None) -> None:
    need = {"pool", "arm_id", "timeout_rate"}
    if timeout_rates is None or timeout_rates.empty or not need <= set(timeout_rates.columns):
        _empty_panel(ax, "no het_timeout_rates.csv data")
        return
    pools = sorted(str(p) for p in timeout_rates["pool"].unique())
    rng = np.random.default_rng(0)
    for i, pool in enumerate(pools):
        g = timeout_rates[timeout_rates["pool"] == pool]
        jitter = rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(np.full(len(g), i) + jitter, g["timeout_rate"], color=FAMILY_COLORS.get(_pool_family(pool), MUTED),
                   s=18, edgecolor=SURFACE, linewidth=0.6)
    ax.set_xticks(range(len(pools)))
    ax.set_xticklabels(pools, fontsize=7.5)
    ax.set_ylabel("per-prompt timeout rate")
    ax.set_title("(c) per-prompt timeout rates", fontsize=9)


def _fig7d_anchor_recovery(ax: plt.Axes, components: pd.DataFrame, timeout_rates: pd.DataFrame | None) -> None:
    comp_cols = _anchor_columns(components)
    rate_cols = _anchor_columns(timeout_rates)
    source, cols = (components, comp_cols) if comp_cols else (timeout_rates, rate_cols)
    if not cols or source is None or source.empty:
        _empty_panel(ax, "anchor rates: see gates output")
        return
    labels = [str(v) for v in source["pool"]] if "pool" in source.columns else [str(i) for i in range(len(source))]
    x = np.arange(len(cols))
    n = len(labels)
    width = 0.8 / max(n, 1)
    for i, (_, row) in enumerate(source.iterrows()):
        vals = [float(row[c]) for c in cols]
        ax.bar(x + i * width, vals, width=width, label=labels[i])
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("anchor recovery rate")
    ax.set_title("(d) anchor recovery", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)


def _fig7_supplement(components: pd.DataFrame, timeout_rates: pd.DataFrame | None, out_dir: Path) -> list[Path]:
    stem = "fig7_supplement"
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
    _fig7a_tail_mass(axes[0][0], components)
    _fig7b_split_half(axes[0][1], components)
    _fig7c_timeout_strip(axes[1][0], timeout_rates)
    _fig7d_anchor_recovery(axes[1][1], components, timeout_rates)
    fig.suptitle("Supplement: tail mass, split-half reliability, timeout rates, anchor recovery",
                color=INK, fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_dir, stem)


# ---- entry point ----------------------------------------------------------------------------------


def make_figures(tables_dir: Path, out_dir: Path, *, calibration_csv: Path | None = None,
                 stage0_dir: Path | None = None, kgrid_csv: Path | None = None,
                 tau_flat: float | None = None) -> list[Path]:
    """The spec section 7 figures, drawn from `tables_dir`'s ``het_*.csv`` tables (`het_verdicts.py`'s
    output directory).

    `het_components.csv` is required (`FileNotFoundError` names it if absent); every other table is
    optional and its absence/emptiness draws an annotated empty panel, never an exception.

    Keyword-only, each defaulting to this study's standard location: `calibration_csv`
    (``tables_dir/calibration.csv``, `calibrate.py`'s output), `stage0_dir` (`tables_dir`, holding
    `stage0.py`'s ``stage0_gmail.csv``/``stage0_gitlab_paired.csv``), `kgrid_csv` (the real deployment
    tree's ``tables/het_kgrid.csv``, `replay.py kgrid`'s output -- a different directory from
    `tables_dir` in production). `tau_flat` is Pre-registration 10's fixed value (figure 2's flat band);
    ``None`` omits that band, annotated.
    """
    tables_dir = Path(tables_dir)
    out_dir = Path(out_dir)
    components_path = tables_dir / "het_components.csv"
    if not components_path.exists():
        raise FileNotFoundError(f"required table missing: {components_path}")
    components = pd.read_csv(components_path)

    classification = _read_optional(tables_dir / "het_classification.csv")
    policy_gaps = _read_optional(tables_dir / "het_policy_gaps.csv")
    portability = _read_optional(tables_dir / "het_portability.csv")
    timeout_rates = _read_optional(tables_dir / "het_timeout_rates.csv")
    calibration = _read_optional(Path(calibration_csv) if calibration_csv is not None else tables_dir / "calibration.csv")
    stage0_root = Path(stage0_dir) if stage0_dir is not None else tables_dir
    stage0_gmail = _read_optional(stage0_root / "stage0_gmail.csv")
    stage0_gitlab = _read_optional(stage0_root / "stage0_gitlab_paired.csv")
    kgrid = _read_optional(Path(kgrid_csv) if kgrid_csv is not None else DEFAULT_KGRID_CSV)

    paths: list[Path] = []
    with plt.rc_context(RC):
        paths += _fig1_variance_decomposition(components, out_dir)
        paths += _fig2_tau_bands(components, calibration, tau_flat, out_dir)
        paths += _fig3_value_of_search(components, classification, calibration, stage0_gmail, stage0_gitlab, out_dir)
        paths += _fig4_regret_vs_k(kgrid, out_dir)
        paths += _fig5_policy_gaps(policy_gaps, classification, out_dir)
        paths += _fig6_portability(portability, out_dir)
        paths += _fig7_supplement(components, timeout_rates, out_dir)
    return paths
