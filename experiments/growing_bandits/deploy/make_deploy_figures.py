#!/usr/bin/env python
"""M7 figures for the deployment study, drawn from the CSV tables `analyze_deployment.py` wrote.

Every figure is built from a table under ``results/growing_bandits/deploy/tables/`` and
writes, next to its PNG and PDF, the exact frame it was drawn from (``<stem>.csv``), so
a number read off a plot can be checked against a file. Nothing here recomputes a
statistic: the CIs come from the tables.

Style: dark surface, one fixed categorical hue per policy (never cycled, never
re-assigned when a policy is absent), 2px lines with a translucent CI wash, hairline
solid gridlines, direct labels at line ends for the key policies plus a legend, and
matplotlib only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import policy_table as pt  # noqa: E402

from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402

log = logging.getLogger("deploy.figures")

DEFAULT_OUT_DIR = ROOT / "results" / "growing_bandits" / "deploy"
KEY_POLICIES: tuple[str, ...] = tuple(pt.TEST_POLICIES["robust"])

# ---- style ------------------------------------------------------------------------------------

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
#: Fixed hue per policy (dark-mode categorical steps, in the palette's validated order).
POLICY_COLORS: dict[str, str] = {
    "phi_k16": "#3987e5",  # blue: the pre-registered learned policy
    "cp0": "#d95926",  # orange: H1a reference
    "p3_star": "#199e70",  # aqua: H1b reference
    "phi_k16_quality": "#c98500",  # yellow: H2 comparator
    "always_search": "#d55181",  # magenta
    "refine_after_init": "#9085e9",  # violet
    "phi_k16_clock": "#e66767",  # red
    "phi_k16_all71": "#008300",  # green
    # The two P11 reservoir rules appear only in the reservoir figure, beside the learned
    # policies; their hues are outside the eight-slot palette so nothing there collides.
    "phi_reservoir_rule": "#b48ead",
    "rule_reservoir": "#e5e2d3",
}
OTHER_COLOR = MUTED
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#1a1a19", "#184f95", "#3987e5", "#9ec5f4", "#cde2fb"]
)
LABELS: dict[str, str] = {
    "phi_k16": "Φ16 (P9)",
    "phi_k16_quality": "Φ16 quality-only (P7)",
    "phi_k16_clock": "Φ16 clock-only (P8)",
    "phi_k16_all71": "Φ16 +history (P10)",
    "cp0": "cp0",
    "p3_star": "P3*",
    "always_search": "always search (P0)",
    "refine_after_init": "refine after init (P1)",
    "phi_reservoir_rule": "reservoir logistic (P11)",
    "rule_reservoir": "reservoir hand rule (P11)",
}
DYNAMICS_METRICS: tuple[tuple[str, str], ...] = (
    ("K_t", "arms held K_t"),
    ("best_discovered", "best discovered μ"),
    ("q_primary", "recommended-now quality"),
    ("n_eliminated", "arms eliminated"),
    ("search_rate", "search rate"),
)

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


def color_of(policy: str) -> str:
    return POLICY_COLORS.get(policy, OTHER_COLOR)


def label_of(policy: str) -> str:
    return LABELS.get(policy, policy)


def save(fig: plt.Figure, frame: pd.DataFrame, out_dir: Path, stem: str) -> dict[str, Path]:
    """PNG + PDF + the CSV the figure was drawn from."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": out_dir / f"{stem}.png",
        "pdf": out_dir / f"{stem}.pdf",
        "csv": out_dir / f"{stem}.csv",
    }
    fig.savefig(paths["png"], bbox_inches="tight", pad_inches=0.25)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    frame.to_csv(paths["csv"], index=False)
    log.info("wrote %s.{png,pdf,csv} (%d rows)", stem, len(frame))
    return paths


LABEL_MIN_GAP_PT = 10.0
#: Alphas of the two stacked segments of `fig_decomposition`. On `SURFACE` the dimmer of
#: the two reads as roughly half the luminance of the brighter, so the caption names them
#: "bright" and "dim" -- calling `R_sel` the "light" segment inverted the key (finding F3).
#: The caption and the key live here beside the alphas so the words and the rendering
#: cannot drift apart again.
DISC_ALPHA, SEL_ALPHA = 0.95, 0.45
DECOMP_CAPTION = (
    "Regret decomposition R_T = R_disc (bright, lower) + R_sel (dim, upper); "
    "whiskers: 95% CI of R_T"
)
DECOMP_KEY: tuple[str, str] = (
    "R_disc (discovery): bright, lower segment",
    "R_sel (selection): dim, upper segment",
)


def _tie_groups(
    items: list[tuple[float, float, str, str]], tols: list[float]
) -> list[list[int]]:
    """Indices of `items` grouped by coincidence: same x, y closer than the tolerance.

    Spreading labels apart vertically invents a ranking the data does not contain
    (finding F2): at T=1000 `always_search`, `p3_star`, `phi_k16` and `phi_k16_quality`
    share their regret to the last bit, and the spreader below used to fan them into a
    column with Phi16 at the top. Labels that coincide are therefore merged into one,
    and only distinct values are spread. `tols[i]` is the caller's idea of "the same"
    -- the CI half-width where it has one, else 0.0, which merges exact ties only.
    """
    groups: list[list[int]] = []
    for i in sorted(range(len(items)), key=lambda i: (items[i][0], items[i][1])):
        for g in groups:
            j = g[0]
            if items[j][0] == items[i][0] and abs(items[i][1] - items[j][1]) <= max(tols[i], tols[j]):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def _direct_labels(
    ax: plt.Axes, items: list[tuple[float, float, str, str]], tols: list[float] | None = None
) -> None:
    """Direct labels at line ends, pushed apart vertically so they never overlap.

    `items` are ``(x, y, text, color)`` in data coordinates. Coincident labels
    (`_tie_groups`, within `tols`) become a single ``a\\n= b\\n= c`` box at their shared
    value; the distinct ones are spread in display space to at least `LABEL_MIN_GAP_PT`
    per line, and a leader marks each true end point, which keeps a tick at every real y.
    """
    if not items:
        return
    tols = list(tols) if tols is not None else [0.0] * len(items)
    fig = ax.figure
    fig.canvas.draw()  # the data->display transform needs final limits
    groups = _tie_groups(items, tols)
    anchors = [(items[g[0]][0], float(np.mean([items[i][1] for i in g]))) for g in groups]
    order = sorted(range(len(groups)), key=lambda k: anchors[k][1])
    disp = np.array([ax.transData.transform(anchors[k])[1] for k in order])
    unit = LABEL_MIN_GAP_PT * fig.dpi / 72.0
    heights = np.array([len(groups[k]) * unit for k in order], dtype=np.float64)
    adj = disp.copy()
    for k in range(1, len(adj)):
        adj[k] = max(adj[k], adj[k - 1] + 0.5 * (heights[k - 1] + heights[k]))
    # Re-centre so the stack does not drift upward as a whole.
    adj -= (adj.mean() - disp.mean())
    for k, gi in enumerate(order):
        group = groups[gi]
        x, y = anchors[gi]
        text = "\n".join(
            items[i][2] if n == 0 else f"= {items[i][2]}" for n, i in enumerate(group)
        )
        y_lab = ax.transData.inverted().transform((0.0, adj[k]))[1]
        ax.annotate(
            text, (x, y), xytext=(x, y_lab), textcoords="data", va="center", ha="left",
            fontsize=8, color=INK_2, annotation_clip=False,
            bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9, "shrinkA": 0, "shrinkB": 3},
        )
        for i in group:
            ax.plot([items[i][0]], [items[i][1]], marker="o", markersize=5, color=items[i][3],
                    markeredgecolor=SURFACE, markeredgewidth=1.2)


def _none_requested(stem: str, frame_columns) -> bool:
    """Log and skip when none of the requested policies is in the table."""
    log.warning("%s: none of the requested policies is in the table (policies present: %s); skipped",
                stem, sorted(frame_columns))
    return True


def _legend(target, policies: Iterable[str], **kw) -> None:
    """A legend below the axes/figure; identity never rides on colour alone."""
    policies = list(policies)
    if not policies:
        return
    handles = [plt.Line2D([0], [0], color=color_of(p), lw=2) for p in policies]
    labels = [label_of(p) for p in policies]
    if isinstance(target, plt.Figure):
        target.legend(handles, labels, loc="lower center", ncol=min(len(policies), 4),
                      bbox_to_anchor=(0.5, -0.02), **kw)
    else:
        target.legend(handles, labels, loc="best", ncol=1, **kw)


def _log_x(ax: plt.Axes, ticks: list) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sorted(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


#: The live-arm cap every learned model's training corpus was harvested at
#: (DEPLOYMENT_PLAN.md "Training corpus"). Caps beyond it are out of training support, so
#: a cap-sweep point to their right is an extrapolation and must not be read as a
#: measurement of a learned rule. The corpus itself dropped K >= 64 states, so support
#: degrades as the boundary is approached rather than ending cleanly at it -- hence a
#: shaded region plus a hairline, not a hard rule.
TRAINING_CAP = 64


def _mark_training_support(ax: plt.Axes, cap: int = TRAINING_CAP, *, annotate: bool = False) -> bool:
    """Shade the out-of-support side of `cap` on an x axis of live-arm caps."""
    lo, hi = ax.get_xlim()
    if hi <= cap:
        return False
    ax.axvspan(cap, hi, color=INK, alpha=0.05, lw=0, zorder=0)
    ax.axvline(cap, color=AXIS, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(lo, hi)
    if annotate:
        ax.text(0.985, 0.03, f"cap > {cap}: outside training support", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=MUTED)
    return True


# ---- figures ------------------------------------------------------------------------------------


def fig_regret_vs_T(
    main: pd.DataFrame, policies: Iterable[str], out_dir: Path, stem: str, rec: str = PRIMARY_RECOMMENDER
) -> pd.DataFrame:
    """Mean regret vs horizon per family (level ``family_horizon``), CI wash from the table."""
    policies = [p for p in policies if p in set(main["policy"])]
    sub = main[(main["level"] == "family_horizon") & (main["policy"].isin(policies))].copy()
    if "recommender" in sub:
        sub = sub[sub["recommender"] == rec]
    if not policies or sub.empty:
        _none_requested(stem, set(main["policy"]))
        return sub
    sub["horizon"] = sub["horizon"].astype(int)
    families = sorted(sub["family"].unique())
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, max(len(families), 1), figsize=(4.6 * max(len(families), 1), 3.6), squeeze=False)
        for ax, fam in zip(axes[0], families, strict=False):
            f = sub[sub["family"] == fam]
            ends: list[tuple[float, float, str, str]] = []
            tols: list[float] = []
            for p in policies:
                g = f[f["policy"] == p].sort_values("horizon")
                if g.empty:
                    continue
                x, y = g["horizon"].to_numpy(), g["regret"].to_numpy()
                ax.plot(x, y, color=color_of(p), marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0)
                ax.fill_between(x, g["regret_lo"].to_numpy(), g["regret_hi"].to_numpy(), color=color_of(p), alpha=0.12, lw=0)
                if p in KEY_POLICIES:
                    ends.append((float(x[-1]), float(y[-1]), label_of(p), color_of(p)))
                    # Two policies the interval cannot separate get one label, not a ranking.
                    half = 0.5 * float(g["regret_hi"].iloc[-1] - g["regret_lo"].iloc[-1])
                    tols.append(half if np.isfinite(half) and half > 0 else 0.0)
            _log_x(ax, list(f["horizon"].unique()))
            ax.set_xlabel("horizon T")
            ax.set_ylabel(f"mean regret ({rec})")
            ax.set_title(f"family {fam}")
            ax.margins(x=0.12)
            _direct_labels(ax, ends, tols)
        _legend(fig, policies)
        fig.suptitle("Deployed regret vs horizon (equal weight per cell, 95% cell-stratified bootstrap)", color=INK, fontsize=11, y=1.02)
        # Wide gutter: the direct labels sit outside the axes and would otherwise run
        # into the next panel's y-axis label.
        fig.subplots_adjust(bottom=0.28, wspace=0.62)
        save(fig, sub, out_dir, stem)
    return sub


def fig_dynamics(dyn: pd.DataFrame, policies: Iterable[str], family: str, out_dir: Path, stem: str) -> pd.DataFrame:
    """Trajectory statistics vs t/T: rows = metrics, columns = horizons, one family."""
    policies = [p for p in policies if p in set(dyn["policy"])]
    sub = dyn[(dyn["family"] == family) & (dyn["policy"].isin(policies))]
    if not policies or sub.empty:
        _none_requested(stem, set(dyn["policy"]))
        return sub
    metrics = [m for m, _ in DYNAMICS_METRICS if m in set(sub["metric"])]
    # Cross-environment mean of the per-cell means at each grid slot (equal weight per cell).
    agg = (
        sub.groupby(["horizon", "policy", "metric", "t_frac"], sort=True)["value"]
        .mean().reset_index()
    )
    horizons = sorted(agg["horizon"].unique())
    if not horizons or not metrics:
        return agg
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(metrics), len(horizons), figsize=(3.2 * len(horizons), 2.3 * len(metrics)), squeeze=False, sharex=True)
        for j, T in enumerate(horizons):
            for i, m in enumerate(metrics):
                ax = axes[i][j]
                g = agg[(agg["horizon"] == T) & (agg["metric"] == m)]
                for p in policies:
                    gp = g[g["policy"] == p].sort_values("t_frac")
                    if gp.empty:
                        continue
                    ax.plot(gp["t_frac"], gp["value"], color=color_of(p))
                if i == 0:
                    ax.set_title(f"T = {int(T)}")
                if j == 0:
                    ax.set_ylabel(dict(DYNAMICS_METRICS)[m])
                if i == len(metrics) - 1:
                    ax.set_xlabel("t / T")
                ax.set_xlim(0, 1)
        fig.suptitle(f"Dynamics, family {family} (cross-episode means, equal weight per environment)", color=INK, fontsize=11, y=1.0)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        _legend(fig, policies)
        save(fig, agg, out_dir, stem)
    return agg


def fig_decomposition(decomp: pd.DataFrame, policies: Iterable[str], out_dir: Path, stem: str) -> pd.DataFrame:
    """`R_T = R_disc + R_sel` as stacked bars per policy, one panel per family x horizon."""
    policies = [p for p in policies if p in set(decomp["policy"])]
    sub = decomp[(decomp["level"] == "family_horizon") & (decomp["policy"].isin(policies))].copy()
    if not policies or sub.empty:
        _none_requested(stem, set(decomp["policy"]))
        return sub
    sub["horizon"] = sub["horizon"].astype(int)
    panels = sorted({(f, T) for f, T in zip(sub["family"], sub["horizon"], strict=False)})
    n = len(panels)
    ncol = min(n, 5)
    nrow = int(np.ceil(n / ncol))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow), squeeze=False, sharey="row")
        for k, (fam, T) in enumerate(panels):
            ax = axes[k // ncol][k % ncol]
            g = sub[(sub["family"] == fam) & (sub["horizon"] == T)].set_index("policy").reindex(policies)
            x = np.arange(len(policies))
            disc = g["regret_disc"].to_numpy(dtype=np.float64)
            sel = g["regret_sel"].to_numpy(dtype=np.float64)
            colors = [color_of(p) for p in policies]
            # A 2px surface gap separates the two segments of every bar. R_disc is the
            # bright segment and R_sel the dim one: on this dark surface the 0.45 alpha
            # composites to about half the luminance of the 0.95 one, so the caption and
            # the key below say "bright"/"dim" and not "solid"/"light" (finding F3).
            ax.bar(x, disc, width=0.62, color=colors, alpha=DISC_ALPHA, edgecolor=SURFACE, linewidth=1.5)
            ax.bar(x, sel, width=0.62, bottom=disc, color=colors, alpha=SEL_ALPHA, edgecolor=SURFACE, linewidth=1.5)
            # A degenerate bootstrap (identical episodes) puts the bound a float epsilon on the
            # wrong side of the mean; matplotlib rejects a negative error bar length.
            total = disc + sel
            yerr = [np.clip(total - g["regret_lo"].to_numpy(dtype=np.float64), 0.0, None),
                    np.clip(g["regret_hi"].to_numpy(dtype=np.float64) - total, 0.0, None)]
            ax.errorbar(x, total, yerr=yerr, fmt="none", ecolor=INK_2, elinewidth=1.0, capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels([label_of(p) for p in policies], rotation=40, ha="right", fontsize=7.5)
            ax.set_title(f"family {fam}, T = {T}")
            ax.grid(axis="x", visible=False)
            if k % ncol == 0:
                ax.set_ylabel("regret")
        for k in range(n, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(DECOMP_CAPTION, color=INK, fontsize=11, y=1.0)
        key = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color_of("phi_k16"), alpha=a, edgecolor=SURFACE)
            for a in (DISC_ALPHA, SEL_ALPHA)
        ]
        fig.legend(key, list(DECOMP_KEY), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        save(fig, sub, out_dir, stem)
    return sub


def fig_offline_vs_deployed(ovd: pd.DataFrame, validity: pd.DataFrame, out_dir: Path, stem: str, rec: str = PRIMARY_RECOMMENDER) -> pd.DataFrame:
    """Offline OOF AUC vs deployed pooled regret across learned variants, Spearman in the title."""
    sub = ovd[ovd["recommender"] == rec].copy() if "recommender" in ovd else ovd.copy()
    sub = sub[sub["oof_auc_env"].notna() & sub["regret_pooled"].notna()]
    rho_txt = ""
    if len(validity):
        v = validity[(validity["recommender"] == rec) & (validity["offline"] == "oof_auc") & (validity["deployed"] == "pooled")]
        if len(v) and np.isfinite(v["rho"].iloc[0]):
            r = v.iloc[0]
            rho_txt = f"  Spearman ρ = {r['rho']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}], n = {int(r['n_variants'])}"
    # Label selectively: the policies with a fixed hue plus the four extremes; the rest
    # are identifiable from the CSV written next to the figure.
    labelled: set[str] = {str(p) for p in sub["policy"] if p in POLICY_COLORS}
    if len(sub):
        for col in ("regret_pooled", "oof_auc_env"):
            labelled.add(str(sub.loc[sub[col].idxmin(), "policy"]))
            labelled.add(str(sub.loc[sub[col].idxmax(), "policy"]))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ends: list[tuple[float, float, str, str]] = []
        for _, r in sub.iterrows():
            p = str(r["policy"])
            yerr = [[max(r["regret_pooled"] - r["regret_lo"], 0.0)], [max(r["regret_hi"] - r["regret_pooled"], 0.0)]]
            ax.errorbar(r["oof_auc_env"], r["regret_pooled"], yerr=yerr,
                        fmt="o", color=color_of(p), markersize=7, markeredgecolor=SURFACE, ecolor=color_of(p),
                        elinewidth=1, capsize=0, alpha=1.0 if p in POLICY_COLORS else 0.7)
            if p in labelled:
                ends.append((float(r["oof_auc_env"]), float(r["regret_pooled"]),
                             str(r["variant"]).replace("clock_quality_evidence", "cqe"), color_of(p)))
        ax.margins(x=0.25)
        _direct_labels(ax, ends)
        ax.set_xlabel("offline OOF AUC (env-grouped folds)")
        ax.set_ylabel(f"deployed pooled regret ({rec})")
        ax.set_title("H3: does the offline surrogate rank deployed regret?" + rho_txt, fontsize=9.5)
        save(fig, sub, out_dir, stem)
    return sub


def fig_tau_curves(tau: pd.DataFrame, out_dir: Path, stem: str) -> pd.DataFrame:
    """Validation tau -> pooled regret per variant; post-hoc test points drawn hollow.

    Learned models share a probability threshold in [0, 1]; the P11 hand rule's tau is a
    gain-per-width ratio on its own scale, so it gets its own panel when present.
    """
    import analyze_deployment as ad

    sub = tau.copy()
    if "heldout_T" not in sub.columns:
        sub["heldout_T"] = False
    val = sub[~sub["posthoc"].astype(bool) & ~sub["heldout_T"].astype(bool)]
    post = sub[sub["posthoc"].astype(bool)]
    canonical = ad.canonical_policy_of_variant()
    hand = val[val["variant"] == "reservoir_rule"]
    models = val[val["variant"] != "reservoir_rule"]
    with plt.rc_context(RC):
        n_panels = 2 if len(hand) else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 4.2), squeeze=False)
        ax = axes[0][0]
        ends: list[tuple[float, float, str, str]] = []
        for v in sorted(models["variant"].unique()):
            g = models[models["variant"] == v].sort_values("tau")
            p = canonical.get(v, "")
            key = p in POLICY_COLORS
            ax.plot(g["tau"], g["regret"], color=color_of(p) if key else OTHER_COLOR,
                    alpha=1.0 if key else 0.45, lw=2 if key else 1.2)
            if key:
                ends.append((float(g["tau"].iloc[-1]), float(g["regret"].iloc[-1]), label_of(p), color_of(p)))
        for _, r in post.iterrows():
            if np.isfinite(r["tau"]):
                ax.plot([r["tau"]], [r["regret"]], marker="o", markersize=9, markerfacecolor=SURFACE,
                        markeredgecolor=color_of("phi_k16"), markeredgewidth=2)
                ends.append((float(r["tau"]), float(r["regret"]), f"{r['policy']} ({r['source']}, post-hoc)", color_of("phi_k16")))
        ax.margins(x=0.25)
        _direct_labels(ax, ends)
        ax.set_xlabel("SEARCH threshold τ (learned models)")
        ax.set_ylabel("pooled regret")
        ax.set_title("validation seeds (lines); hollow = test set, post-hoc")
        if len(hand):
            ax2 = axes[0][1]
            g = hand.sort_values("tau")
            ax2.plot(g["tau"], g["regret"], color=color_of("rule_reservoir"), marker="o", markersize=5, markeredgecolor=SURFACE)
            ax2.set_xlabel("hand-rule τ (gain x remaining / leader width)")
            ax2.set_ylabel("pooled regret")
            ax2.set_title(label_of("rule_reservoir"))
        fig.suptitle("τ sensitivity", color=INK, fontsize=11, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, sub, out_dir, stem)
    return sub


def fig_ood_heatmap(ood_features: pd.DataFrame, ood_summary: pd.DataFrame, policy: str, out_dir: Path, stem: str) -> pd.DataFrame:
    """Left: fraction of `policy`'s on-policy rows outside the corpus [0.5%, 99.5%] range,
    feature x cell. Right: kNN OOD fraction, policy x cell."""
    feat = ood_features[ood_features["policy"] == policy]
    grid = feat.pivot_table(index="feature", columns="cell", values="frac_outside", aggfunc="mean")
    order = [c for c in ood_features["feature"].unique() if c in grid.index]
    grid = grid.reindex(order)
    summ = ood_summary.pivot_table(index="policy", columns="cell", values="ood_frac", aggfunc="mean")
    with plt.rc_context(RC):
        h = max(4.0, 0.16 * len(grid.index) + 1.5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(9.0, 0.5 * (grid.shape[1] + summ.shape[1]) + 5), h),
                                       gridspec_kw={"width_ratios": [max(grid.shape[1], 1), max(summ.shape[1], 1)]})
        # Coverage fractions are usually a few percent, so that panel's scale follows the
        # data; the kNN OOD fraction keeps a fixed [0, 1] scale so cells compare across tests.
        cov_max = float(np.nanmax(grid.to_numpy(dtype=np.float64))) if grid.size else 0.0
        for ax, g, title, vmax in (
            (ax1, grid, f"{label_of(policy)}: frac outside corpus [0.5%, 99.5%] range", max(0.05, cov_max)),
            (ax2, summ, "kNN(5) OOD fraction (policy x cell); flag > 0.10", 1.0),
        ):
            if g.empty:
                ax.axis("off")
                continue
            im = ax.imshow(g.to_numpy(dtype=np.float64), cmap=SEQ_CMAP, vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(g.shape[1]))
            ax.set_xticklabels([str(c) for c in g.columns], rotation=90, fontsize=6.5)
            ax.set_yticks(range(g.shape[0]))
            ax.set_yticklabels([str(i) for i in g.index], fontsize=6.5)
            ax.grid(False)
            ax.set_title(title, fontsize=9.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.ax.tick_params(colors=MUTED, labelsize=7)
            cb.outline.set_edgecolor(AXIS)
        fig.suptitle("On-policy state shift vs the corpus (register #5/#11)", color=INK, fontsize=11, y=1.0)
        fig.tight_layout()
        long = grid.reset_index().melt(id_vars="feature", var_name="cell", value_name="frac_outside")
        long["policy"] = policy
        save(fig, long, out_dir, stem)
    return long


def fig_cap_sweep(cap: pd.DataFrame, policies: Iterable[str], out_dir: Path, stem: str) -> pd.DataFrame:
    """Regret vs live-arm cap per policy; one panel per (environment, horizon).

    The `TRAINING_CAP` boundary is drawn in every panel: this is the one figure a reader
    uses to judge cap sensitivity, and the learned policies' points beyond cap 64 are
    extrapolation, not a measurement of a rule that was ever trained there.
    """
    policies = [p for p in policies if p in set(cap["policy"])]
    sub = cap[cap["policy"].isin(policies)].copy()
    if not policies or sub.empty:
        _none_requested(stem, set(cap["policy"]))
        return sub
    sub["cap"] = sub["cap"].astype(int)
    sub["horizon"] = sub["horizon"].astype(int)
    panels = sorted({(e, T) for e, T in zip(sub["env_id"], sub["horizon"], strict=False)})
    ncol = min(len(panels), 4)
    nrow = int(np.ceil(len(panels) / ncol))
    shaded = False
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), squeeze=False)
        for k, (env, T) in enumerate(panels):
            ax = axes[k // ncol][k % ncol]
            g = sub[(sub["env_id"] == env) & (sub["horizon"] == T)]
            for p in policies:
                gp = g[g["policy"] == p].sort_values("cap")
                if gp.empty:
                    continue
                ax.plot(gp["cap"], gp["regret"], color=color_of(p), marker="o", markersize=5, markeredgecolor=SURFACE)
                ax.fill_between(gp["cap"], gp["regret_lo"], gp["regret_hi"], color=color_of(p), alpha=0.12, lw=0)
            _log_x(ax, list(g["cap"].unique()))
            ax.set_title(f"{env}, T = {T}", fontsize=9)
            ax.set_xlabel("live-arm cap")
            if k % ncol == 0:
                ax.set_ylabel("mean regret")
            shaded = _mark_training_support(ax, annotate=(k == 0)) or shaded
        for k in range(len(panels), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        title = "Cap sensitivity (95% paired-bootstrap CI of the mean)"
        if shaded:
            title += f"\nshaded: cap > {TRAINING_CAP}, beyond the learned models' training support"
        fig.suptitle(title, color=INK, fontsize=11, y=1.0)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        _legend(fig, policies)
        save(fig, sub, out_dir, stem)
    return sub


def fig_reservoir(bins: pd.DataFrame, reservoir: pd.DataFrame, policies: Iterable[str], out_dir: Path, stem: str) -> pd.DataFrame:
    """Left: P(SEARCH) by within-cell decile of oracle `I_t`, per family (pooled over cells).
    Right: AUC of oracle `I_t` for each policy's own decisions, per cell."""
    policies = [p for p in policies if p in set(bins["policy"])]
    sub = bins[bins["policy"].isin(policies)]
    if not policies or sub.empty:
        _none_requested(stem, set(bins["policy"]))
        return sub
    prof = sub.groupby(["family", "policy", "bin"], sort=True).agg(
        search_rate=("search_rate", "mean"), model_p_mean=("model_p_mean", "mean"), I_mean=("I_mean", "mean"), n=("n", "sum")
    ).reset_index()
    families = sorted(prof["family"].unique())
    auc = reservoir[reservoir["policy"].isin(policies)][["cell", "family", "horizon", "policy", "auc_oracle_I_for_decision", "search_rate"]].copy()
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(families) + 1, figsize=(4.2 * (len(families) + 1), 3.6), squeeze=False)
        for ax, fam in zip(axes[0], families, strict=False):
            g = prof[prof["family"] == fam]
            for p in policies:
                gp = g[g["policy"] == p].sort_values("bin")
                if gp.empty:
                    continue
                ax.plot(gp["bin"] + 1, gp["search_rate"], color=color_of(p), marker="o", markersize=5, markeredgecolor=SURFACE)
            ax.set_xlabel("decile of oracle I_t within the cell (1 = smallest tail)")
            ax.set_ylabel("P(SEARCH)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"family {fam}")
        ax = axes[0][-1]
        x = np.arange(len(policies))
        for i, p in enumerate(policies):
            vals = auc[auc["policy"] == p]["auc_oracle_I_for_decision"].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ax.scatter(np.full(vals.size, x[i]) + np.linspace(-0.15, 0.15, vals.size), vals, color=color_of(p), s=22, edgecolor=SURFACE, linewidth=0.8)
                ax.plot([x[i] - 0.25, x[i] + 0.25], [vals.mean()] * 2, color=INK_2, lw=1.5)
        ax.axhline(0.5, color=AXIS, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([label_of(p) for p in policies], rotation=40, ha="right", fontsize=7.5)
        ax.set_ylabel("AUC of oracle I_t for the policy's decisions")
        ax.set_title("per cell (dot) and mean (bar)")
        fig.suptitle("Reservoir tail vs SEARCH decisions (oracle I_t = ∫ P(μ > x) dx above the incumbent)", color=INK, fontsize=11, y=1.02)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        _legend(fig, policies)
        save(fig, prof, out_dir, stem)
    return prof


# ---- CLI ----------------------------------------------------------------------------------------------


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("%s absent; skipping the figures that need it", path.name)
        return None
    return pd.read_csv(path)


def _table(tables_dir: Path, base: str, test: str, rec: str | None = None) -> Path:
    import analyze_deployment as ad

    return tables_dir / ad.table_name(base, test, rec)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--recommender", default=PRIMARY_RECOMMENDER)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tables-dir", type=Path, default=None)
    p.add_argument("--figures-dir", type=Path, default=None)
    p.add_argument("--policies", default=None, help="comma list (default: the six key policies)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> list[str]:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    test, rec = args.test, args.recommender
    out_dir = Path(args.out_dir)
    tables = Path(args.tables_dir) if args.tables_dir else out_dir / "tables"
    figures = Path(args.figures_dir) if args.figures_dir else out_dir / "figures"
    policies = [p.strip() for p in args.policies.split(",")] if args.policies else list(KEY_POLICIES)
    written: list[str] = []

    main_t = _read(_table(tables, "main", test, rec))
    if main_t is not None:
        fig_regret_vs_T(main_t, policies, figures, f"regret_vs_T_{test}", rec)
        written.append(f"regret_vs_T_{test}")
    dyn = _read(tables / f"dynamics_{test}.csv")
    if dyn is not None and len(dyn):
        for fam in sorted(dyn["family"].unique()):
            fig_dynamics(dyn, policies, str(fam), figures, f"dynamics_{test}_family{fam}")
            written.append(f"dynamics_{test}_family{fam}")
    decomp = _read(_table(tables, "decomposition", test, rec))
    if decomp is not None:
        fig_decomposition(decomp, policies, figures, f"decomposition_{test}")
        written.append(f"decomposition_{test}")
    ovd = _read(_table(tables, "offline_vs_deployed", test))
    validity = _read(_table(tables, "surrogate_validity", test))
    if ovd is not None and len(ovd):
        fig_offline_vs_deployed(ovd, validity if validity is not None else pd.DataFrame(), figures, f"offline_vs_deployed_{test}", rec)
        written.append(f"offline_vs_deployed_{test}")
    tau = _read(_table(tables, "tau_curves", test))
    if tau is not None and len(tau):
        fig_tau_curves(tau, figures, f"tau_curves_{test}")
        written.append(f"tau_curves_{test}")
    ood_f = _read(tables / f"ood_features_{test}.csv")
    ood_s = _read(tables / f"ood_{test}.csv")
    if ood_f is not None and ood_s is not None and len(ood_f):
        policy = "phi_k16" if "phi_k16" in set(ood_f["policy"]) else str(ood_f["policy"].iloc[0])
        fig_ood_heatmap(ood_f, ood_s, policy, figures, f"ood_heatmap_{test}")
        written.append(f"ood_heatmap_{test}")
    cap = _read(tables / "cap_sweep.csv") if test == "cap" else None
    if cap is not None and len(cap):
        fig_cap_sweep(cap, policies, figures, "cap_sweep")
        written.append("cap_sweep")
    bins = _read(tables / f"reservoir_bins_{test}.csv")
    res = _read(tables / f"reservoir_{test}.csv")
    if bins is not None and res is not None and len(bins):
        learned = [p for p in policies if pt.POLICIES.get(p, {}).get("group") in ("learned", "rule")]
        extra = [p for p in ("phi_k16_clock", "phi_reservoir_rule", "rule_reservoir") if p in set(bins["policy"]) and p not in learned]
        fig_reservoir(bins, res, learned + extra, figures, f"reservoir_{test}")
        written.append(f"reservoir_{test}")
    log.info("%d figures written to %s", len(written), figures)
    return written


if __name__ == "__main__":
    main()
