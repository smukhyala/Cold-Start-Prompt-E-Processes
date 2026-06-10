"""End-of-run markdown summary.

Reports the per-arm and global statistics formalized in Mukhyala &
Waudby-Smith (2026):

- § 3.3   μ_a (sample mean), a* (empirical best arm), Δ_a = μ̂* − μ̂_a
- § 3.4   per-arm log E_t, confidence-sequence bounds, rejection flag
- § 3.6   τ_α = inf{t : E_t ≥ 1/α} (global first-rejection time)
- § 3.2   prompt-distance matrix d(a,b) over the arm catalog (when arms
          and axes are supplied)
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cold_start.prompts.axes import AxesSpec
from cold_start.prompts.vector import distance
from cold_start.types import Arm


def summary_markdown(
    df: pd.DataFrame,
    arms: list[Arm] | None = None,
    axes: AxesSpec | None = None,
) -> str:
    """Render a markdown summary for a completed run.

    When `arms` and `axes` are supplied, an "Arm geometry" section with the
    pairwise prompt-distance matrix d(a,b) is appended.
    """
    if df.empty:
        return "# Run summary\n\n(empty log)\n"

    last = df.iloc[-1]
    run_id = last.get("run_id", "?")
    T = int(last["t"])
    alpha = float(last["inference"]["alpha"])
    m0 = float(last["inference"]["m0"])
    global_log_e = float(last["global_e"]["log_e"])
    global_rejected = bool(last["global_e"]["rejected"])
    log_thresh = math.log(1.0 / alpha)

    # Stopping time τ_α: first t at which the global log-e crosses log(1/α).
    tau_alpha: int | None = None
    for _, row in df.iterrows():
        if float(row["global_e"].get("log_e", 0.0)) >= log_thresh:
            tau_alpha = int(row["t"])
            break

    lines: list[str] = [
        f"# Run summary — `{run_id}`",
        "",
        f"- T = {T}",
        f"- alpha = {alpha}",
        f"- per-arm null m_0 = {m0}",
        f"- global log-e = {global_log_e:.3f} (e = {math.exp(global_log_e):.3g}); "
        f"global null rejected = **{global_rejected}**",
        f"- τ_α (first global rejection) = "
        f"{tau_alpha if tau_alpha is not None else '— (never crossed log 1/α)'}",
        "",
    ]

    pas = last["per_arm_state"]
    arm_ids = sorted(pas.keys())

    # Empirical best arm a* by sample mean μ̂_a; ties broken by arm_id.
    means: dict[str, float] = {}
    for aid in arm_ids:
        s = pas[aid]
        n = s["pulls"]
        means[aid] = (s["successes"] / n) if n else float("nan")
    finite_means = {aid: mu for aid, mu in means.items() if not math.isnan(mu)}
    if finite_means:
        best_mu = max(finite_means.values())
        a_star = sorted(aid for aid, mu in finite_means.items() if mu == best_mu)[0]
    else:
        a_star = None
        best_mu = float("nan")

    lines += [
        "## Per-arm final state",
        "",
        "| arm | a* | pulls | successes | μ̂_a | Δ_a | log_e | CS low | CS high | rejected |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for aid in arm_ids:
        s = pas[aid]
        pulls = s["pulls"]
        successes = s["successes"]
        mu = means[aid]
        mu_str = "—" if math.isnan(mu) else f"{mu:.3f}"
        if math.isnan(mu) or math.isnan(best_mu):
            delta_str = "—"
        else:
            delta_str = f"{best_mu - mu:.3f}"
        rejected = s["log_e"] >= log_thresh
        star = "★" if aid == a_star else ""
        lines.append(
            f"| {aid} | {star} | {pulls} | {successes} | {mu_str} | {delta_str} | "
            f"{s['log_e']:.3f} | {s['cs_lo']:.3f} | {s['cs_hi']:.3f} | {rejected} |"
        )
    lines.append("")
    if a_star is not None:
        lines.append(f"Empirical best arm a* = **{a_star}** (μ̂ = {best_mu:.3f}).")
        lines.append("")

    # Per-arm first-rejection times
    lines += ["## First-rejection times", ""]
    for aid in arm_ids:
        first_t: int | None = None
        for _, row in df.iterrows():
            le = row["per_arm_state"].get(aid, {}).get("log_e", 0.0)
            if le >= log_thresh:
                first_t = int(row["t"])
                break
        lines.append(f"- {aid}: {first_t if first_t is not None else '—'}")
    lines.append("")

    # Optional arm-geometry section
    if arms is not None and axes is not None:
        lines += _arm_geometry_section(arms, axes)

    return "\n".join(lines) + "\n"


def distance_matrix(arms: list[Arm], axes: AxesSpec) -> pd.DataFrame:
    """K×K weighted Manhattan distance matrix d(a,b) over the arm catalog.

    Rows and columns indexed by arm_id, sorted for deterministic ordering.
    """
    sorted_arms = sorted(arms, key=lambda a: a.arm_id)
    ids = [a.arm_id for a in sorted_arms]
    n = len(sorted_arms)
    data = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(sorted_arms):
        for j, b in enumerate(sorted_arms):
            if i == j:
                continue
            data[i][j] = distance(a.vector, b.vector, axes)
    return pd.DataFrame(data, index=ids, columns=ids)


def _arm_geometry_section(arms: list[Arm], axes: AxesSpec) -> list[str]:
    matrix = distance_matrix(arms, axes)
    ids = list(matrix.columns)
    lines: list[str] = [
        "## Arm geometry — prompt-distance matrix d(a,b)",
        "",
        "Weighted, range-normalized Manhattan over the six axes (Mukhyala &",
        "Waudby-Smith § 3.2). 0 on the diagonal; larger values = more dissimilar.",
        "",
        "| | " + " | ".join(ids) + " |",
        "|---|" + "---|" * len(ids),
    ]
    for aid in ids:
        row = [f"{matrix.loc[aid, other]:.3f}" for other in ids]
        lines.append(f"| **{aid}** | " + " | ".join(row) + " |")
    lines.append("")
    return lines


def write_summary(
    df: pd.DataFrame,
    out: str | Path,
    arms: list[Arm] | None = None,
    axes: AxesSpec | None = None,
) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(summary_markdown(df, arms=arms, axes=axes))
    return p
