"""End-of-run markdown summary."""

from __future__ import annotations

import math

import pandas as pd


def summary_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Run summary\n\n(empty log)\n"

    last = df.iloc[-1]
    run_id = last.get("run_id", "?")
    T = int(last["t"])
    alpha = float(last["inference"]["alpha"])
    m0 = float(last["inference"]["m0"])
    global_log_e = float(last["global_e"]["log_e"])
    global_rejected = bool(last["global_e"]["rejected"])

    lines: list[str] = [
        f"# Run summary — `{run_id}`",
        "",
        f"- T = {T}",
        f"- alpha = {alpha}",
        f"- per-arm null m_0 = {m0}",
        f"- global log-e = {global_log_e:.3f} (e = {math.exp(global_log_e):.3g}); "
        f"global null rejected = **{global_rejected}**",
        "",
        "## Per-arm final state",
        "",
        "| arm | pulls | successes | mean | log_e | CS low | CS high | rejected |",
        "|---|---|---|---|---|---|---|---|",
    ]
    pas = last["per_arm_state"]
    log_thresh = math.log(1.0 / alpha)
    for aid in sorted(pas.keys()):
        s = pas[aid]
        pulls = s["pulls"]
        successes = s["successes"]
        mean = (successes / pulls) if pulls else 0.0
        rejected = s["log_e"] >= log_thresh
        lines.append(
            f"| {aid} | {pulls} | {successes} | {mean:.3f} | "
            f"{s['log_e']:.3f} | {s['cs_lo']:.3f} | {s['cs_hi']:.3f} | {rejected} |"
        )
    lines.append("")

    # First rejection times
    lines.append("## First-rejection times")
    lines.append("")
    arm_ids = sorted(pas.keys())
    for aid in arm_ids:
        first_t = None
        for _, row in df.iterrows():
            le = row["per_arm_state"].get(aid, {}).get("log_e", 0.0)
            if le >= log_thresh:
                first_t = int(row["t"])
                break
        lines.append(f"- {aid}: {first_t if first_t is not None else '—'}")

    return "\n".join(lines) + "\n"
