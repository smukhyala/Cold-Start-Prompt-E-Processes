#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Analyze early best-arm identification in the GitLab 160-task sweep.

This is intentionally post-hoc/read-only with respect to experiment logs. It
can be run while the allocation sweep is still active; partial resumed segments
are merged in memory by timestep.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cold_start.inference.confidence_sequence import ConfidenceSequence
from cold_start.inference.hedged_capital import HedgedCapitalEProcess


RUN_RE = re.compile(
    r"gitlab_strong_allocation_(?P<policy>.+)_budget(?P<budget>\d+)_rep(?P<rep>\d+)_trial0_"
)


@dataclass(frozen=True)
class RunBundle:
    base_run_id: str
    policy: str
    replicate: int
    budget: int
    rows: list[dict[str, Any]]
    source_paths: tuple[Path, ...]
    complete: bool
    contiguous: bool


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    paired = pd.read_csv(args.paired_summary)
    true_best = args.true_best or str(paired.iloc[0]["arm_id"])
    best_nonoracle_rate = float(paired.loc[paired["arm_id"] != true_best, "success_rate"].max())
    oracle_rate = float(paired.loc[paired["arm_id"] == true_best, "success_rate"].iloc[0])
    arms = list(paired["arm_id"])

    bundles = load_run_bundles(Path(args.allocation_dir), args.budget)
    timeseries_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        ts = analyze_run(
            bundle,
            arms=arms,
            true_best=true_best,
            best_nonoracle_rate=best_nonoracle_rate,
            alpha=args.alpha,
            min_leader_pulls=args.min_leader_pulls,
        )
        if ts.empty:
            continue
        timeseries_parts.append(ts)
        summary_rows.append(
            summarize_run(
                bundle,
                ts,
                true_best=true_best,
                oracle_rate=oracle_rate,
                success_targets=args.success_targets,
            )
        )

    analysis_dir = out / "identification_analysis"
    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    timeseries = pd.concat(timeseries_parts, ignore_index=True) if timeseries_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    timeseries.to_csv(analysis_dir / "identification_timeseries.csv", index=False)
    summary.to_csv(analysis_dir / "identification_summary.csv", index=False)
    if not summary.empty:
        policy_summary = summarize_by_policy(summary)
        policy_summary.to_csv(analysis_dir / "identification_summary_by_policy.csv", index=False)
    else:
        policy_summary = pd.DataFrame()

    make_plots(timeseries, summary, plot_dir, true_best)
    write_report(
        analysis_dir,
        timeseries=timeseries,
        summary=summary,
        policy_summary=policy_summary,
        true_best=true_best,
        oracle_rate=oracle_rate,
        best_nonoracle_rate=best_nonoracle_rate,
        alpha=args.alpha,
        min_leader_pulls=args.min_leader_pulls,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation-dir",
        type=Path,
        default=ROOT / "results/gitlab_strong_arm/allocation_160",
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        default=ROOT / "results/gitlab_strong_arm/paired/paired_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gitlab_strong_arm/allocation_160")
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--true-best", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-leader-pulls", type=int, default=3)
    parser.add_argument("--success-targets", type=int, nargs="+", default=[35, 40, 43, 50, 60, 80])
    return parser.parse_args()


def load_run_bundles(allocation_dir: Path, budget: int) -> list[RunBundle]:
    log_dir = allocation_dir / "logs"
    grouped: dict[str, tuple[dict[int, dict[str, Any]], list[Path]]] = {}
    for path in sorted(log_dir.glob("gitlab_strong_allocation_*_budget*_rep*_trial0_*.jsonl")):
        name = path.name
        if name.startswith("INVALID_") or name.endswith("_MERGED_SO_FAR.jsonl"):
            continue
        if path.stat().st_size == 0:
            continue
        for row in read_jsonl(path):
            base = str(row["run_id"]).split("_resume_from_")[0]
            rows_by_t, paths = grouped.setdefault(base, ({}, []))
            rows_by_t[int(row["t"])] = row
            if path not in paths:
                paths.append(path)

    bundles: list[RunBundle] = []
    for base, (rows_by_t, paths) in grouped.items():
        match = RUN_RE.search(base)
        if match is None:
            continue
        ordered_t = sorted(rows_by_t)
        contiguous = ordered_t == list(range(1, max(ordered_t) + 1)) if ordered_t else False
        rows = [rows_by_t[t] for t in ordered_t]
        bundles.append(
            RunBundle(
                base_run_id=base,
                policy=match.group("policy"),
                replicate=int(match.group("rep")),
                budget=int(match.group("budget")),
                rows=rows,
                source_paths=tuple(paths),
                complete=len(rows) >= budget and contiguous,
                contiguous=contiguous,
            )
        )
    return sorted(bundles, key=lambda b: (b.replicate, b.policy, b.base_run_id))


def analyze_run(
    bundle: RunBundle,
    *,
    arms: list[str],
    true_best: str,
    best_nonoracle_rate: float,
    alpha: float,
    min_leader_pulls: int,
) -> pd.DataFrame:
    counts = {arm: 0 for arm in arms}
    successes = {arm: 0.0 for arm in arms}
    cs = {arm: ConfidenceSequence(alpha=alpha, grid_size=64) for arm in arms}
    e_vs_half = {arm: HedgedCapitalEProcess(m0=0.5) for arm in arms}
    e_vs_runner_up = {
        arm: HedgedCapitalEProcess(m0=clamp_m0(best_nonoracle_rate)) for arm in arms
    }
    total_successes = 0
    cumulative_runtime = 0.0
    cumulative_cost = 0.0
    log_threshold = math.log(1.0 / alpha)
    out_rows: list[dict[str, Any]] = []

    for row in bundle.rows:
        arm = str(row["arm_id"])
        reward = float(row.get("reward", 0.0))
        counts.setdefault(arm, 0)
        successes.setdefault(arm, 0.0)
        counts[arm] += 1
        successes[arm] += reward
        total_successes += int(reward >= 0.5)
        cumulative_runtime += float(row.get("wallclock_s", 0.0) or 0.0)
        cumulative_cost += token_cost(row.get("tokens", {}))
        cs.setdefault(arm, ConfidenceSequence(alpha=alpha, grid_size=64)).update(reward)
        e_vs_half.setdefault(arm, HedgedCapitalEProcess(m0=0.5)).update(reward)
        e_vs_runner_up.setdefault(arm, HedgedCapitalEProcess(m0=clamp_m0(best_nonoracle_rate))).update(
            reward
        )

        empirical_rates = {
            aid: successes.get(aid, 0.0) / counts.get(aid, 0)
            for aid in arms
            if counts.get(aid, 0) >= min_leader_pulls
        }
        empirical_leader = unique_argmax(empirical_rates)
        most_pulled = unique_argmax({aid: float(n) for aid, n in counts.items() if n > 0})
        log_e_half = {aid: e_vs_half[aid].log_e_upper for aid in arms}
        log_e_runner_up = {aid: e_vs_runner_up[aid].log_e_upper for aid in arms}
        e_leader_half = unique_argmax(log_e_half)
        e_leader_runner_up = unique_argmax(log_e_runner_up)
        score_leader = policy_score_leader(row)
        cs_bounds = {aid: cs[aid].bounds() for aid in arms}
        oracle_cs_best = (
            cs_bounds[true_best][0] > max(hi for aid, (_lo, hi) in cs_bounds.items() if aid != true_best)
            if true_best in cs_bounds and len(cs_bounds) > 1
            else False
        )
        oracle_log_e_runner_up = log_e_runner_up[true_best]
        t = int(row["t"])
        oracle_pulls = counts.get(true_best, 0)
        oracle_successes = successes.get(true_best, 0.0)
        out_rows.append(
            {
                "base_run_id": bundle.base_run_id,
                "policy": bundle.policy,
                "replicate": bundle.replicate,
                "budget": bundle.budget,
                "complete": bundle.complete,
                "contiguous": bundle.contiguous,
                "timestep": t,
                "task_id": row.get("task_id"),
                "selected_arm": arm,
                "reward": reward,
                "cumulative_successes": total_successes,
                "cumulative_success_rate": total_successes / t,
                "cumulative_runtime": cumulative_runtime,
                "cumulative_cost": cumulative_cost,
                "oracle_pulls": oracle_pulls,
                "oracle_successes": oracle_successes,
                "oracle_success_rate": oracle_successes / oracle_pulls if oracle_pulls else math.nan,
                "oracle_pull_share": oracle_pulls / t,
                "empirical_leader_min_pulls": empirical_leader,
                "empirical_leader_is_oracle": empirical_leader == true_best,
                "most_pulled_arm": most_pulled,
                "most_pulled_is_oracle": most_pulled == true_best,
                "e_leader_m0_0_5": e_leader_half,
                "e_leader_m0_0_5_is_oracle": e_leader_half == true_best,
                "e_leader_vs_runner_up": e_leader_runner_up,
                "e_leader_vs_runner_up_is_oracle": e_leader_runner_up == true_best,
                "policy_score_leader": score_leader,
                "policy_score_leader_is_oracle": score_leader == true_best,
                "oracle_log_e_upper_vs_0_5": log_e_half[true_best],
                "oracle_log_e_upper_vs_runner_up": oracle_log_e_runner_up,
                "oracle_evalue_vs_runner_up": math.exp(min(oracle_log_e_runner_up, 700.0)),
                "oracle_beats_runner_up_eprocess": oracle_log_e_runner_up >= log_threshold,
                "oracle_cs_lo": cs_bounds[true_best][0],
                "oracle_cs_hi": cs_bounds[true_best][1],
                "oracle_cs_best_certified": oracle_cs_best,
                "alpha": alpha,
                "log_e_threshold": log_threshold,
            }
        )
    return pd.DataFrame(out_rows)


def summarize_run(
    bundle: RunBundle,
    ts: pd.DataFrame,
    *,
    true_best: str,
    oracle_rate: float,
    success_targets: list[int],
) -> dict[str, Any]:
    final = ts.sort_values("timestep").iloc[-1]
    row: dict[str, Any] = {
        "base_run_id": bundle.base_run_id,
        "policy": bundle.policy,
        "replicate": bundle.replicate,
        "budget": bundle.budget,
        "observed_timesteps": int(final["timestep"]),
        "complete": bundle.complete,
        "contiguous": bundle.contiguous,
        "true_best_arm": true_best,
        "successes_observed": int(final["cumulative_successes"]),
        "success_rate_observed": float(final["cumulative_success_rate"]),
        "total_runtime_observed": float(final["cumulative_runtime"]),
        "total_cost_observed": float(final["cumulative_cost"]),
        "final_oracle_pulls": int(final["oracle_pulls"]),
        "final_oracle_pull_share": float(final["oracle_pull_share"]),
        "final_oracle_successes": int(final["oracle_successes"]),
        "final_oracle_success_rate": safe_float(final["oracle_success_rate"]),
        "first_oracle_pull_t": first_time(ts["oracle_pulls"] > 0, ts),
        "first_oracle_3_pulls_t": first_time(ts["oracle_pulls"] >= 3, ts),
        "first_oracle_5_pulls_t": first_time(ts["oracle_pulls"] >= 5, ts),
        "first_oracle_10_pulls_t": first_time(ts["oracle_pulls"] >= 10, ts),
        "first_oracle_empirical_leader_t": first_time(ts["empirical_leader_is_oracle"], ts),
        "stable_oracle_empirical_leader_t": first_stable_time(ts["empirical_leader_is_oracle"], ts),
        "first_oracle_e_leader_0_5_t": first_time(ts["e_leader_m0_0_5_is_oracle"], ts),
        "stable_oracle_e_leader_0_5_t": first_stable_time(ts["e_leader_m0_0_5_is_oracle"], ts),
        "first_oracle_policy_score_leader_t": first_time(ts["policy_score_leader_is_oracle"], ts),
        "stable_oracle_policy_score_leader_t": first_stable_time(ts["policy_score_leader_is_oracle"], ts),
        "first_oracle_most_pulled_t": first_time(ts["most_pulled_is_oracle"], ts),
        "stable_oracle_most_pulled_t": first_stable_time(ts["most_pulled_is_oracle"], ts),
        "first_oracle_eprocess_beats_runner_up_t": first_time(
            ts["oracle_beats_runner_up_eprocess"], ts
        ),
        "first_oracle_cs_best_certified_t": first_time(ts["oracle_cs_best_certified"], ts),
    }
    decision_cols = [
        "first_oracle_empirical_leader_t",
        "stable_oracle_empirical_leader_t",
        "first_oracle_e_leader_0_5_t",
        "stable_oracle_e_leader_0_5_t",
        "first_oracle_policy_score_leader_t",
        "stable_oracle_policy_score_leader_t",
        "first_oracle_eprocess_beats_runner_up_t",
        "first_oracle_cs_best_certified_t",
    ]
    for col in decision_cols:
        add_decision_resources(row, col, ts, oracle_rate)
    for target in success_targets:
        hit_t = first_time(ts["cumulative_successes"] >= target, ts)
        row[f"first_successes_{target}_t"] = hit_t
        row[f"tasks_saved_after_successes_{target}"] = bundle.budget - hit_t if hit_t is not None else math.nan
    return row


def add_decision_resources(
    row: dict[str, Any], decision_col: str, ts: pd.DataFrame, oracle_rate: float
) -> None:
    decision_t = row.get(decision_col)
    prefix = decision_col.removesuffix("_t")
    if decision_t is None or pd.isna(decision_t):
        row[f"{prefix}_tasks_saved"] = math.nan
        row[f"{prefix}_cost_at_decision"] = math.nan
        row[f"{prefix}_runtime_at_decision"] = math.nan
        row[f"{prefix}_successes_at_decision"] = math.nan
        row[f"{prefix}_observed_cost_saved"] = math.nan
        row[f"{prefix}_projected_final_successes_if_commit_oracle"] = math.nan
        return
    decision = ts.loc[ts["timestep"] == int(decision_t)].iloc[-1]
    final = ts.sort_values("timestep").iloc[-1]
    budget = int(final["budget"])
    tasks_saved = budget - int(decision_t)
    row[f"{prefix}_tasks_saved"] = tasks_saved
    row[f"{prefix}_cost_at_decision"] = float(decision["cumulative_cost"])
    row[f"{prefix}_runtime_at_decision"] = float(decision["cumulative_runtime"])
    row[f"{prefix}_successes_at_decision"] = int(decision["cumulative_successes"])
    row[f"{prefix}_observed_cost_saved"] = (
        float(final["cumulative_cost"]) - float(decision["cumulative_cost"])
        if bool(final["complete"])
        else math.nan
    )
    row[f"{prefix}_projected_final_successes_if_commit_oracle"] = (
        float(decision["cumulative_successes"]) + tasks_saved * oracle_rate
    )


def summarize_by_policy(summary: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in summary.columns
        if c.endswith("_t")
        or c.endswith("_tasks_saved")
        or c.endswith("_observed_cost_saved")
        or c
        in {
            "successes_observed",
            "success_rate_observed",
            "total_runtime_observed",
            "total_cost_observed",
            "final_oracle_pulls",
            "final_oracle_pull_share",
        }
    ]
    agg = {c: ["mean", "median"] for c in numeric_cols}
    out = summary.groupby("policy").agg(agg)
    out.columns = ["_".join(col).strip("_") for col in out.columns]
    out = out.reset_index()
    for col in [
        "first_oracle_eprocess_beats_runner_up_t",
        "first_oracle_cs_best_certified_t",
        "stable_oracle_empirical_leader_t",
        "stable_oracle_policy_score_leader_t",
    ]:
        if col in summary:
            rate = summary[col].notna().groupby(summary["policy"]).mean()
            out = out.merge(
                rate.rename(f"{col}_rate").reset_index(),
                on="policy",
                how="left",
            )
    return out


def make_plots(timeseries: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path, true_best: str) -> None:
    if timeseries.empty:
        return
    averaged = (
        timeseries.groupby(["policy", "timestep"], as_index=False)
        .agg(
            cumulative_success_rate=("cumulative_success_rate", "mean"),
            oracle_pull_share=("oracle_pull_share", "mean"),
            oracle_log_e_upper_vs_runner_up=("oracle_log_e_upper_vs_runner_up", "mean"),
        )
        .sort_values(["policy", "timestep"])
    )
    lineplot(
        averaged,
        "timestep",
        "cumulative_success_rate",
        "Cumulative Success Rate",
        "Success rate",
        plot_dir / "cumulative_success_rate_time",
    )
    lineplot(
        averaged,
        "timestep",
        "oracle_pull_share",
        f"{true_best} Pull Share",
        "Pull share",
        plot_dir / "oracle_pull_share_time",
    )
    lineplot(
        averaged,
        "timestep",
        "oracle_log_e_upper_vs_runner_up",
        "Oracle Evidence vs Runner-Up Mean",
        "Mean log e-value",
        plot_dir / "oracle_evidence_vs_runner_up_time",
    )
    if summary.empty:
        return
    for col, title in [
        ("first_oracle_eprocess_beats_runner_up_t", "First E-Process Certification Time"),
        ("stable_oracle_empirical_leader_t", "Stable Empirical Leader Time"),
        ("stable_oracle_policy_score_leader_t", "Stable SPRUCE Score Leader Time"),
    ]:
        if col in summary:
            boxplot(summary, col, title, "Task index", plot_dir / col)


def lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    outbase: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for policy, g in df.groupby("policy"):
        ax.plot(g[x], g[y], linewidth=2, label=str(policy))
    ax.set_title(title)
    ax.set_xlabel("Task index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def boxplot(df: pd.DataFrame, y: str, title: str, ylabel: str, outbase: Path) -> None:
    values = []
    labels = []
    for policy, g in df.groupby("policy"):
        vals = pd.to_numeric(g[y], errors="coerce").dropna()
        if len(vals):
            values.append(vals.to_numpy())
            labels.append(str(policy))
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.boxplot(values, tick_labels=labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def write_report(
    analysis_dir: Path,
    *,
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    true_best: str,
    oracle_rate: float,
    best_nonoracle_rate: float,
    alpha: float,
    min_leader_pulls: int,
) -> None:
    lines = [
        "# GitLab 160 Early Identification Analysis",
        "",
        f"- True best arm from paired sweep: `{true_best}`.",
        f"- Paired success rate for true best: `{oracle_rate:.3f}`.",
        f"- Best non-oracle paired success rate: `{best_nonoracle_rate:.3f}`.",
        f"- E-process certification threshold: `log(1 / alpha) = {math.log(1 / alpha):.3f}` with `alpha={alpha}`.",
        f"- Empirical leader metrics require at least `{min_leader_pulls}` pulls for an arm.",
        "",
        "Key definitions:",
        "",
        "- `first_oracle_eprocess_beats_runner_up_t`: first task index where the oracle arm's one-sided hedged-capital e-process rejects the null that its mean is no better than the paired runner-up mean.",
        "- `stable_oracle_empirical_leader_t`: first task index after which the oracle remains the empirical success-rate leader among arms with enough pulls.",
        "- `stable_oracle_policy_score_leader_t`: first task index after which the oracle remains the top SPRUCE score arm, when policy scores are logged.",
        "- `*_tasks_saved`: remaining task budget if we committed to the oracle at that decision time.",
        "- `*_projected_final_successes_if_commit_oracle`: successes so far plus remaining budget times the paired oracle success rate.",
        "",
    ]
    if summary.empty:
        lines.append("No allocation rows were available yet.")
    else:
        visible_cols = [
            "policy",
            "replicate",
            "observed_timesteps",
            "complete",
            "successes_observed",
            "final_oracle_pulls",
            "first_oracle_eprocess_beats_runner_up_t",
            "stable_oracle_empirical_leader_t",
            "stable_oracle_policy_score_leader_t",
        ]
        lines.append("Current run-level summary:")
        lines.append("")
        lines.append(markdown_table(summary[[c for c in visible_cols if c in summary]]))
        lines.append("")
    if not policy_summary.empty:
        lines.append("Policy-level rollup:")
        lines.append("")
        compact_cols = [
            c
            for c in policy_summary.columns
            if c == "policy"
            or c.endswith("_rate")
            or c
            in {
                "success_rate_observed_mean",
                "final_oracle_pull_share_mean",
                "first_oracle_eprocess_beats_runner_up_t_mean",
                "stable_oracle_empirical_leader_t_mean",
                "stable_oracle_policy_score_leader_t_mean",
            }
        ]
        lines.append(markdown_table(policy_summary[compact_cols]))
        lines.append("")
    if not timeseries.empty:
        lines.append("Generated artifacts:")
        lines.append("")
        lines.append("- `identification_timeseries.csv`")
        lines.append("- `identification_summary.csv`")
        lines.append("- `identification_summary_by_policy.csv`")
        lines.append("- `plots/`")
    (analysis_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(format_cell)
    headers = [str(c) for c in clean.columns]
    rows = clean.values.tolist()
    widths = [
        max(len(headers[idx]), *(len(str(row[idx])) for row in rows))
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.3f}"
        return str(value)
    return str(value)


def token_cost(tokens: Any) -> float:
    if not isinstance(tokens, dict):
        return 0.0
    return float(tokens.get("cost_usd", 0.0) or 0.0)


def policy_score_leader(row: dict[str, Any]) -> str | None:
    policy = row.get("policy")
    if not isinstance(policy, dict):
        return None
    scores = policy.get("scores")
    if not isinstance(scores, dict) or not scores:
        return None
    numeric = {str(k): float(v) for k, v in scores.items()}
    return unique_argmax(numeric)


def unique_argmax(values: dict[str, float]) -> str | None:
    finite = {k: v for k, v in values.items() if not pd.isna(v)}
    if not finite:
        return None
    best = max(finite.values())
    winners = sorted(k for k, v in finite.items() if v == best)
    return winners[0] if len(winners) == 1 else None


def first_time(mask: pd.Series, ts: pd.DataFrame) -> int | None:
    hits = ts.loc[mask.fillna(False), "timestep"]
    return int(hits.iloc[0]) if len(hits) else None


def first_stable_time(mask: pd.Series, ts: pd.DataFrame) -> int | None:
    values = list(mask.fillna(False))
    times = list(ts["timestep"])
    for idx, value in enumerate(values):
        if bool(value) and all(bool(v) for v in values[idx:]):
            return int(times[idx])
    return None


def clamp_m0(value: float) -> float:
    return min(max(float(value), 1e-6), 1.0 - 1e-6)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


if __name__ == "__main__":
    main()
