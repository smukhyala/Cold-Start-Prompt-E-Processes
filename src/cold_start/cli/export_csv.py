"""CLI entrypoint: flatten a run JSONL into spreadsheet-friendly CSVs.

Produces two files in `<out-dir>/`:

- `run.csv` — one row per task with the columns most useful for quick
  eyeballing (t, task_id, arm_id, success, reward, steps, wallclock,
  flattened tokens block, global e-process snapshot, run_id).

- `per_arm.csv` — one row per arm with the end-of-run per-arm statistics
  (pulls, successes, μ̂, log_e, CS bounds, rejection flag) plus
  cumulative cost and wallclock by arm. Derived from the *last* record's
  `per_arm_state` snapshot plus a groupby on the per-task frame.

The nested `per_arm_state`, `prompt_vector`, and `policy.scores` blocks are
omitted from `run.csv` — they balloon column counts and the JSONL keeps the
authoritative copy. The markdown report (`cold-start-report`) is the place
for the inference-flavored summary; this CLI is the place for a flat
spreadsheet.

Usage:
    cold-start-export-csv --log logs/<run_id>.jsonl \\
        [--out-dir reports/<run_id>/]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from cold_start.analysis.load import load_run


# Per-task fields lifted to top-level columns in run.csv. Anything nested
# (per_arm_state, prompt_vector, policy.scores, inference, by_model) is
# intentionally dropped — those live in the JSONL and the markdown summary.
_TASK_COLUMNS = [
    "run_id",
    "t",
    "task_id",
    "arm_id",
    "success",
    "reward",
    "steps",
    "wallclock_s",
    "tokens_input",
    "tokens_output",
    "tokens_cache_read",
    "tokens_cache_write",
    "tokens_cost_usd",
    "tokens_invocations",
    "global_log_e",
    "global_rejected",
    "policy_type",
    "config_hash",
]


def _flatten_task_row(row: pd.Series) -> dict[str, object]:
    """Project a single JSONL record into a flat dict for run.csv."""
    tokens = row.get("tokens") or {}
    if not isinstance(tokens, dict):
        tokens = {}
    global_e = row.get("global_e") or {}
    if not isinstance(global_e, dict):
        global_e = {}
    policy = row.get("policy") or {}
    if not isinstance(policy, dict):
        policy = {}
    return {
        "run_id": row.get("run_id"),
        "t": row.get("t"),
        "task_id": row.get("task_id"),
        "arm_id": row.get("arm_id"),
        "success": row.get("success"),
        "reward": row.get("reward"),
        "steps": row.get("steps"),
        "wallclock_s": row.get("wallclock_s"),
        "tokens_input": tokens.get("input"),
        "tokens_output": tokens.get("output"),
        "tokens_cache_read": tokens.get("cache_read"),
        "tokens_cache_write": tokens.get("cache_write"),
        "tokens_cost_usd": tokens.get("cost_usd"),
        "tokens_invocations": tokens.get("invocations"),
        "global_log_e": global_e.get("log_e"),
        "global_rejected": global_e.get("rejected"),
        "policy_type": policy.get("type"),
        "config_hash": row.get("config_hash"),
    }


def _build_task_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [_flatten_task_row(row) for _, row in df.iterrows()]
    out = pd.DataFrame(rows, columns=_TASK_COLUMNS)
    if "t" in out.columns:
        out = out.sort_values("t").reset_index(drop=True)
    return out


def _build_per_arm_frame(df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm end-of-run snapshot + cumulative cost/wallclock from the run."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "arm_id",
                "pulls",
                "successes",
                "mu_hat",
                "log_e",
                "cs_low",
                "cs_high",
                "rejected",
                "alpha",
                "m0",
                "total_cost_usd",
                "total_wallclock_s",
            ]
        )

    last = df.iloc[-1]
    pas = last.get("per_arm_state") or {}
    inference = last.get("inference") or {}
    alpha = float(inference.get("alpha", float("nan")))
    m0 = float(inference.get("m0", float("nan")))
    log_thresh = math.log(1.0 / alpha) if alpha and alpha > 0 else float("inf")

    # Cumulative cost / wallclock by arm — independent of the e-process state
    # because per_arm_state.pulls counts only that arm's draws, but cost
    # accumulates from the per-task tokens block.
    cost_by_arm = (
        task_df.groupby("arm_id")["tokens_cost_usd"].sum(min_count=1).fillna(0.0)
    )
    wall_by_arm = (
        task_df.groupby("arm_id")["wallclock_s"].sum(min_count=1).fillna(0.0)
    )

    rows: list[dict[str, object]] = []
    for arm_id in sorted(pas.keys()):
        s = pas[arm_id]
        pulls = int(s.get("pulls", 0))
        successes = int(s.get("successes", 0))
        mu_hat = (successes / pulls) if pulls else float("nan")
        log_e = float(s.get("log_e", 0.0))
        rows.append(
            {
                "arm_id": arm_id,
                "pulls": pulls,
                "successes": successes,
                "mu_hat": mu_hat,
                "log_e": log_e,
                "cs_low": float(s.get("cs_lo", float("nan"))),
                "cs_high": float(s.get("cs_hi", float("nan"))),
                "rejected": bool(log_e >= log_thresh),
                "alpha": alpha,
                "m0": m0,
                "total_cost_usd": float(cost_by_arm.get(arm_id, 0.0)),
                "total_wallclock_s": float(wall_by_arm.get(arm_id, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cold-start-export-csv",
        description="Flatten a run JSONL into run.csv + per_arm.csv.",
    )
    parser.add_argument("--log", required=True, type=Path, help="JSONL log file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: reports/<log stem>/)",
    )
    args = parser.parse_args()

    df = load_run(args.log)
    out_dir = args.out_dir or (Path("reports") / args.log.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_df = _build_task_frame(df)
    per_arm_df = _build_per_arm_frame(df, task_df)

    run_csv = out_dir / "run.csv"
    per_arm_csv = out_dir / "per_arm.csv"
    task_df.to_csv(run_csv, index=False)
    per_arm_df.to_csv(per_arm_csv, index=False)

    print(f"run.csv     ({len(task_df)} rows)  -> {run_csv}")
    print(f"per_arm.csv ({len(per_arm_df)} rows)  -> {per_arm_csv}")


if __name__ == "__main__":
    main()
