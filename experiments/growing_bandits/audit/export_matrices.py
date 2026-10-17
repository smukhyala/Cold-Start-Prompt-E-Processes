#!/usr/bin/env python3
"""Export long-format CSVs for each real prompt pool."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import sys
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.growing_bandits.empirical import replay as emp_replay
from experiments.growing_bandits.empirical import study as emp_study


@dataclass
class ExportStats:
    pool: str
    n_rows: int
    n_arms: int
    n_tasks: int
    n_primary_rows: int
    n_replicate_rows: int
    mean_success_primary: float
    n_dropped: int
    source: str


def load_snapshot_data(pool_names: list[str]) -> pd.DataFrame:
    """Load empirical pool snapshot and filter by pool names."""
    res_dir = ROOT / "data" / "empirical_pool" / "reservoirs"
    df = emp_replay.load_snapshot(res_dir)
    filtered = df[df["pool"].isin(pool_names)].copy()
    return filtered


def load_glk_data(glk_type: str) -> pd.DataFrame:
    """Load GLK or GLK30 heterogeneity snapshot and filter by pool."""
    if glk_type == "GLK30":
        res_dir = ROOT / "data" / "heterogeneity" / "reservoirs_glk30"
        pool_filter = "GLK"
    else:  # GLK
        res_dir = ROOT / "data" / "heterogeneity" / "reservoirs_glk"
        pool_filter = "GLK"

    rows = []
    path = res_dir / "outcomes_snapshot.jsonl"
    with path.open() as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj["pool"] == pool_filter:
                    rows.append(obj)

    df = pd.DataFrame(rows)
    df["pool"] = glk_type
    return df


def load_gitlab_paired_data() -> pd.DataFrame:
    """Load GitLab paired results (18 arms) with is_cookbook column."""
    path = ROOT / "results" / "gitlab_strong_arm" / "paired" / "paired_results.csv"
    df = pd.read_csv(path)

    # Rename task column to task_id
    df = df.rename(columns={"task": "task_id"})

    # Add replicate column (all primary)
    df["replicate"] = 0

    # Add is_cookbook column
    df["is_cookbook"] = (df["arm_id"] == "gitlab_oracle_operator").astype(int)

    # Add pool column
    df["pool"] = "gitlab18"

    return df


def load_paired_sweep_data(env: str) -> pd.DataFrame:
    """Load paired sweep data (Gmail or Shopping)."""
    if env == "gmail":
        pattern = "*_FULL.jsonl"
        pool_name = "gmail_paired12"
        logs_dir = ROOT / "logs" / "paired_sweep"
        expected_tasks = 60
    else:  # shopping
        pattern = "*FULL80*.jsonl"
        pool_name = "shopping_paired12"
        logs_dir = ROOT / "logs" / "paired_sweep_shopping"
        expected_tasks = 80

    rows = []
    for fpath in sorted(logs_dir.glob(pattern)):
        with fpath.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    row = {
                        "pool": pool_name,
                        "arm_id": obj.get("arm_id"),
                        "task_id": obj.get("task_id"),
                        "replicate": 0,
                        "success": obj.get("success", 0),
                        "steps": obj.get("steps"),
                        "cost_usd": obj.get("tokens", {}).get("cost_usd"),
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


def transform_snapshot_to_csv(df: pd.DataFrame, pool_name: str) -> tuple[pd.DataFrame, int]:
    """Transform snapshot data to CSV format.

    Returns:
        tuple of (output DataFrame, number of dropped rows)
    """
    # Filter by status = "ok" (terminal results)
    dropped = len(df[df["status"] != "ok"])
    df = df[df["status"] == "ok"].copy()

    # Select columns and rename
    result = pd.DataFrame({
        "pool": df["pool"],
        "arm_id": df["arm_id"],
        "task_id": df["task_id"],
        "replicate": df["replicate"],
        "success": df["success"].astype(int),
        "steps": pd.NA,  # not available in snapshot
        "cost_usd": pd.NA,  # not available in snapshot
    })

    return result, dropped


def transform_glk_to_csv(df: pd.DataFrame, pool_name: str) -> tuple[pd.DataFrame, int]:
    """Transform GLK data to CSV format."""
    # Filter by status = "ok"
    dropped = len(df[df["status"] != "ok"])
    df = df[df["status"] == "ok"].copy()

    result = pd.DataFrame({
        "pool": pool_name,
        "arm_id": df["arm_id"],
        "task_id": df["task_id"],
        "replicate": df["replicate"],
        "success": df["success"].astype(int),
        "steps": pd.NA,
        "cost_usd": pd.NA,
    })

    return result, dropped


def transform_gitlab_to_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Transform GitLab paired data to CSV format."""
    # All rows are terminal (already in the CSV)
    dropped = 0

    result = pd.DataFrame({
        "pool": df["pool"],
        "arm_id": df["arm_id"],
        "task_id": df["task_id"],
        "replicate": df["replicate"],
        "success": df["success"].astype(int),
        "steps": df["steps"].fillna(pd.NA),
        "cost_usd": df["cost"].fillna(pd.NA),
    })

    return result, dropped


def transform_paired_sweep_to_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Transform paired sweep data to CSV format."""
    # All rows are terminal
    dropped = 0

    result = pd.DataFrame({
        "pool": df["pool"],
        "arm_id": df["arm_id"],
        "task_id": df["task_id"],
        "replicate": df["replicate"],
        "success": df["success"].astype(int),
        "steps": df["steps"].fillna(pd.NA),
        "cost_usd": df["cost_usd"].fillna(pd.NA),
    })

    return result, dropped


def compute_stats(df: pd.DataFrame, source: str) -> ExportStats:
    """Compute summary statistics from exported DataFrame."""
    pool = df["pool"].iloc[0]
    n_rows = len(df)
    n_arms = df["arm_id"].nunique()
    n_tasks = df["task_id"].nunique()

    primary = df[df["replicate"] == 0]
    n_primary = len(primary)
    n_replicate = n_rows - n_primary

    mean_success_primary = primary["success"].mean()

    return ExportStats(
        pool=pool,
        n_rows=n_rows,
        n_arms=n_arms,
        n_tasks=n_tasks,
        n_primary_rows=n_primary,
        n_replicate_rows=n_replicate,
        mean_success_primary=mean_success_primary,
        n_dropped=0,  # to be filled per pool
        source=source,
    )


def export_pool(pool_name: str, out_dir: Path) -> ExportStats:
    """Export a single pool and return stats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if pool_name == "G":
        df = load_snapshot_data(["G"])
        result, dropped = transform_snapshot_to_csv(df, pool_name)
        source = "data/empirical_pool/reservoirs/outcomes_snapshot.jsonl"
    elif pool_name == "F":
        df = load_snapshot_data(["F"])
        result, dropped = transform_snapshot_to_csv(df, pool_name)
        source = "data/empirical_pool/reservoirs/outcomes_snapshot.jsonl"
    elif pool_name == "gmail_anchor":
        df = load_snapshot_data(["anchor"])
        result, dropped = transform_snapshot_to_csv(df, pool_name)
        result["pool"] = "gmail_anchor"
        source = "data/empirical_pool/reservoirs/outcomes_snapshot.jsonl"
    elif pool_name == "GLK30":
        df = load_glk_data("GLK30")
        result, dropped = transform_glk_to_csv(df, pool_name)
        source = "data/heterogeneity/reservoirs_glk30/outcomes_snapshot.jsonl"
    elif pool_name == "GLK":
        df = load_glk_data("GLK")
        result, dropped = transform_glk_to_csv(df, pool_name)
        source = "data/heterogeneity/reservoirs_glk/outcomes_snapshot.jsonl"
    elif pool_name == "gitlab18":
        df = load_gitlab_paired_data()
        result, dropped = transform_gitlab_to_csv(df)
        source = "results/gitlab_strong_arm/paired/paired_results.csv"
    elif pool_name == "gmail_paired12":
        df = load_paired_sweep_data("gmail")
        result, dropped = transform_paired_sweep_to_csv(df)
        source = "logs/paired_sweep/*_FULL.jsonl"
    elif pool_name == "shopping_paired12":
        df = load_paired_sweep_data("shopping")
        result, dropped = transform_paired_sweep_to_csv(df)
        source = "logs/paired_sweep_shopping/*FULL80*.jsonl"
    else:
        raise ValueError(f"Unknown pool: {pool_name}")

    # Write to CSV
    csv_path = out_dir / f"{pool_name}.csv"
    result.to_csv(csv_path, index=False)

    stats = compute_stats(result, source)
    stats.n_dropped = dropped

    return stats


def export_all(out_dir: Path) -> None:
    """Export all pools and create summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = ["G", "F", "gmail_anchor", "GLK30", "GLK", "gitlab18", "gmail_paired12", "shopping_paired12"]

    all_stats = []
    for pool in pools:
        try:
            stats = export_pool(pool, out_dir)
            all_stats.append(stats)
            print(f"Exported {pool}: {stats.n_rows} rows, {stats.n_arms} arms, {stats.n_tasks} tasks")
        except FileNotFoundError as e:
            print(f"Skipping {pool}: {e}")

    # Write summary
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pool",
                "n_rows",
                "n_arms",
                "n_tasks",
                "n_primary_rows",
                "n_replicate_rows",
                "mean_success_primary",
                "n_dropped",
                "source",
            ],
        )
        writer.writeheader()
        for stats in all_stats:
            writer.writerow({
                "pool": stats.pool,
                "n_rows": stats.n_rows,
                "n_arms": stats.n_arms,
                "n_tasks": stats.n_tasks,
                "n_primary_rows": stats.n_primary_rows,
                "n_replicate_rows": stats.n_replicate_rows,
                "mean_success_primary": f"{stats.mean_success_primary:.4f}",
                "n_dropped": stats.n_dropped,
                "source": stats.source,
            })

    print(f"\nSummary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export long-format CSVs for prompt pools")
    parser.add_argument("--out", default=str(ROOT / "results" / "audit" / "matrices"),
                        help="Output directory for matrices")
    args = parser.parse_args()

    out_dir = Path(args.out)
    export_all(out_dir)


if __name__ == "__main__":
    main()
