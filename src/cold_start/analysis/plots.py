"""Plots for run analysis. All functions return a Matplotlib Figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_log_e_per_arm(df: pd.DataFrame, out: str | Path | None = None):
    """Line chart of each arm's log-e-value over time."""
    arm_ids = _arm_ids(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid in arm_ids:
        series = df["per_arm_state"].apply(lambda d: d.get(aid, {}).get("log_e", 0.0))
        ax.plot(df["t"], series, label=aid)
    ax.set_xlabel("t")
    ax.set_ylabel("log E_t per arm")
    ax.set_title("Per-arm log-e-value over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out)
    return fig


def plot_global_log_e(df: pd.DataFrame, out: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    series = df["global_e"].apply(lambda d: d.get("log_e", 0.0))
    ax.plot(df["t"], series, color="black")
    ax.set_xlabel("t")
    ax.set_ylabel("log E_t global")
    ax.set_title("Global-null log-e-value over time")
    ax.grid(True, alpha=0.3)
    _save(fig, out)
    return fig


def plot_arm_pulls(df: pd.DataFrame, out: str | Path | None = None):
    """Stacked line of pulls per arm over time."""
    arm_ids = _arm_ids(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid in arm_ids:
        pulls = df["per_arm_state"].apply(lambda d: d.get(aid, {}).get("pulls", 0))
        ax.plot(df["t"], pulls, label=aid)
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative pulls")
    ax.set_title("Arm pull counts over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out)
    return fig


def plot_cs_width(df: pd.DataFrame, out: str | Path | None = None):
    arm_ids = _arm_ids(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid in arm_ids:
        width = df["per_arm_state"].apply(
            lambda d: d.get(aid, {}).get("cs_hi", 1.0) - d.get(aid, {}).get("cs_lo", 0.0)
        )
        ax.plot(df["t"], width, label=aid)
    ax.set_xlabel("t")
    ax.set_ylabel("CS width")
    ax.set_title("Confidence-sequence width per arm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out)
    return fig


def plot_prompt_distance(
    matrix: pd.DataFrame,
    out: str | Path | None = None,
):
    """Heatmap of pairwise prompt distances d(a,b) over the arm catalog.

    `matrix` is the DataFrame returned by `analysis.summary.distance_matrix`.
    """
    arr = np.asarray(matrix.values, dtype=float)
    ids = list(matrix.columns)
    fig, ax = plt.subplots(figsize=(0.6 * len(ids) + 2, 0.6 * len(ids) + 2))
    im = ax.imshow(arr, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(ids)), ids, rotation=45, ha="right")
    ax.set_yticks(range(len(ids)), ids)
    ax.set_title("Pairwise prompt distance d(a,b)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Annotate cells when the catalog is small enough to read.
    if len(ids) <= 16:
        for i in range(len(ids)):
            for j in range(len(ids)):
                ax.text(
                    j, i, f"{arr[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() * 0.5 else "black",
                    fontsize=7,
                )
    fig.tight_layout()
    _save(fig, out)
    return fig


def _arm_ids(df: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    for d in df["per_arm_state"]:
        seen.update(d.keys())
    return sorted(seen)


def _save(fig, out):
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=120)
