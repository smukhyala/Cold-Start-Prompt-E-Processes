"""Render a compact thesis figure from the saved offline portfolio tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from task_transfer import OUTPUT


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUTPUT / "agent_portfolio")
    args = parser.parse_args()
    repeats = read_rows(args.results / "repeat_summary.csv")
    portfolios = read_rows(args.results / "portfolio_summary.csv")
    breadth = read_rows(args.results / "breadth_summary.csv")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 200})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), layout="constrained")
    pools = ["G", "F", "GLK30"]
    labels = ["Gmail G", "Gmail F", "GitLab K"]
    lookup = {(r["pool"], r["metric"]): r for r in repeats}
    y = np.arange(len(pools))
    point = np.asarray([float(lookup[p, "within_delta"]["estimate"]) for p in pools]) * 100
    low = np.asarray([float(lookup[p, "within_delta"]["task_boot_lo"]) for p in pools]) * 100
    high = np.asarray([float(lookup[p, "within_delta"]["task_boot_hi"]) for p in pools]) * 100
    axes[0].axvline(0, color="0.55", lw=1)
    axes[0].errorbar(point, y, xerr=[point - low, high - point], fmt="o", color="#23669b",
                     capsize=3, lw=1.6)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Coverage gain, percentage points")
    axes[0].set_title("A  Distinct agent vs rerun")
    axes[0].text(0.02, -0.2, "Task-bootstrap 95% intervals", transform=axes[0].transAxes,
                 fontsize=8, color="0.35")

    pools = ["G", "F", "GLK30", "GitLab17"]
    labels = ["Gmail G", "Gmail F", "GitLab K", "Older GitLab"]
    lookup = {(r["pool"], int(r["team_size"]), r["method"]): r for r in portfolios}
    top = np.asarray([float(lookup[p, 2, "top_marginal"]["minus_random"]) for p in pools]) * 100
    greedy = np.asarray([float(lookup[p, 2, "greedy_coverage"]["minus_random"]) for p in pools]) * 100
    x = np.arange(len(pools))
    axes[1].axhline(0, color="0.55", lw=1)
    axes[1].bar(x - .18, top, width=.34, label="Top two", color="#23669b")
    axes[1].bar(x + .18, greedy, width=.34, label="Coverage greedy", color="#dc8f39")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("Held-out gain vs random pair, pp")
    axes[1].set_title("B  Pair selection on reused tasks")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    for pool, label, color in zip(pools, labels, ("#23669b", "#dc8f39", "#2a8f73", "#7a5ca6"), strict=True):
        rows = sorted((r for r in breadth if r["pool"] == pool and int(r["budget"]) == 96),
                      key=lambda r: int(r["n_candidates"]))
        ks = [int(r["n_candidates"]) for r in rows]
        gain = [(float(r["selected_single_test"]) - float(r["random_candidate_test"])) * 100
                for r in rows]
        axes[2].plot(ks, gain, marker="o", lw=1.5, label=label, color=color)
    axes[2].axhline(0, color="0.55", lw=1)
    axes[2].set_xticks([8, 16, 32])
    axes[2].set_xlabel("Candidates tested; 96 training episodes")
    axes[2].set_ylabel("Selected agent gain vs random, pp")
    axes[2].set_title("C  Breadth at fixed budget")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("Existing trials: limited transferable value in three newer prompt pools",
                 fontsize=12, fontweight="semibold")
    fig.text(.5, -.045, "A: oracle coverage; B and C: exploratory views of the same reused task banks",
             ha="center", fontsize=8, color="0.35")
    for suffix in ("png", "pdf"):
        fig.savefig(args.results / f"agent_portfolio_overview.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
