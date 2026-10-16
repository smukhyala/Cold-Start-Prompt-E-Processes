"""Presentation-only zoom of the saved real evidence; no statistical recomputation."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(directory: Path) -> None:
    evidence = json.loads((directory / "evidence.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if not evidence:
        raise ValueError("No real paired outcomes to plot")
    x = [0] + [p["pair_index"] for p in evidence]
    stop = summary["first_crossing_pair"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
        ax.plot(x, [0] + [r["log_efficacy"] for r in evidence], color="#0072B2", lw=1.8,
                label=f"Efficacy (final E = {summary['final_efficacy']:.2f})")
        ax.plot(x, [0] + [r["log_futility"] for r in evidence], color="#D55E00", lw=1.8,
                ls="--", label=f"Practical futility (final E = {summary['final_futility']:.3f})")
        ax.axhline(math.log(summary["threshold_efficacy"]), color="0.35", ls=":", lw=1.4,
                   label="Registered threshold: log(40)")
        ax.axvline(x[-1], color="0.5", ls="-.", lw=1)
        note = (f"Statistical stop at pair {stop}" if stop else
                f"Operational halt\n{summary['completed_pairs']} complete pairs; next pair incomplete")
        ax.annotate(note, xy=(x[-1], 1.3), xytext=(x[-1] - 0.5, 1.65),
                    ha="right", fontsize=9, arrowprops={"arrowstyle": "-", "color": "0.5"})
        ax.set(xlim=(0, x[-1] + .6), ylim=(-.15, 4.05), xlabel="Completed task pairs",
               ylabel="Natural log evidence",
               title=f"{summary['decision'].capitalize()} — planned maximum {summary['max_pairs']} pairs")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0, .88), fontsize=8.5, frameon=False)
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(directory / f"live_stopping_observed_prefix.{suffix}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/paper/live_stopping"))
    plot(parser.parse_args().results)
