"""Publication figure from the complete saved amended continuation trajectory."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(directory):
    summary = json.loads((directory / "summary.json").read_text())
    evidence = json.loads((directory / "evidence.json").read_text())
    x = [0] + [r["pair_index"] for r in evidence]
    efficacy = [0] + [r["log_efficacy"] for r in evidence]
    futility = [0] + [r["log_futility"] for r in evidence]
    threshold = math.log(summary["threshold_efficacy"])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
        ax.axvspan(0, 13, color="0.94", zorder=0)
        ax.axvline(13, color="0.6", linewidth=.8, linestyle="--", label="Original 13-pair prefix")
        ax.plot(x, efficacy, color="#0072B2", linewidth=1.8, label="Efficacy")
        ax.plot(x, futility, color="#D55E00", linewidth=1.8, linestyle="--", label="Practical futility")
        ax.axhline(threshold, color="0.25", linestyle=":", linewidth=1.4, label="Threshold: log(40)")
        stop = summary["first_crossing_pair"]
        ax.axvline(x[-1], color="0.35", linestyle="-.", linewidth=1.0,
                   label=f"Online statistical stop: pair {stop}" if stop else "Operational / budget endpoint")
        ax.set(xlim=(0, x[-1] + max(1, x[-1] * .03)),
               ylim=(min(-.15, min(efficacy + futility) - .2), max(threshold, max(efficacy + futility)) + .55),
               xlabel="Completed task pairs", ylabel="Natural log evidence",
               title=f"Authorized amended continuation — {summary['decision']}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")
        for suffix in ("pdf", "svg", "png"):
            fig.savefig(directory / f"live_stopping_continuation.{suffix}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    plot(parser.parse_args().results)
