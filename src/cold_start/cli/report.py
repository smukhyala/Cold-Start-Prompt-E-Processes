"""CLI entrypoint: produce a markdown + plot report for a run log.

By default loads the standard arm catalog (configs/arms_initial.yaml and
configs/axes.yaml) so the report can include the prompt-distance section
introduced in Mukhyala & Waudby-Smith § 3.2. Pass `--no-arm-geometry` to skip
the distance computation, or `--arms`/`--axes` to point at non-default files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cold_start.analysis.load import load_run
from cold_start.analysis.plots import (
    plot_arm_pulls,
    plot_cs_width,
    plot_global_log_e,
    plot_log_e_per_arm,
    plot_prompt_distance,
)
from cold_start.analysis.summary import distance_matrix, summary_markdown
from cold_start.prompts.axes import load_axes
from cold_start.prompts.catalog import load_arms


def main() -> None:
    parser = argparse.ArgumentParser(prog="cold-start-report")
    parser.add_argument("--log", required=True, type=Path, help="JSONL log file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for report outputs")
    parser.add_argument(
        "--arms",
        type=Path,
        default=Path("configs/arms_initial.yaml"),
        help="Arm catalog YAML (for the prompt-distance section)",
    )
    parser.add_argument(
        "--axes",
        type=Path,
        default=Path("configs/axes.yaml"),
        help="Axes spec YAML (for the prompt-distance section)",
    )
    parser.add_argument(
        "--no-arm-geometry",
        action="store_true",
        help="Skip loading arms/axes and omit the prompt-distance section",
    )
    args = parser.parse_args()

    df = load_run(args.log)
    out_dir = args.out_dir or (Path("reports") / args.log.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = None
    axes = None
    if not args.no_arm_geometry:
        try:
            axes = load_axes(args.axes)
            arms = load_arms(args.arms, axes)
        except FileNotFoundError:
            # Reporting a run that pre-dates the catalog files shouldn't fail
            # the report — just emit a note and continue without geometry.
            print(
                f"note: arms/axes not found at {args.arms}/{args.axes}; "
                "skipping arm-geometry section"
            )

    summary = summary_markdown(df, arms=arms, axes=axes)
    (out_dir / "summary.md").write_text(summary)

    plot_log_e_per_arm(df, out_dir / "log_e_per_arm.png")
    plot_global_log_e(df, out_dir / "global_log_e.png")
    plot_arm_pulls(df, out_dir / "arm_pulls.png")
    plot_cs_width(df, out_dir / "cs_width.png")

    if arms is not None and axes is not None:
        plot_prompt_distance(distance_matrix(arms, axes), out_dir / "prompt_distance.png")

    print(f"report written to {out_dir}")


if __name__ == "__main__":
    main()
