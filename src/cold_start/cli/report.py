"""CLI entrypoint: produce a markdown + plot report for a run log."""

from __future__ import annotations

import argparse
from pathlib import Path

from cold_start.analysis.load import load_run
from cold_start.analysis.plots import (
    plot_arm_pulls,
    plot_cs_width,
    plot_global_log_e,
    plot_log_e_per_arm,
)
from cold_start.analysis.summary import summary_markdown


def main() -> None:
    parser = argparse.ArgumentParser(prog="cold-start-report")
    parser.add_argument("--log", required=True, type=Path, help="JSONL log file")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for report outputs")
    args = parser.parse_args()

    df = load_run(args.log)
    out_dir = args.out_dir or (Path("reports") / args.log.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summary_markdown(df)
    (out_dir / "summary.md").write_text(summary)

    plot_log_e_per_arm(df, out_dir / "log_e_per_arm.png")
    plot_global_log_e(df, out_dir / "global_log_e.png")
    plot_arm_pulls(df, out_dir / "arm_pulls.png")
    plot_cs_width(df, out_dir / "cs_width.png")

    print(f"report written to {out_dir}")


if __name__ == "__main__":
    main()
