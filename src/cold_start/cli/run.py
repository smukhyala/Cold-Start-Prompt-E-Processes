"""CLI entrypoint: `python -m cold_start.cli.run --config configs/pilot_toy.yaml`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cold_start.cli import _bootstrap  # noqa: F401  (registry population)
from cold_start.config import load_config
from cold_start.runner.orchestrator import run_trial
from cold_start.runner.pre_flight import PreflightError, run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(prog="cold-start-run")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override trial.seed")
    parser.add_argument("--trials", type=int, default=None, help="Override trial.num_trials")
    parser.add_argument("--out", type=Path, default=None, help="Override output directory")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a prior run's JSONL. Its records at t < --start-at are replayed "
        "into e-process/policy state; the run continues from --start-at.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="First task index to execute (1-indexed). Used with --resume-from.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the pre-flight environment checks (API key, port, sibling repo). "
        "Useful when intentionally running against a non-standard setup; the run will "
        "still error inline if something is genuinely broken.",
    )
    args = parser.parse_args()

    if args.resume_from is not None and args.start_at <= 1:
        parser.error("--resume-from requires --start-at >= 2")
    if args.resume_from is None and args.start_at != 1:
        parser.error("--start-at > 1 requires --resume-from")

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.trial.seed = args.seed
    if args.trials is not None:
        cfg.trial.num_trials = args.trials

    if args.resume_from is not None and cfg.trial.num_trials != 1:
        parser.error(
            f"--resume-from only supports num_trials=1 (got {cfg.trial.num_trials}); "
            "resume is per-trial"
        )

    if not args.skip_preflight:
        try:
            run_preflight(cfg, output_dir=args.out)
        except PreflightError as exc:
            sys.stderr.write(f"pre-flight failed: {exc}\n")
            sys.exit(2)

    paths = []
    for i in range(cfg.trial.num_trials):
        p = run_trial(
            cfg,
            trial_idx=i,
            log_dir=args.out or cfg.trial.output_dir,
            resume_from=args.resume_from,
            start_at=args.start_at,
        )
        paths.append(p)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
