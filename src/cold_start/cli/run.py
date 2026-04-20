"""CLI entrypoint: `python -m cold_start.cli.run --config configs/pilot_toy.yaml`."""

from __future__ import annotations

import argparse
from pathlib import Path

from cold_start.cli import _bootstrap  # noqa: F401  (registry population)
from cold_start.config import load_config
from cold_start.runner.orchestrator import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(prog="cold-start-run")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override trial.seed")
    parser.add_argument("--trials", type=int, default=None, help="Override trial.num_trials")
    parser.add_argument("--out", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.trial.seed = args.seed
    if args.trials is not None:
        cfg.trial.num_trials = args.trials

    paths = []
    for i in range(cfg.trial.num_trials):
        p = run_trial(cfg, trial_idx=i, log_dir=args.out or cfg.trial.output_dir)
        paths.append(p)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
