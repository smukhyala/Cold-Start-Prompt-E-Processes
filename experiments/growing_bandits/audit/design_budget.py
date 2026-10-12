#!/usr/bin/env python3
"""Exact paired-grid planning costs; no efficacy or stopping-time prediction."""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challengers", type=int, required=True)
    ap.add_argument("--tasks", type=int, required=True)
    ap.add_argument("--episode-cost", type=float, default=0.0, help="currency units per arm-task episode")
    ap.add_argument("--episode-seconds", type=float, default=0.0)
    ap.add_argument("--parallelism", type=int, default=1)
    a = ap.parse_args()
    if a.challengers < 1 or a.tasks < 1 or a.episode_cost < 0 or a.episode_seconds < 0 or a.parallelism < 1:
        ap.error("counts and parallelism must be positive; costs and seconds nonnegative")
    full = (a.challengers + 1) * a.tasks
    print(json.dumps({"challengers": a.challengers, "paired_tasks": a.tasks,
                      "full_grid_episodes": full, "incumbent_episodes": a.tasks,
                      "challenger_episodes": a.challengers * a.tasks,
                      "full_grid_cost": full * a.episode_cost,
                      "idealized_wall_seconds": math.ceil(full / a.parallelism) * a.episode_seconds,
                      "note": "Actual sequential stopping may use fewer episodes; no stopping-time prediction is validated."}, indent=2))


if __name__ == "__main__":
    main()
