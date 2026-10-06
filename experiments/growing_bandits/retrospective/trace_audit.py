"""Fixed small audit of saved histories; reports observable actions, not causal labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from agent_portfolio import LOGS, load_log_metrics, write_csv
from task_transfer import OUTPUT

SEED = 20261006


def actions(path: str) -> Counter[str]:
    history_path = Path(path) / "history.json"
    history = json.loads(history_path.read_text())["history"]
    if not isinstance(history, list) or not history:
        raise ValueError(f"empty or invalid history: {history_path}")
    counts: Counter[str] = Counter()
    for step in history:
        model_output = step.get("model_output") or {}
        for item in model_output.get("action") or []:
            counts.update(item.keys())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUTPUT / "agent_portfolio")
    args = parser.parse_args()
    with (args.results / "repeat_cells.csv").open(newline="") as handle:
        cells = list(csv.DictReader(handle))
    rng = np.random.default_rng(SEED)
    output = []
    for pool in ("G", "F", "GLK30"):
        logs = load_log_metrics(LOGS[pool], "GLK" if pool == "GLK30" else pool)
        for repeat_success in (0, 1):
            eligible = [r for r in cells if r["pool"] == pool and int(r["first"]) == 0
                        and int(r["repeat"]) == repeat_success]
            if len(eligible) < 2:
                raise ValueError(f"too few cases: {pool}, repeat={repeat_success}")
            picks = rng.choice(len(eligible), size=2, replace=False)
            for pick in picks:
                row = eligible[int(pick)]
                first = logs[row["arm_id"], row["task_id"], 0]
                second = logs[row["arm_id"], row["task_id"], 1]
                a0, a1 = actions(first["trace_path"]), actions(second["trace_path"])
                output.append({"pool": pool, "arm_id": row["arm_id"], "task_id": row["task_id"],
                               "first_scored_success": 0, "repeat_scored_success": repeat_success,
                               "first_raw_success": first["raw_success"],
                               "repeat_raw_success": second["raw_success"],
                               "first_steps": first["steps"], "repeat_steps": second["steps"],
                               "first_timed_out": first["timed_out"],
                               "repeat_timed_out": second["timed_out"],
                               "first_actions": sum(a0.values()), "repeat_actions": sum(a1.values()),
                               "first_action_types": "|".join(sorted(a0)),
                               "repeat_action_types": "|".join(sorted(a1))})
    write_csv(args.results / "trace_audit_sample.csv", output)
    (args.results / "trace_audit_method.json").write_text(json.dumps({
        "seed": SEED, "rule": "two cells per pool from each first-fail/repeat-fail and first-fail/repeat-success stratum",
        "n_cells": len(output), "outcome_role": "descriptive audit; selected on repeat outcome; no causal coding",
    }, indent=2) + "\n")
    print(f"audited {len(output)} saved trace pairs")


if __name__ == "__main__":
    main()
