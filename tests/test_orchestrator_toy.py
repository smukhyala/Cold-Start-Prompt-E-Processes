from __future__ import annotations

import json

from cold_start.config import load_config
from cold_start.runner.orchestrator import run_trial


def test_pilot_toy_end_to_end(tmp_log_dir):
    cfg = load_config("configs/pilot_toy.yaml")
    cfg.trial.num_tasks = 120
    cfg.trial.output_dir = str(tmp_log_dir)
    log_path = run_trial(cfg, trial_idx=0, log_dir=tmp_log_dir)

    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == cfg.trial.num_tasks
    last = json.loads(lines[-1])

    assert last["t"] == cfg.trial.num_tasks
    arms = last["per_arm_state"]
    total_pulls = sum(v["pulls"] for v in arms.values())
    assert total_pulls == cfg.trial.num_tasks

    # Expected: best arm (planner, p=0.6) accumulates more pulls than baseline (p=0.35).
    # Warm-start floors each at min_pulls=5, so check dominance past the warmstart phase.
    assert arms["planner"]["pulls"] > arms["baseline"]["pulls"]
