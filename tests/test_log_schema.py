from __future__ import annotations

import json

from cold_start.analysis.load import load_run
from cold_start.config import load_config
from cold_start.runner.orchestrator import run_trial


REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "t",
    "timestamp_utc",
    "task_id",
    "arm_id",
    "prompt_vector",
    "prompt_id",
    "success",
    "reward",
    "steps",
    "wallclock_s",
    "tokens",
    "policy",
    "per_arm_state",
    "global_e",
    "inference",
    "seed",
    "config_hash",
}


def test_log_schema_fields_present(tmp_log_dir):
    cfg = load_config("configs/pilot_toy.yaml")
    cfg.trial.num_tasks = 30
    cfg.trial.output_dir = str(tmp_log_dir)
    log_path = run_trial(cfg, trial_idx=0, log_dir=tmp_log_dir)

    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        missing = REQUIRED_FIELDS - rec.keys()
        assert not missing, f"record missing fields: {missing}"
    df = load_run(log_path)
    assert len(df) == cfg.trial.num_tasks
