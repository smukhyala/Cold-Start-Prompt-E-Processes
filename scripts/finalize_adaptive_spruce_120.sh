#!/usr/bin/env bash
set -Eeuo pipefail

export STATUS_PATH="docs/adaptive_sweep_120_status.md"
export LOG_DIR="logs/adaptive_sweep_120"
export TRIAL_NAME="webarena_gmail_adaptive_spruce_120"
export REPORT_DIR="reports/adaptive_sweep/spruce_120"
export TARGET_T="120"
export RUN_LABEL="adaptive_spruce_120"
export RUN_TITLE="Adaptive SPRUCE 120-Task Status"
export POLICY_DESCRIPTION="- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 120 task executions over two cycles of the 60-task Gmail bank."

bash scripts/finalize_policy_run.sh
