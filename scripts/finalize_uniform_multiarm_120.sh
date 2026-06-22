#!/usr/bin/env bash
set -Eeuo pipefail

export STATUS_PATH="docs/uniform_multiarm_120_status.md"
export LOG_DIR="logs/uniform_multiarm_120"
export TRIAL_NAME="webarena_gmail_uniform_multiarm_120"
export REPORT_DIR="reports/uniform_multiarm/round_robin_120"
export TARGET_T="120"
export RUN_LABEL="uniform_multiarm_120"
export RUN_TITLE="Uniform Multi-Arm 120-Task Status"
export POLICY_DESCRIPTION="- Sampler: uniform round-robin across all 12 arms. Budget: 120 task executions, giving 10 pulls per arm over two cycles of the 60-task Gmail bank."

bash scripts/finalize_policy_run.sh
