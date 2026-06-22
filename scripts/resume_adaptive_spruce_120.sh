#!/usr/bin/env bash
set -Eeuo pipefail

export CONFIG="configs/webarena_gmail_adaptive_spruce_120.yaml"
export TRIAL_NAME="webarena_gmail_adaptive_spruce_120"
export LOG_DIR="logs/adaptive_sweep_120"
export REPORT_DIR="reports/adaptive_sweep/spruce_120"
export STDOUT_LOG="logs/adaptive_sweep_120/adaptive_stdout.log"
export TARGET_T="120"
export FINALIZE_SCRIPT="scripts/finalize_adaptive_spruce_120.sh"

bash scripts/resume_policy_run.sh
