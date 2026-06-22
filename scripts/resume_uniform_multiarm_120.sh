#!/usr/bin/env bash
set -Eeuo pipefail

export CONFIG="configs/webarena_gmail_uniform_multiarm_120.yaml"
export TRIAL_NAME="webarena_gmail_uniform_multiarm_120"
export LOG_DIR="logs/uniform_multiarm_120"
export REPORT_DIR="reports/uniform_multiarm/round_robin_120"
export STDOUT_LOG="logs/uniform_multiarm_120/uniform_stdout.log"
export TARGET_T="120"
export FINALIZE_SCRIPT="scripts/finalize_uniform_multiarm_120.sh"

bash scripts/resume_policy_run.sh
