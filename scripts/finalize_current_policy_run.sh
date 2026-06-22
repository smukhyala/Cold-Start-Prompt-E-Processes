#!/usr/bin/env bash
# Env-driven adapter for scripts/resume_policy_run.sh.

set -Eeuo pipefail

: "${STATUS_PATH:?}"
: "${LOG_DIR:?}"
: "${TRIAL_NAME:?}"
: "${REPORT_DIR:?}"
: "${TARGET_T:?}"
: "${RUN_LABEL:?}"
: "${RUN_TITLE:?}"
: "${POLICY_DESCRIPTION:?}"

bash scripts/finalize_policy_run.sh
