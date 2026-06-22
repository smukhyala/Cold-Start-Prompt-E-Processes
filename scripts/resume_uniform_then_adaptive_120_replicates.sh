#!/usr/bin/env bash
# Queue two additional 120-task uniform/adaptive replicates after the current pair.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs/comparison_sweeps_120_replicates
TRACE_LOG="logs/comparison_sweeps_120_replicates/resume_trace.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] follow-up replicate queue started pid=$$" >> "$TRACE_LOG"
trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed at line ${LINENO}: ${BASH_COMMAND}" >> "$TRACE_LOG"' ERR

current_pair_done() {
    bash scripts/finalize_uniform_multiarm_120.sh >/dev/null 2>&1 || true
    bash scripts/finalize_adaptive_spruce_120.sh >/dev/null 2>&1 || true
    python3 - <<'PY'
from pathlib import Path

checks = [
    (Path("docs/uniform_multiarm_120_status.md"), "uniform_multiarm_120"),
    (Path("docs/adaptive_sweep_120_status.md"), "adaptive_spruce_120"),
]
for path, run in checks:
    if not path.exists():
        raise SystemExit(1)
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 5 and parts[0] == run and parts[1] == "done":
            break
    else:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

wait_for_current_pair() {
    while ! current_pair_done; do
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for current 120-task pair to finish" | tee -a "$TRACE_LOG"
        sleep 300
    done
}

run_policy() {
    local kind="$1"
    local rep="$2"
    local label status title policy_desc

    if [[ "$kind" == "uniform" ]]; then
        label="uniform_multiarm_120_${rep}"
        status="docs/${label}_status.md"
        title="Uniform Multi-Arm 120-Task ${rep} Status"
        policy_desc="- Sampler: uniform round-robin across all 12 arms. Budget: 120 task executions, giving 10 pulls per arm over two cycles of the 60-task Gmail bank."
        export CONFIG="configs/webarena_gmail_uniform_multiarm_120_${rep}.yaml"
        export TRIAL_NAME="webarena_gmail_uniform_multiarm_120_${rep}"
        export LOG_DIR="logs/uniform_multiarm_120_${rep}"
        export REPORT_DIR="reports/uniform_multiarm/round_robin_120_${rep}"
        export STDOUT_LOG="${LOG_DIR}/uniform_stdout.log"
    else
        label="adaptive_spruce_120_${rep}"
        status="docs/${label}_status.md"
        title="Adaptive SPRUCE 120-Task ${rep} Status"
        policy_desc="- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 120 task executions over two cycles of the 60-task Gmail bank."
        export CONFIG="configs/webarena_gmail_adaptive_spruce_120_${rep}.yaml"
        export TRIAL_NAME="webarena_gmail_adaptive_spruce_120_${rep}"
        export LOG_DIR="logs/adaptive_sweep_120_${rep}"
        export REPORT_DIR="reports/adaptive_sweep/spruce_120_${rep}"
        export STDOUT_LOG="${LOG_DIR}/adaptive_stdout.log"
    fi

    export STATUS_PATH="$status"
    export TARGET_T="120"
    export RUN_LABEL="$label"
    export RUN_TITLE="$title"
    export POLICY_DESCRIPTION="$policy_desc"
    export FINALIZE_SCRIPT="scripts/finalize_current_policy_run.sh"

    bash "$FINALIZE_SCRIPT"
    bash scripts/resume_policy_run.sh
    bash "$FINALIZE_SCRIPT"
}

wait_for_current_pair

for rep in rep2 rep3; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${rep} uniform 120" | tee -a "$TRACE_LOG"
    run_policy uniform "$rep"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${rep} adaptive 120" | tee -a "$TRACE_LOG"
    run_policy adaptive "$rep"
done

echo "120-task follow-up replicates complete"
