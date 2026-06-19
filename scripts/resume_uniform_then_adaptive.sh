#!/usr/bin/env bash
# Run the 60-task uniform multi-arm baseline, then the 60-task adaptive SPRUCE run.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs/comparison_sweeps
TRACE_LOG="logs/comparison_sweeps/resume_uniform_then_adaptive_trace.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] comparison wrapper started pid=$$" >> "$TRACE_LOG"
trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed at line ${LINENO}: ${BASH_COMMAND}" >> "$TRACE_LOG"' ERR

bash scripts/resume_uniform_multiarm.sh
bash scripts/resume_adaptive_spruce.sh

bash scripts/finalize_uniform_multiarm.sh
bash scripts/finalize_adaptive_sweep.sh

echo "uniform + adaptive comparison runs complete"
