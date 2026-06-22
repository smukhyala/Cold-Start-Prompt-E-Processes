#!/usr/bin/env bash
# Run the 120-task uniform baseline, then the 120-task adaptive SPRUCE run.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs/comparison_sweeps_120
TRACE_LOG="logs/comparison_sweeps_120/resume_uniform_then_adaptive_120_trace.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 120-task comparison wrapper started pid=$$" >> "$TRACE_LOG"
trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed at line ${LINENO}: ${BASH_COMMAND}" >> "$TRACE_LOG"' ERR

bash scripts/resume_uniform_multiarm_120.sh
bash scripts/resume_adaptive_spruce_120.sh

bash scripts/finalize_uniform_multiarm_120.sh
bash scripts/finalize_adaptive_spruce_120.sh

echo "120-task uniform + adaptive comparison runs complete"
