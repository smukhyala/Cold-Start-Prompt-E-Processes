#!/usr/bin/env bash
# Start 120-task uniform then adaptive SPRUCE comparison runs with watchdog.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs/comparison_sweeps_120 logs/uniform_multiarm_120 logs/adaptive_sweep_120

RUN_SCREEN="${RUN_SCREEN:-csp_uniform_then_adaptive_120}"
WATCHDOG_SCREEN="${WATCHDOG_SCREEN:-csp_uniform_then_adaptive_120_watchdog}"
CAFFEINE_SCREEN="${CAFFEINE_SCREEN:-csp_comparison_120_caffeinate}"

if ! screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${CAFFEINE_SCREEN}$"; then
    screen -dmS "$CAFFEINE_SCREEN" bash -lc "cd '$REPO_DIR' && while true; do date -u '+[caffeinate-120 %Y-%m-%dT%H:%M:%SZ] renewing 1h assertion' >> logs/comparison_sweeps_120/caffeinate.log; caffeinate -i -t 3600 >> logs/comparison_sweeps_120/caffeinate.log 2>&1; sleep 5; done"
fi

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${RUN_SCREEN}$"; then
    echo "run screen already exists: ${RUN_SCREEN}" >&2
else
    screen -dmS "$RUN_SCREEN" bash -lc "cd '$REPO_DIR' && bash scripts/resume_uniform_then_adaptive_120.sh > logs/comparison_sweeps_120/uniform_then_adaptive_120_stdout.log 2>&1"
fi

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${WATCHDOG_SCREEN}$"; then
    echo "watchdog screen already exists: ${WATCHDOG_SCREEN}" >&2
else
    screen -dmS "$WATCHDOG_SCREEN" bash -lc "cd '$REPO_DIR' && SCREEN_NAME='$RUN_SCREEN' bash scripts/watchdog_uniform_then_adaptive_120.sh >> logs/comparison_sweeps_120/watchdog_uniform_then_adaptive_120_screen.log 2>&1"
fi

cat <<EOF
120-task uniform then adaptive comparison launched.

Tail:
  tail -f logs/comparison_sweeps_120/uniform_then_adaptive_120_stdout.log
  tail -f logs/comparison_sweeps_120/watchdog_uniform_then_adaptive_120.log
  tail -f logs/uniform_multiarm_120/uniform_stdout.log
  tail -f logs/adaptive_sweep_120/adaptive_stdout.log

Attach:
  screen -r ${RUN_SCREEN}
  screen -r ${WATCHDOG_SCREEN}
  screen -r ${CAFFEINE_SCREEN}
EOF
