#!/usr/bin/env bash
# Start uniform multi-arm then adaptive SPRUCE comparison runs with watchdog.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs/comparison_sweeps logs/uniform_multiarm logs/adaptive_sweep

RUN_SCREEN="${RUN_SCREEN:-csp_uniform_then_adaptive}"
WATCHDOG_SCREEN="${WATCHDOG_SCREEN:-csp_uniform_then_adaptive_watchdog}"
CAFFEINE_SCREEN="${CAFFEINE_SCREEN:-csp_comparison_caffeinate}"

if ! screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${CAFFEINE_SCREEN}$"; then
    screen -dmS "$CAFFEINE_SCREEN" bash -lc "cd '$REPO_DIR' && while true; do date -u '+[caffeinate %Y-%m-%dT%H:%M:%SZ] renewing 1h assertion' >> logs/comparison_sweeps/caffeinate.log; caffeinate -i -t 3600 >> logs/comparison_sweeps/caffeinate.log 2>&1; sleep 5; done"
fi

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${RUN_SCREEN}$"; then
    echo "run screen already exists: ${RUN_SCREEN}" >&2
else
    screen -dmS "$RUN_SCREEN" bash -lc "cd '$REPO_DIR' && bash scripts/resume_uniform_then_adaptive.sh > logs/comparison_sweeps/uniform_then_adaptive_stdout.log 2>&1"
fi

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${WATCHDOG_SCREEN}$"; then
    echo "watchdog screen already exists: ${WATCHDOG_SCREEN}" >&2
else
    screen -dmS "$WATCHDOG_SCREEN" bash -lc "cd '$REPO_DIR' && SCREEN_NAME='$RUN_SCREEN' bash scripts/watchdog_uniform_then_adaptive.sh >> logs/comparison_sweeps/watchdog_uniform_then_adaptive_screen.log 2>&1"
fi

cat <<EOF
Uniform then adaptive comparison launched.

Tail:
  tail -f logs/comparison_sweeps/uniform_then_adaptive_stdout.log
  tail -f logs/comparison_sweeps/watchdog_uniform_then_adaptive.log
  tail -f logs/uniform_multiarm/uniform_stdout.log
  tail -f logs/adaptive_sweep/adaptive_stdout.log

Attach:
  screen -r ${RUN_SCREEN}
  screen -r ${WATCHDOG_SCREEN}
  screen -r ${CAFFEINE_SCREEN}
EOF
