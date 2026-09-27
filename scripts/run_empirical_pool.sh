#!/usr/bin/env bash
# Launch the empirical-pool collector under its watchdog, detached and awake.
#   scripts/run_empirical_pool.sh --pilot   # the 660-episode pilot (gate G2)
#   scripts/run_empirical_pool.sh           # everything left in the queue (resumes the pilot)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/empirical_pool
nohup caffeinate -dimsu .venv/bin/python scripts/watchdog_empirical_pool.py "$@" \
  >> logs/empirical_pool/watchdog.out 2>&1 &
echo "watchdog pid $! -- tail -f logs/empirical_pool/watchdog.out logs/empirical_pool/collect.out"
