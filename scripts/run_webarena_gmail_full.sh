#!/usr/bin/env bash
# Drive the full first-real WebArena Gmail trial end-to-end:
#   1. Smoke (4 tasks) as live pre-flight integration test.
#   2. Full 60-task run with the production config.
#   3. CSV export (run.csv + per_arm.csv) of the full run.
#   4. Markdown + plot report of the full run.
#
# Aborts immediately if any phase fails. Designed to be launched via:
#
#   nohup bash scripts/run_webarena_gmail_full.sh \
#       > /tmp/cold_start_webarena.log 2>&1 &
#   echo $! > /tmp/cold_start_webarena.pid
#
# Then the runbook's monitoring snippets (tail, ps) work unchanged.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate

# .env carries ANTHROPIC_API_KEY; export so child processes inherit it.
set -a
# shellcheck disable=SC1091
source .env
set +a

SMOKE_CONFIG="configs/webarena_gmail_smoke.yaml"
FULL_CONFIG="configs/webarena_gmail.yaml"

banner() {
    printf '\n========================================\n'
    printf '== %s\n' "$1"
    printf '== %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '========================================\n\n'
}

newest_log_for() {
    # Newest JSONL log whose stem starts with the supplied trial-name prefix.
    local prefix="$1"
    ls -t "logs/${prefix}"_trial0_*.jsonl 2>/dev/null | head -1
}

# --- Phase 1: smoke ----------------------------------------------------------

banner "Phase 1/4: smoke (4 tasks) — $SMOKE_CONFIG"
cold-start-run --config "$SMOKE_CONFIG"
SMOKE_LOG="$(newest_log_for webarena_gmail_smoke)"
if [[ -z "$SMOKE_LOG" ]]; then
    echo "ERROR: smoke run produced no JSONL log under logs/. Aborting." >&2
    exit 1
fi
echo "smoke log: $SMOKE_LOG"

# --- Phase 2: full real run --------------------------------------------------

banner "Phase 2/4: full (60 tasks) — $FULL_CONFIG"
cold-start-run --config "$FULL_CONFIG"
FULL_LOG="$(newest_log_for webarena_gmail)"
if [[ -z "$FULL_LOG" || "$FULL_LOG" == "$SMOKE_LOG" ]]; then
    echo "ERROR: full run produced no new JSONL log under logs/. Aborting." >&2
    exit 1
fi
echo "full log: $FULL_LOG"

REPORT_DIR="reports/$(basename "$FULL_LOG" .jsonl)"
mkdir -p "$REPORT_DIR"

# --- Phase 3: CSV export -----------------------------------------------------

banner "Phase 3/4: CSV export -> $REPORT_DIR"
cold-start-export-csv --log "$FULL_LOG" --out-dir "$REPORT_DIR"

# --- Phase 4: markdown + plots ----------------------------------------------

banner "Phase 4/4: markdown report + plots -> $REPORT_DIR"
cold-start-report --log "$FULL_LOG" --out-dir "$REPORT_DIR"

banner "Done. Artifacts:"
ls -la "$REPORT_DIR"
