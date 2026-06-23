#!/usr/bin/env bash
# Run Shopping smoke test and generate reports.
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
[[ -f .env ]] && { set -a; source .env; set +a; }
mkdir -p logs/paired_sweep_shopping reports/paired_sweep_shopping/smoke
cold-start-run --config configs/webarena_shopping_smoke.yaml 2>&1 | tee logs/paired_sweep_shopping/smoke_stdout.log
log="$(ls -t logs/paired_sweep_shopping/webarena_shopping_smoke_trial0_*.jsonl | head -1)"
cold-start-export-csv --log "$log" --out-dir reports/paired_sweep_shopping/smoke
cold-start-report --log "$log" --out-dir reports/paired_sweep_shopping/smoke
bash scripts/finalize_paired_sweep_shopping.sh
