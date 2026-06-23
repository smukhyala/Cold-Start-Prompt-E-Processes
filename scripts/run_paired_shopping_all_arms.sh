#!/usr/bin/env bash
# Run or resume all 12 WebArena Shopping paired-sweep arms sequentially.
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
[[ -f .env ]] && { set -a; source .env; set +a; }
source scripts/shopping_paired_common.sh
for arm in "${SHOPPING_ARMS[@]}"; do
  shopping_run_or_resume_arm "$arm" "configs/webarena_shopping_pair_${arm}_60.yaml" "webarena_shopping_pair_${arm}_60"
done
COMMIT_FINAL=1 bash scripts/finalize_paired_sweep_shopping.sh
