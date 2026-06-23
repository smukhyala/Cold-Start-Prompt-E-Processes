#!/usr/bin/env bash
# Run or resume one WebArena Shopping paired-sweep arm.
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
[[ -f .env ]] && { set -a; source .env; set +a; }
source scripts/shopping_paired_common.sh
arm="${1:?usage: bash scripts/run_paired_shopping_arm.sh <arm_id>}"
case " ${SHOPPING_ARMS[*]} " in *" $arm "*) ;; *) echo "unknown arm: $arm" >&2; exit 2;; esac
shopping_run_or_resume_arm "$arm" "configs/webarena_shopping_pair_${arm}_60.yaml" "webarena_shopping_pair_${arm}_60"
