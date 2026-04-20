#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
echo "venv ready. Activate with: source .venv/bin/activate"
