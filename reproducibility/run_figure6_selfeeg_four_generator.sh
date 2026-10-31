#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
"${PYTHON_BIN:-python3}" "$SCRIPT_DIR/build_figure6_selfeeg_four_generator.py" \
  --input-csv "$REPO_ROOT/source_data/Figure6_selfEEG_core3_plus_wgan_metrics_long.csv" \
  --output-dir "$REPO_ROOT/figures/Figure6" \
  --source-data-dir "$REPO_ROOT/outputs/validation/Figure6" \
  "$@"
