#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
"${PYTHON_BIN:-python3}" "$SCRIPT_DIR/build_figure5_openneuro_winners.py" \
  --input-csv "$REPO_ROOT/source_data/OpenNeuro_all_errors_by_subject_seed_model_metric.csv" \
  --figures-dir "$REPO_ROOT/figures/Figure5" \
  --source-data-dir "$REPO_ROOT/outputs/validation/Figure5" \
  "$@"
