#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TABLE4="${REPO_ROOT}/source_data/Table4_E3_metricwise_20seed.csv"
BUILDER="${SCRIPT_DIR}/build_fig4_from_table4_standalone.py"
OUTPUT="${REPO_ROOT}/figures/Figure4"

[[ -f "$BUILDER" ]] || { echo "[ERROR] Missing: $BUILDER"; exit 1; }
[[ -f "$TABLE4" ]] || {
  echo "[ERROR] Missing: $TABLE4"
  echo "Source data missing at: $TABLE4"
  exit 1
}

exec "$PYTHON_BIN" "$BUILDER" \
  --table4 "$TABLE4" \
  --output-dir "$OUTPUT"
