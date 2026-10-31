#!/usr/bin/env bash
set -euo pipefail

# Table 4 (E3) only.
#
# This launcher calls the dedicated Python builder. It does not contain a
# second implementation of the Table 4 aggregation.
#
# Optional overrides are passed through, for example:
#   ./reproducibility/run_table4_e3_v3.sh \
#     --input-csv /path/to/all_metrics_long.csv \
#     --run-config /path/to/run_config.json \
#     --output-dir /path/to/output
#
# Validation without writing publication outputs:
#   ./reproducibility/run_table4_e3_v3.sh --validate-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BUILDER="${SCRIPT_DIR}/build_revision_figures_tables_20260814_table4_fixed_v3.py"
INPUT_CSV="${PROJECT_ROOT}/runs/self_eeg_aligned_ddpm_T1000_E800_B8_20seeds_20260811_200259/summary_aligned_ddpm_T1000_E800_B8/all_metrics_long.csv"
RUN_CONFIG="${PROJECT_ROOT}/runs/self_eeg_aligned_ddpm_T1000_E800_B8_20seeds_20260811_200259/run_config.json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/publication"

if [[ ! -f "${BUILDER}" ]]; then
    echo "[ERROR] Table 4 builder not found: ${BUILDER}" >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${BUILDER}" \
    --input-csv "${INPUT_CSV}" \
    --run-config "${RUN_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
