#!/usr/bin/env bash
set -euo pipefail

# Self-EEG rerun: core three generated conditions, 20 seeds.
# Both DDPM branches use T=1000, 800 optimization updates, batch size 8.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REAL_CSV="${REAL_CSV:-$ROOT_DIR/data/eeg_record_20251226_173831.csv}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/self_eeg_aligned_ddpm_T1000_E800_B8_20seeds_${RUN_STAMP}}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

export ROOT_DIR PYTHON_BIN REAL_CSV OUTPUT_ROOT SEEDS SKIP_COMPLETED
export DATASET_LABEL=self_eeg
export SUBJECT_ID=self_eeg
export RUN_E1_E2=1
export MAKE_FIGURES="${MAKE_FIGURES:-1}"

"$ROOT_DIR/scripts/run_aligned_ddpm_one_dataset.sh"

"$PYTHON_BIN" "$ROOT_DIR/scripts/aggregate_aligned_ddpm.py" \
  --root "$OUTPUT_ROOT" \
  --expected-subjects 1 \
  --expected-seeds 20

echo
echo "[DONE] self-EEG aligned rerun"
echo "Output: $OUTPUT_ROOT"
