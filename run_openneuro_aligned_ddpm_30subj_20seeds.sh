#!/usr/bin/env bash
set -euo pipefail

# OpenNeuro ds005385 rerun: 30 fixed subjects x 20 seeds.
# Both DDPM branches use T=1000, 800 optimization updates, batch size 8.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BIDS_ROOT="${BIDS_ROOT:-$ROOT_DIR/OpenNeuro/ds005385}"
SESSION="${SESSION:-ses-1}"
TASK="${TASK:-EyesClosed}"
ACQUISITION="${ACQUISITION:-pre}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/openneuro30_aligned_ddpm_T1000_E800_B8_20seeds_${RUN_STAMP}}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PREPROCESS_ONLY="${PREPROCESS_ONLY:-0}"

DEFAULT_SUBJECTS="sub-012 sub-050 sub-062 sub-122 sub-140 sub-144 sub-152 sub-157 sub-180 sub-305 sub-308 sub-320 sub-323 sub-331 sub-349 sub-402 sub-407 sub-412 sub-425 sub-449 sub-470 sub-518 sub-529 sub-542 sub-566 sub-573 sub-574 sub-575 sub-597 sub-603"
SUBJECTS_STR="${SUBJECTS:-$DEFAULT_SUBJECTS}"
read -r -a SUBJECT_ARRAY <<< "$SUBJECTS_STR"

if [[ "${#SUBJECT_ARRAY[@]}" -ne 30 && -z "${SUBJECTS:-}" ]]; then
  echo "[ERROR] Internal subject list must contain exactly 30 subjects" >&2
  exit 1
fi
if [[ ! -d "$BIDS_ROOT" ]]; then
  echo "[ERROR] BIDS_ROOT not found: $BIDS_ROOT" >&2
  echo "Set BIDS_ROOT to the ds005385 directory." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/preprocessed" "$OUTPUT_ROOT/subjects"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

"$PYTHON_BIN" - "$OUTPUT_ROOT/batch_config.json" "$BIDS_ROOT" "$SESSION" \
  "$TASK" "$ACQUISITION" "$SUBJECTS_STR" "$SEEDS" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
config = {
    "schema_version": 1,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "dataset": "OpenNeuro ds005385",
    "bids_root": str(Path(sys.argv[2]).resolve()),
    "session": sys.argv[3],
    "task": sys.argv[4],
    "acquisition": sys.argv[5],
    "subjects": sys.argv[6].split(),
    "seeds": [int(x) for x in sys.argv[7].split()],
    "channels": ["F7", "F8"],
    "preprocessing": {"target_fs": 256, "duration_sec": 180},
    "diffusion_only": {"T": 1000, "optimization_updates": 800, "batch_size": 8},
    "add_residual_diffusion": {"T": 1000, "optimization_updates": 800, "batch_size": 8},
    "e1_e2": False,
}
if path.exists():
    old = json.loads(path.read_text(encoding="utf-8"))
    old.pop("created_at", None)
    new = dict(config)
    new.pop("created_at", None)
    if old != new:
        raise SystemExit(
            "Existing batch_config.json differs from this request. "
            "Choose a new OUTPUT_ROOT instead of mixing runs."
        )
else:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

echo "[PHASE 1] Preflight and preprocess all selected OpenNeuro subjects"
for subject in "${SUBJECT_ARRAY[@]}"; do
  if [[ ! "$subject" =~ ^sub-[0-9]+$ ]]; then
    echo "[ERROR] Invalid BIDS subject label: $subject" >&2
    exit 1
  fi
  PREPROCESSED_CSV="$OUTPUT_ROOT/preprocessed/${subject}_${SESSION}_${TASK}_acq-${ACQUISITION}_180s_256Hz.csv"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/preprocess_openneuro_cli.py" \
    --bids-root "$BIDS_ROOT" \
    --subject "$subject" \
    --session "$SESSION" \
    --task "$TASK" \
    --acquisition "$ACQUISITION" \
    --out "$PREPROCESSED_CSV" \
    --fs 256 \
    --duration-s 180 \
    --channels F7 F8
done

if [[ "$PREPROCESS_ONLY" == "1" ]]; then
  echo "[DONE] PREPROCESS_ONLY=1; no generation runs were started."
  echo "Output: $OUTPUT_ROOT/preprocessed"
  exit 0
fi

echo "[PHASE 2] Run 30 subjects x 20 seeds"
for subject in "${SUBJECT_ARRAY[@]}"; do
  PREPROCESSED_CSV="$OUTPUT_ROOT/preprocessed/${subject}_${SESSION}_${TASK}_acq-${ACQUISITION}_180s_256Hz.csv"
  SUBJECT_OUTPUT="$OUTPUT_ROOT/subjects/$subject"
  echo
  echo "################################################################"
  echo "OPENNEURO SUBJECT: $subject"
  echo "OUTPUT: $SUBJECT_OUTPUT"
  echo "################################################################"

  ROOT_DIR="$ROOT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  REAL_CSV="$PREPROCESSED_CSV" \
  OUTPUT_ROOT="$SUBJECT_OUTPUT" \
  DATASET_LABEL="openneuro_ds005385" \
  SUBJECT_ID="$subject" \
  SEEDS="$SEEDS" \
  RUN_E1_E2=0 \
  MAKE_FIGURES=0 \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
    "$ROOT_DIR/scripts/run_aligned_ddpm_one_dataset.sh"
done

echo "[PHASE 3] Validate and aggregate the complete 30-subject batch"
"$PYTHON_BIN" "$ROOT_DIR/scripts/aggregate_aligned_ddpm.py" \
  --root "$OUTPUT_ROOT" \
  --expected-subjects 30 \
  --expected-seeds 20

echo
echo "[DONE] OpenNeuro 30-subject aligned rerun"
echo "Output: $OUTPUT_ROOT"
