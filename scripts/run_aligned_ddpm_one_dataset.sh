#!/usr/bin/env bash
set -euo pipefail

# Run the aligned core-three EEG-generation pipeline for one Real-EEG CSV.
#
# Required environment variables:
#   REAL_CSV       Input CSV with time,F7,F8 columns.
#   OUTPUT_ROOT    Dedicated output directory for this dataset/subject.
#
# Optional environment variables:
#   ROOT_DIR       Repository root (default: parent of this script directory).
#   DATASET_LABEL  Provenance label (default: dataset).
#   SUBJECT_ID     Provenance label (default: self_eeg).
#   SEEDS          Space-separated seeds (default: 0..19).
#   RUN_E1_E2      1 for self-EEG E1/E2 diagnostics, 0 for OpenNeuro (default: 0).
#   MAKE_FIGURES   Must be 0; old exploratory plots removed from public release.
#   SKIP_COMPLETED 1 to resume by skipping seed directories with .complete (default: 1).
#   PYTHON_BIN     Python executable (default: python).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REAL_CSV="${REAL_CSV:?Set REAL_CSV to the Real-EEG CSV path}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a dedicated output directory}"
DATASET_LABEL="${DATASET_LABEL:-dataset}"
SUBJECT_ID="${SUBJECT_ID:-self_eeg}"
SEEDS_STR="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
RUN_E1_E2="${RUN_E1_E2:-0}"
MAKE_FIGURES="${MAKE_FIGURES:-0}"
if [[ "$MAKE_FIGURES" != "0" ]]; then
  echo "[ERROR] MAKE_FIGURES=1 used obsolete exploratory plotting scripts; published figures are generated via reproducibility/run_fig*.sh." >&2
  exit 1
fi
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$REAL_CSV" ]]; then
  echo "[ERROR] REAL_CSV not found: $REAL_CSV" >&2
  exit 1
fi
REAL_CSV="$(cd "$(dirname "$REAL_CSV")" && pwd)/$(basename "$REAL_CSV")"
OUTPUT_ROOT="$(mkdir -p "$OUTPUT_ROOT" && cd "$OUTPUT_ROOT" && pwd)"

for required in \
  scripts/generate_nmm_only.py \
  scripts/generate_diffusion_only.py \
  scripts/make_residual.py \
  scripts/make_add.py \
  scripts/evaluate.py; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] Required repository file not found: $ROOT_DIR/$required" >&2
    exit 1
  fi
done

if [[ "$RUN_E1_E2" == "1" && ! -f scripts/evaluate_cross_term_e2.py ]]; then
  echo "[ERROR] RUN_E1_E2=1 but scripts/evaluate_cross_term_e2.py is missing" >&2
  exit 1
fi

read -r -a SEED_ARRAY <<< "$SEEDS_STR"
if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "[ERROR] SEEDS is empty" >&2
  exit 1
fi
for seed in "${SEED_ARRAY[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid seed: $seed" >&2
    exit 1
  fi
done

# Locked evaluation protocol.
FS=256
CH1=F7
CH2=F8
START_SEC=20
DUR_SEC=139.99609375
WIN_S=10
HOP_S=5
BP_LOW=1
BP_HIGH=45
ZSCORE=global
DFA_MIN_S=0.2
DFA_MAX_S=1.25

# NMM protocol.
NMM_COUPLING=0.02
NMM_DURATION_S=180

# Aligned DDPM protocol for both Diffusion-only and Add residual diffusion.
DIFF_T=1000
DIFF_UPDATES=800
DIFF_BATCH=8
DIFF_LR=1e-3
DIFF_TRAIN_WIN_S=10
DIFF_TRAIN_HOP_S=5
DIFF_DURATION_S=180

ADD_RESIDUAL_T=1000
ADD_RESIDUAL_UPDATES=800
ADD_RESIDUAL_BATCH=8
ADD_RESIDUAL_LR=1e-3
ADD_RESIDUAL_DURATION_S=140

CONFIG_FILE="$OUTPUT_ROOT/run_config.json"
"$PYTHON_BIN" - "$CONFIG_FILE" "$REAL_CSV" "$DATASET_LABEL" "$SUBJECT_ID" \
  "$SEEDS_STR" "$RUN_E1_E2" "$MAKE_FIGURES" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

config_path = Path(sys.argv[1])
config = {
    "schema_version": 1,
    "real_csv": str(Path(sys.argv[2]).resolve()),
    "dataset_label": sys.argv[3],
    "subject_id": sys.argv[4],
    "seeds": [int(x) for x in sys.argv[5].split()],
    "run_e1_e2": bool(int(sys.argv[6])),
    "make_figures": bool(int(sys.argv[7])),
    "channels": ["F7", "F8"],
    "fs": 256,
    "evaluation": {
        "start_sec": 20.0,
        "duration_sec": 139.99609375,
        "window_sec": 10.0,
        "hop_sec": 5.0,
        "bandpass_hz": [1.0, 45.0],
        "zscore": "global",
        "dfa_scale_sec": [0.2, 1.25],
    },
    "nmm": {"coupling": 0.02, "duration_sec": 180.0},
    "diffusion_only": {
        "T": 1000,
        "optimization_updates": 800,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "train_window_sec": 10.0,
        "train_hop_sec": 5.0,
        "generated_duration_sec": 180.0,
    },
    "add_residual_diffusion": {
        "T": 1000,
        "optimization_updates": 800,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "train_window_sec": 10.0,
        "train_hop_sec": 5.0,
        "generated_duration_sec": 140.0,
        "std_match": True,
        "mixing_coefficient_b": 1.0,
    },
}

if config_path.exists():
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    previous.pop("created_at", None)
    if previous != config:
        raise SystemExit(
            "Existing run_config.json differs from the requested configuration. "
            "Choose a new OUTPUT_ROOT instead of mixing runs."
        )
else:
    config["created_at"] = datetime.now().isoformat(timespec="seconds")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

run_eval() {
  local input_csv="$1"
  local outdir="$2"
  local start_sec="$3"
  local duration_sec="$4"

  "$PYTHON_BIN" scripts/evaluate.py \
    --input "$input_csv" \
    --ch1 "$CH1" \
    --ch2 "$CH2" \
    --fs "$FS" \
    --win_s "$WIN_S" \
    --hop_s "$HOP_S" \
    --bp_low "$BP_LOW" \
    --bp_high "$BP_HIGH" \
    --zscore "$ZSCORE" \
    --dfa_min_s "$DFA_MIN_S" \
    --dfa_max_s "$DFA_MAX_S" \
    --start_sec "$start_sec" \
    --duration_sec "$duration_sec" \
    --plots \
    --no_gc \
    --outdir "$outdir"
}

for seed in "${SEED_ARRAY[@]}"; do
  RUN_ROOT="$OUTPUT_ROOT/seed${seed}"
  COMPLETE_MARKER="$RUN_ROOT/.complete"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$COMPLETE_MARKER" ]]; then
    echo "[SKIP][$DATASET_LABEL][$SUBJECT_ID][seed=$seed] complete"
    continue
  fi

  echo
  echo "================================================================"
  echo "[$DATASET_LABEL][$SUBJECT_ID] seed=$seed"
  echo "DDPM: T=1000, updates=800, batch=8 (Diffusion-only and Add)"
  echo "================================================================"

  GENERATED_DIR="$RUN_ROOT/generated"
  mkdir -p "$GENERATED_DIR"

  REAL_DIR="$RUN_ROOT/real_start20_dur${DUR_SEC}"
  NMM_DIR="$RUN_ROOT/nmm_seed${seed}_start20_dur${DUR_SEC}"
  DIFF_DIR="$RUN_ROOT/diffusion_only_start20_dur${DUR_SEC}"
  ADD_DIR="$RUN_ROOT/add_stdmatched_b1.0_start20_dur${DUR_SEC}"

  NMM_OUT_BASE="$GENERATED_DIR/nmm_3min.csv"
  NMM_CSV="$GENERATED_DIR/nmm_3min_nmm_seed${seed}.csv"
  DIFF_CSV="$GENERATED_DIR/diffusion_only_t1000_b8.csv"
  RES_TRUE_CSV="$GENERATED_DIR/res_true_real_minus_nmm.csv"
  RES_HAT_CSV="$GENERATED_DIR/residual_hat_t1000_b8.csv"
  RES_HAT_STDMATCHED_CSV="$GENERATED_DIR/residual_hat_t1000_b8_stdmatched.csv"
  BASE_ALIGNED_CSV="$GENERATED_DIR/nmm_base_aligned_start20.csv"
  ADD_CSV="$GENERATED_DIR/add_t1000_b8_stdmatched_b1.0.csv"

  echo "[1/9] Real evaluation"
  run_eval "$REAL_CSV" "$REAL_DIR" "$START_SEC" "$DUR_SEC"

  echo "[2/9] NMM-only generation and evaluation"
  "$PYTHON_BIN" scripts/generate_nmm_only.py \
    --out "$NMM_OUT_BASE" \
    --tag "nmm_seed${seed}" \
    --fs "$FS" \
    --duration_s "$NMM_DURATION_S" \
    --seed "$seed" \
    --seed_f7 "$((seed * 2))" \
    --seed_f8 "$((seed * 2 + 1))" \
    --coupling "$NMM_COUPLING" \
    --ch1 "$CH1" \
    --ch2 "$CH2"
  run_eval "$NMM_CSV" "$NMM_DIR" "$START_SEC" "$DUR_SEC"

  echo "[3/9] Diffusion-only generation (T=1000, updates=800, batch=8)"
  "$PYTHON_BIN" scripts/generate_diffusion_only.py \
    --input "$REAL_CSV" \
    --out "$DIFF_CSV" \
    --fs "$FS" \
    --duration_s "$DIFF_DURATION_S" \
    --train_win_s "$DIFF_TRAIN_WIN_S" \
    --train_hop_s "$DIFF_TRAIN_HOP_S" \
    --T "$DIFF_T" \
    --epochs "$DIFF_UPDATES" \
    --batch "$DIFF_BATCH" \
    --lr "$DIFF_LR" \
    --seed "$seed"

  echo "[4/9] Diffusion-only evaluation"
  run_eval "$DIFF_CSV" "$DIFF_DIR" "$START_SEC" "$DUR_SEC"

  echo "[5/9] True residual creation"
  "$PYTHON_BIN" scripts/make_residual.py \
    --real "$REAL_CSV" \
    --base "$NMM_CSV" \
    --ch1 "$CH1" \
    --ch2 "$CH2" \
    --fs "$FS" \
    --bp_low "$BP_LOW" \
    --bp_high "$BP_HIGH" \
    --zscore "$ZSCORE" \
    --start_sec "$START_SEC" \
    --duration_sec "$DUR_SEC" \
    --out_csv "$RES_TRUE_CSV"

  echo "[6/9] Add residual generation (T=1000, updates=800, batch=8)"
  "$PYTHON_BIN" scripts/generate_diffusion_only.py \
    --input "$RES_TRUE_CSV" \
    --out "$RES_HAT_CSV" \
    --fs "$FS" \
    --duration_s "$ADD_RESIDUAL_DURATION_S" \
    --train_win_s "$DIFF_TRAIN_WIN_S" \
    --train_hop_s "$DIFF_TRAIN_HOP_S" \
    --T "$ADD_RESIDUAL_T" \
    --epochs "$ADD_RESIDUAL_UPDATES" \
    --batch "$ADD_RESIDUAL_BATCH" \
    --lr "$ADD_RESIDUAL_LR" \
    --seed "$seed"

  echo "[7/9] Residual standard-deviation matching and NMM alignment"
  "$PYTHON_BIN" - "$RES_TRUE_CSV" "$RES_HAT_CSV" \
    "$RES_HAT_STDMATCHED_CSV" "$NMM_CSV" "$BASE_ALIGNED_CSV" \
    "$FS" "$START_SEC" "$DUR_SEC" "$CH1" "$CH2" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

res_true_path, res_hat_path, std_out_path, nmm_path, base_out_path = map(Path, sys.argv[1:6])
fs, start_sec, duration_sec = map(float, sys.argv[6:9])
channels = sys.argv[9:11]

res_true = pd.read_csv(res_true_path)
res_hat = pd.read_csv(res_hat_path)
n = min(len(res_true), len(res_hat))
if n <= 0:
    raise ValueError("Residual inputs are empty")
res_true = res_true.iloc[:n].copy()
res_hat = res_hat.iloc[:n].copy()
stdmatched = res_hat.copy()
for channel in channels:
    if channel not in res_true or channel not in res_hat:
        raise ValueError(f"Missing residual channel: {channel}")
    src_sd = float(np.std(res_true[channel].to_numpy(dtype=float)))
    gen_sd = float(np.std(res_hat[channel].to_numpy(dtype=float)))
    if not np.isfinite(gen_sd) or gen_sd <= 1e-12:
        raise ValueError(f"Generated residual SD is invalid for {channel}: {gen_sd}")
    scale = src_sd / gen_sd
    stdmatched[channel] = res_hat[channel].to_numpy(dtype=float) * scale
    print(f"[std-match] {channel}: scale={scale:.12g}")
stdmatched.to_csv(std_out_path, index=False)

nmm = pd.read_csv(nmm_path)
for channel in channels:
    if channel not in nmm:
        raise ValueError(f"Missing NMM channel: {channel}")
start_idx = int(round(start_sec * fs))
requested = int(round(duration_sec * fs))
end_idx = start_idx + requested
if end_idx > len(nmm):
    raise ValueError(
        f"NMM alignment exceeds signal length: len={len(nmm)}, "
        f"start={start_idx}, end={end_idx}"
    )
nmm.iloc[start_idx:end_idx].copy().to_csv(base_out_path, index=False)
PY

  echo "[8/9] Add composition and evaluation"
  "$PYTHON_BIN" scripts/make_add.py \
    --base "$BASE_ALIGNED_CSV" \
    --residual "$RES_HAT_STDMATCHED_CSV" \
    --fs "$FS" \
    --start_sec 0 \
    --duration_sec "$DUR_SEC" \
    --b 1.0 \
    --no_preproc \
    --out_csv "$ADD_CSV"
  run_eval "$ADD_CSV" "$ADD_DIR" 0 "$DUR_SEC"

  if [[ "$RUN_E1_E2" == "1" ]]; then
    echo "[8.5/9] Self-EEG E1 residual evaluation"
    run_eval "$RES_TRUE_CSV" "$RUN_ROOT/res_true_eval_start0_dur${DUR_SEC}" 0 "$DUR_SEC"
    run_eval "$RES_HAT_CSV" "$RUN_ROOT/residual_hat_eval_start0_dur${DUR_SEC}" 0 "$DUR_SEC"
    run_eval "$RES_HAT_STDMATCHED_CSV" \
      "$RUN_ROOT/residual_hat_stdmatched_eval_start0_dur${DUR_SEC}" 0 "$DUR_SEC"

    echo "[8.7/9] Self-EEG E2 cross-term verification"
    E2_DIR="$RUN_ROOT/e2_cross_term"
    mkdir -p "$E2_DIR"
    "$PYTHON_BIN" scripts/evaluate_cross_term_e2.py \
      --base "$BASE_ALIGNED_CSV" \
      --residual "$RES_HAT_STDMATCHED_CSV" \
      --ch1 "$CH1" \
      --ch2 "$CH2" \
      --fs "$FS" \
      --win_s "$WIN_S" \
      --hop_s "$HOP_S" \
      --lag_secs 0.1 0.5 1.0 \
      --seed "$seed" \
      --out_csv "$E2_DIR/e2_cross_term_summary.csv" \
      --out_window_csv "$E2_DIR/e2_cross_term_by_window.csv" \
      --meta_json "$E2_DIR/e2_cross_term_summary.meta.json" \
      --tag "aligned_ddpm_T1000_E800_B8_seed=${seed}"
  fi


  echo "[9/9] Completion marker"
  "$PYTHON_BIN" - "$RUN_ROOT/seed_config.json" "$seed" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "seed": int(sys.argv[2]),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "diffusion_only": {"T": 1000, "optimization_updates": 800, "batch_size": 8},
            "add_residual_diffusion": {"T": 1000, "optimization_updates": 800, "batch_size": 8},
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
  touch "$COMPLETE_MARKER"
done

echo
echo "[DONE] one-dataset aligned DDPM run: $OUTPUT_ROOT"
