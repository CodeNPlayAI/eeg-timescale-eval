#!/usr/bin/env bash
# run_gan_only_30subj.sh
# GAN-only condition, 30 subjects x 20 seeds, ds005385 (eyes-closed, F7/F8, 180 s).
#
# LOCKED SETTINGS (do not change without re-running the single-subject diagnostics):
#   n_critic=3, gp_lambda=50, gen_steps=3000, batch=16, lr=1e-4,
#   Adam betas=(0.0, 0.9) [default], GroupNorm generator, linear output.
#
# PARITY CONTRACT with generate_diffusion_only.py -- these must NOT be changed:
#   fs=256, train_win_s=10 (win=2560), train_hop_s=5 (hop=1280),
#   per-window z-score, overlap-add (Hanning, 50% COLA), duration_s=180.
#
# NOTE: --holdout is NOT used here. Hold-out was a memorization diagnostic only;
# the production condition trains on ALL windows, exactly like diffusion-only.
#
# Resumable: subjects/seeds whose 180 s CSV already exists are skipped.

set -euo pipefail

# ----------------------------- EDIT THESE -----------------------------
# Two layouts are supported:
#   (a) flat   : IN_DIR/sub-XXX.csv        -> IN_GLOB="sub-*.csv"
#   (b) nested : IN_DIR/sub-XXX/real.csv   -> IN_GLOB="sub-*/real.csv"
IN_DIR="data/openneuro"                        # one CSV per subject, columns: time,F7,F8
IN_GLOB="sub-*/real.csv"
OUT_ROOT="runs/gan_only"
SCRIPT="scripts/generate_gan_only.py"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
# ----------------------------------------------------------------------

FS=256
DUR=180
WIN_S=10
HOP_S=5

mkdir -p "$OUT_ROOT" logs
LOG="logs/gan_only_30subj_$(date +%Y%m%d_%H%M%S).log"
echo "[start] $(date "+%Y-%m-%dT%H:%M:%S%z")  log=$LOG" | tee -a "$LOG"

shopt -s nullglob
CSVS=("$IN_DIR"/$IN_GLOB)
if [ ${#CSVS[@]} -eq 0 ]; then
  echo "[ERROR] no input CSV matched: $IN_DIR/$IN_GLOB" | tee -a "$LOG"; exit 1
fi
echo "[info] subjects found: ${#CSVS[@]}" | tee -a "$LOG"

# ---- derive subject ids and abort on collision (macOS Bash 3.2 compatible) ----
SUBJECT_IDS=()
SUBJECT_PATHS=()
for csv in "${CSVS[@]}"; do
  b="$(basename "$csv" .csv)"
  case "$b" in
    sub-*) sub="$b" ;;                                   # flat layout: sub-XXX.csv
    *)     sub="$(basename "$(dirname "$csv")")" ;;      # nested layout: sub-XXX/real.csv
  esac

  duplicate_index=-1
  i=0
  # Bash 3.2 + set -u: expanding an empty array with ${ARRAY[@]} raises
  # "unbound variable", so only iterate after the first item exists.
  if [ ${#SUBJECT_IDS[@]} -gt 0 ]; then
    for seen_sub in "${SUBJECT_IDS[@]}"; do
      if [ "$seen_sub" = "$sub" ]; then
        duplicate_index=$i
        break
      fi
      i=$((i + 1))
    done
  fi

  if [ "$duplicate_index" -ge 0 ]; then
    echo "[ERROR] duplicate subject id '$sub' from '$csv' and '${SUBJECT_PATHS[$duplicate_index]}'." | tee -a "$LOG"
    echo "        Output dirs would collide and subjects would be silently skipped. Aborting." | tee -a "$LOG"
    exit 1
  fi

  SUBJECT_IDS+=("$sub")
  SUBJECT_PATHS+=("$csv")
done
printf '%s\n' "${SUBJECT_IDS[@]}" | sort > /tmp/gan_subject_ids_$$.txt
echo "[info] subject ids: $(tr '\n' ' ' < /tmp/gan_subject_ids_$$.txt)" | tee -a "$LOG"
rm -f /tmp/gan_subject_ids_$$.txt

for csv in "${CSVS[@]}"; do
  b="$(basename "$csv" .csv)"
  case "$b" in
    sub-*) sub="$b" ;;
    *)     sub="$(basename "$(dirname "$csv")")" ;;
  esac
  outdir="$OUT_ROOT/$sub"
  mkdir -p "$outdir"

  # ---- resume: collect only the seeds that are still missing ----
  todo=""
  for s in $SEEDS; do
    if [ ! -f "$outdir/gan_3min_gan_seed${s}.csv" ]; then
      todo="$todo $s"
    fi
  done
  todo="$(echo "$todo" | xargs || true)"

  if [ -z "$todo" ]; then
    echo "[skip] $sub (all seeds already present)" | tee -a "$LOG"
    continue
  fi

  echo "[run ] $sub  seeds:[$todo]  $(date "+%Y-%m-%dT%H:%M:%S%z")" | tee -a "$LOG"
  python "$SCRIPT" \
    --input "$csv" \
    --out_dir "$outdir" \
    --fs $FS --duration_s $DUR \
    --train_win_s $WIN_S --train_hop_s $HOP_S \
    --seeds "$todo" \
    --gen_steps 3000 --batch 16 --n_critic 3 --gp_lambda 50 \
    --latent 100 --lr 1e-4 --gen_norm group \
    --diag_n_gen 200 --diag_no_save_windows \
    --save_180s 2>&1 | tee -a "$LOG"
done

echo "[done ] $(date "+%Y-%m-%dT%H:%M:%S%z")" | tee -a "$LOG"

# ---- post-run verification ----
echo "" | tee -a "$LOG"
echo "[verify] generated 180 s CSVs: $(find "$OUT_ROOT" -name 'gan_3min_gan_seed*.csv' | wc -l) (expected: subjects x seeds)" | tee -a "$LOG"
echo "[verify] train windows per subject (expect 35 for 180 s @ 256 Hz, win=10 s, hop=5 s):" | tee -a "$LOG"
grep -h "train_windows=" "$LOG" | sed 's/.*train_windows=\([0-9]*\).*/  \1/' | sort | uniq -c | tee -a "$LOG"
echo "[verify] checkpoint footprint: $(du -sh "$OUT_ROOT" 2>/dev/null | cut -f1)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "To reclaim space after verification (keeps seed 0 checkpoints only):" | tee -a "$LOG"
echo "  find $OUT_ROOT -name 'generator_seed*.pt' ! -name 'generator_seed0.pt' -delete" | tee -a "$LOG"
