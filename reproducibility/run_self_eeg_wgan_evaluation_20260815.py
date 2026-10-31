#!/usr/bin/env python3
"""Evaluate the self-EEG WGAN-GP seeds with the EXISTING scripts/evaluate.py
and merge the result with the aligned Core-3 results for Figure 6 (formerly Supplementary Figure S1).

Design constraint
-----------------
This script contains NO metric implementation. It invokes scripts/evaluate.py
as a subprocess and reads back the files that script writes
(metrics_summary.csv and meta.json), so the WGAN-GP numbers are produced by
exactly the same code path that produced Table 4 and Table S1.

Metric extraction (from what evaluate.py writes)
------------------------------------------------
  psd_loglog_slope_all_2_40hz : scalar, whole evaluation signal
  dfa_x_alpha                 : mean over 10-s windows
  acf_x_abs_0_1s              : mean over 10-s windows
  acf_x_abs_0_5s              : mean over 10-s windows
  acf_x_abs_1_0s              : mean over 10-s windows

IMPORTANT: in evaluate.py, `psd_loglog_slope_all_2_40hz` is produced only
inside the `if args.plots:` block. Running without --plots silently yields a
summary with no PSD slope row. This driver therefore always passes --plots and
verifies that the row exists.

Pipeline
--------
1. Evaluate the real self-EEG recording and check the five resulting metric
   values against `real_value` in the aligned all_metrics_long.csv
   (atol=1e-12). Any disagreement stops the run: it means the protocol
   arguments or the reference recording differ, and the WGAN-GP errors would
   not be comparable with the Core-3 errors.
2. Evaluate the 20 WGAN-GP seed files with identical arguments.
3. Emit a WGAN-GP seed-level table (5 metrics x 20 seeds) and a merged
   long-format CSV (aligned Core 3 + WGAN-GP) for Figure 6 (formerly Supplementary Figure S1).

Nothing is inferred, imputed, or corrected. Every inconsistency raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = list(range(20))
PSD_METRIC = "psd_loglog_slope_all_2_40hz"
WINDOW_METRICS = [
    "dfa_x_alpha",
    "acf_x_abs_0_1s",
    "acf_x_abs_0_5s",
    "acf_x_abs_1_0s",
]
METRICS = [PSD_METRIC, *WINDOW_METRICS]
METRIC_LABEL = {
    PSD_METRIC: "PSD slope\n(2–40 Hz)",
    "dfa_x_alpha": "DFA α",
    "acf_x_abs_0_1s": "ACF ≤ 0.1 s",
    "acf_x_abs_0_5s": "ACF ≤ 0.5 s",
    "acf_x_abs_1_0s": "ACF ≤ 1.0 s",
}
CORE_MODELS = ["NMM-only", "Diffusion-only", "Add"]
WGAN_MODEL = "WGAN-GP"
ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Running the existing evaluate.py.
# ---------------------------------------------------------------------------


def protocol_arguments(args) -> list[str]:
    """Protocol arguments passed to every evaluate.py invocation.

    These must be identical for the real recording and for all 20 WGAN-GP
    seeds; the Real reference check in step 1 is what proves they also match
    the aligned Core-3 run.
    """
    return [
        "--ch1", args.ch1,
        "--ch2", args.ch2,
        "--fs", str(args.fs),
        "--win_s", str(args.win_s),
        "--hop_s", str(args.hop_s),
        "--start_sec", str(args.start_sec),
        "--duration_sec", str(args.duration_sec),
        "--bp_low", str(args.bp_low),
        "--bp_high", str(args.bp_high),
        "--zscore", args.zscore,
        "--dfa_min_s", str(args.dfa_min_s),
        "--dfa_max_s", str(args.dfa_max_s),
        # --plots is mandatory: without it evaluate.py never computes the PSD
        # slope. --no_gc only skips Granger causality, which is not used here.
        "--plots",
        "--no_gc",
    ]


def run_evaluate(args, input_csv: Path, outdir: Path, context: str) -> dict[str, float]:
    if not input_csv.is_file():
        raise FileNotFoundError(f"{context}: input signal not found: {input_csv}")
    outdir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(args.evaluate_script),
        "--input", str(input_csv),
        "--outdir", str(outdir),
        *protocol_arguments(args),
    ]
    completed = subprocess.run(
        command,
        cwd=str(args.repo_root),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{context}: evaluate.py failed (exit {completed.returncode}).\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )

    (outdir / "driver_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    return read_metrics(outdir, context)


def read_metrics(outdir: Path, context: str) -> dict[str, float]:
    """Read the five metrics out of what evaluate.py wrote. No recomputation."""
    summary_path = outdir / "metrics_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{context}: evaluate.py wrote no metrics_summary.csv")

    summary = pd.read_csv(summary_path, index_col=0)
    if "mean" not in summary.columns:
        raise ValueError(
            f"{context}: metrics_summary.csv has no 'mean' column; "
            f"columns={list(summary.columns)}"
        )

    missing = [metric for metric in METRICS if metric not in summary.index]
    if missing:
        hint = ""
        if PSD_METRIC in missing:
            hint = (
                " The PSD slope row is only written when evaluate.py is run with "
                "--plots."
            )
        raise KeyError(
            f"{context}: metrics_summary.csv is missing {missing}.{hint} "
            f"Present rows: {list(summary.index)}"
        )

    values: dict[str, float] = {}
    for metric in METRICS:
        value = float(summary.loc[metric, "mean"])
        if not np.isfinite(value):
            raise ValueError(f"{context}: metric {metric} is not finite ({value})")
        values[metric] = value

    # Cross-check the PSD slope against meta.json, which evaluate.py also writes.
    meta_path = outdir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if PSD_METRIC in meta:
            if not np.isclose(float(meta[PSD_METRIC]), values[PSD_METRIC], rtol=0.0, atol=ATOL):
                raise ValueError(
                    f"{context}: PSD slope differs between metrics_summary.csv "
                    f"({values[PSD_METRIC]}) and meta.json ({meta[PSD_METRIC]})"
                )
    return values


def read_used_window(outdir: Path) -> dict[str, object]:
    meta_path = outdir / "meta.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        key: meta.get(key)
        for key in (
            "fs", "bandpass", "zscore", "window_s", "hop_s",
            "start_sec", "used_duration_sec", "used_samples",
            "n_samples_after_trim",
        )
    }


# ---------------------------------------------------------------------------
# Aligned Core-3 reference.
# ---------------------------------------------------------------------------


def load_aligned(csv_path: Path) -> pd.DataFrame:
    required = {
        "subject", "seed", "condition", "metric", "metric_label",
        "value", "real_value", "absolute_error", "relative_error",
    }
    if not csv_path.is_file():
        raise FileNotFoundError(f"Aligned metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Aligned metrics CSV missing columns: {sorted(missing)}")
    if sorted(df["seed"].astype(int).unique().tolist()) != SEEDS:
        raise ValueError("Aligned metrics CSV does not contain seeds 0-19")
    if set(df["condition"]) != {"Real", *CORE_MODELS}:
        raise ValueError(
            f"Aligned metrics CSV condition set unexpected: {sorted(set(df['condition']))}"
        )
    if set(df["metric"]) != set(METRICS):
        raise ValueError("Aligned metrics CSV metric set unexpected")
    if len(df) != len(SEEDS) * 4 * len(METRICS):
        raise ValueError(f"Aligned metrics CSV should have 400 rows, found {len(df)}")
    return df


def aligned_real_vector(aligned: pd.DataFrame) -> dict[str, float]:
    reference: dict[str, float] = {}
    for metric in METRICS:
        values = (
            aligned.loc[aligned["metric"] == metric, "real_value"].astype(float).to_numpy()
        )
        if not np.allclose(values, values[0], rtol=0.0, atol=ATOL):
            raise ValueError(
                f"Aligned real_value is not constant across seeds for {metric}"
            )
        reference[metric] = float(values[0])
    return reference


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-script", type=Path, required=True,
                        help="Path to the existing scripts/evaluate.py.")
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="Working directory for evaluate.py (must make "
                             "`common` importable). Default: evaluate.py's parent's parent.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--real-csv", type=Path, required=True,
                        help="Real self-EEG recording used by the aligned run.")
    parser.add_argument("--gan-dir", type=Path, required=True)
    parser.add_argument("--gan-pattern", default="gan_3min_gan_seed{seed}.csv")
    parser.add_argument("--aligned-metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/self_eeg_wgan_20260815"))

    # Protocol arguments. The defaults are the GAN evaluation convention
    # recorded in the OpenNeuro run paths (gan_start20_dur139.99609375) and the
    # evaluate.py protocol defaults; the Real check verifies them.
    parser.add_argument("--ch1", default="F7")
    parser.add_argument("--ch2", default="F8")
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--win_s", type=float, default=10.0)
    parser.add_argument("--hop_s", type=float, default=5.0)
    parser.add_argument("--start_sec", type=float, default=20.0)
    parser.add_argument("--duration_sec", type=float, default=139.99609375)
    parser.add_argument("--bp_low", type=float, default=1.0)
    parser.add_argument("--bp_high", type=float, default=45.0)
    parser.add_argument("--zscore", default="global", choices=["none", "global", "window"])
    parser.add_argument("--dfa_min_s", type=float, default=0.2)
    parser.add_argument("--dfa_max_s", type=float, default=1.25)
    args = parser.parse_args()

    args.evaluate_script = args.evaluate_script.resolve()
    if args.repo_root is None:
        args.repo_root = args.evaluate_script.parent.parent
    args.repo_root = args.repo_root.resolve()

    out_dir = args.output_dir.resolve()
    tables_dir = out_dir / "tables"
    source_dir = out_dir / "source_data"
    runs_dir = out_dir / "evaluate_runs"
    for directory in (out_dir, tables_dir, source_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    aligned = load_aligned(args.aligned_metrics_csv.resolve())
    reference = aligned_real_vector(aligned)

    # --- Step 1: reproduce Real with the existing evaluation code ------------
    real_dir = runs_dir / "real"
    recomputed_real = run_evaluate(
        args, args.real_csv.resolve(), real_dir, "Real self-EEG"
    )
    mismatches = {
        metric: {
            "aligned_real_value": reference[metric],
            "recomputed": recomputed_real[metric],
            "difference": recomputed_real[metric] - reference[metric],
        }
        for metric in METRICS
        if not np.isclose(recomputed_real[metric], reference[metric], rtol=0.0, atol=ATOL)
    }
    if mismatches:
        raise ValueError(
            "Recomputed Real metric values disagree with the aligned real_value.\n"
            "The evaluation arguments or the reference recording differ from the "
            "aligned Core-3 run. Stopping without correction; do not proceed to "
            "Figure 6 until this is resolved.\n"
            + json.dumps(mismatches, indent=2)
            + "\nProtocol actually used:\n"
            + json.dumps(read_used_window(real_dir), indent=2)
        )

    # --- Step 2: evaluate the 20 WGAN-GP seeds -------------------------------
    rows: list[dict[str, object]] = []
    gan_dir = args.gan_dir.resolve()
    for seed in SEEDS:
        gan_path = gan_dir / args.gan_pattern.format(seed=seed)
        context = f"WGAN-GP seed {seed}"
        seed_dir = runs_dir / f"wgan_seed{seed}"
        values = run_evaluate(args, gan_path, seed_dir, context)
        for metric in METRICS:
            value = values[metric]
            real_value = reference[metric]
            rows.append(
                {
                    "subject": "self_eeg",
                    "seed": seed,
                    "condition": WGAN_MODEL,
                    "metric": metric,
                    "metric_label": METRIC_LABEL[metric],
                    "value": value,
                    "real_value": real_value,
                    "absolute_error": abs(value - real_value),
                    "relative_error": abs(value - real_value) / abs(real_value),
                    "source": str(seed_dir / "metrics_summary.csv"),
                    "signal": str(gan_path),
                }
            )

    wgan = pd.DataFrame(rows)
    expected_rows = len(SEEDS) * len(METRICS)
    if len(wgan) != expected_rows:
        raise ValueError(f"Expected {expected_rows} WGAN-GP rows, built {len(wgan)}")
    wgan.to_csv(source_dir / "selfEEG_WGANGP_all_metrics_long.csv", index=False)

    for column, stem in (
        ("value", "TableS_selfEEG_WGANGP_seedwise_metric_values"),
        ("absolute_error", "TableS_selfEEG_WGANGP_seedwise_absolute_error"),
    ):
        wide = wgan.pivot(index="seed", columns="metric", values=column)[METRICS]
        wide.columns = [METRIC_LABEL[metric] for metric in METRICS]
        wide.to_csv(tables_dir / f"{stem}.csv")

    # --- Step 3: merged input for Figure 6 (formerly Supplementary Figure S1) --------------------------
    core = aligned.copy()
    for column in ("source", "signal"):
        if column not in core.columns:
            core[column] = str(args.aligned_metrics_csv.resolve())
    merged = pd.concat([core, wgan], ignore_index=True)
    merged = merged.sort_values(["seed", "metric", "condition"]).reset_index(drop=True)
    expected_merged = len(SEEDS) * len(METRICS) * 5  # Real + Core 3 + WGAN-GP
    if len(merged) != expected_merged:
        raise ValueError(
            f"Merged Fig. 6 input should have {expected_merged} rows, found {len(merged)}"
        )
    recomputed = (merged["value"].astype(float) - merged["real_value"].astype(float)).abs()
    if not np.allclose(
        recomputed.to_numpy(),
        merged["absolute_error"].astype(float).to_numpy(),
        rtol=0.0, atol=ATOL,
    ):
        raise ValueError("Merged Fig. 6 input failed the absolute_error check")
    merged.to_csv(
        source_dir / "Figure6_selfEEG_core3_plus_wgan_metrics_long.csv", index=False
    )

    report = {
        "protocol": "self-EEG WGAN-GP evaluation via the existing scripts/evaluate.py",
        "evaluate_script": str(args.evaluate_script),
        "evaluate_script_sha256": sha256(args.evaluate_script),
        "repo_root": str(args.repo_root),
        "protocol_arguments": protocol_arguments(args),
        "real_csv": str(args.real_csv.resolve()),
        "real_csv_sha256": sha256(args.real_csv.resolve()),
        "aligned_metrics_csv": str(args.aligned_metrics_csv.resolve()),
        "aligned_metrics_sha256": sha256(args.aligned_metrics_csv.resolve()),
        "real_reference_check": "passed (atol=1e-12)",
        "aligned_real_value": reference,
        "recomputed_real_value": recomputed_real,
        "protocol_actually_used": read_used_window(real_dir),
        "wgan_rows": int(len(wgan)),
        "merged_rows": int(len(merged)),
    }
    (out_dir / "selfEEG_WGANGP_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
