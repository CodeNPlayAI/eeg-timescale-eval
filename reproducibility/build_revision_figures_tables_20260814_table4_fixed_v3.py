#!/usr/bin/env python3
"""Build Table 4 (E3) from the aligned self-EEG 20-seed metrics.

This module owns Table 4 only. It does not build or modify Fig. 4, Fig. 5,
Table 5, or the reporting policy. Scientific definitions are fixed by
REPORTING_POLICY_20260814_REVISED.md and the approved 2026-08-15 Table 4
specification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELF_RUN = (
    ROOT
    / "runs"
    / "self_eeg_aligned_ddpm_T1000_E800_B8_20seeds_20260811_200259"
)
DEFAULT_INPUT_CSV = (
    DEFAULT_SELF_RUN
    / "summary_aligned_ddpm_T1000_E800_B8"
    / "all_metrics_long.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "publication"

EXPECTED_SEEDS = list(range(20))
EXPECTED_SUBJECTS = ["self_eeg"]
EXPECTED_CONDITIONS = ["Real", "NMM-only", "Diffusion-only", "Add"]
EXPECTED_METRICS = [
    "psd_loglog_slope_all_2_40hz",
    "dfa_x_alpha",
    "acf_x_abs_0_1s",
    "acf_x_abs_0_5s",
    "acf_x_abs_1_0s",
]
DISPLAY_LABELS = {
    "psd_loglog_slope_all_2_40hz": "PSD slope (2-40 Hz)",
    "dfa_x_alpha": "DFA alpha",
    "acf_x_abs_0_1s": "ACF ≤ 0.1 s",
    "acf_x_abs_0_5s": "ACF ≤ 0.5 s",
    "acf_x_abs_1_0s": "ACF ≤ 1.0 s",
}
REQUIRED_COLUMNS = {
    "subject",
    "seed",
    "condition",
    "metric",
    "metric_label",
    "value",
    "real_value",
    "absolute_error",
    "relative_error",
}
TABLE_HEADER = [
    "Metric",
    "Real",
    "NMM-only",
    "Diffusion-only",
    "Add",
    "Δ(Diffusion-only - Add)",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_config(run_config_path: Path) -> dict[str, object]:
    """Verify that Table 4 comes from the locked aligned self-EEG run."""
    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"Aligned self-EEG provenance requires run_config.json: {run_config_path}"
        )
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if config.get("dataset_label") != "self_eeg" or config.get("subject_id") != "self_eeg":
        raise ValueError("run_config.json does not identify the canonical self_eeg source")
    if [int(seed) for seed in config.get("seeds", [])] != EXPECTED_SEEDS:
        raise ValueError("run_config.json must list exactly seeds 0-19 in order")
    for key in ("diffusion_only", "add_residual_diffusion"):
        if key not in config:
            raise ValueError(f"run_config.json is missing {key}")
        item = config[key]
        observed = (
            int(item.get("T", -1)),
            int(item.get("optimization_updates", -1)),
            int(item.get("batch_size", -1)),
        )
        if observed != (1000, 800, 8):
            raise ValueError(f"Unaligned DDPM configuration for {key}: {observed}")
    return config


def load_table4_metrics_long(csv_path: Path) -> pd.DataFrame:
    """Load and strictly validate the approved Table 4 long-format input."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Table 4 metrics CSV not found: {csv_path}")
    if csv_path.name != "all_metrics_long.csv":
        raise ValueError(
            f"Table 4 requires the aligned all_metrics_long.csv output; found {csv_path.name}"
        )

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Table 4 input is missing required columns: {sorted(missing)}")

    subjects = sorted(df["subject"].astype(str).unique().tolist())
    if subjects != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected subject={EXPECTED_SUBJECTS}; found {subjects}")

    df["seed"] = df["seed"].astype(int)
    seeds = sorted(df["seed"].unique().tolist())
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds 0-19; found {seeds}")

    conditions = set(df["condition"].astype(str))
    if conditions != set(EXPECTED_CONDITIONS):
        raise ValueError(
            f"Expected conditions {EXPECTED_CONDITIONS}; found {sorted(conditions)}"
        )

    metrics = set(df["metric"].astype(str))
    if metrics != set(EXPECTED_METRICS):
        raise ValueError(f"Expected metrics {EXPECTED_METRICS}; found {sorted(metrics)}")

    expected_rows = len(EXPECTED_SEEDS) * len(EXPECTED_CONDITIONS) * len(EXPECTED_METRICS)
    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows (20 seeds x 4 conditions x 5 metrics); "
            f"found {len(df)}"
        )
    counts = df.groupby(["seed", "condition", "metric"]).size()
    if len(counts) != expected_rows or not counts.eq(1).all():
        raise ValueError("Expected exactly one row per seed x condition x metric")

    numeric = df[["value", "real_value", "absolute_error", "relative_error"]].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Table 4 input contains non-finite metric or error values")

    calculated = (numeric["value"] - numeric["real_value"]).abs()
    if not np.allclose(
        calculated.to_numpy(), numeric["absolute_error"].to_numpy(), rtol=0.0, atol=1e-12
    ):
        bad = np.where(
            ~np.isclose(
                calculated.to_numpy(),
                numeric["absolute_error"].to_numpy(),
                rtol=0.0,
                atol=1e-12,
            )
        )[0]
        raise ValueError(
            f"absolute_error mismatch at {len(bad)} row(s); "
            f"first indices={bad[:10].tolist()}"
        )

    for (seed, metric), group in df.groupby(["seed", "metric"]):
        real_values = group["real_value"].astype(float).to_numpy()
        if not np.allclose(real_values, real_values[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"Inconsistent real_value for seed={seed}, metric={metric}")

    real_rows = df.loc[df["condition"] == "Real"]
    if not np.allclose(
        real_rows["value"].astype(float),
        real_rows["real_value"].astype(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Real rows do not match real_value")
    return df


def build_table4_e3(table4_long: pd.DataFrame) -> pd.DataFrame:
    """Return the approved aligned self-EEG 20-seed E3 table.

    Real and generator columns contain mean raw metric values over seeds 0-19.
    Delta is mean absolute error(Diffusion-only) minus mean absolute error(Add).
    """
    rows: list[list[object]] = []
    for metric in EXPECTED_METRICS:
        group = table4_long.loc[table4_long["metric"] == metric]
        mean_raw: dict[str, float] = {}
        mean_absolute_error: dict[str, float] = {}
        for condition in EXPECTED_CONDITIONS:
            condition_group = group.loc[group["condition"] == condition].sort_values("seed")
            if len(condition_group) != len(EXPECTED_SEEDS):
                raise ValueError(
                    f"{metric}/{condition}: expected 20 rows, found {len(condition_group)}"
                )
            mean_raw[condition] = float(condition_group["value"].astype(float).mean())
            mean_absolute_error[condition] = float(
                condition_group["absolute_error"].astype(float).mean()
            )

        rows.append(
            [
                DISPLAY_LABELS[metric],
                mean_raw["Real"],
                mean_raw["NMM-only"],
                mean_raw["Diffusion-only"],
                mean_raw["Add"],
                mean_absolute_error["Diffusion-only"] - mean_absolute_error["Add"],
            ]
        )
    return pd.DataFrame(rows, columns=TABLE_HEADER)


def write_table4_outputs(
    table4_long: pd.DataFrame,
    table: pd.DataFrame,
    output_dir: Path,
    input_csv: Path,
    run_config_path: Path,
) -> None:
    tables_dir = output_dir / "tables" / "Table4"
    source_dir = output_dir / "source_data" / "Table4"
    documentation_dir = output_dir / "documentation"
    for directory in (tables_dir, source_dir, documentation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    table_csv = tables_dir / "Table4_E3_metricwise_20seed.csv"
    table_md = tables_dir / "Table4_E3_metricwise_20seed.md"
    source_csv = source_dir / "Table4_E3_metricwise_20seed_source.csv"
    meta_json = documentation_dir / "Table4_E3_metricwise_20seed_meta.json"

    table.to_csv(table_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    table4_source = table4_long.copy()
    table4_source["metric_label"] = table4_source["metric"].map(DISPLAY_LABELS)
    table4_source.to_csv(source_csv, index=False)

    markdown = [
        "| " + " | ".join(TABLE_HEADER) + " |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False, name=None):
        markdown.append(
            "| " + " | ".join([str(row[0]), *(f"{float(value):.3f}" for value in row[1:])]) + " |"
        )
    table_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    metadata = {
        "analysis": "Table 4 / E3 metric-wise quantitative comparison",
        "dataset": "self-recorded EEG",
        "input_csv": str(input_csv),
        "input_sha256": sha256(input_csv),
        "run_config": str(run_config_path),
        "run_config_sha256": sha256(run_config_path),
        "seeds": EXPECTED_SEEDS,
        "conditions": EXPECTED_CONDITIONS,
        "metrics": EXPECTED_METRICS,
        "aggregation": {
            "Real/NMM-only/Diffusion-only/Add": "mean raw metric value across seeds 0-19",
            "Δ(Diffusion-only - Add)": (
                "mean absolute_error(Diffusion-only) - mean absolute_error(Add)"
            ),
        },
        "interpretation": {
            "delta_negative": "Diffusion-only has smaller mean absolute error",
            "delta_positive": "Add has smaller mean absolute error",
            "delta_zero": "equal mean absolute error",
        },
        "paper_display_decimals": 3,
    }
    meta_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Aligned self-EEG run_config.json; defaults to input_csv.parent.parent/run_config.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the aligned input and provenance without writing Table 4 outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    run_config_path = (
        args.run_config.resolve()
        if args.run_config is not None
        else input_csv.parent.parent / "run_config.json"
    )

    validate_run_config(run_config_path)
    table4_long = load_table4_metrics_long(input_csv)
    table = build_table4_e3(table4_long)

    summary = {
        "status": "complete",
        "dataset": "self-recorded EEG",
        "rows": int(len(table4_long)),
        "seeds": EXPECTED_SEEDS,
        "conditions": EXPECTED_CONDITIONS,
        "metrics": EXPECTED_METRICS,
        "aligned_ddpm": {"T": 1000, "optimization_updates": 800, "batch_size": 8},
        "aggregation": (
            "20-seed mean raw metric values; delta = mean absolute error(Diffusion-only) "
            "- mean absolute error(Add)"
        ),
        "output_written": not args.validate_only,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return

    write_table4_outputs(
        table4_long,
        table,
        args.output_dir.resolve(),
        input_csv,
        run_config_path,
    )
    print(json.dumps(summary | {"output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
