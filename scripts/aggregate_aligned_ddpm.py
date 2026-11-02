#!/usr/bin/env python3
"""Validate and aggregate aligned-DDPM self-EEG or OpenNeuro runs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    ("psd_loglog_slope_all_2_40hz", "PSD slope (2–40 Hz)"),
    ("dfa_x_alpha", "DFA α"),
    ("acf_x_abs_0_1s", "ACF@0.1s"),
    ("acf_x_abs_0_5s", "ACF@0.5s"),
    ("acf_x_abs_1_0s", "ACF@1.0s"),
]
CONDITIONS = ["Real", "NMM-only", "Diffusion-only", "Add"]
SEED_PATTERN = re.compile(r"seed(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Self run root or OpenNeuro batch root")
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def discover_subject_roots(root: Path) -> list[tuple[str, Path]]:
    subjects_dir = root / "subjects"
    if subjects_dir.is_dir():
        return [(path.name, path) for path in sorted(subjects_dir.glob("sub-*")) if path.is_dir()]
    return [("self_eeg", root)]


def seed_number(path: Path) -> int:
    match = SEED_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Invalid seed directory: {path}")
    return int(match.group(1))


def load_means(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col=0)
    if "mean" not in frame.columns:
        raise ValueError(f"Missing mean column: {path}")
    values: dict[str, float] = {}
    for key, _ in METRICS:
        if key not in frame.index:
            raise ValueError(f"Missing metric {key}: {path}")
        value = float(frame.loc[key, "mean"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite metric {key}: {path}")
        values[key] = value
    return values


def metric_paths(seed_root: Path, seed: int) -> dict[str, Path]:
    duration = "139.99609375"
    return {
        "Real": seed_root / f"real_start20_dur{duration}" / "metrics_summary.csv",
        "NMM-only": seed_root / f"nmm_seed{seed}_start20_dur{duration}" / "metrics_summary.csv",
        "Diffusion-only": seed_root / f"diffusion_only_start20_dur{duration}" / "metrics_summary.csv",
        "Add": seed_root / f"add_stdmatched_b1.0_start20_dur{duration}" / "metrics_summary.csv",
    }


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    e1_rows: list[dict[str, object]] = []
    e2_frames: list[pd.DataFrame] = []
    problems: list[str] = []

    for subject, subject_root in discover_subject_roots(root):
        seed_roots = sorted(
            [
                path
                for path in subject_root.glob("seed*")
                if path.is_dir() and SEED_PATTERN.fullmatch(path.name)
            ],
            key=seed_number,
        )
        for seed_root in seed_roots:
            seed = seed_number(seed_root)
            if not (seed_root / ".complete").is_file():
                problems.append(f"missing completion marker: {seed_root}")
            paths = metric_paths(seed_root, seed)
            try:
                loaded = {condition: load_means(path) for condition, path in paths.items()}
            except (FileNotFoundError, ValueError) as exc:
                problems.append(str(exc))
                continue

            for key, label in METRICS:
                real_value = loaded["Real"][key]
                for condition in CONDITIONS:
                    value = loaded[condition][key]
                    absolute_error = 0.0 if condition == "Real" else abs(value - real_value)
                    relative_error = (
                        0.0
                        if condition == "Real"
                        else absolute_error / abs(real_value)
                        if abs(real_value) > 0
                        else float("nan")
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "condition": condition,
                            "metric": key,
                            "metric_label": label,
                            "value": value,
                            "real_value": real_value,
                            "absolute_error": absolute_error,
                            "relative_error": relative_error,
                        }
                    )

            e1_specs = {
                "residual_true": seed_root / "res_true_eval_start0_dur139.99609375" / "metrics_summary.csv",
                "residual_hat": seed_root / "residual_hat_eval_start0_dur139.99609375" / "metrics_summary.csv",
                "residual_hat_stdmatched": seed_root
                / "residual_hat_stdmatched_eval_start0_dur139.99609375"
                / "metrics_summary.csv",
            }
            if any(path.exists() for path in e1_specs.values()):
                try:
                    for series, path in e1_specs.items():
                        values = load_means(path)
                        for key, label in METRICS[2:]:
                            e1_rows.append(
                                {
                                    "subject": subject,
                                    "seed": seed,
                                    "series": series,
                                    "metric": key,
                                    "metric_label": label,
                                    "value": values[key],
                                }
                            )
                except (FileNotFoundError, ValueError) as exc:
                    problems.append(str(exc))

            e2_path = seed_root / "e2_cross_term" / "e2_cross_term_summary.csv"
            if e2_path.is_file():
                e2 = pd.read_csv(e2_path)
                e2.insert(0, "subject", subject)
                e2_frames.append(e2)

    all_metrics = pd.DataFrame(rows)
    e1 = pd.DataFrame(e1_rows)
    e2 = pd.concat(e2_frames, ignore_index=True) if e2_frames else pd.DataFrame()
    return all_metrics, e1, e2, problems


def validate_counts(
    all_metrics: pd.DataFrame,
    expected_subjects: int,
    expected_seeds: int,
) -> list[str]:
    problems: list[str] = []
    if all_metrics.empty:
        return ["no model metrics were collected"]
    subjects = sorted(all_metrics["subject"].unique())
    if len(subjects) != expected_subjects:
        problems.append(f"subjects: expected {expected_subjects}, found {len(subjects)}")
    expected_seed_set = set(range(expected_seeds))
    expected_rows_per_seed = len(CONDITIONS) * len(METRICS)
    for subject in subjects:
        subject_frame = all_metrics[all_metrics["subject"] == subject]
        found_seed_set = set(subject_frame["seed"].unique())
        if found_seed_set != expected_seed_set:
            problems.append(
                f"{subject}: expected seeds {sorted(expected_seed_set)}, "
                f"found {sorted(found_seed_set)}"
            )
        counts = subject_frame.groupby("seed").size()
        bad = counts[counts != expected_rows_per_seed]
        if not bad.empty:
            problems.append(f"{subject}: incomplete condition/metric rows {bad.to_dict()}")
    return problems


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    summary_dir = root / "summary_aligned_ddpm_T1000_E800_B8"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, e1, e2, problems = collect(root)
    problems.extend(validate_counts(all_metrics, args.expected_subjects, args.expected_seeds))

    all_metrics_path = summary_dir / "all_metrics_long.csv"
    all_metrics.to_csv(all_metrics_path, index=False)

    if not all_metrics.empty:
        generated = all_metrics[all_metrics["condition"] != "Real"].copy()
        subject_seed_summary = (
            generated.groupby(["subject", "seed", "condition", "metric", "metric_label"], as_index=False)
            .agg(
                value=("value", "mean"),
                real_value=("real_value", "mean"),
                absolute_error=("absolute_error", "mean"),
                relative_error=("relative_error", "mean"),
            )
        )
        subject_seed_summary.to_csv(summary_dir / "generated_metrics_by_subject_seed.csv", index=False)

        e3 = generated.pivot_table(
            index=["subject", "seed", "metric", "metric_label"],
            columns="condition",
            values="absolute_error",
            aggfunc="first",
        ).reset_index()
        if {"Diffusion-only", "Add"}.issubset(e3.columns):
            e3["diffusion_minus_add_absolute_error"] = e3["Diffusion-only"] - e3["Add"]
            e3.to_csv(summary_dir / "e3_diffusion_minus_add_by_subject_seed.csv", index=False)

    if not e1.empty:
        e1.to_csv(summary_dir / "e1_residual_acf_by_seed.csv", index=False)
    if not e2.empty:
        e2.to_csv(summary_dir / "e2_cross_term_by_seed.csv", index=False)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "configuration": {
            "diffusion_only": {"T": 1000, "optimization_updates": 800, "batch_size": 8},
            "add_residual_diffusion": {"T": 1000, "optimization_updates": 800, "batch_size": 8},
        },
        "expected_subjects": args.expected_subjects,
        "expected_seeds_per_subject": args.expected_seeds,
        "found_subjects": 0 if all_metrics.empty else int(all_metrics["subject"].nunique()),
        "found_subject_seed_pairs": (
            0 if all_metrics.empty else int(all_metrics[["subject", "seed"]].drop_duplicates().shape[0])
        ),
        "model_metric_rows": int(len(all_metrics)),
        "e1_rows": int(len(e1)),
        "e2_rows": int(len(e2)),
        "problems": sorted(set(problems)),
        "status": "complete" if not problems else "incomplete",
    }
    (summary_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"[OK] metrics: {all_metrics_path}")
    print(f"[OK] validation: {summary_dir / 'validation_report.json'}")
    if problems:
        for problem in sorted(set(problems)):
            print(f"[PROBLEM] {problem}")
        if not args.allow_incomplete:
            raise SystemExit("Aggregation failed completeness checks")
    else:
        print("[OK] completeness checks passed")


if __name__ == "__main__":
    main()
