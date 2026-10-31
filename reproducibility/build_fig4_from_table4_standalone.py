#!/usr/bin/env python3
"""
Standalone Fig. 4 builder for the IEEE Access revision.

Purpose
-------
Rebuild Fig. 4(a)/(b) on a PC that does not contain the original repository.

Required input in the SAME directory:
    Table4_E3_metricwise_20seed.csv

Fig. 4(b):
    Uses the full-precision Delta column from Table 4 directly.

Fig. 4(a):
    Uses the locked 20-seed mean absolute errors previously validated from the
    aligned self-EEG Core-3 analysis. These values are stored here to 4 decimal
    places because the original seed-level source file is not present on this PC.
    They are sufficient for the plotted point locations but are NOT a replacement
    for the full-precision seed-level reproducibility source.

When the original repository is available again, the final audit should compare
these plotted values against Fig4_20seed_mean_source.csv generated from the
seed-level aligned results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORE_MODELS = ["NMM-only", "Diffusion-only", "Add"]
METRICS = [
    "PSD slope (2-40 Hz)",
    "DFA alpha",
    "ACF 0.1 s",
    "ACF 0.5 s",
    "ACF 1.0 s",
]
METRIC_LABEL = {
    "PSD slope (2-40 Hz)": "PSD slope (2-40 Hz)",
    "DFA alpha": r"DFA $\alpha$",
    "ACF 0.1 s": "ACF ≤ 0.1 s",
    "ACF 0.5 s": "ACF ≤ 0.5 s",
    "ACF 1.0 s": "ACF ≤ 1.0 s",
}
MODEL_COLORS = {
    "NMM-only": "#0072B2",
    "Diffusion-only": "#E69F00",
    "Add": "#009E73",
}

# Locked 20-seed mean absolute errors from the final aligned self-EEG analysis.
# Stored to 4 decimals because the full seed-level source is unavailable on this PC.
LOCKED_MAE_4DP = {
    "PSD slope (2-40 Hz)": {
        "NMM-only": 0.2517,
        "Diffusion-only": 0.0439,
        "Add": 0.1919,
    },
    "DFA alpha": {
        "NMM-only": 0.1736,
        "Diffusion-only": 0.0989,
        "Add": 0.1134,
    },
    "ACF 0.1 s": {
        "NMM-only": 0.0283,
        "Diffusion-only": 0.0153,
        "Add": 0.0141,
    },
    "ACF 0.5 s": {
        "NMM-only": 0.0092,
        "Diffusion-only": 0.0149,
        "Add": 0.0151,
    },
    "ACF 1.0 s": {
        "NMM-only": 0.0053,
        "Diffusion-only": 0.0093,
        "Add": 0.0089,
    },
}


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def normalize_metric(s: str) -> str:
    s = str(s).strip()
    s = s.replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)
    aliases = {
        "PSD slope (2-40 Hz)": "PSD slope (2-40 Hz)",
        "DFA alpha": "DFA alpha",
        "DFA α": "DFA alpha",
        "ACF 0.1 s": "ACF 0.1 s",
        "ACF@0.1s": "ACF 0.1 s",
        "ACF@0.1 s": "ACF 0.1 s",
        "ACF ≤ 0.1 s": "ACF 0.1 s",
        "ACF 0.5 s": "ACF 0.5 s",
        "ACF@0.5s": "ACF 0.5 s",
        "ACF@0.5 s": "ACF 0.5 s",
        "ACF ≤ 0.5 s": "ACF 0.5 s",
        "ACF 1.0 s": "ACF 1.0 s",
        "ACF@1.0s": "ACF 1.0 s",
        "ACF@1.0 s": "ACF 1.0 s",
        "ACF ≤ 1.0 s": "ACF 1.0 s",
    }
    if s not in aliases:
        raise ValueError(f"Unexpected Table 4 metric label: {s!r}")
    return aliases[s]


def detect_delta_column(df: pd.DataFrame) -> str:
    candidates = [
        c for c in df.columns
        if "delta" in c.lower() or "Δ" in c
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely identify Table 4 Delta column. "
            f"Columns: {list(df.columns)}"
        )
    return candidates[0]


def load_table4(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"Metric", "Real", "NMM-only", "Diffusion-only", "Add"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Table 4 columns: {sorted(missing)}")

    delta_col = detect_delta_column(df)
    df = df.copy()
    df["metric_norm"] = df["Metric"].map(normalize_metric)

    if set(df["metric_norm"]) != set(METRICS) or len(df) != 5:
        raise ValueError(
            "Table 4 must contain exactly the locked five E3 metrics."
        )

    for c in ["Real", "NMM-only", "Diffusion-only", "Add", delta_col]:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c].to_numpy()).all():
            raise ValueError(f"Non-finite value in Table 4 column {c}")

    df = df.set_index("metric_norm").reindex(METRICS).reset_index()
    df = df.rename(columns={delta_col: "delta_table4"})
    return df


def make_source(table4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in table4.iterrows():
        metric = r["metric_norm"]
        diff_rounded = LOCKED_MAE_4DP[metric]["Diffusion-only"]
        add_rounded = LOCKED_MAE_4DP[metric]["Add"]
        delta_from_rounded_mae = diff_rounded - add_rounded
        delta_table4 = float(r["delta_table4"])

        rows.append({
            "metric": METRIC_LABEL[metric],
            "Real_mean_value_Table4": float(r["Real"]),
            "NMM-only_mean_value_Table4": float(r["NMM-only"]),
            "Diffusion-only_mean_value_Table4": float(r["Diffusion-only"]),
            "Add_mean_value_Table4": float(r["Add"]),
            "NMM-only_mean_absolute_error_locked_4dp": LOCKED_MAE_4DP[metric]["NMM-only"],
            "Diffusion-only_mean_absolute_error_locked_4dp": diff_rounded,
            "Add_mean_absolute_error_locked_4dp": add_rounded,
            "delta_Table4_full_precision": delta_table4,
            "delta_from_locked_MAE_4dp": delta_from_rounded_mae,
            "delta_rounding_difference": delta_from_rounded_mae - delta_table4,
        })

    source = pd.DataFrame(rows)
    max_diff = float(source["delta_rounding_difference"].abs().max())
    # 4-dp MAEs imply at most roughly 1e-4 disagreement in their difference.
    if max_diff > 1.1e-4:
        raise ValueError(
            "Table 4 Delta is inconsistent with the locked 4-dp mean absolute errors. "
            f"max difference={max_diff:.8g}. Refusing to plot."
        )
    return source


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_fig4a(source: pd.DataFrame, out_dir: Path) -> None:
    y = np.arange(len(METRICS))
    offsets = {"NMM-only": -0.22, "Diffusion-only": 0.0, "Add": 0.22}
    markers = {"NMM-only": "o", "Diffusion-only": "s", "Add": "D"}

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    for model in CORE_MODELS:
        values = source[f"{model}_mean_absolute_error_locked_4dp"].to_numpy(float)
        if np.any(values <= 0):
            raise ValueError("Log axis requires strictly positive mean absolute errors.")
        ax.scatter(
            values,
            y + offsets[model],
            s=58,
            color=MODEL_COLORS[model],
            marker=markers[model],
            edgecolor="white",
            linewidth=0.7,
            label=model,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yticks(y, [METRIC_LABEL[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error relative to Real across 20 seeds (log scale)")
    ax.grid(axis="x", which="both", color="#D7DDE2", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, out_dir, "Fig4a_E3_metricwise_absolute_error_20seed")


def plot_fig4b(source: pd.DataFrame, out_dir: Path) -> None:
    y = np.arange(len(METRICS))
    delta = source["delta_Table4_full_precision"].to_numpy(float)
    colors = [
        MODEL_COLORS["Diffusion-only"] if d < 0 else MODEL_COLORS["Add"]
        for d in delta
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    bars = ax.barh(y, delta, color=colors, height=0.72)
    ax.axvline(0, color="#26323D", linewidth=1.25)
    ax.set_yticks(y, [METRIC_LABEL[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlabel(
        r"$\Delta$ mean absolute error = mean|Diffusion-only - Real| - mean|Add - Real|"
    )
    ax.grid(axis="x", color="#D7DDE2", linewidth=0.75)
    ax.set_axisbelow(True)

    max_abs = max(abs(float(v)) for v in delta)
    pad = max_abs * 0.04 if max_abs else 0.01

    for i, (bar, value) in enumerate(zip(bars, delta)):
        if i == 0 and value < 0:
            x, ha = value + pad, "left"
        else:
            x = value - pad if value < 0 else value + pad
            ha = "right" if value < 0 else "left"

        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            va="center",
            ha=ha,
            fontsize=8.5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, out_dir, "Fig4b_E3_diffusion_minus_add_20seed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--table4",
        type=Path,
        default=Path(__file__).resolve().parent / "Table4_E3_metricwise_20seed.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Fig4_standalone_output",
    )
    args = p.parse_args()

    table4 = args.table4.resolve()
    out = args.output_dir.resolve()
    if not table4.is_file():
        raise FileNotFoundError(
            f"Table 4 CSV not found: {table4}\n"
            "Place Table4_E3_metricwise_20seed.csv in the same directory as this script."
        )

    configure_matplotlib()
    t4 = load_table4(table4)
    source = make_source(t4)

    out.mkdir(parents=True, exist_ok=True)
    validation_dir = Path(__file__).resolve().parents[1] / "outputs" / "validation" / "Figure4"
    validation_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(validation_dir / "Fig4_20seed_standalone_source.csv", index=False)

    print("=== Fig. 4 standalone source ===")
    print(source[[
        "metric",
        "NMM-only_mean_absolute_error_locked_4dp",
        "Diffusion-only_mean_absolute_error_locked_4dp",
        "Add_mean_absolute_error_locked_4dp",
        "delta_Table4_full_precision",
        "delta_rounding_difference",
    ]].to_string(index=False))

    plot_fig4a(source, out)
    plot_fig4b(source, out)

    print()
    print("[OK] Fig. 4(a): locked aligned 20-seed mean absolute errors (4-dp)")
    print("[OK] Fig. 4(b): full-precision Delta read directly from Table 4 CSV")
    print("[OK] PDF / SVG / 600-dpi PNG generated")
    print(f"[OK] Output: {out}")
    print("[NOTE] Final audit on the original PC: compare against full-precision seed-level source.")


if __name__ == "__main__":
    main()
