#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

EXPECTED_SEEDS = list(range(20))
GENERATORS = ["NMM-only", "Diffusion-only", "Add", "WGAN-GP"]
COLUMNS = ["Real", *GENERATORS]
METRICS = [
    "psd_loglog_slope_all_2_40hz",
    "dfa_x_alpha",
    "acf_x_abs_0_1s",
    "acf_x_abs_0_5s",
    "acf_x_abs_1_0s",
]
METRIC_LABELS = {
    "psd_loglog_slope_all_2_40hz": "PSD slope\n(2–40 Hz)",
    "dfa_x_alpha": "DFA alpha",
    "acf_x_abs_0_1s": "ACF ≤ 0.1 s",
    "acf_x_abs_0_5s": "ACF ≤ 0.5 s",
    "acf_x_abs_1_0s": "ACF ≤ 1.0 s",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-data-dir", default=None, help="Separate output folder for Figure 6 source data")
    return p.parse_args()


def round_half_away_from_zero(value: float, digits: int) -> float:
    q = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def validate(df: pd.DataFrame) -> None:
    required = {
        "subject", "seed", "condition", "metric",
        "value", "real_value", "absolute_error", "relative_error"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if sorted(df["subject"].astype(str).unique().tolist()) != ["self_eeg"]:
        raise ValueError("Expected subject=self_eeg only")

    seeds = sorted(df["seed"].astype(int).unique().tolist())
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Expected seeds 0-19; found {seeds}")

    if set(df["condition"]) != set(COLUMNS):
        raise ValueError(f"Unexpected conditions: {sorted(df['condition'].unique().tolist())}")

    if set(df["metric"]) != set(METRICS):
        raise ValueError(f"Unexpected metric set: {sorted(df['metric'].unique().tolist())}")

    if len(df) != 20 * 5 * 5:
        raise ValueError(f"Expected 500 rows; found {len(df)}")

    counts = df.groupby(["seed", "condition", "metric"]).size()
    if len(counts) != 500 or not counts.eq(1).all():
        raise ValueError("Expected exactly one row per seed × condition × metric")

    calc_abs = (df["value"].astype(float) - df["real_value"].astype(float)).abs()
    if not np.allclose(
        calc_abs.to_numpy(),
        df["absolute_error"].astype(float).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("absolute_error != |value - real_value|")

    denom = df["real_value"].astype(float).abs()
    mask = denom > 0
    calc_rel = calc_abs[mask] / denom[mask]
    if not np.allclose(
        calc_rel.to_numpy(),
        df.loc[mask, "relative_error"].astype(float).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("relative_error != absolute_error / |real_value|")


def build_mean_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Figure 6 (formerly Supplementary Figure S1): 20-seed means in the Fig.2/Fig.3 layout.

    Rows: metrics
    Columns: Real + four generators
    Cell color: mean relative absolute error vs Real (Real column = 0)
    Cell text: mean raw metric value; generator cells also show mean relative error (%)
    """
    relative = np.zeros((len(METRICS), len(COLUMNS)), dtype=float)
    raw = np.zeros_like(relative)

    source_rows = []
    for i, metric in enumerate(METRICS):
        gm = df[df["metric"] == metric].copy()

        # Real is invariant over seeds; mean is safe and audited.
        real_rows = gm[gm["condition"] == "Real"].sort_values("seed")
        raw[i, 0] = float(real_rows["value"].astype(float).mean())

        for j, model in enumerate(GENERATORS, start=1):
            rows = gm[gm["condition"] == model].sort_values("seed")
            raw[i, j] = float(rows["value"].astype(float).mean())
            relative[i, j] = float(rows["relative_error"].astype(float).mean())

        source_rows.append({
            "metric": metric,
            "metric_label": METRIC_LABELS[metric].replace("\n", " "),
            **{f"{col}_mean_value": float(raw[i, j]) for j, col in enumerate(COLUMNS)},
            **{
                f"{model}_mean_relative_error": float(relative[i, j])
                for j, model in enumerate(GENERATORS, start=1)
            },
        })

    return relative, raw, pd.DataFrame(source_rows)


def winner_map(relative: np.ndarray) -> dict[str, str]:
    winners = {}
    for i, metric in enumerate(METRICS):
        vals = relative[i, 1:]
        minv = float(np.min(vals))
        tied = [
            GENERATORS[j]
            for j, v in enumerate(vals)
            if np.isclose(float(v), minv, rtol=0.0, atol=1e-12)
        ]
        winners[metric] = tied[0] if len(tied) == 1 else "tie:" + "|".join(tied)
    return winners


def draw_heatmap(
    relative: np.ndarray,
    raw: np.ndarray,
    out_base: Path,
    *,
    annotation_fontsize: float = 10.2,
) -> None:
    """Match the agreed Fig.2/Fig.3 heatmap presentation.

    - rows = metrics
    - columns = Real + generators
    - fixed viridis scale 0.00–0.40 (relative error)
    - Real column included
    - Real cell: raw metric value only
    - generator cell: raw metric value + relative error (%) underneath
    - white outline around winner among generators
    - text switches black/white according to cell luminance
    - no embedded figure title/caption
    """
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    im = ax.imshow(
        relative,
        cmap="viridis",
        vmin=0.0,
        vmax=0.40,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=25, ha="right")
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.tick_params(length=0)

    for i in range(relative.shape[0]):
        winner = 1 + int(np.argmin(relative[i, 1:]))
        ax.add_patch(
            Rectangle(
                (winner - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="white",
                linewidth=2.3,
            )
        )

        for j in range(relative.shape[1]):
            error = float(relative[i, j])
            metric_value = round_half_away_from_zero(float(raw[i, j]), 3)

            if j == 0:
                annotation = f"{metric_value:.3f}"
            else:
                annotation = f"{metric_value:.3f}\n({error * 100.0:.1f}%)"

            rgba = im.cmap(im.norm(error))
            luminance = (
                0.2126 * rgba[0]
                + 0.7152 * rgba[1]
                + 0.0722 * rgba[2]
            )
            color = "black" if luminance > 0.56 else "white"

            ax.text(
                j,
                i,
                annotation,
                ha="center",
                va="center",
                color=color,
                fontsize=annotation_fontsize,
                fontweight="normal",
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Relative absolute error vs Real")

    for suffix in ("png", "pdf", "svg"):
        path = out_base.with_suffix("." + suffix)
        if suffix == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")

    plt.close(fig)


def main():
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    source_dir = Path(args.source_data_dir).resolve() if args.source_data_dir else out_dir / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    validate(df)

    # Figure 6: former Supplementary Figure S1, 20-seed mean, same presentation as Fig. 2/3.
    fig6_rel, fig6_raw, fig6_source = build_mean_matrices(df)
    fig6_source.to_csv(source_dir / "Figure6_20seed_mean_source.csv", index=False)
    draw_heatmap(
        fig6_rel,
        fig6_raw,
        out_dir / "Figure6_self_EEG_four_generator_20seed_mean",
    )

    report = {
        "input_csv": (
            str(input_csv.relative_to(Path(__file__).resolve().parents[1]))
            if input_csv.is_relative_to(Path(__file__).resolve().parents[1])
            else str(input_csv)
        ),
        "rows": int(len(df)),
        "figure6": {
            "provenance": "renumbered from the former supplementary Figure S1",
            "aggregation": "20-seed mean raw metric value and mean relative absolute error",
            "winner_by_metric": winner_map(fig6_rel),
        },
        "figure_s2": "not generated or saved in the public repository",
        "heatmap_specification": {
            "orientation": "metrics on rows; Real and generators on columns",
            "columns": COLUMNS,
            "real_column": True,
            "cell_annotation": "raw metric value; generator cells add relative error (%) on second line",
            "color_quantity": "relative absolute error vs Real",
            "color_scale": [0.0, 0.40],
            "colormap": "viridis",
            "winner_outline": "white, generators only",
            "text_color": "luminance-adaptive black/white",
            "embedded_title": False,
        },
    }

    (source_dir / "Figure6_validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Figure 6 winners:", report["figure6"]["winner_by_metric"])
    print("[OK] Fig.2/Fig.3 heatmap specification applied")
    print(f"[OK] outputs: {out_dir}")


if __name__ == "__main__":
    main()
