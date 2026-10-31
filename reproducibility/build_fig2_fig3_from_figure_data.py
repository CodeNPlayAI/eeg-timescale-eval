#!/usr/bin/env python3
"""Replot manuscript Fig. 2 and 3 from four saved, selected-seed metric CSVs.
The selection of seeds 3/2/16 was verified in the original 20-seed analysis;
these files do NOT allow rerunning the seed-selection algorithm independently.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility"))
from build_revision_figures_tables_20260814 import (  # noqa: E402
    configure_matplotlib, self_matrix, draw_self_heatmap, save_figure
)

PLOTS = [(0, "Fig2_self_EEG_seed0_core3"),
         (3, "Fig3a_self_EEG_seed3_core3"),
         (2, "Fig3b_self_EEG_seed2_core3"),
         (16, "Fig3c_self_EEG_seed16_core3")]

def main():
    configure_matplotlib()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for seed, stem in PLOTS:
        name = (f"Fig2_self_EEG_seed{seed}_source.csv" if seed == 0 else
                f"Fig3_self_EEG_seed{seed}_source.csv")
        path = ROOT / "source_data" / name
        df = pd.read_csv(path)
        if len(df) != 15 or set(df.seed) != {seed} or df.duplicated(['metric_key','model']).any():
            raise ValueError(f"Malformed saved selected-seed data: {path}")
        matrix, raw, _ = self_matrix(df, seed)
        fig, ax = plt.subplots(figsize=(9.2, 4.5))
        im = draw_self_heatmap(ax, matrix, raw, title=None,
                               show_ylabels=True, annotation_fontsize=10.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("Relative absolute error vs Real")
        save_figure(fig, out, stem)
        print("[OK]", out / f"{stem}.pdf")

if __name__ == "__main__":
    main()
