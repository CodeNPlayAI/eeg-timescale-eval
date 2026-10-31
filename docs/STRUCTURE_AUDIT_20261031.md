# Publication package audit, 2026-10-31

Changes compared with the previous candidate:
- Removed `legacy/` and the supplementary tables whose outputs are not the manuscript Figure S2 and have no included current builder.
- Preserved **only Figure 5's original verified statistical and plotting functions** in `reproducibility/openneuro_figure5_core.py`, correcting ACF display labels, and removed the obsolete seed-0-only Fig. 4 builder.
- `figures/` is the **only** figure output destination. Removed duplicated frozen and generated trees.
- Discarded derived `outputs/` cache; it is excluded from Git and recreated for Figure 5/6 numeric validation reports.
- Reduced `source_data/` to figure/table-specific inputs, historical winner validation and explicit data-provenance records. Four representative seed CSVs are intentionally included to reproduce Fig. 2/3 and **are not a full 20-seed dataset**.
- Preserved archived 12,000-row OpenNeuro and 500-row self-EEG full evaluation metric grids for Figures 5/6.
- No scientific training or EEG metric formula changed; scripts retain their original full experiment entry points.

Validation completed in container with existing dependencies: Fig. 2, Fig. 3(a-c), Fig. 4(a-b), Fig. 5(a-b), Fig. 6 were regenerated. Fig. 4 uses rounded published Table 4 input, not new EEG evaluations. This is **figure regeneration**, not a clean MacOS or full raw data-to-figure replication.

Check archived values using `bash reproducibility/run_figure5_openneuro_winners.sh --validate-only` and figure generators in README.
