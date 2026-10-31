# Data scope and provenance

- `OpenNeuro_all_errors_by_subject_seed_model_metric.csv`: the complete archived **30 participants × 20 seeds × 4 generators × 5 metrics** evaluation grid (12,000 rows); **not raw EEG**. Fig. 5 can be recomputed from this table.
- `Figure6_selfEEG_core3_plus_wgan_metrics_long.csv`: complete 20 seeds × 5 conditions × 5 metrics (500 rows, including Real); Fig. 6 input.
- `Table4_E3_metricwise_20seed_source.csv`: archived **all 20 seeds × 4 conditions × 5 metrics** (400 rows) for E3. `Table4_E3_metricwise_20seed.csv` is the rounded published summary used by the existing Fig. 4 standalone renderer. Hence Fig. 4 plotting is verifiable but not recreated from original EEG time series.
- `Fig2_self_EEG_seed0_source.csv`, `Fig3_self_EEG_seed{3,2,16}_source.csv`: **four deliberately selected example seeds**, each 15 model-metric rows. Sufficient for the published Fig. 2 / Fig. 3 heatmaps; **not a complete 20-seed experiment**. The original 20-seed representative-seed selection cannot be independently rerun from these four files.
- `Table2_E1_all_seeds_source.csv`, `Table3_self_eeg_E2_aggregated.csv`, `Table5_OpenNeuro_absolute_error_long.csv`: preserved table provenance. The full underlying E2 seed-level records are not bundled.
- `subject_level_winners_extended4.csv`: verified historic reference used to cross-check Fig. 5 aggregation; not trusted as plot input.

Raw EEG recordings, all synthetic EEG traces, training checkpoints, and all E2 seed-level files are not bundled. Re-running the *entire* experimental study requires those inputs.

The original plotting/aggregation functions for Figure 5 are reproduced verbatim in `reproducibility/openneuro_figure5_core.py`, except corrected ACF display strings. The obsolete seed-0-only Fig. 4 builder and `legacy/` are excluded.

The `source` column of `OpenNeuro_all_errors_by_subject_seed_model_metric.csv` and the paths in `reproducibility/self_eeg_*.json` record where each value came from in the authors' project. The authors' local home-directory prefix is replaced by `author-project/`; all numeric values are unchanged.
