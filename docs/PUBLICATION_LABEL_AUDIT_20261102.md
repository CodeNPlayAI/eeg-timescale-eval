# Publication label audit, 2026-11-02

The Figure 5 drawing code still used `OpenNeuro subject` and `Number of subjects`, although the revised manuscript used participant terminology. Figure 6 also spelled out `alpha` after the revised figure had used the Greek symbol. Rebuilding from these strings could reintroduce stale labels. Both the paper-cohort and subset Figure 5 drawing routes are now corrected.

Changes:

- Figure 5: `OpenNeuro participant` and `Number of participants`.
- Figure 6: `DFA α`; the source-data export uses Unicode α rather than math markup.
- Figures 2–6: `PSD slope` with `2–40 Hz` on the second line. Figure 4/5 restore the two-line layout. Figure 1 retains its schematic title-style `PSD Slope`.
- Published tables and their generation code: `DFA α` and participant wording.
- ACF range labels (`ACF ≤ 0.1 s`, `ACF ≤ 0.5 s`, `ACF ≤ 1.0 s`) are preserved. Lag zero is excluded by the metric definition.

No raw EEG, metric values, seed selections, BIDS participant IDs, CSV schema keys, or model settings were changed. Historical `source_data` display strings are retained for provenance and remain accepted by the loaders. Internal `subject` keys and established filenames are intentional compatibility names.

Validation performed:

1. Rebuilt all nine generated figure panels (Figures 2, 3a–c, 4a–b, 5a–b, 6); inspected the fixed Figure 1 separately.
2. Compared numeric text in each PDF with the pre-change repository PDFs: unchanged.
3. `python pipeline/run_openneuro.py selftest`: passed Table 5 means, SDs, medians and 95% CIs and Figure 5 counts (28/30, 526/600, model totals 11/31/18/90).
4. Exercised the subset drawing path with two participants and checked the generated axis labels.
5. Rebuilt Table 4 from the archived 400-row aligned source; all numeric columns agree with the archived table to absolute tolerance 1e-15.
6. `python reproducibility/check_publication_labels.py`: passed. This checks the actual generated SVGs, public table text and both Figure 5 drawing paths. Figure 1 still requires visual inspection.

Run the label check after regenerating figures. This check complements numeric validation; it does not rerun model training or prove the absence of every possible manuscript error.
