# Public-release candidate audit — 2026-10-31

- Checked **33 Python** files by AST parser and **10 shell** scripts with `bash -n`.
- Added author-supplied final Figure 1 PDF (rev4); existing Figures 2–6 retained in the canonical `figures/` folder.
- Added Table 3 CSV and readable Markdown rendering from archived aggregate data; **no claim to recreate missing seed-level E2 inputs**.
- Normalized the local project distribution name to `eeg-timescale-eval`.
- Excluded the limited sub603 reference CSV, which is not a paper-table reconstruction input.
- No `legacy/`, `Supplementary/`, or duplicate `outputs/publication/figures/` tree.
- Existing experimental training scripts contain references to local `runs/` and raw EEG data not distributed in the repository; **not guaranteed to execute standalone**.
- Figure regeneration had been previously verified in the parent clean candidate; these checks do **not** constitute a fresh complete model-training replay.
- No legal license has yet been selected or authorized by the code copyright holders. Do not claim MIT/GPL/Apache permissions.
- GitHub repository has **not** been created or pushed from this file archive.

## Final packaging update

MIT LICENSE added with copyright line "Hiroshi Murai and contributors"; `setup_macos.sh` uses a project-local `venv/`, and `.gitignore` already excludes `venv/`. The complete clean macOS installation has **not** been executed by this packaging step.
