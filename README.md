# EEG multi-timescale evaluation

Code and data for the IEEE Access article **"A Multi-Timescale Evaluation Framework for EEG Generation Reveals Metric-Dependent Trade-offs Across Temporal Scales"** (Murai, Ishihara, Hosokawa, Fujimoto).

The framework scores generated EEG against real EEG on five measures: PSD slope (2–40 Hz), DFA alpha, and the mean absolute autocorrelation up to 0.1, 0.5 and 1.0 s (`ACF ≤ 0.1 s`, `ACF ≤ 0.5 s`, `ACF ≤ 1.0 s`; lag 0 excluded). Four generators are compared: NMM-only (Jansen–Rit neural mass model), Diffusion-only (DDPM), Add (NMM plus a diffusion-generated residual) and WGAN-GP.

## What you can reproduce

| Paper result | Data | How |
|---|---|---|
| **Table 5, Fig. 5** (30 participants × 20 seeds, four generators) | Public: OpenNeuro ds005385 v1.0.2 | **Full pipeline from raw EEG** — `pipeline/run_openneuro.py` (this README) |
| Table 5, Fig. 5 from the archived evaluation table | Included (`source_data/`) | `pipeline/run_openneuro.py selftest` (about 1 minute) |
| Figs. 2–4, 6; Tables 1–4 (self-recorded EEG) | Not public (recorded from an author) | Figures only, from archived summaries — see [Figures from archived data](#figures-from-archived-data) |

The self-recorded EEG cannot be shared, so the experiments behind Figs. 2–4, 6 and Tables 2–4 cannot be rerun by third parties. Their figures can be redrawn from the archived summaries in `source_data/`.

## Quick start

```bash
git clone https://github.com/CodeNPlayAI/eeg-timescale-eval.git
cd eeg-timescale-eval
bash setup_macos.sh --experiments           # venv/ with the pinned packages (macOS or Linux)
source venv/bin/activate

python pipeline/run_openneuro.py selftest                      # 1. installation check, no EEG needed
python pipeline/download_openneuro.py --n-subjects 1           # 2. download one participant
python pipeline/run_openneuro.py run --n-subjects 1 --n-seeds 1 --run-name smoke   # 3. one complete run
```

When the one-participant run succeeds, start the paper-scale run (30 participants × 20 seeds):

```bash
python pipeline/download_openneuro.py
python pipeline/run_openneuro.py run --yes
```

## 1. Requirements

- macOS or Linux with `bash` (on Windows, use WSL).
- Python **3.12** (3.11–3.13 run; the paper used 3.12.13).
- An Apple-silicon GPU (MPS) or an NVIDIA GPU is strongly recommended. The code runs on CPU, but slowly — see [Run time and disk space](#run-time-and-disk-space).
- Disk space for the run outputs (see [Run time and disk space](#run-time-and-disk-space)). Only the recordings of the selected participants are downloaded, not the whole dataset.

## 2. Install

```bash
bash setup_macos.sh --experiments
source venv/bin/activate
```

This creates `venv/` inside the repository and installs `requirements-experiments.txt` (NumPy, SciPy, pandas, Matplotlib, PyTorch, MNE, statsmodels, openneuro-py at the versions used in the paper). Without `--experiments`, only the plotting packages are installed, which is enough for `selftest` and the archived figures.

Manual alternative:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-experiments.txt
python -m pip check
```

## 3. Self-test (about 1 minute)

```bash
python pipeline/run_openneuro.py selftest
```

This rebuilds Table 5 and Fig. 5 from the archived evaluation table (`source_data/OpenNeuro_all_errors_by_subject_seed_model_metric.csv`, 12,000 rows) with the same code that the full pipeline uses, and checks the published numbers: Table 5 means, SDs, medians and bootstrap CIs, 28 of 30 participants and 526 of 600 runs with differing closest generators, and winner counts 11/31/18/90. It ends with `[PASS]`.

## 4. Download the OpenNeuro data

```bash
python pipeline/download_openneuro.py                       # the paper's 30 participants
python pipeline/download_openneuro.py --n-subjects 3         # first 3 of them
python pipeline/download_openneuro.py --subjects sub-012 sub-407
```

Only the eyes-closed, pre-task, session-1 recording of each selected participant is downloaded, plus the dataset's top-level metadata. The files are placed here (the folder is git-ignored):

```text
eeg-timescale-eval/
└── data/
    └── openneuro/
        └── ds005385/                        # BIDS root  (--bids-root)
            ├── dataset_description.json
            ├── participants.tsv
            ├── README, CHANGES
            ├── sub-012/
            │   └── ses-1/
            │       └── eeg/
            │           └── sub-012_ses-1_task-EyesClosed_acq-pre_eeg.edf
            ├── sub-050/ ...
            └── sub-603/ ...
```

If you already have ds005385 elsewhere (for example from DataLad or the OpenNeuro website), keep it there and pass `--bids-root /path/to/ds005385` to every command. It must be version **1.0.2** and keep the BIDS layout above. With DataLad, run `datalad get` on the files first; the pre-flight check reports files that are only placeholders.

## 5. Check before running

```bash
python pipeline/run_openneuro.py check --n-subjects 1 --n-seeds 1
```

`run` performs the same checks automatically and stops before any computation if one fails. The check reports:

| Area | What is checked |
|---|---|
| Python | Version; every required package importable; versions that differ from the paper's (warning) |
| Hardware | GPU (CUDA or MPS) or CPU only (warning) |
| Repository | All scripts and archived files the pipeline uses are present |
| Data | BIDS folder and `dataset_description.json`; for each participant, the EDF exists, is not a broken DataLad link or a truncated download, can be read, contains F7 and F8, and lasts at least 180 s. Missing participants are listed with the exact download command |
| Output | An existing run folder was made with the same settings (otherwise it is never mixed); enough free disk space |

## 6. Run the pipeline

```bash
python pipeline/run_openneuro.py run [selection] [options]
```

### Choosing participants and seeds

| Option | Meaning | Default |
|---|---|---|
| `--n-subjects N` | The first N of the paper's 30 participants, in paper order (sub-012, sub-050, …) | 30 |
| `--subjects sub-012 sub-407 …` | Explicit participants. Labels outside the paper's 30 are allowed, but they cannot be compared with the paper | — |
| `--n-seeds N` | Seeds 0 … N−1 | 20 |
| `--seeds 0 5 7 …` | Explicit seeds | — |

The same selection options work for `download_openneuro.py`, `check`, `run`, `results` and `selftest`.

### Other options

| Option | Meaning |
|---|---|
| `--run-name NAME` | Output folder `runs/NAME` (default `openneuro_<N>subj_<M>seeds`) |
| `--skip-wgan` | Run NMM-only, Diffusion-only and Add only |
| `--stop-after preprocess` / `core3` | Stop after that stage |
| `--keep-going` | If one participant fails, continue with the others and report it at the end |
| `--yes` | Do not ask for confirmation of long runs (more than 4 participant-seeds) |
| `--bids-root PATH`, `--runs-dir PATH` | Data and output locations |

### What `run` does

1. **Pre-flight check** (section 5).
2. **Preprocessing** — F7 and F8, resampled to 256 Hz, first 180 s, converted to µV (`scripts/preprocess_openneuro_cli.py`).
3. **Real-EEG check** — each preprocessed recording must have 46,080 finite samples per channel and a non-zero SD. Constant stretches of 1 s or more and many samples at the minimum or maximum (clipping) are reported as warnings. The SD of each channel is recorded in `reports/real_data_check.csv` for reference.
4. **NMM-only, Diffusion-only and Add** for every participant and seed, followed by evaluation of these generators and of the real recording (`scripts/run_aligned_ddpm_one_dataset.sh`).
5. **WGAN-GP** training and generation (`scripts/generate_gan_only.py`); every generated signal is checked for length and finite values, then evaluated with the same protocol (`scripts/evaluate.py`).
6. **Results** — error table, participant-level winners, Table 5, Fig. 5 and a comparison with the paper's archived values.

The protocol is fixed to the paper's: evaluation interval 20–160 s, 1–45 Hz band-pass, global z-score, 10-s windows with a 5-s hop, DFA scales 0.2–1.25 s; DDPM with T = 1000, 800 updates, batch 8; WGAN-GP with n_critic = 3, λ_GP = 50, 3000 generator updates. As in the paper, each generator is trained on the same participant's recording that it is evaluated against (in-sample temporal fidelity). The settings are written to `run_config.json` in the run folder.

### Resuming

Every step writes its result before the next one starts. If a run stops (error, power loss, Ctrl-C), run **the same command again**: finished participants, seeds and evaluations are detected and skipped. A different participant or seed selection needs a new `--run-name`; runs with different settings are never mixed.

## 7. Outputs

```text
runs/<run-name>/
├── run_config.json                   # selection, protocol, Python and platform
├── reports/
│   ├── preflight.json
│   └── real_data_check.csv / .json
├── logs/                             # full output of every step (preprocess, core3_<sub>, wgan_<sub>)
├── preprocessed/                     # sub-XXX_ses-1_EyesClosed_acq-pre_180s_256Hz.csv
├── subjects/sub-XXX/seed<N>/         # generated signals and per-generator evaluations
│   ├── generated/
│   ├── real_start20_dur139.99609375/metrics_summary.csv
│   ├── nmm_seed<N>_…/  diffusion_only_…/  add_stdmatched_b1.0_…/  gan_…/
├── wgan/sub-XXX/                     # WGAN-GP signals, checkpoints and training diagnostics
└── results/
    ├── errors_by_subject_seed_model_metric.csv   # same columns as the archived table
    ├── subject_mean_errors.csv
    ├── subject_level_winners.csv, seed_level_winners.csv
    ├── Table5_absolute_error_95CI.csv / .md       # Table 5
    ├── Table5_long.csv
    ├── figures/Fig5a_OpenNeuro_subjectwise_winners.{pdf,png,svg}   # Fig. 5(a)
    ├── figures/Fig5b_OpenNeuro_winner_counts.{pdf,png,svg}         # Fig. 5(b)
    ├── summary.json                               # counts reported in the paper
    ├── comparison_with_paper.json
    └── comparison_with_paper_rows.csv
```

To rebuild the results of an existing run, for example after an interruption with `--keep-going`:

```bash
python pipeline/run_openneuro.py results --run-name NAME [--allow-incomplete]
```

## 8. Comparing your run with the paper

For participants and seeds that are part of the paper, `comparison_with_paper.json` sets your values beside the archived ones.

- **Real-EEG metrics and NMM-only** are deterministic. They should agree with the paper to numerical precision on any machine.
- **Diffusion-only, Add and WGAN-GP** are trained neural networks. On the authors' Mac Studio (Apple silicon, MPS) with the pinned packages, a rerun of sub-012, seed 0 reproduced the archived values of all four generators exactly (largest difference 1.7e-11). On other hardware (CPU, CUDA) or with other library versions, their seed-level values can differ from the paper; compare participant-level errors, winners and the counts in `summary.json` instead.

## Run time and disk space

Each participant-seed trains three networks (two DDPMs and one WGAN-GP). Measured per participant-seed:

| Hardware | NMM-only + Diffusion-only + Add | WGAN-GP | Full run (30 × 20) |
|---|---|---|---|
| Mac Studio (2022, Apple silicon, MPS) | 0.8 min | 5.3 min | about 62 hours |
| 2-core Linux CPU, no GPU | 5 min | 67 min | about 30 days |
| Other machines | measure with the one-participant run | | |

Measure your machine with `--n-subjects 1 --n-seeds 1`, then multiply by the number of participant-seeds (600 for the paper). WGAN-GP takes most of the time; `--skip-wgan` runs the other three generators only.

Disk use is about 30 MB per participant-seed, or about **20 GB** for the full run. The WGAN-GP checkpoints (`runs/<run-name>/wgan/sub-XXX/seed<N>/generator_seed<N>.pt`) can be deleted after the run if space is short.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `missing: sub-XXX/ses-1/eeg/...edf` | Not downloaded. Run the `download_openneuro.py` command printed by the check |
| `broken link (DataLad/git-annex placeholder ...)` | The file was cloned with DataLad but not fetched: `datalad get <file>` |
| `file is only N bytes` | Interrupted download. Delete the file and download again |
| `channels ['F7', 'F8'] not in the file` / `recording is N s` | Wrong file or dataset version. Use ds005385 v1.0.2 |
| `... was created with different settings` | The run folder belongs to another selection. Use a new `--run-name` |
| `Command failed ... Last lines of logs/...` | The step's full output is in the named log file. Fix the cause and run the same command again |
| `Package X: ... installed; paper environment used ...` | Warning only. Use `setup_macos.sh --experiments` for the paper's versions |

## Figures from archived data

These scripts redraw the published figures from `source_data/` without EEG or training. Install with `bash setup_macos.sh` (no `--experiments` needed).

```bash
bash reproducibility/run_fig2_fig3.sh                   # Figs. 2-3   (self-recorded EEG, selected seeds)
bash reproducibility/run_fig4_standalone.sh             # Fig. 4      (self-recorded EEG, rounded Table 4 summary)
bash reproducibility/run_figure5_openneuro_winners.sh   # Fig. 5      (OpenNeuro, archived 30 x 20 table)
bash reproducibility/run_figure6_selfeeg_four_generator.sh   # Fig. 6 (self-recorded EEG, 20 seeds)
```

These scripts overwrite the files in `figures/`; the redrawn images preserve the archived numerical results. Publication-label corrections are documented in [the label audit](docs/PUBLICATION_LABEL_AUDIT_20261102.md). After rebuilding, run `python reproducibility/check_publication_labels.py` to check the visible labels. Figure 1 (`figures/Figure1.pdf`) is a schematic. `docs/PUBLIC_DATA_SCOPE.md` describes each archived file.

## Author-only scripts

The following scripts reproduce the self-recorded EEG experiments and need that recording, which is not distributed: `run_self_eeg_aligned_ddpm_20seeds.sh`, `run_table3_e2_v2.sh`, `reproducibility/run_table4_e3_v3.sh`, `reproducibility/run_self_eeg_wgan_evaluation_20260815.py`. The earlier OpenNeuro batch scripts (`run_openneuro_aligned_ddpm_30subj_20seeds.sh`, `run_gan_only_30subj_macos_v2.sh`, `evaluate_gan_30subj_20seeds.py`) are kept for provenance; `pipeline/run_openneuro.py` runs the same steps with the same settings in one command.

## Repository layout

```text
pipeline/          one-command OpenNeuro pipeline (download, check, run, results, selftest)
scripts/, src/     preprocessing, generators, evaluation (used by the pipeline)
reproducibility/   figure and table builders from archived data
source_data/       archived evaluation tables (no raw EEG)
figures/, tables/  published figures and tables
docs/              data scope, dependency notes and audits
```

## Data, license and citation

- OpenNeuro ds005385 is distributed by OpenNeuro under its own terms; cite the dataset as described on its OpenNeuro page.
- Code and documentation: [MIT License](LICENSE).
- Please cite the article; metadata are in `CITATION.cff`.
