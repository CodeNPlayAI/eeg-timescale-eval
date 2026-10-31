# Dependency versions: Mac Studio snapshot reconciliation (2026-10-31)

## Evidence and limitations

The user supplied `requirements-original.txt` from `python -m pip freeze` in the Mac Studio project environment. Its exact-version inventory records **38 packages** plus one editable Git-SSH installation referring to the parent project at commit `6e57bb8f552bc91239d5e5a10b8322ec8eba7c80`. The original supplied snapshot has SHA-256 `165a583264f03f081e631083d953d234209827ffad716d4e2a7b1a5389d7ac84`.

This is the environment **at snapshot time**, not definitive proof of versions installed during every historical EEG experiment. The pip-freeze snapshot alone does not record the Python version, but the user subsequently reported `Python 3.12.13` and `python -m pip check` returning `No broken requirements found.` The OS architecture, interpreter path, and package installation history were not recorded. No new clean-install test was performed with the updated Mac Studio pins in this working container. The previous Linux/Python 3.13.5 tests used NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, Torch 2.10.0, and MNE 1.11.0. Do **not** attribute those tests to this Mac snapshot.

## Direct pins for publication users

`requirements-figures.txt` contains the four directly imported plotting/data packages observed in the Mac snapshot. `requirements-experiments.txt` adds Torch/MNE and other directly used experiment packages. This exact pinning can be checked before release by a clean macOS arm64 Python environment. The figure requirements should not require PyTorch.

```bash
python -m pip install -r requirements-figures.txt
bash reproducibility/run_fig4_standalone.sh
bash reproducibility/run_figure5_openneuro_winners.sh
bash reproducibility/run_figure6_selfeeg_four_generator.sh
```

For model training/evaluation and EDF preprocessing:

```bash
python -m pip install -r requirements-experiments.txt
```

`python -m pip install -e .` installs the package from the **current checkout** (without fetching the project itself over GitHub SSH).

## Full observed inventory

`docs/MACSTUDIO_FREEZE_20261031.txt` contains all ordinary `==` entries from the Mac snapshot, including transitive packages. The original editable Git-SSH line is **not** copied into an executable requirements file: it may require GitHub SSH authorization and fetch a separate revision instead of the current public checkout. Its pinned commit is documented above for provenance. The inventory is **not** guaranteed reproducible across Python versions, Linux, macOS arm64, or future package indices.

## Direct-version comparison against earlier candidate

| Package | Earlier Linux-tested pin | Mac Studio observed pin |
|---|---|---|
| numpy | 2.3.5 | 2.4.4 |
| pandas | 2.2.3 | 3.0.2 |
| scipy | 1.17.0 | 1.17.1 |
| matplotlib | 3.10.8 | 3.10.8 |
| torch | 2.10.0 | 2.11.0 |
| mne | 1.11.0 | 1.12.1 |
| statsmodels | 0.14.6 | 0.14.6 |
| tqdm | 4.67.3 | 4.67.3 |
| PyYAML | 6.0.3 | 6.0.3 |

## Remaining verification

- Mac Studio Python 3.12.13 and `pip check`: user-reported success (existing environment). Architecture and interpreter path still not separately recorded.
- Clean macOS arm64 venv install with matching Python version, then archived-CSV reproduction for Figures 2–6 and tables.
- Any failures caused by differences between pandas 2.x and 3.x or NumPy 2.3 and 2.4 must be diagnosed before calling Mac pins validated.
- Historical 30-participant experiments were **not** rerun.
