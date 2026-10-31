#!/usr/bin/env python3
"""One-command reproduction of the OpenNeuro (ds005385) analysis.

Commands
--------
  selftest  Rebuild Table 5 and Figure 5 from the archived paper data and check that
            the published numbers are reproduced (about 1 minute, no EEG needed).
  check     Pre-flight checks only: Python packages, repository files, downloaded EEG
            files (missing, broken, too short, wrong channels), output folder, disk space.
  run       check -> preprocess -> real-EEG check -> NMM-only / Diffusion-only / Add
            -> WGAN-GP -> evaluation -> Table 5, Figure 5 and comparison with the paper.
            Interrupted runs resume where they stopped when re-run with the same options.
  results   Rebuild the tables and figures of an existing run folder.

Participant and seed selection (all commands)
---------------------------------------------
  --n-subjects N | --subjects sub-012 ...   default: the paper's 30 participants
  --n-seeds N    | --seeds 0 1 ...          default: seeds 0-19

Examples
--------
  python pipeline/run_openneuro.py selftest
  python pipeline/run_openneuro.py check --n-subjects 1 --n-seeds 1
  python pipeline/run_openneuro.py run   --n-subjects 1 --n-seeds 1 --run-name smoke
  python pipeline/run_openneuro.py run   --yes                       # full paper run (30 x 20)
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import (  # noqa: E402
    ALL_MODELS, CHANNELS, CORE_MODELS, DATASET_ID, DATASET_VERSION, DEFAULT_BIDS_ROOT,
    DEFAULT_RUNS_DIR, DURATION_SEC, EVALUATE_ARGS, FS, PAPER_SEEDS, PAPER_SUBJECTS,
    PREPROCESSED_ROWS, PROTOCOL, REPO_ROOT, Report, StepFailed, WGAN_ARGS, add_selection_args,
    edf_relpath, eval_dirname, log, now, preprocessed_name, python_exe, rel, resolve_selection,
    run_logged,
)

PINNED = {"numpy": "2.4.4", "scipy": "1.17.1", "pandas": "3.0.2", "matplotlib": "3.10.8",
          "torch": "2.11.0", "mne": "1.12.1", "statsmodels": "0.14.6", "yaml": "6.0.3",
          "tqdm": "4.67.3"}
REQUIRED_FILES = [
    "scripts/preprocess_openneuro_cli.py", "scripts/run_aligned_ddpm_one_dataset.sh",
    "scripts/generate_nmm_only.py", "scripts/generate_diffusion_only.py", "scripts/make_residual.py",
    "scripts/make_add.py", "scripts/evaluate.py", "scripts/generate_gan_only.py",
    "src/common/metrics.py", "src/common/preprocess.py", "src/common/io.py", "src/common/granger.py",
    "reproducibility/openneuro_figure5_core.py",
    "source_data/OpenNeuro_all_errors_by_subject_seed_model_metric.csv",
    "source_data/Table5_OpenNeuro_absolute_error_long.csv",
]
# Disk use measured in a 1-participant x 1-seed run (rounded up), per participant-seed.
MB_PER_RUN_CORE3 = 20
MB_PER_RUN_WGAN = 10
MB_PER_SUBJECT = 5


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def check_environment(rep: Report, need_training: bool) -> None:
    v = sys.version_info
    if not ((3, 11) <= (v.major, v.minor) < (3, 14)):
        rep.error("Python version", platform.python_version(), "Use Python 3.12 (3.11-3.13 supported).")
    elif (v.major, v.minor) != (3, 12):
        rep.warn("Python version", f"{platform.python_version()} (reference: 3.12)")
    else:
        rep.ok("Python version", platform.python_version())
    if sys.prefix == sys.base_prefix:
        rep.warn("Virtual environment", "not active", "Activate venv/ so the pinned packages are used.")

    needed = ["numpy", "scipy", "pandas", "matplotlib"]
    if need_training:
        needed += ["torch", "mne", "statsmodels", "yaml", "tqdm"]
    for name in needed:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            rep.error(f"Package {name}", f"cannot be imported ({exc.__class__.__name__})",
                      "python -m pip install -r requirements-experiments.txt")
            continue
        version = getattr(module, "__version__", "?").split("+")[0]
        if version != PINNED[name]:
            rep.warn(f"Package {name}", f"{version} installed; paper environment used {PINNED[name]}")
        else:
            rep.ok(f"Package {name}", version)

    if need_training:
        if shutil.which("bash") is None:
            rep.error("bash", "not found", "The generation runner is a bash script (macOS/Linux; on Windows use WSL).")
        try:
            import torch
            if torch.cuda.is_available():
                rep.ok("Compute device", f"CUDA ({torch.cuda.get_device_name(0)})")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                rep.ok("Compute device", "Apple MPS")
            else:
                rep.warn("Compute device", "CPU only",
                         "Runs work on CPU but are slow; measure with --n-subjects 1 --n-seeds 1 first.")
        except Exception:
            pass


def check_repository(rep: Report) -> None:
    missing = [f for f in REQUIRED_FILES if not (REPO_ROOT / f).is_file()]
    for f in missing:
        rep.error("Repository file", f"missing: {f}", "Re-download or re-clone the repository.")
    if not missing:
        rep.ok("Repository files", f"{len(REQUIRED_FILES)} required files present")


def check_bids(rep: Report, bids_root: Path, subjects: list[str], deep: bool = True) -> list[str]:
    """Return the participants whose recordings are usable."""
    download_hint = "python pipeline/download_openneuro.py  (with the same participant options)"
    missing_files: list[str] = []
    if not bids_root.is_dir():
        rep.error("BIDS folder", f"not found: {bids_root}", download_hint)
        return []
    desc = bids_root / "dataset_description.json"
    if not desc.is_file():
        rep.error("dataset_description.json", "missing in the BIDS folder", download_hint)
    else:
        try:
            info = json.loads(desc.read_text(encoding="utf-8"))
            doi = str(info.get("DatasetDOI", ""))
            if doi and DATASET_ID not in doi:
                rep.warn("Dataset identity", f"DatasetDOI is {doi}", f"Expected OpenNeuro {DATASET_ID}.")
            else:
                rep.ok("Dataset", info.get("Name", DATASET_ID))
        except json.JSONDecodeError:
            rep.error("dataset_description.json", "not valid JSON")

    usable = []
    mne = None
    if deep:
        try:
            import mne  # noqa: F811
            mne.set_log_level("ERROR")
        except ImportError:
            rep.warn("EDF header check", "skipped (mne not installed)")
    for subject in subjects:
        path = bids_root / edf_relpath(subject)
        label = f"EEG {subject}"
        if path.is_symlink() and not path.exists():
            rep.error(label, "broken link (DataLad/git-annex placeholder without content)",
                      f"Run `datalad get {edf_relpath(subject)}` in the dataset, or use the download script.")
            continue
        if not path.is_file():
            alt = sorted((bids_root / subject).glob(f"**/*task-EyesClosed*acq-pre*_eeg.*")) \
                if (bids_root / subject).is_dir() else []
            detail = f"missing: {edf_relpath(subject)}"
            if alt:
                detail += f" (found instead: {', '.join(a.name for a in alt[:3])})"
            rep.error(label, detail)
            missing_files.append(subject)
            continue
        size = path.stat().st_size
        if size < 100_000:
            rep.error(label, f"file is only {size} bytes (incomplete download or annex pointer)",
                      "Delete it and download again.")
            continue
        if mne is not None:
            try:
                raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
                missing_ch = [c for c in CHANNELS if c not in raw.ch_names]
                dur = raw.n_times / raw.info["sfreq"]
                if missing_ch:
                    rep.error(label, f"channels {missing_ch} not in the file")
                    continue
                if dur < DURATION_SEC:
                    rep.error(label, f"recording is {dur:.1f} s; {DURATION_SEC} s are required")
                    continue
                rep.ok(label, f"{dur:.0f} s at {raw.info['sfreq']:g} Hz, F7/F8 present")
            except Exception as exc:
                rep.error(label, f"EDF cannot be read ({exc.__class__.__name__}: {exc})",
                          "The file may be corrupted; delete it and download again.")
                continue
        else:
            rep.ok(label, f"{size / 1e6:.1f} MB")
        usable.append(subject)
    if missing_files:
        rep.error("Missing recordings", f"{len(missing_files)} of {len(subjects)} participant(s)",
                  "python pipeline/download_openneuro.py --subjects " + " ".join(missing_files))
    return usable


def check_output(rep: Report, run_dir: Path, config: dict, n_runs: int, n_subjects: int, wgan: bool) -> None:
    cfg_path = run_dir / "run_config.json"
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        same = all(old.get(k) == config.get(k) for k in ("subjects", "seeds", "models", "protocol"))
        if same:
            rep.ok("Run folder", f"{rel(run_dir)} exists with the same settings (resume)")
        else:
            rep.error("Run folder", f"{rel(run_dir)} was created with different settings",
                      "Use a new --run-name; runs with different selections are never mixed.")
    need_mb = n_runs * (MB_PER_RUN_CORE3 + (MB_PER_RUN_WGAN if wgan else 0)) + n_subjects * MB_PER_SUBJECT
    probe = run_dir if run_dir.exists() else run_dir.parent
    while not probe.exists():
        probe = probe.parent
    free_mb = shutil.disk_usage(probe).free / 1e6
    detail = f"about {need_mb / 1000:.1f} GB needed, {free_mb / 1000:.1f} GB free"
    if free_mb < need_mb:
        rep.error("Disk space", detail, "Free space or choose --runs-dir on another disk.")
    elif free_mb < 1.5 * need_mb:
        rep.warn("Disk space", detail)
    else:
        rep.ok("Disk space", detail)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_preprocess(run_dir: Path, bids_root: Path, subjects: list[str]) -> None:
    for i, subject in enumerate(subjects, 1):
        out = run_dir / "preprocessed" / preprocessed_name(subject)
        log(f"[preprocess {i}/{len(subjects)}] {subject}")
        run_logged([python_exe(), "scripts/preprocess_openneuro_cli.py", "--bids-root", str(bids_root),
                    "--subject", subject, "--out", str(out), "--fs", str(FS),
                    "--duration-s", str(DURATION_SEC), "--channels", *CHANNELS],
                   run_dir / "logs" / "preprocess.log")


def stage_real_check(run_dir: Path, subjects: list[str]) -> Report:
    """Signal-level sanity check of every preprocessed recording before any training."""
    import numpy as np
    import pandas as pd
    rep = Report("Real-EEG check (preprocessed recordings)")
    rows = []
    for subject in subjects:
        path = run_dir / "preprocessed" / preprocessed_name(subject)
        frame = pd.read_csv(path)
        label = f"Real {subject}"
        if list(frame.columns) != ["time", *CHANNELS] or len(frame) != PREPROCESSED_ROWS:
            rep.error(label, f"unexpected shape {frame.shape} / columns {list(frame.columns)}")
            continue
        x = frame[list(CHANNELS)].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            rep.error(label, "non-finite samples")
            continue
        row = {"subject": subject}
        problems = []
        for k, ch in enumerate(CHANNELS):
            v = x[:, k]
            sd = float(v.std())
            run_lengths = np.diff(np.flatnonzero(np.diff(np.r_[np.inf, v, np.inf]) != 0))
            longest_flat_s = float(run_lengths.max()) / FS if len(run_lengths) else 0.0
            at_extreme = float(np.mean((v == v.max()) | (v == v.min())))
            row.update({f"{ch}_sd_uV": sd, f"{ch}_longest_flat_s": longest_flat_s,
                        f"{ch}_fraction_at_min_or_max": at_extreme})
            if sd <= 1e-9:
                rep.error(label, f"{ch} is flat (SD = 0)")
                problems.append("flat")
            elif not 1.0 <= sd <= 200.0:
                problems.append(f"{ch} SD {sd:.1f} uV outside 1-200 uV")
            if longest_flat_s >= 1.0:
                problems.append(f"{ch} constant for {longest_flat_s:.1f} s")
            if at_extreme > 0.01:
                problems.append(f"{ch} {100 * at_extreme:.1f}% of samples at min/max (clipping?)")
        rows.append(row)
        if problems and "flat" not in problems:
            rep.warn(label, "; ".join(problems), "Inspect the recording; the run continues.")
        elif not problems:
            rep.ok(label, ", ".join(f"{ch} SD {row[f'{ch}_sd_uV']:.1f} uV" for ch in CHANNELS))
    pd.DataFrame(rows).to_csv(run_dir / "reports" / "real_data_check.csv", index=False)
    return rep


def core3_done(run_dir: Path, subject: str, seeds: list[int]) -> bool:
    root = run_dir / "subjects" / subject
    return all((root / f"seed{s}" / ".complete").is_file() for s in seeds)


def stage_core3(run_dir: Path, subjects: list[str], seeds: list[int], keep_going: bool) -> list[str]:
    failed = []
    for i, subject in enumerate(subjects, 1):
        if core3_done(run_dir, subject, seeds):
            log(f"[core3 {i}/{len(subjects)}] {subject}: already complete")
            continue
        log(f"[core3 {i}/{len(subjects)}] {subject}: NMM-only, Diffusion-only, Add x {len(seeds)} seed(s)")
        env = {
            "ROOT_DIR": str(REPO_ROOT), "PYTHON_BIN": python_exe(),
            "REAL_CSV": str(run_dir / "preprocessed" / preprocessed_name(subject)),
            "OUTPUT_ROOT": str(run_dir / "subjects" / subject),
            "DATASET_LABEL": f"openneuro_{DATASET_ID}", "SUBJECT_ID": subject,
            "SEEDS": " ".join(map(str, seeds)), "RUN_E1_E2": "0", "MAKE_FIGURES": "0",
            "SKIP_COMPLETED": "1",
        }
        start = time.time()
        try:
            run_logged(["bash", "scripts/run_aligned_ddpm_one_dataset.sh"],
                       run_dir / "logs" / f"core3_{subject}.log", env=env)
        except StepFailed as exc:
            failed.append(subject)
            log(f"[ERROR] core3 {subject}: {exc}")
            if not keep_going:
                raise
            continue
        log(f"[core3 {i}/{len(subjects)}] {subject}: done in {(time.time() - start) / 60:.1f} min")
    return failed


def valid_generated_csv(path: Path) -> bool:
    import numpy as np
    import pandas as pd
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return (len(frame) == PREPROCESSED_ROWS and all(c in frame for c in CHANNELS)
            and np.isfinite(frame[list(CHANNELS)].to_numpy(dtype=float)).all())


def valid_metrics(path: Path) -> bool:
    from pipeline.openneuro_results import read_metric_means
    try:
        read_metric_means(path)
        return True
    except Exception:
        return False


def stage_wgan(run_dir: Path, subjects: list[str], seeds: list[int], keep_going: bool) -> list[str]:
    failed = []
    for i, subject in enumerate(subjects, 1):
        out_dir = run_dir / "wgan" / subject
        todo = [s for s in seeds if not valid_generated_csv(out_dir / f"gan_3min_gan_seed{s}.csv")]
        log_path = run_dir / "logs" / f"wgan_{subject}.log"
        try:
            if todo:
                log(f"[wgan {i}/{len(subjects)}] {subject}: training seed(s) {todo}")
                start = time.time()
                run_logged([python_exe(), "scripts/generate_gan_only.py",
                            "--input", str(run_dir / "preprocessed" / preprocessed_name(subject)),
                            "--out_dir", str(out_dir), "--seeds", " ".join(map(str, todo)), *WGAN_ARGS],
                           log_path)
                bad = [s for s in todo if not valid_generated_csv(out_dir / f"gan_3min_gan_seed{s}.csv")]
                if bad:
                    raise StepFailed(f"WGAN-GP output missing or invalid for seed(s) {bad}; see {log_path}")
                log(f"[wgan {i}/{len(subjects)}] {subject}: trained in {(time.time() - start) / 60:.1f} min")
            else:
                log(f"[wgan {i}/{len(subjects)}] {subject}: generated signals already present")
            for s in seeds:
                eval_dir = run_dir / "subjects" / subject / f"seed{s}" / eval_dirname("WGAN-GP", s)
                if valid_metrics(eval_dir / "metrics_summary.csv"):
                    continue
                run_logged([python_exe(), "scripts/evaluate.py",
                            "--input", str(out_dir / f"gan_3min_gan_seed{s}.csv"), *EVALUATE_ARGS,
                            "--outdir", str(eval_dir)], log_path)
                if not valid_metrics(eval_dir / "metrics_summary.csv"):
                    raise StepFailed(f"evaluation of WGAN-GP seed {s} produced no valid metrics; see {log_path}")
        except StepFailed as exc:
            failed.append(subject)
            log(f"[ERROR] wgan {subject}: {exc}")
            if not keep_going:
                raise
    return failed


def stage_results(run_dir: Path, subjects: list[str], seeds: list[int], models: list[str],
                  allow_incomplete: bool) -> dict:
    from pipeline.openneuro_results import build_results, collect_run
    frame, problems = collect_run(run_dir, subjects, seeds, models, allow_incomplete)
    if problems:
        log(f"[WARN] {len(problems)} missing evaluation(s); results use the complete participants only")
        complete = frame.groupby("subject")["seed"].nunique()
        complete = complete[complete == len(seeds)].index
        cells = frame[frame["subject"].isin(complete)].groupby("subject")["model"].nunique()
        subjects = [s for s in subjects if s in cells[cells == len(models)].index]
        frame = frame[frame["subject"].isin(subjects)]
        if not subjects:
            raise SystemExit("[ERROR] No participant has complete results yet.")
    summary = build_results(frame, subjects, seeds, models, run_dir / "results", compare=True)
    print_summary(summary, run_dir / "results")
    return summary


def print_summary(summary: dict, out_dir: Path) -> None:
    p, s = summary["participant_level"], summary["seed_level"]
    print("\n=== Results ===")
    print(f"Participants where the closest generator differs across measures: "
          f"{p['closest_generator_differs']}/{p['units']}")
    print(f"Participant-seed runs where it differs: {s['closest_generator_differs']}/{s['units']}")
    print(f"Participant x measure winner cells: {summary['winner_cells_by_generator']}")
    cmp = summary.get("comparison_with_paper")
    if cmp:
        print(f"Comparison with the paper ({cmp['matched_rows']} matched rows): "
              f"Real-EEG metrics max |diff| = {cmp['real_eeg_metrics_max_abs_diff']:.2e}; "
              f"winner agreement {cmp['participant_metric_winner_agreement']}")
        for model, info in cmp["per_model"].items():
            print(f"    {model:15s} generated-metric max |diff| = {info['generated_metric_max_abs_diff']:.2e}")
    print(f"Outputs: {rel(out_dir)}  (Table5_absolute_error_95CI.md, figures/Fig5a_*, Fig5b_*, summary.json)")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def make_config(subjects, seeds, models, bids_root) -> dict:
    return {"created_at": now(), "subjects": subjects, "seeds": seeds, "models": models,
            "bids_root": rel(bids_root), "dataset_version": DATASET_VERSION, "protocol": PROTOCOL,
            "python": platform.python_version(), "platform": platform.platform()}


def run_dir_for(args, subjects, seeds) -> Path:
    name = args.run_name or f"openneuro_{len(subjects)}subj_{len(seeds)}seeds"
    return (args.runs_dir / name).resolve()


def cmd_check(args, training: bool = True) -> tuple[Report, list[str]]:
    subjects, seeds = resolve_selection(args)
    rep = Report(f"Pre-flight check ({len(subjects)} participant(s) x {len(seeds)} seed(s))")
    check_environment(rep, need_training=training)
    check_repository(rep)
    usable = check_bids(rep, args.bids_root.resolve(), subjects)
    models = CORE_MODELS if args.skip_wgan else ALL_MODELS
    run_dir = run_dir_for(args, subjects, seeds)
    check_output(rep, run_dir, make_config(subjects, seeds, models, args.bids_root),
                 len(subjects) * len(seeds), len(subjects), not args.skip_wgan)
    rep.print(verbose_ok=args.verbose)
    return rep, usable


def cmd_run(args) -> None:
    subjects, seeds = resolve_selection(args)
    models = CORE_MODELS if args.skip_wgan else ALL_MODELS
    run_dir = run_dir_for(args, subjects, seeds)
    n_train = len(subjects) * len(seeds) * (3 if not args.skip_wgan else 2)
    print(f"Plan: {len(subjects)} participant(s) x {len(seeds)} seed(s); generators: {', '.join(models)}")
    print(f"      {n_train} neural-network trainings (2 DDPM{' + 1 WGAN-GP' if not args.skip_wgan else ''} "
          f"per participant-seed). Output: {rel(run_dir)}")
    if len(subjects) * len(seeds) > 4 and not args.yes:
        answer = input("This is a long run (days for the full 30 x 20). Continue? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Stopped. Tip: start with --n-subjects 1 --n-seeds 1 to measure the run time.")

    rep, usable = cmd_check(args)
    if rep.n_errors:
        raise SystemExit("[STOP] Fix the errors above, then run the same command again.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    rep.save(run_dir / "reports" / "preflight.json")
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.is_file():
        cfg_path.write_text(json.dumps(make_config(subjects, seeds, models, args.bids_root), indent=2),
                            encoding="utf-8")

    t0 = time.time()
    stage_preprocess(run_dir, args.bids_root.resolve(), subjects)
    real = stage_real_check(run_dir, subjects)
    real.print()
    real.save(run_dir / "reports" / "real_data_check.json")
    if real.n_errors:
        raise SystemExit("[STOP] Real-EEG check failed; see reports/real_data_check.json")
    if args.stop_after == "preprocess":
        log("Stopped after preprocessing (--stop-after preprocess).")
        return

    failed = stage_core3(run_dir, subjects, seeds, args.keep_going)
    if args.stop_after == "core3":
        log("Stopped after NMM-only / Diffusion-only / Add (--stop-after core3).")
        return
    if not args.skip_wgan:
        ok_subjects = [s for s in subjects if s not in failed]
        failed += stage_wgan(run_dir, ok_subjects, seeds, args.keep_going)

    stage_results(run_dir, subjects, seeds, models, allow_incomplete=bool(failed))
    log(f"Finished in {(time.time() - t0) / 3600:.2f} h. Failed participants: {failed or 'none'}")


def cmd_results(args) -> None:
    subjects, seeds = resolve_selection(args)
    models = CORE_MODELS if args.skip_wgan else ALL_MODELS
    run_dir = run_dir_for(args, subjects, seeds)
    cfg = run_dir / "run_config.json"
    if cfg.is_file():
        stored = json.loads(cfg.read_text(encoding="utf-8"))
        subjects, seeds, models = stored["subjects"], stored["seeds"], stored["models"]
    elif not run_dir.is_dir():
        raise SystemExit(f"[ERROR] Run folder not found: {rel(run_dir)}")
    stage_results(run_dir, subjects, seeds, models, allow_incomplete=args.allow_incomplete)


def cmd_selftest(args) -> None:
    from pipeline.openneuro_results import build_results, load_archived, verify_against_published
    subjects, seeds = resolve_selection(args)
    models = CORE_MODELS if args.skip_wgan else ALL_MODELS
    full = subjects == PAPER_SUBJECTS and seeds == PAPER_SEEDS and models == ALL_MODELS
    out_dir = (args.runs_dir / "selftest_from_archived_data").resolve()
    log(f"Rebuilding results from the archived paper data ({len(subjects)} participants x {len(seeds)} seeds)")
    frame = load_archived(subjects, seeds, models)
    summary = build_results(frame, subjects, seeds, models, out_dir, compare=False)
    print_summary(summary, out_dir)
    if not full:
        print("[INFO] Subset selected: published numbers are only checked for the full 30 x 20 selection.")
        return
    failures = verify_against_published(out_dir)
    if failures:
        print("[FAIL] " + "\n[FAIL] ".join(failures))
        raise SystemExit(1)
    print("[PASS] Table 5 (mean, SD, median, 95% CI) and Figure 5 counts (28/30, 526/600, "
          "11/31/18/90) reproduce the paper.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("selftest", "rebuild Table 5 / Fig. 5 from archived data and verify"),
                            ("check", "pre-flight checks only"),
                            ("run", "full pipeline (resumable)"),
                            ("results", "rebuild tables/figures of an existing run")]:
        p = sub.add_parser(name, help=help_text)
        add_selection_args(p)
        p.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT,
                       help=f"BIDS folder of {DATASET_ID} (default: data/openneuro/{DATASET_ID}).")
        p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Parent folder for runs (default: runs/).")
        p.add_argument("--run-name", help="Run folder name (default: openneuro_<N>subj_<M>seeds).")
        p.add_argument("--skip-wgan", action="store_true", help="Run only NMM-only, Diffusion-only and Add.")
        p.add_argument("--verbose", action="store_true", help="Also list passed checks.")
        if name == "run":
            p.add_argument("--yes", action="store_true", help="Do not ask for confirmation of long runs.")
            p.add_argument("--keep-going", action="store_true",
                           help="Continue with other participants after a failure.")
            p.add_argument("--stop-after", choices=["preprocess", "core3"],
                           help="Stop after this stage (useful for staged runs).")
        if name == "results":
            p.add_argument("--allow-incomplete", action="store_true",
                           help="Summarise the participants that are complete so far.")
    args = parser.parse_args()
    args.runs_dir = args.runs_dir.expanduser()
    args.bids_root = args.bids_root.expanduser()
    try:
        if args.command == "check":
            rep, _ = cmd_check(args)
            sys.exit(1 if rep.n_errors else 0)
        {"run": cmd_run, "results": cmd_results, "selftest": cmd_selftest}[args.command](args)
    except StepFailed as exc:
        raise SystemExit(f"[STOP] {exc}\nRe-run the same command after fixing the problem; "
                         "completed work is kept and skipped.")
    except KeyboardInterrupt:
        raise SystemExit("\n[STOP] Interrupted. Re-run the same command to resume.")


if __name__ == "__main__":
    main()
