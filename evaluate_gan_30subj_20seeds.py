#!/usr/bin/env python3
"""
Evaluate all WGAN-GP EEG CSV files with the existing scripts/evaluate.py.

Expected input layout
---------------------
runs/gan_only/sub-XXX/gan_3min_gan_seed0.csv
...
runs/gan_only/sub-XXX/gan_3min_gan_seed19.csv

Expected existing paper-run layout
----------------------------------
runs/openneuro_sub-XXX_YYYYMMDD_HHMMSS_multiseed20_paper/seed0/
...
runs/openneuro_sub-XXX_YYYYMMDD_HHMMSS_multiseed20_paper/seed19/

Output for each subject/seed
----------------------------
runs/openneuro_sub-XXX_YYYYMMDD_HHMMSS_multiseed20_paper/seedN/
    gan_start20_dur139.99609375/
        metrics_summary.csv
        metrics_windows.csv
        meta.json
        plots/

The default protocol matches the existing Real, NMM-only, Diffusion-only,
and Add evaluation:
    bandpass 1-45 Hz -> global z-score -> 10 s windows / 5 s hop
    evaluation interval: start=20 s, duration=139.99609375 s
    DFA scales: 0.2-1.25 s

PSD slope is calculated by evaluate.py only when --plots is specified, so
this runner always enables --plots.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable


DEFAULT_DURATION_SEC = 139.99609375
DEFAULT_OUTPUT_DIRNAME = "gan_start20_dur139.99609375"

REQUIRED_METRICS = {
    "psd_loglog_slope_all_2_40hz",
    "dfa_x_alpha",
    "acf_x_abs_0_1s",
    "acf_x_abs_0_5s",
    "acf_x_abs_1_0s",
}

INPUT_RE = re.compile(r"^gan_3min_gan_seed(?P<seed>\d+)\.csv$")
SUBJECT_RE = re.compile(r"^sub-\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 30 subjects x 20 WGAN-GP seeds with the existing "
            "EEG evaluation protocol."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--gan-root",
        type=Path,
        default=Path("runs/gan_only"),
        help="GAN result root, relative to repo-root unless absolute.",
    )
    parser.add_argument(
        "--paper-runs-template",
        default="auto",
        help=(
            "Existing per-subject paper-run directory template. Default: auto, "
            "which discovers timestamped directories made by "
            "run_openneuro_20seeds.sh. For a manual template, use {subject} "
            "for names such as sub-012."
        ),
    )
    parser.add_argument(
        "--evaluate-script",
        type=Path,
        default=Path("scripts/evaluate.py"),
        help="Path to the existing evaluate.py, relative to repo-root unless absolute.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke evaluate.py. Default: current Python.",
    )
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--ch1", default="F7")
    parser.add_argument("--ch2", default="F8")
    parser.add_argument("--win-s", type=float, default=10.0)
    parser.add_argument("--hop-s", type=float, default=5.0)
    parser.add_argument("--start-sec", type=float, default=20.0)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--dfa-min-s", type=float, default=0.2)
    parser.add_argument("--dfa-max-s", type=float, default=1.25)
    parser.add_argument("--bp-low", type=float, default=1.0)
    parser.add_argument("--bp-high", type=float, default=45.0)
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_OUTPUT_DIRNAME,
        help="Output subdirectory created under every paper-run seed directory.",
    )
    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=30,
        help="Required number of subjects. Default: 30.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=20,
        help="Required seeds per subject, numbered 0..N-1. Default: 20.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subset such as sub-012 sub-407. Disables the 30-subject count check.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even when a valid output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate discovery and print commands without running evaluate.py.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed run. By default, stop at the first failure.",
    )
    return parser.parse_args()


def resolve_under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def find_evaluate_script(repo_root: Path, requested: Path) -> Path:
    candidate = resolve_under(repo_root, requested)
    if candidate.is_file():
        return candidate

    if requested == Path("scripts/evaluate.py"):
        fallback = (repo_root / "evaluate.py").resolve()
        if fallback.is_file():
            return fallback

    raise FileNotFoundError(
        f"evaluate.py was not found: {candidate}\n"
        "Place this runner at the repository root or pass --evaluate-script."
    )


def discover_inputs(
    gan_root: Path,
    expected_subjects: int,
    expected_seeds: int,
    selected_subjects: list[str] | None,
) -> list[tuple[str, int, Path]]:
    if not gan_root.is_dir():
        raise FileNotFoundError(f"GAN root was not found: {gan_root}")

    if selected_subjects:
        invalid = [s for s in selected_subjects if not SUBJECT_RE.fullmatch(s)]
        if invalid:
            raise ValueError(f"Invalid subject name(s): {', '.join(invalid)}")
        subject_names = sorted(set(selected_subjects))
    else:
        subject_names = sorted(
            p.name
            for p in gan_root.iterdir()
            if p.is_dir() and SUBJECT_RE.fullmatch(p.name)
        )
        if len(subject_names) != expected_subjects:
            raise RuntimeError(
                f"Expected {expected_subjects} subject directories under {gan_root}, "
                f"but found {len(subject_names)}: {', '.join(subject_names)}"
            )

    expected_seed_set = set(range(expected_seeds))
    discovered: list[tuple[str, int, Path]] = []

    for subject in subject_names:
        subject_dir = gan_root / subject
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Subject directory was not found: {subject_dir}")

        seed_to_file: dict[int, Path] = {}
        for path in subject_dir.glob("gan_3min_gan_seed*.csv"):
            match = INPUT_RE.fullmatch(path.name)
            if not match:
                continue
            seed = int(match.group("seed"))
            if seed in seed_to_file:
                raise RuntimeError(
                    f"Duplicate GAN input for {subject}/seed{seed}: "
                    f"{seed_to_file[seed]} and {path}"
                )
            seed_to_file[seed] = path.resolve()

        actual_seed_set = set(seed_to_file)
        missing = sorted(expected_seed_set - actual_seed_set)
        extra = sorted(actual_seed_set - expected_seed_set)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing seeds={missing}")
            if extra:
                details.append(f"unexpected seeds={extra}")
            raise RuntimeError(f"{subject}: " + "; ".join(details))

        for seed in sorted(expected_seed_set):
            discovered.append((subject, seed, seed_to_file[seed]))

    expected_total = len(subject_names) * expected_seeds
    if len(discovered) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} GAN CSV files, but found {len(discovered)}."
        )
    return discovered


def read_metric_names(summary_path: Path) -> set[str]:
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return {row[0] for row in reader if row}


def output_is_valid(outdir: Path, input_csv: Path) -> tuple[bool, str]:
    summary_path = outdir / "metrics_summary.csv"
    windows_path = outdir / "metrics_windows.csv"
    meta_path = outdir / "meta.json"

    for path in (summary_path, windows_path, meta_path):
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"missing or empty {path.name}"

    try:
        metric_names = read_metric_names(summary_path)
    except Exception as exc:
        return False, f"cannot read metrics_summary.csv: {exc}"

    missing_metrics = sorted(REQUIRED_METRICS - metric_names)
    if missing_metrics:
        return False, "missing metrics: " + ", ".join(missing_metrics)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"cannot read meta.json: {exc}"

    meta_input = meta.get("input")
    if not meta_input:
        return False, "meta.json has no input path"
    try:
        if Path(meta_input).resolve() != input_csv.resolve():
            return False, f"meta input does not match: {meta_input}"
    except Exception as exc:
        return False, f"cannot validate meta input: {exc}"

    return True, "complete"


def seed_dir_count(path: Path, expected_seeds: int) -> int:
    return sum((path / f"seed{seed}").is_dir() for seed in range(expected_seeds))


def timestamp_from_paper_run_name(name: str, subject: str) -> str:
    match = re.fullmatch(
        rf"openneuro_{re.escape(subject)}_(\d{{8}}_\d{{6}})_multiseed20_paper",
        name,
    )
    return match.group(1) if match else ""


def paper_run_dir(
    repo_root: Path,
    template: str,
    subject: str,
    expected_seeds: int,
) -> tuple[Path, str | None]:
    if template == "auto":
        runs_root = repo_root / "runs"
        candidates: list[Path] = []

        legacy = runs_root / f"openneuro_{subject}_multiseed20_paper"
        if legacy.is_dir():
            candidates.append(legacy)

        candidates.extend(
            path
            for path in runs_root.glob(
                f"openneuro_{subject}_*_multiseed20_paper"
            )
            if path.is_dir()
        )
        candidates = sorted(set(candidates))

        complete = [
            path
            for path in candidates
            if seed_dir_count(path, expected_seeds) == expected_seeds
        ]
        if not complete:
            if candidates:
                details = ", ".join(
                    f"{path.name} ({seed_dir_count(path, expected_seeds)}/"
                    f"{expected_seeds} seed dirs)"
                    for path in candidates
                )
                raise FileNotFoundError(
                    f"No complete paper-run was found for {subject}. "
                    f"Candidates: {details}"
                )
            raise FileNotFoundError(
                f"No paper-run directory was found for {subject}. Expected a "
                f"directory like runs/openneuro_{subject}_"
                "YYYYMMDD_HHMMSS_multiseed20_paper."
            )

        complete.sort(
            key=lambda path: (
                timestamp_from_paper_run_name(path.name, subject),
                path.stat().st_mtime_ns,
                path.name,
            )
        )
        chosen = complete[-1]
        warning = None
        if len(complete) > 1:
            warning = (
                f"{subject}: found {len(complete)} complete paper-runs; "
                f"using latest {chosen.name}"
            )
        return chosen.resolve(), warning

    try:
        rendered = template.format(subject=subject)
    except KeyError as exc:
        raise ValueError(
            "--paper-runs-template must use only the {subject} placeholder."
        ) from exc
    return resolve_under(repo_root, Path(rendered)), None


def build_command(
    python_executable: str,
    evaluate_script: Path,
    input_csv: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python_executable,
        str(evaluate_script),
        "--input",
        str(input_csv),
        "--ch1",
        args.ch1,
        "--ch2",
        args.ch2,
        "--fs",
        f"{args.fs:g}",
        "--win_s",
        f"{args.win_s:g}",
        "--hop_s",
        f"{args.hop_s:g}",
        "--start_sec",
        f"{args.start_sec:g}",
        "--duration_sec",
        str(args.duration_sec),
        "--bp_low",
        f"{args.bp_low:g}",
        "--bp_high",
        f"{args.bp_high:g}",
        "--zscore",
        "global",
        "--dfa_min_s",
        f"{args.dfa_min_s:g}",
        "--dfa_max_s",
        f"{args.dfa_max_s:g}",
        "--outdir",
        str(outdir),
        "--plots",
        "--no_gc",
    ]


def write_status_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "subject",
        "seed",
        "input_csv",
        "output_dir",
        "status",
        "seconds",
        "message",
    ]
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    gan_root = resolve_under(repo_root, args.gan_root)
    evaluate_script = find_evaluate_script(repo_root, args.evaluate_script)

    inputs = discover_inputs(
        gan_root=gan_root,
        expected_subjects=args.expected_subjects,
        expected_seeds=args.expected_seeds,
        selected_subjects=args.subjects,
    )

    targets: list[tuple[str, int, Path, Path]] = []
    missing_paper_dirs: list[Path] = []
    paper_run_roots: dict[str, Path] = {}
    discovery_warnings: list[str] = []
    for subject, seed, input_csv in inputs:
        if subject not in paper_run_roots:
            subject_run, warning = paper_run_dir(
                repo_root,
                args.paper_runs_template,
                subject,
                args.expected_seeds,
            )
            paper_run_roots[subject] = subject_run
            if warning:
                discovery_warnings.append(warning)
        else:
            subject_run = paper_run_roots[subject]
        seed_dir = subject_run / f"seed{seed}"
        if not seed_dir.is_dir():
            missing_paper_dirs.append(seed_dir)
        targets.append(
            (subject, seed, input_csv, seed_dir / args.output_dirname)
        )

    if missing_paper_dirs:
        preview = "\n".join(f"  {p}" for p in missing_paper_dirs[:10])
        suffix = (
            f"\n  ... and {len(missing_paper_dirs) - 10} more"
            if len(missing_paper_dirs) > 10
            else ""
        )
        raise FileNotFoundError(
            "Existing paper-run seed directories were not found:\n"
            f"{preview}{suffix}\n"
            "Check --paper-runs-template before evaluation."
        )

    expected_total = len(targets)
    print(f"[CHECK] GAN inputs      : {expected_total}")
    print(f"[CHECK] subjects       : {len({x[0] for x in targets})}")
    print(f"[CHECK] seeds/subject  : {args.expected_seeds}")
    print(f"[CHECK] evaluate.py    : {evaluate_script}")
    print(f"[CHECK] paper-run mode : {args.paper_runs_template}")
    print(f"[CHECK] paper-run roots: {len(paper_run_roots)}")
    print(f"[CHECK] output dirname : {args.output_dirname}")
    print(
        "[CHECK] protocol       : "
        f"fs={args.fs:g}, {args.ch1}/{args.ch2}, "
        f"start={args.start_sec:g}s, duration={args.duration_sec}s, "
        f"window={args.win_s:g}s, hop={args.hop_s:g}s, "
        "zscore=global, plots=yes, GC=no"
    )
    for warning in discovery_warnings:
        print(f"[WARNING] {warning}")

    env = os.environ.copy()
    src_path = str((repo_root / "src").resolve())
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not old_pythonpath else src_path + os.pathsep + old_pythonpath
    )

    status_path = gan_root / "gan_evaluation_status.csv"
    log_path = gan_root / (
        "gan_evaluation_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".log"
    )
    status_rows: list[dict[str, object]] = []
    completed = 0
    skipped = 0
    failed = 0

    log_handle = None
    if not args.dry_run:
        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write(f"Started: {datetime.now().isoformat()}\n")
        log_handle.write(f"Repository: {repo_root}\n")
        log_handle.write(f"Inputs: {expected_total}\n\n")
        log_handle.flush()

    try:
        for index, (subject, seed, input_csv, outdir) in enumerate(targets, start=1):
            valid, reason = output_is_valid(outdir, input_csv)
            if valid and not args.force:
                skipped += 1
                row = {
                    "subject": subject,
                    "seed": seed,
                    "input_csv": str(input_csv),
                    "output_dir": str(outdir),
                    "status": "skipped_complete",
                    "seconds": "0.000",
                    "message": reason,
                }
                status_rows.append(row)
                if not args.dry_run:
                    write_status_csv(status_path, status_rows)
                print(
                    f"[{index:03d}/{expected_total}] "
                    f"{subject}/seed{seed}: SKIP (complete)"
                )
                continue

            cmd = build_command(
                python_executable=args.python,
                evaluate_script=evaluate_script,
                input_csv=input_csv,
                outdir=outdir,
                args=args,
            )

            if args.dry_run:
                print(
                    f"[{index:03d}/{expected_total}] "
                    + subprocess.list2cmdline(cmd)
                )
                continue

            print(
                f"[{index:03d}/{expected_total}] "
                f"{subject}/seed{seed}: RUN"
            )
            assert log_handle is not None
            log_handle.write(
                f"\n=== [{index:03d}/{expected_total}] "
                f"{subject}/seed{seed} ===\n"
            )
            log_handle.write(subprocess.list2cmdline(cmd) + "\n")
            log_handle.flush()

            start_time = time.perf_counter()
            result = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - start_time

            if result.returncode == 0:
                valid_after, reason_after = output_is_valid(outdir, input_csv)
            else:
                valid_after = False
                reason_after = f"evaluate.py exit code {result.returncode}"

            if valid_after:
                completed += 1
                status = "completed"
                message = reason_after
                print(
                    f"[{index:03d}/{expected_total}] "
                    f"{subject}/seed{seed}: OK ({elapsed:.1f}s)"
                )
            else:
                failed += 1
                status = "failed"
                message = reason_after
                print(
                    f"[{index:03d}/{expected_total}] "
                    f"{subject}/seed{seed}: FAILED - {reason_after}",
                    file=sys.stderr,
                )

            status_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "input_csv": str(input_csv),
                    "output_dir": str(outdir),
                    "status": status,
                    "seconds": f"{elapsed:.3f}",
                    "message": message,
                }
            )
            write_status_csv(status_path, status_rows)

            if not valid_after and not args.keep_going:
                break
    finally:
        if log_handle is not None:
            log_handle.write(
                "\nFinished: "
                + datetime.now().isoformat()
                + f"\ncompleted={completed}, skipped={skipped}, failed={failed}\n"
            )
            log_handle.close()

    if args.dry_run:
        print(f"[DRY-RUN] validated {expected_total} input/output mappings.")
        return 0

    print("\n=== GAN EVALUATION SUMMARY ===")
    print(f"Newly completed : {completed}")
    print(f"Already complete: {skipped}")
    print(f"Failed          : {failed}")
    print(f"Status CSV      : {status_path}")
    print(f"Log             : {log_path}")

    processed = completed + skipped
    if failed == 0 and processed == expected_total:
        print(f"[DONE] All {expected_total} GAN runs are ready for aggregation.")
        return 0

    print(
        f"[INCOMPLETE] Ready={processed}/{expected_total}, failed={failed}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
