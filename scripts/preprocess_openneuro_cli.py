#!/usr/bin/env python3
"""Convert one ds005385 EDF file to the pipeline's time,F7,F8 CSV format."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve and convert one OpenNeuro ds005385 EyesClosed EDF to "
            "exactly 180 s at 256 Hz with F7/F8 channels."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--edf", help="Explicit EDF file path")
    source.add_argument("--bids-root", help="Root of ds005385")
    parser.add_argument("--subject", required=True, help="BIDS subject label, e.g. sub-012")
    parser.add_argument("--session", default="ses-1")
    parser.add_argument("--task", default="EyesClosed")
    parser.add_argument("--acquisition", default="pre")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--channels", nargs=2, default=["F7", "F8"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV. Without this flag, an existing valid CSV is reused.",
    )
    return parser.parse_args()


def resolve_edf(args: argparse.Namespace) -> Path:
    if args.edf:
        path = Path(args.edf).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EDF not found: {path}")
        return path

    root = Path(args.bids_root).expanduser().resolve()
    expected = (
        root
        / args.subject
        / args.session
        / "eeg"
        / (
            f"{args.subject}_{args.session}_task-{args.task}_"
            f"acq-{args.acquisition}_eeg.edf"
        )
    )
    if expected.is_file():
        return expected

    pattern = (
        f"{args.subject}_*_task-{args.task}_"
        f"acq-{args.acquisition}_eeg.edf"
    )
    matches = sorted((root / args.subject).glob(f"*/eeg/{pattern}"))
    if not matches:
        raise FileNotFoundError(
            f"Expected EDF not found: {expected}. No fallback match for {pattern}."
        )
    if len(matches) != 1:
        joined = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            "EDF selection is ambiguous. Specify --edf explicitly. Matches:\n  " + joined
        )
    if matches[0].parent.parent.name != args.session:
        raise RuntimeError(
            f"Only match is in {matches[0].parent.parent.name}, not requested "
            f"{args.session}: {matches[0]}"
        )
    return matches[0].resolve()


def valid_existing_csv(path: Path, channels: list[str], expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return len(frame) == expected_rows and all(channel in frame for channel in channels)


def main() -> None:
    args = parse_args()
    if not args.subject.startswith("sub-"):
        raise ValueError(f"--subject must use BIDS form sub-XXX: {args.subject}")
    if args.fs <= 0 or args.duration_s <= 0:
        raise ValueError("--fs and --duration-s must be positive")

    expected_rows = int(round(args.fs * args.duration_s))
    out_path = Path(args.out).expanduser().resolve()
    meta_path = Path(str(out_path) + ".meta.json")
    if not args.overwrite and valid_existing_csv(out_path, args.channels, expected_rows):
        print(f"[SKIP] valid preprocessed CSV exists: {out_path}")
        return

    edf_path = resolve_edf(args)

    try:
        import mne
    except ImportError as exc:
        raise RuntimeError(
            "MNE is required for EDF preprocessing. Install it in the execution environment."
        ) from exc

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    original_fs = float(raw.info["sfreq"])
    available_channels = list(raw.ch_names)
    missing = [channel for channel in args.channels if channel not in available_channels]
    if missing:
        raise ValueError(
            f"Missing required channels {missing} in {edf_path}. "
            f"Available channels: {available_channels}"
        )

    raw.pick(args.channels)
    if abs(float(raw.info["sfreq"]) - args.fs) > 1e-9:
        raw.resample(args.fs)

    data = raw.get_data() * 1e6  # V -> microvolt
    if data.shape[1] < expected_rows:
        raise ValueError(
            f"Recording is too short after resampling: {data.shape[1]} samples; "
            f"need {expected_rows} ({args.duration_s}s at {args.fs}Hz)"
        )
    data = data[:, :expected_rows]
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite samples found in {edf_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "time": np.arange(expected_rows, dtype=float) / args.fs,
            args.channels[0]: data[0],
            args.channels[1]: data[1],
        }
    )
    frame.to_csv(out_path, index=False)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_edf": str(edf_path),
        "output_csv": str(out_path),
        "subject": args.subject,
        "session": args.session,
        "task": args.task,
        "acquisition": args.acquisition,
        "channels": args.channels,
        "original_fs": original_fs,
        "target_fs": args.fs,
        "duration_sec": args.duration_s,
        "samples": expected_rows,
        "amplitude_unit": "microvolt",
        "operations": ["channel selection", "resampling", "first-180-s crop", "V-to-uV"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] OpenNeuro CSV: {out_path} ({expected_rows} rows)")
    print(f"[OK] provenance: {meta_path}")


if __name__ == "__main__":
    main()
