#!/usr/bin/env python3
"""Download the OpenNeuro ds005385 (v1.0.2) recordings needed by the pipeline.

Only the eyes-closed, pre-task, session-1 EEG file of each selected participant
is downloaded, plus the dataset's top-level BIDS metadata, into

    data/openneuro/ds005385/
        dataset_description.json, participants.tsv, README, CHANGES, ...
        sub-012/ses-1/eeg/sub-012_ses-1_task-EyesClosed_acq-pre_eeg.edf
        ...

Examples
--------
    python pipeline/download_openneuro.py                  # all 30 paper participants
    python pipeline/download_openneuro.py --n-subjects 1   # sub-012 only
    python pipeline/download_openneuro.py --subjects sub-012 sub-407
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import (  # noqa: E402
    DATASET_ID, DATASET_VERSION, DEFAULT_BIDS_ROOT, SESSION, TASK, ACQUISITION,
    add_selection_args, edf_relpath, resolve_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(parser)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT,
                        help=f"Download target (default: data/openneuro/{DATASET_ID}).")
    args = parser.parse_args()
    subjects, _ = resolve_selection(args)

    try:
        import openneuro
    except ImportError:
        raise SystemExit("[ERROR] openneuro-py is not installed. Run: "
                         "python -m pip install -r requirements-experiments.txt")

    target = args.bids_root.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    pending = [s for s in subjects if not (target / edf_relpath(s)).is_file()]
    print(f"[INFO] Dataset {DATASET_ID} v{DATASET_VERSION} -> {target}")
    print(f"[INFO] Selected participants: {len(subjects)}; already present: {len(subjects) - len(pending)}")
    if not pending:
        print("[OK] Nothing to download.")
        return

    # One glob per participant: the EDF plus any sidecar of the same recording.
    include = [f"{s}/{SESSION}/eeg/{s}_{SESSION}_task-{TASK}_acq-{ACQUISITION}_eeg.*" for s in pending]
    try:
        openneuro.download(dataset=DATASET_ID, tag=DATASET_VERSION, target_dir=target, include=include)
    except RuntimeError as exc:
        raise SystemExit(
            f"[ERROR] Download failed: {exc}\n"
            "Timeouts: check the internet connection and run the same command again "
            "(files already downloaded are kept).\n"
            "'Could not find path': check the participant label against participants.tsv."
        )
    except FileExistsError as exc:
        raise SystemExit(
            f"[ERROR] {exc}\nThe target folder holds a different dataset version. "
            "Move it away or choose another --bids-root."
        )

    missing = [s for s in subjects if not (target / edf_relpath(s)).is_file()]
    if missing:
        raise SystemExit(f"[ERROR] Still missing after download: {missing}")
    print(f"[OK] {len(subjects)} participant recording(s) available under {target}")
    print("     Next: python pipeline/run_openneuro.py check  (with the same participant selection)")


if __name__ == "__main__":
    main()
