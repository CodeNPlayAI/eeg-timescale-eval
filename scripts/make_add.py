#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _zscore_global(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return x * 0.0
    return (x - mu) / sd


def _parse_args():
    p = argparse.ArgumentParser(description="Add composer: base (NMM) + residual_hat -> add.csv")
    p.add_argument("--base", required=True, help="Base CSV (already aligned if needed)")
    p.add_argument("--residual", required=True, help="Residual-hat CSV")
    p.add_argument("--out_csv", required=True, help="Output CSV path for Add (base + residual)")

    p.add_argument("--ch1", default="F7")
    p.add_argument("--ch2", default="F8")
    p.add_argument("--fs", type=float, required=True)

    p.add_argument("--bp_low", type=float, default=1.0)
    p.add_argument("--bp_high", type=float, default=45.0)
    p.add_argument("--zscore", choices=["none", "global"], default="global")

    p.add_argument(
        "--start_sec",
        type=float,
        default=0.0,
        help="Start offset (seconds) applied to both already-prepared inputs before composing."
    )
    p.add_argument(
        "--duration_sec",
        type=float,
        default=None,
        help="Duration (seconds) to keep after start_sec. If omitted, keep until end."
    )

    p.add_argument("--tag", default="", help="Optional tag to record in meta")
    p.add_argument(
        "--b",
        type=float,
        default=1.0,
        help="Residual mixing coefficient: Add = base + b * residual_hat"
    )
    p.add_argument(
        "--no_preproc",
        action="store_true",
        help="If set, do NOT apply bandpass/zscore here; just clip + sum."
    )

    return p.parse_args()


def _load_xy(csv_path: str, ch1: str, ch2: str, fs: float):
    from common.io import load_csv_eeg

    eeg = load_csv_eeg(csv_path, ch1, ch2, fs=fs)
    df = eeg.df
    x = pd.to_numeric(df[ch1], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ch2], errors="coerce").to_numpy(dtype=float)
    return x, y


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    try:
        from common.preprocess import bandpass  # type: ignore
        return np.asarray(bandpass(x, fs, lo, hi), dtype=float)
    except Exception as e:
        raise RuntimeError(
            "bandpass() unavailable. Install/ensure repo common.preprocess.bandpass is importable, "
            "or run with --no_preproc."
        ) from e


def _clip(x: np.ndarray, start_samples: int, dur_samples: Optional[int]) -> np.ndarray:
    if start_samples < 0:
        raise ValueError("start_sec must be >= 0")
    if start_samples >= len(x):
        raise ValueError(f"start_sec too large (start_samples={start_samples}, n={len(x)})")
    x2 = x[start_samples:]
    if dur_samples is not None:
        if dur_samples <= 0:
            raise ValueError("duration_sec must be > 0")
        x2 = x2[:dur_samples]
    return x2


@dataclass
class Meta:
    created_at: str
    base: str
    residual: str
    out_csv: str
    channels: list[str]
    fs: float
    bandpass: list[float]
    zscore: str
    no_preproc: bool
    start_sec: float
    start_samples: int
    requested_duration_sec: Optional[float]
    requested_samples: Optional[int]
    used_samples: int
    used_duration_sec: float
    b: float
    tag: str
    command: str


def main():
    args = _parse_args()

    fs = float(args.fs)
    start_samples = int(round(float(args.start_sec) * fs))
    dur_samples = None
    if args.duration_sec is not None:
        dur_samples = int(round(float(args.duration_sec) * fs))

    xb, yb = _load_xy(args.base, args.ch1, args.ch2, fs)
    xr, yr = _load_xy(args.residual, args.ch1, args.ch2, fs)

    if not args.no_preproc:
        xb = _bandpass(xb, fs, args.bp_low, args.bp_high)
        yb = _bandpass(yb, fs, args.bp_low, args.bp_high)
        xr = _bandpass(xr, fs, args.bp_low, args.bp_high)
        yr = _bandpass(yr, fs, args.bp_low, args.bp_high)

        if args.zscore == "global":
            xb = _zscore_global(xb)
            yb = _zscore_global(yb)
            xr = _zscore_global(xr)
            yr = _zscore_global(yr)

    xb = _clip(xb, start_samples, dur_samples)
    yb = _clip(yb, start_samples, dur_samples)
    xr = _clip(xr, start_samples, dur_samples)
    yr = _clip(yr, start_samples, dur_samples)

    n = min(len(xb), len(yb), len(xr), len(yr))
    xb, yb, xr, yr = xb[:n], yb[:n], xr[:n], yr[:n]

    b = float(args.b)
    xa = xb + b * xr
    ya = yb + b * yr

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({args.ch1: xa, args.ch2: ya}).to_csv(out_csv, index=False)

    used_duration = float(n) / fs

    meta = Meta(
        created_at=datetime.now().isoformat(timespec="seconds"),
        base=str(Path(args.base)),
        residual=str(Path(args.residual)),
        out_csv=str(out_csv),
        channels=[args.ch1, args.ch2],
        fs=fs,
        bandpass=[float(args.bp_low), float(args.bp_high)],
        zscore=str(args.zscore),
        no_preproc=bool(args.no_preproc),
        start_sec=float(args.start_sec),
        start_samples=int(start_samples),
        requested_duration_sec=(None if args.duration_sec is None else float(args.duration_sec)),
        requested_samples=(None if dur_samples is None else int(dur_samples)),
        used_samples=int(n),
        used_duration_sec=float(used_duration),
        b=float(args.b),
        tag=str(args.tag),
        command=" ".join([Path(sys.argv[0]).as_posix(), *sys.argv[1:]]),
    )

    meta_path = Path(str(out_csv) + ".meta.json")
    meta_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Add saved: {out_csv} (b={b}, used_samples={n}, used_duration={used_duration:.3f}s)")
    print(f"[OK] meta saved: {meta_path}")


if __name__ == "__main__":
    main()