#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common.io import load_csv_eeg
from common.preprocess import bandpass


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Create residual CSV: residual = REAL - BASE (e.g., NMM-only), "
            "after applying the SAME preprocessing and the SAME (start_sec, duration_sec) clipping."
        )
    )
    p.add_argument("--real", required=True, help="Real CSV path")
    p.add_argument("--base", required=True, help="Baseline CSV path (e.g., NMM-only CSV)")
    p.add_argument("--ch1", default="F7")
    p.add_argument("--ch2", default="F8")
    p.add_argument("--fs", type=float, default=None, help="Sampling rate (Hz). If omitted, try to infer per file.")
    p.add_argument("--bp_low", type=float, default=1.0)
    p.add_argument("--bp_high", type=float, default=45.0)

    p.add_argument(
        "--zscore",
        choices=["none", "global"],
        default="global",
        help="Normalization after bandpass, before clipping. Use global to match evaluate.py protocol.",
    )

    p.add_argument("--start_sec", type=float, default=0.0, help="Drop seconds from beginning BEFORE residual.")
    p.add_argument(
        "--duration_sec",
        type=float,
        default=None,
        help="Fixed duration AFTER start drop. If too long, safely clip to available.",
    )

    p.add_argument("--out_csv", required=True, help="Output residual CSV path")
    p.add_argument("--meta_json", default=None, help="Optional meta.json output path (default: <out_csv>.meta.json)")
    p.add_argument("--tag", default="", help="Optional tag stored in meta")

    return p.parse_args()


def _fill_nan(v: np.ndarray) -> np.ndarray:
    s = pd.Series(v)
    s = s.interpolate(limit_direction="both")
    return s.to_numpy(dtype=float)


def _zscore_global(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = float(np.nanmean(v))
    sd = float(np.nanstd(v))
    if (not np.isfinite(sd)) or sd < eps:
        return v - mu
    return (v - mu) / (sd + eps)


def _make_time_axis(df: pd.DataFrame, n: int, fs: float, start_sample: int = 0) -> np.ndarray:
    # Prefer a time-like column if present; else fallback to sample index / fs
    for c in df.columns:
        if c.lower() in ("time", "timestamp", "t"):
            t = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(t).sum() > 10:
                s = int(max(0, start_sample))
                e = int(min(len(t), s + int(n)))
                t_seg = t[s:e]
                if len(t_seg) < int(n) or np.isfinite(t_seg).sum() < 2:
                    break
                t0 = t_seg[np.isfinite(t_seg)][0]
                return t_seg - t0
    return (np.arange(n, dtype=float) + float(start_sample)) / float(fs)


def _preproc_and_clip(
    csv_path: str,
    ch1: str,
    ch2: str,
    fs_arg: float | None,
    bp_low: float,
    bp_high: float,
    zscore: str,
    req_start_sec: float,
    req_duration_sec: float | None,
):
    # Use the same loader as evaluate.py for consistency
    eeg = load_csv_eeg(csv_path, ch1, ch2, fs=fs_arg)
    fs = float(eeg.fs)
    df = eeg.df

    x = pd.to_numeric(df[ch1], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ch2], errors="coerce").to_numpy(dtype=float)

    x = _fill_nan(x)
    y = _fill_nan(y)

    x_f = bandpass(x, fs, bp_low, bp_high)
    y_f = bandpass(y, fs, bp_low, bp_high)

    if zscore == "global":
        x_f = _zscore_global(x_f)
        y_f = _zscore_global(y_f)

    # requested start
    start_sample = int(round(float(req_start_sec) * fs))
    if start_sample < 0:
        start_sample = 0
    if start_sample >= len(x_f):
        raise ValueError(f"{Path(csv_path).name}: start_sec={req_start_sec} too large for n={len(x_f)}")

    x_f = x_f[start_sample:]
    y_f = y_f[start_sample:]

    # requested duration with safe clipping
    if req_duration_sec is not None:
        requested_samples = int(round(float(req_duration_sec) * fs))
        if requested_samples < 0:
            requested_samples = 0
        available = len(x_f)
        used = min(requested_samples, available)
        if used <= 0:
            raise ValueError(f"{Path(csv_path).name}: too short after start. available={available} samples")
        x_win = x_f[:used]
        y_win = y_f[:used]
        used_duration_sec = used / fs
        used_samples = used
    else:
        requested_samples = None
        x_win = x_f
        y_win = y_f
        used_samples = len(x_win)
        used_duration_sec = used_samples / fs

    # time axis for the clipped window (relative time)
    t = _make_time_axis(df, n=used_samples, fs=fs, start_sample=start_sample)

    return {
        "fs": fs,
        "df": df,
        "t": t,
        "x_win": x_win,
        "y_win": y_win,
        "req_start_sec": float(req_start_sec),
        "req_start_samples": int(start_sample),
        "requested_duration_sec": float(req_duration_sec) if req_duration_sec is not None else None,
        "requested_samples": int(requested_samples) if requested_samples is not None else None,
        "used_duration_sec": float(used_duration_sec),
        "used_samples": int(used_samples),
    }


def main():
    args = parse_args()

    # preprocess + clip each
    real = _preproc_and_clip(
        args.real, args.ch1, args.ch2, args.fs,
        args.bp_low, args.bp_high, args.zscore,
        args.start_sec, args.duration_sec,
    )
    base = _preproc_and_clip(
        args.base, args.ch1, args.ch2, args.fs,
        args.bp_low, args.bp_high, args.zscore,
        args.start_sec, args.duration_sec,
    )

    # enforce same fs
    if abs(real["fs"] - base["fs"]) > 1e-9:
        raise ValueError(f"fs mismatch: real={real['fs']} base={base['fs']}")

    fs = float(real["fs"])

    # enforce same used length by clipping to the minimum (safest for residual definition)
    used_samples = min(real["used_samples"], base["used_samples"])
    if used_samples <= 0:
        raise ValueError("No samples available after alignment.")

    # define aligned arrays
    t = real["t"][:used_samples]  # time axis from real (relative); base is aligned by sample index
    rx = real["x_win"][:used_samples]
    ry = real["y_win"][:used_samples]
    bx = base["x_win"][:used_samples]
    by = base["y_win"][:used_samples]

    # residual
    res_x = rx - bx
    res_y = ry - by

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({
        "time": t,
        args.ch1: res_x,
        args.ch2: res_y,
    })
    out_df.to_csv(out_csv, index=False)

    # meta
    meta_path = Path(args.meta_json) if args.meta_json else out_csv.with_suffix(out_csv.suffix + ".meta.json")
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "real_input": str(Path(args.real).resolve()),
        "base_input": str(Path(args.base).resolve()),
        "output_csv": str(out_csv.resolve()),
        "channels": [args.ch1, args.ch2],
        "fs": fs,
        "bandpass": [args.bp_low, args.bp_high],
        "zscore": args.zscore,

        "start_sec": float(args.start_sec),
        "start_samples": int(round(float(args.start_sec) * fs)),
        "requested_duration_sec": float(args.duration_sec) if args.duration_sec is not None else None,
        "requested_samples": int(round(float(args.duration_sec) * fs)) if args.duration_sec is not None else None,

        # what each side actually had after safe clipping
        "real_used_samples": int(real["used_samples"]),
        "base_used_samples": int(base["used_samples"]),
        # what residual used (final alignment)
        "residual_used_samples": int(used_samples),
        "residual_used_duration_sec": float(used_samples / fs),

        "tag": args.tag,
        "command": " ".join(sys.argv),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        "[OK] residual saved:",
        str(out_csv),
        f"(used_samples={used_samples}, used_duration={used_samples/fs:.3f}s)",
    )
    print("[OK] meta saved:", str(meta_path))


if __name__ == "__main__":
    main()
