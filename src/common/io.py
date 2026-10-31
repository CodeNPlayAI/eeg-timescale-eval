# src/common/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class EEGData:
    df: pd.DataFrame
    fs: float  # sampling frequency (Hz)


def _infer_fs_from_time(time_series: pd.Series) -> float | None:
    """
    Infer sampling frequency from a time column (seconds or ms or datetime-like).
    Returns None if cannot infer.
    """
    s = time_series.dropna()

    # datetime-like
    if np.issubdtype(s.dtype, np.datetime64):
        t = pd.to_datetime(s).astype("int64") / 1e9  # seconds
        dt = np.diff(t)
    else:
        # numeric: seconds or ms; we guess by scale
        t = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
        if t.size < 3:
            return None
        # if it looks like milliseconds, convert
        if np.nanmedian(np.diff(t)) > 0.5:  # rough heuristic
            # could still be seconds with low fs; leave it
            pass
        dt = np.diff(t)

    dt = dt[np.isfinite(dt)]
    if dt.size < 10:
        return None

    med = float(np.median(dt))
    if med <= 0:
        return None
    return 1.0 / med


def load_csv_eeg(path: str | Path, ch1: str, ch2: str, fs: float | None = None) -> EEGData:
    path = Path(path)
    df = pd.read_csv(path)

    # Ensure channels exist
    for ch in (ch1, ch2):
        if ch not in df.columns:
            raise ValueError(f"Channel '{ch}' not found. columns={list(df.columns)[:20]}...")

    # Infer fs if not provided
    if fs is None:
        # common candidates for time column
        candidates = [c for c in df.columns if c.lower() in ("time", "timestamp", "t", "datetime")]
        inferred = None
        for c in candidates:
            try:
                inferred = _infer_fs_from_time(df[c])
            except Exception:
                inferred = None
            if inferred is not None:
                break
        if inferred is None:
            raise ValueError(
                "fs is not provided and cannot be inferred from a time column. "
                "Please pass --fs explicitly."
            )
        fs = inferred

    return EEGData(df=df, fs=float(fs))
