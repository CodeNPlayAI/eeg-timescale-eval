# src/common/preprocess.py
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(x: np.ndarray, fs: float, low_hz: float = 1.0, high_hz: float = 45.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid bandpass range: low={low_hz}, high={high_hz}, fs={fs}")
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, x).astype(np.float64)


def window_view(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """
    Return shape: (n_windows, win)
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if win <= 0 or hop <= 0:
        raise ValueError("win and hop must be positive")

    n = x.size
    if n < win:
        return np.empty((0, win), dtype=x.dtype)

    starts = np.arange(0, n - win + 1, hop)
    out = np.stack([x[s : s + win] for s in starts], axis=0)
    return out
