# src/common/metrics.py
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


def psd_bandpower(x: np.ndarray, fs: float, fmin: float, fmax: float, nperseg: int | None = None) -> float:
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return float("nan")
    return float(trapezoid(pxx[mask], f[mask]))


def acf_mean_abs(x: np.ndarray, max_lag: int) -> float:
    """
    Simple ACF magnitude summary up to max_lag (excluding lag=0).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return float("nan")

    vals = []
    for lag in range(1, max_lag + 1):
        v = np.dot(x[:-lag], x[lag:]) / denom
        vals.append(abs(v))
    return float(np.mean(vals)) if vals else float("nan")


def dfa_alpha(x: np.ndarray, fs: float, min_scale_s: float = 0.5, max_scale_s: float = 10.0, n_scales: int = 10) -> float:
    """
    Minimal DFA implementation (linear detrend per window).
    Returns alpha (slope in log-log).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    y = np.cumsum(x)  # profile

    min_n = max(8, int(min_scale_s * fs))
    max_n = max(min_n + 1, int(max_scale_s * fs))
    if max_n >= y.size:
        max_n = y.size - 1
    if max_n <= min_n:
        return float("nan")

    scales = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), n_scales).astype(int))
    F = []
    S = []
    for n in scales:
        k = y.size // n
        if k < 2:
            continue
        y0 = y[: k * n].reshape(k, n)
        t = np.arange(n, dtype=float)

        # detrend each segment
        rms = []
        for seg in y0:
            # linear fit
            coef = np.polyfit(t, seg, 1)
            trend = coef[0] * t + coef[1]
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
        F.append(np.mean(rms))
        S.append(n)

    if len(F) < 3:
        return float("nan")

    logS = np.log10(np.asarray(S, dtype=float))
    logF = np.log10(np.asarray(F, dtype=float))
    a, b = np.polyfit(logS, logF, 1)
    return float(a)
