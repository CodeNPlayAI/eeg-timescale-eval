#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2 (Cross-Term Verification) --- post-hoc cross-correlation analysis
=====================================================================

Computes the one-sided absolute mean cross-correlation function (XCF)
between two signals (typically the dynamics-based component
u_NMM(t) and the std-matched generated residual r_hat(t)) per
window, per channel, then aggregates to a seed-level representative.

The normalization scheme mirrors the convention of `acf_mean_abs` in
src/common/metrics.py so that ACF and XCF live on the same scale and
are directly comparable in the paper.

Per window:
    u_c = u - mean(u)
    r_c = r - mean(r)
    denom = sqrt(sum(u_c**2) * sum(r_c**2))
    rho(tau) = sum(u_c[:N-tau] * r_c[tau:]) / denom    for tau = 1..L
    XCF@T   = mean_{tau=1..L_T} |rho(tau)|             where L_T = round(T*fs)

Hierarchy:
    each (channel, window) -> XCF@0.1s, XCF@0.5s, XCF@1.0s
    -> mean over windows  (per channel)
    -> mean over channels (seed representative)

Inputs (CSVs already preprocessed: bandpass + global z-score + start/duration
clipping, possibly std-matched). NO further bandpass/z-score is applied here.

Outputs:
    --out_csv         : seed-level summary (one row, with channel-mean values)
    --out_window_csv  : per-window/per-channel detail (diagnostic)
    --meta_json       : configuration + provenance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from common.preprocess import window_view


# ---------------------------------------------------------------------------
# Core: cross-correlation summary (one-sided, |rho|, mean over lags 1..L_T)
# ---------------------------------------------------------------------------
def xcf_mean_abs(u: np.ndarray, r: np.ndarray, max_lag: int) -> float:
    """
    One-sided mean absolute normalized cross-correlation up to max_lag.

    Mirrors the convention of `acf_mean_abs` in metrics.py so that ACF and XCF
    are directly comparable on the same scale.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        First signal (e.g., u_NMM(t) within a window).
    r : np.ndarray, shape (N,)
        Second signal (e.g., r_hat(t) within a window). Must be the same
        length as u.
    max_lag : int
        Maximum lag in samples (inclusive). Lag = 0 is excluded.

    Returns
    -------
    float
        mean_{tau=1..L} |rho(tau)|, or NaN if undefined.
    """
    u = np.asarray(u, dtype=float)
    r = np.asarray(r, dtype=float)
    if u.shape != r.shape or u.ndim != 1:
        raise ValueError("u and r must be 1-D arrays of the same length")

    u = u - u.mean()
    r = r - r.mean()
    denom = float(np.sqrt(np.dot(u, u) * np.dot(r, r)))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")

    N = u.size
    if max_lag < 1 or max_lag >= N:
        max_lag = min(max_lag, N - 1)
    if max_lag < 1:
        return float("nan")

    vals = []
    for lag in range(1, max_lag + 1):
        # rho(tau) = sum( u[:N-tau] * r[tau:] ) / denom
        v = float(np.dot(u[: N - lag], r[lag:]) / denom)
        vals.append(abs(v))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fill_nan_1d(v: np.ndarray) -> np.ndarray:
    s = pd.Series(v).interpolate(limit_direction="both")
    return s.to_numpy(dtype=float)


def _read_channel(csv_path: Path, ch: str) -> np.ndarray:
    """Load a single channel from a preprocessed CSV using pandas directly.

    Using pd.read_csv() rather than load_csv_eeg() avoids the risk of
    unexpected behaviour when ch1 == ch2 inside that helper.
    """
    df = pd.read_csv(csv_path)
    if ch not in df.columns:
        raise ValueError(
            f"Channel '{ch}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    x = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
    return _fill_nan_1d(x)


def _windowed_xcf(
    u_full: np.ndarray,
    r_full: np.ndarray,
    fs: float,
    win_samples: int,
    hop_samples: int,
    lag_samples_list: list[int],
) -> pd.DataFrame:
    """
    Compute XCF@lag for each window. Returns a DataFrame with one row per window.
    Columns: window_idx, n_samples, XCF_at_<lag_label> ...
    """
    if u_full.shape != r_full.shape:
        raise ValueError(f"u and r length mismatch: {u_full.shape} vs {r_full.shape}")

    U = window_view(u_full, win=win_samples, hop=hop_samples)  # (n_win, win)
    R = window_view(r_full, win=win_samples, hop=hop_samples)
    if U.shape[0] == 0:
        return pd.DataFrame()

    rows = []
    for w_idx in range(U.shape[0]):
        u_w = U[w_idx]
        r_w = R[w_idx]
        row = {"window_idx": w_idx, "n_samples": int(win_samples)}
        for lag in lag_samples_list:
            row[f"xcf_at_{lag}_samples"] = xcf_mean_abs(u_w, r_w, max_lag=lag)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute one-sided mean absolute cross-correlation (XCF) between "
            "u_NMM(t) and r_hat(t) for E2 (Cross-Term Verification). "
            "Inputs are expected to be already preprocessed and aligned."
        )
    )
    p.add_argument("--base", required=True, help="Base CSV (e.g., NMM aligned).")
    p.add_argument("--residual", required=True, help="Residual CSV (e.g., std-matched residual_hat).")
    p.add_argument("--ch1", default="F7")
    p.add_argument("--ch2", default="F8")
    p.add_argument("--fs", type=float, default=256.0)
    p.add_argument("--win_s", type=float, default=10.0)
    p.add_argument("--hop_s", type=float, default=5.0)
    p.add_argument(
        "--lag_secs",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Lag thresholds in seconds; XCF is computed as the mean |rho| "
             "over lags 1..round(T*fs) for each T.",
    )
    p.add_argument("--out_csv", required=True, help="Seed-level summary CSV (one row).")
    p.add_argument(
        "--out_window_csv",
        default=None,
        help="Optional per-window/per-channel detail CSV.",
    )
    p.add_argument("--meta_json", default=None, help="Optional meta JSON output path.")
    p.add_argument("--seed", type=int, default=None, help="Optional seed tag stored in outputs.")
    p.add_argument("--tag", default="", help="Optional tag stored in meta.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fs = float(args.fs)

    win_samples = int(round(float(args.win_s) * fs))
    hop_samples = int(round(float(args.hop_s) * fs))
    lag_samples_list = [int(round(float(lag) * fs)) for lag in args.lag_secs]
    lag_label_map = {
        int(round(float(lag) * fs)): f"{lag:.1f}s" for lag in args.lag_secs
    }

    base_path = Path(args.base)
    res_path = Path(args.residual)
    if not base_path.exists():
        raise FileNotFoundError(f"--base not found: {base_path}")
    if not res_path.exists():
        raise FileNotFoundError(f"--residual not found: {res_path}")

    # Load aligned signals (no further bandpass / z-score; inputs are already preprocessed)
    detail_rows = []
    seed_summary_row = {
        "base_csv": str(base_path),
        "residual_csv": str(res_path),
        "fs": fs,
        "win_s": float(args.win_s),
        "hop_s": float(args.hop_s),
    }
    if args.seed is not None:
        seed_summary_row["seed"] = int(args.seed)

    per_channel_means: dict[str, dict[str, float]] = {}
    n_windows_recorded = False  # record once from the first channel

    for ch in (args.ch1, args.ch2):
        u_full = _read_channel(base_path, ch=ch)
        r_full = _read_channel(res_path, ch=ch)

        # Truncate to the common length defensively
        N = min(u_full.size, r_full.size)
        u_full = u_full[:N]
        r_full = r_full[:N]

        df_win = _windowed_xcf(
            u_full=u_full,
            r_full=r_full,
            fs=fs,
            win_samples=win_samples,
            hop_samples=hop_samples,
            lag_samples_list=lag_samples_list,
        )
        if df_win.empty:
            print(f"[WARN] channel {ch}: no windows produced (signal too short).")
            continue

        df_win.insert(0, "channel", ch)
        if args.seed is not None:
            df_win.insert(0, "seed", int(args.seed))
        detail_rows.append(df_win)

        # Record n_windows once from the first channel (same for all channels).
        if not n_windows_recorded:
            seed_summary_row["n_windows"] = int(df_win["window_idx"].nunique())
            n_windows_recorded = True

        # Channel-level mean over windows
        ch_means = {}
        for lag_samples in lag_samples_list:
            colname = f"xcf_at_{lag_samples}_samples"
            ch_means[lag_label_map[lag_samples]] = float(df_win[colname].mean())
        per_channel_means[ch] = ch_means

    if not per_channel_means:
        raise RuntimeError("No channels produced any windows; aborting.")

    # Seed representative = mean over channels (matches the ACF protocol in evaluate.py)
    for lag_label in [lag_label_map[ls] for ls in lag_samples_list]:
        ch_vals = [per_channel_means[ch].get(lag_label, np.nan)
                   for ch in per_channel_means]
        seed_summary_row[f"XCF_at_{lag_label}_{args.ch1}"] = (
            per_channel_means.get(args.ch1, {}).get(lag_label, np.nan)
        )
        seed_summary_row[f"XCF_at_{lag_label}_{args.ch2}"] = (
            per_channel_means.get(args.ch2, {}).get(lag_label, np.nan)
        )
        seed_summary_row[f"XCF_at_{lag_label}_mean"] = float(np.nanmean(ch_vals))

    # Outputs
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([seed_summary_row]).to_csv(out_csv, index=False)
    print(f"[OK] seed summary written: {out_csv}")

    if args.out_window_csv:
        out_win = Path(args.out_window_csv)
        out_win.parent.mkdir(parents=True, exist_ok=True)
        if detail_rows:
            pd.concat(detail_rows, ignore_index=True).to_csv(out_win, index=False)
            print(f"[OK] per-window detail written: {out_win}")

    # Always write meta for reproducibility; path defaults to out_csv stem.
    meta_path = Path(args.meta_json) if args.meta_json else out_csv.with_suffix(".meta.json")
    meta = {
        "tool": "evaluate_cross_term_e2.py",
        "version": "1.0",
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "base_csv": str(base_path),
            "residual_csv": str(res_path),
            "ch1": args.ch1,
            "ch2": args.ch2,
            "fs": fs,
        },
        "params": {
            "win_s": float(args.win_s),
            "hop_s": float(args.hop_s),
            "win_samples": int(win_samples),
            "hop_samples": int(hop_samples),
            "lag_secs": [float(x) for x in args.lag_secs],
            "lag_samples": [int(x) for x in lag_samples_list],
        },
        "definition": {
            "xcf": (
                "rho(tau) = sum(u_c[:N-tau] * r_c[tau:]) / "
                "sqrt(sum(u_c^2) * sum(r_c^2)); "
                "XCF@T = mean_{tau=1..round(T*fs)} |rho(tau)|. "
                "Mirrors acf_mean_abs in src/common/metrics.py."
            ),
            "hierarchy": "per-window XCF -> mean over windows (per channel) -> mean over channels (seed)",
        },
        "seed": int(args.seed) if args.seed is not None else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] meta written: {meta_path}")


if __name__ == "__main__":
    main()
