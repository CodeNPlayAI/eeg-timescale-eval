from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from common.io import load_csv_eeg
from common.preprocess import bandpass, window_view
from common.metrics import psd_bandpower, acf_mean_abs, dfa_alpha
from common.granger import granger_var_aic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV path")
    p.add_argument("--ch1", default="F7")
    p.add_argument("--ch2", default="F8")
    p.add_argument("--fs", type=float, default=None, help="Sampling rate (Hz). If omitted, try to infer.")
    p.add_argument("--win_s", type=float, default=10.0)
    p.add_argument("--hop_s", type=float, default=5.0)
    p.add_argument(
        "--start_sec",
        type=float,
        default=0.0,
        help="Start offset (seconds) to drop from the beginning BEFORE windowing/metrics. Use to align time bands across conditions.",
    )
    p.add_argument(
        "--duration_sec",
        type=float,
        default=None,
        help=(
            "Optional fixed evaluation duration in seconds AFTER start_sec drop. "
            "If set, signals are truncated to this duration before windowing/metrics to ensure equal length across conditions."
        ),
    )
    p.add_argument("--bp_low", type=float, default=1.0)
    p.add_argument("--bp_high", type=float, default=45.0)
    p.add_argument(
        "--zscore",
        choices=["none", "global", "window"],
        default="global",
        help="Z-score normalization mode. Protocol default: global (after bandpass, before windowing).",
    )
    p.add_argument("--dfa_min_s", type=float, default=0.2, help="DFA min scale in seconds (PoC default: 0.2)")
    p.add_argument("--dfa_max_s", type=float, default=1.25, help="DFA max scale in seconds (PoC default: 2.0)")

    p.add_argument("--outdir", default=None, help="runs directory. default: runs/<timestamp>")

    # --- diagnostics / plotting ---
    p.add_argument("--plots", action="store_true", help="Save diagnostic plots to <outdir>/plots/")
    p.add_argument("--top_k_bp14", type=int, default=3, help="Top-K windows to plot by bp_x_1_4")
    p.add_argument(
        "--exclude_bp14_mult",
        type=float,
        default=3.0,
        help="If >0, exclude windows with bp_x_1_4 > median*mult and plot PSD before/after.",
    )
    p.add_argument("--no_gc", action="store_true", help="Skip Granger causality computation (faster)")
    p.add_argument("--tag", type=str, default="", help="Tag appended to output directory")

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


def _zscore_window(Vw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Window-wise z-score. Vw shape: (n_windows, win_len)."""
    mu = np.nanmean(Vw, axis=1, keepdims=True)
    sd = np.nanstd(Vw, axis=1, keepdims=True)
    sd = np.where((~np.isfinite(sd)) | (sd < eps), 1.0, sd)
    return (Vw - mu) / sd


def _make_time_axis(df: pd.DataFrame, n: int, fs: float, start_sample: int = 0) -> np.ndarray:
    # Prefer a time column if present; else fallback to sample index / fs
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


def _ensure_plot_dir(outdir: Path) -> Path:
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    return plotdir


def psd_loglog_slope(f: np.ndarray, pxx: np.ndarray, fmin: float = 2.0, fmax: float = 40.0) -> float:
    """
    Fit slope of log10(PSD) vs log10(f) within [fmin, fmax].
    Return slope (expected negative for 1/f-like).
    """
    f = np.asarray(f)
    pxx = np.asarray(pxx)
    m = (f >= fmin) & (f <= fmax) & np.isfinite(pxx) & (pxx > 0) & np.isfinite(f) & (f > 0)
    if m.sum() < 5:
        return float("nan")
    x = np.log10(f[m])
    y = np.log10(pxx[m])
    slope = float(np.polyfit(x, y, deg=1)[0])
    return slope


def _plot_average_psd(x_f: np.ndarray, y_f: np.ndarray, fs: float, plot_path: Path, title: str):
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    nperseg = int(round(fs * 2.0))
    noverlap = int(round(fs * 1.0))
    f, pxx_x = welch(x_f, fs=fs, nperseg=min(nperseg, len(x_f)), noverlap=min(noverlap, nperseg // 2))
    _, pxx_y = welch(y_f, fs=fs, nperseg=min(nperseg, len(y_f)), noverlap=min(noverlap, nperseg // 2))
    pxx = 0.5 * (pxx_x + pxx_y)

    slope = psd_loglog_slope(f, pxx, fmin=2.0, fmax=40.0)

    plt.figure()
    plt.semilogy(f, pxx)
    try:
        plt.ylim(bottom=1e-7)
    except Exception:
        pass
    plt.xlim(0, 45)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (a.u.^2/Hz)")
    plt.title(f"{title}\nlog-log slope(2–40Hz)={slope:.3f}")
    plt.axvline(8, linestyle="--")
    plt.axvline(12, linestyle="--")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return slope


def _plot_window_waveform(
    t: np.ndarray,
    x_f: np.ndarray,
    y_f: np.ndarray,
    fs: float,
    win_idx: int,
    win: int,
    hop: int,
    plot_path: Path,
    title: str,
):
    import matplotlib.pyplot as plt

    s = win_idx * hop
    e = s + win
    if e > len(x_f):
        return

    plt.figure()
    plt.plot(t[s:e], x_f[s:e], label="ch1")
    plt.plot(t[s:e], y_f[s:e], label="ch2")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def _mask_from_excluded_windows(n: int, win: int, hop: int, excluded: list[int]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    for w in excluded:
        s = w * hop
        e = s + win
        mask[s:e] = False
    return mask


def _add_scalar_metric_to_summary(summary: pd.DataFrame, metric_name: str, value: float) -> pd.DataFrame:
    summary.loc[metric_name, "count"] = 1.0
    summary.loc[metric_name, "mean"] = float(value)
    summary.loc[metric_name, "std"] = np.nan
    summary.loc[metric_name, "min"] = float(value)
    summary.loc[metric_name, "25%"] = float(value)
    summary.loc[metric_name, "50%"] = float(value)
    summary.loc[metric_name, "75%"] = float(value)
    summary.loc[metric_name, "max"] = float(value)
    return summary


def main():
    args = parse_args()

    eeg = load_csv_eeg(args.input, args.ch1, args.ch2, fs=args.fs)
    fs = eeg.fs
    df = eeg.df

    x = pd.to_numeric(df[args.ch1], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[args.ch2], errors="coerce").to_numpy(dtype=float)

    x = _fill_nan(x)
    y = _fill_nan(y)

    x_f = bandpass(x, fs, args.bp_low, args.bp_high)
    y_f = bandpass(y, fs, args.bp_low, args.bp_high)

    # Protocol: bandpass -> z-score -> windowing
    if args.zscore == "global":
        x_f = _zscore_global(x_f)
        y_f = _zscore_global(y_f)

    start_sample = int(round(float(args.start_sec) * float(fs)))
    req_start_sec = float(args.start_sec)
    req_start_samples = int(start_sample)
    if start_sample > 0:
        if start_sample >= len(x_f):
            raise ValueError(f"start_sec={args.start_sec} is too large for signal length (n={len(x_f)})")
        x_f = x_f[start_sample:]
        y_f = y_f[start_sample:]
    else:
        start_sample = 0

    if args.duration_sec is not None:
        fs = float(fs)
        eval_start_sec = 0.0
        eval_start_samples = 0

        requested_samples = int(round(float(args.duration_sec) * fs))

        x_after_start = x_f[eval_start_samples:]
        y_after_start = y_f[eval_start_samples:]
        available_samples = len(x_after_start)
        used_samples = min(requested_samples, available_samples)

        if used_samples <= 0:
            raise ValueError(
                f"Too short after start. available={available_samples} samples, "
                f"requested={requested_samples} samples, start_samples={eval_start_samples}"
            )

        x_win = x_after_start[:used_samples]
        y_win = y_after_start[:used_samples]
        used_duration_sec = used_samples / fs

    else:
        x_win = x_f
        y_win = y_f
        used_duration_sec = len(x_win) / fs
        used_samples = len(x_win)
        requested_samples = None
        eval_start_sec = 0.0
        eval_start_samples = 0

    win = int(args.win_s * fs)
    hop = int(args.hop_s * fs)

    Xw = window_view(x_win, win, hop)
    Yw = window_view(y_win, win, hop)

    if args.zscore == "window":
        Xw = _zscore_window(Xw)
        Yw = _zscore_window(Yw)

    # output dir
    if args.outdir is None:
        rid = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_outdir = Path("runs") / rid
    else:
        base_outdir = Path(args.outdir)

    if args.tag:
        outdir = base_outdir.parent / f"{base_outdir.name}_{args.tag}"
    else:
        outdir = base_outdir

    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(Xw.shape[0]):
        xw = Xw[i]
        yw = Yw[i]

        bp_x_1_4 = 0.5 * (psd_bandpower(xw, fs, 1, 4) + psd_bandpower(yw, fs, 1, 4))
        bp_x_4_8 = 0.5 * (psd_bandpower(xw, fs, 4, 8) + psd_bandpower(yw, fs, 4, 8))
        bp_x_8_13 = 0.5 * (psd_bandpower(xw, fs, 8, 13) + psd_bandpower(yw, fs, 8, 13))

        acf_x = 0.5 * (
            acf_mean_abs(xw, max_lag=min(int(fs * 2), len(xw) - 2))
            + acf_mean_abs(yw, max_lag=min(int(fs * 2), len(yw) - 2))
        )
        lag01 = min(int(fs * 0.1), len(xw) - 2)
        lag05 = min(int(fs * 0.5), len(xw) - 2)
        lag10 = min(int(fs * 1.0), len(xw) - 2)

        acf_x_01 = 0.5 * (acf_mean_abs(xw, max_lag=lag01) + acf_mean_abs(yw, max_lag=lag01))
        acf_x_05 = 0.5 * (acf_mean_abs(xw, max_lag=lag05) + acf_mean_abs(yw, max_lag=lag05))
        acf_x_10 = 0.5 * (acf_mean_abs(xw, max_lag=lag10) + acf_mean_abs(yw, max_lag=lag10))

        dfa_x = 0.5 * (
            dfa_alpha(
                xw,
                fs,
                min_scale_s=args.dfa_min_s,
                max_scale_s=min(args.dfa_max_s, args.win_s),
                n_scales=10,
            )
            + dfa_alpha(
                yw,
                fs,
                min_scale_s=args.dfa_min_s,
                max_scale_s=min(args.dfa_max_s, args.win_s),
                n_scales=10,
            )
        )

        g_lag = None
        g_x_to_y_p = float("nan")
        g_y_to_x_p = float("nan")
        if not args.no_gc:
            maxlags = min(12, len(xw) // 10)
            g = granger_var_aic(xw, yw, maxlags=maxlags)
            g_lag = g["lag"]
            g_x_to_y_p = g["x->y_p"]
            g_y_to_x_p = g["y->x_p"]

        rows.append(
            {
                "win_idx": i,
                "fs": fs,
                "bp_x_1_4": bp_x_1_4,
                "bp_x_4_8": bp_x_4_8,
                "bp_x_8_13": bp_x_8_13,
                "acf_x_mean_abs": acf_x,
                "acf_x_abs_0_1s": acf_x_01,
                "acf_x_abs_0_5s": acf_x_05,
                "acf_x_abs_1_0s": acf_x_10,
                "dfa_x_alpha": dfa_x,
                "granger_lag": g_lag,
                "granger_x_to_y_p": g_x_to_y_p,
                "granger_y_to_x_p": g_y_to_x_p,
            }
        )

    win_df = pd.DataFrame(rows)
    win_df.to_csv(outdir / "metrics_windows.csv", index=False)

    summary = win_df.drop(columns=["win_idx"]).describe().T

    meta = {
        "input": str(Path(args.input).resolve()),
        "channels": [args.ch1, args.ch2],
        "fs": fs,
        "bandpass": [args.bp_low, args.bp_high],
        "zscore": args.zscore,
        "window_s": args.win_s,
        "hop_s": args.hop_s,
        "start_sec": float(req_start_sec),
        "start_samples": int(req_start_samples),
        "eval_start_sec": float(eval_start_sec),
        "eval_start_samples": int(eval_start_samples),
        "requested_duration_sec": float(args.duration_sec) if args.duration_sec is not None else None,
        "used_duration_sec": float(used_duration_sec),
        "requested_samples": int(requested_samples) if requested_samples is not None else None,
        "used_samples": int(used_samples),
        "duration_sec": args.duration_sec,
        "duration_samples": used_samples if args.duration_sec is not None else None,
        "n_samples_after_trim": len(x_win),
        "tag": args.tag,
        "command": " ".join(sys.argv),
    }

    # -------------------------
    # Diagnostics / plots
    # -------------------------
    if args.plots:
        plotdir = _ensure_plot_dir(outdir)
        t = _make_time_axis(df, n=len(x_win), fs=fs, start_sample=start_sample)

        # (1) Average PSD (all data)
        slope_all = _plot_average_psd(
            x_win,
            y_win,
            fs,
            plot_path=plotdir / "avg_psd_all.png",
            title=f"Average PSD (mean of {args.ch1}/{args.ch2}) - bandpass {args.bp_low:.0f}–{args.bp_high:.0f} Hz",
        )
        meta["psd_loglog_slope_all_2_40hz"] = float(slope_all)
        summary = _add_scalar_metric_to_summary(summary, "psd_loglog_slope_all_2_40hz", slope_all)

        # (2) Waveforms for top-K windows by bp_x_1_4
        top = win_df.sort_values("bp_x_1_4", ascending=False).head(max(1, args.top_k_bp14))
        top[["win_idx", "bp_x_1_4", "bp_x_4_8", "bp_x_8_13"]].to_csv(
            plotdir / "top_windows_by_bp_x_1_4.csv", index=False
        )

        for _, r in top.iterrows():
            widx = int(r["win_idx"])
            _plot_window_waveform(
                t,
                x_win,
                y_win,
                fs,
                win_idx=widx,
                win=win,
                hop=hop,
                plot_path=plotdir / f"waveform_top_bp14_win{widx}.png",
                title=f"Waveform (bandpass {args.bp_low:.0f}–{args.bp_high:.0f} Hz) - window {widx}",
            )

        # (3) Optional: exclusion by bp_x_1_4 threshold and PSD before/after
        if args.exclude_bp14_mult is not None and args.exclude_bp14_mult > 0:
            med = float(np.nanmedian(win_df["bp_x_1_4"].to_numpy()))
            thr = med * float(args.exclude_bp14_mult)
            excluded = win_df.loc[win_df["bp_x_1_4"] > thr, "win_idx"].astype(int).tolist()
            pd.DataFrame({"excluded_win_idx": excluded, "threshold": thr}).to_csv(
                plotdir / "excluded_windows.csv", index=False
            )

            slope_before = _plot_average_psd(
                x_win,
                y_win,
                fs,
                plot_path=plotdir / "avg_psd_before_exclusion.png",
                title=f"Average PSD (before exclusion) - threshold=median*{args.exclude_bp14_mult:g}",
            )
            meta["psd_loglog_slope_before_exclusion_2_40hz"] = float(slope_before)

            if excluded:
                mask = _mask_from_excluded_windows(len(x_win), win, hop, excluded)
                slope_after = _plot_average_psd(
                    x_win[mask],
                    y_win[mask],
                    fs,
                    plot_path=plotdir / "avg_psd_after_exclusion.png",
                    title=f"Average PSD (after exclusion) - removed {len(excluded)} windows",
                )
                meta["psd_loglog_slope_after_exclusion_2_40hz"] = float(slope_after)

    summary.to_csv(outdir / "metrics_summary.csv")
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.duration_sec is not None:
        print(
            f"[OK] saved to: {outdir} "
            f"(requested duration: {args.duration_sec}s, used duration: {used_duration_sec:.3f}s, "
            f"requested samples: {requested_samples}, used samples: {used_samples})"
        )
    else:
        print(f"[OK] saved to: {outdir}")


if __name__ == "__main__":
    main()