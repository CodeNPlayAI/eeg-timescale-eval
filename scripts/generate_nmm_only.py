#!/usr/bin/env python3
"""
generate_nmm_only.py

Generate 2-channel EEG-like signals from Jansen–Rit neural mass model.
Outputs CSV with columns: time, F7, F8

Usage example:
python scripts/generate_nmm_only.py \
    --out data/nmm_3min.csv \
    --tag seed0_c0p02 \
    --fs 256 \
    --duration_s 180 \
    --seed 0 \
    --coupling 0.02 \
    --ch1 F7 \
    --ch2 F8
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# Jansen–Rit neural mass model (simplified)
# ------------------------------------------------------------
def jansen_rit_simulation(fs, duration_s, seed=0, coupling=0.02):
    np.random.seed(seed)

    dt = 1.0 / fs
    n = int(fs * duration_s)

    # Parameters (standard-ish)
    A = 3.25
    B = 22.0
    a = 100.0
    b = 50.0
    C = 135.0

    C1 = C
    C2 = 0.8 * C
    C3 = 0.25 * C
    C4 = 0.25 * C

    # State variables
    y0 = np.zeros(n)
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)
    y4 = np.zeros(n)
    y5 = np.zeros(n)

    # External input (noise)
    p = 120 + 30 * np.random.randn(n)

    def sigmoid(v):
        return 2.0 * 2.5 / (1.0 + np.exp(0.56 * (6.0 - v)))

    for i in range(1, n):
        S1 = sigmoid(y1[i-1] - y2[i-1])
        S2 = sigmoid(C1 * y0[i-1])
        S3 = sigmoid(C3 * y0[i-1])

        y0[i] = y0[i-1] + dt * y3[i-1]
        y3[i] = y3[i-1] + dt * (
            A * a * S1 - 2 * a * y3[i-1] - a**2 * y0[i-1]
        )

        y1[i] = y1[i-1] + dt * y4[i-1]
        y4[i] = y4[i-1] + dt * (
            A * a * (p[i] + coupling * S2)
            - 2 * a * y4[i-1]
            - a**2 * y1[i-1]
        )

        y2[i] = y2[i-1] + dt * y5[i-1]
        y5[i] = y5[i-1] + dt * (
            B * b * (C4 * S3)
            - 2 * b * y5[i-1]
            - b**2 * y2[i-1]
        )

    eeg = y1 - y2
    return eeg


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--tag", type=str, default="",
                    help="Tag appended to output filename to avoid overwriting")
    ap.add_argument("--fs", type=int, default=256)
    ap.add_argument("--duration_s", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_f7", type=int, default=None)
    ap.add_argument("--seed_f8", type=int, default=None)
    ap.add_argument("--coupling", type=float, default=0.02)
    ap.add_argument("--ch1", type=str, default="F7")
    ap.add_argument("--ch2", type=str, default="F8")
    args = ap.parse_args()

    # Generate two channels with explicit seeds (avoid overlap across runs)
    seed_f7 = args.seed_f7 if args.seed_f7 is not None else args.seed * 2
    seed_f8 = args.seed_f8 if args.seed_f8 is not None else args.seed * 2 + 1

    sig1 = jansen_rit_simulation(
        args.fs, args.duration_s, seed=seed_f7, coupling=args.coupling
    )
    sig2 = jansen_rit_simulation(
        args.fs, args.duration_s, seed=seed_f8, coupling=args.coupling
    )

    t = np.arange(len(sig1)) / args.fs

    df = pd.DataFrame({
        "time": t,
        args.ch1: sig1,
        args.ch2: sig2,
    })

    out_path = Path(args.out)

    # ✅ append tag to avoid overwriting
    if args.tag:
        out_path = out_path.with_name(
            f"{out_path.stem}_{args.tag}{out_path.suffix}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[OK] saved: {out_path}")
    print(f"     fs={args.fs}, duration={args.duration_s}s, seed_f7={seed_f7}, seed_f8={seed_f8}, coupling={args.coupling}")


if __name__ == "__main__":
    main()