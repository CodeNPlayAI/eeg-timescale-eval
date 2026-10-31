#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Table 3 (E2) regeneration - self-contained version
#
# Does NOT depend on the CLI options of
# aggregate_e2_cross_term_multiseed.py.
#
# Usage:
#   ./run_table3_e2_v2.sh
#
# or:
#   ./run_table3_e2_v2.sh /path/to/self_eeg_aligned_run
#
# Run from the repository root.
# ============================================================

REPO_ROOT="$(pwd)"
DEFAULT_RUN_ROOT="${REPO_ROOT}/runs/self_eeg_aligned_ddpm_T1000_E800_B8_20seeds_20260811_200259"
RUN_ROOT="${1:-$DEFAULT_RUN_ROOT}"

if [[ ! -d "$RUN_ROOT" ]]; then
    echo "[ERROR] aligned self-EEG run not found:"
    echo "  $RUN_ROOT"
    exit 1
fi

export RUN_ROOT

python - <<'PY'
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd

run_root = Path(os.environ["RUN_ROOT"])
summary_dir = run_root / "multiseed_summary"
summary_dir.mkdir(parents=True, exist_ok=True)

expected_seeds = list(range(20))
lags = ["0.1s", "0.5s", "1.0s"]
n_boot = 10_000
bootstrap_seed = 42

rows = []

for seed in expected_seeds:
    path = run_root / f"seed{seed}" / "e2_cross_term" / "e2_cross_term_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing E2 summary for seed {seed}: {path}"
        )

    df = pd.read_csv(path)

    # Locked protocol expects exactly one summary row per seed.
    if len(df) != 1:
        raise ValueError(
            f"Seed {seed}: expected exactly 1 row in {path.name}, found {len(df)}"
        )

    row = df.iloc[0].to_dict()
    row["seed"] = seed
    rows.append(row)

per_seed = pd.DataFrame(rows)

# Validate required XCF columns.
required_cols = [f"XCF_at_{lag}_mean" for lag in lags]
missing_cols = [c for c in required_cols if c not in per_seed.columns]
if missing_cols:
    raise ValueError(
        "Missing required E2 XCF columns: " + ", ".join(missing_cols)
    )

# Save a canonical 20-seed long source table.
per_seed_out = summary_dir / "e2_cross_term_summary.csv"
per_seed.to_csv(per_seed_out, index=False)

def bootstrap_ci_mean(x, n_boot=10_000, random_state=42):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )

agg_rows = []
for lag in lags:
    col = f"XCF_at_{lag}_mean"
    x = per_seed[col].to_numpy(dtype=float)

    if not np.isfinite(x).all():
        bad = np.where(~np.isfinite(x))[0].tolist()
        raise ValueError(f"{col}: non-finite values at row indices {bad}")

    if len(x) != 20:
        raise ValueError(f"{col}: expected 20 seed values, found {len(x)}")

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    lo, hi = bootstrap_ci_mean(
        x,
        n_boot=n_boot,
        random_state=bootstrap_seed,
    )

    agg_rows.append(
        {
            "Metric": f"XCF @ {lag}",
            "N": 20,
            "Mean": mean,
            "SD": sd,
            "CI95_lower": lo,
            "CI95_upper": hi,
        }
    )

agg = pd.DataFrame(agg_rows)

csv_out = summary_dir / "e2_cross_term_aggregated.csv"
agg.to_csv(csv_out, index=False)

md_lines = [
    "| Metric | Mean | SD | 95% CI |",
    "|---|---:|---:|---:|",
]
for _, r in agg.iterrows():
    md_lines.append(
        f"| {r['Metric']} | {r['Mean']:.4f} | {r['SD']:.4f} | "
        f"[{r['CI95_lower']:.4f}, {r['CI95_upper']:.4f}] |"
    )

md_out = summary_dir / "e2_cross_term_aggregated.md"
md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

meta = {
    "analysis": "Table 3 / E2 cross-term verification",
    "dataset": "self-recorded EEG",
    "run_root": str(run_root),
    "seeds": expected_seeds,
    "n_seed_runs": 20,
    "lag_labels": lags,
    "summary": "mean, sample SD, percentile bootstrap 95% CI",
    "bootstrap_resamples": n_boot,
    "bootstrap_rng_seed": bootstrap_seed,
}
meta_out = summary_dir / "e2_aggregation_meta.json"
meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("=" * 72)
print("Table 3 (E2): aligned self-EEG, 20 seeds")
print("=" * 72)
print(agg.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print()
print("[OK]", per_seed_out)
print("[OK]", csv_out)
print("[OK]", md_out)
print("[OK]", meta_out)
print()
print("Paper-ready preview:")
print(md_out.read_text(encoding="utf-8"))
PY
