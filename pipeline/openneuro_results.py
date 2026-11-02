"""Aggregation, Table 5, Figure 5 and archive comparison for OpenNeuro runs.

The statistics follow the paper exactly:
  * absolute error  = |generator metric - Real metric| per participant and seed;
  * participant-level error = mean over the selected seeds;
  * winner = generator with the smallest participant-level error (exact ties reported);
  * "closest generator differs across measures" = more than one unique winner over
    the five measures for a participant (the paper's operational trade-off);
  * Table 5 = mean, SD, median and percentile 95% bootstrap CI over participants
    (B = 100,000 resamples, RNG seed 20260810, fresh generator per cell).
Running these functions on the archived 12,000-row table reproduces the
published Table 5 and Figure 5 counts (see `run_openneuro.py selftest`).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common import (
    ALL_MODELS, METRICS, METRIC_DISPLAY, PAPER_SUBJECTS, REPO_ROOT, eval_dirname, rel,
)

ARCHIVED_ERRORS = REPO_ROOT / "source_data" / "OpenNeuro_all_errors_by_subject_seed_model_metric.csv"
ARCHIVED_TABLE5 = REPO_ROOT / "source_data" / "Table5_OpenNeuro_absolute_error_long.csv"
BOOTSTRAP_B = 100_000
BOOTSTRAP_SEED = 20260810
COLUMNS = ["subject", "seed", "metric", "model", "real", "value", "absolute_error", "source"]


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def read_metric_means(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, index_col=0)
    if "mean" not in frame.columns:
        raise ValueError(f"no 'mean' column in {path}")
    missing = [m for m in METRICS if m not in frame.index]
    if missing:
        raise ValueError(f"missing metrics {missing} in {path}")
    values = {m: float(frame.at[m, "mean"]) for m in METRICS}
    if not np.isfinite(list(values.values())).all():
        raise ValueError(f"non-finite metric value in {path}")
    return values


def collect_run(run_dir: Path, subjects: list[str], seeds: list[int], models: list[str],
                allow_incomplete: bool = False) -> tuple[pd.DataFrame, list[str]]:
    rows, problems = [], []
    for subject in subjects:
        for seed in seeds:
            seed_dir = run_dir / "subjects" / subject / f"seed{seed}"
            try:
                real = read_metric_means(seed_dir / eval_dirname("Real", seed) / "metrics_summary.csv")
            except (FileNotFoundError, ValueError) as exc:
                problems.append(f"{subject} seed{seed} Real: {exc}")
                continue
            for model in models:
                path = seed_dir / eval_dirname(model, seed) / "metrics_summary.csv"
                try:
                    values = read_metric_means(path)
                except (FileNotFoundError, ValueError) as exc:
                    problems.append(f"{subject} seed{seed} {model}: {exc}")
                    continue
                for metric in METRICS:
                    rows.append({
                        "subject": subject, "seed": seed, "metric": metric, "model": model,
                        "real": real[metric], "value": values[metric],
                        "absolute_error": abs(values[metric] - real[metric]),
                        "source": rel(path),
                    })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if problems and not allow_incomplete:
        raise ValueError("Incomplete run outputs:\n  " + "\n  ".join(problems[:30])
                         + ("\n  ..." if len(problems) > 30 else ""))
    return frame, problems


def load_archived(subjects: list[str], seeds: list[int], models: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(ARCHIVED_ERRORS)
    frame = frame[frame["subject"].isin(subjects) & frame["seed"].isin(seeds) & frame["model"].isin(models)]
    missing = sorted(set(subjects) - set(frame["subject"]))
    if missing:
        raise ValueError(f"participants not in the archived table: {missing}")
    if not set(seeds) <= set(frame["seed"]):
        raise ValueError(f"seeds not in the archived table: {sorted(set(seeds) - set(frame['seed']))}")
    return frame.reset_index(drop=True)


def validate_errors(frame: pd.DataFrame, models: list[str]) -> None:
    if frame.duplicated(["subject", "seed", "metric", "model"]).any():
        raise ValueError("duplicate subject/seed/metric/model rows")
    numeric = frame[["real", "value", "absolute_error"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite values in the error table")
    if not np.allclose(frame["absolute_error"], (frame["value"] - frame["real"]).abs(), rtol=0, atol=1e-12):
        raise ValueError("absolute_error disagrees with |value - real|")
    if not frame.groupby(["subject", "seed", "metric"])["real"].nunique().eq(1).all():
        raise ValueError("Real reference differs across generators for the same participant/seed")
    counts = frame.groupby(["subject", "metric"])["seed"].nunique()
    if counts.nunique() != 1:
        raise ValueError("participants have different numbers of seeds; use a complete selection")
    cells = frame.groupby(["subject", "seed", "metric"])["model"].nunique()
    if not cells.eq(len(models)).all():
        raise ValueError("some participant/seed/metric cells lack a generator")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def exact_winner(values: pd.Series) -> tuple[str, int]:
    minimum = float(values.min())
    winners = values.index[values == minimum].tolist()
    return ("|".join(sorted(winners)), len(winners)) if len(winners) != 1 else (str(winners[0]), 1)


def subject_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.groupby(["subject", "metric", "model"], as_index=False)
            .agg(n_seeds=("seed", "size"), real_reference=("real", "first"),
                 mean_absolute_error=("absolute_error", "mean")))


def winners_table(summary: pd.DataFrame, value_col: str, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in summary.groupby(keys):
        values = group.set_index("model")[value_col]
        winner, tie_n = exact_winner(values)
        ordered = values.sort_values()
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update({
            "metric_label": METRIC_DISPLAY[row["metric"]],
            "unique_winner": winner if tie_n == 1 else "",
            "tie_cardinality": tie_n,
            "best_error": float(ordered.iloc[0]),
            "second_error": float(ordered.iloc[1]) if len(ordered) > 1 else float("nan"),
        })
        row["absolute_margin"] = row["second_error"] - row["best_error"]
        rows.append(row)
    return pd.DataFrame(rows)


def differing_closest(winners: pd.DataFrame, unit: list[str]) -> dict:
    """Paper's operational trade-off: the closest generator differs across the five measures."""
    tied_units = winners.loc[winners["tie_cardinality"] != 1, unit].drop_duplicates()
    clean = winners.merge(tied_units, on=unit, how="left", indicator=True)
    clean = clean[clean["_merge"] == "left_only"]
    wide = clean.pivot_table(index=unit, columns="metric", values="unique_winner", aggfunc="first")
    n_unique = wide.nunique(axis=1)
    single = wide.index[n_unique == 1].tolist()
    return {
        "units": int(len(wide) + len(tied_units)),
        "closest_generator_differs": int((n_unique > 1).sum()),
        "same_generator_for_all_measures": [("/".join(map(str, u)) if isinstance(u, tuple) else u) for u in single],
        "excluded_exact_ties": int(len(tied_units)),
    }


def bootstrap_table(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        means = {}
        for model in models:
            v = summary[(summary["metric"] == metric) & (summary["model"] == model)]
            v = v.sort_values("subject")["mean_absolute_error"].to_numpy(dtype=float)
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boot = v[rng.integers(0, len(v), size=(BOOTSTRAP_B, len(v)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            means[model] = v.mean()
            rows.append({
                "metric": metric, "metric_label": METRIC_DISPLAY[metric], "model": model,
                "n_subjects": len(v), "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else float("nan"),
                "median": float(np.median(v)), "ci_lower": lo, "ci_upper": hi,
                "bootstrap_resamples": BOOTSTRAP_B, "bootstrap_rng_seed": BOOTSTRAP_SEED,
            })
        best = min(means, key=means.get)
        for row in rows[-len(models):]:
            row["smallest_mean_within_metric"] = row["model"] == best
    return pd.DataFrame(rows)


def write_table5(table: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    table.to_csv(out_dir / "Table5_long.csv", index=False)
    wide_rows = []
    for metric in METRICS:
        row = {"Metric": METRIC_DISPLAY[metric]}
        for model in models:
            r = table[(table["metric"] == metric) & (table["model"] == model)].iloc[0]
            row[model] = f"{r['mean']:.4f} ({r['ci_lower']:.4f}, {r['ci_upper']:.4f})"
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(out_dir / "Table5_absolute_error_95CI.csv", index=False)
    lines = ["| " + " | ".join(wide.columns) + " |", "| " + " | ".join("---" for _ in wide.columns) + " |"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in wide.itertuples(index=False)]
    note = (f"\nMean (95% percentile bootstrap CI) of participant-level absolute errors; "
            f"N = {int(table['n_subjects'].iloc[0])} participants; B = {BOOTSTRAP_B:,}; "
            f"RNG seed = {BOOTSTRAP_SEED}.\n")
    (out_dir / "Table5_absolute_error_95CI.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------
def _figure_core():
    path = REPO_ROOT / "reproducibility" / "openneuro_figure5_core.py"
    spec = importlib.util.spec_from_file_location("openneuro_figure5_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_figure5(winners: pd.DataFrame, subjects: list[str], models: list[str], out_dir: Path) -> None:
    """Same drawing routine and style as the published Fig. 5, for any participant/model selection."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    core = _figure_core()
    core.configure_matplotlib()
    if subjects == core.SUBJECTS and models == core.MODELS and (winners["tie_cardinality"] == 1).all():
        core.plot_fig5(winners.rename(columns={}), out_dir)  # the exact published routine
        return

    code = {m: i for i, m in enumerate(models)}
    matrix = np.full((len(METRICS), len(subjects)), np.nan)
    for i, metric in enumerate(METRICS):
        sel = winners[winners["metric"] == metric].set_index("subject")["unique_winner"]
        for j, subject in enumerate(subjects):
            w = sel.get(subject, "")
            matrix[i, j] = code[w] if w in code else np.nan
    wide = winners.pivot(index="subject", columns="metric", values="unique_winner").reindex(subjects)
    single = wide.apply(lambda r: r.ne("").all() and r.nunique() == 1, axis=1)

    cmap = ListedColormap([core.MODEL_COLORS[m] for m in models])
    cmap.set_bad("#BBBBBB")
    fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * len(subjects)), 4.6))
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-0.5, vmax=len(models) - 0.5, aspect="auto")
    ax.set_xticks(range(len(subjects)), subjects, rotation=90)
    ax.set_yticks(range(len(METRICS)), [core.METRIC_LABEL[m] for m in METRICS])
    ax.set_xlabel("OpenNeuro participant", labelpad=8)
    ax.set_xticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    for j, subject in enumerate(subjects):
        if bool(single.loc[subject]):
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(METRICS), fill=False, edgecolor="black", linewidth=1.5))
    handles = [Patch(facecolor=core.MODEL_COLORS[m], label=m) for m in models]
    handles.append(Patch(facecolor="white", edgecolor="black", linewidth=1.5,
                         label="One generator best for all five metrics"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    core.save_figure(fig, out_dir, "Fig5a_OpenNeuro_subjectwise_winners")

    counts = (winners[winners["unique_winner"] != ""].groupby(["metric", "unique_winner"]).size()
              .unstack(fill_value=0).reindex(index=METRICS, columns=models, fill_value=0).astype(int))
    y = np.arange(len(METRICS))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    left = np.zeros(len(METRICS))
    for model in models:
        values = counts[model].to_numpy()
        bars = ax.barh(y, values, left=left, color=core.MODEL_COLORS[model], edgecolor="white",
                       linewidth=0.8, label=model)
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                color = "white" if model in {"NMM-only", "Add", "WGAN-GP"} else "black"
                ax.text(left[i] + value / 2, bar.get_y() + bar.get_height() / 2, str(int(value)),
                        ha="center", va="center", color=color, fontsize=8.6, fontweight="bold")
        left += values
    ax.set_yticks(y, [core.METRIC_LABEL[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlim(0, len(subjects))
    ax.set_xlabel(f"Number of participants (N = {len(subjects)})")
    ax.grid(axis="x", color="#D6DCE2", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    core.save_figure(fig, out_dir, "Fig5b_OpenNeuro_winner_counts")


# ---------------------------------------------------------------------------
# Comparison with the archived paper results
# ---------------------------------------------------------------------------
def compare_with_archive(frame: pd.DataFrame, out_dir: Path) -> dict | None:
    archive = pd.read_csv(ARCHIVED_ERRORS)
    keys = ["subject", "seed", "metric", "model"]
    merged = frame.merge(archive[keys + ["real", "value", "absolute_error"]], on=keys,
                         suffixes=("", "_paper"), how="inner")
    if merged.empty:
        return None
    merged["real_diff"] = (merged["real"] - merged["real_paper"]).abs()
    merged["value_diff"] = (merged["value"] - merged["value_paper"]).abs()
    merged.to_csv(out_dir / "comparison_with_paper_rows.csv", index=False)

    report = {"matched_rows": int(len(merged)),
              "real_eeg_metrics_max_abs_diff": float(merged["real_diff"].max()),
              "per_model": {}}
    mine = subject_summary(frame)
    paper = subject_summary(archive[archive["subject"].isin(frame["subject"].unique())
                                    & archive["seed"].isin(frame["seed"].unique())
                                    & archive["model"].isin(frame["model"].unique())])
    both = mine.merge(paper, on=["subject", "metric", "model"], suffixes=("", "_paper"))
    for model, g in merged.groupby("model"):
        b = both[both["model"] == model]
        report["per_model"][model] = {
            "generated_metric_max_abs_diff": float(g["value_diff"].max()),
            "generated_metric_median_abs_diff": float(g["value_diff"].median()),
            "participant_error_spearman_r": (
                float(b["mean_absolute_error"].rank().corr(b["mean_absolute_error_paper"].rank()))
                if len(b) > 2 else None),
        }
    win_mine = winners_table(mine, "mean_absolute_error", ["subject", "metric"])
    win_paper = winners_table(both.assign(mean_absolute_error=both["mean_absolute_error_paper"]),
                              "mean_absolute_error", ["subject", "metric"])
    agree = win_mine.merge(win_paper, on=["subject", "metric"], suffixes=("", "_paper"))
    report["participant_metric_winner_agreement"] = (
        f"{int((agree['unique_winner'] == agree['unique_winner_paper']).sum())}/{len(agree)}")
    report["interpretation"] = (
        "Real-EEG metrics and NMM-only outputs are deterministic and should agree with the paper to "
        "numerical precision on any machine. Diffusion-only, Add and WGAN-GP are trained neural networks: "
        "on the authors' Mac Studio (Apple silicon, MPS) with the pinned packages they reproduced the "
        "archived values exactly, but on other hardware or library versions their seed-level values can "
        "differ; then compare participant-level errors and winners instead.")
    (out_dir / "comparison_with_paper.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_results(frame: pd.DataFrame, subjects: list[str], seeds: list[int], models: list[str],
                  out_dir: Path, compare: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_errors(frame, models)
    frame = frame.sort_values(["subject", "seed", "metric", "model"]).reset_index(drop=True)
    frame.to_csv(out_dir / "errors_by_subject_seed_model_metric.csv", index=False)

    summary = subject_summary(frame)
    summary.to_csv(out_dir / "subject_mean_errors.csv", index=False)
    subj_win = winners_table(summary, "mean_absolute_error", ["subject", "metric"])
    seed_win = winners_table(frame.rename(columns={"absolute_error": "err"}), "err", ["subject", "seed", "metric"])
    subj_win.to_csv(out_dir / "subject_level_winners.csv", index=False)
    seed_win.to_csv(out_dir / "seed_level_winners.csv", index=False)

    counts = (subj_win[subj_win["unique_winner"] != ""].groupby("unique_winner").size()
              .reindex(models, fill_value=0).astype(int).to_dict())
    summary_json = {
        "participants": len(subjects), "seeds": seeds, "generators": models,
        "participant_level": differing_closest(subj_win, ["subject"]),
        "seed_level": differing_closest(seed_win, ["subject", "seed"]),
        "winner_cells_by_generator": counts,
        "definition": "A participant (or participant-seed run) is counted when the generator closest to "
                      "real EEG differs across the five measures (the paper's operational trade-off).",
    }
    table = bootstrap_table(summary, models)
    write_table5(table, models, out_dir)
    plot_figure5(subj_win, subjects, models, out_dir / "figures")
    if compare:
        summary_json["comparison_with_paper"] = compare_with_archive(frame, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    return summary_json


def verify_against_published(out_dir: Path) -> list[str]:
    """Used by `selftest`: compare a full 30 x 20 x 4 rebuild with the published numbers."""
    failures = []
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    p, s = summary["participant_level"], summary["seed_level"]
    if (p["closest_generator_differs"], p["units"]) != (28, 30):
        failures.append(f"participant-level count {p['closest_generator_differs']}/{p['units']} != 28/30")
    if sorted(p["same_generator_for_all_measures"]) != ["sub-320", "sub-323"]:
        failures.append(f"participants with one generator: {p['same_generator_for_all_measures']}")
    if (s["closest_generator_differs"], s["units"]) != (526, 600):
        failures.append(f"seed-level count {s['closest_generator_differs']}/{s['units']} != 526/600")
    if summary["winner_cells_by_generator"] != {"NMM-only": 11, "Diffusion-only": 31, "Add": 18, "WGAN-GP": 90}:
        failures.append(f"winner counts {summary['winner_cells_by_generator']}")
    mine = pd.read_csv(out_dir / "Table5_long.csv")
    paper = pd.read_csv(ARCHIVED_TABLE5)
    m = mine.merge(paper, on=["metric", "model"], suffixes=("", "_paper"))
    for col in ["mean", "sd", "median", "ci_lower", "ci_upper"]:
        diff = float((m[col] - m[f"{col}_paper"]).abs().max())
        if not diff < 1e-9:
            failures.append(f"Table 5 {col} differs from the paper by {diff:g}")
    if not (m["smallest_mean_within_metric"] == m["smallest_mean_within_metric_paper"]).all():
        failures.append("Table 5 best-generator flags differ")
    return failures
