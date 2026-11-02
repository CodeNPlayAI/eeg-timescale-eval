#!/usr/bin/env python3
"""Original OpenNeuro Fig. 5 statistical and plotting functions.
Copied verbatim from the verified historical builder; the original full run-level
script is intentionally excluded because its Fig. 4 computes seed 0 only.
Only display labels below are updated; metric calculations remain unchanged.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
CORE_MODELS = ["NMM-only", "Diffusion-only", "Add"]
MODELS = [*CORE_MODELS, "WGAN-GP"]
METRICS = ["psd_loglog_slope_all_2_40hz", "dfa_x_alpha", "acf_x_abs_0_1s", "acf_x_abs_0_5s", "acf_x_abs_1_0s"]
METRIC_LABEL = {"psd_loglog_slope_all_2_40hz": "PSD slope\n(2–40 Hz)", "dfa_x_alpha": r"DFA $\alpha$", "acf_x_abs_0_1s": "ACF ≤ 0.1 s", "acf_x_abs_0_5s": "ACF ≤ 0.5 s", "acf_x_abs_1_0s": "ACF ≤ 1.0 s"}
SUBJECTS = ["sub-012", "sub-050", "sub-062", "sub-122", "sub-140", "sub-144", "sub-152", "sub-157", "sub-180", "sub-305", "sub-308", "sub-320", "sub-323", "sub-331", "sub-349", "sub-402", "sub-407", "sub-412", "sub-425", "sub-449", "sub-470", "sub-518", "sub-529", "sub-542", "sub-566", "sub-573", "sub-574", "sub-575", "sub-597", "sub-603"]
MODEL_COLORS = {"NMM-only": "#0072B2", "Diffusion-only": "#E69F00", "Add": "#009E73", "WGAN-GP": "#CC79A7"}

def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def exact_winner(group: pd.DataFrame, value_col: str) -> tuple[str, int]:
    values = group.set_index("model")[value_col]
    minimum = float(values.min())
    winners = values.index[values == minimum].tolist()
    if len(winners) != 1:
        return "|".join(sorted(winners)), len(winners)
    return str(winners[0]), 1


def subject_level_summary(all_errors: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = all_errors[all_errors["model"].isin(models)].copy()
    summary = (
        subset.groupby(["subject", "metric", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "size"),
            real_reference=("real", "first"),
            mean_absolute_error=("absolute_error", "mean"),
        )
    )
    if not summary["n_seeds"].eq(20).all():
        raise ValueError("Every subject/metric/model cell must have exactly 20 seeds")
    winners = []
    for (subject, metric), group in summary.groupby(["subject", "metric"]):
        winner, tie_n = exact_winner(group, "mean_absolute_error")
        ordered = group.sort_values("mean_absolute_error")
        winners.append(
            {
                "subject": subject,
                "metric": metric,
                "metric_label": METRIC_LABEL[metric].replace("$", ""),
                "unique_winner": winner if tie_n == 1 else "",
                "tie_cardinality": tie_n,
                "best_mean_absolute_error": float(ordered.iloc[0]["mean_absolute_error"]),
                "second_mean_absolute_error": float(ordered.iloc[1]["mean_absolute_error"]),
                "absolute_margin": float(ordered.iloc[1]["mean_absolute_error"] - ordered.iloc[0]["mean_absolute_error"]),
            }
        )
    winner_df = pd.DataFrame(winners)
    if len(winner_df) != 150:
        raise ValueError(f"Expected 150 subject/metric winner cells, found {len(winner_df)}")
    return summary, winner_df


def seed_level_winners(all_errors: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    subset = all_errors[all_errors["model"].isin(models)].copy()
    rows = []
    for (subject, seed, metric), group in subset.groupby(["subject", "seed", "metric"]):
        winner, tie_n = exact_winner(group, "absolute_error")
        rows.append(
            {
                "subject": subject,
                "seed": int(seed),
                "metric": metric,
                "unique_winner": winner if tie_n == 1 else "",
                "tie_cardinality": tie_n,
            }
        )
    return pd.DataFrame(rows)


def tradeoff_stats(winners: pd.DataFrame, unit: str) -> tuple[int, int, list[str]]:
    if (winners["tie_cardinality"] != 1).any():
        ties = winners[winners["tie_cardinality"] != 1]
        raise ValueError(f"Exact winner ties found; trade-off classification is locked to unique winners:\n{ties.head()}")
    if unit == "subject":
        wide = winners.pivot(index="subject", columns="metric", values="unique_winner").reindex(SUBJECTS)
        n_unique = wide.nunique(axis=1)
        tradeoff = int((n_unique > 1).sum())
        nontradeoff = wide.index[n_unique == 1].tolist()
        return tradeoff, len(wide), nontradeoff
    if unit == "seed":
        wide = winners.pivot(index=["subject", "seed"], columns="metric", values="unique_winner")
        n_unique = wide.nunique(axis=1)
        tradeoff = int((n_unique > 1).sum())
        return tradeoff, len(wide), []
    raise ValueError(unit)


def plot_fig5(winners: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ordered = winners.copy()
    ordered["subject"] = pd.Categorical(ordered["subject"], SUBJECTS, ordered=True)
    ordered["metric"] = pd.Categorical(ordered["metric"], METRICS, ordered=True)
    ordered = ordered.sort_values(["metric", "subject"]).reset_index(drop=True)

    model_code = {model: idx for idx, model in enumerate(MODELS)}
    matrix = np.empty((len(METRICS), len(SUBJECTS)), dtype=int)
    for i, metric in enumerate(METRICS):
        by_subject = ordered.loc[ordered["metric"] == metric].set_index("subject")["unique_winner"]
        for j, subject in enumerate(SUBJECTS):
            matrix[i, j] = model_code[str(by_subject.loc[subject])]

    wide = ordered.pivot(index="subject", columns="metric", values="unique_winner").reindex(SUBJECTS)
    single_model = wide.nunique(axis=1).eq(1)

    cmap = ListedColormap([MODEL_COLORS[m] for m in MODELS])
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(SUBJECTS)), SUBJECTS, rotation=90)
    ax.set_yticks(range(len(METRICS)), [METRIC_LABEL[m] for m in METRICS])
    ax.set_xlabel("OpenNeuro participant", labelpad=8)
    ax.set_xticks(np.arange(-0.5, len(SUBJECTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    for j, subject in enumerate(SUBJECTS):
        if bool(single_model.loc[subject]):
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(METRICS), fill=False, edgecolor="black", linewidth=1.5))
    handles = [Patch(facecolor=MODEL_COLORS[m], label=m) for m in MODELS]
    handles.append(Patch(facecolor="white", edgecolor="black", linewidth=1.5, label="One generator best for all five metrics"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)
    save_figure(fig, out_dir, "Fig5a_OpenNeuro_subjectwise_winners")

    counts = (
        ordered.groupby(["metric", "unique_winner"], observed=False).size().unstack(fill_value=0)
        .reindex(index=METRICS, columns=MODELS, fill_value=0).astype(int)
    )
    y = np.arange(len(METRICS))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    left = np.zeros(len(METRICS), dtype=float)
    for model in MODELS:
        values = counts[model].to_numpy(dtype=int)
        bars = ax.barh(y, values, left=left, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.8, label=model)
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value <= 0:
                continue
            text_color = "white" if model in {"NMM-only", "Add", "WGAN-GP"} else "black"
            ax.text(left[i] + value / 2, bar.get_y() + bar.get_height() / 2, str(int(value)),
                    ha="center", va="center", color=text_color, fontsize=8.6, fontweight="bold")
        left += values
    ax.set_yticks(y, [METRIC_LABEL[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlabel("Number of participants (N = 30)")
    ax.grid(axis="x", color="#D6DCE2", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    save_figure(fig, out_dir, "Fig5b_OpenNeuro_winner_counts")
    return counts

