#!/usr/bin/env python3
"""Figure 5 (OpenNeuro, N=30): regenerate participant-level winners from raw archived errors.

Scientific quantities and the plotting routine originate in the historical
`reproducibility/openneuro_figure5_core.py`. This
wrapper uses only the original `subject_level_summary`, `seed_level_winners`,
`tradeoff_stats`, `configure_matplotlib` and `plot_fig5` functions; it never
calls the original script's `main()` (which also plots an obsolete seed-0 Fig. 4).

No EEG model training or new metric evaluation occurs in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source_data' / 'OpenNeuro_all_errors_by_subject_seed_model_metric.csv'
FROZEN_WINNERS = ROOT / 'source_data' / 'subject_level_winners_extended4.csv'
EXPECTED_COUNTS = {'NMM-only': 11, 'Diffusion-only': 31, 'Add': 18, 'WGAN-GP': 90}


def plotting_module():
    import openneuro_figure5_core as module
    return module


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-csv', type=Path, default=SOURCE)
    ap.add_argument('--figures-dir', type=Path, default=ROOT / 'figures/Figure5')
    ap.add_argument('--source-data-dir', type=Path, default=ROOT / 'outputs/validation/Figure5')
    ap.add_argument('--validate-only', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    original = plotting_module()
    df = pd.read_csv(args.input_csv)
    expected_cols = {'subject', 'seed', 'metric', 'model', 'real', 'value', 'absolute_error'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(expected_cols - set(df.columns))}')
    if len(df) != 30 * 20 * 5 * 4:
        raise ValueError(f'Expected 12000 rows, got {len(df)}')
    if set(df['subject']) != set(original.SUBJECTS) or set(df['metric']) != set(original.METRICS) or set(df['model']) != set(original.MODELS):
        raise ValueError('Unexpected subject, metric, or model identities')
    if set(pd.to_numeric(df['seed']).astype(int)) != set(range(20)):
        raise ValueError('Expected seeds 0..19')
    if df.duplicated(['subject', 'seed', 'metric', 'model']).any():
        raise ValueError('Duplicate subject/seed/metric/model rows')
    numeric = df[['real', 'value', 'absolute_error']].astype(float).to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite Real/value/absolute-error')
    if not np.allclose(df['absolute_error'].to_numpy(dtype=float), np.abs(df['value'] - df['real']), rtol=0, atol=1e-12):
        raise ValueError('Absolute errors disagree with |value-real|')
    reference_nunique = df.groupby(['subject','seed','metric'])['real'].nunique()
    if not reference_nunique.eq(1).all():
        raise ValueError('Real reference differs across generators')

    summary, winners = original.subject_level_summary(df, original.MODELS)
    run_winners = original.seed_level_winners(df, original.MODELS)
    subject_tradeoff, nsub, nontradeoff = original.tradeoff_stats(winners, 'subject')
    seed_tradeoff, ntrials, _ = original.tradeoff_stats(run_winners, 'seed')
    counts = winners['unique_winner'].value_counts().reindex(original.MODELS, fill_value=0).astype(int).to_dict()
    if (nsub, subject_tradeoff, ntrials, seed_tradeoff) != (30, 28, 600, 526):
        raise ValueError(f'Locked tradeoff mismatch: {subject_tradeoff}/{nsub}, {seed_tradeoff}/{ntrials}')
    if set(nontradeoff) != {'sub-320', 'sub-323'}:
        raise ValueError(f'Nontradeoff participant identity mismatch: {nontradeoff}')
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'Locked winner counts mismatch: {counts}')

    frozen = pd.read_csv(FROZEN_WINNERS).sort_values(['subject','metric']).reset_index(drop=True)
    actual = winners.sort_values(['subject','metric']).reset_index(drop=True)
    if not actual[['subject','metric','unique_winner','tie_cardinality']].equals(frozen[['subject','metric','unique_winner','tie_cardinality']]):
        raise ValueError('Archived subject-level winner identities differ')
    for col in ['best_mean_absolute_error', 'second_mean_absolute_error', 'absolute_margin']:
        if not np.allclose(actual[col], frozen[col], rtol=0, atol=1e-12):
            raise ValueError(f'Archived subject-level errors differ: {col}')

    if args.validate_only:
        print('[OK] Figure 5 archived numeric validation: 12000 rows, 28/30, 526/600, winner counts', counts)
        return

    original.configure_matplotlib()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.source_data_dir.mkdir(parents=True, exist_ok=True)
    plotted_counts = original.plot_fig5(winners, args.figures_dir)
    # Recomputed winner counts are checked above against EXPECTED_COUNTS.
    if plotted_counts.sum(axis=0).astype(int).to_dict() != EXPECTED_COUNTS:
        raise ValueError('Figure 5 panel counts disagree with verified totals')
    winners.to_csv(args.source_data_dir / 'Figure5_subject_level_winners.csv', index=False)
    summary.to_csv(args.source_data_dir / 'Figure5_subject_level_mean_absolute_errors.csv', index=False)
    plotted_counts.reset_index().to_csv(args.source_data_dir / 'Figure5_winner_counts_by_metric.csv', index=False)
    report = {
        'figure': 5,
        'description': 'OpenNeuro participant-level winner structure including WGAN-GP',
        'input_source': 'source_data/OpenNeuro_all_errors_by_subject_seed_model_metric.csv',
        'input_rows': len(df), 'subjects': nsub, 'seeds_per_subject': 20,
        'metric_count': 5, 'model_count': 4,
        'subject_tradeoff_count': subject_tradeoff,
        'seed_tradeoff_count': seed_tradeoff,
        'nontradeoff_subjects': nontradeoff,
        'winner_cells': counts,
        'archived_winners_recomputed': True,
        'archived_count_table_reproduced': True,
        'plot_source': 'reproducibility/openneuro_figure5_core.py::plot_fig5',
        'acf_display': 'mean absolute ACF over positive lags up to indicated lag',
    }
    (args.source_data_dir/'Figure5_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[OK] Figure 5 real source data validated, archived counts unchanged:', counts)
    print('[OK] Figure 5 PNG/PDF/SVG panels:', args.figures_dir)


if __name__ == '__main__':
    main()
