#!/usr/bin/env python3
"""
=================================================================
BENCHMARK COMPOSITION ANALYSIS BY PHASE
=================================================================
Cross-references trades with asset inference, clusters, and metrics
to reveal WHAT the benchmark invests in during each phase, HOW
it allocates, and WHETHER it picks good algos.

Outputs:
  - phase_composition.csv          Per-phase asset class breakdown
  - phase_top_algos.csv            Top algos per phase with full profile
  - phase_cluster_allocation.csv   Cluster distribution per phase
  - phase_quality_comparison.csv   Benchmark vs universe quality metrics
  - composition_report.json        Full analysis

Usage:
  python3 benchmark_composition.py \
    --trades trades_benchmark.csv \
    --inference results/asset_inference_all.csv \
    --metrics results/metrics_all.csv \
    --clusters results/clusters.csv \
    --output results/composition/
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from notebook_paths import default_output_dir, notebook_data_path, raw_benchmark_path


# ============================================================
# PHASE DEFINITIONS
# ============================================================

PHASES = [
    ('Seed',       None,         '2023-12-25'),
    ('Transition', '2023-12-26', '2024-01-31'),
    ('Scale',      '2024-02-01', None),
]


def load_trades(path):
    df = pd.read_csv(path)
    df['dateOpen'] = pd.to_datetime(df['dateOpen'], format='mixed', utc=True).dt.tz_localize(None)
    df['dateClose'] = pd.to_datetime(df['dateClose'], format='mixed', utc=True).dt.tz_localize(None)
    df['open_day'] = df['dateOpen'].dt.normalize()
    df = df.rename(columns={'productname': 'algo'})
    df['holding_days'] = (df['dateClose'] - df['dateOpen']).dt.total_seconds() / 86400
    return df


def slice_phase(df, start, end):
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df['open_day'] >= start
    if end:
        mask &= df['open_day'] <= end
    return df[mask]


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def asset_class_breakdown(trades_phase, inference):
    """What asset classes does the benchmark invest in during this phase?"""
    merged = trades_phase.merge(
        inference[['name', 'asset_class', 'predicted_asset', 'direction', 'confidence']],
        left_on='algo', right_on='name', how='left')

    # By trade count
    by_count = merged['asset_class'].fillna('unmapped').value_counts()

    # By volume (allocation-weighted)
    by_volume = merged.groupby(merged['asset_class'].fillna('unmapped'))['volume'].sum()
    vol_total = by_volume.sum()
    by_volume_pct = (by_volume / vol_total * 100).sort_values(ascending=False)

    # By unique algos
    by_algos = merged.groupby(merged['asset_class'].fillna('unmapped'))['algo'].nunique()

    # Direction breakdown
    direction = merged.groupby(merged['direction'].fillna('unknown'))['volume'].sum()
    dir_total = direction.sum()
    dir_pct = (direction / dir_total * 100).sort_values(ascending=False)

    # Predicted asset detail (top 15)
    by_asset = merged.groupby(merged['predicted_asset'].fillna('unknown'))['volume'].sum()
    by_asset_pct = (by_asset / vol_total * 100).sort_values(ascending=False)

    return {
        'by_count': by_count.to_dict(),
        'by_volume_pct': by_volume_pct.round(2).to_dict(),
        'by_unique_algos': by_algos.to_dict(),
        'direction_pct': dir_pct.round(2).to_dict(),
        'top_assets_pct': by_asset_pct.head(15).round(2).to_dict(),
        'n_trades': len(trades_phase),
        'n_unique_algos': trades_phase['algo'].nunique(),
        'total_volume': float(vol_total),
    }


def top_algos_profile(trades_phase, inference, metrics):
    """Top algos by volume with full profile."""
    vol_by_algo = trades_phase.groupby('algo').agg(
        total_volume=('volume', 'sum'),
        n_trades=('volume', 'count'),
        avg_holding_days=('holding_days', 'mean'),
    ).sort_values('total_volume', ascending=False)

    total_vol = vol_by_algo['total_volume'].sum()
    vol_by_algo['pct_of_portfolio'] = (vol_by_algo['total_volume'] / total_vol * 100).round(2)
    vol_by_algo['cum_pct'] = vol_by_algo['pct_of_portfolio'].cumsum().round(2)

    # Merge with inference
    if inference is not None:
        inf_cols = ['name', 'asset_class', 'predicted_asset', 'direction', 'confidence']
        inf_cols = [c for c in inf_cols if c in inference.columns]
        vol_by_algo = vol_by_algo.merge(inference[inf_cols],
                                         left_index=True, right_on='name', how='left')
        vol_by_algo = vol_by_algo.set_index('name')

    # Merge with metrics
    if metrics is not None:
        met_cols = ['name', 'annualized_return_pct', 'annualized_volatility_pct',
                    'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']
        met_cols = [c for c in met_cols if c in metrics.columns]
        vol_by_algo = vol_by_algo.merge(metrics[met_cols],
                                         left_index=True, right_on='name', how='left')
        vol_by_algo = vol_by_algo.set_index('name')

    return vol_by_algo.head(30)


def cluster_allocation(trades_phase, clusters):
    """How is capital distributed across clusters?"""
    if clusters is None or 'cluster' not in clusters.columns:
        return None

    merged = trades_phase.merge(clusters[['name', 'cluster']],
                                 left_on='algo', right_on='name', how='left')

    by_cluster_vol = merged.groupby('cluster')['volume'].sum()
    total = by_cluster_vol.sum()
    by_cluster_pct = (by_cluster_vol / total * 100).sort_values(ascending=False)

    by_cluster_algos = merged.groupby('cluster')['algo'].nunique()
    by_cluster_trades = merged.groupby('cluster').size()

    result = pd.DataFrame({
        'volume_pct': by_cluster_pct.round(2),
        'n_algos': by_cluster_algos,
        'n_trades': by_cluster_trades,
    })
    return result


def quality_comparison(trades_phase, metrics):
    """Compare metrics of benchmark-selected algos vs full universe."""
    if metrics is None:
        return None

    bench_algos = set(trades_phase['algo'].unique())
    bench_met = metrics[metrics['name'].isin(bench_algos)]
    univ_met = metrics

    compare_cols = ['annualized_return_pct', 'annualized_volatility_pct',
                    'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct',
                    'sortino_ratio', 'calmar_ratio']
    compare_cols = [c for c in compare_cols if c in metrics.columns]

    rows = []
    for col in compare_cols:
        rows.append({
            'metric': col,
            'bench_mean': round(float(bench_met[col].mean()), 4),
            'bench_median': round(float(bench_met[col].median()), 4),
            'bench_p25': round(float(bench_met[col].quantile(0.25)), 4),
            'bench_p75': round(float(bench_met[col].quantile(0.75)), 4),
            'universe_mean': round(float(univ_met[col].mean()), 4),
            'universe_median': round(float(univ_met[col].median()), 4),
            'diff_mean': round(float(bench_met[col].mean() - univ_met[col].mean()), 4),
        })

    return pd.DataFrame(rows)


def concentration_analysis(trades_phase):
    """How concentrated is the portfolio?"""
    vol_by_algo = trades_phase.groupby('algo')['volume'].sum().sort_values(ascending=False)
    total = vol_by_algo.sum()
    cum_pct = (vol_by_algo / total).cumsum()

    top5_pct = float(cum_pct.iloc[min(4, len(cum_pct)-1)] * 100)
    top10_pct = float(cum_pct.iloc[min(9, len(cum_pct)-1)] * 100)
    top20_pct = float(cum_pct.iloc[min(19, len(cum_pct)-1)] * 100)

    # HHI (Herfindahl-Hirschman Index)
    shares = vol_by_algo / total
    hhi = float((shares ** 2).sum() * 10000)

    # Effective N (inverse HHI)
    eff_n = 10000 / hhi if hhi > 0 else len(vol_by_algo)

    return {
        'top_5_pct': round(top5_pct, 1),
        'top_10_pct': round(top10_pct, 1),
        'top_20_pct': round(top20_pct, 1),
        'hhi': round(hhi, 0),
        'effective_n': round(eff_n, 1),
        'total_algos': len(vol_by_algo),
    }


def holding_period_profile(trades_phase):
    """Holding period statistics."""
    h = trades_phase['holding_days']
    return {
        'mean': round(float(h.mean()), 1),
        'median': round(float(h.median()), 1),
        'p10': round(float(h.quantile(0.10)), 1),
        'p25': round(float(h.quantile(0.25)), 1),
        'p75': round(float(h.quantile(0.75)), 1),
        'p90': round(float(h.quantile(0.90)), 1),
        'pct_intraday': round(float((h < 1).mean() * 100), 1),
        'pct_under_7d': round(float((h <= 7).mean() * 100), 1),
        'pct_7_30d': round(float(((h > 7) & (h <= 30)).mean() * 100), 1),
        'pct_30_90d': round(float(((h > 30) & (h <= 90)).mean() * 100), 1),
        'pct_over_90d': round(float((h > 90).mean() * 100), 1),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark composition by phase')
    parser.add_argument('--trades', default=str(raw_benchmark_path('trades_benchmark.csv')))
    parser.add_argument('--inference', default=str(notebook_data_path('pipeline', 'asset_inference_all.csv')),
                        help='asset_inference_all.csv')
    parser.add_argument('--metrics', default=str(notebook_data_path('pipeline', 'metrics_all.csv')),
                        help='metrics_all.csv')
    parser.add_argument('--clusters', default=str(notebook_data_path('clusters', 'clusters.csv')), help='clusters.csv')
    parser.add_argument('--output', default=str(default_output_dir('composition')))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load
    print("\n[1/3] Loading data...")
    trades = load_trades(args.trades)
    inference = pd.read_csv(args.inference)
    metrics = pd.read_csv(args.metrics)
    clusters = pd.read_csv(args.clusters) if args.clusters and os.path.exists(args.clusters) else None

    print(f"  Trades: {len(trades)}, Inference: {len(inference)}, Metrics: {len(metrics)}")
    if clusters is not None:
        print(f"  Clusters: {len(clusters)}, {clusters['cluster'].nunique()} clusters")

    # Analyze each phase
    print("\n[2/3] Analyzing phases...")
    all_composition = []
    all_top_algos = []
    all_cluster_alloc = []
    all_quality = []
    report_phases = {}

    for phase_name, start, end in PHASES:
        print(f"\n{'='*60}")
        print(f"  {phase_name} ({start or 'start'} → {end or 'end'})")
        print(f"{'='*60}")

        phase_trades = slice_phase(trades, start, end)
        if len(phase_trades) == 0:
            print(f"  No trades in this phase")
            continue

        # --- Asset class breakdown ---
        ac = asset_class_breakdown(phase_trades, inference)
        report_phases[phase_name] = {'asset_breakdown': ac}

        print(f"\n  Asset Class (volume-weighted):")
        for cls, pct in ac['by_volume_pct'].items():
            n_algos = ac['by_unique_algos'].get(cls, 0)
            print(f"    {cls:<15} {pct:>6.1f}%  ({n_algos} algos)")

        print(f"\n  Direction: {ac['direction_pct']}")

        print(f"\n  Top predicted assets:")
        for asset, pct in list(ac['top_assets_pct'].items())[:10]:
            print(f"    {asset:<30} {pct:>6.1f}%")

        # Store for CSV
        for cls, pct in ac['by_volume_pct'].items():
            all_composition.append({
                'phase': phase_name,
                'asset_class': cls,
                'volume_pct': pct,
                'n_algos': ac['by_unique_algos'].get(cls, 0),
                'n_trades': ac['by_count'].get(cls, 0),
            })

        # --- Concentration ---
        conc = concentration_analysis(phase_trades)
        report_phases[phase_name]['concentration'] = conc
        print(f"\n  Concentration:")
        print(f"    Top 5: {conc['top_5_pct']}%, Top 10: {conc['top_10_pct']}%, "
              f"Top 20: {conc['top_20_pct']}%")
        print(f"    HHI: {conc['hhi']}, Effective N: {conc['effective_n']} "
              f"(of {conc['total_algos']} total)")

        # --- Holding periods ---
        hp = holding_period_profile(phase_trades)
        report_phases[phase_name]['holding_periods'] = hp
        print(f"\n  Holding periods:")
        print(f"    Median: {hp['median']}d, <7d: {hp['pct_under_7d']}%, "
              f"7-30d: {hp['pct_7_30d']}%, 30-90d: {hp['pct_30_90d']}%, "
              f">90d: {hp['pct_over_90d']}%")

        # --- Top algos ---
        top = top_algos_profile(phase_trades, inference, metrics)
        top_export = top.copy()
        top_export['phase'] = phase_name
        all_top_algos.append(top_export)

        print(f"\n  Top 10 algos:")
        display_cols = ['total_volume', 'pct_of_portfolio', 'n_trades']
        if 'asset_class' in top.columns:
            display_cols.append('asset_class')
        if 'sharpe_ratio' in top.columns:
            display_cols.append('sharpe_ratio')
        if 'annualized_return_pct' in top.columns:
            display_cols.append('annualized_return_pct')
        print(top[display_cols].head(10).to_string())

        # --- Cluster allocation ---
        if clusters is not None:
            cl = cluster_allocation(phase_trades, clusters)
            if cl is not None:
                cl_export = cl.copy()
                cl_export['phase'] = phase_name
                all_cluster_alloc.append(cl_export)
                report_phases[phase_name]['clusters'] = cl['volume_pct'].to_dict()
                print(f"\n  Cluster allocation:")
                for idx, row in cl.iterrows():
                    print(f"    Cluster {idx}: {row['volume_pct']:>5.1f}% "
                          f"({int(row['n_algos'])} algos, {int(row['n_trades'])} trades)")

        # --- Quality ---
        qual = quality_comparison(phase_trades, metrics)
        if qual is not None:
            qual['phase'] = phase_name
            all_quality.append(qual)
            report_phases[phase_name]['quality'] = qual.set_index('metric')['diff_mean'].to_dict()
            print(f"\n  Quality (bench vs universe):")
            for _, row in qual.iterrows():
                better = '✓' if row['diff_mean'] > 0 else '✗'
                print(f"    {row['metric']:<30} bench={row['bench_mean']:>8.2f}  "
                      f"univ={row['universe_mean']:>8.2f}  diff={row['diff_mean']:>+8.2f} {better}")

    # --- Cross-phase comparison ---
    print(f"\n\n{'='*60}")
    print(f"  CROSS-PHASE COMPARISON")
    print(f"{'='*60}")

    # Algo overlap
    phase_algo_sets = {}
    for phase_name, start, end in PHASES:
        phase_trades = slice_phase(trades, start, end)
        phase_algo_sets[phase_name] = set(phase_trades['algo'].unique())

    print(f"\n  Algo overlap:")
    for i, (n1, s1) in enumerate(phase_algo_sets.items()):
        for n2, s2 in list(phase_algo_sets.items())[i+1:]:
            common = s1 & s2
            pct1 = len(common) / max(len(s1), 1) * 100
            pct2 = len(common) / max(len(s2), 1) * 100
            print(f"    {n1} ∩ {n2}: {len(common)} algos "
                  f"({pct1:.0f}% of {n1}, {pct2:.0f}% of {n2})")

    # Save
    print("\n[3/3] Saving...")

    pd.DataFrame(all_composition).to_csv(
        os.path.join(args.output, 'phase_composition.csv'), index=False)

    if all_top_algos:
        pd.concat(all_top_algos).to_csv(
            os.path.join(args.output, 'phase_top_algos.csv'))

    if all_cluster_alloc:
        pd.concat(all_cluster_alloc).to_csv(
            os.path.join(args.output, 'phase_cluster_allocation.csv'))

    if all_quality:
        pd.concat(all_quality).to_csv(
            os.path.join(args.output, 'phase_quality_comparison.csv'), index=False)

    report = {
        'phases': report_phases,
        'algo_overlap': {
            f"{n1}_and_{n2}": len(s1 & s2)
            for i, (n1, s1) in enumerate(phase_algo_sets.items())
            for n2, s2 in list(phase_algo_sets.items())[i+1:]
        },
    }
    with open(os.path.join(args.output, 'composition_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Saved to {args.output}/")
    print("Done!")


if __name__ == '__main__':
    main()
