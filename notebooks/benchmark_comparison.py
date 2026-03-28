#!/usr/bin/env python3
"""
=================================================================
BENCHMARK COMPARISON & DEEP ANALYSIS
=================================================================
Reads reconstruction_daily.csv and benchmark data to produce:
  - Rolling correlation/tracking error analysis
  - Monthly/quarterly/yearly return comparison
  - Drawdown comparison
  - Regime analysis (when does each method work best/worst)
  - Attribution: which algos contribute most to tracking error
  - Full comparison report

Usage:
  python3 benchmark_comparison.py \
    --daily results/reconstruction_daily.csv \
    --monthly benchmark_monthly_returns.csv \
    --yearly benchmark_yearly_returns.csv \
    --trades trades_benchmark.csv \
    --algos ./algorithms/ \
    --output results/comparison/
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_paths import default_output_dir, notebook_data_path, raw_benchmark_path
try:
    from algo_pipeline import load_algorithm_csv, trim_dead_tail
except:
    pass


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_reconstruction(daily_path):
    """Load reconstruction_daily.csv"""
    df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    methods = [c for c in df.columns if c != 'actual']
    print(f"  Methods: {methods}")
    print(f"  Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Actual: {df['actual'].dropna().iloc[0]:.2f} -> {df['actual'].dropna().iloc[-1]:.2f}")
    return df, methods


# ============================================================
# 2. ROLLING ANALYSIS
# ============================================================

def rolling_analysis(df, methods, windows=[30, 60, 90, 180]):
    """
    Compute rolling correlation, tracking error, and beta
    between each method and actual.
    """
    actual_ret = np.log(df['actual'] / df['actual'].shift(1)).dropna()
    results = {}

    for method in methods:
        method_ret = np.log(df[method] / df[method].shift(1)).dropna()
        aligned = pd.DataFrame({'actual': actual_ret, 'method': method_ret}).dropna()

        method_results = {}
        for w in windows:
            if len(aligned) < w:
                continue

            # Rolling correlation
            roll_corr = aligned['actual'].rolling(w).corr(aligned['method'])

            # Rolling tracking error (annualized)
            roll_diff = aligned['actual'] - aligned['method']
            roll_te = roll_diff.rolling(w).std() * np.sqrt(252)

            # Rolling beta
            roll_cov = aligned['actual'].rolling(w).cov(aligned['method'])
            roll_var = aligned['actual'].rolling(w).var()
            roll_beta = roll_cov / roll_var.replace(0, np.nan)

            method_results[f'corr_{w}d'] = roll_corr.dropna()
            method_results[f'te_{w}d'] = roll_te.dropna()
            method_results[f'beta_{w}d'] = roll_beta.dropna()

        results[method] = method_results

    return results


# ============================================================
# 3. MULTI-FREQUENCY RETURN COMPARISON
# ============================================================

def multi_frequency_comparison(df, methods):
    """Compare returns at daily, weekly, monthly, quarterly, yearly."""

    comparisons = {}

    freqs = {
        'weekly': 'W',
        'monthly': 'ME',
        'quarterly': 'QE',
        'yearly': 'YE',
    }

    for freq_name, freq_code in freqs.items():
        resampled = df.resample(freq_code).last().dropna(how='all')

        freq_results = {}
        for method in methods:
            aligned = resampled[['actual', method]].dropna()
            if len(aligned) < 3:
                continue

            a_ret = np.log(aligned['actual'] / aligned['actual'].shift(1)).dropna()
            m_ret = np.log(aligned[method] / aligned[method].shift(1)).dropna()

            both = pd.DataFrame({'actual': a_ret, 'recon': m_ret}).dropna()
            if len(both) < 3:
                continue

            corr = both['actual'].corr(both['recon'])
            diff = both['actual'] - both['recon']
            mae = diff.abs().mean()

            # Count how many periods have the same direction
            same_dir = (np.sign(both['actual']) == np.sign(both['recon'])).mean()

            freq_results[method] = {
                'correlation': round(float(corr), 4),
                'mae': round(float(mae), 6),
                'same_direction_pct': round(float(same_dir * 100), 1),
                'n_periods': len(both),
            }

            # Store the actual vs recon returns for export
            freq_results[f'{method}_returns'] = both

        comparisons[freq_name] = freq_results

    return comparisons


# ============================================================
# 4. DRAWDOWN COMPARISON
# ============================================================

def drawdown_comparison(df, methods):
    """Compare drawdown profiles between actual and reconstructions."""
    results = {}

    for col in ['actual'] + methods:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        # Compute drawdown from cumulative max
        cum_max = series.cummax()
        dd = (series - cum_max) / cum_max

        # Max drawdown
        max_dd = dd.min()
        max_dd_date = dd.idxmin()

        # Recovery time from max DD
        dd_end_idx = dd.loc[max_dd_date:].gt(-0.001)
        recovery_date = dd_end_idx[dd_end_idx].index[0] if dd_end_idx.any() else None
        recovery_days = (recovery_date - max_dd_date).days if recovery_date else None

        # Avg drawdown
        avg_dd = dd[dd < 0].mean() if (dd < 0).any() else 0

        # Time in drawdown
        pct_in_dd = (dd < -0.001).mean()

        # Drawdown curve for export
        results[col] = {
            'max_drawdown_pct': round(float(max_dd * 100), 2),
            'max_dd_date': str(max_dd_date.date()) if max_dd_date else None,
            'recovery_days': recovery_days,
            'avg_drawdown_pct': round(float(avg_dd * 100), 2),
            'time_in_drawdown_pct': round(float(pct_in_dd * 100), 1),
            'dd_series': dd,
        }

    # Drawdown correlation: do they draw down at the same time?
    dd_corrs = {}
    actual_dd = results.get('actual', {}).get('dd_series')
    if actual_dd is not None:
        for method in methods:
            method_dd = results.get(method, {}).get('dd_series')
            if method_dd is not None:
                aligned = pd.DataFrame({'actual': actual_dd, 'recon': method_dd}).dropna()
                if len(aligned) > 30:
                    dd_corrs[method] = round(float(aligned['actual'].corr(aligned['recon'])), 4)

    return results, dd_corrs


# ============================================================
# 5. REGIME ANALYSIS
# ============================================================

def regime_analysis(df, methods):
    """
    Analyze when each method performs best/worst relative to actual.
    Split by market regimes: trending up, trending down, flat, volatile.
    """
    actual = df['actual'].dropna()
    if len(actual) < 60:
        return {}

    # Define regimes using rolling 60-day return and volatility
    actual_ret = np.log(actual / actual.shift(1)).dropna()
    roll_ret_60 = actual_ret.rolling(60).mean() * 252  # annualized
    roll_vol_60 = actual_ret.rolling(60).std() * np.sqrt(252)

    regimes = pd.Series('flat', index=actual_ret.index)
    regimes[roll_ret_60 > 0.05] = 'trending_up'
    regimes[roll_ret_60 < -0.05] = 'trending_down'
    regimes[(roll_vol_60 > roll_vol_60.quantile(0.75))] = 'high_vol'

    results = {}
    for method in methods:
        method_ret = np.log(df[method] / df[method].shift(1)).dropna()
        aligned = pd.DataFrame({
            'actual': actual_ret,
            'method': method_ret,
            'regime': regimes
        }).dropna()

        aligned['diff'] = aligned['actual'] - aligned['method']

        regime_stats = {}
        for regime in ['trending_up', 'trending_down', 'flat', 'high_vol']:
            subset = aligned[aligned['regime'] == regime]
            if len(subset) < 10:
                continue
            regime_stats[regime] = {
                'n_days': len(subset),
                'correlation': round(float(subset['actual'].corr(subset['method'])), 4),
                'tracking_error': round(float(subset['diff'].std() * np.sqrt(252)), 4),
                'mean_diff': round(float(subset['diff'].mean() * 252), 4),
            }

        results[method] = regime_stats

    return results


# ============================================================
# 6. MONTHLY RETURN TABLE
# ============================================================

def build_monthly_return_table(df, best_method):
    """Build a side-by-side monthly return table: actual vs best reconstruction."""
    monthly = df[['actual', best_method]].resample('ME').last().dropna(how='all')

    a_ret = monthly['actual'].pct_change().dropna()
    r_ret = monthly[best_method].pct_change().dropna()

    both = pd.DataFrame({
        'month': a_ret.index.strftime('%Y-%m'),
        'actual_pct': (a_ret * 100).round(4).values,
        'recon_pct': (r_ret.reindex(a_ret.index) * 100).round(4).values,
    }).dropna()

    both['diff_pct'] = (both['recon_pct'] - both['actual_pct']).round(4)
    both['abs_diff'] = both['diff_pct'].abs().round(4)

    return both


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Deep benchmark comparison analysis')
    parser.add_argument('--daily', default=str(notebook_data_path('reconstruction', 'reconstruction_daily.csv')),
                        help='reconstruction_daily.csv')
    parser.add_argument('--monthly', default=str(raw_benchmark_path('benchmark_monthly_returns.csv')),
                        help='benchmark_monthly_returns.csv')
    parser.add_argument('--yearly', default=str(raw_benchmark_path('benchmark_yearly_returns.csv')),
                        help='benchmark_yearly_returns.csv')
    parser.add_argument('--output', default=str(default_output_dir('comparison')), help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Load
    print("\n[1/6] Loading reconstruction data...")
    df, methods = load_reconstruction(args.daily)

    # 2. Rolling analysis
    print("\n[2/6] Rolling analysis...")
    rolling = rolling_analysis(df, methods)

    # Export rolling data
    for method, data in rolling.items():
        roll_df = pd.DataFrame({k: v for k, v in data.items()})
        roll_df.to_csv(os.path.join(args.output, f'rolling_{method}.csv'))
    print(f"  Saved rolling CSVs for {len(methods)} methods")

    # Summary: average rolling corr at each window
    print("\n  Average rolling correlation:")
    for method in methods:
        data = rolling.get(method, {})
        for key, series in data.items():
            if key.startswith('corr_'):
                print(f"    {method} {key}: {series.mean():.4f} (min={series.min():.4f}, max={series.max():.4f})")

    # 3. Multi-frequency comparison
    print("\n[3/6] Multi-frequency comparison...")
    freq_comp = multi_frequency_comparison(df, methods)

    freq_summary = []
    for freq_name, freq_data in freq_comp.items():
        for method, stats in freq_data.items():
            if isinstance(stats, dict) and 'correlation' in stats:
                row = {'frequency': freq_name, 'method': method}
                row.update(stats)
                freq_summary.append(row)
                print(f"  {freq_name:>10} | {method:<20} | corr={stats['correlation']:.4f} | "
                      f"same_dir={stats['same_direction_pct']:.0f}% | n={stats['n_periods']}")

    pd.DataFrame(freq_summary).to_csv(
        os.path.join(args.output, 'frequency_comparison.csv'), index=False)

    # 4. Drawdown comparison
    print("\n[4/6] Drawdown comparison...")
    dd_results, dd_corrs = drawdown_comparison(df, methods)

    dd_summary = []
    for col, stats in dd_results.items():
        row = {'series': col}
        row.update({k: v for k, v in stats.items() if k != 'dd_series'})
        dd_summary.append(row)
        print(f"  {col:<20} MaxDD={stats['max_drawdown_pct']:>7.2f}% | "
              f"AvgDD={stats['avg_drawdown_pct']:>7.2f}% | "
              f"Time in DD={stats['time_in_drawdown_pct']:.0f}%")

    if dd_corrs:
        print(f"\n  Drawdown correlations with actual:")
        for method, corr in dd_corrs.items():
            print(f"    {method}: {corr:.4f}")

    pd.DataFrame(dd_summary).to_csv(
        os.path.join(args.output, 'drawdown_comparison.csv'), index=False)

    # Export drawdown curves
    dd_curves = pd.DataFrame({col: stats['dd_series'] for col, stats in dd_results.items()
                              if 'dd_series' in stats})
    dd_curves.to_csv(os.path.join(args.output, 'drawdown_curves.csv'))

    # 5. Regime analysis
    print("\n[5/6] Regime analysis...")
    regimes = regime_analysis(df, methods)

    regime_rows = []
    for method, regime_data in regimes.items():
        for regime, stats in regime_data.items():
            row = {'method': method, 'regime': regime}
            row.update(stats)
            regime_rows.append(row)
            print(f"  {method:<20} {regime:<15} corr={stats['correlation']:>7.4f} "
                  f"TE={stats['tracking_error']:>7.4f} days={stats['n_days']}")

    pd.DataFrame(regime_rows).to_csv(
        os.path.join(args.output, 'regime_analysis.csv'), index=False)

    # 6. Monthly return table (best method)
    print("\n[6/6] Monthly return comparison...")

    # Find best method by monthly correlation
    best_method = max(
        [(m, s.get('correlation', 0)) for freq_data in freq_comp.values()
         for m, s in freq_data.items() if isinstance(s, dict) and 'correlation' in s
         and freq_data is freq_comp.get('monthly', {})],
        key=lambda x: x[1],
        default=('volume_weighted', 0)
    )[0]

    print(f"  Best monthly method: {best_method}")
    monthly_table = build_monthly_return_table(df, best_method)
    monthly_table.to_csv(os.path.join(args.output, 'monthly_returns_comparison.csv'), index=False)

    print(f"\n  Monthly return comparison ({best_method} vs actual):")
    print(f"  {'Month':<10} {'Actual%':>10} {'Recon%':>10} {'Diff%':>10}")
    print(f"  {'-' * 42}")
    for _, row in monthly_table.iterrows():
        print(f"  {row['month']:<10} {row['actual_pct']:>10.2f} {row['recon_pct']:>10.2f} "
              f"{row['diff_pct']:>+10.2f}")

    avg_abs_diff = monthly_table['abs_diff'].mean()
    print(f"\n  Avg monthly |diff|: {avg_abs_diff:.4f}%")
    print(f"  Max monthly |diff|: {monthly_table['abs_diff'].max():.4f}%")
    print(f"  Months within 0.5%: {(monthly_table['abs_diff'] < 0.5).sum()}/{len(monthly_table)}")

    # --- Full report ---
    report = {
        'best_method_monthly': best_method,
        'frequency_comparison': freq_summary,
        'drawdown_comparison': [r for r in dd_summary],
        'drawdown_correlations': dd_corrs,
        'regime_analysis': regime_rows,
        'monthly_avg_abs_diff': round(float(avg_abs_diff), 4),
        'monthly_max_abs_diff': round(float(monthly_table['abs_diff'].max()), 4),
    }
    with open(os.path.join(args.output, 'comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  All saved to: {args.output}/")
    print("Done!")


if __name__ == '__main__':
    main()
