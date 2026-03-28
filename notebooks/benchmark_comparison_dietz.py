#!/usr/bin/env python3
"""
=================================================================
BENCHMARK COMPARISON — DIETZ (WITH CASHFLOWS)
=================================================================
Deep comparison of Dietz reconstruction methods vs actual benchmark.
Includes comparison against the no-cashflow reconstruction for
quantifying the improvement from cashflow handling.

Usage:
  python3 benchmark_comparison_dietz.py \
    --dietz_daily results/dietz/dietz_reconstruction_daily.csv \
    --nocf_daily results/reconstruction_daily.csv \
    --trades trades_benchmark.csv \
    --monthly benchmark_monthly_returns.csv \
    --output results/dietz/comparison/
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from notebook_paths import default_output_dir, notebook_data_path, raw_benchmark_path


def load_actual(monthly_path):
    """Load actual benchmark."""
    df = pd.read_csv(monthly_path)
    records = []
    for _, row in df.iterrows():
        start = pd.Timestamp(row['month'] + '-01')
        end = start + pd.offsets.MonthEnd(0)
        records.append({'date': start, 'equity': row['start_equity']})
        records.append({'date': end, 'equity': row['end_equity']})
    bench = pd.DataFrame(records).drop_duplicates('date').sort_values('date').set_index('date')
    daily_idx = pd.date_range(bench.index[0], bench.index[-1], freq='D')
    return bench.reindex(daily_idx).interpolate(method='time')['equity']


def compute_stats(actual, recon, method_name):
    """Full comparison stats between actual and a reconstruction."""
    common = actual.dropna().index.intersection(recon.dropna().index)
    if len(common) < 30:
        return {'method': method_name, 'error': 'insufficient_data', 'n_days': len(common)}

    a = actual.loc[common]
    r = recon.loc[common]
    a_norm = a / a.iloc[0] * 100
    r_norm = r / r.iloc[0] * 100

    a_ret = np.log(a_norm / a_norm.shift(1)).dropna()
    r_ret = np.log(r_norm / r_norm.shift(1)).dropna()
    both = pd.DataFrame({'actual': a_ret, 'recon': r_ret}).dropna()

    if len(both) < 10:
        return {'method': method_name, 'error': 'insufficient_returns'}

    diff = both['actual'] - both['recon']

    # Multi-frequency correlations
    freq_corrs = {}
    for freq_name, freq_code in [('weekly', 'W'), ('monthly', 'ME'),
                                   ('quarterly', 'QE')]:
        a_f = a_norm.resample(freq_code).last()
        r_f = r_norm.resample(freq_code).last()
        a_fr = np.log(a_f / a_f.shift(1)).dropna()
        r_fr = np.log(r_f / r_f.shift(1)).dropna()
        aligned = pd.DataFrame({'a': a_fr, 'r': r_fr}).dropna()
        if len(aligned) > 3:
            freq_corrs[freq_name] = round(float(aligned['a'].corr(aligned['r'])), 6)
            freq_corrs[f'{freq_name}_same_dir'] = round(
                float((np.sign(aligned['a']) == np.sign(aligned['r'])).mean() * 100), 1)

    # Drawdown comparison
    a_dd = (a_norm - a_norm.cummax()) / a_norm.cummax()
    r_dd = (r_norm - r_norm.cummax()) / r_norm.cummax()
    dd_corr = a_dd.corr(r_dd) if len(a_dd) > 30 else 0

    return {
        'method': method_name,
        'n_days': len(common),
        'correlation_daily': round(float(both['actual'].corr(both['recon'])), 6),
        'tracking_error_annual': round(float(diff.std() * np.sqrt(252)), 6),
        'rmse_index': round(float(np.sqrt(((r_norm - a_norm)**2).mean())), 4),
        'max_deviation': round(float((r_norm - a_norm).abs().max()), 4),
        'final_actual': round(float(a_norm.iloc[-1]), 2),
        'final_recon': round(float(r_norm.iloc[-1]), 2),
        'final_diff_pct': round(float((r_norm.iloc[-1] - a_norm.iloc[-1]) / a_norm.iloc[-1] * 100), 2),
        'drawdown_correlation': round(float(dd_corr), 4),
        **freq_corrs,
    }


def period_analysis(actual, recon, method_name):
    """
    Analyze accuracy across three fund phases:
    Phase 1 (Seed): 2020-06 to 2023-12-25
    Phase 2 (Transition): 2023-12-26 to 2024-01-31
    Phase 3 (Scale): 2024-02-01 to 2024-12-31
    """
    boundaries = [
        ('Seed (2020-2023)', None, '2023-12-25'),
        ('Transition (Dec23-Jan24)', '2023-12-26', '2024-01-31'),
        ('Scale (Feb-Dec 2024)', '2024-02-01', None),
    ]

    results = []
    for phase_name, start, end in boundaries:
        mask_a = pd.Series(True, index=actual.index)
        mask_r = pd.Series(True, index=recon.index)
        if start:
            mask_a &= actual.index >= start
            mask_r &= recon.index >= start
        if end:
            mask_a &= actual.index <= end
            mask_r &= recon.index <= end

        a_phase = actual[mask_a].dropna()
        r_phase = recon[mask_r].dropna()

        if len(a_phase) < 10 or len(r_phase) < 10:
            results.append({'phase': phase_name, 'method': method_name,
                          'error': 'insufficient_data'})
            continue

        stats = compute_stats(a_phase, r_phase, f"{method_name}|{phase_name}")
        stats['phase'] = phase_name
        stats['method'] = method_name
        results.append(stats)

    return results


def build_monthly_comparison(actual, methods_dict):
    """Build side-by-side monthly return comparison for all methods."""
    a_monthly = actual.resample('ME').last()
    a_ret = a_monthly.pct_change().dropna() * 100

    records = []
    for month_end, a_r in a_ret.items():
        row = {
            'month': month_end.strftime('%Y-%m'),
            'actual_pct': round(float(a_r), 4),
        }
        for name, series in methods_dict.items():
            s_monthly = series.resample('ME').last()
            s_ret = s_monthly.pct_change().dropna() * 100
            if month_end in s_ret.index:
                row[f'{name}_pct'] = round(float(s_ret.loc[month_end]), 4)
                row[f'{name}_diff'] = round(float(s_ret.loc[month_end] - a_r), 4)
        records.append(row)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description='Compare Dietz reconstruction methods')
    parser.add_argument('--dietz_daily', default=str(notebook_data_path('dietz', 'dietz_reconstruction_daily.csv')),
                       help='dietz_reconstruction_daily.csv')
    parser.add_argument('--nocf_daily', default=str(notebook_data_path('reconstruction', 'reconstruction_daily.csv')),
                       help='reconstruction_daily.csv (no-cashflow, for comparison)')
    parser.add_argument('--monthly', default=str(raw_benchmark_path('benchmark_monthly_returns.csv')),
                       help='benchmark_monthly_returns.csv')
    parser.add_argument('--output', default=str(default_output_dir('dietz', 'comparison')))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load actual
    print("\n[1/4] Loading data...")
    actual = load_actual(args.monthly)
    print(f"  Actual: {actual.index[0].date()} → {actual.index[-1].date()}")

    # Load Dietz reconstruction
    dietz = pd.read_csv(args.dietz_daily, index_col=0, parse_dates=True)
    dietz_methods = [c for c in dietz.columns if c != 'actual']
    print(f"  Dietz methods: {dietz_methods}")

    # Load no-cashflow reconstruction (optional)
    nocf_methods = []
    if args.nocf_daily and os.path.exists(args.nocf_daily):
        nocf = pd.read_csv(args.nocf_daily, index_col=0, parse_dates=True)
        nocf_methods = [c for c in nocf.columns if c != 'actual']
        print(f"  No-cashflow methods: {nocf_methods}")

    # 2. Overall comparison
    print("\n[2/4] Overall comparison...")
    all_results = []

    for method in dietz_methods:
        stats = compute_stats(actual, dietz[method], f"[Dietz] {method}")
        all_results.append(stats)
        if 'error' not in stats:
            print(f"\n  [Dietz] {method}:")
            print(f"    Daily corr:   {stats['correlation_daily']:.6f}")
            print(f"    Monthly corr: {stats.get('monthly', 0):.6f}")
            print(f"    Tracking err: {stats['tracking_error_annual']:.6f}")
            print(f"    Final diff:   {stats['final_diff_pct']:+.2f}%")

    for method in nocf_methods:
        stats = compute_stats(actual, nocf[method], f"[NoCF] {method}")
        all_results.append(stats)
        if 'error' not in stats:
            print(f"\n  [NoCF] {method}:")
            print(f"    Daily corr:   {stats['correlation_daily']:.6f}")
            print(f"    Monthly corr: {stats.get('monthly', 0):.6f}")
            print(f"    Tracking err: {stats['tracking_error_annual']:.6f}")
            print(f"    Final diff:   {stats['final_diff_pct']:+.2f}%")

    # 3. Per-phase analysis
    print("\n[3/4] Phase-level analysis...")
    phase_results = []

    for method in dietz_methods:
        phase_stats = period_analysis(actual, dietz[method], f"[Dietz] {method}")
        phase_results.extend(phase_stats)

    for method in nocf_methods:
        phase_stats = period_analysis(actual, nocf[method], f"[NoCF] {method}")
        phase_results.extend(phase_stats)

    phase_df = pd.DataFrame(phase_results)
    if len(phase_df) > 0:
        for _, row in phase_df.iterrows():
            if 'error' not in row or pd.isna(row.get('error')):
                print(f"  {row['method']:<45} {row.get('phase',''):<25} "
                      f"corr={row.get('correlation_daily', 0):>8.4f} "
                      f"TE={row.get('tracking_error_annual', 0):>8.4f}")

    # 4. Monthly comparison
    print("\n[4/4] Monthly return comparison...")
    all_series = {}
    for method in dietz_methods:
        all_series[f"dietz_{method}"] = dietz[method].dropna()
    for method in nocf_methods:
        all_series[f"nocf_{method}"] = nocf[method].dropna()

    monthly_comp = build_monthly_comparison(actual, all_series)

    # Save
    print("\n  Saving...")
    pd.DataFrame(all_results).to_csv(
        os.path.join(args.output, 'dietz_comparison_results.csv'), index=False)
    phase_df.to_csv(
        os.path.join(args.output, 'dietz_phase_analysis.csv'), index=False)
    monthly_comp.to_csv(
        os.path.join(args.output, 'dietz_monthly_comparison.csv'), index=False)

    # Report
    best_dietz = min([r for r in all_results if 'Dietz' in r.get('method', '')
                      and 'error' not in r],
                     key=lambda x: x.get('tracking_error_annual', 999),
                     default={'method': 'none'})
    best_nocf = min([r for r in all_results if 'NoCF' in r.get('method', '')
                     and 'error' not in r],
                    key=lambda x: x.get('tracking_error_annual', 999),
                    default={'method': 'none'})

    improvement = {}
    if best_dietz.get('tracking_error_annual') and best_nocf.get('tracking_error_annual'):
        te_dietz = best_dietz['tracking_error_annual']
        te_nocf = best_nocf['tracking_error_annual']
        improvement = {
            'te_reduction_pct': round((1 - te_dietz / te_nocf) * 100, 2) if te_nocf > 0 else 0,
            'final_diff_improvement': round(
                abs(best_nocf.get('final_diff_pct', 0)) - abs(best_dietz.get('final_diff_pct', 0)), 2),
        }

    report = {
        'overall_comparison': all_results,
        'phase_analysis': phase_results,
        'best_dietz_method': best_dietz,
        'best_nocf_method': best_nocf,
        'improvement_from_cashflow_handling': improvement,
    }
    with open(os.path.join(args.output, 'dietz_comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"  BEST DIETZ: {best_dietz.get('method', '?')}")
    if 'tracking_error_annual' in best_dietz:
        print(f"    TE: {best_dietz['tracking_error_annual']:.6f} | "
              f"Final diff: {best_dietz.get('final_diff_pct', '?'):+.2f}%")
    print(f"\n  BEST NO-CASHFLOW: {best_nocf.get('method', '?')}")
    if 'tracking_error_annual' in best_nocf:
        print(f"    TE: {best_nocf['tracking_error_annual']:.6f} | "
              f"Final diff: {best_nocf.get('final_diff_pct', '?'):+.2f}%")
    if improvement:
        print(f"\n  IMPROVEMENT FROM CASHFLOW HANDLING:")
        print(f"    Tracking error reduction: {improvement['te_reduction_pct']:+.2f}%")
        print(f"    Final diff improvement:   {improvement['final_diff_improvement']:+.2f} pp")
    print(f"{'='*60}")
    print("Done!")


if __name__ == '__main__':
    main()
