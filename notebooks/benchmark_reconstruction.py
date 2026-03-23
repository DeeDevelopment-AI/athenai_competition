#!/usr/bin/env python3
"""
=================================================================
BENCHMARK RECONSTRUCTION ENGINE v2
=================================================================
Reconstructs a fund-of-algorithms index by tracking accumulated
units per asset, exactly as a portfolio manager would in a spreadsheet.

Methodology (no cashflow assumptions needed):
  1. Build holdings matrix: track cumulative units held per asset per day.
     For asset A on day D:
       - If openDate trade exists: accumulated += volume
       - If closeDate trade exists: accumulated -= volume
       - Otherwise: carry forward previous day's value
  2. Calculate weights per day:
     - Volume-weighted: units_A / sum(all_units)
     - Value-weighted:  (units_A * price_A) / sum(units_i * price_i)
  3. Get daily LOG returns per asset from algo CSV close prices.
  4. Portfolio log return = sum(weight_i * log_return_i)
  5. Cumulate log returns -> index level.

Uses log returns throughout for mathematical correctness (additive).
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from algo_pipeline import load_algorithm_csv, trim_dead_tail
except ImportError:
    def load_algorithm_csv(fp):
        try:
            df = pd.read_csv(fp, parse_dates=['datetime'])
            return df.set_index('datetime').resample('D').last().dropna()
        except:
            return None
    def trim_dead_tail(s, **kw):
        return s, {'was_trimmed': False}


def build_holdings_matrix(trades_path):
    df = pd.read_csv(trades_path)
    df['dateOpen'] = pd.to_datetime(df['dateOpen'], format='mixed', utc=True).dt.tz_localize(None)
    df['dateClose'] = pd.to_datetime(df['dateClose'], format='mixed', utc=True).dt.tz_localize(None)
    df['open_day'] = df['dateOpen'].dt.normalize()
    df['close_day'] = df['dateClose'].dt.normalize()
    df = df.rename(columns={'productname': 'algo'})

    print(f"  Trades: {len(df)}, Unique algos: {df['algo'].nunique()}")

    min_date = df['open_day'].min()
    max_date = pd.Timestamp('2024-12-31')
    dates = pd.date_range(min_date, max_date, freq='D')
    algos = sorted(df['algo'].unique())

    opens = df.groupby(['open_day', 'algo'])['volume'].sum().reset_index()
    opens = opens.rename(columns={'open_day': 'date', 'volume': 'add'})
    closes = df.groupby(['close_day', 'algo'])['volume'].sum().reset_index()
    closes = closes.rename(columns={'close_day': 'date', 'volume': 'sub'})

    net_change = pd.DataFrame(0.0, index=dates, columns=algos)
    for _, row in opens.iterrows():
        d, a, v = row['date'], row['algo'], row['add']
        if d in net_change.index and a in net_change.columns:
            net_change.loc[d, a] += v
    for _, row in closes.iterrows():
        d, a, v = row['date'], row['algo'], row['sub']
        if d in net_change.index and a in net_change.columns:
            net_change.loc[d, a] -= v

    holdings = net_change.cumsum().clip(lower=0)
    total_units = holdings.sum(axis=1)
    n_active = (holdings > 0).sum(axis=1)
    print(f"  Holdings: {holdings.shape[0]} days x {holdings.shape[1]} algos")
    print(f"  Avg active/day: {n_active.mean():.1f}, Max: {n_active.max()}")
    return holdings, df


def load_algo_closes(algo_dir, needed_algos):
    closes_dict = {}
    missing = []
    for algo_name in needed_algos:
        filepath = os.path.join(algo_dir, f"{algo_name}.csv")
        if not os.path.exists(filepath):
            missing.append(algo_name)
            continue
        daily = load_algorithm_csv(filepath)
        if daily is None or len(daily) < 2:
            missing.append(algo_name)
            continue
        closes_dict[algo_name] = daily['close']

    print(f"  Loaded: {len(closes_dict)}/{len(needed_algos)} algos")
    if missing:
        print(f"  Missing: {len(missing)} ({', '.join(missing[:10])}{'...' if len(missing)>10 else ''})")
    df_closes = pd.DataFrame(closes_dict)
    return df_closes, missing


def compute_log_returns(df_closes):
    return np.log(df_closes / df_closes.shift(1))


def reconstruct_volume_weighted(holdings, log_returns, start_value=100):
    total_units = holdings.sum(axis=1).replace(0, np.nan)
    weights = holdings.div(total_units, axis=0).fillna(0)
    common_dates = weights.index.intersection(log_returns.index)
    common_algos = weights.columns.intersection(log_returns.columns)
    w = weights.loc[common_dates, common_algos]
    r = log_returns.loc[common_dates, common_algos].fillna(0)
    port_log_ret = (w * r).sum(axis=1)
    cum_log_ret = port_log_ret.cumsum()
    index_values = start_value * np.exp(cum_log_ret)
    has_data = log_returns.loc[common_dates, common_algos].notna()
    coverage = (w * has_data).sum(axis=1)
    return index_values, port_log_ret, coverage, weights


def reconstruct_value_weighted(holdings, df_closes, log_returns, start_value=100):
    common_dates = holdings.index.intersection(df_closes.index).intersection(log_returns.index)
    common_algos = holdings.columns.intersection(df_closes.columns).intersection(log_returns.columns)
    h = holdings.loc[common_dates, common_algos]
    p = df_closes.loc[common_dates, common_algos]
    r = log_returns.loc[common_dates, common_algos].fillna(0)
    position_values = h * p
    total_value = position_values.sum(axis=1).replace(0, np.nan)
    weights = position_values.div(total_value, axis=0).fillna(0)
    port_log_ret = (weights * r).sum(axis=1)
    cum_log_ret = port_log_ret.cumsum()
    index_values = start_value * np.exp(cum_log_ret)
    has_data = df_closes.loc[common_dates, common_algos].notna()
    coverage = (weights * has_data).sum(axis=1)
    return index_values, port_log_ret, coverage, weights


def reconstruct_equal_weighted(holdings, log_returns, start_value=100):
    common_dates = holdings.index.intersection(log_returns.index)
    common_algos = holdings.columns.intersection(log_returns.columns)
    h = holdings.loc[common_dates, common_algos]
    r = log_returns.loc[common_dates, common_algos].fillna(0)
    active = (h > 0).astype(float)
    n_active = active.sum(axis=1).replace(0, np.nan)
    weights = active.div(n_active, axis=0).fillna(0)
    port_log_ret = (weights * r).sum(axis=1)
    cum_log_ret = port_log_ret.cumsum()
    index_values = start_value * np.exp(cum_log_ret)
    coverage = (weights * r.notna()).sum(axis=1)
    return index_values, port_log_ret, coverage, weights


def reconstruct_inv_vol(holdings, log_returns, lookback=60, start_value=100):
    common_dates = holdings.index.intersection(log_returns.index)
    common_algos = holdings.columns.intersection(log_returns.columns)
    h = holdings.loc[common_dates, common_algos]
    r = log_returns.loc[common_dates, common_algos].fillna(0)
    rolling_vol = r.rolling(lookback, min_periods=20).std()
    active = (h > 0).astype(float)
    inv_vol = (1.0 / rolling_vol.replace(0, np.nan)).fillna(0) * active
    total_inv_vol = inv_vol.sum(axis=1).replace(0, np.nan)
    weights = inv_vol.div(total_inv_vol, axis=0).fillna(0)
    no_vol = total_inv_vol.isna()
    if no_vol.any():
        n_active = active.loc[no_vol].sum(axis=1).replace(0, 1)
        eq_w = active.loc[no_vol].div(n_active, axis=0)
        weights.loc[no_vol] = eq_w
    port_log_ret = (weights * r).sum(axis=1)
    cum_log_ret = port_log_ret.cumsum()
    index_values = start_value * np.exp(cum_log_ret)
    coverage = (weights * r.notna()).sum(axis=1)
    return index_values, port_log_ret, coverage, weights


def load_actual_benchmark(monthly_path):
    df = pd.read_csv(monthly_path)
    records = []
    for _, row in df.iterrows():
        month = row['month']
        start_date = pd.Timestamp(month + '-01')
        end_date = start_date + pd.offsets.MonthEnd(0)
        records.append({'date': start_date, 'equity': row['start_equity']})
        records.append({'date': end_date, 'equity': row['end_equity']})
    bench_df = pd.DataFrame(records).drop_duplicates('date').sort_values('date').set_index('date')
    daily_idx = pd.date_range(bench_df.index[0], bench_df.index[-1], freq='D')
    bench_daily = bench_df.reindex(daily_idx).interpolate(method='time')
    bench_daily.columns = ['actual']
    print(f"  Actual: {bench_daily.index[0].date()} -> {bench_daily.index[-1].date()}")
    print(f"  Start: {bench_daily['actual'].iloc[0]:.2f}, End: {bench_daily['actual'].iloc[-1]:.2f}")
    print(f"  Total return: {(bench_daily['actual'].iloc[-1]/bench_daily['actual'].iloc[0]-1)*100:.2f}%")
    return bench_daily


def compare_with_actual(actual_series, recon_series, method_name):
    common = actual_series.dropna().index.intersection(recon_series.dropna().index)
    if len(common) < 30:
        return {'method': method_name, 'error': 'insufficient_overlap', 'n_days': len(common)}
    a = actual_series.loc[common]
    r = recon_series.loc[common]
    a_norm = a / a.iloc[0] * 100
    r_norm = r / r.iloc[0] * 100
    a_logret = np.log(a_norm / a_norm.shift(1)).dropna()
    r_logret = np.log(r_norm / r_norm.shift(1)).dropna()
    common_ret = pd.DataFrame({'actual': a_logret, 'recon': r_logret}).dropna()
    if len(common_ret) < 10:
        return {'method': method_name, 'error': 'insufficient_returns'}
    diff = common_ret['actual'] - common_ret['recon']
    tracking_error = diff.std() * np.sqrt(252)
    corr = common_ret['actual'].corr(common_ret['recon'])
    ss_res = (diff ** 2).sum()
    ss_tot = ((common_ret['actual'] - common_ret['actual'].mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    final_diff = r_norm.iloc[-1] - a_norm.iloc[-1]
    rmse = np.sqrt(((r_norm - a_norm) ** 2).mean())
    max_dev = (r_norm - a_norm).abs().max()
    a_m = a_norm.resample('ME').last()
    r_m = r_norm.resample('ME').last()
    monthly_corr = a_m.corr(r_m) if len(a_m) > 3 else 0
    return {
        'method': method_name,
        'n_days': len(common),
        'correlation_daily': round(float(corr), 6),
        'r_squared': round(float(r_squared), 6),
        'tracking_error_annual': round(float(tracking_error), 6),
        'rmse_index': round(float(rmse), 4),
        'max_deviation': round(float(max_dev), 4),
        'final_actual': round(float(a_norm.iloc[-1]), 2),
        'final_recon': round(float(r_norm.iloc[-1]), 2),
        'final_diff_pct': round(float(final_diff / a_norm.iloc[-1] * 100), 2),
        'monthly_correlation': round(float(monthly_corr), 6),
    }


def main():
    parser = argparse.ArgumentParser(description='Reconstruct fund-of-algorithms benchmark v2')
    parser.add_argument('--trades', required=True, help='trades_benchmark.csv')
    parser.add_argument('--monthly', required=True, help='benchmark_monthly_returns.csv')
    parser.add_argument('--algos', required=True, help='Directory with algorithm CSVs')
    parser.add_argument('--output', default='results', help='Output directory')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n[1/5] Building holdings matrix (cumulative units)...")
    holdings, trades_df = build_holdings_matrix(args.trades)

    print("\n[2/5] Loading algorithm close prices...")
    needed_algos = holdings.columns.tolist()
    df_closes, missing = load_algo_closes(args.algos, needed_algos)

    print("\n[3/5] Computing log returns...")
    log_returns = compute_log_returns(df_closes)
    print(f"  Log returns: {log_returns.shape[0]} days x {log_returns.shape[1]} algos")

    print("\n[4/5] Reconstructing index (4 methods)...")
    methods = {}
    method_weights = {}
    start_val = 100

    print("\n  Method 1: Volume-Weighted...")
    idx, ret, cov, w = reconstruct_volume_weighted(holdings, log_returns, start_val)
    methods['volume_weighted'] = idx
    method_weights['volume_weighted'] = w
    if len(idx) > 0:
        print(f"    Days: {len(idx)}, Final: {idx.iloc[-1]:.2f}, Avg coverage: {cov.mean():.2%}")
    else:
        print(f"    No overlapping data")

    print("\n  Method 2: Value-Weighted (mark-to-market)...")
    idx, ret, cov, w = reconstruct_value_weighted(holdings, df_closes, log_returns, start_val)
    methods['value_weighted'] = idx
    method_weights['value_weighted'] = w
    if len(idx) > 0:
        print(f"    Days: {len(idx)}, Final: {idx.iloc[-1]:.2f}, Avg coverage: {cov.mean():.2%}")
    else:
        print(f"    No overlapping data")

    print("\n  Method 3: Equal-Weighted...")
    idx, ret, cov, w = reconstruct_equal_weighted(holdings, log_returns, start_val)
    methods['equal_weighted'] = idx
    method_weights['equal_weighted'] = w
    if len(idx) > 0:
        print(f"    Days: {len(idx)}, Final: {idx.iloc[-1]:.2f}")
    else:
        print(f"    No overlapping data")

    print("\n  Method 4: Inverse-Volatility...")
    idx, ret, cov, w = reconstruct_inv_vol(holdings, log_returns, lookback=60, start_value=start_val)
    methods['inv_volatility'] = idx
    method_weights['inv_volatility'] = w
    if len(idx) > 0:
        print(f"    Days: {len(idx)}, Final: {idx.iloc[-1]:.2f}")
    else:
        print(f"    No overlapping data")

    print("\n[5/5] Comparing with actual benchmark...")
    actual = load_actual_benchmark(args.monthly)

    results = []
    for method_name, index_series in methods.items():
        stats = compare_with_actual(actual['actual'], index_series, method_name)
        results.append(stats)
        if 'error' not in stats:
            print(f"\n  {method_name}:")
            print(f"    Daily correlation:  {stats['correlation_daily']:.6f}")
            print(f"    R2:                 {stats['r_squared']:.6f}")
            print(f"    Tracking error:     {stats['tracking_error_annual']:.6f}")
            print(f"    RMSE (index):       {stats['rmse_index']:.4f}")
            print(f"    Final diff:         {stats['final_diff_pct']:+.2f}%")
            print(f"    Monthly corr:       {stats['monthly_correlation']:.6f}")

    print("\n  Saving...")
    pd.DataFrame(results).to_csv(os.path.join(args.output, 'reconstruction_results.csv'), index=False)
    daily_df = pd.DataFrame(methods)
    daily_df['actual'] = actual['actual']
    daily_df.to_csv(os.path.join(args.output, 'reconstruction_daily.csv'))

    last_day = holdings.index[-1]
    top_vol = holdings.loc[last_day].sort_values(ascending=False).head(20)
    best = min([r for r in results if 'error' not in r],
               key=lambda x: x.get('tracking_error_annual', 999), default={'method': 'none'})
    best_name = best.get('method', 'volume_weighted')
    weights_snapshot = {}
    if best_name in method_weights:
        last_w = method_weights[best_name]
        if last_day in last_w.index:
            lw = last_w.loc[last_day].sort_values(ascending=False).head(20)
            weights_snapshot = {k: round(float(v), 4) for k, v in lw.items() if v > 0.001}

    report = {
        'comparison': results, 'best_method': best,
        'total_algos': len(needed_algos), 'loaded_algos': len(df_closes.columns),
        'missing_algos': missing,
        'top_holdings_units': {k: float(v) for k, v in top_vol.items() if v > 0},
        'best_method_weights': weights_snapshot,
    }
    with open(os.path.join(args.output, 'reconstruction_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  BEST METHOD: {best.get('method', '?')}")
    if 'tracking_error_annual' in best:
        print(f"  Tracking error:     {best['tracking_error_annual']:.6f}")
        print(f"  Daily correlation:  {best['correlation_daily']:.6f}")
        print(f"  R2:                 {best['r_squared']:.6f}")
        print(f"  Final diff:         {best['final_diff_pct']:+.2f}%")
    print(f"{'='*60}")
    print("Done!")


if __name__ == '__main__':
    main()