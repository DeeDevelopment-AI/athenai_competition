#!/usr/bin/env python3
"""
=================================================================
BENCHMARK RECONSTRUCTION WITH CASHFLOWS (Dietz Methods)
=================================================================
Reconstructs the fund-of-algorithms benchmark accounting for
cashflows, using methods from Christopherson et al. Chapter 5:

  Method 1: True Time-Weighted Return (TWR)
    - Subperiods at each cashflow event
    - r_sub = EV_t / BV_t - 1  (eq 5.2)
    - Link: (1+r_1)(1+r_2)...(1+r_T) - 1

  Method 2: Linked Modified Dietz (monthly)
    - Monthly subperiods
    - Modified Dietz per month (eq 5.5):
        r = (EV - BV - ΣC_k) / (BV + ΣW_k·C_k)
      where W_k = (TD - D_k) / TD
    - Link monthly returns

  Method 3: Modified Dietz (daily granularity)
    - Apply Modified Dietz formula to rolling daily windows
    - Each day: r = (EV - BV - CF) / (BV + 0.5 * CF)

  Method 4: Volume-Weighted + Cashflow Adjustment
    - Same holdings matrix as no-cashflow version
    - BUT scale returns by the fraction of capital actually invested
    - Accounts for money "in transit" during cashflow events

Does NOT modify existing no-cashflow scripts. This is a separate
analysis for comparison.

Usage:
  python3 benchmark_reconstruction_dietz.py \
    --trades trades_benchmark.csv \
    --monthly benchmark_monthly_returns.csv \
    --algos ./algorithms/ \
    --output results/dietz/
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
from notebook_paths import default_output_dir, raw_algos_path, raw_benchmark_path
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


# ============================================================
# 1. BUILD FUND TIMELINE + CASHFLOWS (GIPS)
# ============================================================

def build_fund_timeline(trades_path):
    """
    Build daily fund timeline with GIPS cashflow detection.
    Returns daily DataFrame with equity_EOD, equity_normalized,
    strategy returns, and detected cashflows.
    """
    df = pd.read_csv(trades_path)
    df['dateOpen'] = pd.to_datetime(df['dateOpen'], format='mixed', utc=True).dt.tz_localize(None)
    df['dateClose'] = pd.to_datetime(df['dateClose'], format='mixed', utc=True).dt.tz_localize(None)
    df['obs_date'] = df['dateOpen'].dt.normalize()
    df = df.rename(columns={'productname': 'algo'})

    daily = df.groupby('obs_date').agg(
        equity_EOD=('equity_EOD', 'last'),
        equity_normalized=('equity_normalized', 'last'),
        AUM=('AUM', 'last'),
        total_invested=('total_invested_amount_EOD', 'last'),
    ).sort_index()

    # GIPS cashflow detection
    daily['ret_strategy'] = daily['equity_normalized'].pct_change()
    daily['expected_equity_EOD'] = daily['equity_EOD'].shift(1) * (1 + daily['ret_strategy'])
    daily['cashflow'] = daily['equity_EOD'] - daily['expected_equity_EOD']

    print(f"  Fund timeline: {len(daily)} days")
    cf = daily['cashflow'].dropna()
    print(f"  Cashflows detected: {(cf.abs() > 100).sum()} events "
          f"(IN: {cf[cf > 100].sum():,.0f}, OUT: {cf[cf < -100].sum():,.0f})")

    return daily, df


# ============================================================
# 2. BUILD HOLDINGS MATRIX + LOAD ALGO RETURNS
# ============================================================

def build_holdings_matrix(trades_df):
    """Same cumulative unit tracking as no-cashflow version."""
    min_date = trades_df['obs_date'].min()
    max_date = pd.Timestamp('2024-12-31')
    dates = pd.date_range(min_date, max_date, freq='D')
    algos = sorted(trades_df['algo'].unique())

    opens = trades_df.groupby(['obs_date', 'algo'])['volume'].sum().reset_index()
    opens = opens.rename(columns={'obs_date': 'date', 'volume': 'add'})

    trades_df['close_day'] = trades_df['dateClose'].dt.normalize()
    closes = trades_df.groupby(['close_day', 'algo'])['volume'].sum().reset_index()
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
    return holdings


def load_algo_closes(algo_dir, needed_algos):
    """Load daily closes for constituent algos."""
    closes_dict = {}
    missing = []
    for name in needed_algos:
        fp = os.path.join(algo_dir, f"{name}.csv")
        if not os.path.exists(fp):
            missing.append(name)
            continue
        daily = load_algorithm_csv(fp)
        if daily is None or len(daily) < 2:
            missing.append(name)
            continue
        closes_dict[name] = daily['close']

    print(f"  Loaded: {len(closes_dict)}/{len(needed_algos)} algos, missing: {len(missing)}")
    return pd.DataFrame(closes_dict), missing


# ============================================================
# 3. RECONSTRUCTION METHODS
# ============================================================

def _portfolio_daily_return(holdings, log_returns, date, method='volume'):
    """
    Compute single-day portfolio return using holdings and algo returns.
    Returns log return for the portfolio on this date.
    """
    if date not in holdings.index or date not in log_returns.index:
        return 0.0

    h = holdings.loc[date]
    r = log_returns.loc[date]

    # Common algos
    common = h.index.intersection(r.index)
    h = h[common]
    r = r[common].fillna(0)

    total = h.sum()
    if total <= 0:
        return 0.0

    weights = h / total
    return float((weights * r).sum())


def method_twr(fund_daily, holdings, log_returns, start_value=100):
    """
    Method 1: True Time-Weighted Return (TWR)

    From Chapter 5, eq 5.2:
      r = (1 + r_1)(1 + r_2)...(1 + r_T) - 1

    Create subperiods at each significant cashflow.
    Within each subperiod, compute portfolio return from holdings × algo returns.
    Link subperiod returns.

    This neutralizes cashflow effects because each subperiod starts
    AFTER the cashflow has been absorbed.
    """
    # Identify cashflow boundaries (material cashflows > 1% of equity)
    cf = fund_daily['cashflow'].dropna()
    equity = fund_daily['equity_EOD']
    cf_pct = (cf / equity.shift(1)).abs()

    # Subperiod boundaries: days with cashflows > 1% of equity
    threshold = 0.01
    boundary_dates = cf_pct[cf_pct > threshold].index.tolist()

    # Add start and end
    all_dates = sorted(set(
        [holdings.index[0]] + boundary_dates + [holdings.index[-1]]
    ))

    common_dates = holdings.index.intersection(log_returns.index)

    # Compute daily portfolio returns for all days
    port_daily_logret = pd.Series(0.0, index=common_dates)
    for date in common_dates:
        port_daily_logret.loc[date] = _portfolio_daily_return(
            holdings, log_returns, date)

    # Link within subperiods, then across subperiods
    # TWR: each subperiod gets equal weight regardless of capital
    linked_return = 1.0
    subperiod_returns = []

    for i in range(len(all_dates) - 1):
        start = all_dates[i]
        end = all_dates[i + 1]

        mask = (port_daily_logret.index >= start) & (port_daily_logret.index <= end)
        sub_logret = port_daily_logret[mask]

        if len(sub_logret) == 0:
            continue

        # Subperiod return = exp(sum of log returns) - 1
        sub_return = np.exp(sub_logret.sum()) - 1
        linked_return *= (1 + sub_return)
        subperiod_returns.append({
            'start': start, 'end': end,
            'n_days': len(sub_logret),
            'return': sub_return,
        })

    # Build daily index by cumulating log returns
    cum_logret = port_daily_logret.cumsum()
    index_values = start_value * np.exp(cum_logret)

    print(f"    TWR: {len(subperiod_returns)} subperiods, "
          f"total return: {(linked_return - 1) * 100:.2f}%")

    return index_values, port_daily_logret, subperiod_returns


def method_linked_modified_dietz(fund_daily, holdings, log_returns,
                                 df_closes, start_value=100):
    """
    Method 2: Linked Modified Dietz (monthly subperiods)

    From Chapter 5, eq 5.5:
      r = (EV - BV - ΣC_k) / (BV + ΣW_k·C_k)
    where W_k = (TD - D_k) / TD

    1. Divide into monthly subperiods
    2. For each month, calculate portfolio value at start/end
    3. Account for cashflows within the month using day-weighting
    4. Link monthly Dietz returns
    """
    common_dates = holdings.index.intersection(log_returns.index)
    algos = holdings.columns.intersection(df_closes.columns)

    # Portfolio value = Σ(units × price) for loaded algos
    h = holdings.loc[common_dates, algos]
    p = df_closes.reindex(common_dates)[algos]
    port_value = (h * p).sum(axis=1)
    port_value = port_value[port_value > 0]

    # Get cashflows aligned to portfolio dates
    cf = fund_daily['cashflow'].reindex(port_value.index).fillna(0)

    # Monthly subperiods
    months = port_value.resample('ME').last().index
    monthly_returns = []

    for i, month_end in enumerate(months):
        month_start = pd.Timestamp(month_end.year, month_end.month, 1)
        if i > 0:
            month_start = months[i - 1] + pd.Timedelta(days=1)

        mask = (port_value.index >= month_start) & (port_value.index <= month_end)
        pv = port_value[mask]

        if len(pv) < 2:
            monthly_returns.append({'month': month_end, 'return': 0.0, 'method': 'skip'})
            continue

        BV = pv.iloc[0]
        EV = pv.iloc[-1]

        # Cashflows in this month
        cf_month = cf[mask]
        TD = (pv.index[-1] - pv.index[0]).days
        if TD == 0:
            TD = 1

        # Modified Dietz: day-weighted cashflows
        sum_cf = 0
        sum_wcf = 0
        for date, c in cf_month.items():
            if abs(c) > 100:
                D_k = (date - pv.index[0]).days
                W_k = (TD - D_k) / TD
                sum_cf += c
                sum_wcf += W_k * c

        # ABV = BV + Σ(W_k * C_k)  (eq 5.6)
        ABV = BV + sum_wcf
        # AEV = EV - Σ((1 - W_k) * C_k)  (eq 5.7)
        # which equals EV - sum_cf + sum_wcf

        if ABV > 0:
            # Modified Dietz return (eq 5.5)
            r_dietz = (EV - BV - sum_cf) / ABV
        else:
            r_dietz = 0.0

        monthly_returns.append({
            'month': month_end,
            'return': r_dietz,
            'BV': BV, 'EV': EV,
            'sum_cf': sum_cf, 'ABV': ABV,
            'method': 'dietz',
        })

    # Link monthly returns
    linked = 1.0
    index_points = [{'date': port_value.index[0], 'value': start_value}]

    for mr in monthly_returns:
        linked *= (1 + mr['return'])
        index_points.append({
            'date': mr['month'],
            'value': start_value * linked,
        })

    index_df = pd.DataFrame(index_points).set_index('date')['value']
    # Interpolate to daily
    daily_idx = pd.date_range(index_df.index[0], index_df.index[-1], freq='D')
    index_daily = index_df.reindex(daily_idx).interpolate(method='time')

    total_ret = (linked - 1) * 100
    print(f"    Linked Dietz: {len(monthly_returns)} months, "
          f"total return: {total_ret:.2f}%")

    return index_daily, pd.DataFrame(monthly_returns)


def method_daily_modified_dietz(fund_daily, start_value=100):
    """
    Method 3: Daily Modified Dietz from fund-level data

    Uses equity_EOD directly with daily cashflows:
      r_d = (EV - BV - CF) / (BV + 0.5 * CF)

    (Midpoint Dietz: cashflows assumed mid-day)
    Then links daily returns.
    """
    equity = fund_daily['equity_EOD'].dropna()
    cf = fund_daily['cashflow'].reindex(equity.index).fillna(0)

    daily_returns = pd.Series(0.0, index=equity.index)

    for i in range(1, len(equity)):
        BV = equity.iloc[i - 1]
        EV = equity.iloc[i]
        C = cf.iloc[i]

        if BV <= 0:
            continue

        # Modified Dietz (single-day, midpoint assumption)
        denom = BV + 0.5 * C
        if denom > 0:
            r = (EV - BV - C) / denom
        else:
            r = 0.0

        # Clamp extreme returns
        r = max(min(r, 0.5), -0.5)
        daily_returns.iloc[i] = r

    # Link
    cum_return = (1 + daily_returns).cumprod()
    index_values = start_value * cum_return

    total = (cum_return.iloc[-1] - 1) * 100
    print(f"    Daily Dietz: {len(equity)} days, total return: {total:.2f}%")

    return index_values, daily_returns


def method_volume_weighted_cashflow_adj(holdings, log_returns,
                                        fund_daily, start_value=100):
    """
    Method 4: Volume-Weighted with Cashflow Scaling

    Same as no-cashflow volume-weighted, but scales each day's
    return by the fraction of AUM that is actually invested
    (excluding cash in transit from recent cashflows).
    """
    total_units = holdings.sum(axis=1).replace(0, np.nan)
    weights = holdings.div(total_units, axis=0).fillna(0)

    common_dates = weights.index.intersection(log_returns.index)
    common_algos = weights.columns.intersection(log_returns.columns)

    w = weights.loc[common_dates, common_algos]
    r = log_returns.loc[common_dates, common_algos].fillna(0)

    # Base portfolio log return
    port_log_ret = (w * r).sum(axis=1)

    # On days with large cashflows, dampen the return
    # (new money wasn't invested for the full day)
    if fund_daily is not None:
        cf = fund_daily['cashflow'].reindex(common_dates).fillna(0)
        equity = fund_daily['equity_EOD'].reindex(common_dates).fillna(0)
        cf_ratio = (cf / equity.replace(0, np.nan)).fillna(0).abs()

        # Scale factor: on cashflow days, reduce return proportionally
        # If 50% of equity is new money, only 50% was invested
        scale = (1 - cf_ratio.clip(0, 0.9))
        port_log_ret = port_log_ret * scale

    cum_log_ret = port_log_ret.cumsum()
    index_values = start_value * np.exp(cum_log_ret)

    total = (np.exp(cum_log_ret.iloc[-1]) - 1) * 100
    print(f"    Vol-Weighted + CF adj: {len(index_values)} days, "
          f"total return: {total:.2f}%")

    return index_values, port_log_ret


# ============================================================
# 4. COMPARE WITH ACTUAL
# ============================================================

def load_actual(monthly_path):
    """Load actual benchmark from monthly returns."""
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


def compare(actual, reconstructed, method_name):
    """Compare reconstruction vs actual."""
    common = actual.dropna().index.intersection(reconstructed.dropna().index)
    if len(common) < 30:
        return {'method': method_name, 'error': 'insufficient_overlap'}

    a = actual.loc[common]
    r = reconstructed.loc[common]
    a_norm = a / a.iloc[0] * 100
    r_norm = r / r.iloc[0] * 100

    a_ret = np.log(a_norm / a_norm.shift(1)).dropna()
    r_ret = np.log(r_norm / r_norm.shift(1)).dropna()
    both = pd.DataFrame({'actual': a_ret, 'recon': r_ret}).dropna()

    diff = both['actual'] - both['recon']

    a_m = a_norm.resample('ME').last()
    r_m = r_norm.resample('ME').last()

    return {
        'method': method_name,
        'n_days': len(common),
        'correlation_daily': round(float(both['actual'].corr(both['recon'])), 6),
        'tracking_error_annual': round(float(diff.std() * np.sqrt(252)), 6),
        'rmse_index': round(float(np.sqrt(((r_norm - a_norm) ** 2).mean())), 4),
        'final_actual': round(float(a_norm.iloc[-1]), 2),
        'final_recon': round(float(r_norm.iloc[-1]), 2),
        'final_diff_pct': round(float((r_norm.iloc[-1] - a_norm.iloc[-1]) / a_norm.iloc[-1] * 100), 2),
        'monthly_correlation': round(float(a_m.corr(r_m)), 6),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark reconstruction WITH cashflows (Dietz methods)')
    parser.add_argument('--trades', default=str(raw_benchmark_path('trades_benchmark.csv')))
    parser.add_argument('--monthly', default=str(raw_benchmark_path('benchmark_monthly_returns.csv')),
                        help='benchmark_monthly_returns.csv')
    parser.add_argument('--algos', default=str(raw_algos_path()), help='Directory with algo CSVs')
    parser.add_argument('--output', default=str(default_output_dir('dietz')))
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1. Fund timeline
    print("\n[1/5] Building fund timeline + GIPS cashflows...")
    fund_daily, trades_df = build_fund_timeline(args.trades)

    # 2. Holdings matrix
    print("\n[2/5] Building holdings matrix...")
    holdings = build_holdings_matrix(trades_df)
    print(f"  Holdings: {holdings.shape[0]} days × {holdings.shape[1]} algos")

    # 3. Load algo data
    print("\n[3/5] Loading algorithm prices...")
    df_closes, missing = load_algo_closes(args.algos, holdings.columns.tolist())
    log_returns = np.log(df_closes / df_closes.shift(1))
    print(f"  Log returns: {log_returns.shape}")

    # 4. Reconstruct
    print("\n[4/5] Reconstructing (4 methods)...")
    methods = {}

    print("\n  Method 1: True Time-Weighted Return (TWR)...")
    idx_twr, ret_twr, subs_twr = method_twr(
        fund_daily, holdings, log_returns)
    methods['TWR'] = idx_twr

    print("\n  Method 2: Linked Modified Dietz (monthly)...")
    idx_lmd, monthly_lmd = method_linked_modified_dietz(
        fund_daily, holdings, log_returns, df_closes)
    methods['linked_modified_dietz'] = idx_lmd

    print("\n  Method 3: Daily Modified Dietz (fund-level)...")
    idx_dmd, ret_dmd = method_daily_modified_dietz(fund_daily)
    methods['daily_modified_dietz'] = idx_dmd

    print("\n  Method 4: Volume-Weighted + Cashflow Adjustment...")
    idx_vwcf, ret_vwcf = method_volume_weighted_cashflow_adj(
        holdings, log_returns, fund_daily)
    methods['volume_weighted_cf_adj'] = idx_vwcf

    # 5. Compare
    print("\n[5/5] Comparing with actual benchmark...")
    actual = load_actual(args.monthly)

    results = []
    for name, idx in methods.items():
        stats = compare(actual, idx, name)
        results.append(stats)
        if 'error' not in stats:
            print(f"\n  {name}:")
            print(f"    Daily corr:    {stats['correlation_daily']:.6f}")
            print(f"    Monthly corr:  {stats['monthly_correlation']:.6f}")
            print(f"    Tracking err:  {stats['tracking_error_annual']:.6f}")
            print(f"    Final diff:    {stats['final_diff_pct']:+.2f}%")

    # Save
    print("\n  Saving...")
    pd.DataFrame(results).to_csv(
        os.path.join(args.output, 'dietz_reconstruction_results.csv'), index=False)

    daily_df = pd.DataFrame(methods)
    daily_df['actual'] = actual
    daily_df.to_csv(os.path.join(args.output, 'dietz_reconstruction_daily.csv'))

    # Monthly Dietz details
    monthly_lmd.to_csv(
        os.path.join(args.output, 'dietz_monthly_returns.csv'), index=False)

    best = min([r for r in results if 'error' not in r],
               key=lambda x: x.get('tracking_error_annual', 999),
               default={'method': 'none'})

    report = {
        'comparison': results,
        'best_method': best,
        'n_algos_loaded': len(df_closes.columns),
        'n_algos_missing': len(missing),
        'n_subperiods_twr': len(subs_twr),
        'reference': {
            'book': 'Christopherson, Carino & Ferson - Portfolio Performance Measurement',
            'chapter': '5 - Returns in the Presence of Cash Flows',
            'modified_dietz': 'Equation 5.5: r = (EV - BV - ΣC_k) / (BV + ΣW_k·C_k)',
            'twr': 'Equation 5.2: r = Π(1 + r_t) - 1',
            'day_weight': 'Equation 5.4: W_k = (TD - D_k) / TD',
        },
    }
    with open(os.path.join(args.output, 'dietz_reconstruction_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  BEST DIETZ METHOD: {best.get('method', '?')}")
    if 'tracking_error_annual' in best:
        print(f"  Tracking error:    {best['tracking_error_annual']:.6f}")
        print(f"  Monthly corr:      {best['monthly_correlation']:.6f}")
        print(f"  Final diff:        {best['final_diff_pct']:+.2f}%")
    print(f"{'=' * 60}")
    print("Done!")


if __name__ == '__main__':
    main()
