#!/usr/bin/env python3
"""
=================================================================
BENCHMARK CASHFLOW ANALYSIS (GIPS / Unit Price Method)
=================================================================
Detects inflows and outflows by comparing two columns:

  - equity_normalized: Pure strategy performance (blind to cashflows)
  - equity_EOD: Real equity (includes deposits/withdrawals)

Method:
  1. Strategy return = equity_normalized_t / equity_normalized_{t-1} - 1
  2. Expected equity = equity_EOD_{t-1} × (1 + strategy_return)
  3. Cashflow = equity_EOD_t - expected_equity_t
     - Positive → Inflow (deposit)
     - Negative → Outflow (withdrawal)

Usage:
  python3 benchmark_cashflows.py \
    --trades trades_benchmark.csv \
    --monthly benchmark_monthly_returns.csv \
    --output results/cashflows/
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import warnings
warnings.filterwarnings('ignore')


def build_daily_timeline(trades_path):
    """
    Build daily fund-level timeline from trade snapshots.
    Each trade row contains a snapshot of equity_EOD and equity_normalized
    at the moment the trade was opened.
    """
    df = pd.read_csv(trades_path)
    df['dateOpen'] = pd.to_datetime(df['dateOpen'], format='mixed', utc=True).dt.tz_localize(None)
    df['dateClose'] = pd.to_datetime(df['dateClose'], format='mixed', utc=True).dt.tz_localize(None)
    df['obs_date'] = df['dateOpen'].dt.normalize()
    df = df.rename(columns={'productname': 'algo'})
    df['holding_days'] = (df['dateClose'] - df['dateOpen']).dt.total_seconds() / 86400

    # Last snapshot per day = most up-to-date view of fund state
    daily = df.groupby('obs_date').agg(
        equity_EOD=('equity_EOD', 'last'),
        equity_normalized=('equity_normalized', 'last'),
        AUM=('AUM', 'last'),
        total_invested=('total_invested_amount_EOD', 'last'),
        n_trades=('volume', 'count'),
        volume_opened=('volume', 'sum'),
        n_algos=('algo', 'nunique'),
    ).sort_index()

    daily['leverage'] = (daily['AUM'] / daily['equity_EOD']).round(2)
    return daily, df


def detect_cashflows_gips(daily):
    """
    Detect cashflows using the GIPS Unit Price Method.

    equity_normalized = pure performance (no cashflows)
    equity_EOD = real money (includes deposits/withdrawals)

    If strategy earns 2%, equity_normalized rises 2%.
    If equity_EOD rises 15% while equity_normalized only rises 2%,
    the difference is a 13% inflow.
    """
    # Step 1: Strategy return from normalized equity (pure performance)
    daily['ret_strategy'] = daily['equity_normalized'].pct_change()

    # Step 2: Expected equity if NO cashflow occurred
    # expected = yesterday's real equity × (1 + today's strategy return)
    daily['expected_equity_EOD'] = daily['equity_EOD'].shift(1) * (1 + daily['ret_strategy'])

    # Step 3: Cashflow = actual equity - expected equity
    daily['cashflow'] = daily['equity_EOD'] - daily['expected_equity_EOD']

    # Classify
    daily['cf_type'] = 'none'
    daily.loc[daily['cashflow'] > 100, 'cf_type'] = 'INFLOW'
    daily.loc[daily['cashflow'] < -100, 'cf_type'] = 'OUTFLOW'

    # Cumulative
    daily['cashflow_cumulative'] = daily['cashflow'].cumsum()
    daily['inflow'] = daily['cashflow'].clip(lower=0)
    daily['outflow'] = daily['cashflow'].clip(upper=0)
    daily['inflow_cumulative'] = daily['inflow'].cumsum()
    daily['outflow_cumulative'] = daily['outflow'].cumsum()

    return daily


def compute_monthly_cashflows(daily):
    """Aggregate cashflows at monthly level."""
    daily_copy = daily.copy()
    daily_copy['month'] = daily_copy.index.to_period('M')

    monthly = daily_copy.groupby('month').agg(
        # Cashflows
        inflows=('cashflow', lambda x: x[x > 100].sum()),
        outflows=('cashflow', lambda x: x[x < -100].sum()),
        n_inflows=('cashflow', lambda x: (x > 100).sum()),
        n_outflows=('cashflow', lambda x: (x < -100).sum()),
        # Fund state
        equity_eod_start=('equity_EOD', 'first'),
        equity_eod_end=('equity_EOD', 'last'),
        norm_start=('equity_normalized', 'first'),
        norm_end=('equity_normalized', 'last'),
        avg_aum=('AUM', 'mean'),
        avg_leverage=('leverage', 'mean'),
        # Activity
        total_trades=('n_trades', 'sum'),
        total_volume=('volume_opened', 'sum'),
        n_algos=('n_algos', 'max'),
    )

    monthly['net_cashflow'] = monthly['inflows'] + monthly['outflows']
    monthly['strategy_return_pct'] = ((monthly['norm_end'] / monthly['norm_start']) - 1) * 100
    monthly['equity_change_pct'] = ((monthly['equity_eod_end'] / monthly['equity_eod_start']) - 1) * 100
    monthly['cf_impact_pct'] = monthly['equity_change_pct'] - monthly['strategy_return_pct']

    monthly.index = monthly.index.astype(str)
    return monthly


def detect_phases(daily, trades_df):
    """Detect and characterize fund phases."""
    phases = {}

    boundaries = [
        ('Phase 1: Seed', None, '2023-12-25'),
        ('Phase 2: Transition', '2023-12-26', '2024-01-31'),
        ('Phase 3: Scale', '2024-02-01', None),
    ]

    for name, start, end in boundaries:
        mask = pd.Series(True, index=daily.index)
        if start:
            mask &= daily.index >= start
        if end:
            mask &= daily.index <= end

        p = daily[mask]
        if len(p) == 0:
            continue

        t_mask = pd.Series(True, index=trades_df.index)
        if start:
            t_mask &= trades_df['obs_date'] >= start
        if end:
            t_mask &= trades_df['obs_date'] <= end
        trades_phase = trades_df[t_mask]

        cf = p['cashflow'].dropna()

        phases[name] = {
            'date_range': f"{p.index.min().date()} → {p.index.max().date()}",
            'n_days': len(p),
            'avg_aum': round(float(p['AUM'].mean()), 0),
            'max_aum': round(float(p['AUM'].max()), 0),
            'avg_leverage': round(float(p['leverage'].mean()), 1),
            'n_trades': len(trades_phase),
            'trades_per_day': round(len(trades_phase) / max(len(p), 1), 1),
            'n_unique_algos': int(trades_phase['algo'].nunique()),
            'total_volume': round(float(trades_phase['volume'].sum()), 0),
            'avg_volume_per_trade': round(float(trades_phase['volume'].mean()), 0) if len(trades_phase) > 0 else 0,
            'median_holding_days': round(float(trades_phase['holding_days'].median()), 1) if len(trades_phase) > 0 else 0,
            # Cashflows
            'total_inflows': round(float(cf[cf > 100].sum()), 0),
            'total_outflows': round(float(cf[cf < -100].sum()), 0),
            'net_cashflow': round(float(cf.sum()), 0),
            'n_inflow_events': int((cf > 100).sum()),
            'n_outflow_events': int((cf < -100).sum()),
        }

    return phases


def algo_rotation_analysis(trades_df):
    """Analyze which algos are used in each phase."""
    seed = trades_df[trades_df['obs_date'] < '2023-12-26']
    scale = trades_df[trades_df['obs_date'] >= '2024-02-01']

    seed_algos = set(seed['algo'].unique())
    scale_algos = set(scale['algo'].unique())
    common = seed_algos & scale_algos

    # Top algos per phase by volume
    seed_top = seed.groupby('algo')['volume'].sum().sort_values(ascending=False).head(15)
    scale_top = scale.groupby('algo')['volume'].sum().sort_values(ascending=False).head(15)

    return {
        'seed_only': len(seed_algos - scale_algos),
        'scale_only': len(scale_algos - seed_algos),
        'common': len(common),
        'seed_total': len(seed_algos),
        'scale_total': len(scale_algos),
        'retention_pct': round(len(common) / max(len(seed_algos), 1) * 100, 1),
        'new_in_scale_pct': round(len(scale_algos - seed_algos) / max(len(scale_algos), 1) * 100, 1),
        'seed_top_15': {k: float(v) for k, v in seed_top.items()},
        'scale_top_15': {k: float(v) for k, v in scale_top.items()},
    }


def main():
    parser = argparse.ArgumentParser(description='GIPS cashflow analysis')
    parser.add_argument('--trades', required=True)
    parser.add_argument('--monthly', default=None)
    parser.add_argument('--output', default='results/cashflows')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Build timeline
    print("\n[1/4] Building daily timeline...")
    daily, trades_df = build_daily_timeline(args.trades)
    print(f"  {len(daily)} observation days, {len(trades_df)} trades")

    # Detect cashflows
    print("\n[2/4] Detecting cashflows (GIPS method)...")
    daily = detect_cashflows_gips(daily)

    cf = daily['cashflow'].dropna()
    total_in = cf[cf > 100].sum()
    total_out = cf[cf < -100].sum()
    print(f"  Total inflows:  {total_in:>15,.2f} ({(cf > 100).sum()} events)")
    print(f"  Total outflows: {total_out:>15,.2f} ({(cf < -100).sum()} events)")
    print(f"  Net cashflow:   {cf.sum():>15,.2f}")

    # Monthly
    print("\n[3/4] Computing monthly aggregations...")
    monthly = compute_monthly_cashflows(daily)

    print(f"\n  {'Month':<10} {'Inflows':>12} {'Outflows':>12} {'Net CF':>12} "
          f"{'Strat%':>8} {'Equity%':>8} {'CF Impact%':>10}")
    print(f"  {'-'*78}")
    for m, r in monthly.iterrows():
        print(f"  {m:<10} {r['inflows']:>12,.0f} {r['outflows']:>12,.0f} {r['net_cashflow']:>12,.0f} "
              f"{r['strategy_return_pct']:>+7.2f}% {r['equity_change_pct']:>+7.2f}% "
              f"{r['cf_impact_pct']:>+9.2f}%")

    # Phases
    print("\n[4/4] Phase analysis...")
    phases = detect_phases(daily, trades_df)
    rotation = algo_rotation_analysis(trades_df)

    for name, p in phases.items():
        print(f"\n  {name} ({p['date_range']})")
        print(f"    AUM: avg={p['avg_aum']:,.0f}, max={p['max_aum']:,.0f}, leverage={p['avg_leverage']}x")
        print(f"    Trading: {p['n_trades']} trades ({p['trades_per_day']}/day), "
              f"{p['n_unique_algos']} algos, median holding={p['median_holding_days']}d")
        print(f"    Cashflows: IN={p['total_inflows']:+,.0f} ({p['n_inflow_events']} events) | "
              f"OUT={p['total_outflows']:+,.0f} ({p['n_outflow_events']} events) | "
              f"NET={p['net_cashflow']:+,.0f}")

    print(f"\n  Algo rotation: {rotation['seed_total']} seed → {rotation['scale_total']} scale "
          f"({rotation['common']} common, {rotation['retention_pct']}% retained, "
          f"{rotation['new_in_scale_pct']}% new)")

    # === SAVE ===
    daily.to_csv(os.path.join(args.output, 'daily_fund_timeline.csv'))
    monthly.to_csv(os.path.join(args.output, 'monthly_fund_evolution.csv'))

    # Cashflow events (material only)
    cf_events = daily[daily['cashflow'].abs() > 100][
        ['equity_EOD', 'equity_normalized', 'expected_equity_EOD',
         'cashflow', 'cf_type', 'ret_strategy', 'AUM', 'leverage']
    ].copy()
    cf_events.to_csv(os.path.join(args.output, 'cashflow_events.csv'))

    report = {
        'method': 'GIPS Unit Price Method',
        'formula': {
            'step1': 'ret_strategy = equity_normalized_t / equity_normalized_{t-1} - 1',
            'step2': 'expected_equity = equity_EOD_{t-1} × (1 + ret_strategy)',
            'step3': 'cashflow = equity_EOD_t - expected_equity_t',
        },
        'totals': {
            'total_inflows': round(float(total_in), 2),
            'total_outflows': round(float(total_out), 2),
            'net_cashflow': round(float(cf.sum()), 2),
            'n_inflow_events': int((cf > 100).sum()),
            'n_outflow_events': int((cf < -100).sum()),
        },
        'phases': phases,
        'algo_rotation': rotation,
    }
    with open(os.path.join(args.output, 'cashflow_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Saved to {args.output}/")
    print("  Done!")


if __name__ == '__main__':
    main()