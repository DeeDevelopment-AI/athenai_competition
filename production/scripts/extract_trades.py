#!/usr/bin/env python
"""
Extract Trades from Recommendation
===================================
Converts portfolio recommendation to actionable buy/sell list.

Usage:
    python production/scripts/extract_trades.py
    python production/scripts/extract_trades.py --output trades.csv
    python production/scripts/extract_trades.py --top 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = SCRIPT_DIR.parent


def load_latest_recommendation() -> dict:
    """Load the most recent recommendation."""
    rec_dir = PRODUCTION_DIR / "outputs" / "recommendations"
    rec_files = sorted(rec_dir.glob("recommendation_*.json"), reverse=True)

    if not rec_files:
        raise FileNotFoundError("No recommendations found")

    with open(rec_files[0]) as f:
        return json.load(f)


def extract_trades(
    recommendation: dict,
    current_holdings: dict = None,
    capital: float = 1_000_000,
) -> pd.DataFrame:
    """
    Extract trade list from recommendation.

    Parameters
    ----------
    recommendation : dict
        Portfolio recommendation
    current_holdings : dict, optional
        Current holdings as {asset: weight}
    capital : float
        Total portfolio capital for dollar amounts

    Returns
    -------
    pd.DataFrame
        Trade list with columns: asset, action, target_weight, target_pct,
        vs_equal, target_value, trade_value (if current_holdings provided)
    """
    weights = recommendation["weights"]
    n_assets = len(weights)
    equal_weight = 1.0 / n_assets

    if current_holdings is None:
        current_holdings = {k: 0.0 for k in weights.keys()}

    trades = []
    for asset, target_weight in weights.items():
        current_weight = current_holdings.get(asset, 0.0)
        weight_change = target_weight - current_weight
        vs_equal = target_weight / equal_weight

        # Determine action
        if vs_equal > 2.5:
            action = "STRONG BUY"
        elif vs_equal > 1.5:
            action = "BUY"
        elif vs_equal < 0.5:
            action = "STRONG REDUCE"
        elif vs_equal < 0.7:
            action = "REDUCE"
        else:
            action = "HOLD"

        trades.append({
            "asset": asset,
            "action": action,
            "target_weight": target_weight,
            "target_pct": target_weight * 100,
            "vs_equal": vs_equal,
            "current_weight": current_weight,
            "weight_change": weight_change,
            "target_value": target_weight * capital,
            "trade_value": weight_change * capital,
        })

    df = pd.DataFrame(trades)
    df = df.sort_values("target_weight", ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Extract trades from recommendation")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of top positions to show (default: 50)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000,
        help="Portfolio capital for dollar amounts (default: 1M)",
    )
    parser.add_argument(
        "--buys-only",
        action="store_true",
        help="Only show buy recommendations",
    )

    args = parser.parse_args()

    # Load recommendation
    rec = load_latest_recommendation()
    print(f"Recommendation date: {rec['date']}")
    print(f"Agent: {rec['agent_type']}")
    print()

    # Extract trades
    df = extract_trades(rec, capital=args.capital)

    # Filter if requested
    if args.buys_only:
        df = df[df["action"].isin(["STRONG BUY", "BUY"])]

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved full trade list to: {args.output}")
        print()

    # Display summary
    print("=" * 80)
    print(f"TOP {args.top} POSITIONS TO BUY")
    print("=" * 80)

    buys = df[df["action"].isin(["STRONG BUY", "BUY"])].head(args.top)

    print(f"{'Rank':<6} {'Asset':<12} {'Action':<12} {'Weight %':<10} {'vs Equal':<10} {'$ Value':<12}")
    print("-" * 80)

    for i, (_, row) in enumerate(buys.iterrows(), 1):
        print(f"{i:<6} {row['asset']:<12} {row['action']:<12} {row['target_pct']:.4f}%    {row['vs_equal']:.2f}x       ${row['target_value']:,.0f}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total assets: {len(df):,}")
    print(f"Strong Buy: {(df['action'] == 'STRONG BUY').sum():,}")
    print(f"Buy: {(df['action'] == 'BUY').sum():,}")
    print(f"Hold: {(df['action'] == 'HOLD').sum():,}")
    print(f"Reduce: {(df['action'] == 'REDUCE').sum():,}")
    print(f"Strong Reduce: {(df['action'] == 'STRONG REDUCE').sum():,}")

    if args.output:
        print()
        print(f"Full list saved to: {args.output}")


if __name__ == "__main__":
    main()
