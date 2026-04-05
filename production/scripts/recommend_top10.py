#!/usr/bin/env python
"""
Top 10 Portfolio Recommendation
===============================
Generates a concentrated portfolio of top 10 assets based on RL family weights
and within-family Sortino ranking.

Uses a filtered universe of S&P 500 / large cap stocks.

Usage:
    python production/scripts/recommend_top10.py
    python production/scripts/recommend_top10.py --top 20
    python production/scripts/recommend_top10.py --capital 100000
    python production/scripts/recommend_top10.py --all-assets  # Use full universe
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PRODUCTION_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import trading universe
from production.config.universe import TRADING_UNIVERSE

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.trading_env import TradingEnvironment
from src.environment.cost_model import CostModel
from src.environment.constraints import PortfolioConstraints
from src.environment.reward import RewardFunction, RewardType

logger = logging.getLogger(__name__)

AGENT_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}


def compute_sortino(returns: pd.DataFrame, window: int = 63, max_sortino: float = 10.0) -> pd.Series:
    """Compute Sortino ratio for each asset over rolling window."""
    recent = returns.iloc[-window:]
    mean_ret = recent.mean() * 252  # Annualized
    downside = recent.copy()
    downside[downside > 0] = 0
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    sortino = mean_ret / downside_std.replace(0, np.nan)
    sortino = sortino.fillna(0)
    # Cap extreme values (likely data anomalies)
    sortino = sortino.clip(upper=max_sortino)
    return sortino


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_latest_prices(symbols: list, db_path: Path) -> pd.Series:
    """Load the latest closing prices for given symbols from DuckDB."""
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return pd.Series(dtype=float)

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        # Get latest price for each symbol
        symbols_str = ", ".join(f"'{s}'" for s in symbols)
        query = f"""
            WITH latest AS (
                SELECT symbol, close, date,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
                FROM equity_prices_daily
                WHERE symbol IN ({symbols_str})
            )
            SELECT symbol, close, date
            FROM latest
            WHERE rn = 1
        """
        df = conn.execute(query).fetchdf()
        conn.close()

        if df.empty:
            return pd.Series(dtype=float)

        prices = df.set_index('symbol')['close']
        logger.info(f"Loaded prices for {len(prices)} assets (as of {df['date'].max()})")
        return prices
    except Exception as e:
        logger.warning(f"Error loading prices: {e}")
        return pd.Series(dtype=float)


def get_top_n_portfolio(
    returns: pd.DataFrame,
    family_weights: np.ndarray,
    family_labels: pd.Series,
    n_top: int = 10,
    min_family_weight: float = 0.05,
    only_good_families: bool = True,  # Only pick from families 0 and 1
    sortino_window: int = 63,  # Window for Sortino calculation
) -> dict:
    """
    Select top N assets based on family weights and within-family Sortino ranking.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns
    family_weights : np.ndarray
        RL-determined weights for each family (4 values)
    family_labels : pd.Series
        Family assignment for each asset (0-3, or -1 for unassigned)
    n_top : int
        Number of top assets to select
    min_family_weight : float
        Minimum family weight to consider for selection

    Returns
    -------
    dict
        {asset: weight} for top N assets
    """
    # Normalize family weights
    fw = np.clip(family_weights, 0, None)

    # If only_good_families, zero out families 2 and 3 (Moderate and Poor)
    if only_good_families:
        fw[2] = 0  # Moderate (Sortino 0-1)
        fw[3] = 0  # Poor (Sortino < 0)

    if fw.sum() > 0:
        fw = fw / fw.sum()
    else:
        # Fallback to equal weight on good families only
        fw = np.array([0.5, 0.5, 0.0, 0.0])

    # Compute Sortino for all assets
    sortino = compute_sortino(returns, window=sortino_window)

    # For each family, get top assets by Sortino
    candidates = []

    for fi in range(4):
        family_w = fw[fi]
        if family_w < min_family_weight:
            continue

        # Get assets in this family
        family_mask = family_labels == fi
        family_assets = family_labels[family_mask].index.tolist()

        if not family_assets:
            continue

        # Rank by Sortino within family
        family_sortino = sortino[family_assets].sort_values(ascending=False)

        # Take top assets from this family proportional to family weight
        n_from_family = max(1, int(np.ceil(n_top * family_w)))
        top_in_family = family_sortino.head(n_from_family)

        for asset, sort_val in top_in_family.items():
            candidates.append({
                'asset': asset,
                'family': fi,
                'family_weight': family_w,
                'sortino': sort_val,
                'score': family_w * (1 + sort_val),  # Combined score
            })

    # Sort by score and take top N
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:n_top]

    # Allocate weights proportional to score
    total_score = sum(c['score'] for c in candidates)

    portfolio = {}
    for c in candidates:
        weight = c['score'] / total_score if total_score > 0 else 1.0 / len(candidates)
        portfolio[c['asset']] = {
            'weight': weight,
            'family': c['family'],
            'sortino': c['sortino'],
            'family_weight': c['family_weight'],
        }

    return portfolio


def main():
    parser = argparse.ArgumentParser(description="Generate top N portfolio recommendation")
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of top assets to select (default: 10)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Portfolio capital (default: 100,000)",
    )
    parser.add_argument(
        "--all-assets",
        action="store_true",
        help="Use all assets instead of filtered S&P 500 universe",
    )
    parser.add_argument(
        "--all-families",
        action="store_true",
        help="Include all families (by default only picks from Excellent/Good families)",
    )
    parser.add_argument(
        "--sortino-window",
        type=int,
        default=63,
        choices=[5, 21, 63],
        help="Sortino ratio window in days for ranking (default: 63)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PRODUCTION_DIR / "config" / "production.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PRODUCTION_DIR / "data" / "processed",
        help="Processed data directory",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PRODUCTION_DIR / "data" / "raw" / "quant.duckdb",
        help="DuckDB database path for prices",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Find model directory
    if args.model_dir is None:
        latest_file = PRODUCTION_DIR / "models" / "latest_run.txt"
        if latest_file.exists():
            latest_name = latest_file.read_text().strip()
            args.model_dir = PRODUCTION_DIR / "models" / latest_name
        else:
            model_dirs = sorted((PRODUCTION_DIR / "models").glob("2*_*"), reverse=True)
            if not model_dirs:
                logger.error("No trained models found")
                sys.exit(1)
            args.model_dir = model_dirs[0]

    logger.info(f"Using model: {args.model_dir.name}")

    # Load config
    config = load_config(args.config)

    # Load returns
    returns_path = args.data_dir / "returns.parquet"
    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days x {returns.shape[1]} assets")

    # Determine trading universe (S&P 500 / large caps) - filter at selection time
    if not args.all_assets:
        available_assets = [a for a in TRADING_UNIVERSE if a in returns.columns]
        missing_assets = [a for a in TRADING_UNIVERSE if a not in returns.columns]
        logger.info(f"Trading universe: {len(available_assets)} assets ({len(missing_assets)} not in data)")
    else:
        available_assets = list(returns.columns)
        logger.info("Using all assets (--all-assets flag)")

    # Load features to get family labels
    features_path = args.data_dir / "features.parquet"
    features = pd.read_parquet(features_path)

    # Extract family_63d for all assets first
    all_family_labels = {}
    for col in features.columns:
        if col[1] == "family_63d":
            asset = col[0]
            if asset != "_cross":
                val = features[col].dropna().iloc[-1] if len(features[col].dropna()) > 0 else 2
                all_family_labels[asset] = int(val)

    # Compute Sortino for filtering universe
    recent_sortino = compute_sortino(returns[available_assets], window=args.sortino_window)

    # Build family labels for our trading universe
    family_labels = {}
    for asset in available_assets:
        if asset in all_family_labels:
            family_labels[asset] = all_family_labels[asset]
        else:
            # Assign based on recent Sortino
            s = recent_sortino.get(asset, 0)
            if s > 2:
                family_labels[asset] = 0
            elif s > 1:
                family_labels[asset] = 1
            elif s > 0:
                family_labels[asset] = 2
            else:
                family_labels[asset] = 3

    family_labels = pd.Series(family_labels)
    logger.info(f"Family labels for {len(family_labels)} assets in trading universe")

    # Load model and encoder
    model_path = args.model_dir / "checkpoints" / "ppo" / "best_model.zip"
    encoder_path = args.model_dir / "checkpoints" / "ppo" / "universe_encoder.pkl"

    model = PPO.load(str(model_path))
    logger.info("Model loaded")

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    logger.info(f"Encoder loaded: {encoder._n_families} families")

    # Align returns columns with encoder's trained columns
    # This handles the case where new assets were added after model training
    if hasattr(encoder, '_trained_columns') and encoder._trained_columns is not None:
        trained_cols = encoder._trained_columns
        common_cols = [c for c in trained_cols if c in returns.columns]
        if len(common_cols) < len(trained_cols):
            logger.warning(
                f"Missing {len(trained_cols) - len(common_cols)} columns from training data. "
                f"Reindexing to {len(trained_cols)} trained columns."
            )
        returns = returns.reindex(columns=trained_cols)
        logger.info(f"Aligned returns to encoder: {returns.shape[1]} columns")
    else:
        # Fallback for older encoders without _trained_columns
        logger.warning(
            "Encoder was trained with older version (no _trained_columns). "
            "Attempting to infer columns from family_labels..."
        )
        if hasattr(encoder, 'family_labels') and encoder.family_labels is not None:
            trained_cols = list(encoder.family_labels.index)
            if len(trained_cols) == encoder._n_total_algos:
                common_cols = [c for c in trained_cols if c in returns.columns]
                logger.info(
                    f"Inferred {len(trained_cols)} columns from family_labels, "
                    f"{len(common_cols)} present in current data"
                )
                returns = returns.reindex(columns=trained_cols)
                logger.info(f"Aligned returns to encoder: {returns.shape[1]} columns")
            else:
                raise RuntimeError(
                    f"Column count mismatch: encoder expects {encoder._n_total_algos} columns, "
                    f"but family_labels has {len(trained_cols)}. Please retrain the model."
                )
        elif encoder._n_total_algos != returns.shape[1]:
            raise RuntimeError(
                f"Column count mismatch: encoder expects {encoder._n_total_algos} columns, "
                f"but returns has {returns.shape[1]}. Please retrain the model with --force-retrain."
            )

    # Update available_assets and family_labels to only include those in aligned returns
    available_assets = [a for a in available_assets if a in returns.columns]
    family_labels = family_labels[[a for a in family_labels.index if a in available_assets]]
    logger.info(f"Available assets after alignment: {len(available_assets)}")

    # Create environment to get observation
    n_assets = returns.shape[1]
    benchmark_weights = pd.DataFrame(
        np.ones((len(returns), n_assets)) / n_assets,
        index=returns.index,
        columns=returns.columns,
    )

    costs_cfg = config.get("costs", {})
    cost_model = CostModel(
        spread_bps=costs_cfg.get("spread_bps", 5),
        slippage_bps=costs_cfg.get("slippage_bps", 2),
        impact_coefficient=costs_cfg.get("market_impact_coef", 0.1),
    )

    constraints_cfg = config.get("constraints", {})
    constraints = PortfolioConstraints(
        max_weight=constraints_cfg.get("max_weight", 0.40),
        min_weight=constraints_cfg.get("min_weight", 0.0),
        max_turnover=constraints_cfg.get("max_turnover", 0.30),
        max_exposure=constraints_cfg.get("max_exposure", 1.0),
    )

    reward_cfg = config.get("reward", {})
    reward_fn = RewardFunction(
        reward_type=RewardType.RISK_CALIBRATED_RETURNS,
        cost_penalty_weight=reward_cfg.get("cost_penalty", 1.0),
        turnover_penalty_weight=reward_cfg.get("turnover_penalty", 0.1),
        drawdown_penalty_weight=reward_cfg.get("drawdown_penalty", 0.5),
    )

    env = TradingEnvironment(
        algo_returns=returns,
        benchmark_weights=benchmark_weights,
        cost_model=cost_model,
        constraints=constraints,
        reward_function=reward_fn,
        initial_capital=1_000_000.0,
        rebalance_frequency="weekly",
        train_start=returns.index[0],
        train_end=returns.index[-1],
        encoder=encoder,
    )

    vec_env = DummyVecEnv([lambda: env])

    # Load VecNormalize if available
    vec_norm_path = args.model_dir / "checkpoints" / "ppo" / "vecnormalize.pkl"
    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Get observation and predict action
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    family_weights = action[0]
    n_families = len(family_weights)

    logger.info(f"RL Family weights ({n_families} families): {family_weights}")
    family_names = ["Sortino >2", "Sortino 1-2", "Sortino 0-1", "Sortino <0"]
    for i in range(n_families):
        logger.info(f"  Family {i} ({family_names[i]}): {family_weights[i]:.2%}")

    # Pad family_weights to 4 if model was trained with fewer families
    # Missing families get weight 0 (they won't be selected)
    full_family_weights = np.zeros(4)
    full_family_weights[:n_families] = family_weights

    # Get top N portfolio (using only assets in trading universe)
    only_good = not args.all_families
    portfolio = get_top_n_portfolio(
        returns=returns[available_assets],
        family_weights=full_family_weights,
        family_labels=family_labels,
        n_top=args.top,
        only_good_families=only_good,
        sortino_window=args.sortino_window,
    )

    if only_good:
        logger.info("Filtering to only Excellent/Good families (Sortino > 1)")

    # Load latest prices for recommended assets
    portfolio_assets = list(portfolio.keys())
    latest_prices = load_latest_prices(portfolio_assets, args.db_path)

    # Build recommendation
    current_date = returns.index[-1]
    recommendation = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "data_date": current_date.strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "n_positions": args.top,
        "capital": args.capital,
        "n_families_trained": n_families,
        "sortino_window": args.sortino_window,
        "family_weights": {
            "excellent_sortino_gt2": float(full_family_weights[0]),
            "good_sortino_1to2": float(full_family_weights[1]),
            "moderate_sortino_0to1": float(full_family_weights[2]),
            "poor_sortino_lt0": float(full_family_weights[3]),
        },
        "positions": [],
    }

    # Sort by weight descending
    sorted_positions = sorted(portfolio.items(), key=lambda x: x[1]['weight'], reverse=True)

    for asset, info in sorted_positions:
        dollar_amount = float(info['weight'] * args.capital)
        price = latest_prices.get(asset, None)
        shares = int(dollar_amount / price) if price and price > 0 else None

        position = {
            "asset": str(asset),
            "weight": float(info['weight']),
            "weight_pct": float(info['weight'] * 100),
            "dollar_amount": dollar_amount,
            "price": float(price) if price else None,
            "shares": shares,
            "family": int(info['family']),
            f"sortino_{args.sortino_window}d": float(info['sortino']),
        }
        recommendation["positions"].append(position)

    # Save if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(recommendation, f, indent=2)
        logger.info(f"Saved to {args.output}")

    # Also save to default location
    default_output = PRODUCTION_DIR / "outputs" / "recommendations" / f"top{args.top}_{datetime.now().strftime('%Y-%m-%d')}.json"
    default_output.parent.mkdir(parents=True, exist_ok=True)
    with open(default_output, 'w') as f:
        json.dump(recommendation, f, indent=2)

    # Display results
    print()
    print("=" * 105)
    print(f"TOP {args.top} PORTFOLIO RECOMMENDATION - {recommendation['date']}")
    print("=" * 105)
    print(f"Data as of: {recommendation['data_date']}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Model trained with: {n_families} families")
    print(f"Sortino window: {args.sortino_window} days")
    print()
    print("RL FAMILY ALLOCATION:")
    family_display = [
        ("Family 0 (Excellent Sortino >2):", full_family_weights[0]),
        ("Family 1 (Good Sortino 1-2):    ", full_family_weights[1]),
        ("Family 2 (Moderate Sortino 0-1):", full_family_weights[2]),
        ("Family 3 (Poor Sortino <0):     ", full_family_weights[3]),
    ]
    for i, (name, weight) in enumerate(family_display):
        status = "" if i < n_families else " (not trained)"
        print(f"  {name} {weight:>6.1%}{status}")
    print()
    sortino_key = f"sortino_{args.sortino_window}d"
    print("-" * 105)
    print(f"{'Rank':<4} {'Asset':<6} {'Weight':>7} {'$Amount':>10} {'Price':>9} {'Shares':>7} {'Family':>9} {f'Sortino{args.sortino_window}d':>9}")
    print("-" * 105)

    total_shares_value = 0
    for i, pos in enumerate(recommendation["positions"], 1):
        family_name = ["Excellent", "Good", "Moderate", "Poor"][pos["family"]]
        price_str = f"${pos['price']:>7.2f}" if pos['price'] else "    N/A"
        shares_str = f"{pos['shares']:>7,d}" if pos['shares'] else "    N/A"
        if pos['shares'] and pos['price']:
            total_shares_value += pos['shares'] * pos['price']
        print(f"{i:<4} {pos['asset']:<6} {pos['weight_pct']:>6.1f}% ${pos['dollar_amount']:>8,.0f} {price_str} {shares_str} {family_name:>9} {pos[sortino_key]:>9.2f}")

    print("-" * 105)
    total_weight = sum(p['weight'] for p in recommendation["positions"])
    total_amount = sum(p['dollar_amount'] for p in recommendation["positions"])
    print(f"{'TOTAL':<4} {'':<6} {total_weight*100:>6.1f}% ${total_amount:>8,.0f}            Invested: ${total_shares_value:>10,.2f}")
    print("=" * 105)
    print()
    print(f"Saved to: {default_output}")


if __name__ == "__main__":
    main()
