#!/usr/bin/env python
"""
Weekly Production Pipeline
==========================
Handles the complete weekly workflow:
1. Download data from Nasdaq Data Link (Sharadar) OR load weekly CSV
2. Calculate/update metrics (returns, volatility, Sharpe, Sortino, etc.)
3. Generate prediction for next week
4. Retrain model every N weeks (default: 4)

Usage:
    # Download from Sharadar and predict (RECOMMENDED)
    python production/scripts/weekly_pipeline.py --sharadar

    # Use CSV instead of Sharadar API
    python production/scripts/weekly_pipeline.py --csv weekly_quotes.csv

    # Force retrain regardless of schedule
    python production/scripts/weekly_pipeline.py --sharadar --force-retrain

    # Skip prediction (just update data)
    python production/scripts/weekly_pipeline.py --sharadar --no-predict

    # Check status only
    python production/scripts/weekly_pipeline.py --status

Environment:
    Set NASDAQ_DATA_LINK_API_KEY or create production/config/api_keys.yaml
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PRODUCTION_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# State file to track pipeline runs
STATE_FILE = PRODUCTION_DIR / "pipeline_state.json"
DEFAULT_DUCKDB_PATH = PRODUCTION_DIR / "data" / "raw" / "quant.duckdb"
DEFAULT_TABLE_NAME = "daily_metrics"  # Main table with prices + rolling metrics


def load_state() -> dict:
    """Load pipeline state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "last_run": None,
        "last_retrain": None,
        "weeks_since_retrain": 0,
        "total_runs": 0,
        "history": [],
    }


def save_state(state: dict) -> None:
    """Save pipeline state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def show_status() -> None:
    """Show current pipeline status."""
    state = load_state()

    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)
    print(f"Total runs: {state['total_runs']}")
    print(f"Last run: {state['last_run'] or 'Never'}")
    print(f"Last retrain: {state['last_retrain'] or 'Never'}")
    print(f"Weeks since retrain: {state['weeks_since_retrain']}")

    # Check if DuckDB exists
    if DEFAULT_DUCKDB_PATH.exists():
        conn = duckdb.connect(str(DEFAULT_DUCKDB_PATH), read_only=True)
        try:
            info = conn.execute(f"""
                SELECT
                    COUNT(*) as rows,
                    COUNT(DISTINCT symbol) as symbols,
                    MIN(date) as min_date,
                    MAX(date) as max_date
                FROM {DEFAULT_TABLE_NAME}
            """).fetchone()
            print(f"\nDatabase: {DEFAULT_DUCKDB_PATH}")
            print(f"  Rows: {info[0]:,}")
            print(f"  Symbols: {info[1]:,}")
            print(f"  Date range: {info[2]} to {info[3]}")
        except Exception as e:
            print(f"\nDatabase exists but error reading: {e}")
        finally:
            conn.close()
    else:
        print(f"\nDatabase: Not found at {DEFAULT_DUCKDB_PATH}")

    # Check model
    latest_file = PRODUCTION_DIR / "models" / "latest_run.txt"
    if latest_file.exists():
        latest_model = latest_file.read_text().strip()
        model_dir = PRODUCTION_DIR / "models" / latest_model
        print(f"\nLatest model: {latest_model}")
        if (model_dir / "checkpoints" / "ppo" / "best_model.zip").exists():
            print("  Status: Ready")
        else:
            print("  Status: Missing checkpoint")
    else:
        print("\nLatest model: None trained yet")

    # Recent history
    if state["history"]:
        print("\nRecent runs:")
        for run in state["history"][-5:]:
            print(f"  {run['date']}: {run['action']} - {run['status']}")

    print("=" * 60)


def load_csv_to_duckdb(
    csv_path: Path,
    db_path: Path = DEFAULT_DUCKDB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
) -> int:
    """
    Load weekly CSV data into DuckDB, appending to existing data.

    Expected CSV format:
        symbol,date,open,high,low,close,volume
        AAPL,2024-01-15,185.50,186.20,184.80,185.90,50000000
        ...

    Returns number of rows inserted.
    """
    logger.info(f"Loading CSV: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Normalize column names (case-insensitive)
    df.columns = df.columns.str.lower().str.strip()

    # Detect and rename columns
    col_mapping = {}
    for col in df.columns:
        if col in ["symbol", "ticker", "asset", "name"]:
            col_mapping[col] = "symbol"
        elif col in ["date", "datetime", "timestamp", "time"]:
            col_mapping[col] = "date"
        elif col in ["open", "o"]:
            col_mapping[col] = "open"
        elif col in ["high", "h"]:
            col_mapping[col] = "high"
        elif col in ["low", "l"]:
            col_mapping[col] = "low"
        elif col in ["close", "c", "adj_close", "adjusted_close"]:
            col_mapping[col] = "close"
        elif col in ["volume", "vol", "v"]:
            col_mapping[col] = "volume"

    df = df.rename(columns=col_mapping)

    # Ensure required columns
    required = ["symbol", "date", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Parse dates
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Add optional columns if missing
    for col in ["open", "high", "low", "volume"]:
        if col not in df.columns:
            df[col] = None

    logger.info(f"CSV loaded: {len(df)} rows, {df['symbol'].nunique()} symbols")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Connect to DuckDB (create if not exists)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    try:
        # Create table if not exists
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE NOT NULL,
                volume BIGINT,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Get existing date range for these symbols
        existing = conn.execute(f"""
            SELECT DISTINCT date FROM {table_name}
            WHERE symbol IN (SELECT DISTINCT symbol FROM df)
        """).fetchdf()

        if len(existing) > 0:
            existing_dates = set(pd.to_datetime(existing["date"]).dt.date)
            new_dates = set(df["date"].unique())
            overlap = existing_dates & new_dates
            if overlap:
                logger.warning(f"Overlapping dates found: {len(overlap)} days")
                logger.info("Removing duplicates (keeping new data)...")
                # Delete overlapping data
                for date in overlap:
                    conn.execute(f"""
                        DELETE FROM {table_name}
                        WHERE date = '{date}'
                        AND symbol IN (SELECT DISTINCT symbol FROM df WHERE date = '{date}')
                    """)

        # Insert new data
        conn.execute(f"""
            INSERT INTO {table_name} (symbol, date, open, high, low, close, volume)
            SELECT symbol, date, open, high, low, close, volume FROM df
        """)

        inserted = len(df)
        logger.info(f"Inserted {inserted} rows into {table_name}")

        # Get updated stats
        info = conn.execute(f"""
            SELECT
                COUNT(*) as rows,
                COUNT(DISTINCT symbol) as symbols,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM {table_name}
        """).fetchone()

        logger.info(f"Database now has: {info[0]:,} rows, {info[1]:,} symbols")
        logger.info(f"Date range: {info[2]} to {info[3]}")

        return inserted

    finally:
        conn.close()


def calculate_metrics(
    db_path: Path = DEFAULT_DUCKDB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    output_dir: Optional[Path] = None,
    min_observations: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export metrics from DuckDB (daily_metrics has pre-computed rolling features).

    Returns (returns, features) DataFrames for training.
    """
    if output_dir is None:
        output_dir = PRODUCTION_DIR / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting metrics from DuckDB...")

    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        # Get symbols with sufficient history
        symbols_df = conn.execute(f"""
            SELECT symbol, COUNT(*) as n_obs
            FROM {table_name}
            WHERE close IS NOT NULL
            GROUP BY symbol
            HAVING n_obs >= {min_observations}
            ORDER BY symbol
        """).fetchdf()

        symbols = symbols_df["symbol"].tolist()
        logger.info(f"Found {len(symbols)} symbols with >= {min_observations} observations")

        if not symbols:
            raise ValueError("No symbols with sufficient history")

        symbols_str = ", ".join(f"'{s}'" for s in symbols)

        # Load returns directly (already calculated as daily_return in some cases)
        # For daily_metrics, we compute from close prices
        logger.info("Loading returns...")
        returns_df = conn.execute(f"""
            WITH prices AS (
                SELECT symbol, date, close
                FROM {table_name}
                WHERE symbol IN ({symbols_str})
                AND close IS NOT NULL
                ORDER BY symbol, date
            ),
            returns AS (
                SELECT
                    symbol,
                    date,
                    (close / LAG(close) OVER (PARTITION BY symbol ORDER BY date)) - 1 AS daily_return
                FROM prices
            )
            SELECT symbol, date, daily_return
            FROM returns
            WHERE daily_return IS NOT NULL
            ORDER BY date, symbol
        """).fetchdf()

        # Pivot returns to wide format
        returns = returns_df.pivot(index="date", columns="symbol", values="daily_return")
        returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index()

        logger.info(f"Returns: {returns.shape[0]} dates x {returns.shape[1]} symbols")

        # Load pre-computed features from daily_metrics
        # The table already has: return_rolling_*, volatility_rolling_*, sharpe_*, sortino_*, calmar_*
        logger.info("Loading pre-computed features...")

        feature_cols = [
            "return_rolling_5d", "return_rolling_21d", "return_rolling_63d",
            "volatility_rolling_5d", "volatility_rolling_21d", "volatility_rolling_63d",
            "sharpe_5d", "sharpe_21d", "sharpe_63d",
            "sortino_5d", "sortino_21d", "sortino_63d",
        ]

        features_df = conn.execute(f"""
            SELECT symbol, date, {', '.join(feature_cols)}
            FROM {table_name}
            WHERE symbol IN ({symbols_str})
            ORDER BY date, symbol
        """).fetchdf()

        features_df["date"] = pd.to_datetime(features_df["date"])

        # Pivot each feature column to create MultiIndex DataFrame
        features = {}
        for feat in feature_cols:
            if feat in features_df.columns:
                pivoted = features_df.pivot(index="date", columns="symbol", values=feat)
                for symbol in pivoted.columns:
                    features[(symbol, feat)] = pivoted[symbol]

        # Add family classifications based on Sortino
        for symbol in symbols:
            for window in ["5d", "21d", "63d"]:
                sortino_col = (symbol, f"sortino_{window}")
                if sortino_col in features:
                    sortino_series = features[sortino_col]

                    def classify_family(s):
                        if pd.isna(s):
                            return -1
                        if s > 2:
                            return 0  # Excellent
                        elif s > 1:
                            return 1  # Good
                        elif s > 0:
                            return 2  # Moderate
                        else:
                            return 3  # Poor

                    features[(symbol, f"family_{window}")] = sortino_series.apply(classify_family)

        feature_df = pd.DataFrame(features)
        feature_df.columns = pd.MultiIndex.from_tuples(
            feature_df.columns, names=["symbol", "feature"]
        )
        feature_df = feature_df.sort_index()

        logger.info(f"Features: {feature_df.shape[0]} dates x {feature_df.shape[1]} columns")

        # Save
        returns.to_parquet(output_dir / "returns.parquet")
        feature_df.to_parquet(output_dir / "features.parquet")
        (output_dir / "assets.txt").write_text("\n".join(returns.columns))

        # Save stats
        stats = pd.DataFrame({
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "sharpe": returns.mean() / returns.std() * np.sqrt(252),
            "n_observations": returns.notna().sum(),
        })
        stats.to_csv(output_dir / "asset_stats.csv")

        logger.info(f"Saved to {output_dir}")

        return returns, feature_df

    finally:
        conn.close()


def calculate_rolling_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling features for all assets."""
    windows = [5, 21, 63]
    features = {}

    for symbol in returns.columns:
        r = returns[symbol].dropna()

        for w in windows:
            if len(r) < w:
                continue

            # Rolling return (annualized)
            roll_ret = r.rolling(w).mean() * 252
            features[(symbol, f"return_{w}d")] = roll_ret

            # Rolling volatility (annualized)
            roll_vol = r.rolling(w).std() * np.sqrt(252)
            features[(symbol, f"volatility_{w}d")] = roll_vol

            # Sharpe ratio
            sharpe = roll_ret / roll_vol.replace(0, np.nan)
            features[(symbol, f"sharpe_{w}d")] = sharpe.clip(-10, 10)

            # Sortino ratio
            downside = r.copy()
            downside[downside > 0] = 0
            downside_vol = downside.rolling(w).apply(
                lambda x: np.sqrt((x**2).mean()) * np.sqrt(252), raw=True
            )
            sortino = roll_ret / downside_vol.replace(0, np.nan)
            features[(symbol, f"sortino_{w}d")] = sortino.clip(-10, 10)

            # Family classification based on Sortino
            def classify_family(s):
                if pd.isna(s):
                    return -1
                if s > 2:
                    return 0  # Excellent
                elif s > 1:
                    return 1  # Good
                elif s > 0:
                    return 2  # Moderate
                else:
                    return 3  # Poor

            family = features[(symbol, f"sortino_{w}d")].apply(classify_family)
            features[(symbol, f"family_{w}d")] = family

            # Max drawdown (rolling)
            cumret = (1 + r).cumprod()
            roll_max = cumret.rolling(w, min_periods=1).max()
            drawdown = (cumret - roll_max) / roll_max
            features[(symbol, f"max_dd_{w}d")] = drawdown.rolling(w).min()

    feature_df = pd.DataFrame(features)
    feature_df.columns = pd.MultiIndex.from_tuples(
        feature_df.columns, names=["symbol", "feature"]
    )

    return feature_df


def generate_prediction(
    capital: float = 100_000,
    n_top: int = 10,
    output_dir: Optional[Path] = None,
    only_good_families: bool = True,
) -> dict:
    """Generate weekly prediction using trained model."""
    if output_dir is None:
        output_dir = PRODUCTION_DIR / "outputs" / "recommendations"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Import recommendation logic
    from production.scripts.recommend_top10 import (
        get_top_n_portfolio,
        compute_sortino,
    )
    from production.config.universe import TRADING_UNIVERSE

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from src.environment.gpu_vec_env import GPUVecTradingEnv, GPUEnvConfig
    from src.environment.universe_encoder import FamilyEncoder

    # Find latest model
    latest_file = PRODUCTION_DIR / "models" / "latest_run.txt"
    if not latest_file.exists():
        raise FileNotFoundError("No trained model found. Run training first.")

    model_name = latest_file.read_text().strip()
    model_dir = PRODUCTION_DIR / "models" / model_name

    logger.info(f"Using model: {model_name}")

    # Load data
    data_dir = PRODUCTION_DIR / "data" / "processed"
    returns = pd.read_parquet(data_dir / "returns.parquet")
    features = pd.read_parquet(data_dir / "features.parquet")

    logger.info(f"Data: {returns.shape[0]} days x {returns.shape[1]} assets")
    logger.info(f"Latest date: {returns.index[-1].strftime('%Y-%m-%d')}")

    # Filter to trading universe
    available_assets = [a for a in TRADING_UNIVERSE if a in returns.columns]
    logger.info(f"Trading universe: {len(available_assets)} assets")

    # Get family labels for ALL assets (needed for encoder)
    all_family_labels = {}
    for col in features.columns:
        if col[1] == "family_63d":
            asset = col[0]
            if asset != "_cross":
                val = features[col].dropna().iloc[-1] if len(features[col].dropna()) > 0 else 2
                all_family_labels[asset] = int(val)

    # Compute Sortino for assets without labels
    recent_sortino = compute_sortino(returns, window=63)
    for asset in returns.columns:
        if asset not in all_family_labels:
            s = recent_sortino.get(asset, 0)
            if s > 2:
                all_family_labels[asset] = 0
            elif s > 1:
                all_family_labels[asset] = 1
            elif s > 0:
                all_family_labels[asset] = 2
            else:
                all_family_labels[asset] = 3

    all_family_labels_series = pd.Series(all_family_labels)

    # Filter to trading universe for portfolio selection
    trading_family_labels = pd.Series({
        a: all_family_labels[a] for a in available_assets if a in all_family_labels
    })

    # Load model and encoder
    model_path = model_dir / "checkpoints" / "ppo" / "best_model.zip"
    encoder_path = model_dir / "checkpoints" / "ppo" / "universe_encoder.pkl"

    if not model_path.exists():
        model_path = model_dir / "checkpoints" / "ppo" / "final_model.zip"

    # Load model on CPU for inference (faster for small batches)
    model = PPO.load(str(model_path), device="cpu")
    logger.info("Model loaded")

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    logger.info(f"Encoder loaded: {encoder._n_families} families")

    # Align returns columns with encoder's trained columns
    # This handles the case where new assets were added after model training
    if hasattr(encoder, '_trained_columns') and encoder._trained_columns is not None:
        trained_cols = encoder._trained_columns
        # Find columns that exist in both
        common_cols = [c for c in trained_cols if c in returns.columns]
        if len(common_cols) < len(trained_cols):
            logger.warning(
                f"Missing {len(trained_cols) - len(common_cols)} columns from training data. "
                f"Reindexing to {len(trained_cols)} trained columns."
            )
        # Reindex returns to match trained columns (fill missing with NaN)
        returns = returns.reindex(columns=trained_cols)
        logger.info(f"Aligned returns to encoder: {returns.shape[1]} columns")
    else:
        # Fallback for older encoders without _trained_columns
        logger.warning(
            "Encoder was trained with older version (no _trained_columns). "
            "Attempting to infer columns from family_labels..."
        )
        # Try to use family_labels index to infer trained columns
        if hasattr(encoder, 'family_labels') and encoder.family_labels is not None:
            trained_cols = list(encoder.family_labels.index)
            if len(trained_cols) == encoder._n_total_algos:
                # Reindex returns to match family_labels index
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

    # Create environment with SAME configuration as training (GPUVecTradingEnv + FamilyEncoder)
    n_assets = returns.shape[1]
    benchmark_weights = pd.DataFrame(
        np.ones((len(returns), n_assets)) / n_assets,
        index=returns.index,
        columns=returns.columns,
    )

    # Create GPU config for inference
    gpu_config = GPUEnvConfig(
        initial_capital=1_000_000.0,
        rebalance_frequency="weekly",
        episode_length=52,
        random_start=False,  # Deterministic for inference
        max_weight=0.40,
        min_weight=0.0,
        max_turnover=0.30,
        max_exposure=1.0,
        spread_bps=5.0,
        slippage_bps=2.0,
        impact_coefficient=0.1,
        reward_scale=100.0,
        cost_penalty=1.0,
        turnover_penalty=0.1,
        drawdown_penalty=0.5,
        bad_family_penalty=2.0,
    )

    # Create GPU environment (uses same obs space as training)
    vec_env = GPUVecTradingEnv(
        n_envs=1,
        algo_returns=returns,
        benchmark_weights=benchmark_weights,
        family_encoder=encoder,
        train_start=returns.index[0],
        train_end=returns.index[-1],
        config=gpu_config,
        device="cpu",  # Use CPU for inference
    )

    # Load VecNormalize if available
    vec_norm_path = model_dir / "checkpoints" / "ppo" / "vecnormalize.pkl"
    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Get prediction
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    family_weights = action[0]
    n_families = len(family_weights)

    logger.info(f"RL Family weights ({n_families} families): {family_weights}")

    # Pad family_weights to 4 if model was trained with fewer families
    full_family_weights = np.zeros(4)
    full_family_weights[:n_families] = family_weights

    # Get top N portfolio
    portfolio = get_top_n_portfolio(
        returns=returns[available_assets],
        family_weights=full_family_weights,
        family_labels=trading_family_labels,
        n_top=n_top,
        only_good_families=only_good_families,
    )

    # Build recommendation
    current_date = returns.index[-1]
    recommendation = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "data_date": current_date.strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "model": model_name,
        "n_positions": n_top,
        "capital": capital,
        "n_families_trained": n_families,
        "family_weights": {
            "excellent_sortino_gt2": float(full_family_weights[0]),
            "good_sortino_1to2": float(full_family_weights[1]),
            "moderate_sortino_0to1": float(full_family_weights[2]),
            "poor_sortino_lt0": float(full_family_weights[3]),
        },
        "positions": [],
    }

    # Sort by weight
    sorted_positions = sorted(portfolio.items(), key=lambda x: x[1]["weight"], reverse=True)

    for asset, info in sorted_positions:
        recommendation["positions"].append({
            "asset": str(asset),
            "weight": float(info["weight"]),
            "weight_pct": float(info["weight"] * 100),
            "dollar_amount": float(info["weight"] * capital),
            "family": int(info["family"]),
            "sortino_63d": float(info["sortino"]),
        })

    # Save recommendation
    output_file = output_dir / f"top{n_top}_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(output_file, "w") as f:
        json.dump(recommendation, f, indent=2)

    logger.info(f"Saved recommendation to {output_file}")

    return recommendation


def retrain_model(
    timesteps: int = 500_000,
    n_envs: int = 12,
    use_gpu: bool = True,
) -> Path:
    """Retrain the model with all available data."""
    logger.info("Starting model retraining...")

    # Import training function
    from production.scripts.train import train_agent, load_config

    # Load config
    config_path = PRODUCTION_DIR / "config" / "production.yaml"
    config = load_config(config_path)

    # Load returns
    returns_path = PRODUCTION_DIR / "data" / "processed" / "returns.parquet"
    returns = pd.read_parquet(returns_path)

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ppo"
    output_dir = PRODUCTION_DIR / "models" / run_id

    # Train
    train_agent(
        returns=returns,
        config=config,
        output_dir=output_dir,
        agent_type="ppo",
        total_timesteps=timesteps,
        n_envs=n_envs,
        use_gpu_env=use_gpu,
    )

    return output_dir


def update_from_sharadar(
    tables: Optional[list[str]] = None,
) -> dict:
    """
    Download updates from Nasdaq Data Link (Sharadar).

    Parameters
    ----------
    tables : list[str], optional
        Tables to update. Default: SEP, DAILY (prices and metrics)

    Returns
    -------
    dict
        Update results
    """
    from production.scripts.sharadar_updater import (
        get_api_key,
        update_all_tables,
        DEFAULT_DUCKDB_PATH as SHARADAR_DB_PATH,
    )

    if tables is None:
        # Default: update prices and daily metrics
        tables = ["SEP", "DAILY"]

    api_key = get_api_key()
    results = update_all_tables(
        api_key=api_key,
        db_path=SHARADAR_DB_PATH,
        tables=tables,
        full_refresh=False,
        recalculate=True,
    )

    return results


def run_weekly_pipeline(
    csv_path: Optional[Path] = None,
    use_sharadar: bool = False,
    sharadar_tables: Optional[list[str]] = None,
    retrain_every_n_weeks: int = 4,
    force_retrain: bool = False,
    skip_predict: bool = False,
    capital: float = 100_000,
    n_top: int = 10,
    retrain_timesteps: int = 500_000,
    retrain_envs: int = 12,
    use_gpu: bool = True,
) -> dict:
    """
    Run the complete weekly pipeline.

    Parameters
    ----------
    csv_path : Path, optional
        Path to weekly CSV with new quotes (alternative to Sharadar)
    use_sharadar : bool
        Download data from Nasdaq Data Link (Sharadar) API
    sharadar_tables : list[str], optional
        Sharadar tables to update (default: SEP, DAILY)
    retrain_every_n_weeks : int
        Retrain model every N weeks
    force_retrain : bool
        Force retraining regardless of schedule
    skip_predict : bool
        Skip prediction step
    capital : float
        Portfolio capital for recommendation
    n_top : int
        Number of top assets in recommendation
    retrain_timesteps : int
        Training timesteps when retraining
    retrain_envs : int
        Number of parallel environments for training
    use_gpu : bool
        Use GPU for training

    Returns
    -------
    dict
        Pipeline results
    """
    state = load_state()
    results = {
        "date": datetime.now().isoformat(),
        "data_updated": False,
        "metrics_updated": False,
        "prediction_generated": False,
        "model_retrained": False,
        "errors": [],
    }

    print("\n" + "=" * 60)
    print(f"WEEKLY PIPELINE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Step 1: Update data (Sharadar API or CSV)
    if use_sharadar:
        print("\n[1/4] Downloading from Nasdaq Data Link (Sharadar)...")
        try:
            sharadar_results = update_from_sharadar(tables=sharadar_tables)
            results["data_updated"] = sharadar_results.get("success", False)
            results["sharadar_tables"] = list(sharadar_results.get("tables", {}).keys())
            total_added = sum(
                r.get("rows_added", 0)
                for r in sharadar_results.get("tables", {}).values()
            )
            print(f"✓ Downloaded {total_added:,} new rows")
        except Exception as e:
            logger.error(f"Failed to download from Sharadar: {e}")
            results["errors"].append(f"Sharadar error: {str(e)}")
            raise
    elif csv_path:
        print("\n[1/4] Loading CSV data into DuckDB...")
        try:
            rows_inserted = load_csv_to_duckdb(csv_path)
            results["data_updated"] = True
            results["rows_inserted"] = rows_inserted
            print(f"✓ Loaded {rows_inserted} rows")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            results["errors"].append(f"CSV load error: {str(e)}")
            raise
    else:
        raise ValueError("Either --sharadar or --csv must be specified")

    # Step 2: Calculate/export metrics for training
    print("\n[2/4] Preparing training data...")
    try:
        returns, features = calculate_metrics()
        results["metrics_updated"] = True
        results["n_assets"] = returns.shape[1]
        results["n_dates"] = returns.shape[0]
        print(f"✓ Exported {returns.shape[1]} assets, {returns.shape[0]} dates")
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        results["errors"].append(f"Metrics error: {str(e)}")
        raise

    # Step 3: Check if retraining needed
    state["weeks_since_retrain"] += 1
    needs_retrain = force_retrain or (state["weeks_since_retrain"] >= retrain_every_n_weeks)

    # Check if model exists
    latest_file = PRODUCTION_DIR / "models" / "latest_run.txt"
    if not latest_file.exists():
        logger.info("No existing model found - training required")
        needs_retrain = True

    if needs_retrain:
        print(f"\n[3/4] Retraining model (every {retrain_every_n_weeks} weeks)...")
        try:
            model_dir = retrain_model(
                timesteps=retrain_timesteps,
                n_envs=retrain_envs,
                use_gpu=use_gpu,
            )
            results["model_retrained"] = True
            results["model_dir"] = str(model_dir)
            state["last_retrain"] = datetime.now().isoformat()
            state["weeks_since_retrain"] = 0
            print(f"✓ Model retrained: {model_dir.name}")
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            results["errors"].append(f"Retrain error: {str(e)}")
            # Continue to prediction if possible
    else:
        print(f"\n[3/4] Skipping retrain (week {state['weeks_since_retrain']}/{retrain_every_n_weeks})")

    # Step 4: Generate prediction
    if not skip_predict:
        print("\n[4/4] Generating prediction...")
        try:
            recommendation = generate_prediction(
                capital=capital,
                n_top=n_top,
            )
            results["prediction_generated"] = True
            results["prediction_file"] = f"top{n_top}_{datetime.now().strftime('%Y-%m-%d')}.json"

            # Display recommendation
            print("\n" + "-" * 60)
            print(f"TOP {n_top} RECOMMENDATIONS")
            print("-" * 60)
            print(f"Data date: {recommendation['data_date']}")
            print(f"Capital: ${capital:,.0f}\n")

            for i, pos in enumerate(recommendation["positions"], 1):
                family_names = ["Excellent", "Good", "Moderate", "Poor"]
                print(f"{i:2}. {pos['asset']:<6} {pos['weight_pct']:>6.1f}%  ${pos['dollar_amount']:>10,.0f}  {family_names[pos['family']]:<10} Sortino: {pos['sortino_63d']:.2f}")

            print("-" * 60)

        except Exception as e:
            logger.error(f"Failed to generate prediction: {e}")
            results["errors"].append(f"Prediction error: {str(e)}")
    else:
        print("\n[4/4] Skipping prediction (--no-predict)")

    # Update state
    state["last_run"] = datetime.now().isoformat()
    state["total_runs"] += 1
    state["history"].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "retrain+predict" if results["model_retrained"] else "predict",
        "status": "success" if not results["errors"] else "partial",
    })
    # Keep only last 50 history entries
    state["history"] = state["history"][-50:]
    save_state(state)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Data updated: {'✓' if results['data_updated'] else '✗'}")
    print(f"Metrics exported: {'✓' if results['metrics_updated'] else '✗'}")
    print(f"Model retrained: {'✓' if results['model_retrained'] else '-'}")
    print(f"Prediction generated: {'✓' if results['prediction_generated'] else '✗'}")
    if results["errors"]:
        print(f"\nErrors: {len(results['errors'])}")
        for err in results["errors"]:
            print(f"  - {err}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Weekly production pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download from Sharadar and predict (RECOMMENDED)
    python weekly_pipeline.py --sharadar

    # Download all Sharadar tables
    python weekly_pipeline.py --sharadar --sharadar-tables SEP SFP SF1 DAILY

    # Use CSV instead of Sharadar
    python weekly_pipeline.py --csv weekly_quotes.csv

    # Force retrain
    python weekly_pipeline.py --sharadar --force-retrain

    # Just update data, no prediction
    python weekly_pipeline.py --sharadar --no-predict

    # Check status
    python weekly_pipeline.py --status

Environment:
    Set NASDAQ_DATA_LINK_API_KEY or create production/config/api_keys.yaml
        """,
    )

    # Data source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--sharadar",
        action="store_true",
        help="Download data from Nasdaq Data Link (Sharadar) API",
    )
    source_group.add_argument(
        "--csv",
        type=Path,
        help="Path to weekly CSV file with quotes",
    )

    parser.add_argument(
        "--sharadar-tables",
        nargs="+",
        default=["SEP", "DAILY"],
        help="Sharadar tables to update (default: SEP DAILY)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force model retraining",
    )
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Skip prediction generation",
    )
    parser.add_argument(
        "--retrain-weeks",
        type=int,
        default=4,
        help="Retrain model every N weeks (default: 4)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Portfolio capital (default: 100,000)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top assets (default: 10)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Training timesteps when retraining (default: 500,000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=12,
        help="Parallel environments for training (default: 12)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU training",
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

    if args.status:
        show_status()
        return

    if not args.sharadar and not args.csv:
        parser.error("Either --sharadar or --csv is required (use --status to check status)")

    if args.csv and not args.csv.exists():
        logger.error(f"CSV file not found: {args.csv}")
        sys.exit(1)

    # Run pipeline
    try:
        results = run_weekly_pipeline(
            csv_path=args.csv,
            use_sharadar=args.sharadar,
            sharadar_tables=args.sharadar_tables,
            retrain_every_n_weeks=args.retrain_weeks,
            force_retrain=args.force_retrain,
            skip_predict=args.no_predict,
            capital=args.capital,
            n_top=args.top,
            retrain_timesteps=args.timesteps,
            retrain_envs=args.n_envs,
            use_gpu=not args.no_gpu,
        )

        if results["errors"]:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
