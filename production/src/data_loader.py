"""
Production Data Loader
======================
Loads and preprocesses OHLCV data from CSV files for production use.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_date_column(df: pd.DataFrame) -> str:
    """Detect the date column in a DataFrame (case-insensitive)."""
    candidates = ["date", "datetime", "time", "timestamp", "index"]
    cols_lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in cols_lower:
            return cols_lower[candidate]

    # Try first column if it looks like dates
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col].head())
        return first_col
    except Exception:
        pass

    raise ValueError(f"Could not detect date column. Columns: {list(df.columns)}")


def detect_ohlcv_columns(df: pd.DataFrame) -> dict:
    """Detect OHLCV columns in a DataFrame (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}

    mapping = {}
    for col_type in ["open", "high", "low", "close", "volume"]:
        # Try exact match first
        if col_type in cols_lower:
            mapping[col_type] = cols_lower[col_type]
        # Try abbreviated versions
        elif col_type[0] in cols_lower:
            mapping[col_type] = cols_lower[col_type[0]]
        # Try with prefixes
        else:
            for key, val in cols_lower.items():
                if col_type in key:
                    mapping[col_type] = val
                    break

    if "close" not in mapping:
        raise ValueError(f"Could not detect 'close' column. Columns: {list(df.columns)}")

    return mapping


def load_single_asset(
    filepath: Path,
    date_column: Optional[str] = None,
    min_history_days: int = 252,
) -> Optional[pd.DataFrame]:
    """
    Load a single asset's OHLCV data from CSV.

    Parameters
    ----------
    filepath : Path
        Path to CSV file
    date_column : str, optional
        Name of date column (auto-detected if None)
    min_history_days : int
        Minimum required trading days

    Returns
    -------
    pd.DataFrame or None
        DataFrame with DatetimeIndex and OHLCV columns, or None if invalid
    """
    try:
        df = pd.read_csv(filepath)

        if df.empty or len(df) < min_history_days:
            logger.warning(f"Skipping {filepath.name}: insufficient data ({len(df)} rows)")
            return None

        # Detect and parse date column
        date_col = date_column or detect_date_column(df)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Detect OHLCV columns
        ohlcv_map = detect_ohlcv_columns(df)

        # Rename to standard names
        rename_map = {v: k for k, v in ohlcv_map.items()}
        df = df.rename(columns=rename_map)

        # Keep only OHLCV columns
        cols_to_keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[cols_to_keep]

        # Ensure numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Remove rows with NaN close
        df = df.dropna(subset=["close"])

        if len(df) < min_history_days:
            logger.warning(f"Skipping {filepath.name}: insufficient data after cleaning ({len(df)} rows)")
            return None

        # Add asset name
        df.index.name = "date"

        return df

    except Exception as e:
        logger.error(f"Error loading {filepath.name}: {e}")
        return None


def load_all_assets(
    data_dir: Path,
    min_history_days: int = 252,
) -> dict[str, pd.DataFrame]:
    """
    Load all OHLCV CSVs from a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing CSV files
    min_history_days : int
        Minimum required trading days per asset

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping asset names to DataFrames
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    assets = {}
    for filepath in sorted(csv_files):
        asset_name = filepath.stem  # filename without extension
        df = load_single_asset(filepath, min_history_days=min_history_days)
        if df is not None:
            assets[asset_name] = df
            logger.debug(f"Loaded {asset_name}: {len(df)} rows, {df.index.min()} to {df.index.max()}")

    logger.info(f"Successfully loaded {len(assets)} assets")

    return assets


def compute_returns(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute daily returns matrix from asset price data.

    Parameters
    ----------
    assets : dict[str, pd.DataFrame]
        Dictionary of asset DataFrames with 'close' column

    Returns
    -------
    pd.DataFrame
        Matrix of daily returns [dates x assets]
    """
    # Extract close prices
    closes = {}
    for name, df in assets.items():
        closes[name] = df["close"]

    # Combine into single DataFrame
    prices = pd.DataFrame(closes)

    # Compute returns
    returns = prices.pct_change()

    # Drop first row (NaN from pct_change)
    returns = returns.iloc[1:]

    # Replace inf with NaN
    returns = returns.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Computed returns: {returns.shape[0]} dates x {returns.shape[1]} assets")

    return returns


def align_to_common_dates(returns: pd.DataFrame, min_coverage: float = 0.8) -> pd.DataFrame:
    """
    Align returns to common trading dates.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix [dates x assets]
    min_coverage : float
        Minimum fraction of assets required per date

    Returns
    -------
    pd.DataFrame
        Aligned returns matrix
    """
    n_assets = returns.shape[1]
    min_assets = int(n_assets * min_coverage)

    # Count non-NaN values per row
    valid_counts = returns.notna().sum(axis=1)

    # Keep dates with sufficient coverage
    mask = valid_counts >= min_assets
    aligned = returns.loc[mask].copy()

    logger.info(f"Aligned to {len(aligned)} common dates (min {min_assets} assets per date)")

    return aligned


def prepare_production_data(
    raw_dir: Path,
    output_dir: Path,
    min_history_days: int = 252,
    min_coverage: float = 0.8,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full pipeline: load raw CSVs, compute returns, save processed data.

    Parameters
    ----------
    raw_dir : Path
        Directory with raw OHLCV CSVs
    output_dir : Path
        Directory to save processed data
    min_history_days : int
        Minimum trading days per asset
    min_coverage : float
        Minimum asset coverage per date

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Returns matrix and list of asset names
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load assets
    assets = load_all_assets(raw_dir, min_history_days=min_history_days)

    if not assets:
        raise ValueError("No valid assets loaded")

    # Compute returns
    returns = compute_returns(assets)

    # Align to common dates
    returns = align_to_common_dates(returns, min_coverage=min_coverage)

    # Save
    returns_path = output_dir / "returns.parquet"
    returns.to_parquet(returns_path)
    logger.info(f"Saved returns to {returns_path}")

    # Save asset list
    asset_names = list(returns.columns)
    asset_list_path = output_dir / "assets.txt"
    asset_list_path.write_text("\n".join(asset_names))
    logger.info(f"Saved asset list to {asset_list_path}")

    # Save summary stats
    stats = pd.DataFrame({
        "mean_return": returns.mean(),
        "std_return": returns.std(),
        "sharpe": returns.mean() / returns.std() * np.sqrt(252),
        "min_return": returns.min(),
        "max_return": returns.max(),
        "n_observations": returns.notna().sum(),
    })
    stats_path = output_dir / "asset_stats.csv"
    stats.to_csv(stats_path)
    logger.info(f"Saved asset stats to {stats_path}")

    return returns, asset_names


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Default paths relative to production/
    production_dir = Path(__file__).parent.parent
    raw_dir = production_dir / "data" / "raw"
    output_dir = production_dir / "data" / "processed"

    if len(sys.argv) > 1:
        raw_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    returns, assets = prepare_production_data(raw_dir, output_dir)
    print(f"\nProcessed {len(assets)} assets, {len(returns)} trading days")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
