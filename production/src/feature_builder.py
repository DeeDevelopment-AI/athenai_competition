"""
Production Feature Builder
==========================
Self-contained feature computation for production use.

Includes Sortino-based family classification:
- Family 0: Sortino > 2 (excellent risk-adjusted returns)
- Family 1: Sortino 1-2 (good risk-adjusted returns)
- Family 2: Sortino 0-1 (moderate risk-adjusted returns)
- Family 3: Sortino < 0 (poor risk-adjusted returns)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

# Sortino family thresholds
SORTINO_THRESHOLDS = [2.0, 1.0, 0.0]  # Family boundaries


# =============================================================================
# Core Feature Functions (Numba-accelerated with fallbacks)
# =============================================================================

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _cumulative_returns_numba(returns: np.ndarray) -> np.ndarray:
        """Cumulative returns using compounding."""
        n_dates, n_assets = returns.shape
        result = np.zeros((n_dates, n_assets), dtype=np.float32)

        for j in range(n_assets):
            cum = 1.0
            for i in range(n_dates):
                r = returns[i, j]
                if not np.isnan(r):
                    cum *= (1.0 + r)
                result[i, j] = cum - 1.0

        return result

    @numba.njit(cache=True)
    def _rolling_returns_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """Rolling returns over window."""
        n_dates, n_assets = returns.shape
        result = np.full((n_dates, n_assets), np.nan, dtype=np.float32)

        for j in range(n_assets):
            for i in range(window - 1, n_dates):
                cum = 1.0
                valid = True
                for k in range(i - window + 1, i + 1):
                    r = returns[k, j]
                    if np.isnan(r):
                        valid = False
                        break
                    cum *= (1.0 + r)
                if valid:
                    result[i, j] = cum - 1.0

        return result

    @numba.njit(cache=True)
    def _cumulative_volatility_numba(returns: np.ndarray) -> np.ndarray:
        """Expanding window volatility (annualized)."""
        n_dates, n_assets = returns.shape
        result = np.full((n_dates, n_assets), np.nan, dtype=np.float32)

        for j in range(n_assets):
            sum_r = 0.0
            sum_sq = 0.0
            count = 0

            for i in range(n_dates):
                r = returns[i, j]
                if not np.isnan(r):
                    sum_r += r
                    sum_sq += r * r
                    count += 1

                if count >= 2:
                    mean = sum_r / count
                    var = (sum_sq / count) - (mean * mean)
                    if var > 0:
                        result[i, j] = np.sqrt(var * 252)
                    else:
                        result[i, j] = 0.0

        return result

    @numba.njit(cache=True)
    def _rolling_volatility_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """Rolling volatility (annualized)."""
        n_dates, n_assets = returns.shape
        result = np.full((n_dates, n_assets), np.nan, dtype=np.float32)

        for j in range(n_assets):
            for i in range(window - 1, n_dates):
                sum_r = 0.0
                sum_sq = 0.0
                count = 0

                for k in range(i - window + 1, i + 1):
                    r = returns[k, j]
                    if not np.isnan(r):
                        sum_r += r
                        sum_sq += r * r
                        count += 1

                if count >= 2:
                    mean = sum_r / count
                    var = (sum_sq / count) - (mean * mean)
                    if var > 0:
                        result[i, j] = np.sqrt(var * 252)
                    else:
                        result[i, j] = 0.0

        return result

    compute_cumulative_returns = _cumulative_returns_numba
    compute_rolling_returns = _rolling_returns_numba
    compute_cumulative_volatility = _cumulative_volatility_numba
    compute_rolling_volatility = _rolling_volatility_numba

else:
    # NumPy fallbacks
    def compute_cumulative_returns(returns: np.ndarray) -> np.ndarray:
        """Cumulative returns using compounding."""
        returns = np.nan_to_num(returns, nan=0.0)
        cum = np.cumprod(1.0 + returns, axis=0) - 1.0
        return cum.astype(np.float32)

    def compute_rolling_returns(returns: np.ndarray, window: int) -> np.ndarray:
        """Rolling returns over window."""
        df = pd.DataFrame(returns)
        result = df.rolling(window).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )
        return result.values.astype(np.float32)

    def compute_cumulative_volatility(returns: np.ndarray) -> np.ndarray:
        """Expanding window volatility (annualized)."""
        df = pd.DataFrame(returns)
        result = df.expanding().std() * np.sqrt(252)
        return result.values.astype(np.float32)

    def compute_rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
        """Rolling volatility (annualized)."""
        df = pd.DataFrame(returns)
        result = df.rolling(window).std() * np.sqrt(252)
        return result.values.astype(np.float32)


# =============================================================================
# Sortino Ratio Computation
# =============================================================================

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _compute_downside_deviation(returns: np.ndarray, target: float = 0.0) -> float:
        """Compute downside deviation (only negative returns below target)."""
        n = len(returns)
        if n < 2:
            return np.nan

        sum_sq = 0.0
        count = 0
        for i in range(n):
            if not np.isnan(returns[i]):
                diff = min(returns[i] - target, 0.0)
                sum_sq += diff * diff
                count += 1

        if count < 2:
            return np.nan

        return np.sqrt(sum_sq / count)

    @numba.njit(cache=True)
    def _compute_sortino_single(returns: np.ndarray, target: float = 0.0, annualize: bool = True) -> float:
        """Compute Sortino ratio for a single return series."""
        n = len(returns)
        if n < 5:
            return np.nan

        # Mean return
        sum_ret = 0.0
        count = 0
        for i in range(n):
            if not np.isnan(returns[i]):
                sum_ret += returns[i]
                count += 1

        if count < 5:
            return np.nan

        mean_ret = sum_ret / count

        # Downside deviation
        downside_dev = _compute_downside_deviation(returns, target)

        if np.isnan(downside_dev) or downside_dev < 1e-10:
            if mean_ret > target:
                return 10.0
            return 0.0

        sortino = (mean_ret - target) / downside_dev

        if annualize:
            sortino = sortino * np.sqrt(252)

        return max(min(sortino, 10.0), -10.0)

    @numba.njit(cache=True, parallel=True)
    def compute_cumulative_sortino(returns: np.ndarray, target: float = 0.0) -> np.ndarray:
        """Compute cumulative (expanding window) Sortino ratio."""
        n_dates, n_assets = returns.shape
        result = np.full((n_dates, n_assets), np.nan, dtype=np.float32)

        for j in numba.prange(n_assets):
            for i in range(5, n_dates):
                result[i, j] = _compute_sortino_single(returns[:i+1, j], target, True)

        return result

    @numba.njit(cache=True, parallel=True)
    def compute_rolling_sortino(returns: np.ndarray, window: int, target: float = 0.0) -> np.ndarray:
        """Compute rolling Sortino ratio."""
        n_dates, n_assets = returns.shape
        result = np.full((n_dates, n_assets), np.nan, dtype=np.float32)

        for j in numba.prange(n_assets):
            for i in range(window - 1, n_dates):
                start_idx = i - window + 1
                result[i, j] = _compute_sortino_single(returns[start_idx:i+1, j], target, True)

        return result

    @numba.njit(cache=True)
    def classify_sortino_family(sortino: float) -> int:
        """Classify asset into family based on Sortino ratio."""
        if np.isnan(sortino):
            return -1
        if sortino > 2.0:
            return 0
        elif sortino > 1.0:
            return 1
        elif sortino > 0.0:
            return 2
        else:
            return 3

    @numba.njit(cache=True, parallel=True)
    def classify_sortino_families(sortino_matrix: np.ndarray) -> np.ndarray:
        """Classify all assets into families based on Sortino ratios."""
        n_dates, n_assets = sortino_matrix.shape
        result = np.full((n_dates, n_assets), -1, dtype=np.int8)

        for j in numba.prange(n_assets):
            for i in range(n_dates):
                result[i, j] = classify_sortino_family(sortino_matrix[i, j])

        return result

else:
    # NumPy fallbacks for Sortino
    def compute_cumulative_sortino(returns: np.ndarray, target: float = 0.0) -> np.ndarray:
        """Compute cumulative Sortino ratio (NumPy fallback)."""
        n_dates, n_assets = returns.shape
        result = np.full((n_dates, n_assets), np.nan, dtype=np.float32)

        for j in range(n_assets):
            for i in range(5, n_dates):
                rets = returns[:i+1, j]
                rets = rets[~np.isnan(rets)]
                if len(rets) < 5:
                    continue

                mean_ret = np.mean(rets)
                downside = rets[rets < target] - target
                if len(downside) < 2:
                    result[i, j] = 10.0 if mean_ret > target else 0.0
                    continue

                downside_std = np.sqrt(np.mean(downside ** 2))
                if downside_std < 1e-10:
                    result[i, j] = 10.0 if mean_ret > target else 0.0
                else:
                    sortino = (mean_ret - target) / downside_std * np.sqrt(252)
                    result[i, j] = np.clip(sortino, -10.0, 10.0)

        return result

    def compute_rolling_sortino(returns: np.ndarray, window: int, target: float = 0.0) -> np.ndarray:
        """Compute rolling Sortino ratio (NumPy fallback)."""
        n_dates, n_assets = returns.shape
        result = np.full((n_dates, n_assets), np.nan, dtype=np.float32)

        for j in range(n_assets):
            for i in range(window - 1, n_dates):
                rets = returns[i - window + 1:i + 1, j]
                rets = rets[~np.isnan(rets)]
                if len(rets) < 5:
                    continue

                mean_ret = np.mean(rets)
                downside = rets[rets < target] - target
                if len(downside) < 2:
                    result[i, j] = 10.0 if mean_ret > target else 0.0
                    continue

                downside_std = np.sqrt(np.mean(downside ** 2))
                if downside_std < 1e-10:
                    result[i, j] = 10.0 if mean_ret > target else 0.0
                else:
                    sortino = (mean_ret - target) / downside_std * np.sqrt(252)
                    result[i, j] = np.clip(sortino, -10.0, 10.0)

        return result

    def classify_sortino_family(sortino: float) -> int:
        """Classify asset into family based on Sortino ratio."""
        if np.isnan(sortino):
            return -1
        if sortino > 2.0:
            return 0
        elif sortino > 1.0:
            return 1
        elif sortino > 0.0:
            return 2
        else:
            return 3

    def classify_sortino_families(sortino_matrix: np.ndarray) -> np.ndarray:
        """Classify all assets into families."""
        vectorized_classify = np.vectorize(classify_sortino_family)
        return vectorized_classify(sortino_matrix).astype(np.int8)


# =============================================================================
# Main Feature Building Function
# =============================================================================

def build_features(
    returns: pd.DataFrame,
    rolling_windows: list[int] = [5, 21, 63],
    include_sortino_families: bool = True,
) -> pd.DataFrame:
    """
    Build feature matrix from returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix [dates x assets]
    rolling_windows : list[int]
        Windows for rolling calculations (days)
    include_sortino_families : bool
        Whether to include Sortino-based family classification

    Returns
    -------
    pd.DataFrame
        Feature matrix with MultiIndex columns [(asset, feature)]

    Notes
    -----
    Family classification based on Sortino ratio:
    - Family 0: Sortino > 2 (excellent)
    - Family 1: Sortino 1-2 (good)
    - Family 2: Sortino 0-1 (moderate)
    - Family 3: Sortino < 0 (poor)
    - Family -1: Insufficient data
    """
    logger.info(f"Building features for {returns.shape[1]} assets...")

    # Convert to numpy for core functions
    returns_arr = returns.values.astype(np.float32)
    n_dates, n_assets = returns_arr.shape

    features = {}

    # Cumulative features
    logger.debug("Computing cumulative returns...")
    cum_ret = compute_cumulative_returns(returns_arr)

    logger.debug("Computing cumulative volatility...")
    cum_vol = compute_cumulative_volatility(returns_arr)

    for i, asset in enumerate(returns.columns):
        features[(asset, "cum_return")] = cum_ret[:, i]
        features[(asset, "cum_volatility")] = cum_vol[:, i]

    # Rolling features
    for window in rolling_windows:
        logger.debug(f"Computing rolling features (window={window})...")

        roll_ret = compute_rolling_returns(returns_arr, window)
        roll_vol = compute_rolling_volatility(returns_arr, window)

        for i, asset in enumerate(returns.columns):
            features[(asset, f"ret_{window}d")] = roll_ret[:, i]
            features[(asset, f"vol_{window}d")] = roll_vol[:, i]

    # Sortino ratios and family classification
    if include_sortino_families:
        logger.debug("Computing cumulative Sortino ratios...")
        cum_sortino = compute_cumulative_sortino(returns_arr)
        cum_families = classify_sortino_families(cum_sortino)

        for i, asset in enumerate(returns.columns):
            features[(asset, "sortino_cumulative")] = cum_sortino[:, i]
            features[(asset, "family_cumulative")] = cum_families[:, i]

        for window in rolling_windows:
            logger.debug(f"Computing rolling Sortino (window={window})...")
            roll_sortino = compute_rolling_sortino(returns_arr, window)
            roll_families = classify_sortino_families(roll_sortino)

            for i, asset in enumerate(returns.columns):
                features[(asset, f"sortino_{window}d")] = roll_sortino[:, i]
                features[(asset, f"family_{window}d")] = roll_families[:, i]

    # Build DataFrame
    feature_df = pd.DataFrame(features, index=returns.index)
    feature_df.columns = pd.MultiIndex.from_tuples(
        feature_df.columns, names=["asset", "feature"]
    )

    # Cross-sectional features
    logger.debug("Computing cross-sectional features...")
    for window in rolling_windows:
        ret_cols = [(a, f"ret_{window}d") for a in returns.columns]
        ret_matrix = feature_df[ret_cols]

        feature_df[("_cross", f"mean_ret_{window}d")] = ret_matrix.mean(axis=1)
        feature_df[("_cross", f"std_ret_{window}d")] = ret_matrix.std(axis=1)

    # Cross-sectional family distribution
    if include_sortino_families:
        logger.debug("Computing cross-sectional family distributions...")
        for suffix in ["cumulative"] + [f"{w}d" for w in rolling_windows]:
            family_cols = [(a, f"family_{suffix}") for a in returns.columns]
            family_matrix = feature_df[family_cols].values

            for fam_id in range(4):
                counts = (family_matrix == fam_id).sum(axis=1)
                feature_df[("_cross", f"n_family{fam_id}_{suffix}")] = counts

            top_count = ((family_matrix == 0) | (family_matrix == 1)).sum(axis=1)
            feature_df[("_cross", f"frac_top_families_{suffix}")] = top_count / n_assets

    logger.info(f"Built {feature_df.shape[1]} features")

    return feature_df


def build_observation(
    returns: pd.DataFrame,
    current_weights: np.ndarray,
    portfolio_value: float,
    benchmark_value: float,
    rolling_windows: list[int] = [5, 21, 63],
    lookback_days: int = 63,
) -> np.ndarray:
    """Build observation vector for inference."""
    recent_returns = returns.iloc[-lookback_days:]
    returns_arr = recent_returns.values.astype(np.float32)
    n_assets = returns_arr.shape[1]

    obs_parts = [current_weights]

    for window in rolling_windows:
        if window <= len(returns_arr):
            roll_ret = compute_rolling_returns(returns_arr, window)[-1]
            roll_vol = compute_rolling_volatility(returns_arr, window)[-1]
        else:
            roll_ret = np.zeros(n_assets)
            roll_vol = np.zeros(n_assets)

        obs_parts.append(roll_ret)
        obs_parts.append(roll_vol)

    cum_return = portfolio_value - 1.0
    peak = max(1.0, portfolio_value)
    drawdown = (peak - portfolio_value) / peak if peak > 0 else 0.0

    scalars = np.array([portfolio_value, benchmark_value, cum_return, drawdown])
    obs_parts.append(scalars)

    observation = np.concatenate(obs_parts).astype(np.float32)
    observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)

    return observation


def save_features(features: pd.DataFrame, output_path: Path) -> None:
    """Save features to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)
    logger.info(f"Saved features to {output_path}")


def get_family_summary(features: pd.DataFrame, date: pd.Timestamp = None) -> dict:
    """Get summary of family distributions for a specific date."""
    if date is None:
        date = features.index[-1]

    row = features.loc[date]
    assets = [c[0] for c in features.columns if c[0] != "_cross" and c[1] == "family_cumulative"]

    summary = {"date": str(date.date()), "n_assets": len(assets), "windows": {}}

    family_names = {
        0: "excellent (>2)",
        1: "good (1-2)",
        2: "moderate (0-1)",
        3: "poor (<0)",
        -1: "no_data",
    }

    for suffix in ["cumulative", "5d", "21d", "63d"]:
        family_col = f"family_{suffix}"
        families = [int(row[(a, family_col)]) for a in assets if (a, family_col) in row.index]

        counts = {family_names[i]: families.count(i) for i in [-1, 0, 1, 2, 3]}
        total_valid = len(families) - counts["no_data"]

        pcts = {}
        for name, count in counts.items():
            if name != "no_data" and total_valid > 0:
                pcts[name] = round(count / total_valid * 100, 1)

        summary["windows"][suffix] = {"counts": counts, "percentages": pcts}

    return summary


def print_family_summary(features: pd.DataFrame, date: pd.Timestamp = None) -> None:
    """Print formatted family summary."""
    summary = get_family_summary(features, date)

    print(f"\n{'=' * 60}")
    print(f"SORTINO FAMILY DISTRIBUTION - {summary['date']}")
    print(f"{'=' * 60}")
    print(f"Total assets: {summary['n_assets']}")
    print()

    for window, data in summary["windows"].items():
        print(f"  {window.upper():15} | ", end="")
        for fam, pct in data["percentages"].items():
            print(f"{fam}: {pct:5.1f}%  ", end="")
        print()

    print(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    production_dir = Path(__file__).parent.parent
    returns_path = production_dir / "data" / "processed" / "returns.parquet"
    output_path = production_dir / "data" / "processed" / "features.parquet"

    if not returns_path.exists():
        print(f"Error: {returns_path} not found. Run data_loader.py first.")
        sys.exit(1)

    returns = pd.read_parquet(returns_path)
    features = build_features(returns)
    save_features(features, output_path)
    print_family_summary(features)
