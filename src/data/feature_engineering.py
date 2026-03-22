"""
Feature engineering para estado del agente RL.

New feature specification:
- Cumulative features from start
- Rolling features with windows [5, 21, 63] days
- Regime features (cross-sectional market stats)

Performance: Uses numba @njit for vectorized calculations when available,
falls back to pure numpy otherwise.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# Number of parallel jobs (default: use all CPUs minus 1)
N_JOBS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4

# Try to import numba, fall back to identity decorator if not available
try:
    from numba import njit
except ImportError:
    logger.info("Numba not available, using pure numpy implementations")
    # Create a no-op decorator that mimics njit
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

# Rolling windows (trading days): 1 week, 1 month, 3 months
ROLLING_WINDOWS = [5, 21, 63]

# Annualization factor
TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Memory-efficient helper functions
# =============================================================================

def _replace_inf_inplace(df: pd.DataFrame) -> None:
    """
    Replace inf/-inf values with NaN in-place.

    This avoids creating a copy of the DataFrame like .replace() does.
    """
    for col in df.columns:
        arr = df[col].values
        # Get mask of inf values
        mask = np.isinf(arr)
        if mask.any():
            # Set inf values to NaN in-place
            arr[mask] = np.nan


# =============================================================================
# Numba JIT-compiled functions for performance
# =============================================================================

@njit
def _cumulative_return(prices: np.ndarray) -> np.ndarray:
    """Cumulative return from start."""
    n = len(prices)
    result = np.empty(n)
    if n == 0:
        return result

    p0 = prices[0]
    for i in range(n):
        result[i] = (prices[i] / p0) - 1.0 if p0 != 0 else 0.0
    return result


@njit
def _rolling_return(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling return over window."""
    n = len(prices)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window, n):
        p_prev = prices[i - window]
        if p_prev != 0:
            result[i] = (prices[i] / p_prev) - 1.0
        else:
            result[i] = 0.0
    return result


@njit
def _cumulative_volatility(returns: np.ndarray) -> np.ndarray:
    """Cumulative volatility (annualized) from start."""
    n = len(returns)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(1, n):
        # Compute std of returns[0:i+1]
        window = returns[:i + 1]
        mean = 0.0
        for j in range(len(window)):
            mean += window[j]
        mean /= len(window)

        var = 0.0
        for j in range(len(window)):
            diff = window[j] - mean
            var += diff * diff
        var /= len(window)

        result[i] = np.sqrt(var) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return result


@njit
def _rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling volatility (annualized)."""
    n = len(returns)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        # Compute std of returns[i-window+1:i+1]
        start = i - window + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += returns[j]
        mean /= window

        var = 0.0
        for j in range(start, i + 1):
            diff = returns[j] - mean
            var += diff * diff
        var /= window

        result[i] = np.sqrt(var) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return result


@njit
def _cumulative_drawdown(prices: np.ndarray) -> np.ndarray:
    """Current drawdown from running maximum."""
    n = len(prices)
    result = np.empty(n)

    running_max = prices[0]
    for i in range(n):
        if prices[i] > running_max:
            running_max = prices[i]
        if running_max != 0:
            result[i] = (prices[i] - running_max) / running_max
        else:
            result[i] = 0.0
    return result


@njit
def _rolling_max_drawdown(prices: np.ndarray, window: int) -> np.ndarray:
    """Maximum drawdown within rolling window."""
    n = len(prices)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        start = i - window + 1
        # Find max drawdown in window
        max_dd = 0.0
        running_max = prices[start]

        for j in range(start, i + 1):
            if prices[j] > running_max:
                running_max = prices[j]
            if running_max != 0:
                dd = (prices[j] - running_max) / running_max
                if dd < max_dd:
                    max_dd = dd

        result[i] = max_dd  # negative value

    return result


@njit
def _cumulative_sharpe(returns: np.ndarray) -> np.ndarray:
    """Cumulative Sharpe ratio (annualized) from start."""
    n = len(returns)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(1, n):
        window = returns[:i + 1]
        mean = 0.0
        for j in range(len(window)):
            mean += window[j]
        mean /= len(window)

        var = 0.0
        for j in range(len(window)):
            diff = window[j] - mean
            var += diff * diff
        var /= len(window)

        std = np.sqrt(var)
        if std > 1e-10:
            result[i] = (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            result[i] = 0.0

    return result


@njit
def _rolling_sharpe(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling Sharpe ratio (annualized)."""
    n = len(returns)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        start = i - window + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += returns[j]
        mean /= window

        var = 0.0
        for j in range(start, i + 1):
            diff = returns[j] - mean
            var += diff * diff
        var /= window

        std = np.sqrt(var)
        if std > 1e-10:
            result[i] = (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            result[i] = 0.0

    return result


@njit
def _cumulative_profit_factor(returns: np.ndarray) -> np.ndarray:
    """Cumulative profit factor (gross profit / gross loss) from start."""
    n = len(returns)
    result = np.empty(n)
    result[:] = np.nan

    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(n):
        if returns[i] > 0:
            gross_profit += returns[i]
        elif returns[i] < 0:
            gross_loss += abs(returns[i])

        if gross_loss > 1e-10:
            result[i] = gross_profit / gross_loss
        else:
            result[i] = np.nan if gross_profit == 0 else np.inf

    return result


@njit
def _rolling_profit_factor(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling profit factor."""
    n = len(returns)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        start = i - window + 1
        gross_profit = 0.0
        gross_loss = 0.0

        for j in range(start, i + 1):
            if returns[j] > 0:
                gross_profit += returns[j]
            elif returns[j] < 0:
                gross_loss += abs(returns[j])

        if gross_loss > 1e-10:
            result[i] = gross_profit / gross_loss
        else:
            result[i] = np.nan if gross_profit == 0 else np.inf

    return result


@njit
def _cumulative_calmar(returns: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Cumulative Calmar ratio (annualized return / max drawdown) from start."""
    n = len(returns)
    result = np.empty(n)
    result[:] = np.nan

    running_max = prices[0]
    max_dd = 0.0  # Most negative drawdown

    for i in range(1, n):
        # Update running max and max drawdown
        if prices[i] > running_max:
            running_max = prices[i]
        if running_max != 0:
            dd = (prices[i] - running_max) / running_max
            if dd < max_dd:
                max_dd = dd

        # Compute annualized return
        if i > 0 and prices[0] != 0:
            total_return = (prices[i] / prices[0]) - 1.0
            years = (i + 1) / TRADING_DAYS_PER_YEAR
            if years > 0:
                ann_return = (1 + total_return) ** (1 / years) - 1
            else:
                ann_return = 0.0
        else:
            ann_return = 0.0

        # Calmar = ann_return / abs(max_dd)
        if abs(max_dd) > 1e-10:
            result[i] = ann_return / abs(max_dd)
        else:
            result[i] = 0.0

    return result


@njit
def _rolling_calmar(returns: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling Calmar ratio."""
    n = len(prices)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        start = i - window + 1

        # Find max drawdown in window
        running_max = prices[start]
        max_dd = 0.0

        for j in range(start, i + 1):
            if prices[j] > running_max:
                running_max = prices[j]
            if running_max != 0:
                dd = (prices[j] - running_max) / running_max
                if dd < max_dd:
                    max_dd = dd

        # Compute return over window
        if prices[start] != 0:
            window_return = (prices[i] / prices[start]) - 1.0
            # Annualize
            years = window / TRADING_DAYS_PER_YEAR
            if years > 0 and window_return > -1:
                ann_return = (1 + window_return) ** (1 / years) - 1
            else:
                ann_return = 0.0
        else:
            ann_return = 0.0

        if abs(max_dd) > 1e-10:
            result[i] = ann_return / abs(max_dd)
        else:
            result[i] = 0.0

    return result


@njit
def _max_drawdown_to_date(prices: np.ndarray) -> np.ndarray:
    """Maximum drawdown up to each point."""
    n = len(prices)
    result = np.empty(n)

    running_max = prices[0]
    max_dd = 0.0

    for i in range(n):
        if prices[i] > running_max:
            running_max = prices[i]
        if running_max != 0:
            dd = (prices[i] - running_max) / running_max
            if dd < max_dd:
                max_dd = dd
        result[i] = max_dd

    return result


# =============================================================================
# FeatureEngineer class
# =============================================================================

class FeatureEngineer:
    """
    Calcula features rolling y cumulativos para cada algoritmo.

    Features:
    - Cumulative: return, volatility, drawdown, sharpe, profit_factor, calmar, max_drawdown
    - Rolling (per window): return, volatility, drawdown, sharpe, profit_factor, calmar
    - Regime: market_vol, market_return (cross-sectional)
    """

    def __init__(
        self,
        windows: Optional[list[int]] = None,
    ):
        self.windows = windows or ROLLING_WINDOWS

    def compute_cumulative_features(
        self,
        returns: pd.Series,
        algo_id: str = "",
    ) -> pd.DataFrame:
        """
        Compute cumulative features from start.

        Args:
            returns: Series of daily returns.
            algo_id: Algorithm identifier for column naming.

        Returns:
            DataFrame with cumulative features.
        """
        prefix = f"{algo_id}_" if algo_id else ""
        features = pd.DataFrame(index=returns.index)

        # Build prices from returns
        prices = (1 + returns).cumprod().values
        ret_vals = returns.values

        # Cumulative return
        features[f"{prefix}return"] = _cumulative_return(prices)

        # Cumulative volatility
        features[f"{prefix}volatility"] = _cumulative_volatility(ret_vals)

        # Current drawdown
        features[f"{prefix}drawdown"] = _cumulative_drawdown(prices)

        # Cumulative Sharpe
        features[f"{prefix}sharpe"] = _cumulative_sharpe(ret_vals)

        # Cumulative profit factor
        features[f"{prefix}profit_factor"] = _cumulative_profit_factor(ret_vals)

        # Cumulative Calmar
        features[f"{prefix}calmar_ratio"] = _cumulative_calmar(ret_vals, prices)

        # Max drawdown to date
        features[f"{prefix}max_drawdown"] = _max_drawdown_to_date(prices)

        return features

    def compute_rolling_features(
        self,
        returns: pd.Series,
        algo_id: str = "",
    ) -> pd.DataFrame:
        """
        Compute rolling features for all windows.

        Args:
            returns: Series of daily returns.
            algo_id: Algorithm identifier for column naming.

        Returns:
            DataFrame with rolling features.
        """
        prefix = f"{algo_id}_" if algo_id else ""
        features = pd.DataFrame(index=returns.index)

        # Build prices from returns
        prices = (1 + returns).cumprod().values
        ret_vals = returns.values

        for w in self.windows:
            # Rolling return
            features[f"{prefix}rolling_return_{w}d"] = _rolling_return(prices, w)

            # Rolling volatility
            features[f"{prefix}rolling_volatility_{w}d"] = _rolling_volatility(ret_vals, w)

            # Rolling max drawdown
            features[f"{prefix}rolling_drawdown_{w}d"] = _rolling_max_drawdown(prices, w)

            # Rolling Sharpe
            features[f"{prefix}rolling_sharpe_{w}d"] = _rolling_sharpe(ret_vals, w)

            # Rolling profit factor
            features[f"{prefix}rolling_profit_factor_{w}d"] = _rolling_profit_factor(ret_vals, w)

            # Rolling Calmar
            features[f"{prefix}rolling_calmar_{w}d"] = _rolling_calmar(ret_vals, prices, w)

        return features

    def compute_algo_features(
        self,
        returns: pd.Series,
        algo_id: str = "",
    ) -> pd.DataFrame:
        """
        Compute all features for a single algorithm.

        Args:
            returns: Series of daily returns.
            algo_id: Algorithm identifier for column naming.

        Returns:
            DataFrame with all features by date.
        """
        cumulative = self.compute_cumulative_features(returns, algo_id)
        rolling = self.compute_rolling_features(returns, algo_id)

        return pd.concat([cumulative, rolling], axis=1)

    def compute_regime_features(
        self,
        returns_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute cross-sectional market-wide regime features.

        Args:
            returns_matrix: DataFrame [date x algo_id] with returns.

        Returns:
            DataFrame with regime features.
        """
        features = pd.DataFrame(index=returns_matrix.index)

        # For each window, compute cross-sectional market stats
        for w in self.windows:
            # Cross-sectional mean volatility (average of rolling vols across algos)
            rolling_vols = returns_matrix.rolling(w).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            features[f"rolling_market_vol_{w}d"] = rolling_vols.mean(axis=1)

            # Cross-sectional mean return
            rolling_returns = returns_matrix.rolling(w).sum()
            features[f"rolling_market_return_{w}d"] = rolling_returns.mean(axis=1)

        return features

    def build_feature_matrix(
        self,
        returns_matrix: pd.DataFrame,
        show_progress: bool = True,
        n_jobs: Optional[int] = None,
        use_float32: bool = True,
        batch_size: int = 500,
    ) -> pd.DataFrame:
        """
        Build complete feature matrix for RL agent.

        Memory-optimized version that:
        - Uses float32 by default (halves memory)
        - Processes in batches to reduce peak memory
        - Replaces inf values in-place

        Args:
            returns_matrix: DataFrame [date x algo_id] with returns.
            show_progress: Show progress updates.
            n_jobs: Number of parallel jobs (default: N_JOBS global setting).
            use_float32: Use float32 instead of float64 (default True, halves memory).
            batch_size: Number of algorithms to process per batch.

        Returns:
            DataFrame with all features by date.
            Columns: {algo_id}_{feature_name} for each algo,
                     plus regime features.
        """
        import gc

        n_algos = len(returns_matrix.columns)
        n_jobs = n_jobs or N_JOBS
        dtype = np.float32 if use_float32 else np.float64

        if show_progress:
            logger.info(
                f"Building feature matrix for {n_algos} algos "
                f"(dtype={dtype.__name__}, batch_size={batch_size})..."
            )

        # Helper function for single algo (returns float32 arrays)
        def compute_single_algo_features_optimized(algo_id: str) -> Optional[pd.DataFrame]:
            algo_returns = returns_matrix[algo_id].dropna()
            if len(algo_returns) > 0:
                features = self.compute_algo_features(algo_returns, algo_id)
                # Convert to target dtype immediately
                if use_float32:
                    features = features.astype(np.float32, copy=False)
                # Replace inf in-place
                _replace_inf_inplace(features)
                return features
            return None

        # Process in batches to reduce peak memory
        algo_ids = list(returns_matrix.columns)
        n_batches = (n_algos + batch_size - 1) // batch_size

        combined_features = []

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_algos)
            batch_algo_ids = algo_ids[start_idx:end_idx]

            if show_progress and n_batches > 1:
                logger.info(f"  Processing batch {batch_idx + 1}/{n_batches} ({len(batch_algo_ids)} algos)...")

            # Parallel feature computation for batch
            if n_jobs > 1 and len(batch_algo_ids) > 10:
                batch_features = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(compute_single_algo_features_optimized)(algo_id)
                    for algo_id in batch_algo_ids
                )
                batch_features = [f for f in batch_features if f is not None]
            else:
                batch_features = []
                for i, algo_id in enumerate(batch_algo_ids):
                    result = compute_single_algo_features_optimized(algo_id)
                    if result is not None:
                        batch_features.append(result)

            if batch_features:
                # Concat batch and add to combined list
                batch_df = pd.concat(batch_features, axis=1)
                combined_features.append(batch_df)

                # Free memory from individual DataFrames
                del batch_features
                gc.collect()

        if show_progress:
            logger.info(f"  Computed features for {n_batches} batches")

        # Regime features (cross-sectional)
        regime_features = self.compute_regime_features(returns_matrix)
        if use_float32:
            regime_features = regime_features.astype(np.float32, copy=False)
        _replace_inf_inplace(regime_features)
        combined_features.append(regime_features)

        # Final concat of batches (fewer, larger DataFrames)
        if show_progress:
            logger.info("  Concatenating batches...")

        feature_matrix = pd.concat(combined_features, axis=1)

        # Free batch DataFrames
        del combined_features
        gc.collect()

        if show_progress:
            mem_mb = feature_matrix.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                f"Built feature matrix: {feature_matrix.shape[0]} rows, "
                f"{feature_matrix.shape[1]} features ({mem_mb:.1f} MB)"
            )

        return feature_matrix

    def get_feature_names(self, algo_ids: list[str]) -> list[str]:
        """
        Get list of all feature names that will be generated.

        Args:
            algo_ids: List of algorithm IDs.

        Returns:
            List of feature column names.
        """
        feature_names = []

        # Per-algorithm features
        cumulative_features = [
            "return", "volatility", "drawdown", "sharpe",
            "profit_factor", "calmar_ratio", "max_drawdown"
        ]
        rolling_features = [
            "rolling_return", "rolling_volatility", "rolling_drawdown",
            "rolling_sharpe", "rolling_profit_factor", "rolling_calmar"
        ]

        for algo_id in algo_ids:
            for feat in cumulative_features:
                feature_names.append(f"{algo_id}_{feat}")
            for feat in rolling_features:
                for w in self.windows:
                    feature_names.append(f"{algo_id}_{feat}_{w}d")

        # Regime features
        for w in self.windows:
            feature_names.append(f"rolling_market_vol_{w}d")
            feature_names.append(f"rolling_market_return_{w}d")

        return feature_names
