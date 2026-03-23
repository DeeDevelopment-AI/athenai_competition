"""
Numba-optimized utility functions for financial calculations.

This module provides high-performance implementations of common financial
calculations using Numba JIT compilation. Falls back to pure NumPy when
Numba is not available.

Usage:
    from src.utils.numba_utils import (
        rolling_mean, rolling_std, rolling_sharpe,
        max_drawdown, max_drawdown_duration,
        portfolio_return, portfolio_variance,
    )

Performance Notes:
- Functions are JIT-compiled on first call (slight delay)
- Subsequent calls are extremely fast (near C speed)
- Use parallel=True functions for large arrays (>10000 elements)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import numba
try:
    from numba import njit, prange, float64, int64, boolean
    HAS_NUMBA = True
    logger.debug("Numba available, using JIT-compiled functions")
except ImportError:
    HAS_NUMBA = False
    logger.info("Numba not available, using pure numpy implementations")

    # Create no-op decorators
    def njit(func=None, parallel=False, cache=True, fastmath=False, **kwargs):
        """No-op njit decorator when numba is not available."""
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

    prange = range
    float64 = None
    int64 = None
    boolean = None


# =============================================================================
# Constants
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
EPSILON = 1e-10


# =============================================================================
# Basic Rolling Statistics (Numba-optimized)
# =============================================================================

@njit(cache=True, fastmath=True)
def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling sum with O(n) complexity.

    Uses the sliding window technique for efficiency.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    # Initial sum
    window_sum = 0.0
    for i in range(window):
        window_sum += arr[i]
    result[window - 1] = window_sum

    # Slide the window
    for i in range(window, n):
        window_sum += arr[i] - arr[i - window]
        result[i] = window_sum

    return result


@njit(cache=True, fastmath=True)
def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with O(n) complexity."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    window_sum = 0.0
    for i in range(window):
        window_sum += arr[i]
    result[window - 1] = window_sum / window

    for i in range(window, n):
        window_sum += arr[i] - arr[i - window]
        result[i] = window_sum / window

    return result


@njit(cache=True, fastmath=True)
def rolling_std(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """
    Compute rolling standard deviation using Welford's online algorithm.

    This is numerically stable and efficient.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    for i in range(window - 1, n):
        start = i - window + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += arr[j]
        mean /= window

        var = 0.0
        for j in range(start, i + 1):
            diff = arr[j] - mean
            var += diff * diff

        if ddof == 0:
            var /= window
        else:
            var /= (window - 1)

        result[i] = np.sqrt(var) if var > 0 else 0.0

    return result


@njit(cache=True, fastmath=True)
def rolling_var(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Compute rolling variance."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    for i in range(window - 1, n):
        start = i - window + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += arr[j]
        mean /= window

        var = 0.0
        for j in range(start, i + 1):
            diff = arr[j] - mean
            var += diff * diff

        if ddof == 0:
            result[i] = var / window
        else:
            result[i] = var / (window - 1)

    return result


# =============================================================================
# Financial Metrics (Numba-optimized)
# =============================================================================

@njit(cache=True)
def cumulative_return(prices: np.ndarray) -> np.ndarray:
    """Compute cumulative return from prices."""
    n = len(prices)
    result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    p0 = prices[0]
    if p0 == 0:
        result[:] = 0.0
        return result

    for i in range(n):
        result[i] = (prices[i] / p0) - 1.0

    return result


@njit(cache=True)
def returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from prices."""
    n = len(prices)
    if n < 2:
        return np.empty(0, dtype=np.float64)

    result = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        if prices[i - 1] != 0:
            result[i - 1] = (prices[i] / prices[i - 1]) - 1.0
        else:
            result[i - 1] = 0.0

    return result


@njit(cache=True)
def log_returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from prices."""
    n = len(prices)
    if n < 2:
        return np.empty(0, dtype=np.float64)

    result = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        if prices[i - 1] > 0 and prices[i] > 0:
            result[i - 1] = np.log(prices[i] / prices[i - 1])
        else:
            result[i - 1] = 0.0

    return result


@njit(cache=True)
def drawdown_series(prices: np.ndarray) -> np.ndarray:
    """Compute drawdown series from prices."""
    n = len(prices)
    result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    running_max = prices[0]
    for i in range(n):
        if prices[i] > running_max:
            running_max = prices[i]
        if running_max > 0:
            result[i] = (prices[i] - running_max) / running_max
        else:
            result[i] = 0.0

    return result


@njit(cache=True)
def max_drawdown(prices: np.ndarray) -> float:
    """Compute maximum drawdown from prices."""
    n = len(prices)
    if n == 0:
        return 0.0

    running_max = prices[0]
    max_dd = 0.0

    for i in range(n):
        if prices[i] > running_max:
            running_max = prices[i]
        if running_max > 0:
            dd = (prices[i] - running_max) / running_max
            if dd < max_dd:
                max_dd = dd

    return max_dd


@njit(cache=True)
def max_drawdown_duration(prices: np.ndarray) -> int:
    """
    Compute maximum drawdown duration in periods.

    Returns the longest number of consecutive periods in drawdown.
    """
    n = len(prices)
    if n < 2:
        return 0

    running_max = prices[0]
    current_duration = 0
    max_duration = 0

    for i in range(n):
        if prices[i] >= running_max:
            running_max = prices[i]
            current_duration = 0
        else:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration

    return max_duration


@njit(cache=True)
def rolling_max_drawdown(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling maximum drawdown over window."""
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    for i in range(window - 1, n):
        start = i - window + 1
        running_max = prices[start]
        max_dd = 0.0

        for j in range(start, i + 1):
            if prices[j] > running_max:
                running_max = prices[j]
            if running_max > 0:
                dd = (prices[j] - running_max) / running_max
                if dd < max_dd:
                    max_dd = dd

        result[i] = max_dd

    return result


@njit(cache=True, fastmath=True)
def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Array of daily returns
        risk_free: Annual risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
    n = len(returns)
    if n < 2:
        return 0.0

    daily_rf = risk_free / TRADING_DAYS_PER_YEAR

    mean = 0.0
    for i in range(n):
        mean += returns[i] - daily_rf
    mean /= n

    var = 0.0
    for i in range(n):
        diff = (returns[i] - daily_rf) - mean
        var += diff * diff
    var /= (n - 1)

    std = np.sqrt(var)
    if std < EPSILON:
        return 0.0

    return (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


@njit(cache=True, fastmath=True)
def rolling_sharpe(returns: np.ndarray, window: int, risk_free: float = 0.0) -> np.ndarray:
    """Compute rolling Sharpe ratio."""
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    daily_rf = risk_free / TRADING_DAYS_PER_YEAR

    for i in range(window - 1, n):
        start = i - window + 1

        mean = 0.0
        for j in range(start, i + 1):
            mean += returns[j] - daily_rf
        mean /= window

        var = 0.0
        for j in range(start, i + 1):
            diff = (returns[j] - daily_rf) - mean
            var += diff * diff
        var /= (window - 1)

        std = np.sqrt(var)
        if std > EPSILON:
            result[i] = (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            result[i] = 0.0

    return result


@njit(cache=True, fastmath=True)
def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute annualized Sortino ratio."""
    n = len(returns)
    if n < 2:
        return 0.0

    daily_rf = risk_free / TRADING_DAYS_PER_YEAR

    mean = 0.0
    for i in range(n):
        mean += returns[i] - daily_rf
    mean /= n

    # Downside deviation
    downside_var = 0.0
    downside_count = 0
    for i in range(n):
        if returns[i] < 0:
            downside_var += returns[i] * returns[i]
            downside_count += 1

    if downside_count < 2:
        return 0.0

    downside_std = np.sqrt(downside_var / downside_count)
    if downside_std < EPSILON:
        return 0.0

    return (mean / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


@njit(cache=True, fastmath=True)
def calmar_ratio(returns: np.ndarray) -> float:
    """Compute Calmar ratio (annualized return / max drawdown)."""
    n = len(returns)
    if n < 2:
        return 0.0

    # Compute prices from returns
    prices = np.empty(n + 1, dtype=np.float64)
    prices[0] = 1.0
    for i in range(n):
        prices[i + 1] = prices[i] * (1.0 + returns[i])

    # Max drawdown
    max_dd = max_drawdown(prices)
    if max_dd >= 0 or abs(max_dd) < EPSILON:
        return 0.0

    # Annualized return
    total_return = prices[-1] / prices[0] - 1.0
    years = n / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0

    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0

    return ann_return / abs(max_dd)


@njit(cache=True, fastmath=True)
def profit_factor(returns: np.ndarray) -> float:
    """Compute profit factor (gross profit / gross loss)."""
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(len(returns)):
        if returns[i] > 0:
            gross_profit += returns[i]
        elif returns[i] < 0:
            gross_loss += abs(returns[i])

    if gross_loss < EPSILON:
        return 10.0 if gross_profit > 0 else 1.0

    return gross_profit / gross_loss


@njit(cache=True, fastmath=True)
def rolling_profit_factor(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling profit factor."""
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
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

        if gross_loss > EPSILON:
            result[i] = gross_profit / gross_loss
        else:
            result[i] = 10.0 if gross_profit > 0 else 1.0

    return result


# =============================================================================
# Portfolio Calculations (Numba-optimized)
# =============================================================================

@njit(cache=True, fastmath=True)
def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """Compute portfolio return given weights and asset returns."""
    n = len(weights)
    result = 0.0
    for i in range(n):
        result += weights[i] * returns[i]
    return result


@njit(cache=True, fastmath=True)
def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio variance given weights and covariance matrix."""
    n = len(weights)
    result = 0.0
    for i in range(n):
        for j in range(n):
            result += weights[i] * weights[j] * cov_matrix[i, j]
    return result


@njit(cache=True, fastmath=True)
def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio volatility (annualized)."""
    var = portfolio_variance(weights, cov_matrix)
    return np.sqrt(var * TRADING_DAYS_PER_YEAR) if var > 0 else 0.0


@njit(cache=True, fastmath=True, parallel=True)
def backtest_portfolio_returns(
    returns_matrix: np.ndarray,
    weights_matrix: np.ndarray,
    rebalance_indices: np.ndarray,
) -> np.ndarray:
    """
    Vectorized portfolio backtest with periodic rebalancing.

    Args:
        returns_matrix: 2D array [n_days x n_assets] of daily returns
        weights_matrix: 2D array [n_rebalances x n_assets] of target weights at each rebalance
        rebalance_indices: 1D array of day indices when rebalancing occurs

    Returns:
        1D array [n_days] of portfolio daily returns
    """
    n_days, n_assets = returns_matrix.shape
    n_rebalances = len(rebalance_indices)
    portfolio_returns = np.zeros(n_days, dtype=np.float64)
    current_weights = np.zeros(n_assets, dtype=np.float64)

    rebalance_idx = 0

    for day in prange(n_days):
        # Check if we should rebalance
        if rebalance_idx < n_rebalances and day == rebalance_indices[rebalance_idx]:
            for i in range(n_assets):
                current_weights[i] = weights_matrix[rebalance_idx, i]
            rebalance_idx += 1

        # Compute portfolio return
        port_ret = 0.0
        for i in range(n_assets):
            port_ret += current_weights[i] * returns_matrix[day, i]
        portfolio_returns[day] = port_ret

    return portfolio_returns


@njit(cache=True, fastmath=True)
def backtest_with_static_weights(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Fast backtest with static weights (no rebalancing).

    Args:
        returns_matrix: 2D array [n_days x n_assets] of daily returns
        weights: 1D array [n_assets] of static weights

    Returns:
        1D array [n_days] of portfolio daily returns
    """
    n_days, n_assets = returns_matrix.shape
    portfolio_returns = np.zeros(n_days, dtype=np.float64)

    for day in range(n_days):
        port_ret = 0.0
        for i in range(n_assets):
            port_ret += weights[i] * returns_matrix[day, i]
        portfolio_returns[day] = port_ret

    return portfolio_returns


@njit(cache=True, fastmath=True)
def compute_sharpe_from_returns(returns: np.ndarray, annualize: bool = True) -> float:
    """Compute Sharpe ratio from returns array."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean_ret = 0.0
    for i in range(n):
        mean_ret += returns[i]
    mean_ret /= n

    var = 0.0
    for i in range(n):
        diff = returns[i] - mean_ret
        var += diff * diff
    var /= (n - 1)

    std = np.sqrt(var)
    if std < EPSILON:
        return 0.0

    sharpe = mean_ret / std
    if annualize:
        sharpe *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe


@njit(cache=True)
def turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """Compute portfolio turnover."""
    n = len(old_weights)
    total = 0.0
    for i in range(n):
        total += abs(new_weights[i] - old_weights[i])
    return total / 2.0


@njit(cache=True, fastmath=True)
def herfindahl_index(weights: np.ndarray) -> float:
    """Compute Herfindahl-Hirschman Index (concentration)."""
    result = 0.0
    for i in range(len(weights)):
        result += weights[i] * weights[i]
    return result


# =============================================================================
# Correlation Functions (Numba-optimized)
# =============================================================================

@njit(cache=True, fastmath=True)
def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    mean_x = 0.0
    mean_y = 0.0
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    if var_x < EPSILON or var_y < EPSILON:
        return 0.0

    return cov / np.sqrt(var_x * var_y)


@njit(cache=True, fastmath=True)
def rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling correlation between two arrays."""
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window or len(y) != n:
        return result

    for i in range(window - 1, n):
        start = i - window + 1

        mean_x = 0.0
        mean_y = 0.0
        for j in range(start, i + 1):
            mean_x += x[j]
            mean_y += y[j]
        mean_x /= window
        mean_y /= window

        cov = 0.0
        var_x = 0.0
        var_y = 0.0
        for j in range(start, i + 1):
            dx = x[j] - mean_x
            dy = y[j] - mean_y
            cov += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        if var_x > EPSILON and var_y > EPSILON:
            result[i] = cov / np.sqrt(var_x * var_y)

    return result


@njit(cache=True, fastmath=True)
def covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix from returns matrix.

    Args:
        returns: 2D array [n_samples, n_assets]

    Returns:
        Covariance matrix [n_assets, n_assets]
    """
    n_samples, n_assets = returns.shape
    cov = np.empty((n_assets, n_assets), dtype=np.float64)

    # Compute means
    means = np.empty(n_assets, dtype=np.float64)
    for j in range(n_assets):
        total = 0.0
        for i in range(n_samples):
            total += returns[i, j]
        means[j] = total / n_samples

    # Compute covariance
    for j1 in range(n_assets):
        for j2 in range(j1, n_assets):
            cov_sum = 0.0
            for i in range(n_samples):
                cov_sum += (returns[i, j1] - means[j1]) * (returns[i, j2] - means[j2])
            cov_val = cov_sum / (n_samples - 1)
            cov[j1, j2] = cov_val
            cov[j2, j1] = cov_val

    return cov


@njit(cache=True, fastmath=True)
def correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix from returns matrix.

    Args:
        returns: 2D array [n_samples, n_assets]

    Returns:
        Correlation matrix [n_assets, n_assets]
    """
    n_samples, n_assets = returns.shape
    corr = np.empty((n_assets, n_assets), dtype=np.float64)

    # Compute means and stds
    means = np.empty(n_assets, dtype=np.float64)
    stds = np.empty(n_assets, dtype=np.float64)

    for j in range(n_assets):
        total = 0.0
        for i in range(n_samples):
            total += returns[i, j]
        means[j] = total / n_samples

        var = 0.0
        for i in range(n_samples):
            diff = returns[i, j] - means[j]
            var += diff * diff
        stds[j] = np.sqrt(var / (n_samples - 1))

    # Compute correlation
    for j1 in range(n_assets):
        corr[j1, j1] = 1.0
        for j2 in range(j1 + 1, n_assets):
            cov_sum = 0.0
            for i in range(n_samples):
                cov_sum += (returns[i, j1] - means[j1]) * (returns[i, j2] - means[j2])

            if stds[j1] > EPSILON and stds[j2] > EPSILON:
                corr_val = cov_sum / ((n_samples - 1) * stds[j1] * stds[j2])
            else:
                corr_val = 0.0

            corr[j1, j2] = corr_val
            corr[j2, j1] = corr_val

    return corr


# =============================================================================
# Risk Metrics (Numba-optimized)
# =============================================================================

@njit(cache=True)
def var_historical(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute Value at Risk using historical simulation.

    Args:
        returns: Array of returns
        alpha: Confidence level (e.g., 0.05 for 95% VaR)

    Returns:
        VaR value (negative for losses)
    """
    n = len(returns)
    if n < 10:
        return 0.0

    # Sort returns
    sorted_returns = np.sort(returns)
    idx = int(n * alpha)
    return sorted_returns[idx]


@njit(cache=True)
def cvar_historical(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Array of returns
        alpha: Confidence level

    Returns:
        CVaR value
    """
    n = len(returns)
    if n < 10:
        return 0.0

    var = var_historical(returns, alpha)

    # Mean of returns below VaR
    total = 0.0
    count = 0
    for i in range(n):
        if returns[i] <= var:
            total += returns[i]
            count += 1

    if count == 0:
        return var

    return total / count


# =============================================================================
# Parallel Versions for Large Arrays
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def parallel_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Parallel version of rolling standard deviation.

    Use this for arrays with >10000 elements.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan

    for i in prange(window - 1, n):
        start = i - window + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += arr[j]
        mean /= window

        var = 0.0
        for j in range(start, i + 1):
            diff = arr[j] - mean
            var += diff * diff
        var /= (window - 1)

        result[i] = np.sqrt(var) if var > 0 else 0.0

    return result


@njit(parallel=True, cache=True, fastmath=True)
def parallel_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Parallel version of correlation matrix computation.

    Use this for matrices with >100 assets.
    """
    n_samples, n_assets = returns.shape
    corr = np.empty((n_assets, n_assets), dtype=np.float64)

    # Compute means and stds (not parallelized, small relative cost)
    means = np.empty(n_assets, dtype=np.float64)
    stds = np.empty(n_assets, dtype=np.float64)

    for j in range(n_assets):
        total = 0.0
        for i in range(n_samples):
            total += returns[i, j]
        means[j] = total / n_samples

        var = 0.0
        for i in range(n_samples):
            diff = returns[i, j] - means[j]
            var += diff * diff
        stds[j] = np.sqrt(var / (n_samples - 1))

    # Compute correlation in parallel
    for j1 in prange(n_assets):
        corr[j1, j1] = 1.0
        for j2 in range(j1 + 1, n_assets):
            cov_sum = 0.0
            for i in range(n_samples):
                cov_sum += (returns[i, j1] - means[j1]) * (returns[i, j2] - means[j2])

            if stds[j1] > EPSILON and stds[j2] > EPSILON:
                corr_val = cov_sum / ((n_samples - 1) * stds[j1] * stds[j2])
            else:
                corr_val = 0.0

            corr[j1, j2] = corr_val
            corr[j2, j1] = corr_val

    return corr


# =============================================================================
# Environment Hot-Path Kernels (Numba-optimized)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def compound_returns_2d(period: np.ndarray) -> np.ndarray:
    """
    Compound returns along axis=0 (rows=days, cols=algos), NaN treated as 0.

    Equivalent to ``(1 + fillna(period, 0)).prod(axis=0) - 1`` but:
    - Avoids the (n_days, n_algos) temporary allocation.
    - Parallelized over columns (algos) for large universes (13k+ algos).

    Args:
        period: (n_days, n_algos) float32 array of daily returns (C-contiguous).

    Returns:
        (n_algos,) float32 compound returns.
    """
    n_days, n_algos = period.shape
    result = np.empty(n_algos, dtype=np.float32)
    for a in prange(n_algos):
        prod = np.float32(1.0)
        for d in range(n_days):
            v = period[d, a]
            if not np.isnan(v):
                prod = prod * (np.float32(1.0) + v)
        result[a] = prod - np.float32(1.0)
    return result


@njit(parallel=True, cache=True, fastmath=True)
def weighted_sum_2d(weights: np.ndarray, values: np.ndarray) -> float:
    """
    Compute sum(weights * values) over a 2D array without a temporary allocation.

    Parallelized over columns (algos). Useful for benchmark weighted returns.

    Args:
        weights: (n_days, n_algos) float32 normalized benchmark weights.
        values:  (n_days, n_algos) float32 daily algo returns (NaN→0 pre-applied).

    Returns:
        Scalar weighted return sum.
    """
    n_days, n_algos = weights.shape
    col_totals = np.zeros(n_algos, dtype=np.float32)
    for a in prange(n_algos):
        s = np.float32(0.0)
        for d in range(n_days):
            s += weights[d, a] * values[d, a]
        col_totals[a] = s
    return float(col_totals.sum())


# =============================================================================
# Utility Functions
# =============================================================================

def is_numba_available() -> bool:
    """Check if numba is available."""
    return HAS_NUMBA


def warm_up_jit():
    """
    Warm up JIT compilation by calling each function once.

    Call this at application startup to avoid compilation delays during
    critical operations.
    """
    if not HAS_NUMBA:
        return

    # Create small test arrays
    test_arr = np.random.randn(100).astype(np.float64)
    test_prices = np.abs(test_arr.cumsum()) + 100
    test_weights = np.ones(10, dtype=np.float64) / 10
    test_cov = np.eye(10, dtype=np.float64)
    test_returns_2d = np.random.randn(50, 10).astype(np.float64)

    # Call each function to trigger compilation
    try:
        rolling_sum(test_arr, 10)
        rolling_mean(test_arr, 10)
        rolling_std(test_arr, 10)
        cumulative_return(test_prices)
        returns_from_prices(test_prices)
        drawdown_series(test_prices)
        max_drawdown(test_prices)
        max_drawdown_duration(test_prices)
        sharpe_ratio(test_arr)
        sortino_ratio(test_arr)
        calmar_ratio(test_arr)
        profit_factor(test_arr)
        portfolio_return(test_weights, test_arr[:10])
        portfolio_variance(test_weights, test_cov)
        turnover(test_weights, test_weights)
        herfindahl_index(test_weights)
        correlation(test_arr, test_arr)
        covariance_matrix(test_returns_2d)
        correlation_matrix(test_returns_2d)
        var_historical(test_arr)
        cvar_historical(test_arr)
        # Environment hot-path kernels
        test_period = np.random.randn(5, 10).astype(np.float32)
        compound_returns_2d(test_period)
        weighted_sum_2d(test_period, test_period)
        logger.debug("JIT warm-up completed")
    except Exception as e:
        logger.warning(f"JIT warm-up failed: {e}")
