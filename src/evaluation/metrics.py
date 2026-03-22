"""
Metricas financieras para evaluacion.

Optimized version with:
- Numba JIT-compiled functions for core calculations
- Vectorized numpy operations
- Fallback to pure numpy when numba not available
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import numba
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func=None, cache=True, fastmath=False, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

# Constants
EPSILON = 1e-10


# =============================================================================
# Numba-optimized core functions
# =============================================================================

@njit(cache=True, fastmath=True)
def _max_drawdown_numba(equity: np.ndarray) -> float:
    """Numba-optimized max drawdown calculation."""
    n = len(equity)
    if n == 0:
        return 0.0

    running_max = equity[0]
    max_dd = 0.0

    for i in range(n):
        if equity[i] > running_max:
            running_max = equity[i]
        if running_max > 0:
            dd = (equity[i] - running_max) / running_max
            if dd < max_dd:
                max_dd = dd

    return max_dd


@njit(cache=True)
def _max_drawdown_duration_numba(equity: np.ndarray) -> int:
    """Numba-optimized max drawdown duration calculation."""
    n = len(equity)
    if n < 2:
        return 0

    running_max = equity[0]
    current_duration = 0
    max_duration = 0

    for i in range(n):
        if equity[i] >= running_max:
            running_max = equity[i]
            current_duration = 0
        else:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration

    return max_duration


@njit(cache=True, fastmath=True)
def _sharpe_ratio_numba(returns: np.ndarray, risk_free_daily: float) -> float:
    """Numba-optimized Sharpe ratio calculation."""
    n = len(returns)
    if n < 2:
        return 0.0

    # Compute mean of excess returns
    mean = 0.0
    for i in range(n):
        mean += returns[i] - risk_free_daily
    mean /= n

    # Compute variance
    var = 0.0
    for i in range(n):
        diff = (returns[i] - risk_free_daily) - mean
        var += diff * diff
    var /= (n - 1)

    std = np.sqrt(var)
    if std < EPSILON:
        return 0.0

    return (mean / std) * np.sqrt(252)


@njit(cache=True, fastmath=True)
def _sortino_ratio_numba(returns: np.ndarray, risk_free_daily: float) -> float:
    """Numba-optimized Sortino ratio calculation."""
    n = len(returns)
    if n < 2:
        return 0.0

    # Compute mean of excess returns
    mean = 0.0
    for i in range(n):
        mean += returns[i] - risk_free_daily
    mean /= n

    # Compute downside deviation
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

    return (mean / downside_std) * np.sqrt(252)


@njit(cache=True, fastmath=True)
def _turnover_numba(weights_history: np.ndarray) -> float:
    """
    Numba-optimized turnover calculation.

    Args:
        weights_history: 2D array [n_periods, n_assets]

    Returns:
        Total turnover
    """
    n_periods = weights_history.shape[0]
    if n_periods < 2:
        return 0.0

    total_turnover = 0.0
    n_assets = weights_history.shape[1]

    for t in range(1, n_periods):
        period_turnover = 0.0
        for j in range(n_assets):
            period_turnover += abs(weights_history[t, j] - weights_history[t - 1, j])
        total_turnover += period_turnover / 2

    return total_turnover


@njit(cache=True, fastmath=True)
def _annualized_return_numba(returns: np.ndarray, trading_days: int) -> float:
    """Numba-optimized annualized return calculation."""
    n = len(returns)
    if n == 0:
        return 0.0

    # Compute cumulative product
    cum_prod = 1.0
    for i in range(n):
        cum_prod *= (1.0 + returns[i])

    total_return = cum_prod - 1.0
    years = n / trading_days

    if years <= 0:
        return 0.0

    if total_return <= -1.0:
        return -1.0  # Total loss

    return (1.0 + total_return) ** (1.0 / years) - 1.0


@njit(cache=True, fastmath=True)
def _beta_numba(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Numba-optimized beta calculation."""
    n = len(returns)
    if n < 2 or len(benchmark_returns) != n:
        return 0.0

    # Compute means
    mean_r = 0.0
    mean_b = 0.0
    for i in range(n):
        mean_r += returns[i]
        mean_b += benchmark_returns[i]
    mean_r /= n
    mean_b /= n

    # Compute covariance and benchmark variance
    cov = 0.0
    var_b = 0.0
    for i in range(n):
        dr = returns[i] - mean_r
        db = benchmark_returns[i] - mean_b
        cov += dr * db
        var_b += db * db

    if var_b < EPSILON:
        return 0.0

    return cov / var_b


# =============================================================================
# Public API functions
# =============================================================================

def compute_full_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Calcula todas las metricas necesarias para comparacion justa.

    Args:
        returns: Serie de retornos del portfolio.
        benchmark_returns: Serie de retornos del benchmark (opcional).
        trading_days: Dias de trading por ano.
        risk_free_rate: Tasa libre de riesgo anualizada.

    Returns:
        Dict con todas las metricas.
    """
    metrics = {}

    # Convert to numpy for numba functions
    ret_arr = returns.values.astype(np.float64)

    # Performance absoluta
    metrics["annualized_return"] = annualized_return(returns, trading_days)
    metrics["annualized_volatility"] = annualized_volatility(returns, trading_days)
    metrics["sharpe_ratio"] = sharpe_ratio(returns, risk_free_rate, trading_days)
    metrics["sortino_ratio"] = sortino_ratio(returns, risk_free_rate, trading_days)
    metrics["calmar_ratio"] = calmar_ratio(returns, trading_days)
    metrics["max_drawdown"] = max_drawdown(returns)
    metrics["max_drawdown_duration_days"] = max_drawdown_duration(returns)

    # Riesgo
    metrics["var_95"] = var(returns, 0.05)
    metrics["cvar_95"] = cvar(returns, 0.05)
    metrics["skewness"] = returns.skew()
    metrics["kurtosis"] = returns.kurtosis()

    # Estadisticas mensuales
    monthly_returns = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    metrics["worst_month"] = monthly_returns.min()
    metrics["best_month"] = monthly_returns.max()
    metrics["pct_positive_months"] = (monthly_returns > 0).mean() * 100

    # Performance relativa al benchmark
    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner")
        aligned.columns = ["portfolio", "benchmark"]

        excess = aligned["portfolio"] - aligned["benchmark"]

        metrics["excess_return"] = annualized_return(excess, trading_days)
        metrics["tracking_error"] = annualized_volatility(excess, trading_days)
        metrics["information_ratio"] = information_ratio(returns, benchmark_returns, trading_days)
        metrics["beta_vs_benchmark"] = beta(returns, benchmark_returns)
        metrics["alpha_vs_benchmark"] = alpha(returns, benchmark_returns, risk_free_rate, trading_days)

    return metrics


def annualized_return(returns: pd.Series, trading_days: int = 252) -> float:
    """Retorno anualizado (optimized)."""
    ret_arr = returns.values.astype(np.float64)
    return _annualized_return_numba(ret_arr, trading_days)


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Volatilidad anualizada."""
    return returns.std() * np.sqrt(trading_days)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Sharpe ratio anualizado (optimized)."""
    ret_arr = returns.values.astype(np.float64)
    daily_rf = risk_free_rate / trading_days
    return _sharpe_ratio_numba(ret_arr, daily_rf)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Sortino ratio anualizado (optimized)."""
    ret_arr = returns.values.astype(np.float64)
    daily_rf = risk_free_rate / trading_days
    return _sortino_ratio_numba(ret_arr, daily_rf)


def calmar_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """Calmar ratio (retorno anualizado / max drawdown)."""
    max_dd = abs(max_drawdown(returns))
    if max_dd < EPSILON:
        return 0.0
    return annualized_return(returns, trading_days) / max_dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximo drawdown (optimized)."""
    equity = (1 + returns).cumprod().values.astype(np.float64)
    return _max_drawdown_numba(equity)


def max_drawdown_duration(returns: pd.Series) -> int:
    """Duracion del maximo drawdown en dias (optimized)."""
    equity = (1 + returns).cumprod().values.astype(np.float64)
    return _max_drawdown_duration_numba(equity)


def var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Value at Risk al percentil alpha."""
    return returns.quantile(alpha)


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall)."""
    var_threshold = var(returns, alpha)
    below_var = returns[returns <= var_threshold]
    if len(below_var) == 0:
        return var_threshold
    return below_var.mean()


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Information ratio (excess return / tracking error)."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner")
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]

    excess_ann = annualized_return(excess, trading_days)
    te = annualized_volatility(excess, trading_days)

    if te < EPSILON:
        return 0.0
    return excess_ann / te


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta vs benchmark (optimized)."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner")
    ret_arr = aligned.iloc[:, 0].values.astype(np.float64)
    bench_arr = aligned.iloc[:, 1].values.astype(np.float64)
    return _beta_numba(ret_arr, bench_arr)


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Alpha de Jensen (alpha de regresion CAPM)."""
    b = beta(returns, benchmark_returns)
    r_portfolio = annualized_return(returns, trading_days)
    r_benchmark = annualized_return(benchmark_returns, trading_days)

    return r_portfolio - (risk_free_rate + b * (r_benchmark - risk_free_rate))


def turnover(weights_history: list[np.ndarray]) -> float:
    """Turnover total (optimized)."""
    if len(weights_history) < 2:
        return 0.0

    # Convert list to 2D array for numba
    weights_arr = np.array(weights_history, dtype=np.float64)
    return _turnover_numba(weights_arr)


def concentration_hhi(weights: np.ndarray) -> float:
    """Indice Herfindahl-Hirschman de concentracion (vectorized)."""
    return np.sum(weights ** 2)


# =============================================================================
# Additional optimized metrics
# =============================================================================

def rolling_sharpe(returns: pd.Series, window: int = 63, risk_free_rate: float = 0.0) -> pd.Series:
    """Rolling Sharpe ratio (optimized)."""
    daily_rf = risk_free_rate / 252

    def _rolling_sharpe_window(x):
        if len(x) < 10:
            return np.nan
        excess = x - daily_rf
        if excess.std() < EPSILON:
            return 0.0
        return (excess.mean() / excess.std()) * np.sqrt(252)

    return returns.rolling(window).apply(_rolling_sharpe_window, raw=True)


def rolling_max_drawdown(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling maximum drawdown."""
    equity = (1 + returns).cumprod()

    def _rolling_mdd(eq_window):
        if len(eq_window) < 2:
            return 0.0
        running_max = np.maximum.accumulate(eq_window)
        drawdowns = (eq_window - running_max) / running_max
        return drawdowns.min()

    return equity.rolling(window).apply(_rolling_mdd, raw=True)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega ratio: probability-weighted ratio of gains vs losses.

    Args:
        returns: Return series
        threshold: Minimum acceptable return (MAR)

    Returns:
        Omega ratio (>1 is good)
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()

    if losses < EPSILON:
        return 10.0 if gains > 0 else 1.0

    return gains / losses


def tail_ratio(returns: pd.Series, percentile: float = 0.95) -> float:
    """
    Tail ratio: ratio of right tail to left tail.

    Measures asymmetry of return distribution.
    """
    right_tail = returns.quantile(percentile)
    left_tail = returns.quantile(1 - percentile)

    if abs(left_tail) < EPSILON:
        return 0.0

    return abs(right_tail / left_tail)


def common_sense_ratio(returns: pd.Series) -> float:
    """
    Common sense ratio: tail_ratio * profit_factor.

    A simple measure of strategy quality.
    """
    tr = tail_ratio(returns)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses < EPSILON:
        pf = 10.0 if gains > 0 else 1.0
    else:
        pf = gains / losses

    return tr * pf
