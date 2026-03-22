"""
Perfil de algoritmos: métricas de performance, operativa y riesgo.

Optimized version with:
- Numba JIT-compiled functions for core metrics
- Vectorized operations across all algorithms
- Memory-efficient processing (float32, batching)
"""

import gc
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import numba, fall back to identity decorator if not available
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    logger.info("Numba not available, using pure numpy implementations")
    HAS_NUMBA = False
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator
    prange = range

# Constants
TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Numba JIT-compiled functions
# =============================================================================

@njit(cache=True)
def _annualized_return_numba(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized return from daily returns array."""
    n = len(returns)
    if n == 0:
        return 0.0

    total_return = 1.0
    for i in range(n):
        total_return *= (1.0 + returns[i])
    total_return -= 1.0

    years = n / trading_days
    if years <= 0:
        return 0.0

    if total_return <= -1:
        return -1.0

    return (1.0 + total_return) ** (1.0 / years) - 1.0


@njit(cache=True)
def _volatility_numba(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized volatility."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean = 0.0
    for i in range(n):
        mean += returns[i]
    mean /= n

    var = 0.0
    for i in range(n):
        diff = returns[i] - mean
        var += diff * diff
    var /= (n - 1)

    return np.sqrt(var) * np.sqrt(trading_days)


@njit(cache=True)
def _sharpe_numba(returns: np.ndarray, risk_free: float = 0.0, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio."""
    n = len(returns)
    if n < 2:
        return 0.0

    daily_rf = risk_free / trading_days

    mean = 0.0
    for i in range(n):
        mean += (returns[i] - daily_rf)
    mean /= n

    var = 0.0
    for i in range(n):
        diff = (returns[i] - daily_rf) - mean
        var += diff * diff
    var /= (n - 1)

    std = np.sqrt(var)
    if std < 1e-10:
        return 0.0

    return (mean / std) * np.sqrt(trading_days)


@njit(cache=True)
def _sortino_numba(returns: np.ndarray, risk_free: float = 0.0, trading_days: int = 252) -> float:
    """Annualized Sortino ratio."""
    n = len(returns)
    if n < 2:
        return 0.0

    daily_rf = risk_free / trading_days

    # Mean excess return
    mean_excess = 0.0
    for i in range(n):
        mean_excess += (returns[i] - daily_rf)
    mean_excess /= n

    # Downside deviation
    downside_sum = 0.0
    downside_count = 0
    for i in range(n):
        if returns[i] < 0:
            downside_sum += returns[i] * returns[i]
            downside_count += 1

    if downside_count < 2:
        return 0.0

    downside_std = np.sqrt(downside_sum / downside_count)
    if downside_std < 1e-10:
        return 0.0

    return (mean_excess / downside_std) * np.sqrt(trading_days)


@njit(cache=True)
def _max_drawdown_numba(returns: np.ndarray) -> float:
    """Maximum drawdown from returns array."""
    n = len(returns)
    if n == 0:
        return 0.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0

    for i in range(n):
        equity *= (1.0 + returns[i])
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (equity - peak) / peak
            if dd < max_dd:
                max_dd = dd

    return max_dd


@njit(cache=True)
def _max_drawdown_duration_numba(returns: np.ndarray) -> int:
    """Maximum drawdown duration in days."""
    n = len(returns)
    if n == 0:
        return 0

    equity = 1.0
    peak = 1.0
    current_dd_length = 0
    max_dd_length = 0

    for i in range(n):
        equity *= (1.0 + returns[i])
        if equity >= peak:
            peak = equity
            if current_dd_length > max_dd_length:
                max_dd_length = current_dd_length
            current_dd_length = 0
        else:
            current_dd_length += 1

    # Check final period
    if current_dd_length > max_dd_length:
        max_dd_length = current_dd_length

    return max_dd_length


@njit(cache=True)
def _var_numba(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Value at Risk at percentile alpha."""
    n = len(returns)
    if n == 0:
        return 0.0

    # Sort returns
    sorted_returns = np.sort(returns)
    idx = int(alpha * n)
    if idx >= n:
        idx = n - 1

    return sorted_returns[idx]


@njit(cache=True)
def _cvar_numba(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional VaR (Expected Shortfall)."""
    n = len(returns)
    if n == 0:
        return 0.0

    sorted_returns = np.sort(returns)
    idx = int(alpha * n)
    if idx < 1:
        idx = 1

    total = 0.0
    for i in range(idx):
        total += sorted_returns[i]

    return total / idx


@njit(cache=True)
def _tail_ratio_numba(returns: np.ndarray) -> float:
    """Tail ratio: p95 gain / p5 loss."""
    n = len(returns)
    if n < 20:
        return 0.0

    sorted_returns = np.sort(returns)

    idx_05 = int(0.05 * n)
    idx_95 = int(0.95 * n)
    if idx_95 >= n:
        idx_95 = n - 1

    p05 = abs(sorted_returns[idx_05])
    p95 = sorted_returns[idx_95]

    if p05 < 1e-10:
        return 0.0

    return p95 / p05


@njit(cache=True)
def _calmar_numba(returns: np.ndarray, trading_days: int = 252) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    ann_ret = _annualized_return_numba(returns, trading_days)
    max_dd = abs(_max_drawdown_numba(returns))

    if max_dd < 1e-10:
        return 0.0

    return ann_ret / max_dd


@njit(cache=True, parallel=True)
def _compute_all_metrics_batch(
    returns_2d: np.ndarray,
    trading_days: int = 252,
    risk_free: float = 0.0,
) -> np.ndarray:
    """
    Compute all metrics for multiple algorithms in parallel.

    Args:
        returns_2d: 2D array [n_days, n_algos] with returns
        trading_days: Trading days per year
        risk_free: Risk-free rate

    Returns:
        2D array [n_algos, 10] with metrics:
        [ann_return, ann_vol, sharpe, sortino, max_dd, max_dd_dur, calmar, var95, cvar95, tail_ratio]
    """
    n_days, n_algos = returns_2d.shape
    results = np.empty((n_algos, 10), dtype=np.float64)

    for i in prange(n_algos):
        returns = returns_2d[:, i]

        # Filter out NaN values
        valid_mask = ~np.isnan(returns)
        valid_count = np.sum(valid_mask)

        if valid_count < 10:
            results[i, :] = 0.0
            continue

        # Extract valid returns
        valid_returns = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for j in range(n_days):
            if valid_mask[j]:
                valid_returns[idx] = returns[j]
                idx += 1

        # Compute all metrics
        results[i, 0] = _annualized_return_numba(valid_returns, trading_days)
        results[i, 1] = _volatility_numba(valid_returns, trading_days)
        results[i, 2] = _sharpe_numba(valid_returns, risk_free, trading_days)
        results[i, 3] = _sortino_numba(valid_returns, risk_free, trading_days)
        results[i, 4] = _max_drawdown_numba(valid_returns)
        results[i, 5] = _max_drawdown_duration_numba(valid_returns)
        results[i, 6] = _calmar_numba(valid_returns, trading_days)
        results[i, 7] = _var_numba(valid_returns, 0.05)
        results[i, 8] = _cvar_numba(valid_returns, 0.05)
        results[i, 9] = _tail_ratio_numba(valid_returns)

    return results


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class AlgoProfile:
    """Perfil completo de un algoritmo."""

    algo_id: str

    # Performance
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # días

    # Operativa
    avg_trades_per_month: float = 0.0
    avg_trade_duration_days: float = 0.0
    median_trade_duration_days: float = 0.0
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    turnover_annualized: float = 0.0

    # Riesgo
    var_95: float = 0.0
    cvar_95: float = 0.0
    tail_ratio: float = 0.0  # ganancia p95 / pérdida p5

    # Comportamiento por régimen
    performance_by_regime: dict = field(default_factory=dict)

    # Correlaciones
    correlation_with_benchmark: float = 0.0
    correlation_with_others: dict = field(default_factory=dict)


# =============================================================================
# AlgoProfiler class
# =============================================================================

class AlgoProfiler:
    """
    Calcula perfil completo de cada algoritmo.

    Optimized version:
    - Uses numba JIT for core metric calculations
    - Vectorized profile_all_vectorized() for batch processing
    - Memory-efficient with float32 and gc.collect()
    """

    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def profile(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        regime_labels: Optional[pd.Series] = None,
        algo_id: str = "unknown",
    ) -> AlgoProfile:
        """
        Genera perfil completo de un algoritmo.
        """
        profile = AlgoProfile(algo_id=algo_id)

        # Convert to numpy for numba
        ret_arr = returns.dropna().values.astype(np.float64)

        if len(ret_arr) < 10:
            return profile

        # Performance metrics using numba
        profile.annualized_return = _annualized_return_numba(ret_arr, self.trading_days)
        profile.annualized_volatility = _volatility_numba(ret_arr, self.trading_days)
        profile.sharpe_ratio = _sharpe_numba(ret_arr, self.risk_free_rate, self.trading_days)
        profile.sortino_ratio = _sortino_numba(ret_arr, self.risk_free_rate, self.trading_days)
        profile.max_drawdown = _max_drawdown_numba(ret_arr)
        profile.max_drawdown_duration = _max_drawdown_duration_numba(ret_arr)
        profile.calmar_ratio = _calmar_numba(ret_arr, self.trading_days)

        # Risk metrics
        profile.var_95 = _var_numba(ret_arr, 0.05)
        profile.cvar_95 = _cvar_numba(ret_arr, 0.05)
        profile.tail_ratio = _tail_ratio_numba(ret_arr)

        # Operativa (si hay datos de trades)
        if trades is not None and not trades.empty:
            profile.avg_trades_per_month = self._avg_trades_per_month(trades, returns)
            profile.avg_trade_duration_days = self._avg_trade_duration(trades)
            profile.median_trade_duration_days = self._median_trade_duration(trades)

        # Correlación con benchmark
        if benchmark_returns is not None:
            aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner")
            if len(aligned) > 10:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    profile.correlation_with_benchmark = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

        # Performance por régimen
        if regime_labels is not None:
            profile.performance_by_regime = self._performance_by_regime(returns, regime_labels)

        return profile

    def profile_all(
        self,
        returns_matrix: pd.DataFrame,
        trades_dict: Optional[dict[str, pd.DataFrame]] = None,
        benchmark_returns: Optional[pd.Series] = None,
        regime_labels: Optional[pd.Series] = None,
        use_float32: bool = True,
    ) -> dict[str, AlgoProfile]:
        """
        Genera perfiles para todos los algoritmos (fully vectorized).

        Uses numba parallel processing for core metrics.
        """
        algo_ids = list(returns_matrix.columns)
        n_algos = len(algo_ids)

        logger.info(f"Profiling {n_algos} algorithms (vectorized)...")

        # Convert to numpy array for numba (transpose: [n_days, n_algos])
        dtype = np.float32 if use_float32 else np.float64
        returns_2d = returns_matrix.values.astype(np.float64)  # Numba needs float64

        # Compute all metrics in parallel using numba
        logger.info("  Computing core metrics with numba...")
        metrics = _compute_all_metrics_batch(
            returns_2d,
            trading_days=self.trading_days,
            risk_free=self.risk_free_rate,
        )

        # Pre-compute benchmark correlations vectorized
        benchmark_corrs = {}
        if benchmark_returns is not None:
            logger.info("  Computing benchmark correlations...")
            combined = returns_matrix.join(benchmark_returns.rename('_benchmark_'), how='inner')
            if len(combined) > 10:
                benchmark_col = combined['_benchmark_']
                algo_cols = combined.drop(columns=['_benchmark_'])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    benchmark_corrs = algo_cols.corrwith(benchmark_col).to_dict()

        # Build profiles from computed metrics
        logger.info("  Building profile objects...")
        profiles = {}
        for i, algo_id in enumerate(algo_ids):
            profile = AlgoProfile(algo_id=algo_id)

            profile.annualized_return = float(metrics[i, 0])
            profile.annualized_volatility = float(metrics[i, 1])
            profile.sharpe_ratio = float(metrics[i, 2])
            profile.sortino_ratio = float(metrics[i, 3])
            profile.max_drawdown = float(metrics[i, 4])
            profile.max_drawdown_duration = int(metrics[i, 5])
            profile.calmar_ratio = float(metrics[i, 6])
            profile.var_95 = float(metrics[i, 7])
            profile.cvar_95 = float(metrics[i, 8])
            profile.tail_ratio = float(metrics[i, 9])

            # Benchmark correlation
            if algo_id in benchmark_corrs:
                corr = benchmark_corrs[algo_id]
                profile.correlation_with_benchmark = corr if not np.isnan(corr) else 0.0

            # Trades (if provided)
            if trades_dict and algo_id in trades_dict:
                trades = trades_dict[algo_id]
                if trades is not None and not trades.empty:
                    profile.avg_trades_per_month = self._avg_trades_per_month(
                        trades, returns_matrix[algo_id]
                    )
                    profile.avg_trade_duration_days = self._avg_trade_duration(trades)
                    profile.median_trade_duration_days = self._median_trade_duration(trades)

            profiles[algo_id] = profile

        # Skip correlation_with_others to save memory (can be computed on-demand)
        # If needed, compute correlation matrix separately

        gc.collect()
        logger.info(f"  Profiled {len(profiles)} algorithms")
        return profiles

    def _avg_trades_per_month(
        self, trades: pd.DataFrame, returns: pd.Series
    ) -> float:
        """Número promedio de trades por mes."""
        n_trades = len(trades)
        months = len(returns) / 21
        return n_trades / months if months > 0 else 0.0

    def _avg_trade_duration(self, trades: pd.DataFrame) -> float:
        """Duración promedio de trades en días."""
        if "duration" in trades.columns:
            return float(trades["duration"].mean())
        if "entry_date" in trades.columns and "exit_date" in trades.columns:
            durations = (trades["exit_date"] - trades["entry_date"]).dt.days
            return float(durations.mean())
        return 0.0

    def _median_trade_duration(self, trades: pd.DataFrame) -> float:
        """Duración mediana de trades en días."""
        if "duration" in trades.columns:
            return float(trades["duration"].median())
        if "entry_date" in trades.columns and "exit_date" in trades.columns:
            durations = (trades["exit_date"] - trades["entry_date"]).dt.days
            return float(durations.median())
        return 0.0

    def _performance_by_regime(
        self, returns: pd.Series, regime_labels: pd.Series
    ) -> dict:
        """Calcula métricas de performance por régimen."""
        aligned = pd.concat([returns, regime_labels], axis=1, join="inner")
        aligned.columns = ["returns", "regime"]

        results = {}
        for regime in aligned["regime"].unique():
            regime_returns = aligned[aligned["regime"] == regime]["returns"]
            if len(regime_returns) > 10:
                ret_arr = regime_returns.values.astype(np.float64)
                results[regime] = {
                    "annualized_return": float(_annualized_return_numba(ret_arr, self.trading_days)),
                    "annualized_volatility": float(_volatility_numba(ret_arr, self.trading_days)),
                    "sharpe_ratio": float(_sharpe_numba(ret_arr, self.risk_free_rate, self.trading_days)),
                    "max_drawdown": float(_max_drawdown_numba(ret_arr)),
                    "n_days": len(regime_returns),
                }

        return results

    def generate_summary_table(self, profiles: dict[str, AlgoProfile]) -> pd.DataFrame:
        """
        Genera tabla resumen de todos los perfiles.

        Returns numeric values for computation. Use generate_summary_table_formatted
        for display-friendly output with percentage formatting.
        """
        rows = []
        for algo_id, profile in profiles.items():
            rows.append({
                "algo_id": algo_id,
                "ann_return": profile.annualized_return,
                "ann_volatility": profile.annualized_volatility,
                "sharpe": profile.sharpe_ratio,
                "sortino": profile.sortino_ratio,
                "calmar": profile.calmar_ratio,
                "max_dd": profile.max_drawdown,
                "max_dd_days": profile.max_drawdown_duration,
                "var_95": profile.var_95,
                "cvar_95": profile.cvar_95,
                "corr_benchmark": profile.correlation_with_benchmark,
            })

        df = pd.DataFrame(rows).set_index("algo_id")
        # Ensure numeric types
        numeric_cols = ['ann_return', 'ann_volatility', 'sharpe', 'sortino',
                        'calmar', 'max_dd', 'var_95', 'cvar_95', 'corr_benchmark']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def generate_summary_table_formatted(self, profiles: dict[str, AlgoProfile]) -> pd.DataFrame:
        """
        Genera tabla resumen con valores formateados para display.
        """
        rows = []
        for algo_id, profile in profiles.items():
            rows.append({
                "algo_id": algo_id,
                "ann_return": f"{profile.annualized_return:.2%}",
                "ann_volatility": f"{profile.annualized_volatility:.2%}",
                "sharpe": f"{profile.sharpe_ratio:.2f}",
                "sortino": f"{profile.sortino_ratio:.2f}",
                "calmar": f"{profile.calmar_ratio:.2f}",
                "max_dd": f"{profile.max_drawdown:.2%}",
                "max_dd_days": profile.max_drawdown_duration,
                "var_95": f"{profile.var_95:.2%}",
                "cvar_95": f"{profile.cvar_95:.2%}",
                "corr_benchmark": f"{profile.correlation_with_benchmark:.2f}",
            })

        return pd.DataFrame(rows).set_index("algo_id")
