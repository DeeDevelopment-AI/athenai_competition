"""
Reverse engineering del benchmark: política de sizing, temporal y riesgo.

Optimized version with numba JIT compilation for core metrics.
"""

import logging
import warnings
from dataclasses import dataclass, field
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
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator


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
        if not np.isnan(returns[i]):
            total_return *= (1.0 + returns[i])
    total_return -= 1.0

    years = n / trading_days
    if years <= 0 or total_return <= -1:
        return 0.0

    return (1.0 + total_return) ** (1.0 / years) - 1.0


@njit(cache=True)
def _volatility_numba(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized volatility."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean = 0.0
    count = 0
    for i in range(n):
        if not np.isnan(returns[i]):
            mean += returns[i]
            count += 1

    if count < 2:
        return 0.0

    mean /= count

    var = 0.0
    for i in range(n):
        if not np.isnan(returns[i]):
            diff = returns[i] - mean
            var += diff * diff
    var /= (count - 1)

    return np.sqrt(var) * np.sqrt(trading_days)


@njit(cache=True)
def _sharpe_numba(returns: np.ndarray, risk_free: float = 0.0, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio."""
    n = len(returns)
    if n < 2:
        return 0.0

    daily_rf = risk_free / trading_days

    mean = 0.0
    count = 0
    for i in range(n):
        if not np.isnan(returns[i]):
            mean += (returns[i] - daily_rf)
            count += 1

    if count < 2:
        return 0.0

    mean /= count

    var = 0.0
    for i in range(n):
        if not np.isnan(returns[i]):
            diff = (returns[i] - daily_rf) - mean
            var += diff * diff
    var /= (count - 1)

    std = np.sqrt(var)
    if std < 1e-10:
        return 0.0

    return (mean / std) * np.sqrt(trading_days)


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
        if not np.isnan(returns[i]):
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
        if not np.isnan(returns[i]):
            equity *= (1.0 + returns[i])
            if equity >= peak:
                peak = equity
                if current_dd_length > max_dd_length:
                    max_dd_length = current_dd_length
                current_dd_length = 0
            else:
                current_dd_length += 1

    if current_dd_length > max_dd_length:
        max_dd_length = current_dd_length

    return max_dd_length


@dataclass
class BenchmarkProfile:
    """Perfil completo del benchmark."""

    # Performance
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Política de sizing
    avg_weights: dict = field(default_factory=dict)
    std_weights: dict = field(default_factory=dict)
    min_weights: dict = field(default_factory=dict)
    max_weights: dict = field(default_factory=dict)
    concentration_hhi: float = 0.0
    concentration_hhi_avg: float = 0.0

    # Política temporal
    rebalance_frequency_days: float = 0.0
    avg_weight_change_per_rebalance: float = 0.0
    weight_autocorrelation: float = 0.0
    avg_holding_period_days: float = 0.0

    # Política de riesgo
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_total_exposure: float = 0.0
    avg_total_exposure: float = 0.0

    # Turnover
    turnover_annualized: float = 0.0
    n_rebalances_per_year: float = 0.0

    # Comportamiento por régimen
    weights_by_regime: dict = field(default_factory=dict)
    performance_by_regime: dict = field(default_factory=dict)


class BenchmarkProfiler:
    """
    Reverse engineering del benchmark.

    Infiere la política implícita del benchmark analizando:
    - Política de sizing (cómo distribuye entre algoritmos)
    - Política temporal (cuándo y cómo rebalancea)
    - Política de riesgo (límites de exposición, drawdown)
    - Comportamiento por régimen de mercado
    """

    def __init__(self, trading_days: int = 252, risk_free_rate: float = 0.0):
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate

    def profile(
        self,
        returns: pd.Series,
        weights: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
    ) -> BenchmarkProfile:
        """
        Genera perfil completo del benchmark.

        Args:
            returns: Serie de retornos del benchmark.
            weights: DataFrame [fecha x algo_id] con pesos.
            regime_labels: Serie con etiquetas de régimen (opcional).

        Returns:
            BenchmarkProfile con análisis completo.
        """
        profile = BenchmarkProfile()

        # Performance
        profile.annualized_return = self._annualized_return(returns)
        profile.annualized_volatility = self._annualized_volatility(returns)
        profile.sharpe_ratio = self._sharpe_ratio(returns)
        profile.sortino_ratio = self._sortino_ratio(returns)
        profile.max_drawdown = self._max_drawdown(returns)
        profile.max_drawdown_duration = self._max_drawdown_duration(returns)
        profile.calmar_ratio = self._calmar_ratio(returns)
        profile.var_95 = returns.quantile(0.05)
        profile.cvar_95 = returns[returns <= profile.var_95].mean()

        # Política de sizing
        if not weights.empty:
            profile.avg_weights = weights.mean().to_dict()
            profile.std_weights = weights.std().to_dict()
            profile.min_weights = weights.min().to_dict()
            profile.max_weights = weights.max().to_dict()
            profile.concentration_hhi = self._current_hhi(weights.iloc[-1])
            profile.concentration_hhi_avg = self._avg_hhi(weights)

            # Política temporal
            profile.rebalance_frequency_days = self._rebalance_frequency(weights)
            profile.avg_weight_change_per_rebalance = self._avg_weight_change(weights)
            profile.weight_autocorrelation = self._weight_autocorrelation(weights)
            profile.avg_holding_period_days = self._avg_holding_period(weights)

            # Exposición
            total_exposure = weights.abs().sum(axis=1)
            profile.max_total_exposure = total_exposure.max()
            profile.avg_total_exposure = total_exposure.mean()

            # Turnover
            turnover = self._calculate_turnover(weights)
            profile.turnover_annualized = turnover.sum() * (self.trading_days / len(turnover))
            profile.n_rebalances_per_year = self._count_rebalances(weights) * (
                self.trading_days / len(weights)
            )

            # Comportamiento por régimen
            if regime_labels is not None:
                profile.weights_by_regime = self._weights_by_regime(weights, regime_labels)
                profile.performance_by_regime = self._performance_by_regime(
                    returns, regime_labels
                )

        return profile

    def _annualized_return(self, returns: pd.Series) -> float:
        """Annualized return using numba."""
        ret_arr = returns.dropna().values.astype(np.float64)
        if len(ret_arr) < 2:
            return 0.0
        return float(_annualized_return_numba(ret_arr, self.trading_days))

    def _annualized_volatility(self, returns: pd.Series) -> float:
        """Annualized volatility using numba."""
        ret_arr = returns.dropna().values.astype(np.float64)
        if len(ret_arr) < 2:
            return 0.0
        return float(_volatility_numba(ret_arr, self.trading_days))

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Sharpe ratio using numba."""
        ret_arr = returns.dropna().values.astype(np.float64)
        if len(ret_arr) < 2:
            return 0.0
        return float(_sharpe_numba(ret_arr, self.risk_free_rate, self.trading_days))

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Sortino ratio (using vectorized numpy for downside)."""
        ret_arr = returns.dropna().values.astype(np.float64)
        if len(ret_arr) < 2:
            return 0.0

        daily_rf = self.risk_free_rate / self.trading_days
        excess_mean = np.mean(ret_arr - daily_rf)

        downside = ret_arr[ret_arr < 0]
        if len(downside) < 2:
            return 0.0

        downside_std = np.std(downside, ddof=1)
        if downside_std < 1e-10:
            return 0.0

        return float((excess_mean / downside_std) * np.sqrt(self.trading_days))

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Maximum drawdown using numba."""
        ret_arr = returns.dropna().values.astype(np.float64)
        if len(ret_arr) < 2:
            return 0.0
        return float(_max_drawdown_numba(ret_arr))

    def _max_drawdown_duration(self, returns: pd.Series) -> int:
        """Maximum drawdown duration using numba."""
        ret_arr = returns.dropna().values.astype(np.float64)
        if len(ret_arr) < 2:
            return 0
        return int(_max_drawdown_duration_numba(ret_arr))

    def _calmar_ratio(self, returns: pd.Series) -> float:
        """Calmar ratio."""
        max_dd = abs(self._max_drawdown(returns))
        if max_dd < 1e-10:
            return 0.0
        return self._annualized_return(returns) / max_dd

    def _current_hhi(self, weights: pd.Series) -> float:
        """Índice Herfindahl-Hirschman de concentración."""
        weights_squared = (weights ** 2).sum()
        return weights_squared

    def _avg_hhi(self, weights: pd.DataFrame) -> float:
        """HHI promedio a lo largo del tiempo."""
        hhi_series = (weights ** 2).sum(axis=1)
        return hhi_series.mean()

    def _calculate_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """Turnover: suma de cambios absolutos de peso / 2."""
        weight_changes = weights.diff().abs()
        return weight_changes.sum(axis=1) / 2

    def _rebalance_frequency(self, weights: pd.DataFrame) -> float:
        """Frecuencia media de rebalanceo en días."""
        # Detectar cambios significativos de peso
        weight_changes = weights.diff().abs()
        total_change = weight_changes.sum(axis=1)

        # Umbral: considerar rebalanceo si cambio total > 1%
        rebalance_threshold = 0.01
        rebalance_dates = total_change[total_change > rebalance_threshold].index

        if len(rebalance_dates) < 2:
            return len(weights)  # No hay rebalanceos detectados

        # Calcular días entre rebalanceos
        days_between = pd.Series(rebalance_dates).diff().dt.days.dropna()
        return days_between.mean()

    def _count_rebalances(self, weights: pd.DataFrame, threshold: float = 0.01) -> int:
        """Cuenta número de rebalanceos."""
        weight_changes = weights.diff().abs()
        total_change = weight_changes.sum(axis=1)
        return (total_change > threshold).sum()

    def _avg_weight_change(self, weights: pd.DataFrame) -> float:
        """Cambio promedio de peso por rebalanceo."""
        weight_changes = weights.diff().abs()
        total_change = weight_changes.sum(axis=1)
        # Solo considerar días con cambios
        changes = total_change[total_change > 0.001]
        return changes.mean() if len(changes) > 0 else 0.0

    def _weight_autocorrelation(self, weights: pd.DataFrame) -> float:
        """Autocorrelación de los cambios de peso (velocidad de reacción)."""
        weight_changes = weights.diff()
        autocorrs = []
        for col in weight_changes.columns:
            ac = weight_changes[col].autocorr(lag=1)
            if not np.isnan(ac):
                autocorrs.append(ac)
        return np.mean(autocorrs) if autocorrs else 0.0

    def _avg_holding_period(self, weights: pd.DataFrame) -> float:
        """
        Periodo medio de holding (días que un peso se mantiene ~constante).
        """
        weight_changes = weights.diff().abs()
        total_change = weight_changes.sum(axis=1)

        # Detectar días sin cambio significativo
        no_change = total_change < 0.01
        holding_periods = []

        current_period = 0
        for is_stable in no_change:
            if is_stable:
                current_period += 1
            else:
                if current_period > 0:
                    holding_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            holding_periods.append(current_period)

        return np.mean(holding_periods) if holding_periods else 1.0

    def _weights_by_regime(
        self, weights: pd.DataFrame, regime_labels: pd.Series
    ) -> dict:
        """Analiza cómo cambian los pesos por régimen."""
        aligned = pd.concat([weights, regime_labels.rename("regime")], axis=1, join="inner")

        results = {}
        for regime in aligned["regime"].unique():
            regime_weights = aligned[aligned["regime"] == regime].drop("regime", axis=1)
            results[regime] = {
                "avg_weights": regime_weights.mean().to_dict(),
                "std_weights": regime_weights.std().to_dict(),
                "avg_hhi": self._avg_hhi(regime_weights),
                "n_days": len(regime_weights),
            }

        return results

    def _performance_by_regime(
        self, returns: pd.Series, regime_labels: pd.Series
    ) -> dict:
        """Performance del benchmark por régimen."""
        aligned = pd.concat([returns.rename("returns"), regime_labels.rename("regime")], axis=1, join="inner")

        results = {}
        for regime in aligned["regime"].unique():
            regime_returns = aligned[aligned["regime"] == regime]["returns"]
            if len(regime_returns) > 10:
                results[regime] = {
                    "annualized_return": self._annualized_return(regime_returns),
                    "annualized_volatility": self._annualized_volatility(regime_returns),
                    "sharpe_ratio": self._sharpe_ratio(regime_returns),
                    "max_drawdown": self._max_drawdown(regime_returns),
                    "n_days": len(regime_returns),
                }

        return results

    def generate_report(self, profile: BenchmarkProfile) -> str:
        """
        Genera informe textual del perfil del benchmark.

        Args:
            profile: BenchmarkProfile con análisis.

        Returns:
            String con informe formateado.
        """
        report = """
BENCHMARK PROFILE
═══════════════════════════════════════════════════════

PERFORMANCE
───────────────────────────────────────────────────────
Retorno anualizado:        {ann_return:>10.2%}
Volatilidad anualizada:    {ann_vol:>10.2%}
Sharpe Ratio:              {sharpe:>10.2f}
Sortino Ratio:             {sortino:>10.2f}
Calmar Ratio:              {calmar:>10.2f}
Max Drawdown:              {max_dd:>10.2%}
Max Drawdown Duration:     {max_dd_days:>10} días
VaR 95%:                   {var:>10.2%}
CVaR 95%:                  {cvar:>10.2%}

POLÍTICA DE SIZING
───────────────────────────────────────────────────────
Concentración HHI (actual): {hhi_current:>10.3f}
Concentración HHI (media):  {hhi_avg:>10.3f}
Exposición total máxima:    {max_exp:>10.2%}
Exposición total media:     {avg_exp:>10.2%}

POLÍTICA TEMPORAL
───────────────────────────────────────────────────────
Frecuencia rebalanceo:     {rebal_freq:>10.1f} días
Cambio medio por rebal:    {avg_change:>10.2%}
Holding period medio:      {holding:>10.1f} días
Autocorr. cambios peso:    {autocorr:>10.2f}

TURNOVER
───────────────────────────────────────────────────────
Turnover anualizado:       {turnover:>10.2%}
Rebalanceos por año:       {n_rebal:>10.1f}
""".format(
            ann_return=profile.annualized_return,
            ann_vol=profile.annualized_volatility,
            sharpe=profile.sharpe_ratio,
            sortino=profile.sortino_ratio,
            calmar=profile.calmar_ratio,
            max_dd=profile.max_drawdown,
            max_dd_days=profile.max_drawdown_duration,
            var=profile.var_95,
            cvar=profile.cvar_95,
            hhi_current=profile.concentration_hhi,
            hhi_avg=profile.concentration_hhi_avg,
            max_exp=profile.max_total_exposure,
            avg_exp=profile.avg_total_exposure,
            rebal_freq=profile.rebalance_frequency_days,
            avg_change=profile.avg_weight_change_per_rebalance,
            holding=profile.avg_holding_period_days,
            autocorr=profile.weight_autocorrelation,
            turnover=profile.turnover_annualized,
            n_rebal=profile.n_rebalances_per_year,
        )

        # Añadir pesos por algoritmo
        if profile.avg_weights:
            report += "\nPESOS POR ALGORITMO\n"
            report += "───────────────────────────────────────────────────────\n"
            report += f"{'Algo':<15} {'Media':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n"
            for algo_id in profile.avg_weights:
                report += f"{algo_id:<15} {profile.avg_weights[algo_id]:>10.2%} "
                report += f"{profile.std_weights.get(algo_id, 0):>10.2%} "
                report += f"{profile.min_weights.get(algo_id, 0):>10.2%} "
                report += f"{profile.max_weights.get(algo_id, 0):>10.2%}\n"

        return report
