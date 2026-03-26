"""
Feature engineering avanzado para algoritmos.

Genera features en tres bloques:
1. Actividad: cuándo vive, duración, patrones de actividad
2. Rendimiento: métricas financieras sobre periodos activos
3. Transición: cómo evoluciona el comportamiento con la edad
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class AlgoFeatureConfig:
    """Configuración para extracción de features."""
    min_active_days: int = 60  # Mínimo días activos para incluir
    tercile_split: bool = True  # Dividir vida en tercios
    compute_autocorr: bool = True  # Calcular autocorrelaciones
    dd_threshold: float = 0.05  # Umbral para "primer drawdown relevante"


class AlgoFeatureExtractor:
    """
    Extrae features avanzadas de algoritmos considerando su ciclo de vida.

    Bloques de features:
    - Actividad: start_idx, end_idx, duration, active_ratio, n_gaps
    - Rendimiento: return, vol, sharpe, sortino, max_dd, var, skew, kurt
    - Transición: decay, stability, time_to_dd, tercile performance
    """

    def __init__(self, config: Optional[AlgoFeatureConfig] = None):
        self.config = config or AlgoFeatureConfig()

    def extract_all_features(
        self,
        returns_matrix: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Extrae features completas para todos los algoritmos.

        Args:
            returns_matrix: DataFrame [dates x algos] con NaN donde no activo
            benchmark_returns: Serie de retornos del benchmark (opcional)

        Returns:
            DataFrame con features por algoritmo
        """
        all_features = []
        total = len(returns_matrix.columns)

        for i, algo_id in enumerate(returns_matrix.columns):
            if (i + 1) % 1000 == 0:
                logger.info(f"Procesando features: {i+1}/{total}")

            returns = returns_matrix[algo_id]

            try:
                features = self.extract_algo_features(
                    algo_id, returns, benchmark_returns
                )
                if features is not None:
                    all_features.append(features)
            except Exception as e:
                logger.debug(f"Error en {algo_id}: {e}")
                continue

        df = pd.DataFrame(all_features).set_index('algo_id')
        logger.info(f"Features extraídas para {len(df)} algoritmos")

        return df

    def extract_algo_features(
        self,
        algo_id: str,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Optional[dict]:
        """
        Extrae features para un algoritmo individual.

        Args:
            algo_id: Identificador del algoritmo
            returns: Serie de retornos (con NaN donde no activo)
            benchmark_returns: Retornos del benchmark

        Returns:
            Dict con todas las features o None si no cumple mínimos
        """
        # Identificar periodos activos
        active_mask = returns.notna() & (returns != 0)
        active_returns = returns[active_mask]

        if len(active_returns) < self.config.min_active_days:
            return None

        features = {'algo_id': algo_id}

        # 1. Features de actividad
        features.update(self._activity_features(returns, active_mask))

        # 2. Features de rendimiento
        features.update(self._performance_features(active_returns))

        # 3. Features de transición/ciclo de vida
        features.update(self._transition_features(active_returns))

        # 4. Features vs benchmark (si disponible)
        if benchmark_returns is not None:
            features.update(self._benchmark_features(active_returns, benchmark_returns))

        return features

    def _activity_features(
        self,
        returns: pd.Series,
        active_mask: pd.Series
    ) -> dict:
        """
        Features de patrón de actividad.

        - start_idx: índice relativo de inicio (0-1)
        - end_idx: índice relativo de fin (0-1)
        - duration_ratio: proporción del estudio que dura
        - active_ratio: proporción de tiempo activo sobre su duración
        - n_gaps: número de interrupciones
        - gap_ratio: proporción de gaps sobre duración
        """
        n_total = len(returns)
        active_indices = np.where(active_mask.values)[0]

        if len(active_indices) == 0:
            return {
                'start_idx': 1.0,
                'end_idx': 1.0,
                'duration_ratio': 0.0,
                'active_ratio': 0.0,
                'n_gaps': 0,
                'gap_ratio': 0.0,
                'n_active_days': 0,
            }

        start_idx = active_indices[0]
        end_idx = active_indices[-1]
        duration = end_idx - start_idx + 1
        n_active = len(active_indices)

        # Contar gaps (interrupciones)
        if len(active_indices) > 1:
            diffs = np.diff(active_indices)
            n_gaps = np.sum(diffs > 1)
            total_gap_days = np.sum(diffs[diffs > 1] - 1)
        else:
            n_gaps = 0
            total_gap_days = 0

        return {
            'start_idx': start_idx / n_total,  # Normalizado 0-1
            'end_idx': end_idx / n_total,
            'duration_ratio': duration / n_total,
            'active_ratio': n_active / duration if duration > 0 else 0,
            'n_gaps': n_gaps,
            'gap_ratio': total_gap_days / duration if duration > 0 else 0,
            'n_active_days': n_active,
        }

    def _performance_features(self, returns: pd.Series) -> dict:
        """
        Features de rendimiento sobre periodos activos.

        Incluye métricas estándar + algunas avanzadas:
        - Ulcer Index
        - Mejor/peor racha
        - Profit factor
        """
        n = len(returns)

        # Básicas
        total_return = (1 + returns).prod() - 1
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Downside
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Drawdown
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Ulcer Index (RMS of drawdowns)
        ulcer_index = np.sqrt((drawdown ** 2).mean())

        # VaR y CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        # Distribución
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Hit ratio
        hit_ratio = (returns > 0).mean()

        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else (10 if gains > 0 else 1)

        # Rachas
        best_streak, worst_streak = self._compute_streaks(returns)

        # Pendiente del equity curve (tendencia)
        if n > 10:
            x = np.arange(n)
            slope, _, r_value, _, _ = stats.linregress(x, equity.values)
            equity_slope = slope * 252  # Anualizado
            equity_r2 = r_value ** 2
        else:
            equity_slope = 0
            equity_r2 = 0

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd,
            'ulcer_index': ulcer_index,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hit_ratio': hit_ratio,
            'profit_factor': min(profit_factor, 10),  # Cap para evitar infinitos
            'best_streak': best_streak,
            'worst_streak': worst_streak,
            'equity_slope': equity_slope,
            'equity_r2': equity_r2,
        }

    def _transition_features(self, returns: pd.Series) -> dict:
        """
        Features de transición y ciclo de vida.

        Analiza cómo cambia el comportamiento con la edad del algoritmo:
        - Performance por tercios de vida
        - Degradación/mejora
        - Estabilidad
        - Tiempo hasta primer drawdown relevante
        """
        n = len(returns)
        features = {}

        # Dividir en tercios
        if self.config.tercile_split and n >= 30:
            tercile_size = n // 3

            first_third = returns.iloc[:tercile_size]
            middle_third = returns.iloc[tercile_size:2*tercile_size]
            last_third = returns.iloc[2*tercile_size:]

            # Retornos por tercio (anualizados)
            features['return_first_third'] = first_third.mean() * 252
            features['return_middle_third'] = middle_third.mean() * 252
            features['return_last_third'] = last_third.mean() * 252

            # Volatilidad por tercio
            features['vol_first_third'] = first_third.std() * np.sqrt(252)
            features['vol_last_third'] = last_third.std() * np.sqrt(252)

            # Sharpe por tercio
            features['sharpe_first_third'] = (
                features['return_first_third'] / features['vol_first_third']
                if features['vol_first_third'] > 0 else 0
            )
            features['sharpe_last_third'] = (
                features['return_last_third'] / features['vol_last_third']
                if features['vol_last_third'] > 0 else 0
            )

            # Degradación: diferencia entre último y primer tercio
            features['return_decay'] = features['return_last_third'] - features['return_first_third']
            features['sharpe_decay'] = features['sharpe_last_third'] - features['sharpe_first_third']
            features['vol_change'] = features['vol_last_third'] - features['vol_first_third']
        else:
            # Valores por defecto si no hay suficientes datos
            features['return_first_third'] = returns.mean() * 252
            features['return_middle_third'] = returns.mean() * 252
            features['return_last_third'] = returns.mean() * 252
            features['vol_first_third'] = returns.std() * np.sqrt(252)
            features['vol_last_third'] = returns.std() * np.sqrt(252)
            features['sharpe_first_third'] = 0
            features['sharpe_last_third'] = 0
            features['return_decay'] = 0
            features['sharpe_decay'] = 0
            features['vol_change'] = 0

        # Estabilidad del rendimiento (rolling sharpe std)
        if n >= 63:
            rolling_ret = returns.rolling(21).mean() * 252
            rolling_vol = returns.rolling(21).std() * np.sqrt(252)
            rolling_sharpe = rolling_ret / rolling_vol.replace(0, np.nan)
            features['sharpe_stability'] = 1 / (rolling_sharpe.std() + 0.01)  # Inverso de std
            features['return_stability'] = 1 / (rolling_ret.std() + 0.01)
        else:
            features['sharpe_stability'] = 0
            features['return_stability'] = 0

        # Tiempo hasta primer drawdown relevante
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max

        dd_threshold = -self.config.dd_threshold
        first_dd_idx = np.where(drawdown.values <= dd_threshold)[0]
        if len(first_dd_idx) > 0:
            features['time_to_first_dd'] = first_dd_idx[0] / n  # Normalizado 0-1
        else:
            features['time_to_first_dd'] = 1.0  # Nunca tuvo DD relevante

        # Autocorrelación (persistencia de retornos)
        if self.config.compute_autocorr and n > 10:
            features['autocorr_1'] = returns.autocorr(lag=1)
            features['autocorr_5'] = returns.autocorr(lag=5) if n > 15 else 0
        else:
            features['autocorr_1'] = 0
            features['autocorr_5'] = 0

        # Rendimiento en primeros N días (early performance)
        early_days = min(21, n // 3)
        if early_days > 5:
            early_returns = returns.iloc[:early_days]
            features['early_return'] = early_returns.mean() * 252
            features['early_sharpe'] = (
                features['early_return'] / (early_returns.std() * np.sqrt(252))
                if early_returns.std() > 0 else 0
            )
        else:
            features['early_return'] = 0
            features['early_sharpe'] = 0

        return features

    def _benchmark_features(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> dict:
        """Features relativas al benchmark."""
        # Alinear
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 20:
            return {
                'corr_benchmark': 0,
                'beta': 1,
                'alpha': 0,
                'trend_score': 0,
                'tracking_error': 0,
                'info_ratio': 0,
            }

        algo_ret = returns.reindex(common_idx).fillna(0)
        bench_ret = benchmark_returns.reindex(common_idx).fillna(0)

        # Correlación (with warning suppression for constant series)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = algo_ret.corr(bench_ret)

        # Beta y Alpha (regresión)
        if bench_ret.std() > 0:
            cov = algo_ret.cov(bench_ret)
            beta = cov / (bench_ret.var())
            alpha = (algo_ret.mean() - beta * bench_ret.mean()) * 252
        else:
            beta = 1
            alpha = 0

        # Trend following score (correlación con benchmark rezagado)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trend_score = algo_ret.corr(bench_ret.shift(1).fillna(0))

        # Tracking error e information ratio
        excess = algo_ret - bench_ret
        tracking_error = excess.std() * np.sqrt(252)
        info_ratio = excess.mean() * 252 / tracking_error if tracking_error > 0 else 0

        return {
            'corr_benchmark': corr if not np.isnan(corr) else 0,
            'beta': beta if not np.isnan(beta) else 1,
            'alpha': alpha if not np.isnan(alpha) else 0,
            'trend_score': trend_score if not np.isnan(trend_score) else 0,
            'tracking_error': tracking_error,
            'info_ratio': info_ratio,
        }

    def _compute_streaks(self, returns: pd.Series) -> tuple[int, int]:
        """Calcula mejor y peor racha consecutiva."""
        signs = np.sign(returns.values)

        best_streak = 0
        worst_streak = 0
        current_pos = 0
        current_neg = 0

        for s in signs:
            if s > 0:
                current_pos += 1
                current_neg = 0
                best_streak = max(best_streak, current_pos)
            elif s < 0:
                current_neg += 1
                current_pos = 0
                worst_streak = max(worst_streak, current_neg)
            else:
                current_pos = 0
                current_neg = 0

        return best_streak, worst_streak


# Feature groups for clustering
ACTIVITY_FEATURES = [
    'start_idx', 'end_idx', 'duration_ratio', 'active_ratio',
    'n_gaps', 'gap_ratio', 'n_active_days'
]

PERFORMANCE_FEATURES = [
    'ann_return', 'ann_vol', 'sharpe', 'sortino', 'max_dd',
    'ulcer_index', 'var_95', 'skewness', 'hit_ratio',
    'profit_factor', 'equity_slope', 'equity_r2'
]

TRANSITION_FEATURES = [
    'return_first_third', 'return_last_third', 'return_decay',
    'sharpe_first_third', 'sharpe_last_third', 'sharpe_decay',
    'vol_change', 'sharpe_stability', 'time_to_first_dd',
    'autocorr_1', 'early_return', 'early_sharpe'
]

BENCHMARK_FEATURES = [
    'corr_benchmark', 'beta', 'alpha', 'trend_score',
    'tracking_error', 'info_ratio'
]

# Convenient combined groups
LIFE_PROFILE_FEATURES = ACTIVITY_FEATURES + ['duration_ratio', 'start_idx', 'end_idx']

FINANCIAL_BEHAVIOR_FEATURES = (
    PERFORMANCE_FEATURES +
    TRANSITION_FEATURES +
    BENCHMARK_FEATURES
)
