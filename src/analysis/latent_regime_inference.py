"""
Inferencia de Regimen Latente para Meta-Allocator.

Este modulo replantea el problema de deteccion de regimenes:
NO intenta adivinar el ciclo macro real, sino inferir un estado latente
del mercado que explique el patron de seleccion del benchmark sobre algoritmos.

Arquitectura de dos capas:
- Capa A: Regimen de mercado global (latente)
- Capa B: Comportamiento del benchmark condicionado al regimen

Pipeline:
1. Preparacion: Grid temporal comun con masks de actividad
2. Taxonomia: Clustering de algoritmos en familias de comportamiento
3. Features agregadas: Series temporales por familia + benchmark + universo
4. Inferencia de regimen: HMM con interpretacion difusa
5. Analisis condicional: Modelar seleccion del benchmark dado el regimen

Referencia: Investment Clock como marco interpretativo, no como etiqueta rigida.
"""

import gc
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats

logger = logging.getLogger(__name__)

# Try to import numba
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


# =============================================================================
# NUMBA JIT-COMPILED FUNCTIONS
# =============================================================================

@njit(cache=True)
def _rolling_mean_correlation_numba(
    returns: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling mean correlation across all pairs (numba optimized)."""
    n_samples, n_assets = returns.shape
    result = np.empty(n_samples, dtype=np.float64)
    result[:] = np.nan

    if n_assets < 2:
        return result

    for t in range(window, n_samples):
        start = t - window

        # Compute mean correlation for this window
        total_corr = 0.0
        n_pairs = 0

        for j1 in range(n_assets):
            for j2 in range(j1 + 1, n_assets):
                # Check if both have valid data
                mean1 = 0.0
                mean2 = 0.0
                count = 0

                for i in range(start, t):
                    v1 = returns[i, j1]
                    v2 = returns[i, j2]
                    if not (np.isnan(v1) or np.isnan(v2)):
                        mean1 += v1
                        mean2 += v2
                        count += 1

                if count < 5:
                    continue

                mean1 /= count
                mean2 /= count

                # Compute correlation
                cov = 0.0
                var1 = 0.0
                var2 = 0.0

                for i in range(start, t):
                    v1 = returns[i, j1]
                    v2 = returns[i, j2]
                    if not (np.isnan(v1) or np.isnan(v2)):
                        d1 = v1 - mean1
                        d2 = v2 - mean2
                        cov += d1 * d2
                        var1 += d1 * d1
                        var2 += d2 * d2

                if var1 > 1e-10 and var2 > 1e-10:
                    corr = cov / np.sqrt(var1 * var2)
                    total_corr += corr
                    n_pairs += 1

        if n_pairs > 0:
            result[t] = total_corr / n_pairs

    return result


@njit(cache=True)
def _compute_leadership_rotation_numba(
    returns: np.ndarray,
    lookback: int = 21,
) -> np.ndarray:
    """Compute leadership rotation (numba optimized)."""
    n_samples, n_assets = returns.shape
    result = np.empty(n_samples, dtype=np.float64)
    result[:] = np.nan

    if n_assets < 5:
        return result

    for t in range(lookback * 2, n_samples):
        # Compute mean returns for previous period
        prev_start = t - lookback * 2
        prev_end = t - lookback
        prev_means = np.empty(n_assets, dtype=np.float64)

        for j in range(n_assets):
            total = 0.0
            count = 0
            for i in range(prev_start, prev_end):
                if not np.isnan(returns[i, j]):
                    total += returns[i, j]
                    count += 1
            prev_means[j] = total / count if count > 0 else np.nan

        # Compute mean returns for current period
        curr_start = t - lookback
        curr_end = t
        curr_means = np.empty(n_assets, dtype=np.float64)

        for j in range(n_assets):
            total = 0.0
            count = 0
            for i in range(curr_start, curr_end):
                if not np.isnan(returns[i, j]):
                    total += returns[i, j]
                    count += 1
            curr_means[j] = total / count if count > 0 else np.nan

        # Filter valid assets
        valid_count = 0
        for j in range(n_assets):
            if not (np.isnan(prev_means[j]) or np.isnan(curr_means[j])):
                valid_count += 1

        if valid_count < 5:
            continue

        # Create arrays for valid assets
        prev_valid = np.empty(valid_count, dtype=np.float64)
        curr_valid = np.empty(valid_count, dtype=np.float64)
        idx = 0
        for j in range(n_assets):
            if not (np.isnan(prev_means[j]) or np.isnan(curr_means[j])):
                prev_valid[idx] = prev_means[j]
                curr_valid[idx] = curr_means[j]
                idx += 1

        # Compute Spearman correlation (simplified rank correlation)
        # Sort and get ranks
        prev_ranks = np.empty(valid_count, dtype=np.float64)
        curr_ranks = np.empty(valid_count, dtype=np.float64)

        for i in range(valid_count):
            prev_rank = 1.0
            curr_rank = 1.0
            for k in range(valid_count):
                if prev_valid[k] > prev_valid[i]:
                    prev_rank += 1
                if curr_valid[k] > curr_valid[i]:
                    curr_rank += 1
            prev_ranks[i] = prev_rank
            curr_ranks[i] = curr_rank

        # Pearson correlation of ranks
        mean_prev = 0.0
        mean_curr = 0.0
        for i in range(valid_count):
            mean_prev += prev_ranks[i]
            mean_curr += curr_ranks[i]
        mean_prev /= valid_count
        mean_curr /= valid_count

        cov = 0.0
        var_prev = 0.0
        var_curr = 0.0
        for i in range(valid_count):
            d_prev = prev_ranks[i] - mean_prev
            d_curr = curr_ranks[i] - mean_curr
            cov += d_prev * d_curr
            var_prev += d_prev * d_prev
            var_curr += d_curr * d_curr

        if var_prev > 1e-10 and var_curr > 1e-10:
            spearman_corr = cov / np.sqrt(var_prev * var_curr)
            result[t] = 1.0 - spearman_corr  # Rotation = 1 - correlation
        else:
            result[t] = 0.0

    return result


@njit(cache=True)
def _compute_activity_time_since_last(
    is_active: np.ndarray,
    timestamps_days: np.ndarray,
) -> np.ndarray:
    """Compute time since last activity (numba optimized)."""
    n_samples, n_assets = is_active.shape
    result = np.empty((n_samples, n_assets), dtype=np.float64)
    result[:] = np.nan

    for j in range(n_assets):
        last_active_time = -1e10

        for i in range(n_samples):
            if is_active[i, j]:
                last_active_time = timestamps_days[i]
                result[i, j] = 0.0
            elif last_active_time > -1e9:
                result[i, j] = timestamps_days[i] - last_active_time

    return result


@njit(cache=True)
def _compute_activity_time_until_next(
    is_active: np.ndarray,
    timestamps_days: np.ndarray,
) -> np.ndarray:
    """Compute time until next activity (numba optimized)."""
    n_samples, n_assets = is_active.shape
    result = np.empty((n_samples, n_assets), dtype=np.float64)
    result[:] = np.nan

    for j in range(n_assets):
        next_active_time = 1e10

        # Iterate backwards
        for i in range(n_samples - 1, -1, -1):
            if is_active[i, j]:
                next_active_time = timestamps_days[i]
                result[i, j] = 0.0
            elif next_active_time < 1e9:
                result[i, j] = next_active_time - timestamps_days[i]

    return result


# =============================================================================
# CONFIGURACION Y ENUMERACIONES
# =============================================================================

class InferenceMethod(Enum):
    """Metodos de inferencia de regimen latente."""
    HMM = "hmm"                          # Hidden Markov Model
    MARKOV_SWITCHING = "markov_switching"  # Markov Switching VAR
    TEMPORAL_CLUSTERING = "temporal_clustering"  # Clustering sobre ventanas
    FUZZY_SCORECARD = "fuzzy_scorecard"   # Scorecard difuso


@dataclass
class ActivityMask:
    """
    Mascara de actividad para algoritmos.

    IMPORTANTE: No usar 9999999 para inactivos.
    Separar explicitamente:
    - is_active: bool, si el algoritmo esta activo
    - value: float, valor observado (solo valido si is_active=True)
    """
    is_active: pd.DataFrame      # [timestamps x algos] bool
    time_since_last: pd.DataFrame  # [timestamps x algos] dias desde ultima obs
    time_until_next: pd.DataFrame  # [timestamps x algos] dias hasta proxima obs


@dataclass
class FamilyAggregates:
    """Metricas agregadas por familia de algoritmos."""
    returns: pd.DataFrame         # [timestamps x families] retorno medio
    n_active: pd.DataFrame        # [timestamps x families] numero activos
    weights: pd.DataFrame         # [timestamps x families] peso benchmark
    dispersion: pd.DataFrame      # [timestamps x families] dispersion interna
    contribution: pd.DataFrame    # [timestamps x families] contribucion al benchmark


@dataclass
class UniverseFeatures:
    """Features agregadas del universo de algoritmos."""
    # Dispersion y correlacion
    cross_sectional_dispersion: pd.Series   # std de retornos entre algos activos
    mean_correlation: pd.Series             # correlacion media entre algos
    pca_first_component: pd.Series          # fuerza del PC1
    pct_positive_returns: pd.Series         # % de algos con retorno positivo

    # Concentracion
    return_concentration_hhi: pd.Series     # HHI de retornos (concentrado en pocos?)
    leadership_rotation: pd.Series          # cambio en ranking de top performers

    # Amplitud
    n_algos_active: pd.Series               # numero de algoritmos activos
    breadth_advance_decline: pd.Series      # (positivos - negativos) / total


@dataclass
class BenchmarkFeatures:
    """Features del comportamiento del benchmark."""
    returns: pd.Series
    volatility: pd.Series             # rolling
    drawdown: pd.Series               # current drawdown
    turnover: pd.Series
    concentration_hhi: pd.Series
    n_positions: pd.Series            # numero de posiciones activas
    exposure_total: pd.Series         # sum of abs weights
    weight_changes: pd.DataFrame      # cambios de peso por algo/familia


@dataclass
class LatentRegimeState:
    """Estado de regimen latente inferido."""
    regime_id: int
    regime_name: str

    # Memberships difusas (0-1 para cada dimension)
    growth_membership: float          # expansion vs contraccion
    risk_membership: float            # risk-on vs risk-off
    volatility_membership: float      # alta vs baja volatilidad
    stress_membership: float          # estrés de mercado

    # Probabilidades (si HMM)
    regime_probabilities: np.ndarray  # P(regime_i | observations)

    # Interpretacion
    description: str


@dataclass
class LatentRegimeResult:
    """Resultado completo de inferencia de regimen."""
    # Series temporales
    regime_labels: pd.Series              # etiqueta por timestamp
    regime_probabilities: pd.DataFrame    # [timestamps x n_regimes]
    fuzzy_memberships: pd.DataFrame       # [timestamps x dimensions]

    # Caracterizacion
    regime_profiles: Dict[int, Dict[str, float]]  # perfil medio por regimen
    regime_names: Dict[int, str]          # nombres interpretativos
    transition_matrix: pd.DataFrame        # matriz de transicion

    # Metricas de calidad
    log_likelihood: float                 # si HMM
    bic: float                            # Bayesian Information Criterion
    silhouette_temporal: float            # coherencia temporal


@dataclass
class BenchmarkConditionalAnalysis:
    """Analisis del benchmark condicionado al regimen."""
    # Por regimen: que familias favorece
    family_weights_by_regime: Dict[str, Dict[str, float]]  # {regime: {family: avg_weight}}
    family_overweight_by_regime: Dict[str, Dict[str, float]]  # vs neutro

    # Probabilidad de seleccion
    selection_probability: pd.DataFrame   # P(familia seleccionada | regimen)

    # Performance condicional
    family_returns_by_regime: Dict[str, Dict[str, float]]
    contribution_by_regime: Dict[str, Dict[str, float]]

    # Modelo de seleccion
    selection_model: Optional[Any]        # modelo entrenado (XGBoost, logistic, etc.)
    selection_features_importance: Dict[str, float]


# =============================================================================
# CLASE PRINCIPAL: LatentRegimeInference
# =============================================================================

class LatentRegimeInference:
    """
    Inferencia de regimen latente basada en el comportamiento del benchmark.

    Objetivo: NO predecir el ciclo macro real, sino inferir un estado de mercado
    que explique por que el benchmark rota entre familias de algoritmos.

    Uso:
        inference = LatentRegimeInference(n_regimes=4)

        # 1. Preparar datos con masks de actividad
        mask = inference.build_activity_mask(algo_returns)

        # 2. Crear familias de algoritmos
        family_labels = inference.assign_families(algo_features, method='cluster')

        # 3. Construir features agregadas
        features = inference.build_temporal_features(
            algo_returns, benchmark_weights, benchmark_returns, family_labels, mask
        )

        # 4. Inferir regimenes
        result = inference.infer_regimes(features, method=InferenceMethod.HMM)

        # 5. Analizar benchmark condicionado a regimenes
        conditional = inference.analyze_benchmark_conditional(
            benchmark_weights, family_labels, result.regime_labels
        )
    """

    def __init__(
        self,
        n_regimes: int = 4,
        min_active_algos: int = 10,
        resampling_freq: str = '4H',  # Grid temporal comun
        vol_window: int = 21,
        correlation_window: int = 63,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.min_active_algos = min_active_algos
        self.resampling_freq = resampling_freq
        self.vol_window = vol_window
        self.correlation_window = correlation_window
        self.random_state = random_state

        self._scaler = RobustScaler()
        self._hmm_model = None
        self._regime_result = None

    # =========================================================================
    # FASE 1: PREPARACION Y MASKS DE ACTIVIDAD
    # =========================================================================

    def build_activity_mask(
        self,
        algo_returns: pd.DataFrame,
        fill_method: str = 'mask',  # 'mask', 'forward_fill', 'zero'
    ) -> ActivityMask:
        """
        Construye mascara de actividad para algoritmos (numba optimized).

        NO usar valores como 9999999 para indicar inactividad.
        Mantener separacion explicita entre:
        - Mascara de actividad (is_active)
        - Valores observados (solo donde is_active=True)

        Args:
            algo_returns: DataFrame [timestamps x algo_ids] con retornos
            fill_method: Como tratar los missing values
                - 'mask': Mantener NaN, usar solo con mascara
                - 'forward_fill': Forward fill limitado
                - 'zero': Rellenar con 0 (NO recomendado)

        Returns:
            ActivityMask con is_active, time_since_last, time_until_next
        """
        # An algorithm is "active" if return is NOT NaN AND NOT zero
        is_active = (~algo_returns.isna()) & (algo_returns != 0)

        # Convert index to numeric (days since epoch) for efficient computation
        timestamps = algo_returns.index
        timestamps_numeric = (timestamps - pd.Timestamp('1970-01-01')).days.values.astype(np.float64)

        # Use numba-optimized functions
        is_active_arr = is_active.values

        if HAS_NUMBA:
            time_since_last_arr = _compute_activity_time_since_last(
                is_active_arr, timestamps_numeric
            )
            time_until_next_arr = _compute_activity_time_until_next(
                is_active_arr, timestamps_numeric
            )
        else:
            # Fallback to loop-based implementation
            n_rows, n_cols = algo_returns.shape
            time_since_last_arr = np.full((n_rows, n_cols), np.nan)
            time_until_next_arr = np.full((n_rows, n_cols), np.nan)

            for col_idx, col in enumerate(algo_returns.columns):
                active_mask = is_active[col].values
                if not active_mask.any():
                    continue

                active_indices = np.where(active_mask)[0]
                active_timestamps = timestamps_numeric[active_indices]

                insert_positions = np.searchsorted(active_timestamps, timestamps_numeric, side='right')
                for i in range(n_rows):
                    pos = insert_positions[i]
                    if pos > 0:
                        time_since_last_arr[i, col_idx] = timestamps_numeric[i] - active_timestamps[pos - 1]

                insert_positions_left = np.searchsorted(active_timestamps, timestamps_numeric, side='left')
                for i in range(n_rows):
                    pos = insert_positions_left[i]
                    if pos < len(active_timestamps):
                        if active_timestamps[pos] == timestamps_numeric[i]:
                            if pos + 1 < len(active_timestamps):
                                time_until_next_arr[i, col_idx] = active_timestamps[pos + 1] - timestamps_numeric[i]
                        else:
                            time_until_next_arr[i, col_idx] = active_timestamps[pos] - timestamps_numeric[i]

        # Convert back to DataFrames
        time_since_last = pd.DataFrame(
            time_since_last_arr,
            index=algo_returns.index,
            columns=algo_returns.columns,
        )
        time_until_next = pd.DataFrame(
            time_until_next_arr,
            index=algo_returns.index,
            columns=algo_returns.columns,
        )

        logger.info(
            f"Activity mask built: {len(algo_returns)} timestamps, "
            f"{len(algo_returns.columns)} algos, "
            f"avg active: {is_active.sum(axis=1).mean():.1f}"
        )

        return ActivityMask(
            is_active=is_active,
            time_since_last=time_since_last,
            time_until_next=time_until_next,
        )

    def resample_to_common_grid(
        self,
        algo_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.Series,
        freq: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Resamplea todas las series a un grid temporal comun.

        Args:
            algo_returns: Retornos de algoritmos
            benchmark_weights: Pesos del benchmark
            benchmark_returns: Retornos del benchmark
            freq: Frecuencia de resampleo (default: self.resampling_freq)

        Returns:
            Tuple con (algo_returns, benchmark_weights, benchmark_returns) resampleados
        """
        freq = freq or self.resampling_freq

        # Resamplear retornos (sumar para acumular)
        algo_returns_resampled = algo_returns.resample(freq).sum()
        benchmark_returns_resampled = benchmark_returns.resample(freq).sum()

        # Resamplear pesos (ultimo del periodo)
        benchmark_weights_resampled = benchmark_weights.resample(freq).last()

        # Alinear indices
        common_index = algo_returns_resampled.index.intersection(
            benchmark_weights_resampled.index
        ).intersection(benchmark_returns_resampled.index)

        logger.info(f"Resampled to {freq}: {len(common_index)} common timestamps")

        return (
            algo_returns_resampled.loc[common_index],
            benchmark_weights_resampled.loc[common_index],
            benchmark_returns_resampled.loc[common_index],
        )

    # =========================================================================
    # FASE 2: TAXONOMIA DE ALGORITMOS (FAMILIAS)
    # =========================================================================

    def compute_algo_behavioral_features(
        self,
        algo_returns: pd.DataFrame,
        mask: Optional[ActivityMask] = None,
    ) -> pd.DataFrame:
        """
        Calcula features de comportamiento para clusterizar algoritmos en familias (vectorized).

        Features (por algoritmo):
        - Performance: return, vol, sharpe, sortino, max_dd
        - Distribucion: skew, kurtosis, tail_ratio
        - Persistencia: autocorr_1, autocorr_5
        - Sensibilidades: beta_universe, beta_drawdowns
        - Estabilidad: sharpe_stability, return_stability

        Args:
            algo_returns: DataFrame de retornos
            mask: Mascara de actividad (opcional)

        Returns:
            DataFrame [algo_ids x features]
        """
        # Count valid data points per column
        valid_counts = algo_returns.notna().sum()
        valid_mask = valid_counts >= 30

        # Initialize result DataFrame
        result = pd.DataFrame(index=algo_returns.columns)

        # Calcular retorno del universo como proxy
        universe_return = algo_returns.mean(axis=1, skipna=True)
        universe_drawdown = self._compute_drawdown_series(universe_return)

        # ===== Vectorized basic performance metrics =====
        result['ann_return'] = algo_returns.mean() * 252
        result['ann_vol'] = algo_returns.std() * np.sqrt(252)

        # Sharpe ratio (vectorized with safe division)
        with np.errstate(divide='ignore', invalid='ignore'):
            result['sharpe'] = np.where(
                result['ann_vol'] > 0,
                result['ann_return'] / result['ann_vol'],
                0
            )

        # Sortino (downside vol) - needs per-column computation for downside
        downside_returns = algo_returns.clip(upper=0)
        downside_vol = downside_returns.std() * np.sqrt(252)
        with np.errstate(divide='ignore', invalid='ignore'):
            result['sortino'] = np.where(
                downside_vol > 0,
                result['ann_return'] / downside_vol,
                result['sharpe']  # Fall back to Sharpe if no downside
            )

        # Max drawdown (vectorized)
        equity = (1 + algo_returns.fillna(0)).cumprod()
        running_max = equity.cummax()
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (equity - running_max) / running_max
        result['max_dd'] = drawdown.min()

        # ===== Vectorized distribution metrics =====
        result['skewness'] = algo_returns.skew()
        result['kurtosis'] = algo_returns.kurtosis()

        # Tail ratio (vectorized)
        p95 = algo_returns.quantile(0.95)
        p05 = algo_returns.quantile(0.05)
        with np.errstate(divide='ignore', invalid='ignore'):
            tail_ratio = np.abs(p95 / p05)
        result['tail_ratio'] = np.clip(tail_ratio.fillna(1.0), 0, 10)

        # ===== Autocorrelation (needs loop but optimized) =====
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autocorr_1 = algo_returns.apply(lambda x: x.autocorr(lag=1) if x.notna().sum() > 5 else 0)
            autocorr_5 = algo_returns.apply(lambda x: x.autocorr(lag=5) if x.notna().sum() > 10 else 0)
        result['autocorr_1'] = autocorr_1.fillna(0)
        result['autocorr_5'] = autocorr_5.fillna(0)

        # ===== Beta to universe (vectorized correlation) =====
        # Compute beta_universe using corrwith and covariance
        import warnings
        combined = algo_returns.join(universe_return.rename('_universe_'), how='inner')
        if len(combined) > 30:
            universe_col = combined['_universe_']
            algo_cols = combined.drop(columns=['_universe_'])

            # Covariance with universe (suppress warnings for small samples)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cov_with_universe = algo_cols.apply(lambda x: x.cov(universe_col))
                var_universe = universe_col.var()

            with np.errstate(divide='ignore', invalid='ignore'):
                result['beta_universe'] = np.where(
                    var_universe > 1e-10,
                    cov_with_universe / var_universe,
                    0
                )

            # Beta to drawdowns (correlation with universe drawdown)
            combined_dd = algo_cols.join(universe_drawdown.rename('_dd_'), how='inner')
            if len(combined_dd) > 30:
                dd_col = combined_dd['_dd_']
                algo_dd_cols = combined_dd.drop(columns=['_dd_'])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result['beta_drawdowns'] = algo_dd_cols.corrwith(dd_col).fillna(0)
            else:
                result['beta_drawdowns'] = 0
        else:
            result['beta_universe'] = 0
            result['beta_drawdowns'] = 0

        # ===== Stability metrics (rolling Sharpe stability) =====
        # Compute rolling mean and std for Sharpe
        rolling_mean = algo_returns.rolling(63, min_periods=30).mean()
        rolling_std = algo_returns.rolling(63, min_periods=30).std()

        with np.errstate(divide='ignore', invalid='ignore'):
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        sharpe_stability = 1 / (rolling_sharpe.std() + 0.01)
        result['sharpe_stability'] = sharpe_stability.fillna(0)

        # Return stability
        rolling_vol = algo_returns.rolling(21, min_periods=10).std()
        return_stability = 1 / (rolling_vol.std() + 0.01)
        result['return_stability'] = return_stability.fillna(0)

        # ===== Filter out algos with insufficient data =====
        result.loc[~valid_mask] = np.nan

        # Replace any remaining NaN/inf with 0
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

        return result

    def assign_families(
        self,
        algo_features: pd.DataFrame,
        n_families: int = 8,
        method: str = 'gmm',
        family_names: Optional[Dict[int, str]] = None,
    ) -> pd.Series:
        """
        Asigna algoritmos a familias de comportamiento.

        Familias tipicas:
        - trend_following: alto autocorr, beta positivo a tendencias
        - mean_reversion: autocorr negativo, apuesta contra momentum
        - low_vol_carry: baja vol, bajo max_dd, sharpe estable
        - crisis_alpha: beta negativo a drawdowns, convexidad
        - momentum: alto retorno en periodos de alta dispersion
        - defensive: baja vol, correlacion baja con universo

        Args:
            algo_features: Features por algoritmo
            n_families: Numero de familias
            method: 'gmm', 'kmeans', 'hdbscan'
            family_names: Nombres custom para familias

        Returns:
            Series [algo_id] -> family_id
        """
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import KMeans

        # Preparar features
        X = algo_features.replace([np.inf, -np.inf], np.nan).fillna(algo_features.median())
        X_scaled = self._scaler.fit_transform(X)

        # Clusterizar
        if method == 'gmm':
            model = GaussianMixture(
                n_components=n_families,
                covariance_type='full',
                random_state=self.random_state,
                n_init=3,
            )
            labels = model.fit_predict(X_scaled)
        elif method == 'kmeans':
            model = KMeans(
                n_clusters=n_families,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(X_scaled)
        else:
            try:
                import hdbscan
                model = hdbscan.HDBSCAN(
                    min_cluster_size=max(15, len(X) // 50),
                    min_samples=5,
                )
                labels = model.fit_predict(X_scaled)
            except ImportError:
                logger.warning("hdbscan not installed, using GMM")
                model = GaussianMixture(n_components=n_families, random_state=self.random_state)
                labels = model.fit_predict(X_scaled)

        # Asignar nombres interpretativos
        if family_names is None:
            family_names = self._generate_family_names(algo_features, labels)

        logger.info(f"Assigned {len(set(labels))} families to {len(labels)} algorithms")
        for fam_id in sorted(set(labels)):
            if fam_id >= 0:
                count = (labels == fam_id).sum()
                name = family_names.get(fam_id, f"family_{fam_id}")
                logger.info(f"  {fam_id}: {name} ({count} algos)")

        return pd.Series(labels, index=algo_features.index, name='family')

    def _generate_family_names(
        self,
        algo_features: pd.DataFrame,
        labels: np.ndarray,
    ) -> Dict[int, str]:
        """
        Genera nombres interpretativos para familias basados en características distintivas.

        Naming strategy:
        1. Rank each family on key dimensions
        2. Identify the most distinctive features for each family
        3. Combine into a unique, descriptive name
        """
        df = algo_features.copy()
        df['family'] = labels

        family_means = df.groupby('family').mean()
        n_families = len([f for f in set(labels) if f >= 0])

        # For each feature, rank families (0 = lowest, n-1 = highest)
        family_ranks = family_means.rank(ascending=True)

        names = {}
        for fam_id in sorted(set(labels)):
            if fam_id == -1:
                names[fam_id] = "noise"
                continue

            row = family_means.loc[fam_id]
            ranks = family_ranks.loc[fam_id]

            # Identify distinctive features (top or bottom quartile among families)
            top_threshold = n_families * 0.75
            bottom_threshold = n_families * 0.25

            traits = []

            # Performance dimension (sharpe + sortino combined)
            perf_rank = (ranks.get('sharpe', n_families/2) + ranks.get('sortino', n_families/2)) / 2
            if perf_rank >= top_threshold:
                traits.append(('perf', 'top', perf_rank))
            elif perf_rank <= bottom_threshold:
                traits.append(('perf', 'low', -perf_rank))

            # Volatility dimension
            vol_rank = ranks.get('ann_vol', n_families/2)
            if vol_rank >= top_threshold:
                traits.append(('vol', 'high', vol_rank))
            elif vol_rank <= bottom_threshold:
                traits.append(('vol', 'low', -vol_rank))

            # Drawdown dimension
            dd_rank = ranks.get('max_dd', n_families/2)  # Higher rank = less negative = better
            if dd_rank >= top_threshold:
                traits.append(('dd', 'shallow', dd_rank))
            elif dd_rank <= bottom_threshold:
                traits.append(('dd', 'deep', -dd_rank))

            # Trend vs mean-reversion (autocorrelation)
            if 'autocorr_1' in ranks:
                ac_rank = ranks['autocorr_1']
                if ac_rank >= top_threshold:
                    traits.append(('style', 'trend', ac_rank))
                elif ac_rank <= bottom_threshold:
                    traits.append(('style', 'meanrev', -ac_rank))

            # Crisis behavior (beta to drawdowns)
            if 'beta_drawdowns' in ranks:
                crisis_rank = ranks['beta_drawdowns']
                # Low beta_drawdowns = crisis alpha, high = pro-cyclical
                if crisis_rank <= bottom_threshold:
                    traits.append(('crisis', 'hedge', -crisis_rank))
                elif crisis_rank >= top_threshold:
                    traits.append(('crisis', 'procyc', crisis_rank))

            # Stability dimension
            if 'sharpe_stability' in ranks:
                stab_rank = ranks['sharpe_stability']
                if stab_rank >= top_threshold:
                    traits.append(('stab', 'stable', stab_rank))
                elif stab_rank <= bottom_threshold:
                    traits.append(('stab', 'erratic', -stab_rank))

            # Tail behavior
            if 'tail_ratio' in ranks:
                tail_rank = ranks['tail_ratio']
                if tail_rank >= top_threshold:
                    traits.append(('tail', 'convex', tail_rank))
                elif tail_rank <= bottom_threshold:
                    traits.append(('tail', 'concave', -tail_rank))

            # Select top 2-3 most distinctive traits
            traits.sort(key=lambda x: abs(x[2]), reverse=True)
            selected = traits[:3] if traits else []

            # Build name from selected traits
            name_parts = []
            for dim, label, _ in selected:
                name_parts.append(label)

            if name_parts:
                names[fam_id] = "_".join(name_parts)
            else:
                # Fallback: use most distinctive single feature
                deviations = {}
                for col in family_means.columns:
                    if col != 'family':
                        z = (row[col] - family_means[col].mean()) / (family_means[col].std() + 1e-10)
                        deviations[col] = abs(z)
                if deviations:
                    top_feat = max(deviations, key=deviations.get)
                    direction = "high" if row[top_feat] > family_means[top_feat].mean() else "low"
                    names[fam_id] = f"{direction}_{top_feat}"
                else:
                    names[fam_id] = f"family_{fam_id}"

        # Resolve remaining duplicates by adding cluster size rank
        seen = {}
        family_sizes = df.groupby('family').size()
        for fam_id in sorted(names.keys()):
            name = names[fam_id]
            if name in seen:
                # Add size descriptor: L=large, M=medium, S=small
                size_rank = family_sizes.rank(ascending=False).get(fam_id, 0)
                if size_rank <= len(family_sizes) * 0.33:
                    size_label = "L"
                elif size_rank <= len(family_sizes) * 0.66:
                    size_label = "M"
                else:
                    size_label = "S"
                names[fam_id] = f"{name}_{size_label}"
                # If still duplicate, add number
                if names[fam_id] in seen:
                    seen[name] += 1
                    names[fam_id] = f"{name}_{seen[name]}"
            seen[name] = seen.get(name, 0) + 1

        return names

    # =========================================================================
    # FASE 3: FEATURES TEMPORALES AGREGADAS
    # =========================================================================

    def build_temporal_features(
        self,
        algo_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.Series,
        family_labels: pd.Series,
        mask: Optional[ActivityMask] = None,
    ) -> Tuple[FamilyAggregates, UniverseFeatures, BenchmarkFeatures]:
        """
        Construye features temporales agregadas para inferencia de regimen.

        Features por familia:
        - Retorno medio de algoritmos en la familia
        - Numero de algoritmos activos
        - Peso total del benchmark en la familia
        - Dispersion interna

        Features del universo:
        - Dispersion cross-sectional
        - Correlacion media
        - PCA first component
        - Breadth (advance/decline)

        Features del benchmark:
        - Returns, volatility, drawdown
        - Turnover, concentration
        - Weight changes

        Note: All input indexes are normalized to tz-naive to avoid
        "Cannot join tz-naive with tz-aware DatetimeIndex" errors.

        Args:
            algo_returns: Retornos de algoritmos
            benchmark_weights: Pesos del benchmark
            benchmark_returns: Retornos del benchmark
            family_labels: Asignacion familia -> algo
            mask: Mascara de actividad

        Returns:
            Tuple (FamilyAggregates, UniverseFeatures, BenchmarkFeatures)
        """
        # Normalize timezones to avoid "Cannot join tz-naive with tz-aware" errors
        if hasattr(algo_returns.index, 'tz') and algo_returns.index.tz is not None:
            algo_returns = algo_returns.copy()
            algo_returns.index = algo_returns.index.tz_convert(None)
        if hasattr(benchmark_weights.index, 'tz') and benchmark_weights.index.tz is not None:
            benchmark_weights = benchmark_weights.copy()
            benchmark_weights.index = benchmark_weights.index.tz_convert(None)
        if hasattr(benchmark_returns.index, 'tz') and benchmark_returns.index.tz is not None:
            benchmark_returns = benchmark_returns.copy()
            benchmark_returns.index = benchmark_returns.index.tz_convert(None)

        # Construir mascara si no existe
        if mask is None:
            mask = self.build_activity_mask(algo_returns)

        # 1. Features por familia
        family_features = self._compute_family_aggregates(
            algo_returns, benchmark_weights, family_labels, mask
        )

        # 2. Features del universo
        universe_features = self._compute_universe_features(algo_returns, mask)

        # 3. Features del benchmark
        benchmark_features = self._compute_benchmark_features(
            benchmark_returns, benchmark_weights, family_labels
        )

        return family_features, universe_features, benchmark_features

    def _compute_family_aggregates(
        self,
        algo_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        family_labels: pd.Series,
        mask: ActivityMask,
    ) -> FamilyAggregates:
        """Calcula agregados por familia de algoritmos."""
        # Normalize timezone to avoid "Cannot join tz-naive with tz-aware" errors
        if hasattr(algo_returns.index, 'tz') and algo_returns.index.tz is not None:
            algo_returns = algo_returns.copy()
            algo_returns.index = algo_returns.index.tz_convert(None)
        if hasattr(benchmark_weights.index, 'tz') and benchmark_weights.index.tz is not None:
            benchmark_weights = benchmark_weights.copy()
            benchmark_weights.index = benchmark_weights.index.tz_convert(None)

        families = sorted(set(family_labels[family_labels >= 0]))

        returns_by_family = pd.DataFrame(index=algo_returns.index, columns=families)
        n_active_by_family = pd.DataFrame(index=algo_returns.index, columns=families)
        weights_by_family = pd.DataFrame(index=algo_returns.index, columns=families)
        dispersion_by_family = pd.DataFrame(index=algo_returns.index, columns=families)
        contribution_by_family = pd.DataFrame(index=algo_returns.index, columns=families)

        for family_id in families:
            # Algoritmos en esta familia
            family_algos = family_labels[family_labels == family_id].index
            family_algos = [a for a in family_algos if a in algo_returns.columns]

            if len(family_algos) == 0:
                continue

            # Retornos de la familia
            family_returns = algo_returns[family_algos]

            # Numero de activos
            n_active_by_family[family_id] = mask.is_active[family_algos].sum(axis=1)

            # Retorno medio (solo activos)
            returns_by_family[family_id] = family_returns.mean(axis=1, skipna=True)

            # Dispersion interna
            dispersion_by_family[family_id] = family_returns.std(axis=1, skipna=True)

            # Pesos del benchmark en esta familia
            family_weights_cols = [c for c in family_algos if c in benchmark_weights.columns]
            if family_weights_cols:
                weights_by_family[family_id] = benchmark_weights[family_weights_cols].sum(axis=1)
                # Contribucion al retorno
                weighted_returns = (
                    algo_returns[family_weights_cols] *
                    benchmark_weights[family_weights_cols]
                ).sum(axis=1, skipna=True)
                contribution_by_family[family_id] = weighted_returns

        return FamilyAggregates(
            returns=returns_by_family.astype(float),
            n_active=n_active_by_family.astype(float),
            weights=weights_by_family.astype(float),
            dispersion=dispersion_by_family.astype(float),
            contribution=contribution_by_family.astype(float),
        )

    def _compute_universe_features(
        self,
        algo_returns: pd.DataFrame,
        mask: ActivityMask,
    ) -> UniverseFeatures:
        """Calcula features agregadas del universo de algoritmos."""
        # Dispersion cross-sectional
        cross_sectional_dispersion = algo_returns.std(axis=1, skipna=True)

        # Correlacion media (rolling)
        mean_correlation = self._rolling_mean_correlation(algo_returns, self.correlation_window)

        # PCA first component (proporcion de varianza explicada)
        pca_first = self._rolling_pca_variance(algo_returns, self.correlation_window)

        # % con retorno positivo
        pct_positive = (algo_returns > 0).sum(axis=1) / mask.is_active.sum(axis=1)

        # Concentracion de retornos (HHI)
        abs_returns = algo_returns.abs()
        total_abs = abs_returns.sum(axis=1, skipna=True)
        return_concentration = (
            (abs_returns.div(total_abs + 1e-10, axis=0) ** 2).sum(axis=1, skipna=True)
        )

        # Rotacion de liderazgo
        leadership_rotation = self._compute_leadership_rotation(algo_returns)

        # Numero de activos
        n_active = mask.is_active.sum(axis=1)

        # Breadth
        n_positive = (algo_returns > 0).sum(axis=1)
        n_negative = (algo_returns < 0).sum(axis=1)
        breadth = (n_positive - n_negative) / (n_positive + n_negative + 1)

        return UniverseFeatures(
            cross_sectional_dispersion=cross_sectional_dispersion,
            mean_correlation=mean_correlation,
            pca_first_component=pca_first,
            pct_positive_returns=pct_positive,
            return_concentration_hhi=return_concentration,
            leadership_rotation=leadership_rotation,
            n_algos_active=n_active,
            breadth_advance_decline=breadth,
        )

    def _compute_benchmark_features(
        self,
        benchmark_returns: pd.Series,
        benchmark_weights: pd.DataFrame,
        family_labels: pd.Series,
    ) -> BenchmarkFeatures:
        """Calcula features del benchmark."""
        # Volatilidad rolling
        volatility = benchmark_returns.rolling(self.vol_window).std() * np.sqrt(252)

        # Drawdown actual
        drawdown = self._compute_drawdown_series(benchmark_returns)

        # Turnover
        weight_changes = benchmark_weights.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2

        # Concentracion HHI
        concentration = (benchmark_weights ** 2).sum(axis=1)

        # Numero de posiciones
        n_positions = (benchmark_weights.abs() > 0.001).sum(axis=1)

        # Exposicion total
        exposure = benchmark_weights.abs().sum(axis=1)

        return BenchmarkFeatures(
            returns=benchmark_returns,
            volatility=volatility,
            drawdown=drawdown,
            turnover=turnover,
            concentration_hhi=concentration,
            n_positions=n_positions,
            exposure_total=exposure,
            weight_changes=weight_changes,
        )

    def _rolling_mean_correlation(
        self,
        returns: pd.DataFrame,
        window: int,
        max_assets: int = 50,
    ) -> pd.Series:
        """Calcula correlacion media rolling entre todos los algoritmos (numba optimized)."""
        # Limit assets for efficiency
        if len(returns.columns) > max_assets:
            valid_counts = returns.notna().sum()
            top_assets = valid_counts.nlargest(max_assets).index
            returns_subset = returns[top_assets]
        else:
            returns_subset = returns

        data = returns_subset.values.astype(np.float64)

        if HAS_NUMBA:
            result_arr = _rolling_mean_correlation_numba(data, window)
        else:
            # Fallback to loop-based implementation
            result_arr = np.empty(len(returns), dtype=np.float64)
            result_arr[:] = np.nan

            for i in range(window, len(returns)):
                window_data = returns_subset.iloc[i-window:i].dropna(axis=1, how='all')
                if window_data.shape[1] < 3:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr_matrix = window_data.corr()
                    upper_tri = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    result_arr[i] = upper_tri.stack().mean()

        return pd.Series(result_arr, index=returns.index, dtype=float)

    def _rolling_pca_variance(
        self,
        returns: pd.DataFrame,
        window: int,
    ) -> pd.Series:
        """Calcula varianza explicada por PC1 (rolling)."""
        result = pd.Series(index=returns.index, dtype=float)

        for i in range(window, len(returns)):
            window_data = returns.iloc[i-window:i].dropna(axis=1, how='all')
            if window_data.shape[1] < 3 or window_data.shape[0] < 10:
                result.iloc[i] = np.nan
                continue

            try:
                # Normalizar
                X = (window_data - window_data.mean()) / (window_data.std() + 1e-10)
                X = X.fillna(0)

                pca = PCA(n_components=1)
                pca.fit(X)
                result.iloc[i] = pca.explained_variance_ratio_[0]
            except Exception:
                result.iloc[i] = np.nan

        return result

    def _compute_leadership_rotation(
        self,
        returns: pd.DataFrame,
        lookback: int = 21,
        max_assets: int = 100,
    ) -> pd.Series:
        """
        Calcula rotacion de liderazgo (cambio en ranking de top performers).
        Numba optimized.
        """
        # Limit assets for efficiency
        if len(returns.columns) > max_assets:
            valid_counts = returns.notna().sum()
            top_assets = valid_counts.nlargest(max_assets).index
            returns_subset = returns[top_assets]
        else:
            returns_subset = returns

        data = returns_subset.values.astype(np.float64)

        if HAS_NUMBA:
            result_arr = _compute_leadership_rotation_numba(data, lookback)
        else:
            # Fallback to loop-based implementation
            result_arr = np.empty(len(returns), dtype=np.float64)
            result_arr[:] = np.nan

            for i in range(lookback * 2, len(returns)):
                prev_returns = returns_subset.iloc[i-lookback*2:i-lookback].mean(skipna=True)
                prev_rank = prev_returns.rank(ascending=False)

                curr_returns = returns_subset.iloc[i-lookback:i].mean(skipna=True)
                curr_rank = curr_returns.rank(ascending=False)

                common = prev_rank.dropna().index.intersection(curr_rank.dropna().index)
                if len(common) < 5:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = stats.spearmanr(prev_rank[common], curr_rank[common])
                result_arr[i] = 1 - corr

        return pd.Series(result_arr, index=returns.index, dtype=float)

    def _compute_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """Calcula serie de drawdown actual."""
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max

    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Calcula max drawdown."""
        if len(returns) == 0:
            return 0.0
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown.min()

    # =========================================================================
    # FASE 4: INFERENCIA DE REGIMEN LATENTE
    # =========================================================================

    def infer_regimes(
        self,
        family_features: FamilyAggregates,
        universe_features: UniverseFeatures,
        benchmark_features: BenchmarkFeatures,
        method: InferenceMethod = InferenceMethod.HMM,
    ) -> LatentRegimeResult:
        """
        Infiere regimenes latentes a partir de features temporales.

        El objetivo NO es predecir el ciclo macro real, sino encontrar
        un estado latente que explique el patron de seleccion del benchmark.

        Args:
            family_features: Agregados por familia
            universe_features: Features del universo
            benchmark_features: Features del benchmark
            method: Metodo de inferencia

        Returns:
            LatentRegimeResult con labels, probabilidades y caracterizacion
        """
        # Construir matriz de observacion
        obs_matrix = self._build_observation_matrix(
            family_features, universe_features, benchmark_features
        )

        # Inferir segun metodo
        if method == InferenceMethod.HMM:
            result = self._infer_hmm(obs_matrix)
        elif method == InferenceMethod.FUZZY_SCORECARD:
            result = self._infer_fuzzy_scorecard(obs_matrix, benchmark_features)
        elif method == InferenceMethod.TEMPORAL_CLUSTERING:
            result = self._infer_temporal_clustering(obs_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Interpretar regimenes con capa difusa
        result = self._interpret_regimes_fuzzy(result, obs_matrix, benchmark_features)

        self._regime_result = result
        return result

    def _build_observation_matrix(
        self,
        family_features: FamilyAggregates,
        universe_features: UniverseFeatures,
        benchmark_features: BenchmarkFeatures,
    ) -> pd.DataFrame:
        """Construye matriz de observacion para inferencia."""
        obs = pd.DataFrame(index=benchmark_features.returns.index)

        # Features del benchmark
        obs['bench_return'] = benchmark_features.returns
        obs['bench_vol'] = benchmark_features.volatility
        obs['bench_dd'] = benchmark_features.drawdown
        obs['bench_turnover'] = benchmark_features.turnover
        obs['bench_concentration'] = benchmark_features.concentration_hhi

        # Features del universo
        obs['univ_dispersion'] = universe_features.cross_sectional_dispersion
        obs['univ_correlation'] = universe_features.mean_correlation
        obs['univ_pca1'] = universe_features.pca_first_component
        obs['univ_breadth'] = universe_features.breadth_advance_decline
        obs['univ_rotation'] = universe_features.leadership_rotation

        # Retornos por familia
        for col in family_features.returns.columns:
            obs[f'fam_{col}_ret'] = family_features.returns[col]

        # Pesos por familia
        for col in family_features.weights.columns:
            obs[f'fam_{col}_wgt'] = family_features.weights[col]

        # Limpiar
        obs = obs.replace([np.inf, -np.inf], np.nan)
        obs = obs.dropna(how='all')
        obs = obs.ffill().bfill().fillna(0)

        return obs

    def _infer_hmm(self, obs_matrix: pd.DataFrame) -> LatentRegimeResult:
        """Infiere regimenes con Hidden Markov Model."""
        try:
            from hmmlearn import hmm
        except ImportError:
            logger.warning("hmmlearn not installed, falling back to temporal clustering")
            return self._infer_temporal_clustering(obs_matrix)

        # Normalizar
        X = self._scaler.fit_transform(obs_matrix.values)

        # Ajustar HMM
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
            init_params="stmc",
        )

        try:
            model.fit(X)
            self._hmm_model = model

            # Predecir estados y probabilidades
            states = model.predict(X)
            probs = model.predict_proba(X)

            # Log likelihood y BIC
            log_likelihood = model.score(X)
            n_params = (
                self.n_regimes * obs_matrix.shape[1] +  # means
                self.n_regimes * obs_matrix.shape[1] * (obs_matrix.shape[1] + 1) / 2 +  # covs
                self.n_regimes ** 2  # transitions
            )
            bic = -2 * log_likelihood + n_params * np.log(len(X))

            # Matriz de transicion
            trans_matrix = pd.DataFrame(
                model.transmat_,
                index=range(self.n_regimes),
                columns=range(self.n_regimes),
            )

            # Caracterizar estados
            regime_profiles = {}
            for state_id in range(self.n_regimes):
                state_mask = states == state_id
                if state_mask.sum() > 0:
                    state_obs = obs_matrix.iloc[state_mask]
                    regime_profiles[state_id] = state_obs.mean().to_dict()

            # Labels y probabilidades como Series/DataFrame
            regime_labels = pd.Series(states, index=obs_matrix.index, name='regime')
            regime_probs = pd.DataFrame(
                probs,
                index=obs_matrix.index,
                columns=[f'regime_{i}' for i in range(self.n_regimes)]
            )

            logger.info(f"HMM fitted: {self.n_regimes} regimes, log_lik={log_likelihood:.2f}")

            return LatentRegimeResult(
                regime_labels=regime_labels,
                regime_probabilities=regime_probs,
                fuzzy_memberships=pd.DataFrame(),  # Se llena en interpret_regimes_fuzzy
                regime_profiles=regime_profiles,
                regime_names={},  # Se llena en interpret_regimes_fuzzy
                transition_matrix=trans_matrix,
                log_likelihood=log_likelihood,
                bic=bic,
                silhouette_temporal=self._compute_temporal_silhouette(states, obs_matrix),
            )

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return self._infer_temporal_clustering(obs_matrix)

    def _infer_temporal_clustering(self, obs_matrix: pd.DataFrame) -> LatentRegimeResult:
        """Infiere regimenes con clustering temporal sobre ventanas rolling."""
        from sklearn.mixture import GaussianMixture

        # Rolling features
        window = 21
        rolling_features = obs_matrix.rolling(window).mean().dropna()

        # Normalizar
        X = self._scaler.fit_transform(rolling_features.values)

        # Clusterizar
        model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=self.random_state,
            n_init=3,
        )
        labels = model.fit_predict(X)
        probs = model.predict_proba(X)

        # Mapear a indice original
        regime_labels = pd.Series(
            np.nan,
            index=obs_matrix.index,
            name='regime'
        )
        regime_labels.loc[rolling_features.index] = labels
        regime_labels = regime_labels.bfill().fillna(0).astype(int)

        regime_probs = pd.DataFrame(
            index=obs_matrix.index,
            columns=[f'regime_{i}' for i in range(self.n_regimes)],
            dtype=float
        )
        regime_probs.loc[rolling_features.index] = probs
        regime_probs = regime_probs.bfill().fillna(1/self.n_regimes)

        # Caracterizar
        regime_profiles = {}
        for state_id in range(self.n_regimes):
            state_mask = regime_labels == state_id
            if state_mask.sum() > 0:
                state_obs = obs_matrix.loc[state_mask]
                regime_profiles[state_id] = state_obs.mean().to_dict()

        return LatentRegimeResult(
            regime_labels=regime_labels,
            regime_probabilities=regime_probs,
            fuzzy_memberships=pd.DataFrame(),
            regime_profiles=regime_profiles,
            regime_names={},
            transition_matrix=pd.DataFrame(),
            log_likelihood=0,
            bic=0,
            silhouette_temporal=self._compute_temporal_silhouette(
                regime_labels.values, obs_matrix
            ),
        )

    def _infer_fuzzy_scorecard(
        self,
        obs_matrix: pd.DataFrame,
        benchmark_features: BenchmarkFeatures,
    ) -> LatentRegimeResult:
        """
        Infiere regimenes usando scorecard difuso.

        Define factores continuos y calcula memberships difusas.
        """
        memberships = pd.DataFrame(index=obs_matrix.index)

        # Factor 1: Crecimiento implicito (breadth + retornos positivos)
        if 'univ_breadth' in obs_matrix.columns:
            growth_raw = obs_matrix['univ_breadth'].rolling(21).mean()
        else:
            growth_raw = benchmark_features.returns.rolling(63).mean() * 252

        memberships['growth'] = self._fuzzy_membership_sigmoid(growth_raw, center=0, scale=0.5)

        # Factor 2: Volatilidad
        vol = benchmark_features.volatility
        vol_median = vol.expanding().median()
        memberships['high_vol'] = self._fuzzy_membership_sigmoid(
            vol - vol_median, center=0, scale=vol_median.mean() * 0.5
        )

        # Factor 3: Stress (correlacion alta + dispersion baja = panico)
        if 'univ_correlation' in obs_matrix.columns and 'univ_dispersion' in obs_matrix.columns:
            corr = obs_matrix['univ_correlation']
            disp = obs_matrix['univ_dispersion']
            stress_raw = corr - disp.rank(pct=True)
            memberships['stress'] = self._fuzzy_membership_sigmoid(stress_raw, center=0, scale=0.3)
        else:
            memberships['stress'] = benchmark_features.drawdown.abs()

        # Factor 4: Risk-on (turnover bajo + breadth alto)
        if 'bench_turnover' in obs_matrix.columns:
            turnover_low = 1 - obs_matrix['bench_turnover'].rank(pct=True)
            breadth_high = obs_matrix.get('univ_breadth', pd.Series(0, index=obs_matrix.index)).rank(pct=True)
            memberships['risk_on'] = (turnover_low + breadth_high) / 2
        else:
            memberships['risk_on'] = 1 - memberships['stress']

        # Asignar regimenes basado en combinaciones
        regime_labels = self._assign_regimes_from_fuzzy(memberships)

        # Probabilidades simples basadas en memberships
        regime_probs = self._memberships_to_probs(memberships)

        return LatentRegimeResult(
            regime_labels=regime_labels,
            regime_probabilities=regime_probs,
            fuzzy_memberships=memberships,
            regime_profiles={},
            regime_names={},
            transition_matrix=pd.DataFrame(),
            log_likelihood=0,
            bic=0,
            silhouette_temporal=0,
        )

    def _fuzzy_membership_sigmoid(
        self,
        x: pd.Series,
        center: float = 0,
        scale: float = 1,
    ) -> pd.Series:
        """Funcion de membresia sigmoide."""
        return 1 / (1 + np.exp(-(x - center) / scale))

    def _assign_regimes_from_fuzzy(self, memberships: pd.DataFrame) -> pd.Series:
        """Asigna regimenes basado en memberships difusas."""
        regimes = pd.Series(index=memberships.index, dtype=int)

        for i, row in memberships.iterrows():
            growth = row.get('growth', 0.5)
            high_vol = row.get('high_vol', 0.5)
            stress = row.get('stress', 0.5)
            risk_on = row.get('risk_on', 0.5)

            # Expansion: growth + risk_on + low_vol
            expansion_score = growth * risk_on * (1 - high_vol)

            # Slowdown: growth pero vol subiendo
            slowdown_score = growth * high_vol * (1 - stress)

            # Recession: stress + high_vol
            recession_score = stress * high_vol * (1 - growth)

            # Recovery: low stress + vol bajando + growth mejorando
            recovery_score = (1 - stress) * (1 - high_vol) * (1 - growth) * 0.5 + growth * 0.5

            scores = {
                0: expansion_score,
                1: slowdown_score,
                2: recession_score,
                3: recovery_score,
            }
            regimes.loc[i] = max(scores, key=scores.get)

        return regimes

    def _memberships_to_probs(self, memberships: pd.DataFrame) -> pd.DataFrame:
        """Convierte memberships a probabilidades de regimen."""
        probs = pd.DataFrame(index=memberships.index)

        # Simplificacion: usar memberships como base para probs
        growth = memberships.get('growth', pd.Series(0.5, index=memberships.index))
        high_vol = memberships.get('high_vol', pd.Series(0.5, index=memberships.index))
        stress = memberships.get('stress', pd.Series(0.5, index=memberships.index))

        probs['regime_0'] = growth * (1 - high_vol)  # expansion
        probs['regime_1'] = growth * high_vol  # slowdown
        probs['regime_2'] = stress * high_vol  # recession
        probs['regime_3'] = (1 - stress) * (1 - growth)  # recovery

        # Normalizar
        row_sums = probs.sum(axis=1)
        probs = probs.div(row_sums + 1e-10, axis=0)

        return probs

    def _interpret_regimes_fuzzy(
        self,
        result: LatentRegimeResult,
        obs_matrix: pd.DataFrame,
        benchmark_features: BenchmarkFeatures,
    ) -> LatentRegimeResult:
        """
        Interpreta regimenes con capa difusa para asignar nombres.

        Los regimenes descubiertos (ya sea por HMM o clustering) se interpretan
        usando caracteristicas observadas para asignar nombres significativos.
        """
        regime_names = {}

        # If regime_profiles is empty, compute it from regime_labels
        regime_profiles = result.regime_profiles.copy()
        if not regime_profiles:
            for regime_id in result.regime_labels.unique():
                regime_mask = result.regime_labels == regime_id
                # Align mask with obs_matrix index
                common_idx = result.regime_labels.index.intersection(obs_matrix.index)
                mask_aligned = regime_mask.loc[common_idx]
                obs_aligned = obs_matrix.loc[common_idx]

                if mask_aligned.sum() > 0:
                    regime_obs = obs_aligned.loc[mask_aligned]
                    regime_profiles[regime_id] = regime_obs.mean().to_dict()

        for regime_id in sorted(regime_profiles.keys()):
            profile = regime_profiles[regime_id]

            # Extraer caracteristicas clave
            bench_vol = profile.get('bench_vol', 0)
            bench_ret = profile.get('bench_return', 0) if 'bench_return' in profile else profile.get('bench_ret', 0)
            bench_dd = profile.get('bench_dd', 0)
            univ_breadth = profile.get('univ_breadth', 0)
            univ_correlation = profile.get('univ_correlation', 0)

            # Clasificar
            parts = []

            # Trend
            all_bench_ret = obs_matrix.get('bench_return', obs_matrix.get('bench_ret', pd.Series()))
            if len(all_bench_ret) > 0:
                ret_median = all_bench_ret.median()
                if bench_ret > ret_median:
                    parts.append("positive_trend")
                else:
                    parts.append("negative_trend")

            # Volatilidad
            all_bench_vol = obs_matrix.get('bench_vol', pd.Series())
            if len(all_bench_vol) > 0:
                vol_median = all_bench_vol.median()
                if bench_vol > vol_median:
                    parts.append("high_vol")
                else:
                    parts.append("low_vol")

            # Breadth
            if univ_breadth > 0.2:
                parts.append("broad_advance")
            elif univ_breadth < -0.2:
                parts.append("broad_decline")

            # Stress
            if bench_dd < -0.1 or univ_correlation > 0.6:
                parts.append("stress")

            regime_names[regime_id] = "_".join(parts) if parts else f"regime_{regime_id}"

        # Mapear a nombres de Investment Clock si es posible
        regime_names = self._map_to_investment_clock(regime_names, regime_profiles)

        # Actualizar result
        result.regime_names = regime_names
        result.regime_profiles = regime_profiles  # Update with computed profiles

        # Calcular fuzzy memberships si no existen
        if result.fuzzy_memberships.empty:
            result.fuzzy_memberships = self._compute_fuzzy_memberships(obs_matrix)

        return result

    def _map_to_investment_clock(
        self,
        regime_names: Dict[int, str],
        regime_profiles: Dict[int, Dict[str, float]],
    ) -> Dict[int, str]:
        """
        Mapea regimenes descubiertos a fases del Investment Clock.

        Investment Clock:
        - EXPANSION: crecimiento + baja vol + momentum positivo
        - SLOWDOWN: crecimiento desacelerando + vol subiendo
        - RECESSION: contraccion + alta vol
        - RECOVERY: mejora + vol cayendo
        """
        ic_names = ['expansion', 'slowdown', 'recession', 'recovery']

        # Scoring para cada regimen descubierto
        scores = {}
        for regime_id, profile in regime_profiles.items():
            bench_ret = profile.get('bench_return', profile.get('bench_ret', 0))
            bench_vol = profile.get('bench_vol', 0)
            bench_dd = profile.get('bench_dd', 0)

            # Normalizar (asumimos que ya estan en escala razonable)
            ret_norm = 1 / (1 + np.exp(-bench_ret * 10))  # sigmoid
            vol_norm = 1 / (1 + np.exp(-bench_vol * 5))
            dd_norm = abs(bench_dd) if bench_dd < 0 else 0

            # Scores para cada fase IC
            scores[regime_id] = {
                'expansion': ret_norm * (1 - vol_norm) * (1 - dd_norm),
                'slowdown': ret_norm * vol_norm,
                'recession': (1 - ret_norm) * vol_norm * dd_norm,
                'recovery': (1 - dd_norm) * (1 - vol_norm) * (1 - ret_norm) * 0.5,
            }

        # Asignar greedily
        assigned = {}
        used_ic = set()
        used_regimes = set()

        all_pairs = []
        for regime_id, ic_scores in scores.items():
            for ic_name, score in ic_scores.items():
                all_pairs.append((score, regime_id, ic_name))

        all_pairs.sort(reverse=True)

        for score, regime_id, ic_name in all_pairs:
            if regime_id not in used_regimes and ic_name not in used_ic:
                assigned[regime_id] = ic_name
                used_regimes.add(regime_id)
                used_ic.add(ic_name)

            if len(assigned) == len(regime_profiles):
                break

        # Regimenes no asignados mantienen nombre descriptivo
        for regime_id in regime_profiles:
            if regime_id not in assigned:
                assigned[regime_id] = regime_names.get(regime_id, f"regime_{regime_id}")

        return assigned

    def _compute_fuzzy_memberships(self, obs_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calcula memberships difusas para features clave."""
        memberships = pd.DataFrame(index=obs_matrix.index)

        for col in obs_matrix.columns:
            series = obs_matrix[col]
            q25 = series.expanding().quantile(0.25)
            q75 = series.expanding().quantile(0.75)

            # Membership "high"
            memberships[f'{col}_high'] = (series - q25) / (q75 - q25 + 1e-10)
            memberships[f'{col}_high'] = memberships[f'{col}_high'].clip(0, 1)

        return memberships

    def _compute_temporal_silhouette(
        self,
        labels: np.ndarray,
        obs_matrix: pd.DataFrame,
    ) -> float:
        """Calcula silhouette score considerando coherencia temporal."""
        from sklearn.metrics import silhouette_score

        try:
            X = self._scaler.fit_transform(obs_matrix.values)
            # Filtrar ruido si existe
            valid_mask = labels >= 0
            if valid_mask.sum() < 10:
                return 0

            return silhouette_score(X[valid_mask], labels[valid_mask])
        except Exception:
            return 0

    # =========================================================================
    # FASE 5: ANALISIS DEL BENCHMARK CONDICIONADO AL REGIMEN
    # =========================================================================

    def analyze_benchmark_conditional(
        self,
        benchmark_weights: pd.DataFrame,
        family_labels: pd.Series,
        regime_labels: pd.Series,
        algo_returns: Optional[pd.DataFrame] = None,
    ) -> BenchmarkConditionalAnalysis:
        """
        Analiza el comportamiento del benchmark condicionado al regimen.

        Responde a:
        - Dado el regimen inferido, que familias sobrepondera el benchmark?
        - Que probabilidad tiene cada familia de ser seleccionada por regimen?
        - Como es el performance de cada familia por regimen?

        Args:
            benchmark_weights: Pesos del benchmark
            family_labels: Asignacion familia -> algo
            regime_labels: Labels de regimen inferidos
            algo_returns: Retornos de algoritmos (opcional, para performance)

        Returns:
            BenchmarkConditionalAnalysis
        """
        # Normalize timezones to avoid "Cannot join tz-naive with tz-aware" errors
        if hasattr(benchmark_weights.index, 'tz') and benchmark_weights.index.tz is not None:
            benchmark_weights = benchmark_weights.copy()
            benchmark_weights.index = benchmark_weights.index.tz_convert(None)
        if hasattr(regime_labels.index, 'tz') and regime_labels.index.tz is not None:
            regime_labels = regime_labels.copy()
            regime_labels.index = regime_labels.index.tz_convert(None)
        if algo_returns is not None:
            if hasattr(algo_returns.index, 'tz') and algo_returns.index.tz is not None:
                algo_returns = algo_returns.copy()
                algo_returns.index = algo_returns.index.tz_convert(None)

        # Calcular pesos por familia
        weights_by_family = self._compute_weights_by_family(benchmark_weights, family_labels)

        # Align indices between weights_by_family and regime_labels
        common_idx = weights_by_family.index.intersection(regime_labels.index)
        weights_by_family_aligned = weights_by_family.loc[common_idx]
        regime_labels_aligned = regime_labels.loc[common_idx]

        # Pesos promedio por familia y regimen
        family_weights_by_regime = {}
        family_overweight_by_regime = {}

        # Peso neutro (equal weight across families)
        n_families = len(weights_by_family_aligned.columns)
        neutral_weight = 1.0 / n_families if n_families > 0 else 0

        for regime in regime_labels_aligned.unique():
            regime_mask = regime_labels_aligned == regime
            regime_weights = weights_by_family_aligned.loc[regime_mask]

            avg_weights = regime_weights.mean().to_dict()
            family_weights_by_regime[str(regime)] = avg_weights

            # Overweight vs neutro
            overweight = {k: v - neutral_weight for k, v in avg_weights.items()}
            family_overweight_by_regime[str(regime)] = overweight

        # Probabilidad de seleccion (peso > threshold)
        selection_prob = self._compute_selection_probability(
            weights_by_family_aligned, regime_labels_aligned, threshold=0.01
        )

        # Performance por familia y regimen
        family_returns_by_regime = {}
        contribution_by_regime = {}

        if algo_returns is not None:
            returns_by_family = self._compute_returns_by_family(algo_returns, family_labels)

            # Find common index across all three: weights, returns, regimes
            all_common_idx = (
                weights_by_family_aligned.index
                .intersection(returns_by_family.index)
                .intersection(regime_labels_aligned.index)
            )

            weights_aligned = weights_by_family_aligned.loc[all_common_idx]
            returns_aligned = returns_by_family.loc[all_common_idx]
            regimes_aligned = regime_labels_aligned.loc[all_common_idx]

            for regime in regimes_aligned.unique():
                regime_mask = regimes_aligned == regime
                regime_returns = returns_aligned.loc[regime_mask]
                regime_weights = weights_aligned.loc[regime_mask]

                avg_returns = regime_returns.mean().to_dict()
                family_returns_by_regime[str(regime)] = avg_returns

                # Contribucion = peso * retorno
                if len(regime_weights) > 0 and len(regime_returns) > 0:
                    contribution = (regime_weights * regime_returns).mean().to_dict()
                    contribution_by_regime[str(regime)] = contribution

        # Modelo de seleccion (simple: logistic por ahora)
        selection_model, feature_importance = self._fit_selection_model(
            weights_by_family_aligned, regime_labels_aligned
        )

        return BenchmarkConditionalAnalysis(
            family_weights_by_regime=family_weights_by_regime,
            family_overweight_by_regime=family_overweight_by_regime,
            selection_probability=selection_prob,
            family_returns_by_regime=family_returns_by_regime,
            contribution_by_regime=contribution_by_regime,
            selection_model=selection_model,
            selection_features_importance=feature_importance,
        )

    def _compute_weights_by_family(
        self,
        benchmark_weights: pd.DataFrame,
        family_labels: pd.Series,
    ) -> pd.DataFrame:
        """Agrega pesos del benchmark por familia."""
        families = sorted(set(family_labels[family_labels >= 0]))
        weights_by_family = pd.DataFrame(index=benchmark_weights.index, columns=families)

        for family_id in families:
            family_algos = family_labels[family_labels == family_id].index
            family_cols = [c for c in family_algos if c in benchmark_weights.columns]
            if family_cols:
                weights_by_family[family_id] = benchmark_weights[family_cols].sum(axis=1)

        return weights_by_family.fillna(0).astype(float)

    def _compute_returns_by_family(
        self,
        algo_returns: pd.DataFrame,
        family_labels: pd.Series,
    ) -> pd.DataFrame:
        """Calcula retornos promedio por familia."""
        families = sorted(set(family_labels[family_labels >= 0]))
        returns_by_family = pd.DataFrame(index=algo_returns.index, columns=families)

        for family_id in families:
            family_algos = family_labels[family_labels == family_id].index
            family_cols = [c for c in family_algos if c in algo_returns.columns]
            if family_cols:
                returns_by_family[family_id] = algo_returns[family_cols].mean(axis=1)

        return returns_by_family.fillna(0).astype(float)

    def _compute_selection_probability(
        self,
        weights_by_family: pd.DataFrame,
        regime_labels: pd.Series,
        threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Calcula P(familia seleccionada | regimen).

        Una familia se considera "seleccionada" si peso > threshold.
        """
        regimes = sorted(regime_labels.unique())
        families = weights_by_family.columns

        prob_df = pd.DataFrame(index=regimes, columns=families, dtype=float)

        for regime in regimes:
            regime_mask = regime_labels == regime
            regime_weights = weights_by_family.loc[regime_mask]

            for family in families:
                # Probabilidad de que peso > threshold
                selected = (regime_weights[family] > threshold).sum()
                total = len(regime_weights)
                prob_df.loc[regime, family] = selected / total if total > 0 else 0

        return prob_df

    def _fit_selection_model(
        self,
        weights_by_family: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> Tuple[Optional[Any], Dict[str, float]]:
        """
        Ajusta un modelo simple para predecir seleccion de familias dado el regimen.

        Returns:
            Tuple (modelo, importancia de features)
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder

            # Preparar datos
            X = pd.get_dummies(regime_labels, prefix='regime')
            y_multi = (weights_by_family > 0.01).astype(int)  # Multi-label

            # Ajustar un modelo por familia
            models = {}
            importance = {}

            for family in weights_by_family.columns:
                y = y_multi[family]
                if y.nunique() < 2:
                    continue

                common_idx = X.index.intersection(y.index)
                X_fit = X.loc[common_idx]
                y_fit = y.loc[common_idx]

                model = LogisticRegression(random_state=self.random_state, max_iter=200)
                model.fit(X_fit, y_fit)
                models[family] = model

                # Importancia (coeficientes)
                for i, col in enumerate(X_fit.columns):
                    key = f"{family}_{col}"
                    importance[key] = model.coef_[0][i]

            return models, importance

        except Exception as e:
            logger.warning(f"Could not fit selection model: {e}")
            return None, {}

    # =========================================================================
    # REPORTING Y VISUALIZACION
    # =========================================================================

    def generate_report(
        self,
        result: LatentRegimeResult,
        conditional: BenchmarkConditionalAnalysis,
    ) -> str:
        """Genera reporte textual del analisis."""
        report = """
LATENT REGIME INFERENCE REPORT
═══════════════════════════════════════════════════════════════════

OBJETIVO
────────────────────────────────────────────────────────────────────
Inferir un estado latente del mercado que explique el patron de
seleccion y rotacion del benchmark entre familias de algoritmos.
(NO intentamos predecir el ciclo macro real)

REGIMENES DETECTADOS
────────────────────────────────────────────────────────────────────
"""
        # Estadisticas por regimen
        for regime_id, name in result.regime_names.items():
            n_days = (result.regime_labels == regime_id).sum()
            pct_time = n_days / len(result.regime_labels) * 100
            report += f"\n{regime_id}. {name.upper()}\n"
            report += f"   Dias: {n_days} ({pct_time:.1f}% del tiempo)\n"

            if regime_id in result.regime_profiles:
                profile = result.regime_profiles[regime_id]
                if 'bench_vol' in profile:
                    report += f"   Vol benchmark: {profile['bench_vol']:.2%}\n"
                if 'bench_ret' in profile or 'bench_return' in profile:
                    ret = profile.get('bench_ret', profile.get('bench_return', 0))
                    report += f"   Retorno benchmark: {ret:.4f}\n"

        report += """
COMPORTAMIENTO DEL BENCHMARK POR REGIMEN
────────────────────────────────────────────────────────────────────
"""
        for regime, weights in conditional.family_weights_by_regime.items():
            report += f"\nRegimen {regime}:\n"
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for family, weight in sorted_weights[:5]:
                overweight = conditional.family_overweight_by_regime[regime].get(family, 0)
                sign = "+" if overweight > 0 else ""
                report += f"   {family}: {weight:.1%} (vs neutro: {sign}{overweight:.1%})\n"

        report += """
PROBABILIDAD DE SELECCION (P(familia | regimen))
────────────────────────────────────────────────────────────────────
"""
        report += conditional.selection_probability.to_string()

        report += """

METRICAS DE CALIDAD
────────────────────────────────────────────────────────────────────
"""
        report += f"Log-likelihood: {result.log_likelihood:.2f}\n"
        report += f"BIC: {result.bic:.2f}\n"
        report += f"Silhouette temporal: {result.silhouette_temporal:.3f}\n"

        return report


# =============================================================================
# INVESTMENT CLOCK REGIME INFERENCE (Asset-Class Based)
# =============================================================================

class BusinessCyclePhase(Enum):
    """Four phases of the business cycle / Investment Clock."""
    RECOVERY = 0      # Below trend but rebounding
    EXPANSION = 1     # Healthy growth, neutral policy
    SLOWDOWN = 2      # Peak growth, decelerating
    CONTRACTION = 3   # Declining activity


@dataclass
class AssetClassPerformance:
    """Rolling performance by asset class."""
    returns: pd.DataFrame          # [date x asset_class]
    volatility: pd.DataFrame       # [date x asset_class]
    n_algorithms: pd.DataFrame     # [date x asset_class] count of algos


@dataclass
class RegimeIndicators:
    """Investment Clock regime indicators computed from asset class performance."""
    cyclical_vs_defensive: pd.Series    # Spread: cyclical - defensive
    credit_spread: pd.Series            # Spread: HY - Gov (risk appetite)
    equity_vs_bonds: pd.Series          # Spread: equity - bonds
    volatility_level: pd.Series         # Cross-sectional volatility
    correlation_level: pd.Series        # Cross-sectional correlation
    momentum_breadth: pd.Series         # % of asset classes with positive momentum


@dataclass
class InvestmentClockResult:
    """Result of Investment Clock regime inference."""
    regime_labels: pd.Series              # [date] -> BusinessCyclePhase
    regime_probabilities: pd.DataFrame    # [date x phase] -> probability
    indicators: RegimeIndicators          # Underlying indicators
    phase_statistics: Dict[int, Dict]     # Statistics per phase
    transition_matrix: np.ndarray         # Phase transition probabilities


class InvestmentClockRegimeInference:
    """
    Infers business cycle regime from asset class performance patterns.

    This approach follows the Investment Clock framework:
    - Recovery: Cyclicals outperform, Value > Growth, Credit > Gov
    - Expansion: Broad equity outperformance, Small > Large
    - Slowdown: Defensives start leading, Long bonds > Short
    - Contraction: Defensives dominate, Gov > Credit, High vol

    Key difference from LatentRegimeInference:
    - Uses CAUSE (asset class performance) not EFFECT (benchmark selection)
    - Avoids lookahead bias by using only rolling windows
    - Groups algorithms by inferred asset class, not behavioral clustering
    """

    # Asset class categories for Investment Clock
    ASSET_CLASS_MAPPING = {
        # Cyclical sectors (do well in recovery/expansion)
        'cyclical': [
            'technology', 'tech', 'materials', 'industrials', 'industrial',
            'consumer_discretionary', 'consumer_disc', 'financials', 'financial'
        ],
        # Defensive sectors (do well in slowdown/contraction)
        'defensive': [
            'utilities', 'utility', 'consumer_staples', 'staples',
            'healthcare', 'health', 'telecom', 'telecommunications'
        ],
        # High yield / credit (risk appetite indicator)
        'high_yield': [
            'high_yield', 'hy', 'junk', 'corporate_bond', 'corp_bond',
            'investment_grade', 'ig'
        ],
        # Government / Treasuries (safe haven)
        'government': [
            'treasury', 'treasuries', 'government', 'gov_bond', 'sovereign'
        ],
        # Equity indices
        'equity_index': [
            'sp500', 's&p500', 'nasdaq', 'dow', 'russell', 'equity', 'stock'
        ],
        # Commodities
        'commodities': [
            'commodity', 'commodities', 'gold', 'silver', 'oil', 'crude',
            'natural_gas', 'copper', 'agricultural'
        ],
        # Forex (currency)
        'forex': [
            'forex', 'fx', 'currency', 'usd', 'eur', 'jpy', 'gbp'
        ],
    }

    def __init__(
        self,
        rolling_window: int = 21,        # ~1 month for regime indicators
        vol_window: int = 63,             # ~3 months for volatility
        rebalance_freq: str = 'W',        # Weekly regime updates
        random_state: int = 42,
    ):
        self.rolling_window = rolling_window
        self.vol_window = vol_window
        self.rebalance_freq = rebalance_freq
        self.random_state = random_state
        self._scaler = RobustScaler()

    def infer_regime_from_assets(
        self,
        algo_returns: pd.DataFrame,
        asset_inferences: Dict[str, 'AssetInference'],
    ) -> InvestmentClockResult:
        """
        Infer business cycle regime from algorithm returns and their inferred assets.

        Args:
            algo_returns: Daily returns [date x algo_id]
            asset_inferences: Mapping algo_id -> AssetInference from asset inference engine

        Returns:
            InvestmentClockResult with regime labels and indicators
        """
        logger.info("Starting Investment Clock regime inference...")

        # 1. Map algorithms to asset class categories
        algo_to_category = self._map_algos_to_categories(asset_inferences)
        logger.info(f"Mapped {len(algo_to_category)} algorithms to asset categories")

        # 2. Compute rolling performance by asset class
        asset_performance = self._compute_asset_class_performance(
            algo_returns, algo_to_category
        )
        logger.info(f"Computed performance for {len(asset_performance.returns.columns)} asset classes")

        # 3. Build regime indicators
        indicators = self._build_regime_indicators(asset_performance)
        logger.info("Built regime indicators")

        # 4. Infer regime using HMM
        result = self._infer_regime_hmm(indicators, asset_performance)
        logger.info(f"Inferred {len(result.regime_labels.unique())} distinct regimes")

        return result

    def _map_algos_to_categories(
        self,
        asset_inferences: Dict[str, 'AssetInference'],
    ) -> Dict[str, str]:
        """Map algorithms to Investment Clock asset categories."""
        algo_to_category = {}

        for algo_id, inference in asset_inferences.items():
            # Get asset class and name
            asset_class = inference.asset_class.lower() if inference.asset_class else 'unknown'
            asset_name = inference.predicted_asset.lower() if inference.predicted_asset else ''

            # Try to match to a category
            category = self._classify_asset(asset_class, asset_name)
            algo_to_category[algo_id] = category

        return algo_to_category

    def _classify_asset(self, asset_class: str, asset_name: str) -> str:
        """Classify an asset into Investment Clock category."""
        combined = f"{asset_class} {asset_name}".lower()

        for category, keywords in self.ASSET_CLASS_MAPPING.items():
            for keyword in keywords:
                if keyword in combined:
                    return category

        # Fallback classification based on asset class
        if asset_class in ['equity', 'equities', 'stock', 'stocks']:
            return 'equity_index'
        elif asset_class in ['forex', 'fx', 'currency']:
            return 'forex'
        elif asset_class in ['commodities', 'commodity']:
            return 'commodities'
        elif asset_class in ['fixed_income', 'bonds', 'bond']:
            return 'government'  # Default to safer category

        return 'other'

    def _compute_asset_class_performance(
        self,
        algo_returns: pd.DataFrame,
        algo_to_category: Dict[str, str],
    ) -> AssetClassPerformance:
        """Compute rolling performance by asset class category."""
        # Group algorithms by category
        categories = list(set(algo_to_category.values()))
        categories = [c for c in categories if c != 'other']  # Exclude 'other'

        returns_by_class = pd.DataFrame(index=algo_returns.index, columns=categories)
        vol_by_class = pd.DataFrame(index=algo_returns.index, columns=categories)
        count_by_class = pd.DataFrame(index=algo_returns.index, columns=categories)

        for category in categories:
            # Get algos in this category
            category_algos = [a for a, c in algo_to_category.items()
                             if c == category and a in algo_returns.columns]

            if len(category_algos) == 0:
                continue

            # Equal-weighted returns for the category
            category_returns = algo_returns[category_algos].mean(axis=1, skipna=True)
            returns_by_class[category] = category_returns

            # Rolling volatility
            vol_by_class[category] = category_returns.rolling(
                self.vol_window, min_periods=self.vol_window // 2
            ).std() * np.sqrt(252)

            # Count of active algorithms
            count_by_class[category] = algo_returns[category_algos].notna().sum(axis=1)

        return AssetClassPerformance(
            returns=returns_by_class.astype(float),
            volatility=vol_by_class.astype(float),
            n_algorithms=count_by_class.astype(float),
        )

    def _build_regime_indicators(
        self,
        performance: AssetClassPerformance,
    ) -> RegimeIndicators:
        """Build Investment Clock regime indicators from asset class performance."""
        returns = performance.returns
        window = self.rolling_window

        # 1. Cyclical vs Defensive spread
        cyclical_ret = returns.get('cyclical', pd.Series(index=returns.index))
        defensive_ret = returns.get('defensive', pd.Series(index=returns.index))

        if cyclical_ret is not None and defensive_ret is not None:
            cyclical_roll = cyclical_ret.rolling(window, min_periods=window//2).mean()
            defensive_roll = defensive_ret.rolling(window, min_periods=window//2).mean()
            cyclical_vs_defensive = cyclical_roll - defensive_roll
        else:
            cyclical_vs_defensive = pd.Series(0, index=returns.index)

        # 2. Credit spread (HY vs Government) - risk appetite
        hy_ret = returns.get('high_yield', pd.Series(index=returns.index))
        gov_ret = returns.get('government', pd.Series(index=returns.index))

        if hy_ret is not None and gov_ret is not None:
            hy_roll = hy_ret.rolling(window, min_periods=window//2).mean()
            gov_roll = gov_ret.rolling(window, min_periods=window//2).mean()
            credit_spread = hy_roll - gov_roll
        else:
            credit_spread = pd.Series(0, index=returns.index)

        # 3. Equity vs Bonds
        equity_ret = returns.get('equity_index', pd.Series(index=returns.index))
        bond_ret = gov_ret if gov_ret is not None else pd.Series(index=returns.index)

        if equity_ret is not None and bond_ret is not None:
            equity_roll = equity_ret.rolling(window, min_periods=window//2).mean()
            bond_roll = bond_ret.rolling(window, min_periods=window//2).mean()
            equity_vs_bonds = equity_roll - bond_roll
        else:
            equity_vs_bonds = pd.Series(0, index=returns.index)

        # 4. Volatility level (cross-sectional)
        volatility_level = performance.volatility.mean(axis=1, skipna=True)

        # 5. Correlation level (rolling cross-sectional)
        correlation_level = self._compute_rolling_cross_correlation(returns, window)

        # 6. Momentum breadth (% of asset classes with positive rolling return)
        rolling_returns = returns.rolling(window, min_periods=window//2).mean()
        momentum_breadth = (rolling_returns > 0).sum(axis=1) / rolling_returns.notna().sum(axis=1)

        return RegimeIndicators(
            cyclical_vs_defensive=cyclical_vs_defensive,
            credit_spread=credit_spread,
            equity_vs_bonds=equity_vs_bonds,
            volatility_level=volatility_level,
            correlation_level=correlation_level,
            momentum_breadth=momentum_breadth,
        )

    def _compute_rolling_cross_correlation(
        self,
        returns: pd.DataFrame,
        window: int,
    ) -> pd.Series:
        """Compute rolling mean correlation across asset classes."""
        result = pd.Series(index=returns.index, dtype=float)

        for i in range(window, len(returns)):
            window_data = returns.iloc[i-window:i].dropna(axis=1, how='all')
            if window_data.shape[1] >= 2:
                corr_matrix = window_data.corr()
                # Mean of upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                mean_corr = corr_matrix.values[mask].mean()
                result.iloc[i] = mean_corr

        return result.fillna(0)

    def _infer_regime_hmm(
        self,
        indicators: RegimeIndicators,
        performance: AssetClassPerformance,
    ) -> InvestmentClockResult:
        """Infer regime using HMM on regime indicators."""
        from hmmlearn import hmm

        # Build observation matrix
        obs = pd.DataFrame(index=indicators.cyclical_vs_defensive.index)
        obs['cyclical_vs_defensive'] = indicators.cyclical_vs_defensive
        obs['credit_spread'] = indicators.credit_spread
        obs['equity_vs_bonds'] = indicators.equity_vs_bonds
        obs['volatility'] = indicators.volatility_level
        obs['correlation'] = indicators.correlation_level
        obs['momentum_breadth'] = indicators.momentum_breadth

        # Clean and scale
        obs = obs.replace([np.inf, -np.inf], np.nan)
        obs = obs.ffill().bfill().fillna(0)
        obs_scaled = self._scaler.fit_transform(obs)

        # Fit HMM with 4 states (4 business cycle phases)
        n_states = 4
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=200,
            random_state=self.random_state,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(obs_scaled)

        # Predict regimes
        labels = model.predict(obs_scaled)
        probabilities = model.predict_proba(obs_scaled)

        # Map HMM states to business cycle phases based on characteristics
        phase_mapping = self._map_hmm_states_to_phases(
            labels, obs, indicators
        )

        # Apply mapping
        mapped_labels = pd.Series(
            [phase_mapping.get(l, l) for l in labels],
            index=obs.index,
            name='regime'
        )

        prob_df = pd.DataFrame(
            probabilities,
            index=obs.index,
            columns=[f'phase_{i}' for i in range(n_states)]
        )

        # Compute phase statistics
        phase_stats = self._compute_phase_statistics(
            mapped_labels, indicators, performance
        )

        return InvestmentClockResult(
            regime_labels=mapped_labels,
            regime_probabilities=prob_df,
            indicators=indicators,
            phase_statistics=phase_stats,
            transition_matrix=model.transmat_,
        )

    def _map_hmm_states_to_phases(
        self,
        labels: np.ndarray,
        obs: pd.DataFrame,
        indicators: RegimeIndicators,
    ) -> Dict[int, int]:
        """
        Map HMM states to business cycle phases based on characteristics.

        Investment Clock characteristics:
        - Recovery: High cyclical_vs_defensive, high credit_spread, moderate vol
        - Expansion: High equity_vs_bonds, high momentum_breadth, low vol
        - Slowdown: Negative cyclical_vs_defensive, negative credit_spread
        - Contraction: Low equity_vs_bonds, low breadth, high vol, high correlation
        """
        state_characteristics = {}

        for state in range(4):
            mask = labels == state
            if mask.sum() == 0:
                continue

            state_characteristics[state] = {
                'cyclical_vs_defensive': obs.loc[mask, 'cyclical_vs_defensive'].mean(),
                'credit_spread': obs.loc[mask, 'credit_spread'].mean(),
                'equity_vs_bonds': obs.loc[mask, 'equity_vs_bonds'].mean(),
                'volatility': obs.loc[mask, 'volatility'].mean(),
                'correlation': obs.loc[mask, 'correlation'].mean(),
                'momentum_breadth': obs.loc[mask, 'momentum_breadth'].mean(),
            }

        # Score each state for each phase
        phase_scores = {state: {} for state in range(4)}

        for state, chars in state_characteristics.items():
            # Recovery: cyclicals winning, credit winning, moderate vol
            phase_scores[state][BusinessCyclePhase.RECOVERY.value] = (
                chars.get('cyclical_vs_defensive', 0) * 2 +
                chars.get('credit_spread', 0) * 2 +
                -abs(chars.get('volatility', 0)) * 0.5
            )

            # Expansion: equity winning, high breadth, low vol
            phase_scores[state][BusinessCyclePhase.EXPANSION.value] = (
                chars.get('equity_vs_bonds', 0) * 2 +
                chars.get('momentum_breadth', 0) * 2 +
                -chars.get('volatility', 0) * 1
            )

            # Slowdown: defensives winning, credit losing, rising vol
            phase_scores[state][BusinessCyclePhase.SLOWDOWN.value] = (
                -chars.get('cyclical_vs_defensive', 0) * 2 +
                -chars.get('credit_spread', 0) * 1 +
                chars.get('volatility', 0) * 0.5
            )

            # Contraction: equity losing, high vol, high correlation
            phase_scores[state][BusinessCyclePhase.CONTRACTION.value] = (
                -chars.get('equity_vs_bonds', 0) * 2 +
                chars.get('volatility', 0) * 2 +
                chars.get('correlation', 0) * 1 +
                -chars.get('momentum_breadth', 0) * 1
            )

        # Greedy assignment: each state gets its best-scoring phase
        mapping = {}
        assigned_phases = set()

        for _ in range(4):
            best_score = -np.inf
            best_state = None
            best_phase = None

            for state in range(4):
                if state in mapping:
                    continue
                for phase in range(4):
                    if phase in assigned_phases:
                        continue
                    score = phase_scores.get(state, {}).get(phase, -np.inf)
                    if score > best_score:
                        best_score = score
                        best_state = state
                        best_phase = phase

            if best_state is not None and best_phase is not None:
                mapping[best_state] = best_phase
                assigned_phases.add(best_phase)

        return mapping

    def _compute_phase_statistics(
        self,
        labels: pd.Series,
        indicators: RegimeIndicators,
        performance: AssetClassPerformance,
    ) -> Dict[int, Dict]:
        """Compute statistics for each business cycle phase."""
        stats = {}

        phase_names = {
            0: 'recovery',
            1: 'expansion',
            2: 'slowdown',
            3: 'contraction',
        }

        for phase in labels.unique():
            mask = labels == phase
            if mask.sum() == 0:
                continue

            stats[int(phase)] = {
                'name': phase_names.get(phase, f'phase_{phase}'),
                'n_days': int(mask.sum()),
                'pct_time': float(mask.sum() / len(labels) * 100),
                'avg_cyclical_vs_defensive': float(indicators.cyclical_vs_defensive.loc[mask].mean()),
                'avg_credit_spread': float(indicators.credit_spread.loc[mask].mean()),
                'avg_volatility': float(indicators.volatility_level.loc[mask].mean()),
                'avg_correlation': float(indicators.correlation_level.loc[mask].mean()),
                'avg_momentum_breadth': float(indicators.momentum_breadth.loc[mask].mean()),
            }

            # Asset class performance in this phase
            for col in performance.returns.columns:
                phase_returns = performance.returns.loc[mask, col].dropna()
                if len(phase_returns) > 0:
                    ann_ret = phase_returns.mean() * 252
                    stats[int(phase)][f'{col}_ann_return'] = float(ann_ret)

        return stats

    def generate_report(self, result: InvestmentClockResult) -> str:
        """Generate text report for Investment Clock regime inference."""
        report = """
INVESTMENT CLOCK REGIME INFERENCE REPORT
═══════════════════════════════════════════════════════════════════════════════

METHODOLOGY
───────────────────────────────────────────────────────────────────────────────
This analysis infers the business cycle regime from ASSET CLASS PERFORMANCE,
NOT from benchmark allocation decisions. This avoids reverse causality:

  Asset Performance → Regime State → Benchmark Selection (consequence)

Regime indicators are computed using rolling windows to avoid lookahead bias.

REGIME INDICATORS USED
───────────────────────────────────────────────────────────────────────────────
1. Cyclical vs Defensive spread (Tech/Industrials vs Utilities/Staples)
2. Credit spread (High Yield vs Government bonds)
3. Equity vs Bonds spread
4. Cross-sectional volatility level
5. Cross-sectional correlation level
6. Momentum breadth (% of asset classes with positive momentum)

BUSINESS CYCLE PHASES DETECTED
───────────────────────────────────────────────────────────────────────────────
"""
        phase_names = {
            0: 'RECOVERY',
            1: 'EXPANSION',
            2: 'SLOWDOWN',
            3: 'CONTRACTION',
        }

        phase_descriptions = {
            0: 'Below-trend but rebounding. Cyclicals lead, credit outperforms.',
            1: 'Healthy growth. Broad equity outperformance, low volatility.',
            2: 'Peak growth, decelerating. Defensives start leading.',
            3: 'Declining activity. Safe havens dominate, high volatility.',
        }

        for phase, stats in sorted(result.phase_statistics.items()):
            name = phase_names.get(phase, f'Phase {phase}')
            desc = phase_descriptions.get(phase, '')
            report += f"\n{phase}. {name}\n"
            report += f"   {desc}\n"
            report += f"   Days: {stats['n_days']} ({stats['pct_time']:.1f}% of time)\n"
            report += f"   Cyclical vs Defensive: {stats['avg_cyclical_vs_defensive']:.4f}\n"
            report += f"   Credit Spread: {stats['avg_credit_spread']:.4f}\n"
            report += f"   Volatility: {stats['avg_volatility']:.2%}\n"
            report += f"   Correlation: {stats['avg_correlation']:.2f}\n"
            report += f"   Momentum Breadth: {stats['avg_momentum_breadth']:.1%}\n"

        report += """
TRANSITION MATRIX
───────────────────────────────────────────────────────────────────────────────
Probability of transitioning from row phase to column phase:
"""
        phases = list(phase_names.values())
        header = "         " + "  ".join(f"{p[:8]:>8}" for p in phases)
        report += header + "\n"

        for i, from_phase in enumerate(phases):
            row = f"{from_phase[:8]:>8} "
            for j in range(4):
                prob = result.transition_matrix[i, j] if i < len(result.transition_matrix) else 0
                row += f"  {prob:>7.1%}"
            report += row + "\n"

        return report
