"""
Análisis de correlaciones entre algoritmos.

Optimized version with:
- Numba JIT-compiled functions for rolling correlations
- Vectorized operations using numpy
- Memory-efficient processing
"""

import gc
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

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
# Numba JIT-compiled functions
# =============================================================================

@njit(cache=True)
def _rolling_correlation_single_pair(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling correlation between two arrays."""
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    for i in range(window - 1, n):
        start = i - window + 1

        # Compute means
        mean_x = 0.0
        mean_y = 0.0
        valid_count = 0

        for j in range(start, i + 1):
            if not (np.isnan(x[j]) or np.isnan(y[j])):
                mean_x += x[j]
                mean_y += y[j]
                valid_count += 1

        if valid_count < 5:
            continue

        mean_x /= valid_count
        mean_y /= valid_count

        # Compute correlation
        cov = 0.0
        var_x = 0.0
        var_y = 0.0

        for j in range(start, i + 1):
            if not (np.isnan(x[j]) or np.isnan(y[j])):
                dx = x[j] - mean_x
                dy = y[j] - mean_y
                cov += dx * dy
                var_x += dx * dx
                var_y += dy * dy

        if var_x > 1e-10 and var_y > 1e-10:
            result[i] = cov / np.sqrt(var_x * var_y)

    return result


@njit(cache=True)
def _compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute correlation matrix for 2D array [n_samples, n_features]."""
    n_samples, n_features = data.shape
    corr = np.empty((n_features, n_features), dtype=np.float64)

    # Compute means
    means = np.empty(n_features, dtype=np.float64)
    for j in range(n_features):
        total = 0.0
        count = 0
        for i in range(n_samples):
            if not np.isnan(data[i, j]):
                total += data[i, j]
                count += 1
        means[j] = total / count if count > 0 else 0.0

    # Compute correlation matrix
    for j1 in range(n_features):
        corr[j1, j1] = 1.0
        for j2 in range(j1 + 1, n_features):
            cov = 0.0
            var1 = 0.0
            var2 = 0.0
            count = 0

            for i in range(n_samples):
                if not (np.isnan(data[i, j1]) or np.isnan(data[i, j2])):
                    d1 = data[i, j1] - means[j1]
                    d2 = data[i, j2] - means[j2]
                    cov += d1 * d2
                    var1 += d1 * d1
                    var2 += d2 * d2
                    count += 1

            if count > 2 and var1 > 1e-10 and var2 > 1e-10:
                c = cov / np.sqrt(var1 * var2)
            else:
                c = 0.0

            corr[j1, j2] = c
            corr[j2, j1] = c

    return corr


@njit(cache=True)
def _diversification_ratio_numba(
    returns: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute diversification ratio.

    DR = sum(w_i * vol_i) / portfolio_vol
    """
    n_samples, n_assets = returns.shape

    if n_samples < 2 or n_assets < 2:
        return 1.0

    # Compute individual volatilities
    vols = np.empty(n_assets, dtype=np.float64)
    for j in range(n_assets):
        total = 0.0
        count = 0
        for i in range(n_samples):
            if not np.isnan(returns[i, j]):
                total += returns[i, j]
                count += 1
        mean = total / count if count > 0 else 0.0

        var = 0.0
        for i in range(n_samples):
            if not np.isnan(returns[i, j]):
                var += (returns[i, j] - mean) ** 2
        vols[j] = np.sqrt(var / (count - 1)) if count > 1 else 0.0

    # Weighted volatility sum
    weighted_vol_sum = 0.0
    for j in range(n_assets):
        weighted_vol_sum += weights[j] * vols[j]

    # Portfolio variance
    # First compute covariance matrix
    means = np.empty(n_assets, dtype=np.float64)
    for j in range(n_assets):
        total = 0.0
        count = 0
        for i in range(n_samples):
            if not np.isnan(returns[i, j]):
                total += returns[i, j]
                count += 1
        means[j] = total / count if count > 0 else 0.0

    portfolio_var = 0.0
    for j1 in range(n_assets):
        for j2 in range(n_assets):
            cov = 0.0
            count = 0
            for i in range(n_samples):
                if not (np.isnan(returns[i, j1]) or np.isnan(returns[i, j2])):
                    cov += (returns[i, j1] - means[j1]) * (returns[i, j2] - means[j2])
                    count += 1
            if count > 1:
                cov /= (count - 1)
            portfolio_var += weights[j1] * weights[j2] * cov

    portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else 0.0

    if portfolio_vol < 1e-10:
        return 1.0

    return weighted_vol_sum / portfolio_vol


@njit(cache=True)
def _rolling_mean_correlation_numba(
    returns: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling mean correlation across all pairs."""
    n_samples, n_assets = returns.shape
    result = np.empty(n_samples, dtype=np.float64)
    result[:] = np.nan

    if n_assets < 2:
        return result

    for t in range(window - 1, n_samples):
        start = t - window + 1

        # Compute mean correlation for this window
        total_corr = 0.0
        n_pairs = 0

        for j1 in range(n_assets):
            for j2 in range(j1 + 1, n_assets):
                # Compute correlation between j1 and j2 in window
                mean1 = 0.0
                mean2 = 0.0
                count = 0

                for i in range(start, t + 1):
                    if not (np.isnan(returns[i, j1]) or np.isnan(returns[i, j2])):
                        mean1 += returns[i, j1]
                        mean2 += returns[i, j2]
                        count += 1

                if count < 5:
                    continue

                mean1 /= count
                mean2 /= count

                cov = 0.0
                var1 = 0.0
                var2 = 0.0

                for i in range(start, t + 1):
                    if not (np.isnan(returns[i, j1]) or np.isnan(returns[i, j2])):
                        d1 = returns[i, j1] - mean1
                        d2 = returns[i, j2] - mean2
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


# =============================================================================
# CorrelationAnalyzer class
# =============================================================================

class CorrelationAnalyzer:
    """
    Analiza correlaciones entre algoritmos y su estabilidad.

    Optimized with numba JIT compilation for:
    - Rolling correlations
    - Diversification ratio
    - Mean correlation computation
    """

    def __init__(self, default_window: int = 63):
        self.default_window = default_window

    def correlation_matrix(
        self,
        returns_matrix: pd.DataFrame,
        method: str = "pearson",
        use_numba: bool = True,
    ) -> pd.DataFrame:
        """
        Calcula matriz de correlaciones.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            method: "pearson", "spearman" o "kendall".
            use_numba: Use numba implementation (faster for large matrices)

        Returns:
            Matriz de correlaciones.
        """
        if use_numba and method == "pearson" and HAS_NUMBA:
            data = returns_matrix.values.astype(np.float64)
            corr_arr = _compute_correlation_matrix(data)
            return pd.DataFrame(
                corr_arr,
                index=returns_matrix.columns,
                columns=returns_matrix.columns,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return returns_matrix.corr(method=method)

    def rolling_correlation(
        self,
        returns_matrix: pd.DataFrame,
        window: Optional[int] = None,
        max_pairs: int = 500,
    ) -> dict[tuple[str, str], pd.Series]:
        """
        Calcula correlaciones rolling entre pares de algoritmos.

        Uses numba for efficient computation.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            window: Ventana rolling.
            max_pairs: Maximum number of pairs to compute (for memory efficiency)

        Returns:
            Dict {(algo1, algo2): serie_correlacion}.
        """
        window = window or self.default_window
        columns = list(returns_matrix.columns)
        n_cols = len(columns)

        # Limit number of pairs for memory efficiency
        n_pairs = n_cols * (n_cols - 1) // 2
        if n_pairs > max_pairs:
            logger.warning(
                f"Too many pairs ({n_pairs}), sampling {max_pairs} pairs"
            )
            # Sample columns to reduce pairs
            sample_size = int(np.sqrt(max_pairs * 2)) + 1
            if sample_size < n_cols:
                np.random.seed(42)
                columns = list(np.random.choice(columns, sample_size, replace=False))

        rolling_corrs = {}

        # Convert to numpy for numba
        data = returns_matrix[columns].values.astype(np.float64)

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i + 1:], start=i + 1):
                x = data[:, i]
                y = data[:, j]

                if HAS_NUMBA:
                    corr_arr = _rolling_correlation_single_pair(x, y, window)
                else:
                    corr_arr = returns_matrix[col1].rolling(window).corr(
                        returns_matrix[col2]
                    ).values

                rolling_corrs[(col1, col2)] = pd.Series(
                    corr_arr, index=returns_matrix.index
                )

        return rolling_corrs

    def rolling_mean_correlation(
        self,
        returns_matrix: pd.DataFrame,
        window: Optional[int] = None,
        max_assets: int = 50,
    ) -> pd.Series:
        """
        Compute rolling mean correlation across all algorithm pairs.

        Uses numba for efficient computation.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            window: Rolling window size.
            max_assets: Maximum number of assets to include (for efficiency)

        Returns:
            Series with mean correlation over time.
        """
        window = window or self.default_window

        # Limit number of assets for efficiency
        if len(returns_matrix.columns) > max_assets:
            # Select assets with most data
            valid_counts = returns_matrix.notna().sum()
            top_assets = valid_counts.nlargest(max_assets).index
            returns_subset = returns_matrix[top_assets]
        else:
            returns_subset = returns_matrix

        data = returns_subset.values.astype(np.float64)

        if HAS_NUMBA:
            mean_corr = _rolling_mean_correlation_numba(data, window)
        else:
            # Fallback to pandas (slower)
            rolling_corrs = returns_subset.rolling(window).corr()
            mean_corr = []
            for t in range(len(returns_subset)):
                if t < window - 1:
                    mean_corr.append(np.nan)
                else:
                    corr_matrix = rolling_corrs.iloc[t * len(returns_subset.columns):(t + 1) * len(returns_subset.columns)]
                    if hasattr(corr_matrix, 'values'):
                        upper_tri = np.triu(corr_matrix.values, k=1)
                        mean_corr.append(upper_tri[upper_tri != 0].mean())
                    else:
                        mean_corr.append(np.nan)
            mean_corr = np.array(mean_corr)

        return pd.Series(mean_corr, index=returns_matrix.index, name='mean_correlation')

    def correlation_stability(
        self,
        returns_matrix: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        window: Optional[int] = None,
        max_pairs: int = 200,
    ) -> pd.DataFrame:
        """
        Analiza estabilidad de correlaciones.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            regime_labels: Serie con etiquetas de régimen (opcional).
            window: Ventana rolling.
            max_pairs: Maximum pairs to analyze.

        Returns:
            DataFrame con métricas de estabilidad por par.
        """
        window = window or self.default_window
        rolling_corrs = self.rolling_correlation(returns_matrix, window, max_pairs)

        stability_metrics = []
        for (algo1, algo2), corr_series in rolling_corrs.items():
            corr_clean = corr_series.dropna()

            if len(corr_clean) < 10:
                continue

            metrics = {
                "algo1": algo1,
                "algo2": algo2,
                "mean_corr": corr_clean.mean(),
                "std_corr": corr_clean.std(),
                "min_corr": corr_clean.min(),
                "max_corr": corr_clean.max(),
                "range_corr": corr_clean.max() - corr_clean.min(),
            }

            # Estabilidad por régimen
            if regime_labels is not None:
                aligned = pd.concat([corr_series, regime_labels], axis=1, join="inner")
                aligned.columns = ["corr", "regime"]

                for regime in aligned["regime"].unique():
                    regime_corr = aligned[aligned["regime"] == regime]["corr"]
                    if len(regime_corr) > 5:
                        metrics[f"corr_{regime}"] = regime_corr.mean()

            stability_metrics.append(metrics)

        return pd.DataFrame(stability_metrics)

    def diversification_ratio(
        self,
        returns_matrix: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calcula ratio de diversificación.

        DR = (sum of weighted vols) / portfolio vol
        DR > 1 indica beneficio de diversificación.

        Uses numba for efficient computation.
        """
        n_assets = len(returns_matrix.columns)

        if weights is None:
            weights = np.ones(n_assets, dtype=np.float64) / n_assets
        else:
            weights = weights.astype(np.float64)

        data = returns_matrix.values.astype(np.float64)

        if HAS_NUMBA:
            return _diversification_ratio_numba(data, weights)
        else:
            # Fallback to pandas/numpy
            vols = returns_matrix.std().values
            weighted_vol_sum = np.dot(weights, vols)

            cov = returns_matrix.cov().values
            portfolio_var = np.dot(weights, np.dot(cov, weights))
            portfolio_vol = np.sqrt(portfolio_var)

            if portfolio_vol == 0:
                return 1.0

            return weighted_vol_sum / portfolio_vol

    def rolling_diversification_ratio(
        self,
        returns_matrix: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        window: Optional[int] = None,
    ) -> pd.Series:
        """
        Calcula ratio de diversificación rolling.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            weights: Vector de pesos (default: equal weight).
            window: Ventana rolling.

        Returns:
            Serie con DR por fecha.
        """
        window = window or self.default_window
        n_assets = len(returns_matrix.columns)

        if weights is None:
            weights = np.ones(n_assets, dtype=np.float64) / n_assets
        else:
            weights = weights.astype(np.float64)

        result = []
        data = returns_matrix.values.astype(np.float64)

        for t in range(len(returns_matrix)):
            if t < window - 1:
                result.append(np.nan)
            else:
                window_data = data[t - window + 1:t + 1, :]
                if HAS_NUMBA:
                    dr = _diversification_ratio_numba(window_data, weights)
                else:
                    dr = self._compute_dr_window(window_data, weights)
                result.append(dr)

        return pd.Series(result, index=returns_matrix.index, name="diversification_ratio")

    def _compute_dr_window(self, window_data: np.ndarray, weights: np.ndarray) -> float:
        """Compute DR for a single window (fallback when no numba)."""
        n_samples, n_assets = window_data.shape
        if n_samples < 2:
            return 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vols = np.nanstd(window_data, axis=0)
            weighted_vol_sum = np.dot(weights, vols)

            cov = np.nancov(window_data.T)
            if np.isscalar(cov):
                return 1.0

            portfolio_var = np.dot(weights, np.dot(cov, weights))
            portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else 0

            if portfolio_vol < 1e-10:
                return 1.0

            return weighted_vol_sum / portfolio_vol

    def correlation_with_benchmark(
        self,
        returns_matrix: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> pd.Series:
        """
        Calcula correlación de cada algoritmo con el benchmark (vectorized).
        """
        combined = returns_matrix.join(benchmark_returns.rename('_benchmark_'), how='inner')
        if len(combined) < 10:
            return pd.Series(0.0, index=returns_matrix.columns, name="corr_with_benchmark")

        benchmark_col = combined['_benchmark_']
        algo_cols = combined.drop(columns=['_benchmark_'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlations = algo_cols.corrwith(benchmark_col)

        correlations.name = "corr_with_benchmark"
        return correlations.fillna(0)

    def cluster_by_correlation(
        self,
        returns_matrix: pd.DataFrame,
        n_clusters: int = 3,
    ) -> dict[str, int]:
        """
        Agrupa algoritmos por similitud de correlación.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            n_clusters: Número de clusters.

        Returns:
            Dict {algo_id: cluster_id}.
        """
        from sklearn.cluster import AgglomerativeClustering

        corr_matrix = self.correlation_matrix(returns_matrix)

        # Convertir correlación a distancia
        distance_matrix = 1 - corr_matrix.abs()

        # Ensure symmetric and no NaN
        distance_matrix = distance_matrix.fillna(1.0)

        # Clustering jerárquico
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        return dict(zip(returns_matrix.columns, labels))

    def get_low_correlation_pairs(
        self,
        returns_matrix: pd.DataFrame,
        threshold: float = 0.3,
    ) -> list[tuple[str, str, float]]:
        """
        Encuentra pares de algoritmos con baja correlación.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            threshold: Umbral máximo de correlación.

        Returns:
            Lista de (algo1, algo2, correlacion) ordenada.
        """
        corr_matrix = self.correlation_matrix(returns_matrix)
        columns = corr_matrix.columns

        pairs = []
        for i, col1 in enumerate(columns):
            for col2 in columns[i + 1:]:
                corr = corr_matrix.loc[col1, col2]
                if not np.isnan(corr) and abs(corr) < threshold:
                    pairs.append((col1, col2, corr))

        # Ordenar por correlación absoluta (menor primero)
        pairs.sort(key=lambda x: abs(x[2]))

        return pairs

    def generate_correlation_report(
        self,
        returns_matrix: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        regime_labels: Optional[pd.Series] = None,
        max_display: int = 20,
    ) -> str:
        """
        Genera informe de correlaciones.

        Args:
            returns_matrix: DataFrame [fecha x algo_id] con retornos.
            benchmark_returns: Serie de retornos del benchmark (opcional).
            regime_labels: Serie con etiquetas de régimen (opcional).
            max_display: Maximum items to display in report.

        Returns:
            String con informe formateado.
        """
        # Limit to top assets for display
        if len(returns_matrix.columns) > max_display:
            valid_counts = returns_matrix.notna().sum()
            top_assets = valid_counts.nlargest(max_display).index
            returns_subset = returns_matrix[top_assets]
        else:
            returns_subset = returns_matrix

        corr_matrix = self.correlation_matrix(returns_subset)
        dr = self.diversification_ratio(returns_subset)
        low_corr_pairs = self.get_low_correlation_pairs(returns_subset)

        report = f"""
CORRELATION ANALYSIS
===============================================================

DIVERSIFICATION
---------------------------------------------------------------
Diversification Ratio (EW): {dr:.3f}
  (>1 indicates diversification benefit)

CORRELATION MATRIX (top {len(returns_subset.columns)} assets)
---------------------------------------------------------------
{corr_matrix.round(3).to_string()}

"""

        # Pares de baja correlación
        if low_corr_pairs:
            report += "LOW CORRELATION PAIRS (<0.3)\n"
            report += "---------------------------------------------------------------\n"
            for algo1, algo2, corr in low_corr_pairs[:10]:
                report += f"  {algo1} vs {algo2}: {corr:.3f}\n"
            report += "\n"

        # Correlación con benchmark
        if benchmark_returns is not None:
            corr_benchmark = self.correlation_with_benchmark(
                returns_subset, benchmark_returns
            )
            report += "CORRELATION WITH BENCHMARK\n"
            report += "---------------------------------------------------------------\n"
            sorted_corr = corr_benchmark.sort_values(ascending=False)
            for algo_id in sorted_corr.head(10).index:
                report += f"  {algo_id}: {sorted_corr[algo_id]:.3f}\n"
            report += "  ...\n"
            for algo_id in sorted_corr.tail(5).index:
                report += f"  {algo_id}: {sorted_corr[algo_id]:.3f}\n"
            report += "\n"

        # Estabilidad por régimen
        if regime_labels is not None:
            stability = self.correlation_stability(
                returns_subset, regime_labels, max_pairs=50
            )
            regime_cols = [c for c in stability.columns if c.startswith("corr_")]
            if regime_cols:
                report += "CORRELATION BY REGIME (sample)\n"
                report += "---------------------------------------------------------------\n"
                display_df = stability[["algo1", "algo2"] + regime_cols].head(10)
                report += display_df.to_string(index=False)
                report += "\n"

        return report
