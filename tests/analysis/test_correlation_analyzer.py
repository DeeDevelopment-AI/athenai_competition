"""
Tests for src/analysis/correlation_analyzer.py

Tests the numba-optimized correlation analysis functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.correlation_analyzer import (
    CorrelationAnalyzer,
    _rolling_correlation_single_pair,
    _compute_correlation_matrix,
    _diversification_ratio_numba,
    _rolling_mean_correlation_numba,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns_matrix():
    """Generate sample returns matrix for multiple algorithms."""
    np.random.seed(42)
    n_days = 252
    n_algos = 5

    # Create returns with some correlation structure
    base = np.random.normal(0, 0.01, n_days)
    returns = np.zeros((n_days, n_algos), dtype=np.float64)

    for i in range(n_algos):
        # Each algo has some exposure to base factor plus idiosyncratic
        beta = 0.3 + 0.1 * i
        idio = np.random.normal(0, 0.01, n_days)
        returns[:, i] = beta * base + idio

    index = pd.date_range('2023-01-01', periods=n_days, freq='B')
    columns = [f'algo_{i}' for i in range(n_algos)]

    return pd.DataFrame(returns, index=index, columns=columns)


@pytest.fixture
def uncorrelated_returns():
    """Generate uncorrelated returns matrix."""
    np.random.seed(42)
    n_days = 252
    n_algos = 4

    returns = np.random.normal(0, 0.01, (n_days, n_algos)).astype(np.float64)

    index = pd.date_range('2023-01-01', periods=n_days, freq='B')
    columns = [f'algo_{i}' for i in range(n_algos)]

    return pd.DataFrame(returns, index=index, columns=columns)


@pytest.fixture
def perfectly_correlated_returns():
    """Generate perfectly correlated returns."""
    np.random.seed(42)
    n_days = 252
    n_algos = 3

    base = np.random.normal(0.001, 0.01, n_days)
    returns = np.zeros((n_days, n_algos), dtype=np.float64)

    for i in range(n_algos):
        returns[:, i] = base * (1 + 0.1 * i)  # Same direction, different scale

    index = pd.date_range('2023-01-01', periods=n_days, freq='B')
    columns = [f'algo_{i}' for i in range(n_algos)]

    return pd.DataFrame(returns, index=index, columns=columns)


@pytest.fixture
def benchmark_returns(sample_returns_matrix):
    """Generate benchmark returns correlated with algos."""
    # Simple equal-weight portfolio
    return sample_returns_matrix.mean(axis=1)


# =============================================================================
# Numba Function Tests
# =============================================================================

class TestRollingCorrelationSinglePair:
    """Tests for _rolling_correlation_single_pair."""

    def test_rolling_correlation_basic(self):
        """Test basic rolling correlation computation."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n).astype(np.float64)
        y = 0.7 * x + 0.3 * np.random.normal(0, 1, n)
        y = y.astype(np.float64)

        result = _rolling_correlation_single_pair(x, y, 21)

        # First 20 values should be NaN
        assert np.all(np.isnan(result[:20]))
        # Rest should be valid correlations
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid >= -1) and np.all(valid <= 1)
        # Should be positive correlation
        assert np.mean(valid) > 0.3

    def test_rolling_correlation_identical_series(self):
        """Test with identical series (perfect correlation)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = x.copy()

        result = _rolling_correlation_single_pair(x, y, 5)

        # Perfect correlation should be 1
        valid = result[~np.isnan(result)]
        assert np.allclose(valid, 1.0)

    def test_rolling_correlation_opposite_series(self):
        """Test with perfectly negatively correlated series."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = -x

        result = _rolling_correlation_single_pair(x, y, 5)

        # Perfect negative correlation
        valid = result[~np.isnan(result)]
        assert np.allclose(valid, -1.0)


class TestComputeCorrelationMatrix:
    """Tests for _compute_correlation_matrix."""

    def test_correlation_matrix_shape(self):
        """Test output shape."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 5)).astype(np.float64)

        result = _compute_correlation_matrix(data)

        assert result.shape == (5, 5)

    def test_correlation_matrix_diagonal(self):
        """Test diagonal is all 1s."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 4)).astype(np.float64)

        result = _compute_correlation_matrix(data)

        np.testing.assert_array_almost_equal(np.diag(result), np.ones(4))

    def test_correlation_matrix_symmetric(self):
        """Test matrix is symmetric."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 4)).astype(np.float64)

        result = _compute_correlation_matrix(data)

        np.testing.assert_array_almost_equal(result, result.T)

    def test_correlation_matrix_bounds(self):
        """Test all values are in [-1, 1]."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 5)).astype(np.float64)

        result = _compute_correlation_matrix(data)

        assert np.all(result >= -1)
        assert np.all(result <= 1)


class TestDiversificationRatioNumba:
    """Tests for _diversification_ratio_numba."""

    def test_dr_single_asset(self):
        """Test DR with single asset is 1."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, (100, 1)).astype(np.float64)
        weights = np.array([1.0], dtype=np.float64)

        result = _diversification_ratio_numba(returns, weights)

        assert abs(result - 1.0) < 0.01

    def test_dr_perfectly_correlated(self):
        """Test DR with perfectly correlated assets is ~1."""
        np.random.seed(42)
        base = np.random.normal(0, 0.01, 100)
        returns = np.column_stack([base, base, base]).astype(np.float64)
        weights = np.array([1/3, 1/3, 1/3], dtype=np.float64)

        result = _diversification_ratio_numba(returns, weights)

        # Perfect correlation means no diversification benefit
        assert abs(result - 1.0) < 0.1

    def test_dr_uncorrelated(self):
        """Test DR with uncorrelated assets is > 1."""
        np.random.seed(42)
        n = 252
        returns = np.random.normal(0, 0.01, (n, 4)).astype(np.float64)
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        result = _diversification_ratio_numba(returns, weights)

        # Uncorrelated assets should have DR > 1
        assert result > 1.0


class TestRollingMeanCorrelationNumba:
    """Tests for _rolling_mean_correlation_numba."""

    def test_rolling_mean_corr_shape(self):
        """Test output shape."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, (100, 5)).astype(np.float64)

        result = _rolling_mean_correlation_numba(returns, 21)

        assert len(result) == 100

    def test_rolling_mean_corr_nan_start(self):
        """Test first window-1 values are NaN."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, (50, 3)).astype(np.float64)
        window = 10

        result = _rolling_mean_correlation_numba(returns, window)

        assert np.all(np.isnan(result[:window - 1]))

    def test_rolling_mean_corr_bounds(self):
        """Test values are in valid range."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, (100, 4)).astype(np.float64)

        result = _rolling_mean_correlation_numba(returns, 21)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1) and np.all(valid <= 1)


# =============================================================================
# CorrelationAnalyzer Class Tests
# =============================================================================

class TestCorrelationAnalyzerMatrix:
    """Tests for correlation matrix computation."""

    def test_correlation_matrix_basic(self, sample_returns_matrix):
        """Test basic correlation matrix computation."""
        analyzer = CorrelationAnalyzer()

        corr = analyzer.correlation_matrix(sample_returns_matrix)

        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (5, 5)
        assert list(corr.columns) == list(sample_returns_matrix.columns)

    def test_correlation_matrix_numba_vs_pandas(self, sample_returns_matrix):
        """Test numba implementation matches pandas."""
        analyzer = CorrelationAnalyzer()

        corr_numba = analyzer.correlation_matrix(sample_returns_matrix, use_numba=True)
        corr_pandas = analyzer.correlation_matrix(sample_returns_matrix, use_numba=False)

        np.testing.assert_array_almost_equal(
            corr_numba.values, corr_pandas.values, decimal=5
        )


class TestCorrelationAnalyzerRolling:
    """Tests for rolling correlation computation."""

    def test_rolling_correlation_basic(self, sample_returns_matrix):
        """Test basic rolling correlation computation."""
        analyzer = CorrelationAnalyzer(default_window=21)

        rolling_corrs = analyzer.rolling_correlation(sample_returns_matrix)

        assert isinstance(rolling_corrs, dict)
        # Number of pairs = n*(n-1)/2 = 5*4/2 = 10
        assert len(rolling_corrs) == 10

        # Check each series
        for (algo1, algo2), series in rolling_corrs.items():
            assert isinstance(series, pd.Series)
            assert len(series) == len(sample_returns_matrix)

    def test_rolling_correlation_max_pairs_limit(self, sample_returns_matrix):
        """Test max_pairs limits computation."""
        analyzer = CorrelationAnalyzer()

        rolling_corrs = analyzer.rolling_correlation(
            sample_returns_matrix, max_pairs=3
        )

        # Should be limited
        assert len(rolling_corrs) <= 3


class TestCorrelationAnalyzerMeanCorrelation:
    """Tests for rolling mean correlation."""

    def test_rolling_mean_correlation_basic(self, sample_returns_matrix):
        """Test rolling mean correlation."""
        analyzer = CorrelationAnalyzer()

        mean_corr = analyzer.rolling_mean_correlation(
            sample_returns_matrix, window=21
        )

        assert isinstance(mean_corr, pd.Series)
        assert len(mean_corr) == len(sample_returns_matrix)
        assert mean_corr.name == 'mean_correlation'

    def test_rolling_mean_correlation_uncorrelated(self, uncorrelated_returns):
        """Test mean correlation for uncorrelated assets."""
        analyzer = CorrelationAnalyzer()

        mean_corr = analyzer.rolling_mean_correlation(
            uncorrelated_returns, window=63
        )

        # Uncorrelated assets should have mean correlation near 0
        valid = mean_corr.dropna()
        assert abs(valid.mean()) < 0.2

    def test_rolling_mean_correlation_correlated(self, perfectly_correlated_returns):
        """Test mean correlation for correlated assets."""
        analyzer = CorrelationAnalyzer()

        mean_corr = analyzer.rolling_mean_correlation(
            perfectly_correlated_returns, window=21
        )

        # Highly correlated assets
        valid = mean_corr.dropna()
        assert valid.mean() > 0.9


class TestCorrelationAnalyzerDiversification:
    """Tests for diversification ratio."""

    def test_diversification_ratio_basic(self, sample_returns_matrix):
        """Test basic diversification ratio."""
        analyzer = CorrelationAnalyzer()

        dr = analyzer.diversification_ratio(sample_returns_matrix)

        assert isinstance(dr, float)
        assert dr > 0

    def test_diversification_ratio_custom_weights(self, sample_returns_matrix):
        """Test DR with custom weights."""
        analyzer = CorrelationAnalyzer()
        weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05])

        dr = analyzer.diversification_ratio(sample_returns_matrix, weights=weights)

        assert isinstance(dr, float)
        assert dr > 0

    def test_rolling_diversification_ratio(self, sample_returns_matrix):
        """Test rolling diversification ratio."""
        analyzer = CorrelationAnalyzer()

        rolling_dr = analyzer.rolling_diversification_ratio(
            sample_returns_matrix, window=63
        )

        assert isinstance(rolling_dr, pd.Series)
        assert len(rolling_dr) == len(sample_returns_matrix)
        valid = rolling_dr.dropna()
        assert (valid > 0).all()


class TestCorrelationAnalyzerBenchmark:
    """Tests for benchmark correlation."""

    def test_correlation_with_benchmark(self, sample_returns_matrix, benchmark_returns):
        """Test correlation with benchmark."""
        analyzer = CorrelationAnalyzer()

        corr_bench = analyzer.correlation_with_benchmark(
            sample_returns_matrix, benchmark_returns
        )

        assert isinstance(corr_bench, pd.Series)
        assert len(corr_bench) == len(sample_returns_matrix.columns)
        assert corr_bench.name == 'corr_with_benchmark'
        # All correlations should be valid
        assert corr_bench.notna().all()


class TestCorrelationAnalyzerStability:
    """Tests for correlation stability analysis."""

    def test_correlation_stability_basic(self, sample_returns_matrix):
        """Test basic correlation stability."""
        analyzer = CorrelationAnalyzer()

        stability = analyzer.correlation_stability(sample_returns_matrix)

        assert isinstance(stability, pd.DataFrame)
        assert 'algo1' in stability.columns
        assert 'algo2' in stability.columns
        assert 'mean_corr' in stability.columns
        assert 'std_corr' in stability.columns

    def test_correlation_stability_with_regime(self, sample_returns_matrix):
        """Test correlation stability with regime labels."""
        analyzer = CorrelationAnalyzer()

        # Create regime labels
        n = len(sample_returns_matrix)
        regime_labels = pd.Series(
            [0] * (n // 2) + [1] * (n - n // 2),
            index=sample_returns_matrix.index,
            name='regime'
        )

        stability = analyzer.correlation_stability(
            sample_returns_matrix, regime_labels=regime_labels
        )

        assert isinstance(stability, pd.DataFrame)
        # Should have regime-specific columns
        assert any('corr_' in c for c in stability.columns)


class TestCorrelationAnalyzerClustering:
    """Tests for correlation-based clustering."""

    def test_cluster_by_correlation(self, sample_returns_matrix):
        """Test correlation-based clustering."""
        analyzer = CorrelationAnalyzer()

        clusters = analyzer.cluster_by_correlation(
            sample_returns_matrix, n_clusters=2
        )

        assert isinstance(clusters, dict)
        assert len(clusters) == len(sample_returns_matrix.columns)
        # Cluster labels should be 0 or 1
        assert all(v in [0, 1] for v in clusters.values())


class TestCorrelationAnalyzerLowCorrelation:
    """Tests for finding low correlation pairs."""

    def test_get_low_correlation_pairs(self, uncorrelated_returns):
        """Test finding low correlation pairs."""
        analyzer = CorrelationAnalyzer()

        pairs = analyzer.get_low_correlation_pairs(
            uncorrelated_returns, threshold=0.3
        )

        assert isinstance(pairs, list)
        for algo1, algo2, corr in pairs:
            assert abs(corr) < 0.3


class TestCorrelationAnalyzerReport:
    """Tests for report generation."""

    def test_generate_report_basic(self, sample_returns_matrix):
        """Test basic report generation."""
        analyzer = CorrelationAnalyzer()

        report = analyzer.generate_correlation_report(sample_returns_matrix)

        assert isinstance(report, str)
        assert 'CORRELATION ANALYSIS' in report
        assert 'DIVERSIFICATION' in report
        assert 'CORRELATION MATRIX' in report

    def test_generate_report_with_benchmark(
        self, sample_returns_matrix, benchmark_returns
    ):
        """Test report with benchmark."""
        analyzer = CorrelationAnalyzer()

        report = analyzer.generate_correlation_report(
            sample_returns_matrix,
            benchmark_returns=benchmark_returns,
        )

        assert 'CORRELATION WITH BENCHMARK' in report


# =============================================================================
# Edge Cases
# =============================================================================

class TestCorrelationEdgeCases:
    """Test edge cases."""

    def test_single_asset(self):
        """Test with single asset."""
        analyzer = CorrelationAnalyzer()
        returns = pd.DataFrame(
            {'algo_0': np.random.normal(0, 0.01, 100)},
            index=pd.date_range('2023-01-01', periods=100, freq='B')
        )

        corr = analyzer.correlation_matrix(returns)
        assert corr.shape == (1, 1)
        assert corr.iloc[0, 0] == 1.0

    def test_short_series(self):
        """Test with very short series."""
        analyzer = CorrelationAnalyzer()
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (5, 3)),
            columns=['a', 'b', 'c']
        )

        # Should not crash
        corr = analyzer.correlation_matrix(returns)
        assert corr.shape == (3, 3)

    def test_nan_handling(self, sample_returns_matrix):
        """Test handling of NaN values."""
        analyzer = CorrelationAnalyzer()

        # Add some NaNs
        returns_with_nan = sample_returns_matrix.copy()
        returns_with_nan.iloc[:10, 0] = np.nan
        returns_with_nan.iloc[50:60, 2] = np.nan

        corr = analyzer.correlation_matrix(returns_with_nan)

        # Should still compute valid correlations
        assert corr.notna().all().all()

    def test_constant_returns(self):
        """Test with constant returns (zero variance)."""
        analyzer = CorrelationAnalyzer()
        returns = pd.DataFrame({
            'algo_0': np.zeros(100),
            'algo_1': np.random.normal(0, 0.01, 100),
        })

        corr = analyzer.correlation_matrix(returns)

        # Correlation with zero-variance series should be 0 or NaN
        # The numba implementation returns 0 for these cases
        assert corr.iloc[0, 1] == 0 or np.isnan(corr.iloc[0, 1])
