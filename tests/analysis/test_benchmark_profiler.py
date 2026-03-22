"""
Tests for src/analysis/benchmark_profiler.py

Tests the benchmark reverse engineering functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.benchmark_profiler import (
    BenchmarkProfiler,
    BenchmarkProfile,
    _annualized_return_numba,
    _volatility_numba,
    _sharpe_numba,
    _max_drawdown_numba,
    _max_drawdown_duration_numba,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_benchmark_returns():
    """Generate sample benchmark daily returns."""
    np.random.seed(42)
    n_days = 504  # 2 years
    returns = np.random.normal(0.0003, 0.008, n_days)  # Lower vol than individual algos

    index = pd.date_range('2022-01-01', periods=n_days, freq='B')
    return pd.Series(returns, index=index, name='benchmark_return')


@pytest.fixture
def sample_benchmark_weights():
    """Generate sample benchmark weights over time."""
    np.random.seed(42)
    n_days = 504
    n_algos = 5

    # Generate weights that sum to ~1
    raw_weights = np.random.dirichlet(np.ones(n_algos), n_days)

    index = pd.date_range('2022-01-01', periods=n_days, freq='B')
    columns = [f'algo_{i}' for i in range(n_algos)]

    weights = pd.DataFrame(raw_weights, index=index, columns=columns)

    # Add some rebalancing events (sudden weight changes)
    for i in range(0, n_days, 63):  # Quarterly rebalancing
        if i + 1 < n_days:
            weights.iloc[i] = np.random.dirichlet(np.ones(n_algos))

    return weights


@pytest.fixture
def sample_regime_labels(sample_benchmark_returns):
    """Generate sample regime labels."""
    n_days = len(sample_benchmark_returns)

    # Create 4 regimes
    labels = np.zeros(n_days, dtype=int)
    labels[n_days // 4:n_days // 2] = 1
    labels[n_days // 2:3 * n_days // 4] = 2
    labels[3 * n_days // 4:] = 3

    return pd.Series(labels, index=sample_benchmark_returns.index, name='regime')


# =============================================================================
# Numba Function Tests
# =============================================================================

class TestBenchmarkNumbaFunctions:
    """Test numba-compiled functions."""

    def test_annualized_return(self, sample_benchmark_returns):
        """Test annualized return calculation."""
        returns = sample_benchmark_returns.values.astype(np.float64)
        result = _annualized_return_numba(returns, 252)

        assert isinstance(result, float)
        assert -1 < result < 2

    def test_volatility(self, sample_benchmark_returns):
        """Test volatility calculation."""
        returns = sample_benchmark_returns.values.astype(np.float64)
        result = _volatility_numba(returns, 252)

        assert result > 0
        assert result < 0.5  # Reasonable annualized vol

    def test_sharpe(self, sample_benchmark_returns):
        """Test Sharpe ratio calculation."""
        returns = sample_benchmark_returns.values.astype(np.float64)
        result = _sharpe_numba(returns, 0.0, 252)

        assert -5 < result < 5  # Reasonable Sharpe range

    def test_max_drawdown(self, sample_benchmark_returns):
        """Test max drawdown calculation."""
        returns = sample_benchmark_returns.values.astype(np.float64)
        result = _max_drawdown_numba(returns)

        assert result <= 0
        assert result > -1


# =============================================================================
# BenchmarkProfiler Tests
# =============================================================================

class TestBenchmarkProfiler:
    """Tests for BenchmarkProfiler class."""

    def test_profile_basic(self, sample_benchmark_returns, sample_benchmark_weights):
        """Test basic profiling."""
        profiler = BenchmarkProfiler()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=sample_benchmark_weights,
        )

        assert isinstance(profile, BenchmarkProfile)

    def test_performance_metrics(self, sample_benchmark_returns, sample_benchmark_weights):
        """Test performance metrics are calculated."""
        profiler = BenchmarkProfiler()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=sample_benchmark_weights,
        )

        assert profile.annualized_return != 0 or len(sample_benchmark_returns) < 10
        assert profile.annualized_volatility >= 0
        assert profile.max_drawdown <= 0
        assert profile.max_drawdown_duration >= 0

    def test_sizing_policy(self, sample_benchmark_returns, sample_benchmark_weights):
        """Test sizing policy metrics are calculated."""
        profiler = BenchmarkProfiler()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=sample_benchmark_weights,
        )

        # Average weights should exist for each algo
        assert len(profile.avg_weights) == 5
        assert all(0 <= v <= 1 for v in profile.avg_weights.values())

        # HHI should be between 1/N and 1
        assert 0.2 <= profile.concentration_hhi <= 1.0
        assert 0.2 <= profile.concentration_hhi_avg <= 1.0

    def test_temporal_policy(self, sample_benchmark_returns, sample_benchmark_weights):
        """Test temporal policy metrics."""
        profiler = BenchmarkProfiler()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=sample_benchmark_weights,
        )

        assert profile.rebalance_frequency_days > 0
        assert profile.avg_holding_period_days > 0
        assert profile.turnover_annualized >= 0

    def test_exposure_metrics(self, sample_benchmark_returns, sample_benchmark_weights):
        """Test exposure metrics."""
        profiler = BenchmarkProfiler()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=sample_benchmark_weights,
        )

        # Weights sum to ~1, so exposure should be ~1
        assert 0.5 < profile.avg_total_exposure < 1.5
        assert 0.5 < profile.max_total_exposure < 1.5

    def test_regime_analysis(
        self,
        sample_benchmark_returns,
        sample_benchmark_weights,
        sample_regime_labels,
    ):
        """Test regime-based analysis."""
        profiler = BenchmarkProfiler()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=sample_benchmark_weights,
            regime_labels=sample_regime_labels,
        )

        # Should have analysis for each regime
        assert len(profile.weights_by_regime) == 4
        assert len(profile.performance_by_regime) == 4

    def test_generate_report(self, sample_benchmark_returns, sample_benchmark_weights):
        """Test report generation."""
        profiler = BenchmarkProfiler()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=sample_benchmark_weights,
        )

        report = profiler.generate_report(profile)

        assert isinstance(report, str)
        assert 'BENCHMARK PROFILE' in report
        assert 'PERFORMANCE' in report
        assert 'TURNOVER' in report


# =============================================================================
# Edge Cases
# =============================================================================

class TestBenchmarkEdgeCases:
    """Test edge cases."""

    def test_empty_weights(self, sample_benchmark_returns):
        """Test with empty weights DataFrame."""
        profiler = BenchmarkProfiler()

        empty_weights = pd.DataFrame()

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=empty_weights,
        )

        # Should still compute performance metrics
        assert profile.annualized_return != 0 or len(sample_benchmark_returns) < 10

    def test_single_algo_weights(self, sample_benchmark_returns):
        """Test with single algorithm (100% weight)."""
        profiler = BenchmarkProfiler()

        weights = pd.DataFrame(
            {'algo_0': np.ones(len(sample_benchmark_returns))},
            index=sample_benchmark_returns.index,
        )

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=weights,
        )

        # HHI should be 1 (maximum concentration)
        assert abs(profile.concentration_hhi - 1.0) < 0.01

    def test_equal_weights(self, sample_benchmark_returns):
        """Test with equal weights."""
        profiler = BenchmarkProfiler()
        n_algos = 4

        weights = pd.DataFrame(
            {f'algo_{i}': np.full(len(sample_benchmark_returns), 1.0 / n_algos)
             for i in range(n_algos)},
            index=sample_benchmark_returns.index,
        )

        profile = profiler.profile(
            returns=sample_benchmark_returns,
            weights=weights,
        )

        # HHI should be 1/N
        expected_hhi = 1.0 / n_algos
        assert abs(profile.concentration_hhi - expected_hhi) < 0.01

        # No rebalancing needed for constant weights
        # Turnover should be very low
        assert profile.turnover_annualized < 0.1


# =============================================================================
# Helper Method Tests
# =============================================================================

class TestHelperMethods:
    """Test helper methods."""

    def test_calculate_turnover(self, sample_benchmark_weights):
        """Test turnover calculation."""
        profiler = BenchmarkProfiler()

        turnover = profiler._calculate_turnover(sample_benchmark_weights)

        assert isinstance(turnover, pd.Series)
        assert len(turnover) == len(sample_benchmark_weights)
        assert (turnover >= 0).all()

    def test_current_hhi(self, sample_benchmark_weights):
        """Test HHI calculation."""
        profiler = BenchmarkProfiler()

        hhi = profiler._current_hhi(sample_benchmark_weights.iloc[-1])

        # HHI should be between 1/N and 1
        assert 0.2 <= hhi <= 1.0

    def test_rebalance_frequency(self, sample_benchmark_weights):
        """Test rebalance frequency calculation."""
        profiler = BenchmarkProfiler()

        freq = profiler._rebalance_frequency(sample_benchmark_weights)

        assert freq > 0
        # Should detect quarterly rebalancing
        assert freq < 100  # Less than 100 days between rebalances
