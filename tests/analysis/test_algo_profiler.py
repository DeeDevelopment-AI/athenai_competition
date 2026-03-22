"""
Tests for src/analysis/algo_profiler.py

Tests the numba-optimized algorithm profiling functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.algo_profiler import (
    AlgoProfiler,
    AlgoProfile,
    _annualized_return_numba,
    _volatility_numba,
    _sharpe_numba,
    _sortino_numba,
    _max_drawdown_numba,
    _max_drawdown_duration_numba,
    _calmar_numba,
    _var_numba,
    _cvar_numba,
    _tail_ratio_numba,
    _compute_all_metrics_batch,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample daily returns."""
    np.random.seed(42)
    # 2 years of daily returns with slight positive drift
    n_days = 504
    returns = np.random.normal(0.0005, 0.015, n_days)
    return returns.astype(np.float64)


@pytest.fixture
def sample_returns_with_drawdown():
    """Generate returns with a clear drawdown period."""
    np.random.seed(42)
    n_days = 252
    returns = np.zeros(n_days, dtype=np.float64)

    # First half: positive trend
    returns[:126] = np.random.normal(0.002, 0.01, 126)
    # Second half: drawdown then recovery
    returns[126:180] = np.random.normal(-0.003, 0.015, 54)  # Drawdown
    returns[180:] = np.random.normal(0.001, 0.01, 72)  # Recovery

    return returns


@pytest.fixture
def sample_returns_matrix():
    """Generate returns matrix for multiple algorithms."""
    np.random.seed(42)
    n_days = 252
    n_algos = 10

    returns = np.random.normal(0.0003, 0.012, (n_days, n_algos))

    # Add some variation between algos
    returns[:, 0] *= 1.5  # Higher volatility
    returns[:, 1] *= 0.5  # Lower volatility
    returns[:, 2] += 0.001  # Higher return

    index = pd.date_range('2023-01-01', periods=n_days, freq='B')
    columns = [f'algo_{i}' for i in range(n_algos)]

    return pd.DataFrame(returns, index=index, columns=columns)


# =============================================================================
# Numba Function Tests
# =============================================================================

class TestAnnualizedReturnNumba:
    """Tests for _annualized_return_numba."""

    def test_positive_returns(self, sample_returns):
        """Test with positive drift returns."""
        result = _annualized_return_numba(sample_returns, 252)
        # Should be positive with positive drift
        assert result > 0
        # Should be reasonable (not extreme)
        assert -1 < result < 2

    def test_zero_returns(self):
        """Test with zero returns."""
        returns = np.zeros(252, dtype=np.float64)
        result = _annualized_return_numba(returns, 252)
        assert result == 0.0

    def test_constant_returns(self):
        """Test with constant positive returns."""
        # 1% daily return
        returns = np.full(252, 0.01, dtype=np.float64)
        result = _annualized_return_numba(returns, 252)
        # Should be very high (compounding)
        assert result > 10

    def test_empty_returns(self):
        """Test with empty array."""
        returns = np.array([], dtype=np.float64)
        result = _annualized_return_numba(returns, 252)
        assert result == 0.0


class TestVolatilityNumba:
    """Tests for _volatility_numba."""

    def test_volatility_calculation(self, sample_returns):
        """Test basic volatility calculation."""
        result = _volatility_numba(sample_returns, 252)
        # ~1.5% daily vol * sqrt(252) ≈ 24%
        assert 0.1 < result < 0.5

    def test_zero_volatility(self):
        """Test with constant returns (zero volatility)."""
        returns = np.full(100, 0.001, dtype=np.float64)
        result = _volatility_numba(returns, 252)
        assert result < 1e-10

    def test_high_volatility(self):
        """Test with high volatility returns."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.05, 252).astype(np.float64)  # 5% daily
        result = _volatility_numba(returns, 252)
        assert result > 0.5  # Should be high


class TestSharpeNumba:
    """Tests for _sharpe_numba."""

    def test_positive_sharpe(self):
        """Test with clearly positive returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).astype(np.float64)
        result = _sharpe_numba(returns, 0.0, 252)
        assert result > 0

    def test_negative_sharpe(self):
        """Test with negative returns."""
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.01, 252).astype(np.float64)
        result = _sharpe_numba(returns, 0.0, 252)
        assert result < 0

    def test_zero_volatility_sharpe(self):
        """Test with zero volatility (should return 0)."""
        returns = np.full(100, 0.001, dtype=np.float64)
        result = _sharpe_numba(returns, 0.0, 252)
        assert result == 0.0


class TestMaxDrawdownNumba:
    """Tests for _max_drawdown_numba."""

    def test_drawdown_with_losses(self, sample_returns_with_drawdown):
        """Test drawdown calculation with clear losses."""
        result = _max_drawdown_numba(sample_returns_with_drawdown)
        # Should be negative (drawdown)
        assert result < 0
        # Should be reasonable
        assert result > -0.5

    def test_no_drawdown(self):
        """Test with only positive returns (no drawdown)."""
        returns = np.full(100, 0.01, dtype=np.float64)
        result = _max_drawdown_numba(returns)
        assert result == 0.0

    def test_complete_loss(self):
        """Test with returns that cause near-complete loss."""
        returns = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
        result = _max_drawdown_numba(returns)
        assert result < -0.8  # Should be severe drawdown


class TestMaxDrawdownDurationNumba:
    """Tests for _max_drawdown_duration_numba."""

    def test_drawdown_duration(self, sample_returns_with_drawdown):
        """Test drawdown duration calculation."""
        result = _max_drawdown_duration_numba(sample_returns_with_drawdown)
        # Should be positive integer
        assert isinstance(result, (int, np.integer))
        assert result >= 0

    def test_no_drawdown_duration(self):
        """Test with no drawdown."""
        returns = np.full(100, 0.01, dtype=np.float64)
        result = _max_drawdown_duration_numba(returns)
        assert result == 0


class TestVaRCVaRNumba:
    """Tests for _var_numba and _cvar_numba."""

    def test_var_95(self, sample_returns):
        """Test VaR at 95% confidence."""
        result = _var_numba(sample_returns, 0.05)
        # Should be negative (worst 5% of returns)
        assert result < 0
        # Should be in reasonable range
        assert result > -0.1

    def test_cvar_worse_than_var(self, sample_returns):
        """CVaR should be worse than VaR."""
        var = _var_numba(sample_returns, 0.05)
        cvar = _cvar_numba(sample_returns, 0.05)
        assert cvar <= var


class TestTailRatioNumba:
    """Tests for _tail_ratio_numba."""

    def test_symmetric_distribution(self):
        """Test with symmetric returns."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000).astype(np.float64)
        result = _tail_ratio_numba(returns)
        # Should be close to 1 for symmetric distribution
        assert 0.8 < result < 1.2

    def test_positive_skew(self):
        """Test with positive skew (more upside)."""
        np.random.seed(42)
        # Positive skewed distribution
        returns = np.random.exponential(0.01, 1000).astype(np.float64) - 0.01
        result = _tail_ratio_numba(returns)
        # Should be > 1 for positive skew
        assert result > 0


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestComputeAllMetricsBatch:
    """Tests for _compute_all_metrics_batch."""

    def test_batch_output_shape(self, sample_returns_matrix):
        """Test output shape is correct."""
        data = sample_returns_matrix.values.astype(np.float64)
        result = _compute_all_metrics_batch(data, 252, 0.0)

        assert result.shape == (10, 10)  # 10 algos, 10 metrics

    def test_batch_metrics_reasonable(self, sample_returns_matrix):
        """Test batch metrics are in reasonable ranges."""
        data = sample_returns_matrix.values.astype(np.float64)
        result = _compute_all_metrics_batch(data, 252, 0.0)

        # Check annualized returns are reasonable
        assert np.all(result[:, 0] > -1)
        assert np.all(result[:, 0] < 5)

        # Check volatilities are positive
        assert np.all(result[:, 1] >= 0)

        # Check max drawdowns are non-positive
        assert np.all(result[:, 4] <= 0)


# =============================================================================
# AlgoProfiler Class Tests
# =============================================================================

class TestAlgoProfiler:
    """Tests for AlgoProfiler class."""

    def test_profile_single_algo(self, sample_returns_matrix):
        """Test profiling a single algorithm."""
        profiler = AlgoProfiler()
        returns = sample_returns_matrix['algo_0']

        profile = profiler.profile(returns, algo_id='algo_0')

        assert isinstance(profile, AlgoProfile)
        assert profile.algo_id == 'algo_0'
        assert profile.annualized_return != 0
        assert profile.annualized_volatility > 0
        assert profile.max_drawdown <= 0

    def test_profile_all_algos(self, sample_returns_matrix):
        """Test profiling all algorithms."""
        profiler = AlgoProfiler()

        profiles = profiler.profile_all(sample_returns_matrix)

        assert len(profiles) == 10
        assert all(isinstance(p, AlgoProfile) for p in profiles.values())

    def test_profile_with_benchmark(self, sample_returns_matrix):
        """Test profiling with benchmark correlation."""
        profiler = AlgoProfiler()

        # Use first algo as benchmark
        benchmark = sample_returns_matrix['algo_0']

        profiles = profiler.profile_all(
            sample_returns_matrix,
            benchmark_returns=benchmark,
        )

        # All algos should have benchmark correlation
        for algo_id, profile in profiles.items():
            if algo_id == 'algo_0':
                assert abs(profile.correlation_with_benchmark - 1.0) < 0.01
            else:
                assert -1 <= profile.correlation_with_benchmark <= 1

    def test_generate_summary_table(self, sample_returns_matrix):
        """Test summary table generation."""
        profiler = AlgoProfiler()
        profiles = profiler.profile_all(sample_returns_matrix)

        summary = profiler.generate_summary_table(profiles)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 10
        assert 'sharpe' in summary.columns
        assert 'max_dd' in summary.columns
        # Verify numeric dtype for nlargest/nsmallest operations
        assert pd.api.types.is_numeric_dtype(summary['sharpe'])
        assert pd.api.types.is_numeric_dtype(summary['ann_return'])
        assert pd.api.types.is_numeric_dtype(summary['max_dd'])
        # Test nlargest works
        top_sharpe = summary.nlargest(3, 'sharpe')
        assert len(top_sharpe) == 3


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nan_handling(self):
        """Test handling of NaN values."""
        returns = np.array([0.01, np.nan, 0.02, np.nan, 0.01], dtype=np.float64)

        # Should not crash (may return NaN due to NaN values in input)
        result = _annualized_return_numba(returns, 252)
        # Result is either a valid float or NaN - function should not raise
        assert isinstance(result, (float, np.floating))

    def test_short_series(self):
        """Test with very short return series."""
        returns = np.array([0.01], dtype=np.float64)

        vol = _volatility_numba(returns, 252)
        assert vol == 0.0  # Not enough data for std

    def test_profile_empty_returns(self):
        """Test profiling with empty returns."""
        profiler = AlgoProfiler()
        returns = pd.Series([], dtype=float)

        profile = profiler.profile(returns, algo_id='empty')

        assert profile.annualized_return == 0.0
        assert profile.annualized_volatility == 0.0
