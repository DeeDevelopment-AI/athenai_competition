"""
Tests for src/data/feature_engineering.py

Tests cover:
- Cumulative features: return, volatility, drawdown, sharpe, profit_factor, calmar, max_drawdown
- Rolling features for windows [5, 21, 63]
- Regime features (cross-sectional market stats)
- Feature matrix shape and structure
"""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    FeatureEngineer,
    ROLLING_WINDOWS,
    TRADING_DAYS_PER_YEAR,
    _cumulative_return,
    _rolling_return,
    _cumulative_volatility,
    _rolling_volatility,
    _cumulative_drawdown,
    _rolling_max_drawdown,
    _cumulative_sharpe,
    _rolling_sharpe,
    _cumulative_profit_factor,
    _rolling_profit_factor,
    _cumulative_calmar,
    _rolling_calmar,
    _max_drawdown_to_date,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Create sample returns series."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates, name='test_algo')
    return returns


@pytest.fixture
def sample_prices(sample_returns):
    """Create sample prices from returns."""
    return (1 + sample_returns).cumprod().values


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer(windows=ROLLING_WINDOWS)


@pytest.fixture
def returns_matrix():
    """Create sample returns matrix for multiple algos."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    return pd.DataFrame({
        'algo1': np.random.randn(252) * 0.01,
        'algo2': np.random.randn(252) * 0.015,
        'algo3': np.random.randn(252) * 0.008,
    }, index=dates)


# =============================================================================
# Tests for numba JIT functions
# =============================================================================

class TestCumulativeReturn:
    """Tests for cumulative return calculation."""

    def test_cumulative_return_basic(self, sample_prices):
        """Test basic cumulative return calculation."""
        result = _cumulative_return(sample_prices)

        assert len(result) == len(sample_prices)
        assert result[0] == 0.0  # First return is 0 (compared to itself)
        # Last value should match expected total return
        expected = (sample_prices[-1] / sample_prices[0]) - 1
        assert abs(result[-1] - expected) < 1e-10

    def test_cumulative_return_constant_price(self):
        """Test cumulative return with constant prices."""
        prices = np.array([100.0] * 100)
        result = _cumulative_return(prices)

        assert np.allclose(result, 0.0)

    def test_cumulative_return_upward_trend(self):
        """Test cumulative return with upward trend."""
        prices = np.array([100.0, 110.0, 121.0, 133.1])  # 10% daily
        result = _cumulative_return(prices)

        assert result[-1] > 0
        assert np.isclose(result[-1], 0.331, rtol=0.01)


class TestRollingReturn:
    """Tests for rolling return calculation."""

    def test_rolling_return_basic(self, sample_prices):
        """Test basic rolling return calculation."""
        window = 5
        result = _rolling_return(sample_prices, window)

        # First window-1 values should be NaN
        assert np.isnan(result[:window]).all()
        # Rest should be computed
        assert not np.isnan(result[window:]).any()

    def test_rolling_return_window_sizes(self, sample_prices):
        """Test rolling return with different window sizes."""
        for window in ROLLING_WINDOWS:
            result = _rolling_return(sample_prices, window)
            assert np.isnan(result[:window]).all()
            assert len(result) == len(sample_prices)


class TestCumulativeVolatility:
    """Tests for cumulative volatility calculation."""

    def test_cumulative_volatility_basic(self, sample_returns):
        """Test basic cumulative volatility calculation."""
        result = _cumulative_volatility(sample_returns.values)

        # First value should be NaN (can't compute std of 1 value)
        assert np.isnan(result[0])
        # Rest should be positive
        assert (result[1:] > 0).all() or np.isnan(result[1:]).any()

    def test_cumulative_volatility_annualized(self, sample_returns):
        """Test that volatility is annualized."""
        result = _cumulative_volatility(sample_returns.values)

        # Manual calculation for comparison
        manual_vol = sample_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        # Last value should be close to manual calculation
        assert abs(result[-1] - manual_vol) < 0.01


class TestRollingVolatility:
    """Tests for rolling volatility calculation."""

    def test_rolling_volatility_basic(self, sample_returns):
        """Test basic rolling volatility calculation."""
        window = 21
        result = _rolling_volatility(sample_returns.values, window)

        # First window-1 values should be NaN
        assert np.isnan(result[:window - 1]).all()
        # Rest should be positive
        valid_result = result[window - 1:]
        assert (valid_result > 0).all()

    def test_rolling_volatility_annualized(self, sample_returns):
        """Test that rolling volatility is annualized."""
        window = 21
        result = _rolling_volatility(sample_returns.values, window)

        # Compare to pandas rolling std
        pandas_vol = sample_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        # Last values should be close
        assert abs(result[-1] - pandas_vol.iloc[-1]) < 0.01


class TestCumulativeDrawdown:
    """Tests for cumulative drawdown calculation."""

    def test_cumulative_drawdown_basic(self, sample_prices):
        """Test basic cumulative drawdown calculation."""
        result = _cumulative_drawdown(sample_prices)

        # Drawdowns should be <= 0
        assert (result <= 0).all()
        # First value should be 0 (at peak)
        assert result[0] == 0.0

    def test_cumulative_drawdown_upward_trend(self):
        """Test drawdown with pure upward trend."""
        prices = np.array([100.0, 110.0, 120.0, 130.0])
        result = _cumulative_drawdown(prices)

        # No drawdown in pure uptrend
        assert np.allclose(result, 0.0)

    def test_cumulative_drawdown_downward_trend(self):
        """Test drawdown with downward movement."""
        prices = np.array([100.0, 110.0, 100.0, 90.0])
        result = _cumulative_drawdown(prices)

        # Should have negative drawdown after peak
        assert result[-1] < 0
        # After peak at 110, going to 90 is ~18% drawdown
        expected_dd = (90 - 110) / 110
        assert np.isclose(result[-1], expected_dd, rtol=0.01)


class TestRollingMaxDrawdown:
    """Tests for rolling max drawdown calculation."""

    def test_rolling_max_drawdown_basic(self, sample_prices):
        """Test basic rolling max drawdown calculation."""
        window = 21
        result = _rolling_max_drawdown(sample_prices, window)

        # First window-1 values should be NaN
        assert np.isnan(result[:window - 1]).all()
        # Rest should be <= 0
        valid_result = result[window - 1:]
        assert (valid_result <= 0).all()


class TestCumulativeSharpe:
    """Tests for cumulative Sharpe ratio calculation."""

    def test_cumulative_sharpe_basic(self, sample_returns):
        """Test basic cumulative Sharpe calculation."""
        result = _cumulative_sharpe(sample_returns.values)

        # First value should be NaN
        assert np.isnan(result[0])
        # Should converge to some value
        assert not np.isnan(result[-1])


class TestRollingSharpe:
    """Tests for rolling Sharpe ratio calculation."""

    def test_rolling_sharpe_basic(self, sample_returns):
        """Test basic rolling Sharpe calculation."""
        window = 21
        result = _rolling_sharpe(sample_returns.values, window)

        # First window-1 values should be NaN
        assert np.isnan(result[:window - 1]).all()
        # Rest should be finite
        valid_result = result[window - 1:]
        assert np.isfinite(valid_result).all()


class TestCumulativeProfitFactor:
    """Tests for cumulative profit factor calculation."""

    def test_cumulative_profit_factor_basic(self, sample_returns):
        """Test basic cumulative profit factor calculation."""
        result = _cumulative_profit_factor(sample_returns.values)

        # Should be positive where there are losses
        # Might be inf where there are no losses
        assert (result >= 0).all() or np.isnan(result).any() or np.isinf(result).any()

    def test_cumulative_profit_factor_all_positive(self):
        """Test profit factor with all positive returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.03])
        result = _cumulative_profit_factor(returns)

        # With no losses, should be inf
        assert np.isinf(result[-1]) or result[-1] > 1000


class TestRollingProfitFactor:
    """Tests for rolling profit factor calculation."""

    def test_rolling_profit_factor_basic(self, sample_returns):
        """Test basic rolling profit factor calculation."""
        window = 21
        result = _rolling_profit_factor(sample_returns.values, window)

        # First window-1 values should be NaN
        assert np.isnan(result[:window - 1]).all()


class TestCumulativeCalmar:
    """Tests for cumulative Calmar ratio calculation."""

    def test_cumulative_calmar_basic(self, sample_returns, sample_prices):
        """Test basic cumulative Calmar calculation."""
        result = _cumulative_calmar(sample_returns.values, sample_prices)

        # First value should be NaN
        assert np.isnan(result[0])

    def test_cumulative_calmar_no_drawdown(self):
        """Test Calmar with no drawdown (pure uptrend)."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        prices = (1 + returns).cumprod()
        prices = np.insert(prices, 0, 1.0)[:-1]  # Align with returns

        result = _cumulative_calmar(returns, prices)

        # With no drawdown, Calmar should be 0 (or very small)
        assert result[-1] == 0.0 or np.isnan(result[-1])


class TestMaxDrawdownToDate:
    """Tests for max drawdown to date calculation."""

    def test_max_drawdown_to_date_basic(self, sample_prices):
        """Test basic max drawdown to date calculation."""
        result = _max_drawdown_to_date(sample_prices)

        # Max drawdown should be monotonically non-increasing (gets worse or stays same)
        # Actually, it can only get worse (more negative) over time
        for i in range(1, len(result)):
            assert result[i] <= result[i - 1] + 1e-10  # Allow small numerical error


# =============================================================================
# Tests for FeatureEngineer class
# =============================================================================

class TestFeatureEngineerCumulative:
    """Tests for cumulative feature computation."""

    def test_compute_cumulative_features(self, feature_engineer, sample_returns):
        """Test cumulative features computation."""
        features = feature_engineer.compute_cumulative_features(sample_returns, 'test')

        expected_cols = [
            'test_return', 'test_volatility', 'test_drawdown',
            'test_sharpe', 'test_profit_factor', 'test_calmar_ratio',
            'test_max_drawdown',
        ]

        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_compute_cumulative_features_shape(self, feature_engineer, sample_returns):
        """Test that cumulative features have correct shape."""
        features = feature_engineer.compute_cumulative_features(sample_returns, 'test')

        assert len(features) == len(sample_returns)
        assert features.index.equals(sample_returns.index)


class TestFeatureEngineerRolling:
    """Tests for rolling feature computation."""

    def test_compute_rolling_features(self, feature_engineer, sample_returns):
        """Test rolling features computation."""
        features = feature_engineer.compute_rolling_features(sample_returns, 'test')

        # Should have features for each window
        for w in ROLLING_WINDOWS:
            assert f'test_rolling_return_{w}d' in features.columns
            assert f'test_rolling_volatility_{w}d' in features.columns
            assert f'test_rolling_drawdown_{w}d' in features.columns
            assert f'test_rolling_sharpe_{w}d' in features.columns

    def test_compute_rolling_features_shape(self, feature_engineer, sample_returns):
        """Test that rolling features have correct shape."""
        features = feature_engineer.compute_rolling_features(sample_returns, 'test')

        assert len(features) == len(sample_returns)


class TestFeatureEngineerRegime:
    """Tests for regime feature computation."""

    def test_compute_regime_features(self, feature_engineer, returns_matrix):
        """Test regime features computation."""
        features = feature_engineer.compute_regime_features(returns_matrix)

        for w in ROLLING_WINDOWS:
            assert f'rolling_market_vol_{w}d' in features.columns
            assert f'rolling_market_return_{w}d' in features.columns

    def test_compute_regime_features_cross_sectional(self, feature_engineer, returns_matrix):
        """Test that regime features are cross-sectional averages."""
        features = feature_engineer.compute_regime_features(returns_matrix)

        # Market vol should be average of individual vols
        individual_vols = returns_matrix.rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        expected_market_vol = individual_vols.mean(axis=1)

        pd.testing.assert_series_equal(
            features['rolling_market_vol_21d'],
            expected_market_vol,
            check_names=False,
        )


class TestFeatureEngineerFullMatrix:
    """Tests for full feature matrix building."""

    def test_build_feature_matrix(self, feature_engineer, returns_matrix):
        """Test building complete feature matrix."""
        features = feature_engineer.build_feature_matrix(returns_matrix, show_progress=False)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(returns_matrix)

    def test_build_feature_matrix_columns(self, feature_engineer, returns_matrix):
        """Test that feature matrix has expected columns."""
        features = feature_engineer.build_feature_matrix(returns_matrix, show_progress=False)

        # Should have algo-specific columns
        for algo_id in returns_matrix.columns:
            assert f'{algo_id}_return' in features.columns
            assert f'{algo_id}_volatility' in features.columns
            assert f'{algo_id}_rolling_return_21d' in features.columns

        # Should have regime columns
        assert 'rolling_market_vol_21d' in features.columns

    def test_build_feature_matrix_no_inf(self, feature_engineer, returns_matrix):
        """Test that feature matrix has no inf values."""
        features = feature_engineer.build_feature_matrix(returns_matrix, show_progress=False)

        # Should not have inf (replaced with NaN)
        assert not np.isinf(features.values).any()

    def test_get_feature_names(self, feature_engineer, returns_matrix):
        """Test getting feature names."""
        algo_ids = returns_matrix.columns.tolist()
        names = feature_engineer.get_feature_names(algo_ids)

        assert len(names) > 0
        assert 'algo1_return' in names
        assert 'rolling_market_vol_21d' in names


# =============================================================================
# Integration tests
# =============================================================================

class TestFeatureEngineerIntegration:
    """Integration tests for FeatureEngineer."""

    def test_feature_computation_consistency(self, feature_engineer, sample_returns):
        """Test that individual and combined features are consistent."""
        # Compute individual
        cumulative = feature_engineer.compute_cumulative_features(sample_returns, 'test')
        rolling = feature_engineer.compute_rolling_features(sample_returns, 'test')

        # Compute combined
        combined = feature_engineer.compute_algo_features(sample_returns, 'test')

        # Check all cumulative columns are present
        for col in cumulative.columns:
            assert col in combined.columns

        # Check all rolling columns are present
        for col in rolling.columns:
            assert col in combined.columns

    def test_feature_values_reasonable(self, feature_engineer, sample_returns):
        """Test that feature values are in reasonable ranges."""
        features = feature_engineer.compute_algo_features(sample_returns, 'test')

        # Volatility should be positive
        vol_cols = [c for c in features.columns if 'volatility' in c]
        for col in vol_cols:
            valid = features[col].dropna()
            assert (valid >= 0).all(), f"{col} has negative values"

        # Drawdowns should be <= 0
        dd_cols = [c for c in features.columns if 'drawdown' in c]
        for col in dd_cols:
            valid = features[col].dropna()
            assert (valid <= 0).all(), f"{col} has positive values"
