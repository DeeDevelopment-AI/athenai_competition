"""
Tests for src/data/preprocessor.py

Tests cover:
- trim_dead_tail functionality
- process_algorithm_returns calculation
- process_algorithm_equity_curve construction
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import AlgorithmData
from src.data.preprocessor import (
    DataPreprocessor,
    ProcessedAlgoData,
    TrimInfo,
    trim_dead_tail,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance."""
    return DataPreprocessor(initial_capital=100.0)


@pytest.fixture
def sample_ohlc():
    """Create sample OHLC DataFrame."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 * (1 + np.random.randn(100).cumsum() * 0.01)

    return pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
    }, index=dates)


@pytest.fixture
def algo_data(sample_ohlc):
    """Create sample AlgorithmData."""
    return AlgorithmData(
        algo_id='test_algo',
        ohlc=sample_ohlc,
        raw_path=Path('/tmp/test_algo.csv'),
    )


# =============================================================================
# Tests for trim_dead_tail
# =============================================================================

class TestTrimDeadTail:
    """Tests for trim_dead_tail function."""

    def test_no_dead_tail(self):
        """Test algorithm with no dead tail."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

        trimmed, info = trim_dead_tail(prices)

        assert not info.was_trimmed
        assert info.original_days == 100
        assert info.alive_days == 100
        assert info.dead_days == 0
        assert info.death_date is None
        assert len(trimmed) == len(prices)

    def test_with_dead_tail(self):
        """Test algorithm that stopped trading."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Price moves for first 50 days, then flatlines
        prices = np.concatenate([
            100 * (1 + np.random.randn(50).cumsum() * 0.01),
            [105.0] * 50  # Flatlined at 105
        ])
        prices = pd.Series(prices, index=dates)

        trimmed, info = trim_dead_tail(prices, max_flat_pct=0.10)

        assert info.was_trimmed
        assert info.original_days == 100
        assert info.alive_days < 100
        assert info.dead_days > 0
        assert info.death_date is not None
        assert info.death_price is not None

    def test_short_dead_tail_not_trimmed(self):
        """Test that short dead tails are not trimmed."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Price moves for first 95 days, flatlines for only 5 days
        prices = np.concatenate([
            100 * (1 + np.random.randn(95).cumsum() * 0.01),
            [105.0] * 5
        ])
        prices = pd.Series(prices, index=dates)

        trimmed, info = trim_dead_tail(prices, max_flat_pct=0.10)

        # 5/100 = 5% < 10% threshold, so should not trim
        assert not info.was_trimmed

    def test_completely_dead_algo(self):
        """Test algorithm that never moved."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series([100.0] * 100, index=dates)

        trimmed, info = trim_dead_tail(prices)

        assert not info.was_trimmed
        assert info.alive_days == 0
        assert info.dead_days == 100

    def test_short_series(self):
        """Test with very short series."""
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        prices = pd.Series([100.0, 101.0, 102.0], index=dates)

        trimmed, info = trim_dead_tail(prices)

        assert not info.was_trimmed
        assert len(trimmed) == 3

    def test_trim_info_types(self):
        """Test TrimInfo dataclass has correct types."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.concatenate([
            100 * (1 + np.random.randn(30).cumsum() * 0.01),
            [105.0] * 70
        ])
        prices = pd.Series(prices, index=dates)

        _, info = trim_dead_tail(prices, max_flat_pct=0.10)

        assert isinstance(info, TrimInfo)
        assert isinstance(info.was_trimmed, (bool, np.bool_))
        assert isinstance(info.original_days, (int, np.integer))
        assert isinstance(info.alive_days, (int, np.integer))
        assert isinstance(info.dead_days, (int, np.integer))


# =============================================================================
# Tests for DataPreprocessor.process_algorithm
# =============================================================================

class TestProcessAlgorithm:
    """Tests for process_algorithm method."""

    def test_process_algorithm_basic(self, preprocessor, algo_data):
        """Test basic algorithm processing."""
        processed = preprocessor.process_algorithm(algo_data)

        assert isinstance(processed, ProcessedAlgoData)
        assert processed.algo_id == 'test_algo'
        assert len(processed.returns) > 0
        assert len(processed.equity_curve) > 0

    def test_process_algorithm_returns(self, preprocessor, algo_data):
        """Test return calculation from close prices."""
        processed = preprocessor.process_algorithm(algo_data)

        returns = processed.returns

        # Returns should be percentage changes
        assert returns.mean() != 0 or returns.std() != 0
        # Returns should be reasonable (not huge)
        assert abs(returns).max() < 1.0  # Less than 100% daily move

    def test_process_algorithm_equity_curve(self, preprocessor, algo_data):
        """Test equity curve construction."""
        processed = preprocessor.process_algorithm(algo_data)

        equity = processed.equity_curve

        # Should start at initial capital
        assert equity.iloc[0] == preprocessor.initial_capital
        # Should be monotonically related to cumulative returns
        assert len(equity) == len(processed.returns) + 1

    def test_process_algorithm_with_trim(self, preprocessor):
        """Test processing with dead tail trimming."""
        # Create algo with dead tail
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.concatenate([
            100 * (1 + np.random.randn(30).cumsum() * 0.01),
            [105.0] * 70
        ])

        ohlc = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
        }, index=dates)

        algo_data = AlgorithmData(
            algo_id='dead_tail_algo',
            ohlc=ohlc,
            raw_path=Path('/tmp/dead_tail.csv'),
        )

        processed = preprocessor.process_algorithm(algo_data, trim_dead=True)

        assert processed.trim_info is not None
        assert processed.trim_info.was_trimmed
        assert len(processed.ohlc) < 100

    def test_process_algorithm_preserves_ohlc(self, preprocessor, algo_data):
        """Test that OHLC data is preserved."""
        processed = preprocessor.process_algorithm(algo_data)

        assert 'open' in processed.ohlc.columns
        assert 'high' in processed.ohlc.columns
        assert 'low' in processed.ohlc.columns
        assert 'close' in processed.ohlc.columns

    def test_process_algorithm_no_close_raises(self, preprocessor):
        """Test error when no close column."""
        ohlc = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
        }, index=pd.date_range('2020-01-01', periods=3, freq='D'))

        algo_data = AlgorithmData(
            algo_id='no_close',
            ohlc=ohlc,
            raw_path=Path('/tmp/no_close.csv'),
        )

        with pytest.raises(ValueError, match="No 'close' column"):
            preprocessor.process_algorithm(algo_data)


# =============================================================================
# Tests for DataPreprocessor.process_all_algorithms
# =============================================================================

class TestProcessAllAlgorithms:
    """Tests for process_all_algorithms method."""

    def test_process_all_algorithms(self, preprocessor, sample_ohlc):
        """Test processing multiple algorithms."""
        algos = {
            'algo1': AlgorithmData('algo1', sample_ohlc.copy(), Path('/tmp/algo1.csv')),
            'algo2': AlgorithmData('algo2', sample_ohlc.copy(), Path('/tmp/algo2.csv')),
        }

        processed = preprocessor.process_all_algorithms(algos, show_progress=False)

        assert len(processed) == 2
        assert 'algo1' in processed
        assert 'algo2' in processed

    def test_process_all_algorithms_with_errors(self, preprocessor, sample_ohlc):
        """Test that errors in one algo don't stop others."""
        bad_ohlc = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            # No close column
        }, index=pd.date_range('2020-01-01', periods=1, freq='D'))

        algos = {
            'good_algo': AlgorithmData('good_algo', sample_ohlc.copy(), Path('/tmp/good.csv')),
            'bad_algo': AlgorithmData('bad_algo', bad_ohlc, Path('/tmp/bad.csv')),
        }

        processed = preprocessor.process_all_algorithms(algos, show_progress=False)

        assert 'good_algo' in processed
        assert 'bad_algo' not in processed


# =============================================================================
# Tests for returns matrix creation
# =============================================================================

class TestCreateReturnsMatrix:
    """Tests for create_returns_matrix method."""

    def test_create_returns_matrix(self, preprocessor, sample_ohlc):
        """Test creating returns matrix from processed algos."""
        algos = {
            'algo1': AlgorithmData('algo1', sample_ohlc.copy(), Path('/tmp/algo1.csv')),
            'algo2': AlgorithmData('algo2', sample_ohlc.copy(), Path('/tmp/algo2.csv')),
        }

        processed = preprocessor.process_all_algorithms(algos, show_progress=False)
        returns_matrix = preprocessor.create_returns_matrix(processed)

        assert isinstance(returns_matrix, pd.DataFrame)
        assert 'algo1' in returns_matrix.columns
        assert 'algo2' in returns_matrix.columns
        assert returns_matrix.index.is_monotonic_increasing

    def test_create_returns_matrix_fillna(self, preprocessor):
        """Test NaN filling in returns matrix."""
        # Create algos with different date ranges
        dates1 = pd.date_range('2020-01-01', periods=50, freq='D')
        dates2 = pd.date_range('2020-02-01', periods=50, freq='D')

        ohlc1 = pd.DataFrame({
            'open': [100] * 50, 'high': [101] * 50,
            'low': [99] * 50, 'close': [100 + i * 0.1 for i in range(50)],
        }, index=dates1)

        ohlc2 = pd.DataFrame({
            'open': [100] * 50, 'high': [101] * 50,
            'low': [99] * 50, 'close': [100 + i * 0.1 for i in range(50)],
        }, index=dates2)

        algos = {
            'algo1': AlgorithmData('algo1', ohlc1, Path('/tmp/algo1.csv')),
            'algo2': AlgorithmData('algo2', ohlc2, Path('/tmp/algo2.csv')),
        }

        processed = preprocessor.process_all_algorithms(algos, show_progress=False)
        returns_matrix = preprocessor.create_returns_matrix(processed, fillna=True)

        # Should have no NaNs when fillna=True
        assert not returns_matrix.isna().any().any()

        # Without fillna, should have NaNs
        returns_matrix_nan = preprocessor.create_returns_matrix(processed, fillna=False)
        assert returns_matrix_nan.isna().any().any()


# =============================================================================
# Tests for summary stats
# =============================================================================

class TestGetSummaryStats:
    """Tests for get_summary_stats method."""

    def test_get_summary_stats(self, preprocessor, sample_ohlc):
        """Test summary stats generation."""
        algos = {
            'algo1': AlgorithmData('algo1', sample_ohlc.copy(), Path('/tmp/algo1.csv')),
        }

        processed = preprocessor.process_all_algorithms(algos, show_progress=False)
        stats = preprocessor.get_summary_stats(processed)

        assert isinstance(stats, pd.DataFrame)
        assert 'algo1' in stats.index
        assert 'n_days' in stats.columns
        assert 'mean_return_daily' in stats.columns
        assert 'sharpe' in stats.columns

    def test_get_summary_stats_with_trim(self, preprocessor):
        """Test summary stats include trim info when available."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = np.concatenate([
            100 * (1 + np.random.randn(30).cumsum() * 0.01),
            [105.0] * 70
        ])

        ohlc = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
        }, index=dates)

        algo = AlgorithmData('trimmed', ohlc, Path('/tmp/trimmed.csv'))
        processed = {'trimmed': preprocessor.process_algorithm(algo, trim_dead=True)}

        stats = preprocessor.get_summary_stats(processed)

        assert 'was_trimmed' in stats.columns
        assert stats.loc['trimmed', 'was_trimmed'] == True

    def test_get_summary_stats_annualized_return_is_compounded(self, preprocessor):
        """Annualized return should use geometric compounding, not arithmetic mean * 252."""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        daily_return = 0.001
        close = 100 * (1 + daily_return) ** np.arange(len(dates))
        ohlc = pd.DataFrame({
            'open': close,
            'high': close,
            'low': close,
            'close': close,
        }, index=dates)

        algo = AlgorithmData('compounded', ohlc, Path('/tmp/compounded.csv'))
        processed = {'compounded': preprocessor.process_algorithm(algo)}

        stats = preprocessor.get_summary_stats(processed)
        expected = (1 + daily_return) ** 252 - 1

        assert stats.loc['compounded', 'annualized_return'] == pytest.approx(expected, rel=1e-4)


class TestBenchmarkReturns:
    """Tests for benchmark daily return reconstruction."""

    def test_calculate_benchmark_daily_returns_does_not_backfill_future_weights(self, preprocessor):
        """The first return should remain NaN without ex-ante weights."""
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        returns_matrix = pd.DataFrame({
            'algo1': [0.01, 0.02, 0.03],
        }, index=dates)
        weights = pd.DataFrame({
            'algo1': [1.0, 1.0, 1.0],
        }, index=dates)

        benchmark_returns = preprocessor.calculate_benchmark_daily_returns(returns_matrix, weights)

        assert pd.isna(benchmark_returns.iloc[0])
        assert benchmark_returns.iloc[1] == pytest.approx(0.02)
