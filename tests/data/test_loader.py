"""
Tests for src/data/loader.py

Tests cover:
- Basic algorithm loading
- Multiple date format detection
- Empty/small file handling
- Resample to daily functionality
- Listing available algorithms
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    DataLoader,
    AlgorithmData,
    _parse_datetime_robust,
    _detect_datetime_column,
    _detect_ohlc_columns,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create algorithms subdirectory
        algos_dir = Path(tmpdir) / "algorithms"
        algos_dir.mkdir()

        # Create a basic algorithm CSV
        basic_algo = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='D'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
        })
        basic_algo.to_csv(algos_dir / "basic_algo.csv", index=False)

        # Create algorithm with different date format
        diff_date_algo = pd.DataFrame({
            'date': ['01/01/2020', '02/01/2020', '03/01/2020'] + [f'{i:02d}/01/2020' for i in range(4, 32)],
            'Open': [100 + i * 0.1 for i in range(31)],
            'High': [101 + i * 0.1 for i in range(31)],
            'Low': [99 + i * 0.1 for i in range(31)],
            'Close': [100.5 + i * 0.1 for i in range(31)],
        })
        diff_date_algo.to_csv(algos_dir / "diff_date_algo.csv", index=False)

        # Create empty file
        (algos_dir / "empty_algo.csv").write_text("datetime,open,high,low,close\n")

        # Create tiny file
        (algos_dir / "tiny_algo.csv").write_text("d")

        # Create intraday data
        intraday_algo = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='h'),
            'open': [100 + i * 0.01 for i in range(1000)],
            'high': [101 + i * 0.01 for i in range(1000)],
            'low': [99 + i * 0.01 for i in range(1000)],
            'close': [100.5 + i * 0.01 for i in range(1000)],
        })
        intraday_algo.to_csv(algos_dir / "intraday_algo.csv", index=False)

        yield Path(tmpdir)


@pytest.fixture
def loader(temp_data_dir):
    """Create DataLoader for temp directory."""
    return DataLoader(temp_data_dir)


# =============================================================================
# Tests for helper functions
# =============================================================================

class TestParseDatetimeRobust:
    """Tests for _parse_datetime_robust function."""

    def test_standard_format(self):
        """Test parsing standard datetime format."""
        series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'])
        result = _parse_datetime_robust(series)

        assert result.notna().all()
        assert result.iloc[0] == pd.Timestamp('2020-01-01')

    def test_european_format(self):
        """Test parsing dd/mm/yyyy format."""
        series = pd.Series(['01/01/2020', '15/06/2020', '31/12/2020'])
        result = _parse_datetime_robust(series)

        # May parse as mm/dd or dd/mm depending on values
        # Just check it parses without error
        assert result.notna().sum() >= 2

    def test_mixed_formats(self):
        """Test parsing with some invalid values."""
        series = pd.Series(['2020-01-01', 'invalid', '2020-01-03'])
        result = _parse_datetime_robust(series)

        assert result.notna().sum() >= 2
        assert result.iloc[0] == pd.Timestamp('2020-01-01')

    def test_timestamp_format(self):
        """Test parsing YYYYMMDD HHMMSS format."""
        series = pd.Series(['20200101 120000', '20200102 130000'])
        result = _parse_datetime_robust(series)

        # Should parse successfully
        assert result.notna().sum() >= 1


class TestDetectDatetimeColumn:
    """Tests for _detect_datetime_column function."""

    def test_datetime_column(self):
        """Test detection of 'datetime' column."""
        df = pd.DataFrame({'datetime': [], 'open': [], 'close': []})
        assert _detect_datetime_column(df) == 'datetime'

    def test_date_column(self):
        """Test detection of 'date' column."""
        df = pd.DataFrame({'date': [], 'open': [], 'close': []})
        assert _detect_datetime_column(df) == 'date'

    def test_gmt_time_column(self):
        """Test detection of 'Gmt time' column."""
        df = pd.DataFrame({'Gmt time': [], 'open': [], 'close': []})
        assert _detect_datetime_column(df) == 'Gmt time'

    def test_no_datetime_column(self):
        """Test when no datetime column exists."""
        df = pd.DataFrame({'price': [], 'volume': []})
        assert _detect_datetime_column(df) is None


class TestDetectOHLCColumns:
    """Tests for _detect_ohlc_columns function."""

    def test_lowercase_columns(self):
        """Test detection of lowercase OHLC columns."""
        df = pd.DataFrame({'open': [], 'high': [], 'low': [], 'close': []})
        col_map = _detect_ohlc_columns(df)

        assert col_map == {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
        }

    def test_uppercase_columns(self):
        """Test detection of uppercase OHLC columns."""
        df = pd.DataFrame({'OPEN': [], 'HIGH': [], 'LOW': [], 'CLOSE': []})
        col_map = _detect_ohlc_columns(df)

        assert col_map == {
            'open': 'OPEN',
            'high': 'HIGH',
            'low': 'LOW',
            'close': 'CLOSE',
        }

    def test_mixed_case_columns(self):
        """Test detection of mixed case OHLC columns."""
        df = pd.DataFrame({'Open': [], 'High': [], 'Low': [], 'Close': []})
        col_map = _detect_ohlc_columns(df)

        assert 'close' in col_map
        assert col_map['close'] == 'Close'

    def test_partial_columns(self):
        """Test detection when only some columns exist."""
        df = pd.DataFrame({'close': [], 'volume': []})
        col_map = _detect_ohlc_columns(df)

        assert col_map == {'close': 'close'}


# =============================================================================
# Tests for DataLoader
# =============================================================================

class TestDataLoaderListAlgorithms:
    """Tests for listing available algorithms."""

    def test_list_algorithms(self, loader):
        """Test listing algorithm files."""
        algos = loader.list_algorithms()

        assert 'basic_algo' in algos
        assert 'diff_date_algo' in algos
        assert 'empty_algo' in algos
        assert 'intraday_algo' in algos
        assert len(algos) >= 4

    def test_list_algorithms_empty_dir(self, temp_data_dir):
        """Test listing algorithms when directory is empty."""
        empty_dir = temp_data_dir / "empty"
        empty_dir.mkdir()
        (empty_dir / "algorithms").mkdir()

        loader = DataLoader(empty_dir)
        algos = loader.list_algorithms()

        assert algos == []


class TestDataLoaderLoadAlgorithm:
    """Tests for loading individual algorithms."""

    def test_load_algorithm_basic(self, loader):
        """Test loading a simple algorithm CSV."""
        algo = loader.load_algorithm('basic_algo')

        assert algo is not None
        assert isinstance(algo, AlgorithmData)
        assert algo.algo_id == 'basic_algo'
        assert len(algo.ohlc) == 100
        assert 'close' in algo.ohlc.columns

    def test_load_algorithm_uppercase_columns(self, loader):
        """Test loading algorithm with uppercase OHLC columns."""
        algo = loader.load_algorithm('diff_date_algo')

        assert algo is not None
        # Should have standardized column names
        assert 'close' in algo.ohlc.columns

    def test_load_algorithm_empty_file(self, loader):
        """Test handling of empty/header-only files."""
        algo = loader.load_algorithm('empty_algo', validate=True)

        assert algo is None

    def test_load_algorithm_tiny_file(self, loader):
        """Test handling of tiny files."""
        algo = loader.load_algorithm('tiny_algo', validate=True)

        assert algo is None

    def test_load_algorithm_not_found(self, loader):
        """Test loading non-existent algorithm."""
        with pytest.raises(FileNotFoundError):
            loader.load_algorithm('nonexistent_algo')

    def test_load_algorithm_resample_to_daily(self, loader):
        """Test intraday data resampling."""
        algo = loader.load_algorithm('intraday_algo', resample_to_daily=True)

        assert algo is not None
        # 1000 hours should resample to ~42 days
        assert len(algo.ohlc) < 100
        assert len(algo.ohlc) >= 30


class TestDataLoaderLoadAllAlgorithms:
    """Tests for loading multiple algorithms."""

    def test_load_all_algorithms(self, loader):
        """Test loading all available algorithms."""
        algos = loader.load_all_algorithms(show_progress=False)

        # Should load valid ones, skip invalid
        assert 'basic_algo' in algos
        assert 'intraday_algo' in algos
        # Empty and tiny should be skipped
        assert 'empty_algo' not in algos

    def test_load_all_algorithms_subset(self, loader):
        """Test loading a subset of algorithms."""
        algos = loader.load_all_algorithms(
            algo_ids=['basic_algo', 'intraday_algo'],
            show_progress=False,
        )

        assert len(algos) == 2
        assert 'basic_algo' in algos
        assert 'intraday_algo' in algos

    def test_load_all_algorithms_skip_invalid(self, loader):
        """Test skipping invalid files."""
        algos = loader.load_all_algorithms(
            algo_ids=['basic_algo', 'empty_algo', 'tiny_algo'],
            show_progress=False,
            skip_invalid=True,
        )

        assert 'basic_algo' in algos
        assert 'empty_algo' not in algos
        assert 'tiny_algo' not in algos


class TestDataLoaderInspectFormat:
    """Tests for data format inspection."""

    def test_inspect_data_format(self, loader):
        """Test inspecting data format."""
        info = loader.inspect_data_format()

        assert 'algorithms' in info
        assert info['algorithms']['n_files'] >= 4
        assert 'columns' in info['algorithms']


# =============================================================================
# Integration tests
# =============================================================================

class TestLoaderIntegration:
    """Integration tests for full loading workflow."""

    def test_load_and_verify_ohlc(self, loader):
        """Test that loaded OHLC data is valid."""
        algo = loader.load_algorithm('basic_algo')

        # Check OHLC relationships
        ohlc = algo.ohlc
        assert (ohlc['high'] >= ohlc['low']).all()
        assert (ohlc['high'] >= ohlc['open']).all()
        assert (ohlc['high'] >= ohlc['close']).all()
        assert (ohlc['low'] <= ohlc['open']).all()
        assert (ohlc['low'] <= ohlc['close']).all()

    def test_datetime_index(self, loader):
        """Test that datetime index is properly set."""
        algo = loader.load_algorithm('basic_algo')

        assert isinstance(algo.ohlc.index, pd.DatetimeIndex)
        assert algo.ohlc.index.is_monotonic_increasing
