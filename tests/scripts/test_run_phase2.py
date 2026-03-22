"""
Tests for scripts/run_phase2.py

Tests cover:
- Phase2Runner initialization and configuration
- Phase 2 pipeline with mock data
- Output file generation in organized folder structure
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_phase2 import Phase2Runner


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_phase1_data(tmp_path):
    """Create mock Phase 1 data files in organized structure."""
    np.random.seed(42)

    # Create returns matrix (252 days x 20 algos)
    n_days = 252
    n_algos = 20
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')

    returns_data = {}
    for i in range(n_algos):
        returns_data[f'algo_{i}'] = np.random.normal(0.0005, 0.01, n_days)

    returns_matrix = pd.DataFrame(returns_data, index=dates)

    # Create organized directory structure
    algorithms_dir = tmp_path / 'algorithms'
    algorithms_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = tmp_path / 'benchmark'
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Save to organized paths
    returns_matrix.to_parquet(algorithms_dir / 'returns.parquet')

    # Also save to flat paths for backward compatibility testing
    returns_matrix.to_parquet(tmp_path / 'algo_returns.parquet')

    # Create benchmark returns
    benchmark_returns = pd.DataFrame({
        'return': returns_matrix.mean(axis=1) * 0.8
    }, index=dates)
    benchmark_returns.to_csv(benchmark_dir / 'daily_returns.csv')
    benchmark_returns.to_csv(tmp_path / 'benchmark_daily_returns.csv')

    # Create benchmark weights
    weights_data = {f'algo_{i}': np.full(n_days, 1.0 / n_algos) for i in range(n_algos)}
    benchmark_weights = pd.DataFrame(weights_data, index=dates)
    benchmark_weights.to_parquet(benchmark_dir / 'weights.parquet')
    benchmark_weights.to_parquet(tmp_path / 'benchmark_weights.parquet')

    return tmp_path


@pytest.fixture
def mock_phase1_data_minimal(tmp_path):
    """Create minimal mock Phase 1 data (no weights)."""
    np.random.seed(42)

    n_days = 100
    n_algos = 10
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')

    returns_data = {}
    for i in range(n_algos):
        returns_data[f'algo_{i}'] = np.random.normal(0.0003, 0.012, n_days)

    returns_matrix = pd.DataFrame(returns_data, index=dates)

    # Create directories
    algorithms_dir = tmp_path / 'algorithms'
    algorithms_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = tmp_path / 'benchmark'
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Save returns
    returns_matrix.to_parquet(algorithms_dir / 'returns.parquet')
    returns_matrix.to_parquet(tmp_path / 'algo_returns.parquet')

    # Benchmark returns only (no weights)
    benchmark_returns = pd.DataFrame({
        'return': returns_matrix.mean(axis=1) * 0.9
    }, index=dates)
    benchmark_returns.to_csv(benchmark_dir / 'daily_returns.csv')
    benchmark_returns.to_csv(tmp_path / 'benchmark_daily_returns.csv')

    return tmp_path


# =============================================================================
# Tests for Phase2Runner
# =============================================================================

class TestPhase2Runner:
    """Tests for Phase2Runner class."""

    def test_runner_initialization(self):
        """Test that runner initializes correctly."""
        runner = Phase2Runner()

        assert runner.phase_name == "Phase 2: Analysis & Reverse Engineering"
        assert runner.phase_number == 2
        assert runner.dp is not None
        assert runner.op is not None

    def test_runner_creates_parser(self):
        """Test that runner creates argument parser with expected args."""
        runner = Phase2Runner()
        parser = runner.create_parser()

        # Parse minimal args
        args = parser.parse_args(['--dry-run'])

        assert args.dry_run is True
        assert hasattr(args, 'sample')
        assert hasattr(args, 'n_regimes')
        assert hasattr(args, 'n_families')
        assert hasattr(args, 'skip_inference')
        assert hasattr(args, 'skip_correlations')
        assert hasattr(args, 'skip_temporal')

    def test_output_dir_is_analysis(self):
        """Test that output directory is set to analysis folder."""
        runner = Phase2Runner()
        runner.args = MagicMock()
        runner.args.output_dir = None

        output_dir = runner.get_output_dir()

        assert 'analysis' in str(output_dir)


class TestPhase2Pipeline:
    """Tests for Phase 2 pipeline execution."""

    def test_run_phase2_basic(self, mock_phase1_data):
        """Test basic Phase 2 run with mock data."""
        with tempfile.TemporaryDirectory() as output_dir:
            runner = Phase2Runner()

            # Mock the data paths to use our test data
            with patch.object(runner, 'dp') as mock_dp:
                # Setup mock paths
                mock_dp.processed.root = mock_phase1_data
                mock_dp.algorithms.returns = mock_phase1_data / 'algorithms' / 'returns.parquet'
                mock_dp.benchmark.daily_returns = mock_phase1_data / 'benchmark' / 'daily_returns.csv'
                mock_dp.benchmark.weights = mock_phase1_data / 'benchmark' / 'weights.parquet'
                mock_dp.processed.analysis.root = Path(output_dir)

                # Execute with minimal options
                results = runner.execute([
                    '--output-dir', output_dir,
                    '--sample', '10',
                    '--skip-inference',
                    '--skip-correlations',
                    '--skip-temporal',
                    '--no-report',
                ])

                assert results is not None
                assert results.get('status') == 'completed'

    def test_run_phase2_creates_output_dirs(self, mock_phase1_data):
        """Test that Phase 2 creates organized output directories."""
        with tempfile.TemporaryDirectory() as output_dir:
            runner = Phase2Runner()

            with patch.object(runner, 'dp') as mock_dp:
                mock_dp.processed.root = mock_phase1_data
                mock_dp.algorithms.returns = mock_phase1_data / 'algorithms' / 'returns.parquet'
                mock_dp.benchmark.daily_returns = mock_phase1_data / 'benchmark' / 'daily_returns.csv'
                mock_dp.benchmark.weights = mock_phase1_data / 'benchmark' / 'weights.parquet'
                mock_dp.processed.analysis.root = Path(output_dir)

                runner.execute([
                    '--output-dir', output_dir,
                    '--sample', '5',
                    '--skip-inference',
                    '--skip-correlations',
                    '--skip-temporal',
                    '--no-report',
                ])

                output_path = Path(output_dir)

                # Check that organized directories were created
                assert (output_path / 'profiles').exists()
                assert (output_path / 'benchmark_profile').exists()


class TestPhase2OutputFiles:
    """Tests for Phase 2 output file generation."""

    def test_profiles_output_files(self, mock_phase1_data):
        """Test that profile output files are created."""
        with tempfile.TemporaryDirectory() as output_dir:
            runner = Phase2Runner()

            with patch.object(runner, 'dp') as mock_dp:
                mock_dp.processed.root = mock_phase1_data
                mock_dp.algorithms.returns = mock_phase1_data / 'algorithms' / 'returns.parquet'
                mock_dp.benchmark.daily_returns = mock_phase1_data / 'benchmark' / 'daily_returns.csv'
                mock_dp.benchmark.weights = mock_phase1_data / 'benchmark' / 'weights.parquet'
                mock_dp.processed.analysis.root = Path(output_dir)

                runner.execute([
                    '--output-dir', output_dir,
                    '--sample', '10',
                    '--skip-inference',
                    '--skip-correlations',
                    '--skip-temporal',
                    '--no-report',
                ])

                output_path = Path(output_dir)

                # Check profiles directory
                profiles_dir = output_path / 'profiles'
                assert (profiles_dir / 'summary.csv').exists()
                assert (profiles_dir / 'full.json').exists()

    def test_summary_csv_has_numeric_columns(self, mock_phase1_data):
        """Test that summary CSV has numeric columns for analysis."""
        with tempfile.TemporaryDirectory() as output_dir:
            runner = Phase2Runner()

            with patch.object(runner, 'dp') as mock_dp:
                mock_dp.processed.root = mock_phase1_data
                mock_dp.algorithms.returns = mock_phase1_data / 'algorithms' / 'returns.parquet'
                mock_dp.benchmark.daily_returns = mock_phase1_data / 'benchmark' / 'daily_returns.csv'
                mock_dp.benchmark.weights = mock_phase1_data / 'benchmark' / 'weights.parquet'
                mock_dp.processed.analysis.root = Path(output_dir)

                runner.execute([
                    '--output-dir', output_dir,
                    '--sample', '10',
                    '--skip-inference',
                    '--skip-correlations',
                    '--skip-temporal',
                    '--no-report',
                ])

                # Load the generated summary CSV
                summary_path = Path(output_dir) / 'profiles' / 'summary.csv'
                if summary_path.exists():
                    summary_df = pd.read_csv(summary_path, index_col=0)

                    # Verify numeric types for key columns (if they exist)
                    numeric_cols = ['sharpe', 'ann_return', 'max_dd']
                    for col in numeric_cols:
                        if col in summary_df.columns:
                            assert pd.api.types.is_numeric_dtype(summary_df[col])

    def test_profiles_json_valid(self, mock_phase1_data):
        """Test that profiles JSON is valid and contains expected fields."""
        with tempfile.TemporaryDirectory() as output_dir:
            runner = Phase2Runner()

            with patch.object(runner, 'dp') as mock_dp:
                mock_dp.processed.root = mock_phase1_data
                mock_dp.algorithms.returns = mock_phase1_data / 'algorithms' / 'returns.parquet'
                mock_dp.benchmark.daily_returns = mock_phase1_data / 'benchmark' / 'daily_returns.csv'
                mock_dp.benchmark.weights = mock_phase1_data / 'benchmark' / 'weights.parquet'
                mock_dp.processed.analysis.root = Path(output_dir)

                runner.execute([
                    '--output-dir', output_dir,
                    '--sample', '5',
                    '--skip-inference',
                    '--skip-correlations',
                    '--skip-temporal',
                    '--no-report',
                ])

                profiles_path = Path(output_dir) / 'profiles' / 'full.json'
                if profiles_path.exists():
                    with open(profiles_path) as f:
                        profiles = json.load(f)
                        assert len(profiles) >= 1


# =============================================================================
# Tests for edge cases
# =============================================================================

class TestPhase2EdgeCases:
    """Test edge cases for Phase 2."""

    def test_dry_run_no_execution(self, mock_phase1_data):
        """Test that dry run doesn't execute pipeline."""
        runner = Phase2Runner()

        results = runner.execute([
            '--dry-run',
            '--sample', '5',
        ])

        # Dry run returns empty dict
        assert results == {}

    def test_missing_returns_file_error(self, tmp_path):
        """Test error when returns file is missing."""
        runner = Phase2Runner()

        # Mock dp to point to empty tmp_path where no files exist
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()

        with patch.object(runner, 'dp') as mock_dp:
            # Point all paths to non-existent locations
            mock_dp.algorithms.returns = empty_dir / 'algorithms' / 'returns.parquet'
            mock_dp.benchmark.daily_returns = empty_dir / 'benchmark' / 'daily_returns.csv'
            mock_dp.benchmark.weights = empty_dir / 'benchmark' / 'weights.parquet'
            mock_dp.processed.root = empty_dir
            mock_dp.processed.analysis.root = empty_dir / 'analysis'

            with pytest.raises(FileNotFoundError):
                runner.execute([
                    '--output-dir', str(tmp_path),
                    '--no-report',
                ])


# =============================================================================
# Integration tests
# =============================================================================

class TestPhase2Integration:
    """Integration tests for full Phase 2 pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_with_inference(self, mock_phase1_data):
        """Test Phase 2 pipeline with regime inference enabled."""
        with tempfile.TemporaryDirectory() as output_dir:
            runner = Phase2Runner()

            with patch.object(runner, 'dp') as mock_dp:
                mock_dp.processed.root = mock_phase1_data
                mock_dp.algorithms.returns = mock_phase1_data / 'algorithms' / 'returns.parquet'
                mock_dp.benchmark.daily_returns = mock_phase1_data / 'benchmark' / 'daily_returns.csv'
                mock_dp.benchmark.weights = mock_phase1_data / 'benchmark' / 'weights.parquet'
                mock_dp.processed.analysis.root = Path(output_dir)

                results = runner.execute([
                    '--output-dir', output_dir,
                    '--sample', '10',
                    '--n-regimes', '3',
                    '--n-families', '3',
                    '--skip-correlations',
                    '--skip-temporal',
                    '--no-report',
                ])

                assert results.get('status') == 'completed'

                output_path = Path(output_dir)

                # Check regime outputs if inference ran
                regimes_dir = output_path / 'regimes'
                if regimes_dir.exists():
                    # At least the directory was created
                    pass

    @pytest.mark.slow
    def test_full_pipeline_with_correlations(self, mock_phase1_data):
        """Test Phase 2 pipeline with correlation analysis enabled."""
        with tempfile.TemporaryDirectory() as output_dir:
            runner = Phase2Runner()

            with patch.object(runner, 'dp') as mock_dp:
                mock_dp.processed.root = mock_phase1_data
                mock_dp.algorithms.returns = mock_phase1_data / 'algorithms' / 'returns.parquet'
                mock_dp.benchmark.daily_returns = mock_phase1_data / 'benchmark' / 'daily_returns.csv'
                mock_dp.benchmark.weights = mock_phase1_data / 'benchmark' / 'weights.parquet'
                mock_dp.processed.analysis.root = Path(output_dir)

                results = runner.execute([
                    '--output-dir', output_dir,
                    '--sample', '10',
                    '--skip-inference',
                    '--skip-temporal',
                    '--no-report',
                ])

                assert results.get('status') == 'completed'

                output_path = Path(output_dir)

                # Check correlation outputs
                correlation_dir = output_path / 'clustering' / 'correlation'
                if correlation_dir.exists():
                    # Matrix should exist if correlation analysis ran
                    pass
