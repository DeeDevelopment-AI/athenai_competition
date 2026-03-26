"""
Tests for src/analysis/algo_clusterer.py - TemporalAlgoClusterer

Tests cover:
- Temporal clustering with three horizons
- Handling of inactive/missing algorithms
- Method comparison
- Cluster stability analysis
- Cluster transitions
- Save/load functionality
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.algo_clusterer import (
    TemporalAlgoClusterer,
    TemporalClusterResult,
    TemporalClusteringOutput,
    ClusterMethod,
    ScalerType,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns_matrix():
    """Create sample returns matrix for testing."""
    np.random.seed(42)

    # 2 years of daily data
    dates = pd.date_range('2020-01-01', periods=504, freq='B')  # Business days
    n_algos = 50

    # Generate returns with different characteristics
    returns_data = {}

    # Group 1: High performers (10 algos)
    for i in range(10):
        returns_data[f'high_perf_{i}'] = np.random.normal(0.001, 0.01, len(dates))

    # Group 2: Low performers (10 algos)
    for i in range(10):
        returns_data[f'low_perf_{i}'] = np.random.normal(-0.0005, 0.015, len(dates))

    # Group 3: High volatility (10 algos)
    for i in range(10):
        returns_data[f'high_vol_{i}'] = np.random.normal(0.0002, 0.03, len(dates))

    # Group 4: Low volatility (10 algos)
    for i in range(10):
        returns_data[f'low_vol_{i}'] = np.random.normal(0.0003, 0.005, len(dates))

    # Group 5: Mixed/Inactive (10 algos) - some with gaps
    for i in range(10):
        returns = np.random.normal(0.0001, 0.012, len(dates))
        # Introduce gaps
        if i < 5:
            start_idx = np.random.randint(50, 200)
            returns[:start_idx] = np.nan
        if i >= 7:
            end_idx = np.random.randint(300, 450)
            returns[end_idx:] = np.nan
        returns_data[f'mixed_{i}'] = returns

    df = pd.DataFrame(returns_data, index=dates)
    return df


@pytest.fixture
def small_returns_matrix():
    """Create smaller returns matrix for quick tests."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    n_algos = 15

    data = {}
    for i in range(n_algos):
        data[f'algo_{i}'] = np.random.normal(0.0005, 0.01, len(dates))

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def clusterer(sample_returns_matrix):
    """Create TemporalAlgoClusterer instance."""
    return TemporalAlgoClusterer(
        returns_matrix=sample_returns_matrix,
        start_date='2020-01-01',
        n_clusters=4,
        method=ClusterMethod.KMEANS,
    )


@pytest.fixture
def small_clusterer(small_returns_matrix):
    """Create smaller clusterer for quick tests."""
    return TemporalAlgoClusterer(
        returns_matrix=small_returns_matrix,
        start_date='2020-01-01',
        n_clusters=3,
        method=ClusterMethod.KMEANS,
    )


# =============================================================================
# Tests for TemporalAlgoClusterer initialization
# =============================================================================

class TestTemporalClustererInit:
    """Tests for TemporalAlgoClusterer initialization."""

    def test_init_basic(self, sample_returns_matrix):
        """Test basic initialization."""
        clusterer = TemporalAlgoClusterer(
            returns_matrix=sample_returns_matrix,
        )

        assert clusterer.n_clusters == 5  # Default
        assert clusterer.method == ClusterMethod.KMEANS
        assert len(clusterer.week_ends) > 0
        assert clusterer.start_date == sample_returns_matrix.index.min()

    def test_init_custom_params(self, sample_returns_matrix):
        """Test initialization with custom parameters."""
        clusterer = TemporalAlgoClusterer(
            returns_matrix=sample_returns_matrix,
            start_date='2020-03-01',
            n_clusters=3,
            method=ClusterMethod.GMM,
            scaler_type=ScalerType.STANDARD,
            min_data_points=10,
        )

        assert clusterer.n_clusters == 3
        assert clusterer.method == ClusterMethod.GMM
        assert clusterer.scaler_type == ScalerType.STANDARD
        assert clusterer.min_data_points == 10

    def test_init_filters_start_date(self, sample_returns_matrix):
        """Test that data is filtered from start_date."""
        clusterer = TemporalAlgoClusterer(
            returns_matrix=sample_returns_matrix,
            start_date='2020-06-01',
        )

        assert clusterer.returns_matrix.index.min() >= pd.Timestamp('2020-06-01')

    def test_get_week_ends(self, clusterer):
        """Test week ends generation."""
        week_ends = clusterer.week_ends

        assert len(week_ends) > 0
        # All should be Fridays (or closest business day)
        for we in week_ends:
            assert isinstance(we, pd.Timestamp)


# =============================================================================
# Tests for single week clustering
# =============================================================================

class TestSingleWeekClustering:
    """Tests for _cluster_single_week method."""

    def test_cluster_single_week(self, clusterer):
        """Test clustering a single week."""
        week_end = clusterer.week_ends[10]  # Use a week in the middle
        result = clusterer._cluster_single_week(week_end)

        assert isinstance(result, TemporalClusterResult)
        assert result.week_end == week_end
        assert result.n_algos == len(clusterer.returns_matrix.columns)
        assert result.n_active > 0

    def test_cluster_single_week_has_three_horizons(self, clusterer):
        """Test that result has all three horizon clusters."""
        week_end = clusterer.week_ends[10]
        result = clusterer._cluster_single_week(week_end)

        assert isinstance(result.cluster_cumulative, pd.Series)
        assert isinstance(result.cluster_weekly, pd.Series)
        assert isinstance(result.cluster_monthly, pd.Series)

        # All should have same length
        assert len(result.cluster_cumulative) == len(clusterer.returns_matrix.columns)
        assert len(result.cluster_weekly) == len(clusterer.returns_matrix.columns)
        assert len(result.cluster_monthly) == len(clusterer.returns_matrix.columns)

    def test_cluster_single_week_has_features(self, clusterer):
        """Test that features are computed."""
        week_end = clusterer.week_ends[10]
        result = clusterer._cluster_single_week(week_end)

        for features_df in [
            result.features_cumulative,
            result.features_weekly,
            result.features_monthly,
        ]:
            assert isinstance(features_df, pd.DataFrame)
            assert len(features_df) == len(clusterer.returns_matrix.columns)

    def test_cluster_single_week_handles_inactive(self, sample_returns_matrix):
        """Test handling of inactive algorithms."""
        # Create matrix with some algos starting late
        matrix = sample_returns_matrix.copy()
        matrix.iloc[:50, :5] = np.nan  # First 5 algos inactive for first 50 days

        clusterer = TemporalAlgoClusterer(
            returns_matrix=matrix,
            start_date='2020-01-01',
            n_clusters=3,
        )

        # First week - some algos should be inactive or have insufficient data
        week_end = clusterer.week_ends[0]
        result = clusterer._cluster_single_week(week_end)

        # Check that inactive algos are marked as inactive or insufficient_data
        for i in range(5):
            algo_id = matrix.columns[i]
            assert result.cluster_cumulative[algo_id] in ['inactive', 'insufficient_data']


# =============================================================================
# Tests for feature computation
# =============================================================================

class TestFeatureComputation:
    """Tests for _compute_period_features method."""

    def test_compute_period_features_basic(self, clusterer):
        """Test basic feature computation."""
        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-03-01')

        features = clusterer._compute_period_features(start, end, 'test')

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(clusterer.returns_matrix.columns)

        # Check expected columns
        expected_cols = ['return', 'volatility', 'sharpe', 'max_drawdown',
                         'calmar_ratio', 'profit_factor']
        for col in expected_cols:
            assert col in features.columns

    def test_compute_period_features_values(self, small_returns_matrix):
        """Test that computed features have reasonable values."""
        clusterer = TemporalAlgoClusterer(
            returns_matrix=small_returns_matrix,
            start_date='2020-01-01',
            n_clusters=3,
        )

        start = pd.Timestamp('2020-01-01')
        end = pd.Timestamp('2020-04-01')
        features = clusterer._compute_period_features(start, end, 'test')

        # Volatility should be positive
        assert (features['volatility'].dropna() >= 0).all()

        # Max drawdown should be <= 0
        assert (features['max_drawdown'].dropna() <= 0).all()

        # Profit factor should be >= 0
        assert (features['profit_factor'].dropna() >= 0).all()

    def test_compute_period_features_short_period(self, clusterer):
        """Test feature computation for very short period."""
        start = pd.Timestamp('2020-01-06')
        end = pd.Timestamp('2020-01-10')

        features = clusterer._compute_period_features(start, end, 'test')

        # Some features may be NaN due to insufficient data
        assert isinstance(features, pd.DataFrame)


# =============================================================================
# Tests for full run
# =============================================================================

class TestFullRun:
    """Tests for run() method."""

    def test_run_basic(self, small_clusterer):
        """Test basic clustering run."""
        output = small_clusterer.run()

        assert isinstance(output, TemporalClusteringOutput)
        assert len(output.weekly_results) > 0
        assert isinstance(output.cluster_history, pd.DataFrame)

    def test_run_cluster_history_format(self, small_clusterer):
        """Test cluster history DataFrame format."""
        output = small_clusterer.run()

        df = output.cluster_history
        expected_cols = ['week_end', 'algo_id', 'cluster_cumulative',
                         'cluster_weekly', 'cluster_monthly']

        for col in expected_cols:
            assert col in df.columns

        # Check we have entries for all algos and weeks
        n_algos = len(small_clusterer.returns_matrix.columns)
        n_weeks = len(small_clusterer.week_ends)
        assert len(df) == n_algos * n_weeks

    def test_run_with_save(self, small_clusterer):
        """Test saving results during run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = small_clusterer.run(save_path=tmpdir)

            # Check files were created
            assert (Path(tmpdir) / 'cluster_history.parquet').exists()
            assert (Path(tmpdir) / 'cluster_history.csv').exists()
            assert (Path(tmpdir) / 'params.json').exists()

    def test_run_params_stored(self, small_clusterer):
        """Test that parameters are stored in output."""
        output = small_clusterer.run()

        assert 'start_date' in output.params
        assert 'n_clusters' in output.params
        assert 'method' in output.params
        assert output.params['n_clusters'] == small_clusterer.n_clusters


# =============================================================================
# Tests for method comparison
# =============================================================================

class TestMethodComparison:
    """Tests for compare_all_methods."""

    def test_compare_all_methods(self, small_clusterer):
        """Test method comparison."""
        methods = [ClusterMethod.KMEANS, ClusterMethod.GMM]
        comparison = small_clusterer.compare_all_methods(methods)

        assert 'cumulative' in comparison
        assert 'weekly' in comparison
        assert 'monthly' in comparison

        for horizon, df in comparison.items():
            assert isinstance(df, pd.DataFrame)
            assert 'avg_silhouette' in df.columns
            assert len(df) == len(methods)

    def test_compare_all_methods_metrics(self, small_clusterer):
        """Test that comparison returns expected metrics."""
        methods = [ClusterMethod.KMEANS]
        comparison = small_clusterer.compare_all_methods(methods)

        df = comparison['cumulative']
        assert 'avg_silhouette' in df.columns
        assert 'avg_calinski_harabasz' in df.columns
        assert 'avg_davies_bouldin' in df.columns

    def test_run_with_method_comparison(self, small_clusterer):
        """Test running with multiple methods."""
        methods = [ClusterMethod.KMEANS, ClusterMethod.HIERARCHICAL]
        output = small_clusterer.run(methods=methods)

        assert len(output.method_comparison) > 0
        assert len(output.best_methods) == 3  # One per horizon


# =============================================================================
# Tests for cluster analysis utilities
# =============================================================================

class TestClusterAnalysis:
    """Tests for cluster analysis utilities."""

    def test_get_cluster_transitions(self, small_clusterer):
        """Test cluster transition analysis."""
        output = small_clusterer.run()

        transitions = small_clusterer.get_cluster_transitions(output, 'cumulative')

        assert isinstance(transitions, pd.DataFrame)
        # Should be a square-ish matrix of transitions
        assert len(transitions) > 0

    def test_get_cluster_stability(self, small_clusterer):
        """Test cluster stability computation."""
        output = small_clusterer.run()

        stability = small_clusterer.get_cluster_stability(output, 'cumulative')

        assert isinstance(stability, pd.DataFrame)
        assert 'algo_id' in stability.columns
        assert 'stability_ratio' in stability.columns
        assert 'n_cluster_changes' in stability.columns
        assert 'dominant_cluster' in stability.columns

        # Stability ratio should be between 0 and 1
        valid_ratios = stability['stability_ratio'].dropna()
        assert (valid_ratios >= 0).all()
        assert (valid_ratios <= 1).all()


# =============================================================================
# Tests for save/load
# =============================================================================

class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_results(self, small_clusterer):
        """Test saving results."""
        output = small_clusterer.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            small_clusterer.save_results(output, tmpdir)

            # Check all files exist
            assert (Path(tmpdir) / 'cluster_history.parquet').exists()
            assert (Path(tmpdir) / 'cluster_history.csv').exists()
            assert (Path(tmpdir) / 'params.json').exists()
            assert (Path(tmpdir) / 'best_methods.json').exists()

    def test_load_results(self, small_clusterer):
        """Test loading results."""
        output = small_clusterer.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            small_clusterer.save_results(output, tmpdir)

            # Load back
            loaded = TemporalAlgoClusterer.load_results(tmpdir)

            assert isinstance(loaded, TemporalClusteringOutput)
            assert len(loaded.cluster_history) == len(output.cluster_history)
            assert loaded.params == output.params

    def test_load_results_preserves_data(self, small_clusterer):
        """Test that loaded data matches original."""
        output = small_clusterer.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            small_clusterer.save_results(output, tmpdir)
            loaded = TemporalAlgoClusterer.load_results(tmpdir)

            # Check cluster history matches
            pd.testing.assert_frame_equal(
                output.cluster_history.reset_index(drop=True),
                loaded.cluster_history.reset_index(drop=True),
            )


# =============================================================================
# Tests for different clustering methods
# =============================================================================

class TestClusteringMethods:
    """Tests for different clustering methods."""

    @pytest.mark.parametrize("method", [
        ClusterMethod.KMEANS,
        ClusterMethod.GMM,
        ClusterMethod.HIERARCHICAL,
    ])
    def test_method_works(self, small_returns_matrix, method):
        """Test that each method produces valid results."""
        clusterer = TemporalAlgoClusterer(
            returns_matrix=small_returns_matrix,
            start_date='2020-01-01',
            n_clusters=3,
            method=method,
        )

        # Run for a single week to test method
        week_end = clusterer.week_ends[5]
        result = clusterer._cluster_single_week(week_end)

        assert isinstance(result, TemporalClusterResult)
        assert result.n_active > 0

    def test_dbscan_method(self, small_returns_matrix):
        """Test DBSCAN (no fixed n_clusters)."""
        clusterer = TemporalAlgoClusterer(
            returns_matrix=small_returns_matrix,
            start_date='2020-01-01',
            n_clusters=3,  # Ignored by DBSCAN
            method=ClusterMethod.DBSCAN,
        )

        week_end = clusterer.week_ends[5]
        result = clusterer._cluster_single_week(week_end)

        assert isinstance(result, TemporalClusterResult)


# =============================================================================
# Tests for edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_period(self, sample_returns_matrix):
        """Test handling of period with no data."""
        clusterer = TemporalAlgoClusterer(
            returns_matrix=sample_returns_matrix,
            start_date='2020-01-01',
        )

        # Far future date - no data
        start = pd.Timestamp('2030-01-01')
        end = pd.Timestamp('2030-01-07')

        features = clusterer._compute_period_features(start, end, 'test')
        assert isinstance(features, pd.DataFrame)

    def test_single_algo_active(self, sample_returns_matrix):
        """Test with only one algorithm active."""
        matrix = sample_returns_matrix.copy()
        matrix.iloc[:, 1:] = np.nan  # Only first algo has data

        clusterer = TemporalAlgoClusterer(
            returns_matrix=matrix,
            start_date='2020-01-01',
            n_clusters=3,
        )

        # Should not crash, but clustering will fail gracefully
        week_end = clusterer.week_ends[5]
        result = clusterer._cluster_single_week(week_end)

        assert isinstance(result, TemporalClusterResult)

    def test_all_nan_returns(self):
        """Test handling of all NaN returns."""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        data = {f'algo_{i}': [np.nan] * 100 for i in range(10)}
        matrix = pd.DataFrame(data, index=dates)

        clusterer = TemporalAlgoClusterer(
            returns_matrix=matrix,
            start_date='2020-01-01',
            n_clusters=3,
        )

        week_end = clusterer.week_ends[5]
        result = clusterer._cluster_single_week(week_end)

        # All should be marked as inactive or insufficient_data
        for label in result.cluster_cumulative:
            assert label in ['inactive', 'insufficient_data']

    def test_few_weeks_data(self):
        """Test with very few weeks of data."""
        dates = pd.date_range('2020-01-01', periods=20, freq='B')
        data = {f'algo_{i}': np.random.normal(0, 0.01, 20) for i in range(10)}
        matrix = pd.DataFrame(data, index=dates)

        clusterer = TemporalAlgoClusterer(
            returns_matrix=matrix,
            start_date='2020-01-01',
            n_clusters=3,
        )

        output = clusterer.run()
        assert isinstance(output, TemporalClusteringOutput)


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, sample_returns_matrix):
        """Test complete workflow."""
        # Initialize
        clusterer = TemporalAlgoClusterer(
            returns_matrix=sample_returns_matrix,
            start_date='2020-01-01',
            n_clusters=4,
        )

        # Run clustering
        output = clusterer.run()

        # Analyze results
        stability = clusterer.get_cluster_stability(output, 'cumulative')
        transitions = clusterer.get_cluster_transitions(output, 'cumulative')

        # Verify output structure
        assert len(output.weekly_results) == len(clusterer.week_ends)
        assert len(stability) == len(sample_returns_matrix.columns)
        assert isinstance(transitions, pd.DataFrame)

    def test_workflow_with_comparison_and_save(self, small_returns_matrix):
        """Test workflow with method comparison and saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clusterer = TemporalAlgoClusterer(
                returns_matrix=small_returns_matrix,
                start_date='2020-01-01',
                n_clusters=3,
            )

            # Run with multiple methods
            output = clusterer.run(
                methods=[ClusterMethod.KMEANS, ClusterMethod.GMM],
                save_path=tmpdir,
            )

            # Load back
            loaded = TemporalAlgoClusterer.load_results(tmpdir)

            # Verify
            assert len(output.method_comparison) > 0
            assert len(loaded.best_methods) == 3
            assert len(loaded.cluster_history) == len(output.cluster_history)


# =============================================================================
# Performance tests (optional, skipped by default)
# =============================================================================

class TestPerformance:
    """Performance tests - skipped in normal runs."""

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        dates = pd.date_range('2018-01-01', periods=1260, freq='B')  # 5 years
        n_algos = 500

        data = {f'algo_{i}': np.random.normal(0, 0.01, len(dates))
                for i in range(n_algos)}
        matrix = pd.DataFrame(data, index=dates)

        clusterer = TemporalAlgoClusterer(
            returns_matrix=matrix,
            start_date='2018-01-01',
            n_clusters=8,
        )

        output = clusterer.run()
        assert len(output.weekly_results) > 0
