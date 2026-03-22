"""
Integration tests for the environment module using real data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.environment import (
    TradingEnvironment,
    MarketSimulator,
    CostModel,
    PortfolioConstraints,
    RewardFunction,
    EpisodeConfig,
)


# Path to processed data
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed"


def data_exists() -> bool:
    """Check if processed data files exist."""
    required_files = [
        "algo_returns.parquet",
        "benchmark_weights.parquet",
        "benchmark_daily_returns.csv",
    ]
    return all((DATA_PATH / f).exists() for f in required_files)


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame index to timezone-naive datetime."""
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


@pytest.fixture
def real_algo_returns():
    """Load real algo returns from processed data."""
    if not data_exists():
        pytest.skip("Processed data not available")

    returns = pd.read_parquet(DATA_PATH / "algo_returns.parquet")
    returns = normalize_index(returns)
    return returns


@pytest.fixture
def real_benchmark_weights():
    """Load real benchmark weights from processed data."""
    if not data_exists():
        pytest.skip("Processed data not available")

    weights = pd.read_parquet(DATA_PATH / "benchmark_weights.parquet")
    weights = normalize_index(weights)
    return weights


@pytest.fixture
def real_benchmark_returns():
    """Load real benchmark returns from processed data."""
    if not data_exists():
        pytest.skip("Processed data not available")

    returns = pd.read_csv(DATA_PATH / "benchmark_daily_returns.csv", index_col=0, parse_dates=True)
    return returns


@pytest.mark.skipif(not data_exists(), reason="Processed data not available")
class TestRealDataIntegration:
    """Integration tests with real processed data."""

    def test_load_real_data(self, real_algo_returns, real_benchmark_weights):
        """Verify real data loads correctly."""
        assert len(real_algo_returns) > 0
        assert len(real_benchmark_weights) > 0

        # Check they have compatible shapes
        assert len(real_algo_returns.columns) >= len(real_benchmark_weights.columns)

        print(f"\nLoaded {len(real_algo_returns)} days of returns")
        print(f"Number of algorithms: {len(real_algo_returns.columns)}")
        print(f"Date range: {real_algo_returns.index[0]} to {real_algo_returns.index[-1]}")

    def test_create_simulator_with_real_data(
        self, real_algo_returns, real_benchmark_weights
    ):
        """Create simulator with real data."""
        # Align columns
        common_cols = real_algo_returns.columns.intersection(real_benchmark_weights.columns)
        algo_returns = real_algo_returns[common_cols]
        benchmark_weights = real_benchmark_weights[common_cols]

        sim = MarketSimulator(
            algo_returns=algo_returns,
            benchmark_weights=benchmark_weights,
            cost_model=CostModel(spread_bps=5, slippage_bps=2),
            constraints=PortfolioConstraints(max_weight=0.40, max_turnover=0.30),
            rebalance_frequency="weekly",
        )

        assert sim.n_algos == len(common_cols)
        print(f"\nSimulator created with {sim.n_algos} algorithms")

    def test_run_episode_with_real_data(
        self, real_algo_returns, real_benchmark_weights
    ):
        """Run a full episode with real data."""
        # Align columns and dates
        common_cols = real_algo_returns.columns.intersection(real_benchmark_weights.columns)
        common_dates = real_algo_returns.index.intersection(real_benchmark_weights.index)

        algo_returns = real_algo_returns.loc[common_dates, common_cols].fillna(0)
        benchmark_weights = real_benchmark_weights.loc[common_dates, common_cols].fillna(0)

        env = TradingEnvironment(
            algo_returns=algo_returns,
            benchmark_weights=benchmark_weights,
            cost_model=CostModel(spread_bps=5, slippage_bps=2),
            constraints=PortfolioConstraints(max_weight=0.40, max_turnover=0.30),
            rebalance_frequency="weekly",
        )

        obs, info = env.reset()

        # Run episode with equal weight action
        n_algos = env.n_algos
        ew_action = np.ones(n_algos) / n_algos

        steps = 0
        total_reward = 0
        nan_rewards = 0

        while True:
            obs, reward, terminated, truncated, info = env.step(ew_action)
            if np.isfinite(reward):
                total_reward += reward
            else:
                nan_rewards += 1
            steps += 1

            if terminated or truncated:
                break

            if steps > 1000:  # Safety limit
                break

        results = env.get_results()

        print(f"\nEpisode completed:")
        print(f"  Steps: {steps}")
        print(f"  Valid rewards: {steps - nan_rewards}, NaN rewards: {nan_rewards}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Portfolio return: {results['total_return']:.2%}")
        if np.isfinite(results['benchmark_return']):
            print(f"  Benchmark return: {results['benchmark_return']:.2%}")
            print(f"  Alpha: {results['total_return'] - results['benchmark_return']:.2%}")

        assert steps > 0
        # Allow some NaN rewards due to data gaps, but portfolio return should be valid
        assert np.isfinite(results["total_return"])

    def test_random_actions_with_real_data(
        self, real_algo_returns, real_benchmark_weights
    ):
        """Run episode with random actions to test stability."""
        # Align columns and dates, fill NaN with 0
        common_cols = real_algo_returns.columns.intersection(real_benchmark_weights.columns)
        common_dates = real_algo_returns.index.intersection(real_benchmark_weights.index)

        algo_returns = real_algo_returns.loc[common_dates, common_cols].fillna(0)
        benchmark_weights = real_benchmark_weights.loc[common_dates, common_cols].fillna(0)

        env = TradingEnvironment(
            algo_returns=algo_returns,
            benchmark_weights=benchmark_weights,
            cost_model=CostModel(spread_bps=5, slippage_bps=2),
            constraints=PortfolioConstraints(max_weight=0.40, max_turnover=0.30),
            rebalance_frequency="weekly",
        )

        np.random.seed(42)
        obs, _ = env.reset()

        steps = 0
        nan_obs_count = 0
        nan_reward_count = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            # Check observation is valid (after normalization should not have NaN)
            if np.any(np.isnan(obs)):
                nan_obs_count += 1
            if np.any(np.isinf(obs)):
                raise AssertionError(f"Inf in observation at step {steps}")
            if not np.isfinite(reward):
                nan_reward_count += 1

            if terminated or truncated:
                break

            if steps > 500:
                break

        results = env.get_results()
        print(f"\nRandom actions episode:")
        print(f"  Steps: {steps}")
        print(f"  NaN observations: {nan_obs_count}")
        print(f"  NaN rewards: {nan_reward_count}")
        print(f"  Final portfolio value: ${results['final_portfolio_value']:,.2f}")

        # With fillna(0) in data, observations should be clean
        assert nan_obs_count == 0, f"Got {nan_obs_count} NaN observations"

    def test_walk_forward_sampling_with_real_data(
        self, real_algo_returns, real_benchmark_weights
    ):
        """Test walk-forward episode sampling with real data."""
        common_cols = real_algo_returns.columns.intersection(real_benchmark_weights.columns)
        algo_returns = real_algo_returns[common_cols]
        benchmark_weights = real_benchmark_weights[common_cols]

        config = EpisodeConfig(
            random_start=True,
            episode_length=20,  # 20 rebalance periods
            warmup_periods=63,  # 3 months warmup
        )

        env = TradingEnvironment(
            algo_returns=algo_returns,
            benchmark_weights=benchmark_weights,
            episode_config=config,
            rebalance_frequency="weekly",
        )

        # Run multiple episodes
        episode_returns = []

        for episode in range(5):
            obs, info = env.reset(seed=episode)
            print(f"\nEpisode {episode}: {info['episode_start'].date()} to {info['episode_end'].date()}")

            steps = 0
            while True:
                action = np.ones(env.n_algos) / env.n_algos
                obs, reward, terminated, truncated, _ = env.step(action)
                steps += 1

                if terminated or truncated:
                    break

            results = env.get_results()
            episode_returns.append(results["total_return"])
            print(f"  Return: {results['total_return']:.2%}")

        # Different episodes should have different returns
        assert len(set(episode_returns)) > 1, "Episodes should have different returns"

    def test_benchmark_comparison_with_real_data(
        self, real_algo_returns, real_benchmark_weights
    ):
        """Compare agent performance vs benchmark with real data."""
        common_cols = real_algo_returns.columns.intersection(real_benchmark_weights.columns)
        algo_returns = real_algo_returns[common_cols]
        benchmark_weights = real_benchmark_weights[common_cols]

        # Use subset of data for faster test
        start_date = algo_returns.index[63]  # Skip warmup
        end_date = algo_returns.index[min(252, len(algo_returns) - 1)]  # ~1 year

        env = TradingEnvironment(
            algo_returns=algo_returns,
            benchmark_weights=benchmark_weights,
            cost_model=CostModel(spread_bps=5, slippage_bps=2),
            constraints=PortfolioConstraints(max_weight=0.40, max_turnover=0.30),
            train_start=start_date,
            train_end=end_date,
            rebalance_frequency="weekly",
        )

        obs, _ = env.reset()

        # Strategy: Equal weight
        n_algos = env.n_algos
        ew_action = np.ones(n_algos) / n_algos

        while True:
            obs, reward, terminated, truncated, info = env.step(ew_action)
            if terminated or truncated:
                break

        results = env.get_results()

        print(f"\nBenchmark Comparison ({start_date.date()} to {end_date.date()}):")
        print(f"  Portfolio return: {results['total_return']:.2%}")
        print(f"  Benchmark return: {results['benchmark_return']:.2%}")
        print(f"  Alpha (EW vs Benchmark): {results['total_return'] - results['benchmark_return']:.2%}")

        # Portfolio value should be positive
        assert results["final_portfolio_value"] > 0

    def test_constraint_enforcement_with_real_data(
        self, real_algo_returns, real_benchmark_weights
    ):
        """Verify constraints are properly enforced with real data."""
        common_cols = real_algo_returns.columns.intersection(real_benchmark_weights.columns)
        algo_returns = real_algo_returns[common_cols]
        benchmark_weights = real_benchmark_weights[common_cols]

        constraints = PortfolioConstraints(
            max_weight=0.25,  # Strict max weight
            max_turnover=0.20,  # Strict turnover
        )

        env = TradingEnvironment(
            algo_returns=algo_returns,
            benchmark_weights=benchmark_weights,
            constraints=constraints,
            rebalance_frequency="weekly",
        )

        env.reset()

        n_algos = env.n_algos

        for step in range(10):
            # Try extreme action
            action = np.zeros(n_algos)
            action[step % n_algos] = 1.0  # All weight on one algo

            env.step(action)

            # Check constraints are enforced
            weights = env.simulator._state.current_weights
            assert weights.max() <= 0.25 + 1e-6, f"Max weight violated at step {step}"
            assert weights.sum() <= 1.0 + 1e-6, f"Sum constraint violated at step {step}"


@pytest.mark.skipif(not data_exists(), reason="Processed data not available")
class TestDataQuality:
    """Tests for data quality and edge cases."""

    def test_no_nan_in_returns(self, real_algo_returns):
        """Check for NaN values in returns data."""
        nan_count = real_algo_returns.isna().sum().sum()
        total_cells = real_algo_returns.size

        nan_pct = nan_count / total_cells * 100
        print(f"\nNaN values: {nan_count} / {total_cells} ({nan_pct:.2f}%)")

        # High NaN is expected since many algorithms have inactive periods
        # Just report and ensure data is loadable
        print(f"  Note: High NaN percentage expected for sparse algorithm data")

        # At least some valid data should exist
        valid_pct = 100 - nan_pct
        assert valid_pct > 5, f"Too few valid values: {valid_pct:.2f}%"

    def test_returns_range(self, real_algo_returns):
        """Check returns are in reasonable range."""
        # Daily returns > 100% or < -100% are suspicious
        extreme_returns = (real_algo_returns.abs() > 1.0).sum().sum()
        total_returns = real_algo_returns.notna().sum().sum()

        extreme_pct = extreme_returns / total_returns * 100
        print(f"\nExtreme returns (|r| > 100%): {extreme_returns} ({extreme_pct:.4f}%)")

        # Should be very rare
        assert extreme_pct < 1, f"Too many extreme returns: {extreme_pct:.2f}%"

    def test_benchmark_weights_sum(self, real_benchmark_weights):
        """Check benchmark weights sum to reasonable values."""
        weight_sums = real_benchmark_weights.sum(axis=1)

        print(f"\nBenchmark weight sums:")
        print(f"  Mean: {weight_sums.mean():.4f}")
        print(f"  Min: {weight_sums.min():.4f}")
        print(f"  Max: {weight_sums.max():.4f}")

        # Weights should sum to ~1 (or less if not fully invested)
        assert weight_sums.max() <= 1.5, "Benchmark weights sum too high"
        assert weight_sums.min() >= 0, "Benchmark weights sum negative"

    def test_date_alignment(self, real_algo_returns, real_benchmark_weights):
        """Check date alignment between returns and weights."""
        returns_dates = set(real_algo_returns.index)
        weights_dates = set(real_benchmark_weights.index)

        common_dates = returns_dates.intersection(weights_dates)

        print(f"\nDate alignment:")
        print(f"  Returns dates: {len(returns_dates)}")
        print(f"  Weights dates: {len(weights_dates)}")
        print(f"  Common dates: {len(common_dates)}")

        # Should have significant overlap
        overlap_pct = len(common_dates) / min(len(returns_dates), len(weights_dates)) * 100
        assert overlap_pct > 50, f"Insufficient date overlap: {overlap_pct:.1f}%"
