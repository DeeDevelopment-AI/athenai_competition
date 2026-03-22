"""
Tests for RL agents (PPO, SAC, TD3).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

# Import agents
from src.agents import (
    BaseAgent,
    PPOAllocator,
    SACAllocator,
    TD3Allocator,
    TrainingMetrics,
    compute_sharpe_from_rewards,
)
from src.agents.callbacks import (
    CSVLoggerCallback,
    FinancialMetricsCallback,
    ProgressCallback,
)
from src.agents.offline_rl import OfflineDataset, OfflineDatasetBuilder


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_returns():
    """Create sample algorithm returns DataFrame."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    n_algos = 5

    # Generate random returns with some correlation
    returns = np.random.randn(len(dates), n_algos) * 0.01  # ~1% daily vol
    returns = pd.DataFrame(
        returns,
        index=dates,
        columns=[f'algo_{i}' for i in range(n_algos)]
    )
    return returns


@pytest.fixture
def sample_benchmark_weights(sample_returns):
    """Create sample benchmark weights DataFrame."""
    dates = sample_returns.index
    n_algos = len(sample_returns.columns)

    # Equal weight with some noise
    weights = np.ones((len(dates), n_algos)) / n_algos
    weights += np.random.randn(len(dates), n_algos) * 0.02
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum(axis=1, keepdims=True)

    return pd.DataFrame(
        weights,
        index=dates,
        columns=sample_returns.columns
    )


@pytest.fixture
def sample_benchmark_returns(sample_returns, sample_benchmark_weights):
    """Create sample benchmark returns Series."""
    returns = (sample_returns * sample_benchmark_weights).sum(axis=1)
    return returns


@pytest.fixture
def mock_env(sample_returns):
    """Create a mock trading environment."""
    from src.environment import TradingEnvironment

    env = TradingEnvironment(
        algo_returns=sample_returns,
        initial_capital=1_000_000.0,
        rebalance_frequency='weekly',
    )

    # Wrap in DummyVecEnv for SB3 compatibility
    vec_env = DummyVecEnv([lambda: env])
    return vec_env


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==============================================================================
# Base Agent Tests
# ==============================================================================


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_empty_metrics(self):
        """Test empty metrics initialization."""
        metrics = TrainingMetrics()
        assert len(metrics.episode_rewards) == 0
        assert len(metrics.timesteps) == 0

    def test_append_metrics(self):
        """Test appending metrics."""
        metrics = TrainingMetrics()
        metrics.episode_rewards.append(100.0)
        metrics.episode_rewards.append(150.0)
        metrics.timesteps.append(1000)
        metrics.timesteps.append(2000)

        assert len(metrics.episode_rewards) == 2
        assert len(metrics.timesteps) == 2

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        metrics = TrainingMetrics()
        metrics.episode_rewards = [100.0, 150.0, 200.0]
        metrics.timesteps = [1000, 2000, 3000]
        metrics.sharpe_estimates = [1.0, 1.5]  # Different length

        df = metrics.to_dataframe()
        assert len(df) == 3  # Max length
        assert 'timestep' in df.columns
        assert 'episode_reward' in df.columns
        assert 'sharpe_estimate' in df.columns

    def test_get_summary(self):
        """Test summary statistics."""
        metrics = TrainingMetrics()
        metrics.episode_rewards = [100.0, 150.0, 200.0]
        metrics.sharpe_estimates = [1.0, 1.5, 2.0]
        metrics.timesteps = [1000, 2000, 3000]

        summary = metrics.get_summary()
        assert 'mean_reward' in summary
        assert 'max_reward' in summary
        assert 'mean_sharpe' in summary
        assert summary['mean_reward'] == 150.0
        assert summary['max_reward'] == 200.0


class TestComputeSharpe:
    """Tests for Sharpe ratio computation."""

    def test_basic_sharpe(self):
        """Test basic Sharpe calculation."""
        rewards = [0.01, 0.02, 0.015, 0.01, 0.02]
        sharpe = compute_sharpe_from_rewards(rewards, periods_per_year=52)
        assert sharpe > 0  # Positive returns should give positive Sharpe

    def test_negative_sharpe(self):
        """Test negative Sharpe."""
        rewards = [-0.01, -0.02, -0.015, -0.01, -0.02]
        sharpe = compute_sharpe_from_rewards(rewards, periods_per_year=52)
        assert sharpe < 0

    def test_zero_volatility(self):
        """Test handling of zero volatility."""
        rewards = [0.01, 0.01, 0.01, 0.01]  # Constant returns
        sharpe = compute_sharpe_from_rewards(rewards, periods_per_year=52)
        assert sharpe == 0.0  # Should return 0 for zero volatility

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        rewards = [0.01]
        sharpe = compute_sharpe_from_rewards(rewards)
        assert sharpe == 0.0


# ==============================================================================
# PPO Agent Tests
# ==============================================================================


class TestPPOAllocator:
    """Tests for PPO agent."""

    def test_initialization(self, mock_env):
        """Test PPO agent initialization."""
        agent = PPOAllocator(
            env=mock_env,
            learning_rate=1e-4,
            n_steps=64,
            batch_size=32,
            verbose=0,
        )
        assert agent is not None
        assert agent.model is not None
        assert not agent.is_trained

    def test_predict_before_training(self, mock_env):
        """Test prediction before training (should work with random policy)."""
        agent = PPOAllocator(env=mock_env, verbose=0)
        obs = mock_env.reset()
        action = agent.predict(obs)

        assert action is not None
        assert action.shape[1] == mock_env.action_space.shape[0]

    def test_short_training(self, mock_env, temp_dir):
        """Test short training run."""
        agent = PPOAllocator(
            env=mock_env,
            learning_rate=1e-3,
            n_steps=64,
            batch_size=32,
            verbose=0,
        )

        agent.train(
            total_timesteps=128,
            save_path=str(temp_dir / "ppo"),
            log_to_csv=False,
            progress_bar=False,
        )

        assert agent.is_trained

    def test_save_load(self, mock_env, temp_dir):
        """Test model saving and loading."""
        agent = PPOAllocator(env=mock_env, verbose=0)

        # Train briefly
        agent.train(total_timesteps=64, log_to_csv=False, progress_bar=False)

        # Save
        save_path = temp_dir / "ppo_model"
        agent.save(save_path)
        assert (save_path.with_suffix('.zip')).exists()

        # Load
        agent2 = PPOAllocator(env=mock_env, verbose=0)
        agent2.load(save_path)
        assert agent2.is_trained

    def test_from_pretrained(self, mock_env, temp_dir):
        """Test loading pretrained model."""
        agent = PPOAllocator(env=mock_env, verbose=0)
        agent.train(total_timesteps=64, log_to_csv=False, progress_bar=False)

        save_path = temp_dir / "ppo_model"
        agent.save(save_path)

        # Load with class method
        loaded = PPOAllocator.from_pretrained(save_path, mock_env)
        assert loaded.is_trained

        # Verify prediction works
        obs = mock_env.reset()
        action = loaded.predict(obs)
        assert action is not None

    def test_get_hyperparameters(self, mock_env):
        """Test getting hyperparameters."""
        agent = PPOAllocator(
            env=mock_env,
            learning_rate=1e-4,
            gamma=0.95,
            verbose=0,
        )
        params = agent.get_hyperparameters()

        assert params['algorithm'] == 'PPO'
        assert params['learning_rate'] == 1e-4
        assert params['gamma'] == 0.95

    def test_training_metrics(self, mock_env):
        """Test training metrics collection."""
        agent = PPOAllocator(env=mock_env, verbose=0)
        agent.train(total_timesteps=128, log_to_csv=False, progress_bar=False)

        metrics = agent.get_training_metrics()
        assert isinstance(metrics, TrainingMetrics)


# ==============================================================================
# SAC Agent Tests
# ==============================================================================


class TestSACAllocator:
    """Tests for SAC agent."""

    def test_initialization(self, mock_env):
        """Test SAC agent initialization."""
        agent = SACAllocator(
            env=mock_env,
            learning_rate=1e-4,
            buffer_size=1000,
            verbose=0,
        )
        assert agent is not None
        assert agent.model is not None
        assert not agent.is_trained

    def test_predict_before_training(self, mock_env):
        """Test prediction before training."""
        agent = SACAllocator(env=mock_env, verbose=0)
        obs = mock_env.reset()
        action = agent.predict(obs)

        assert action is not None
        assert action.shape[1] == mock_env.action_space.shape[0]

    def test_short_training(self, mock_env, temp_dir):
        """Test short training run."""
        agent = SACAllocator(
            env=mock_env,
            learning_rate=1e-3,
            buffer_size=500,
            learning_starts=50,
            verbose=0,
        )

        agent.train(
            total_timesteps=100,
            save_path=str(temp_dir / "sac"),
            log_to_csv=False,
            progress_bar=False,
        )

        assert agent.is_trained

    def test_save_load(self, mock_env, temp_dir):
        """Test model saving and loading."""
        agent = SACAllocator(
            env=mock_env,
            buffer_size=500,
            learning_starts=50,
            verbose=0,
        )
        agent.train(total_timesteps=100, log_to_csv=False, progress_bar=False)

        save_path = temp_dir / "sac_model"
        agent.save(save_path)

        agent2 = SACAllocator(
            env=mock_env,
            buffer_size=500,
            verbose=0,
        )
        agent2.load(save_path)
        assert agent2.is_trained

    def test_get_hyperparameters(self, mock_env):
        """Test getting hyperparameters."""
        agent = SACAllocator(
            env=mock_env,
            learning_rate=1e-4,
            tau=0.01,
            verbose=0,
        )
        params = agent.get_hyperparameters()

        assert params['algorithm'] == 'SAC'
        assert params['learning_rate'] == 1e-4
        assert params['tau'] == 0.01


# ==============================================================================
# TD3 Agent Tests
# ==============================================================================


class TestTD3Allocator:
    """Tests for TD3 agent."""

    def test_initialization(self, mock_env):
        """Test TD3 agent initialization."""
        agent = TD3Allocator(
            env=mock_env,
            learning_rate=1e-4,
            buffer_size=1000,
            verbose=0,
        )
        assert agent is not None
        assert agent.model is not None
        assert not agent.is_trained

    def test_predict_before_training(self, mock_env):
        """Test prediction before training."""
        agent = TD3Allocator(env=mock_env, verbose=0)
        obs = mock_env.reset()
        action = agent.predict(obs)

        assert action is not None
        assert action.shape[1] == mock_env.action_space.shape[0]

    def test_short_training(self, mock_env, temp_dir):
        """Test short training run."""
        agent = TD3Allocator(
            env=mock_env,
            learning_rate=1e-3,
            buffer_size=500,
            learning_starts=50,
            verbose=0,
        )

        agent.train(
            total_timesteps=100,
            save_path=str(temp_dir / "td3"),
            log_to_csv=False,
            progress_bar=False,
        )

        assert agent.is_trained

    def test_action_noise_types(self, mock_env):
        """Test different action noise types."""
        # Normal noise
        agent1 = TD3Allocator(
            env=mock_env,
            action_noise_type="normal",
            action_noise_std=0.1,
            verbose=0,
        )
        assert agent1.action_noise_type == "normal"

        # OU noise
        agent2 = TD3Allocator(
            env=mock_env,
            action_noise_type="ou",
            action_noise_std=0.1,
            verbose=0,
        )
        assert agent2.action_noise_type == "ou"

    def test_get_hyperparameters(self, mock_env):
        """Test getting hyperparameters."""
        agent = TD3Allocator(
            env=mock_env,
            learning_rate=1e-4,
            policy_delay=3,
            verbose=0,
        )
        params = agent.get_hyperparameters()

        assert params['algorithm'] == 'TD3'
        assert params['learning_rate'] == 1e-4
        assert params['policy_delay'] == 3


# ==============================================================================
# Offline RL Tests
# ==============================================================================


class TestOfflineDataset:
    """Tests for OfflineDataset."""

    def test_create_dataset(self):
        """Test creating offline dataset."""
        n = 100
        obs_dim = 10
        action_dim = 5

        dataset = OfflineDataset(
            observations=np.random.randn(n, obs_dim).astype(np.float32),
            actions=np.random.rand(n, action_dim).astype(np.float32),
            rewards=np.random.randn(n).astype(np.float32),
            next_observations=np.random.randn(n, obs_dim).astype(np.float32),
            terminals=np.zeros(n, dtype=np.float32),
        )

        assert len(dataset) == n
        assert dataset.observations.shape == (n, obs_dim)
        assert dataset.actions.shape == (n, action_dim)

    def test_save_load(self, temp_dir):
        """Test saving and loading dataset."""
        n = 50
        dataset = OfflineDataset(
            observations=np.random.randn(n, 10).astype(np.float32),
            actions=np.random.rand(n, 5).astype(np.float32),
            rewards=np.random.randn(n).astype(np.float32),
            next_observations=np.random.randn(n, 10).astype(np.float32),
            terminals=np.zeros(n, dtype=np.float32),
        )

        path = temp_dir / "dataset.npz"
        dataset.save(path)

        loaded = OfflineDataset.load(path)
        assert len(loaded) == len(dataset)
        np.testing.assert_array_almost_equal(
            loaded.observations, dataset.observations
        )

    def test_get_statistics(self):
        """Test dataset statistics."""
        dataset = OfflineDataset(
            observations=np.ones((100, 10), dtype=np.float32),
            actions=np.ones((100, 5), dtype=np.float32),
            rewards=np.array([1.0, -1.0] * 50, dtype=np.float32),
            next_observations=np.ones((100, 10), dtype=np.float32),
            terminals=np.array([0] * 99 + [1], dtype=np.float32),
        )

        stats = dataset.get_statistics()
        assert stats['n_transitions'] == 100
        assert stats['obs_dim'] == 10
        assert stats['action_dim'] == 5
        assert stats['reward_mean'] == 0.0
        assert abs(stats['terminal_rate'] - 0.01) < 1e-6  # Float comparison


class TestOfflineDatasetBuilder:
    """Tests for OfflineDatasetBuilder."""

    def test_build_dataset(self, sample_returns, sample_benchmark_weights, sample_benchmark_returns):
        """Test building dataset from benchmark data."""
        builder = OfflineDatasetBuilder(
            algo_returns=sample_returns,
            benchmark_weights=sample_benchmark_weights,
            benchmark_returns=sample_benchmark_returns,
        )

        dataset = builder.build_dataset(lookback=21)

        assert len(dataset) > 0
        # Check dimensions
        n_algos = len(sample_returns.columns)
        assert dataset.actions.shape[1] == n_algos

    def test_insufficient_data(self):
        """Test error with insufficient data."""
        dates = pd.date_range('2020-01-01', periods=10, freq='B')
        returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01,
            index=dates,
            columns=['a', 'b', 'c']
        )
        weights = pd.DataFrame(
            np.ones((10, 3)) / 3,
            index=dates,
            columns=['a', 'b', 'c']
        )
        bench_returns = returns.mean(axis=1)

        builder = OfflineDatasetBuilder(
            algo_returns=returns,
            benchmark_weights=weights,
            benchmark_returns=bench_returns,
        )

        with pytest.raises(ValueError, match="Not enough data"):
            builder.build_dataset(lookback=21)


# ==============================================================================
# Callback Tests
# ==============================================================================


class TestCallbacks:
    """Tests for training callbacks."""

    def test_financial_metrics_callback(self):
        """Test financial metrics callback initialization."""
        metrics = TrainingMetrics()
        callback = FinancialMetricsCallback(
            training_metrics=metrics,
            eval_freq=100,
            verbose=0,
        )
        assert callback is not None

    def test_csv_logger_callback(self, temp_dir):
        """Test CSV logger callback initialization."""
        callback = CSVLoggerCallback(
            log_path=temp_dir / "log.csv",
            log_freq=100,
            verbose=0,
        )
        assert callback is not None

    def test_progress_callback(self):
        """Test progress callback initialization."""
        callback = ProgressCallback(
            total_timesteps=10000,
            log_freq=1000,
            verbose=0,
        )
        assert callback is not None


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestAgentIntegration:
    """Integration tests for agents with environment."""

    def test_all_agents_same_env(self, mock_env, temp_dir):
        """Test that all agents work with the same environment."""
        agents = [
            ('PPO', PPOAllocator(env=mock_env, n_steps=64, batch_size=32, verbose=0)),
            ('SAC', SACAllocator(env=mock_env, buffer_size=500, learning_starts=50, verbose=0)),
            ('TD3', TD3Allocator(env=mock_env, buffer_size=500, learning_starts=50, verbose=0)),
        ]

        for name, agent in agents:
            # Test prediction
            obs = mock_env.reset()
            action = agent.predict(obs)
            assert action is not None, f"{name} prediction failed"

            # Test action shape
            assert action.shape[1] == mock_env.action_space.shape[0], \
                f"{name} action shape mismatch"

            # Test action bounds (after normalization in env, should be in [0, 1])
            assert np.all(action >= 0), f"{name} action below 0"
            assert np.all(action <= 1), f"{name} action above 1"

    def test_deterministic_vs_stochastic(self, mock_env):
        """Test deterministic vs stochastic predictions."""
        agent = PPOAllocator(env=mock_env, verbose=0)
        agent.train(total_timesteps=64, log_to_csv=False, progress_bar=False)

        obs = mock_env.reset()

        # Deterministic should be consistent
        action1 = agent.predict(obs, deterministic=True)
        action2 = agent.predict(obs, deterministic=True)
        np.testing.assert_array_almost_equal(action1, action2)

        # Stochastic may vary (though with same obs they might still be close)
        # Just check it doesn't crash
        action3 = agent.predict(obs, deterministic=False)
        assert action3 is not None
