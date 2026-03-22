"""
Base agent class defining common interface for all RL agents.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics collected during RL training."""

    # Episode-level metrics
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)

    # Financial metrics (computed at eval)
    portfolio_returns: list[float] = field(default_factory=list)
    sharpe_estimates: list[float] = field(default_factory=list)
    max_drawdowns: list[float] = field(default_factory=list)
    turnovers: list[float] = field(default_factory=list)

    # Training progress
    timesteps: list[int] = field(default_factory=list)
    wall_time_seconds: list[float] = field(default_factory=list)

    # Loss components (if available)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropy_losses: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a DataFrame for analysis."""
        # Find the longest list to determine number of records
        max_len = max(
            len(self.episode_rewards),
            len(self.timesteps),
            len(self.sharpe_estimates),
            1
        )

        def pad_list(lst: list, length: int, fill_value=np.nan) -> list:
            """Pad list to specified length."""
            return lst + [fill_value] * (length - len(lst))

        return pd.DataFrame({
            'timestep': pad_list(self.timesteps, max_len),
            'episode_reward': pad_list(self.episode_rewards, max_len),
            'episode_length': pad_list(self.episode_lengths, max_len),
            'portfolio_return': pad_list(self.portfolio_returns, max_len),
            'sharpe_estimate': pad_list(self.sharpe_estimates, max_len),
            'max_drawdown': pad_list(self.max_drawdowns, max_len),
            'turnover': pad_list(self.turnovers, max_len),
            'policy_loss': pad_list(self.policy_losses, max_len),
            'value_loss': pad_list(self.value_losses, max_len),
            'wall_time': pad_list(self.wall_time_seconds, max_len),
        })

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics of training."""
        summary = {}

        if self.episode_rewards:
            summary['mean_reward'] = np.mean(self.episode_rewards)
            summary['std_reward'] = np.std(self.episode_rewards)
            summary['max_reward'] = np.max(self.episode_rewards)
            summary['min_reward'] = np.min(self.episode_rewards)

        if self.sharpe_estimates:
            summary['mean_sharpe'] = np.mean(self.sharpe_estimates)
            summary['max_sharpe'] = np.max(self.sharpe_estimates)
            summary['final_sharpe'] = self.sharpe_estimates[-1]

        if self.portfolio_returns:
            summary['mean_return'] = np.mean(self.portfolio_returns)
            summary['total_return'] = np.sum(self.portfolio_returns)

        if self.max_drawdowns:
            summary['worst_drawdown'] = np.min(self.max_drawdowns)

        if self.timesteps:
            summary['total_timesteps'] = self.timesteps[-1]

        return summary


class BaseAgent(ABC):
    """
    Abstract base class for all RL allocation agents.

    Defines common interface that all agents must implement.
    """

    def __init__(self, env: VecEnv, seed: int = 42):
        """
        Args:
            env: Vectorized trading environment.
            seed: Random seed for reproducibility.
        """
        self.env = env
        self.seed = seed
        self._training_metrics = TrainingMetrics()
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if agent has been trained."""
        return self._is_trained

    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_path: Optional[str] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        """
        Train the agent.

        Args:
            total_timesteps: Total number of training timesteps.
            eval_env: Optional environment for evaluation.
            eval_freq: Evaluation frequency in timesteps.
            n_eval_episodes: Number of evaluation episodes.
            save_path: Path to save best model.
            callbacks: Additional callbacks.
        """
        pass

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict action (portfolio weights) for given observation.

        Args:
            observation: Current state observation.
            deterministic: Whether to use deterministic policy.

        Returns:
            Array of target portfolio weights.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save model.
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Load a trained model.

        Args:
            path: Path to load model from.
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path, env: VecEnv) -> "BaseAgent":
        """
        Load a pre-trained agent.

        Args:
            path: Path to saved model.
            env: Environment for the agent.

        Returns:
            Loaded agent instance.
        """
        pass

    def get_training_metrics(self) -> TrainingMetrics:
        """
        Get training metrics collected during training.

        Returns:
            TrainingMetrics containing episode rewards, losses, etc.
        """
        return self._training_metrics

    def reset_metrics(self) -> None:
        """Reset training metrics."""
        self._training_metrics = TrainingMetrics()

    def evaluate(
        self,
        env: VecEnv,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate agent on environment.

        Args:
            env: Environment to evaluate on.
            n_episodes: Number of evaluation episodes.
            deterministic: Whether to use deterministic policy.

        Returns:
            Dictionary of evaluation metrics.
        """
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        episode_drawdowns = []

        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            length = 0

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                length += 1

                # Handle vectorized env done
                if isinstance(done, np.ndarray):
                    done = done[0]

            episode_rewards.append(total_reward)
            episode_lengths.append(length)

            # Extract financial metrics from info if available
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            if isinstance(info, dict):
                if 'portfolio_value' in info:
                    episode_returns.append(
                        info.get('total_return', info.get('portfolio_value', 1.0) - 1.0)
                    )
                if 'drawdown' in info:
                    episode_drawdowns.append(info['drawdown'])

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'n_episodes': n_episodes,
        }

        if episode_returns:
            metrics['mean_return'] = np.mean(episode_returns)
            # Estimate Sharpe (simplified)
            if len(episode_returns) > 1 and np.std(episode_returns) > 0:
                metrics['sharpe_estimate'] = (
                    np.mean(episode_returns) / np.std(episode_returns) * np.sqrt(252 / np.mean(episode_lengths))
                )

        if episode_drawdowns:
            metrics['worst_drawdown'] = np.min(episode_drawdowns)

        return metrics


def compute_sharpe_from_rewards(
    rewards: list[float],
    periods_per_year: float = 52,  # Weekly rebalancing
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute Sharpe ratio from a list of rewards.

    Args:
        rewards: List of periodic rewards/returns.
        periods_per_year: Number of periods per year for annualization.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Annualized Sharpe ratio.
    """
    if len(rewards) < 2:
        return 0.0

    returns = np.array(rewards)
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return < 1e-8:
        return 0.0

    # Convert to annualized Sharpe
    excess_return = mean_return - risk_free_rate / periods_per_year
    sharpe = excess_return / std_return * np.sqrt(periods_per_year)

    return float(sharpe)
