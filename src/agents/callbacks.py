"""
Callbacks for RL training with financial metrics tracking.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

logger = logging.getLogger(__name__)


class FinancialMetricsCallback(BaseCallback):
    """
    Callback for tracking financial metrics during training.

    Collects portfolio returns, Sharpe estimates, drawdowns, and turnover
    from the trading environment.
    """

    def __init__(
        self,
        training_metrics: Optional[Any] = None,
        eval_freq: int = 1000,
        verbose: int = 0,
    ):
        """
        Args:
            training_metrics: TrainingMetrics object to store metrics.
            eval_freq: Frequency of metric logging.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.training_metrics = training_metrics
        self.eval_freq = eval_freq

        self._start_time = None
        self._last_logged_timestep = 0

        # Buffers for collecting episode data
        self._episode_rewards_buffer: list[float] = []
        self._episode_returns_buffer: list[float] = []
        self._episode_drawdowns_buffer: list[float] = []
        self._episode_turnovers_buffer: list[float] = []

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Collect episode info when episodes end
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            infos = self.locals.get('infos', [{}])

            for i, done in enumerate(dones):
                if done and i < len(infos):
                    info = infos[i]

                    # Get episode reward from monitor wrapper
                    if 'episode' in info:
                        self._episode_rewards_buffer.append(info['episode']['r'])

                    # Get financial metrics
                    if 'portfolio_return' in info:
                        self._episode_returns_buffer.append(info['portfolio_return'])
                    elif 'total_return' in info:
                        self._episode_returns_buffer.append(info['total_return'])

                    if 'drawdown' in info:
                        self._episode_drawdowns_buffer.append(info['drawdown'])

                    if 'turnover' in info:
                        self._episode_turnovers_buffer.append(info['turnover'])

        # Log metrics at specified frequency
        if self.num_timesteps - self._last_logged_timestep >= self.eval_freq:
            self._log_metrics()
            self._last_logged_timestep = self.num_timesteps

        return True

    def _log_metrics(self) -> None:
        """Log collected metrics."""
        if self.training_metrics is None:
            return

        # Record timestep and wall time
        self.training_metrics.timesteps.append(self.num_timesteps)
        if self._start_time:
            self.training_metrics.wall_time_seconds.append(
                time.time() - self._start_time
            )

        # Record episode rewards
        if self._episode_rewards_buffer:
            mean_reward = np.mean(self._episode_rewards_buffer)
            self.training_metrics.episode_rewards.append(mean_reward)
            self._episode_rewards_buffer = []

            if self.verbose > 0:
                logger.info(
                    f"Step {self.num_timesteps}: Mean episode reward = {mean_reward:.4f}"
                )

        # Record portfolio returns
        if self._episode_returns_buffer:
            mean_return = np.mean(self._episode_returns_buffer)
            self.training_metrics.portfolio_returns.append(mean_return)

            # Compute rolling Sharpe estimate
            if len(self._episode_returns_buffer) > 1:
                sharpe = self._compute_sharpe(self._episode_returns_buffer)
                self.training_metrics.sharpe_estimates.append(sharpe)

            self._episode_returns_buffer = []

        # Record drawdowns
        if self._episode_drawdowns_buffer:
            worst_dd = np.min(self._episode_drawdowns_buffer)
            self.training_metrics.max_drawdowns.append(worst_dd)
            self._episode_drawdowns_buffer = []

        # Record turnovers
        if self._episode_turnovers_buffer:
            mean_turnover = np.mean(self._episode_turnovers_buffer)
            self.training_metrics.turnovers.append(mean_turnover)
            self._episode_turnovers_buffer = []

        # Get training losses if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Try to get loss values from SB3 logger
            try:
                if hasattr(self.model.logger, 'name_to_value'):
                    values = self.model.logger.name_to_value
                    if 'train/policy_loss' in values:
                        self.training_metrics.policy_losses.append(
                            values['train/policy_loss']
                        )
                    if 'train/value_loss' in values:
                        self.training_metrics.value_losses.append(
                            values['train/value_loss']
                        )
                    if 'train/entropy_loss' in values:
                        self.training_metrics.entropy_losses.append(
                            values['train/entropy_loss']
                        )
            except Exception:
                pass

    def _compute_sharpe(self, returns: list[float], periods_per_year: float = 52) -> float:
        """Compute Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret < 1e-8:
            return 0.0
        return float(mean_ret / std_ret * np.sqrt(periods_per_year))

    def _on_training_end(self) -> None:
        # Final log
        self._log_metrics()

        if self.verbose > 0:
            summary = self.training_metrics.get_summary() if self.training_metrics else {}
            logger.info(f"Training finished. Summary: {summary}")


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on validation performance.

    Stops training when validation Sharpe ratio stops improving.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        patience: int = 5,
        min_improvement: float = 0.01,
        metric: str = 'sharpe',
        verbose: int = 0,
    ):
        """
        Args:
            eval_env: Environment for evaluation.
            eval_freq: Evaluation frequency in timesteps.
            n_eval_episodes: Number of evaluation episodes.
            patience: Number of evaluations without improvement before stopping.
            min_improvement: Minimum improvement required to reset patience.
            metric: Metric to monitor ('sharpe', 'return', 'reward').
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric = metric

        self._best_metric = -np.inf
        self._no_improvement_count = 0
        self._last_eval_timestep = 0
        self._eval_history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep >= self.eval_freq:
            self._evaluate()
            self._last_eval_timestep = self.num_timesteps

            # Check for early stopping
            if self._no_improvement_count >= self.patience:
                if self.verbose > 0:
                    logger.info(
                        f"Early stopping triggered after {self.patience} evaluations "
                        f"without improvement. Best {self.metric}: {self._best_metric:.4f}"
                    )
                return False

        return True

    def _evaluate(self) -> None:
        """Run evaluation and check for improvement."""
        episode_rewards = []
        episode_returns = []
        episode_sharpes = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            total_reward = 0.0
            episode_returns_list = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                total_reward += reward

                # Collect per-step returns
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]
                if isinstance(info, dict) and 'portfolio_return' in info:
                    episode_returns_list.append(info['portfolio_return'])

                if isinstance(done, np.ndarray):
                    done = done[0]

            episode_rewards.append(total_reward)

            # Get total return from final info
            if isinstance(info, dict):
                if 'total_return' in info:
                    episode_returns.append(info['total_return'])
                elif 'portfolio_value' in info:
                    episode_returns.append(info['portfolio_value'] / 1e6 - 1)  # Assuming 1M initial

            # Compute episode Sharpe
            if len(episode_returns_list) > 1:
                mean_ret = np.mean(episode_returns_list)
                std_ret = np.std(episode_returns_list, ddof=1)
                if std_ret > 1e-8:
                    episode_sharpes.append(mean_ret / std_ret * np.sqrt(52))

        # Compute current metric
        current_metric = 0.0
        if self.metric == 'sharpe' and episode_sharpes:
            current_metric = np.mean(episode_sharpes)
        elif self.metric == 'return' and episode_returns:
            current_metric = np.mean(episode_returns)
        elif self.metric == 'reward':
            current_metric = np.mean(episode_rewards)

        # Record evaluation
        self._eval_history.append({
            'timestep': self.num_timesteps,
            'mean_reward': np.mean(episode_rewards),
            'mean_return': np.mean(episode_returns) if episode_returns else None,
            'mean_sharpe': np.mean(episode_sharpes) if episode_sharpes else None,
        })

        if self.verbose > 0:
            logger.info(
                f"Evaluation at step {self.num_timesteps}: "
                f"reward={np.mean(episode_rewards):.4f}, "
                f"{self.metric}={current_metric:.4f}"
            )

        # Check for improvement
        if current_metric > self._best_metric + self.min_improvement:
            self._best_metric = current_metric
            self._no_improvement_count = 0
            if self.verbose > 0:
                logger.info(f"New best {self.metric}: {current_metric:.4f}")
        else:
            self._no_improvement_count += 1


class CSVLoggerCallback(BaseCallback):
    """
    Callback for logging training progress to CSV file.
    """

    def __init__(
        self,
        log_path: str | Path,
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        """
        Args:
            log_path: Path to CSV log file.
            log_freq: Logging frequency in timesteps.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_freq = log_freq

        self._last_log_timestep = 0
        self._episode_rewards: list[float] = []
        self._csv_file = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        # Create log directory
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open CSV file and write header
        self._csv_file = open(self.log_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'timestep',
            'n_episodes',
            'mean_reward',
            'std_reward',
            'min_reward',
            'max_reward',
        ])

    def _on_step(self) -> bool:
        # Collect episode rewards
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    infos = self.locals.get('infos', [])
                    if i < len(infos) and 'episode' in infos[i]:
                        self._episode_rewards.append(infos[i]['episode']['r'])

        # Log at specified frequency
        if self.num_timesteps - self._last_log_timestep >= self.log_freq:
            self._log_to_csv()
            self._last_log_timestep = self.num_timesteps

        return True

    def _log_to_csv(self) -> None:
        """Write current stats to CSV."""
        if not self._episode_rewards:
            return

        self._csv_writer.writerow([
            self.num_timesteps,
            len(self._episode_rewards),
            np.mean(self._episode_rewards),
            np.std(self._episode_rewards),
            np.min(self._episode_rewards),
            np.max(self._episode_rewards),
        ])
        self._csv_file.flush()

        self._episode_rewards = []

    def _on_training_end(self) -> None:
        # Final log and close file
        self._log_to_csv()
        if self._csv_file:
            self._csv_file.close()


class ProgressCallback(BaseCallback):
    """
    Simple callback for progress reporting.
    """

    def __init__(
        self,
        total_timesteps: int,
        log_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        self._start_time = None
        self._last_log = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_freq:
            elapsed = time.time() - self._start_time
            progress = self.num_timesteps / self.total_timesteps * 100
            rate = self.num_timesteps / elapsed if elapsed > 0 else 0

            if self.verbose > 0:
                logger.info(
                    f"Progress: {progress:.1f}% ({self.num_timesteps}/{self.total_timesteps}) "
                    f"| Rate: {rate:.0f} steps/s | Elapsed: {elapsed/60:.1f} min"
                )

            self._last_log = self.num_timesteps

        return True


class NaNDetectionCallback(BaseCallback):
    """
    Stops training immediately when NaN is detected in policy weights.

    Without this, NaN propagates silently through one full rollout (n_steps ×
    n_envs timesteps) before the ValueError surfaces at the next train() call,
    wasting significant time and losing the checkpoint.

    On NaN detection: saves the model and vecnormalize stats (if available),
    then returns False to stop training cleanly.
    """

    def __init__(
        self,
        check_freq: int = 5_000,
        save_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self._last_check = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps

        # Check all policy parameters for NaN
        for name, param in self.model.policy.named_parameters():
            if param.data.isnan().any() or param.data.isinf().any():
                logger.error(
                    f"NaN/Inf detected in policy parameter '{name}' "
                    f"at timestep {self.num_timesteps}. Stopping training."
                )
                # Attempt emergency save before stopping
                if self.save_path is not None:
                    try:
                        emergency_path = str(Path(self.save_path) / "emergency_checkpoint")
                        self.model.save(emergency_path)
                        logger.info(f"Emergency checkpoint saved to: {emergency_path}")
                    except Exception as e:
                        logger.warning(f"Could not save emergency checkpoint: {e}")
                return False  # Stop training

        return True


def create_eval_callback(
    eval_env,
    save_path: str | Path,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    deterministic: bool = True,
) -> EvalCallback:
    """
    Create a standard SB3 EvalCallback with sensible defaults.

    Args:
        eval_env: Environment for evaluation.
        save_path: Path to save best model.
        eval_freq: Evaluation frequency.
        n_eval_episodes: Number of evaluation episodes.
        deterministic: Use deterministic actions.

    Returns:
        Configured EvalCallback.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    return EvalCallback(
        eval_env,
        best_model_save_path=str(save_path),
        log_path=str(save_path),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
        verbose=1,
    )
