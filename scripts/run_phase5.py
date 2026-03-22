#!/usr/bin/env python3
"""
=================================================================
PHASE 5: RL Agent Training
=================================================================
Complete Phase 5 pipeline to train RL meta-allocators:
  5.1 Load processed data from Phase 1
  5.2 Configure training environment with walk-forward splits
  5.3 Train RL agents (PPO, SAC, TD3)
  5.4 Evaluate on validation set
  5.5 Save checkpoints and training logs

Agents:
  - PPO: On-policy, stable (main baseline)
  - SAC: Off-policy, sample-efficient
  - TD3: Off-policy, handles noisy rewards

Usage:
  python scripts/run_phase5.py                  # Train PPO (default)
  python scripts/run_phase5.py --quick          # Quick test (10k steps, 20 algos)
  python scripts/run_phase5.py --agent sac      # Train SAC
  python scripts/run_phase5.py --agent all      # Train all agents
  python scripts/run_phase5.py --timesteps 1M   # 1 million timesteps
  python scripts/run_phase5.py --sample 50      # Use 50 algorithms

Options:
  --agent NAME         Agent to train: ppo, sac, td3, all (default: ppo)
  --quick              Quick test mode (10k steps, 20 algos)
  --timesteps N        Total training timesteps (default: 500k)
  --sample N           Use only N algorithms
  --eval-freq N        Evaluation frequency in timesteps (default: 10k)
  --n-eval-episodes N  Episodes per evaluation (default: 5)
  --rebalance-freq F   Rebalance frequency (default: weekly)
  --seed N             Random seed (default: 42)
"""

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Agent
    agent: str = "ppo"
    learning_rate: float = 3e-4
    net_arch: List[int] = field(default_factory=lambda: [256, 256])

    # Training
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    seed: int = 42

    # Environment
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "weekly"
    episode_length: int = 52  # weeks
    random_start: bool = True

    # Constraints
    max_weight: float = 0.40
    min_weight: float = 0.00
    max_turnover: float = 0.30
    max_exposure: float = 1.0

    # Costs
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    impact_coefficient: float = 0.1

    # Reward
    reward_scale: float = 100.0
    cost_penalty: float = 1.0
    turnover_penalty: float = 0.1
    drawdown_penalty: float = 0.5

    # Walk-forward
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


def parse_timesteps(value: str) -> int:
    """Parse timesteps with K/M suffix."""
    value = value.upper().strip()
    if value.endswith('K'):
        return int(float(value[:-1]) * 1_000)
    elif value.endswith('M'):
        return int(float(value[:-1]) * 1_000_000)
    else:
        return int(value)


# =============================================================================
# Phase 5 Runner
# =============================================================================

class Phase5Runner(PhaseRunner):
    """Phase 5: RL Agent Training."""

    phase_name = "Phase 5: RL Agent Training"
    phase_number = 5

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 5 specific arguments."""
        parser.add_argument(
            '--agent', type=str, default='ppo',
            choices=['ppo', 'sac', 'td3', 'all'],
            help='Agent to train (default: ppo)'
        )
        parser.add_argument(
            '--quick', '-q', action='store_true',
            help='Quick test mode (10k steps, 20 algos)'
        )
        parser.add_argument(
            '--timesteps', type=str, default='500K',
            help='Total training timesteps (e.g., 500K, 1M)'
        )
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Use only N algorithms'
        )
        parser.add_argument(
            '--eval-freq', type=int, default=10_000,
            help='Evaluation frequency in timesteps'
        )
        parser.add_argument(
            '--n-eval-episodes', type=int, default=5,
            help='Episodes per evaluation'
        )
        parser.add_argument(
            '--rebalance-freq', type=str, default='weekly',
            choices=['daily', 'weekly', 'monthly'],
            help='Rebalance frequency'
        )
        parser.add_argument(
            '--seed', type=int, default=42,
            help='Random seed'
        )
        parser.add_argument(
            '--learning-rate', type=float, default=3e-4,
            help='Learning rate'
        )
        parser.add_argument(
            '--input-dir', type=str, default=None,
            help='Override input directory (Phase 1 outputs)'
        )
        parser.add_argument(
            '--resume', type=str, default=None,
            help='Path to checkpoint to resume from'
        )

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 5 pipeline."""
        # Quick mode overrides
        if args.quick:
            args.timesteps = '10K'
            args.sample = args.sample or 20
            args.eval_freq = 2_000
            args.n_eval_episodes = 2

        # Parse timesteps
        total_timesteps = parse_timesteps(args.timesteps)

        # Build config
        config = TrainingConfig(
            agent=args.agent,
            total_timesteps=total_timesteps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            rebalance_frequency=args.rebalance_freq,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )

        results = {
            'mode': 'quick' if args.quick else 'standard',
            'config': asdict(config),
            'agents_trained': [],
        }

        # Determine paths
        input_dir = Path(args.input_dir) if args.input_dir else self.dp.processed.root
        output_dir = self.get_output_dir()

        # ==================================================================
        # STEP 5.1: LOAD DATA
        # ==================================================================
        with self.step("5.1 Load Phase 1 Data"):
            algo_returns, benchmark_weights, benchmark_returns = self._load_data(input_dir)

            # Sample if requested - only from algorithms that have benchmark weights
            if args.sample and args.sample < len(algo_returns.columns):
                np.random.seed(config.seed)
                if benchmark_weights is not None:
                    # Sample only from algorithms in both returns and benchmark
                    common_algos = [c for c in algo_returns.columns if c in benchmark_weights.columns]
                    if len(common_algos) >= args.sample:
                        sampled_cols = list(np.random.choice(common_algos, args.sample, replace=False))
                    else:
                        sampled_cols = common_algos
                        self.logger.warning(f"Only {len(common_algos)} common algorithms available")
                    benchmark_weights = benchmark_weights[sampled_cols]
                else:
                    sampled_cols = list(np.random.choice(
                        algo_returns.columns, args.sample, replace=False
                    ))
                algo_returns = algo_returns[sampled_cols]
                self.logger.info(f"Sampled {len(sampled_cols)} algorithms")

            self.logger.info(f"Returns: {algo_returns.shape[0]} days x {algo_returns.shape[1]} algos")
            results['n_algos'] = algo_returns.shape[1]
            results['n_days'] = algo_returns.shape[0]

        # ==================================================================
        # STEP 5.2: COMPUTE TRAIN/VAL/TEST SPLITS
        # ==================================================================
        with self.step("5.2 Compute Data Splits"):
            dates = algo_returns.index
            n = len(dates)

            train_end = int(n * config.train_ratio)
            val_end = int(n * (config.train_ratio + config.val_ratio))

            train_dates = (dates[0], dates[train_end - 1])
            val_dates = (dates[train_end], dates[val_end - 1])
            test_dates = (dates[val_end], dates[-1])

            self.logger.info(f"Train: {train_dates[0].date()} to {train_dates[1].date()} ({train_end} days)")
            self.logger.info(f"Val:   {val_dates[0].date()} to {val_dates[1].date()} ({val_end - train_end} days)")
            self.logger.info(f"Test:  {test_dates[0].date()} to {test_dates[1].date()} ({n - val_end} days)")

            results['splits'] = {
                'train': [str(train_dates[0].date()), str(train_dates[1].date())],
                'val': [str(val_dates[0].date()), str(val_dates[1].date())],
                'test': [str(test_dates[0].date()), str(test_dates[1].date())],
            }

        # ==================================================================
        # STEP 5.3: CREATE ENVIRONMENTS
        # ==================================================================
        with self.step("5.3 Create Training Environments"):
            from src.environment.trading_env import TradingEnvironment, EpisodeConfig, VecTradingEnv
            from src.environment.cost_model import CostModel
            from src.environment.constraints import PortfolioConstraints
            from src.environment.reward import RewardFunction, RewardType

            # Components
            cost_model = CostModel(
                spread_bps=config.spread_bps,
                slippage_bps=config.slippage_bps,
                impact_coefficient=config.impact_coefficient,
            )
            constraints = PortfolioConstraints(
                max_weight=config.max_weight,
                min_weight=config.min_weight,
                max_turnover=config.max_turnover,
                max_exposure=config.max_exposure,
            )
            reward_fn = RewardFunction(
                reward_type=RewardType.ALPHA_PENALIZED,
                cost_penalty_weight=config.cost_penalty,
                turnover_penalty_weight=config.turnover_penalty,
                drawdown_penalty_weight=config.drawdown_penalty,
            )

            episode_config = EpisodeConfig(
                random_start=config.random_start,
                episode_length=config.episode_length,
                min_episode_length=26,
                warmup_periods=4,
            )

            # Training environment (vectorized)
            train_env = VecTradingEnv(
                n_envs=4,
                algo_returns=algo_returns,
                benchmark_weights=benchmark_weights,
                train_start=train_dates[0],
                train_end=train_dates[1],
                initial_capital=config.initial_capital,
                rebalance_frequency=config.rebalance_frequency,
                cost_model=cost_model,
                constraints=constraints,
                reward_function=reward_fn,
                episode_config=episode_config,
                reward_scale=config.reward_scale,
            )

            # Evaluation environment (validation period)
            eval_episode_config = EpisodeConfig(
                random_start=False,
                episode_length=config.episode_length,
                min_episode_length=26,
            )
            eval_env = VecTradingEnv(
                n_envs=1,
                algo_returns=algo_returns,
                benchmark_weights=benchmark_weights,
                train_start=val_dates[0],
                train_end=val_dates[1],
                initial_capital=config.initial_capital,
                rebalance_frequency=config.rebalance_frequency,
                cost_model=cost_model,
                constraints=constraints,
                reward_function=reward_fn,
                episode_config=eval_episode_config,
                reward_scale=config.reward_scale,
            )

            # Extract the underlying DummyVecEnv from the wrapper
            train_vec_env = train_env.envs
            eval_vec_env = eval_env.envs

            self.logger.info(f"Training env: {train_vec_env.num_envs} parallel envs")
            self.logger.info(f"Observation shape: {train_vec_env.observation_space.shape}")
            self.logger.info(f"Action shape: {train_vec_env.action_space.shape}")

            results['env'] = {
                'observation_shape': list(train_vec_env.observation_space.shape),
                'action_shape': list(train_vec_env.action_space.shape),
                'n_parallel_envs': train_vec_env.num_envs,
            }

        # ==================================================================
        # STEP 5.4: TRAIN AGENTS
        # ==================================================================
        agents_to_train = ['ppo', 'sac', 'td3'] if args.agent == 'all' else [args.agent]

        for agent_name in agents_to_train:
            with self.step(f"5.4 Train {agent_name.upper()} Agent"):
                agent_results = self._train_agent(
                    agent_name=agent_name,
                    train_env=train_vec_env,
                    eval_env=eval_vec_env,
                    config=config,
                    output_dir=output_dir,
                    resume_path=args.resume if agent_name == args.agent else None,
                )
                results['agents_trained'].append(agent_results)

            # Cleanup between agents
            gc.collect()

        # ==================================================================
        # STEP 5.5: GENERATE REPORT
        # ==================================================================
        with self.step("5.5 Generate Report"):
            self._generate_report(results, output_dir)

        return results

    def _train_agent(
        self,
        agent_name: str,
        train_env,
        eval_env,
        config: TrainingConfig,
        output_dir: Path,
        resume_path: Optional[str] = None,
    ) -> dict:
        """Train a single agent."""
        from src.agents.ppo_agent import PPOAllocator
        from src.agents.sac_agent import SACAllocator
        from src.agents.td3_agent import TD3Allocator

        agent_dir = output_dir / "checkpoints" / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs" / agent_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create agent
        if agent_name == 'ppo':
            agent = PPOAllocator(
                env=train_env,
                learning_rate=config.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                net_arch=config.net_arch,
                seed=config.seed,
            )
        elif agent_name == 'sac':
            agent = SACAllocator(
                env=train_env,
                learning_rate=config.learning_rate,
                buffer_size=100_000,
                batch_size=256,
                tau=0.005,
                ent_coef='auto',
                net_arch=config.net_arch,
                seed=config.seed,
            )
        elif agent_name == 'td3':
            agent = TD3Allocator(
                env=train_env,
                learning_rate=config.learning_rate,
                buffer_size=100_000,
                batch_size=256,
                tau=0.005,
                policy_delay=2,
                net_arch=config.net_arch,
                seed=config.seed,
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        self.logger.info(f"Created {agent_name.upper()} agent")
        self.logger.info(f"Hyperparameters: {agent.get_hyperparameters()}")

        # Resume if checkpoint provided
        if resume_path:
            agent.load(resume_path)
            self.logger.info(f"Resumed from: {resume_path}")

        # Train
        self.logger.info(f"Training for {config.total_timesteps:,} timesteps...")
        agent.train(
            total_timesteps=config.total_timesteps,
            eval_env=eval_env,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            save_path=str(agent_dir),
            log_to_csv=True,
            progress_bar=False,  # Avoid tqdm/rich dependency
        )

        # Save final model
        final_path = agent_dir / "final_model"
        agent.save(final_path)
        self.logger.info(f"Saved final model to: {final_path}")

        # Get training metrics
        metrics = agent.get_training_metrics()
        metrics_df = metrics.to_dataframe()

        if not metrics_df.empty:
            metrics_df.to_csv(logs_dir / "metrics.csv", index=False)

            # Summary stats
            summary = metrics.get_summary()
            self.logger.info(f"Training summary:")
            self.logger.info(f"  Mean reward: {summary.get('mean_reward', 0):.2f}")
            self.logger.info(f"  Final Sharpe: {summary.get('final_sharpe', 0):.3f}")
            self.logger.info(f"  Max drawdown: {summary.get('max_drawdown', 0):.2%}")
        else:
            summary = {}

        # Evaluate on validation set
        self.logger.info("Evaluating on validation set...")
        eval_metrics = agent.evaluate(eval_env, n_episodes=config.n_eval_episodes)
        self.logger.info(f"Validation mean reward: {eval_metrics['mean_reward']:.2f}")
        self.logger.info(f"Validation mean return: {eval_metrics.get('mean_portfolio_return', 0):.2%}")

        return {
            'agent': agent_name,
            'timesteps': config.total_timesteps,
            'model_path': str(final_path),
            'training_summary': summary,
            'validation': {
                'mean_reward': eval_metrics['mean_reward'],
                'std_reward': eval_metrics['std_reward'],
                'mean_return': eval_metrics.get('mean_portfolio_return', 0),
            }
        }

    def _load_data(self, input_dir: Path):
        """Load Phase 1 data."""
        # Load returns matrix - try new path first, then legacy
        returns_path = self.dp.algorithms.returns
        if not returns_path.exists():
            returns_path = input_dir / 'algo_returns.parquet'
        if not returns_path.exists():
            raise FileNotFoundError(f"Returns matrix not found: {returns_path}")

        algo_returns = pd.read_parquet(returns_path)
        algo_returns = algo_returns.astype(np.float32, copy=False)
        # Normalize timezone (remove if present)
        if algo_returns.index.tz is not None:
            algo_returns.index = algo_returns.index.tz_localize(None)
        self.logger.info(f"Loaded returns: {algo_returns.shape}")

        # Load benchmark weights
        weights_path = self.dp.benchmark.weights
        if not weights_path.exists():
            weights_path = input_dir / 'benchmark_weights.parquet'

        benchmark_weights = None
        if weights_path.exists():
            benchmark_weights = pd.read_parquet(weights_path)
            benchmark_weights = benchmark_weights.astype(np.float32, copy=False)
            # Normalize timezone (remove if present)
            if benchmark_weights.index.tz is not None:
                benchmark_weights.index = benchmark_weights.index.tz_localize(None)
            self.logger.info(f"Loaded benchmark weights: {benchmark_weights.shape}")

        # Load benchmark returns
        bench_path = self.dp.benchmark.daily_returns
        if not bench_path.exists():
            bench_path = input_dir / 'benchmark_daily_returns.csv'

        benchmark_returns = None
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path, index_col=0, parse_dates=True)
            if 'return' in bench_df.columns:
                benchmark_returns = bench_df['return']
            else:
                benchmark_returns = bench_df.iloc[:, 0]
            self.logger.info(f"Loaded benchmark returns: {len(benchmark_returns)} days")

        return algo_returns, benchmark_weights, benchmark_returns

    def _generate_report(self, results: dict, output_dir: Path):
        """Generate markdown report."""
        config = results['config']

        report = f"""# Phase 5: RL Agent Training Report

**Generated**: {datetime.now().isoformat()}
**Mode**: {results['mode']}

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Timesteps | {config['total_timesteps']:,} |
| Learning Rate | {config['learning_rate']} |
| Network Architecture | {config['net_arch']} |
| Rebalance Frequency | {config['rebalance_frequency']} |
| Episode Length | {config['episode_length']} |
| Seed | {config['seed']} |

## Data Summary

| Metric | Value |
|--------|-------|
| Algorithms | {results['n_algos']} |
| Trading Days | {results['n_days']} |

## Train/Val/Test Splits

| Split | Start | End |
|-------|-------|-----|
| Train | {results['splits']['train'][0]} | {results['splits']['train'][1]} |
| Val | {results['splits']['val'][0]} | {results['splits']['val'][1]} |
| Test | {results['splits']['test'][0]} | {results['splits']['test'][1]} |

## Agents Trained

"""
        for agent_result in results['agents_trained']:
            report += f"""### {agent_result['agent'].upper()}

| Metric | Value |
|--------|-------|
| Timesteps | {agent_result['timesteps']:,} |
| Model Path | `{agent_result['model_path']}` |
| Val Mean Reward | {agent_result['validation']['mean_reward']:.2f} |
| Val Std Reward | {agent_result['validation']['std_reward']:.2f} |
| Val Mean Return | {agent_result['validation']['mean_return']:.2%} |

"""
            if agent_result.get('training_summary'):
                summary = agent_result['training_summary']
                report += f"""**Training Summary:**
- Mean Reward: {summary.get('mean_reward', 'N/A')}
- Final Sharpe: {summary.get('final_sharpe', 'N/A')}
- Max Drawdown: {summary.get('max_drawdown', 'N/A')}

"""

        report += """## Next Steps

1. Run Phase 6 (Evaluation) to compare against baselines
2. Perform walk-forward validation
3. Analyze training curves in `outputs/rl_training/logs/`

## Files Generated

- `checkpoints/*/final_model.zip` - Trained model
- `logs/*/training_log.csv` - Training logs
- `logs/*/metrics.csv` - Training metrics
"""

        # Save report
        report_path = output_dir / "PHASE5_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        self.logger.info(f"Report saved to: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    runner = Phase5Runner()
    runner.execute()


if __name__ == "__main__":
    main()
