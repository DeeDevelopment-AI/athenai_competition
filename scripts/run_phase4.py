#!/usr/bin/env python3
"""
=================================================================
PHASE 4: Environment Validation & Testing
=================================================================
Complete Phase 4 pipeline to validate the RL environment:
  4.1 Load processed data from Phase 1
  4.2 Initialize and validate MarketSimulator
  4.3 Test TradingEnvironment (Gymnasium wrapper)
  4.4 Validate constraints, costs, and reward functions
  4.5 Run sanity checks with random/constant policies
  4.6 Generate environment specification report

This phase ensures the RL environment is correctly configured before
training agents in Phase 5.

Usage:
  python scripts/run_phase4.py                  # Standard validation
  python scripts/run_phase4.py --sample 50      # Use 50 algorithms
  python scripts/run_phase4.py --episodes 10    # Run 10 test episodes
  python scripts/run_phase4.py --full           # Full validation + stress tests
  python scripts/run_phase4.py --dry-run        # Show configuration only

Options:
  --sample N           Use only N algorithms (for faster testing)
  --episodes N         Number of test episodes (default: 5)
  --full               Full validation with stress tests
  --rebalance-freq F   Rebalance frequency: daily, weekly, monthly (default: weekly)
"""

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnvConfig:
    """Environment configuration for testing."""
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "weekly"
    max_weight: float = 0.40
    min_weight: float = 0.00
    max_turnover: float = 0.30
    max_exposure: float = 1.0
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    impact_coefficient: float = 0.1
    reward_scale: float = 100.0
    episode_length: int = 52  # weeks


# =============================================================================
# Phase 4 Runner
# =============================================================================

class Phase4Runner(PhaseRunner):
    """Phase 4: Environment Validation & Testing."""

    phase_name = "Phase 4: Environment Validation"
    phase_number = 4

    def _run_tag(self) -> str:
        freq = getattr(self.args, 'rebalance_freq', 'weekly')
        mode = "full" if getattr(self.args, 'full', False) else "std"
        return f"{freq}_{mode}"

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 4 specific arguments."""
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Use only N algorithms (for faster testing)'
        )
        parser.add_argument(
            '--episodes', type=int, default=5,
            help='Number of test episodes to run'
        )
        parser.add_argument(
            '--full', '-f', action='store_true',
            help='Full validation with stress tests'
        )
        parser.add_argument(
            '--rebalance-freq', type=str, default='weekly',
            choices=['daily', 'weekly', 'monthly', 'quarterly'],
            help='Rebalance frequency'
        )
        parser.add_argument(
            '--input-dir', type=str, default=None,
            help='Override input directory (Phase 1 outputs)'
        )

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 4 pipeline."""
        results = {
            'mode': 'full' if args.full else 'standard',
            'n_episodes': args.episodes,
            'sample_size': args.sample,
        }

        # Configuration
        config = EnvConfig(rebalance_frequency=args.rebalance_freq)
        results['config'] = asdict(config)

        # Determine paths
        input_dir = Path(args.input_dir) if args.input_dir else self.dp.processed.root
        output_dir = self.get_output_dir()

        # ==================================================================
        # STEP 4.1: LOAD DATA
        # ==================================================================
        with self.step("4.1 Load Phase 1 Data"):
            algo_returns, benchmark_weights, benchmark_returns = self._load_data(input_dir)

            # Sample if requested - only from algorithms that have benchmark weights
            if args.sample and args.sample < len(algo_returns.columns):
                np.random.seed(42)
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
        # STEP 4.2: VALIDATE MARKET SIMULATOR
        # ==================================================================
        with self.step("4.2 Validate MarketSimulator"):
            from src.environment.market_simulator import MarketSimulator
            from src.environment.cost_model import CostModel
            from src.environment.constraints import PortfolioConstraints

            # Initialize components
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

            # Create simulator
            simulator = MarketSimulator(
                algo_returns=algo_returns,
                benchmark_weights=benchmark_weights,
                initial_capital=config.initial_capital,
                rebalance_frequency=config.rebalance_frequency,
                cost_model=cost_model,
                constraints=constraints,
            )

            # Test reset
            obs = simulator.reset()
            self.logger.info(f"Simulator initialized: {simulator.n_algos} algorithms")
            self.logger.info(f"Rebalance dates: {len(simulator._rebalance_dates)}")
            self.logger.info(f"Initial observation shape: {obs.shape}")

            results['simulator'] = {
                'n_algos': simulator.n_algos,
                'n_rebalance_dates': len(simulator._rebalance_dates),
                'initial_capital': config.initial_capital,
            }

            # Test single step with equal weights
            equal_weights = np.ones(simulator.n_algos) / simulator.n_algos
            step_result = simulator.step(equal_weights)

            self.logger.info(f"Test step completed: reward={step_result.reward:.4f}")
            self.logger.info(f"Portfolio value after step: ${step_result.info['portfolio_value']:,.0f}")

        # ==================================================================
        # STEP 4.3: VALIDATE TRADING ENVIRONMENT
        # ==================================================================
        with self.step("4.3 Validate TradingEnvironment"):
            from src.environment.trading_env import TradingEnvironment, EpisodeConfig

            episode_config = EpisodeConfig(
                random_start=False,
                episode_length=config.episode_length,
                min_episode_length=26,
                warmup_periods=4,
            )

            env = TradingEnvironment(
                algo_returns=algo_returns,
                benchmark_weights=benchmark_weights,
                initial_capital=config.initial_capital,
                rebalance_frequency=config.rebalance_frequency,
                cost_model=cost_model,
                constraints=constraints,
                episode_config=episode_config,
                reward_scale=config.reward_scale,
            )

            # Test reset and spaces
            obs, info = env.reset()
            self.logger.info(f"Observation space: {env.observation_space.shape}")
            self.logger.info(f"Action space: {env.action_space.shape}")
            self.logger.info(f"Initial observation shape: {obs.shape}")

            results['env'] = {
                'observation_shape': list(env.observation_space.shape),
                'action_shape': list(env.action_space.shape),
                'obs_range': [float(obs.min()), float(obs.max())],
            }

            # Test single step
            action = env.action_space.sample()
            obs2, reward, terminated, truncated, info = env.step(action)
            self.logger.info(f"Step result: reward={reward:.4f}, terminated={terminated}")

        # ==================================================================
        # STEP 4.4: VALIDATE REWARD FUNCTION
        # ==================================================================
        with self.step("4.4 Validate Reward Function"):
            from src.environment.reward import RewardFunction, RewardType

            reward_types = [
                RewardType.ABSOLUTE_RETURNS,
                RewardType.RISK_CALIBRATED_RETURNS,
                RewardType.RISK_ADJUSTED,
                RewardType.DIVERSIFIED,
            ]

            reward_results = {}
            for reward_type in reward_types:
                reward_fn = RewardFunction(
                    reward_type=reward_type,
                    cost_penalty_weight=1.0,
                    turnover_penalty_weight=0.1,
                    drawdown_penalty_weight=0.5,
                )

                # Test reward computation
                reward_components = reward_fn.compute(
                    portfolio_return=0.01,
                    benchmark_return=0.005,
                    transaction_costs=0.0001,
                    turnover=0.1,
                    current_drawdown=0.05,
                    portfolio_vol=0.15,
                )
                reward_results[reward_type.name] = float(reward_components.total)
                self.logger.info(f"  {reward_type.name}: {reward_components.total:.4f}")

            results['reward_tests'] = reward_results

        # ==================================================================
        # STEP 4.5: RUN TEST EPISODES
        # ==================================================================
        with self.step("4.5 Run Test Episodes"):
            episode_metrics = []

            for ep in range(args.episodes):
                obs, info = env.reset(seed=ep)
                done = False
                total_reward = 0
                steps = 0

                while not done:
                    # Random policy
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                episode_metrics.append({
                    'episode': ep,
                    'total_reward': total_reward,
                    'steps': steps,
                    'final_value': info.get('portfolio_value', 0),
                })
                self.logger.info(f"  Episode {ep}: reward={total_reward:.2f}, steps={steps}")

            results['test_episodes'] = episode_metrics

            # Summary statistics
            rewards = [m['total_reward'] for m in episode_metrics]
            self.logger.info(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

        # ==================================================================
        # STEP 4.6: SANITY CHECKS
        # ==================================================================
        with self.step("4.6 Sanity Checks"):
            sanity_results = {}

            # Test 1: Equal weight policy should be stable
            self.logger.info("Testing equal weight policy...")
            obs, _ = env.reset(seed=100)
            equal_weights = np.ones(env.action_space.shape[0]) / env.action_space.shape[0]
            rewards_ew = []
            for _ in range(min(52, config.episode_length)):
                obs, reward, terminated, truncated, info = env.step(equal_weights)
                rewards_ew.append(reward)
                if terminated or truncated:
                    break
            sanity_results['equal_weight_mean_reward'] = float(np.mean(rewards_ew))
            self.logger.info(f"  Equal weight mean reward: {np.mean(rewards_ew):.4f}")

            # Test 2: Zero action should fail gracefully
            self.logger.info("Testing zero weights handling...")
            obs, _ = env.reset(seed=101)
            zero_weights = np.zeros(env.action_space.shape[0])
            obs, reward, terminated, truncated, info = env.step(zero_weights)
            sanity_results['zero_weights_handled'] = not np.isnan(reward)
            self.logger.info(f"  Zero weights handled: {sanity_results['zero_weights_handled']}")

            # Test 3: Extreme weights should be constrained
            self.logger.info("Testing constraint enforcement...")
            obs, _ = env.reset(seed=102)
            extreme_weights = np.zeros(env.action_space.shape[0])
            extreme_weights[0] = 1.0  # All in one algo
            obs, reward, terminated, truncated, info = env.step(extreme_weights)
            actual_max = info.get('weights', extreme_weights).max()
            sanity_results['max_weight_enforced'] = actual_max <= config.max_weight + 0.01
            self.logger.info(f"  Max weight enforced: {sanity_results['max_weight_enforced']} (max={actual_max:.2f})")

            results['sanity_checks'] = sanity_results

        # ==================================================================
        # STEP 4.7: STRESS TESTS (FULL MODE ONLY)
        # ==================================================================
        if args.full:
            with self.step("4.7 Stress Tests"):
                stress_results = {}

                # Long episode test
                self.logger.info("Testing long episode (200 steps)...")
                episode_config_long = EpisodeConfig(
                    random_start=False,
                    episode_length=200,
                    min_episode_length=50,
                )
                env_long = TradingEnvironment(
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    episode_config=episode_config_long,
                )
                obs, _ = env_long.reset()
                for step in range(200):
                    action = env_long.action_space.sample()
                    obs, reward, terminated, truncated, info = env_long.step(action)
                    if terminated or truncated:
                        break
                stress_results['long_episode_steps'] = step + 1
                self.logger.info(f"  Long episode completed: {step + 1} steps")

                # Multiple resets test
                self.logger.info("Testing multiple resets (100x)...")
                for i in range(100):
                    obs, info = env.reset(seed=i)
                    if np.any(np.isnan(obs)):
                        stress_results['nan_on_reset'] = True
                        break
                else:
                    stress_results['nan_on_reset'] = False
                self.logger.info(f"  No NaN on reset: {not stress_results['nan_on_reset']}")

                results['stress_tests'] = stress_results

        # ==================================================================
        # GENERATE REPORT
        # ==================================================================
        self._generate_report(results, output_dir)

        return results

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
        report = f"""# Phase 4: Environment Validation Report

**Generated**: {datetime.now().isoformat()}
**Mode**: {results['mode']}

## Data Summary

| Metric | Value |
|--------|-------|
| Algorithms | {results['n_algos']} |
| Trading Days | {results['n_days']} |
| Sample Size | {results.get('sample_size', 'All')} |

## Environment Specifications

### Observation Space
- Shape: {results['env']['observation_shape']}
- Range: [{results['env']['obs_range'][0]:.2f}, {results['env']['obs_range'][1]:.2f}]

### Action Space
- Shape: {results['env']['action_shape']}
- Type: Continuous (normalized weights)

### Simulator Configuration
- Rebalance Dates: {results['simulator']['n_rebalance_dates']}
- Initial Capital: ${results['simulator']['initial_capital']:,.0f}

## Reward Function Tests

| Type | Test Value |
|------|-----------|
"""
        for name, value in results.get('reward_tests', {}).items():
            report += f"| {name} | {value:.4f} |\n"

        report += f"""
## Test Episodes (Random Policy)

| Episode | Total Reward | Steps | Final Value |
|---------|-------------|-------|-------------|
"""
        for ep in results.get('test_episodes', []):
            report += f"| {ep['episode']} | {ep['total_reward']:.2f} | {ep['steps']} | ${ep['final_value']:,.0f} |\n"

        report += f"""
## Sanity Checks

| Check | Result |
|-------|--------|
"""
        for check, passed in results.get('sanity_checks', {}).items():
            status = "PASS" if passed else "FAIL"
            report += f"| {check} | {status} |\n"

        if 'stress_tests' in results:
            report += f"""
## Stress Tests

| Test | Result |
|------|--------|
"""
            for test, value in results['stress_tests'].items():
                report += f"| {test} | {value} |\n"

        report += """
## Conclusion

Environment validation completed. Ready for Phase 5 (RL Training).
"""

        # Save report
        report_path = output_dir / "PHASE4_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        self.logger.info(f"Report saved to: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    runner = Phase4Runner()
    runner.execute()


if __name__ == "__main__":
    main()
