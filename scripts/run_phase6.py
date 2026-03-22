#!/usr/bin/env python3
"""
=================================================================
PHASE 6: Walk-Forward Evaluation & Comparison
=================================================================
Complete Phase 6 pipeline to evaluate trained RL agents:
  6.1 Load trained models from Phase 5
  6.2 Run walk-forward validation
  6.3 Evaluate baselines with same protocol
  6.4 Compare all strategies
  6.5 Generate final report

Walk-Forward Protocol:
  - Train window: 252 days (1 year)
  - Validation window: 63 days (1 quarter)
  - Test window: 63 days (1 quarter)
  - Step size: 63 days (rolling quarterly)

Usage:
  python scripts/run_phase6.py                      # Evaluate PPO (default)
  python scripts/run_phase6.py --agent all          # Evaluate all trained agents
  python scripts/run_phase6.py --quick              # Quick test (shorter windows)
  python scripts/run_phase6.py --include-baselines  # Include baseline comparison
  python scripts/run_phase6.py --folds 3            # Limit number of folds

Options:
  --agent NAME           Agent to evaluate: ppo, sac, td3, all (default: ppo)
  --quick                Quick test mode (shorter windows)
  --include-baselines    Also evaluate classical baselines
  --folds N              Maximum number of folds (default: all)
  --train-window N       Training window in days (default: 252)
  --val-window N         Validation window in days (default: 63)
  --test-window N        Test window in days (default: 63)
  --step-size N          Step size between folds (default: 63)
  --expanding            Use expanding window (default: rolling)
"""

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

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
class EvalConfig:
    """Evaluation configuration."""
    # Walk-forward windows
    train_window: int = 252      # 1 year
    val_window: int = 63         # 1 quarter
    test_window: int = 63        # 1 quarter
    step_size: int = 63          # quarterly step
    expanding: bool = False      # rolling vs expanding
    max_folds: Optional[int] = None

    # Environment
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "weekly"

    # Constraints (same as training)
    max_weight: float = 0.40
    min_weight: float = 0.00
    max_turnover: float = 0.30
    max_exposure: float = 1.0

    # Costs (same as training)
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    impact_coefficient: float = 0.1

    # Reward (for consistency)
    reward_scale: float = 100.0

    # Seed
    seed: int = 42


# =============================================================================
# Phase 6 Runner
# =============================================================================

class Phase6Runner(PhaseRunner):
    """Phase 6: Walk-Forward Evaluation."""

    phase_name = "Phase 6: Walk-Forward Evaluation"
    phase_number = 6

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 6 specific arguments."""
        parser.add_argument(
            '--agent', type=str, default='ppo',
            choices=['ppo', 'sac', 'td3', 'all'],
            help='Agent to evaluate (default: ppo)'
        )
        parser.add_argument(
            '--quick', '-q', action='store_true',
            help='Quick test mode (shorter windows)'
        )
        parser.add_argument(
            '--include-baselines', '-b', action='store_true',
            help='Include classical baselines in comparison'
        )
        parser.add_argument(
            '--folds', type=int, default=None,
            help='Maximum number of folds'
        )
        parser.add_argument(
            '--train-window', type=int, default=252,
            help='Training window in days (default: 252)'
        )
        parser.add_argument(
            '--val-window', type=int, default=63,
            help='Validation window in days (default: 63)'
        )
        parser.add_argument(
            '--test-window', type=int, default=63,
            help='Test window in days (default: 63)'
        )
        parser.add_argument(
            '--step-size', type=int, default=63,
            help='Step size between folds (default: 63)'
        )
        parser.add_argument(
            '--expanding', action='store_true',
            help='Use expanding window instead of rolling'
        )
        parser.add_argument(
            '--models-dir', type=str, default=None,
            help='Override directory containing trained models'
        )
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Use only N algorithms (must match training)'
        )
        parser.add_argument(
            '--seed', type=int, default=42,
            help='Random seed'
        )

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 6 pipeline."""
        # Quick mode overrides
        if args.quick:
            args.train_window = 126  # 6 months
            args.val_window = 21     # 1 month
            args.test_window = 21    # 1 month
            args.step_size = 21      # monthly
            args.folds = args.folds or 3
            args.sample = args.sample or 20  # Match Phase 5 quick mode

        # Build config
        config = EvalConfig(
            train_window=args.train_window,
            val_window=args.val_window,
            test_window=args.test_window,
            step_size=args.step_size,
            expanding=args.expanding,
            max_folds=args.folds,
            seed=args.seed,
        )

        results = {
            'mode': 'quick' if args.quick else 'standard',
            'config': asdict(config),
            'agents': {},
            'baselines': {},
            'comparison': {},
        }

        # Determine paths
        output_dir = self.get_output_dir()
        models_dir = Path(args.models_dir) if args.models_dir else self.op.rl_training.root / "checkpoints"

        # ==================================================================
        # STEP 6.1: LOAD DATA
        # ==================================================================
        with self.step("6.1 Load Data"):
            algo_returns, benchmark_weights, benchmark_returns = self._load_data()

            # Sample if requested - must match training configuration
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
                self.logger.info(f"Sampled {len(sampled_cols)} algorithms (matching training)")

            self.logger.info(f"Returns: {algo_returns.shape[0]} days x {algo_returns.shape[1]} algos")
            results['n_algos'] = algo_returns.shape[1]
            results['n_days'] = algo_returns.shape[0]

        # ==================================================================
        # STEP 6.2: GENERATE WALK-FORWARD FOLDS
        # ==================================================================
        with self.step("6.2 Generate Walk-Forward Folds"):
            from src.evaluation.walk_forward import WalkForwardValidator

            validator = WalkForwardValidator(
                train_window=config.train_window,
                val_window=config.val_window,
                test_window=config.test_window,
                step_size=config.step_size,
                expanding=config.expanding,
            )

            folds = validator.generate_folds(algo_returns.index)

            if config.max_folds is not None:
                folds = folds[:config.max_folds]

            self.logger.info(f"Generated {len(folds)} walk-forward folds")
            for i, fold in enumerate(folds):
                self.logger.info(
                    f"  Fold {i}: train {fold['train_start'].date()} - {fold['train_end'].date()}, "
                    f"test {fold['test_start'].date()} - {fold['test_end'].date()}"
                )

            results['n_folds'] = len(folds)
            results['folds'] = [
                {k: str(v.date()) if hasattr(v, 'date') else v for k, v in fold.items()}
                for fold in folds
            ]

        # ==================================================================
        # STEP 6.3: EVALUATE RL AGENTS
        # ==================================================================
        agents_to_eval = ['ppo', 'sac', 'td3'] if args.agent == 'all' else [args.agent]

        for agent_name in agents_to_eval:
            model_path = models_dir / agent_name / "final_model"

            if not model_path.exists() and not (model_path.parent / "final_model.zip").exists():
                self.logger.warning(f"Model not found for {agent_name}: {model_path}")
                continue

            with self.step(f"6.3 Evaluate {agent_name.upper()} Agent"):
                agent_results = self._evaluate_agent(
                    agent_name=agent_name,
                    model_path=model_path,
                    folds=folds,
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    benchmark_returns=benchmark_returns,
                    config=config,
                    output_dir=output_dir,
                )
                results['agents'][agent_name] = agent_results

            gc.collect()

        # ==================================================================
        # STEP 6.4: EVALUATE BASELINES (optional)
        # ==================================================================
        if args.include_baselines:
            baselines = ['equal_weight', 'risk_parity', 'min_variance', 'max_sharpe', 'momentum', 'vol_targeting']

            for baseline_name in baselines:
                with self.step(f"6.4 Evaluate {baseline_name.replace('_', ' ').title()} Baseline"):
                    baseline_results = self._evaluate_baseline(
                        baseline_name=baseline_name,
                        folds=folds,
                        algo_returns=algo_returns,
                        benchmark_weights=benchmark_weights,
                        benchmark_returns=benchmark_returns,
                        config=config,
                    )
                    results['baselines'][baseline_name] = baseline_results

                gc.collect()

        # ==================================================================
        # STEP 6.5: COMPARE ALL STRATEGIES
        # ==================================================================
        with self.step("6.5 Compare Strategies"):
            comparison = self._compare_strategies(results, benchmark_returns)
            results['comparison'] = comparison

            # Print summary table
            self._print_comparison_table(comparison)

        # ==================================================================
        # STEP 6.6: GENERATE REPORT
        # ==================================================================
        with self.step("6.6 Generate Report"):
            self._generate_report(results, output_dir)

        return results

    def _load_data(self):
        """Load Phase 1 data."""
        # Load returns matrix
        returns_path = self.dp.algorithms.returns
        if not returns_path.exists():
            returns_path = self.dp.processed.root / 'algo_returns.parquet'
        if not returns_path.exists():
            raise FileNotFoundError(f"Returns matrix not found: {returns_path}")

        algo_returns = pd.read_parquet(returns_path)
        algo_returns = algo_returns.astype(np.float32, copy=False)
        if algo_returns.index.tz is not None:
            algo_returns.index = algo_returns.index.tz_localize(None)
        self.logger.info(f"Loaded returns: {algo_returns.shape}")

        # Load benchmark weights
        weights_path = self.dp.benchmark.weights
        if not weights_path.exists():
            weights_path = self.dp.processed.root / 'benchmark_weights.parquet'

        benchmark_weights = None
        if weights_path.exists():
            benchmark_weights = pd.read_parquet(weights_path)
            benchmark_weights = benchmark_weights.astype(np.float32, copy=False)
            if benchmark_weights.index.tz is not None:
                benchmark_weights.index = benchmark_weights.index.tz_localize(None)
            self.logger.info(f"Loaded benchmark weights: {benchmark_weights.shape}")

        # Load benchmark returns
        bench_path = self.dp.benchmark.daily_returns
        if not bench_path.exists():
            bench_path = self.dp.processed.root / 'benchmark_daily_returns.csv'

        benchmark_returns = None
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path, index_col=0, parse_dates=True)
            if 'return' in bench_df.columns:
                benchmark_returns = bench_df['return']
            else:
                benchmark_returns = bench_df.iloc[:, 0]
            self.logger.info(f"Loaded benchmark returns: {len(benchmark_returns)} days")

        return algo_returns, benchmark_weights, benchmark_returns

    def _evaluate_agent(
        self,
        agent_name: str,
        model_path: Path,
        folds: List[dict],
        algo_returns: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        config: EvalConfig,
        output_dir: Path,
    ) -> dict:
        """Evaluate an RL agent using walk-forward."""
        from src.agents.ppo_agent import PPOAllocator
        from src.agents.sac_agent import SACAllocator
        from src.agents.td3_agent import TD3Allocator
        from src.environment.trading_env import TradingEnvironment, EpisodeConfig
        from src.environment.cost_model import CostModel
        from src.environment.constraints import PortfolioConstraints
        from src.environment.reward import RewardFunction, RewardType
        from src.evaluation.metrics import compute_full_metrics

        # Create components
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
        )

        # Load agent
        if agent_name == 'ppo':
            AgentClass = PPOAllocator
        elif agent_name == 'sac':
            AgentClass = SACAllocator
        elif agent_name == 'td3':
            AgentClass = TD3Allocator
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Check for .zip extension
        actual_path = model_path
        if not model_path.exists() and (model_path.parent / "final_model.zip").exists():
            actual_path = model_path.parent / "final_model.zip"

        fold_results = []
        all_test_returns = []

        for fold in folds:
            self.logger.info(f"  Processing fold {fold['fold_id']}...")

            # Create test environment
            episode_config = EpisodeConfig(
                random_start=False,
                episode_length=None,  # Run full period
            )

            test_env = TradingEnvironment(
                algo_returns=algo_returns,
                benchmark_weights=benchmark_weights,
                train_start=fold['test_start'],
                train_end=fold['test_end'],
                initial_capital=config.initial_capital,
                rebalance_frequency=config.rebalance_frequency,
                cost_model=cost_model,
                constraints=constraints,
                reward_function=reward_fn,
                episode_config=episode_config,
                reward_scale=config.reward_scale,
            )

            # Load and evaluate agent
            agent = AgentClass.from_pretrained(str(actual_path), env=test_env)

            # Run episode
            obs, _ = test_env.reset()
            done = False
            episode_returns = []
            episode_dates = []
            total_reward = 0

            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)

                total_reward += reward
                if 'portfolio_return' in info:
                    episode_returns.append(info['portfolio_return'])
                if 'date' in info:
                    episode_dates.append(info['date'])

                done = terminated or truncated

            # Compute metrics for this fold
            if episode_returns:
                returns_series = pd.Series(episode_returns, index=episode_dates[:len(episode_returns)])

                # Get benchmark returns for this period
                bench_fold = None
                if benchmark_returns is not None:
                    bench_fold = benchmark_returns.loc[fold['test_start']:fold['test_end']]

                metrics = compute_full_metrics(returns_series, bench_fold)

                fold_results.append({
                    'fold_id': fold['fold_id'],
                    'test_start': str(fold['test_start'].date()),
                    'test_end': str(fold['test_end'].date()),
                    'total_reward': total_reward,
                    'n_steps': len(episode_returns),
                    **metrics,
                })

                all_test_returns.extend(episode_returns)

        # Aggregate metrics across folds
        aggregated = self._aggregate_fold_metrics(fold_results)

        # Save fold results
        fold_df = pd.DataFrame(fold_results)
        fold_path = output_dir / "walk_forward" / f"{agent_name}_folds.csv"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(fold_path, index=False)

        self.logger.info(f"  Aggregated Sharpe: {aggregated.get('sharpe_ratio_mean', 0):.3f}")
        self.logger.info(f"  Aggregated Return: {aggregated.get('annualized_return_mean', 0):.2%}")
        self.logger.info(f"  Max Drawdown: {aggregated.get('max_drawdown_mean', 0):.2%}")

        return {
            'model_path': str(actual_path),
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'aggregated': aggregated,
        }

    def _evaluate_baseline(
        self,
        baseline_name: str,
        folds: List[dict],
        algo_returns: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        config: EvalConfig,
    ) -> dict:
        """Evaluate a classical baseline using walk-forward."""
        from src.baselines.equal_weight import EqualWeightAllocator
        from src.baselines.risk_parity import RiskParityAllocator
        from src.baselines.min_variance import MinVarianceAllocator
        from src.baselines.max_sharpe import MaxSharpeAllocator
        from src.baselines.momentum_allocator import MomentumAllocator
        from src.baselines.vol_targeting import VolTargetingAllocator
        from src.evaluation.metrics import compute_full_metrics

        # Select baseline
        baseline_map = {
            'equal_weight': EqualWeightAllocator,
            'risk_parity': RiskParityAllocator,
            'min_variance': MinVarianceAllocator,
            'max_sharpe': MaxSharpeAllocator,
            'momentum': MomentumAllocator,
            'vol_targeting': VolTargetingAllocator,
        }

        AllocatorClass = baseline_map.get(baseline_name)
        if AllocatorClass is None:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        fold_results = []

        for fold in folds:
            self.logger.info(f"  Processing fold {fold['fold_id']}...")

            # Get data for this fold
            train_data = algo_returns.loc[fold['train_start']:fold['train_end']]
            test_data = algo_returns.loc[fold['test_start']:fold['test_end']]

            # Create and fit allocator
            allocator = AllocatorClass(
                max_weight=config.max_weight,
                min_weight=config.min_weight,
            )

            # Compute weights using training data
            weights = allocator.compute_weights(train_data)

            # Apply to test period
            test_returns = (test_data * weights).sum(axis=1)

            # Get benchmark returns for this period
            bench_fold = None
            if benchmark_returns is not None:
                bench_fold = benchmark_returns.loc[fold['test_start']:fold['test_end']]

            metrics = compute_full_metrics(test_returns, bench_fold)

            fold_results.append({
                'fold_id': fold['fold_id'],
                'test_start': str(fold['test_start'].date()),
                'test_end': str(fold['test_end'].date()),
                **metrics,
            })

        # Aggregate metrics across folds
        aggregated = self._aggregate_fold_metrics(fold_results)

        self.logger.info(f"  Aggregated Sharpe: {aggregated.get('sharpe_ratio_mean', 0):.3f}")
        self.logger.info(f"  Aggregated Return: {aggregated.get('annualized_return_mean', 0):.2%}")

        return {
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'aggregated': aggregated,
        }

    def _aggregate_fold_metrics(self, fold_results: List[dict]) -> dict:
        """Aggregate metrics across folds."""
        if not fold_results:
            return {}

        aggregated = {}

        # Get all metric keys (excluding non-numeric)
        exclude_keys = {'fold_id', 'test_start', 'test_end', 'model_path'}
        metric_keys = [k for k in fold_results[0].keys() if k not in exclude_keys]

        for key in metric_keys:
            values = [f.get(key) for f in fold_results if f.get(key) is not None]
            values = [v for v in values if not np.isnan(v)]

            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)

        return aggregated

    def _compare_strategies(self, results: dict, benchmark_returns: Optional[pd.Series]) -> dict:
        """Compare all strategies."""
        comparison = {
            'strategies': [],
            'ranking': {},
        }

        # Add benchmark
        if benchmark_returns is not None:
            from src.evaluation.metrics import compute_full_metrics
            bench_metrics = compute_full_metrics(benchmark_returns)
            comparison['strategies'].append({
                'name': 'Benchmark',
                'type': 'benchmark',
                'sharpe_ratio': bench_metrics.get('sharpe_ratio', 0),
                'annualized_return': bench_metrics.get('annualized_return', 0),
                'max_drawdown': bench_metrics.get('max_drawdown', 0),
                'volatility': bench_metrics.get('annualized_volatility', 0),
            })

        # Add RL agents
        for agent_name, agent_results in results.get('agents', {}).items():
            agg = agent_results.get('aggregated', {})
            comparison['strategies'].append({
                'name': agent_name.upper(),
                'type': 'rl_agent',
                'sharpe_ratio': agg.get('sharpe_ratio_mean', 0),
                'annualized_return': agg.get('annualized_return_mean', 0),
                'max_drawdown': agg.get('max_drawdown_mean', 0),
                'volatility': agg.get('annualized_volatility_mean', 0),
                'information_ratio': agg.get('information_ratio_mean', 0),
            })

        # Add baselines
        for baseline_name, baseline_results in results.get('baselines', {}).items():
            agg = baseline_results.get('aggregated', {})
            comparison['strategies'].append({
                'name': baseline_name.replace('_', ' ').title(),
                'type': 'baseline',
                'sharpe_ratio': agg.get('sharpe_ratio_mean', 0),
                'annualized_return': agg.get('annualized_return_mean', 0),
                'max_drawdown': agg.get('max_drawdown_mean', 0),
                'volatility': agg.get('annualized_volatility_mean', 0),
            })

        # Rank by Sharpe ratio
        strategies = comparison['strategies']
        ranked = sorted(strategies, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        comparison['ranking']['by_sharpe'] = [s['name'] for s in ranked]

        # Rank by return
        ranked = sorted(strategies, key=lambda x: x.get('annualized_return', 0), reverse=True)
        comparison['ranking']['by_return'] = [s['name'] for s in ranked]

        return comparison

    def _print_comparison_table(self, comparison: dict):
        """Print comparison table to console."""
        strategies = comparison.get('strategies', [])
        if not strategies:
            return

        self.logger.info("\n" + "=" * 80)
        self.logger.info("STRATEGY COMPARISON")
        self.logger.info("=" * 80)
        self.logger.info(f"{'Strategy':<20} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Vol':>10} {'IR':>10}")
        self.logger.info("-" * 80)

        for s in strategies:
            ir = s.get('information_ratio', 0) or 0
            self.logger.info(
                f"{s['name']:<20} "
                f"{s.get('sharpe_ratio', 0):>10.3f} "
                f"{s.get('annualized_return', 0):>11.2%} "
                f"{s.get('max_drawdown', 0):>10.2%} "
                f"{s.get('volatility', 0):>10.2%} "
                f"{ir:>10.3f}"
            )

        self.logger.info("=" * 80)

        # Print ranking
        ranking = comparison.get('ranking', {})
        if ranking.get('by_sharpe'):
            self.logger.info(f"\nRanking by Sharpe: {' > '.join(ranking['by_sharpe'])}")

    def _generate_report(self, results: dict, output_dir: Path):
        """Generate markdown report."""
        config = results['config']

        report = f"""# Phase 6: Walk-Forward Evaluation Report

**Generated**: {datetime.now().isoformat()}
**Mode**: {results['mode']}

## Configuration

| Parameter | Value |
|-----------|-------|
| Train Window | {config['train_window']} days |
| Validation Window | {config['val_window']} days |
| Test Window | {config['test_window']} days |
| Step Size | {config['step_size']} days |
| Window Type | {'Expanding' if config['expanding'] else 'Rolling'} |
| Number of Folds | {results['n_folds']} |

## Data Summary

| Metric | Value |
|--------|-------|
| Algorithms | {results['n_algos']} |
| Trading Days | {results['n_days']} |

## Walk-Forward Folds

| Fold | Train Period | Test Period |
|------|--------------|-------------|
"""
        for fold in results.get('folds', []):
            report += f"| {fold['fold_id']} | {fold['train_start']} - {fold['train_end']} | {fold['test_start']} - {fold['test_end']} |\n"

        report += "\n## RL Agents\n\n"

        for agent_name, agent_results in results.get('agents', {}).items():
            agg = agent_results.get('aggregated', {})
            report += f"""### {agent_name.upper()}

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Sharpe Ratio | {agg.get('sharpe_ratio_mean', 0):.3f} | {agg.get('sharpe_ratio_std', 0):.3f} | {agg.get('sharpe_ratio_min', 0):.3f} | {agg.get('sharpe_ratio_max', 0):.3f} |
| Ann. Return | {agg.get('annualized_return_mean', 0):.2%} | {agg.get('annualized_return_std', 0):.2%} | {agg.get('annualized_return_min', 0):.2%} | {agg.get('annualized_return_max', 0):.2%} |
| Max Drawdown | {agg.get('max_drawdown_mean', 0):.2%} | {agg.get('max_drawdown_std', 0):.2%} | {agg.get('max_drawdown_min', 0):.2%} | {agg.get('max_drawdown_max', 0):.2%} |
| Info Ratio | {agg.get('information_ratio_mean', 0):.3f} | {agg.get('information_ratio_std', 0):.3f} | {agg.get('information_ratio_min', 0):.3f} | {agg.get('information_ratio_max', 0):.3f} |

"""

        if results.get('baselines'):
            report += "\n## Classical Baselines\n\n"

            for baseline_name, baseline_results in results.get('baselines', {}).items():
                agg = baseline_results.get('aggregated', {})
                report += f"""### {baseline_name.replace('_', ' ').title()}

| Metric | Mean | Std |
|--------|------|-----|
| Sharpe Ratio | {agg.get('sharpe_ratio_mean', 0):.3f} | {agg.get('sharpe_ratio_std', 0):.3f} |
| Ann. Return | {agg.get('annualized_return_mean', 0):.2%} | {agg.get('annualized_return_std', 0):.2%} |
| Max Drawdown | {agg.get('max_drawdown_mean', 0):.2%} | {agg.get('max_drawdown_std', 0):.2%} |

"""

        # Comparison table
        comparison = results.get('comparison', {})
        strategies = comparison.get('strategies', [])

        if strategies:
            report += """## Strategy Comparison

| Strategy | Type | Sharpe | Return | Max DD | Vol |
|----------|------|--------|--------|--------|-----|
"""
            for s in strategies:
                report += f"| {s['name']} | {s['type']} | {s.get('sharpe_ratio', 0):.3f} | {s.get('annualized_return', 0):.2%} | {s.get('max_drawdown', 0):.2%} | {s.get('volatility', 0):.2%} |\n"

        ranking = comparison.get('ranking', {})
        if ranking.get('by_sharpe'):
            report += f"\n**Ranking by Sharpe Ratio**: {' > '.join(ranking['by_sharpe'])}\n"

        report += """
## Files Generated

- `walk_forward/*_folds.csv` - Per-fold metrics for each agent
- `PHASE6_SUMMARY.md` - This report
- `results.json` - Full results in JSON format
"""

        # Save report
        report_path = output_dir / "PHASE6_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        self.logger.info(f"Report saved to: {report_path}")

        # Save JSON results
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    runner = Phase6Runner()
    runner.execute()


if __name__ == "__main__":
    main()
