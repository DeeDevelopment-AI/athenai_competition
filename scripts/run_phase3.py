#!/usr/bin/env python3
"""
=================================================================
PHASE 3: Baseline Backtesting
=================================================================
Complete Phase 3 pipeline for classical allocation strategies:
  3.1 Load processed data from Phase 1
  3.2 Configure trial specifications (factor x optimizer combinations)
  3.3 Run systematic backtests with train/val/test splits
  3.4 Optionally run walk-forward validation
  3.5 Generate comparison report

Baselines tested:
  - Equal Weight
  - Risk Parity
  - Min Variance
  - Max Sharpe
  - Momentum
  - Vol Targeting

Usage:
  python scripts/run_phase3.py                  # Standard run
  python scripts/run_phase3.py --quick          # Quick test (fewer configs)
  python scripts/run_phase3.py --full           # Full run + walk-forward
  python scripts/run_phase3.py --baseline risk_parity  # Single baseline
  python scripts/run_phase3.py --dry-run        # Show configurations

Options:
  --quick              Run reduced set of configurations
  --full               Full run with walk-forward validation
  --baseline NAME      Run only specific baseline
  --fresh              Clear previous results
  --walk-forward       Add walk-forward validation to standard run
"""

import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import dataclass
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
# Constants and Configuration
# =============================================================================

BASELINES = [
    'equal_weight',
    'risk_parity',
    'min_variance',
    'max_sharpe',
    'momentum',
    'vol_targeting',
]

# Available factors for selection
FACTORS = {
    # Momentum factors (higher is better -> top_n)
    'momentum_5d': 'rolling_return_5d',
    'momentum_21d': 'rolling_return_21d',
    'momentum_63d': 'rolling_return_63d',
    # Quality factors (higher is better -> top_n)
    'sharpe_21d': 'rolling_sharpe_21d',
    'sharpe_63d': 'rolling_sharpe_63d',
    # Defensive factors
    'low_vol_21d': 'rolling_volatility_21d',
    'low_vol_63d': 'rolling_volatility_63d',
    'min_drawdown_21d': 'rolling_drawdown_21d',
}

# Default constraints (calibrated from benchmark analysis)
DEFAULT_CONSTRAINTS = {
    'max_weight': 0.40,
    'min_weight': 0.00,
    'max_turnover': 0.30,
    'max_exposure': 1.0,
}


@dataclass
class TrialSpec:
    """Specification for a single backtest trial."""
    factor_name: str
    selection_method: str = "top_n"
    selection_param: float = 100  # Top N algorithms
    optimizer: str = "equal_weight"
    lookback_window: int = 63
    shrinkage: float = 0.1
    rebalance_frequency: str = "weekly"
    target_vol: float = 0.10

    def __str__(self):
        return f"{self.optimizer}|{self.factor_name}|top{int(self.selection_param)}|lb{self.lookback_window}"


# =============================================================================
# Phase 3 Runner
# =============================================================================

class Phase3Runner(PhaseRunner):
    """Phase 3: Baseline Backtesting."""

    phase_name = "Phase 3: Baseline Backtesting"
    phase_number = 3

    def _run_tag(self) -> str:
        if getattr(self.args, 'full', False):
            return "full"
        if getattr(self.args, 'quick', False):
            return "quick"
        return "standard"

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 3 specific arguments."""
        parser.add_argument(
            '--quick', '-q', action='store_true',
            help='Quick mode: fewer configurations'
        )
        parser.add_argument(
            '--full', '-f', action='store_true',
            help='Full mode: all configs + walk-forward'
        )
        parser.add_argument(
            '--baseline', type=str, default=None,
            choices=BASELINES,
            help='Run only specific baseline'
        )
        parser.add_argument(
            '--fresh', action='store_true',
            help='Clear previous results'
        )
        parser.add_argument(
            '--walk-forward', action='store_true',
            help='Include walk-forward validation'
        )
        parser.add_argument(
            '--n-top', type=int, default=5,
            help='Number of top configs for walk-forward'
        )
        parser.add_argument(
            '--input-dir', type=str, default=None,
            help='Override input directory (Phase 1 outputs)'
        )

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 3 pipeline."""
        results = {
            'mode': 'full' if args.full else ('quick' if args.quick else 'standard'),
            'baseline_filter': args.baseline,
        }

        # Determine paths
        input_dir = Path(args.input_dir) if args.input_dir else self.dp.processed.root
        output_dir = self.get_output_dir()

        # Clear previous results if requested
        if args.fresh:
            for f in output_dir.glob("phase3_*"):
                f.unlink()
                self.logger.info(f"Removed {f.name}")

        # ==================================================================
        # STEP 3.1: LOAD DATA
        # ==================================================================
        with self.step("3.1 Load Data"):
            algo_returns, features, benchmark_returns = self._load_data(input_dir)
            self.logger.info(f"Returns: {algo_returns.shape[0]} days x {algo_returns.shape[1]} algos")
            if features is not None:
                self.logger.info(f"Features: {features.shape}")
            if benchmark_returns is not None:
                self.logger.info(f"Benchmark: {len(benchmark_returns)} days")

        # ==================================================================
        # STEP 3.2: GENERATE TRIAL SPECS
        # ==================================================================
        with self.step("3.2 Generate Trial Configurations"):
            specs = self._generate_trial_specs(
                quick_mode=args.quick,
                baseline_filter=args.baseline,
            )
            self.logger.info(f"Generated {len(specs)} trial configurations")
            results['n_trials_planned'] = len(specs)

        # ==================================================================
        # STEP 3.3: COMPUTE DATE SPLITS
        # ==================================================================
        with self.step("3.3 Compute Date Splits"):
            dates = algo_returns.index
            n = len(dates)
            train_end_idx = int(n * 0.60)
            val_end_idx = int(n * 0.80)

            splits = {
                'train_start': str(dates[0].date()),
                'train_end': str(dates[train_end_idx - 1].date()),
                'val_start': str(dates[train_end_idx].date()),
                'val_end': str(dates[val_end_idx - 1].date()),
                'test_start': str(dates[val_end_idx].date()),
                'test_end': str(dates[-1].date()),
            }

            self.logger.info(f"Train: {splits['train_start']} to {splits['train_end']}")
            self.logger.info(f"Val:   {splits['val_start']} to {splits['val_end']}")
            self.logger.info(f"Test:  {splits['test_start']} to {splits['test_end']}")

            results['date_splits'] = splits

        # ==================================================================
        # STEP 3.4: RUN BACKTESTS
        # ==================================================================
        with self.step("3.4 Run Backtests"):
            trial_results = []

            for i, spec in enumerate(specs):
                self.logger.info(f"[{i+1}/{len(specs)}] {spec}")

                result = self._run_single_trial(
                    spec=spec,
                    algo_returns=algo_returns,
                    features=features,
                    benchmark_returns=benchmark_returns,
                    splits=splits,
                )

                if result['status'] == 'completed':
                    self.logger.info(
                        f"    Sharpe: train={result['sharpe_train']:.3f}, "
                        f"val={result['sharpe_val']:.3f}, test={result['sharpe_test']:.3f}"
                    )
                else:
                    self.logger.warning(f"    Failed: {result.get('error', 'Unknown')}")

                trial_results.append(result)

                # Memory cleanup every 50 trials
                if (i + 1) % 50 == 0:
                    gc.collect()

            # Create organized output directories
            trials_dir = output_dir / 'trials'
            trials_dir.mkdir(parents=True, exist_ok=True)
            figures_dir = output_dir / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Save trial results to organized path
            results_df = pd.DataFrame(trial_results)
            trials_csv_path = trials_dir / 'results.csv'
            results_df.to_csv(trials_csv_path, index=False)
            self.logger.info(f"Saved {trials_csv_path.relative_to(output_dir)} ({len(results_df)} trials)")

            # Compute summary statistics
            completed = results_df[results_df['status'] == 'completed']
            results['n_trials_completed'] = len(completed)
            results['n_trials_failed'] = len(results_df) - len(completed)

            if len(completed) > 0:
                best = completed.loc[completed['sharpe_val'].idxmax()]
                results['best_config'] = {
                    'optimizer': best['optimizer'],
                    'factor': best['factor'],
                    'n_select': int(best['n_select']),
                    'lookback': int(best['lookback']),
                    'sharpe_val': float(best['sharpe_val']),
                    'sharpe_test': float(best['sharpe_test']),
                }

        # ==================================================================
        # STEP 3.5: WALK-FORWARD VALIDATION (optional)
        # ==================================================================
        if args.full or args.walk_forward:
            with self.step("3.5 Walk-Forward Validation"):
                wf_results = self._run_walk_forward(
                    results_df=results_df,
                    algo_returns=algo_returns,
                    features=features,
                    benchmark_returns=benchmark_returns,
                    n_top=args.n_top,
                )
                if wf_results:
                    wf_df = pd.DataFrame(wf_results)
                    wf_csv_path = trials_dir / 'walk_forward.csv'
                    wf_df.to_csv(wf_csv_path, index=False)
                    self.logger.info(f"Saved {wf_csv_path.relative_to(output_dir)} ({len(wf_df)} folds)")
                    results['walk_forward'] = {
                        'n_folds': len(wf_df),
                        'avg_sharpe': float(wf_df['sharpe_test'].mean()),
                    }

        # ==================================================================
        # GENERATE REPORTS
        # ==================================================================
        with self.step("Generate Reports"):
            self._generate_markdown_report(results, results_df, output_dir)
            self._print_summary(results_df)

        return results

    def _load_data(self, input_dir: Path) -> tuple:
        """Load Phase 1 data from organized folder structure."""
        # Try organized paths first, fallback to flat paths for compatibility

        # Load returns - try organized path first
        returns_path = self.dp.algorithms.returns
        if not returns_path.exists():
            # Fallback to flat path
            returns_path = input_dir / 'algo_returns.parquet'
        if not returns_path.exists():
            raise FileNotFoundError(
                f"Run Phase 1 first. Expected returns at: {self.dp.algorithms.returns}"
            )
        algo_returns = pd.read_parquet(returns_path)

        # Load features (optional) - try organized path first
        features = None
        features_path = self.dp.algorithms.features
        if not features_path.exists():
            features_path = input_dir / 'algo_features.parquet'
        if features_path.exists():
            features = pd.read_parquet(features_path)

        # Load benchmark - try organized path first
        benchmark_returns = None
        bench_path = self.dp.benchmark.daily_returns
        if not bench_path.exists():
            bench_path = input_dir / 'benchmark_daily_returns.csv'
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path, index_col=0, parse_dates=True)
            benchmark_returns = bench_df.iloc[:, 0]

        return algo_returns, features, benchmark_returns

    def _generate_trial_specs(
        self,
        quick_mode: bool = False,
        baseline_filter: Optional[str] = None,
    ) -> list[TrialSpec]:
        """Generate trial specifications."""
        specs = []

        if quick_mode:
            factors = ['sharpe_21d', 'low_vol_21d']
            selection_params = [100]
            lookbacks = [63]
            optimizers = BASELINES if baseline_filter is None else [baseline_filter]
        else:
            factors = [
                'sharpe_21d', 'sharpe_63d',
                'momentum_21d', 'momentum_63d',
                'low_vol_21d', 'low_vol_63d',
                'min_drawdown_21d',
            ]
            selection_params = [50, 100, 200]
            lookbacks = [21, 63, 126]
            optimizers = [baseline_filter] if baseline_filter else BASELINES

        for factor_key in factors:
            factor_name = FACTORS.get(factor_key, factor_key)

            for n_select in selection_params:
                for lookback in lookbacks:
                    for optimizer in optimizers:
                        # Skip certain combinations to reduce trials
                        if optimizer in ['min_variance', 'max_sharpe'] and n_select > 200:
                            continue

                        specs.append(TrialSpec(
                            factor_name=factor_name,
                            selection_param=n_select,
                            optimizer=optimizer,
                            lookback_window=lookback,
                        ))

        return specs

    def _run_single_trial(
        self,
        spec: TrialSpec,
        algo_returns: pd.DataFrame,
        features: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        splits: dict,
    ) -> dict:
        """Run a single backtest trial."""
        from src.evaluation.metrics import compute_full_metrics

        start_time = time.time()

        try:
            if features is not None:
                # Use fast backtester
                results = self._run_fast_backtest(
                    spec, algo_returns, features, splits
                )
            else:
                # Fallback to allocator-based backtest
                results = self._run_allocator_backtest(
                    spec, algo_returns, splits
                )

            # Compute metrics for validation period
            val_returns = results['val']['portfolio_returns']
            bench_slice = None
            if benchmark_returns is not None:
                mask = (benchmark_returns.index >= splits['val_start']) & \
                       (benchmark_returns.index <= splits['val_end'])
                bench_slice = benchmark_returns.loc[mask]

            metrics = compute_full_metrics(val_returns, bench_slice)

            return {
                'spec': str(spec),
                'optimizer': spec.optimizer,
                'factor': spec.factor_name,
                'n_select': int(spec.selection_param),
                'lookback': spec.lookback_window,
                'sharpe_train': results['train']['sharpe'],
                'sharpe_val': results['val']['sharpe'],
                'sharpe_test': results['test']['sharpe'],
                'return_ann_val': metrics.get('annualized_return', np.nan),
                'vol_ann_val': metrics.get('annualized_volatility', np.nan),
                'max_dd_val': metrics.get('max_drawdown', np.nan),
                'sortino_val': metrics.get('sortino_ratio', np.nan),
                'calmar_val': metrics.get('calmar_ratio', np.nan),
                'information_ratio': metrics.get('information_ratio', np.nan),
                'n_selected_avg': results['val'].get('n_selected_avg', 0),
                'status': 'completed',
                'elapsed_seconds': time.time() - start_time,
            }

        except Exception as e:
            return {
                'spec': str(spec),
                'optimizer': spec.optimizer,
                'factor': spec.factor_name,
                'n_select': int(spec.selection_param),
                'lookback': spec.lookback_window,
                'status': 'failed',
                'error': str(e),
                'elapsed_seconds': time.time() - start_time,
            }

    def _run_fast_backtest(
        self,
        spec: TrialSpec,
        algo_returns: pd.DataFrame,
        features: pd.DataFrame,
        splits: dict,
    ) -> dict:
        """Run backtest using FastBacktester."""
        from src.evaluation.fast_backtester import FastBacktester

        # Determine selection method (volatility factors use bottom_n)
        selection_method = spec.selection_method
        factor_lower = spec.factor_name.lower()
        if 'volatility' in factor_lower or 'vol' in factor_lower:
            if selection_method == 'top_n':
                selection_method = 'bottom_n'

        backtester = FastBacktester(
            algo_returns=algo_returns,
            features=features,
            factor_name=spec.factor_name,
            rebalance_frequency=spec.rebalance_frequency,
            lookback_window=spec.lookback_window,
            precompute_cov=(spec.optimizer in ['min_variance', 'max_sharpe']),
        )

        results = {}
        for period_name, start_key, end_key in [
            ('train', 'train_start', 'train_end'),
            ('val', 'val_start', 'val_end'),
            ('test', 'test_start', 'test_end'),
        ]:
            try:
                cache = backtester.prepare_period(splits[start_key], splits[end_key])
                result = backtester.run_with_cache(
                    cache,
                    optimizer=spec.optimizer,
                    selection_method=selection_method,
                    selection_param=int(spec.selection_param),
                    max_weight=DEFAULT_CONSTRAINTS['max_weight'],
                    shrinkage=spec.shrinkage,
                )
                results[period_name] = result
            except Exception:
                results[period_name] = {
                    'portfolio_returns': pd.Series(dtype=float),
                    'sharpe': np.nan,
                    'n_selected_avg': 0,
                }

        return results

    def _run_allocator_backtest(
        self,
        spec: TrialSpec,
        algo_returns: pd.DataFrame,
        splits: dict,
    ) -> dict:
        """Run backtest using allocator directly."""
        from src.baselines import (
            EqualWeightAllocator, RiskParityAllocator,
            MinVarianceAllocator, MaxSharpeAllocator,
            MomentumAllocator, VolTargetingAllocator,
        )
        from src.utils.numba_utils import compute_sharpe_from_returns

        # Create allocator
        common_kwargs = {
            'factor_name': spec.factor_name,
            'selection_method': spec.selection_method,
            'selection_param': spec.selection_param,
            'rebalance_frequency': spec.rebalance_frequency,
            **DEFAULT_CONSTRAINTS,
        }

        allocator_map = {
            'equal_weight': EqualWeightAllocator,
            'risk_parity': RiskParityAllocator,
            'min_variance': MinVarianceAllocator,
            'max_sharpe': MaxSharpeAllocator,
            'momentum': MomentumAllocator,
            'vol_targeting': VolTargetingAllocator,
        }

        allocator_class = allocator_map[spec.optimizer]
        allocator = allocator_class(lookback_window=spec.lookback_window, **common_kwargs)

        results = {}
        for period_name, start_key, end_key in [
            ('train', 'train_start', 'train_end'),
            ('val', 'val_start', 'val_end'),
            ('test', 'test_start', 'test_end'),
        ]:
            mask = (algo_returns.index >= splits[start_key]) & \
                   (algo_returns.index <= splits[end_key])
            period_returns = algo_returns.loc[mask]

            if len(period_returns) < 5:
                results[period_name] = {
                    'portfolio_returns': pd.Series(dtype=float),
                    'sharpe': np.nan,
                    'n_selected_avg': 0,
                }
                continue

            # Run backtest
            n_algos = period_returns.shape[1]
            weights = np.zeros(n_algos)
            portfolio_returns = []
            n_selected_list = []
            last_rebalance = None

            for date in period_returns.index:
                if allocator.should_rebalance(date, last_rebalance):
                    try:
                        hist_returns = algo_returns.loc[:date]
                        result = allocator.allocate(date, hist_returns, weights)
                        weights = result.weights
                        n_selected_list.append(result.n_selected)
                        last_rebalance = date
                    except Exception:
                        pass

                day_return = np.dot(weights, period_returns.loc[date].fillna(0).values)
                portfolio_returns.append(day_return)

            port_returns_arr = np.array(portfolio_returns)
            sharpe = compute_sharpe_from_returns(port_returns_arr, annualize=True)

            results[period_name] = {
                'portfolio_returns': pd.Series(portfolio_returns, index=period_returns.index),
                'sharpe': sharpe,
                'n_selected_avg': np.mean(n_selected_list) if n_selected_list else 0,
            }

        return results

    def _run_walk_forward(
        self,
        results_df: pd.DataFrame,
        algo_returns: pd.DataFrame,
        features: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        n_top: int = 5,
    ) -> list:
        """Run walk-forward validation for top configurations."""
        completed = results_df[results_df['status'] == 'completed']
        if len(completed) == 0:
            return []

        top_configs = completed.nlargest(n_top, 'sharpe_val')
        self.logger.info(f"Running walk-forward for top {len(top_configs)} configs")

        # Walk-forward parameters
        train_window = 252
        val_window = 63
        test_window = 63
        step_size = 63

        dates = algo_returns.index
        wf_results = []

        for _, config_row in top_configs.iterrows():
            spec = TrialSpec(
                factor_name=config_row['factor'],
                selection_param=config_row['n_select'],
                optimizer=config_row['optimizer'],
                lookback_window=config_row['lookback'],
            )

            fold = 0
            start_idx = 0

            while start_idx + train_window + val_window + test_window <= len(dates):
                train_end_idx = start_idx + train_window
                val_end_idx = train_end_idx + val_window
                test_end_idx = val_end_idx + test_window

                splits = {
                    'train_start': str(dates[start_idx].date()),
                    'train_end': str(dates[train_end_idx - 1].date()),
                    'val_start': str(dates[train_end_idx].date()),
                    'val_end': str(dates[val_end_idx - 1].date()),
                    'test_start': str(dates[val_end_idx].date()),
                    'test_end': str(dates[test_end_idx - 1].date()),
                }

                result = self._run_single_trial(
                    spec=spec,
                    algo_returns=algo_returns,
                    features=features,
                    benchmark_returns=benchmark_returns,
                    splits=splits,
                )

                result['fold'] = fold
                result['wf_train_start'] = splits['train_start']
                result['wf_test_end'] = splits['test_end']
                wf_results.append(result)

                fold += 1
                start_idx += step_size

        return wf_results

    def _generate_markdown_report(
        self,
        results: dict,
        results_df: pd.DataFrame,
        output_dir: Path,
    ):
        """Generate markdown summary report."""
        completed = results_df[results_df['status'] == 'completed']

        # Top 5 by Sharpe
        top5 = completed.nlargest(5, 'sharpe_val') if len(completed) > 0 else pd.DataFrame()
        top5_lines = []
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            top5_lines.append(
                f"| {i} | {row['optimizer']} | {row['factor']} | "
                f"{int(row['n_select'])} | {row['sharpe_val']:.4f} | {row['sharpe_test']:.4f} |"
            )

        # Best per baseline
        baseline_best = []
        for baseline in BASELINES:
            baseline_df = completed[completed['optimizer'] == baseline]
            if len(baseline_df) > 0:
                best = baseline_df.loc[baseline_df['sharpe_val'].idxmax()]
                baseline_best.append(
                    f"| {baseline} | {best['sharpe_val']:.4f} | {best['sharpe_test']:.4f} | "
                    f"{best['factor']} | {int(best['n_select'])} |"
                )

        report = f"""# Phase 3: Baseline Backtesting Summary

**Generated**: {datetime.now().isoformat()}

## Overview

Phase 3 tests classical portfolio allocation strategies as baselines
for comparison with the RL meta-allocator.

## Configuration

- Mode: {results.get('mode', 'standard')}
- Baseline filter: {results.get('baseline_filter', 'all')}
- Trials planned: {results.get('n_trials_planned', 0)}
- Trials completed: {results.get('n_trials_completed', 0)}
- Trials failed: {results.get('n_trials_failed', 0)}

## Date Splits

| Period | Start | End |
|--------|-------|-----|
| Train | {results.get('date_splits', {}).get('train_start', 'N/A')} | {results.get('date_splits', {}).get('train_end', 'N/A')} |
| Val | {results.get('date_splits', {}).get('val_start', 'N/A')} | {results.get('date_splits', {}).get('val_end', 'N/A')} |
| Test | {results.get('date_splits', {}).get('test_start', 'N/A')} | {results.get('date_splits', {}).get('test_end', 'N/A')} |

## Top 5 Configurations by Validation Sharpe

| Rank | Optimizer | Factor | N Select | Sharpe (Val) | Sharpe (Test) |
|------|-----------|--------|----------|--------------|---------------|
{chr(10).join(top5_lines)}

## Best Configuration per Baseline

| Baseline | Sharpe (Val) | Sharpe (Test) | Factor | N Select |
|----------|--------------|---------------|--------|----------|
{chr(10).join(baseline_best)}

## Best Overall Configuration

- **Optimizer**: {results.get('best_config', {}).get('optimizer', 'N/A')}
- **Factor**: {results.get('best_config', {}).get('factor', 'N/A')}
- **N Selected**: {results.get('best_config', {}).get('n_select', 'N/A')}
- **Lookback**: {results.get('best_config', {}).get('lookback', 'N/A')}
- **Sharpe (Val)**: {results.get('best_config', {}).get('sharpe_val', 0):.4f}
- **Sharpe (Test)**: {results.get('best_config', {}).get('sharpe_test', 0):.4f}

## Output Files

```
{output_dir}/
├── trials/
│   ├── results.csv            # All trial results
│   └── walk_forward.csv       # Walk-forward results (if run)
├── figures/                   # Visualizations (generated separately)
├── phase3_results.json        # Summary metrics
├── phase3_metrics.json        # Performance metrics
└── PHASE3_SUMMARY.md          # This report
```

## Next Steps

1. Review top configurations for consistency (val vs test Sharpe)
2. Use best baselines as benchmark for RL agent
3. Proceed to Phase 4: RL Environment Setup
4. Then Phase 5: RL Training
"""
        with open(output_dir / 'PHASE3_SUMMARY.md', 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info("Report saved to PHASE3_SUMMARY.md")

    def _print_summary(self, results_df: pd.DataFrame):
        """Print summary to console."""
        completed = results_df[results_df['status'] == 'completed']

        if len(completed) == 0:
            self.logger.warning("No completed trials to summarize")
            return

        self.logger.info("\n" + "=" * 60)
        self.logger.info("RESULTS SUMMARY")
        self.logger.info("=" * 60)

        best = completed.loc[completed['sharpe_val'].idxmax()]
        self.logger.info(f"Best configuration:")
        self.logger.info(f"  Optimizer: {best['optimizer']}")
        self.logger.info(f"  Factor: {best['factor']}")
        self.logger.info(f"  N Selected: {best['n_select']}")
        self.logger.info(f"  Sharpe (val): {best['sharpe_val']:.4f}")
        self.logger.info(f"  Sharpe (test): {best['sharpe_test']:.4f}")

        self.logger.info("\nTop 5 by Validation Sharpe:")
        for _, row in completed.nlargest(5, 'sharpe_val').iterrows():
            self.logger.info(
                f"  {row['optimizer']}|{row['factor']}|{int(row['n_select'])}: "
                f"val={row['sharpe_val']:.3f}, test={row['sharpe_test']:.3f}"
            )


def main():
    runner = Phase3Runner()
    runner.execute()


if __name__ == '__main__':
    main()
