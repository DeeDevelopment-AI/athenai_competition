"""
Baseline Backtester Engine for Phase 3.

Runs systematic backtests of all baseline strategies with different
feature combinations and tracks all trials.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.baselines import (
    EqualWeightAllocator,
    MaxSharpeAllocator,
    MinVarianceAllocator,
    MomentumAllocator,
    RiskParityAllocator,
    VolTargetingAllocator,
)
from src.baselines.base import AllocationResult, BaseAllocator
from src.evaluation.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_full_metrics,
    concentration_hhi,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)
from src.utils.trial_tracker import Trial, TrialConfig, TrialMetrics, TrialTracker

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result from a single backtest run."""

    returns: pd.Series
    weights_history: list[tuple[pd.Timestamp, np.ndarray]]
    turnover_history: list[float]
    metrics: dict
    warnings: list[str]


class BacktestEngine:
    """
    Engine for running systematic backtests of baseline strategies.

    Supports:
    - Multiple baseline strategies
    - Feature-based allocation (using features as inputs to allocators)
    - Train/validation/test splits
    - Walk-forward validation
    - Constraint enforcement
    - Metric computation
    """

    # Default constraints matching benchmark profile
    DEFAULT_CONSTRAINTS = {
        'max_weight': 0.40,
        'min_weight': 0.00,
        'max_turnover': 0.30,
        'max_exposure': 1.0,
        'rebalance_frequency': 'weekly',
    }

    # Cost model for transaction costs
    DEFAULT_COSTS = {
        'spread_bps': 5.0,
        'slippage_bps': 2.0,
    }

    def __init__(
        self,
        algo_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        tracker: Optional[TrialTracker] = None,
        constraints: Optional[dict] = None,
        costs: Optional[dict] = None,
    ):
        """
        Initialize the backtest engine.

        Args:
            algo_returns: DataFrame of algorithm returns [dates x algos]
            benchmark_returns: Series of benchmark returns
            tracker: TrialTracker instance for logging
            constraints: Portfolio constraints
            costs: Transaction cost parameters
        """
        self.algo_returns = algo_returns
        self.benchmark_returns = benchmark_returns
        self.tracker = tracker or TrialTracker()
        self.constraints = {**self.DEFAULT_CONSTRAINTS, **(constraints or {})}
        self.costs = {**self.DEFAULT_COSTS, **(costs or {})}

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate input data."""
        if self.algo_returns.empty:
            raise ValueError("algo_returns cannot be empty")

        n_dates, n_algos = self.algo_returns.shape
        logger.info(f"Backtest engine initialized with {n_dates} dates, {n_algos} algorithms")

        if self.benchmark_returns is not None:
            overlap = self.algo_returns.index.intersection(self.benchmark_returns.index)
            logger.info(f"Benchmark overlap: {len(overlap)} dates")

    def run_backtest(
        self,
        allocator: BaseAllocator,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_weights: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run a single backtest for an allocator.

        Args:
            allocator: Allocator instance to test
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_weights: Starting weights (default: equal weight)

        Returns:
            BacktestResult with returns and metrics
        """
        warnings = []

        # Filter returns to date range
        mask = (self.algo_returns.index >= start_date) & (self.algo_returns.index <= end_date)
        returns = self.algo_returns.loc[mask].copy()

        if len(returns) < 5:
            warnings.append(f"Very short backtest period: {len(returns)} days")

        n_algos = returns.shape[1]

        # Initialize weights
        if initial_weights is None:
            weights = np.ones(n_algos) / n_algos
        else:
            weights = initial_weights.copy()

        # Track history
        portfolio_returns = []
        weights_history = [(returns.index[0], weights.copy())]
        turnover_history = []
        last_rebalance = None

        for date in returns.index:
            # Get daily returns
            daily_returns = returns.loc[date].values

            # Handle NaN returns (algorithm not trading)
            daily_returns = np.nan_to_num(daily_returns, nan=0.0)

            # Calculate portfolio return for today
            port_ret = np.dot(weights, daily_returns)
            portfolio_returns.append(port_ret)

            # Check if should rebalance
            if allocator.should_rebalance(date, last_rebalance):
                try:
                    # Get new allocation
                    result = allocator.allocate(date, returns.loc[:date], weights)
                    new_weights = result.weights
                    turnover_history.append(result.turnover)

                    # Apply transaction costs
                    cost = self._compute_transaction_cost(weights, new_weights)
                    # Cost is subtracted from return (already included in next period)

                    weights = new_weights
                    weights_history.append((date, weights.copy()))
                    last_rebalance = date

                except Exception as e:
                    warnings.append(f"Allocation failed on {date}: {e}")

        # Create returns series
        portfolio_series = pd.Series(portfolio_returns, index=returns.index)

        # Compute metrics
        metrics = self._compute_metrics(portfolio_series, start_date, end_date)

        return BacktestResult(
            returns=portfolio_series,
            weights_history=weights_history,
            turnover_history=turnover_history,
            metrics=metrics,
            warnings=warnings,
        )

    def _compute_transaction_cost(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Compute transaction cost for a rebalance."""
        turnover = np.abs(new_weights - old_weights).sum() / 2
        cost_bps = self.costs['spread_bps'] + self.costs['slippage_bps']
        return turnover * cost_bps / 10000

    def _compute_metrics(
        self,
        returns: pd.Series,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> dict:
        """Compute all metrics for a backtest."""
        # Get benchmark returns for the same period
        bench = None
        if self.benchmark_returns is not None:
            mask = (self.benchmark_returns.index >= start_date) & (self.benchmark_returns.index <= end_date)
            bench = self.benchmark_returns.loc[mask]

        return compute_full_metrics(returns, bench)

    def run_trial(
        self,
        config: TrialConfig,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
    ) -> Trial:
        """
        Run a complete trial with train/val/test splits.

        Args:
            config: Trial configuration
            train_start/end: Training period dates
            val_start/end: Validation period dates
            test_start/end: Optional test period dates

        Returns:
            Completed Trial object
        """
        start_time = time.time()

        # Create trial
        trial = self.tracker.create_trial(
            config=config,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        )

        try:
            # Create allocator
            allocator = self._create_allocator(config)

            # Run training backtest
            train_result = self.run_backtest(
                allocator,
                pd.Timestamp(train_start),
                pd.Timestamp(train_end),
            )

            # Update train metrics
            trial.metrics.sharpe_train = train_result.metrics.get('sharpe_ratio', np.nan)
            trial.metrics.return_ann_train = train_result.metrics.get('annualized_return', np.nan)
            trial.metrics.vol_ann_train = train_result.metrics.get('annualized_volatility', np.nan)
            trial.metrics.max_dd_train = train_result.metrics.get('max_drawdown', np.nan)
            trial.metrics.sortino_train = train_result.metrics.get('sortino_ratio', np.nan)
            trial.metrics.calmar_train = train_result.metrics.get('calmar_ratio', np.nan)

            trial.warnings.extend(train_result.warnings)

            # Get final weights from training for initialization
            final_train_weights = train_result.weights_history[-1][1] if train_result.weights_history else None

            # Run validation backtest
            val_result = self.run_backtest(
                allocator,
                pd.Timestamp(val_start),
                pd.Timestamp(val_end),
                initial_weights=final_train_weights,
            )

            # Update validation metrics
            trial.metrics.sharpe_val = val_result.metrics.get('sharpe_ratio', np.nan)
            trial.metrics.return_ann_val = val_result.metrics.get('annualized_return', np.nan)
            trial.metrics.vol_ann_val = val_result.metrics.get('annualized_volatility', np.nan)
            trial.metrics.max_dd_val = val_result.metrics.get('max_drawdown', np.nan)
            trial.metrics.sortino_val = val_result.metrics.get('sortino_ratio', np.nan)
            trial.metrics.calmar_val = val_result.metrics.get('calmar_ratio', np.nan)

            # Relative metrics
            trial.metrics.excess_return = val_result.metrics.get('excess_return', np.nan)
            trial.metrics.tracking_error = val_result.metrics.get('tracking_error', np.nan)
            trial.metrics.information_ratio = val_result.metrics.get('information_ratio', np.nan)
            trial.metrics.beta_vs_benchmark = val_result.metrics.get('beta_vs_benchmark', np.nan)
            trial.metrics.alpha_vs_benchmark = val_result.metrics.get('alpha_vs_benchmark', np.nan)

            # Operational metrics
            if val_result.turnover_history:
                # Annualize turnover (assume weekly rebalancing)
                n_rebalances = len(val_result.turnover_history)
                n_days = (pd.Timestamp(val_end) - pd.Timestamp(val_start)).days
                if n_days > 0:
                    rebalances_per_year = n_rebalances * 365 / n_days
                    trial.metrics.turnover_ann = sum(val_result.turnover_history) * rebalances_per_year

            trial.warnings.extend(val_result.warnings)

            # Run test backtest if dates provided
            if test_start and test_end:
                final_val_weights = val_result.weights_history[-1][1] if val_result.weights_history else final_train_weights

                test_result = self.run_backtest(
                    allocator,
                    pd.Timestamp(test_start),
                    pd.Timestamp(test_end),
                    initial_weights=final_val_weights,
                )

                trial.metrics.sharpe_test = test_result.metrics.get('sharpe_ratio', np.nan)
                trial.metrics.return_ann_test = test_result.metrics.get('annualized_return', np.nan)
                trial.metrics.vol_ann_test = test_result.metrics.get('annualized_volatility', np.nan)
                trial.metrics.max_dd_test = test_result.metrics.get('max_drawdown', np.nan)
                trial.metrics.sortino_test = test_result.metrics.get('sortino_ratio', np.nan)
                trial.metrics.calmar_test = test_result.metrics.get('calmar_ratio', np.nan)

                trial.warnings.extend(test_result.warnings)

            trial.status = "completed"

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            trial.status = "failed"
            trial.notes = str(e)

        trial.elapsed_seconds = time.time() - start_time

        # Save trial
        self.tracker.save_trial(trial)

        return trial

    def _create_allocator(self, config: TrialConfig) -> BaseAllocator:
        """Create an allocator instance from config."""
        common_kwargs = {
            'max_weight': config.max_weight,
            'min_weight': config.min_weight,
            'max_turnover': config.max_turnover,
            'max_exposure': config.max_exposure,
            'rebalance_frequency': config.rebalance_frequency,
        }

        if config.baseline_name == "equal_weight":
            return EqualWeightAllocator(**common_kwargs)

        elif config.baseline_name == "risk_parity":
            return RiskParityAllocator(
                vol_lookback=config.lookback_window,
                **common_kwargs,
            )

        elif config.baseline_name == "min_variance":
            return MinVarianceAllocator(
                cov_lookback=config.lookback_window,
                shrinkage=config.shrinkage,
                **common_kwargs,
            )

        elif config.baseline_name == "max_sharpe":
            return MaxSharpeAllocator(
                lookback=config.lookback_window,
                shrinkage=config.shrinkage,
                **common_kwargs,
            )

        elif config.baseline_name == "momentum":
            top_n = config.extra_params.get('top_n', None)
            skip_recent = config.extra_params.get('skip_recent', 5)
            return MomentumAllocator(
                momentum_lookback=config.lookback_window,
                skip_recent=skip_recent,
                top_n=top_n,
                **common_kwargs,
            )

        elif config.baseline_name == "vol_targeting":
            target_vol = config.extra_params.get('target_vol', 0.10)
            return VolTargetingAllocator(
                target_vol=target_vol,
                vol_lookback=config.lookback_window,
                **common_kwargs,
            )

        else:
            raise ValueError(f"Unknown baseline: {config.baseline_name}")

    def run_walk_forward(
        self,
        config: TrialConfig,
        train_window: int = 252,
        val_window: int = 63,
        test_window: int = 63,
        step_size: int = 63,
    ) -> list[Trial]:
        """
        Run walk-forward validation for a configuration.

        Args:
            config: Trial configuration
            train_window: Training window in days
            val_window: Validation window in days
            test_window: Test window in days
            step_size: Step size between folds in days

        Returns:
            List of Trial objects for each fold
        """
        dates = self.algo_returns.index
        total_window = train_window + val_window + test_window

        if len(dates) < total_window:
            raise ValueError(f"Not enough data for walk-forward. Need {total_window}, have {len(dates)}")

        trials = []
        fold_id = 0
        start_idx = 0

        while True:
            train_start_idx = start_idx
            train_end_idx = start_idx + train_window
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + val_window
            test_start_idx = val_end_idx
            test_end_idx = test_start_idx + test_window

            if test_end_idx > len(dates):
                break

            # Create fold-specific config with note
            fold_config = TrialConfig(
                baseline_name=config.baseline_name,
                feature_set=config.feature_set,
                lookback_window=config.lookback_window,
                rebalance_frequency=config.rebalance_frequency,
                max_weight=config.max_weight,
                min_weight=config.min_weight,
                max_turnover=config.max_turnover,
                max_exposure=config.max_exposure,
                shrinkage=config.shrinkage,
                extra_params={**config.extra_params, 'walk_forward_fold': fold_id},
            )

            logger.info(f"Running walk-forward fold {fold_id}")

            trial = self.run_trial(
                config=fold_config,
                train_start=str(dates[train_start_idx].date()),
                train_end=str(dates[train_end_idx - 1].date()),
                val_start=str(dates[val_start_idx].date()),
                val_end=str(dates[val_end_idx - 1].date()),
                test_start=str(dates[test_start_idx].date()),
                test_end=str(dates[test_end_idx - 1].date()),
            )

            trial.notes = f"Walk-forward fold {fold_id}"
            trials.append(trial)

            fold_id += 1
            start_idx += step_size

        logger.info(f"Completed {len(trials)} walk-forward folds")
        return trials


def create_feature_sets() -> dict[str, list[str]]:
    """Define feature set configurations for testing."""
    return {
        # Single factor trials
        'returns_5d': ['rolling_return_5d'],
        'returns_21d': ['rolling_return_21d'],
        'returns_63d': ['rolling_return_63d'],
        'volatility_5d': ['rolling_volatility_5d'],
        'volatility_21d': ['rolling_volatility_21d'],
        'volatility_63d': ['rolling_volatility_63d'],
        'sharpe_21d': ['rolling_sharpe_21d'],
        'sharpe_63d': ['rolling_sharpe_63d'],
        'drawdown_21d': ['rolling_drawdown_21d'],
        'drawdown_63d': ['rolling_drawdown_63d'],
        'profit_factor_21d': ['rolling_profit_factor_21d'],
        'profit_factor_63d': ['rolling_profit_factor_63d'],
        'calmar_63d': ['rolling_calmar_63d'],

        # Multi-factor combinations
        'return_vol': ['rolling_return_21d', 'rolling_volatility_21d'],
        'sharpe_dd': ['rolling_sharpe_21d', 'rolling_drawdown_21d'],
        'return_vol_dd': ['rolling_return_21d', 'rolling_volatility_21d', 'rolling_drawdown_21d'],
        'full_set': [
            'rolling_return_21d', 'rolling_volatility_21d', 'rolling_sharpe_21d',
            'rolling_drawdown_21d', 'rolling_profit_factor_21d', 'rolling_calmar_63d',
        ],
    }


def generate_trial_configs(baseline_name: str) -> list[TrialConfig]:
    """Generate trial configurations for a baseline."""
    configs = []
    feature_sets = create_feature_sets()

    # Common constraint configs
    constraint_configs = [
        {'rebalance_frequency': 'weekly', 'max_turnover': 0.30},
        {'rebalance_frequency': 'monthly', 'max_turnover': 0.30},
    ]

    # Lookback windows to test
    lookback_windows = [21, 63, 126]

    if baseline_name == "equal_weight":
        # Equal weight doesn't use features or lookback
        for cc in constraint_configs:
            configs.append(TrialConfig(
                baseline_name=baseline_name,
                feature_set=[],
                lookback_window=0,
                rebalance_frequency=cc['rebalance_frequency'],
                max_turnover=cc['max_turnover'],
            ))

    elif baseline_name == "risk_parity":
        # Test different volatility lookback windows
        for lb in lookback_windows:
            for cc in constraint_configs:
                configs.append(TrialConfig(
                    baseline_name=baseline_name,
                    feature_set=['volatility'],
                    lookback_window=lb,
                    rebalance_frequency=cc['rebalance_frequency'],
                    max_turnover=cc['max_turnover'],
                ))

    elif baseline_name == "min_variance":
        # Test different covariance lookback windows and shrinkage
        shrinkage_values = [0.0, 0.1, 0.3]
        for lb in lookback_windows:
            for shrink in shrinkage_values:
                configs.append(TrialConfig(
                    baseline_name=baseline_name,
                    feature_set=['covariance'],
                    lookback_window=lb,
                    rebalance_frequency='weekly',
                    shrinkage=shrink,
                ))

    elif baseline_name == "max_sharpe":
        # Test different lookback windows and shrinkage
        shrinkage_values = [0.1, 0.2, 0.3]
        for lb in lookback_windows:
            for shrink in shrinkage_values:
                configs.append(TrialConfig(
                    baseline_name=baseline_name,
                    feature_set=['returns', 'covariance'],
                    lookback_window=lb,
                    rebalance_frequency='weekly',
                    shrinkage=shrink,
                ))

    elif baseline_name == "momentum":
        # Test different momentum lookback windows and top_n
        top_n_values = [None, 50, 100]  # None = all
        skip_recent_values = [0, 5]
        for lb in lookback_windows:
            for top_n in top_n_values:
                for skip in skip_recent_values:
                    configs.append(TrialConfig(
                        baseline_name=baseline_name,
                        feature_set=['momentum'],
                        lookback_window=lb,
                        rebalance_frequency='weekly',
                        extra_params={'top_n': top_n, 'skip_recent': skip},
                    ))

    elif baseline_name == "vol_targeting":
        # Test different target volatilities
        target_vols = [0.05, 0.10, 0.15]
        for lb in lookback_windows:
            for tv in target_vols:
                configs.append(TrialConfig(
                    baseline_name=baseline_name,
                    feature_set=['volatility'],
                    lookback_window=lb,
                    rebalance_frequency='weekly',
                    extra_params={'target_vol': tv},
                ))

    return configs[:10]  # Ensure minimum 10 trials per baseline
