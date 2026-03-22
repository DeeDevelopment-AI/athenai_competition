"""
Base interfaces for allocators.

Provides:
- BaseAllocator: Simple allocator interface
- FactorBasedAllocator: Factor selection + portfolio optimization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AllocationResult:
    """Result of an allocation."""

    weights: np.ndarray
    turnover: float
    constraint_violations: list[str]
    n_selected: int = 0  # Number of algorithms selected by factor


class BaseAllocator(ABC):
    """
    Base class for all allocators.

    All allocators must implement compute_weights().
    Constraints are applied automatically by apply_constraints().
    """

    def __init__(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.00,
        max_turnover: float = 0.30,
        max_exposure: float = 1.0,
        rebalance_frequency: str = "weekly",
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_turnover = max_turnover
        self.max_exposure = max_exposure
        self.rebalance_frequency = rebalance_frequency

    @abstractmethod
    def compute_weights(
        self,
        date: pd.Timestamp,
        algo_data: pd.DataFrame,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute target weights for the given date.

        Args:
            date: Calculation date.
            algo_data: DataFrame with algorithm returns up to date.
            current_weights: Current portfolio weights.

        Returns:
            Target weight vector (before constraints).
        """
        pass

    def apply_constraints(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        n_selected: int = 0,
    ) -> AllocationResult:
        """
        Apply constraints to target weights.

        Args:
            target_weights: Desired weights.
            current_weights: Current weights.
            n_selected: Number of algorithms selected by factor.

        Returns:
            AllocationResult with adjusted weights.
        """
        violations = []
        weights = target_weights.copy()

        # 1. Clip by min/max weight
        weights = np.clip(weights, self.min_weight, self.max_weight)
        if not np.allclose(weights, target_weights):
            violations.append("weight_bounds")

        # 2. Normalize if sum > max_exposure
        total = weights.sum()
        if total > self.max_exposure:
            weights = weights * (self.max_exposure / total)
            violations.append("max_exposure")

        # 3. Limit turnover
        turnover = np.abs(weights - current_weights).sum() / 2
        if turnover > self.max_turnover:
            # Proportionally reduce the change
            scale = self.max_turnover / turnover
            weights = current_weights + scale * (weights - current_weights)
            turnover = self.max_turnover
            violations.append("max_turnover")

        return AllocationResult(
            weights=weights,
            turnover=turnover,
            constraint_violations=violations,
            n_selected=n_selected,
        )

    def allocate(
        self,
        date: pd.Timestamp,
        algo_data: pd.DataFrame,
        current_weights: np.ndarray,
    ) -> AllocationResult:
        """
        Main method: compute weights and apply constraints.

        Args:
            date: Calculation date.
            algo_data: DataFrame with algorithm returns.
            current_weights: Current weights.

        Returns:
            AllocationResult with final weights.
        """
        target_weights = self.compute_weights(date, algo_data, current_weights)
        return self.apply_constraints(target_weights, current_weights)

    def should_rebalance(self, date: pd.Timestamp, last_rebalance: Optional[pd.Timestamp]) -> bool:
        """
        Determine if it's time to rebalance.

        Args:
            date: Current date.
            last_rebalance: Date of last rebalance.

        Returns:
            True if should rebalance.
        """
        if last_rebalance is None:
            return True

        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return (date - last_rebalance).days >= 5
        elif self.rebalance_frequency == "monthly":
            return (date - last_rebalance).days >= 21
        elif self.rebalance_frequency == "quarterly":
            return (date - last_rebalance).days >= 63

        return True


class FactorBasedAllocator(BaseAllocator):
    """
    Factor-based allocator: Factor Selection + Portfolio Optimization.

    Architecture:
    1. Factor Model: Select/rank algorithms using features (like Fama-French)
    2. Portfolio Optimizer: Weight selected algorithms
    3. Constraints: Match benchmark characteristics

    Subclasses implement _optimize_portfolio() for different optimization methods.

    Factor Direction:
    - Some factors should select TOP values (higher is better): sharpe, return, profit_factor, calmar, drawdown (less negative is better)
    - Some factors should select BOTTOM values (lower is better): volatility
    - Use auto_direction=True to automatically detect factor direction

    Note on drawdown: Drawdown values are negative (e.g., -0.10 = 10% drawdown).
    Higher values (closer to 0) are better, so drawdown uses top_n selection.
    """

    # Factors where LOWER is BETTER (select bottom N)
    INVERSE_FACTORS = {
        'volatility', 'vol', 'rolling_volatility',
    }

    def __init__(
        self,
        # Factor selection parameters
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,
        min_observations: int = 20,
        auto_direction: bool = True,  # Auto-detect if factor should use bottom_n

        # Optimization parameters
        lookback_window: int = 63,

        # Features data (set externally)
        features: pd.DataFrame = None,

        # Base constraints
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.factor_name = factor_name
        self.secondary_factors = secondary_factors or []
        self.selection_param = selection_param
        self.min_observations = min_observations
        self.lookback_window = lookback_window
        self.auto_direction = auto_direction

        # Auto-detect selection method based on factor name
        if auto_direction and selection_method in ("top_n", "bottom_n"):
            self.selection_method = self._detect_selection_method(factor_name, selection_method)
        else:
            self.selection_method = selection_method

        # Features DataFrame - must be set before allocating
        self._features = features

    def _detect_selection_method(self, factor_name: str, default: str) -> str:
        """Auto-detect if factor should use bottom_n (inverse selection)."""
        factor_lower = factor_name.lower()
        for inverse_key in self.INVERSE_FACTORS:
            if inverse_key in factor_lower:
                # For volatility/drawdown, we want LOWEST values
                if default == "top_n":
                    return "bottom_n"
                elif default == "bottom_n":
                    return "top_n"  # Double negative
        return default

    def set_features(self, features: pd.DataFrame):
        """Set the features DataFrame for factor selection."""
        self._features = features

    def compute_weights(
        self,
        date: pd.Timestamp,
        algo_data: pd.DataFrame,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weights using factor selection + optimization.

        Args:
            date: Calculation date.
            algo_data: DataFrame with algorithm returns.
            current_weights: Current weights.

        Returns:
            Target weight vector.
        """
        n_algos = len(current_weights)

        # Step 1: Select algorithms using factor model
        selected_indices, factor_scores = self._select_by_factor(date, algo_data)

        if len(selected_indices) == 0:
            # Fallback to equal weight if no selection
            return np.ones(n_algos) / n_algos

        # Step 2: Get returns for selected algorithms
        selected_returns = algo_data.iloc[:, selected_indices]

        # Step 3: Optimize portfolio over selected algorithms
        selected_weights = self._optimize_portfolio(
            date=date,
            returns=selected_returns,
            factor_scores=factor_scores,
            current_weights=current_weights[selected_indices] if len(current_weights) > 0 else None,
        )

        # Step 4: Expand back to full universe
        full_weights = np.zeros(n_algos)
        full_weights[selected_indices] = selected_weights

        return full_weights

    def _select_by_factor(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select algorithms based on factor scores.

        Returns:
            Tuple of (selected_indices, factor_scores)
        """
        algo_ids = returns.columns.tolist()
        n_algos = len(algo_ids)

        # Compute factor scores
        factor_scores = self._compute_factor_scores(date, algo_ids, returns)

        # Apply pre-filters
        valid_mask = self._apply_filters(date, returns)
        factor_scores[~valid_mask] = np.nan

        # Select based on method
        selected_mask = self._select_by_method(factor_scores)

        selected_indices = np.where(selected_mask)[0]
        selected_scores = factor_scores[selected_mask]

        # Normalize scores to [0, 1]
        if len(selected_scores) > 0 and not np.all(np.isnan(selected_scores)):
            min_s = np.nanmin(selected_scores)
            max_s = np.nanmax(selected_scores)
            if max_s > min_s:
                selected_scores = (selected_scores - min_s) / (max_s - min_s)
            else:
                selected_scores = np.ones_like(selected_scores)

        return selected_indices, selected_scores

    def _compute_factor_scores(
        self,
        date: pd.Timestamp,
        algo_ids: list[str],
        returns: pd.DataFrame,
    ) -> np.ndarray:
        """Compute factor scores for all algorithms."""

        n_algos = len(algo_ids)
        all_factors = [self.factor_name] + self.secondary_factors
        n_factors = len(all_factors)
        weights = [1.0 / n_factors] * n_factors

        scores = np.zeros(n_algos)

        for factor_name, weight in zip(all_factors, weights):
            factor_values = self._get_factor_values(date, algo_ids, factor_name, returns)

            # Z-score normalize
            valid = ~np.isnan(factor_values)
            if valid.sum() > 1:
                mean = np.nanmean(factor_values)
                std = np.nanstd(factor_values)
                if std > 1e-8:
                    factor_values = (factor_values - mean) / std

            scores += weight * np.nan_to_num(factor_values, nan=0.0)

        return scores

    def _get_factor_values(
        self,
        date: pd.Timestamp,
        algo_ids: list[str],
        factor_name: str,
        returns: pd.DataFrame,
    ) -> np.ndarray:
        """Extract factor values from features DataFrame or compute from returns."""

        n_algos = len(algo_ids)
        values = np.full(n_algos, np.nan)

        # Try to get from features DataFrame
        if self._features is not None:
            # Find the row for this date
            if date in self._features.index:
                row = self._features.loc[date]
            else:
                valid_dates = self._features.index[self._features.index <= date]
                if len(valid_dates) > 0:
                    row = self._features.loc[valid_dates[-1]]
                else:
                    row = None

            if row is not None:
                for i, algo_id in enumerate(algo_ids):
                    col_name = f"{algo_id}_{factor_name}"
                    if col_name in row.index:
                        values[i] = row[col_name]

                # If we got values, return them
                if not np.all(np.isnan(values)):
                    return values

        # Fallback: compute factor from returns
        return self._compute_factor_from_returns(date, returns, factor_name)

    def _compute_factor_from_returns(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_name: str,
    ) -> np.ndarray:
        """Compute factor values directly from returns when features unavailable."""

        hist = returns.loc[:date].tail(self.lookback_window)

        if len(hist) < self.min_observations:
            return np.full(len(returns.columns), np.nan)

        if "return" in factor_name:
            # Cumulative return
            return (1 + hist).prod().values - 1

        elif "volatility" in factor_name or "vol" in factor_name:
            return hist.std().values * np.sqrt(252)

        elif "sharpe" in factor_name:
            mean = hist.mean().values
            std = hist.std().values
            std = np.where(std < 1e-8, np.nan, std)
            return (mean / std) * np.sqrt(252)

        elif "drawdown" in factor_name:
            # Max drawdown (negative values, higher is better)
            cum = (1 + hist).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / running_max
            return dd.min().values

        elif "calmar" in factor_name:
            ret = (1 + hist).prod().values - 1
            cum = (1 + hist).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / running_max
            max_dd = dd.min().values
            max_dd = np.where(np.abs(max_dd) < 1e-8, np.nan, max_dd)
            return ret / np.abs(max_dd)

        elif "profit_factor" in factor_name:
            gains = hist.where(hist > 0, 0).sum()
            losses = (-hist.where(hist < 0, 0)).sum()
            losses = losses.where(losses > 1e-8, np.nan)
            return (gains / losses).values

        else:
            # Default: cumulative return
            return (1 + hist).prod().values - 1

    def _apply_filters(self, date: pd.Timestamp, returns: pd.DataFrame) -> np.ndarray:
        """Apply pre-selection filters."""

        n_algos = len(returns.columns)
        valid = np.ones(n_algos, dtype=bool)

        hist = returns.loc[:date]

        # Minimum observations
        if self.min_observations > 0:
            obs_count = hist.notna().sum()
            valid &= (obs_count >= self.min_observations).values

        # Minimum volatility (filter out dead algorithms)
        min_vol = 1e-8
        vols = hist.std()
        valid &= (vols >= min_vol).values

        return valid

    def _select_by_method(self, scores: np.ndarray) -> np.ndarray:
        """Select algorithms based on selection method."""

        n_algos = len(scores)
        valid_mask = ~np.isnan(scores)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            return np.zeros(n_algos, dtype=bool)

        param = self.selection_param
        selected = np.zeros(n_algos, dtype=bool)

        if self.selection_method == "top_n":
            n_select = min(int(param), n_valid)
            if n_select == 0:
                return selected
            # Use argsort to get exact top N (handles ties properly)
            scores_copy = np.copy(scores)
            scores_copy[~valid_mask] = -np.inf
            top_indices = np.argsort(scores_copy)[-n_select:]
            selected[top_indices] = True
            return selected

        elif self.selection_method == "top_percentile":
            threshold = np.nanpercentile(scores, 100 - param)
            return (scores >= threshold) & valid_mask

        elif self.selection_method == "threshold":
            return (scores >= param) & valid_mask

        elif self.selection_method == "bottom_n":
            n_select = min(int(param), n_valid)
            if n_select == 0:
                return selected
            scores_copy = np.copy(scores)
            scores_copy[~valid_mask] = np.inf
            bottom_indices = np.argsort(scores_copy)[:n_select]
            selected[bottom_indices] = True
            return selected

        return valid_mask

    @abstractmethod
    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Optimize portfolio over selected algorithms.

        Args:
            date: Calculation date.
            returns: Returns of SELECTED algorithms only.
            factor_scores: Normalized factor scores for selected algorithms.
            current_weights: Current weights for selected algorithms.

        Returns:
            Weight vector for selected algorithms (sums to 1).
        """
        pass

    def allocate(
        self,
        date: pd.Timestamp,
        algo_data: pd.DataFrame,
        current_weights: np.ndarray,
    ) -> AllocationResult:
        """Main allocation method with n_selected tracking."""
        target_weights = self.compute_weights(date, algo_data, current_weights)
        n_selected = (target_weights > 0).sum()
        return self.apply_constraints(target_weights, current_weights, n_selected)
