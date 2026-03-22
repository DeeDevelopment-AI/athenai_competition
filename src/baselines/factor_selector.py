"""
Factor-based algorithm selection.

Implements factor models (like Fama-French) to select/rank algorithms
before portfolio optimization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class SelectionMethod(Enum):
    """How to select algorithms based on factor scores."""
    TOP_N = "top_n"                    # Select top N by factor
    TOP_PERCENTILE = "top_percentile"  # Select top X percentile
    THRESHOLD = "threshold"            # Select where factor > threshold
    BOTTOM_N = "bottom_n"              # Select bottom N (for short/inverse factors)
    LONG_SHORT = "long_short"          # Long top N, short bottom N


@dataclass
class FactorConfig:
    """Configuration for factor-based selection."""

    # Factor specification
    factor_name: str                          # Column name in features DataFrame
    selection_method: SelectionMethod = SelectionMethod.TOP_N
    selection_param: float = 100              # N for top_n, percentile for top_percentile, etc.

    # Multi-factor support
    secondary_factors: list[str] = None       # Additional factors for composite scoring
    factor_weights: list[float] = None        # Weights for composite (default: equal)

    # Filters before factor ranking
    min_observations: int = 20                # Minimum days of data required
    min_volatility: float = 1e-6              # Filter out zero-vol algos
    exclude_negative_return: bool = False     # Only consider positive return algos

    def __post_init__(self):
        if self.secondary_factors is None:
            self.secondary_factors = []
        if self.factor_weights is None:
            # Equal weight all factors
            n_factors = 1 + len(self.secondary_factors)
            self.factor_weights = [1.0 / n_factors] * n_factors


class FactorSelector:
    """
    Selects algorithms based on factor scores.

    Similar to Fama-French factor models:
    - Rank all algorithms by factor(s)
    - Select top performers
    - Pass to portfolio optimizer
    """

    def __init__(self, config: FactorConfig):
        self.config = config

    def select(
        self,
        date: pd.Timestamp,
        features: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select algorithms based on factor scores.

        Args:
            date: Current date for selection
            features: DataFrame with factor values [dates x (algos * features)]
            returns: DataFrame with returns [dates x algos] for filtering

        Returns:
            Tuple of (selected_indices, factor_scores)
            - selected_indices: Array of column indices in returns DataFrame
            - factor_scores: Normalized scores for selected algorithms
        """
        # Get available algorithms (columns in returns)
        algo_ids = returns.columns.tolist()
        n_algos = len(algo_ids)

        # Extract factor values for this date
        factor_scores = self._compute_factor_scores(date, features, algo_ids)

        # Apply pre-filters
        valid_mask = self._apply_filters(date, returns, algo_ids)

        # Combine: only rank valid algorithms
        factor_scores[~valid_mask] = np.nan

        # Select based on method
        selected_mask = self._select_by_method(factor_scores)

        # Get indices and scores for selected
        selected_indices = np.where(selected_mask)[0]
        selected_scores = factor_scores[selected_mask]

        # Normalize scores to [0, 1] for weighting
        if len(selected_scores) > 0 and not np.all(np.isnan(selected_scores)):
            min_score = np.nanmin(selected_scores)
            max_score = np.nanmax(selected_scores)
            if max_score > min_score:
                selected_scores = (selected_scores - min_score) / (max_score - min_score)
            else:
                selected_scores = np.ones_like(selected_scores)

        return selected_indices, selected_scores

    def _compute_factor_scores(
        self,
        date: pd.Timestamp,
        features: pd.DataFrame,
        algo_ids: list[str],
    ) -> np.ndarray:
        """Compute composite factor scores for all algorithms."""

        n_algos = len(algo_ids)
        scores = np.zeros(n_algos)

        # Get all factors to use
        all_factors = [self.config.factor_name] + self.config.secondary_factors
        weights = self.config.factor_weights

        for factor_name, weight in zip(all_factors, weights):
            factor_values = self._get_factor_values(date, features, algo_ids, factor_name)

            # Standardize factor (z-score) before combining
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
        features: pd.DataFrame,
        algo_ids: list[str],
        factor_name: str,
    ) -> np.ndarray:
        """Extract factor values for all algorithms at given date."""

        n_algos = len(algo_ids)
        values = np.full(n_algos, np.nan)

        # Features are stored as {algo}_{feature_name}
        # e.g., "fpJbh_rolling_sharpe_21d"

        # Find the row for this date (or closest before)
        if date in features.index:
            row = features.loc[date]
        else:
            # Get most recent date before this one
            valid_dates = features.index[features.index <= date]
            if len(valid_dates) == 0:
                return values
            row = features.loc[valid_dates[-1]]

        for i, algo_id in enumerate(algo_ids):
            col_name = f"{algo_id}_{factor_name}"
            if col_name in row.index:
                values[i] = row[col_name]

        return values

    def _apply_filters(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        algo_ids: list[str],
    ) -> np.ndarray:
        """Apply pre-selection filters."""

        n_algos = len(algo_ids)
        valid = np.ones(n_algos, dtype=bool)

        # Get returns up to date
        hist_returns = returns.loc[:date]

        # Filter by minimum observations
        if self.config.min_observations > 0:
            obs_count = hist_returns.notna().sum()
            valid &= (obs_count >= self.config.min_observations).values

        # Filter by minimum volatility
        if self.config.min_volatility > 0:
            vols = hist_returns.std()
            valid &= (vols >= self.config.min_volatility).values

        # Filter to positive return only
        if self.config.exclude_negative_return:
            total_ret = (1 + hist_returns.fillna(0)).prod() - 1
            valid &= (total_ret > 0).values

        return valid

    def _select_by_method(self, scores: np.ndarray) -> np.ndarray:
        """Select algorithms based on selection method."""

        n_algos = len(scores)
        valid_scores = ~np.isnan(scores)
        n_valid = valid_scores.sum()

        if n_valid == 0:
            return np.zeros(n_algos, dtype=bool)

        method = self.config.selection_method
        param = self.config.selection_param

        if method == SelectionMethod.TOP_N:
            n_select = min(int(param), n_valid)
            threshold = np.nanpercentile(scores, 100 * (1 - n_select / n_valid))
            return (scores >= threshold) & valid_scores

        elif method == SelectionMethod.TOP_PERCENTILE:
            threshold = np.nanpercentile(scores, 100 - param)
            return (scores >= threshold) & valid_scores

        elif method == SelectionMethod.THRESHOLD:
            return (scores >= param) & valid_scores

        elif method == SelectionMethod.BOTTOM_N:
            n_select = min(int(param), n_valid)
            threshold = np.nanpercentile(scores, 100 * n_select / n_valid)
            return (scores <= threshold) & valid_scores

        elif method == SelectionMethod.LONG_SHORT:
            # Select both top and bottom N
            n_select = min(int(param), n_valid // 2)
            top_threshold = np.nanpercentile(scores, 100 * (1 - n_select / n_valid))
            bottom_threshold = np.nanpercentile(scores, 100 * n_select / n_valid)
            return ((scores >= top_threshold) | (scores <= bottom_threshold)) & valid_scores

        return valid_scores


# Predefined factor configurations for common strategies
MOMENTUM_FACTORS = FactorConfig(
    factor_name="rolling_return_21d",
    secondary_factors=["rolling_return_63d"],
    selection_method=SelectionMethod.TOP_N,
    selection_param=100,
)

QUALITY_FACTORS = FactorConfig(
    factor_name="rolling_sharpe_21d",
    secondary_factors=["rolling_profit_factor_21d"],
    selection_method=SelectionMethod.TOP_N,
    selection_param=100,
)

LOW_VOL_FACTORS = FactorConfig(
    factor_name="rolling_volatility_21d",
    selection_method=SelectionMethod.BOTTOM_N,  # Select lowest volatility
    selection_param=100,
)

TREND_FACTORS = FactorConfig(
    factor_name="rolling_return_63d",
    secondary_factors=["rolling_sharpe_63d", "rolling_calmar_63d"],
    selection_method=SelectionMethod.TOP_N,
    selection_param=100,
)
