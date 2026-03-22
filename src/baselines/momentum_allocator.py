"""
Momentum Allocator with Factor Selection.

Architecture:
1. Factor Model: Select top N algorithms by momentum factor
2. Portfolio Optimizer: Weight by momentum score or equal weight
3. Constraints: Match benchmark characteristics
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import FactorBasedAllocator


class MomentumAllocator(FactorBasedAllocator):
    """
    Momentum-based allocation on factor-selected algorithms.

    Steps:
    1. Rank all algorithms by momentum (return over lookback)
    2. Select top N by momentum
    3. Weight by momentum score (normalized) or equal weight
    4. Apply benchmark constraints
    """

    def __init__(
        self,
        # Factor selection - use momentum as the factor
        factor_name: str = "rolling_return_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # Momentum parameters
        momentum_lookback: int = 63,
        skip_recent: int = 5,
        weight_by_score: bool = True,  # Weight by momentum or equal weight

        # Base parameters
        **kwargs,
    ):
        # Default to momentum factors
        if secondary_factors is None:
            secondary_factors = ["rolling_return_63d"]

        super().__init__(
            factor_name=factor_name,
            secondary_factors=secondary_factors,
            selection_method=selection_method,
            selection_param=selection_param,
            lookback_window=momentum_lookback,
            **kwargs,
        )
        self.momentum_lookback = momentum_lookback
        self.skip_recent = skip_recent
        self.weight_by_score = weight_by_score

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Momentum-weighted optimization.

        Args:
            date: Calculation date.
            returns: Returns of selected algorithms.
            factor_scores: Normalized momentum scores.
            current_weights: Current weights (not used).

        Returns:
            Momentum-weighted or equal weights summing to 1.
        """
        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        if not self.weight_by_score:
            # Equal weight among selected
            return np.ones(n_selected) / n_selected

        # Weight by momentum score
        # factor_scores are already normalized [0, 1]
        scores = np.maximum(factor_scores, 0)

        if scores.sum() < 1e-10:
            return np.ones(n_selected) / n_selected

        weights = scores / scores.sum()
        return weights


class MomentumVolAdjusted(FactorBasedAllocator):
    """
    Sharpe-weighted momentum (volatility-adjusted).

    Uses Sharpe ratio as the factor for both selection and weighting.
    """

    def __init__(
        self,
        # Factor selection - use Sharpe as the factor
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # Parameters
        lookback: int = 63,

        # Base parameters
        **kwargs,
    ):
        if secondary_factors is None:
            secondary_factors = ["rolling_sharpe_63d"]

        super().__init__(
            factor_name=factor_name,
            secondary_factors=secondary_factors,
            selection_method=selection_method,
            selection_param=selection_param,
            lookback_window=lookback,
            **kwargs,
        )
        self.lookback = lookback

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Sharpe-weighted optimization.

        Args:
            date: Calculation date.
            returns: Returns of selected algorithms.
            factor_scores: Normalized Sharpe scores.
            current_weights: Current weights (not used).

        Returns:
            Sharpe-weighted portfolios summing to 1.
        """
        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        # Weight by Sharpe score
        scores = np.maximum(factor_scores, 0)

        if scores.sum() < 1e-10:
            return np.ones(n_selected) / n_selected

        weights = scores / scores.sum()
        return weights
