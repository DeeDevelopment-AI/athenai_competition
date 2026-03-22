"""
Equal Weight Allocator with Factor Selection.

Architecture:
1. Factor Model: Select top N algorithms by factor
2. Portfolio Optimizer: Equal weight on selected algorithms
3. Constraints: Match benchmark characteristics
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import FactorBasedAllocator


class EqualWeightAllocator(FactorBasedAllocator):
    """
    Equal weight allocation on factor-selected algorithms.

    Steps:
    1. Rank all algorithms by factor (e.g., Sharpe, momentum)
    2. Select top N algorithms
    3. Equal weight among selected
    4. Apply benchmark constraints

    Reference: DeMiguel et al. (2009) "Optimal Versus Naive Diversification"
    """

    def __init__(
        self,
        # Factor selection
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # Base parameters
        **kwargs,
    ):
        super().__init__(
            factor_name=factor_name,
            secondary_factors=secondary_factors,
            selection_method=selection_method,
            selection_param=selection_param,
            **kwargs,
        )

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Equal weight optimization: 1/N for each selected algorithm.

        Args:
            date: Calculation date.
            returns: Returns of selected algorithms.
            factor_scores: Normalized factor scores (not used for EW).
            current_weights: Current weights (not used for EW).

        Returns:
            Equal weights summing to 1.
        """
        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        return np.ones(n_selected) / n_selected
