"""
Minimum Variance Allocator with Factor Selection.

Architecture:
1. Factor Model: Select top N algorithms by factor
2. Portfolio Optimizer: Minimize portfolio variance
3. Constraints: Match benchmark characteristics
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import FactorBasedAllocator


class MinVarianceAllocator(FactorBasedAllocator):
    """
    Minimum Variance allocation on factor-selected algorithms.

    Steps:
    1. Rank all algorithms by factor (e.g., low volatility)
    2. Select top N algorithms
    3. Solve quadratic optimization to minimize variance
    4. Apply benchmark constraints
    """

    def __init__(
        self,
        # Factor selection
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # Optimization parameters
        cov_lookback: int = 126,
        shrinkage: float = 0.1,

        # Base parameters
        **kwargs,
    ):
        super().__init__(
            factor_name=factor_name,
            secondary_factors=secondary_factors,
            selection_method=selection_method,
            selection_param=selection_param,
            lookback_window=cov_lookback,
            **kwargs,
        )
        self.cov_lookback = cov_lookback
        self.shrinkage = shrinkage

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Minimum variance optimization using CVXPY.

        Args:
            date: Calculation date.
            returns: Returns of selected algorithms.
            factor_scores: Normalized factor scores (not used).
            current_weights: Current weights (not used).

        Returns:
            Minimum variance weights summing to 1.
        """
        import cvxpy as cp

        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        recent = returns.tail(self.cov_lookback)

        if len(recent) < 20:
            return np.ones(n_selected) / n_selected

        # Shrunk covariance matrix
        cov = self._shrunk_covariance(recent)

        # Optimization variable
        w = cp.Variable(n_selected)

        # Objective: minimize variance
        portfolio_var = cp.quad_form(w, cov)

        # Constraints
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= self.max_weight,
        ]

        # Solve
        problem = cp.Problem(cp.Minimize(portfolio_var), constraints)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem.solve(solver=cp.OSQP, verbose=False, max_iter=1000)

            if problem.status == "optimal" and w.value is not None:
                return w.value
            else:
                # Fallback to inverse volatility
                vols = recent.std().values
                vols = np.maximum(vols, 1e-8)
                inv_vols = 1.0 / vols
                return inv_vols / inv_vols.sum()

        except Exception:
            return np.ones(n_selected) / n_selected

    def _shrunk_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute shrunk covariance matrix."""
        returns_clean = returns.fillna(0)
        sample_cov = returns_clean.cov().values
        n = sample_cov.shape[0]

        sample_cov = np.nan_to_num(sample_cov, nan=0.0)
        sample_cov = (sample_cov + sample_cov.T) / 2

        trace = np.trace(sample_cov) / n
        if np.isnan(trace) or trace <= 0:
            trace = 1e-4
        target = trace * np.eye(n)

        shrunk_cov = (1 - self.shrinkage) * sample_cov + self.shrinkage * target

        # Ensure PSD
        min_eig = np.min(np.linalg.eigvalsh(shrunk_cov))
        if min_eig < 1e-8:
            shrunk_cov += (1e-8 - min_eig) * np.eye(n)

        return shrunk_cov
