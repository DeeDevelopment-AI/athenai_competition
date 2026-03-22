"""
Risk Parity Allocator with Factor Selection.

Architecture:
1. Factor Model: Select top N algorithms by factor
2. Portfolio Optimizer: Risk parity weighting on selected algorithms
3. Constraints: Match benchmark characteristics
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import FactorBasedAllocator


class RiskParityAllocator(FactorBasedAllocator):
    """
    Risk Parity allocation on factor-selected algorithms.

    Steps:
    1. Rank all algorithms by factor (e.g., Sharpe, momentum)
    2. Select top N algorithms
    3. Weight inversely proportional to volatility (equal risk contribution)
    4. Apply benchmark constraints
    """

    def __init__(
        self,
        # Factor selection
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # Risk parity parameters
        vol_lookback: int = 63,

        # Base parameters
        **kwargs,
    ):
        super().__init__(
            factor_name=factor_name,
            secondary_factors=secondary_factors,
            selection_method=selection_method,
            selection_param=selection_param,
            lookback_window=vol_lookback,
            **kwargs,
        )
        self.vol_lookback = vol_lookback

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Risk parity optimization: weights inversely proportional to volatility.

        Args:
            date: Calculation date.
            returns: Returns of selected algorithms.
            factor_scores: Normalized factor scores (not used).
            current_weights: Current weights (not used).

        Returns:
            Risk parity weights summing to 1.
        """
        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        # Get recent returns
        recent = returns.tail(self.vol_lookback)

        if len(recent) < 10:
            return np.ones(n_selected) / n_selected

        # Compute volatilities
        vols = recent.std().values

        # Handle zero volatility
        vols = np.maximum(vols, 1e-8)

        # Weights inversely proportional to volatility
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()

        return weights


class RiskParityERC(FactorBasedAllocator):
    """
    Equal Risk Contribution (exact Risk Parity) on factor-selected algorithms.

    Each asset contributes exactly equal risk to the portfolio.
    Requires iterative optimization.
    """

    def __init__(
        self,
        # Factor selection
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # ERC parameters
        vol_lookback: int = 63,
        max_iter: int = 100,
        tol: float = 1e-6,

        # Base parameters
        **kwargs,
    ):
        super().__init__(
            factor_name=factor_name,
            secondary_factors=secondary_factors,
            selection_method=selection_method,
            selection_param=selection_param,
            lookback_window=vol_lookback,
            **kwargs,
        )
        self.vol_lookback = vol_lookback
        self.max_iter = max_iter
        self.tol = tol

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        ERC optimization via iterative method.

        Args:
            date: Calculation date.
            returns: Returns of selected algorithms.
            factor_scores: Normalized factor scores (not used).
            current_weights: Current weights (not used).

        Returns:
            ERC weights summing to 1.
        """
        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        recent = returns.tail(self.vol_lookback)

        if len(recent) < 10:
            return np.ones(n_selected) / n_selected

        # Covariance matrix
        cov = recent.cov().values
        cov = np.nan_to_num(cov, nan=0.0)

        # Make symmetric and PSD
        cov = (cov + cov.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 1e-8:
            cov += (1e-8 - min_eig) * np.eye(n_selected)

        # Initialize with equal weights
        weights = np.ones(n_selected) / n_selected

        # Iterative optimization
        for _ in range(self.max_iter):
            portfolio_var = np.dot(weights, np.dot(cov, weights))
            portfolio_vol = np.sqrt(max(portfolio_var, 1e-16))

            marginal_risk = np.dot(cov, weights) / portfolio_vol
            risk_contrib = weights * marginal_risk
            total_risk = risk_contrib.sum()

            if total_risk < 1e-10:
                break

            # Target: equal contribution
            target_contrib = total_risk / n_selected

            # Update weights
            adjustment = np.sqrt(target_contrib / (risk_contrib + 1e-10))
            new_weights = weights * adjustment
            new_weights = np.maximum(new_weights, 0)
            new_weights = new_weights / new_weights.sum()

            # Check convergence
            if np.abs(new_weights - weights).max() < self.tol:
                break

            weights = new_weights

        return weights
