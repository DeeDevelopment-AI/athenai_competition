"""
Volatility Targeting Allocator with Factor Selection.

Architecture:
1. Factor Model: Select top N algorithms by factor
2. Portfolio Optimizer: Base allocation (EW, RP, etc.) + vol scaling
3. Constraints: Match benchmark characteristics
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import FactorBasedAllocator


class VolTargetingAllocator(FactorBasedAllocator):
    """
    Volatility Targeting on factor-selected algorithms.

    Steps:
    1. Rank all algorithms by factor
    2. Select top N algorithms
    3. Apply base weighting (equal weight, risk parity, etc.)
    4. Scale exposure to hit target volatility
    5. Apply benchmark constraints
    """

    def __init__(
        self,
        # Factor selection
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # Vol targeting parameters
        target_vol: float = 0.10,
        vol_lookback: int = 21,
        leverage_cap: float = 2.0,
        base_weighting: str = "equal",  # "equal", "risk_parity", "momentum"

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
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.leverage_cap = leverage_cap
        self.base_weighting = base_weighting

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Vol-targeted optimization.

        Args:
            date: Calculation date.
            returns: Returns of selected algorithms.
            factor_scores: Normalized factor scores.
            current_weights: Current weights (not used).

        Returns:
            Vol-scaled weights.
        """
        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        # Step 1: Get base weights
        base_weights = self._compute_base_weights(returns, factor_scores)

        # Step 2: Compute current portfolio volatility
        recent = returns.tail(self.vol_lookback)

        if len(recent) < 5:
            return base_weights

        portfolio_returns = (recent * base_weights).sum(axis=1)
        current_vol = portfolio_returns.std() * np.sqrt(252)

        if current_vol < 1e-8:
            return base_weights

        # Step 3: Scale to target volatility
        scale = self.target_vol / current_vol
        scale = min(scale, self.leverage_cap)
        scale = max(scale, 0.1)

        scaled_weights = base_weights * scale

        # Step 4: Cap at max_exposure
        if scaled_weights.sum() > self.max_exposure:
            scaled_weights = scaled_weights * (self.max_exposure / scaled_weights.sum())

        return scaled_weights

    def _compute_base_weights(
        self,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
    ) -> np.ndarray:
        """Compute base weights before vol scaling."""
        n = len(returns.columns)

        if self.base_weighting == "equal":
            return np.ones(n) / n

        elif self.base_weighting == "risk_parity":
            recent = returns.tail(self.vol_lookback)
            vols = recent.std().values
            vols = np.maximum(vols, 1e-8)
            inv_vols = 1.0 / vols
            return inv_vols / inv_vols.sum()

        elif self.base_weighting == "momentum":
            scores = np.maximum(factor_scores, 0)
            if scores.sum() < 1e-10:
                return np.ones(n) / n
            return scores / scores.sum()

        else:
            return np.ones(n) / n


class AdaptiveVolTargeting(VolTargetingAllocator):
    """
    Adaptive Volatility Targeting.

    Reduces exposure aggressively when volatility spikes,
    increases gradually when volatility is low.
    """

    def __init__(
        self,
        # Factor selection
        factor_name: str = "rolling_sharpe_21d",
        secondary_factors: list[str] = None,
        selection_method: str = "top_n",
        selection_param: float = 100,

        # Adaptive parameters
        target_vol: float = 0.10,
        vol_lookback: int = 21,
        fast_lookback: int = 5,
        ramp_up_speed: float = 0.1,
        ramp_down_speed: float = 0.5,

        # Base parameters
        **kwargs,
    ):
        super().__init__(
            factor_name=factor_name,
            secondary_factors=secondary_factors,
            selection_method=selection_method,
            selection_param=selection_param,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            **kwargs,
        )
        self.fast_lookback = fast_lookback
        self.ramp_up_speed = ramp_up_speed
        self.ramp_down_speed = ramp_down_speed
        self._last_scale = 1.0

    def _optimize_portfolio(
        self,
        date: pd.Timestamp,
        returns: pd.DataFrame,
        factor_scores: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Adaptive vol-targeted optimization.
        """
        n_selected = len(returns.columns)

        if n_selected == 0:
            return np.array([])

        base_weights = self._compute_base_weights(returns, factor_scores)

        if len(returns) < self.vol_lookback:
            return base_weights

        # Long-term volatility
        portfolio_returns_long = (returns.tail(self.vol_lookback) * base_weights).sum(axis=1)
        long_vol = portfolio_returns_long.std() * np.sqrt(252)

        # Short-term volatility (spike detection)
        portfolio_returns_short = (returns.tail(self.fast_lookback) * base_weights).sum(axis=1)
        short_vol = portfolio_returns_short.std() * np.sqrt(252)

        if long_vol < 1e-8:
            return base_weights

        # Use higher of two volatilities
        current_vol = max(long_vol, short_vol)

        # Target scale
        target_scale = self.target_vol / current_vol
        target_scale = min(target_scale, self.leverage_cap)
        target_scale = max(target_scale, 0.1)

        # Asymmetric adjustment: fast down, slow up
        if target_scale < self._last_scale:
            new_scale = self._last_scale + self.ramp_down_speed * (target_scale - self._last_scale)
        else:
            new_scale = self._last_scale + self.ramp_up_speed * (target_scale - self._last_scale)

        self._last_scale = new_scale

        scaled_weights = base_weights * new_scale

        if scaled_weights.sum() > self.max_exposure:
            scaled_weights = scaled_weights * (self.max_exposure / scaled_weights.sum())

        return scaled_weights
