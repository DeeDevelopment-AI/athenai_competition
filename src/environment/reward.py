"""
Reward functions for RL agents.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RewardType(Enum):
    """Supported reward modes."""

    ABSOLUTE_RETURNS = "absolute_returns"
    RISK_CALIBRATED_RETURNS = "risk_calibrated_returns"
    ALPHA_PENALIZED = "alpha_penalized"  # legacy alias
    INFORMATION_RATIO = "info_ratio"  # legacy alias
    RISK_ADJUSTED = "risk_adjusted"
    DIVERSIFIED = "diversified"
    PURE_RETURNS = "pure_returns"
    CALIBRATED_ALPHA = "calibrated_alpha"  # legacy alias


@dataclass
class RewardComponents:
    """Reward breakdown."""

    base_reward: float
    cost_penalty: float
    turnover_penalty: float
    drawdown_penalty: float
    risk_penalty: float
    tracking_error_penalty: float
    total: float


class RewardFunction:
    """
    Absolute-return reward with optional risk calibration.

    The benchmark is not a target to beat. It is only a reference that can
    anchor the acceptable risk envelope.
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.RISK_CALIBRATED_RETURNS,
        cost_penalty_weight: float = 1.0,
        turnover_penalty_weight: float = 0.1,
        drawdown_penalty_weight: float = 0.5,
        drawdown_threshold: float = 0.05,
        risk_penalty_weight: float = 0.2,
        risk_tolerance: float = 1.2,
        tracking_error_penalty: float = 0.0,
        diversification_bonus: float = 0.0,
    ):
        self.reward_type = reward_type
        self.cost_penalty_weight = cost_penalty_weight
        self.turnover_penalty_weight = turnover_penalty_weight
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.drawdown_threshold = drawdown_threshold
        self.risk_penalty_weight = risk_penalty_weight
        self.risk_tolerance = risk_tolerance
        self.tracking_error_penalty = tracking_error_penalty
        self.diversification_bonus = diversification_bonus

    def compute(
        self,
        portfolio_return: float,
        benchmark_return: float,
        transaction_costs: float = 0.0,
        turnover: float = 0.0,
        current_drawdown: float = 0.0,
        portfolio_vol: Optional[float] = None,
        benchmark_vol: Optional[float] = None,
        tracking_error: float = 0.0,
        diversification_ratio: float = 1.0,
    ) -> RewardComponents:
        """
        Compute reward.

        `benchmark_return` is kept only for backward compatibility. It does not
        define the base objective.
        """

        def safe_float(x: Optional[float], default: float = 0.0) -> float:
            if x is None or np.isnan(x) or np.isinf(x):
                return default
            return float(x)

        portfolio_return = safe_float(portfolio_return)
        _ = safe_float(benchmark_return)
        transaction_costs = safe_float(transaction_costs)
        turnover = safe_float(turnover)
        current_drawdown = safe_float(current_drawdown)
        tracking_error = safe_float(tracking_error)
        diversification_ratio = safe_float(diversification_ratio, 1.0)
        portfolio_vol = safe_float(portfolio_vol) if portfolio_vol is not None else None
        benchmark_vol = safe_float(benchmark_vol) if benchmark_vol is not None else None

        if self.reward_type == RewardType.PURE_RETURNS:
            return RewardComponents(
                base_reward=portfolio_return,
                cost_penalty=0.0,
                turnover_penalty=0.0,
                drawdown_penalty=0.0,
                risk_penalty=0.0,
                tracking_error_penalty=0.0,
                total=portfolio_return,
            )

        if self.reward_type in (
            RewardType.ABSOLUTE_RETURNS,
            RewardType.ALPHA_PENALIZED,
        ):
            base_reward = portfolio_return
        elif self.reward_type == RewardType.INFORMATION_RATIO:
            base_reward = portfolio_return - 0.5 * tracking_error
        elif self.reward_type == RewardType.RISK_ADJUSTED:
            if portfolio_vol and portfolio_vol > 1e-6:
                base_reward = portfolio_return / (portfolio_vol / np.sqrt(252.0))
            else:
                base_reward = portfolio_return
        elif self.reward_type == RewardType.DIVERSIFIED:
            base_reward = portfolio_return
        elif self.reward_type in (
            RewardType.RISK_CALIBRATED_RETURNS,
            RewardType.CALIBRATED_ALPHA,
        ):
            norm_cost = transaction_costs
            norm_turnover = turnover * 0.01
            dd_excess = max(0.0, abs(current_drawdown) - self.drawdown_threshold)
            norm_drawdown = dd_excess * 0.05
            risk_penalty = 0.0
            if benchmark_vol is not None and benchmark_vol > 1e-6 and portfolio_vol is not None:
                max_allowed_vol = benchmark_vol * self.risk_tolerance
                risk_penalty = max(0.0, portfolio_vol - max_allowed_vol) * 0.05
            total = portfolio_return - norm_cost - norm_turnover - norm_drawdown - risk_penalty
            return RewardComponents(
                base_reward=portfolio_return,
                cost_penalty=norm_cost,
                turnover_penalty=norm_turnover,
                drawdown_penalty=norm_drawdown,
                risk_penalty=risk_penalty,
                tracking_error_penalty=0.0,
                total=safe_float(total),
            )
        else:
            base_reward = portfolio_return

        cost_penalty = self.cost_penalty_weight * transaction_costs
        turnover_penalty = self.turnover_penalty_weight * turnover
        dd_excess = max(0.0, abs(current_drawdown) - self.drawdown_threshold)
        drawdown_penalty = self.drawdown_penalty_weight * dd_excess

        risk_penalty = 0.0
        if portfolio_vol is not None and benchmark_vol is not None and benchmark_vol > 1e-6:
            max_allowed_vol = benchmark_vol * self.risk_tolerance
            risk_penalty = self.risk_penalty_weight * max(0.0, portfolio_vol - max_allowed_vol)

        te_penalty = self.tracking_error_penalty * tracking_error
        total = base_reward - cost_penalty - turnover_penalty - drawdown_penalty - risk_penalty - te_penalty

        if self.diversification_bonus > 0:
            total += self.diversification_bonus * max(0.0, diversification_ratio - 1.0)

        return RewardComponents(
            base_reward=safe_float(base_reward),
            cost_penalty=safe_float(cost_penalty),
            turnover_penalty=safe_float(turnover_penalty),
            drawdown_penalty=safe_float(drawdown_penalty),
            risk_penalty=safe_float(risk_penalty),
            tracking_error_penalty=safe_float(te_penalty),
            total=safe_float(total),
        )

    def scale_reward(self, reward: float, scale: float = 100.0) -> float:
        return reward * scale

    def clip_reward(self, reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        return np.clip(reward, min_val, max_val)
