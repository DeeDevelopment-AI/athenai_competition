"""
Funciones de reward para el agente RL.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RewardType(Enum):
    """Tipos de función de reward."""

    ALPHA_PENALIZED = "alpha_penalized"
    INFORMATION_RATIO = "info_ratio"
    RISK_ADJUSTED = "risk_adjusted"
    DIVERSIFIED = "diversified"
    PURE_RETURNS = "pure_returns"  # Maximize returns only, no penalties
    CALIBRATED_ALPHA = "calibrated_alpha"  # Alpha with normalized penalties


@dataclass
class RewardComponents:
    """Componentes individuales del reward."""

    base_reward: float
    cost_penalty: float
    turnover_penalty: float
    drawdown_penalty: float
    risk_penalty: float
    tracking_error_penalty: float
    total: float


class RewardFunction:
    """
    Función de reward para el agente RL.

    Reward = alpha vs benchmark - penalizaciones

    Diseño clave: el reward NO es PnL bruto.
    Es exceso de retorno penalizado por costes, turnover, drawdown y riesgo.
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.ALPHA_PENALIZED,
        cost_penalty_weight: float = 1.0,
        turnover_penalty_weight: float = 0.1,
        drawdown_penalty_weight: float = 0.5,
        drawdown_threshold: float = 0.05,
        risk_penalty_weight: float = 0.2,
        risk_tolerance: float = 1.2,
        tracking_error_penalty: float = 0.0,
        diversification_bonus: float = 0.0,
    ):
        """
        Args:
            reward_type: Tipo de función de reward base.
            cost_penalty_weight: Peso de penalización por costes.
            turnover_penalty_weight: Peso de penalización por turnover.
            drawdown_penalty_weight: Peso de penalización por drawdown.
            drawdown_threshold: Umbral de drawdown antes de penalizar.
            risk_penalty_weight: Peso de penalización por exceso de riesgo.
            risk_tolerance: Factor sobre volatilidad del benchmark.
            tracking_error_penalty: Peso de penalización por tracking error.
            diversification_bonus: Bonus por diversificación.
        """
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
        Calcula reward y sus componentes.

        Args:
            portfolio_return: Retorno del portfolio en el periodo.
            benchmark_return: Retorno del benchmark en el periodo.
            transaction_costs: Costes de transacción (como fracción).
            turnover: Turnover del rebalanceo (0-1).
            current_drawdown: Drawdown actual del portfolio (negativo).
            portfolio_vol: Volatilidad del portfolio (anualizada).
            benchmark_vol: Volatilidad del benchmark (anualizada).
            tracking_error: Tracking error vs benchmark.
            diversification_ratio: Ratio de diversificación.

        Returns:
            RewardComponents con desglose del reward.
        """
        # Sanitize inputs - replace NaN/Inf with 0
        def safe_float(x: float, default: float = 0.0) -> float:
            if x is None or np.isnan(x) or np.isinf(x):
                return default
            return float(x)

        portfolio_return = safe_float(portfolio_return)
        benchmark_return = safe_float(benchmark_return)
        transaction_costs = safe_float(transaction_costs)
        turnover = safe_float(turnover)
        current_drawdown = safe_float(current_drawdown)
        tracking_error = safe_float(tracking_error)
        diversification_ratio = safe_float(diversification_ratio, 1.0)

        if portfolio_vol is not None:
            portfolio_vol = safe_float(portfolio_vol)
        if benchmark_vol is not None:
            benchmark_vol = safe_float(benchmark_vol)

        # Base reward según tipo
        if self.reward_type == RewardType.PURE_RETURNS:
            # Pure returns: only maximize portfolio return, no penalties
            base_reward = portfolio_return
            return RewardComponents(
                base_reward=safe_float(base_reward),
                cost_penalty=0.0,
                turnover_penalty=0.0,
                drawdown_penalty=0.0,
                risk_penalty=0.0,
                tracking_error_penalty=0.0,
                total=safe_float(base_reward),
            )

        elif self.reward_type == RewardType.ALPHA_PENALIZED:
            base_reward = portfolio_return - benchmark_return

        elif self.reward_type == RewardType.INFORMATION_RATIO:
            # IR incremental (aproximado)
            if tracking_error > 1e-6:
                base_reward = (portfolio_return - benchmark_return) / tracking_error
            else:
                base_reward = portfolio_return - benchmark_return

        elif self.reward_type == RewardType.RISK_ADJUSTED:
            # Sharpe diferencial
            if portfolio_vol and portfolio_vol > 1e-6:
                portfolio_sharpe = portfolio_return / (portfolio_vol / np.sqrt(252))
                if benchmark_vol and benchmark_vol > 1e-6:
                    benchmark_sharpe = benchmark_return / (benchmark_vol / np.sqrt(252))
                else:
                    benchmark_sharpe = 0
                base_reward = portfolio_sharpe - benchmark_sharpe
            else:
                base_reward = portfolio_return - benchmark_return

        elif self.reward_type == RewardType.DIVERSIFIED:
            base_reward = portfolio_return - benchmark_return
            # Bonus se añade abajo

        elif self.reward_type == RewardType.CALIBRATED_ALPHA:
            # Calibrated alpha: penalties normalized to be comparable in magnitude
            # All terms scaled to be in similar range (~0.001 to ~0.01 for typical values)
            alpha = portfolio_return - benchmark_return

            # Normalize penalties to alpha scale (weekly returns ~0.1% to 1%)
            # Costs are already in return units, no change needed
            norm_cost = transaction_costs
            # Turnover 0-30% → penalty ~0-0.003 (10x less than alpha)
            norm_turnover = turnover * 0.01
            # Drawdown 0-20% → penalty ~0-0.01 when exceeds threshold
            dd_excess = max(0, abs(current_drawdown) - self.drawdown_threshold)
            norm_dd = dd_excess * 0.05

            # Total with soft penalties (don't dominate alpha signal)
            total = alpha - norm_cost - norm_turnover - norm_dd

            return RewardComponents(
                base_reward=safe_float(alpha),
                cost_penalty=safe_float(norm_cost),
                turnover_penalty=safe_float(norm_turnover),
                drawdown_penalty=safe_float(norm_dd),
                risk_penalty=0.0,
                tracking_error_penalty=0.0,
                total=safe_float(total),
            )

        else:
            base_reward = portfolio_return - benchmark_return

        # Penalizaciones
        cost_penalty = self.cost_penalty_weight * transaction_costs

        turnover_penalty = self.turnover_penalty_weight * turnover

        # Penalización de drawdown (solo si excede umbral)
        dd_excess = max(0, abs(current_drawdown) - self.drawdown_threshold)
        drawdown_penalty = self.drawdown_penalty_weight * dd_excess

        # Penalización de riesgo (solo si excede tolerancia)
        risk_penalty = 0.0
        if portfolio_vol and benchmark_vol:
            vol_excess = max(0, portfolio_vol - benchmark_vol * self.risk_tolerance)
            risk_penalty = self.risk_penalty_weight * vol_excess

        # Penalización de tracking error
        te_penalty = self.tracking_error_penalty * tracking_error

        # Total
        total = base_reward - cost_penalty - turnover_penalty - drawdown_penalty - risk_penalty - te_penalty

        # Bonus por diversificación
        if self.diversification_bonus > 0:
            div_bonus = self.diversification_bonus * max(0, diversification_ratio - 1)
            total += div_bonus

        # Final safety check - ensure no NaN in output
        base_reward = safe_float(base_reward)
        cost_penalty = safe_float(cost_penalty)
        turnover_penalty = safe_float(turnover_penalty)
        drawdown_penalty = safe_float(drawdown_penalty)
        risk_penalty = safe_float(risk_penalty)
        te_penalty = safe_float(te_penalty)
        total = safe_float(total)

        return RewardComponents(
            base_reward=base_reward,
            cost_penalty=cost_penalty,
            turnover_penalty=turnover_penalty,
            drawdown_penalty=drawdown_penalty,
            risk_penalty=risk_penalty,
            tracking_error_penalty=te_penalty,
            total=total,
        )

    def scale_reward(self, reward: float, scale: float = 100.0) -> float:
        """
        Escala el reward para estabilidad de entrenamiento.

        Los retornos diarios son pequeños (~0.001), escalar ayuda al gradiente.

        Args:
            reward: Reward sin escalar.
            scale: Factor de escala.

        Returns:
            Reward escalado.
        """
        return reward * scale

    def clip_reward(self, reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """
        Clipea el reward para evitar gradientes explosivos.

        Args:
            reward: Reward a clipear.
            min_val: Valor mínimo.
            max_val: Valor máximo.

        Returns:
            Reward clipeado.
        """
        return np.clip(reward, min_val, max_val)
