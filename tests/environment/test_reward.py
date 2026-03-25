"""
Tests para las funciones de reward.
"""

import numpy as np
import pytest

from src.environment.reward import RewardFunction, RewardType, RewardComponents


class TestRewardFunction:
    """Tests para RewardFunction."""

    def test_default_values(self):
        """Verificar valores por defecto."""
        reward_fn = RewardFunction()

        assert reward_fn.reward_type == RewardType.ALPHA_PENALIZED
        assert reward_fn.cost_penalty_weight == 1.0
        assert reward_fn.turnover_penalty_weight == 0.1
        assert reward_fn.drawdown_penalty_weight == 0.5

    def test_custom_reward_type(self):
        """Verificar tipos de reward personalizados."""
        for reward_type in RewardType:
            reward_fn = RewardFunction(reward_type=reward_type)
            assert reward_fn.reward_type == reward_type


class TestAlphaPenalizedReward:
    """Tests para reward tipo ALPHA_PENALIZED."""

    def test_positive_alpha(self):
        """Alpha positivo debe dar reward positivo."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
        )

        assert result.base_reward == pytest.approx(0.01)
        assert result.total == pytest.approx(0.01)

    def test_negative_alpha(self):
        """Alpha negativo debe dar reward negativo."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.01,
            benchmark_return=0.02,
        )

        assert result.base_reward == pytest.approx(-0.01)
        assert result.total == pytest.approx(-0.01)

    def test_zero_alpha(self):
        """Match con benchmark debe dar reward 0."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.01,
            benchmark_return=0.01,
        )

        assert result.base_reward == pytest.approx(0.0)
        assert result.total == pytest.approx(0.0)


class TestCostPenalty:
    """Tests para penalizacion por costes."""

    def test_cost_penalty_reduces_reward(self):
        """Costes deben reducir el reward."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=1.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            transaction_costs=0.005,
        )

        assert result.cost_penalty == pytest.approx(0.005)
        assert result.total == pytest.approx(0.01 - 0.005)

    def test_cost_penalty_weight(self):
        """Peso de penalizacion debe escalar costes."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=2.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            transaction_costs=0.005,
        )

        assert result.cost_penalty == pytest.approx(0.01)

    def test_zero_cost_no_penalty(self):
        """Sin costes no hay penalizacion."""
        reward_fn = RewardFunction(cost_penalty_weight=1.0)

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            transaction_costs=0.0,
        )

        assert result.cost_penalty == 0.0


class TestTurnoverPenalty:
    """Tests para penalizacion por turnover."""

    def test_turnover_penalty_reduces_reward(self):
        """Turnover alto debe reducir reward."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=1.0,
            drawdown_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            turnover=0.20,
        )

        assert result.turnover_penalty == pytest.approx(0.20)

    def test_zero_turnover_no_penalty(self):
        """Sin turnover no hay penalizacion."""
        reward_fn = RewardFunction(turnover_penalty_weight=1.0)

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            turnover=0.0,
        )

        assert result.turnover_penalty == 0.0


class TestDrawdownPenalty:
    """Tests para penalizacion por drawdown."""

    def test_drawdown_above_threshold(self):
        """Drawdown sobre umbral debe penalizar."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=1.0,
            drawdown_threshold=0.05,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            current_drawdown=-0.10,  # 10% drawdown
        )

        # Exceso = 0.10 - 0.05 = 0.05
        assert result.drawdown_penalty == pytest.approx(0.05)

    def test_drawdown_below_threshold(self):
        """Drawdown bajo umbral no penaliza."""
        reward_fn = RewardFunction(
            drawdown_penalty_weight=1.0,
            drawdown_threshold=0.05,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            current_drawdown=-0.03,  # 3% drawdown, bajo umbral
        )

        assert result.drawdown_penalty == 0.0

    def test_no_drawdown_no_penalty(self):
        """Sin drawdown no hay penalizacion."""
        reward_fn = RewardFunction(drawdown_penalty_weight=1.0)

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            current_drawdown=0.0,
        )

        assert result.drawdown_penalty == 0.0


class TestRiskPenalty:
    """Tests para penalizacion por exceso de riesgo."""

    def test_excess_volatility_penalized(self):
        """Volatilidad excesiva debe penalizar."""
        reward_fn = RewardFunction(
            reward_type=RewardType.ALPHA_PENALIZED,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
            risk_penalty_weight=1.0,
            risk_tolerance=1.2,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            portfolio_vol=0.20,
            benchmark_vol=0.10,  # Portfolio vol = 2x benchmark
        )

        # Exceso = 0.20 - (0.10 * 1.2) = 0.08
        assert result.risk_penalty == pytest.approx(0.08)

    def test_volatility_within_tolerance(self):
        """Volatilidad dentro de tolerancia no penaliza."""
        reward_fn = RewardFunction(
            risk_penalty_weight=1.0,
            risk_tolerance=1.5,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            portfolio_vol=0.12,
            benchmark_vol=0.10,  # 1.2x, dentro de 1.5x tolerancia
        )

        assert result.risk_penalty == 0.0

    def test_no_vol_data_no_penalty(self):
        """Sin datos de volatilidad no hay penalizacion."""
        reward_fn = RewardFunction(risk_penalty_weight=1.0)

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
        )

        assert result.risk_penalty == 0.0


class TestPureReturnsReward:
    """Tests para reward tipo PURE_RETURNS."""

    def test_pure_returns_ignores_benchmark(self):
        """Pure returns solo usa portfolio return, ignora benchmark."""
        reward_fn = RewardFunction(reward_type=RewardType.PURE_RETURNS)

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.05,  # Benchmark higher, but ignored
        )

        assert result.base_reward == pytest.approx(0.02)
        assert result.total == pytest.approx(0.02)

    def test_pure_returns_ignores_penalties(self):
        """Pure returns ignora todas las penalizaciones."""
        reward_fn = RewardFunction(
            reward_type=RewardType.PURE_RETURNS,
            cost_penalty_weight=1.0,
            turnover_penalty_weight=1.0,
            drawdown_penalty_weight=1.0,
            risk_penalty_weight=1.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.03,
            benchmark_return=0.01,
            transaction_costs=0.01,
            turnover=0.50,
            current_drawdown=-0.20,
            portfolio_vol=0.30,
            benchmark_vol=0.10,
        )

        # All penalties should be 0
        assert result.cost_penalty == 0.0
        assert result.turnover_penalty == 0.0
        assert result.drawdown_penalty == 0.0
        assert result.risk_penalty == 0.0
        assert result.tracking_error_penalty == 0.0
        # Total should equal base_reward (portfolio return)
        assert result.total == pytest.approx(0.03)

    def test_pure_returns_negative(self):
        """Pure returns con retorno negativo."""
        reward_fn = RewardFunction(reward_type=RewardType.PURE_RETURNS)

        result = reward_fn.compute(
            portfolio_return=-0.02,
            benchmark_return=0.01,
        )

        assert result.base_reward == pytest.approx(-0.02)
        assert result.total == pytest.approx(-0.02)


class TestInformationRatioReward:
    """Tests para reward tipo INFORMATION_RATIO."""

    def test_ir_calculation(self):
        """IR debe ser alpha / tracking error."""
        reward_fn = RewardFunction(
            reward_type=RewardType.INFORMATION_RATIO,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            tracking_error=0.05,
        )

        # IR = (0.02 - 0.01) / 0.05 = 0.2
        assert result.base_reward == pytest.approx(0.2)

    def test_ir_with_zero_te(self):
        """Con TE = 0, usar alpha directamente."""
        reward_fn = RewardFunction(
            reward_type=RewardType.INFORMATION_RATIO,
            cost_penalty_weight=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            tracking_error=0.0,
        )

        # Fallback a alpha
        assert result.base_reward == pytest.approx(0.01)


class TestDiversifiedReward:
    """Tests para reward con bonus de diversificacion."""

    def test_diversification_bonus(self):
        """Ratio de diversificacion alto debe dar bonus."""
        reward_fn = RewardFunction(
            reward_type=RewardType.DIVERSIFIED,
            cost_penalty_weight=0.0,
            turnover_penalty_weight=0.0,
            drawdown_penalty_weight=0.0,
            diversification_bonus=0.1,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            diversification_ratio=1.5,  # Bien diversificado
        )

        # Base = 0.01, bonus = 0.1 * (1.5 - 1) = 0.05
        assert result.total == pytest.approx(0.01 + 0.05)

    def test_no_bonus_undiversified(self):
        """Ratio < 1 no da bonus."""
        reward_fn = RewardFunction(
            reward_type=RewardType.DIVERSIFIED,
            diversification_bonus=0.1,
        )

        result = reward_fn.compute(
            portfolio_return=0.02,
            benchmark_return=0.01,
            diversification_ratio=0.8,
        )

        # No hay bonus por ratio < 1
        # (bonus formula: max(0, ratio - 1))
        assert result.total <= result.base_reward


class TestRewardScaling:
    """Tests para escalado y clipping de reward."""

    def test_scale_reward(self):
        """Escalar reward por factor."""
        reward_fn = RewardFunction()

        scaled = reward_fn.scale_reward(0.01, scale=100.0)

        assert scaled == pytest.approx(1.0)

    def test_clip_reward(self):
        """Clipear reward a rango."""
        reward_fn = RewardFunction()

        # Valor extremo positivo
        clipped_pos = reward_fn.clip_reward(5.0, min_val=-1.0, max_val=1.0)
        assert clipped_pos == 1.0

        # Valor extremo negativo
        clipped_neg = reward_fn.clip_reward(-5.0, min_val=-1.0, max_val=1.0)
        assert clipped_neg == -1.0

        # Valor dentro de rango
        clipped_normal = reward_fn.clip_reward(0.5, min_val=-1.0, max_val=1.0)
        assert clipped_normal == 0.5


class TestRewardComponents:
    """Tests para RewardComponents dataclass."""

    def test_components_sum(self):
        """Total debe ser base_reward - penalizaciones."""
        reward_fn = RewardFunction(
            cost_penalty_weight=1.0,
            turnover_penalty_weight=1.0,
            drawdown_penalty_weight=1.0,
            drawdown_threshold=0.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.05,
            benchmark_return=0.01,
            transaction_costs=0.005,
            turnover=0.10,
            current_drawdown=-0.03,
        )

        expected_total = (
            result.base_reward
            - result.cost_penalty
            - result.turnover_penalty
            - result.drawdown_penalty
            - result.risk_penalty
            - result.tracking_error_penalty
        )

        assert result.total == pytest.approx(expected_total, rel=1e-6)

    def test_all_penalties_combined(self):
        """Todas las penalizaciones deben sumarse."""
        reward_fn = RewardFunction(
            cost_penalty_weight=1.0,
            turnover_penalty_weight=1.0,
            drawdown_penalty_weight=1.0,
            risk_penalty_weight=1.0,
            tracking_error_penalty=1.0,
            drawdown_threshold=0.0,
            risk_tolerance=1.0,
        )

        result = reward_fn.compute(
            portfolio_return=0.05,
            benchmark_return=0.01,
            transaction_costs=0.01,
            turnover=0.10,
            current_drawdown=-0.05,
            portfolio_vol=0.20,
            benchmark_vol=0.10,
            tracking_error=0.05,
        )

        # Todas las penalizaciones deben ser positivas
        assert result.cost_penalty > 0
        assert result.turnover_penalty > 0
        assert result.drawdown_penalty > 0
        assert result.risk_penalty > 0
        assert result.tracking_error_penalty > 0

        # Total debe ser menor que base_reward
        assert result.total < result.base_reward
