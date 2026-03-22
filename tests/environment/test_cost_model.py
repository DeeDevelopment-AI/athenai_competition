"""
Tests para el modelo de costes.
"""

import numpy as np
import pytest

from src.environment.cost_model import CostModel


class TestCostModel:
    """Tests para CostModel."""

    def test_zero_turnover_zero_cost(self):
        """Turnover 0 debe dar coste 0."""
        model = CostModel(spread_bps=10, slippage_bps=5)

        old_weights = np.array([0.25, 0.25, 0.25, 0.25])
        new_weights = np.array([0.25, 0.25, 0.25, 0.25])

        cost = model.compute_cost(old_weights, new_weights, 1_000_000)
        assert cost == 0.0

    def test_cost_increases_with_turnover(self):
        """Coste debe aumentar con turnover."""
        model = CostModel(spread_bps=10, slippage_bps=5)

        old_weights = np.array([0.25, 0.25, 0.25, 0.25])

        # Pequeño cambio
        small_change = np.array([0.30, 0.20, 0.25, 0.25])
        cost_small = model.compute_cost(old_weights, small_change, 1_000_000)

        # Grande cambio
        large_change = np.array([0.50, 0.10, 0.20, 0.20])
        cost_large = model.compute_cost(old_weights, large_change, 1_000_000)

        assert cost_large > cost_small

    def test_cost_proportional_to_portfolio_value(self):
        """Coste debe ser proporcional al valor del portfolio."""
        model = CostModel(spread_bps=10, slippage_bps=5)

        old_weights = np.array([0.25, 0.25, 0.25, 0.25])
        new_weights = np.array([0.40, 0.20, 0.20, 0.20])

        cost_1m = model.compute_cost(old_weights, new_weights, 1_000_000)
        cost_2m = model.compute_cost(old_weights, new_weights, 2_000_000)

        assert abs(cost_2m - 2 * cost_1m) < 1e-6

    def test_cost_as_return(self):
        """Coste como retorno debe estar entre 0 y 1."""
        model = CostModel(spread_bps=10, slippage_bps=5)

        old_weights = np.array([0.25, 0.25, 0.25, 0.25])
        new_weights = np.array([0.40, 0.20, 0.20, 0.20])

        cost_pct = model.compute_cost_as_return(old_weights, new_weights)

        assert 0 <= cost_pct <= 1

    def test_fixed_cost_per_trade(self):
        """Coste fijo debe aplicarse por trade."""
        model = CostModel(fixed_cost_per_trade=100, spread_bps=0, slippage_bps=0)

        old_weights = np.array([0.25, 0.25, 0.25, 0.25])
        new_weights = np.array([0.30, 0.20, 0.25, 0.25])  # 2 trades

        cost = model.compute_cost(old_weights, new_weights, 1_000_000)

        # 2 trades * 100 = 200 (aproximadamente, puede haber costes de impacto)
        assert cost >= 200

    def test_estimate_breakeven_alpha(self):
        """Breakeven alpha debe ser positivo para turnover positivo."""
        model = CostModel(spread_bps=10, slippage_bps=5)

        breakeven = model.estimate_breakeven_alpha(turnover_annual=2.0)

        assert breakeven > 0
