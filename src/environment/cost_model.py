"""
Modelo de costes de transacción.
"""

import numpy as np


class CostModel:
    """
    Modela costes de implementación de transacciones.

    Componentes:
    1. Comisión fija por rebalanceo
    2. Spread proporcional
    3. Slippage proporcional al tamaño
    4. Impacto de mercado (función creciente del tamaño)
    """

    def __init__(
        self,
        fixed_cost_per_trade: float = 0.0,
        spread_bps: float = 5.0,
        slippage_bps: float = 2.0,
        impact_coefficient: float = 0.1,
    ):
        """
        Args:
            fixed_cost_per_trade: Coste fijo por operación (en unidades monetarias).
            spread_bps: Spread bid-ask en puntos básicos.
            slippage_bps: Slippage esperado en puntos básicos.
            impact_coefficient: Coeficiente de impacto de mercado.
        """
        self.fixed_cost = fixed_cost_per_trade
        self.spread = spread_bps / 10000
        self.slippage = slippage_bps / 10000
        self.impact_coef = impact_coefficient

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        """
        Calcula coste total de transición (vectorized).

        Args:
            old_weights: Pesos antes del rebalanceo.
            new_weights: Pesos después del rebalanceo.
            portfolio_value: Valor total del portfolio.

        Returns:
            Coste total en unidades monetarias.
        """
        delta_weights = np.abs(new_weights - old_weights)

        # Mask for significant trades (avoid numerical noise)
        trade_mask = delta_weights > 1e-6

        # Number of trades
        n_trades = trade_mask.sum()

        # Fixed cost
        fixed_cost = n_trades * self.fixed_cost

        # Vectorized variable cost computation
        # Only compute for significant trades
        significant_deltas = delta_weights[trade_mask]

        if len(significant_deltas) == 0:
            return fixed_cost

        # Trade values (vectorized)
        trade_values = significant_deltas * portfolio_value

        # Spread cost (vectorized)
        spread_costs = trade_values * self.spread

        # Slippage cost (vectorized)
        slippage_costs = trade_values * self.slippage

        # Impact cost (vectorized): proportional to trade_value * delta
        # impact = trade_value * impact_coef * delta = delta^2 * portfolio_value * impact_coef
        impact_costs = trade_values * self.impact_coef * significant_deltas

        # Total variable cost (sum of all components)
        variable_cost = spread_costs.sum() + slippage_costs.sum() + impact_costs.sum()

        return fixed_cost + variable_cost

    def compute_cost_as_return(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> float:
        """
        Calcula coste como porcentaje del portfolio.

        Útil para penalizaciones en el reward.

        Args:
            old_weights: Pesos antes del rebalanceo.
            new_weights: Pesos después del rebalanceo.

        Returns:
            Coste como fracción del portfolio value.
        """
        # Normalizar asumiendo portfolio_value = 1
        cost = self.compute_cost(old_weights, new_weights, 1.0)
        return cost

    def estimate_breakeven_alpha(
        self,
        turnover_annual: float,
    ) -> float:
        """
        Estima alpha mínimo necesario para cubrir costes.

        Args:
            turnover_annual: Turnover anualizado (suma de |delta_w|).

        Returns:
            Alpha mínimo anual necesario.
        """
        # Coste medio por unidad de turnover
        avg_cost_per_unit = self.spread + self.slippage + self.impact_coef * 0.1

        return turnover_annual * avg_cost_per_unit
