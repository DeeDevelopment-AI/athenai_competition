"""
Restricciones de cartera.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConstraintViolation:
    """Información sobre violación de restricción."""

    name: str
    severity: float  # 0-1, qué tan grave es la violación
    message: str


class PortfolioConstraints:
    """
    Aplica y verifica restricciones de cartera.

    Restricciones soportadas:
    - Peso máximo y mínimo por algoritmo
    - Exposición total máxima
    - Turnover máximo por rebalanceo
    - Volatilidad objetivo (opcional)
    """

    def __init__(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.00,
        max_turnover: float = 0.30,
        max_exposure: float = 1.0,
        target_volatility: Optional[float] = None,
        vol_tolerance: float = 0.02,
    ):
        """
        Args:
            max_weight: Peso máximo permitido por algoritmo.
            min_weight: Peso mínimo permitido por algoritmo.
            max_turnover: Turnover máximo por rebalanceo.
            max_exposure: Exposición total máxima (sum of weights).
            target_volatility: Volatilidad objetivo del portfolio (opcional).
            vol_tolerance: Tolerancia alrededor de la volatilidad objetivo.
        """
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_turnover = max_turnover
        self.max_exposure = max_exposure
        self.target_volatility = target_volatility
        self.vol_tolerance = vol_tolerance

    def apply(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Proyecta pesos objetivo al espacio factible.

        Args:
            target_weights: Pesos deseados.
            current_weights: Pesos actuales.

        Returns:
            Pesos ajustados que cumplen todas las restricciones.
        """
        weights = target_weights.copy()

        # 1. Clip por min/max peso
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # 2. Asegurar no negativos
        weights = np.maximum(weights, 0)

        # 3. Normalizar si suma > max_exposure
        total = weights.sum()
        if total > self.max_exposure:
            weights = weights * (self.max_exposure / total)

        # 4. Limitar turnover
        turnover = np.abs(weights - current_weights).sum() / 2
        if turnover > self.max_turnover:
            scale = self.max_turnover / turnover
            weights = current_weights + scale * (weights - current_weights)

        return weights

    def check_violations(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
    ) -> list[ConstraintViolation]:
        """
        Reporta violaciones de restricciones.

        Args:
            weights: Pesos a verificar.
            current_weights: Pesos anteriores (para calcular turnover).

        Returns:
            Lista de violaciones encontradas.
        """
        violations = []

        # Verificar pesos máximos
        max_violation = np.maximum(weights - self.max_weight, 0).max()
        if max_violation > 1e-6:
            violations.append(ConstraintViolation(
                name="max_weight",
                severity=min(max_violation / self.max_weight, 1.0),
                message=f"Weight exceeds max by {max_violation:.4f}",
            ))

        # Verificar pesos mínimos
        min_violation = np.maximum(self.min_weight - weights, 0).max()
        if min_violation > 1e-6:
            violations.append(ConstraintViolation(
                name="min_weight",
                severity=min(min_violation / max(self.min_weight, 0.01), 1.0),
                message=f"Weight below min by {min_violation:.4f}",
            ))

        # Verificar exposición total
        total_exposure = weights.sum()
        if total_exposure > self.max_exposure + 1e-6:
            excess = total_exposure - self.max_exposure
            violations.append(ConstraintViolation(
                name="max_exposure",
                severity=min(excess / self.max_exposure, 1.0),
                message=f"Total exposure exceeds max by {excess:.4f}",
            ))

        # Verificar turnover
        turnover = np.abs(weights - current_weights).sum() / 2
        if turnover > self.max_turnover + 1e-6:
            excess = turnover - self.max_turnover
            violations.append(ConstraintViolation(
                name="max_turnover",
                severity=min(excess / self.max_turnover, 1.0),
                message=f"Turnover exceeds max by {excess:.4f}",
            ))

        return violations

    def is_feasible(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
    ) -> bool:
        """
        Verifica si los pesos cumplen todas las restricciones.

        Args:
            weights: Pesos a verificar.
            current_weights: Pesos anteriores.

        Returns:
            True si cumple todas las restricciones.
        """
        violations = self.check_violations(weights, current_weights)
        return len(violations) == 0

    def project_to_feasible(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray,
        max_iterations: int = 10,
    ) -> np.ndarray:
        """
        Proyecta pesos al espacio factible iterativamente.

        Método de proyección alternada (Dykstra's algorithm simplificado).

        Args:
            weights: Pesos iniciales.
            current_weights: Pesos actuales.
            max_iterations: Número máximo de iteraciones.

        Returns:
            Pesos proyectados.
        """
        w = weights.copy()

        for _ in range(max_iterations):
            w_prev = w.copy()

            # Proyectar a box constraints
            w = np.clip(w, self.min_weight, self.max_weight)

            # Proyectar a simplex (sum <= max_exposure)
            if w.sum() > self.max_exposure:
                w = w * (self.max_exposure / w.sum())

            # Proyectar a constraint de turnover
            turnover = np.abs(w - current_weights).sum() / 2
            if turnover > self.max_turnover:
                scale = self.max_turnover / turnover
                w = current_weights + scale * (w - current_weights)

            # Verificar convergencia
            if np.allclose(w, w_prev):
                break

        return w
