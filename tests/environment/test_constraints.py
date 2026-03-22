"""
Tests para las restricciones de cartera.
"""

import numpy as np
import pytest

from src.environment.constraints import PortfolioConstraints, ConstraintViolation


class TestPortfolioConstraints:
    """Tests para PortfolioConstraints."""

    def test_default_values(self):
        """Verificar valores por defecto."""
        constraints = PortfolioConstraints()

        assert constraints.max_weight == 0.40
        assert constraints.min_weight == 0.00
        assert constraints.max_turnover == 0.30
        assert constraints.max_exposure == 1.0

    def test_custom_values(self):
        """Verificar valores personalizados."""
        constraints = PortfolioConstraints(
            max_weight=0.50,
            min_weight=0.05,
            max_turnover=0.20,
            max_exposure=0.90,
        )

        assert constraints.max_weight == 0.50
        assert constraints.min_weight == 0.05
        assert constraints.max_turnover == 0.20
        assert constraints.max_exposure == 0.90


class TestConstraintsApply:
    """Tests para el metodo apply()."""

    def test_apply_no_changes_needed(self):
        """Pesos ya validos no deben cambiar."""
        constraints = PortfolioConstraints()

        target = np.array([0.25, 0.25, 0.25, 0.25])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.apply(target, current)

        np.testing.assert_array_almost_equal(result, target)

    def test_apply_clips_max_weight(self):
        """Pesos que exceden max_weight deben recortarse."""
        constraints = PortfolioConstraints(max_weight=0.30)

        target = np.array([0.50, 0.30, 0.10, 0.10])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.apply(target, current)

        assert result.max() <= 0.30

    def test_apply_clips_negative_weights(self):
        """Pesos negativos deben recortarse a 0."""
        constraints = PortfolioConstraints()

        target = np.array([0.60, -0.10, 0.30, 0.20])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.apply(target, current)

        assert result.min() >= 0

    def test_apply_normalizes_if_exceeds_exposure(self):
        """Si suma > max_exposure, normalizar."""
        constraints = PortfolioConstraints(max_exposure=1.0, max_weight=1.0)

        target = np.array([0.40, 0.40, 0.40, 0.40])  # Suma = 1.6
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.apply(target, current)

        assert result.sum() <= 1.0 + 1e-6

    def test_apply_limits_turnover(self):
        """Turnover debe limitarse a max_turnover."""
        constraints = PortfolioConstraints(max_turnover=0.10, max_weight=1.0)

        # Cambio total de 0.50 (turnover = 0.25)
        target = np.array([0.50, 0.0, 0.25, 0.25])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.apply(target, current)

        # Turnover = sum(|delta|) / 2
        actual_turnover = np.abs(result - current).sum() / 2
        assert actual_turnover <= 0.10 + 1e-6

    def test_apply_preserves_sum_constraint(self):
        """Despues de aplicar restricciones, suma <= max_exposure."""
        constraints = PortfolioConstraints(max_exposure=0.90)

        target = np.array([0.30, 0.30, 0.30, 0.30])  # Suma = 1.2
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.apply(target, current)

        assert result.sum() <= 0.90 + 1e-6

    def test_apply_with_zero_current_weights(self):
        """Aplicar desde cash (todos pesos = 0)."""
        constraints = PortfolioConstraints()

        target = np.array([0.25, 0.25, 0.25, 0.25])
        current = np.zeros(4)

        result = constraints.apply(target, current)

        # Con max_turnover=0.30, solo se puede mover 0.30
        actual_turnover = np.abs(result - current).sum() / 2
        assert actual_turnover <= 0.30 + 1e-6


class TestConstraintViolations:
    """Tests para check_violations()."""

    def test_no_violations(self):
        """Pesos validos no deben tener violaciones."""
        constraints = PortfolioConstraints()

        weights = np.array([0.25, 0.25, 0.25, 0.25])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        violations = constraints.check_violations(weights, current)

        assert len(violations) == 0

    def test_max_weight_violation(self):
        """Detectar violacion de peso maximo."""
        constraints = PortfolioConstraints(max_weight=0.30)

        weights = np.array([0.50, 0.20, 0.15, 0.15])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        violations = constraints.check_violations(weights, current)

        assert any(v.name == "max_weight" for v in violations)

    def test_max_exposure_violation(self):
        """Detectar violacion de exposicion maxima."""
        constraints = PortfolioConstraints(max_exposure=0.80)

        weights = np.array([0.30, 0.30, 0.20, 0.20])  # Suma = 1.0
        current = np.array([0.25, 0.25, 0.25, 0.25])

        violations = constraints.check_violations(weights, current)

        assert any(v.name == "max_exposure" for v in violations)

    def test_turnover_violation(self):
        """Detectar violacion de turnover."""
        constraints = PortfolioConstraints(max_turnover=0.10)

        # Cambio de 0.50 de turnover
        weights = np.array([0.50, 0.0, 0.25, 0.25])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        violations = constraints.check_violations(weights, current)

        assert any(v.name == "max_turnover" for v in violations)

    def test_violation_severity(self):
        """Severidad debe reflejar magnitud de violacion."""
        constraints = PortfolioConstraints(max_weight=0.30)

        # Violacion pequena
        small_violation = np.array([0.35, 0.25, 0.20, 0.20])
        violations_small = constraints.check_violations(small_violation, small_violation)

        # Violacion grande
        large_violation = np.array([0.60, 0.20, 0.10, 0.10])
        violations_large = constraints.check_violations(large_violation, large_violation)

        severity_small = [v.severity for v in violations_small if v.name == "max_weight"][0]
        severity_large = [v.severity for v in violations_large if v.name == "max_weight"][0]

        assert severity_large > severity_small


class TestIsFeasible:
    """Tests para is_feasible()."""

    def test_feasible_weights(self):
        """Pesos validos son feasible."""
        constraints = PortfolioConstraints()

        weights = np.array([0.25, 0.25, 0.25, 0.25])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        assert constraints.is_feasible(weights, current)

    def test_infeasible_max_weight(self):
        """Pesos excediendo max_weight no son feasible."""
        constraints = PortfolioConstraints(max_weight=0.30)

        weights = np.array([0.50, 0.20, 0.15, 0.15])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        assert not constraints.is_feasible(weights, current)

    def test_infeasible_turnover(self):
        """Turnover excedido no es feasible."""
        constraints = PortfolioConstraints(max_turnover=0.05)

        weights = np.array([0.40, 0.10, 0.25, 0.25])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        assert not constraints.is_feasible(weights, current)


class TestProjectToFeasible:
    """Tests para project_to_feasible()."""

    def test_project_already_feasible(self):
        """Pesos ya feasible no cambian."""
        constraints = PortfolioConstraints()

        weights = np.array([0.25, 0.25, 0.25, 0.25])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.project_to_feasible(weights, current)

        np.testing.assert_array_almost_equal(result, weights)

    def test_project_converges(self):
        """Proyeccion debe converger a solucion feasible."""
        constraints = PortfolioConstraints()

        # Pesos claramente infeasibles
        weights = np.array([0.80, 0.60, 0.40, 0.20])
        current = np.zeros(4)

        result = constraints.project_to_feasible(weights, current)

        # Verificar que cumple restricciones (excepto turnover desde cero)
        assert result.max() <= constraints.max_weight + 1e-6
        assert result.min() >= constraints.min_weight - 1e-6
        assert result.sum() <= constraints.max_exposure + 1e-6

    def test_project_respects_turnover(self):
        """Proyeccion debe respetar turnover limite."""
        constraints = PortfolioConstraints(max_turnover=0.15)

        weights = np.array([0.50, 0.10, 0.20, 0.20])
        current = np.array([0.25, 0.25, 0.25, 0.25])

        result = constraints.project_to_feasible(weights, current)

        actual_turnover = np.abs(result - current).sum() / 2
        assert actual_turnover <= 0.15 + 1e-6
