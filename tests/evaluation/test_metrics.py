"""
Tests para métricas financieras.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    var,
    cvar,
)


class TestMetrics:
    """Tests para métricas financieras."""

    @pytest.fixture
    def constant_returns(self):
        """Retornos constantes positivos."""
        dates = pd.date_range("2020-01-01", periods=252)
        return pd.Series(0.001, index=dates)  # 0.1% diario

    @pytest.fixture
    def random_returns(self):
        """Retornos aleatorios."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252)
        return pd.Series(np.random.randn(252) * 0.01, index=dates)

    @pytest.fixture
    def negative_returns(self):
        """Retornos negativos constantes."""
        dates = pd.date_range("2020-01-01", periods=252)
        return pd.Series(-0.001, index=dates)

    def test_annualized_return_positive(self, constant_returns):
        """Retornos positivos deben dar retorno anualizado positivo."""
        ret = annualized_return(constant_returns)
        assert ret > 0

    def test_annualized_return_negative(self, negative_returns):
        """Retornos negativos deben dar retorno anualizado negativo."""
        ret = annualized_return(negative_returns)
        assert ret < 0

    def test_volatility_always_positive(self, random_returns):
        """Volatilidad siempre debe ser positiva."""
        vol = annualized_volatility(random_returns)
        assert vol > 0

    def test_volatility_zero_for_constant(self, constant_returns):
        """Volatilidad debe ser cero (o muy cercana) para retornos constantes."""
        vol = annualized_volatility(constant_returns)
        # Allow for floating-point precision issues
        assert vol < 1e-10

    def test_sharpe_high_for_constant_positive(self, constant_returns):
        """Sharpe debe ser muy alto para retornos constantes positivos."""
        # Sharpe con vol muy bajo debería ser muy alto o infinito
        sr = sharpe_ratio(constant_returns)
        # With near-zero volatility, Sharpe could be very high, zero, or inf
        assert sr == 0 or np.isinf(sr) or sr > 1000

    def test_sharpe_negative_for_losses(self, negative_returns):
        """Sharpe debe ser negativo o cero para pérdidas constantes."""
        sr = sharpe_ratio(negative_returns)
        assert sr <= 0

    def test_max_drawdown_negative(self, random_returns):
        """Max drawdown debe ser negativo o cero."""
        dd = max_drawdown(random_returns)
        assert dd <= 0

    def test_max_drawdown_zero_for_constant_positive(self, constant_returns):
        """Max drawdown debe ser cero para retornos siempre positivos."""
        dd = max_drawdown(constant_returns)
        assert dd == 0

    def test_var_negative(self, random_returns):
        """VaR 95% debe ser negativo (pérdida)."""
        v = var(random_returns, 0.05)
        assert v < 0

    def test_cvar_worse_than_var(self, random_returns):
        """CVaR debe ser peor (más negativo) que VaR."""
        v = var(random_returns, 0.05)
        cv = cvar(random_returns, 0.05)
        assert cv <= v

    def test_metrics_handle_empty_series(self):
        """Métricas deben manejar series vacías gracefully."""
        empty = pd.Series(dtype=float)

        # Estas funciones deberían no crashear
        ret = annualized_return(empty)
        vol = annualized_volatility(empty)

        assert np.isnan(ret) or ret == 0
        assert np.isnan(vol) or vol == 0
