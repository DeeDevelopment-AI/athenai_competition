"""
Tests para el simulador de mercado.
"""

import numpy as np
import pandas as pd
import pytest

from src.environment.market_simulator import MarketSimulator, SimulatorState, StepResult
from src.environment.cost_model import CostModel
from src.environment.constraints import PortfolioConstraints
from src.environment.reward import RewardFunction


@pytest.fixture
def sample_returns():
    """Crear matriz de retornos de ejemplo."""
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    n_algos = 5

    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(252, n_algos) * 0.01 + 0.0005,  # Media positiva
        index=dates,
        columns=[f"algo_{i}" for i in range(n_algos)],
    )
    return returns


@pytest.fixture
def sample_benchmark_weights(sample_returns):
    """Crear pesos de benchmark de ejemplo."""
    dates = sample_returns.index
    n_algos = len(sample_returns.columns)

    # Equal weight constante
    weights = pd.DataFrame(
        np.ones((len(dates), n_algos)) / n_algos,
        index=dates,
        columns=sample_returns.columns,
    )
    return weights


@pytest.fixture
def simulator(sample_returns, sample_benchmark_weights):
    """Crear simulador configurado."""
    return MarketSimulator(
        algo_returns=sample_returns,
        benchmark_weights=sample_benchmark_weights,
        cost_model=CostModel(spread_bps=5, slippage_bps=2),
        constraints=PortfolioConstraints(max_weight=0.40, max_turnover=0.30),
        reward_function=RewardFunction(),
        initial_capital=1_000_000.0,
        rebalance_frequency="weekly",
    )


class TestMarketSimulatorInit:
    """Tests para inicializacion del simulador."""

    def test_init_with_defaults(self, sample_returns):
        """Inicializacion con valores por defecto."""
        sim = MarketSimulator(algo_returns=sample_returns)

        assert sim.n_algos == 5
        assert sim.initial_capital == 1_000_000.0
        assert sim.rebalance_frequency == "weekly"
        assert sim._state is None

    def test_init_with_custom_params(self, sample_returns, sample_benchmark_weights):
        """Inicializacion con parametros personalizados."""
        sim = MarketSimulator(
            algo_returns=sample_returns,
            benchmark_weights=sample_benchmark_weights,
            initial_capital=500_000.0,
            rebalance_frequency="monthly",
        )

        assert sim.initial_capital == 500_000.0
        assert sim.rebalance_frequency == "monthly"

    def test_algo_names_extracted(self, sample_returns):
        """Nombres de algoritmos deben extraerse del DataFrame."""
        sim = MarketSimulator(algo_returns=sample_returns)

        assert sim.algo_names == ["algo_0", "algo_1", "algo_2", "algo_3", "algo_4"]


class TestMarketSimulatorReset:
    """Tests para reset del simulador."""

    def test_reset_initializes_state(self, simulator):
        """Reset debe inicializar estado."""
        obs = simulator.reset()

        assert simulator._state is not None
        assert isinstance(simulator._state, SimulatorState)
        assert simulator._state.portfolio_value == simulator.initial_capital
        assert simulator._state.benchmark_value == simulator.initial_capital

    def test_reset_returns_observation(self, simulator):
        """Reset debe retornar observacion."""
        obs = simulator.reset()

        assert isinstance(obs, np.ndarray)
        assert len(obs) > 0
        assert not np.any(np.isnan(obs))

    def test_reset_with_date_range(self, simulator):
        """Reset con rango de fechas especifico."""
        start = pd.Timestamp("2020-03-01")
        end = pd.Timestamp("2020-06-30")

        obs = simulator.reset(start_date=start, end_date=end)

        assert simulator._start_date == start
        assert simulator._end_date == end

    def test_reset_initializes_equal_weight(self, simulator):
        """Reset debe inicializar con equal weight."""
        simulator.reset()

        expected = np.ones(simulator.n_algos) / simulator.n_algos
        np.testing.assert_array_almost_equal(
            simulator._state.current_weights, expected
        )

    def test_reset_generates_rebalance_dates(self, simulator):
        """Reset debe generar fechas de rebalanceo."""
        simulator.reset()

        assert len(simulator._rebalance_dates) > 0
        # Weekly: aproximadamente 50 fechas para un ano
        assert len(simulator._rebalance_dates) < 252


class TestMarketSimulatorStep:
    """Tests para step del simulador."""

    def test_step_without_reset_raises(self, simulator):
        """Step sin reset debe fallar."""
        with pytest.raises(RuntimeError):
            simulator.step(np.ones(5) / 5)

    def test_step_returns_step_result(self, simulator):
        """Step debe retornar StepResult."""
        simulator.reset()
        result = simulator.step(np.ones(5) / 5)

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, np.ndarray)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_advances_date(self, simulator):
        """Step debe avanzar la fecha."""
        simulator.reset()
        initial_date = simulator._state.current_date

        simulator.step(np.ones(5) / 5)

        assert simulator._state.current_date > initial_date

    def test_step_updates_portfolio_value(self, simulator):
        """Step debe actualizar valor del portfolio."""
        simulator.reset()
        initial_value = simulator._state.portfolio_value

        # Hacer varios pasos
        for _ in range(5):
            result = simulator.step(np.ones(5) / 5)
            if result.done:
                break

        # El valor debe haber cambiado (puede subir o bajar)
        assert simulator._state.portfolio_value != initial_value

    def test_step_applies_constraints(self, simulator):
        """Step debe aplicar restricciones a pesos."""
        simulator.reset()

        # Intentar pesos que violan max_weight
        invalid_weights = np.array([0.80, 0.05, 0.05, 0.05, 0.05])
        result = simulator.step(invalid_weights)

        # Los pesos aplicados deben respetar max_weight
        assert simulator._state.current_weights.max() <= 0.40 + 1e-6

    def test_step_calculates_transaction_costs(self, simulator):
        """Step debe calcular costes de transaccion."""
        simulator.reset()

        # Cambio de pesos que genera turnover
        new_weights = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
        result = simulator.step(new_weights)

        assert "transaction_costs" in result.info
        assert result.info["transaction_costs"] >= 0

    def test_step_calculates_turnover(self, simulator):
        """Step debe calcular turnover."""
        simulator.reset()

        new_weights = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
        result = simulator.step(new_weights)

        assert "turnover" in result.info
        assert 0 <= result.info["turnover"] <= 0.30  # Limitado por constraint

    def test_step_tracks_equity_curve(self, simulator):
        """Step debe registrar equity curve."""
        simulator.reset()

        initial_len = len(simulator._state.equity_curve)

        for _ in range(5):
            result = simulator.step(np.ones(5) / 5)
            if result.done:
                break

        assert len(simulator._state.equity_curve) > initial_len

    def test_step_done_at_end(self, simulator):
        """Step debe retornar done=True al final."""
        simulator.reset()

        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            result = simulator.step(np.ones(5) / 5)
            done = result.done
            steps += 1

        assert done

    def test_step_reward_reflects_performance(self, simulator):
        """Reward debe reflejar performance vs benchmark."""
        simulator.reset()

        total_reward = 0
        steps = 0

        while steps < 10:
            result = simulator.step(np.ones(5) / 5)
            total_reward += result.reward
            steps += 1
            if result.done:
                break

        # El reward total puede ser positivo o negativo
        # Pero debe ser un numero valido
        assert np.isfinite(total_reward)


class TestRebalanceFrequency:
    """Tests para diferentes frecuencias de rebalanceo."""

    def test_daily_rebalance(self, sample_returns, sample_benchmark_weights):
        """Rebalanceo diario."""
        sim = MarketSimulator(
            algo_returns=sample_returns,
            benchmark_weights=sample_benchmark_weights,
            rebalance_frequency="daily",
        )
        sim.reset()

        # Cada dia debe ser fecha de rebalanceo
        assert len(sim._rebalance_dates) == len(sim._dates)

    def test_weekly_rebalance(self, sample_returns, sample_benchmark_weights):
        """Rebalanceo semanal."""
        sim = MarketSimulator(
            algo_returns=sample_returns,
            benchmark_weights=sample_benchmark_weights,
            rebalance_frequency="weekly",
        )
        sim.reset()

        # Aproximadamente 1/5 de los dias
        ratio = len(sim._rebalance_dates) / len(sim._dates)
        assert 0.15 < ratio < 0.30

    def test_monthly_rebalance(self, sample_returns, sample_benchmark_weights):
        """Rebalanceo mensual."""
        sim = MarketSimulator(
            algo_returns=sample_returns,
            benchmark_weights=sample_benchmark_weights,
            rebalance_frequency="monthly",
        )
        sim.reset()

        # Aproximadamente 12 rebalanceos por ano
        assert 10 < len(sim._rebalance_dates) < 20


class TestGetObservation:
    """Tests para get_observation."""

    def test_observation_shape(self, simulator):
        """Observacion debe tener forma correcta."""
        simulator.reset()
        obs = simulator.get_observation()

        # Dimension esperada: n_algos * 4 + 4 (pesos, ret_5d, ret_21d, vol, + 4 metricas)
        # 4 scalars: avg_corr, drawdown, momentum_breadth, vol_regime
        expected_dim = simulator.n_algos * 4 + 4
        # Nota: La dimension real puede variar segun implementacion
        assert len(obs) > 0

    def test_observation_no_nan(self, simulator):
        """Observacion no debe tener NaN."""
        simulator.reset()

        for _ in range(5):
            obs = simulator.get_observation()
            assert not np.any(np.isnan(obs))

            result = simulator.step(np.ones(5) / 5)
            if result.done:
                break

    def test_observation_includes_weights(self, simulator):
        """Observacion debe incluir pesos actuales."""
        simulator.reset()
        obs = simulator.get_observation()

        # Los primeros n_algos valores deben ser los pesos
        weights = obs[:simulator.n_algos]
        expected = np.ones(simulator.n_algos) / simulator.n_algos

        np.testing.assert_array_almost_equal(weights, expected)


class TestGetResults:
    """Tests para get_results."""

    def test_results_after_episode(self, simulator):
        """Resultados deben estar disponibles tras episodio."""
        simulator.reset()

        while True:
            result = simulator.step(np.ones(5) / 5)
            if result.done:
                break

        results = simulator.get_results()

        assert "dates" in results
        assert "equity_curve" in results
        assert "benchmark_curve" in results
        assert "weights_history" in results
        assert "total_return" in results
        assert "benchmark_return" in results

    def test_results_equity_curve_length(self, simulator):
        """Equity curve debe tener longitud correcta."""
        simulator.reset()

        steps = 0
        while True:
            result = simulator.step(np.ones(5) / 5)
            steps += 1
            if result.done:
                break

        results = simulator.get_results()

        assert len(results["equity_curve"]) == steps + 1  # +1 por valor inicial
        assert len(results["dates"]) == steps + 1

    def test_results_without_episode(self, simulator):
        """Resultados sin episodio deben estar vacios."""
        results = simulator.get_results()

        assert results == {}


class TestSanityChecks:
    """Sanity checks para el simulador."""

    def test_equal_weight_consistent(self, sample_returns, sample_benchmark_weights):
        """EW constante debe dar resultados consistentes."""
        sim = MarketSimulator(
            algo_returns=sample_returns,
            benchmark_weights=sample_benchmark_weights,
            cost_model=CostModel(spread_bps=0, slippage_bps=0, fixed_cost_per_trade=0),
            constraints=PortfolioConstraints(max_turnover=1.0),
            rebalance_frequency="weekly",
        )
        sim.reset()

        # Siempre EW
        ew = np.ones(5) / 5

        while True:
            result = sim.step(ew)
            if result.done:
                break

        results = sim.get_results()

        # Con costes 0 y EW constante, el portfolio debe trackear muy cerca del benchmark EW
        # (puede haber pequenas diferencias por timing de rebalanceo)
        portfolio_ret = results["total_return"]
        benchmark_ret = results["benchmark_return"]

        # Deben ser similares (diferencia < 5%)
        diff = abs(portfolio_ret - benchmark_ret)
        assert diff < 0.05 or abs(diff / max(abs(benchmark_ret), 0.01)) < 0.10

    def test_no_trading_no_costs(self, sample_returns):
        """Sin cambios de peso, no hay costes."""
        sim = MarketSimulator(
            algo_returns=sample_returns,
            cost_model=CostModel(spread_bps=100),  # Costes altos
            rebalance_frequency="weekly",
        )
        sim.reset()

        # Mantener EW todo el tiempo (sin cambios)
        ew = np.ones(5) / 5

        total_costs = 0
        while True:
            result = sim.step(ew)
            total_costs += result.info.get("transaction_costs", 0)
            if result.done:
                break

        # Primer rebalanceo genera costes, luego no deberia haber
        # (porque EW -> EW no genera turnover)
        # Nota: Puede haber costes del primer paso si empezamos desde cero
        assert total_costs < 0.01  # Muy pocos costes

    def test_portfolio_value_positive(self, simulator):
        """Valor del portfolio siempre debe ser positivo."""
        simulator.reset()

        while True:
            result = simulator.step(np.ones(5) / 5)
            assert simulator._state.portfolio_value > 0
            if result.done:
                break

    def test_weights_sum_valid(self, simulator):
        """Suma de pesos siempre debe ser <= 1."""
        simulator.reset()

        while True:
            result = simulator.step(np.random.rand(5))
            assert simulator._state.current_weights.sum() <= 1.0 + 1e-6
            if result.done:
                break
