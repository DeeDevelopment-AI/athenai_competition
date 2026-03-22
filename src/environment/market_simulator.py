"""
Motor de simulación de mercado event-driven.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .constraints import PortfolioConstraints
from .cost_model import CostModel
from .reward import RewardComponents, RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class SimulatorState:
    """Estado interno del simulador."""

    current_date: pd.Timestamp
    current_weights: np.ndarray
    portfolio_value: float
    benchmark_value: float
    cash: float
    equity_curve: list[float] = field(default_factory=list)
    benchmark_curve: list[float] = field(default_factory=list)
    weights_history: list[np.ndarray] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)


@dataclass
class StepResult:
    """Resultado de un paso de simulación."""

    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class MarketSimulator:
    """
    Simulador event-driven para meta-allocation.

    No simula mercados directamente (los algos son caja negra).
    Lo que simula es:
    - Los retornos de cada algoritmo (a partir de históricos)
    - Los costes de cambiar de pesos
    - Las restricciones de cartera
    - El paso temporal según frecuencia de rebalanceo
    """

    def __init__(
        self,
        algo_returns: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame] = None,
        cost_model: Optional[CostModel] = None,
        constraints: Optional[PortfolioConstraints] = None,
        reward_function: Optional[RewardFunction] = None,
        initial_capital: float = 1_000_000.0,
        rebalance_frequency: str = "weekly",
    ):
        """
        Args:
            algo_returns: DataFrame [dates x algos] con retornos diarios.
            benchmark_weights: DataFrame [dates x algos] con pesos del benchmark.
            cost_model: Modelo de costes de transacción.
            constraints: Restricciones de cartera.
            reward_function: Función de reward.
            initial_capital: Capital inicial.
            rebalance_frequency: "daily", "weekly", "monthly".
        """
        self.algo_returns = algo_returns.sort_index()

        # Align benchmark_weights columns to match algo_returns columns
        if benchmark_weights is not None:
            common_cols = [c for c in algo_returns.columns if c in benchmark_weights.columns]
            if common_cols:
                # Reindex to match algo_returns columns, fill missing with 0
                self.benchmark_weights = benchmark_weights.reindex(
                    columns=algo_returns.columns, fill_value=0.0
                )
            else:
                # No common columns - use None (will fall back to equal weight)
                logger.warning("No common columns between algo_returns and benchmark_weights")
                self.benchmark_weights = None
        else:
            self.benchmark_weights = benchmark_weights

        self.cost_model = cost_model or CostModel()
        self.constraints = constraints or PortfolioConstraints()
        self.reward_function = reward_function or RewardFunction()
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

        self.n_algos = len(algo_returns.columns)
        self.algo_names = list(algo_returns.columns)

        # Estado
        self._state: Optional[SimulatorState] = None
        self._start_date: Optional[pd.Timestamp] = None
        self._end_date: Optional[pd.Timestamp] = None
        self._date_index: int = 0
        self._rebalance_dates: list[pd.Timestamp] = []

    def reset(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """
        Inicializa episodio entre dos fechas.

        Args:
            start_date: Fecha de inicio (default: primera fecha disponible).
            end_date: Fecha de fin (default: última fecha disponible).

        Returns:
            Observación inicial.
        """
        # Determinar rango de fechas
        available_dates = self.algo_returns.index
        self._start_date = start_date or available_dates[0]
        self._end_date = end_date or available_dates[-1]

        # Filtrar fechas en el rango
        mask = (available_dates >= self._start_date) & (available_dates <= self._end_date)
        self._dates = available_dates[mask].tolist()

        if len(self._dates) < 2:
            raise ValueError("Not enough dates in the specified range")

        # Generar fechas de rebalanceo
        self._rebalance_dates = self._generate_rebalance_dates()
        self._date_index = 0

        # Inicializar estado
        initial_weights = np.ones(self.n_algos) / self.n_algos  # Equal weight inicial
        self._state = SimulatorState(
            current_date=self._dates[0],
            current_weights=initial_weights,
            portfolio_value=self.initial_capital,
            benchmark_value=self.initial_capital,
            cash=0.0,
            equity_curve=[self.initial_capital],
            benchmark_curve=[self.initial_capital],
            weights_history=[initial_weights.copy()],
            dates=[self._dates[0]],
        )

        logger.info(
            f"Simulator reset: {self._start_date} to {self._end_date}, "
            f"{len(self._dates)} days, {len(self._rebalance_dates)} rebalance dates"
        )

        return self.get_observation()

    def _generate_rebalance_dates(self) -> list[pd.Timestamp]:
        """Genera lista de fechas de rebalanceo según frecuencia."""
        if self.rebalance_frequency == "daily":
            return self._dates.copy()

        rebalance_dates = [self._dates[0]]
        last_rebalance = self._dates[0]

        for date in self._dates[1:]:
            days_since = (date - last_rebalance).days

            if self.rebalance_frequency == "weekly" and days_since >= 5:
                rebalance_dates.append(date)
                last_rebalance = date
            elif self.rebalance_frequency == "monthly" and days_since >= 21:
                rebalance_dates.append(date)
                last_rebalance = date
            elif self.rebalance_frequency == "quarterly" and days_since >= 63:
                rebalance_dates.append(date)
                last_rebalance = date

        return rebalance_dates

    def step(self, target_weights: np.ndarray) -> StepResult:
        """
        Ejecuta un paso de simulación.

        Args:
            target_weights: Pesos objetivo (pre-restricciones).

        Returns:
            StepResult con observación, reward, done, info.
        """
        if self._state is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")

        current_date = self._state.current_date
        old_weights = self._state.current_weights.copy()

        # 1. Aplicar restricciones
        new_weights = self.constraints.apply(target_weights, old_weights)

        # 2. Calcular costes de transición
        turnover = np.abs(new_weights - old_weights).sum() / 2
        transaction_costs = self.cost_model.compute_cost_as_return(old_weights, new_weights)

        # 3. Avanzar al siguiente periodo de rebalanceo
        next_rebalance_idx = self._find_next_rebalance_index()
        if next_rebalance_idx >= len(self._rebalance_dates):
            # Fin del episodio
            return StepResult(
                observation=self.get_observation(),
                reward=0.0,
                done=True,
                info={"reason": "end_of_data"},
            )

        next_date = self._rebalance_dates[next_rebalance_idx]

        # 4. Calcular retornos del periodo
        period_returns = self._get_period_returns(current_date, next_date)
        portfolio_return = np.dot(new_weights, period_returns) - transaction_costs
        benchmark_return = self._get_benchmark_return(current_date, next_date)

        # 5. Actualizar valores
        self._state.portfolio_value *= (1 + portfolio_return)
        self._state.benchmark_value *= (1 + benchmark_return)

        # 6. Calcular métricas para reward
        current_drawdown = self._calculate_drawdown()
        portfolio_vol = self._calculate_rolling_vol()
        benchmark_vol = self._calculate_benchmark_rolling_vol()

        # 7. Calcular reward
        reward_components = self.reward_function.compute(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            transaction_costs=transaction_costs,
            turnover=turnover,
            current_drawdown=current_drawdown,
            portfolio_vol=portfolio_vol,
            benchmark_vol=benchmark_vol,
        )

        # 8. Actualizar estado
        self._state.current_date = next_date
        self._state.current_weights = new_weights
        self._state.equity_curve.append(self._state.portfolio_value)
        self._state.benchmark_curve.append(self._state.benchmark_value)
        self._state.weights_history.append(new_weights.copy())
        self._state.dates.append(next_date)
        self._date_index = self._dates.index(next_date)

        # 9. Verificar si terminó
        done = next_rebalance_idx >= len(self._rebalance_dates) - 1

        info = {
            "date": next_date,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "transaction_costs": transaction_costs,
            "turnover": turnover,
            "portfolio_value": self._state.portfolio_value,
            "benchmark_value": self._state.benchmark_value,
            "drawdown": current_drawdown,
            "reward_components": reward_components,
            "weights": new_weights.copy(),  # Include constrained weights for debugging
        }

        return StepResult(
            observation=self.get_observation(),
            reward=reward_components.total,
            done=done,
            info=info,
        )

    def _find_next_rebalance_index(self) -> int:
        """Encuentra índice de la siguiente fecha de rebalanceo."""
        current_date = self._state.current_date
        for i, date in enumerate(self._rebalance_dates):
            if date > current_date:
                return i
        return len(self._rebalance_dates)

    def _get_period_returns(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> np.ndarray:
        """Obtiene retornos acumulados del periodo para cada algoritmo."""
        mask = (self.algo_returns.index > start_date) & (self.algo_returns.index <= end_date)
        period_returns = self.algo_returns.loc[mask]

        if len(period_returns) == 0:
            return np.zeros(self.n_algos)

        # Retorno acumulado (handle NaN by filling with 0)
        period_returns_clean = period_returns.fillna(0.0)
        cumulative = (1 + period_returns_clean).prod() - 1
        result = cumulative.values

        # Replace any remaining NaN/Inf with 0
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _get_benchmark_return(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> float:
        """Calcula retorno del benchmark en el periodo."""
        if self.benchmark_weights is None:
            # Usar equal weight como proxy
            period_returns = self._get_period_returns(start_date, end_date)
            result = float(np.nanmean(period_returns))
            return 0.0 if np.isnan(result) else result

        # Usar pesos del benchmark
        mask = (self.algo_returns.index > start_date) & (self.algo_returns.index <= end_date)
        dates_in_period = self.algo_returns.index[mask]

        total_return = 0.0
        for date in dates_in_period:
            if date in self.benchmark_weights.index:
                weights = self.benchmark_weights.loc[date].values
            else:
                # Usar último peso conocido
                prev_dates = self.benchmark_weights.index[self.benchmark_weights.index <= date]
                if len(prev_dates) > 0:
                    weights = self.benchmark_weights.loc[prev_dates[-1]].values
                else:
                    weights = np.ones(self.n_algos) / self.n_algos

            daily_return = self.algo_returns.loc[date].values

            # Handle NaN in weights and returns
            weights = np.nan_to_num(weights, nan=0.0)
            daily_return = np.nan_to_num(daily_return, nan=0.0)

            # Normalize weights if needed
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum

            total_return += np.dot(weights, daily_return)

        return float(np.nan_to_num(total_return, nan=0.0))

    def _calculate_drawdown(self) -> float:
        """Calcula drawdown actual del portfolio."""
        if len(self._state.equity_curve) < 2:
            return 0.0

        peak = max(self._state.equity_curve)
        current = self._state.portfolio_value

        if peak <= 0:
            return 0.0

        return (current - peak) / peak

    def _calculate_rolling_vol(self, window: int = 21) -> float:
        """Calcula volatilidad rolling del portfolio."""
        if len(self._state.equity_curve) < window:
            return 0.0

        values = np.array(self._state.equity_curve[-window:])
        returns = np.diff(values) / values[:-1]

        return returns.std() * np.sqrt(252)

    def _calculate_benchmark_rolling_vol(self, window: int = 21) -> float:
        """Calcula volatilidad rolling del benchmark."""
        if len(self._state.benchmark_curve) < window:
            return 0.0

        values = np.array(self._state.benchmark_curve[-window:])
        returns = np.diff(values) / values[:-1]

        return returns.std() * np.sqrt(252)

    def get_observation(self) -> np.ndarray:
        """
        Construye vector de observación actual.

        Incluye:
        - Pesos actuales
        - Retornos recientes de cada algoritmo
        - Volatilidades
        - Drawdown actual
        - Features de régimen
        """
        if self._state is None:
            return np.zeros(self.n_algos * 4 + 3, dtype=np.float32)  # Placeholder

        current_date = self._state.current_date
        lookback = 21

        # Obtener datos históricos
        idx_mask = self.algo_returns.index <= current_date
        historical = self.algo_returns.loc[idx_mask].tail(lookback)

        # Fill NaN with 0 for calculations
        historical_clean = historical.fillna(0.0)

        # Features
        obs = []

        # 1. Pesos actuales
        obs.extend(self._state.current_weights)

        # 2. Retornos recientes (5d, 21d)
        if len(historical_clean) >= 5:
            ret_5d = (1 + historical_clean.tail(5)).prod() - 1
            obs.extend(np.nan_to_num(ret_5d.values, nan=0.0))
        else:
            obs.extend([0.0] * self.n_algos)

        if len(historical_clean) >= 21:
            ret_21d = (1 + historical_clean).prod() - 1
            obs.extend(np.nan_to_num(ret_21d.values, nan=0.0))
        else:
            obs.extend([0.0] * self.n_algos)

        # 3. Volatilidades (21d)
        if len(historical_clean) >= 10:
            vol_21d = historical_clean.std() * np.sqrt(252)
            obs.extend(np.nan_to_num(vol_21d.values, nan=0.0))
        else:
            obs.extend([0.0] * self.n_algos)

        # 4. Correlación media
        if len(historical_clean) >= 10:
            corr = historical_clean.corr().values
            corr = np.nan_to_num(corr, nan=0.0)
            triu_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            corr_vals = corr[triu_mask]
            avg_corr = corr_vals.mean() if len(corr_vals) > 0 else 0.0
            obs.append(float(np.nan_to_num(avg_corr, nan=0.0)))
        else:
            obs.append(0.0)

        # 5. Drawdown actual
        obs.append(self._calculate_drawdown())

        # 6. Exceso de retorno acumulado vs benchmark
        if self._state.benchmark_value > 0:
            excess = (
                self._state.portfolio_value / self._state.benchmark_value - 1
            )
            obs.append(float(np.nan_to_num(excess, nan=0.0)))
        else:
            obs.append(0.0)

        # Final cleanup - ensure no NaN/Inf in observation
        result = np.array(obs, dtype=np.float32)
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

        return result

    def get_state(self) -> SimulatorState:
        """Devuelve estado interno (para debugging)."""
        return self._state

    def get_results(self) -> dict:
        """
        Devuelve resultados del episodio.

        Returns:
            Dict con equity curves, pesos, métricas.
        """
        if self._state is None:
            return {}

        return {
            "dates": self._state.dates,
            "equity_curve": self._state.equity_curve,
            "benchmark_curve": self._state.benchmark_curve,
            "weights_history": self._state.weights_history,
            "final_portfolio_value": self._state.portfolio_value,
            "final_benchmark_value": self._state.benchmark_value,
            "total_return": (
                self._state.portfolio_value / self.initial_capital - 1
            ),
            "benchmark_return": (
                self._state.benchmark_value / self.initial_capital - 1
            ),
        }
