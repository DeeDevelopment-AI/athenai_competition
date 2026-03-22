"""
Interfaz de entorno para agentes RL (compatible con SB3).
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .constraints import PortfolioConstraints
from .cost_model import CostModel
from .market_simulator import MarketSimulator
from .reward import RewardFunction

logger = logging.getLogger(__name__)


@dataclass
class EpisodeConfig:
    """Configuration for episode sampling."""

    random_start: bool = False  # If True, sample random start within window
    episode_length: Optional[int] = None  # Fixed episode length (in rebalance periods)
    min_episode_length: int = 10  # Minimum episode length (rebalance periods)
    warmup_periods: int = 21  # Days to skip at start for feature calculation


class TradingEnvironment(gym.Env):
    """
    Interfaz de entorno compatible con Gymnasium/SB3.

    Wraps MarketSimulator para proporcionar la interfaz estándar
    que espera Stable Baselines 3.

    NOTA: No heredamos lógica de gym.Env predefinida.
    Solo usamos gym.Env para compatibilidad de interfaz con SB3.
    El motor de simulación es completamente custom.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        algo_returns: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame] = None,
        cost_model: Optional[CostModel] = None,
        constraints: Optional[PortfolioConstraints] = None,
        reward_function: Optional[RewardFunction] = None,
        initial_capital: float = 1_000_000.0,
        rebalance_frequency: str = "weekly",
        train_start: Optional[pd.Timestamp] = None,
        train_end: Optional[pd.Timestamp] = None,
        reward_scale: float = 100.0,
        normalize_obs: bool = True,
        episode_config: Optional[EpisodeConfig] = None,
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
            train_start: Fecha de inicio para entrenamiento.
            train_end: Fecha de fin para entrenamiento.
            reward_scale: Factor de escala para reward.
            normalize_obs: Si normalizar observaciones.
            episode_config: Configuration for episode sampling (walk-forward).
        """
        super().__init__()

        self.algo_returns = algo_returns
        self.benchmark_weights = benchmark_weights
        self.cost_model = cost_model
        self.constraints = constraints
        self.reward_function = reward_function
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

        self.simulator = MarketSimulator(
            algo_returns=algo_returns,
            benchmark_weights=benchmark_weights,
            cost_model=cost_model,
            constraints=constraints,
            reward_function=reward_function,
            initial_capital=initial_capital,
            rebalance_frequency=rebalance_frequency,
        )

        self.train_start = train_start
        self.train_end = train_end
        self.reward_scale = reward_scale
        self.normalize_obs = normalize_obs
        self.episode_config = episode_config or EpisodeConfig()

        self.n_algos = self.simulator.n_algos

        # Pre-compute available dates for episode sampling
        self._available_dates = algo_returns.index.sort_values()
        self._setup_date_ranges()

        # Definir espacios de observación y acción
        # Observación: pesos + retornos + vols + métricas
        obs_dim = self.n_algos * 4 + 3  # Aproximado, ajustar según get_observation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Acción: pesos objetivo para cada algoritmo
        # Rango [0, 1] que luego se normaliza
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_algos,),
            dtype=np.float32,
        )

        # Estado interno
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_std: Optional[np.ndarray] = None
        self._rng = np.random.default_rng()
        self._episode_start: Optional[pd.Timestamp] = None
        self._episode_end: Optional[pd.Timestamp] = None
        self._steps_in_episode: int = 0

    def _setup_date_ranges(self) -> None:
        """Pre-compute valid date ranges for episode sampling."""
        dates = self._available_dates

        # Apply warmup offset
        warmup = self.episode_config.warmup_periods
        if len(dates) > warmup:
            self._valid_start_dates = dates[warmup:]
        else:
            self._valid_start_dates = dates

        # Filter by train_start/train_end if specified
        if self.train_start is not None:
            self._valid_start_dates = self._valid_start_dates[
                self._valid_start_dates >= self.train_start
            ]
        if self.train_end is not None:
            self._valid_start_dates = self._valid_start_dates[
                self._valid_start_dates <= self.train_end
            ]

    def _sample_episode_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Sample start and end dates for an episode.

        Returns:
            Tuple of (start_date, end_date).
        """
        config = self.episode_config

        if not config.random_start:
            # Use fixed window
            start = self.train_start or self._valid_start_dates[0]
            end = self.train_end or self._available_dates[-1]
            return start, end

        # Random start within valid range
        valid_dates = self._valid_start_dates

        if config.episode_length is not None:
            # Ensure we have enough dates for fixed episode length
            # Estimate days per rebalance period
            days_per_period = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}
            approx_days = config.episode_length * days_per_period.get(
                self.rebalance_frequency, 5
            )

            # Find valid start dates that allow full episode
            max_start_idx = max(0, len(valid_dates) - approx_days)
            if max_start_idx == 0:
                # Not enough data, use all
                start_idx = 0
            else:
                start_idx = self._rng.integers(0, max_start_idx + 1)

            start = valid_dates[start_idx]

            # Calculate end date
            end_idx = min(start_idx + approx_days, len(self._available_dates) - 1)
            end = self._available_dates[end_idx]
        else:
            # Random start, run to end of training window
            if len(valid_dates) <= config.min_episode_length:
                start_idx = 0
            else:
                # Leave room for minimum episode length
                days_per_period = {"daily": 1, "weekly": 5, "monthly": 21}
                min_days = config.min_episode_length * days_per_period.get(
                    self.rebalance_frequency, 5
                )
                max_start_idx = max(0, len(valid_dates) - min_days)
                start_idx = self._rng.integers(0, max_start_idx + 1)

            start = valid_dates[start_idx]
            end = self.train_end or self._available_dates[-1]

        return start, end

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reinicia el entorno.

        Args:
            seed: Semilla aleatoria para reproducibilidad.
            options: Opciones adicionales:
                - start_date: Override start date for this episode
                - end_date: Override end date for this episode

        Returns:
            Tuple (observación, info).
        """
        super().reset(seed=seed)

        # Set RNG seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Determine episode dates
        options = options or {}
        if "start_date" in options and "end_date" in options:
            self._episode_start = options["start_date"]
            self._episode_end = options["end_date"]
        else:
            self._episode_start, self._episode_end = self._sample_episode_dates()

        self._steps_in_episode = 0

        obs = self.simulator.reset(
            start_date=self._episode_start,
            end_date=self._episode_end,
        )

        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        return obs, {
            "date": self.simulator._state.current_date,
            "episode_start": self._episode_start,
            "episode_end": self._episode_end,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Ejecuta un paso en el entorno.

        Args:
            action: Vector de pesos objetivo (pre-normalización).

        Returns:
            Tuple (obs, reward, terminated, truncated, info).
        """
        # Normalizar acción para que sume a 1 (o menos)
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 1:
            action = action / action_sum
        elif action_sum == 0:
            # If all zeros, use equal weight
            action = np.ones(len(action)) / len(action)

        # Ejecutar paso en el simulador
        result = self.simulator.step(action)

        self._steps_in_episode += 1

        # Check for fixed episode length truncation
        truncated = False
        if self.episode_config.episode_length is not None:
            if self._steps_in_episode >= self.episode_config.episode_length:
                truncated = True

        # Procesar observación
        obs = result.observation
        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        # Escalar reward (handle NaN/Inf)
        reward = result.reward * self.reward_scale
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        # Add step count to info
        result.info["step"] = self._steps_in_episode

        return obs, float(reward), result.done, truncated, result.info

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normaliza observación para estabilidad de entrenamiento.

        Usa normalización running si se ha calculado previamente.
        """
        # Clippear valores extremos
        obs = np.clip(obs, -10, 10)

        # Reemplazar NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs.astype(np.float32)

    def render(self) -> None:
        """Renderiza el estado actual (modo texto)."""
        if self.simulator._state is None:
            print("Environment not initialized")
            return

        state = self.simulator._state
        print(f"\nDate: {state.current_date}")
        print(f"Portfolio Value: ${state.portfolio_value:,.2f}")
        print(f"Benchmark Value: ${state.benchmark_value:,.2f}")
        print(f"Current Weights: {state.current_weights.round(3)}")
        print(f"Drawdown: {self.simulator._calculate_drawdown():.2%}")

    def close(self) -> None:
        """Cierra el entorno."""
        pass

    def get_results(self) -> dict:
        """Devuelve resultados del episodio."""
        return self.simulator.get_results()


class VecTradingEnv:
    """
    Wrapper para crear entornos vectorizados.

    Útil para entrenamiento paralelo con SB3.
    """

    def __init__(
        self,
        n_envs: int,
        **env_kwargs,
    ):
        """
        Args:
            n_envs: Número de entornos paralelos.
            **env_kwargs: Argumentos para TradingEnvironment.
        """
        from stable_baselines3.common.vec_env import DummyVecEnv

        self.envs = DummyVecEnv([
            lambda: TradingEnvironment(**env_kwargs)
            for _ in range(n_envs)
        ])

    def __getattr__(self, name: str) -> Any:
        """Delega atributos al VecEnv interno."""
        return getattr(self.envs, name)
