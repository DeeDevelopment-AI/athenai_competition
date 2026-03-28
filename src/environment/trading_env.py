"""
Interfaz de entorno para agentes RL (compatible con SB3).

Supports two modes:
1. Standard mode: RL agent outputs absolute portfolio weights
2. Hybrid mode: MPT base allocation + RL tilts (more stable training)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .constraints import PortfolioConstraints
from .cost_model import CostModel
from .market_simulator import MarketSimulator
from .reward import RewardFunction
from .universe_encoder import AlgoUniverseEncoder

logger = logging.getLogger(__name__)


class BaseAllocatorType(Enum):
    """Types of base allocators for hybrid mode."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"  # Inverse volatility weighting
    MIN_VARIANCE = "min_variance"  # Minimum variance (requires covariance)


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
        encoder: Optional[AlgoUniverseEncoder] = None,
        # Hybrid mode parameters
        hybrid_mode: bool = False,
        base_allocator: str = "risk_parity",
        max_tilt: float = 0.15,
        vol_lookback: int = 63,
        # Cluster-based reward shaping (soft mode)
        cluster_filter=None,
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
            encoder: Optional fitted AlgoUniverseEncoder.  When provided, the
                     observation/action spaces use the compressed PCA dimensions
                     and observations/actions are encoded/decoded automatically.
            hybrid_mode: If True, use MPT base + RL tilts instead of absolute weights.
            base_allocator: Base allocator type: "risk_parity", "equal_weight", "min_variance".
            max_tilt: Maximum tilt per asset (e.g., 0.15 = ±15%).
            vol_lookback: Lookback window for volatility calculation (risk parity).
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
        self.encoder = encoder

        # Hybrid mode: MPT base + RL tilts
        self.hybrid_mode = hybrid_mode
        self.max_tilt = max_tilt
        self.vol_lookback = vol_lookback
        self._base_allocator_type = BaseAllocatorType(base_allocator)
        self.cluster_filter = cluster_filter

        self.n_algos = self.simulator.n_algos

        # Pre-compute available dates for episode sampling
        self._available_dates = algo_returns.index.sort_values()
        self._setup_date_ranges()

        # Observation and action spaces — compressed when encoder is provided
        if encoder is not None:
            obs_dim = encoder.obs_dim
            action_dim = encoder.action_dim
        else:
            obs_dim = self.n_algos * 4 + 4  # weights, ret5d, ret21d, vol + 4 scalars
            action_dim = self.n_algos

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space depends on mode
        if hybrid_mode:
            # Hybrid mode: tilts in [-1, 1], will be scaled by max_tilt
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                dtype=np.float32,
            )
            logger.info(f"Hybrid mode enabled: {base_allocator} base + RL tilts (max_tilt={max_tilt})")
        else:
            # Standard mode: absolute weights in [0, 1]
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(action_dim,),
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

        current_date = self.simulator._state.current_date

        if self.encoder is not None:
            obs = self.encoder.encode_obs(obs, current_date)

        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        return obs, {
            "date": current_date,
            "episode_start": self._episode_start,
            "episode_end": self._episode_end,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Ejecuta un paso en el entorno.

        Args:
            action: In standard mode: target weights [0, 1].
                    In hybrid mode: tilts [-1, 1] to add to base allocation.

        Returns:
            Tuple (obs, reward, terminated, truncated, info).
        """
        if self.encoder is not None:
            # Decode PCA action → full algo weight vector before simulator
            decoded_action = self.encoder.decode_action(action, self.simulator._state.current_date)
        else:
            decoded_action = action

        if self.hybrid_mode:
            # Hybrid mode: base allocation + RL tilts
            base_weights = self._compute_base_weights()

            # Scale tilts by max_tilt and apply
            tilts = np.clip(decoded_action, -1.0, 1.0) * self.max_tilt
            sim_action = base_weights + tilts

            # Ensure non-negative and normalize
            sim_action = np.maximum(sim_action, 0.0)
            action_sum = sim_action.sum()
            if action_sum > 0:
                sim_action = sim_action / action_sum
            else:
                sim_action = np.ones(len(sim_action)) / len(sim_action)
        else:
            # Standard mode: absolute weights
            sim_action = np.clip(decoded_action, 0, 1)
            action_sum = sim_action.sum()
            if action_sum > 1:
                sim_action = sim_action / action_sum
            elif action_sum == 0:
                sim_action = np.ones(len(sim_action)) / len(sim_action)

        # Ejecutar paso en el simulador
        result = self.simulator.step(sim_action)

        self._steps_in_episode += 1

        # Check for fixed episode length truncation
        truncated = False
        if self.episode_config.episode_length is not None:
            if self._steps_in_episode >= self.episode_config.episode_length:
                truncated = True

        # Procesar observación
        obs = result.observation
        if self.encoder is not None:
            obs = self.encoder.encode_obs(obs, self.simulator._state.current_date)
        if self.normalize_obs:
            obs = self._normalize_observation(obs)

        # Escalar reward (handle NaN/Inf)
        raw_reward = result.reward
        if self.cluster_filter is not None:
            raw_reward += self.cluster_filter.compute_reward_bonus(sim_action)
        reward = raw_reward * self.reward_scale
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

    def _compute_base_weights(self) -> np.ndarray:
        """
        Compute base allocation weights using the configured allocator.

        Returns:
            np.ndarray of weights summing to 1.
        """
        n = self.n_algos

        if self._base_allocator_type == BaseAllocatorType.EQUAL_WEIGHT:
            return np.ones(n, dtype=np.float32) / n

        elif self._base_allocator_type == BaseAllocatorType.RISK_PARITY:
            # Inverse volatility weighting
            current_date = self.simulator._state.current_date

            # Find current index in the returns array using searchsorted
            current_idx = int(np.searchsorted(
                self.simulator._returns_index_np,
                np.datetime64(current_date),
                side='right'
            ))

            # Get recent returns for volatility calculation
            start_idx = max(0, current_idx - self.vol_lookback)
            if start_idx >= current_idx or current_idx == 0:
                return np.ones(n, dtype=np.float32) / n

            recent_returns = self.simulator._returns_np[start_idx:current_idx, :]

            if len(recent_returns) < 10:
                return np.ones(n, dtype=np.float32) / n

            # Compute volatilities (std of returns)
            with np.errstate(invalid='ignore'):
                vols = np.nanstd(recent_returns, axis=0)

            # Handle zero or NaN volatility
            vols = np.nan_to_num(vols, nan=1e-6, posinf=1e-6, neginf=1e-6)
            vols = np.maximum(vols, 1e-8)

            # Inverse volatility weights
            inv_vols = 1.0 / vols
            weights = inv_vols / inv_vols.sum()

            return weights.astype(np.float32)

        elif self._base_allocator_type == BaseAllocatorType.MIN_VARIANCE:
            # Minimum variance (simplified: use inverse variance)
            current_date = self.simulator._state.current_date

            # Find current index in the returns array
            current_idx = int(np.searchsorted(
                self.simulator._returns_index_np,
                np.datetime64(current_date),
                side='right'
            ))

            start_idx = max(0, current_idx - self.vol_lookback)
            if start_idx >= current_idx or current_idx == 0:
                return np.ones(n, dtype=np.float32) / n

            recent_returns = self.simulator._returns_np[start_idx:current_idx, :]

            if len(recent_returns) < 10:
                return np.ones(n, dtype=np.float32) / n

            # Compute variances
            with np.errstate(invalid='ignore'):
                variances = np.nanvar(recent_returns, axis=0)

            variances = np.nan_to_num(variances, nan=1e-6)
            variances = np.maximum(variances, 1e-10)

            # Inverse variance weights (simplified min variance)
            inv_vars = 1.0 / variances
            weights = inv_vars / inv_vars.sum()

            return weights.astype(np.float32)

        else:
            # Default to equal weight
            return np.ones(n, dtype=np.float32) / n

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
    Usa SubprocVecEnv (true parallelism) cuando use_subproc=True.
    """

    def __init__(
        self,
        n_envs: int,
        use_subproc: bool = False,
        **env_kwargs,
    ):
        """
        Args:
            n_envs: Número de entornos paralelos.
            use_subproc: If True, use SubprocVecEnv (true parallelism via subprocesses).
                         Requires the environment and its data to be picklable.
                         Recommended when n_envs > 4 and on Linux (fork).
            **env_kwargs: Argumentos para TradingEnvironment.
        """
        import functools

        def _make_env(kwargs: dict):
            return TradingEnvironment(**kwargs)

        env_fns = [functools.partial(_make_env, env_kwargs) for _ in range(n_envs)]

        if use_subproc and n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            self.envs = SubprocVecEnv(env_fns)
            logger.info(f"VecTradingEnv: {n_envs} envs using SubprocVecEnv (true parallelism)")
        else:
            from stable_baselines3.common.vec_env import DummyVecEnv
            self.envs = DummyVecEnv(env_fns)
            logger.info(f"VecTradingEnv: {n_envs} envs using DummyVecEnv (serial)")

    def __getattr__(self, name: str) -> Any:
        """Delega atributos al VecEnv interno."""
        return getattr(self.envs, name)
