"""
Placeholder para Offline RL (CQL/IQL).

Para este proyecto, offline RL tiene mucho sentido porque:
1. Solo disponemos de datos historicos
2. No podemos explorar online (riesgo financiero)
3. Queremos aprender de la politica del benchmark

Opciones de implementacion:
- d3rlpy: Libreria especializada en offline RL
- RLlib: Soporte para CQL
- Implementacion manual sobre SB3 (mas laborioso)

Ventajas del offline RL para meta-allocation:
- Aprende de datos historicos sin necesidad de interaccion online
- Puede aprender de la politica del benchmark (imitation learning)
- Evita el problema de exploration en entornos financieros
- Conservador: no sobreestima el valor de acciones no vistas

Algoritmos recomendados:
- CQL (Conservative Q-Learning): Penaliza Q-values de acciones fuera del dataset
- IQL (Implicit Q-Learning): Evita maximizacion sobre acciones fuera del dataset
- BCQ (Batch-Constrained Q-Learning): Restringe acciones al soporte del dataset
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecEnv

from .base import BaseAgent, TrainingMetrics

logger = logging.getLogger(__name__)


@dataclass
class OfflineDataset:
    """
    Dataset para offline RL.

    Formato compatible con d3rlpy y otras librerias de offline RL.

    Attributes:
        observations: Array de observaciones [N, obs_dim]
        actions: Array de acciones [N, action_dim]
        rewards: Array de rewards [N]
        next_observations: Array de observaciones siguientes [N, obs_dim]
        terminals: Array de flags de terminacion [N]
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray

    def __len__(self) -> int:
        """Return number of transitions in dataset."""
        return len(self.observations)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_observations': self.next_observations,
            'terminals': self.terminals,
        }

    def save(self, path: str | Path) -> None:
        """Save dataset to npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_observations=self.next_observations,
            terminals=self.terminals,
        )
        logger.info(f"Dataset saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "OfflineDataset":
        """Load dataset from npz file."""
        data = np.load(path)
        return cls(
            observations=data['observations'],
            actions=data['actions'],
            rewards=data['rewards'],
            next_observations=data['next_observations'],
            terminals=data['terminals'],
        )

    def get_statistics(self) -> dict:
        """Get basic statistics about the dataset."""
        return {
            'n_transitions': len(self),
            'obs_dim': self.observations.shape[1] if len(self.observations.shape) > 1 else 1,
            'action_dim': self.actions.shape[1] if len(self.actions.shape) > 1 else 1,
            'reward_mean': float(np.mean(self.rewards)),
            'reward_std': float(np.std(self.rewards)),
            'reward_min': float(np.min(self.rewards)),
            'reward_max': float(np.max(self.rewards)),
            'terminal_rate': float(np.mean(self.terminals)),
        }


class OfflineDatasetBuilder:
    """
    Construye dataset offline a partir de historicos del benchmark.

    Convierte los logs del benchmark en transiciones (s, a, r, s', done)
    para entrenamiento offline.

    El dataset resultante puede usarse para:
    1. Behavioral cloning (imitar al benchmark)
    2. CQL/IQL (mejorar sobre el benchmark de forma conservadora)
    3. Analisis de la politica del benchmark
    """

    def __init__(
        self,
        algo_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.Series,
    ):
        """
        Args:
            algo_returns: DataFrame [dates x algos] con retornos.
            benchmark_weights: DataFrame [dates x algos] con pesos del benchmark.
            benchmark_returns: Serie con retornos del benchmark.
        """
        self.algo_returns = algo_returns
        self.benchmark_weights = benchmark_weights
        self.benchmark_returns = benchmark_returns

    def build_dataset(
        self,
        lookback: int = 21,
        include_volatility: bool = True,
        include_momentum: bool = True,
        reward_scale: float = 100.0,
    ) -> OfflineDataset:
        """
        Construye dataset de transiciones.

        Args:
            lookback: Ventana para features de estado.
            include_volatility: Incluir volatilidades en observacion.
            include_momentum: Incluir momentum en observacion.
            reward_scale: Factor de escala para rewards.

        Returns:
            OfflineDataset con transiciones.
        """
        # Alinear indices
        common_idx = (
            self.algo_returns.index
            .intersection(self.benchmark_weights.index)
            .intersection(self.benchmark_returns.index)
        )
        common_idx = sorted(common_idx)

        if len(common_idx) < lookback + 2:
            raise ValueError(
                f"Not enough data points. Have {len(common_idx)}, "
                f"need at least {lookback + 2}"
            )

        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []

        for i in range(lookback, len(common_idx) - 1):
            date = common_idx[i]
            next_date = common_idx[i + 1]

            # Estado: features de los ultimos 'lookback' dias
            obs = self._build_observation(
                date, lookback, include_volatility, include_momentum
            )
            next_obs = self._build_observation(
                next_date, lookback, include_volatility, include_momentum
            )

            # Accion: pesos del benchmark en esa fecha
            action = self.benchmark_weights.loc[date].values

            # Reward: retorno del benchmark (scaled)
            reward = self.benchmark_returns.loc[next_date] * reward_scale

            # Terminal: ultimo dia del dataset
            terminal = i == len(common_idx) - 2

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(terminal)

        dataset = OfflineDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
        )

        logger.info(f"Built offline dataset with {len(dataset)} transitions")
        logger.info(f"Statistics: {dataset.get_statistics()}")

        return dataset

    def _build_observation(
        self,
        date: pd.Timestamp,
        lookback: int,
        include_volatility: bool = True,
        include_momentum: bool = True,
    ) -> np.ndarray:
        """
        Construye vector de observacion para una fecha.

        Incluye:
        - Pesos actuales del benchmark
        - Retornos acumulados recientes
        - Volatilidades (opcional)
        - Momentum (opcional)
        """
        returns = self.algo_returns.loc[:date].tail(lookback)
        weights = self.benchmark_weights.loc[date].values

        # Features base
        obs_parts = [weights]

        # Retornos acumulados
        ret_sum = returns.sum().values
        obs_parts.append(ret_sum)

        # Volatilidades anualizadas
        if include_volatility:
            vol = returns.std().values * np.sqrt(252)
            obs_parts.append(vol)

        # Momentum (retorno ultimos 5 dias vs lookback)
        if include_momentum and lookback >= 5:
            short_ret = returns.tail(5).sum().values
            long_ret = ret_sum
            momentum = short_ret - long_ret / (lookback / 5)
            obs_parts.append(momentum)

        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32)


class CQLAllocatorPlaceholder(BaseAgent):
    """
    Placeholder para CQL (Conservative Q-Learning).

    Para implementacion real, usar d3rlpy:

    ```python
    import d3rlpy

    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    cql = d3rlpy.algos.CQL()
    cql.fit(dataset, n_steps=100000)
    ```

    Parametros clave de CQL:
    - alpha: Peso de la penalizacion conservadora (default: 1.0)
    - n_critics: Numero de criticos (default: 2)
    - actor_learning_rate: LR del actor (default: 3e-4)
    - critic_learning_rate: LR de los criticos (default: 3e-4)
    """

    def __init__(
        self,
        env: VecEnv,
        dataset: Optional[OfflineDataset] = None,
        seed: int = 42,
    ):
        """
        Args:
            env: Environment (for compatibility).
            dataset: Dataset offline para entrenamiento.
            seed: Random seed.
        """
        super().__init__(env, seed)
        self.dataset = dataset
        logger.warning(
            "CQL not implemented. Install d3rlpy for offline RL support."
        )

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_path: Optional[str] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        """Entrena el agente con CQL."""
        raise NotImplementedError(
            "CQL training requires d3rlpy. "
            "Install with: pip install d3rlpy"
        )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predice accion."""
        raise NotImplementedError("Model not trained")

    def save(self, path: str | Path) -> None:
        """Guarda el modelo."""
        raise NotImplementedError("Model not trained")

    def load(self, path: str | Path) -> None:
        """Carga el modelo."""
        raise NotImplementedError("CQL loading not implemented")

    @classmethod
    def from_pretrained(cls, path: str | Path, env: VecEnv) -> "CQLAllocatorPlaceholder":
        """Carga un agente pre-entrenado."""
        raise NotImplementedError("CQL loading not implemented")


def create_behavioral_cloning_dataset(
    algo_returns: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create dataset for behavioral cloning (supervised learning).

    This is a simpler alternative to full offline RL when you just
    want to imitate the benchmark policy.

    Args:
        algo_returns: DataFrame [dates x algos] con retornos.
        benchmark_weights: DataFrame [dates x algos] con pesos.

    Returns:
        Tuple of (X, y) for supervised learning.
    """
    common_idx = algo_returns.index.intersection(benchmark_weights.index)
    common_idx = sorted(common_idx)

    X = []
    y = []

    lookback = 21

    for i in range(lookback, len(common_idx)):
        date = common_idx[i]

        # Features: recent returns and current weights
        returns = algo_returns.loc[:date].tail(lookback)
        prev_weights = benchmark_weights.loc[common_idx[i-1]].values

        # Simple features
        ret_sum = returns.sum().values
        vol = returns.std().values * np.sqrt(252)
        features = np.concatenate([prev_weights, ret_sum, vol])

        # Target: next weights
        target = benchmark_weights.loc[date].values

        X.append(features)
        y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
