"""
SAC Agent para meta-allocation.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv

from .base import BaseAgent, TrainingMetrics
from .callbacks import (
    CSVLoggerCallback,
    FinancialMetricsCallback,
    ProgressCallback,
    create_eval_callback,
)

logger = logging.getLogger(__name__)


class SACAllocator(BaseAgent):
    """
    Wrapper sobre SB3 SAC adaptado al entorno financiero.

    SAC: off-policy, maximum entropy, acciones continuas.
    Más sample-efficient que PPO pero puede ser menos estable.

    Hiperparámetros clave:
    - learning_rate: 3e-4
    - buffer_size: 100000 (replay buffer)
    - batch_size: 256
    - tau: 0.005 (soft update)
    - gamma: 0.99
    - ent_coef: 'auto' (ajuste automático de temperatura)
    """

    def __init__(
        self,
        env: VecEnv,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        train_freq: int = 1,
        gradient_steps: int = 1,
        net_arch: Optional[list[int]] = None,
        seed: int = 42,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Args:
            env: Entorno de trading.
            learning_rate: Tasa de aprendizaje.
            buffer_size: Tamaño del replay buffer.
            learning_starts: Pasos antes de empezar a entrenar.
            batch_size: Tamaño de batch.
            tau: Coeficiente de soft update.
            gamma: Factor de descuento.
            ent_coef: Coeficiente de entropía ("auto" para ajuste automático).
            target_entropy: Entropía objetivo ("auto" para calcular).
            train_freq: Frecuencia de entrenamiento.
            gradient_steps: Pasos de gradiente por update.
            net_arch: Arquitectura de red [hidden_layers].
            seed: Semilla aleatoria.
            device: "cpu", "cuda" o "auto".
            verbose: Nivel de verbosidad.
            tensorboard_log: Directorio para logs de TensorBoard.
        """
        super().__init__(env, seed)

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log

        if net_arch is None:
            net_arch = [256, 256]
        self.net_arch = net_arch

        policy_kwargs = {
            "net_arch": dict(pi=net_arch, qf=net_arch),
            "activation_fn": th.nn.Tanh,  # Bounded — avoids NaN with large obs_dim
        }

        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
        )

        logger.info(f"SAC agent initialized with {net_arch} architecture")

    def train(
        self,
        total_timesteps: int = 500_000,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_path: Optional[str] = None,
        callbacks: Optional[list[BaseCallback]] = None,
        log_to_csv: bool = True,
        progress_bar: bool = True,
    ) -> None:
        """
        Entrena el agente.

        Args:
            total_timesteps: Número total de pasos de entrenamiento.
            eval_env: Entorno de evaluación (opcional).
            eval_freq: Frecuencia de evaluación (en timesteps).
            n_eval_episodes: Número de episodios por evaluación.
            save_path: Ruta para guardar el mejor modelo.
            callbacks: Callbacks adicionales.
            log_to_csv: Si guardar logs en CSV.
            progress_bar: Mostrar barra de progreso.
        """
        # Reset training metrics
        self.reset_metrics()

        # Build callback list
        callback_list = []

        # Financial metrics callback
        financial_callback = FinancialMetricsCallback(
            training_metrics=self._training_metrics,
            eval_freq=eval_freq,
            verbose=self.verbose,
        )
        callback_list.append(financial_callback)

        # Evaluation callback
        if eval_env is not None and save_path is not None:
            eval_callback = create_eval_callback(
                eval_env=eval_env,
                save_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
            callback_list.append(eval_callback)

        # CSV logger callback
        if log_to_csv and save_path is not None:
            csv_path = Path(save_path) / "training_log.csv"
            csv_callback = CSVLoggerCallback(
                log_path=csv_path,
                log_freq=eval_freq,
                verbose=self.verbose,
            )
            callback_list.append(csv_callback)

        # Progress callback
        if self.verbose > 0:
            progress_callback = ProgressCallback(
                total_timesteps=total_timesteps,
                log_freq=eval_freq,
                verbose=self.verbose,
            )
            callback_list.append(progress_callback)

        # Add user callbacks
        if callbacks:
            callback_list.extend(callbacks)

        logger.info(f"Starting SAC training for {total_timesteps} timesteps")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callback_list) if callback_list else None,
            progress_bar=progress_bar,
        )

        self._is_trained = True
        logger.info("Training completed")

        # Log summary
        summary = self._training_metrics.get_summary()
        if summary:
            logger.info(f"Training summary: {summary}")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predice acción para una observación."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str | Path) -> None:
        """Guarda el modelo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Carga el modelo."""
        self.model = SAC.load(path, env=self.env)
        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(cls, path: str | Path, env: VecEnv) -> "SACAllocator":
        """Carga un agente pre-entrenado."""
        allocator = cls.__new__(cls)
        allocator.env = env
        allocator.seed = 42
        allocator._training_metrics = TrainingMetrics()
        allocator._is_trained = True
        allocator.model = SAC.load(path, env=env)
        logger.info(f"Loaded pre-trained SAC from {path}")
        return allocator

    def get_hyperparameters(self) -> dict[str, Any]:
        """Get agent hyperparameters."""
        return {
            "algorithm": "SAC",
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "target_entropy": self.target_entropy,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "net_arch": self.net_arch,
        }

    def get_replay_buffer_size(self) -> int:
        """Get current replay buffer size."""
        if hasattr(self.model, 'replay_buffer'):
            return self.model.replay_buffer.size()
        return 0
