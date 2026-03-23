"""
PPO Agent para meta-allocation.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv

from .base import BaseAgent, TrainingMetrics
from .callbacks import (
    CSVLoggerCallback,
    FinancialMetricsCallback,
    NaNDetectionCallback,
    ProgressCallback,
    create_eval_callback,
)

logger = logging.getLogger(__name__)


class PPOAllocator(BaseAgent):
    """
    Wrapper sobre SB3 PPO adaptado al entorno financiero.

    Hiperparámetros clave (defaults razonables para finanzas):
    - learning_rate: 3e-4
    - n_steps: 2048
    - batch_size: 64
    - n_epochs: 10
    - gamma: 0.99
    - gae_lambda: 0.95
    - clip_range: 0.2
    - ent_coef: 0.01 (regularización entrópica)
    """

    def __init__(
        self,
        env: VecEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        net_arch: Optional[list[int]] = None,
        seed: int = 42,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Args:
            env: Entorno de trading (vectorizado).
            learning_rate: Tasa de aprendizaje.
            n_steps: Pasos por update.
            batch_size: Tamaño de batch.
            n_epochs: Epochs por update.
            gamma: Factor de descuento.
            gae_lambda: Lambda para GAE.
            clip_range: Rango de clipping para PPO.
            ent_coef: Coeficiente de entropía.
            vf_coef: Coeficiente de value function.
            max_grad_norm: Clip de gradiente.
            net_arch: Arquitectura de red [hidden_layers].
            seed: Semilla aleatoria.
            device: "cpu", "cuda" o "auto".
            verbose: Nivel de verbosidad.
            tensorboard_log: Directorio para logs de TensorBoard.
        """
        super().__init__(env, seed)

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log

        # Arquitectura de red
        if net_arch is None:
            net_arch = [256, 256]
        self.net_arch = net_arch

        policy_kwargs = {
            "net_arch": dict(pi=net_arch, vf=net_arch),
            # Tanh bounds pre-activations — critical with obs_dim=54055 where input
            # L2 norm can reach ~2300, causing ReLU networks to produce NaN on first
            # gradient step (gradient explosion through unbounded activations).
            "activation_fn": th.nn.Tanh,
            # ortho_init scales by sqrt(2)*gain; with 54055-dim input this produces
            # pre-activation magnitudes >>100 before any training — disable it.
            "ortho_init": False,
            # Small initial log_std → lower action variance → stable first rollout
            "log_std_init": -1.0,
        }

        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
        )

        logger.info(f"PPO agent initialized with {net_arch} architecture")

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

        # NaN detection — stops training and saves emergency checkpoint instead of
        # crashing at the next train() call after a full wasted rollout.
        nan_callback = NaNDetectionCallback(
            check_freq=eval_freq,
            save_path=save_path,
            verbose=self.verbose,
        )
        callback_list.append(nan_callback)

        # Add user callbacks
        if callbacks:
            callback_list.extend(callbacks)

        logger.info(f"Starting PPO training for {total_timesteps} timesteps")

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
        """
        Predice acción para una observación.

        Args:
            observation: Vector de estado.
            deterministic: Si usar política determinista.

        Returns:
            Vector de pesos (acción).
        """
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
        self.model = PPO.load(path, env=self.env)
        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(cls, path: str | Path, env: VecEnv) -> "PPOAllocator":
        """
        Carga un agente pre-entrenado.

        Args:
            path: Ruta al modelo guardado.
            env: Entorno de trading.

        Returns:
            Instancia de PPOAllocator con modelo cargado.
        """
        allocator = cls.__new__(cls)
        allocator.env = env
        allocator.seed = 42
        allocator._training_metrics = TrainingMetrics()
        allocator._is_trained = True
        allocator.model = PPO.load(path, env=env)
        logger.info(f"Loaded pre-trained PPO from {path}")
        return allocator

    def get_hyperparameters(self) -> dict[str, Any]:
        """Get agent hyperparameters."""
        return {
            "algorithm": "PPO",
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "net_arch": self.net_arch,
        }


class TrainingCallback(BaseCallback):
    """
    Callback personalizado para logging durante entrenamiento.

    DEPRECATED: Use FinancialMetricsCallback instead.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Logging de episodios completados
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if "r" in ep_info:
                self.episode_rewards.append(ep_info["r"])
            if "l" in ep_info:
                self.episode_lengths.append(ep_info["l"])

        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            logger.info(
                f"Training finished. "
                f"Mean reward: {np.mean(self.episode_rewards):.4f}, "
                f"Mean episode length: {np.mean(self.episode_lengths):.1f}"
            )
