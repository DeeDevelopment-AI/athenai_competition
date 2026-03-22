"""
Validación walk-forward para evaluación temporal.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .metrics import compute_full_metrics

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Resultado de un fold de validación."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    model_path: Optional[str] = None


@dataclass
class WalkForwardResult:
    """Resultado completo de validación walk-forward."""

    folds: list[FoldResult]
    aggregated_metrics: dict = field(default_factory=dict)


class WalkForwardValidator:
    """
    Validación temporal walk-forward.

    Esquema:
    |--- train ---|-- val --|-- test --|
                  |--- train ---|-- val --|-- test --|
                                |--- train ---|-- val --|-- test --|

    IMPORTANTE: Nunca usar train/test aleatorio en datos financieros.
    Siempre separación temporal para evitar data leakage.
    """

    def __init__(
        self,
        train_window: int = 252,
        val_window: int = 63,
        test_window: int = 63,
        step_size: int = 63,
        expanding: bool = False,
        min_train_size: int = 126,
    ):
        """
        Args:
            train_window: Tamaño de ventana de entrenamiento (días).
            val_window: Tamaño de ventana de validación (días).
            test_window: Tamaño de ventana de test (días).
            step_size: Pasos entre folds (días).
            expanding: Si usar ventana expansiva (True) o fija (False).
            min_train_size: Tamaño mínimo de entrenamiento.
        """
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.step_size = step_size
        self.expanding = expanding
        self.min_train_size = min_train_size

    def generate_folds(self, dates: pd.DatetimeIndex) -> list[dict]:
        """
        Genera índices de folds para walk-forward.

        Args:
            dates: Índice de fechas disponibles.

        Returns:
            Lista de dicts con índices de cada fold.
        """
        folds = []
        total_window = self.train_window + self.val_window + self.test_window

        if len(dates) < total_window:
            raise ValueError(
                f"Not enough data. Need {total_window} days, have {len(dates)}"
            )

        fold_id = 0
        start_idx = 0

        while True:
            if self.expanding:
                train_start = 0
            else:
                train_start = start_idx

            train_end = start_idx + self.train_window
            val_start = train_end
            val_end = val_start + self.val_window
            test_start = val_end
            test_end = test_start + self.test_window

            if test_end > len(dates):
                break

            folds.append({
                "fold_id": fold_id,
                "train_start": dates[train_start],
                "train_end": dates[train_end - 1],
                "val_start": dates[val_start],
                "val_end": dates[val_end - 1],
                "test_start": dates[test_start],
                "test_end": dates[test_end - 1],
            })

            fold_id += 1
            start_idx += self.step_size

        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds

    def run(
        self,
        agent_factory: Callable,
        env_factory: Callable,
        data: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        save_dir: Optional[str] = None,
    ) -> WalkForwardResult:
        """
        Ejecuta validación walk-forward completa.

        Args:
            agent_factory: Función que crea un agente nuevo.
            env_factory: Función que crea un entorno dado (start, end).
            data: DataFrame con datos de retornos.
            benchmark_returns: Serie de retornos del benchmark.
            save_dir: Directorio para guardar modelos.

        Returns:
            WalkForwardResult con métricas de todos los folds.
        """
        dates = data.index
        folds_config = self.generate_folds(dates)

        fold_results = []

        for fold in folds_config:
            logger.info(
                f"Processing fold {fold['fold_id']}: "
                f"train {fold['train_start']} - {fold['train_end']}, "
                f"test {fold['test_start']} - {fold['test_end']}"
            )

            result = self._run_fold(
                fold=fold,
                agent_factory=agent_factory,
                env_factory=env_factory,
                data=data,
                benchmark_returns=benchmark_returns,
                save_dir=save_dir,
            )
            fold_results.append(result)

        # Agregar métricas
        aggregated = self._aggregate_metrics(fold_results)

        return WalkForwardResult(
            folds=fold_results,
            aggregated_metrics=aggregated,
        )

    def _run_fold(
        self,
        fold: dict,
        agent_factory: Callable,
        env_factory: Callable,
        data: pd.DataFrame,
        benchmark_returns: Optional[pd.Series],
        save_dir: Optional[str],
    ) -> FoldResult:
        """Ejecuta un fold individual."""
        # Crear entornos
        train_env = env_factory(fold["train_start"], fold["train_end"])
        val_env = env_factory(fold["val_start"], fold["val_end"])
        test_env = env_factory(fold["test_start"], fold["test_end"])

        # Crear y entrenar agente
        agent = agent_factory(train_env)
        agent.train(eval_env=val_env)

        # Evaluar en cada split
        train_returns = self._evaluate_agent(agent, train_env)
        val_returns = self._evaluate_agent(agent, val_env)
        test_returns = self._evaluate_agent(agent, test_env)

        # Calcular métricas
        benchmark_train = benchmark_returns.loc[
            fold["train_start"]:fold["train_end"]
        ] if benchmark_returns is not None else None

        benchmark_test = benchmark_returns.loc[
            fold["test_start"]:fold["test_end"]
        ] if benchmark_returns is not None else None

        result = FoldResult(
            fold_id=fold["fold_id"],
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            test_start=fold["test_start"],
            test_end=fold["test_end"],
            train_metrics=compute_full_metrics(train_returns, benchmark_train),
            val_metrics=compute_full_metrics(val_returns),
            test_metrics=compute_full_metrics(test_returns, benchmark_test),
        )

        # Guardar modelo
        if save_dir:
            model_path = f"{save_dir}/model_fold_{fold['fold_id']}"
            agent.save(model_path)
            result.model_path = model_path

        return result

    def _evaluate_agent(self, agent, env) -> pd.Series:
        """Evalúa un agente en un entorno y devuelve retornos."""
        obs, _ = env.reset()
        done = False
        returns = []
        dates = []

        while not done:
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

            if "portfolio_return" in info:
                returns.append(info["portfolio_return"])
            if "date" in info:
                dates.append(info["date"])

            done = done or truncated

        return pd.Series(returns, index=dates[:len(returns)])

    def _aggregate_metrics(self, fold_results: list[FoldResult]) -> dict:
        """Agrega métricas de todos los folds."""
        test_metrics = [f.test_metrics for f in fold_results]

        aggregated = {}
        if not test_metrics:
            return aggregated

        # Calcular media y std de cada métrica
        for key in test_metrics[0].keys():
            values = [m.get(key, np.nan) for m in test_metrics]
            values = [v for v in values if not np.isnan(v)]

            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)

        return aggregated
