"""
Generación de informes.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .comparison import StrategyComparison
from .metrics import compute_full_metrics

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Genera informes completos de evaluación.

    Incluye:
    - Tabla comparativa de métricas
    - Gráficos de equity curves y drawdowns
    - Análisis por régimen
    - Distribución de retornos
    """

    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Args:
            output_dir: Directorio para guardar informes.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        strategies: dict[str, pd.Series],
        benchmark_returns: pd.Series,
        weights_history: Optional[dict[str, list[np.ndarray]]] = None,
        regime_labels: Optional[pd.Series] = None,
        report_name: str = "evaluation_report",
    ) -> str:
        """
        Genera informe completo.

        Args:
            strategies: Dict {nombre: retornos}.
            benchmark_returns: Retornos del benchmark.
            weights_history: Dict {nombre: historial_pesos} (opcional).
            regime_labels: Etiquetas de régimen (opcional).
            report_name: Nombre del informe.

        Returns:
            Ruta al informe generado.
        """
        # Crear comparación
        comparison = StrategyComparison(benchmark_returns)
        for name, returns in strategies.items():
            comparison.add_strategy(name, returns)

        # Generar tablas
        comparison_table = comparison.get_comparison_table()
        detailed_table = comparison.get_detailed_comparison()

        # Generar gráficos
        self._plot_equity_curves(strategies, benchmark_returns, report_name)
        self._plot_drawdowns(strategies, benchmark_returns, report_name)
        self._plot_rolling_sharpe(strategies, benchmark_returns, report_name)
        self._plot_return_distribution(strategies, benchmark_returns, report_name)

        if weights_history:
            for name, weights in weights_history.items():
                self._plot_weights_evolution(weights, f"{report_name}_{name}_weights")

        if regime_labels is not None:
            self._plot_regime_analysis(strategies, benchmark_returns, regime_labels, report_name)

        # Generar texto del informe
        report_text = self._generate_text_report(
            comparison, detailed_table, regime_labels
        )

        # Guardar informe
        report_path = self.output_dir / f"{report_name}.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        # Guardar tabla como CSV
        detailed_table.to_csv(self.output_dir / f"{report_name}_metrics.csv")

        logger.info(f"Report generated: {report_path}")
        return str(report_path)

    def _plot_equity_curves(
        self,
        strategies: dict[str, pd.Series],
        benchmark_returns: pd.Series,
        name: str,
    ) -> None:
        """Grafica curvas de equity."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Benchmark
        benchmark_equity = (1 + benchmark_returns).cumprod()
        ax.plot(benchmark_equity.index, benchmark_equity.values, label="Benchmark", linewidth=2, color="black")

        # Estrategias
        colors = plt.cm.tab10.colors
        for i, (strategy_name, returns) in enumerate(strategies.items()):
            equity = (1 + returns).cumprod()
            ax.plot(equity.index, equity.values, label=strategy_name, color=colors[i % len(colors)])

        ax.set_title("Equity Curves")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / f"{name}_equity_curves.png", dpi=150)
        plt.close()

    def _plot_drawdowns(
        self,
        strategies: dict[str, pd.Series],
        benchmark_returns: pd.Series,
        name: str,
    ) -> None:
        """Grafica drawdowns."""
        fig, ax = plt.subplots(figsize=(12, 6))

        def calc_drawdown(returns):
            equity = (1 + returns).cumprod()
            rolling_max = equity.cummax()
            return (equity - rolling_max) / rolling_max

        # Benchmark
        benchmark_dd = calc_drawdown(benchmark_returns)
        ax.fill_between(benchmark_dd.index, benchmark_dd.values, 0, alpha=0.3, label="Benchmark", color="black")

        # Estrategias
        colors = plt.cm.tab10.colors
        for i, (strategy_name, returns) in enumerate(strategies.items()):
            dd = calc_drawdown(returns)
            ax.plot(dd.index, dd.values, label=strategy_name, color=colors[i % len(colors)])

        ax.set_title("Drawdowns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / f"{name}_drawdowns.png", dpi=150)
        plt.close()

    def _plot_rolling_sharpe(
        self,
        strategies: dict[str, pd.Series],
        benchmark_returns: pd.Series,
        name: str,
        window: int = 63,
    ) -> None:
        """Grafica Sharpe rolling."""
        fig, ax = plt.subplots(figsize=(12, 6))

        def rolling_sharpe(returns, window):
            roll_mean = returns.rolling(window).mean() * 252
            roll_std = returns.rolling(window).std() * np.sqrt(252)
            return roll_mean / roll_std

        # Benchmark
        benchmark_sharpe = rolling_sharpe(benchmark_returns, window)
        ax.plot(benchmark_sharpe.index, benchmark_sharpe.values, label="Benchmark", linewidth=2, color="black")

        # Estrategias
        colors = plt.cm.tab10.colors
        for i, (strategy_name, returns) in enumerate(strategies.items()):
            sharpe = rolling_sharpe(returns, window)
            ax.plot(sharpe.index, sharpe.values, label=strategy_name, color=colors[i % len(colors)])

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Rolling Sharpe Ratio ({window}d)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.output_dir / f"{name}_rolling_sharpe.png", dpi=150)
        plt.close()

    def _plot_return_distribution(
        self,
        strategies: dict[str, pd.Series],
        benchmark_returns: pd.Series,
        name: str,
    ) -> None:
        """Grafica distribución de retornos mensuales."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convertir a retornos mensuales
        benchmark_monthly = benchmark_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        ax.hist(benchmark_monthly.values, bins=30, alpha=0.5, label="Benchmark", color="black")

        colors = plt.cm.tab10.colors
        for i, (strategy_name, returns) in enumerate(strategies.items()):
            monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            ax.hist(monthly.values, bins=30, alpha=0.5, label=strategy_name, color=colors[i % len(colors)])

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Monthly Return Distribution")
        ax.set_xlabel("Monthly Return")
        ax.set_ylabel("Frequency")
        ax.legend()

        plt.tight_layout()
        fig.savefig(self.output_dir / f"{name}_return_distribution.png", dpi=150)
        plt.close()

    def _plot_weights_evolution(
        self,
        weights_history: list[np.ndarray],
        name: str,
    ) -> None:
        """Grafica evolución de pesos."""
        if not weights_history:
            return

        weights_array = np.array(weights_history)
        n_assets = weights_array.shape[1]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.stackplot(
            range(len(weights_history)),
            weights_array.T,
            labels=[f"Algo {i+1}" for i in range(n_assets)],
            alpha=0.8,
        )

        ax.set_title("Portfolio Weights Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

        plt.tight_layout()
        fig.savefig(self.output_dir / f"{name}.png", dpi=150)
        plt.close()

    def _plot_regime_analysis(
        self,
        strategies: dict[str, pd.Series],
        benchmark_returns: pd.Series,
        regime_labels: pd.Series,
        name: str,
    ) -> None:
        """Grafica análisis por régimen."""
        # TODO: Implementar análisis por régimen
        pass

    def _generate_text_report(
        self,
        comparison: StrategyComparison,
        detailed_table: pd.DataFrame,
        regime_labels: Optional[pd.Series],
    ) -> str:
        """Genera texto del informe."""
        report = comparison.generate_report()

        report += "\n\nDETAILED METRICS\n"
        report += "═" * 80 + "\n"
        report += detailed_table.to_string()

        return report
