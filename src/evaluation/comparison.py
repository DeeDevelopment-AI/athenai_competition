"""
Comparación de estrategias.
"""

import logging
from typing import Optional

import pandas as pd

from .metrics import compute_full_metrics

logger = logging.getLogger(__name__)


class StrategyComparison:
    """
    Compara múltiples estrategias contra el benchmark.

    Genera tabla comparativa con métricas de retorno Y riesgo
    para evaluación justa ("peras con peras").
    """

    def __init__(
        self,
        benchmark_returns: pd.Series,
        trading_days: int = 252,
        risk_free_rate: float = 0.0,
    ):
        """
        Args:
            benchmark_returns: Serie de retornos del benchmark.
            trading_days: Días de trading por año.
            risk_free_rate: Tasa libre de riesgo.
        """
        self.benchmark_returns = benchmark_returns
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate

        self.strategies: dict[str, pd.Series] = {}
        self.metrics: dict[str, dict] = {}

    def add_strategy(self, name: str, returns: pd.Series) -> None:
        """
        Añade una estrategia a comparar.

        Args:
            name: Nombre de la estrategia.
            returns: Serie de retornos.
        """
        self.strategies[name] = returns

        # Calcular métricas
        self.metrics[name] = compute_full_metrics(
            returns=returns,
            benchmark_returns=self.benchmark_returns,
            trading_days=self.trading_days,
            risk_free_rate=self.risk_free_rate,
        )

        logger.info(f"Added strategy: {name}")

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Genera tabla comparativa de todas las estrategias.

        Returns:
            DataFrame con métricas principales.
        """
        # Añadir benchmark primero
        benchmark_metrics = compute_full_metrics(
            returns=self.benchmark_returns,
            trading_days=self.trading_days,
            risk_free_rate=self.risk_free_rate,
        )

        rows = [self._format_row("Benchmark", benchmark_metrics)]

        for name in self.strategies:
            rows.append(self._format_row(name, self.metrics[name]))

        df = pd.DataFrame(rows)
        df = df.set_index("Strategy")

        return df

    def _format_row(self, name: str, metrics: dict) -> dict:
        """Formatea una fila de la tabla."""
        return {
            "Strategy": name,
            "Return (ann.)": f"{metrics.get('annualized_return', 0):.2%}",
            "Volatility (ann.)": f"{metrics.get('annualized_volatility', 0):.2%}",
            "Sharpe": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino": f"{metrics.get('sortino_ratio', 0):.2f}",
            "Calmar": f"{metrics.get('calmar_ratio', 0):.2f}",
            "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
            "Tracking Error": f"{metrics.get('tracking_error', 0):.2%}" if "tracking_error" in metrics else "-",
            "Info Ratio": f"{metrics.get('information_ratio', 0):.2f}" if "information_ratio" in metrics else "-",
            "VaR 95%": f"{metrics.get('var_95', 0):.2%}",
        }

    def get_detailed_comparison(self) -> pd.DataFrame:
        """
        Genera tabla detallada con todas las métricas.

        Returns:
            DataFrame con todas las métricas numéricas.
        """
        benchmark_metrics = compute_full_metrics(
            returns=self.benchmark_returns,
            trading_days=self.trading_days,
            risk_free_rate=self.risk_free_rate,
        )

        all_metrics = {"Benchmark": benchmark_metrics}
        all_metrics.update(self.metrics)

        return pd.DataFrame(all_metrics).T

    def rank_strategies(self, metric: str = "sharpe_ratio", ascending: bool = False) -> pd.DataFrame:
        """
        Rankea estrategias por una métrica.

        Args:
            metric: Métrica para ranking.
            ascending: Si menor es mejor.

        Returns:
            DataFrame ordenado.
        """
        df = self.get_detailed_comparison()

        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found")

        df = df.sort_values(metric, ascending=ascending)
        df["Rank"] = range(1, len(df) + 1)

        return df[["Rank", metric]]

    def generate_report(self) -> str:
        """
        Genera informe textual de comparación.

        Returns:
            String con informe formateado.
        """
        table = self.get_comparison_table()

        report = """
STRATEGY COMPARISON
═══════════════════════════════════════════════════════════════════════════════

{table}

SUMMARY
───────────────────────────────────────────────────────────────────────────────
""".format(table=table.to_string())

        # Encontrar mejor estrategia por Sharpe
        detailed = self.get_detailed_comparison()
        best_sharpe = detailed["sharpe_ratio"].idxmax()
        best_ir = detailed.get("information_ratio", pd.Series()).idxmax() if "information_ratio" in detailed else None

        report += f"\nBest Sharpe Ratio: {best_sharpe}"
        if best_ir:
            report += f"\nBest Information Ratio: {best_ir}"

        # Verificar si alguna estrategia bate al benchmark
        for name in self.strategies:
            if "excess_return" in self.metrics[name]:
                excess = self.metrics[name]["excess_return"]
                if excess > 0:
                    report += f"\n✓ {name} beats benchmark by {excess:.2%} annually"
                else:
                    report += f"\n✗ {name} underperforms benchmark by {abs(excess):.2%} annually"

        return report
