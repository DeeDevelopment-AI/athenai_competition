"""
Utilidades de visualización.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_equity_curves(
    returns_dict: dict[str, pd.Series],
    title: str = "Equity Curves",
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grafica curvas de equity para múltiples estrategias.

    Args:
        returns_dict: {nombre: serie_retornos}.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        save_path: Ruta para guardar (opcional).

    Returns:
        Figura de matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, returns in returns_dict.items():
        equity = (1 + returns).cumprod()
        ax.plot(equity.index, equity.values, label=name)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drawdowns(
    returns_dict: dict[str, pd.Series],
    title: str = "Drawdowns",
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grafica drawdowns para múltiples estrategias.

    Args:
        returns_dict: {nombre: serie_retornos}.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        save_path: Ruta para guardar (opcional).

    Returns:
        Figura de matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, returns in returns_dict.items():
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=name)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_weights_evolution(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Weights Over Time",
    figsize: tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grafica evolución de pesos en el tiempo (área apilada).

    Args:
        weights_df: DataFrame [fechas x algoritmos] con pesos.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        save_path: Ruta para guardar (opcional).

    Returns:
        Figura de matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.stackplot(
        weights_df.index,
        weights_df.T.values,
        labels=weights_df.columns,
        alpha=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grafica matriz de correlaciones como heatmap.

    Args:
        corr_matrix: Matriz de correlaciones.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        save_path: Ruta para guardar (opcional).

    Returns:
        Figura de matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
