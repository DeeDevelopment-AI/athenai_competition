"""
Módulo de evaluación: walk-forward, métricas y comparación.
"""

from .walk_forward import WalkForwardValidator
from .metrics import compute_full_metrics
from .comparison import StrategyComparison
from .reporting import ReportGenerator

__all__ = [
    "WalkForwardValidator",
    "compute_full_metrics",
    "StrategyComparison",
    "ReportGenerator",
]
