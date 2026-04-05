"""
Production modules for RL Meta-Allocator inference.

Modules:
- data_loader: Load data from CSV files
- duckdb_loader: Load data from DuckDB database
- feature_builder: Build features and Sortino family classifications
- inference: RL agent inference (requires full ML stack)
"""

import sys
from pathlib import Path

# Add project root to path for modules that need it
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Core data loading (always available)
from .data_loader import (
    load_single_asset,
    load_all_assets,
    compute_returns,
    prepare_production_data,
)

from .duckdb_loader import (
    DuckDBLoader,
    load_from_duckdb,
)

from .feature_builder import (
    build_features,
    build_observation,
    compute_cumulative_sortino,
    compute_rolling_sortino,
    classify_sortino_families,
    get_family_summary,
    print_family_summary,
    SORTINO_THRESHOLDS,
)

# Inference module loaded lazily (requires torch, stable-baselines3, etc.)
# Import explicitly when needed:
#   from production.src.inference import ProductionInferenceEngine

__all__ = [
    # Data loading (CSV)
    "load_single_asset",
    "load_all_assets",
    "compute_returns",
    "prepare_production_data",
    # Data loading (DuckDB)
    "DuckDBLoader",
    "load_from_duckdb",
    # Feature building
    "build_features",
    "build_observation",
    # Sortino & family classification
    "compute_cumulative_sortino",
    "compute_rolling_sortino",
    "classify_sortino_families",
    "get_family_summary",
    "print_family_summary",
    "SORTINO_THRESHOLDS",
]


def __getattr__(name):
    """Lazy import for inference components."""
    if name in ("ProductionInferenceEngine", "HybridInferenceEngine",
                "save_recommendation", "load_latest_recommendation"):
        from . import inference
        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
