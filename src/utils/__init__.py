"""
Modulo de utilidades: configuracion, logging, visualizacion, y optimizacion.

This module provides:
- Configuration management (config.py)
- Logging utilities (logging_utils.py)
- Plotting functions (plotting.py)
- GPU/CUDA device management (device.py)
- Numba-optimized calculations (numba_utils.py)
- Memory-efficient dtype utilities (dtypes.py)
- Performance monitoring (performance_monitor.py)
- Centralized path management (paths.py)
"""

from .config import load_config, get_config
from .logging_utils import setup_logging, get_logger
from .plotting import plot_equity_curves, plot_drawdowns, plot_weights_evolution

# Path management
from .paths import (
    PROJECT_ROOT,
    data_paths,
    output_paths,
    ensure_dir,
    ensure_parent_dir,
    DataPaths,
    OutputPaths,
)

# Performance monitoring
from .performance_monitor import (
    PerformanceMonitor,
    PhaseMonitor,
    PerformanceReport,
    monitor_function,
    is_gpu_available,
    get_gpu_backend,
    get_memory_info,
    get_system_memory_info,
    get_gpu_memory_info,
    force_garbage_collection,
)

# Device management
from .device import (
    get_device,
    get_torch_device,
    DeviceManager,
    DeviceInfo,
    to_device,
    ensure_cpu,
    clear_cuda_cache,
    get_memory_stats,
    get_sb3_policy_kwargs,
)

# Numba utilities
from .numba_utils import (
    is_numba_available,
    warm_up_jit,
    rolling_mean,
    rolling_std,
    rolling_sum,
    cumulative_return,
    returns_from_prices,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    profit_factor,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    turnover,
    herfindahl_index,
    correlation,
    rolling_correlation,
    covariance_matrix,
    correlation_matrix,
    var_historical,
    cvar_historical,
)

# Dtype utilities
from .dtypes import (
    optimize_dtypes,
    downcast_floats,
    upcast_floats,
    get_memory_usage,
    memory_report,
    optimize_parquet_compression,
    convert_array_dtype,
    estimate_memory_for_features,
    MemoryTracker,
    efficient_concat,
)

__all__ = [
    # Config
    "load_config",
    "get_config",
    # Logging
    "setup_logging",
    "get_logger",
    # Plotting
    "plot_equity_curves",
    "plot_drawdowns",
    "plot_weights_evolution",
    # Paths
    "PROJECT_ROOT",
    "data_paths",
    "output_paths",
    "ensure_dir",
    "ensure_parent_dir",
    "DataPaths",
    "OutputPaths",
    # Performance monitoring
    "PerformanceMonitor",
    "PhaseMonitor",
    "PerformanceReport",
    "monitor_function",
    "is_gpu_available",
    "get_gpu_backend",
    "get_memory_info",
    "get_system_memory_info",
    "get_gpu_memory_info",
    "force_garbage_collection",
    # Device
    "get_device",
    "get_torch_device",
    "DeviceManager",
    "DeviceInfo",
    "to_device",
    "ensure_cpu",
    "clear_cuda_cache",
    "get_memory_stats",
    "get_sb3_policy_kwargs",
    # Numba
    "is_numba_available",
    "warm_up_jit",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "cumulative_return",
    "returns_from_prices",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_duration",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "profit_factor",
    "portfolio_return",
    "portfolio_variance",
    "portfolio_volatility",
    "turnover",
    "herfindahl_index",
    "correlation",
    "rolling_correlation",
    "covariance_matrix",
    "correlation_matrix",
    "var_historical",
    "cvar_historical",
    # Dtypes
    "optimize_dtypes",
    "downcast_floats",
    "upcast_floats",
    "get_memory_usage",
    "memory_report",
    "optimize_parquet_compression",
    "convert_array_dtype",
    "estimate_memory_for_features",
    "MemoryTracker",
    "efficient_concat",
]
