#!/usr/bin/env python
"""
Benchmark script to verify performance improvements from optimizations.

This script compares the performance of:
1. Numba-optimized functions vs pure numpy
2. Vectorized operations vs loops
3. Memory-efficient dtypes

Usage:
    python scripts/benchmark_optimizations.py
"""

import time
import numpy as np
import pandas as pd
from typing import Callable, Any

# Import from our modules
from src.utils.numba_utils import (
    rolling_mean,
    rolling_std,
    rolling_sharpe,
    max_drawdown,
    max_drawdown_duration,
    correlation_matrix,
    is_numba_available,
    warm_up_jit,
)
from src.utils.dtypes import optimize_dtypes, get_memory_usage
from src.evaluation.metrics import (
    sharpe_ratio as metrics_sharpe,
    max_drawdown as metrics_max_dd,
    turnover,
)
from src.environment.cost_model import CostModel


def benchmark(func: Callable, *args, n_runs: int = 10, **kwargs) -> tuple[float, Any]:
    """Run a function multiple times and return average time and result."""
    # Warm-up run
    result = func(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time, result


def benchmark_rolling_calculations():
    """Benchmark rolling calculation functions."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Rolling Calculations")
    print("=" * 60)

    sizes = [1000, 10000, 100000]
    window = 21

    for size in sizes:
        arr = np.random.randn(size).astype(np.float64)
        prices = np.abs(arr.cumsum()) + 100

        # Rolling mean
        time_numba, _ = benchmark(rolling_mean, arr, window)
        time_pandas, _ = benchmark(lambda x, w: pd.Series(x).rolling(w).mean().values, arr, window)
        speedup = time_pandas / time_numba if time_numba > 0 else float('inf')

        print(f"\nRolling mean (n={size:,}):")
        print(f"  Numba:  {time_numba * 1000:.3f} ms")
        print(f"  Pandas: {time_pandas * 1000:.3f} ms")
        print(f"  Speedup: {speedup:.1f}x")

        # Rolling std
        time_numba, _ = benchmark(rolling_std, arr, window)
        time_pandas, _ = benchmark(lambda x, w: pd.Series(x).rolling(w).std().values, arr, window)
        speedup = time_pandas / time_numba if time_numba > 0 else float('inf')

        print(f"\nRolling std (n={size:,}):")
        print(f"  Numba:  {time_numba * 1000:.3f} ms")
        print(f"  Pandas: {time_pandas * 1000:.3f} ms")
        print(f"  Speedup: {speedup:.1f}x")


def benchmark_financial_metrics():
    """Benchmark financial metric calculations."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Financial Metrics")
    print("=" * 60)

    sizes = [252, 1260, 5040]  # 1 year, 5 years, 20 years

    for size in sizes:
        returns = np.random.randn(size) * 0.01  # ~1% daily vol
        returns_series = pd.Series(returns)
        prices = (1 + returns).cumprod() * 100

        # Sharpe ratio
        time_optimized, _ = benchmark(metrics_sharpe, returns_series)

        print(f"\nSharpe ratio (n={size:,}):")
        print(f"  Optimized: {time_optimized * 1000:.3f} ms")

        # Max drawdown
        time_optimized, result = benchmark(max_drawdown, prices)
        print(f"\nMax drawdown (n={size:,}):")
        print(f"  Optimized: {time_optimized * 1000:.3f} ms")
        print(f"  Result: {result:.4f}")

        # Max drawdown duration
        time_optimized, result = benchmark(max_drawdown_duration, prices)
        print(f"\nMax drawdown duration (n={size:,}):")
        print(f"  Optimized: {time_optimized * 1000:.3f} ms")
        print(f"  Result: {result} days")


def benchmark_correlation_matrix():
    """Benchmark correlation matrix computation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Correlation Matrix")
    print("=" * 60)

    n_samples = 252
    n_assets_list = [10, 50, 100, 200]

    for n_assets in n_assets_list:
        returns = np.random.randn(n_samples, n_assets).astype(np.float64)
        returns_df = pd.DataFrame(returns)

        # Numba version
        time_numba, _ = benchmark(correlation_matrix, returns)

        # Pandas version
        time_pandas, _ = benchmark(lambda df: df.corr().values, returns_df)

        speedup = time_pandas / time_numba if time_numba > 0 else float('inf')

        print(f"\nCorrelation matrix ({n_samples}x{n_assets}):")
        print(f"  Numba:  {time_numba * 1000:.3f} ms")
        print(f"  Pandas: {time_pandas * 1000:.3f} ms")
        print(f"  Speedup: {speedup:.1f}x")


def benchmark_cost_model():
    """Benchmark cost model computation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Cost Model")
    print("=" * 60)

    n_assets_list = [10, 50, 100, 500]
    cost_model = CostModel()

    for n_assets in n_assets_list:
        old_weights = np.random.dirichlet(np.ones(n_assets))
        new_weights = np.random.dirichlet(np.ones(n_assets))
        portfolio_value = 1_000_000

        time_cost, result = benchmark(
            cost_model.compute_cost, old_weights, new_weights, portfolio_value,
            n_runs=100
        )

        print(f"\nCost computation (n_assets={n_assets}):")
        print(f"  Time: {time_cost * 1000:.4f} ms")
        print(f"  Cost: ${result:,.2f}")


def benchmark_turnover():
    """Benchmark turnover calculation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Turnover Calculation")
    print("=" * 60)

    n_assets = 50
    n_periods_list = [52, 252, 1260]  # 1 year weekly, 1 year daily, 5 years daily

    for n_periods in n_periods_list:
        weights_history = [np.random.dirichlet(np.ones(n_assets)) for _ in range(n_periods)]

        time_optimized, result = benchmark(turnover, weights_history, n_runs=50)

        print(f"\nTurnover ({n_periods} periods, {n_assets} assets):")
        print(f"  Time: {time_optimized * 1000:.3f} ms")
        print(f"  Result: {result:.2f}")


def benchmark_memory_optimization():
    """Benchmark memory optimization."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Memory Optimization")
    print("=" * 60)

    # Create a large DataFrame
    n_rows = 100000
    n_cols = 100

    df = pd.DataFrame({
        f'float_{i}': np.random.randn(n_rows).astype(np.float64)
        for i in range(n_cols // 2)
    })
    for i in range(n_cols // 2):
        df[f'int_{i}'] = np.random.randint(0, 1000, n_rows).astype(np.int64)

    original_memory = get_memory_usage(df, detailed=False)

    # Optimize
    start = time.perf_counter()
    df_optimized = optimize_dtypes(df, verbose=False)
    optimization_time = time.perf_counter() - start

    optimized_memory = get_memory_usage(df_optimized, detailed=False)

    reduction = (1 - optimized_memory / original_memory) * 100

    print(f"\nDataFrame size: {n_rows:,} rows x {n_cols} columns")
    print(f"Original memory: {original_memory:.2f} MB")
    print(f"Optimized memory: {optimized_memory:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Optimization time: {optimization_time * 1000:.1f} ms")


def main():
    print("=" * 60)
    print("RL Meta-Allocator Optimization Benchmark")
    print("=" * 60)

    print(f"\nNumba available: {is_numba_available()}")

    if is_numba_available():
        print("Warming up JIT compilation...")
        warm_up_jit()
        print("JIT warm-up complete.")

    benchmark_rolling_calculations()
    benchmark_financial_metrics()
    benchmark_correlation_matrix()
    benchmark_cost_model()
    benchmark_turnover()
    benchmark_memory_optimization()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
