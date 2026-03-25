"""Tests for the ACO meta allocator."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.swarm.aco_allocator import ACOAllocatorBacktester, ACOConfig, ACOMetaAllocator


class _NearOneRng:
    def random(self, size=None, dtype=None):
        dtype = np.float32 if dtype is None else dtype
        return np.full(size, np.nextafter(dtype(1.0), dtype(0.0)), dtype=dtype)


def test_aco_sampler_clamps_bucket_indices():
    allocator = ACOMetaAllocator(config=ACOConfig(weight_buckets=21, seed=7))
    allocator._rng = _NearOneRng()

    pheromone = np.ones((3, 21), dtype=np.float32)
    heuristic = np.ones((3, 21), dtype=np.float32)

    sampled = allocator._sample_weight_buckets(pheromone=pheromone, heuristic_matrix=heuristic)

    assert sampled.shape == (3,)
    assert sampled.min() >= 0
    assert sampled.max() <= 20


def test_aco_meta_allocator_produces_feasible_weights():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    returns = pd.DataFrame(
        {
            "algo_strong": rng.normal(0.0016, 0.008, size=len(dates)),
            "algo_good": rng.normal(0.0012, 0.009, size=len(dates)),
            "algo_flat": rng.normal(0.0002, 0.011, size=len(dates)),
            "algo_weak": rng.normal(-0.0004, 0.012, size=len(dates)),
        },
        index=dates,
    )
    previous_weights = np.full(returns.shape[1], 1.0 / returns.shape[1], dtype=np.float32)
    family_labels = pd.Series(
        {
            "algo_strong": "trend",
            "algo_good": "trend",
            "algo_flat": "blend",
            "algo_weak": "defensive",
        }
    )

    allocator = ACOMetaAllocator(
        config=ACOConfig(
            n_ants=32,
            n_iterations=10,
            weight_buckets=11,
            max_weight=0.55,
            target_portfolio_vol=0.30,
            min_gross_exposure=0.50,
            seed=7,
        ),
        family_labels=family_labels,
        family_alpha_scores={"trend": 0.5, "blend": 0.1, "defensive": -0.2},
    )

    result = allocator.optimize(returns_window=returns, previous_weights=previous_weights)

    assert result.weights.shape == (4,)
    assert np.isfinite(result.score)
    assert float(result.weights.sum()) <= 1.0 + 1e-6
    assert float(result.weights.sum()) >= 0.50
    assert float(result.weights.max()) <= 0.55 + 1e-6
    assert result.weights[0] > result.weights[-1]
    assert "sharpe_ratio" in result.diagnostics
    assert result.diagnostics["best_iteration"] >= 0


def test_backtester_combines_multiple_selection_factors_with_directional_normalization():
    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    returns = pd.DataFrame(
        {
            "algo_a": [0.01, 0.00, 0.01, 0.00, 0.01],
            "algo_b": [0.00, 0.01, 0.00, 0.01, 0.00],
            "algo_c": [-0.01, -0.01, 0.00, -0.01, 0.00],
        },
        index=dates,
    )
    features = pd.DataFrame(
        {
            "algo_a_rolling_sharpe_21d": [2.0, 2.0, 2.0, 2.0, 2.0],
            "algo_b_rolling_sharpe_21d": [1.0, 1.0, 1.0, 1.0, 1.0],
            "algo_c_rolling_sharpe_21d": [0.0, 0.0, 0.0, 0.0, 0.0],
            "algo_a_rolling_volatility_21d": [0.10, 0.10, 0.10, 0.10, 0.10],
            "algo_b_rolling_volatility_21d": [0.20, 0.20, 0.20, 0.20, 0.20],
            "algo_c_rolling_volatility_21d": [0.30, 0.30, 0.30, 0.30, 0.30],
        },
        index=dates,
    )

    backtester = ACOAllocatorBacktester(
        algo_returns=returns,
        features=features,
        benchmark_returns=None,
        benchmark_weights=None,
        config=ACOConfig(min_history=3, lookback_window=3, n_ants=8, n_iterations=2, weight_buckets=5, seed=7),
        selection_factor=["rolling_sharpe_21d", "rolling_volatility_21d"],
    )

    selected_signal = backtester._compute_selected_signal(features.iloc[-1], ["algo_a", "algo_b", "algo_c"])

    assert backtester.selection_factors == ("rolling_sharpe_21d", "rolling_volatility_21d")
    assert selected_signal["algo_a"] > selected_signal["algo_b"] > selected_signal["algo_c"]
