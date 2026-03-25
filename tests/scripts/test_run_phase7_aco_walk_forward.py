"""Tests for scripts/run_phase7_aco_walk_forward.py."""

import json
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_phase7_aco_walk_forward import Phase7ACOWalkForwardRunner


def _build_test_data_paths(root: Path) -> SimpleNamespace:
    algorithms_dir = root / "algorithms"
    benchmark_dir = root / "benchmark"
    analysis_dir = root / "analysis"
    return SimpleNamespace(
        algorithms=SimpleNamespace(
            returns=algorithms_dir / "returns.parquet",
            features=algorithms_dir / "features.parquet",
        ),
        benchmark=SimpleNamespace(
            daily_returns=benchmark_dir / "daily_returns.csv",
            weights=benchmark_dir / "weights.parquet",
        ),
        processed=SimpleNamespace(
            root=root,
            analysis=SimpleNamespace(
                root=analysis_dir,
                family_labels=analysis_dir / "clustering" / "behavioral" / "family_labels.csv",
                regime_labels=analysis_dir / "regimes" / "labels.csv",
            ),
        ),
    )


@pytest.fixture
def mock_phase_inputs():
    root = Path.cwd() / ".tmp_test_artifacts" / f"wf_phase7_aco_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(55)
    dates = pd.date_range("2020-01-01", periods=320, freq="B")
    algos = [f"algo_{i}" for i in range(8)]

    algorithms_dir = root / "algorithms"
    benchmark_dir = root / "benchmark"
    analysis_behavioral_dir = root / "analysis" / "clustering" / "behavioral"
    analysis_temporal_dir = root / "analysis" / "clustering" / "temporal"
    analysis_regimes_dir = root / "analysis" / "regimes"
    analysis_profiles_dir = root / "analysis" / "profiles"
    algorithms_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    analysis_behavioral_dir.mkdir(parents=True, exist_ok=True)
    analysis_temporal_dir.mkdir(parents=True, exist_ok=True)
    analysis_regimes_dir.mkdir(parents=True, exist_ok=True)
    analysis_profiles_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(len(dates), len(algos))),
        index=dates,
        columns=algos,
    )
    returns.iloc[:, :2] += 0.0008
    returns.to_parquet(algorithms_dir / "returns.parquet")

    features = {}
    rolling_sharpe = returns.rolling(21).mean() / returns.rolling(21).std().replace(0, np.nan)
    for algo in algos:
        features[f"{algo}_rolling_sharpe_21d"] = rolling_sharpe[algo].fillna(0.0)
        features[f"{algo}_rolling_return_21d"] = returns[algo].rolling(21).sum().fillna(0.0)
        features[f"{algo}_rolling_profit_factor_21d"] = (1.1 + rolling_sharpe[algo].fillna(0.0)).clip(lower=0.5)
        features[f"{algo}_rolling_calmar_21d"] = rolling_sharpe[algo].fillna(0.0) * 0.5
        features[f"{algo}_rolling_drawdown_21d"] = (-returns[algo].rolling(21).sum().abs()).fillna(0.02)
    pd.DataFrame(features, index=dates).to_parquet(algorithms_dir / "features.parquet")

    pd.DataFrame({"benchmark": returns.mean(axis=1)}, index=dates).to_csv(benchmark_dir / "daily_returns.csv")
    pd.DataFrame(1.0 / len(algos), index=dates, columns=algos).to_parquet(benchmark_dir / "weights.parquet")

    pd.Series([f"family_{i % 2}" for i in range(len(algos))], index=algos, name="family").to_csv(
        analysis_behavioral_dir / "family_labels.csv"
    )
    cluster_dates = pd.date_range(dates[20], periods=20, freq="W-FRI")
    cluster_rows = []
    for week_end in cluster_dates:
        for algo_idx, algo in enumerate(algos):
            cluster_id = 1 if algo_idx < 4 else 2
            cluster_rows.append(
                {
                    "week_end": week_end,
                    "algo_id": algo,
                    "cluster_cumulative": cluster_id,
                    "cluster_weekly": cluster_id,
                    "cluster_monthly": cluster_id,
                }
            )
    pd.DataFrame(cluster_rows).to_csv(analysis_temporal_dir / "cluster_history.csv", index=False)
    pd.DataFrame({"algo_id": algos, "stability_ratio": np.linspace(0.8, 0.4, len(algos))}).to_csv(
        analysis_temporal_dir / "cluster_stability.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "algo_id": algos,
            "ann_return": np.linspace(0.16, 0.04, len(algos)),
            "ann_volatility": np.linspace(0.18, 0.26, len(algos)),
            "sharpe": np.linspace(0.9, 0.2, len(algos)),
        }
    ).to_csv(analysis_profiles_dir / "summary.csv", index=False)
    pd.Series([i % 3 for i in range(len(dates))], index=dates, name="regime").to_csv(
        analysis_regimes_dir / "labels.csv"
    )

    config_dir = root / "outputs" / "tuning" / "factor_search"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "best_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_factor": "rolling_sharpe_21d",
                "lookback_window": 84,
                "top_k": 6,
                "min_history": 42,
                "n_ants": 24,
                "n_iterations": 6,
                "weight_buckets": 11,
                "max_weight": 0.40,
                "max_family_exposure": 0.30,
                "expected_return_weight": 0.80,
                "volatility_weight": 0.35,
                "turnover_weight": 0.15,
                "concentration_weight": 0.10,
                "diversification_weight": 0.12,
                "family_penalty_weight": 0.20,
                "family_alpha_reward_weight": 0.15,
                "risk_budget_weight": 0.25,
                "sparsity_penalty_weight": 0.01,
                "entropy_reward_weight": 0.05,
                "sharpe_weight": 1.75,
                "regime_focus": 1.50,
                "target_portfolio_vol": 0.16,
                "min_active_weight": 0.0025,
                "min_gross_exposure": 0.85,
                "under_investment_penalty_weight": 0.35,
                "objective_name": "factor_search_sharpe_21d",
                "selection_mode": "legacy",
                "normalize_objective_metrics": False,
                "seed": 42,
                "use_gpu": False,
            },
            handle,
            indent=2,
        )

    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_phase7_aco_walk_forward_runner_compares_with_benchmark(mock_phase_inputs):
    runner = Phase7ACOWalkForwardRunner()
    runner.dp = _build_test_data_paths(mock_phase_inputs)
    output_dir = mock_phase_inputs / "outputs"

    results = runner.execute(
        [
            "--input-dir",
            str(mock_phase_inputs),
            "--analysis-dir",
            str(mock_phase_inputs / "analysis"),
            "--output-dir",
            str(output_dir),
            "--sample",
            "8",
            "--train-window",
            "126",
            "--validation-window",
            "42",
            "--test-window",
            "42",
            "--step-size",
            "42",
            "--max-folds",
            "2",
            "--no-report",
        ]
    )

    wf_dir = output_dir / "walk_forward"
    assert results["status"] == "completed"
    assert results["n_folds"] == 2
    assert (wf_dir / "folds.csv").exists()
    assert (wf_dir / "summary.json").exists()
    assert (wf_dir / "comparison.csv").exists()
    assert (wf_dir / "comparison_detailed.csv").exists()
    assert (wf_dir / "portfolio_test_returns.csv").exists()

    folds_df = pd.read_csv(wf_dir / "folds.csv")
    assert "test_excess_total_return" in folds_df.columns
    assert "test_daily_hit_rate_vs_benchmark" in folds_df.columns
    assert "test_beat_benchmark_total_return" in folds_df.columns
