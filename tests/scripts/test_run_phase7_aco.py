"""Tests for scripts/run_phase7_aco.py."""

import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_phase7_aco import Phase7ACORunner


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
    root = Path.cwd() / ".tmp_test_artifacts" / f"phase7_aco_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=260, freq="B")
    algos = [f"algo_{i}" for i in range(12)]

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
    returns.iloc[:, :3] += 0.0007
    returns.to_parquet(algorithms_dir / "returns.parquet")

    features_data = {}
    rolling_sharpe = returns.rolling(21).mean() / returns.rolling(21).std().replace(0, np.nan)
    for algo in algos:
        features_data[f"{algo}_rolling_sharpe_21d"] = rolling_sharpe[algo].fillna(0.0)
        features_data[f"{algo}_rolling_calmar_21d"] = rolling_sharpe[algo].fillna(0.0) * 0.5
        features_data[f"{algo}_rolling_profit_factor_21d"] = (1.1 + rolling_sharpe[algo].fillna(0.0)).clip(lower=0.5)
        features_data[f"{algo}_rolling_drawdown_63d"] = (-returns[algo].rolling(63).sum().abs()).fillna(0.02)
    features = pd.DataFrame(features_data, index=dates)
    features.to_parquet(algorithms_dir / "features.parquet")

    benchmark_returns = pd.DataFrame({"benchmark": returns.mean(axis=1)}, index=dates)
    benchmark_returns.to_csv(benchmark_dir / "daily_returns.csv")

    benchmark_weights = pd.DataFrame(1.0 / len(algos), index=dates, columns=algos)
    benchmark_weights.to_parquet(benchmark_dir / "weights.parquet")

    family_labels = pd.Series([f"family_{i % 3}" for i in range(len(algos))], index=algos, name="family")
    family_labels.to_csv(analysis_behavioral_dir / "family_labels.csv")

    cluster_dates = pd.date_range(dates[20], periods=20, freq="W-FRI")
    cluster_rows = []
    for week_idx, week_end in enumerate(cluster_dates):
        for algo_idx, algo in enumerate(algos):
            cluster_id = 2 if algo_idx < 4 else (4 if algo_idx < 8 else 0)
            cluster_rows.append(
                {
                    "week_end": week_end,
                    "algo_id": algo,
                    "cluster_cumulative": cluster_id,
                    "cluster_weekly": cluster_id if week_idx % 3 else cluster_id,
                    "cluster_monthly": cluster_id,
                }
            )
    pd.DataFrame(cluster_rows).to_csv(analysis_temporal_dir / "cluster_history.csv", index=False)

    stability = pd.DataFrame(
        {
            "algo_id": algos,
            "n_weeks_active": [len(cluster_dates)] * len(algos),
            "n_cluster_changes": [2, 3, 2, 1, 4, 5, 4, 3, 10, 11, 12, 13],
            "stability_ratio": [0.80, 0.76, 0.78, 0.82, 0.66, 0.64, 0.61, 0.60, 0.25, 0.22, 0.18, 0.15],
            "dominant_cluster": [2, 2, 2, 2, 4, 4, 4, 4, 0, 0, 1, 1],
        }
    )
    stability.to_csv(analysis_temporal_dir / "cluster_stability.csv", index=False)

    profiles = pd.DataFrame(
        {
            "algo_id": algos,
            "ann_return": [0.18, 0.17, 0.16, 0.15, 0.12, 0.11, 0.10, 0.09, -0.02, -0.03, -0.01, -0.04],
            "ann_volatility": [0.20, 0.21, 0.19, 0.18, 0.23, 0.24, 0.25, 0.26, 0.18, 0.19, 0.20, 0.22],
            "sharpe": [0.90, 0.85, 0.84, 0.83, 0.55, 0.50, 0.45, 0.40, -0.10, -0.12, -0.05, -0.15],
        }
    )
    profiles.to_csv(analysis_profiles_dir / "summary.csv", index=False)

    regime_labels = pd.Series([i % 4 for i in range(len(dates))], index=dates, name="regime")
    regime_labels.to_csv(analysis_regimes_dir / "labels.csv")

    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_phase7_aco_runner_executes_with_existing_artifacts(mock_phase_inputs):
    runner = Phase7ACORunner()
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
            "--top-k",
            "4",
            "--ants",
            "24",
            "--iterations",
            "6",
            "--weight-buckets",
            "11",
            "--no-report",
        ]
    )

    assert results["status"] == "completed"
    # Outputs are in a timestamped subdir; use paths from results dict
    assert Path(results["outputs"]["weights_path"]).exists()
    assert Path(results["outputs"]["portfolio_returns_path"]).exists()
    assert Path(results["outputs"]["summary_path"]).exists()
    assert Path(results["outputs"]["comparison_path"]).exists()
    assert results["summary"]["objective_name"] == "aco_sharpe_balanced"
    assert results["artifacts"]["cluster_history_available"] is True
    assert results["artifacts"]["cluster_stability_available"] is True
    assert results["artifacts"]["cluster_alpha_scores_available"] is True
