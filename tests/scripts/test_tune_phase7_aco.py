"""Tests for scripts/tune_phase7_aco.py."""

import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.tune_phase7_aco import Phase7ACOTuningRunner


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
    root = Path.cwd() / ".tmp_test_artifacts" / f"tune_phase7_aco_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    dates = pd.date_range("2021-01-01", periods=220, freq="B")
    algos = [f"algo_{i}" for i in range(10)]

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
        rng.normal(0.0004, 0.01, size=(len(dates), len(algos))),
        index=dates,
        columns=algos,
    )
    returns.iloc[:, :2] += 0.0010
    returns.to_parquet(algorithms_dir / "returns.parquet")

    features_data = {}
    rolling_sharpe = returns.rolling(21).mean() / returns.rolling(21).std().replace(0, np.nan)
    rolling_return = returns.rolling(21).sum()
    rolling_profit = (1.1 + rolling_sharpe.fillna(0.0)).clip(lower=0.5)
    rolling_calmar = (rolling_return / returns.clip(upper=0.0).abs().rolling(21).mean().replace(0, np.nan)).fillna(0.0)
    for algo in algos:
        features_data[f"{algo}_rolling_sharpe_21d"] = rolling_sharpe[algo].fillna(0.0)
        features_data[f"{algo}_rolling_return_21d"] = rolling_return[algo].fillna(0.0)
        features_data[f"{algo}_rolling_profit_factor_21d"] = rolling_profit[algo].fillna(1.0)
        features_data[f"{algo}_rolling_calmar_21d"] = rolling_calmar[algo].fillna(0.0)
        features_data[f"{algo}_rolling_drawdown_21d"] = (-returns[algo].rolling(21).sum().abs()).fillna(0.02)
    pd.DataFrame(features_data, index=dates).to_parquet(algorithms_dir / "features.parquet")

    pd.DataFrame({"benchmark": returns.mean(axis=1)}, index=dates).to_csv(benchmark_dir / "daily_returns.csv")
    pd.DataFrame(1.0 / len(algos), index=dates, columns=algos).to_parquet(benchmark_dir / "weights.parquet")

    pd.Series([f"family_{i % 3}" for i in range(len(algos))], index=algos, name="family").to_csv(
        analysis_behavioral_dir / "family_labels.csv"
    )
    cluster_dates = pd.date_range(dates[20], periods=16, freq="W-FRI")
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
    pd.DataFrame(
        {
            "algo_id": algos,
            "stability_ratio": [0.8, 0.78, 0.76, 0.75, 0.5, 0.48, 0.46, 0.44, 0.3, 0.28],
        }
    ).to_csv(analysis_temporal_dir / "cluster_stability.csv", index=False)
    pd.DataFrame(
        {
            "algo_id": algos,
            "ann_return": np.linspace(0.18, 0.02, len(algos)),
            "ann_volatility": np.linspace(0.20, 0.28, len(algos)),
            "sharpe": np.linspace(0.9, 0.1, len(algos)),
        }
    ).to_csv(analysis_profiles_dir / "summary.csv", index=False)
    pd.Series([i % 3 for i in range(len(dates))], index=dates, name="regime").to_csv(
        analysis_regimes_dir / "labels.csv"
    )

    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_phase7_aco_tuning_runner_searches_factors(mock_phase_inputs):
    runner = Phase7ACOTuningRunner()
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
            "--quick",
            "--max-trials",
            "0",
            "--max-combination-size",
            "2",
            "--train-window",
            "84",
            "--validation-window",
            "21",
            "--test-window",
            "21",
            "--step-size",
            "21",
            "--max-folds",
            "1",
            "--factor",
            "sharpe_21d",
            "momentum_21d",
            "--no-report",
        ]
    )

    factor_search_dir = output_dir / "tuning" / "factor_search"
    assert results["status"] == "completed"
    assert results["n_trials"] == 3
    assert results["selection_scheme"] == "walk_forward"
    assert results["n_folds"] == 1
    assert (factor_search_dir / "trials.csv").exists()
    assert (factor_search_dir / "trial_split_metrics.csv").exists()
    assert (factor_search_dir / "relationship_summary.csv").exists()
    assert (factor_search_dir / "relationship_summary.json").exists()
    assert (factor_search_dir / "best_config.json").exists()
    assert (factor_search_dir / "SUMMARY.md").exists()
    trials_df = pd.read_csv(factor_search_dir / "trials.csv")
    split_metrics_df = pd.read_csv(factor_search_dir / "trial_split_metrics.csv")
    relationship_df = pd.read_csv(factor_search_dir / "relationship_summary.csv")
    assert set(trials_df["factor_alias"]) == {"sharpe_21d", "momentum_21d", "sharpe_21d+momentum_21d"}
    assert results["best_trial"]["selection_uses_test_data"] is False
    assert results["best_trial"]["test_role"] == "report_only_holdout"
    assert results["best_trial"]["selection_scheme"] == "walk_forward"
    assert int(results["best_trial"]["n_folds"]) == 1
    assert int(trials_df["selection_factor_count"].max()) == 2
    assert "train_sharpe_ratio" in trials_df.columns
    assert "train_annualized_volatility" in trials_df.columns
    assert "validation_sharpe_ratio" in trials_df.columns
    assert "test_sharpe_ratio" in trials_df.columns
    assert "validation_test_return_gap" in trials_df.columns
    assert "selection_scheme" in trials_df.columns
    assert "n_folds" in trials_df.columns
    assert set(split_metrics_df["split"]) == {"train", "validation", "test"}
    assert "annualized_return" in split_metrics_df.columns
    assert "annualized_volatility" in split_metrics_df.columns
    assert "validation_test_correlation" in relationship_df.columns


def test_phase7_aco_tuning_runner_rejects_blended_objectives_without_explicit_opt_in():
    runner = Phase7ACOTuningRunner()
    args = runner.create_parser().parse_args(["--selection-objective", "blended_sharpe"])

    with pytest.raises(ValueError, match="allow-test-in-selection"):
        runner._validate_selection_policy(args)
