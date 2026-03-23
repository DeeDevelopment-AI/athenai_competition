#!/usr/bin/env python3
"""
Tune Phase 7 swarm hyperparameters using temporal validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner
from scripts.run_phase7 import Phase7Runner
from src.swarm import SwarmAllocatorBacktester, SwarmConfig


class Phase7TuningRunner(PhaseRunner):
    phase_name = "Phase 7 Tuning: Temporal Validation"
    phase_number = 77

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--sample", type=int, default=None, help="Use only N algorithms")
        parser.add_argument("--max-trials", type=int, default=8, help="Maximum candidate configurations to evaluate")
        parser.add_argument("--cpu-only", action="store_true")
        parser.add_argument("--input-dir", type=str, default=None)
        parser.add_argument("--analysis-dir", type=str, default=None)
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--end-date", type=str, default=None)
        parser.add_argument("--train-ratio", type=float, default=0.60)
        parser.add_argument("--validation-ratio", type=float, default=0.20)
        parser.add_argument("--max-validation-drawdown", type=float, default=0.10)
        parser.add_argument("--max-test-drawdown", type=float, default=0.12)
        parser.add_argument("--validation-weight", type=float, default=0.70)
        parser.add_argument("--test-weight", type=float, default=0.30)
        parser.add_argument(
            "--selection-objective",
            type=str,
            default="blended_excess_return",
            choices=["validation_excess_return", "validation_sharpe", "blended_excess_return", "blended_sharpe"],
            help="How to choose the best trial",
        )

    def run(self, args: argparse.Namespace) -> dict:
        phase7 = Phase7Runner()
        input_dir = Path(args.input_dir) if args.input_dir else phase7.dp.processed.root
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else phase7.dp.processed.analysis.root
        output_dir = self.get_output_dir()

        with self.step("Load Shared Artifacts"):
            shared = self._load_shared_artifacts(phase7, input_dir, analysis_dir, args.sample)

        candidates = self._candidate_configs(cpu_only=args.cpu_only)[: args.max_trials]
        trial_rows = []
        best_row = None

        for idx, candidate in enumerate(candidates, start=1):
            with self.step(f"Trial {idx}: {candidate['name']}"):
                config = SwarmConfig(**candidate["config"])
                backtester = SwarmAllocatorBacktester(
                    algo_returns=shared["algo_returns"],
                    features=shared["features"],
                    benchmark_returns=shared["benchmark_returns"],
                    benchmark_weights=shared["benchmark_weights"],
                    config=config,
                    family_labels=shared["family_labels"],
                    family_alpha_scores=shared["family_alpha_scores"],
                    cluster_history=shared["cluster_history"],
                    cluster_stability=shared["cluster_stability"],
                    cluster_alpha_scores=shared["cluster_alpha_scores"],
                    regime_labels=shared["regime_labels"],
                    selection_factor=candidate["selection_factor"],
                )
                result = backtester.run(start_date=args.start_date, end_date=args.end_date)
                split_summary = phase7._build_temporal_split_summary(
                    portfolio_returns=result.portfolio_returns,
                    benchmark_returns=result.benchmark_returns,
                    evaluation_start=result.weights.index.min(),
                    train_ratio=args.train_ratio,
                    validation_ratio=args.validation_ratio,
                )
                validation = split_summary.get("validation", {})
                test = split_summary.get("test", {})
                selection_score = self._compute_selection_score(
                    validation=validation,
                    test=test,
                    max_validation_drawdown=args.max_validation_drawdown,
                    max_test_drawdown=args.max_test_drawdown,
                    validation_weight=args.validation_weight,
                    test_weight=args.test_weight,
                    objective=args.selection_objective,
                )

                row = {
                    "trial_name": candidate["name"],
                    "selection_factor": candidate["selection_factor"],
                    "selection_score": selection_score,
                    "validation_excess_return": float(validation.get("excess_return", 0.0)),
                    "validation_annualized_return": float(validation.get("annualized_return", 0.0)),
                    "validation_sharpe_ratio": float(validation.get("sharpe_ratio", 0.0)),
                    "validation_max_drawdown": float(validation.get("max_drawdown", 0.0)),
                    "validation_benchmark_annualized_return": float(validation.get("benchmark_annualized_return", 0.0)),
                    "test_annualized_return": float(test.get("annualized_return", 0.0)),
                    "test_sharpe_ratio": float(test.get("sharpe_ratio", 0.0)),
                    "test_benchmark_annualized_return": float(test.get("benchmark_annualized_return", 0.0)),
                    **candidate["config"],
                    **result.summary,
                }
                trial_rows.append(row)

                if best_row is None or row["selection_score"] > best_row["selection_score"]:
                    best_row = row

        tuning_dir = output_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        trials_df = pd.DataFrame(trial_rows).sort_values(
            by=["selection_score", "validation_excess_return", "test_annualized_return"],
            ascending=[False, False, False],
        )
        trials_path = tuning_dir / "trials.csv"
        trials_df.to_csv(trials_path, index=False)

        best_path = tuning_dir / "best_config.json"
        with open(best_path, "w", encoding="utf-8") as handle:
            json.dump(best_row, handle, indent=2, default=str)

        report_path = tuning_dir / "SUMMARY.md"
        report_path.write_text(
            self._build_report(
                trials_df,
                best_row,
                args.selection_objective,
                args.validation_weight,
                args.test_weight,
            ),
            encoding="utf-8",
        )

        return {
            "n_trials": len(trial_rows),
            "best_trial": best_row,
            "trials_path": str(trials_path),
            "best_config_path": str(best_path),
            "report_path": str(report_path),
        }

    def _load_shared_artifacts(
        self,
        phase7: Phase7Runner,
        input_dir: Path,
        analysis_dir: Path,
        sample: int | None,
    ) -> dict:
        algo_returns = phase7._load_returns(input_dir)
        features = phase7._load_optional_features(input_dir)
        benchmark_returns = phase7._load_optional_benchmark_returns(input_dir)
        benchmark_weights = phase7._load_optional_benchmark_weights(input_dir)

        if sample and sample < len(algo_returns.columns):
            sampled_cols = algo_returns.columns[:sample]
            algo_returns = algo_returns[sampled_cols]
            if features is not None:
                feature_cols = [
                    c for c in features.columns
                    if any(c.startswith(f"{algo}_") for algo in sampled_cols)
                    or c.startswith("rolling_market")
                ]
                features = features[feature_cols]
            if benchmark_weights is not None:
                benchmark_weights = benchmark_weights.reindex(columns=sampled_cols, fill_value=0.0)

        family_labels = phase7._load_optional_family_labels(
            analysis_dir,
            algo_returns.columns,
            allow_default_fallback=True,
        )
        family_alpha_scores = phase7._load_optional_family_alpha_scores(
            analysis_dir,
            family_labels,
            allow_default_fallback=True,
        )
        cluster_history = phase7._load_optional_cluster_history(
            analysis_dir,
            allow_default_fallback=True,
        )
        cluster_stability = phase7._load_optional_cluster_stability(
            analysis_dir,
            allow_default_fallback=True,
        )
        cluster_alpha_scores = phase7._load_optional_cluster_alpha_scores(
            analysis_dir,
            cluster_history,
            allow_default_fallback=True,
        )
        regime_labels = phase7._load_optional_regime_labels(
            analysis_dir,
            allow_default_fallback=True,
        )

        return {
            "algo_returns": algo_returns,
            "features": features,
            "benchmark_returns": benchmark_returns,
            "benchmark_weights": benchmark_weights,
            "family_labels": family_labels,
            "family_alpha_scores": family_alpha_scores,
            "cluster_history": cluster_history,
            "cluster_stability": cluster_stability,
            "cluster_alpha_scores": cluster_alpha_scores,
            "regime_labels": regime_labels,
        }

    def _compute_selection_score(
        self,
        validation: dict,
        test: dict,
        max_validation_drawdown: float,
        max_test_drawdown: float,
        validation_weight: float,
        test_weight: float,
        objective: str,
    ) -> float:
        validation_drawdown = abs(float(validation.get("max_drawdown", 0.0)))
        validation_penalty = max(0.0, validation_drawdown - max_validation_drawdown) * 5.0
        test_drawdown = abs(float(test.get("max_drawdown", 0.0)))
        test_penalty = max(0.0, test_drawdown - max_test_drawdown) * 5.0

        validation_benchmark_return = float(validation.get("benchmark_annualized_return", 0.0))
        test_benchmark_return = float(test.get("benchmark_annualized_return", 0.0))
        validation_excess = float(validation.get("annualized_return", 0.0)) - validation_benchmark_return
        test_excess = float(test.get("annualized_return", 0.0)) - test_benchmark_return
        validation_sharpe = float(validation.get("sharpe_ratio", 0.0))
        test_sharpe = float(test.get("sharpe_ratio", 0.0))

        if objective == "validation_sharpe":
            return validation_sharpe - validation_penalty
        if objective == "validation_excess_return":
            return validation_excess - validation_penalty
        if objective == "blended_sharpe":
            return (
                validation_weight * validation_sharpe
                + test_weight * test_sharpe
                - validation_penalty
                - test_penalty
            )
        return (
            validation_weight * validation_excess
            + test_weight * test_excess
            - validation_penalty
            - test_penalty
        )

    def _candidate_configs(self, cpu_only: bool) -> list[dict]:
        use_gpu = not cpu_only
        common = {
            "rebalance_frequency": "weekly",
            "seed": 42,
            "use_gpu": use_gpu,
            "selection_mode": "legacy",
            "normalize_objective_metrics": False,
            "excess_return_weight": 0.0,
            "information_ratio_weight": 0.0,
            "benchmark_hit_rate_weight": 0.0,
            "downside_excess_weight": 0.0,
        }
        return [
            {
                "name": "gate_balanced",
                "selection_factor": "rolling_sharpe_21d",
                "config": {
                    **common,
                    "lookback_window": 126,
                    "top_k": 128,
                    "min_history": 63,
                    "n_particles": 128,
                    "n_iterations": 80,
                    "inertia": 0.70,
                    "cognitive_weight": 1.40,
                    "social_weight": 1.30,
                    "max_weight": 0.40,
                    "expected_return_weight": 1.00,
                    "volatility_weight": 0.50,
                    "tracking_error_weight": 0.35,
                    "turnover_weight": 0.15,
                    "concentration_weight": 0.10,
                    "diversification_weight": 0.10,
                    "family_penalty_weight": 0.20,
                    "risk_budget_weight": 0.30,
                    "sparsity_penalty_weight": 0.01,
                    "regime_focus": 1.50,
                    "target_portfolio_vol": 0.16,
                    "min_active_weight": 0.0025,
                    "min_gross_exposure": 0.85,
                    "under_investment_penalty_weight": 0.35,
                    "cluster_alpha_weight": 1.25,
                    "cluster_gate_min_signal": 0.05,
                    "cluster_gate_min_stability": 0.20,
                    "cluster_gate_score_quantile": 0.55,
                    "min_selection_pass_ratio": 0.75,
                    "objective_name": "gate_balanced",
                },
            },
            {
                "name": "gate_strict",
                "selection_factor": "rolling_sharpe_21d",
                "config": {
                    **common,
                    "lookback_window": 126,
                    "top_k": 96,
                    "min_history": 63,
                    "n_particles": 128,
                    "n_iterations": 80,
                    "inertia": 0.70,
                    "cognitive_weight": 1.40,
                    "social_weight": 1.30,
                    "max_weight": 0.40,
                    "expected_return_weight": 1.00,
                    "volatility_weight": 0.55,
                    "tracking_error_weight": 0.35,
                    "turnover_weight": 0.15,
                    "concentration_weight": 0.10,
                    "diversification_weight": 0.10,
                    "family_penalty_weight": 0.20,
                    "risk_budget_weight": 0.30,
                    "sparsity_penalty_weight": 0.01,
                    "regime_focus": 1.50,
                    "target_portfolio_vol": 0.15,
                    "min_active_weight": 0.0030,
                    "min_gross_exposure": 0.85,
                    "under_investment_penalty_weight": 0.40,
                    "cluster_alpha_weight": 1.45,
                    "cluster_gate_min_signal": 0.12,
                    "cluster_gate_min_stability": 0.28,
                    "cluster_gate_score_quantile": 0.70,
                    "min_selection_pass_ratio": 0.85,
                    "objective_name": "gate_strict",
                },
            },
            {
                "name": "gate_loose_alpha",
                "selection_factor": "rolling_return_63d",
                "config": {
                    **common,
                    "lookback_window": 126,
                    "top_k": 144,
                    "min_history": 63,
                    "n_particles": 128,
                    "n_iterations": 80,
                    "inertia": 0.68,
                    "cognitive_weight": 1.50,
                    "social_weight": 1.20,
                    "max_weight": 0.40,
                    "expected_return_weight": 1.10,
                    "volatility_weight": 0.45,
                    "tracking_error_weight": 0.30,
                    "turnover_weight": 0.10,
                    "concentration_weight": 0.10,
                    "diversification_weight": 0.12,
                    "family_penalty_weight": 0.15,
                    "risk_budget_weight": 0.25,
                    "sparsity_penalty_weight": 0.005,
                    "regime_focus": 1.40,
                    "target_portfolio_vol": 0.17,
                    "min_active_weight": 0.0020,
                    "min_gross_exposure": 0.82,
                    "under_investment_penalty_weight": 0.25,
                    "cluster_alpha_weight": 1.10,
                    "cluster_gate_min_signal": 0.02,
                    "cluster_gate_min_stability": 0.15,
                    "cluster_gate_score_quantile": 0.45,
                    "min_selection_pass_ratio": 0.65,
                    "objective_name": "gate_loose_alpha",
                },
            },
            {
                "name": "quality_momentum",
                "selection_factor": "rolling_sharpe_63d",
                "config": {
                    **common,
                    "lookback_window": 126,
                    "top_k": 96,
                    "min_history": 84,
                    "n_particles": 144,
                    "n_iterations": 90,
                    "inertia": 0.66,
                    "cognitive_weight": 1.45,
                    "social_weight": 1.25,
                    "max_weight": 0.35,
                    "expected_return_weight": 1.15,
                    "volatility_weight": 0.48,
                    "tracking_error_weight": 0.30,
                    "turnover_weight": 0.10,
                    "concentration_weight": 0.12,
                    "diversification_weight": 0.14,
                    "family_penalty_weight": 0.15,
                    "risk_budget_weight": 0.28,
                    "sparsity_penalty_weight": 0.01,
                    "regime_focus": 1.60,
                    "target_portfolio_vol": 0.16,
                    "min_active_weight": 0.0025,
                    "min_gross_exposure": 0.86,
                    "under_investment_penalty_weight": 0.30,
                    "cluster_alpha_weight": 1.35,
                    "cluster_gate_min_signal": 0.08,
                    "cluster_gate_min_stability": 0.22,
                    "cluster_gate_score_quantile": 0.60,
                    "min_selection_pass_ratio": 0.80,
                    "objective_name": "quality_momentum",
                },
            },
            {
                "name": "stability_first",
                "selection_factor": "rolling_calmar_63d",
                "config": {
                    **common,
                    "lookback_window": 252,
                    "top_k": 80,
                    "min_history": 84,
                    "n_particles": 128,
                    "n_iterations": 90,
                    "inertia": 0.72,
                    "cognitive_weight": 1.20,
                    "social_weight": 1.45,
                    "max_weight": 0.30,
                    "expected_return_weight": 0.95,
                    "volatility_weight": 0.60,
                    "tracking_error_weight": 0.30,
                    "turnover_weight": 0.08,
                    "concentration_weight": 0.12,
                    "diversification_weight": 0.16,
                    "family_penalty_weight": 0.10,
                    "risk_budget_weight": 0.35,
                    "sparsity_penalty_weight": 0.015,
                    "regime_focus": 1.70,
                    "target_portfolio_vol": 0.14,
                    "min_active_weight": 0.0030,
                    "min_gross_exposure": 0.88,
                    "under_investment_penalty_weight": 0.35,
                    "cluster_alpha_weight": 1.40,
                    "cluster_gate_min_signal": 0.10,
                    "cluster_gate_min_stability": 0.30,
                    "cluster_gate_score_quantile": 0.65,
                    "min_selection_pass_ratio": 0.85,
                    "objective_name": "stability_first",
                },
            },
            {
                "name": "cluster_conviction",
                "selection_factor": "rolling_profit_factor_63d",
                "config": {
                    **common,
                    "lookback_window": 126,
                    "top_k": 72,
                    "min_history": 63,
                    "n_particles": 160,
                    "n_iterations": 100,
                    "inertia": 0.64,
                    "cognitive_weight": 1.55,
                    "social_weight": 1.10,
                    "max_weight": 0.40,
                    "expected_return_weight": 1.20,
                    "volatility_weight": 0.42,
                    "tracking_error_weight": 0.25,
                    "turnover_weight": 0.08,
                    "concentration_weight": 0.14,
                    "diversification_weight": 0.12,
                    "family_penalty_weight": 0.10,
                    "risk_budget_weight": 0.25,
                    "sparsity_penalty_weight": 0.005,
                    "regime_focus": 1.50,
                    "target_portfolio_vol": 0.17,
                    "min_active_weight": 0.0020,
                    "min_gross_exposure": 0.88,
                    "under_investment_penalty_weight": 0.20,
                    "cluster_alpha_weight": 1.70,
                    "cluster_gate_min_signal": 0.12,
                    "cluster_gate_min_stability": 0.24,
                    "cluster_gate_score_quantile": 0.72,
                    "min_selection_pass_ratio": 0.75,
                    "objective_name": "cluster_conviction",
                },
            },
            {
                "name": "regime_responsive",
                "selection_factor": "rolling_return_21d",
                "config": {
                    **common,
                    "lookback_window": 84,
                    "top_k": 96,
                    "min_history": 42,
                    "n_particles": 144,
                    "n_iterations": 90,
                    "inertia": 0.62,
                    "cognitive_weight": 1.60,
                    "social_weight": 1.15,
                    "max_weight": 0.35,
                    "expected_return_weight": 1.10,
                    "volatility_weight": 0.44,
                    "tracking_error_weight": 0.28,
                    "turnover_weight": 0.12,
                    "concentration_weight": 0.10,
                    "diversification_weight": 0.14,
                    "family_penalty_weight": 0.10,
                    "risk_budget_weight": 0.22,
                    "sparsity_penalty_weight": 0.005,
                    "regime_focus": 1.90,
                    "target_portfolio_vol": 0.17,
                    "min_active_weight": 0.0020,
                    "min_gross_exposure": 0.84,
                    "under_investment_penalty_weight": 0.20,
                    "cluster_alpha_weight": 1.30,
                    "cluster_gate_min_signal": 0.08,
                    "cluster_gate_min_stability": 0.18,
                    "cluster_gate_score_quantile": 0.58,
                    "min_selection_pass_ratio": 0.70,
                    "objective_name": "regime_responsive",
                },
            },
            {
                "name": "defensive_quality",
                "selection_factor": "rolling_calmar_21d",
                "config": {
                    **common,
                    "lookback_window": 126,
                    "top_k": 88,
                    "min_history": 63,
                    "n_particles": 128,
                    "n_iterations": 80,
                    "inertia": 0.74,
                    "cognitive_weight": 1.15,
                    "social_weight": 1.50,
                    "max_weight": 0.28,
                    "expected_return_weight": 0.90,
                    "volatility_weight": 0.62,
                    "tracking_error_weight": 0.32,
                    "turnover_weight": 0.09,
                    "concentration_weight": 0.10,
                    "diversification_weight": 0.18,
                    "family_penalty_weight": 0.12,
                    "risk_budget_weight": 0.34,
                    "sparsity_penalty_weight": 0.01,
                    "regime_focus": 1.60,
                    "target_portfolio_vol": 0.14,
                    "min_active_weight": 0.0035,
                    "min_gross_exposure": 0.90,
                    "under_investment_penalty_weight": 0.32,
                    "cluster_alpha_weight": 1.25,
                    "cluster_gate_min_signal": 0.10,
                    "cluster_gate_min_stability": 0.28,
                    "cluster_gate_score_quantile": 0.68,
                    "min_selection_pass_ratio": 0.82,
                    "objective_name": "defensive_quality",
                },
            },
        ]

    def _build_report(
        self,
        trials_df: pd.DataFrame,
        best_row: dict,
        objective: str,
        validation_weight: float,
        test_weight: float,
    ) -> str:
        top_rows = trials_df.head(5)
        lines = [
            "# Phase 7 Tuning Summary",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Selection objective: {objective}",
            f"Validation weight: {validation_weight:.2f}",
            f"Test weight: {test_weight:.2f}",
            "",
            "## Best Trial",
            "",
            f"- Name: {best_row['trial_name']}",
            f"- Selection factor: {best_row['selection_factor']}",
            f"- Validation annualized return: {best_row['validation_annualized_return']:.2%}",
            f"- Validation benchmark annualized return: {best_row['validation_benchmark_annualized_return']:.2%}",
            f"- Validation Sharpe: {best_row['validation_sharpe_ratio']:.3f}",
            f"- Validation max drawdown: {best_row['validation_max_drawdown']:.2%}",
            f"- Test annualized return: {best_row['test_annualized_return']:.2%}",
            f"- Test Sharpe: {best_row['test_sharpe_ratio']:.3f}",
            f"- Test benchmark annualized return: {best_row['test_benchmark_annualized_return']:.2%}",
            "",
            "## Top Trials",
            "",
            "| Trial | Factor | Selection Score | Val Return | Val Bench | Val Sharpe | Test Return | Test Sharpe |",
            "|------|--------|-----------------|------------|-----------|------------|-------------|-------------|",
        ]
        for _, row in top_rows.iterrows():
            lines.append(
                f"| {row['trial_name']} | {row['selection_factor']} | {row['selection_score']:.4f} | "
                f"{row['validation_annualized_return']:.2%} | {row['validation_benchmark_annualized_return']:.2%} | "
                f"{row['validation_sharpe_ratio']:.3f} | {row['test_annualized_return']:.2%} | {row['test_sharpe_ratio']:.3f} |"
            )
        lines.append("")
        return "\n".join(lines)


def main():
    runner = Phase7TuningRunner()
    runner.execute()


if __name__ == "__main__":
    main()
