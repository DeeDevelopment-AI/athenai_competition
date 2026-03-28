#!/usr/bin/env python3
"""
Phase 7 Walk-Forward: PSO walk-forward evaluation on unseen data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner
from scripts.run_phase7 import Phase7Runner
from src.evaluation.audit import split_periodic_allocation_rows
from src.evaluation.comparison import StrategyComparison
from src.evaluation.metrics import compute_full_metrics
from src.evaluation.walk_forward import WalkForwardValidator
from src.swarm import SwarmAllocatorBacktester, SwarmConfig


class Phase7WalkForwardRunner(PhaseRunner):
    phase_name = "Phase 7 Walk-Forward Evaluation"
    phase_number = 78

    def get_output_dir(self) -> Path:
        if self.args and self.args.output_dir:
            return Path(self.args.output_dir)
        return self.op.swarm_pso.root

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--sample", type=int, default=None)
        parser.add_argument("--input-dir", type=str, default=None)
        parser.add_argument("--analysis-dir", type=str, default=None)
        parser.add_argument("--phase2-cluster-filter", action="store_true")
        parser.add_argument(
            "--phase2-cluster-source",
            type=str,
            default="temporal_cumulative",
            choices=["behavioral_family", "temporal_cumulative", "temporal_weekly", "temporal_monthly"],
        )
        parser.add_argument(
            "--phase2-cluster-score-mode",
            type=str,
            default="return_low_vol",
            choices=["return_low_vol", "return", "sharpe", "sortino"],
        )
        parser.add_argument("--phase2-cluster-top-k", type=int, default=1)
        parser.add_argument("--phase2-cluster-min-size", type=int, default=20)
        parser.add_argument("--phase2-cluster-min-return", type=float, default=0.01)
        parser.add_argument("--phase2-cluster-max-vol", type=float, default=0.12)
        parser.add_argument("--phase2-cluster-full-history", action="store_true")
        parser.add_argument("--selection-factor", type=str, default="rolling_sharpe_21d")
        parser.add_argument("--lookback-window", type=int, default=63)
        parser.add_argument("--min-history", type=int, default=42)
        parser.add_argument("--particles", type=int, default=96)
        parser.add_argument("--iterations", type=int, default=70)
        parser.add_argument("--rebalance-freq", type=str, default="weekly", choices=["daily", "weekly", "monthly"])
        parser.add_argument("--max-weight", type=float, default=0.40)
        parser.add_argument("--max-family-exposure", type=float, default=0.30)
        parser.add_argument("--expected-return-weight", type=float, default=1.00)
        parser.add_argument("--volatility-weight", type=float, default=0.50)
        parser.add_argument("--tracking-error-weight", type=float, default=0.35)
        parser.add_argument("--turnover-weight", type=float, default=0.15)
        parser.add_argument("--concentration-weight", type=float, default=0.10)
        parser.add_argument("--diversification-weight", type=float, default=0.10)
        parser.add_argument("--family-penalty-weight", type=float, default=0.20)
        parser.add_argument("--risk-budget-weight", type=float, default=0.30)
        parser.add_argument("--sparsity-penalty-weight", type=float, default=0.01)
        parser.add_argument("--regime-focus", type=float, default=1.50)
        parser.add_argument("--target-portfolio-vol", type=float, default=0.16)
        parser.add_argument("--min-active-weight", type=float, default=0.0025)
        parser.add_argument("--min-gross-exposure", type=float, default=0.85)
        parser.add_argument("--under-investment-penalty-weight", type=float, default=0.35)
        parser.add_argument("--objective-name", type=str, default="risk_calibrated_return")
        parser.add_argument("--selection-mode", type=str, default="legacy", choices=["legacy", "benchmark_aware"])
        parser.add_argument("--normalize-objective-metrics", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--cpu-only", action="store_true")
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--end-date", type=str, default=None)
        parser.add_argument("--train-window", type=int, default=252)
        parser.add_argument("--validation-window", type=int, default=63)
        parser.add_argument("--test-window", type=int, default=63)
        parser.add_argument("--step-size", type=int, default=63)
        parser.add_argument("--expanding", action="store_true")
        parser.add_argument("--max-folds", type=int, default=None)

    def run(self, args: argparse.Namespace) -> dict:
        phase7 = Phase7Runner()
        input_dir = Path(args.input_dir) if args.input_dir else phase7.dp.processed.root
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else phase7.dp.processed.analysis.root
        output_dir = self.get_output_dir()

        with self.step("78.1 Load Shared Artifacts"):
            sampled_cols = phase7._resolve_sampled_algorithms(input_dir, args.sample)
            algo_returns = phase7._load_returns(input_dir, sampled_cols=sampled_cols)
            features = phase7._load_optional_features(input_dir, sampled_algorithms=sampled_cols)
            benchmark_returns = phase7._load_optional_benchmark_returns(input_dir)
            benchmark_weights = phase7._load_optional_benchmark_weights(input_dir, sampled_cols=sampled_cols)
            if args.phase2_cluster_filter:
                algo_returns, features, benchmark_weights, cluster_selection = phase7._apply_phase2_cluster_universe(
                    algo_returns=algo_returns,
                    features=features,
                    benchmark_weights=benchmark_weights,
                    analysis_dir=analysis_dir,
                    output_dir=output_dir,
                    source=args.phase2_cluster_source,
                    score_mode=args.phase2_cluster_score_mode,
                    top_k=args.phase2_cluster_top_k,
                    min_size=args.phase2_cluster_min_size,
                    min_return=args.phase2_cluster_min_return,
                    max_vol=args.phase2_cluster_max_vol,
                    latest_only=not args.phase2_cluster_full_history,
                )
            else:
                cluster_selection = None
            family_labels = phase7._load_optional_family_labels(analysis_dir, algo_returns.columns, allow_default_fallback=True)
            family_alpha_scores = phase7._load_optional_family_alpha_scores(analysis_dir, family_labels, allow_default_fallback=True)
            cluster_history = phase7._load_optional_cluster_history(analysis_dir, allow_default_fallback=True)
            cluster_stability = phase7._load_optional_cluster_stability(analysis_dir, allow_default_fallback=True)
            cluster_alpha_scores = phase7._load_optional_cluster_alpha_scores(analysis_dir, cluster_history, allow_default_fallback=True)
            regime_labels = phase7._load_optional_regime_labels(analysis_dir, allow_default_fallback=True)

        with self.step("78.2 Generate Walk-Forward Folds"):
            evaluation_returns = algo_returns.copy()
            if args.start_date or args.end_date:
                start_ts = pd.Timestamp(args.start_date) if args.start_date else evaluation_returns.index.min()
                end_ts = pd.Timestamp(args.end_date) if args.end_date else evaluation_returns.index.max()
                evaluation_returns = evaluation_returns.loc[(evaluation_returns.index >= start_ts) & (evaluation_returns.index <= end_ts)]
            validator = WalkForwardValidator(
                train_window=args.train_window,
                val_window=args.validation_window,
                test_window=args.test_window,
                step_size=args.step_size,
                expanding=args.expanding,
            )
            folds = validator.generate_folds(evaluation_returns.index)
            if args.max_folds is not None:
                folds = folds[: args.max_folds]

        with self.step("78.3 Run Walk-Forward Backtests"):
            config = SwarmConfig(
                lookback_window=args.lookback_window,
                rebalance_frequency=args.rebalance_freq,
                top_k=256,
                min_history=args.min_history,
                n_particles=args.particles,
                n_iterations=args.iterations,
                max_weight=args.max_weight,
                max_family_exposure=args.max_family_exposure,
                expected_return_weight=args.expected_return_weight,
                volatility_weight=args.volatility_weight,
                tracking_error_weight=args.tracking_error_weight,
                turnover_weight=args.turnover_weight,
                concentration_weight=args.concentration_weight,
                diversification_weight=args.diversification_weight,
                family_penalty_weight=args.family_penalty_weight,
                risk_budget_weight=args.risk_budget_weight,
                sparsity_penalty_weight=args.sparsity_penalty_weight,
                regime_focus=args.regime_focus,
                target_portfolio_vol=args.target_portfolio_vol,
                min_active_weight=args.min_active_weight,
                min_gross_exposure=args.min_gross_exposure,
                under_investment_penalty_weight=args.under_investment_penalty_weight,
                objective_name=args.objective_name,
                selection_mode=args.selection_mode,
                normalize_objective_metrics=args.normalize_objective_metrics,
                seed=args.seed,
                use_gpu=not args.cpu_only,
            )

            fold_rows = []
            allocation_rows = []
            stitched_portfolio = []
            stitched_benchmark = []
            for fold in folds:
                backtester = SwarmAllocatorBacktester(
                    algo_returns=algo_returns,
                    features=features,
                    benchmark_returns=benchmark_returns,
                    benchmark_weights=benchmark_weights,
                    config=config,
                    family_labels=family_labels,
                    family_alpha_scores=family_alpha_scores,
                    cluster_history=cluster_history,
                    cluster_stability=cluster_stability,
                    cluster_alpha_scores=cluster_alpha_scores,
                    regime_labels=regime_labels,
                    selection_factor=args.selection_factor,
                )
                result = backtester.run(
                    start_date=str(fold["train_start"].date()),
                    end_date=str(fold["test_end"].date()),
                )
                test_returns = result.portfolio_returns.loc[fold["test_start"]:fold["test_end"]]
                benchmark_test = result.benchmark_returns.loc[fold["test_start"]:fold["test_end"]] if result.benchmark_returns is not None else None
                test_metrics = compute_full_metrics(test_returns, benchmark_test)
                fold_rows.append(
                    {
                        "fold_id": fold["fold_id"],
                        "test_start": str(fold["test_start"].date()),
                        "test_end": str(fold["test_end"].date()),
                        "test_sharpe_ratio": float(test_metrics.get("sharpe_ratio", 0.0)),
                        "test_annualized_return": float(test_metrics.get("annualized_return", 0.0)),
                        "test_benchmark_annualized_volatility": float(test_metrics.get("benchmark_annualized_volatility", 0.0)),
                        "test_max_drawdown": float(test_metrics.get("max_drawdown", 0.0)),
                    }
                )
                allocation_rows.extend(
                    split_periodic_allocation_rows(
                        result.weights.index,
                        result.weights.to_numpy(dtype=float, copy=False),
                        result.weights.columns.tolist(),
                        split_windows=[
                            ("train", fold["train_start"], fold["train_end"]),
                            ("validation", fold["val_start"], fold["val_end"]),
                            ("test", fold["test_start"], fold["test_end"]),
                        ],
                        final_period_end=fold["test_end"],
                        metadata={
                            "strategy": "pso_walk_forward",
                            "fold_id": int(fold["fold_id"]),
                            "selection_factor": args.selection_factor,
                        },
                    )
                )
                stitched_portfolio.append(test_returns)
                if benchmark_test is not None:
                    stitched_benchmark.append(benchmark_test)

        stitched_portfolio_returns = self._stitch_returns(stitched_portfolio)
        stitched_benchmark_returns = self._stitch_returns(stitched_benchmark) if stitched_benchmark else None

        with self.step("78.4 Build Summary"):
            comparison_table = pd.DataFrame()
            if stitched_benchmark_returns is not None and not stitched_benchmark_returns.empty:
                comparison = StrategyComparison(stitched_benchmark_returns)
                comparison.add_strategy("pso_walk_forward", stitched_portfolio_returns)
                comparison_table = comparison.get_comparison_table()
            overall_metrics = compute_full_metrics(stitched_portfolio_returns, stitched_benchmark_returns)

        wf_dir = output_dir / "walk_forward"
        wf_dir.mkdir(parents=True, exist_ok=True)
        folds_df = pd.DataFrame(fold_rows)
        folds_df.to_csv(wf_dir / "folds.csv", index=False)
        if allocation_rows:
            pd.DataFrame(allocation_rows).to_csv(wf_dir / "rebalance_allocations.csv", index=False)
        stitched_portfolio_returns.to_csv(wf_dir / "portfolio_test_returns.csv", header=True)
        if stitched_benchmark_returns is not None and not stitched_benchmark_returns.empty:
            stitched_benchmark_returns.to_csv(wf_dir / "benchmark_test_returns.csv", header=True)
        if not comparison_table.empty:
            comparison_table.to_csv(wf_dir / "comparison.csv")

        summary = {
            "selection_factor": args.selection_factor,
            "n_folds": len(fold_rows),
            "overall_metrics": overall_metrics,
            "phase2_cluster_selection": cluster_selection,
            "rebalance_allocations_path": str(wf_dir / "rebalance_allocations.csv") if allocation_rows else None,
            "mean_test_sharpe": float(folds_df["test_sharpe_ratio"].mean()) if not folds_df.empty else 0.0,
            "mean_test_return": float(folds_df["test_annualized_return"].mean()) if not folds_df.empty else 0.0,
        }
        (wf_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        return {
            "summary": summary,
            "walk_forward_dir": str(wf_dir),
        }

    @staticmethod
    def _stitch_returns(segments: list[pd.Series]) -> pd.Series:
        if not segments:
            return pd.Series(dtype=float)
        combined = pd.concat(segments).sort_index()
        return combined[~combined.index.duplicated(keep="first")]


def main():
    runner = Phase7WalkForwardRunner()
    runner.execute()


if __name__ == "__main__":
    main()
