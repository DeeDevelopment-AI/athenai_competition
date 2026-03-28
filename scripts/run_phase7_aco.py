#!/usr/bin/env python3
"""
=================================================================
PHASE 7A: Ant Colony Meta-Allocator
=================================================================
Build an ACO-based meta allocator on top of prior phase artifacts.

This variant reuses the Phase 7 data plumbing and selection pipeline, but
replaces the portfolio optimizer with an ant-colony search inspired by the
ACO reference paper in `data/references/ACO Portfolio optimization.pdf`.

Usage:
  python scripts/run_phase7_aco.py
  python scripts/run_phase7_aco.py --sample 100 --top-k 32 --ants 64
  python scripts/run_phase7_aco.py --start-date 2021-01-01 --end-date 2023-12-31
  python scripts/run_phase7_aco.py --selection-factor rolling_sharpe_21d rolling_return_63d

Options:
  --sample N                    Use only N algorithms (for faster testing)
  --top-k N                     Candidate pool size passed to optimizer (default: 256)
  --lookback-window N           Lookback window in days (default: 126)
  --min-history N               Minimum history required in days (default: 63)
  --rebalance-freq F            Rebalance frequency: daily, weekly, monthly (default: weekly)
  --selection-factor S [S ...]  Feature(s) for candidate selection (default: rolling_sharpe_21d)
  --start-date YYYY-MM-DD       Start date for backtest
  --end-date YYYY-MM-DD         End date for backtest

ACO Parameters:
  --ants N                      Number of ants per iteration (default: 96)
  --iterations N                Number of ACO iterations (default: 60)
  --weight-buckets N            Discrete weight buckets per algorithm (default: 21)
  --pheromone-power F           Alpha exponent on pheromone (default: 1.0)
  --heuristic-power F           Beta exponent on heuristic desirability (default: 2.0)
  --evaporation-rate F          Pheromone evaporation rate (default: 0.30)
  --pheromone-deposit-scale F   Base pheromone deposit scale (default: 1.0)
  --elite-ants N                Top ants for pheromone reinforcement (default: 8)

Constraint Options:
  --max-weight F                Max weight per algorithm (default: 0.40)
  --max-family-exposure F       Max family exposure (default: 0.30)
  --min-active-weight F         Min weight to count as active (default: 0.0025)
  --min-gross-exposure F        Min total portfolio exposure (default: 0.85)
  --target-portfolio-vol F      Target annualized volatility (default: 0.16)

Objective Weight Options:
  --expected-return-weight F    Weight on expected return (default: 0.80)
  --volatility-weight F         Weight on volatility penalty (default: 0.35)
  --turnover-weight F           Weight on turnover penalty (default: 0.15)
  --concentration-weight F      Weight on concentration penalty (default: 0.10)
  --diversification-weight F    Weight on diversification reward (default: 0.12)
  --family-penalty-weight F     Weight on family concentration (default: 0.20)
  --family-alpha-reward-weight F  Weight on family alpha reward (default: 0.15)
  --risk-budget-weight F        Weight on risk budget deviation (default: 0.25)
  --sparsity-penalty-weight F   Weight on sparsity penalty (default: 0.01)
  --entropy-reward-weight F     Weight on entropy reward (default: 0.05)
  --sharpe-weight F             Weight on Sharpe ratio in objective (default: 1.75)
  --under-investment-penalty-weight F  Penalty for under-investment (default: 0.35)
  --regime-focus F              Regime focus multiplier (default: 1.50)

Advanced Options:
  --objective-name S            Name of the objective function (default: aco_sharpe_balanced)
  --selection-mode S            Universe selection: legacy, benchmark_aware (default: legacy)
  --normalize-objective-metrics Normalize objective components cross-sectionally
  --temporal-split S            Temporal eval split: none, auto (default: auto)
  --train-ratio F               Train ratio for temporal split (default: 0.60)
  --validation-ratio F          Validation ratio for temporal split (default: 0.20)
  --seed N                      Random seed (default: 42)
  --cpu-only                    Force CPU execution even if CUDA is available
  --input-dir PATH              Override Phase 1 processed directory
  --analysis-dir PATH           Override Phase 2 analysis directory
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase7 import Phase7Runner
from src.swarm import ACOAllocatorBacktester, ACOConfig


class Phase7ACORunner(Phase7Runner):
    """Phase 7A: Ant-colony-based meta allocation."""

    phase_name = "Phase 7A: Ant Colony Meta-Allocator"
    phase_number = 8

    def _run_tag(self) -> str:
        if not self.args:
            return ""
        ants = getattr(self.args, 'ants', 96)
        iterations = getattr(self.args, 'iterations', 60)
        factors = getattr(self.args, 'selection_factor', ['rolling_sharpe_21d'])
        factor = factors[0] if isinstance(factors, list) else factors
        return f"aco_{ants}a_{iterations}i_{factor}"

    def get_output_dir(self) -> Path:
        if self.args and self.args.output_dir:
            return Path(self.args.output_dir)
        root = self.op.swarm_aco.root
        run_id = self._run_id or self._make_run_id()
        return root / run_id

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--sample", type=int, default=None, help="Use only N algorithms")
        parser.add_argument("--top-k", type=int, default=256, help="Candidate pool size passed to the optimizer")
        parser.add_argument("--lookback-window", type=int, default=126, help="Lookback window in days")
        parser.add_argument("--min-history", type=int, default=63, help="Minimum history required")
        parser.add_argument("--ants", type=int, default=96, help="Number of ants per ACO iteration")
        parser.add_argument("--iterations", type=int, default=60, help="ACO iterations")
        parser.add_argument("--weight-buckets", type=int, default=21, help="Discrete weight buckets per algorithm")
        parser.add_argument("--pheromone-power", type=float, default=1.0, help="Alpha exponent on pheromone")
        parser.add_argument("--heuristic-power", type=float, default=2.0, help="Beta exponent on heuristic desirability")
        parser.add_argument("--evaporation-rate", type=float, default=0.30, help="Pheromone evaporation rate")
        parser.add_argument("--pheromone-deposit-scale", type=float, default=1.0, help="Base pheromone deposit scale")
        parser.add_argument("--elite-ants", type=int, default=8, help="Top ants used for pheromone reinforcement")
        parser.add_argument(
            "--rebalance-freq",
            type=str,
            default="weekly",
            choices=["daily", "weekly", "monthly"],
            help="Rebalance frequency",
        )
        parser.add_argument("--selection-factor", nargs="+", default=["rolling_sharpe_21d"])
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--end-date", type=str, default=None)
        parser.add_argument("--max-weight", type=float, default=0.40)
        parser.add_argument("--max-family-exposure", type=float, default=0.30)
        parser.add_argument("--expected-return-weight", type=float, default=0.80)
        parser.add_argument("--volatility-weight", type=float, default=0.35)
        parser.add_argument("--turnover-weight", type=float, default=0.15)
        parser.add_argument("--concentration-weight", type=float, default=0.10)
        parser.add_argument("--diversification-weight", type=float, default=0.12)
        parser.add_argument("--family-penalty-weight", type=float, default=0.20)
        parser.add_argument("--family-alpha-reward-weight", type=float, default=0.15)
        parser.add_argument("--risk-budget-weight", type=float, default=0.25)
        parser.add_argument("--sparsity-penalty-weight", type=float, default=0.01)
        parser.add_argument("--entropy-reward-weight", type=float, default=0.05)
        parser.add_argument("--sharpe-weight", type=float, default=1.75)
        parser.add_argument("--regime-focus", type=float, default=1.50)
        parser.add_argument("--target-portfolio-vol", type=float, default=0.16)
        parser.add_argument("--min-active-weight", type=float, default=0.0025)
        parser.add_argument("--min-gross-exposure", type=float, default=0.85)
        parser.add_argument("--under-investment-penalty-weight", type=float, default=0.35)
        parser.add_argument("--objective-name", type=str, default="aco_sharpe_balanced")
        parser.add_argument(
            "--selection-mode",
            type=str,
            default="legacy",
            choices=["legacy", "benchmark_aware"],
            help="Universe selection mode",
        )
        parser.add_argument(
            "--normalize-objective-metrics",
            action="store_true",
            help="Normalize objective components cross-sectionally inside the optimizer",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--cpu-only",
            action="store_true",
            help="Force CPU execution even if CUDA is available",
        )
        parser.add_argument("--input-dir", type=str, default=None, help="Override Phase 1 processed directory")
        parser.add_argument("--analysis-dir", type=str, default=None, help="Override Phase 2 analysis directory")
        parser.add_argument("--phase2-cluster-filter", action="store_true", help="Restrict the ACO universe to the best Phase 2 clusters")
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
        parser.add_argument(
            "--temporal-split",
            type=str,
            default="auto",
            choices=["none", "auto"],
            help="Generate train/validation/test temporal evaluation summary",
        )
        parser.add_argument("--train-ratio", type=float, default=0.60, help="Train ratio for temporal split evaluation")
        parser.add_argument("--validation-ratio", type=float, default=0.20, help="Validation ratio for temporal split evaluation")

    def run(self, args: argparse.Namespace) -> dict:
        results = {
            "config": {},
            "artifacts": {},
        }

        input_dir = Path(args.input_dir) if args.input_dir else self.dp.processed.root
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else self.dp.processed.analysis.root

        # Output dir and run_id are managed by PhaseRunner.execute() via _run_tag()
        output_dir = self.get_output_dir()
        self.logger.info(f"Swarm ACO run ID: {self._run_id or output_dir.name}")

        with self.step("7A.1 Load Phase 1 Artifacts"):
            sampled_cols = self._resolve_sampled_algorithms(input_dir, args.sample)
            algo_returns = self._load_returns(input_dir, sampled_cols=sampled_cols)
            features = self._load_optional_features(
                input_dir,
                sampled_algorithms=sampled_cols,
                extra_suffixes=set(args.selection_factor),
            )
            benchmark_returns = self._load_optional_benchmark_returns(input_dir)
            benchmark_weights = self._load_optional_benchmark_weights(input_dir, sampled_cols=sampled_cols)

            results["artifacts"]["returns_shape"] = list(algo_returns.shape)
            results["artifacts"]["features_available"] = features is not None
            results["artifacts"]["benchmark_returns_available"] = benchmark_returns is not None
            results["artifacts"]["benchmark_weights_available"] = benchmark_weights is not None

        if args.phase2_cluster_filter:
            with self.step("7A.1.5 Apply Phase 2 Cluster Universe"):
                algo_returns, features, benchmark_weights, cluster_selection = self._apply_phase2_cluster_universe(
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
                results["phase2_cluster_selection"] = cluster_selection
                results["artifacts"]["returns_shape"] = list(algo_returns.shape)
                results["artifacts"]["features_available"] = features is not None
                results["artifacts"]["benchmark_weights_available"] = benchmark_weights is not None

        with self.step("7A.2 Load Optional Phase 2 Artifacts"):
            family_labels = self._load_optional_family_labels(
                analysis_dir,
                algo_returns.columns,
                allow_default_fallback=args.analysis_dir is None,
            )
            family_alpha_scores = self._load_optional_family_alpha_scores(
                analysis_dir,
                family_labels,
                allow_default_fallback=args.analysis_dir is None,
            )
            cluster_history = self._load_optional_cluster_history(
                analysis_dir,
                allow_default_fallback=args.analysis_dir is None,
            )
            cluster_stability = self._load_optional_cluster_stability(
                analysis_dir,
                allow_default_fallback=args.analysis_dir is None,
            )
            cluster_alpha_scores = self._load_optional_cluster_alpha_scores(
                analysis_dir,
                cluster_history,
                allow_default_fallback=args.analysis_dir is None,
            )
            regime_labels = self._load_optional_regime_labels(
                analysis_dir,
                allow_default_fallback=args.analysis_dir is None,
            )

            results["artifacts"]["family_labels_available"] = family_labels is not None
            results["artifacts"]["family_alpha_scores_available"] = bool(family_alpha_scores)
            results["artifacts"]["cluster_history_available"] = cluster_history is not None
            results["artifacts"]["cluster_stability_available"] = cluster_stability is not None
            results["artifacts"]["cluster_alpha_scores_available"] = bool(cluster_alpha_scores)
            results["artifacts"]["regime_labels_available"] = regime_labels is not None

        with self.step("7A.3 Configure ACO Allocator"):
            config = ACOConfig(
                lookback_window=args.lookback_window,
                rebalance_frequency=args.rebalance_freq,
                top_k=args.top_k,
                min_history=args.min_history,
                n_iterations=args.iterations,
                n_ants=args.ants,
                weight_buckets=args.weight_buckets,
                pheromone_power=args.pheromone_power,
                heuristic_power=args.heuristic_power,
                evaporation_rate=args.evaporation_rate,
                pheromone_deposit_scale=args.pheromone_deposit_scale,
                elite_ants=args.elite_ants,
                max_weight=args.max_weight,
                max_family_exposure=args.max_family_exposure,
                expected_return_weight=args.expected_return_weight,
                volatility_weight=args.volatility_weight,
                turnover_weight=args.turnover_weight,
                concentration_weight=args.concentration_weight,
                diversification_weight=args.diversification_weight,
                family_penalty_weight=args.family_penalty_weight,
                family_alpha_reward_weight=args.family_alpha_reward_weight,
                risk_budget_weight=args.risk_budget_weight,
                sparsity_penalty_weight=args.sparsity_penalty_weight,
                entropy_reward_weight=args.entropy_reward_weight,
                sharpe_weight=args.sharpe_weight,
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
            results["config"] = asdict(config)

        with self.step("7A.4 Run ACO Backtest"):
            backtester = ACOAllocatorBacktester(
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
            backtest = backtester.run(start_date=args.start_date, end_date=args.end_date)

            weights_dir = output_dir / "weights"
            backtests_dir = output_dir / "backtests"
            diagnostics_dir = output_dir / "diagnostics"
            reports_dir = output_dir / "reports"
            for directory in [weights_dir, backtests_dir, diagnostics_dir, reports_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            weights_path = weights_dir / "weights.parquet"
            returns_path = backtests_dir / "portfolio_returns.csv"
            benchmark_path = backtests_dir / "benchmark_returns.csv"
            diagnostics_path = diagnostics_dir / "optimization_diagnostics.csv"
            summary_path = reports_dir / "summary.json"
            comparison_path = reports_dir / "comparison.json"
            split_summary_path = reports_dir / "split_summary.json"

            backtest.weights.to_parquet(weights_path)
            backtest.portfolio_returns.to_csv(returns_path, header=True)
            if backtest.benchmark_returns is not None:
                backtest.benchmark_returns.to_csv(benchmark_path, header=True)
            if not backtest.diagnostics.empty:
                backtest.diagnostics.to_csv(diagnostics_path)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(backtest.summary, handle, indent=2, default=str)
            with open(comparison_path, "w", encoding="utf-8") as handle:
                json.dump(backtest.comparison, handle, indent=2, default=str)

            split_summary = {}
            if args.temporal_split == "auto":
                split_summary = self._build_temporal_split_summary(
                    portfolio_returns=backtest.portfolio_returns,
                    benchmark_returns=backtest.benchmark_returns,
                    evaluation_start=backtest.weights.index.min(),
                    train_ratio=args.train_ratio,
                    validation_ratio=args.validation_ratio,
                )
                with open(split_summary_path, "w", encoding="utf-8") as handle:
                    json.dump(split_summary, handle, indent=2, default=str)

            results["outputs"] = {
                "weights_path": str(weights_path),
                "portfolio_returns_path": str(returns_path),
                "summary_path": str(summary_path),
                "comparison_path": str(comparison_path),
            }
            if split_summary:
                results["outputs"]["split_summary_path"] = str(split_summary_path)
            results["summary"] = backtest.summary
            results["comparison"] = backtest.comparison
            results["split_summary"] = split_summary

        with self.step("7A.5 Generate Report"):
            self._generate_markdown_report(results, output_dir)
            self._print_result_summary(results)

        return results

    def _generate_markdown_report(self, results: dict, output_dir: Path):
        summary = results.get("summary", {})
        comparison = results.get("comparison", {})
        split_summary = results.get("split_summary", {})
        report = f"""# Phase 7A: Ant Colony Meta-Allocator Summary

**Generated**: {datetime.now().isoformat()}

## Overview

Phase 7A reuses the Phase 7 candidate-selection pipeline and replaces the
weight optimizer with an ant-colony search over discrete allocation buckets.

## Inputs Reused

- Returns shape: {results.get("artifacts", {}).get("returns_shape", "N/A")}
- Features available: {results.get("artifacts", {}).get("features_available", False)}
- Benchmark returns available: {results.get("artifacts", {}).get("benchmark_returns_available", False)}
- Family labels available: {results.get("artifacts", {}).get("family_labels_available", False)}
- Regime labels available: {results.get("artifacts", {}).get("regime_labels_available", False)}

## Performance Summary

| Metric | Value |
|--------|-------|
| Annualized Return | {summary.get("annualized_return", 0):.2%} |
| Annualized Volatility | {summary.get("annualized_volatility", 0):.2%} |
| Sharpe Ratio | {summary.get("sharpe_ratio", 0):.3f} |
| Max Drawdown | {summary.get("max_drawdown", 0):.2%} |
| Tracking Error | {summary.get("tracking_error", 0):.2%} |
| Information Ratio | {summary.get("information_ratio", 0):.3f} |
| Rebalances | {summary.get("n_rebalances", 0)} |
| Mean Candidates | {summary.get("mean_n_candidates", 0):.1f} |
| Mean Active Positions | {summary.get("mean_n_active", 0):.1f} |
| Valid Return Observations | {summary.get("valid_return_observations", 0)} |
| Objective | {summary.get("objective_name", "aco_sharpe_balanced")} |
| Device | {summary.get("device", "cpu")} |

## Risk Reference

| Metric | ACO | Benchmark Reference | Gap |
|--------|-----|---------------------|-----|
| Annualized Volatility | {summary.get("annualized_volatility", 0):.2%} | {summary.get("benchmark_annualized_volatility", 0):.2%} | {comparison.get("volatility_gap", 0):.2%} |
| Max Drawdown | {summary.get("max_drawdown", 0):.2%} | {summary.get("benchmark_max_drawdown", 0):.2%} | {comparison.get("drawdown_gap", 0):.2%} |
| Risk Alignment Score | {comparison.get("risk_alignment_score", 0):.3f} | n/a | n/a |

## Temporal Split

"""
        if split_summary:
            for split_name in ["train", "validation", "test"]:
                split = split_summary.get(split_name, {})
                report += (
                    f"- {split_name.title()}: "
                    f"{split.get('start_date', 'n/a')} -> {split.get('end_date', 'n/a')} | "
                    f"return={split.get('annualized_return', 0):.2%} | "
                    f"sharpe={split.get('sharpe_ratio', 0):.3f} | "
                    f"benchmark_vol={split.get('benchmark_annualized_volatility', 0):.2%} | "
                    f"benchmark_max_dd={split.get('benchmark_max_drawdown', 0):.2%}\n"
                )
        else:
            report += "- Temporal split evaluation disabled\n"

        report += """

## Output Files

```
outputs/swarm_allocator_aco/
|-- weights/weights.parquet
|-- backtests/portfolio_returns.csv
|-- diagnostics/optimization_diagnostics.csv
`-- reports/summary.json, comparison.json, split_summary.json
```
"""
        report_path = output_dir / "PHASE7A_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

    def _print_result_summary(self, results: dict):
        summary = results.get("summary", {})
        comparison = results.get("comparison", {})

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("PHASE 7A RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(
            "ACO: return=%s | vol=%s | sharpe=%.3f | max_dd=%s",
            f"{summary.get('annualized_return', 0):.2%}",
            f"{summary.get('annualized_volatility', 0):.2%}",
            summary.get("sharpe_ratio", 0.0),
            f"{summary.get('max_drawdown', 0):.2%}",
        )
        if comparison:
            self.logger.info(
                "Benchmark risk anchor: vol=%s | max_dd=%s",
                f"{summary.get('benchmark_annualized_volatility', 0):.2%}",
                f"{summary.get('benchmark_max_drawdown', 0):.2%}",
            )
            self.logger.info(
                "Risk alignment: vol_gap=%s | drawdown_gap=%s | score=%.3f",
                f"{comparison.get('volatility_gap', 0):.2%}",
                f"{comparison.get('drawdown_gap', 0):.2%}",
                comparison.get("risk_alignment_score", 0.0),
            )
        split_summary = results.get("split_summary", {})
        test_split = split_summary.get("test", {}) if split_summary else {}
        if test_split:
            self.logger.info(
                "Temporal test: %s -> %s | return=%s | sharpe=%.3f | benchmark_vol=%s",
                test_split.get("start_date", "n/a"),
                test_split.get("end_date", "n/a"),
                f"{test_split.get('annualized_return', 0):.2%}",
                test_split.get("sharpe_ratio", 0.0),
                f"{test_split.get('benchmark_annualized_volatility', 0):.2%}",
            )
        self.logger.info(
            "Artifacts: %s",
            results.get("outputs", {}).get("summary_path", "outputs/swarm_allocator_aco/reports/summary.json"),
        )
        self.logger.info("=" * 70)


def main():
    runner = Phase7ACORunner()
    runner.execute()


if __name__ == "__main__":
    main()
