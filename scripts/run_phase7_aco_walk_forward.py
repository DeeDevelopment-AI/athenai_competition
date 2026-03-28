#!/usr/bin/env python3
"""
=================================================================
PHASE 7A Walk-Forward: ACO Walk-Forward Evaluation
=================================================================
Run walk-forward evaluation using the best Phase 7A ACO configuration
found by tune_phase7_aco.py.

This script loads the best configuration from the tuning output and
evaluates it across multiple temporal folds for robust out-of-sample
performance estimation.

Usage:
  python scripts/run_phase7_aco_walk_forward.py
  python scripts/run_phase7_aco_walk_forward.py --config-path path/to/best_config.json
  python scripts/run_phase7_aco_walk_forward.py --max-folds 5 --expanding
  python scripts/run_phase7_aco_walk_forward.py --train-window 252 --test-window 63

Options:
  --config-path PATH            Path to best_config.json from factor search
                                (default: auto-discover from tuning outputs)
  --sample N                    Use only N algorithms (for faster testing)
  --input-dir PATH              Override Phase 1 processed directory
  --analysis-dir PATH           Override Phase 2 analysis directory
  --start-date YYYY-MM-DD       Start date for evaluation window
  --end-date YYYY-MM-DD         End date for evaluation window

Walk-Forward Configuration:
  --train-window N              Training window in days (default: 252)
  --validation-window N         Validation window in days (default: 63)
  --test-window N               Test window in days (default: 63)
  --step-size N                 Step size between folds in days (default: 63)
  --expanding                   Use expanding window instead of rolling
  --max-folds N                 Maximum number of folds to evaluate
  --cpu-only                    Force CPU execution even if CUDA is available

Output Files:
  walk_forward/folds.csv                 Per-fold metrics
  walk_forward/portfolio_test_returns.csv  Stitched test returns
  walk_forward/benchmark_test_returns.csv  Stitched benchmark returns
  walk_forward/summary.json              Summary statistics
  walk_forward/comparison.json           Benchmark comparison
  walk_forward/comparison.csv            Comparison table
  walk_forward/SUMMARY.md                Human-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
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
from src.swarm import ACOAllocatorBacktester, ACOConfig


class Phase7ACOWalkForwardRunner(PhaseRunner):
    phase_name = "Phase 7A Walk-Forward Evaluation"
    phase_number = 79

    def get_output_dir(self) -> Path:
        if self.args and self.args.output_dir:
            return Path(self.args.output_dir)
        return self.op.swarm_aco.root

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--config-path", type=str, default=None, help="Path to best_config.json from factor search")
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
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--end-date", type=str, default=None)
        parser.add_argument("--train-window", type=int, default=252)
        parser.add_argument("--validation-window", type=int, default=63)
        parser.add_argument("--test-window", type=int, default=63)
        parser.add_argument("--step-size", type=int, default=63)
        parser.add_argument("--expanding", action="store_true")
        parser.add_argument("--max-folds", type=int, default=None)
        parser.add_argument("--cpu-only", action="store_true")

    def run(self, args: argparse.Namespace) -> dict:
        phase7 = Phase7Runner()
        input_dir = Path(args.input_dir) if args.input_dir else phase7.dp.processed.root
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else phase7.dp.processed.analysis.root
        output_dir = self.get_output_dir()

        with self.step("79.1 Load Selected Configuration"):
            config_payload = self._load_config_payload(output_dir, args.config_path)
            selection_factor = config_payload.get("selection_factor")
            if not selection_factor:
                raise ValueError("Config payload does not contain selection_factor")
            config = self._build_aco_config(config_payload)

        with self.step("79.2 Load Shared Artifacts"):
            sampled_cols = phase7._resolve_sampled_algorithms(input_dir, args.sample)
            algo_returns = phase7._load_returns(input_dir, sampled_cols=sampled_cols)
            features = phase7._load_optional_features(
                input_dir,
                sampled_algorithms=sampled_cols,
                extra_suffixes=self._selection_factor_suffixes(selection_factor),
            )
            benchmark_returns = phase7._load_optional_benchmark_returns(input_dir)
            benchmark_weights = phase7._load_optional_benchmark_weights(input_dir, sampled_cols=sampled_cols)
            if args.phase2_cluster_filter:
                algo_returns, features, benchmark_weights, _ = phase7._apply_phase2_cluster_universe(
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
            family_labels = phase7._load_optional_family_labels(analysis_dir, algo_returns.columns, allow_default_fallback=True)
            family_alpha_scores = phase7._load_optional_family_alpha_scores(
                analysis_dir,
                family_labels,
                allow_default_fallback=True,
            )
            cluster_history = phase7._load_optional_cluster_history(analysis_dir, allow_default_fallback=True)
            cluster_stability = phase7._load_optional_cluster_stability(analysis_dir, allow_default_fallback=True)
            cluster_alpha_scores = phase7._load_optional_cluster_alpha_scores(
                analysis_dir,
                cluster_history,
                allow_default_fallback=True,
            )
            regime_labels = phase7._load_optional_regime_labels(analysis_dir, allow_default_fallback=True)

        with self.step("79.3 Generate Walk-Forward Folds"):
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

        with self.step("79.4 Run Walk-Forward Backtests"):
            fold_rows = []
            allocation_rows = []
            stitched_portfolio = []
            stitched_benchmark = []
            for fold in folds:
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
                    selection_factor=selection_factor,
                )
                result = backtester.run(
                    start_date=str(fold["train_start"].date()),
                    end_date=str(fold["test_end"].date()),
                )
                train_returns = result.portfolio_returns.loc[fold["train_start"] : fold["train_end"]]
                val_returns = result.portfolio_returns.loc[fold["val_start"] : fold["val_end"]]
                test_returns = result.portfolio_returns.loc[fold["test_start"] : fold["test_end"]]
                benchmark_train = result.benchmark_returns.loc[fold["train_start"] : fold["train_end"]] if result.benchmark_returns is not None else None
                benchmark_val = result.benchmark_returns.loc[fold["val_start"] : fold["val_end"]] if result.benchmark_returns is not None else None
                benchmark_test = result.benchmark_returns.loc[fold["test_start"] : fold["test_end"]] if result.benchmark_returns is not None else None

                train_metrics = compute_full_metrics(train_returns, benchmark_train)
                val_metrics = compute_full_metrics(val_returns, benchmark_val)
                test_metrics = compute_full_metrics(test_returns, benchmark_test)
                fold_comparison = self._build_benchmark_comparison(test_returns, benchmark_test)
                fold_rows.append(
                    {
                        "fold_id": fold["fold_id"],
                        "train_start": str(fold["train_start"].date()),
                        "train_end": str(fold["train_end"].date()),
                        "val_start": str(fold["val_start"].date()),
                        "val_end": str(fold["val_end"].date()),
                        "test_start": str(fold["test_start"].date()),
                        "test_end": str(fold["test_end"].date()),
                        "selection_factor": selection_factor,
                        "train_sharpe_ratio": float(train_metrics.get("sharpe_ratio", 0.0)),
                        "validation_sharpe_ratio": float(val_metrics.get("sharpe_ratio", 0.0)),
                        "test_sharpe_ratio": float(test_metrics.get("sharpe_ratio", 0.0)),
                        "test_annualized_return": float(test_metrics.get("annualized_return", 0.0)),
                        "test_benchmark_annualized_volatility": float(test_metrics.get("benchmark_annualized_volatility", 0.0)),
                        "test_max_drawdown": float(test_metrics.get("max_drawdown", 0.0)),
                        "test_portfolio_total_return": float(fold_comparison.get("portfolio_total_return", 0.0)),
                        "test_benchmark_max_drawdown": float(fold_comparison.get("benchmark_max_drawdown", 0.0)),
                        "test_volatility_gap": float(fold_comparison.get("volatility_gap", 0.0)),
                        "test_drawdown_gap": float(fold_comparison.get("drawdown_gap", 0.0)),
                        "test_risk_alignment_score": float(fold_comparison.get("risk_alignment_score", 0.0)),
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
                            "strategy": "aco_walk_forward",
                            "fold_id": int(fold["fold_id"]),
                            "selection_factor": selection_factor,
                        },
                    )
                )
                stitched_portfolio.append(test_returns)
                if benchmark_test is not None:
                    stitched_benchmark.append(benchmark_test)

            stitched_portfolio_returns = self._stitch_returns(stitched_portfolio)
            stitched_benchmark_returns = self._stitch_returns(stitched_benchmark) if stitched_benchmark else None

        with self.step("79.5 Build Risk Reference Summary"):
            comparison_table = pd.DataFrame()
            detailed_table = pd.DataFrame()
            benchmark_comparison = {}
            if stitched_benchmark_returns is not None and not stitched_benchmark_returns.empty:
                comparison = StrategyComparison(stitched_benchmark_returns)
                comparison.add_strategy("aco_walk_forward", stitched_portfolio_returns)
                comparison_table = comparison.get_comparison_table()
                detailed_table = comparison.get_detailed_comparison()
            overall_metrics = compute_full_metrics(stitched_portfolio_returns, stitched_benchmark_returns)
            benchmark_comparison = self._build_benchmark_comparison(
                stitched_portfolio_returns,
                stitched_benchmark_returns,
            )

        wf_dir = output_dir / "walk_forward"
        wf_dir.mkdir(parents=True, exist_ok=True)
        folds_path = wf_dir / "folds.csv"
        portfolio_path = wf_dir / "portfolio_test_returns.csv"
        benchmark_path = wf_dir / "benchmark_test_returns.csv"
        summary_path = wf_dir / "summary.json"
        comparison_json_path = wf_dir / "comparison.json"
        comparison_path = wf_dir / "comparison.csv"
        detailed_path = wf_dir / "comparison_detailed.csv"
        report_path = wf_dir / "SUMMARY.md"

        folds_df = pd.DataFrame(fold_rows)
        folds_df.to_csv(folds_path, index=False)
        if allocation_rows:
            pd.DataFrame(allocation_rows).to_csv(wf_dir / "rebalance_allocations.csv", index=False)
        stitched_portfolio_returns.to_csv(portfolio_path, header=True)
        if stitched_benchmark_returns is not None and not stitched_benchmark_returns.empty:
            stitched_benchmark_returns.to_csv(benchmark_path, header=True)
        if not comparison_table.empty:
            comparison_table.to_csv(comparison_path)
        if not detailed_table.empty:
            detailed_table.to_csv(detailed_path)

        summary = {
            "config_path": str(args.config_path or (output_dir / "tuning" / "factor_search" / "best_config.json")),
            "selection_factor": selection_factor,
            "n_folds": len(fold_rows),
            "overall_metrics": overall_metrics,
            "risk_reference": benchmark_comparison,
            "rebalance_allocations_path": str(wf_dir / "rebalance_allocations.csv") if allocation_rows else None,
            "mean_test_sharpe": float(folds_df["test_sharpe_ratio"].mean()) if not folds_df.empty else 0.0,
            "mean_test_return": float(folds_df["test_annualized_return"].mean()) if not folds_df.empty else 0.0,
            "mean_test_benchmark_volatility": float(folds_df["test_benchmark_annualized_volatility"].mean()) if not folds_df.empty else 0.0,
            "mean_test_risk_alignment_score": float(folds_df["test_risk_alignment_score"].mean()) if not folds_df.empty else 0.0,
            "mean_test_max_drawdown": float(folds_df["test_max_drawdown"].mean()) if not folds_df.empty else 0.0,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        with open(comparison_json_path, "w", encoding="utf-8") as handle:
            json.dump(benchmark_comparison, handle, indent=2, default=str)
        report_path.write_text(self._build_report(summary, folds_df, comparison_table), encoding="utf-8")
        self._print_summary(summary, comparison_table)

        return {
            "n_folds": len(fold_rows),
            "selection_factor": selection_factor,
            "folds_path": str(folds_path),
            "summary_path": str(summary_path),
            "comparison_json_path": str(comparison_json_path),
            "comparison_path": str(comparison_path),
            "report_path": str(report_path),
        }

    def _load_config_payload(self, output_dir: Path, config_path: str | None) -> dict:
        path = self._resolve_config_path(output_dir, config_path)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_config_path(self, output_dir: Path, config_path: str | None) -> Path:
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Best config not found: {path}")
            return path

        candidate_paths = [
            output_dir / "tuning" / "factor_search" / "best_config.json",
            self.op.root / "phase78" / "tuning" / "factor_search" / "best_config.json",
            self.op.root / "swarm_allocator_aco" / "tuning" / "factor_search" / "best_config.json",
            self.op.root / "phase7_aco" / "tuning" / "factor_search" / "best_config.json",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        matches = sorted(
            self.op.root.glob("**/tuning/factor_search/best_config.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]

        searched = "\n".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            "Best config not found. Checked:\n"
            f"{searched}\n"
            "You can also pass --config-path explicitly."
        )

    def _build_aco_config(self, payload: dict) -> ACOConfig:
        valid_fields = {field.name for field in fields(ACOConfig)}
        config_kwargs = {key: value for key, value in payload.items() if key in valid_fields}
        config_kwargs["use_gpu"] = not getattr(self.args, "cpu_only", False)
        return ACOConfig(**config_kwargs)

    def _selection_factor_suffixes(self, selection_factor: str | list[str]) -> set[str]:
        if isinstance(selection_factor, str):
            return {selection_factor}
        return {str(factor) for factor in selection_factor if str(factor)}

    def _stitch_returns(self, pieces: list[pd.Series]) -> pd.Series:
        if not pieces:
            return pd.Series(dtype=float)
        stitched = pd.concat(pieces).sort_index()
        if stitched.index.has_duplicates:
            stitched = stitched.groupby(level=0).mean()
        return stitched

    def _build_benchmark_comparison(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series | None,
    ) -> dict:
        comparison = {
            "portfolio_total_return": float((1.0 + portfolio_returns).prod() - 1.0),
        }
        if benchmark_returns is None or benchmark_returns.empty:
            return comparison

        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").fillna(0.0)
        aligned.columns = ["portfolio", "benchmark"]
        benchmark_metrics = compute_full_metrics(aligned["benchmark"])
        portfolio_total = float((1.0 + aligned["portfolio"]).prod() - 1.0)
        benchmark_total = float((1.0 + aligned["benchmark"]).prod() - 1.0)
        comparison.update(
            {
                "benchmark_total_return": benchmark_total,
                "benchmark_annualized_volatility": float(benchmark_metrics.get("annualized_volatility", 0.0)),
                "benchmark_max_drawdown": float(benchmark_metrics.get("max_drawdown", 0.0)),
            }
        )
        portfolio_metrics = compute_full_metrics(aligned["portfolio"])
        volatility_gap = float(
            portfolio_metrics.get("annualized_volatility", 0.0)
            - benchmark_metrics.get("annualized_volatility", 0.0)
        )
        drawdown_gap = float(
            abs(portfolio_metrics.get("max_drawdown", 0.0))
            - abs(benchmark_metrics.get("max_drawdown", 0.0))
        )
        comparison.update(
            {
                "volatility_gap": volatility_gap,
                "drawdown_gap": drawdown_gap,
                "risk_alignment_score": float(1.0 / (1.0 + abs(volatility_gap) + abs(drawdown_gap))),
            }
        )
        return comparison

    def _build_report(self, summary: dict, folds_df: pd.DataFrame, comparison_table: pd.DataFrame) -> str:
        overall = summary.get("overall_metrics", {})
        benchmark = summary.get("risk_reference", {})
        lines = [
            "# Phase 7A Walk-Forward Summary",
            "",
            f"Selection factor: {self._format_selection_factor(summary['selection_factor'])}",
            f"Folds: {summary['n_folds']}",
            f"Mean test Sharpe: {summary['mean_test_sharpe']:.3f}",
            f"Mean test annualized return: {summary['mean_test_return']:.2%}",
            f"Mean benchmark volatility reference: {summary['mean_test_benchmark_volatility']:.2%}",
            f"Mean risk alignment score: {summary['mean_test_risk_alignment_score']:.3f}",
            f"Mean test max drawdown: {summary['mean_test_max_drawdown']:.2%}",
            "",
            "## Overall Out-of-Sample Metrics",
            "",
        ]
        metric_rows = [
            ("Annualized Return", overall.get("annualized_return", 0.0), "pct"),
            ("Annualized Volatility", overall.get("annualized_volatility", 0.0), "pct"),
            ("Sharpe Ratio", overall.get("sharpe_ratio", 0.0), "num"),
            ("Sortino Ratio", overall.get("sortino_ratio", 0.0), "num"),
            ("Calmar Ratio", overall.get("calmar_ratio", 0.0), "num"),
            ("Max Drawdown", overall.get("max_drawdown", 0.0), "pct"),
        ]
        lines.extend(
            [
                "| Metric | Value |",
                "|--------|-------|",
            ]
        )
        for label, value, fmt in metric_rows:
            lines.append(f"| {label} | {self._format_metric_value(value, fmt)} |")
        if benchmark:
            lines.extend(
                [
                    "",
                    "## Risk Reference",
                    "",
                    "| Metric | ACO Walk-Forward | Benchmark Reference | Gap |",
                    "|--------|------------------|---------------------|-----|",
                    (
                        f"| Annualized Volatility | {self._format_metric_value(overall.get('annualized_volatility', 0.0), 'pct')} | "
                        f"{self._format_metric_value(benchmark.get('benchmark_annualized_volatility', 0.0), 'pct')} | "
                        f"{self._format_metric_value(benchmark.get('volatility_gap', 0.0), 'pct')} |"
                    ),
                    (
                        f"| Max Drawdown | {self._format_metric_value(overall.get('max_drawdown', 0.0), 'pct')} | "
                        f"{self._format_metric_value(benchmark.get('benchmark_max_drawdown', 0.0), 'pct')} | "
                        f"{self._format_metric_value(benchmark.get('drawdown_gap', 0.0), 'pct')} |"
                    ),
                    f"| Risk Alignment Score | {self._format_metric_value(benchmark.get('risk_alignment_score', 0.0), 'num')} | - | - |",
                ]
            )
        if not folds_df.empty:
            lines.extend(
                [
                    "",
                    "## Fold Results",
                    "",
                    folds_df.head(10).to_markdown(index=False),
                ]
            )
        if not comparison_table.empty:
            lines.extend(
                [
                    "",
                    "## Comparison Table",
                    "",
                    comparison_table.to_markdown(),
                ]
            )
        lines.append("")
        return "\n".join(lines)

    def _format_metric_value(self, value: float | int | None, fmt: str) -> str:
        if value is None:
            return "n/a"
        numeric = float(value)
        if fmt == "pct":
            return f"{numeric:.2%}"
        return f"{numeric:.3f}"

    def _print_summary(self, summary: dict, comparison_table: pd.DataFrame):
        overall = summary.get("overall_metrics", {})
        benchmark = summary.get("risk_reference", {})

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("PHASE 7A WALK-FORWARD RESULTS")
        self.logger.info("=" * 80)
        self.logger.info(
            "Selection factor: %s",
            self._format_selection_factor(summary.get("selection_factor", "")),
        )
        self.logger.info(
            "Out-of-sample: return=%s | vol=%s | sharpe=%.3f | max_dd=%s",
            self._format_metric_value(overall.get("annualized_return", 0.0), "pct"),
            self._format_metric_value(overall.get("annualized_volatility", 0.0), "pct"),
            float(overall.get("sharpe_ratio", 0.0)),
            self._format_metric_value(overall.get("max_drawdown", 0.0), "pct"),
        )
        if benchmark:
            self.logger.info(
                "Benchmark risk anchor: vol=%s | max_dd=%s | score=%.3f",
                self._format_metric_value(benchmark.get("benchmark_annualized_volatility", 0.0), "pct"),
                self._format_metric_value(benchmark.get("benchmark_max_drawdown", 0.0), "pct"),
                float(benchmark.get("risk_alignment_score", 0.0)),
            )
        if not comparison_table.empty:
            self.logger.info("Comparison table saved.")
        self.logger.info("=" * 80)

    def _format_selection_factor(self, selection_factor: str | list[str]) -> str:
        if isinstance(selection_factor, str):
            return selection_factor
        return " + ".join(str(factor) for factor in selection_factor)


def main():
    runner = Phase7ACOWalkForwardRunner()
    runner.execute()


if __name__ == "__main__":
    main()
