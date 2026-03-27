#!/usr/bin/env python3
"""
=================================================================
PHASE 7: Swarm Meta-Allocator
=================================================================
Build a swarm-based meta allocator on top of prior phase artifacts.

This phase reuses:
  - Phase 1 returns/features and benchmark artifacts
  - Optional Phase 2 family/regime outputs

Usage:
  python scripts/run_phase7.py
  python scripts/run_phase7.py --sample 100 --top-k 32 --particles 64
  python scripts/run_phase7.py --start-date 2021-01-01 --end-date 2023-12-31
  python scripts/run_phase7.py --cpu-only --rebalance-freq monthly

Options:
  --sample N                    Use only N algorithms (for faster testing)
  --top-k N                     Candidate pool size passed to the optimizer (default: 256)
  --lookback-window N           Lookback window in days (default: 126)
  --min-history N               Minimum history required in days (default: 63)
  --particles N                 Number of PSO swarm particles (default: 128)
  --iterations N                Number of PSO iterations (default: 80)
  --inertia F                   PSO inertia term (default: 0.70)
  --cognitive-weight F          PSO cognitive coefficient (default: 1.40)
  --social-weight F             PSO social coefficient (default: 1.30)
  --rebalance-freq F            Rebalance frequency: daily, weekly, monthly (default: weekly)
  --selection-factor S          Feature for candidate selection (default: rolling_sharpe_21d)
  --start-date YYYY-MM-DD       Start date for backtest
  --end-date YYYY-MM-DD         End date for backtest

Constraint Options:
  --max-weight F                Max weight per algorithm (default: 0.40)
  --max-family-exposure F       Max family exposure (default: 0.30)
  --min-active-weight F         Min weight to count as active (default: 0.0025)
  --min-gross-exposure F        Min total portfolio exposure (default: 0.85)
  --target-portfolio-vol F      Target annualized volatility (default: 0.16)

Objective Weight Options:
  --expected-return-weight F    Weight on expected return in objective (default: 1.00)
  --volatility-weight F         Weight on volatility penalty (default: 0.50)
  --tracking-error-weight F     Weight on tracking error penalty (default: 0.35)
  --turnover-weight F           Weight on turnover penalty (default: 0.15)
  --concentration-weight F      Weight on concentration penalty (default: 0.10)
  --diversification-weight F    Weight on diversification reward (default: 0.10)
  --family-penalty-weight F     Weight on family concentration penalty (default: 0.20)
  --risk-budget-weight F        Weight on risk budget deviation (default: 0.30)
  --sparsity-penalty-weight F   Weight on sparsity penalty (default: 0.01)
  --under-investment-penalty-weight F  Penalty for under-investment (default: 0.35)
  --regime-focus F              Regime focus multiplier (default: 1.50)

Advanced Options:
  --objective-name S            Name of the objective function (default: alpha_risk_balanced)
  --selection-mode S            Universe selection: legacy, benchmark_aware (default: legacy)
  --normalize-objective-metrics Normalize objective components cross-sectionally
  --temporal-split S            Temporal eval split: none, auto (default: auto)
  --train-ratio F               Train ratio for temporal split (default: 0.60)
  --validation-ratio F          Validation ratio for temporal split (default: 0.20)
  --seed N                      Random seed (default: 42)
  --cpu-only                    Disable GPU usage
  --input-dir PATH              Override Phase 1 processed directory
  --analysis-dir PATH           Override Phase 2 analysis directory
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner
from src.evaluation.metrics import compute_full_metrics
from src.swarm import SwarmAllocatorBacktester, SwarmConfig


def _normalize_cluster_key(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            numeric = float(stripped)
            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            return stripped
        return stripped
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    return str(value).strip()


class Phase7Runner(PhaseRunner):
    """Phase 7: Swarm-based meta allocation."""

    phase_name = "Phase 7: Swarm Meta-Allocator"
    phase_number = 7

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--sample", type=int, default=None, help="Use only N algorithms")
        parser.add_argument("--top-k", type=int, default=256, help="Candidate pool size passed to the optimizer")
        parser.add_argument("--lookback-window", type=int, default=126, help="Lookback window in days")
        parser.add_argument("--min-history", type=int, default=63, help="Minimum history required")
        parser.add_argument("--particles", type=int, default=128, help="Number of swarm particles")
        parser.add_argument("--iterations", type=int, default=80, help="PSO iterations")
        parser.add_argument("--inertia", type=float, default=0.70, help="PSO inertia term")
        parser.add_argument("--cognitive-weight", type=float, default=1.40, help="PSO cognitive coefficient")
        parser.add_argument("--social-weight", type=float, default=1.30, help="PSO social coefficient")
        parser.add_argument(
            "--rebalance-freq",
            type=str,
            default="weekly",
            choices=["daily", "weekly", "monthly"],
            help="Rebalance frequency",
        )
        parser.add_argument("--selection-factor", type=str, default="rolling_sharpe_21d")
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--end-date", type=str, default=None)
        parser.add_argument("--max-weight", type=float, default=0.40)
        parser.add_argument("--max-family-exposure", type=float, default=0.30)
        parser.add_argument("--expected-return-weight", type=float, default=1.00)
        parser.add_argument("--excess-return-weight", type=float, default=0.0)
        parser.add_argument("--information-ratio-weight", type=float, default=0.0)
        parser.add_argument("--benchmark-hit-rate-weight", type=float, default=0.0)
        parser.add_argument("--downside-excess-weight", type=float, default=0.0)
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
        parser.add_argument("--objective-name", type=str, default="alpha_risk_balanced")
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
            help="Normalize objective components cross-sectionally inside the swarm",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--cpu-only", action="store_true", help="Disable GPU usage")
        parser.add_argument("--input-dir", type=str, default=None, help="Override Phase 1 processed directory")
        parser.add_argument("--analysis-dir", type=str, default=None, help="Override Phase 2 analysis directory")
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
        output_dir = self.get_output_dir()

        with self.step("7.1 Load Phase 1 Artifacts"):
            sampled_cols = self._resolve_sampled_algorithms(input_dir, args.sample)
            algo_returns = self._load_returns(input_dir, sampled_cols=sampled_cols)
            features = self._load_optional_features(input_dir, sampled_algorithms=sampled_cols)
            benchmark_returns = self._load_optional_benchmark_returns(input_dir)
            benchmark_weights = self._load_optional_benchmark_weights(input_dir, sampled_cols=sampled_cols)

            results["artifacts"]["returns_shape"] = list(algo_returns.shape)
            results["artifacts"]["features_available"] = features is not None
            results["artifacts"]["benchmark_returns_available"] = benchmark_returns is not None
            results["artifacts"]["benchmark_weights_available"] = benchmark_weights is not None

        with self.step("7.2 Load Optional Phase 2 Artifacts"):
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

        with self.step("7.3 Configure Swarm Allocator"):
            config = SwarmConfig(
                lookback_window=args.lookback_window,
                rebalance_frequency=args.rebalance_freq,
                top_k=args.top_k,
                min_history=args.min_history,
                n_particles=args.particles,
                n_iterations=args.iterations,
                inertia=args.inertia,
                cognitive_weight=args.cognitive_weight,
                social_weight=args.social_weight,
                max_weight=args.max_weight,
                max_family_exposure=args.max_family_exposure,
                expected_return_weight=args.expected_return_weight,
                excess_return_weight=args.excess_return_weight,
                information_ratio_weight=args.information_ratio_weight,
                benchmark_hit_rate_weight=args.benchmark_hit_rate_weight,
                downside_excess_weight=args.downside_excess_weight,
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
            results["config"] = asdict(config)

        with self.step("7.4 Run Swarm Backtest"):
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

        with self.step("7.5 Generate Report"):
            self._generate_markdown_report(results, output_dir)
            self._print_result_summary(results)

        return results

    def _resolve_sampled_algorithms(self, input_dir: Path, sample: int | None) -> list[str] | None:
        if sample is None:
            return None
        path = self.dp.algorithms.returns
        if not path.exists():
            path = input_dir / "algo_returns.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Returns matrix not found: {path}")
        columns = self._peek_parquet_columns(path)
        if sample >= len(columns):
            return None
        return columns[:sample]

    def _peek_parquet_columns(self, path: Path) -> list[str]:
        return list(pq.ParquetFile(path).schema.names)

    def _load_returns(self, input_dir: Path, sampled_cols: list[str] | None = None) -> pd.DataFrame:
        path = self.dp.algorithms.returns
        if not path.exists():
            path = input_dir / "algo_returns.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Returns matrix not found: {path}")
        read_columns = sampled_cols if sampled_cols else None
        returns = pd.read_parquet(path, columns=read_columns).astype(np.float32, copy=False)
        if getattr(returns.index, "tz", None) is not None:
            returns.index = returns.index.tz_localize(None)
        return returns

    def _feature_columns_for_algorithms(
        self,
        path: Path,
        sampled_algorithms: list[str] | None,
        extra_suffixes: set[str] | None = None,
    ) -> list[str] | None:
        columns = self._peek_parquet_columns(path)
        required_suffixes = self._required_feature_suffixes(extra_suffixes=extra_suffixes)
        prefixes = tuple(f"{algo}_" for algo in sampled_algorithms) if sampled_algorithms else None
        selected = []
        for column in columns:
            if column.startswith("rolling_market"):
                selected.append(column)
                continue
            if prefixes is not None and not column.startswith(prefixes):
                continue
            if any(column.endswith(f"_{suffix}") for suffix in required_suffixes):
                selected.append(column)
        return selected or None

    def _required_feature_suffixes(self, extra_suffixes: set[str] | None = None) -> set[str]:
        suffixes = {
            "rolling_sharpe_5d",
            "rolling_sharpe_21d",
            "rolling_sharpe_63d",
            "rolling_return_21d",
            "rolling_return_63d",
            "rolling_volatility_21d",
            "rolling_volatility_63d",
            "rolling_profit_factor_21d",
            "rolling_profit_factor_63d",
            "rolling_calmar_21d",
            "rolling_calmar_63d",
            "rolling_drawdown_21d",
            "rolling_drawdown_63d",
            "volatility",
            "max_drawdown",
            "drawdown",
        }
        selection_factor = getattr(self.args, "selection_factor", None)
        if selection_factor:
            if isinstance(selection_factor, str):
                suffixes.add(selection_factor)
            else:
                suffixes.update(str(factor) for factor in selection_factor if str(factor))
        if extra_suffixes:
            suffixes.update(extra_suffixes)
        return suffixes

    def _load_optional_features(
        self,
        input_dir: Path,
        sampled_algorithms: list[str] | None = None,
        extra_suffixes: set[str] | None = None,
    ) -> pd.DataFrame | None:
        path = self.dp.algorithms.features
        if not path.exists():
            path = input_dir / "algo_features.parquet"
        if not path.exists():
            return None
        read_columns = self._feature_columns_for_algorithms(path, sampled_algorithms, extra_suffixes=extra_suffixes)
        features = pd.read_parquet(path, columns=read_columns)
        if getattr(features.index, "tz", None) is not None:
            features.index = features.index.tz_localize(None)
        return features

    def _load_optional_benchmark_returns(self, input_dir: Path) -> pd.Series | None:
        path = self.dp.benchmark.daily_returns
        if not path.exists():
            path = input_dir / "benchmark_daily_returns.csv"
        if not path.exists():
            return None
        bench_df = pd.read_csv(path, index_col=0, parse_dates=True)
        series = bench_df.iloc[:, 0]
        if getattr(series.index, "tz", None) is not None:
            series.index = series.index.tz_localize(None)
        return series.astype(np.float32)

    def _load_optional_benchmark_weights(
        self,
        input_dir: Path,
        sampled_cols: list[str] | None = None,
    ) -> pd.DataFrame | None:
        path = self.dp.benchmark.weights
        if not path.exists():
            path = input_dir / "benchmark_weights.parquet"
        if not path.exists():
            return None
        read_columns = sampled_cols if sampled_cols else None
        weights = pd.read_parquet(path, columns=read_columns).astype(np.float32, copy=False)
        if getattr(weights.index, "tz", None) is not None:
            weights.index = weights.index.tz_localize(None)
        return weights

    def _load_optional_family_labels(
        self,
        analysis_dir: Path,
        algo_columns: pd.Index,
        allow_default_fallback: bool = True,
    ) -> pd.Series | None:
        path = analysis_dir / "clustering" / "behavioral" / "family_labels.csv"
        if not path.exists() and allow_default_fallback:
            path = self.dp.processed.analysis.family_labels
        if not path.exists():
            return None
        family_df = pd.read_csv(path, index_col=0)
        series = family_df.iloc[:, 0]
        series = series.reindex(algo_columns)
        return series

    def _load_optional_regime_labels(
        self,
        analysis_dir: Path,
        allow_default_fallback: bool = True,
    ) -> pd.Series | None:
        path = analysis_dir / "regimes" / "labels.csv"
        if not path.exists() and allow_default_fallback:
            path = self.dp.processed.analysis.regime_labels
        if not path.exists():
            return None
        labels = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        if getattr(labels.index, "tz", None) is not None:
            labels.index = labels.index.tz_localize(None)
        return labels

    def _load_optional_cluster_history(
        self,
        analysis_dir: Path,
        allow_default_fallback: bool = True,
    ) -> pd.DataFrame | None:
        path = analysis_dir / "clustering" / "temporal" / "cluster_history.csv"
        if not path.exists() and allow_default_fallback:
            path = self.dp.processed.analysis.root / "clustering" / "temporal" / "cluster_history.csv"
        if not path.exists():
            return None
        history = pd.read_csv(path, parse_dates=["week_end"])
        return history

    def _load_optional_cluster_stability(
        self,
        analysis_dir: Path,
        allow_default_fallback: bool = True,
    ) -> pd.DataFrame | None:
        path = analysis_dir / "clustering" / "temporal" / "cluster_stability.csv"
        if not path.exists() and allow_default_fallback:
            path = self.dp.processed.analysis.root / "clustering" / "temporal" / "cluster_stability.csv"
        if not path.exists():
            return None
        stability = pd.read_csv(path)
        if "algo_id" in stability.columns:
            stability = stability.set_index("algo_id")
        return stability

    def _load_optional_family_alpha_scores(
        self,
        analysis_dir: Path,
        family_labels: pd.Series | None,
        allow_default_fallback: bool = True,
    ) -> dict[str, float]:
        if family_labels is None or family_labels.empty:
            return {}

        path = analysis_dir / "profiles" / "summary.csv"
        if not path.exists() and allow_default_fallback:
            path = self.dp.processed.analysis.root / "profiles" / "summary.csv"
        if not path.exists():
            return {}

        profile_df = pd.read_csv(path)
        if "algo_id" not in profile_df.columns:
            return {}

        merged = profile_df.merge(
            family_labels.rename("family"),
            left_on="algo_id",
            right_index=True,
            how="inner",
        )
        if merged.empty:
            return {}

        family_summary = merged.groupby("family").agg(
            ann_return=("ann_return", "mean"),
            ann_volatility=("ann_volatility", "mean"),
            sharpe=("sharpe", "mean"),
        )
        if family_summary.empty:
            return {}

        family_summary["quality"] = (
            0.65 * family_summary["sharpe"]
            + 0.35 * family_summary["ann_return"] / family_summary["ann_volatility"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        ranked = family_summary.sort_values(["quality", "ann_return"], ascending=[False, False])
        scores: dict[str, float] = {}
        if ranked.empty:
            return scores

        top_family = str(ranked.index[0])
        scores[top_family] = 1.0
        if len(ranked) > 1:
            second_family = str(ranked.index[1])
            scores[second_family] = 0.6
        for family, row in ranked.iloc[2:].iterrows():
            if row["ann_return"] <= 0 or row["sharpe"] <= 0:
                scores[str(family)] = -0.75
            else:
                scores[str(family)] = -0.25
        return scores

    def _load_optional_cluster_alpha_scores(
        self,
        analysis_dir: Path,
        cluster_history: pd.DataFrame | None,
        allow_default_fallback: bool = True,
    ) -> dict[str, dict[str, float]]:
        if cluster_history is None or cluster_history.empty:
            return {}

        path = analysis_dir / "profiles" / "summary.csv"
        if not path.exists() and allow_default_fallback:
            path = self.dp.processed.analysis.root / "profiles" / "summary.csv"
        if not path.exists():
            return {}

        profile_df = pd.read_csv(path)
        if "algo_id" not in profile_df.columns:
            return {}

        cluster_history = cluster_history.copy()
        cluster_history["week_end"] = pd.to_datetime(cluster_history["week_end"])
        latest_week = cluster_history["week_end"].max()
        horizons = ["cluster_cumulative", "cluster_weekly", "cluster_monthly"]
        score_maps: dict[str, dict[str, float]] = {}

        for horizon in horizons:
            horizon_df = cluster_history[["week_end", "algo_id", horizon]].rename(columns={horizon: "cluster"})
            horizon_df["cluster"] = horizon_df["cluster"].map(_normalize_cluster_key)
            horizon_df = horizon_df[~horizon_df["cluster"].isin(["", "inactive", "insufficient_data", "error"])].copy()
            if horizon_df.empty:
                continue

            # Weight recent assignments higher while still using the whole temporal history.
            age_days = (latest_week - horizon_df["week_end"]).dt.days.clip(lower=0)
            horizon_df["recency_weight"] = np.exp(-age_days / 90.0)

            merged = profile_df.merge(horizon_df, on="algo_id", how="inner")
            merged["cluster"] = merged["cluster"].map(_normalize_cluster_key)
            merged = merged[~merged["cluster"].isin(["", "inactive", "insufficient_data", "error"])].copy()
            if merged.empty:
                continue

            weighted_rows = []
            for cluster_id, group in merged.groupby("cluster"):
                weights = group["recency_weight"].to_numpy(dtype=np.float64, copy=False)
                weight_sum = float(weights.sum())
                if weight_sum <= 0:
                    continue
                ann_return = float(np.average(group["ann_return"], weights=weights))
                ann_volatility = float(np.average(group["ann_volatility"], weights=weights))
                sharpe = float(np.average(group["sharpe"], weights=weights))
                weighted_rows.append(
                    {
                        "cluster": str(cluster_id),
                        "ann_return": ann_return,
                        "ann_volatility": ann_volatility,
                        "sharpe": sharpe,
                        "weight_sum": weight_sum,
                    }
                )

            cluster_summary = pd.DataFrame(weighted_rows).set_index("cluster") if weighted_rows else pd.DataFrame()
            if cluster_summary.empty or "sharpe" not in cluster_summary.columns:
                continue

            cluster_summary["quality"] = (
                0.80 * cluster_summary["sharpe"]
                + 0.20 * (
                    cluster_summary["ann_return"] / cluster_summary["ann_volatility"].replace(0, np.nan)
                )
            ).replace([np.inf, -np.inf], np.nan).fillna(-1.0)
            cluster_summary["quality"] = cluster_summary["quality"] * np.log1p(cluster_summary["weight_sum"])

            ranked = cluster_summary.sort_values(["quality", "ann_return"], ascending=[False, False])
            horizon_scores: dict[str, float] = {}
            if not ranked.empty:
                best_cluster = _normalize_cluster_key(ranked.index[0])
                horizon_scores[best_cluster] = 1.0
            if len(ranked) > 1:
                second_cluster = _normalize_cluster_key(ranked.index[1])
                second_row = ranked.iloc[1]
                horizon_scores[second_cluster] = 0.6 if second_row["quality"] > 0 else 0.2
            for cluster_id, row in ranked.iloc[2:].iterrows():
                normalized_cluster = _normalize_cluster_key(cluster_id)
                if row["quality"] > 0:
                    horizon_scores[normalized_cluster] = -0.05
                elif row["ann_return"] > 0:
                    horizon_scores[normalized_cluster] = -0.20
                else:
                    horizon_scores[normalized_cluster] = -0.50
            score_maps[horizon] = horizon_scores

        return score_maps

    def _generate_markdown_report(self, results: dict, output_dir: Path):
        summary = results.get("summary", {})
        comparison = results.get("comparison", {})
        split_summary = results.get("split_summary", {})
        report = f"""# Phase 7: Swarm Meta-Allocator Summary

**Generated**: {datetime.now().isoformat()}

## Overview

Phase 7 builds a particle-swarm meta allocator directly from existing processed returns,
features, and optional Phase 2 family/regime artifacts.

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
| Device | {summary.get("device", "cpu")} |

## Benchmark Comparison

| Metric | Swarm | Benchmark | Delta |
|--------|-------|-----------|-------|
| Total Return | {summary.get("portfolio_total_return", 0):.2%} | {summary.get("benchmark_total_return", 0):.2%} | {summary.get("excess_total_return", 0):.2%} |
| Annualized Return | {summary.get("annualized_return", 0):.2%} | {summary.get("benchmark_annualized_return", 0):.2%} | {(summary.get("annualized_return", 0) - summary.get("benchmark_annualized_return", 0)):.2%} |
| Sharpe Ratio | {summary.get("sharpe_ratio", 0):.3f} | {summary.get("benchmark_sharpe_ratio", 0):.3f} | {(summary.get("sharpe_ratio", 0) - summary.get("benchmark_sharpe_ratio", 0)):.3f} |
| Max Drawdown | {summary.get("max_drawdown", 0):.2%} | {summary.get("benchmark_max_drawdown", 0):.2%} | {(summary.get("max_drawdown", 0) - summary.get("benchmark_max_drawdown", 0)):.2%} |

## Headline

- Beat benchmark by total return: {comparison.get("beat_benchmark_total_return", False)}
- Daily hit rate vs benchmark: {comparison.get("daily_hit_rate_vs_benchmark", 0):.2%}

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
                    f"benchmark_return={split.get('benchmark_annualized_return', 0):.2%}\n"
                )
        else:
            report += "- Temporal split evaluation disabled\n"

        report += """

## Output Files

```
outputs/swarm_allocator/
|-- weights/weights.parquet
|-- backtests/portfolio_returns.csv
|-- diagnostics/optimization_diagnostics.csv
`-- reports/summary.json, comparison.json, split_summary.json
```
"""
        report_path = output_dir / "PHASE7_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

    def _print_result_summary(self, results: dict):
        """Print a concise performance summary to the console."""
        summary = results.get("summary", {})
        comparison = results.get("comparison", {})

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("PHASE 7 RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(
            "Swarm: return=%s | vol=%s | sharpe=%.3f | max_dd=%s",
            f"{summary.get('annualized_return', 0):.2%}",
            f"{summary.get('annualized_volatility', 0):.2%}",
            summary.get("sharpe_ratio", 0.0),
            f"{summary.get('max_drawdown', 0):.2%}",
        )
        if comparison:
            self.logger.info(
                "Benchmark: return=%s | sharpe=%.3f | max_dd=%s",
                f"{summary.get('benchmark_annualized_return', 0):.2%}",
                summary.get("benchmark_sharpe_ratio", 0.0),
                f"{summary.get('benchmark_max_drawdown', 0):.2%}",
            )
            self.logger.info(
                "Comparison: total_return_delta=%s | beat_benchmark=%s | daily_hit_rate=%s",
                f"{summary.get('excess_total_return', 0):.2%}",
                comparison.get("beat_benchmark_total_return", False),
                f"{comparison.get('daily_hit_rate_vs_benchmark', 0):.2%}",
            )
        split_summary = results.get("split_summary", {})
        test_split = split_summary.get("test", {}) if split_summary else {}
        if test_split:
            self.logger.info(
                "Temporal test: %s -> %s | return=%s | sharpe=%.3f | benchmark=%s",
                test_split.get("start_date", "n/a"),
                test_split.get("end_date", "n/a"),
                f"{test_split.get('annualized_return', 0):.2%}",
                test_split.get("sharpe_ratio", 0.0),
                f"{test_split.get('benchmark_annualized_return', 0):.2%}",
            )
        self.logger.info(
            "Artifacts: %s",
            results.get("outputs", {}).get("summary_path", "outputs/swarm_allocator/reports/summary.json"),
        )
        self.logger.info("=" * 70)

    def _build_temporal_split_summary(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series | None,
        evaluation_start: pd.Timestamp,
        train_ratio: float,
        validation_ratio: float,
    ) -> dict:
        eval_portfolio = portfolio_returns.loc[evaluation_start:].copy()
        if eval_portfolio.empty:
            return {}

        eval_benchmark = benchmark_returns.loc[evaluation_start:].copy() if benchmark_returns is not None else None
        n_obs = len(eval_portfolio)
        train_end_idx = max(int(n_obs * train_ratio), 1)
        val_end_idx = max(train_end_idx + int(n_obs * validation_ratio), train_end_idx + 1)
        val_end_idx = min(val_end_idx, n_obs - 1)

        segments = {
            "train": eval_portfolio.iloc[:train_end_idx],
            "validation": eval_portfolio.iloc[train_end_idx:val_end_idx],
            "test": eval_portfolio.iloc[val_end_idx:],
        }
        benchmark_segments = {
            "train": eval_benchmark.iloc[:train_end_idx] if eval_benchmark is not None else None,
            "validation": eval_benchmark.iloc[train_end_idx:val_end_idx] if eval_benchmark is not None else None,
            "test": eval_benchmark.iloc[val_end_idx:] if eval_benchmark is not None else None,
        }

        summary: dict[str, dict] = {}
        for name, segment in segments.items():
            if segment.empty:
                continue
            segment_benchmark = benchmark_segments.get(name)
            metrics = compute_full_metrics(segment, segment_benchmark)
            metrics["start_date"] = str(segment.index.min().date())
            metrics["end_date"] = str(segment.index.max().date())
            if segment_benchmark is not None and not segment_benchmark.empty:
                benchmark_metrics = compute_full_metrics(segment_benchmark)
                metrics["benchmark_annualized_return"] = float(benchmark_metrics.get("annualized_return", 0.0))
                metrics["benchmark_sharpe_ratio"] = float(benchmark_metrics.get("sharpe_ratio", 0.0))
            summary[name] = metrics

        summary["split_meta"] = {
            "train_ratio": float(train_ratio),
            "validation_ratio": float(validation_ratio),
            "test_ratio": float(max(0.0, 1.0 - train_ratio - validation_ratio)),
        }
        return summary


def main():
    runner = Phase7Runner()
    runner.execute()


if __name__ == "__main__":
    main()
