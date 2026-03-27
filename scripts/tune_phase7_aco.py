#!/usr/bin/env python3
"""
=================================================================
PHASE 7A Tuning: ACO Factor Search
=================================================================
Tune Phase 7A ACO configurations with a Phase-3-style factor search.

This script evaluates multiple selection factors against the ACO allocator,
using temporal validation/test splits and ranking trials by out-of-sample
performance.

Usage:
  python scripts/tune_phase7_aco.py
  python scripts/tune_phase7_aco.py --quick --sample 100
  python scripts/tune_phase7_aco.py --factor sharpe_21d momentum_63d
  python scripts/tune_phase7_aco.py --selection-scheme walk_forward --max-folds 3

Options:
  --sample N                    Use only N algorithms (for faster testing)
  --quick                       Run reduced factor search (fewer parameter combos)
  --max-trials N                Maximum number of factor trials; 0 = all (default: 24)
  --factor ALIAS [ALIAS ...]    Specific factor aliases to search (default: all)
  --min-combination-size N      Minimum factors per combination (default: 1)
  --max-combination-size N      Maximum factors per combination (default: all)
  --input-dir PATH              Override Phase 1 processed directory
  --analysis-dir PATH           Override Phase 2 analysis directory
  --start-date YYYY-MM-DD       Start date for evaluation
  --end-date YYYY-MM-DD         End date for evaluation

Selection Scheme Options:
  --selection-scheme S          How to score factors: walk_forward, single_split
                                (default: walk_forward)

Single-Split Options (when --selection-scheme single_split):
  --train-ratio F               Train ratio for split (default: 0.60)
  --validation-ratio F          Validation ratio for split (default: 0.20)

Walk-Forward Options (when --selection-scheme walk_forward):
  --train-window N              Training window in days (default: 252)
  --validation-window N         Validation window in days (default: 63)
  --test-window N               Test window in days (default: 63)
  --step-size N                 Step size between folds (default: 63)
  --expanding                   Use expanding window instead of rolling
  --max-folds N                 Maximum number of folds

Selection Criteria:
  --selection-objective S       How to choose best trial (default: validation_excess_return)
                                Options: validation_sharpe, validation_excess_return,
                                         blended_excess_return, blended_sharpe
  --validation-weight F         Weight on validation metrics (default: 0.70)
  --test-weight F               Weight on test metrics (default: 0.30)
  --max-validation-drawdown F   Max allowed validation drawdown (default: 0.12)
  --max-test-drawdown F         Max allowed test drawdown (default: 0.15)
  --allow-test-in-selection     Allow blended objectives that use test data in selection

Available Factors:
  momentum_21d, momentum_63d    Rolling return (21d, 63d windows)
  sharpe_21d, sharpe_63d        Rolling Sharpe ratio
  profit_factor_21d/63d         Rolling profit factor
  calmar_21d, calmar_63d        Rolling Calmar ratio
  drawdown_21d, drawdown_63d    Rolling drawdown
  volatility_21d, volatility_63d  Rolling volatility

Output Files:
  tuning/factor_search/trials.csv               All trial results
  tuning/factor_search/trial_split_metrics.csv  Per-split metrics
  tuning/factor_search/relationship_summary.csv Validation-test relationships
  tuning/factor_search/best_config.json         Best configuration
  tuning/factor_search/SUMMARY.md               Human-readable report
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner
from scripts.run_phase7 import Phase7Runner
from src.evaluation.metrics import compute_full_metrics
from src.evaluation.walk_forward import WalkForwardValidator
from src.swarm import ACOAllocatorBacktester, ACOConfig


PHASE3_STYLE_FACTORS = {
    "momentum_21d": "rolling_return_21d",
    "momentum_63d": "rolling_return_63d",
    "sharpe_21d": "rolling_sharpe_21d",
    "sharpe_63d": "rolling_sharpe_63d",
    "profit_factor_21d": "rolling_profit_factor_21d",
    "profit_factor_63d": "rolling_profit_factor_63d",
    "calmar_21d": "rolling_calmar_21d",
    "calmar_63d": "rolling_calmar_63d",
    "drawdown_21d": "rolling_drawdown_21d",
    "drawdown_63d": "rolling_drawdown_63d",
    "volatility_21d": "rolling_volatility_21d",
    "volatility_63d": "rolling_volatility_63d",
}

SPLIT_METRIC_KEYS = (
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "tracking_error",
    "information_ratio",
    "alpha_vs_benchmark",
    "beta_vs_benchmark",
    "benchmark_annualized_return",
)


@dataclass
class ACOFactorTrialSpec:
    factor_alias: str
    selection_factors: tuple[str, ...]
    top_k: int
    lookback_window: int
    ants: int
    iterations: int
    weight_buckets: int
    target_portfolio_vol: float

    def __str__(self) -> str:
        return (
            f"{self.factor_alias}|top{self.top_k}|lb{self.lookback_window}|"
            f"ants{self.ants}|it{self.iterations}|wb{self.weight_buckets}"
        )


class Phase7ACOTuningRunner(PhaseRunner):
    phase_name = "Phase 7A Tuning: Factor Search"
    phase_number = 78

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--sample", type=int, default=None, help="Use only N algorithms")
        parser.add_argument("--quick", action="store_true", help="Run a reduced factor search")
        parser.add_argument("--max-trials", type=int, default=24, help="Maximum number of factor trials; 0 means all")
        parser.add_argument(
            "--factor",
            dest="factors",
            nargs="*",
            default=None,
            help="Optional factor aliases to search",
        )
        parser.add_argument("--min-combination-size", type=int, default=1)
        parser.add_argument("--max-combination-size", type=int, default=None)
        parser.add_argument("--input-dir", type=str, default=None)
        parser.add_argument("--analysis-dir", type=str, default=None)
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--end-date", type=str, default=None)
        parser.add_argument(
            "--selection-scheme",
            type=str,
            default="walk_forward",
            choices=["walk_forward", "single_split"],
            help="How to score factors during tuning",
        )
        parser.add_argument("--train-ratio", type=float, default=0.60)
        parser.add_argument("--validation-ratio", type=float, default=0.20)
        parser.add_argument("--train-window", type=int, default=252)
        parser.add_argument("--validation-window", type=int, default=63)
        parser.add_argument("--test-window", type=int, default=63)
        parser.add_argument("--step-size", type=int, default=63)
        parser.add_argument("--expanding", action="store_true")
        parser.add_argument("--max-folds", type=int, default=None)
        parser.add_argument("--max-validation-drawdown", type=float, default=0.12)
        parser.add_argument("--max-test-drawdown", type=float, default=0.15)
        parser.add_argument("--validation-weight", type=float, default=0.70)
        parser.add_argument("--test-weight", type=float, default=0.30)
        parser.add_argument(
            "--selection-objective",
            type=str,
            default="validation_excess_return",
            choices=["validation_sharpe", "validation_excess_return", "blended_excess_return", "blended_sharpe"],
        )
        parser.add_argument(
            "--allow-test-in-selection",
            action="store_true",
            help="Explicitly allow blended objectives that use holdout test metrics during factor selection",
        )

    def run(self, args: argparse.Namespace) -> dict:
        phase7 = Phase7Runner()
        input_dir = Path(args.input_dir) if args.input_dir else phase7.dp.processed.root
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else phase7.dp.processed.analysis.root
        output_dir = self.get_output_dir()
        self._validate_selection_policy(args)

        with self.step("78.1 Discover Factors"):
            factor_map = self._discover_factors(phase7, input_dir, args.factors)
            if not factor_map:
                raise ValueError("No candidate factors found in the features parquet")

        with self.step("78.2 Load Shared Artifacts"):
            shared = self._load_shared_artifacts(
                phase7=phase7,
                input_dir=input_dir,
                analysis_dir=analysis_dir,
                sample=args.sample,
                extra_suffixes=set(factor_map.values()),
            )

        with self.step("78.3 Prepare Selection Splits"):
            selection_folds = self._build_selection_folds(shared["algo_returns"], args)

        specs = self._generate_trial_specs(
            factor_map,
            quick=args.quick,
            min_combination_size=args.min_combination_size,
            max_combination_size=args.max_combination_size,
        )
        if args.max_trials > 0:
            specs = specs[: args.max_trials]
        if not specs:
            raise ValueError("No factor trials generated")

        rows = []
        best_row = None
        for idx, spec in enumerate(specs, start=1):
            with self.step(f"78.4 Trial {idx}: {spec.factor_alias}"):
                row = self._run_trial(
                    phase7=phase7,
                    shared=shared,
                    spec=spec,
                    args=args,
                    selection_folds=selection_folds,
                )
                rows.append(row)
                if best_row is None or row["selection_score"] > best_row["selection_score"]:
                    best_row = row

        trials_df = pd.DataFrame(rows).sort_values(
            by=["selection_score", "validation_sharpe_ratio", "validation_annualized_return"],
            ascending=[False, False, False],
        )
        best_row = trials_df.iloc[0].to_dict()

        tuning_dir = output_dir / "tuning" / "factor_search"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        trials_path = tuning_dir / "trials.csv"
        split_metrics_path = tuning_dir / "trial_split_metrics.csv"
        relationships_json_path = tuning_dir / "relationship_summary.json"
        relationships_csv_path = tuning_dir / "relationship_summary.csv"
        best_path = tuning_dir / "best_config.json"
        report_path = tuning_dir / "SUMMARY.md"
        split_metrics_df = self._build_split_metrics_frame(trials_df)
        relationship_df = self._build_relationship_summary(trials_df)
        trials_df.to_csv(trials_path, index=False)
        split_metrics_df.to_csv(split_metrics_path, index=False)
        relationship_df.to_csv(relationships_csv_path, index=False)
        with open(best_path, "w", encoding="utf-8") as handle:
            json.dump(best_row, handle, indent=2, default=str)
        with open(relationships_json_path, "w", encoding="utf-8") as handle:
            json.dump(relationship_df.to_dict(orient="records"), handle, indent=2, default=str)
        report_path.write_text(
            self._build_report(
                trials_df=trials_df,
                split_metrics_df=split_metrics_df,
                relationship_df=relationship_df,
                best_row=best_row,
                n_factors=len(factor_map),
                selection_objective=args.selection_objective,
                selection_scheme=args.selection_scheme,
                n_folds=len(selection_folds),
                selection_uses_test_data=bool(
                    args.allow_test_in_selection and args.selection_objective.startswith("blended_")
                ),
            ),
            encoding="utf-8",
        )

        return {
            "n_trials": len(rows),
            "n_factors": len(factor_map),
            "selection_scheme": args.selection_scheme,
            "n_folds": len(selection_folds),
            "best_trial": best_row,
            "trials_path": str(trials_path),
            "split_metrics_path": str(split_metrics_path),
            "relationship_summary_json_path": str(relationships_json_path),
            "relationship_summary_csv_path": str(relationships_csv_path),
            "best_config_path": str(best_path),
            "report_path": str(report_path),
        }

    def _validate_selection_policy(self, args: argparse.Namespace) -> None:
        if args.selection_objective.startswith("blended_") and not args.allow_test_in_selection:
            raise ValueError(
                "Blended selection objectives use holdout test metrics. "
                "Pass --allow-test-in-selection to enable them explicitly."
            )

    def _build_selection_folds(self, algo_returns: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
        evaluation_returns = algo_returns.copy()
        if args.start_date or args.end_date:
            start_ts = pd.Timestamp(args.start_date) if args.start_date else evaluation_returns.index.min()
            end_ts = pd.Timestamp(args.end_date) if args.end_date else evaluation_returns.index.max()
            evaluation_returns = evaluation_returns.loc[
                (evaluation_returns.index >= start_ts) & (evaluation_returns.index <= end_ts)
            ]
        if evaluation_returns.empty:
            raise ValueError("No return observations remain after applying start/end date filters")

        if args.selection_scheme == "single_split":
            return [
                {
                    "fold_id": 0,
                    "train_start": evaluation_returns.index.min(),
                    "train_end": evaluation_returns.index.max(),
                    "val_start": evaluation_returns.index.min(),
                    "val_end": evaluation_returns.index.max(),
                    "test_start": evaluation_returns.index.min(),
                    "test_end": evaluation_returns.index.max(),
                    "scheme": "single_split",
                }
            ]

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
        if not folds:
            raise ValueError("No walk-forward folds were generated for tuning")
        return folds

    def _discover_factors(
        self,
        phase7: Phase7Runner,
        input_dir: Path,
        requested_aliases: list[str] | None,
    ) -> dict[str, str]:
        path = phase7.dp.algorithms.features
        if not path.exists():
            path = input_dir / "algo_features.parquet"
        if not path.exists():
            return {}

        requested = requested_aliases or list(PHASE3_STYLE_FACTORS.keys())
        columns = phase7._peek_parquet_columns(path)
        discovered = {}
        for alias in requested:
            suffix = PHASE3_STYLE_FACTORS.get(alias)
            if suffix is None:
                continue
            if any(column.endswith(f"_{suffix}") for column in columns):
                discovered[alias] = suffix
        return discovered

    def _load_shared_artifacts(
        self,
        phase7: Phase7Runner,
        input_dir: Path,
        analysis_dir: Path,
        sample: int | None,
        extra_suffixes: set[str],
    ) -> dict:
        sampled_cols = phase7._resolve_sampled_algorithms(input_dir, sample)
        algo_returns = phase7._load_returns(input_dir, sampled_cols=sampled_cols)
        features = phase7._load_optional_features(
            input_dir,
            sampled_algorithms=sampled_cols,
            extra_suffixes=extra_suffixes,
        )
        benchmark_returns = phase7._load_optional_benchmark_returns(input_dir)
        benchmark_weights = phase7._load_optional_benchmark_weights(input_dir, sampled_cols=sampled_cols)
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

    def _generate_trial_specs(
        self,
        factor_map: dict[str, str],
        quick: bool,
        min_combination_size: int,
        max_combination_size: int | None,
    ) -> list[ACOFactorTrialSpec]:
        if quick:
            top_ks = [64]
            lookbacks = [84]
            ants = [24]
            iterations = [6]
            buckets = [11]
            target_vols = [0.16]
        else:
            top_ks = [64, 128]
            lookbacks = [84, 126]
            ants = [48]
            iterations = [12]
            buckets = [11, 21]
            target_vols = [0.14, 0.16]

        factor_items = list(factor_map.items())
        if not factor_items:
            return []
        max_size = max_combination_size or len(factor_items)
        max_size = max(1, min(max_size, len(factor_items)))
        min_size = max(1, min(min_combination_size, max_size))

        factor_combinations = []
        for combo_size in range(min_size, max_size + 1):
            for combo in combinations(factor_items, combo_size):
                aliases = tuple(alias for alias, _ in combo)
                suffixes = tuple(suffix for _, suffix in combo)
                factor_combinations.append((aliases, suffixes))

        specs = []
        for aliases, suffixes in factor_combinations:
            factor_alias = "+".join(aliases)
            for top_k in top_ks:
                for lookback in lookbacks:
                    for n_ants in ants:
                        for n_iterations in iterations:
                            for weight_buckets in buckets:
                                for target_vol in target_vols:
                                    specs.append(
                                        ACOFactorTrialSpec(
                                            factor_alias=factor_alias,
                                            selection_factors=suffixes,
                                            top_k=top_k,
                                            lookback_window=lookback,
                                            ants=n_ants,
                                            iterations=n_iterations,
                                            weight_buckets=weight_buckets,
                                            target_portfolio_vol=target_vol,
                                        )
                                    )
        return specs

    def _run_trial(
        self,
        phase7: Phase7Runner,
        shared: dict,
        spec: ACOFactorTrialSpec,
        args: argparse.Namespace,
        selection_folds: list[dict],
    ) -> dict:
        config = ACOConfig(
            lookback_window=spec.lookback_window,
            rebalance_frequency="weekly",
            top_k=spec.top_k,
            min_history=max(42, spec.lookback_window // 2),
            n_ants=spec.ants,
            n_iterations=spec.iterations,
            weight_buckets=spec.weight_buckets,
            max_weight=0.40,
            max_family_exposure=0.30,
            expected_return_weight=0.80,
            volatility_weight=0.35,
            turnover_weight=0.15,
            concentration_weight=0.10,
            diversification_weight=0.12,
            family_penalty_weight=0.20,
            family_alpha_reward_weight=0.15,
            risk_budget_weight=0.25,
            sparsity_penalty_weight=0.01,
            entropy_reward_weight=0.05,
            sharpe_weight=1.75,
            regime_focus=1.50,
            target_portfolio_vol=spec.target_portfolio_vol,
            min_active_weight=0.0025,
            min_gross_exposure=0.85,
            under_investment_penalty_weight=0.35,
            objective_name=f"factor_search_{spec.factor_alias}",
            selection_mode="legacy",
            normalize_objective_metrics=False,
            seed=42,
            use_gpu=False,
        )
        if args.selection_scheme == "single_split":
            backtester = self._build_backtester(shared, config, spec.selection_factors)
            result = backtester.run(start_date=args.start_date, end_date=args.end_date)
            split_summary = phase7._build_temporal_split_summary(
                portfolio_returns=result.portfolio_returns,
                benchmark_returns=result.benchmark_returns,
                evaluation_start=result.weights.index.min(),
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
            )
            train = split_summary.get("train", {})
            validation = split_summary.get("validation", {})
            test = split_summary.get("test", {})
            n_folds = 1
            trial_summary = result.summary
        else:
            fold_metrics = []
            fold_summaries = []
            for fold in selection_folds:
                backtester = self._build_backtester(shared, config, spec.selection_factors)
                result = backtester.run(
                    start_date=str(fold["train_start"].date()),
                    end_date=str(fold["test_end"].date()),
                )
                train_returns = result.portfolio_returns.loc[fold["train_start"] : fold["train_end"]]
                val_returns = result.portfolio_returns.loc[fold["val_start"] : fold["val_end"]]
                test_returns = result.portfolio_returns.loc[fold["test_start"] : fold["test_end"]]
                benchmark_train = (
                    result.benchmark_returns.loc[fold["train_start"] : fold["train_end"]]
                    if result.benchmark_returns is not None
                    else None
                )
                benchmark_val = (
                    result.benchmark_returns.loc[fold["val_start"] : fold["val_end"]]
                    if result.benchmark_returns is not None
                    else None
                )
                benchmark_test = (
                    result.benchmark_returns.loc[fold["test_start"] : fold["test_end"]]
                    if result.benchmark_returns is not None
                    else None
                )
                fold_metrics.append(
                    {
                        "train": compute_full_metrics(train_returns, benchmark_train),
                        "validation": compute_full_metrics(val_returns, benchmark_val),
                        "test": compute_full_metrics(test_returns, benchmark_test),
                    }
                )
                fold_summaries.append(result.summary)
            train = self._aggregate_metric_dicts([fold["train"] for fold in fold_metrics])
            validation = self._aggregate_metric_dicts([fold["validation"] for fold in fold_metrics])
            test = self._aggregate_metric_dicts([fold["test"] for fold in fold_metrics])
            n_folds = len(fold_metrics)
            trial_summary = self._aggregate_metric_dicts(fold_summaries)

        score = self._compute_selection_score(
            validation=validation,
            test=test,
            max_validation_drawdown=args.max_validation_drawdown,
            max_test_drawdown=args.max_test_drawdown,
            validation_weight=args.validation_weight,
            test_weight=args.test_weight,
            objective=args.selection_objective,
        )

        return {
            "trial_name": str(spec),
            "factor_alias": spec.factor_alias,
            "selection_factor": (
                spec.selection_factors[0] if len(spec.selection_factors) == 1 else list(spec.selection_factors)
            ),
            "selection_factor_count": int(len(spec.selection_factors)),
            "selection_score": score,
            "top_k": spec.top_k,
            "lookback_window": spec.lookback_window,
            "ants": spec.ants,
            "iterations": spec.iterations,
            "weight_buckets": spec.weight_buckets,
            "target_portfolio_vol": spec.target_portfolio_vol,
            "selection_scheme": args.selection_scheme,
            "n_folds": int(n_folds),
            **self._flatten_split_metrics("train", train),
            **self._flatten_split_metrics("validation", validation),
            **self._flatten_split_metrics("test", test),
            "validation_test_return_gap": float(validation.get("annualized_return", 0.0) - test.get("annualized_return", 0.0)),
            "validation_test_sharpe_gap": float(validation.get("sharpe_ratio", 0.0) - test.get("sharpe_ratio", 0.0)),
            "validation_test_drawdown_gap": float(validation.get("max_drawdown", 0.0) - test.get("max_drawdown", 0.0)),
            "selection_uses_test_data": bool(
                args.allow_test_in_selection and args.selection_objective.startswith("blended_")
            ),
            "test_role": (
                "selection"
                if args.allow_test_in_selection and args.selection_objective.startswith("blended_")
                else "report_only_holdout"
            ),
            **trial_summary,
        }

    def _build_backtester(
        self,
        shared: dict,
        config: ACOConfig,
        selection_factor: str | tuple[str, ...],
    ) -> ACOAllocatorBacktester:
        return ACOAllocatorBacktester(
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
            selection_factor=selection_factor,
        )

    def _aggregate_metric_dicts(self, metrics_list: list[dict]) -> dict:
        if not metrics_list:
            return {}

        numeric_keys = set()
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, Real):
                    numeric_keys.add(key)

        aggregated = {}
        for key in sorted(numeric_keys):
            values = [float(metrics.get(key, 0.0)) for metrics in metrics_list]
            if not values:
                continue
            aggregated[key] = float(sum(values) / len(values))
        return aggregated

    def _flatten_split_metrics(self, split_name: str, metrics: dict) -> dict[str, float]:
        flattened = {}
        for key in SPLIT_METRIC_KEYS:
            flattened[f"{split_name}_{key}"] = float(metrics.get(key, 0.0))
        return flattened

    def _build_split_metrics_frame(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        base_columns = [
            "trial_name",
            "factor_alias",
            "selection_factor",
            "selection_factor_count",
            "selection_score",
            "selection_scheme",
            "n_folds",
            "top_k",
            "lookback_window",
            "ants",
            "iterations",
            "weight_buckets",
            "target_portfolio_vol",
        ]
        for _, row in trials_df.iterrows():
            for split_name in ("train", "validation", "test"):
                split_row = {column: row[column] for column in base_columns}
                split_row["split"] = split_name
                for key in SPLIT_METRIC_KEYS:
                    split_row[key] = float(row.get(f"{split_name}_{key}", 0.0))
                rows.append(split_row)
        return pd.DataFrame(rows)

    def _build_relationship_summary(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        metrics = [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "tracking_error",
            "information_ratio",
            "alpha_vs_benchmark",
            "beta_vs_benchmark",
        ]
        rows = []
        for metric in metrics:
            train_series = pd.to_numeric(trials_df.get(f"train_{metric}"), errors="coerce")
            validation_series = pd.to_numeric(trials_df.get(f"validation_{metric}"), errors="coerce")
            test_series = pd.to_numeric(trials_df.get(f"test_{metric}"), errors="coerce")
            rows.append(
                {
                    "metric": metric,
                    "train_validation_correlation": self._safe_series_correlation(train_series, validation_series),
                    "validation_test_correlation": self._safe_series_correlation(validation_series, test_series),
                    "train_test_correlation": self._safe_series_correlation(train_series, test_series),
                    "mean_train": float(train_series.mean()) if train_series.notna().any() else 0.0,
                    "mean_validation": float(validation_series.mean()) if validation_series.notna().any() else 0.0,
                    "mean_test": float(test_series.mean()) if test_series.notna().any() else 0.0,
                    "validation_minus_test_mean_gap": float((validation_series - test_series).mean())
                    if (validation_series.notna() & test_series.notna()).any()
                    else 0.0,
                    "validation_minus_test_mae": float((validation_series - test_series).abs().mean())
                    if (validation_series.notna() & test_series.notna()).any()
                    else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _safe_series_correlation(self, left: pd.Series, right: pd.Series) -> float:
        aligned = pd.concat([left, right], axis=1).dropna()
        if len(aligned) <= 1:
            return 0.0
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if pd.isna(corr):
            return 0.0
        return float(corr)

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
        if objective == "blended_excess_return":
            return validation_weight * validation_excess + test_weight * test_excess - validation_penalty - test_penalty
        return validation_weight * validation_sharpe + test_weight * test_sharpe - validation_penalty - test_penalty

    def _build_report(
        self,
        trials_df: pd.DataFrame,
        split_metrics_df: pd.DataFrame,
        relationship_df: pd.DataFrame,
        best_row: dict,
        n_factors: int,
        selection_objective: str,
        selection_scheme: str,
        n_folds: int,
        selection_uses_test_data: bool,
    ) -> str:
        top_rows = trials_df.head(10)
        lines = [
            "# Phase 7A Factor Search",
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Selection scheme: {selection_scheme}",
            f"Selection folds: {n_folds}",
            f"Selection objective: {selection_objective}",
            f"Selection uses holdout test data: {selection_uses_test_data}",
            f"Factors searched: {n_factors}",
            f"Trials run: {len(trials_df)}",
            "",
            "## Best Trial",
            "",
            f"- Trial: {best_row['trial_name']}",
            f"- Factor alias: {best_row['factor_alias']}",
            f"- Selection factor: {best_row['selection_factor']}",
            f"- Test role: {best_row.get('test_role', 'report_only_holdout')}",
            f"- Selection scheme: {best_row.get('selection_scheme', selection_scheme)}",
            f"- Folds evaluated: {int(best_row.get('n_folds', n_folds))}",
            f"- Train Sharpe: {best_row['train_sharpe_ratio']:.3f}",
            f"- Validation Sharpe: {best_row['validation_sharpe_ratio']:.3f}",
            f"- Test Sharpe: {best_row['test_sharpe_ratio']:.3f}",
            f"- Train annualized return: {best_row['train_annualized_return']:.2%}",
            f"- Validation annualized return: {best_row['validation_annualized_return']:.2%}",
            f"- Test annualized return: {best_row['test_annualized_return']:.2%}",
            "",
            "## Top Trials",
            "",
            "| Trial | Factor | Score | Train Sharpe | Val Sharpe | Test Sharpe | Val Return | Test Return |",
            "|------|--------|-------|--------------|------------|-------------|------------|-------------|",
        ]
        for _, row in top_rows.iterrows():
            lines.append(
                f"| {row['trial_name']} | {row['factor_alias']} | {row['selection_score']:.4f} | "
                f"{row['train_sharpe_ratio']:.3f} | {row['validation_sharpe_ratio']:.3f} | {row['test_sharpe_ratio']:.3f} | "
                f"{row['validation_annualized_return']:.2%} | {row['test_annualized_return']:.2%} |"
            )
        if not relationship_df.empty:
            lines.extend(
                [
                    "",
                    "## Validation vs Test Relationships",
                    "",
                    relationship_df.to_markdown(index=False),
                ]
            )
        best_trial_split_metrics = split_metrics_df[split_metrics_df["trial_name"] == best_row["trial_name"]]
        if not best_trial_split_metrics.empty:
            lines.extend(
                [
                    "",
                    "## Best Trial Split Metrics",
                    "",
                    best_trial_split_metrics.to_markdown(index=False),
                ]
            )
        lines.append("")
        return "\n".join(lines)


def main():
    runner = Phase7ACOTuningRunner()
    runner.execute()


if __name__ == "__main__":
    main()
