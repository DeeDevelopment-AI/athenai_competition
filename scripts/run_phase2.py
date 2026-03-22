#!/usr/bin/env python3
"""
=================================================================
PHASE 2: Analysis and Reverse Engineering
=================================================================
Complete Phase 2 pipeline:
  2.1 Profile all algorithms (performance, risk, operational metrics)
  2.2 Reverse engineer the benchmark (sizing, temporal, risk policies)
  2.3 Infer latent regimes (family clustering + HMM/fuzzy inference)
  2.4 Analyze correlations (stability, diversification, clustering)
  2.5 Temporal clustering (weekly cluster evolution analysis)

Usage:
  python scripts/run_phase2.py
  python scripts/run_phase2.py --sample 100 --skip-inference
  python scripts/run_phase2.py --n-regimes 6 --n-families 10

Options:
  --sample N               Use only N algorithms (for testing)
  --n-regimes N            Number of latent regimes (default: 4)
  --n-families N           Number of algorithm families (default: 8)
  --n-clusters N           Number of temporal clusters (default: 5)
  --skip-inference         Skip latent regime inference (faster)
  --skip-correlations      Skip correlation analysis (faster)
  --skip-temporal          Skip temporal clustering analysis (faster)
  --clustering-method M    Clustering method: kmeans, gmm, hierarchical (default: kmeans)
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner


class Phase2Runner(PhaseRunner):
    """Phase 2: Analysis and Reverse Engineering."""

    phase_name = "Phase 2: Analysis & Reverse Engineering"
    phase_number = 2

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 2 specific arguments."""
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Sample N algorithms for testing'
        )
        parser.add_argument(
            '--n-regimes', type=int, default=4,
            help='Number of latent regimes'
        )
        parser.add_argument(
            '--n-families', type=int, default=8,
            help='Number of algorithm families'
        )
        parser.add_argument(
            '--skip-inference', action='store_true',
            help='Skip latent regime inference'
        )
        parser.add_argument(
            '--skip-correlations', action='store_true',
            help='Skip correlation analysis'
        )
        parser.add_argument(
            '--skip-temporal', action='store_true',
            help='Skip temporal clustering analysis'
        )
        parser.add_argument(
            '--n-clusters', type=int, default=5,
            help='Number of temporal clusters'
        )
        parser.add_argument(
            '--clustering-method', type=str, default='kmeans',
            choices=['kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan'],
            help='Temporal clustering method'
        )
        parser.add_argument(
            '--input-dir', type=str, default=None,
            help='Override input directory (Phase 1 outputs)'
        )

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 2 pipeline."""
        results = {
            'params': {
                'sample': args.sample,
                'n_regimes': args.n_regimes,
                'n_families': args.n_families,
                'n_clusters': args.n_clusters,
                'clustering_method': args.clustering_method,
            }
        }

        # Determine paths - use organized structure
        input_dir = Path(args.input_dir) if args.input_dir else self.dp.processed.root
        analysis_dir = self.dp.processed.analysis.root

        # Create subdirectories
        profiles_dir = analysis_dir / "profiles"
        clustering_dir = analysis_dir / "clustering"
        regimes_dir = analysis_dir / "regimes"
        benchmark_profile_dir = analysis_dir / "benchmark_profile"

        for d in [profiles_dir, clustering_dir, regimes_dir, benchmark_profile_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # ==================================================================
        # LOAD PHASE 1 DATA
        # ==================================================================
        with self.step("Load Phase 1 Data"):
            data = self._load_phase1_data(input_dir)
            returns_matrix = data['returns_matrix']
            benchmark_returns = data.get('benchmark_returns')
            benchmark_weights = data.get('benchmark_weights')

            # Sample if requested
            if args.sample and args.sample < len(returns_matrix.columns):
                np.random.seed(42)
                sampled_cols = list(np.random.choice(
                    returns_matrix.columns, args.sample, replace=False
                ))
                returns_matrix = returns_matrix[sampled_cols]
                if benchmark_weights is not None:
                    benchmark_weights = benchmark_weights[
                        [c for c in sampled_cols if c in benchmark_weights.columns]
                    ]
                self.logger.info(f"Sampled {args.sample} algorithms")

            self.logger.info(f"Returns matrix: {returns_matrix.shape}")

        # ==================================================================
        # STEP 2.1: ALGORITHM PROFILING
        # ==================================================================
        with self.step("2.1 Algorithm Profiling"):
            from src.analysis.algo_profiler import AlgoProfiler

            profiler = AlgoProfiler()
            self.logger.info(f"Profiling {len(returns_matrix.columns)} algorithms...")

            profiles = profiler.profile_all(
                returns_matrix=returns_matrix,
                benchmark_returns=benchmark_returns,
            )
            self.logger.info(f"Profiled {len(profiles)} algorithms")

            # Generate summary table
            summary_df = profiler.generate_summary_table(profiles)
            summary_df.to_csv(profiles_dir / 'summary.csv')
            self.logger.info("Saved profiles/summary.csv")

            # Save full profiles as JSON
            profiles_dict = {}
            for algo_id, profile in profiles.items():
                profiles_dict[algo_id] = {
                    'annualized_return': profile.annualized_return,
                    'annualized_volatility': profile.annualized_volatility,
                    'sharpe_ratio': profile.sharpe_ratio,
                    'sortino_ratio': profile.sortino_ratio,
                    'calmar_ratio': profile.calmar_ratio,
                    'max_drawdown': profile.max_drawdown,
                    'max_drawdown_duration': profile.max_drawdown_duration,
                    'var_95': profile.var_95,
                    'cvar_95': profile.cvar_95,
                    'tail_ratio': profile.tail_ratio,
                    'correlation_with_benchmark': profile.correlation_with_benchmark,
                }

            with open(profiles_dir / 'full.json', 'w', encoding='utf-8') as f:
                json.dump(profiles_dict, f, indent=2, default=str)
            self.logger.info("Saved profiles/full.json")

            # Top performers
            top_sharpe = summary_df.nlargest(10, 'sharpe')
            self.logger.info("\nTop 10 by Sharpe Ratio:")
            for idx, row in top_sharpe.iterrows():
                self.logger.info(f"  {idx}: Sharpe={row['sharpe']:.2f}, Return={row['ann_return']:.2%}")

            results['algo_profiles'] = {
                'n_profiled': len(profiles),
                'top_sharpe': top_sharpe.index.tolist(),
            }

        gc.collect()

        # ==================================================================
        # STEP 2.2: BENCHMARK REVERSE ENGINEERING
        # ==================================================================
        if benchmark_returns is not None:
            with self.step("2.2 Benchmark Reverse Engineering"):
                from src.analysis.benchmark_profiler import BenchmarkProfiler

                bench_profiler = BenchmarkProfiler()

                # Create synthetic weights if not available
                if benchmark_weights is None:
                    self.logger.warning("No benchmark weights, using equal weights")
                    benchmark_weights = pd.DataFrame(
                        1.0 / len(returns_matrix.columns),
                        index=returns_matrix.index,
                        columns=returns_matrix.columns
                    )

                self.logger.info("Analyzing benchmark behavior...")
                bench_profile = bench_profiler.profile(
                    returns=benchmark_returns,
                    weights=benchmark_weights,
                )

                # Generate report
                report = bench_profiler.generate_report(bench_profile)
                with open(benchmark_profile_dir / 'report.txt', 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info("Saved benchmark_profile_report.txt")

                # Save key metrics
                bench_metrics = {
                    'annualized_return': bench_profile.annualized_return,
                    'annualized_volatility': bench_profile.annualized_volatility,
                    'sharpe_ratio': bench_profile.sharpe_ratio,
                    'sortino_ratio': bench_profile.sortino_ratio,
                    'calmar_ratio': bench_profile.calmar_ratio,
                    'max_drawdown': bench_profile.max_drawdown,
                    'max_drawdown_duration': bench_profile.max_drawdown_duration,
                    'concentration_hhi': bench_profile.concentration_hhi,
                    'concentration_hhi_avg': bench_profile.concentration_hhi_avg,
                    'rebalance_frequency_days': bench_profile.rebalance_frequency_days,
                    'turnover_annualized': bench_profile.turnover_annualized,
                    'avg_holding_period_days': bench_profile.avg_holding_period_days,
                }

                with open(benchmark_profile_dir / 'metrics.json', 'w', encoding='utf-8') as f:
                    json.dump(bench_metrics, f, indent=2, default=str)
                self.logger.info("Saved benchmark_metrics.json")

                results['benchmark_profile'] = bench_metrics

        gc.collect()

        # ==================================================================
        # STEP 2.3: LATENT REGIME INFERENCE
        # ==================================================================
        if not args.skip_inference and benchmark_returns is not None:
            with self.step("2.3 Latent Regime Inference"):
                from src.analysis.latent_regime_inference import (
                    LatentRegimeInference,
                    InferenceMethod,
                )

                inference = LatentRegimeInference(
                    n_regimes=args.n_regimes,
                    random_state=42,
                )

                self.logger.info("Building activity mask...")
                mask = inference.build_activity_mask(returns_matrix)

                self.logger.info("Computing algorithm behavioral features...")
                algo_features = inference.compute_algo_behavioral_features(returns_matrix, mask)
                (clustering_dir / 'behavioral').mkdir(parents=True, exist_ok=True)
                algo_features.to_csv(clustering_dir / 'behavioral' / 'features.csv')
                self.logger.info("Saved clustering/behavioral/features.csv")

                self.logger.info(f"Assigning algorithms to {args.n_families} families...")
                family_labels = inference.assign_families(
                    algo_features,
                    n_families=args.n_families,
                    method='gmm'
                )
                family_labels.to_csv(clustering_dir / 'behavioral' / 'family_labels.csv')
                self.logger.info("Saved algo_family_labels.csv")

                # Family distribution
                family_counts = family_labels.value_counts().sort_index()
                self.logger.info("Family distribution:")
                for fam_id, count in family_counts.items():
                    self.logger.info(f"  Family {fam_id}: {count} algorithms")

                self.logger.info("Building temporal features...")
                family_features, universe_features, bench_features = inference.build_temporal_features(
                    algo_returns=returns_matrix,
                    benchmark_weights=benchmark_weights,
                    benchmark_returns=benchmark_returns,
                    family_labels=family_labels,
                    mask=mask,
                )

                self.logger.info("Inferring latent regimes (HMM)...")
                try:
                    regime_result = inference.infer_regimes(
                        family_features=family_features,
                        universe_features=universe_features,
                        benchmark_features=bench_features,
                        method=InferenceMethod.HMM,
                    )
                    self.logger.info(f"Log-likelihood: {regime_result.log_likelihood:.2f}")
                    self.logger.info(f"BIC: {regime_result.bic:.2f}")
                except Exception as e:
                    self.logger.warning(f"HMM failed ({e}), using temporal clustering...")
                    regime_result = inference.infer_regimes(
                        family_features=family_features,
                        universe_features=universe_features,
                        benchmark_features=bench_features,
                        method=InferenceMethod.TEMPORAL_CLUSTERING,
                    )

                # Regime distribution
                self.logger.info("Regime distribution:")
                regime_counts = regime_result.regime_labels.value_counts().sort_index()
                for regime_id, count in regime_counts.items():
                    name = regime_result.regime_names.get(regime_id, f"regime_{regime_id}")
                    pct = count / len(regime_result.regime_labels) * 100
                    self.logger.info(f"  {regime_id} ({name}): {count} days ({pct:.1f}%)")

                # Save outputs
                regime_result.regime_labels.to_csv(regimes_dir / 'labels.csv')
                regime_result.regime_probabilities.to_parquet(regimes_dir / 'probabilities.parquet')
                self.logger.info("Saved regime_labels.csv and regime_probabilities.parquet")

                # Conditional analysis
                self.logger.info("Analyzing benchmark behavior by regime...")
                conditional = inference.analyze_benchmark_conditional(
                    benchmark_weights=benchmark_weights,
                    family_labels=family_labels,
                    regime_labels=regime_result.regime_labels,
                    algo_returns=returns_matrix,
                )

                # Generate report
                regime_report = inference.generate_report(regime_result, conditional)
                with open(regimes_dir / 'report.txt', 'w', encoding='utf-8') as f:
                    f.write(regime_report)
                self.logger.info("Saved regime_inference_report.txt")

                # Save conditional analysis
                conditional_data = {
                    'family_weights_by_regime': conditional.family_weights_by_regime,
                    'family_overweight_by_regime': conditional.family_overweight_by_regime,
                    'family_returns_by_regime': conditional.family_returns_by_regime,
                }
                with open(regimes_dir / 'benchmark_conditional.json', 'w', encoding='utf-8') as f:
                    json.dump(conditional_data, f, indent=2, default=str)
                self.logger.info("Saved benchmark_conditional_analysis.json")

                conditional.selection_probability.to_csv(regimes_dir / 'family_selection_prob.csv')
                self.logger.info("Saved family_selection_probability.csv")

                results['regime_inference'] = {
                    'n_regimes': args.n_regimes,
                    'n_families': args.n_families,
                    'regime_names': regime_result.regime_names,
                    'log_likelihood': regime_result.log_likelihood,
                    'bic': regime_result.bic,
                    'silhouette': regime_result.silhouette_temporal,
                }

        gc.collect()

        # ==================================================================
        # STEP 2.4: CORRELATION ANALYSIS
        # ==================================================================
        if not args.skip_correlations:
            with self.step("2.4 Correlation Analysis"):
                from src.analysis.correlation_analyzer import CorrelationAnalyzer

                corr_analyzer = CorrelationAnalyzer()

                # Limit to top algorithms by data availability
                valid_counts = returns_matrix.notna().sum()
                top_algos = valid_counts.nlargest(min(100, len(returns_matrix.columns))).index
                returns_subset = returns_matrix[top_algos]

                self.logger.info(f"Analyzing correlations for top {len(top_algos)} algorithms...")

                # Create correlation directory
                (clustering_dir / 'correlation').mkdir(parents=True, exist_ok=True)

                # Static correlation matrix
                corr_matrix = corr_analyzer.correlation_matrix(returns_subset)
                corr_matrix.to_parquet(clustering_dir / 'correlation' / 'matrix.parquet')
                self.logger.info("Saved clustering/correlation/matrix.parquet")

                # Diversification ratio
                div_ratio = corr_analyzer.diversification_ratio(returns_subset)
                self.logger.info(f"Diversification ratio (EW): {div_ratio:.3f}")

                # Low correlation pairs
                low_corr_pairs = corr_analyzer.get_low_correlation_pairs(returns_subset, threshold=0.3)
                self.logger.info(f"Found {len(low_corr_pairs)} pairs with |corr| < 0.3")

                # Cluster by correlation
                self.logger.info("Clustering algorithms by correlation...")
                n_corr_clusters = min(5, len(top_algos) // 10 + 1)
                corr_clusters = corr_analyzer.cluster_by_correlation(returns_subset, n_clusters=n_corr_clusters)

                corr_cluster_df = pd.Series(corr_clusters, name='corr_cluster')
                corr_cluster_df.to_csv(clustering_dir / 'correlation' / 'clusters.csv')
                self.logger.info("Saved correlation_clusters.csv")

                # Generate correlation report
                regime_labels_for_corr = None
                if not args.skip_inference:
                    try:
                        regime_labels_for_corr = pd.read_csv(
                            regimes_dir / 'labels.csv',
                            index_col=0,
                            parse_dates=True
                        ).squeeze()
                    except Exception:
                        pass

                corr_report = corr_analyzer.generate_correlation_report(
                    returns_matrix=returns_subset,
                    benchmark_returns=benchmark_returns,
                    regime_labels=regime_labels_for_corr,
                )
                with open(clustering_dir / 'correlation' / 'report.txt', 'w', encoding='utf-8') as f:
                    f.write(corr_report)
                self.logger.info("Saved correlation_report.txt")

                results['correlation_analysis'] = {
                    'n_algos_analyzed': len(top_algos),
                    'diversification_ratio': div_ratio,
                    'n_low_corr_pairs': len(low_corr_pairs),
                    'n_corr_clusters': n_corr_clusters,
                }

        gc.collect()

        # ==================================================================
        # STEP 2.5: TEMPORAL CLUSTERING
        # ==================================================================
        if not args.skip_temporal:
            with self.step("2.5 Temporal Clustering"):
                try:
                    from src.analysis.algo_clusterer import TemporalAlgoClusterer, ClusterMethod

                    # Map method string to enum
                    method_map = {
                        'kmeans': ClusterMethod.KMEANS,
                        'gmm': ClusterMethod.GMM,
                        'hierarchical': ClusterMethod.HIERARCHICAL,
                        'dbscan': ClusterMethod.DBSCAN,
                        'hdbscan': ClusterMethod.HDBSCAN,
                    }
                    method = method_map.get(args.clustering_method, ClusterMethod.KMEANS)

                    self.logger.info(f"Running temporal clustering (method={args.clustering_method}, k={args.n_clusters})...")

                    # Create output directory for temporal clusters
                    temporal_output = clustering_dir / 'temporal'
                    temporal_output.mkdir(parents=True, exist_ok=True)

                    clusterer = TemporalAlgoClusterer(
                        returns_matrix=returns_matrix,
                        n_clusters=args.n_clusters,
                        method=method,
                    )

                    # Run clustering with comparison of methods
                    output = clusterer.run(
                        methods=[ClusterMethod.KMEANS, ClusterMethod.GMM],
                        save_path=str(temporal_output),
                    )

                    self.logger.info(f"Analyzed {len(output.weekly_results)} weeks")

                    # Analyze stability
                    stability = clusterer.get_cluster_stability(output, 'cumulative')
                    self.logger.info(f"Mean stability ratio: {stability['stability_ratio'].mean():.3f}")
                    self.logger.info(f"Median cluster changes: {stability['n_cluster_changes'].median():.0f}")

                    # Most stable algorithms
                    most_stable = stability.nlargest(5, 'stability_ratio')
                    self.logger.info("\nMost stable algorithms:")
                    for _, row in most_stable.iterrows():
                        self.logger.info(
                            f"  {row['algo_id']}: stability={row['stability_ratio']:.3f}, "
                            f"dominant={row['dominant_cluster']}"
                        )

                    # Least stable algorithms
                    least_stable = stability.nsmallest(5, 'stability_ratio')
                    self.logger.info("\nLeast stable algorithms:")
                    for _, row in least_stable.iterrows():
                        self.logger.info(
                            f"  {row['algo_id']}: stability={row['stability_ratio']:.3f}, "
                            f"changes={row['n_cluster_changes']}"
                        )

                    # Save stability analysis
                    stability.to_csv(temporal_output / 'cluster_stability.csv', index=False)
                    self.logger.info("Saved cluster_stability.csv")

                    results['temporal_clustering'] = {
                        'n_weeks': len(output.weekly_results),
                        'n_clusters': args.n_clusters,
                        'method': args.clustering_method,
                        'mean_stability': float(stability['stability_ratio'].mean()),
                        'median_changes': float(stability['n_cluster_changes'].median()),
                    }

                except ImportError as e:
                    self.logger.warning(f"Temporal clustering not available: {e}")
                except Exception as e:
                    self.logger.error(f"Temporal clustering failed: {e}")

        # ==================================================================
        # GENERATE SUMMARY REPORT
        # ==================================================================
        self._generate_markdown_report(results, analysis_dir)

        return results

    def _load_phase1_data(self, input_dir: Path) -> dict:
        """Load processed data from Phase 1.

        Uses centralized paths from self.dp (DataPaths) for consistent
        data organization. Falls back to legacy flat structure if new
        paths don't exist.
        """
        data = {}

        # Load returns matrix - try new path first, then legacy
        returns_path = self.dp.algorithms.returns
        if not returns_path.exists():
            # Fallback to legacy flat structure
            returns_path = input_dir / 'algo_returns.parquet'
        if not returns_path.exists():
            raise FileNotFoundError(f"Returns matrix not found: {returns_path}")

        returns_matrix = pd.read_parquet(returns_path)
        returns_matrix = returns_matrix.astype(np.float32, copy=False)
        data['returns_matrix'] = returns_matrix
        self.logger.info(f"Loaded returns matrix: {returns_matrix.shape}")

        # Load benchmark returns - try new path first, then legacy
        bench_path = self.dp.benchmark.daily_returns
        if not bench_path.exists():
            bench_path = input_dir / 'benchmark_daily_returns.csv'
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path, index_col=0, parse_dates=True)
            if 'return' in bench_df.columns:
                data['benchmark_returns'] = bench_df['return']
            else:
                data['benchmark_returns'] = bench_df.iloc[:, 0]
            self.logger.info(f"Loaded benchmark returns: {len(data['benchmark_returns'])} days")

        # Load benchmark weights - try new path first, then legacy
        weights_path = self.dp.benchmark.weights
        if not weights_path.exists():
            weights_path = input_dir / 'benchmark_weights.parquet'
        if weights_path.exists():
            benchmark_weights = pd.read_parquet(weights_path)
            benchmark_weights = benchmark_weights.astype(np.float32, copy=False)
            data['benchmark_weights'] = benchmark_weights
            self.logger.info(f"Loaded benchmark weights: {benchmark_weights.shape}")
        else:
            data['benchmark_weights'] = None
            self.logger.info("Benchmark weights not found")

        return data

    def _generate_markdown_report(self, results: dict, output_dir: Path):
        """Generate markdown summary report."""
        report = f"""# Phase 2: Analysis and Reverse Engineering Summary

**Generated**: {datetime.now().isoformat()}

## Overview

Phase 2 profiles algorithms, reverse engineers the benchmark strategy,
infers latent market regimes, and analyzes correlations.

## 2.1 Algorithm Profiling

| Metric | Value |
|--------|-------|
| Algorithms profiled | {results.get('algo_profiles', {}).get('n_profiled', 'N/A')} |

### Top 10 Algorithms by Sharpe
{', '.join(results.get('algo_profiles', {}).get('top_sharpe', [])[:5])}...

## 2.2 Benchmark Reverse Engineering

| Metric | Value |
|--------|-------|
| Annualized Return | {results.get('benchmark_profile', {}).get('annualized_return', 0):.2%} |
| Annualized Volatility | {results.get('benchmark_profile', {}).get('annualized_volatility', 0):.2%} |
| Sharpe Ratio | {results.get('benchmark_profile', {}).get('sharpe_ratio', 0):.3f} |
| Max Drawdown | {results.get('benchmark_profile', {}).get('max_drawdown', 0):.2%} |
| Rebalance Frequency | {results.get('benchmark_profile', {}).get('rebalance_frequency_days', 'N/A')} days |
| Annualized Turnover | {results.get('benchmark_profile', {}).get('turnover_annualized', 0):.2%} |

## 2.3 Latent Regime Inference

| Metric | Value |
|--------|-------|
| N Regimes | {results.get('regime_inference', {}).get('n_regimes', 'N/A')} |
| N Families | {results.get('regime_inference', {}).get('n_families', 'N/A')} |
| Log-Likelihood | {results.get('regime_inference', {}).get('log_likelihood', 0):.2f} |
| BIC | {results.get('regime_inference', {}).get('bic', 0):.2f} |

## 2.4 Correlation Analysis

| Metric | Value |
|--------|-------|
| Algorithms analyzed | {results.get('correlation_analysis', {}).get('n_algos_analyzed', 'N/A')} |
| Diversification ratio | {results.get('correlation_analysis', {}).get('diversification_ratio', 0):.3f} |
| Low correlation pairs | {results.get('correlation_analysis', {}).get('n_low_corr_pairs', 'N/A')} |

## 2.5 Temporal Clustering

| Metric | Value |
|--------|-------|
| Weeks analyzed | {results.get('temporal_clustering', {}).get('n_weeks', 'N/A')} |
| N Clusters | {results.get('temporal_clustering', {}).get('n_clusters', 'N/A')} |
| Method | {results.get('temporal_clustering', {}).get('method', 'N/A')} |
| Mean stability | {results.get('temporal_clustering', {}).get('mean_stability', 0):.3f} |
| Median changes | {results.get('temporal_clustering', {}).get('median_changes', 0):.1f} |

## Output Files

```
data/processed/analysis/
├── profiles/
│   ├── summary.csv             # Algorithm profile summary
│   └── full.json               # Full profile details
├── benchmark_profile/
│   ├── metrics.json            # Benchmark key metrics
│   └── report.txt              # Benchmark analysis report
├── clustering/
│   ├── behavioral/
│   │   ├── features.csv        # Behavioral features
│   │   └── family_labels.csv   # Algorithm family assignments
│   ├── correlation/
│   │   ├── matrix.parquet      # Correlation matrix
│   │   ├── clusters.csv        # Correlation-based clusters
│   │   └── report.txt          # Correlation analysis
│   └── temporal/
│       ├── cluster_history.parquet
│       └── cluster_stability.csv
└── regimes/
    ├── labels.csv              # Regime assignments
    ├── probabilities.parquet   # Regime probabilities
    ├── report.txt              # Regime inference report
    ├── benchmark_conditional.json
    └── family_selection_prob.csv

data/processed/reports/phase2/
└── SUMMARY.md                  # This report
```

## Next Steps

1. Review regime analysis in `analysis/regimes/report.txt`
2. Use family labels for strategy grouping
3. Proceed to Phase 3: Baseline Backtesting
"""
        # Create reports directory and save
        report_path = self.dp.processed.reports.root / 'phase2' / 'SUMMARY.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info(f"Report saved to {report_path.relative_to(self.dp.processed.root)}")


def main():
    runner = Phase2Runner()
    runner.execute()


if __name__ == '__main__':
    main()
