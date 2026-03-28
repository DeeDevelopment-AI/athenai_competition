#!/usr/bin/env python3
"""
=================================================================
PHASE 1: Data Loading, Reconstruction and Feature Engineering
=================================================================
Complete Phase 1 pipeline:
  1.1 Load algorithm OHLC data and benchmark transactions
  1.2 Reconstruct equity curves, returns, positions, weights
  1.3 Calculate rolling features for RL state representation
  1.4 Run asset inference to identify underlying assets

Usage:
  python scripts/run_phase1.py
  python scripts/run_phase1.py --sample 100 --skip-features
  python scripts/run_phase1.py --benchmark-only --no-inference

Options:
  --sample N           Process only N algorithms (for testing)
  --benchmark-only     Process only benchmark products
  --skip-features      Skip feature engineering (faster)
  --no-trim            Disable dead tail trimming
  --no-inference       Skip asset inference
  --no-trade-features  Skip trade-based features
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
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor, trim_dead_tail
from src.data.feature_engineering import FeatureEngineer


class Phase1Runner(PhaseRunner):
    """Phase 1: Data Loading, Reconstruction, and Feature Engineering."""

    phase_name = "Phase 1: Data Loading & Feature Engineering"
    phase_number = 1

    def _run_tag(self) -> str:
        parts = []
        if getattr(self.args, 'sample', None):
            parts.append(f"s{self.args.sample}")
        if getattr(self.args, 'skip_features', False):
            parts.append("nofeat")
        return "_".join(parts)

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 1 specific arguments."""
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Process only N algorithms (for testing)'
        )
        parser.add_argument(
            '--benchmark-only', action='store_true',
            help='Process only benchmark products'
        )
        parser.add_argument(
            '--skip-features', action='store_true',
            help='Skip feature engineering (faster)'
        )
        parser.add_argument(
            '--no-trim', action='store_true',
            help='Disable dead tail trimming'
        )
        parser.add_argument(
            '--no-inference', action='store_true',
            help='Skip asset inference'
        )
        parser.add_argument(
            '--no-trade-features', action='store_true',
            help='Skip trade-based features'
        )
        parser.add_argument(
            '--data-path', type=str, default=None,
            help='Override raw data path'
        )

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 1 pipeline."""
        results = {
            'options': {
                'sample_size': args.sample,
                'benchmark_only': args.benchmark_only,
                'skip_features': args.skip_features,
                'trim_dead': not args.no_trim,
                'run_inference': not args.no_inference,
            }
        }

        # Determine paths
        data_path = Path(args.data_path) if args.data_path else self.dp.raw.root
        output_path = self.get_output_dir()

        # ==================================================================
        # STEP 1.1: DATA LOADING
        # ==================================================================
        with self.step("1.1 Data Loading"):
            loader = DataLoader(raw_path=str(data_path))

            # List available algorithms
            all_algo_ids = loader.list_algorithms()
            self.logger.info(f"Found {len(all_algo_ids):,} algorithm files")
            results['n_algorithms_found'] = len(all_algo_ids)

            # Load benchmark data
            self.logger.info("Loading benchmark data...")
            try:
                benchmark_raw = loader.load_benchmark()
                self.logger.info(f"  Trades: {len(benchmark_raw.trades):,} records")
                self.logger.info(f"  Unique products: {benchmark_raw.trades['productname'].nunique()}")

                trade_start = benchmark_raw.trades['dateOpen'].min()
                trade_end = benchmark_raw.trades['dateClose'].max()
                self.logger.info(f"  Period: {trade_start.date()} to {trade_end.date()}")

                benchmark_products = set(benchmark_raw.trades['productname'].unique())
                results['benchmark'] = {
                    'n_trades': len(benchmark_raw.trades),
                    'n_unique_products': len(benchmark_products),
                    'trade_start': str(trade_start.date()),
                    'trade_end': str(trade_end.date()),
                }
            except Exception as e:
                self.logger.error(f"Failed to load benchmark: {e}")
                benchmark_raw = None
                benchmark_products = set()
                results['benchmark_error'] = str(e)

            # Determine which algorithms to load
            available_algos = set(all_algo_ids)
            benchmark_in_algos = benchmark_products & available_algos
            self.logger.info(f"Benchmark products found: {len(benchmark_in_algos)}/{len(benchmark_products)}")

            if args.benchmark_only:
                algos_to_load = list(benchmark_in_algos)
            else:
                algos_to_load = list(benchmark_in_algos) + [a for a in all_algo_ids if a not in benchmark_in_algos]

            if args.sample and args.sample < len(algos_to_load):
                algos_to_load = algos_to_load[:args.sample]
                self.logger.info(f"Sampling: {args.sample} algorithms")

            # Load algorithms
            algorithms = {}
            load_errors = []
            for i, algo_id in enumerate(algos_to_load):
                try:
                    algo = loader.load_algorithm(algo_id, validate=True)
                    if algo is not None:
                        algorithms[algo_id] = algo
                    if (i + 1) % 1000 == 0:
                        self.logger.info(f"  Loaded {i + 1}/{len(algos_to_load)}...")
                except Exception as e:
                    load_errors.append((algo_id, str(e)))

            self.logger.info(f"Successfully loaded: {len(algorithms):,} algorithms")
            self.logger.info(f"Load errors: {len(load_errors)}")
            results['loading'] = {
                'n_loaded': len(algorithms),
                'n_errors': len(load_errors),
                'n_benchmark_products_loaded': len(benchmark_in_algos & set(algorithms.keys())),
            }

        # ==================================================================
        # STEP 1.2: DATA RECONSTRUCTION
        # ==================================================================
        with self.step("1.2 Data Reconstruction"):
            preprocessor = DataPreprocessor(initial_capital=100.0, resample_freq='D')
            trim_dead = not args.no_trim

            # Process algorithms
            self.logger.info("Processing algorithm OHLC -> returns + equity curves...")
            if trim_dead:
                self.logger.info("  Dead tail trimming: ENABLED")

            processed_algos = {}
            process_errors = []
            n_trimmed = 0
            total_dead_days = 0

            for i, (algo_id, algo_data) in enumerate(algorithms.items()):
                try:
                    processed = preprocessor.process_algorithm(algo_data, trim_dead=trim_dead)
                    processed_algos[algo_id] = processed

                    if trim_dead and processed.trim_info and processed.trim_info.was_trimmed:
                        n_trimmed += 1
                        total_dead_days += processed.trim_info.dead_days

                    if (i + 1) % 1000 == 0:
                        self.logger.info(f"  Processed {i + 1}/{len(algorithms)}...")
                except Exception as e:
                    process_errors.append((algo_id, str(e)))

            self.logger.info(f"Successfully processed: {len(processed_algos):,} algorithms")
            if trim_dead:
                self.logger.info(f"Algorithms trimmed: {n_trimmed} ({n_trimmed/max(len(processed_algos),1)*100:.1f}%)")
                self.logger.info(f"Total dead days removed: {total_dead_days:,}")

            results['processing'] = {
                'n_processed': len(processed_algos),
                'n_errors': len(process_errors),
                'n_trimmed': n_trimmed,
                'total_dead_days': int(total_dead_days),
            }

            # Process benchmark
            benchmark_processed = None
            if benchmark_raw:
                self.logger.info("Processing benchmark transactions -> positions + weights...")
                try:
                    benchmark_processed = preprocessor.process_benchmark(benchmark_raw)
                    self.logger.info(f"  Positions matrix: {benchmark_processed.positions.shape}")
                    self.logger.info(f"  Weights matrix: {benchmark_processed.weights.shape}")

                    turnover = preprocessor.calculate_benchmark_turnover(benchmark_processed.weights)
                    concentration = preprocessor.calculate_benchmark_concentration(benchmark_processed.weights)
                    self.logger.info(f"  Daily turnover: mean={turnover.mean():.4f}")
                    self.logger.info(f"  Concentration (HHI): mean={concentration['hhi'].mean():.4f}")
                except Exception as e:
                    self.logger.error(f"Failed to process benchmark: {e}")
                    results['benchmark_processing_error'] = str(e)

            # Create returns matrix
            if processed_algos:
                returns_matrix = preprocessor.create_returns_matrix(processed_algos, fillna=False)
                self.logger.info(f"Returns matrix: {returns_matrix.shape}")
                coverage = (~returns_matrix.isna()).mean().mean()
                self.logger.info(f"Data coverage: {coverage:.1%}")

                results['returns_matrix'] = {
                    'shape': list(returns_matrix.shape),
                    'date_start': str(returns_matrix.index.min().date()),
                    'date_end': str(returns_matrix.index.max().date()),
                    'coverage': float(coverage),
                }

            # Generate statistics
            if processed_algos:
                stats_df = preprocessor.get_summary_stats(processed_algos)
                results['algo_stats'] = {
                    'n_processed': len(stats_df),
                    'mean_sharpe': float(stats_df['sharpe'].mean()),
                    'median_sharpe': float(stats_df['sharpe'].median()),
                    'mean_ann_return': float(stats_df['annualized_return'].mean()),
                }

        # ==================================================================
        # STEP 1.3: FEATURE ENGINEERING
        # ==================================================================
        if not args.skip_features and processed_algos:
            with self.step("1.3 Feature Engineering"):
                fe = FeatureEngineer(windows=[5, 21, 63])
                self.logger.info(f"Rolling windows: {fe.windows}")

                gc.collect()

                all_algo_ids_list = list(processed_algos.keys())
                self.logger.info(f"Computing features for {len(all_algo_ids_list)} algorithms...")

                all_returns = returns_matrix[all_algo_ids_list].dropna(how='all')
                all_returns = all_returns.astype(np.float32, copy=False)
                self.logger.info(f"Returns matrix: {all_returns.shape} (float32)")

                feature_matrix = fe.build_feature_matrix(
                    all_returns,
                    show_progress=True,
                    use_float32=True,
                    batch_size=300,
                )

                del all_returns
                gc.collect()

                self.logger.info(f"Feature matrix: {feature_matrix.shape}")
                results['features'] = {
                    'feature_matrix_shape': list(feature_matrix.shape),
                    'n_algos_with_features': len(all_algo_ids_list),
                }

                # Save benchmark-only features separately
                benchmark_algo_ids = [a for a in benchmark_products if a in processed_algos]
                if benchmark_algo_ids:
                    benchmark_cols = [c for c in feature_matrix.columns
                                     if any(c.startswith(aid + '_') for aid in benchmark_algo_ids)
                                     or c.startswith('rolling_market')]
                    benchmark_features = feature_matrix[benchmark_cols]

        # ==================================================================
        # STEP 1.4: ASSET INFERENCE
        # ==================================================================
        if not args.no_inference and algorithms:
            with self.step("1.4 Asset Inference"):
                bench_dirs = ['forex', 'indices', 'commodities', 'futures', 'sharadar']
                has_benchmarks = any((data_path / d).exists() for d in bench_dirs)

                if not has_benchmarks:
                    self.logger.warning("No benchmark directories found, skipping inference")
                else:
                    try:
                        from src.analysis import AssetInferenceEngine

                        self.logger.info("Loading benchmarks...")
                        engine = AssetInferenceEngine.from_directory(str(data_path))
                        self.logger.info(f"Loaded {len(engine.bench_returns)} benchmarks")

                        if len(engine.bench_returns) > 0:
                            self.logger.info(f"Running inference on {len(algorithms)} algorithms...")
                            inference_results = []

                            for i, (algo_id, algo_data) in enumerate(algorithms.items()):
                                if (i + 1) % 500 == 0:
                                    self.logger.info(f"  Progress: {i + 1}/{len(algorithms)}...")

                                try:
                                    ohlc = algo_data.ohlc
                                    trimmed_closes, trim_info = trim_dead_tail(ohlc['close'])

                                    if len(trimmed_closes) < 15:
                                        inference_results.append({
                                            'algo_id': algo_id,
                                            'predicted_asset': 'unknown',
                                            'asset_class': 'unknown',
                                            'confidence': 0,
                                        })
                                        continue

                                    trimmed_ohlc = ohlc.loc[trimmed_closes.index]
                                    result = engine.infer(trimmed_ohlc)
                                    inference_results.append({
                                        'algo_id': algo_id,
                                        'predicted_asset': result.predicted_asset,
                                        'asset_class': result.asset_class,
                                        'asset_label': result.asset_label,
                                        'direction': result.direction,
                                        'confidence': result.confidence,
                                        'best_composite': result.best_composite,
                                        'n_exposures': len(result.significant_exposures),
                                    })
                                except Exception:
                                    inference_results.append({
                                        'algo_id': algo_id,
                                        'predicted_asset': 'error',
                                        'asset_class': 'error',
                                        'confidence': 0,
                                    })

                            inference_df = pd.DataFrame(inference_results)
                            conf = inference_df['confidence']

                            # Asset class distribution
                            self.logger.info("\nAsset class distribution:")
                            asset_class_dist = inference_df['asset_class'].value_counts().to_dict()
                            for cls, count in asset_class_dist.items():
                                pct = count / len(inference_df) * 100
                                self.logger.info(f"  {cls:<15} {count:>5} ({pct:.1f}%)")

                            # Confidence distribution
                            confidence_dist = {
                                'very_high_80_100': int((conf >= 80).sum()),
                                'high_60_80': int(((conf >= 60) & (conf < 80)).sum()),
                                'medium_40_60': int(((conf >= 40) & (conf < 60)).sum()),
                                'low_20_40': int(((conf >= 20) & (conf < 40)).sum()),
                                'very_low_0_20': int((conf < 20).sum()),
                            }

                            self.logger.info("\nConfidence distribution:")
                            for bucket, count in confidence_dist.items():
                                pct = count / len(conf) * 100
                                self.logger.info(f"  {bucket:<20} {count:>5} ({pct:.1f}%)")

                            results['inference'] = {
                                'n_inferred': len(inference_df),
                                'asset_class_distribution': asset_class_dist,
                                'confidence_distribution': confidence_dist,
                                'mean_confidence': float(conf.mean()),
                                'median_confidence': float(conf.median()),
                            }
                    except Exception as e:
                        self.logger.error(f"Asset inference failed: {e}")

        # ==================================================================
        # SAVE OUTPUTS (using organized folder structure)
        # ==================================================================
        with self.step("Save Outputs"):
            from src.utils.paths import ensure_parent_dir

            # --- Algorithm data -> data/processed/algorithms/ ---
            algo_dir = self.dp.processed.algorithms.root
            algo_dir.mkdir(parents=True, exist_ok=True)

            if processed_algos:
                stats_df.to_csv(self.dp.algorithms.stats)
                self.logger.info(f"  algorithms/stats.csv ({len(stats_df)} rows)")

                returns_matrix.to_parquet(self.dp.algorithms.returns)
                self.logger.info(f"  algorithms/returns.parquet {returns_matrix.shape}")

            if not args.skip_features and 'feature_matrix' in dir():
                feature_matrix.to_parquet(self.dp.algorithms.features)
                self.logger.info(f"  algorithms/features.parquet {feature_matrix.shape}")

            if not args.no_inference and 'inference_df' in dir():
                inference_df.to_csv(self.dp.algorithms.asset_inference, index=False)
                self.logger.info(f"  algorithms/asset_inference.csv ({len(inference_df)} rows)")

            # --- Benchmark data -> data/processed/benchmark/ ---
            bench_dir = self.dp.processed.benchmark.root
            bench_dir.mkdir(parents=True, exist_ok=True)

            if benchmark_processed:
                benchmark_processed.weights.to_parquet(self.dp.benchmark.weights)
                self.logger.info(f"  benchmark/weights.parquet")

                benchmark_processed.positions.to_parquet(self.dp.benchmark.positions)
                self.logger.info(f"  benchmark/positions.parquet")

                turnover = preprocessor.calculate_benchmark_turnover(benchmark_processed.weights)
                turnover.to_csv(self.dp.benchmark.turnover)
                self.logger.info(f"  benchmark/turnover.csv")

                concentration = preprocessor.calculate_benchmark_concentration(benchmark_processed.weights)
                concentration.to_csv(self.dp.benchmark.concentration)
                self.logger.info(f"  benchmark/concentration.csv")

                if processed_algos:
                    bench_returns = preprocessor.calculate_benchmark_daily_returns(
                        returns_matrix, benchmark_processed.weights
                    )
                    if len(bench_returns) > 0:
                        bench_returns.to_csv(self.dp.benchmark.daily_returns)
                        self.logger.info(f"  benchmark/daily_returns.csv")

            if processed_algos and benchmark_products:
                equity_curves = {}
                for algo_id in benchmark_products:
                    if algo_id in processed_algos:
                        equity_curves[algo_id] = processed_algos[algo_id].equity_curve
                if equity_curves:
                    equity_df = pd.DataFrame(equity_curves)
                    equity_df.to_parquet(self.dp.benchmark.algo_equity)
                    self.logger.info(f"  benchmark/algo_equity.parquet")

            if not args.skip_features and 'benchmark_features' in dir():
                benchmark_features.to_parquet(self.dp.benchmark.algo_features)
                self.logger.info(f"  benchmark/algo_features.parquet")

            # --- Reports -> outputs/data_pipeline/ ---
            reports_dir = self.op.data_pipeline.root
            reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate markdown report
        self._generate_markdown_report(results, output_path)

        return results

    def _generate_markdown_report(self, results: dict, output_path: Path):
        """Generate markdown summary report."""
        report = f"""# Phase 1: Data Loading and Reconstruction Summary

**Generated**: {datetime.now().isoformat()}

## Overview

Phase 1 loads raw algorithm and benchmark data, reconstructs equity curves and returns,
and computes rolling features for the RL state representation.

## 1.1 Data Loading

| Metric | Value |
|--------|-------|
| Files found | {results.get('n_algorithms_found', 'N/A'):,} |
| Successfully loaded | {results.get('loading', {}).get('n_loaded', 'N/A'):,} |
| Load errors | {results.get('loading', {}).get('n_errors', 'N/A')} |
| Benchmark products | {results.get('benchmark', {}).get('n_unique_products', 'N/A')} |

## 1.2 Data Reconstruction

| Metric | Value |
|--------|-------|
| Algorithms processed | {results.get('processing', {}).get('n_processed', 'N/A'):,} |
| Algorithms trimmed | {results.get('processing', {}).get('n_trimmed', 'N/A')} |
| Dead days removed | {results.get('processing', {}).get('total_dead_days', 'N/A'):,} |

### Algorithm Statistics

| Metric | Value |
|--------|-------|
| Mean Sharpe | {results.get('algo_stats', {}).get('mean_sharpe', 0):.3f} |
| Median Sharpe | {results.get('algo_stats', {}).get('median_sharpe', 0):.3f} |
| Mean Ann. Return | {results.get('algo_stats', {}).get('mean_ann_return', 0):.2%} |

## 1.3 Feature Engineering

| Metric | Value |
|--------|-------|
| Feature matrix shape | {results.get('features', {}).get('feature_matrix_shape', 'N/A')} |
| Algorithms with features | {results.get('features', {}).get('n_algos_with_features', 'N/A')} |

## 1.4 Asset Inference

| Metric | Value |
|--------|-------|
| Algorithms inferred | {results.get('inference', {}).get('n_inferred', 'N/A')} |
| Mean confidence | {results.get('inference', {}).get('mean_confidence', 0):.1f}% |
| Median confidence | {results.get('inference', {}).get('median_confidence', 0):.1f}% |

### Confidence Distribution

| Confidence Level | Count | Percentage |
|------------------|-------|------------|
| Very High (80-100%) | {results.get('inference', {}).get('confidence_distribution', {}).get('very_high_80_100', 0)} | {results.get('inference', {}).get('confidence_distribution', {}).get('very_high_80_100', 0) / max(results.get('inference', {}).get('n_inferred', 1), 1) * 100:.1f}% |
| High (60-80%) | {results.get('inference', {}).get('confidence_distribution', {}).get('high_60_80', 0)} | {results.get('inference', {}).get('confidence_distribution', {}).get('high_60_80', 0) / max(results.get('inference', {}).get('n_inferred', 1), 1) * 100:.1f}% |
| Medium (40-60%) | {results.get('inference', {}).get('confidence_distribution', {}).get('medium_40_60', 0)} | {results.get('inference', {}).get('confidence_distribution', {}).get('medium_40_60', 0) / max(results.get('inference', {}).get('n_inferred', 1), 1) * 100:.1f}% |
| Low (20-40%) | {results.get('inference', {}).get('confidence_distribution', {}).get('low_20_40', 0)} | {results.get('inference', {}).get('confidence_distribution', {}).get('low_20_40', 0) / max(results.get('inference', {}).get('n_inferred', 1), 1) * 100:.1f}% |
| Very Low (0-20%) | {results.get('inference', {}).get('confidence_distribution', {}).get('very_low_0_20', 0)} | {results.get('inference', {}).get('confidence_distribution', {}).get('very_low_0_20', 0) / max(results.get('inference', {}).get('n_inferred', 1), 1) * 100:.1f}% |

### Asset Class Distribution

{self._format_asset_class_table(results.get('inference', {}).get('asset_class_distribution', {}))}

## Output Files

```
data/processed/
├── algorithms/
│   ├── returns.parquet         # Daily returns matrix
│   ├── features.parquet        # Rolling features
│   ├── stats.csv               # Algorithm statistics
│   └── asset_inference.csv     # Inferred underlying assets
├── benchmark/
│   ├── weights.parquet         # Daily benchmark weights
│   ├── positions.parquet       # Daily positions
│   ├── daily_returns.csv       # Reconstructed returns
│   ├── turnover.csv            # Turnover metrics
│   ├── concentration.csv       # HHI concentration
│   ├── algo_equity.parquet     # Equity curves (benchmark products)
│   └── algo_features.parquet   # Features (benchmark products)
outputs/data_pipeline/
    ├── PHASE1_SUMMARY.md       # This report
    ├── phase1_results.json     # Results JSON
    └── phase1_metrics.json     # Performance metrics
```

## Next Steps

1. Review `algorithms/stats.csv` for algorithm quality
2. Use `algorithms/features.parquet` for RL training
3. Check `algorithms/asset_inference.csv` for underlying asset analysis
4. Proceed to Phase 2: Analysis and Regime Detection
"""
        # Save to outputs/data_pipeline/
        report_path = self.op.data_pipeline.root / "PHASE1_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        self.logger.info(f"Report saved to {report_path}")

    def _format_asset_class_table(self, asset_class_dist: dict) -> str:
        """Format asset class distribution as markdown table."""
        if not asset_class_dist:
            return "No asset inference data available."

        total = sum(asset_class_dist.values())
        lines = [
            "| Asset Class | Count | Percentage |",
            "|-------------|-------|------------|",
        ]
        for cls, count in sorted(asset_class_dist.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"| {cls} | {count} | {pct:.1f}% |")

        return "\n".join(lines)


def main():
    runner = Phase1Runner()
    runner.execute()


if __name__ == '__main__':
    main()
