#!/usr/bin/env python
"""
Run temporal clustering analysis on algorithm data.

This script clusters algorithms on a weekly basis using three time horizons:
1. Cumulative: From start date to current week
2. Weekly: Current week only
3. Monthly: Current month

Usage:
    python scripts/run_temporal_clustering.py --data-path data/raw --output data/processed/temporal_clusters

Options:
    --data-path: Path to raw data directory (default: data/raw)
    --output: Path to save results (default: data/processed/temporal_clusters)
    --start-date: Start date for analysis (default: 2020-01-01)
    --n-clusters: Number of clusters (default: 5)
    --sample: Number of algorithms to sample (default: all)
    --compare-methods: Compare all clustering methods
    --method: Clustering method (see below)

Clustering Methods:
    Traditional:
    - kmeans: K-Means clustering (fast, spherical clusters)
    - gmm: Gaussian Mixture Model (probabilistic, elliptical clusters)
    - hierarchical: Agglomerative clustering (dendrograms)
    - dbscan: Density-based clustering (detects outliers, no k required)
    - hdbscan: Hierarchical DBSCAN (variable density)

    Deep Learning:
    - autoencoder: Basic autoencoder + K-Means (captures nonlinear patterns)
    - vae: Variational Autoencoder (probabilistic latent space)
    - sparse_ae: Sparse Autoencoder (interpretable sparse features)
    - deep_infomax: Deep InfoMax (mutual information maximization)
    - bigan: Bidirectional GAN (adversarial representation learning)
    - iic: Invariant Information Clustering (direct MI-based clustering)
    - dac: Deep Adaptive Clustering (pairwise similarity learning)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.analysis.algo_clusterer import (
    TemporalAlgoClusterer,
    ClusterMethod,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_returns_matrix(
    data_path: str,
    sample: int = None,
) -> pd.DataFrame:
    """Load and process algorithm returns into a matrix."""
    loader = DataLoader(data_path)
    preprocessor = DataPreprocessor()

    # List available algorithms
    algo_ids = loader.list_algorithms()
    logger.info(f"Found {len(algo_ids)} algorithms")

    if sample and sample < len(algo_ids):
        np.random.seed(42)
        algo_ids = list(np.random.choice(algo_ids, sample, replace=False))
        logger.info(f"Sampling {sample} algorithms")

    # Load and process
    algos = loader.load_all_algorithms(
        algo_ids=algo_ids,
        show_progress=True,
        skip_invalid=True,
    )
    logger.info(f"Loaded {len(algos)} valid algorithms")

    # Process to get returns
    processed = preprocessor.process_all_algorithms(algos, show_progress=True)
    logger.info(f"Processed {len(processed)} algorithms")

    # Create returns matrix
    returns_dict = {
        algo_id: data.returns
        for algo_id, data in processed.items()
    }

    # Align all returns to common index
    returns_matrix = pd.DataFrame(returns_dict)

    logger.info(
        f"Returns matrix: {returns_matrix.shape[0]} dates x {returns_matrix.shape[1]} algos"
    )

    return returns_matrix


def main():
    parser = argparse.ArgumentParser(description='Run temporal clustering analysis')
    parser.add_argument('--data-path', type=str, default='data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--output', type=str, default='data/processed/temporal_clusters',
                        help='Path to save results')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for analysis')
    parser.add_argument('--n-clusters', type=int, default=5,
                        help='Number of clusters')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N algorithms (default: all)')
    parser.add_argument('--compare-methods', action='store_true',
                        help='Compare all clustering methods')
    parser.add_argument('--method', type=str, default='kmeans',
                        choices=['kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan',
                                 'autoencoder', 'vae', 'sparse_ae', 'deep_infomax', 'bigan', 'iic', 'dac'],
                        help='Clustering method (see --help for full list)')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("TEMPORAL CLUSTERING ANALYSIS")
    logger.info("="*60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"N clusters: {args.n_clusters}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Compare methods: {args.compare_methods}")

    # Load data
    logger.info("\n--- Loading Data ---")
    returns_matrix = load_returns_matrix(args.data_path, args.sample)

    # Map method string to enum
    method_map = {
        # Traditional methods
        'kmeans': ClusterMethod.KMEANS,
        'gmm': ClusterMethod.GMM,
        'hierarchical': ClusterMethod.HIERARCHICAL,
        'dbscan': ClusterMethod.DBSCAN,
        'hdbscan': ClusterMethod.HDBSCAN,
        # Deep learning methods
        'autoencoder': ClusterMethod.AUTOENCODER,
        'vae': ClusterMethod.VAE,
        'sparse_ae': ClusterMethod.SPARSE_AE,
        'deep_infomax': ClusterMethod.DEEP_INFOMAX,
        'bigan': ClusterMethod.BIGAN,
        'iic': ClusterMethod.IIC,
        'dac': ClusterMethod.DAC,
    }
    method = method_map[args.method]

    # Initialize clusterer
    logger.info("\n--- Initializing Clusterer ---")
    clusterer = TemporalAlgoClusterer(
        returns_matrix=returns_matrix,
        start_date=args.start_date,
        n_clusters=args.n_clusters,
        method=method,
    )

    # Determine methods to run
    if args.compare_methods:
        methods = [
            # Traditional methods
            ClusterMethod.KMEANS,
            ClusterMethod.GMM,
            ClusterMethod.HIERARCHICAL,
            ClusterMethod.DBSCAN,
            # Deep learning methods
            ClusterMethod.AUTOENCODER,
            ClusterMethod.VAE,
            ClusterMethod.SPARSE_AE,
            ClusterMethod.IIC,
            ClusterMethod.DAC,
        ]
    else:
        methods = [method]

    # Run clustering
    logger.info("\n--- Running Clustering ---")
    output = clusterer.run(
        methods=methods,
        save_path=args.output,
    )

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)

    logger.info(f"\nWeeks analyzed: {len(output.weekly_results)}")
    logger.info(f"Algorithms: {output.params['n_algos']}")

    # Cluster distribution for last week
    if output.weekly_results:
        last_week = output.weekly_results[-1]
        logger.info(f"\nLast week ({last_week.week_end.date()}):")
        logger.info(f"  Active algorithms: {last_week.n_active}")

        for horizon in ['cumulative', 'weekly', 'monthly']:
            cluster_col = getattr(last_week, f'cluster_{horizon}')
            counts = cluster_col.value_counts()
            logger.info(f"\n  {horizon.capitalize()} clusters:")
            for cluster, count in counts.head(10).items():
                logger.info(f"    {cluster}: {count}")

    # Method comparison
    if args.compare_methods and output.method_comparison:
        logger.info("\n--- Method Comparison ---")
        for horizon, df in output.method_comparison.items():
            logger.info(f"\n{horizon.capitalize()}:")
            logger.info(df.to_string())

        logger.info("\nBest methods per horizon:")
        for horizon, method in output.best_methods.items():
            logger.info(f"  {horizon}: {method.value}")

    # Stability analysis
    logger.info("\n--- Cluster Stability ---")
    stability = clusterer.get_cluster_stability(output, 'cumulative')

    logger.info(f"\nMean stability ratio: {stability['stability_ratio'].mean():.3f}")
    logger.info(f"Median cluster changes: {stability['n_cluster_changes'].median():.0f}")

    # Most stable algorithms
    most_stable = stability.nlargest(5, 'stability_ratio')
    logger.info("\nMost stable algorithms:")
    for _, row in most_stable.iterrows():
        logger.info(
            f"  {row['algo_id']}: stability={row['stability_ratio']:.3f}, "
            f"changes={row['n_cluster_changes']}, dominant={row['dominant_cluster']}"
        )

    # Least stable algorithms
    least_stable = stability.nsmallest(5, 'stability_ratio')
    logger.info("\nLeast stable algorithms:")
    for _, row in least_stable.iterrows():
        logger.info(
            f"  {row['algo_id']}: stability={row['stability_ratio']:.3f}, "
            f"changes={row['n_cluster_changes']}"
        )

    # Cluster transitions
    logger.info("\n--- Cluster Transitions ---")
    transitions = clusterer.get_cluster_transitions(output, 'cumulative')
    if len(transitions) > 0:
        logger.info(f"\nTransition matrix (top transitions):")
        # Flatten to show top transitions
        transitions_flat = transitions.stack().sort_values(ascending=False)
        for (from_cluster, to_cluster), count in transitions_flat.head(10).items():
            if from_cluster != to_cluster:
                logger.info(f"  {from_cluster} -> {to_cluster}: {count}")

    logger.info(f"\n--- Results saved to: {args.output} ---")

    # Generate summary report
    report_path = Path(args.output) / 'CLUSTERING_REPORT.md'
    _generate_report(output, clusterer, stability, report_path)
    logger.info(f"Report saved to: {report_path}")

    logger.info("\n" + "="*60)
    logger.info("DONE")
    logger.info("="*60)


def _generate_report(output, clusterer, stability, report_path):
    """Generate markdown report of clustering results."""
    with open(report_path, 'w') as f:
        f.write("# Temporal Clustering Report\n\n")

        f.write("## Parameters\n\n")
        f.write(f"- Start date: {output.params['start_date']}\n")
        f.write(f"- N clusters: {output.params['n_clusters']}\n")
        f.write(f"- Method: {output.params['method']}\n")
        f.write(f"- Weeks analyzed: {output.params['n_weeks']}\n")
        f.write(f"- Algorithms: {output.params['n_algos']}\n\n")

        f.write("## Cluster Stability\n\n")
        f.write(f"- Mean stability ratio: {stability['stability_ratio'].mean():.3f}\n")
        f.write(f"- Median cluster changes: {stability['n_cluster_changes'].median():.0f}\n\n")

        f.write("### Most Stable Algorithms\n\n")
        f.write("| Algorithm | Stability | Changes | Dominant Cluster |\n")
        f.write("|-----------|-----------|---------|------------------|\n")
        for _, row in stability.nlargest(10, 'stability_ratio').iterrows():
            f.write(
                f"| {row['algo_id']} | {row['stability_ratio']:.3f} | "
                f"{int(row['n_cluster_changes'])} | {row['dominant_cluster']} |\n"
            )
        f.write("\n")

        f.write("### Least Stable Algorithms\n\n")
        f.write("| Algorithm | Stability | Changes |\n")
        f.write("|-----------|-----------|--------|\n")
        for _, row in stability.nsmallest(10, 'stability_ratio').iterrows():
            f.write(
                f"| {row['algo_id']} | {row['stability_ratio']:.3f} | "
                f"{int(row['n_cluster_changes'])} |\n"
            )
        f.write("\n")

        if output.method_comparison:
            f.write("## Method Comparison\n\n")
            for horizon, df in output.method_comparison.items():
                f.write(f"### {horizon.capitalize()}\n\n")
                f.write(df.to_markdown())
                f.write("\n\n")

            f.write("### Best Methods\n\n")
            for horizon, method in output.best_methods.items():
                f.write(f"- {horizon}: **{method.value}**\n")
            f.write("\n")

        f.write("## Files Generated\n\n")
        f.write("- `cluster_history.parquet` - Full cluster history\n")
        f.write("- `cluster_history.csv` - Full cluster history (CSV)\n")
        f.write("- `params.json` - Parameters used\n")
        f.write("- `best_methods.json` - Best methods per horizon\n")
        f.write("- `method_comparison_*.csv` - Method comparison tables\n")


if __name__ == '__main__':
    main()
