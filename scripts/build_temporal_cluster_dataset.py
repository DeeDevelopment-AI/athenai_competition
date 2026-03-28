#!/usr/bin/env python3
"""
Build a processed dataset restricted to a temporal cumulative cluster.

Default behavior:
  - looks at the latest available `cluster_cumulative` assignment
  - scores each cluster by high annualized return and low volatility
  - keeps only algorithms from the best cluster
  - writes a self-contained processed dataset under the chosen output dir

Example:
  python scripts/build_temporal_cluster_dataset.py ^
    --output-dir data/processed/cluster_highret_lowvol
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered processed dataset from temporal cumulative cluster.")
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed",
        help="Root of processed data (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the filtered dataset",
    )
    parser.add_argument(
        "--cluster-id",
        type=str,
        default=None,
        help="Explicit cumulative cluster id to keep. If omitted, auto-select best high-return/low-vol cluster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.processed_root)
    output_dir = Path(args.output_dir)

    returns_path = root / "algorithms" / "returns.parquet"
    weights_path = root / "benchmark" / "weights.parquet"
    benchmark_returns_path = root / "benchmark" / "daily_returns.csv"
    cluster_history_path = root / "analysis" / "clustering" / "temporal" / "cluster_history.csv"
    profiles_path = root / "analysis" / "profiles" / "summary.csv"

    algo_returns = pd.read_parquet(returns_path)
    benchmark_weights = pd.read_parquet(weights_path) if weights_path.exists() else None
    benchmark_returns = pd.read_csv(benchmark_returns_path, index_col=0)
    cluster_history = pd.read_csv(
        cluster_history_path,
        usecols=["week_end", "algo_id", "cluster_cumulative"],
    )
    profiles = pd.read_csv(profiles_path)

    cluster_history = cluster_history[
        cluster_history["cluster_cumulative"].notna()
        & ~cluster_history["cluster_cumulative"].isin(["inactive", "insufficient_data", "error"])
    ].copy()
    latest_week = cluster_history["week_end"].max()
    latest = cluster_history[cluster_history["week_end"] == latest_week].copy()
    latest["cluster_cumulative"] = latest["cluster_cumulative"].astype(str)

    summary = latest.merge(
        profiles[["algo_id", "ann_return", "ann_volatility", "sharpe"]],
        on="algo_id",
        how="left",
    )
    cluster_summary = (
        summary.groupby("cluster_cumulative")
        .agg(
            n_algos=("algo_id", "count"),
            ann_return_mean=("ann_return", "mean"),
            ann_return_median=("ann_return", "median"),
            ann_vol_mean=("ann_volatility", "mean"),
            sharpe_mean=("sharpe", "mean"),
        )
        .sort_values(["ann_return_mean", "ann_vol_mean"], ascending=[False, True])
    )

    cluster_id = str(args.cluster_id) if args.cluster_id is not None else str(cluster_summary.index[0])
    selected_algos = (
        latest.loc[latest["cluster_cumulative"] == cluster_id, "algo_id"]
        .drop_duplicates()
        .tolist()
    )

    if not selected_algos:
        raise ValueError(f"No algorithms found for cluster {cluster_id}")

    selected_cols = [c for c in algo_returns.columns if c in selected_algos]
    filtered_returns = algo_returns[selected_cols].copy()
    filtered_weights = None
    if benchmark_weights is not None:
        filtered_weights = benchmark_weights.reindex(columns=selected_cols, fill_value=0.0).copy()

    (output_dir / "algorithms").mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    filtered_returns.to_parquet(output_dir / "algorithms" / "returns.parquet")
    if filtered_weights is not None:
        filtered_weights.to_parquet(output_dir / "benchmark" / "weights.parquet")
    benchmark_returns.to_csv(output_dir / "benchmark" / "daily_returns.csv")

    pd.DataFrame({"algo_id": filtered_returns.columns}).to_csv(
        output_dir / "meta" / "selected_algos.csv",
        index=False,
    )
    cluster_summary.to_csv(output_dir / "meta" / "cluster_summary.csv")

    manifest = {
        "source_processed_root": str(root.resolve()),
        "latest_week": latest_week,
        "cluster_id": cluster_id,
        "n_algos": int(filtered_returns.shape[1]),
        "selection_rule": "latest cumulative cluster with highest mean ann_return and lowest mean ann_volatility"
        if args.cluster_id is None
        else "explicit cluster_id",
    }
    (output_dir / "meta" / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
