"""
Select "good" Phase 2 clusters programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


INVALID_CLUSTER_LABELS = {"", "inactive", "insufficient_data", "error", "nan", "none"}


@dataclass
class Phase2ClusterSelectionConfig:
    source: str = "behavioral_family"
    score_mode: str = "return_low_vol"
    top_k: int = 1
    min_cluster_size: int = 25
    min_return: Optional[float] = 0.0
    max_vol: Optional[float] = None
    latest_only: bool = True


class Phase2ClusterSelector:
    def __init__(self, analysis_root: Path):
        self.analysis_root = self._resolve_analysis_root(Path(analysis_root))

    @staticmethod
    def _resolve_analysis_root(path: Path) -> Path:
        path = Path(path)
        direct = path / "clustering"
        snapshot = path / "analysis_snapshot"
        nested = path / "analysis"

        if direct.exists():
            return path
        if (snapshot / "clustering").exists():
            return snapshot
        if (nested / "clustering").exists():
            return nested
        raise FileNotFoundError(
            f"Could not find Phase 2 analysis artifacts under {path}. "
            "Expected one of: <dir>/clustering, <dir>/analysis_snapshot/clustering, <dir>/analysis/clustering"
        )

    def select(self, config: Phase2ClusterSelectionConfig) -> tuple[list[str], pd.DataFrame, list[str]]:
        summary, assignments = self._load_cluster_source(config)
        if summary.empty or assignments.empty:
            raise ValueError(f"No clusters available for source={config.source}")

        filtered = summary.copy()
        if config.min_cluster_size is not None:
            filtered = filtered[filtered["n_algos"] >= config.min_cluster_size]
        if config.min_return is not None:
            filtered = filtered[filtered["ann_return_mean"] >= config.min_return]
        if config.max_vol is not None:
            filtered = filtered[filtered["ann_vol_mean"] <= config.max_vol]
        if filtered.empty:
            raise ValueError("No Phase 2 clusters satisfy the requested filters.")

        filtered = filtered.copy()
        filtered["selection_score"] = filtered.apply(
            lambda row: self._compute_score(row, config.score_mode),
            axis=1,
        )
        filtered = filtered.sort_values(
            ["selection_score", "ann_return_mean", "sharpe_mean"],
            ascending=[False, False, False],
        )

        selected_clusters = filtered.head(max(int(config.top_k), 1)).index.astype(str).tolist()
        selected_algos = (
            assignments.loc[assignments["cluster_id"].isin(selected_clusters), "algo_id"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        return selected_algos, filtered, selected_clusters

    def _compute_score(self, row: pd.Series, score_mode: str) -> float:
        ann_return = float(row.get("ann_return_mean", 0.0))
        ann_vol = float(row.get("ann_vol_mean", 0.0))
        sharpe = float(row.get("sharpe_mean", 0.0))
        sortino = float(row.get("sortino_mean", 0.0))
        if score_mode == "return":
            return ann_return
        if score_mode == "sharpe":
            return sharpe
        if score_mode == "sortino":
            return sortino
        if score_mode == "return_low_vol":
            return ann_return / max(ann_vol, 1e-6)
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    def _load_cluster_source(
        self,
        config: Phase2ClusterSelectionConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if config.source == "behavioral_family":
            return self._load_behavioral_families()

        horizon_map = {
            "temporal_cumulative": "cluster_cumulative",
            "temporal_weekly": "cluster_weekly",
            "temporal_monthly": "cluster_monthly",
        }
        if config.source in horizon_map:
            return self._load_temporal_clusters(horizon_map[config.source], latest_only=config.latest_only)

        raise ValueError(f"Unsupported Phase 2 cluster source: {config.source}")

    def _load_behavioral_families(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        labels_path = self.analysis_root / "clustering" / "behavioral" / "family_labels.csv"
        features_path = self.analysis_root / "clustering" / "behavioral" / "features.csv"

        labels_df = pd.read_csv(labels_path, index_col=0)
        features_df = pd.read_csv(features_path, index_col=0)
        family_col = "family" if "family" in labels_df.columns else labels_df.columns[0]

        assignments = labels_df[[family_col]].rename(columns={family_col: "cluster_id"}).reset_index(names="algo_id")
        assignments["cluster_id"] = assignments["cluster_id"].astype(str)

        merged = assignments.merge(features_df, left_on="algo_id", right_index=True, how="inner")
        vol_col = "ann_volatility" if "ann_volatility" in merged.columns else "ann_vol"
        summary = merged.groupby("cluster_id").agg(
            n_algos=("algo_id", "count"),
            ann_return_mean=("ann_return", "mean"),
            ann_vol_mean=(vol_col, "mean"),
            sharpe_mean=("sharpe", "mean"),
            sortino_mean=("sortino", "mean"),
        )
        return summary, assignments

    def _load_temporal_clusters(
        self,
        cluster_col: str,
        latest_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        history_path = self.analysis_root / "clustering" / "temporal" / "cluster_history.csv"
        profiles_path = self.analysis_root / "profiles" / "summary.csv"

        history_df = pd.read_csv(history_path, usecols=["week_end", "algo_id", cluster_col])
        history_df = history_df.rename(columns={cluster_col: "cluster_id"})
        history_df["cluster_id"] = history_df["cluster_id"].astype(str).str.strip()
        history_df = history_df[~history_df["cluster_id"].str.lower().isin(INVALID_CLUSTER_LABELS)].copy()

        if latest_only and not history_df.empty:
            latest_week = history_df["week_end"].max()
            history_df = history_df[history_df["week_end"] == latest_week].copy()

        assignments = history_df[["algo_id", "cluster_id"]].drop_duplicates()
        profiles_df = pd.read_csv(profiles_path)
        merged = assignments.merge(profiles_df, on="algo_id", how="inner")
        summary = merged.groupby("cluster_id").agg(
            n_algos=("algo_id", "count"),
            ann_return_mean=("ann_return", "mean"),
            ann_vol_mean=("ann_volatility", "mean"),
            sharpe_mean=("sharpe", "mean"),
            sortino_mean=("sortino", "mean"),
        )
        return summary, assignments
