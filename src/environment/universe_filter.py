"""
ClusterUniverseFilter: Filter or shape the algorithm universe using Phase 2 cluster scores.

Two modes
---------
- 'hard'  Remove algorithms whose family's median score is below a threshold.
          Applied once before environment creation; reduces the action/obs dimensionality.
- 'soft'  Add a per-step reward bonus proportional to the portfolio's weighted cluster quality.
          Encourages allocating to high-scoring families without hard exclusion.

Family quality is scored using Phase 2 behavioral features (Sharpe, return, Sortino).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClusterFilterConfig:
    mode: str = "hard"           # "hard" or "soft"
    score_metric: str = "sharpe" # "sharpe", "return", "sortino"
    threshold: float = 0.0       # Remove families with median score below this
    bonus_weight: float = 0.001  # Reward bonus scale (soft mode — pre reward_scale)
    min_coverage: float = 0.20   # Min fraction of universe to keep (hard mode safety valve)


class ClusterUniverseFilter:
    """
    Filter/shape the algorithm universe using Phase 2 behavioral cluster quality scores.

    Usage::

        cfg = ClusterFilterConfig(mode="hard", score_metric="sharpe", threshold=0.0)
        f = ClusterUniverseFilter(cfg)
        f.load_cluster_data(family_labels_path, behavioral_features_path)

        # Hard mode: filter returns matrix before creating the RL environment
        filtered_returns, filtered_bw = f.apply_hard_filter(algo_returns, benchmark_weights)

        # Soft mode: add the filter to TradingEnvironment; bonus computed inside step()
        env = TradingEnvironment(..., cluster_filter=f)
    """

    def __init__(self, config: ClusterFilterConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._family_labels: Optional[pd.Series] = None
        self._family_scores: Optional[dict] = None
        # Pre-computed per-algo score for fast lookup in compute_reward_bonus()
        self._algo_family_score: Optional[np.ndarray] = None
        self._algo_columns: Optional[list] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_cluster_data(
        self,
        family_labels_path: Path,
        behavioral_features_path: Path,
    ) -> None:
        """Load Phase 2 family labels and compute cluster quality scores."""
        if not Path(family_labels_path).exists():
            raise FileNotFoundError(f"family_labels not found: {family_labels_path}")
        if not Path(behavioral_features_path).exists():
            raise FileNotFoundError(f"behavioral_features not found: {behavioral_features_path}")

        family_df = pd.read_csv(family_labels_path, index_col=0)
        fam_col = "family" if "family" in family_df.columns else family_df.columns[0]
        self._family_labels = family_df[fam_col].astype(int)

        features_df = pd.read_csv(behavioral_features_path, index_col=0)

        _metric_map = {
            "sharpe": "sharpe",
            "return": "ann_return",
            "sortino": "sortino",
        }
        metric_col = _metric_map.get(self.config.score_metric, "sharpe")
        if metric_col not in features_df.columns:
            candidates = [
                c for c in features_df.columns
                if any(k in c.lower() for k in ("sharpe", "return", "sortino"))
            ]
            if not candidates:
                raise ValueError(
                    f"No score column found (tried '{metric_col}'). "
                    f"Available: {features_df.columns.tolist()}"
                )
            metric_col = candidates[0]
            self.logger.warning(
                f"Column '{_metric_map.get(self.config.score_metric)}' not found; "
                f"using '{metric_col}'."
            )

        common = self._family_labels.index.intersection(features_df.index)
        family_s = self._family_labels.loc[common]
        score_s = features_df.loc[common, metric_col]

        self._family_scores = (
            pd.concat([family_s.rename("family"), score_s.rename("score")], axis=1)
            .groupby("family")["score"]
            .median()
            .to_dict()
        )

        self.logger.info(
            f"Cluster data loaded: {len(self._family_scores)} families, "
            f"metric={metric_col}, mode={self.config.mode}"
        )
        for fid, sc in sorted(self._family_scores.items()):
            n_algos = int((family_s == fid).sum())
            flag = "✓" if sc >= self.config.threshold else "✗"
            self.logger.info(f"  {flag} Family {fid}: {n_algos} algos, {metric_col}={sc:.3f}")

    def prepare_for_env(self, algo_columns: list) -> None:
        """
        Pre-compute per-algo family scores aligned to the environment's column order.
        Call this once after load_cluster_data() and before using as soft-mode filter.
        """
        if self._family_labels is None or self._family_scores is None:
            raise RuntimeError("Call load_cluster_data() first.")
        self._algo_columns = list(algo_columns)
        self._algo_family_score = np.array([
            self._family_scores.get(
                int(self._family_labels[col]) if col in self._family_labels.index else -1,
                0.0,
            )
            for col in algo_columns
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Hard mode
    # ------------------------------------------------------------------

    def apply_hard_filter(
        self,
        algo_returns: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Remove algorithms from families whose median score is below the threshold.

        Returns filtered (algo_returns, benchmark_weights).
        """
        if self._family_labels is None:
            raise RuntimeError("Call load_cluster_data() first.")

        threshold = self.config.threshold
        good_families = {fid for fid, sc in self._family_scores.items() if sc >= threshold}

        if not good_families:
            fallback = float(np.median(list(self._family_scores.values())))
            self.logger.warning(
                f"No families above threshold={threshold}. "
                f"Falling back to median threshold={fallback:.3f}."
            )
            good_families = {fid for fid, sc in self._family_scores.items() if sc >= fallback}

        labels_in_universe = (
            self._family_labels.reindex(algo_returns.columns).dropna().astype(int)
        )
        keep_algos = labels_in_universe[labels_in_universe.isin(good_families)].index.tolist()

        # Safety: enforce minimum universe coverage
        min_keep = max(int(len(algo_returns.columns) * self.config.min_coverage), 1)
        if len(keep_algos) < min_keep:
            algo_scores = labels_in_universe.map(self._family_scores).fillna(-np.inf)
            keep_algos = algo_scores.nlargest(min_keep).index.tolist()
            self.logger.warning(
                f"Hard filter would keep only {len(keep_algos)} algos; "
                f"enforcing min_coverage={self.config.min_coverage:.0%}, keeping {min_keep}."
            )

        filtered_returns = algo_returns[keep_algos]
        filtered_bw = benchmark_weights[keep_algos] if benchmark_weights is not None else None

        self.logger.info(
            f"Hard filter applied: "
            f"{len(keep_algos)}/{len(algo_returns.columns)} algos kept "
            f"({len(good_families)}/{len(self._family_scores)} families above {threshold:.3f})"
        )
        return filtered_returns, filtered_bw

    # ------------------------------------------------------------------
    # Soft mode
    # ------------------------------------------------------------------

    def compute_reward_bonus(self, weights: np.ndarray) -> float:
        """
        Weighted-average family score bonus for soft mode.

        Returns: bonus_weight * Σ(w_i * family_score_i) / Σ(w_i)

        This is pre-reward_scale — the TradingEnvironment multiplies the full reward
        (including this bonus) by reward_scale.

        Args:
            weights: Portfolio weights aligned to the columns passed to prepare_for_env().

        Returns:
            Scalar reward bonus (small positive for good portfolios).
        """
        if self._algo_family_score is None:
            return 0.0
        total = float(weights.sum())
        if total < 1e-8:
            return 0.0
        # Dot product: Σ w_i * score_i
        weighted_score = float(np.dot(weights, self._algo_family_score)) / total
        return float(self.config.bonus_weight * weighted_score)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_family_scores(self) -> dict:
        """Return {family_id: score} mapping."""
        return dict(self._family_scores) if self._family_scores else {}

    def get_included_families(self) -> set:
        """Return set of family IDs above the threshold (for logging/reporting)."""
        if not self._family_scores:
            return set()
        return {fid for fid, sc in self._family_scores.items() if sc >= self.config.threshold}
