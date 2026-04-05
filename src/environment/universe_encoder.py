"""
AlgoUniverseEncoder: Three-stage dimensionality reduction for the algo universe.

Stage 1 — Static quality filter:
    Remove algos with < min_days_active non-NaN days in the training window.
    Fit once; stable across episodes. 13,513 → ~9,616 algos.

Stage 2 — Dynamic alive mask:
    Zero out algos that had no returns in the last activity_window days.
    Applied at every step using only past data (no look-ahead).
    ~9,616 → ~2,650 active per step.

Stage 3 — Walk-forward PCA:
    Project each per-algo feature vector (weights, ret5d, ret21d, vol) onto the
    principal components of the training-window return matrix.
    Fit on training window only. obs: 9,616*4+4 → n_pca*4+4; action: 9,616 → n_pca.

Look-ahead safeguards:
  - Stage 1: counts only within [train_start, train_end].
  - Stage 2: uses side='left' searchsorted so current_date is excluded.
  - Stage 3: PCA.fit() is called only on training-window rows.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class AlgoUniverseEncoder:
    """
    Encode/decode observations and actions through a three-stage funnel.

    Designed to be:
    - Picklable (required for SubprocVecEnv).
    - Free of look-ahead bias.
    - Backward-compatible: pass encoder=None to TradingEnvironment to use raw dims.

    Usage::

        encoder = AlgoUniverseEncoder(n_components=50, min_days_active=21)
        encoder.fit(algo_returns, train_start, train_end)

        # Encode raw observation from MarketSimulator
        encoded_obs = encoder.encode_obs(raw_obs, current_date)

        # Decode PCA action from agent to full algo weight vector
        full_weights = encoder.decode_action(pc_action, current_date)

        stats = encoder.get_filter_stats()
    """

    def __init__(
        self,
        n_components: int = 50,
        min_days_active: int = 21,
        activity_window: int = 63,
    ):
        """
        Args:
            n_components: Target number of PCA components.  Capped at n_static_algos.
            min_days_active: Minimum non-NaN days in training window to survive Stage 1.
            activity_window: Number of calendar days to look back for Stage 2 alive mask.
        """
        self.n_components = n_components
        self.min_days_active = min_days_active
        self.activity_window = activity_window

        # Populated by fit()
        self._is_fitted: bool = False
        self._n_total_algos: int = 0
        self._static_indices: np.ndarray = np.array([], dtype=np.intp)
        self._n_components_actual: int = 0
        self._pca: Optional[PCA] = None
        # Fast numpy path for alive mask (avoids pandas overhead per step)
        self._returns_index_np: Optional[np.ndarray] = None
        self._is_active_np: Optional[np.ndarray] = None  # (n_days, n_static) bool
        # Pre-extracted PCA matrices for batched transform (avoids sklearn overhead per step)
        self._pca_mean: Optional[np.ndarray] = None       # (n_static,)
        self._pca_components: Optional[np.ndarray] = None  # (n_pca, n_static)
        # Alive mask cache: avoid recomputing within the same date (decode+encode share date)
        self._alive_cache_date: Optional[np.datetime64] = None
        self._alive_cache_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        algo_returns: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> "AlgoUniverseEncoder":
        """
        Fit the encoder using only data in [train_start, train_end].

        Args:
            algo_returns: Full DataFrame [dates × algos] (may include future dates;
                          only [train_start, train_end] is used for fitting).
            train_start:  Start of training window (inclusive).
            train_end:    End of training window (inclusive).

        Returns:
            self
        """
        algo_returns = algo_returns.sort_index()
        self._n_total_algos = len(algo_returns.columns)
        self._returns_index_np = algo_returns.index.values

        # Extract training window (no future leakage)
        train_mask = (algo_returns.index >= train_start) & (algo_returns.index <= train_end)
        train_data = algo_returns.loc[train_mask]

        if len(train_data) == 0:
            raise ValueError(
                f"No data in training window [{train_start.date()}, {train_end.date()}]"
            )

        # ---- Stage 1: Static quality filter ----
        active_days = train_data.notna().sum(axis=0)
        static_mask = (active_days >= self.min_days_active).values
        self._static_indices = np.where(static_mask)[0].astype(np.intp)
        n_static = len(self._static_indices)

        logger.info(
            f"Stage 1 filter: {self._n_total_algos} algos → {n_static} survive "
            f"(≥{self.min_days_active} active days in training window, "
            f"{self._n_total_algos - n_static} removed)"
        )

        if n_static == 0:
            raise ValueError(
                f"No algos survive Stage 1 filter (min_days_active={self.min_days_active}). "
                f"Max active days in window: {int(active_days.max())}"
            )

        # ---- Stage 3: Fit PCA on training returns ----
        self._n_components_actual = min(self.n_components, n_static)
        if self._n_components_actual < self.n_components:
            logger.warning(
                f"PCA n_components capped: {self.n_components} → {self._n_components_actual} "
                f"(only {n_static} static algos)"
            )

        train_static = (
            train_data.iloc[:, self._static_indices]
            .fillna(0.0)
            .values
            .astype(np.float32)
        )  # shape: (n_train_days, n_static)

        self._pca = PCA(n_components=self._n_components_actual, random_state=42)
        self._pca.fit(train_static)

        # Pre-extract PCA matrices as float32 for fast batched transforms (no sklearn overhead)
        self._pca_mean = self._pca.mean_.astype(np.float32)           # (n_static,)
        self._pca_components = self._pca.components_.astype(np.float32)  # (n_pca, n_static)

        # Pre-compute activity bool matrix for fast alive mask (avoids float NaN checks per step)
        # bool is 8× smaller than float32 and .any() on bool is faster than isnan + all on float
        static_vals = (
            algo_returns.iloc[:, self._static_indices]
            .values
        )  # (n_days, n_static) — float, keep original dtype for notna check
        self._is_active_np: np.ndarray = ~np.isnan(static_vals)  # (n_days, n_static) bool
        del static_vals

        # Invalidate alive mask cache
        self._alive_cache_date = None
        self._alive_cache_mask = None

        explained = float(self._pca.explained_variance_ratio_.sum())
        logger.info(
            f"Stage 3 PCA: {n_static} → {self._n_components_actual} components "
            f"({explained:.1%} variance explained on training window)"
        )

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Encoded observation dimension = n_pca*4 + 4 (4 scalars)."""
        self._check_fitted()
        return self._n_components_actual * 4 + 4

    @property
    def action_dim(self) -> int:
        """Encoded action dimension = n_pca."""
        self._check_fitted()
        return self._n_components_actual

    @property
    def n_static_algos(self) -> int:
        """Number of algos after Stage 1 static filter."""
        return int(len(self._static_indices))

    @property
    def n_total_algos(self) -> int:
        """Total algos in the original universe."""
        return self._n_total_algos

    # ------------------------------------------------------------------
    # Stage 2: Dynamic alive mask
    # ------------------------------------------------------------------

    def get_alive_mask(self, current_date: pd.Timestamp) -> np.ndarray:
        """
        Boolean mask of shape (n_static_algos,) indicating which static algos
        had at least one non-NaN return in the last activity_window days
        strictly before current_date.

        No future leakage: uses side='left' so current_date row is excluded.
        Result is cached per date — decode_action + encode_obs share the same date
        within one step, so the mask is computed only once per step.
        """
        self._check_fitted()

        # Cache hit: same date within the same environment step
        date_key = np.datetime64(current_date)
        if self._alive_cache_date is not None and self._alive_cache_date == date_key:
            return self._alive_cache_mask  # type: ignore[return-value]

        # Find the row index for current_date (exclusive — side='left')
        idx = int(self._returns_index_np.searchsorted(date_key, side="left"))
        start_idx = max(0, idx - self.activity_window)

        if start_idx >= idx:
            # No history yet → treat all as alive
            alive = np.ones(self.n_static_algos, dtype=bool)
        else:
            # Bool slice (no allocation for the window itself, .any on bool is fast)
            window = self._is_active_np[start_idx:idx]  # (window, n_static) bool view
            alive = window.any(axis=0)                  # (n_static,) bool — tiny allocation

        # Cache for reuse within this step
        self._alive_cache_date = date_key
        self._alive_cache_mask = alive
        return alive

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode_obs(self, raw_obs: np.ndarray, current_date: pd.Timestamp) -> np.ndarray:
        """
        Encode a raw MarketSimulator observation to PCA space.

        raw_obs layout (from MarketSimulator.get_observation):
            [weights(n), ret5d(n), ret21d(n), vol(n), avg_corr, drawdown, excess]
        where n = n_total_algos.

        Returns:
            Encoded observation of shape (n_pca*4 + 3,).
        """
        self._check_fitted()
        n = self._n_total_algos

        weights = raw_obs[:n]
        ret5d   = raw_obs[n : 2 * n]
        ret21d  = raw_obs[2 * n : 3 * n]
        vol     = raw_obs[3 * n : 4 * n]
        scalars = raw_obs[4 * n :]  # (4,): avg_corr, drawdown, momentum_breadth, vol_regime

        # Stage 1: extract static algos
        w_s   = weights[self._static_indices]
        r5_s  = ret5d[self._static_indices]
        r21_s = ret21d[self._static_indices]
        vol_s = vol[self._static_indices]

        # Stage 2: apply alive mask (zero out inactive algos) + sanitize NaN/Inf
        # get_alive_mask is cached by date — called once even if decode_action ran first
        alive = self.get_alive_mask(current_date)
        # Stack 4 feature vectors as rows: (4, n_static)
        features = np.stack([w_s, r5_s, r21_s, vol_s], axis=0).astype(np.float32)
        np.nan_to_num(features, nan=0.0, copy=False)
        features *= alive  # broadcast alive mask over rows in one shot

        # Stage 3: batched PCA transform — in-place subtraction avoids a 4×n_static temporary
        # transform(X) = (X - mean) @ components.T  →  (4, n_pca)
        features -= self._pca_mean                          # in-place, no extra allocation
        features_pc = features @ self._pca_components.T    # (4, n_pca)

        scalars_clean = np.nan_to_num(scalars.astype(np.float32), nan=0.0)
        encoded = np.concatenate([features_pc.ravel(), scalars_clean])
        return encoded.astype(np.float32)

    def decode_action(
        self, pc_weights: np.ndarray, current_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Decode a PCA-space action to a full algo weight vector.

        Args:
            pc_weights:   (n_pca,) raw action from agent (any scale; will be clipped).
            current_date: Current simulation date for alive mask.

        Returns:
            (n_total_algos,) non-negative weights that sum to ≤ 1.
        """
        self._check_fitted()

        # Reconstruct in static-algo space: inverse_transform(x) = x @ components + mean
        # In-place addition avoids a second (n_static,) temporary allocation
        raw_static = pc_weights.astype(np.float32) @ self._pca_components  # (n_static,)
        raw_static += self._pca_mean                                         # in-place

        # Stage 2: zero out inactive algos
        alive = self.get_alive_mask(current_date)
        raw_static = np.where(alive, raw_static, 0.0)

        # Clip to non-negative (weights can't be short)
        raw_static = np.clip(raw_static, 0.0, None)

        # Normalize so weights sum ≤ 1
        total = float(raw_static.sum())
        if total > 1.0:
            raw_static = raw_static / total
        elif total < 1e-8:
            # Fallback: equal weight over alive algos
            n_alive = int(alive.sum())
            if n_alive > 0:
                raw_static = alive.astype(np.float32) / n_alive

        # Map to full universe (zero weight for filtered-out algos)
        full_weights = np.zeros(self._n_total_algos, dtype=np.float32)
        full_weights[self._static_indices] = raw_static.astype(np.float32)
        return full_weights

    def encode_weights(self, full_weights: np.ndarray) -> np.ndarray:
        """
        Project full algo weights into PCA action space (approximate left-inverse of decode_action).

        Useful for behavioral cloning: maps expert full-space allocations to the compressed
        action representation that the policy outputs.

        Unlike decode_action, no alive mask is applied — the caller should ensure that
        full_weights are already zero for inactive algos if needed.

        Args:
            full_weights: (n_total_algos,) weight vector in the original algo space.

        Returns:
            (n_pca,) PCA-space action vector.
        """
        self._check_fitted()
        # Extract static algos, center by PCA mean, project onto principal components
        w_static = full_weights[self._static_indices].astype(np.float32)
        w_centered = w_static - self._pca_mean                          # (n_static,)
        pc_weights = self._pca_components @ w_centered                  # (n_pca,)
        return pc_weights.astype(np.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_filter_stats(self) -> dict:
        """Return a summary dict of filter pipeline statistics."""
        if not self._is_fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "n_total_algos": self._n_total_algos,
            "n_static_algos": self.n_static_algos,
            "static_filter_ratio": round(self.n_static_algos / max(self._n_total_algos, 1), 4),
            "n_pca_components": self._n_components_actual,
            "pca_explained_variance": round(
                float(self._pca.explained_variance_ratio_.sum()), 4
            ),
            "obs_dim_raw": self._n_total_algos * 4 + 4,
            "obs_dim_encoded": self.obs_dim,
            "action_dim_raw": self._n_total_algos,
            "action_dim_encoded": self.action_dim,
            "compression_ratio_obs": round(
                (self._n_total_algos * 4 + 4) / max(self.obs_dim, 1), 1
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "AlgoUniverseEncoder is not fitted. Call fit(algo_returns, train_start, train_end) first."
            )


class FamilyEncoder:
    """
    Family-based encoder: maps the full algo universe to N-family aggregate signals.

    Stage 1 — Family assignment (static, from pre-computed labels):
        Each algo is assigned to one of N families based on behavioral clustering.
        Fitted once. No look-ahead (assignments computed on training data only).

    Stage 2 — Dynamic alive mask:
        Zero out algos that had no returns in the last activity_window days.
        Applied at every step using only past data (no look-ahead).

    Obs: (N_families × 5) + 4 scalars
        Per family: avg_ret5d, avg_ret21d, avg_vol, avg_current_weight, frac_active
        Scalars: avg_corr, drawdown, momentum_breadth, vol_regime (from MarketSimulator raw obs)

    Action: N_families weights → equal weight within each family among alive algos.

    Interface is identical to AlgoUniverseEncoder (obs_dim, action_dim,
    encode_obs, decode_action, get_filter_stats, get_alive_mask) so TradingEnvironment
    works without any modification.
    """

    def __init__(
        self,
        family_labels: "pd.Series",
        activity_window: int = 63,
    ):
        """
        Args:
            family_labels: pd.Series indexed by algo_id, values are int family ids.
                           Computed from training data only (no look-ahead).
            activity_window: Days to look back for alive mask (default 63).
        """
        self.family_labels = family_labels
        self.activity_window = activity_window

        self._is_fitted: bool = False
        self._n_total_algos: int = 0
        self._n_families: int = 0
        self._family_ids: Optional[np.ndarray] = None  # sorted unique int family ids
        self._algo_to_family: Optional[np.ndarray] = None  # (n_total,) int, -1 = unassigned
        self._family_sizes: Optional[np.ndarray] = None  # (n_families,) total size
        # Per-family masks: (n_families, n_total_algos) bool — precomputed for speed
        self._family_masks: Optional[np.ndarray] = None
        # Activity matrix for alive mask
        self._returns_index_np: Optional[np.ndarray] = None
        self._is_active_np: Optional[np.ndarray] = None  # (n_days, n_total) bool
        # Alive mask cache (per step)
        self._alive_cache_date: Optional[np.datetime64] = None
        self._alive_cache_mask: Optional[np.ndarray] = None
        # Store column names for inference-time alignment
        self._trained_columns: Optional[list] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        algo_returns: "pd.DataFrame",
        train_start: "pd.Timestamp",
        train_end: "pd.Timestamp",
    ) -> "FamilyEncoder":
        """
        Fit the encoder. Only algo_returns.columns and .index are used here —
        no future data leaks because family_labels were assigned externally on
        training data.
        """
        algo_returns = algo_returns.sort_index()
        self._n_total_algos = len(algo_returns.columns)
        self._returns_index_np = algo_returns.index.values
        # Store column names for inference-time alignment
        self._trained_columns = list(algo_returns.columns)

        # Map family labels to column positions
        self._algo_to_family = np.full(self._n_total_algos, -1, dtype=np.intp)
        for i, col in enumerate(algo_returns.columns):
            if col in self.family_labels.index:
                self._algo_to_family[i] = int(self.family_labels[col])

        # Sorted unique family ids (skip -1 = unassigned)
        assigned = self._algo_to_family[self._algo_to_family >= 0]
        self._family_ids = np.array(sorted(set(assigned.tolist())), dtype=np.intp)
        self._n_families = len(self._family_ids)

        # Build per-family boolean masks: (n_families, n_total_algos)
        self._family_masks = np.zeros((self._n_families, self._n_total_algos), dtype=bool)
        self._family_sizes = np.zeros(self._n_families, dtype=np.intp)
        for fi, fam_id in enumerate(self._family_ids):
            mask = self._algo_to_family == fam_id
            self._family_masks[fi] = mask
            self._family_sizes[fi] = int(mask.sum())

        # Pre-compute activity bool array for alive mask
        static_vals = algo_returns.values  # (n_days, n_total_algos) float
        self._is_active_np = ~np.isnan(static_vals).astype(bool)
        del static_vals

        self._alive_cache_date = None
        self._alive_cache_mask = None
        self._is_fitted = True

        n_assigned = int((self._algo_to_family >= 0).sum())
        logger.info(
            f"FamilyEncoder: {self._n_total_algos} algos, "
            f"{n_assigned} assigned to {self._n_families} families, "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}"
        )
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Encoded observation dimension = n_families * 5 + 4 (4 scalars)."""
        self._check_fitted()
        return self._n_families * 5 + 4

    @property
    def action_dim(self) -> int:
        """Encoded action dimension = n_families."""
        self._check_fitted()
        return self._n_families

    @property
    def n_static_algos(self) -> int:
        """Number of assigned algos (compatibility with AlgoUniverseEncoder)."""
        return int((self._algo_to_family >= 0).sum()) if self._is_fitted else 0

    @property
    def n_total_algos(self) -> int:
        return self._n_total_algos

    @property
    def trained_columns(self) -> list:
        """Return the column names from training data (for inference-time alignment)."""
        self._check_fitted()
        return self._trained_columns

    # ------------------------------------------------------------------
    # Stage 2: Dynamic alive mask
    # ------------------------------------------------------------------

    def get_alive_mask(self, current_date: "pd.Timestamp") -> np.ndarray:
        """
        Boolean mask (n_total_algos,) — algos with activity in the last
        activity_window days strictly before current_date.
        Cached per date.
        """
        self._check_fitted()
        date_key = np.datetime64(current_date)
        if self._alive_cache_date is not None and self._alive_cache_date == date_key:
            return self._alive_cache_mask  # type: ignore[return-value]

        idx = int(self._returns_index_np.searchsorted(date_key, side="left"))
        start_idx = max(0, idx - self.activity_window)

        if start_idx >= idx:
            alive = np.ones(self._n_total_algos, dtype=bool)
        else:
            window = self._is_active_np[start_idx:idx]
            alive = window.any(axis=0)

        self._alive_cache_date = date_key
        self._alive_cache_mask = alive
        return alive

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode_obs(self, raw_obs: np.ndarray, current_date: "pd.Timestamp") -> np.ndarray:
        """
        Aggregate raw MarketSimulator obs to family-level features.

        raw_obs layout: [weights(n), ret5d(n), ret21d(n), vol(n), avg_corr, drawdown, momentum_breadth, vol_regime]
        where n = n_total_algos (4 scalars at end).

        Returns (n_families * 5 + 4,):
            For each family: [avg_ret5d, avg_ret21d, avg_vol, avg_weight, frac_active]
            Then scalars: [avg_corr, drawdown, momentum_breadth, vol_regime]
        """
        self._check_fitted()
        n = self._n_total_algos

        weights = raw_obs[:n].astype(np.float32)
        ret5d   = raw_obs[n : 2 * n].astype(np.float32)
        ret21d  = raw_obs[2 * n : 3 * n].astype(np.float32)
        vol     = raw_obs[3 * n : 4 * n].astype(np.float32)
        scalars = raw_obs[4 * n :].astype(np.float32)

        alive = self.get_alive_mask(current_date)

        family_features = np.zeros(self._n_families * 5, dtype=np.float32)
        for fi in range(self._n_families):
            fmask = self._family_masks[fi] & alive
            n_alive_in_family = int(fmask.sum())
            base = fi * 5

            if n_alive_in_family == 0:
                # All zeros for this family — no active algos
                continue

            r5   = ret5d[fmask]
            r21  = ret21d[fmask]
            v    = vol[fmask]
            w    = weights[fmask]

            family_features[base]     = float(np.nanmean(r5))   if np.any(np.isfinite(r5))  else 0.0
            family_features[base + 1] = float(np.nanmean(r21))  if np.any(np.isfinite(r21)) else 0.0
            family_features[base + 2] = float(np.nanmean(v))    if np.any(np.isfinite(v))   else 0.0
            family_features[base + 3] = float(np.nanmean(w))    if np.any(np.isfinite(w))   else 0.0
            family_features[base + 4] = n_alive_in_family / max(self._family_sizes[fi], 1)

        np.nan_to_num(family_features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        scalars_clean = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([family_features, scalars_clean]).astype(np.float32)

    def decode_action(
        self, family_weights: np.ndarray, current_date: "pd.Timestamp"
    ) -> np.ndarray:
        """
        Decode N-family allocation weights to a full algo weight vector.

        Strategy: equal weight within each family among currently alive algos.
        Family budget is proportional to the agent's softmax-normalized output.

        Returns (n_total_algos,) non-negative weights summing to ≤ 1.
        """
        self._check_fitted()

        # Clip and normalize family weights to sum to 1
        fw = np.clip(family_weights.astype(np.float32), 0.0, None)
        total = float(fw.sum())
        if total > 1e-8:
            fw = fw / total
        else:
            fw = np.ones(self._n_families, dtype=np.float32) / self._n_families

        alive = self.get_alive_mask(current_date)
        full_weights = np.zeros(self._n_total_algos, dtype=np.float32)

        # Track families with no alive algos — redistribute their budget
        no_alive_budget = 0.0
        families_with_alive = []
        alive_counts = []

        for fi in range(self._n_families):
            fmask = self._family_masks[fi] & alive
            n_alive = int(fmask.sum())
            if n_alive == 0:
                no_alive_budget += float(fw[fi])
            else:
                families_with_alive.append(fi)
                alive_counts.append(n_alive)

        if len(families_with_alive) == 0:
            # Fallback: equal weight over all alive algos
            n_alive_total = int(alive.sum())
            if n_alive_total > 0:
                full_weights[alive] = 1.0 / n_alive_total
            return full_weights

        # Redistribute no_alive_budget proportionally
        redistributed = fw.copy()
        if no_alive_budget > 1e-8:
            total_active_budget = sum(fw[fi] for fi in families_with_alive)
            if total_active_budget > 1e-8:
                scale = (total_active_budget + no_alive_budget) / total_active_budget
                for fi in families_with_alive:
                    redistributed[fi] = fw[fi] * scale

        # Assign equal weight within each family
        for fi, n_alive in zip(families_with_alive, alive_counts):
            fmask = self._family_masks[fi] & alive
            per_algo_weight = redistributed[fi] / n_alive
            full_weights[fmask] = per_algo_weight

        # Normalize to ensure sum ≤ 1 (floating point safety)
        total = float(full_weights.sum())
        if total > 1.0:
            full_weights /= total

        return full_weights

    def encode_weights(self, full_weights: np.ndarray) -> np.ndarray:
        """
        Project full algo weights into family action space (approximate left-inverse of decode_action).

        Family weight f = sum of the individual algo weights belonging to family f,
        then normalized to sum to 1.  Useful for behavioral cloning: maps expert
        full-space allocations to the N-family action vector the policy outputs.

        Args:
            full_weights: (n_total_algos,) weight vector in the original algo space.

        Returns:
            (n_families,) family-space weight vector.
        """
        self._check_fitted()
        # Vectorized: family_weight[f] = Σ full_weights[family_masks[f]]
        family_weights = (
            self._family_masks.astype(np.float32) @ full_weights.astype(np.float32)
        )
        total = float(family_weights.sum())
        if total > 1e-8:
            family_weights = family_weights / total
        return family_weights.astype(np.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_filter_stats(self) -> dict:
        """Return a summary dict of encoder statistics."""
        if not self._is_fitted:
            return {"fitted": False, "encoder_type": "family"}
        return {
            "fitted": True,
            "encoder_type": "family",
            "n_total_algos": self._n_total_algos,
            "n_families": self._n_families,
            "n_assigned_algos": self.n_static_algos,
            "family_sizes": {
                int(fid): int(sz)
                for fid, sz in zip(self._family_ids, self._family_sizes)
            },
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FamilyEncoder is not fitted. Call fit(algo_returns, train_start, train_end) first."
            )
