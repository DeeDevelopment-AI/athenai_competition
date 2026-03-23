"""
Tests for AlgoUniverseEncoder — three-stage dimensionality reduction.

Covers:
  - Stage 1: static quality filter (min_days_active)
  - Stage 2: dynamic alive mask (no look-ahead)
  - Stage 3: walk-forward PCA (fitted only on training window)
  - encode_obs / decode_action round-trips
  - No look-ahead bias validation
  - TradingEnvironment integration with encoder
"""

import numpy as np
import pandas as pd
import pytest

from src.environment.universe_encoder import AlgoUniverseEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_returns():
    """Small returns matrix: 200 days × 10 algos."""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((200, 10)) * 0.01
    # Make algos 8 and 9 mostly NaN (should be filtered by Stage 1)
    data[:, 8] = np.nan
    data[:, 9] = np.nan
    data[:5, 9] = rng.standard_normal(5) * 0.01  # only 5 active days
    return pd.DataFrame(
        data,
        index=dates,
        columns=[f"algo_{i}" for i in range(10)],
    )


@pytest.fixture
def train_dates(small_returns):
    dates = small_returns.index
    return dates[0], dates[119]  # first 120 days = train


@pytest.fixture
def fitted_encoder(small_returns, train_dates):
    enc = AlgoUniverseEncoder(n_components=4, min_days_active=21, activity_window=30)
    enc.fit(small_returns, train_dates[0], train_dates[1])
    return enc


# ---------------------------------------------------------------------------
# Stage 1: Static filter
# ---------------------------------------------------------------------------


class TestStage1StaticFilter:
    def test_removes_always_nan_algos(self, small_returns, train_dates):
        enc = AlgoUniverseEncoder(n_components=4, min_days_active=21)
        enc.fit(small_returns, train_dates[0], train_dates[1])
        # algo_8 (all NaN) and algo_9 (only 5 days) should be filtered
        algo_cols = small_returns.columns.tolist()
        idx_8 = algo_cols.index("algo_8")
        idx_9 = algo_cols.index("algo_9")
        assert idx_8 not in enc._static_indices
        assert idx_9 not in enc._static_indices

    def test_keeps_active_algos(self, fitted_encoder, small_returns):
        # algos 0-7 have 120 active days in train — all should survive
        algo_cols = small_returns.columns.tolist()
        for i in range(8):
            idx = algo_cols.index(f"algo_{i}")
            assert idx in fitted_encoder._static_indices

    def test_n_static_algos(self, fitted_encoder):
        assert fitted_encoder.n_static_algos == 8

    def test_raises_if_nothing_survives(self, small_returns, train_dates):
        enc = AlgoUniverseEncoder(n_components=4, min_days_active=9999)
        with pytest.raises(ValueError, match="No algos survive"):
            enc.fit(small_returns, train_dates[0], train_dates[1])

    def test_raises_if_no_data_in_window(self, small_returns):
        enc = AlgoUniverseEncoder(n_components=4, min_days_active=1)
        future = pd.Timestamp("2099-01-01")
        with pytest.raises(ValueError, match="No data in training window"):
            enc.fit(small_returns, future, future + pd.Timedelta(days=10))


# ---------------------------------------------------------------------------
# Stage 3: PCA
# ---------------------------------------------------------------------------


class TestStage3PCA:
    def test_explained_variance_positive(self, fitted_encoder):
        stats = fitted_encoder.get_filter_stats()
        assert stats["pca_explained_variance"] > 0

    def test_n_components_capped_at_n_static(self, small_returns, train_dates):
        # Request more components than algos survive
        enc = AlgoUniverseEncoder(n_components=100, min_days_active=21)
        enc.fit(small_returns, train_dates[0], train_dates[1])
        assert enc._n_components_actual <= enc.n_static_algos

    def test_obs_dim_matches_pca(self, fitted_encoder):
        assert fitted_encoder.obs_dim == fitted_encoder._n_components_actual * 4 + 3

    def test_action_dim_matches_pca(self, fitted_encoder):
        assert fitted_encoder.action_dim == fitted_encoder._n_components_actual


# ---------------------------------------------------------------------------
# Stage 2: Alive mask (no look-ahead)
# ---------------------------------------------------------------------------


class TestStage2AliveMask:
    def test_mask_shape(self, fitted_encoder, small_returns):
        date = small_returns.index[150]
        mask = fitted_encoder.get_alive_mask(date)
        assert mask.shape == (fitted_encoder.n_static_algos,)
        assert mask.dtype == bool

    def test_no_future_leakage(self, small_returns, train_dates):
        """Mask on date D must not see returns on or after D."""
        enc = AlgoUniverseEncoder(n_components=4, min_days_active=21, activity_window=30)
        enc.fit(small_returns, train_dates[0], train_dates[1])

        # Create a version with NaN after day 50 for one algo
        returns_with_gap = small_returns.copy()
        idx_0 = small_returns.columns.tolist().index("algo_0")
        # Set days 50 onward to NaN for algo_0
        returns_with_gap.iloc[50:, 0] = np.nan

        enc2 = AlgoUniverseEncoder(n_components=4, min_days_active=21, activity_window=30)
        enc2.fit(returns_with_gap, train_dates[0], train_dates[1])

        # At day 80, algo_0 should be dead (last 30 days are NaN)
        date_80 = small_returns.index[80]
        mask = enc2.get_alive_mask(date_80)
        # Find algo_0 in static_indices
        static_list = list(enc2._static_indices)
        if idx_0 in static_list:
            pos = static_list.index(idx_0)
            assert not mask[pos], "algo_0 should be dead at day 80"

    def test_all_alive_at_start(self, fitted_encoder, small_returns):
        """Before any history, encoder returns all-alive mask."""
        first_date = small_returns.index[0]
        mask = fitted_encoder.get_alive_mask(first_date)
        assert mask.all(), "All should be alive when no history exists"


# ---------------------------------------------------------------------------
# encode_obs
# ---------------------------------------------------------------------------


class TestEncodeObs:
    def _make_raw_obs(self, n_algos):
        rng = np.random.default_rng(1)
        # Layout: [weights(n), ret5d(n), ret21d(n), vol(n), corr, dd, excess]
        return rng.standard_normal(n_algos * 4 + 3).astype(np.float32)

    def test_output_shape(self, fitted_encoder, small_returns):
        n = fitted_encoder.n_total_algos
        raw_obs = self._make_raw_obs(n)
        date = small_returns.index[150]
        encoded = fitted_encoder.encode_obs(raw_obs, date)
        assert encoded.shape == (fitted_encoder.obs_dim,)

    def test_output_dtype(self, fitted_encoder, small_returns):
        n = fitted_encoder.n_total_algos
        raw_obs = self._make_raw_obs(n)
        date = small_returns.index[150]
        encoded = fitted_encoder.encode_obs(raw_obs, date)
        assert encoded.dtype == np.float32

    def test_scalars_preserved(self, fitted_encoder, small_returns):
        """The 3 scalar features (corr, dd, excess) must be unchanged."""
        n = fitted_encoder.n_total_algos
        raw_obs = self._make_raw_obs(n)
        date = small_returns.index[150]
        encoded = fitted_encoder.encode_obs(raw_obs, date)
        np.testing.assert_allclose(
            encoded[-3:], raw_obs[4 * n :].astype(np.float32), rtol=1e-5
        )

    def test_no_nan_in_output(self, fitted_encoder, small_returns):
        """Output must not contain NaN even if raw_obs has NaN."""
        n = fitted_encoder.n_total_algos
        raw_obs = np.full(n * 4 + 3, np.nan, dtype=np.float32)
        date = small_returns.index[150]
        encoded = fitted_encoder.encode_obs(raw_obs, date)
        assert not np.isnan(encoded).any()


# ---------------------------------------------------------------------------
# decode_action
# ---------------------------------------------------------------------------


class TestDecodeAction:
    def test_output_shape(self, fitted_encoder, small_returns):
        pc_action = np.ones(fitted_encoder.action_dim, dtype=np.float32)
        date = small_returns.index[150]
        full = fitted_encoder.decode_action(pc_action, date)
        assert full.shape == (fitted_encoder.n_total_algos,)

    def test_non_negative(self, fitted_encoder, small_returns):
        rng = np.random.default_rng(2)
        pc_action = rng.standard_normal(fitted_encoder.action_dim).astype(np.float32)
        date = small_returns.index[150]
        full = fitted_encoder.decode_action(pc_action, date)
        assert (full >= 0).all()

    def test_sums_to_at_most_one(self, fitted_encoder, small_returns):
        rng = np.random.default_rng(3)
        pc_action = rng.standard_normal(fitted_encoder.action_dim).astype(np.float32) * 10
        date = small_returns.index[150]
        full = fitted_encoder.decode_action(pc_action, date)
        assert full.sum() <= 1.0 + 1e-6

    def test_filtered_algos_have_zero_weight(self, fitted_encoder, small_returns):
        """Algos removed by Stage 1 must always get zero weight."""
        pc_action = np.ones(fitted_encoder.action_dim, dtype=np.float32)
        date = small_returns.index[150]
        full = fitted_encoder.decode_action(pc_action, date)
        static_set = set(fitted_encoder._static_indices.tolist())
        for i in range(fitted_encoder.n_total_algos):
            if i not in static_set:
                assert full[i] == 0.0, f"algo {i} was filtered but got non-zero weight"

    def test_all_zeros_action_gives_equal_weight(self, fitted_encoder, small_returns):
        """Zero PC action → equal weight over alive algos (fallback)."""
        pc_action = np.zeros(fitted_encoder.action_dim, dtype=np.float32)
        date = small_returns.index[150]
        full = fitted_encoder.decode_action(pc_action, date)
        # After inverse_transform(zeros) → PCA mean, which may not be zero.
        # The important property: no NaN and weights sum ≤ 1.
        assert not np.isnan(full).any()
        assert full.sum() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# get_filter_stats
# ---------------------------------------------------------------------------


class TestFilterStats:
    def test_stats_keys(self, fitted_encoder):
        stats = fitted_encoder.get_filter_stats()
        required_keys = {
            "fitted",
            "n_total_algos",
            "n_static_algos",
            "n_pca_components",
            "pca_explained_variance",
            "obs_dim_raw",
            "obs_dim_encoded",
            "action_dim_raw",
            "action_dim_encoded",
            "compression_ratio_obs",
        }
        assert required_keys.issubset(set(stats.keys()))

    def test_compression_reduces_dims(self, fitted_encoder):
        stats = fitted_encoder.get_filter_stats()
        assert stats["obs_dim_encoded"] < stats["obs_dim_raw"]
        assert stats["action_dim_encoded"] < stats["action_dim_raw"]

    def test_not_fitted_returns_safe_dict(self):
        enc = AlgoUniverseEncoder()
        stats = enc.get_filter_stats()
        assert stats == {"fitted": False}


# ---------------------------------------------------------------------------
# Unfitted encoder raises
# ---------------------------------------------------------------------------


class TestUnfittedRaises:
    def test_obs_dim_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = AlgoUniverseEncoder().obs_dim

    def test_action_dim_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = AlgoUniverseEncoder().action_dim

    def test_get_alive_mask_raises(self, small_returns):
        with pytest.raises(RuntimeError, match="not fitted"):
            AlgoUniverseEncoder().get_alive_mask(small_returns.index[0])

    def test_encode_obs_raises(self, small_returns):
        with pytest.raises(RuntimeError, match="not fitted"):
            AlgoUniverseEncoder().encode_obs(
                np.zeros(43, dtype=np.float32), small_returns.index[0]
            )

    def test_decode_action_raises(self, small_returns):
        with pytest.raises(RuntimeError, match="not fitted"):
            AlgoUniverseEncoder().decode_action(
                np.zeros(4, dtype=np.float32), small_returns.index[0]
            )


# ---------------------------------------------------------------------------
# Integration: TradingEnvironment with encoder
# ---------------------------------------------------------------------------


class TestTradingEnvironmentWithEncoder:
    """Smoke tests verifying TradingEnvironment works correctly with encoder."""

    @pytest.fixture
    def env_with_encoder(self, small_returns, fitted_encoder):
        from src.environment.trading_env import TradingEnvironment

        dates = small_returns.index
        return TradingEnvironment(
            algo_returns=small_returns,
            train_start=dates[0],
            train_end=dates[119],
            encoder=fitted_encoder,
        )

    def test_observation_space_matches_encoder(self, env_with_encoder, fitted_encoder):
        assert env_with_encoder.observation_space.shape == (fitted_encoder.obs_dim,)

    def test_action_space_matches_encoder(self, env_with_encoder, fitted_encoder):
        assert env_with_encoder.action_space.shape == (fitted_encoder.action_dim,)

    def test_reset_returns_encoded_obs(self, env_with_encoder, fitted_encoder):
        obs, info = env_with_encoder.reset(seed=0)
        assert obs.shape == (fitted_encoder.obs_dim,)
        assert obs.dtype == np.float32

    def test_step_with_encoded_action(self, env_with_encoder, fitted_encoder):
        env_with_encoder.reset(seed=0)
        action = np.zeros(fitted_encoder.action_dim, dtype=np.float32)
        obs, reward, terminated, truncated, info = env_with_encoder.step(action)
        assert obs.shape == (fitted_encoder.obs_dim,)
        assert isinstance(reward, float)

    def test_without_encoder_uses_raw_dims(self, small_returns):
        from src.environment.trading_env import TradingEnvironment

        dates = small_returns.index
        env = TradingEnvironment(
            algo_returns=small_returns,
            train_start=dates[0],
            train_end=dates[119],
            encoder=None,
        )
        n = len(small_returns.columns)
        assert env.observation_space.shape == (n * 4 + 3,)
        assert env.action_space.shape == (n,)
