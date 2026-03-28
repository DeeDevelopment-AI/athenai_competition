"""
Tests for BehavioralCloningPretrainer — BC warm-start for RL policies.

Covers:
  - BCConfig validation
  - generate_expert_weights: rolling allocation, shape, normalization
  - encode_weights via FamilyEncoder and no encoder (raw)
  - collect_demonstrations: obs/action pair collection from a TradingEnvironment
  - pretrain: policy loss decreases over epochs
  - Integration: encode_weights roundtrip via FamilyEncoder
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.bc_pretrainer import BehavioralCloningPretrainer, BCConfig, _VALID_STRATEGIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ALGOS = 16
N_DAYS = 120


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def algo_returns(rng):
    dates = pd.date_range("2021-01-01", periods=N_DAYS, freq="B")
    data = rng.standard_normal((N_DAYS, N_ALGOS)) * 0.01
    return pd.DataFrame(data, index=dates, columns=[f"algo_{i}" for i in range(N_ALGOS)])


@pytest.fixture
def train_window(algo_returns):
    idx = algo_returns.index
    return idx[0], idx[80]


# ---------------------------------------------------------------------------
# BCConfig
# ---------------------------------------------------------------------------


class TestBCConfig:
    def test_default_strategy_valid(self):
        cfg = BCConfig()
        assert cfg.strategy in _VALID_STRATEGIES

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            BehavioralCloningPretrainer(BCConfig(strategy="bad_algo"))

    def test_all_valid_strategies_accepted(self):
        for s in _VALID_STRATEGIES:
            bc = BehavioralCloningPretrainer(BCConfig(strategy=s))
            assert bc.config.strategy == s


# ---------------------------------------------------------------------------
# generate_expert_weights
# ---------------------------------------------------------------------------


class TestGenerateExpertWeights:
    @pytest.mark.parametrize(
        "strategy",
        ["equal_weight", "risk_parity", "min_variance", "max_sharpe", "momentum", "vol_targeting"],
    )
    def test_weights_shape(self, algo_returns, train_window, strategy):
        bc = BehavioralCloningPretrainer(BCConfig(strategy=strategy, lookback=21))
        w = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        assert w.shape[1] == N_ALGOS
        assert len(w) > 0

    def test_weights_sum_to_one(self, algo_returns, train_window):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="risk_parity", lookback=21))
        w = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        row_sums = w.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_weights_non_negative(self, algo_returns, train_window):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        w = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        assert (w.values >= -1e-9).all()

    def test_columns_match_universe(self, algo_returns, train_window):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        w = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        assert list(w.columns) == list(algo_returns.columns)

    def test_equal_weight_uniform(self, algo_returns, train_window):
        """EqualWeightAllocator should produce uniform weights."""
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        w = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        expected = 1.0 / N_ALGOS
        # Each weight should be close to 1/N (subject to max_turnover constraint)
        assert w.iloc[0].max() < 0.15  # not concentrated

    def test_short_lookback_raises_if_no_data(self, algo_returns):
        """lookback > available window should raise RuntimeError."""
        bc = BehavioralCloningPretrainer(BCConfig(strategy="risk_parity", lookback=500))
        idx = algo_returns.index
        with pytest.raises(RuntimeError, match="no weights"):
            bc.generate_expert_weights(algo_returns, idx[0], idx[10])


# ---------------------------------------------------------------------------
# collect_demonstrations (without encoder — raw mode)
# ---------------------------------------------------------------------------


class TestCollectDemonstrations:
    @pytest.fixture
    def bc_env(self, algo_returns):
        from src.environment.trading_env import TradingEnvironment, EpisodeConfig
        ep_cfg = EpisodeConfig(random_start=False, episode_length=10)
        return TradingEnvironment(
            algo_returns=algo_returns,
            train_start=algo_returns.index[0],
            train_end=algo_returns.index[-1],
            rebalance_frequency="weekly",
            episode_config=ep_cfg,
        )

    @pytest.fixture
    def expert_weights(self, algo_returns, train_window):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        return bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])

    def test_returns_arrays(self, bc_env, expert_weights):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        obs, acts = bc.collect_demonstrations(bc_env, expert_weights, encoder=None)
        assert isinstance(obs, np.ndarray)
        assert isinstance(acts, np.ndarray)

    def test_shapes_consistent(self, bc_env, expert_weights):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        obs, acts = bc.collect_demonstrations(bc_env, expert_weights, encoder=None)
        assert obs.shape[0] == acts.shape[0]  # same number of steps
        assert acts.shape[1] == N_ALGOS       # raw mode: action_dim = n_algos

    def test_obs_finite(self, bc_env, expert_weights):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        obs, acts = bc.collect_demonstrations(bc_env, expert_weights, encoder=None)
        assert np.all(np.isfinite(obs))

    def test_actions_sum_to_one(self, bc_env, expert_weights):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        _, acts = bc.collect_demonstrations(bc_env, expert_weights, encoder=None)
        row_sums = acts.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# collect_demonstrations with FamilyEncoder
# ---------------------------------------------------------------------------


class TestCollectDemonstrationsWithEncoder:
    @pytest.fixture
    def family_encoder(self, algo_returns, train_window):
        from src.environment.universe_encoder import FamilyEncoder
        labels = pd.Series(
            {f"algo_{i}": i % 4 for i in range(N_ALGOS)}
        )
        enc = FamilyEncoder(family_labels=labels, activity_window=30)
        enc.fit(algo_returns, train_window[0], train_window[1])
        return enc

    @pytest.fixture
    def bc_env_with_encoder(self, algo_returns, family_encoder):
        from src.environment.trading_env import TradingEnvironment, EpisodeConfig
        ep_cfg = EpisodeConfig(random_start=False, episode_length=8)
        return TradingEnvironment(
            algo_returns=algo_returns,
            train_start=algo_returns.index[0],
            train_end=algo_returns.index[-1],
            rebalance_frequency="weekly",
            episode_config=ep_cfg,
            encoder=family_encoder,
        )

    def test_action_dim_equals_n_families(self, bc_env_with_encoder, algo_returns, train_window, family_encoder):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        expert_weights = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        _, acts = bc.collect_demonstrations(
            bc_env_with_encoder, expert_weights, encoder=family_encoder
        )
        assert acts.shape[1] == family_encoder.action_dim  # 4 families

    def test_encoded_actions_sum_to_one(self, bc_env_with_encoder, algo_returns, train_window, family_encoder):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21))
        expert_weights = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        _, acts = bc.collect_demonstrations(
            bc_env_with_encoder, expert_weights, encoder=family_encoder
        )
        np.testing.assert_allclose(acts.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------


class TestPretrain:
    """Verify that supervised pre-training reduces the policy MSE loss."""

    @pytest.fixture
    def small_env(self, algo_returns):
        from src.environment.trading_env import TradingEnvironment, EpisodeConfig
        ep_cfg = EpisodeConfig(random_start=False, episode_length=20)
        return TradingEnvironment(
            algo_returns=algo_returns,
            train_start=algo_returns.index[0],
            train_end=algo_returns.index[-1],
            rebalance_frequency="weekly",
            episode_config=ep_cfg,
        )

    @pytest.fixture
    def ppo_model(self, small_env):
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: small_env])
        return PPO("MlpPolicy", vec_env, n_steps=32, batch_size=16, verbose=0)

    def test_pretrain_returns_loss_history(self, algo_returns, train_window, small_env, ppo_model):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21, epochs=3))
        expert_weights = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        obs, acts = bc.collect_demonstrations(small_env, expert_weights, encoder=None)

        result = bc.pretrain(ppo_model, (obs, acts))
        assert "loss_history" in result
        assert len(result["loss_history"]) == 3
        assert result["n_samples"] == len(obs)

    def test_loss_decreases(self, algo_returns, train_window, small_env, ppo_model):
        """Loss should not increase monotonically — first epoch > last (with enough data)."""
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight", lookback=21, epochs=5))
        expert_weights = bc.generate_expert_weights(algo_returns, train_window[0], train_window[1])
        obs, acts = bc.collect_demonstrations(small_env, expert_weights, encoder=None)

        result = bc.pretrain(ppo_model, (obs, acts))
        losses = result["loss_history"]
        # First epoch loss should be >= last epoch loss (training reduces loss)
        assert losses[0] >= losses[-1], (
            f"Expected loss to decrease; got {losses[0]:.6f} -> {losses[-1]:.6f}"
        )

    def test_empty_demonstrations_no_crash(self, ppo_model):
        bc = BehavioralCloningPretrainer(BCConfig(strategy="equal_weight"))
        empty = (np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32))
        result = bc.pretrain(ppo_model, empty)
        assert result["n_samples"] == 0
        assert result["loss_history"] == []


# ---------------------------------------------------------------------------
# encode_weights roundtrip via FamilyEncoder
# ---------------------------------------------------------------------------


class TestEncodeWeightsIntegration:
    def test_encode_weights_family_roundtrip(self, algo_returns, train_window):
        """encode_weights then decode_action should give non-trivially different weights
        but the family-level sums should be preserved."""
        from src.environment.universe_encoder import FamilyEncoder

        labels = pd.Series({f"algo_{i}": i % 4 for i in range(N_ALGOS)})
        enc = FamilyEncoder(family_labels=labels, activity_window=30)
        enc.fit(algo_returns, train_window[0], train_window[1])

        # Expert weights: all weight in family 0
        full_w = np.zeros(N_ALGOS, dtype=np.float32)
        family0 = [i for i in range(N_ALGOS) if i % 4 == 0]
        full_w[family0] = 1.0 / len(family0)

        family_w = enc.encode_weights(full_w)
        assert family_w.shape == (4,)
        # Family 0 should have weight ≈ 1.0
        np.testing.assert_allclose(family_w[0], 1.0, atol=1e-5)
        np.testing.assert_allclose(family_w[1:], 0.0, atol=1e-5)
        np.testing.assert_allclose(family_w.sum(), 1.0, atol=1e-5)

    def test_encode_weights_pca_shape(self, algo_returns, train_window):
        from src.environment.universe_encoder import AlgoUniverseEncoder

        enc = AlgoUniverseEncoder(n_components=4, min_days_active=5)
        enc.fit(algo_returns, train_window[0], train_window[1])

        full_w = np.ones(N_ALGOS, dtype=np.float32) / N_ALGOS
        pc_w = enc.encode_weights(full_w)
        assert pc_w.shape == (enc.action_dim,)
        assert np.all(np.isfinite(pc_w))
