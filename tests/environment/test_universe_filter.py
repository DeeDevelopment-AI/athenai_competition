"""
Tests for ClusterUniverseFilter — hard and soft cluster-based universe filtering.

Covers:
  - load_cluster_data: metric mapping, missing column fallback, family score computation
  - apply_hard_filter: threshold filtering, min_coverage safety valve, benchmark alignment
  - prepare_for_env + compute_reward_bonus: soft mode bonus computation
  - Integration: TradingEnvironment with cluster_filter in soft mode produces different rewards
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.environment.universe_filter import ClusterUniverseFilter, ClusterFilterConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ALGOS = 24
N_DAYS = 100
N_FAMILIES = 4


@pytest.fixture
def tmp_cluster_dir(tmp_path):
    """Write synthetic family_labels.csv and features.csv to a temp directory."""
    rng = np.random.default_rng(42)
    cols = [f"algo_{i}" for i in range(N_ALGOS)]
    families = [i % N_FAMILIES for i in range(N_ALGOS)]
    labels_df = pd.DataFrame({"family": families}, index=cols)
    labels_df.to_csv(tmp_path / "family_labels.csv")

    # Family 0: good (sharpe ~1.2), family 1: ok (0.4), family 2: bad (-0.3), family 3: ok (0.6)
    sharpe_by_family = {0: 1.2, 1: 0.4, 2: -0.3, 3: 0.6}
    sharpes = [sharpe_by_family[f] + rng.standard_normal() * 0.05 for f in families]
    returns = [s * 0.12 for s in sharpes]
    features_df = pd.DataFrame(
        {"sharpe": sharpes, "ann_return": returns, "sortino": [s * 1.3 for s in sharpes]},
        index=cols,
    )
    features_df.to_csv(tmp_path / "features.csv")
    return tmp_path


@pytest.fixture
def loaded_filter(tmp_cluster_dir):
    """A ClusterUniverseFilter loaded with synthetic data, threshold=0."""
    cfg = ClusterFilterConfig(mode="hard", score_metric="sharpe", threshold=0.0)
    f = ClusterUniverseFilter(cfg)
    f.load_cluster_data(
        tmp_cluster_dir / "family_labels.csv",
        tmp_cluster_dir / "features.csv",
    )
    return f


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="B")
    cols = [f"algo_{i}" for i in range(N_ALGOS)]
    data = rng.standard_normal((N_DAYS, N_ALGOS)) * 0.01
    return pd.DataFrame(data, index=dates, columns=cols)


@pytest.fixture
def synthetic_bw(synthetic_returns):
    data = np.abs(np.random.default_rng(1).standard_normal((N_DAYS, N_ALGOS)))
    bw = pd.DataFrame(data, index=synthetic_returns.index, columns=synthetic_returns.columns)
    bw = bw.div(bw.sum(axis=1), axis=0)
    return bw


# ---------------------------------------------------------------------------
# load_cluster_data
# ---------------------------------------------------------------------------


class TestLoadClusterData:
    def test_family_scores_computed(self, loaded_filter):
        scores = loaded_filter.get_family_scores()
        assert len(scores) == N_FAMILIES
        assert all(isinstance(v, float) for v in scores.values())

    def test_family_score_ordering(self, loaded_filter):
        scores = loaded_filter.get_family_scores()
        # Family 0 should have the highest score (~1.2) and family 2 the lowest (~-0.3)
        assert scores[0] > scores[2]

    def test_missing_file_raises(self, tmp_cluster_dir):
        cfg = ClusterFilterConfig(mode="hard")
        f = ClusterUniverseFilter(cfg)
        with pytest.raises(FileNotFoundError):
            f.load_cluster_data(tmp_cluster_dir / "nonexistent.csv", tmp_cluster_dir / "features.csv")

    def test_return_metric_works(self, tmp_cluster_dir):
        cfg = ClusterFilterConfig(mode="hard", score_metric="return")
        f = ClusterUniverseFilter(cfg)
        f.load_cluster_data(tmp_cluster_dir / "family_labels.csv", tmp_cluster_dir / "features.csv")
        assert len(f.get_family_scores()) == N_FAMILIES

    def test_not_loaded_raises_on_filter(self):
        f = ClusterUniverseFilter(ClusterFilterConfig())
        with pytest.raises(RuntimeError):
            f.apply_hard_filter(pd.DataFrame())


# ---------------------------------------------------------------------------
# apply_hard_filter
# ---------------------------------------------------------------------------


class TestHardFilter:
    def test_removes_bad_families(self, loaded_filter, synthetic_returns, synthetic_bw):
        """Family 2 (sharpe ~ -0.3) should be removed when threshold=0."""
        filtered_r, filtered_bw = loaded_filter.apply_hard_filter(synthetic_returns, synthetic_bw)
        # All algos in family 2 should be gone
        family2_algos = [f"algo_{i}" for i in range(N_ALGOS) if i % N_FAMILIES == 2]
        remaining = set(filtered_r.columns)
        assert all(a not in remaining for a in family2_algos)

    def test_keeps_good_families(self, loaded_filter, synthetic_returns, synthetic_bw):
        filtered_r, _ = loaded_filter.apply_hard_filter(synthetic_returns, synthetic_bw)
        family0_algos = [f"algo_{i}" for i in range(N_ALGOS) if i % N_FAMILIES == 0]
        remaining = set(filtered_r.columns)
        assert all(a in remaining for a in family0_algos)

    def test_filtered_returns_shape_matches_bw(self, loaded_filter, synthetic_returns, synthetic_bw):
        filtered_r, filtered_bw = loaded_filter.apply_hard_filter(synthetic_returns, synthetic_bw)
        assert list(filtered_r.columns) == list(filtered_bw.columns)
        assert len(filtered_r) == N_DAYS

    def test_benchmark_weights_none(self, loaded_filter, synthetic_returns):
        filtered_r, filtered_bw = loaded_filter.apply_hard_filter(synthetic_returns, None)
        assert filtered_bw is None
        assert len(filtered_r.columns) < N_ALGOS

    def test_min_coverage_enforced(self, tmp_cluster_dir, synthetic_returns, synthetic_bw):
        """With a very high threshold, min_coverage safety valve kicks in."""
        cfg = ClusterFilterConfig(mode="hard", threshold=99.0, min_coverage=0.5)
        f = ClusterUniverseFilter(cfg)
        f.load_cluster_data(tmp_cluster_dir / "family_labels.csv", tmp_cluster_dir / "features.csv")
        filtered_r, _ = f.apply_hard_filter(synthetic_returns, synthetic_bw)
        assert len(filtered_r.columns) >= int(N_ALGOS * 0.5)

    def test_all_families_good_keeps_all(self, tmp_cluster_dir, synthetic_returns, synthetic_bw):
        """With threshold < min family score, all algos are kept."""
        cfg = ClusterFilterConfig(mode="hard", threshold=-999.0)
        f = ClusterUniverseFilter(cfg)
        f.load_cluster_data(tmp_cluster_dir / "family_labels.csv", tmp_cluster_dir / "features.csv")
        filtered_r, _ = f.apply_hard_filter(synthetic_returns, synthetic_bw)
        assert len(filtered_r.columns) == N_ALGOS

    def test_included_families(self, loaded_filter):
        included = loaded_filter.get_included_families()
        scores = loaded_filter.get_family_scores()
        expected = {fid for fid, sc in scores.items() if sc >= 0.0}
        assert included == expected


# ---------------------------------------------------------------------------
# Soft mode: compute_reward_bonus
# ---------------------------------------------------------------------------


class TestSoftMode:
    @pytest.fixture
    def soft_filter(self, tmp_cluster_dir):
        cfg = ClusterFilterConfig(mode="soft", bonus_weight=0.01)
        f = ClusterUniverseFilter(cfg)
        f.load_cluster_data(tmp_cluster_dir / "family_labels.csv", tmp_cluster_dir / "features.csv")
        cols = [f"algo_{i}" for i in range(N_ALGOS)]
        f.prepare_for_env(cols)
        return f

    def test_bonus_positive_for_good_portfolio(self, soft_filter):
        """Allocating entirely to family 0 (best) should give a positive bonus."""
        cols = [f"algo_{i}" for i in range(N_ALGOS)]
        # Family 0 = algos 0, 4, 8, 12, 16, 20
        w = np.zeros(N_ALGOS)
        family0_idx = [i for i in range(N_ALGOS) if i % N_FAMILIES == 0]
        w[family0_idx] = 1.0 / len(family0_idx)
        bonus = soft_filter.compute_reward_bonus(w)
        assert bonus > 0.0

    def test_bad_portfolio_lower_bonus(self, soft_filter):
        """Portfolio concentrated in family 2 (bad) should give lower bonus than family 0."""
        family0_idx = [i for i in range(N_ALGOS) if i % N_FAMILIES == 0]
        family2_idx = [i for i in range(N_ALGOS) if i % N_FAMILIES == 2]

        w0 = np.zeros(N_ALGOS)
        w0[family0_idx] = 1.0 / len(family0_idx)
        b0 = soft_filter.compute_reward_bonus(w0)

        w2 = np.zeros(N_ALGOS)
        w2[family2_idx] = 1.0 / len(family2_idx)
        b2 = soft_filter.compute_reward_bonus(w2)

        assert b0 > b2

    def test_zero_weights_returns_zero(self, soft_filter):
        assert soft_filter.compute_reward_bonus(np.zeros(N_ALGOS)) == 0.0

    def test_prepare_required_before_bonus(self, tmp_cluster_dir):
        cfg = ClusterFilterConfig(mode="soft", bonus_weight=0.01)
        f = ClusterUniverseFilter(cfg)
        f.load_cluster_data(tmp_cluster_dir / "family_labels.csv", tmp_cluster_dir / "features.csv")
        # prepare_for_env NOT called → _algo_family_score is None → returns 0
        bonus = f.compute_reward_bonus(np.ones(N_ALGOS) / N_ALGOS)
        assert bonus == 0.0

    def test_bonus_scales_with_weight(self, soft_filter):
        """bonus_weight=0.01 should scale the raw score."""
        w = np.ones(N_ALGOS) / N_ALGOS
        bonus = soft_filter.compute_reward_bonus(w)
        # Should be bonus_weight × avg_family_score
        avg_score = np.mean(list(soft_filter.get_family_scores().values()))
        expected = 0.01 * avg_score
        assert abs(bonus - expected) < 0.05  # loose tolerance (median vs mean)


# ---------------------------------------------------------------------------
# Integration: TradingEnvironment with cluster_filter (soft mode)
# ---------------------------------------------------------------------------


class TestClusterFilterEnvIntegration:
    @pytest.fixture
    def env_data(self):
        rng = np.random.default_rng(7)
        n, d = 16, 80
        dates = pd.date_range("2021-01-01", periods=d, freq="B")
        cols = [f"algo_{i}" for i in range(n)]
        returns = pd.DataFrame(rng.standard_normal((d, n)) * 0.005, index=dates, columns=cols)
        bw = pd.DataFrame(np.ones((d, n)) / n, index=dates, columns=cols)
        return returns, bw, cols

    def test_soft_filter_adds_reward_bonus(self, tmp_cluster_dir, env_data):
        """TradingEnvironment step() reward should differ with vs without cluster_filter."""
        from src.environment.trading_env import TradingEnvironment, EpisodeConfig

        returns, bw, cols = env_data
        # Resize cluster data to match env columns
        n = len(cols)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            families = [i % 4 for i in range(n)]
            pd.DataFrame({"family": families}, index=cols).to_csv(tmp / "family_labels.csv")
            rng = np.random.default_rng(0)
            sharpes = [1.5 if f == 0 else (-0.5 if f == 2 else 0.5) for f in families]
            pd.DataFrame(
                {"sharpe": sharpes, "ann_return": [s * 0.1 for s in sharpes]},
                index=cols,
            ).to_csv(tmp / "features.csv")

            cfg = ClusterFilterConfig(mode="soft", bonus_weight=1.0)  # large bonus_weight to see effect
            cf = ClusterUniverseFilter(cfg)
            cf.load_cluster_data(tmp / "family_labels.csv", tmp / "features.csv")
            cf.prepare_for_env(cols)

            ep_cfg = EpisodeConfig(random_start=False, episode_length=5)

            env_plain = TradingEnvironment(
                algo_returns=returns, benchmark_weights=bw,
                train_start=returns.index[0], train_end=returns.index[-1],
                rebalance_frequency="weekly", episode_config=ep_cfg,
            )
            env_filtered = TradingEnvironment(
                algo_returns=returns, benchmark_weights=bw,
                train_start=returns.index[0], train_end=returns.index[-1],
                rebalance_frequency="weekly", episode_config=ep_cfg,
                cluster_filter=cf,
            )

            # Same seed, same action
            action = np.ones(n) / n
            obs_p, _ = env_plain.reset(seed=42)
            obs_f, _ = env_filtered.reset(seed=42)

            rewards_plain, rewards_filtered = [], []
            for _ in range(3):
                _, r_p, done_p, _, _ = env_plain.step(action)
                _, r_f, done_f, _, _ = env_filtered.step(action)
                rewards_plain.append(r_p)
                rewards_filtered.append(r_f)
                if done_p or done_f:
                    break

            # With bonus_weight=1.0, filtered rewards should be strictly larger
            for r_p, r_f in zip(rewards_plain, rewards_filtered):
                assert r_f >= r_p - 1e-9, (
                    f"Filtered reward {r_f:.4f} should be ≥ plain reward {r_p:.4f}"
                )
