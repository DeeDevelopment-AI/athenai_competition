"""
Behavioral Cloning (BC) pre-trainer for RL policies.

Generates expert demonstrations from a Phase 3 baseline allocator and
pre-trains the RL policy via supervised learning (MSE loss on the policy mean)
before PPO/SAC/TD3 fine-tuning.

Motivation
----------
RL from scratch requires exploring bad strategies before finding good ones.
BC warm-start seeds the policy near a known-good allocator (e.g., risk_parity,
max_sharpe), dramatically reducing the "cold start" exploration problem.

Limitation
----------
BC is a *ceiling*, not the goal: the RL agent cannot beat the expert if it
only imitates. The warm-start phase is short (10-20 epochs); RL fine-tuning
then improves beyond the expert.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = [
    "equal_weight",
    "risk_parity",
    "min_variance",
    "max_sharpe",
    "momentum",
    "vol_targeting",
]


@dataclass
class BCConfig:
    strategy: str = "risk_parity"  # Which Phase 3 allocator to imitate
    epochs: int = 10               # Supervised training epochs
    lr: float = 1e-3              # Learning rate (higher than RL — one-shot supervised)
    batch_size: int = 256          # Mini-batch size
    lookback: int = 63             # Lookback window for the expert allocator (days)
    n_select: int = 50             # Top-N selection for factor-based allocators
    max_weight: float = 0.40
    min_weight: float = 0.00
    max_turnover: float = 0.30


class BehavioralCloningPretrainer:
    """
    Pre-trains an SB3 policy to imitate a Phase 3 baseline strategy.

    Typical usage in run_phase5.py::

        bc_cfg = BCConfig(strategy="max_sharpe", epochs=10)
        pretrainer = BehavioralCloningPretrainer(bc_cfg)

        # Step 1: generate expert weight time series
        expert_weights = pretrainer.generate_expert_weights(
            algo_returns, train_dates[0], train_dates[1]
        )

        # Step 2: collect (obs, encoded_action) pairs by rolling through the env
        single_env = TradingEnvironment(..., random_start=False)
        demos = pretrainer.collect_demonstrations(single_env, expert_weights, encoder)

        # Step 3: supervised pre-training of the agent's policy
        bc_results = pretrainer.pretrain(agent.model, demos)
    """

    def __init__(self, config: BCConfig):
        if config.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{config.strategy}'. "
                f"Valid options: {_VALID_STRATEGIES}"
            )
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Step 1: Generate expert weight time series
    # ------------------------------------------------------------------

    def generate_expert_weights(
        self,
        algo_returns: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run the expert allocator in a rolling fashion to build a weight time series.

        Args:
            algo_returns: Returns matrix [dates × algos].
            train_start: Start of training period.
            train_end:   End of training period.
            features:    Optional algo features DataFrame (required for factor-based strategies
                         such as max_sharpe and momentum; loaded from Phase 1 if available).

        Returns:
            DataFrame [rebalance_dates × algos] of expert weights.
        """
        allocator = self._create_allocator(features)
        train_returns = algo_returns.loc[train_start:train_end]
        dates = train_returns.index
        lb = self.config.lookback
        n_algos = len(algo_returns.columns)
        prev_weights = np.ones(n_algos, dtype=np.float32) / n_algos
        weight_records: List[pd.Series] = []

        for i in range(lb, len(dates)):
            # Rebalance approximately weekly (every 5 trading days)
            if (i - lb) % 5 != 0:
                continue
            date = dates[i]
            window = train_returns.iloc[i - lb : i]
            try:
                result = allocator.allocate(date, window, prev_weights)
                w = result.weights.astype(np.float32)
                prev_weights = w.copy()
            except Exception as exc:
                self.logger.debug(f"Expert allocator failed at {date}: {exc}")
                w = prev_weights.copy()

            weight_records.append(pd.Series(w, index=window.columns, name=date))

        if not weight_records:
            raise RuntimeError(
                f"Expert allocator '{self.config.strategy}' produced no weights. "
                f"Check train period and lookback ({self.config.lookback} days)."
            )

        expert_df = pd.DataFrame(weight_records)
        # Reindex to full universe; zeros for algos not in the selected set
        expert_df = expert_df.reindex(columns=algo_returns.columns, fill_value=0.0)
        self.logger.info(
            f"Expert weights: {len(expert_df)} rebalancing steps, "
            f"strategy={self.config.strategy}, lookback={self.config.lookback}d"
        )
        return expert_df

    # ------------------------------------------------------------------
    # Step 2: Collect (obs, encoded_action) demonstrations
    # ------------------------------------------------------------------

    def collect_demonstrations(
        self,
        env,
        expert_weights: pd.DataFrame,
        encoder=None,
        hybrid_mode: bool = False,
        max_tilt: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Roll through a non-vectorized TradingEnvironment using expert actions to collect
        (observation, encoded_action) pairs.

        Args:
            env:            Non-vectorized TradingEnvironment (random_start=False).
            expert_weights: DataFrame [rebalance_dates × algos] from generate_expert_weights().
            encoder:        Optional fitted encoder (AlgoUniverseEncoder or FamilyEncoder).
                            If provided, expert weights are projected to the encoder's action space.
            hybrid_mode:    If True, actions are tilts (Δw), not absolute weights.
            max_tilt:       Tilt scale for hybrid mode (±max_tilt).

        Returns:
            Tuple (obs_array [N × obs_dim], action_array [N × action_dim]).
        """
        obs_list: List[np.ndarray] = []
        action_list: List[np.ndarray] = []
        obs, _ = env.reset()
        done = False

        while not done:
            current_date = env.simulator._state.current_date

            # Find the closest expert allocation on or before current_date
            past = expert_weights.index <= current_date
            expert_w = (
                expert_weights[past].iloc[-1].values.astype(np.float32)
                if past.any()
                else expert_weights.iloc[0].values.astype(np.float32)
            )
            # Normalize
            total = expert_w.sum()
            if total > 1e-8:
                expert_w = expert_w / total

            # Compute encoded action for the current mode
            if hybrid_mode and getattr(env, "hybrid_mode", False):
                base_w = env._compute_base_weights()
                # Tilt = (expert - base) / max_tilt, clipped to [-1, 1]
                tilt = (expert_w - base_w)
                tilt_normalized = np.clip(tilt / max(float(max_tilt), 1e-8), -1.0, 1.0)
                action = self._encode(tilt_normalized, encoder)
            else:
                action = self._encode(expert_w, encoder)

            obs_list.append(obs.copy())
            action_list.append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if not obs_list:
            self.logger.warning("No demonstrations collected (episode was empty).")
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        obs_arr = np.array(obs_list, dtype=np.float32)
        act_arr = np.array(action_list, dtype=np.float32)
        self.logger.info(
            f"Demonstrations collected: {len(obs_arr)} steps, "
            f"obs_dim={obs_arr.shape[1]}, action_dim={act_arr.shape[1]}"
        )
        return obs_arr, act_arr

    # ------------------------------------------------------------------
    # Step 3: Supervised pre-training
    # ------------------------------------------------------------------

    def pretrain(
        self,
        model,
        demonstrations: Tuple[np.ndarray, np.ndarray],
    ) -> dict:
        """
        Pre-train the SB3 policy via supervised MSE loss on the policy mean.

        Works for PPO (DiagGaussian), SAC (SquashedGaussian), and TD3 (deterministic actor).
        The policy mean is trained to match the expert action; the RL objective takes over
        during the subsequent fine-tuning phase.

        Args:
            model:          Fitted SB3 model (PPO / SAC / TD3).
            demonstrations: (obs_array, action_array) from collect_demonstrations().

        Returns:
            Dict with 'loss_history' (list of per-epoch average MSE) and 'n_samples'.
        """
        import torch

        obs_arr, act_arr = demonstrations
        if len(obs_arr) == 0:
            self.logger.warning("Empty demonstrations — skipping BC pre-training.")
            return {"loss_history": [], "n_samples": 0}

        policy = model.policy
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.lr)

        obs_t = torch.FloatTensor(obs_arr).to(model.device)
        act_t = torch.FloatTensor(act_arr).to(model.device)
        n = len(obs_t)
        loss_history: List[float] = []

        for epoch in range(self.config.epochs):
            perm = torch.randperm(n, device=model.device)
            epoch_losses: List[float] = []

            for start in range(0, n, self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                obs_b = obs_t[idx]
                act_b = act_t[idx]

                optimizer.zero_grad()

                # Get policy's predicted action distribution
                try:
                    dist = policy.get_distribution(obs_b)
                    # DiagGaussian (PPO) → distribution.mean
                    if hasattr(dist, "distribution") and hasattr(dist.distribution, "mean"):
                        pred = dist.distribution.mean
                    elif hasattr(dist, "mean_actions"):
                        pred = dist.mean_actions
                    else:
                        pred = dist.mode()
                except Exception:
                    # SAC / TD3: use the actor directly
                    if hasattr(policy, "actor"):
                        pred = policy.actor(obs_b)
                    else:
                        pred = policy.predict_values(obs_b)  # fallback

                loss = torch.nn.functional.mse_loss(pred, act_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            avg = float(np.mean(epoch_losses))
            loss_history.append(avg)
            self.logger.info(
                f"BC epoch {epoch + 1}/{self.config.epochs}: loss={avg:.6f}"
            )

        final_loss = loss_history[-1] if loss_history else float("nan")
        self.logger.info(
            f"BC pre-training complete: {n} samples, {self.config.epochs} epochs, "
            f"final_loss={final_loss:.6f}"
        )
        return {"loss_history": loss_history, "n_samples": n}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_allocator(self, features: Optional[pd.DataFrame]):
        """Instantiate the expert baseline allocator by name."""
        from src.baselines import (
            EqualWeightAllocator,
            RiskParityAllocator,
            MinVarianceAllocator,
            MaxSharpeAllocator,
            MomentumAllocator,
            VolTargetingAllocator,
        )

        shared = dict(
            max_weight=self.config.max_weight,
            min_weight=self.config.min_weight,
            max_turnover=self.config.max_turnover,
            rebalance_frequency="weekly",
        )
        name = self.config.strategy.lower().replace("-", "_")

        if name == "equal_weight":
            return EqualWeightAllocator(**shared)
        elif name == "risk_parity":
            return RiskParityAllocator(vol_lookback=self.config.lookback, **shared)
        elif name == "min_variance":
            return MinVarianceAllocator(cov_lookback=self.config.lookback, **shared)

        # Factor-based allocators need features.
        # Do NOT pass lookback_window here — each subclass maps its own lookback
        # param to lookback_window internally; passing it again via **kwargs would
        # cause a "multiple values" TypeError.
        factor_kw = dict(
            selection_param=self.config.n_select,
            features=features,
            **shared,
        )
        if name == "max_sharpe":
            return MaxSharpeAllocator(**factor_kw)
        elif name == "momentum":
            return MomentumAllocator(**factor_kw)
        elif name == "vol_targeting":
            return VolTargetingAllocator(**factor_kw)

        # Unreachable after __init__ validation, but keeps mypy happy
        self.logger.warning(f"Unknown strategy '{name}'; falling back to risk_parity.")
        return RiskParityAllocator(vol_lookback=self.config.lookback, **shared)

    def _encode(self, full_weights: np.ndarray, encoder) -> np.ndarray:
        """Project full-space weights to the encoder's action space."""
        if encoder is None:
            return full_weights.astype(np.float32)
        if hasattr(encoder, "encode_weights"):
            return encoder.encode_weights(full_weights)
        # Encoder does not support encode_weights — use raw (rare fallback)
        self.logger.debug(
            "Encoder has no encode_weights(); using raw weights for BC. "
            "Consider adding encode_weights() to your encoder."
        )
        return full_weights.astype(np.float32)
