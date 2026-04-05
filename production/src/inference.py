"""
Production Inference Engine
===========================
Loads trained model and generates portfolio recommendations.
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add parent project to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.trading_env import TradingEnvironment
from src.environment.cost_model import CostModel
from src.environment.constraints import PortfolioConstraints
from src.environment.reward import RewardFunction, RewardType
from src.baselines.risk_parity import RiskParityAllocator

logger = logging.getLogger(__name__)

AGENT_CLASSES = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


class ProductionInferenceEngine:
    """
    Inference engine for production portfolio recommendations.
    """

    def __init__(
        self,
        model_dir: Path,
        agent_type: str = "ppo",
        config: Optional[dict] = None,
    ):
        """
        Initialize inference engine.

        Parameters
        ----------
        model_dir : Path
            Directory containing model checkpoints
        agent_type : str
            Type of agent (ppo, sac, td3)
        config : dict, optional
            Configuration overrides
        """
        self.model_dir = Path(model_dir)
        self.agent_type = agent_type
        self.config = config or {}

        self.model = None
        self.encoder = None
        self.vec_normalize = None
        self.env = None

        self._load_model()

    def _load_model(self) -> None:
        """Load trained model and associated artifacts."""
        agent_dir = self.model_dir / "checkpoints" / self.agent_type

        # Load model
        model_path = agent_dir / "best_model.zip"
        if not model_path.exists():
            model_path = agent_dir / "final_model.zip"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found in {agent_dir}")

        logger.info(f"Loading model from {model_path}")

        agent_class = AGENT_CLASSES.get(self.agent_type)
        if agent_class is None:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Load without environment first
        self.model = agent_class.load(str(model_path))

        # Load encoder if exists
        encoder_path = agent_dir / "universe_encoder.pkl"
        if encoder_path.exists():
            logger.info(f"Loading encoder from {encoder_path}")
            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)

        # Load VecNormalize stats if exists
        self.vec_normalize_path = agent_dir / "vecnormalize.pkl"

        logger.info("Model loaded successfully")

    def setup_environment(
        self,
        returns: pd.DataFrame,
        initial_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Setup inference environment with current data.

        Parameters
        ----------
        returns : pd.DataFrame
            Recent returns data [dates x assets]
        initial_weights : np.ndarray, optional
            Starting portfolio weights (defaults to equal weight)
        """
        n_assets = returns.shape[1]

        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets

        # Setup constraints
        constraints_cfg = self.config.get("constraints", {})
        constraints = PortfolioConstraints(
            max_weight=constraints_cfg.get("max_weight", 0.40),
            min_weight=constraints_cfg.get("min_weight", 0.0),
            max_turnover=constraints_cfg.get("max_turnover", 0.30),
            max_exposure=constraints_cfg.get("max_exposure", 1.0),
        )

        # Setup cost model
        costs_cfg = self.config.get("costs", {})
        cost_model = CostModel(
            spread_bps=costs_cfg.get("spread_bps", 5),
            slippage_bps=costs_cfg.get("slippage_bps", 2),
            impact_coefficient=costs_cfg.get("market_impact_coef", 0.1),
        )

        # Setup reward (not used in inference but required by env)
        reward_cfg = self.config.get("reward", {})
        reward_fn = RewardFunction(
            reward_type=RewardType.RISK_CALIBRATED_RETURNS,
            cost_penalty_weight=reward_cfg.get("cost_penalty", 1.0),
            turnover_penalty_weight=reward_cfg.get("turnover_penalty", 0.1),
            drawdown_penalty_weight=reward_cfg.get("drawdown_penalty", 0.5),
        )

        # Create benchmark weights (equal weight as placeholder)
        benchmark_weights = pd.DataFrame(
            np.ones((len(returns), n_assets)) / n_assets,
            index=returns.index,
            columns=returns.columns,
        )

        # Create environment
        train_start = returns.index.min()
        train_end = returns.index.max()

        # Re-fit encoder if it exists and assets are different
        encoder_to_use = None
        if self.encoder is not None:
            # Check if we need to re-fit (different number of assets)
            if hasattr(self.encoder, 'n_algos') and self.encoder.n_algos != n_assets:
                logger.warning(
                    f"Asset count mismatch: encoder expects {self.encoder.n_algos}, "
                    f"got {n_assets}. Will use raw observations."
                )
            else:
                encoder_to_use = self.encoder

        self.env = TradingEnvironment(
            algo_returns=returns,
            benchmark_weights=benchmark_weights,
            cost_model=cost_model,
            constraints=constraints,
            reward_function=reward_fn,
            initial_capital=1_000_000.0,
            rebalance_frequency="weekly",
            train_start=train_start,
            train_end=train_end,
            encoder=encoder_to_use,
        )

        # Wrap in DummyVecEnv
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Apply VecNormalize if available
        if self.vec_normalize_path.exists():
            try:
                self.vec_env = VecNormalize.load(
                    str(self.vec_normalize_path),
                    self.vec_env,
                )
                self.vec_env.training = False
                self.vec_env.norm_reward = False
                logger.info("Applied observation normalization from training")
            except Exception as e:
                logger.warning(f"Could not load VecNormalize: {e}")

        logger.info(f"Environment setup complete: {n_assets} assets")

    def get_recommendation(
        self,
        returns: pd.DataFrame,
        current_weights: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> dict:
        """
        Generate portfolio weight recommendation.

        Parameters
        ----------
        returns : pd.DataFrame
            Recent returns data (at least lookback_days)
        current_weights : np.ndarray, optional
            Current portfolio weights
        deterministic : bool
            Use deterministic policy (no exploration)

        Returns
        -------
        dict
            Recommendation with weights, date, and metadata
        """
        # Setup environment if needed
        if self.env is None:
            self.setup_environment(returns, current_weights)

        # Reset environment to get initial observation
        obs, info = self.vec_env.reset()

        # Get action from model
        action, _ = self.model.predict(obs, deterministic=deterministic)

        # Step environment to apply action and get weights
        obs_next, reward, done, infos = self.vec_env.step(action)

        # Extract weights from info
        info = infos[0]
        raw_weights = info.get("weights", action[0])

        # Ensure weights are valid
        asset_names = list(returns.columns)

        if len(raw_weights) != len(asset_names):
            logger.warning(
                f"Weight dimension mismatch: {len(raw_weights)} vs {len(asset_names)} assets. "
                "Using equal weights."
            )
            raw_weights = np.ones(len(asset_names)) / len(asset_names)

        # Apply constraints
        weights = self._apply_constraints(raw_weights)

        # Build recommendation
        recommendation = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "agent_type": self.agent_type,
            "weights": {name: float(w) for name, w in zip(asset_names, weights)},
            "summary": {
                "n_assets": len(asset_names),
                "n_active": int(np.sum(weights > 0.01)),
                "max_weight": float(np.max(weights)),
                "hhi": float(np.sum(weights ** 2)),
            },
        }

        return recommendation

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply portfolio constraints to weights."""
        constraints_cfg = self.config.get("constraints", {})

        # Ensure non-negative
        weights = np.maximum(weights, constraints_cfg.get("min_weight", 0.0))

        # Apply max weight
        max_weight = constraints_cfg.get("max_weight", 0.40)
        weights = np.minimum(weights, max_weight)

        # Normalize to sum to 1
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(len(weights)) / len(weights)

        return weights


class HybridInferenceEngine(ProductionInferenceEngine):
    """
    Hybrid inference: base allocator + RL tilts.
    """

    def __init__(
        self,
        model_dir: Path,
        agent_type: str = "ppo",
        config: Optional[dict] = None,
        base_allocator: str = "risk_parity",
        max_tilt: float = 0.15,
    ):
        super().__init__(model_dir, agent_type, config)
        self.base_allocator = base_allocator
        self.max_tilt = max_tilt
        self._base_allocator_instance = None

    def _get_base_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute base allocation weights."""
        n_assets = returns.shape[1]

        if self.base_allocator == "risk_parity":
            try:
                if self._base_allocator_instance is None:
                    self._base_allocator_instance = RiskParityAllocator(
                        vol_lookback=63,
                        min_weight=0.0,
                        max_weight=0.40,
                        selection_param=len(returns.columns),  # Select all assets
                    )
                # Use the allocate interface
                current_weights = np.ones(n_assets) / n_assets
                date = returns.index[-1]
                result = self._base_allocator_instance.allocate(date, returns, current_weights)
                weights = result.weights

                # Handle NaN values (from inactive assets with no volatility)
                if np.any(np.isnan(weights)):
                    # Replace NaN with 0, renormalize over valid assets
                    weights = np.nan_to_num(weights, nan=0.0)
                    total = weights.sum()
                    if total > 1e-8:
                        weights = weights / total
                    else:
                        # All weights invalid, fall back to equal weight
                        logger.warning("Risk Parity returned all NaN, using equal weight")
                        weights = np.ones(n_assets) / n_assets

                return weights
            except Exception as e:
                logger.warning(f"Risk Parity failed: {e}, using equal weight")
                return np.ones(n_assets) / n_assets

        elif self.base_allocator == "equal_weight":
            return np.ones(n_assets) / n_assets

        else:
            raise ValueError(f"Unknown base allocator: {self.base_allocator}")

    def get_recommendation(
        self,
        returns: pd.DataFrame,
        current_weights: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> dict:
        """
        Generate hybrid recommendation (base + RL tilts).
        """
        # Get base weights
        base_weights = self._get_base_weights(returns)

        # Setup environment with base weights
        if self.env is None:
            self.setup_environment(returns, base_weights)

        # Reset and get observation (VecEnv returns just obs, not (obs, info))
        obs = self.vec_env.reset()

        # Get current date (latest date in returns for recommendation)
        current_date = returns.index[-1]

        # Get action from RL model
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action = action[0] if len(action.shape) > 1 else action

        # Decode action to full portfolio weights
        # FamilyEncoder.decode_action returns full portfolio weights (sum to 1), not tilts
        if self.encoder is not None and hasattr(self.encoder, 'decode_action'):
            rl_weights = self.encoder.decode_action(action, current_date)
        else:
            rl_weights = action

        # Hybrid mode: blend base_weights with rl_weights using max_tilt
        # final = base + (rl - base) * max_tilt = (1 - max_tilt) * base + max_tilt * rl
        # This means max_tilt=0 gives pure base, max_tilt=1 gives pure RL
        tilts = rl_weights - base_weights
        final_weights = base_weights + tilts * self.max_tilt

        # Apply constraints
        final_weights = self._apply_constraints(final_weights)

        # Build recommendation
        asset_names = list(returns.columns)

        # Calculate active threshold dynamically (assets with > 0.1% of equal weight)
        equal_weight = 1.0 / len(asset_names)
        active_threshold = equal_weight * 0.1  # Positions with at least 10% of equal weight

        recommendation = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "agent_type": f"{self.agent_type}_hybrid",
            "base_allocator": self.base_allocator,
            "max_tilt": self.max_tilt,
            "weights": {name: float(w) for name, w in zip(asset_names, final_weights)},
            "base_weights": {name: float(w) for name, w in zip(asset_names, base_weights)},
            "summary": {
                "n_assets": len(asset_names),
                "n_active": int(np.sum(final_weights > active_threshold)),
                "n_overweight": int(np.sum(final_weights > equal_weight * 1.5)),
                "n_underweight": int(np.sum((final_weights > 0) & (final_weights < equal_weight * 0.5))),
                "max_weight": float(np.max(final_weights)),
                "max_weight_pct": float(np.max(final_weights) * 100),
                "hhi": float(np.sum(final_weights ** 2)),
                "max_tilt_applied": float(np.max(np.abs(final_weights - base_weights))),
                "equal_weight_ref": float(equal_weight),
            },
        }

        return recommendation


def save_recommendation(recommendation: dict, output_dir: Path) -> Path:
    """Save recommendation to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = recommendation["date"]
    output_path = output_dir / f"recommendation_{date_str}.json"

    with open(output_path, "w") as f:
        json.dump(recommendation, f, indent=2)

    logger.info(f"Saved recommendation to {output_path}")

    return output_path


def load_latest_recommendation(output_dir: Path) -> Optional[dict]:
    """Load most recent recommendation."""
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return None

    files = sorted(output_dir.glob("recommendation_*.json"), reverse=True)

    if not files:
        return None

    with open(files[0]) as f:
        return json.load(f)
