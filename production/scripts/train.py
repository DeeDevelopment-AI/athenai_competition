#!/usr/bin/env python
"""
Train Production Agent
======================
Trains an RL agent on production asset data.

Usage:
    python production/scripts/train.py
    python production/scripts/train.py --agent ppo --timesteps 100000
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Setup paths - use resolve() to get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PRODUCTION_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import from project root's src (the main RL environment)
from src.environment.trading_env import TradingEnvironment
from src.environment.cost_model import CostModel
from src.environment.constraints import PortfolioConstraints
from src.environment.reward import RewardFunction, RewardType
from src.environment.universe_encoder import AlgoUniverseEncoder

# GPU environment (optional, for faster training)
try:
    from src.environment.gpu_vec_env import GPUVecTradingEnv, GPUEnvConfig
    from src.environment.universe_encoder import FamilyEncoder
    HAS_GPU_ENV = True
except ImportError:
    HAS_GPU_ENV = False

logger = logging.getLogger(__name__)

AGENT_CLASSES = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_env(
    returns: pd.DataFrame,
    config: dict,
    encoder: AlgoUniverseEncoder = None,
    is_eval: bool = False,
) -> TradingEnvironment:
    """Create trading environment from config."""
    constraints_cfg = config.get("constraints", {})
    constraints = PortfolioConstraints(
        max_weight=constraints_cfg.get("max_weight", 0.40),
        min_weight=constraints_cfg.get("min_weight", 0.0),
        max_turnover=constraints_cfg.get("max_turnover", 0.30),
        max_exposure=constraints_cfg.get("max_exposure", 1.0),
    )

    costs_cfg = config.get("costs", {})
    cost_model = CostModel(
        spread_bps=costs_cfg.get("spread_bps", 5),
        slippage_bps=costs_cfg.get("slippage_bps", 2),
        impact_coefficient=costs_cfg.get("market_impact_coef", 0.1),
    )

    reward_cfg = config.get("reward", {})
    reward_fn = RewardFunction(
        reward_type=RewardType.RISK_CALIBRATED_RETURNS,
        cost_penalty_weight=reward_cfg.get("cost_penalty", 1.0),
        turnover_penalty_weight=reward_cfg.get("turnover_penalty", 0.1),
        drawdown_penalty_weight=reward_cfg.get("drawdown_penalty", 0.5),
    )

    # Create equal-weight benchmark
    n_assets = returns.shape[1]
    benchmark_weights = pd.DataFrame(
        np.ones((len(returns), n_assets)) / n_assets,
        index=returns.index,
        columns=returns.columns,
    )

    train_cfg = config.get("training", {})
    train_window = train_cfg.get("train_window_days", 504)

    # Split data for training
    train_end_idx = len(returns) - train_cfg.get("val_window_days", 63)
    train_start_idx = max(0, train_end_idx - train_window)

    train_start = returns.index[train_start_idx]
    train_end = returns.index[train_end_idx]

    if is_eval:
        # Use validation period
        train_start = returns.index[train_end_idx]
        train_end = returns.index[-1]

    env = TradingEnvironment(
        algo_returns=returns,
        benchmark_weights=benchmark_weights,
        cost_model=cost_model,
        constraints=constraints,
        reward_function=reward_fn,
        initial_capital=1_000_000.0,
        rebalance_frequency="weekly",
        train_start=train_start,
        train_end=train_end,
        encoder=encoder,
    )

    return env


def train_agent(
    returns: pd.DataFrame,
    config: dict,
    output_dir: Path,
    agent_type: str = "ppo",
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    use_gpu_env: bool = False,
    use_subproc: bool = False,
    device: str = "auto",
) -> None:
    """
    Train RL agent on production data.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns matrix [dates x assets]
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory for checkpoints
    agent_type : str
        Agent type (ppo, sac, td3)
    total_timesteps : int
        Total training timesteps
    n_envs : int
        Number of parallel environments
    use_gpu_env : bool
        Use GPU-accelerated vectorized environment
    use_subproc : bool
        Use SubprocVecEnv for CPU training
    device : str
        Device for training (auto, cpu, cuda)
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints" / agent_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_type = "GPU" if use_gpu_env else ("SubprocVecEnv" if use_subproc else "DummyVecEnv")
    logger.info(f"Training {agent_type.upper()} agent...")
    logger.info(f"Data: {returns.shape[0]} days x {returns.shape[1]} assets")
    logger.info(f"Timesteps: {total_timesteps:,}")
    logger.info(f"Environment: {env_type} x {n_envs}")

    train_cfg = config.get("training", {})
    constraints_cfg = config.get("constraints", {})
    costs_cfg = config.get("costs", {})
    reward_cfg = config.get("reward", {})

    # Determine device
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    # Get top_families setting
    top_families = train_cfg.get("top_families", 4)  # Default: use all 4 families
    logger.info(f"Using top {top_families} families for training")

    # GPU-accelerated environment
    if use_gpu_env:
        if not HAS_GPU_ENV:
            raise RuntimeError("GPU environment not available. Check torch/CUDA installation.")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        if device == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Create equal-weight benchmark
        n_assets = returns.shape[1]
        benchmark_weights = pd.DataFrame(
            np.ones((len(returns), n_assets)) / n_assets,
            index=returns.index,
            columns=returns.columns,
        )

        # Compute train window
        train_window = train_cfg.get("train_window_days", 504)
        train_end_idx = len(returns) - train_cfg.get("val_window_days", 63)
        train_start_idx = max(0, train_end_idx - train_window)
        train_start = returns.index[train_start_idx]
        train_end = returns.index[train_end_idx]

        # Create family labels from Sortino-based classification
        # Use the latest available family assignments (based on sortino_63d)
        logger.info("Creating family labels for GPU encoder...")

        # Always compute Sortino-based families from returns (most reliable)
        logger.info("Computing family labels from Sortino ratios...")
        recent_returns = returns.iloc[train_start_idx:train_end_idx]

        # Compute Sortino ratio for each asset
        mean_ret = recent_returns.mean() * 252  # Annualized
        downside = recent_returns.copy()
        downside[downside > 0] = 0
        downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(252)  # Annualized
        sortino = mean_ret / downside_std.replace(0, np.nan)
        sortino = sortino.fillna(0).clip(-10, 10)  # Clip extreme values

        # Classify into families based on Sortino
        family_labels_series = pd.Series(index=returns.columns, dtype=int)
        family_labels_series[sortino > 2] = 0      # Excellent
        family_labels_series[(sortino > 1) & (sortino <= 2)] = 1  # Good
        family_labels_series[(sortino > 0) & (sortino <= 1)] = 2  # Moderate
        family_labels_series[sortino <= 0] = 3     # Poor

        logger.info(f"Computed {len(family_labels_series)} family labels")

        # Log family distribution
        family_counts = family_labels_series.value_counts().sort_index()
        family_names = ["Excellent (Sortino>2)", "Good (Sortino 1-2)", "Moderate (Sortino 0-1)", "Poor (Sortino<0)"]
        logger.info("Family distribution:")
        for fam_idx in range(4):
            count = family_counts.get(fam_idx, 0)
            logger.info(f"  Family {fam_idx} ({family_names[fam_idx]}): {count} assets")

        # Filter to top N families if specified
        if top_families < 4:
            # Set assets in excluded families to -1 (will be ignored by encoder)
            excluded_families = list(range(top_families, 4))
            logger.info(f"Filtering to top {top_families} families, excluding: {excluded_families}")
            mask = family_labels_series.isin(excluded_families)
            family_labels_series = family_labels_series.copy()
            family_labels_series[mask] = -1  # Mark as excluded
            n_remaining = (family_labels_series >= 0).sum()
            logger.info(f"Excluded {mask.sum()} assets, {n_remaining} remaining for training")

            if n_remaining == 0:
                raise ValueError(
                    f"No assets remaining after filtering to top {top_families} families! "
                    f"All assets are in families {excluded_families}. Try --top-families 4 or check your data."
                )

        # Verify we have valid families
        valid_families = family_labels_series[family_labels_series >= 0].unique()
        logger.info(f"Active families after filtering: {sorted(valid_families.tolist())}")

        # Create and fit FamilyEncoder
        # The encoder will only use families present in family_labels_series
        # Assets with family=-1 are excluded automatically
        family_encoder = FamilyEncoder(
            family_labels=family_labels_series,
            activity_window=63,
        )
        family_encoder.fit(returns, train_start, train_end)
        logger.info(f"FamilyEncoder: obs_dim={family_encoder.obs_dim}, action_dim={family_encoder.action_dim}, n_families={family_encoder._n_families}")

        if family_encoder.action_dim == 0:
            raise ValueError(
                "FamilyEncoder has 0 action dimensions! No valid families found. "
                "Check that your data has assets with positive Sortino ratios."
            )

        # GPU environment config
        gpu_config = GPUEnvConfig(
            initial_capital=1_000_000.0,
            rebalance_frequency="weekly",
            episode_length=52,
            random_start=True,
            max_weight=constraints_cfg.get("max_weight", 0.40),
            min_weight=constraints_cfg.get("min_weight", 0.0),
            max_turnover=constraints_cfg.get("max_turnover", 0.30),
            max_exposure=constraints_cfg.get("max_exposure", 1.0),
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            impact_coefficient=costs_cfg.get("market_impact_coef", 0.1),
            reward_scale=reward_cfg.get("scale", 100.0),
            cost_penalty=reward_cfg.get("cost_penalty", 1.0),
            turnover_penalty=reward_cfg.get("turnover_penalty", 0.1),
            drawdown_penalty=reward_cfg.get("drawdown_penalty", 0.5),
            bad_family_penalty=reward_cfg.get("bad_family_penalty", 2.0),  # Penalize bad families
        )

        # Create GPU vectorized environment
        train_vec_env = GPUVecTradingEnv(
            n_envs=n_envs,
            algo_returns=returns,
            benchmark_weights=benchmark_weights,
            family_encoder=family_encoder,
            train_start=train_start,
            train_end=train_end,
            config=gpu_config,
            device=device,
        )

        # Apply observation normalization
        train_vec_env = VecNormalize(
            train_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # Eval environment - must use same observation space as training (FamilyEncoder)
        encoder = family_encoder  # Save for later

        # Use validation period for eval
        val_start = returns.index[train_end_idx]
        val_end = returns.index[-1]

        # Create eval GPU environment with same family encoder
        eval_gpu_config = GPUEnvConfig(
            initial_capital=1_000_000.0,
            rebalance_frequency="weekly",
            episode_length=52,
            random_start=False,  # Deterministic for eval
            max_weight=constraints_cfg.get("max_weight", 0.40),
            min_weight=constraints_cfg.get("min_weight", 0.0),
            max_turnover=constraints_cfg.get("max_turnover", 0.30),
            max_exposure=constraints_cfg.get("max_exposure", 1.0),
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            impact_coefficient=costs_cfg.get("market_impact_coef", 0.1),
            reward_scale=reward_cfg.get("scale", 100.0),
            cost_penalty=reward_cfg.get("cost_penalty", 1.0),
            turnover_penalty=reward_cfg.get("turnover_penalty", 0.1),
            drawdown_penalty=reward_cfg.get("drawdown_penalty", 0.5),
            bad_family_penalty=reward_cfg.get("bad_family_penalty", 2.0),  # Penalize bad families
        )

        eval_env = GPUVecTradingEnv(
            n_envs=1,  # Single env for eval
            algo_returns=returns,
            benchmark_weights=benchmark_weights,
            family_encoder=family_encoder,
            train_start=val_start,
            train_end=val_end,
            config=eval_gpu_config,
            device=device,
        )

        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
        # Share running stats - both envs have same obs shape now
        eval_env.obs_rms = train_vec_env.obs_rms

    else:
        # CPU-based environment (SubprocVecEnv or DummyVecEnv)
        # Create encoder if configured
        encoder = None

        if train_cfg.get("use_encoder", True):
            logger.info("Creating universe encoder...")
            n_components = train_cfg.get("n_pca_components", 20)

            # Compute train window
            train_window = train_cfg.get("train_window_days", 504)
            train_end_idx = len(returns) - train_cfg.get("val_window_days", 63)
            train_start_idx = max(0, train_end_idx - train_window)
            train_start = returns.index[train_start_idx]
            train_end = returns.index[train_end_idx]

            encoder = AlgoUniverseEncoder(
                n_components=n_components,
                min_days_active=21,
                activity_window=63,
            )

            # Fit encoder on training data
            encoder.fit(returns, train_start, train_end)

            logger.info(f"Encoder fitted: {encoder.obs_dim} obs dims, {encoder.action_dim} action dims")

        # Create parallel training environments
        def make_train_env():
            return create_env(returns, config, encoder, is_eval=False)

        if n_envs > 1 and use_subproc:
            logger.info("Using SubprocVecEnv for parallel training")
            train_vec_env = SubprocVecEnv([make_train_env for _ in range(n_envs)])
        else:
            logger.info("Using DummyVecEnv")
            train_vec_env = DummyVecEnv([make_train_env for _ in range(n_envs)])

        # Apply observation normalization
        train_vec_env = VecNormalize(
            train_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # Create eval environment (always DummyVecEnv for eval)
        eval_env = DummyVecEnv([lambda: create_env(returns, config, encoder, is_eval=True)])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
        eval_env.obs_rms = train_vec_env.obs_rms  # Share running stats

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(output_dir / "logs" / agent_type),
        eval_freq=max(5000 // n_envs, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 2000),
        save_path=str(checkpoint_dir),
        name_prefix="model",
    )

    # Create agent
    agent_class = AGENT_CLASSES[agent_type]

    policy_kwargs = {
        "net_arch": train_cfg.get("net_arch", [256, 256]),
    }

    if agent_type == "ppo":
        policy_kwargs["activation_fn"] = __import__("torch").nn.Tanh
        policy_kwargs["log_std_init"] = -1.0
        policy_kwargs["ortho_init"] = False

    # Check if tensorboard is available
    try:
        import tensorboard
        tb_log = str(output_dir / "tensorboard")
    except ImportError:
        logger.warning("TensorBoard not installed, skipping tensorboard logging")
        tb_log = None

    # PPO hyperparameters - reduce memory footprint for large n_envs
    n_steps = train_cfg.get("n_steps", 1024)  # Reduced from default 2048
    batch_size = train_cfg.get("batch_size", 256)  # Smaller batches

    agent = agent_class(
        "MlpPolicy",
        train_vec_env,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log,
    )

    # Train
    logger.info("Starting training...")
    agent.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_model_path = checkpoint_dir / "final_model.zip"
    agent.save(str(final_model_path))
    logger.info(f"Saved final model to {final_model_path}")

    # Save VecNormalize stats
    vec_norm_path = checkpoint_dir / "vecnormalize.pkl"
    train_vec_env.save(str(vec_norm_path))
    logger.info(f"Saved normalization stats to {vec_norm_path}")

    # Save encoder
    if encoder is not None:
        encoder_path = checkpoint_dir / "universe_encoder.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        logger.info(f"Saved encoder to {encoder_path}")

    # Save run info
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "agent_type": agent_type,
        "total_timesteps": total_timesteps,
        "n_assets": returns.shape[1],
        "n_training_days": len(returns),
        "date_range": {
            "start": returns.index.min().isoformat(),
            "end": returns.index.max().isoformat(),
        },
        "config": config,
    }

    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    # Create latest pointer (file instead of symlink for Windows compatibility)
    latest_link = output_dir.parent / "latest_run.txt"
    with open(latest_link, "w") as f:
        f.write(output_dir.name)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Agent: {agent_type.upper()}")
    print(f"Timesteps: {total_timesteps}")
    print(f"Output: {output_dir}")
    print(f"\nModel files:")
    print(f"  - {final_model_path}")
    print(f"  - {vec_norm_path}")
    if encoder is not None:
        print(f"  - {encoder_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train production RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training with GPU environment (recommended)
    python train.py --gpu-env --n-envs 12 --timesteps 500000

    # Train with only good families (Sortino > 1)
    python train.py --gpu-env --n-envs 12 --timesteps 1000000 --good-families-only

    # Use top N families only
    python train.py --gpu-env --top-families 2

    # Use SubprocVecEnv for CPU training
    python train.py --use-subproc --n-envs 8

    # Custom reward type
    python train.py --gpu-env --reward-type pure_returns

    # Full training with all options
    python train.py --gpu-env --n-envs 24 --timesteps 10000000 \\
        --good-families-only --bad-family-penalty 5.0 \\
        --learning-rate 0.0001 --batch-size 256 --n-steps 1024
        """,
    )

    # Agent selection
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="Agent type to train (default: ppo)",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500,000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=12,
        help="Number of parallel environments (default: 12)",
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=None,
        help="Learning rate (default: from config or 0.0001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="Steps per environment per update (default: 1024)",
    )

    # Environment options
    parser.add_argument(
        "--gpu-env",
        action="store_true",
        help="Use GPU-accelerated vectorized environment (recommended, 50-200x faster)",
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        help="Use SubprocVecEnv for CPU training (better parallelism)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training (default: auto)",
    )

    # Family/cluster selection
    parser.add_argument(
        "--good-families-only",
        action="store_true",
        help="Only use families 0 and 1 (Sortino > 1) for training",
    )
    parser.add_argument(
        "--top-families",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of top families to use (1=Excellent only, 2=Excellent+Good, etc.)",
    )
    parser.add_argument(
        "--bad-family-penalty",
        type=float,
        default=None,
        help="Penalty for allocating to bad families (default: from config or 2.0)",
    )

    # Reward configuration
    parser.add_argument(
        "--reward-type",
        type=str,
        default=None,
        choices=["pure_returns", "risk_calibrated", "absolute_returns", "risk_adjusted"],
        help="Reward function type (default: from config)",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Reward scaling factor (default: from config or 100.0)",
    )

    # Constraints
    parser.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="Maximum weight per asset (default: from config or 0.40)",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="Maximum turnover per rebalance (default: from config or 0.30)",
    )

    # Paths
    parser.add_argument(
        "--config",
        type=Path,
        default=PRODUCTION_DIR / "config" / "production.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PRODUCTION_DIR / "data" / "processed",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated with timestamp)",
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Check GPU environment availability
    if args.gpu_env and not HAS_GPU_ENV:
        logger.error("GPU environment not available. Install torch with CUDA support.")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    if args.config.exists():
        config = load_config(args.config)
    else:
        logger.warning(f"Config not found at {args.config}, using defaults")
        config = {}

    # Override config with command line arguments
    if args.learning_rate is not None:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.n_steps is not None:
        config.setdefault("training", {})["n_steps"] = args.n_steps
    if args.reward_type is not None:
        config.setdefault("reward", {})["type"] = args.reward_type
    if args.reward_scale is not None:
        config.setdefault("reward", {})["scale"] = args.reward_scale
    if args.bad_family_penalty is not None:
        config.setdefault("reward", {})["bad_family_penalty"] = args.bad_family_penalty
    if args.max_weight is not None:
        config.setdefault("constraints", {})["max_weight"] = args.max_weight
    if args.max_turnover is not None:
        config.setdefault("constraints", {})["max_turnover"] = args.max_turnover

    # Handle family filtering
    if args.good_families_only:
        args.top_families = 2  # Only families 0 and 1
    config["training"] = config.get("training", {})
    config["training"]["top_families"] = args.top_families

    # Load returns
    returns_path = args.data_dir / "returns.parquet"
    if not returns_path.exists():
        logger.error(f"Returns not found at {returns_path}")
        logger.info("Run 'python production/scripts/prepare_data.py' first")
        sys.exit(1)

    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape}")

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.agent}"
        output_dir = PRODUCTION_DIR / "models" / run_id

    # Print training configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Agent: {args.agent.upper()}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Environments: {args.n_envs}")
    print(f"Environment type: {'GPU' if args.gpu_env else 'CPU (SubprocVecEnv)' if args.use_subproc else 'CPU (DummyVecEnv)'}")
    print(f"Top families: {args.top_families} (0={'Excellent'}, 1={'Good'}, 2={'Moderate'}, 3={'Poor'})")
    print(f"Bad family penalty: {config.get('reward', {}).get('bad_family_penalty', 2.0)}")
    print(f"Reward type: {config.get('reward', {}).get('type', 'risk_calibrated')}")
    print(f"Learning rate: {config.get('training', {}).get('learning_rate', 0.0001)}")
    print(f"Batch size: {config.get('training', {}).get('batch_size', 256)}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")

    # Train
    train_agent(
        returns=returns,
        config=config,
        output_dir=output_dir,
        agent_type=args.agent,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        use_gpu_env=args.gpu_env,
        use_subproc=args.use_subproc,
        device=args.device,
    )


if __name__ == "__main__":
    main()
