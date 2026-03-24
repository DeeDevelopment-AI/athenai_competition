#!/usr/bin/env python3
"""
=================================================================
PHASE 5: RL Agent Training
=================================================================
Complete Phase 5 pipeline to train RL meta-allocators:
  5.1 Load processed data from Phase 1
  5.2 Configure training environment with walk-forward splits
  5.3 Train RL agents (PPO, SAC, TD3)
  5.4 Evaluate on validation set
  5.5 Save checkpoints and training logs

Agents:
  - PPO: On-policy, stable (main baseline)
  - SAC: Off-policy, sample-efficient
  - TD3: Off-policy, handles noisy rewards

Usage:
  python scripts/run_phase5.py                  # Train PPO (default)
  python scripts/run_phase5.py --quick          # Quick test (10k steps, 20 algos)
  python scripts/run_phase5.py --agent sac      # Train SAC
  python scripts/run_phase5.py --agent all      # Train all agents
  python scripts/run_phase5.py --timesteps 1M   # 1 million timesteps
  python scripts/run_phase5.py --sample 50      # Use 50 algorithms

Options:
  --agent NAME         Agent to train: ppo, sac, td3, all (default: ppo)
  --quick              Quick test mode (10k steps, 20 algos)
  --timesteps N        Total training timesteps (default: 500k)
  --sample N           Use only N algorithms
  --eval-freq N        Evaluation frequency in timesteps (default: 10k)
  --n-eval-episodes N  Episodes per evaluation (default: 5)
  --rebalance-freq F   Rebalance frequency (default: weekly)
  --seed N             Random seed (default: 42)
  --max-resources      Auto-configure to fully use GPU/CPU/RAM (detects VRAM and sets
                       n_envs, batch_size, net_arch, SubprocVecEnv automatically)
  --n-envs N           Override number of parallel envs (default: 4)
  --use-subproc        Use SubprocVecEnv for true process parallelism
"""

import argparse
import gc
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Agent
    agent: str = "ppo"
    # Reduced from 3e-4: with obs_dim=54055 and VecNormalize, 1e-4 is more stable
    learning_rate: float = 1e-4
    net_arch: List[int] = field(default_factory=lambda: [256, 256])

    # Training
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    seed: int = 42

    # Environment
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "weekly"
    episode_length: int = 52  # weeks
    random_start: bool = True

    # Constraints
    max_weight: float = 0.40
    min_weight: float = 0.00
    max_turnover: float = 0.30
    max_exposure: float = 1.0

    # Costs
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    impact_coefficient: float = 0.1

    # Reward
    # Calibración: alpha semanal típico ~0.1% → reward_scale=100 → señal ~0.10
    # cost_penalty=1.0: incluye costes reales (7bps×turnover ≈ 0.01 escalado) — mantener
    # turnover_penalty: penalización ADICIONAL sobre costes. Con 0.1 y max_turnover=0.30
    #   la penalización (3.0 escalada) aplasta la señal (0.10). Reducir 20×.
    # drawdown_penalty: con 0.5 y 5% DD excess, penalización (2.5 escalada) >> señal.
    reward_scale: float = 100.0
    cost_penalty: float = 1.0
    turnover_penalty: float = 0.005   # was 0.1 — 20× reduction to let agent learn to trade
    drawdown_penalty: float = 0.1     # was 0.5 — proportional to alpha signal magnitude

    # Walk-forward
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Universe encoder (dimensionality reduction)
    use_encoder: bool = True
    use_family_encoder: bool = True   # True = FamilyEncoder (8-dim), False = PCA encoder
    n_pca_components: int = 50
    min_days_active: int = 21
    n_families: int = 8


def parse_timesteps(value: str) -> int:
    """Parse timesteps with K/M suffix."""
    value = value.upper().strip()
    if value.endswith('K'):
        return int(float(value[:-1]) * 1_000)
    elif value.endswith('M'):
        return int(float(value[:-1]) * 1_000_000)
    else:
        return int(value)


def detect_hardware() -> dict:
    """Detect available GPU VRAM, RAM, and CPU cores."""
    import logging
    log = logging.getLogger(__name__)

    info = {
        "gpu_available": False,
        "gpu_vram_gb": 0.0,
        "gpu_name": "none",
        "ram_gb": 8.0,
        "cpu_count": os.cpu_count() or 4,
    }

    # RAM
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / 1e9
    except ImportError:
        pass

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_available"] = True
            info["gpu_vram_gb"] = props.total_memory / 1e9
            info["gpu_name"] = props.name
    except Exception:
        pass

    log.info(
        f"Hardware detected: GPU={info['gpu_name']} ({info['gpu_vram_gb']:.0f} GB VRAM), "
        f"RAM={info['ram_gb']:.0f} GB, CPU={info['cpu_count']} cores"
    )
    return info


def build_max_resources_config(hw: dict, agent: str) -> dict:
    """
    Return hyperparameter overrides to maximize hardware utilization.

    Tiers based on GPU VRAM:
      >= 80 GB : massive config (n_envs=64, batch=8192, net=[1024,512,256])
      >= 40 GB : large config  (n_envs=32, batch=4096, net=[512,512])
      >= 16 GB : medium config (n_envs=16, batch=2048, net=[512,256])
      CPU only : cpu config    (n_envs=cpu//2, batch=512, net=[256,256])
    """
    vram = hw["gpu_vram_gb"]
    cpu = hw["cpu_count"]

    if vram >= 80:
        n_envs, batch_size, n_steps = 64, 8192, 8192
        net_arch = [1024, 512, 256]
        buf_size, grad_steps = 2_000_000, 16
    elif vram >= 40:
        n_envs, batch_size, n_steps = 32, 4096, 4096
        net_arch = [512, 512]
        buf_size, grad_steps = 1_000_000, 8
    elif vram >= 16:
        n_envs, batch_size, n_steps = 16, 2048, 4096
        net_arch = [512, 256]
        buf_size, grad_steps = 500_000, 4
    else:
        # CPU-only: parallelism via SubprocVecEnv with many envs
        n_envs = max(4, cpu // 2)
        batch_size, n_steps = 512, 2048
        net_arch = [256, 256]
        buf_size, grad_steps = 200_000, 2

    # SubprocVecEnv uses fork on Linux (fast) but spawn on Windows (very slow —
    # each subprocess re-imports Python, pickles the full DataFrame, and sends it
    # over IPC).  On Windows, DummyVecEnv with more envs is faster overall.
    import sys as _sys
    use_subproc = _sys.platform != "win32"

    cfg: dict = {
        "n_envs": n_envs,
        "use_subproc": use_subproc,
        "net_arch": net_arch,
    }

    if agent in ("ppo", "all"):
        cfg.update({
            "ppo_n_steps": n_steps,
            "ppo_batch_size": batch_size,
        })
    if agent in ("sac", "td3", "all"):
        cfg.update({
            "off_policy_batch_size": batch_size,
            "buffer_size": buf_size,
            "gradient_steps": grad_steps,
        })

    return cfg


# =============================================================================
# Phase 5 Runner
# =============================================================================

class Phase5Runner(PhaseRunner):
    """Phase 5: RL Agent Training."""

    phase_name = "Phase 5: RL Agent Training"
    phase_number = 5

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 5 specific arguments."""
        parser.add_argument(
            '--agent', type=str, default='ppo',
            choices=['ppo', 'sac', 'td3', 'all'],
            help='Agent to train (default: ppo)'
        )
        parser.add_argument(
            '--quick', '-q', action='store_true',
            help='Quick test mode (10k steps, 20 algos)'
        )
        parser.add_argument(
            '--timesteps', type=str, default='500K',
            help='Total training timesteps (e.g., 500K, 1M)'
        )
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Use only N algorithms'
        )
        parser.add_argument(
            '--eval-freq', type=int, default=10_000,
            help='Evaluation frequency in timesteps'
        )
        parser.add_argument(
            '--n-eval-episodes', type=int, default=5,
            help='Episodes per evaluation'
        )
        parser.add_argument(
            '--rebalance-freq', type=str, default='weekly',
            choices=['daily', 'weekly', 'monthly'],
            help='Rebalance frequency'
        )
        parser.add_argument(
            '--seed', type=int, default=42,
            help='Random seed'
        )
        parser.add_argument(
            '--learning-rate', type=float, default=1e-4,
            help='Learning rate (default: 1e-4; 3e-4 causes NaN with 13k-dim action space)'
        )
        parser.add_argument(
            '--input-dir', type=str, default=None,
            help='Override input directory (Phase 1 outputs)'
        )
        parser.add_argument(
            '--resume', type=str, default=None,
            help='Path to checkpoint to resume from'
        )
        parser.add_argument(
            '--max-resources', action='store_true',
            help=(
                'Auto-configure hyperparameters to maximally use available GPU/CPU/RAM. '
                'Detects VRAM, RAM, and CPU cores to set n_envs, batch_size, net_arch. '
                'Uses SubprocVecEnv for true parallelism.'
            )
        )
        parser.add_argument(
            '--n-envs', type=int, default=None,
            help='Override number of parallel environments (default: 4, or auto with --max-resources)'
        )
        parser.add_argument(
            '--use-subproc', action='store_true',
            help='Use SubprocVecEnv for true process-level parallelism'
        )
        parser.add_argument(
            '--no-encoder', action='store_true',
            help='Disable AlgoUniverseEncoder (use raw 13k-dim obs/action spaces)'
        )
        parser.add_argument(
            '--pca-components', type=int, default=50,
            help='Number of PCA components for AlgoUniverseEncoder (default: 50)'
        )
        parser.add_argument(
            '--min-days-active', type=int, default=21,
            help='Min active days in training window for Stage 1 filter (default: 21)'
        )
        parser.add_argument(
            '--pca-encoder', action='store_true',
            help='Use PCA-based AlgoUniverseEncoder instead of FamilyEncoder (default: FamilyEncoder)'
        )
        parser.add_argument(
            '--run-id', type=str, default=None,
            help=(
                'Identifier for this training run (default: auto YYYYMMDD_HHMMSS). '
                'Outputs go to outputs/rl_training/{run_id}/. '
                'Pass the same run-id to run_phase6.py --run-id to evaluate this specific run.'
            )
        )
        parser.add_argument(
            '--gpu-env', action='store_true',
            help='Use GPU-accelerated batched environment (requires CUDA)'
        )

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 5 pipeline."""
        # Quick mode overrides
        if args.quick:
            args.timesteps = '10K'
            args.sample = args.sample or 20
            args.eval_freq = 2_000
            args.n_eval_episodes = 2

        # Parse timesteps
        total_timesteps = parse_timesteps(args.timesteps)

        # --max-resources: detect hardware and override hyperparameters
        n_envs = args.n_envs or 4
        use_subproc = args.use_subproc
        ppo_n_steps = 2048
        ppo_batch_size = 64
        off_policy_batch_size = 256
        buffer_size = 100_000
        gradient_steps = 1
        net_arch = [256, 256]

        if args.max_resources:
            hw = detect_hardware()
            mr = build_max_resources_config(hw, args.agent)
            n_envs = mr["n_envs"]
            use_subproc = mr["use_subproc"]
            net_arch = mr["net_arch"]
            ppo_n_steps = mr.get("ppo_n_steps", ppo_n_steps)
            ppo_batch_size = mr.get("ppo_batch_size", ppo_batch_size)
            off_policy_batch_size = mr.get("off_policy_batch_size", off_policy_batch_size)
            buffer_size = mr.get("buffer_size", buffer_size)
            gradient_steps = mr.get("gradient_steps", gradient_steps)
            self.logger.info(
                f"Max-resources mode: n_envs={n_envs}, subproc={use_subproc}, "
                f"ppo_batch={ppo_batch_size}, net_arch={net_arch}"
            )

        # Build config
        config = TrainingConfig(
            agent=args.agent,
            total_timesteps=total_timesteps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            rebalance_frequency=args.rebalance_freq,
            seed=args.seed,
            learning_rate=args.learning_rate,
            net_arch=net_arch,
            use_encoder=not args.no_encoder,
            use_family_encoder=not getattr(args, 'pca_encoder', False),
            n_pca_components=args.pca_components,
            min_days_active=args.min_days_active,
        )

        # ── Run ID: unique identifier for this training run ──────────────────
        # Lets you keep multiple runs and always know which model was evaluated.
        # Format: YYYYMMDD_HHMMSS  (or custom via --run-id)
        from datetime import datetime as _dt
        import json as _json

        run_id = args.run_id or _dt.now().strftime("%Y%m%d_%H%M%S")
        if args.agent != 'all':
            # Suffix with agent name when training a single agent for clarity
            run_id = f"{run_id}_{args.agent}"

        # Output lives under outputs/rl_training/{run_id}/
        rl_root = self.op.rl_training.root
        output_dir = rl_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── latest_run.txt — Phase 6 reads this by default ───────────────────
        latest_path = rl_root / "latest_run.txt"
        latest_path.write_text(run_id)

        self.logger.info(f"Run ID: {run_id}")
        self.logger.info(f"Output: {output_dir}")
        self.logger.info(f"Latest pointer updated: {latest_path}")

        results = {
            'run_id': run_id,
            'mode': 'quick' if args.quick else 'standard',
            'config': asdict(config),
            'agents_trained': [],
        }

        # Determine paths
        input_dir = Path(args.input_dir) if args.input_dir else self.dp.processed.root

        # ==================================================================
        # STEP 5.1: LOAD DATA
        # ==================================================================
        with self.step("5.1 Load Phase 1 Data"):
            algo_returns, benchmark_weights, benchmark_returns = self._load_data(input_dir)

            # Sample if requested - only from algorithms that have benchmark weights
            if args.sample and args.sample < len(algo_returns.columns):
                np.random.seed(config.seed)
                if benchmark_weights is not None:
                    # Sample only from algorithms in both returns and benchmark
                    common_algos = [c for c in algo_returns.columns if c in benchmark_weights.columns]
                    if len(common_algos) >= args.sample:
                        sampled_cols = list(np.random.choice(common_algos, args.sample, replace=False))
                    else:
                        sampled_cols = common_algos
                        self.logger.warning(f"Only {len(common_algos)} common algorithms available")
                    benchmark_weights = benchmark_weights[sampled_cols]
                else:
                    sampled_cols = list(np.random.choice(
                        algo_returns.columns, args.sample, replace=False
                    ))
                algo_returns = algo_returns[sampled_cols]
                self.logger.info(f"Sampled {len(sampled_cols)} algorithms")

            self.logger.info(f"Returns: {algo_returns.shape[0]} days x {algo_returns.shape[1]} algos")
            results['n_algos'] = algo_returns.shape[1]
            results['n_days'] = algo_returns.shape[0]

        # ==================================================================
        # STEP 5.2: COMPUTE TRAIN/VAL/TEST SPLITS
        # ==================================================================
        with self.step("5.2 Compute Data Splits"):
            dates = algo_returns.index
            n = len(dates)

            train_end = int(n * config.train_ratio)
            val_end = int(n * (config.train_ratio + config.val_ratio))

            train_dates = (dates[0], dates[train_end - 1])
            val_dates = (dates[train_end], dates[val_end - 1])
            test_dates = (dates[val_end], dates[-1])

            self.logger.info(f"Train: {train_dates[0].date()} to {train_dates[1].date()} ({train_end} days)")
            self.logger.info(f"Val:   {val_dates[0].date()} to {val_dates[1].date()} ({val_end - train_end} days)")
            self.logger.info(f"Test:  {test_dates[0].date()} to {test_dates[1].date()} ({n - val_end} days)")

            results['splits'] = {
                'train': [str(train_dates[0].date()), str(train_dates[1].date())],
                'val': [str(val_dates[0].date()), str(val_dates[1].date())],
                'test': [str(test_dates[0].date()), str(test_dates[1].date())],
            }

        # ==================================================================
        # STEP 5.2.5: FIT UNIVERSE ENCODER
        # ==================================================================
        encoder = None
        with self.step("5.2.5 Fit Universe Encoder"):
            if config.use_encoder:
                if config.use_family_encoder:
                    from src.environment.universe_encoder import FamilyEncoder
                    # Load pre-computed family labels (from Phase 2 behavioral clustering)
                    family_labels_path = self.dp.processed.root / 'analysis' / 'clustering' / 'behavioral' / 'family_labels.csv'
                    if not family_labels_path.exists():
                        self.logger.warning(
                            f"family_labels.csv not found at {family_labels_path}. "
                            "Falling back to PCA encoder."
                        )
                        config.use_family_encoder = False
                    else:
                        import pandas as _pd
                        family_df = _pd.read_csv(family_labels_path, index_col=0)
                        # Column may be 'family' or first column
                        fam_col = 'family' if 'family' in family_df.columns else family_df.columns[0]
                        family_labels_series = family_df[fam_col].astype(int)
                        encoder = FamilyEncoder(
                            family_labels=family_labels_series,
                            activity_window=63,
                        )
                        encoder.fit(algo_returns, train_dates[0], train_dates[1])
                        stats = encoder.get_filter_stats()
                        self.logger.info(
                            f"FamilyEncoder: {stats['n_total_algos']} algos -> "
                            f"{stats['n_families']} families, "
                            f"obs_dim={stats['obs_dim']}, action_dim={stats['action_dim']}"
                        )
                        for fid, sz in stats['family_sizes'].items():
                            self.logger.info(f"  Family {fid}: {sz} algos")
                        results['encoder'] = stats

                if not config.use_family_encoder:
                    from src.environment.universe_encoder import AlgoUniverseEncoder
                    encoder = AlgoUniverseEncoder(
                        n_components=config.n_pca_components,
                        min_days_active=config.min_days_active,
                    )
                    encoder.fit(algo_returns, train_dates[0], train_dates[1])
                    stats = encoder.get_filter_stats()
                    self.logger.info(
                        f"PCA Encoder: {stats['n_total_algos']} algos -> "
                        f"{stats['n_static_algos']} static -> "
                        f"{stats['n_pca_components']} PCA components "
                        f"({stats['pca_explained_variance']:.1%} var explained)"
                    )
                    results['encoder'] = stats
            else:
                self.logger.info("Universe encoder disabled (--no-encoder). Using raw dimensions.")
                results['encoder'] = {"fitted": False}

        # ==================================================================
        # STEP 5.3: CREATE ENVIRONMENTS
        # ==================================================================
        with self.step("5.3 Create Training Environments"):
            from src.environment.trading_env import TradingEnvironment, EpisodeConfig, VecTradingEnv
            from src.environment.cost_model import CostModel
            from src.environment.constraints import PortfolioConstraints
            from src.environment.reward import RewardFunction, RewardType

            # --- Get encoder (fitted in step 5.2.5) ---
            # NOTE: encoder is already available from step 5.2.5
            # If not, retrieve it from where step 5.2.5 stores it

            use_gpu_env = getattr(args, 'gpu_env', False) and torch.cuda.is_available()

            if use_gpu_env:
                # ============================================================
                # GPU-ACCELERATED PATH
                # ============================================================
                from src.environment.gpu_vec_env import GPUVecTradingEnv, GPUEnvConfig

                gpu_config = GPUEnvConfig(
                    initial_capital=config.initial_capital,
                    rebalance_frequency=config.rebalance_frequency,
                    episode_length=config.episode_length,
                    random_start=config.random_start,
                    max_weight=config.max_weight,
                    min_weight=config.min_weight,
                    max_turnover=config.max_turnover,
                    max_exposure=config.max_exposure,
                    spread_bps=config.spread_bps,
                    slippage_bps=config.slippage_bps,
                    impact_coefficient=config.impact_coefficient,
                    reward_scale=config.reward_scale,
                    cost_penalty=config.cost_penalty,
                    turnover_penalty=config.turnover_penalty,
                    drawdown_penalty=config.drawdown_penalty,
                    seed=config.seed,
                )

                # Training environment (GPU batched)
                train_vec_env = GPUVecTradingEnv(
                    n_envs=n_envs,
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    family_encoder=encoder,  # from step 5.2.5
                    train_start=train_dates[0],
                    train_end=train_dates[1],
                    config=gpu_config,
                    device="cuda",
                )

                # Eval environment (can also use GPU, smaller)
                eval_gpu_config = GPUEnvConfig(
                    initial_capital=config.initial_capital,
                    rebalance_frequency=config.rebalance_frequency,
                    episode_length=config.episode_length,
                    random_start=False,  # Deterministic eval
                    max_weight=config.max_weight,
                    min_weight=config.min_weight,
                    max_turnover=config.max_turnover,
                    max_exposure=config.max_exposure,
                    spread_bps=config.spread_bps,
                    slippage_bps=config.slippage_bps,
                    impact_coefficient=config.impact_coefficient,
                    reward_scale=config.reward_scale,
                    cost_penalty=config.cost_penalty,
                    turnover_penalty=config.turnover_penalty,
                    drawdown_penalty=config.drawdown_penalty,
                    seed=config.seed,
                )

                eval_vec_env = GPUVecTradingEnv(
                    n_envs=1,
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    family_encoder=encoder,
                    train_start=val_dates[0],
                    train_end=val_dates[1],
                    config=eval_gpu_config,
                    device="cuda",
                )

                # Wrap with VecNormalize as before
                from stable_baselines3.common.vec_env import VecNormalize

                train_vec_env = VecNormalize(
                    train_vec_env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    gamma=0.99,
                )
                eval_vec_env = VecNormalize(
                    eval_vec_env,
                    norm_obs=True,
                    norm_reward=False,
                    clip_obs=10.0,
                    training=False,
                )
                eval_vec_env.obs_rms = train_vec_env.obs_rms

                self.logger.info(f"GPU Training env: {n_envs} batched envs on CUDA")

            else:
                # ============================================================
                # ORIGINAL CPU PATH (unchanged)
                # ============================================================
                cost_model = CostModel(
                    spread_bps=config.spread_bps,
                    slippage_bps=config.slippage_bps,
                    impact_coefficient=config.impact_coefficient,
                )
                constraints = PortfolioConstraints(
                    max_weight=config.max_weight,
                    min_weight=config.min_weight,
                    max_turnover=config.max_turnover,
                    max_exposure=config.max_exposure,
                )
                reward_fn = RewardFunction(
                    reward_type=RewardType.ALPHA_PENALIZED,
                    cost_penalty_weight=config.cost_penalty,
                    turnover_penalty_weight=config.turnover_penalty,
                    drawdown_penalty_weight=config.drawdown_penalty,
                )
                episode_config = EpisodeConfig(
                    random_start=config.random_start,
                    episode_length=config.episode_length,
                    min_episode_length=26,
                    warmup_periods=4,
                )

                train_env = VecTradingEnv(
                    n_envs=n_envs,
                    use_subproc=use_subproc,
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    train_start=train_dates[0],
                    train_end=train_dates[1],
                    initial_capital=config.initial_capital,
                    rebalance_frequency=config.rebalance_frequency,
                    cost_model=cost_model,
                    constraints=constraints,
                    reward_function=reward_fn,
                    episode_config=episode_config,
                    reward_scale=config.reward_scale,
                )
                eval_episode_config = EpisodeConfig(
                    random_start=False,
                    episode_length=config.episode_length,
                    min_episode_length=26,
                )
                eval_env = VecTradingEnv(
                    n_envs=1,
                    use_subproc=False,
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    train_start=val_dates[0],
                    train_end=val_dates[1],
                    initial_capital=config.initial_capital,
                    rebalance_frequency=config.rebalance_frequency,
                    cost_model=cost_model,
                    constraints=constraints,
                    reward_function=reward_fn,
                    episode_config=eval_episode_config,
                    reward_scale=config.reward_scale,
                )

                from stable_baselines3.common.vec_env import VecNormalize
                train_vec_env = VecNormalize(train_env.envs, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
                eval_vec_env = VecNormalize(eval_env.envs, norm_obs=True, norm_reward=False, clip_obs=10.0,
                                            training=False)
                eval_vec_env.obs_rms = train_vec_env.obs_rms

            self.logger.info(f"Training env: {train_vec_env.num_envs} parallel envs (VecNormalize)")
            self.logger.info(f"Observation shape: {train_vec_env.observation_space.shape}")
            self.logger.info(f"Action shape: {train_vec_env.action_space.shape}")

            results['env'] = {
                'observation_shape': list(train_vec_env.observation_space.shape),
                'action_shape': list(train_vec_env.action_space.shape),
                'n_parallel_envs': train_vec_env.num_envs,
                'gpu_accelerated': use_gpu_env,
            }

        # ==================================================================
        # STEP 5.4: TRAIN AGENTS
        # ==================================================================
        agents_to_train = ['ppo', 'sac', 'td3'] if args.agent == 'all' else [args.agent]

        for agent_name in agents_to_train:
            with self.step(f"5.4 Train {agent_name.upper()} Agent"):
                agent_results = self._train_agent(
                    agent_name=agent_name,
                    train_env=train_vec_env,
                    eval_env=eval_vec_env,
                    config=config,
                    output_dir=output_dir,
                    resume_path=args.resume if agent_name == args.agent else None,
                    ppo_n_steps=ppo_n_steps,
                    ppo_batch_size=ppo_batch_size,
                    off_policy_batch_size=off_policy_batch_size,
                    buffer_size=buffer_size,
                    gradient_steps=gradient_steps,
                )
                results['agents_trained'].append(agent_results)

            # Cleanup between agents
            gc.collect()

        # ==================================================================
        # STEP 5.5: GENERATE REPORT
        # ==================================================================
        with self.step("5.5 Generate Report"):
            self._generate_report(results, output_dir)

            # ── run_info.json: lightweight manifest for comparison ────────
            # Captures the key settings so you can diff runs without reading logs.
            run_info = {
                "run_id": run_id,
                "timestamp": run_id[:15],  # YYYYMMDD_HHMMSS prefix
                "agents": [r["agent"] for r in results.get("agents_trained", [])],
                "total_timesteps": config.total_timesteps,
                "use_encoder": config.use_encoder,
                "encoder_type": "family" if config.use_family_encoder else "pca",
                "n_pca_components": config.n_pca_components,
                "reward": {
                    "scale": config.reward_scale,
                    "cost_penalty": config.cost_penalty,
                    "turnover_penalty": config.turnover_penalty,
                    "drawdown_penalty": config.drawdown_penalty,
                },
                "training": {
                    "learning_rate": config.learning_rate,
                    "net_arch": config.net_arch,
                    "n_envs": n_envs,
                },
                "validation": {
                    a["agent"]: a.get("validation", {})
                    for a in results.get("agents_trained", [])
                },
            }
            run_info_path = output_dir / "run_info.json"
            import json as _json
            import numpy as _np

            class _NumpyEncoder(_json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, _np.integer):
                        return int(obj)
                    if isinstance(obj, _np.floating):
                        return float(obj)
                    if isinstance(obj, _np.ndarray):
                        return obj.tolist()
                    return super().default(obj)

            run_info_path.write_text(_json.dumps(run_info, indent=2, cls=_NumpyEncoder))
            self.logger.info(f"Run manifest: {run_info_path}")

        return results

    def _train_agent(
        self,
        agent_name: str,
        train_env,
        eval_env,
        config: TrainingConfig,
        output_dir: Path,
        resume_path: Optional[str] = None,
        ppo_n_steps: int = 2048,
        ppo_batch_size: int = 64,
        off_policy_batch_size: int = 256,
        buffer_size: int = 100_000,
        gradient_steps: int = 1,
    ) -> dict:
        """Train a single agent."""
        from src.agents.ppo_agent import PPOAllocator
        from src.agents.sac_agent import SACAllocator
        from src.agents.td3_agent import TD3Allocator

        agent_dir = output_dir / "checkpoints" / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs" / agent_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create agent
        if agent_name == 'ppo':
            agent = PPOAllocator(
                env=train_env,
                learning_rate=config.learning_rate,
                n_steps=ppo_n_steps,
                batch_size=ppo_batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                # ent_coef=0: with 13513 action dims the entropy of a Normal is ~12k nats,
                # making entropy_loss ≈ -115 and dominating the policy gradient signal.
                ent_coef=0.0,
                vf_coef=0.25,   # Value fn already well-trained (expl_var≈0.97); reduce weight
                # max_grad_norm=0.05: output layer has 13513×256=3.5M params; tighter clip
                # prevents accumulated weight drift to NaN over hundreds of thousands of steps.
                max_grad_norm=0.05,
                net_arch=config.net_arch,
                seed=config.seed,
            )
        elif agent_name == 'sac':
            agent = SACAllocator(
                env=train_env,
                learning_rate=config.learning_rate,
                buffer_size=buffer_size,
                batch_size=off_policy_batch_size,
                tau=0.005,
                ent_coef='auto',
                gradient_steps=gradient_steps,
                net_arch=config.net_arch,
                seed=config.seed,
            )
        elif agent_name == 'td3':
            agent = TD3Allocator(
                env=train_env,
                learning_rate=config.learning_rate,
                buffer_size=buffer_size,
                batch_size=off_policy_batch_size,
                tau=0.005,
                policy_delay=2,
                gradient_steps=gradient_steps,
                net_arch=config.net_arch,
                seed=config.seed,
            )
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        self.logger.info(f"Created {agent_name.upper()} agent")
        self.logger.info(f"Hyperparameters: {agent.get_hyperparameters()}")

        # Resume if checkpoint provided
        if resume_path:
            agent.load(resume_path)
            self.logger.info(f"Resumed from: {resume_path}")

        # Train
        self.logger.info(f"Training for {config.total_timesteps:,} timesteps...")
        agent.train(
            total_timesteps=config.total_timesteps,
            eval_env=eval_env,
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            save_path=str(agent_dir),
            log_to_csv=True,
            progress_bar=False,  # Avoid tqdm/rich dependency
        )

        # Save final model
        final_path = agent_dir / "final_model"
        agent.save(final_path)
        self.logger.info(f"Saved final model to: {final_path}")

        # Save VecNormalize stats alongside the model so they can be restored on load
        from stable_baselines3.common.vec_env import VecNormalize
        if isinstance(train_env, VecNormalize):
            norm_path = agent_dir / "vecnormalize.pkl"
            train_env.save(str(norm_path))
            self.logger.info(f"Saved VecNormalize stats to: {norm_path}")

        # Save encoder so it can be reloaded for inference / evaluation
        import pickle
        from stable_baselines3.common.vec_env import VecNormalize as _VN  # local alias
        _inner = train_env.venv if isinstance(train_env, _VN) else train_env
        _env0 = _inner.envs[0] if hasattr(_inner, 'envs') else None
        # Try to get encoder from CPU env or GPU env
        _encoder = None
        if _env0 is not None and hasattr(_env0, 'encoder') and _env0.encoder is not None:
            _encoder = _env0.encoder
        elif hasattr(_inner, 'encoder') and _inner.encoder is not None:
            _encoder = _inner.encoder
        if _encoder is not None:
            enc_path = agent_dir / "universe_encoder.pkl"
            with open(enc_path, "wb") as f:
                pickle.dump(_encoder, f)
            self.logger.info(f"Saved universe encoder to: {enc_path}")

        # Get training metrics
        metrics = agent.get_training_metrics()
        metrics_df = metrics.to_dataframe()

        if not metrics_df.empty:
            metrics_df.to_csv(logs_dir / "metrics.csv", index=False)

            # Summary stats
            summary = metrics.get_summary()
            self.logger.info(f"Training summary:")
            self.logger.info(f"  Mean reward: {summary.get('mean_reward', 0):.2f}")
            self.logger.info(f"  Final Sharpe: {summary.get('final_sharpe', 0):.3f}")
            self.logger.info(f"  Max drawdown: {summary.get('max_drawdown', 0):.2%}")
        else:
            summary = {}

        # Evaluate on validation set
        self.logger.info("Evaluating on validation set...")
        eval_metrics = agent.evaluate(eval_env, n_episodes=config.n_eval_episodes)
        self.logger.info(f"Validation mean reward: {eval_metrics['mean_reward']:.2f}")
        self.logger.info(f"Validation mean return: {eval_metrics.get('mean_portfolio_return', 0):.2%}")

        return {
            'agent': agent_name,
            'timesteps': config.total_timesteps,
            'model_path': str(final_path),
            'training_summary': summary,
            'validation': {
                'mean_reward': eval_metrics['mean_reward'],
                'std_reward': eval_metrics['std_reward'],
                'mean_return': eval_metrics.get('mean_portfolio_return', 0),
            }
        }

    def _load_data(self, input_dir: Path):
        """Load Phase 1 data."""
        # Load returns matrix - try new path first, then legacy
        returns_path = self.dp.algorithms.returns
        if not returns_path.exists():
            returns_path = input_dir / 'algo_returns.parquet'
        if not returns_path.exists():
            raise FileNotFoundError(f"Returns matrix not found: {returns_path}")

        algo_returns = pd.read_parquet(returns_path)
        algo_returns = algo_returns.astype(np.float32, copy=False)
        # Normalize timezone (remove if present)
        if algo_returns.index.tz is not None:
            algo_returns.index = algo_returns.index.tz_localize(None)
        self.logger.info(f"Loaded returns: {algo_returns.shape}")

        # Load benchmark weights
        weights_path = self.dp.benchmark.weights
        if not weights_path.exists():
            weights_path = input_dir / 'benchmark_weights.parquet'

        benchmark_weights = None
        if weights_path.exists():
            benchmark_weights = pd.read_parquet(weights_path)
            benchmark_weights = benchmark_weights.astype(np.float32, copy=False)
            # Normalize timezone (remove if present)
            if benchmark_weights.index.tz is not None:
                benchmark_weights.index = benchmark_weights.index.tz_localize(None)
            self.logger.info(f"Loaded benchmark weights: {benchmark_weights.shape}")

        # Load benchmark returns
        bench_path = self.dp.benchmark.daily_returns
        if not bench_path.exists():
            bench_path = input_dir / 'benchmark_daily_returns.csv'

        benchmark_returns = None
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path, index_col=0, parse_dates=True)
            if 'return' in bench_df.columns:
                benchmark_returns = bench_df['return']
            else:
                benchmark_returns = bench_df.iloc[:, 0]
            self.logger.info(f"Loaded benchmark returns: {len(benchmark_returns)} days")

        return algo_returns, benchmark_weights, benchmark_returns

    def _generate_report(self, results: dict, output_dir: Path):
        """Generate markdown report."""
        config = results['config']

        report = f"""# Phase 5: RL Agent Training Report

**Generated**: {datetime.now().isoformat()}
**Mode**: {results['mode']}

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Timesteps | {config['total_timesteps']:,} |
| Learning Rate | {config['learning_rate']} |
| Network Architecture | {config['net_arch']} |
| Rebalance Frequency | {config['rebalance_frequency']} |
| Episode Length | {config['episode_length']} |
| Seed | {config['seed']} |

## Data Summary

| Metric | Value |
|--------|-------|
| Algorithms | {results['n_algos']} |
| Trading Days | {results['n_days']} |

## Train/Val/Test Splits

| Split | Start | End |
|-------|-------|-----|
| Train | {results['splits']['train'][0]} | {results['splits']['train'][1]} |
| Val | {results['splits']['val'][0]} | {results['splits']['val'][1]} |
| Test | {results['splits']['test'][0]} | {results['splits']['test'][1]} |

## Agents Trained

"""
        for agent_result in results['agents_trained']:
            report += f"""### {agent_result['agent'].upper()}

| Metric | Value |
|--------|-------|
| Timesteps | {agent_result['timesteps']:,} |
| Model Path | `{agent_result['model_path']}` |
| Val Mean Reward | {agent_result['validation']['mean_reward']:.2f} |
| Val Std Reward | {agent_result['validation']['std_reward']:.2f} |
| Val Mean Return | {agent_result['validation']['mean_return']:.2%} |

"""
            if agent_result.get('training_summary'):
                summary = agent_result['training_summary']
                report += f"""**Training Summary:**
- Mean Reward: {summary.get('mean_reward', 'N/A')}
- Final Sharpe: {summary.get('final_sharpe', 'N/A')}
- Max Drawdown: {summary.get('max_drawdown', 'N/A')}

"""

        report += """## Next Steps

1. Run Phase 6 (Evaluation) to compare against baselines
2. Perform walk-forward validation
3. Analyze training curves in `outputs/rl_training/logs/`

## Files Generated

- `checkpoints/*/final_model.zip` - Trained model
- `logs/*/training_log.csv` - Training logs
- `logs/*/metrics.csv` - Training metrics
"""

        # Save report
        report_path = output_dir / "PHASE5_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        self.logger.info(f"Report saved to: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    runner = Phase5Runner()
    runner.execute()


if __name__ == "__main__":
    main()
