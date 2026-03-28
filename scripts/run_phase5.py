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
  python scripts/run_phase5.py --run-id exp01   # Custom run identifier

Options:
  --agent NAME         Agent to train: ppo, sac, td3, all (default: ppo)
  --quick              Quick test mode (10k steps, 20 algos)
  --timesteps N        Total training timesteps (default: 500K, supports K/M suffix)
  --sample N           Use only N algorithms
  --eval-freq N        Evaluation frequency in timesteps (default: 10k)
  --n-eval-episodes N  Episodes per evaluation (default: 5)
  --rebalance-freq F   Rebalance frequency: daily, weekly, monthly (default: weekly)
  --seed N             Random seed (default: 42)
  --learning-rate F    Learning rate (default: 1e-4)
  --input-dir PATH     Override input directory (Phase 1 outputs)
  --resume PATH        Path to checkpoint to resume training from

Run Management:
  --run-id ID          Identifier for this training run (default: auto YYYYMMDD_HHMMSS)
                       Outputs go to outputs/rl_training/{run_id}/
                       Use same ID with run_phase6.py --run-id to evaluate

Parallelization:
  --max-resources      Auto-configure to fully use GPU/CPU/RAM
                       Detects VRAM to set n_envs, batch_size, net_arch automatically
  --n-envs N           Override number of parallel envs (default: 4)
  --use-subproc        Use SubprocVecEnv for true process parallelism
  --gpu-env            Use GPU-accelerated batched environment (requires CUDA)

Encoder Options:
  --no-encoder         Disable AlgoUniverseEncoder (use raw 13k-dim obs/action spaces)
  --pca-encoder        Use PCA-based encoder instead of FamilyEncoder (default: FamilyEncoder)
  --pca-components N   Number of PCA components for encoder (default: 50)
  --min-days-active N  Min active days for Stage 1 filter (default: 21)

Hybrid Mode (MPT base + RL tilts):
  --no-hybrid          Disable hybrid mode (use absolute weights instead)
  --base-allocator S   Base allocator: risk_parity, equal_weight, min_variance
                       (default: risk_parity)
  --max-tilt F         Maximum tilt per asset in hybrid mode (default: 0.15 = ±15%)
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
    episode_length: int = 104  # weeks (2 years - longer episodes for better learning)
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
    # The benchmark is only a risk anchor. The objective is absolute return.
    reward_type: str = "risk_calibrated_returns"
    reward_scale: float = 100.0
    # Penalty weights for absolute-return rewards
    cost_penalty: float = 1.0
    turnover_penalty: float = 0.005
    drawdown_penalty: float = 0.1

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

    # Hybrid mode: MPT base + RL tilts (recommended for stability)
    hybrid_mode: bool = True  # Enable by default for better training stability
    base_allocator: str = "risk_parity"  # "risk_parity", "equal_weight", "min_variance"
    max_tilt: float = 0.15  # Maximum tilt per asset (±15%)

    # Behavioral Cloning warm-start (--pretrain-bc)
    bc_pretrain: bool = False
    bc_strategy: str = "risk_parity"        # Baseline allocator to imitate
    bc_epochs: int = 10                      # Supervised epochs before RL
    bc_lr: float = 1e-3                      # BC learning rate (higher than RL)
    bc_batch_size: int = 256
    bc_lookback: int = 63                    # Lookback for the expert allocator

    # Cluster-based universe filter (--cluster-filter)
    cluster_filter: bool = False
    cluster_filter_mode: str = "hard"        # "hard" (pre-filter) or "soft" (reward shaping)
    cluster_score_metric: str = "sharpe"     # "sharpe", "return", "sortino"
    cluster_score_threshold: float = 0.0     # Remove families with score < threshold
    cluster_bonus_weight: float = 0.001      # Reward bonus weight (soft mode)
    phase2_cluster_filter: bool = False
    phase2_analysis_dir: Optional[str] = None
    phase2_cluster_source: str = "behavioral_family"
    phase2_cluster_score_mode: str = "return_low_vol"
    phase2_cluster_top_k: int = 1
    phase2_cluster_min_size: int = 25
    phase2_cluster_min_return: Optional[float] = 0.0
    phase2_cluster_max_vol: Optional[float] = None
    phase2_cluster_latest_only: bool = True


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

    def _make_run_id(self) -> str:
        """Phase 5 run ID: timestamp + agent + optional bc/cluster flags."""
        if getattr(self.args, 'run_id', None):
            base = self.args.run_id
        else:
            base = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [base]
        agent = getattr(self.args, 'agent', 'ppo')
        if agent != 'all':
            parts.append(agent)
        if getattr(self.args, 'pretrain_bc', False):
            strategy = getattr(self.args, 'bc_strategy', 'rp')
            parts.append(f"bc_{strategy[:2]}")
        if getattr(self.args, 'cluster_filter', False):
            mode = getattr(self.args, 'cluster_filter_mode', 'hard')[:1]
            parts.append(f"cf{mode}")
        return "_".join(parts)

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
            help='Override input directory (processed dataset root for training/eval universe)'
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
        # Hybrid mode options
        parser.add_argument(
            '--no-hybrid', action='store_true',
            help='Disable hybrid mode (use absolute weights instead of MPT base + tilts)'
        )
        parser.add_argument(
            '--base-allocator', type=str, default='risk_parity',
            choices=['risk_parity', 'equal_weight', 'min_variance'],
            help='Base allocator for hybrid mode (default: risk_parity)'
        )
        parser.add_argument(
            '--max-tilt', type=float, default=0.15,
            help='Maximum tilt per asset in hybrid mode (default: 0.15 = ±15%%)'
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
        # ------------------------------------------------------------------
        # Behavioral Cloning warm-start
        # ------------------------------------------------------------------
        parser.add_argument(
            '--pretrain-bc', action='store_true',
            help=(
                'Pre-train the RL policy to imitate a Phase 3 baseline via behavioral cloning '
                'before PPO/SAC/TD3 fine-tuning. Reduces cold-start exploration.'
            )
        )
        parser.add_argument(
            '--bc-strategy', type=str, default='risk_parity',
            choices=['equal_weight', 'risk_parity', 'min_variance', 'max_sharpe', 'momentum', 'vol_targeting'],
            help='Phase 3 allocator to imitate during BC pre-training (default: risk_parity)'
        )
        parser.add_argument(
            '--bc-epochs', type=int, default=10,
            help='Number of supervised BC epochs (default: 10)'
        )
        parser.add_argument(
            '--bc-lr', type=float, default=1e-3,
            help='Learning rate for BC pre-training (default: 1e-3, higher than RL)'
        )
        parser.add_argument(
            '--bc-batch-size', type=int, default=256,
            help='Mini-batch size for BC pre-training (default: 256)'
        )
        parser.add_argument(
            '--bc-lookback', type=int, default=63,
            help='Lookback window (days) for the expert allocator (default: 63)'
        )
        # ------------------------------------------------------------------
        # Cluster-based universe filter
        # ------------------------------------------------------------------
        parser.add_argument(
            '--cluster-filter', action='store_true',
            help=(
                'Filter the algorithm universe using Phase 2 cluster quality scores. '
                'Hard mode removes low-scoring families; soft mode shapes rewards.'
            )
        )
        parser.add_argument(
            '--cluster-filter-mode', type=str, default='hard', choices=['hard', 'soft'],
            help=(
                'hard: remove algorithms from low-scoring families before training. '
                'soft: add a per-step reward bonus for allocating to high-scoring families. '
                '(default: hard)'
            )
        )
        parser.add_argument(
            '--cluster-score-metric', type=str, default='sharpe',
            choices=['sharpe', 'return', 'sortino'],
            help='Metric used to score families (default: sharpe)'
        )
        parser.add_argument(
            '--cluster-score-threshold', type=float, default=0.0,
            help='Remove families with median score below this threshold (default: 0.0)'
        )
        parser.add_argument(
            '--cluster-bonus-weight', type=float, default=0.001,
            help='Reward bonus weight for soft mode (default: 0.001)'
        )
        parser.add_argument(
            '--reward-type', type=str, default='risk_calibrated_returns',
            choices=['risk_calibrated_returns', 'absolute_returns', 'calibrated_alpha', 'pure_returns', 'alpha_penalized', 'info_ratio', 'risk_adjusted'],
            help='Reward function type (default: risk_calibrated_returns)'
        )
        parser.add_argument(
            '--phase2-cluster-filter', action='store_true',
            help='Filter the Phase 5 universe using the best clusters detected in Phase 2'
        )
        parser.add_argument(
            '--phase2-analysis-dir', type=str, default=None,
            help='Path to a concrete Phase 2 analysis directory, snapshot, or outputs/analysis run dir'
        )
        parser.add_argument(
            '--phase2-cluster-source', type=str, default='behavioral_family',
            choices=['behavioral_family', 'temporal_cumulative', 'temporal_weekly', 'temporal_monthly'],
            help='Phase 2 cluster source used to define the tradable universe'
        )
        parser.add_argument(
            '--phase2-cluster-score-mode', type=str, default='return_low_vol',
            choices=['return_low_vol', 'return', 'sharpe', 'sortino'],
            help='How to score Phase 2 clusters before selecting the top ones'
        )
        parser.add_argument(
            '--phase2-cluster-top-k', type=int, default=1,
            help='Number of best Phase 2 clusters to include in the training universe'
        )
        parser.add_argument(
            '--phase2-cluster-min-size', type=int, default=25,
            help='Minimum size required for a Phase 2 cluster to be eligible'
        )
        parser.add_argument(
            '--phase2-cluster-min-return', type=float, default=0.0,
            help='Optional minimum mean annualized return required at cluster level'
        )
        parser.add_argument(
            '--phase2-cluster-max-vol', type=float, default=None,
            help='Optional maximum mean annualized volatility allowed at cluster level'
        )
        parser.add_argument(
            '--phase2-cluster-full-history', action='store_true',
            help='For temporal clusters, score clusters over full history instead of only the latest assignment'
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
            # Hybrid mode
            hybrid_mode=not getattr(args, 'no_hybrid', False),
            base_allocator=getattr(args, 'base_allocator', 'risk_parity'),
            max_tilt=getattr(args, 'max_tilt', 0.15),
            # BC warm-start
            bc_pretrain=getattr(args, 'pretrain_bc', False),
            bc_strategy=getattr(args, 'bc_strategy', 'risk_parity'),
            bc_epochs=getattr(args, 'bc_epochs', 10),
            bc_lr=getattr(args, 'bc_lr', 1e-3),
            bc_batch_size=getattr(args, 'bc_batch_size', 256),
            bc_lookback=getattr(args, 'bc_lookback', 63),
            # Cluster filter
            cluster_filter=getattr(args, 'cluster_filter', False),
            cluster_filter_mode=getattr(args, 'cluster_filter_mode', 'hard'),
            cluster_score_metric=getattr(args, 'cluster_score_metric', 'sharpe'),
            cluster_score_threshold=getattr(args, 'cluster_score_threshold', 0.0),
            cluster_bonus_weight=getattr(args, 'cluster_bonus_weight', 0.001),
            reward_type=getattr(args, 'reward_type', 'risk_calibrated_returns'),
            phase2_cluster_filter=getattr(args, 'phase2_cluster_filter', False),
            phase2_analysis_dir=getattr(args, 'phase2_analysis_dir', None),
            phase2_cluster_source=getattr(args, 'phase2_cluster_source', 'behavioral_family'),
            phase2_cluster_score_mode=getattr(args, 'phase2_cluster_score_mode', 'return_low_vol'),
            phase2_cluster_top_k=getattr(args, 'phase2_cluster_top_k', 1),
            phase2_cluster_min_size=getattr(args, 'phase2_cluster_min_size', 25),
            phase2_cluster_min_return=getattr(args, 'phase2_cluster_min_return', None),
            phase2_cluster_max_vol=getattr(args, 'phase2_cluster_max_vol', None),
            phase2_cluster_latest_only=not getattr(args, 'phase2_cluster_full_history', False),
        )

        # Output dir and run_id are managed by PhaseRunner.execute() via _make_run_id()
        output_dir = self.get_output_dir()
        run_id = self._run_id or output_dir.name
        self.logger.info(f"Output: {output_dir}")

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

            if config.phase2_cluster_filter:
                from src.analysis.phase2_cluster_selector import (
                    Phase2ClusterSelectionConfig,
                    Phase2ClusterSelector,
                )

                analysis_root = Path(config.phase2_analysis_dir) if config.phase2_analysis_dir else self.dp.processed.analysis.root
                selector = Phase2ClusterSelector(analysis_root)
                selection_cfg = Phase2ClusterSelectionConfig(
                    source=config.phase2_cluster_source,
                    score_mode=config.phase2_cluster_score_mode,
                    top_k=config.phase2_cluster_top_k,
                    min_cluster_size=config.phase2_cluster_min_size,
                    min_return=config.phase2_cluster_min_return,
                    max_vol=config.phase2_cluster_max_vol,
                    latest_only=config.phase2_cluster_latest_only,
                )
                selected_algos, cluster_ranking, selected_clusters = selector.select(selection_cfg)
                selected_cols = [c for c in algo_returns.columns if c in selected_algos]
                if not selected_cols:
                    raise ValueError(
                        "Phase 2 cluster selection produced an empty universe after intersecting with returns."
                    )
                algo_returns = algo_returns[selected_cols]
                if benchmark_weights is not None:
                    benchmark_weights = benchmark_weights.reindex(columns=selected_cols, fill_value=0.0)

                cluster_selection_dir = output_dir / 'cluster_selection'
                cluster_selection_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({'algo_id': selected_cols}).to_csv(
                    cluster_selection_dir / 'selected_algos.csv',
                    index=False,
                )
                cluster_ranking.reset_index().to_csv(
                    cluster_selection_dir / 'cluster_ranking.csv',
                    index=False,
                )

                results['phase2_cluster_selection'] = {
                    'analysis_dir': str(selector.analysis_root.resolve()),
                    'source': config.phase2_cluster_source,
                    'score_mode': config.phase2_cluster_score_mode,
                    'top_k': config.phase2_cluster_top_k,
                    'selected_clusters': selected_clusters,
                    'selected_algo_count': len(selected_cols),
                    'latest_only': config.phase2_cluster_latest_only,
                }
                self.logger.info(
                    f"Phase 2 cluster selection applied: analysis={selector.analysis_root}, "
                    f"source={config.phase2_cluster_source}, "
                    f"clusters={selected_clusters}, algos={len(selected_cols)}"
                )

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
        # STEP 5.1.5: CLUSTER UNIVERSE FILTER (optional)
        # ==================================================================
        cluster_filter_obj = None
        if config.cluster_filter:
            with self.step("5.1.5 Cluster Universe Filter"):
                from src.environment.universe_filter import ClusterUniverseFilter, ClusterFilterConfig

                family_labels_path = (
                    self.dp.processed.root / 'analysis' / 'clustering' / 'behavioral' / 'family_labels.csv'
                )
                behavioral_features_path = (
                    self.dp.processed.root / 'analysis' / 'clustering' / 'behavioral' / 'features.csv'
                )

                if not family_labels_path.exists() or not behavioral_features_path.exists():
                    self.logger.warning(
                        "Cluster data not found (Phase 2 must be run first). "
                        "Skipping cluster filter."
                    )
                    config.cluster_filter = False
                else:
                    cf_config = ClusterFilterConfig(
                        mode=config.cluster_filter_mode,
                        score_metric=config.cluster_score_metric,
                        threshold=config.cluster_score_threshold,
                        bonus_weight=config.cluster_bonus_weight,
                    )
                    cluster_filter_obj = ClusterUniverseFilter(cf_config)
                    cluster_filter_obj.load_cluster_data(family_labels_path, behavioral_features_path)

                    if config.cluster_filter_mode == "hard":
                        algo_returns, benchmark_weights = cluster_filter_obj.apply_hard_filter(
                            algo_returns, benchmark_weights
                        )
                        results['cluster_filter'] = {
                            'mode': 'hard',
                            'n_algos_after': algo_returns.shape[1],
                            'included_families': sorted(cluster_filter_obj.get_included_families()),
                            'family_scores': {
                                str(k): round(v, 4)
                                for k, v in cluster_filter_obj.get_family_scores().items()
                            },
                        }
                        self.logger.info(
                            f"Universe after hard filter: "
                            f"{algo_returns.shape[1]} algos, "
                            f"{algo_returns.shape[0]} days"
                        )
                    else:
                        # Soft mode: prepare per-algo scores; pass filter to env later
                        cluster_filter_obj.prepare_for_env(algo_returns.columns.tolist())
                        results['cluster_filter'] = {
                            'mode': 'soft',
                            'bonus_weight': config.cluster_bonus_weight,
                            'family_scores': {
                                str(k): round(v, 4)
                                for k, v in cluster_filter_obj.get_family_scores().items()
                            },
                        }
                        self.logger.info("Soft cluster filter ready (reward shaping enabled)")

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

            # Define cost_model, constraints, reward_fn unconditionally — the BC
            # pre-training step always uses a CPU TradingEnvironment regardless of
            # whether the main training env is GPU-accelerated.
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
            _reward_type_map = {
                "risk_calibrated_returns": RewardType.RISK_CALIBRATED_RETURNS,
                "absolute_returns": RewardType.ABSOLUTE_RETURNS,
                "calibrated_alpha": RewardType.RISK_CALIBRATED_RETURNS,
                "pure_returns": RewardType.PURE_RETURNS,
                "alpha_penalized": RewardType.ABSOLUTE_RETURNS,
                "info_ratio": RewardType.INFORMATION_RATIO,
                "risk_adjusted": RewardType.RISK_ADJUSTED,
            }
            reward_fn = RewardFunction(
                reward_type=_reward_type_map.get(config.reward_type, RewardType.RISK_CALIBRATED_RETURNS),
                cost_penalty_weight=config.cost_penalty,
                turnover_penalty_weight=config.turnover_penalty,
                drawdown_penalty_weight=config.drawdown_penalty,
            )
            self.logger.info(f"Using reward type: {config.reward_type}")

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
                    risk_penalty=0.2,
                    risk_tolerance=1.2,
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
                    risk_penalty=0.2,
                    risk_tolerance=1.2,
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
                # ORIGINAL CPU PATH
                # ============================================================
                # cost_model, constraints, reward_fn already defined above
                self.logger.info(f"Using reward type: {config.reward_type}")
                episode_config = EpisodeConfig(
                    random_start=config.random_start,
                    episode_length=config.episode_length,
                    min_episode_length=26,
                    warmup_periods=4,
                )

                # Log hybrid mode status
                if config.hybrid_mode:
                    self.logger.info(f"Hybrid mode: {config.base_allocator} base + RL tilts (max_tilt={config.max_tilt})")
                else:
                    self.logger.info("Standard mode: RL outputs absolute weights")

                # Pass soft-mode cluster filter to env (None for hard mode or disabled)
                _soft_filter = (
                    cluster_filter_obj
                    if config.cluster_filter and config.cluster_filter_mode == "soft"
                    else None
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
                    # Hybrid mode parameters
                    hybrid_mode=config.hybrid_mode,
                    base_allocator=config.base_allocator,
                    max_tilt=config.max_tilt,
                    # Encoder for dimensionality reduction (critical for stability!)
                    encoder=encoder,
                    # Soft cluster filter for reward shaping
                    cluster_filter=_soft_filter,
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
                    # Hybrid mode parameters
                    hybrid_mode=config.hybrid_mode,
                    base_allocator=config.base_allocator,
                    max_tilt=config.max_tilt,
                    # Encoder for dimensionality reduction
                    encoder=encoder,
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
        # STEP 5.3.5: BEHAVIORAL CLONING PRE-TRAINING (optional)
        # ==================================================================
        bc_demonstrations = None
        if config.bc_pretrain:
            with self.step("5.3.5 Behavioral Cloning Pre-training"):
                from src.agents.bc_pretrainer import BehavioralCloningPretrainer, BCConfig
                from src.environment.trading_env import TradingEnvironment, EpisodeConfig as _EC

                bc_cfg = BCConfig(
                    strategy=config.bc_strategy,
                    epochs=config.bc_epochs,
                    lr=config.bc_lr,
                    batch_size=config.bc_batch_size,
                    lookback=config.bc_lookback,
                    max_weight=config.max_weight,
                    min_weight=config.min_weight,
                    max_turnover=config.max_turnover,
                )
                bc = BehavioralCloningPretrainer(bc_cfg)

                # Try to load Phase 1 features for factor-based strategies
                features_for_bc = None
                if config.bc_strategy in ("max_sharpe", "momentum", "vol_targeting"):
                    features_path = input_dir / 'algo_features.parquet'
                    if not features_path.exists():
                        features_path = input_dir / 'algorithms' / 'features.parquet'
                    if features_path.exists():
                        features_for_bc = pd.read_parquet(features_path)
                        self.logger.info(f"Loaded algo features for BC: {features_path.name}")
                    else:
                        self.logger.warning(
                            f"Features file not found for strategy '{config.bc_strategy}'. "
                            "BC will proceed without features (may fall back to rank-based selection)."
                        )

                self.logger.info(
                    f"BC strategy={config.bc_strategy}, "
                    f"epochs={config.bc_epochs}, lr={config.bc_lr}"
                )
                expert_weights = bc.generate_expert_weights(
                    algo_returns=algo_returns,
                    train_start=train_dates[0],
                    train_end=train_dates[1],
                    features=features_for_bc,
                )

                # Create a single non-vectorized env for rollout collection
                bc_episode_cfg = _EC(
                    random_start=False,
                    episode_length=None,  # run full training window
                )
                bc_env = TradingEnvironment(
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    cost_model=cost_model,
                    constraints=constraints,
                    reward_function=reward_fn,
                    initial_capital=config.initial_capital,
                    rebalance_frequency=config.rebalance_frequency,
                    train_start=train_dates[0],
                    train_end=train_dates[1],
                    reward_scale=config.reward_scale,
                    episode_config=bc_episode_cfg,
                    encoder=encoder,
                    hybrid_mode=config.hybrid_mode,
                    base_allocator=config.base_allocator,
                    max_tilt=config.max_tilt,
                )
                bc_demonstrations = bc.collect_demonstrations(
                    env=bc_env,
                    expert_weights=expert_weights,
                    encoder=encoder,
                    hybrid_mode=config.hybrid_mode,
                    max_tilt=config.max_tilt,
                )
                results['bc_pretrain'] = {
                    'strategy': config.bc_strategy,
                    'n_demonstrations': int(bc_demonstrations[0].shape[0])
                    if bc_demonstrations[0].ndim > 0
                    else 0,
                    'epochs': config.bc_epochs,
                }
                del bc_env
                gc.collect()

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
                    bc_demonstrations=bc_demonstrations,
                    bc_config=config if config.bc_pretrain else None,
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
                "input_dir": str(input_dir.resolve()),
                "phase2_cluster_selection": results.get("phase2_cluster_selection"),
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
        bc_demonstrations=None,
        bc_config=None,
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

        # Behavioral Cloning warm-start (runs before RL training)
        bc_results = None
        if bc_demonstrations is not None and bc_config is not None:
            obs_arr, act_arr = bc_demonstrations
            if len(obs_arr) > 0:
                from src.agents.bc_pretrainer import BehavioralCloningPretrainer, BCConfig
                bc_cfg = BCConfig(
                    strategy=bc_config.bc_strategy,
                    epochs=bc_config.bc_epochs,
                    lr=bc_config.bc_lr,
                    batch_size=bc_config.bc_batch_size,
                )
                bc = BehavioralCloningPretrainer(bc_cfg)
                self.logger.info(
                    f"Starting BC pre-training ({bc_config.bc_epochs} epochs, "
                    f"lr={bc_config.bc_lr}, strategy={bc_config.bc_strategy})"
                )
                bc_results = bc.pretrain(agent.model, bc_demonstrations)
                self.logger.info(
                    f"BC pre-training done: "
                    f"final_loss={bc_results['loss_history'][-1]:.6f}"
                    if bc_results.get('loss_history')
                    else "BC pre-training done (no loss recorded)"
                )
            else:
                self.logger.warning("BC demonstrations empty — skipping BC pre-training.")

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
        # Load returns matrix - honor input_dir override first
        if input_dir != self.dp.processed.root:
            candidates = [
                input_dir / 'algorithms' / 'returns.parquet',
                input_dir / 'algo_returns.parquet',
                input_dir / 'returns.parquet',
            ]
            returns_path = next((p for p in candidates if p.exists()), self.dp.algorithms.returns)
        else:
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
        if input_dir != self.dp.processed.root:
            candidates = [
                input_dir / 'benchmark' / 'weights.parquet',
                input_dir / 'benchmark_weights.parquet',
                input_dir / 'weights.parquet',
            ]
            weights_path = next((p for p in candidates if p.exists()), self.dp.benchmark.weights)
        else:
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
        if input_dir != self.dp.processed.root:
            candidates = [
                input_dir / 'benchmark' / 'daily_returns.csv',
                input_dir / 'benchmark_daily_returns.csv',
                input_dir / 'daily_returns.csv',
            ]
            bench_path = next((p for p in candidates if p.exists()), self.dp.benchmark.daily_returns)
        else:
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
