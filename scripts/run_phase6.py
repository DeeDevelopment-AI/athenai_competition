#!/usr/bin/env python3
"""
=================================================================
PHASE 6: Walk-Forward Evaluation & Comparison
=================================================================
Complete Phase 6 pipeline to evaluate trained RL agents:
  6.1 Load trained models from Phase 5
  6.2 Run walk-forward validation
  6.3 Evaluate baselines with same protocol
  6.4 Compare all strategies
  6.5 Generate final report

Walk-Forward Protocol:
  - Train window: 252 days (1 year)
  - Validation window: 63 days (1 quarter)
  - Test window: 63 days (1 quarter)
  - Step size: 63 days (rolling quarterly)

Usage:
  python scripts/run_phase6.py                      # Evaluate PPO from latest run
  python scripts/run_phase6.py --agent all          # Evaluate all trained agents
  python scripts/run_phase6.py --quick              # Quick test (shorter windows)
  python scripts/run_phase6.py --include-baselines  # Include baseline comparison
  python scripts/run_phase6.py --folds 3            # Limit number of folds
  python scripts/run_phase6.py --run-id 20260323_143000_ppo  # Specific training run
  python scripts/run_phase6.py --list-runs          # List all available runs

Options:
  --agent NAME           Agent to evaluate: ppo, sac, td3, all (default: ppo)
  --quick                Quick test mode (shorter windows, smaller sample)
  --include-baselines    Also evaluate classical baselines
  --sample N             Use only N algorithms (must match training)
  --seed N               Random seed (default: 42)

Model Selection:
  --list-runs            List all available training runs with parameters and exit
  --run-id ID            Evaluate models from a specific training run
                         Default: reads outputs/rl_training/latest_run.txt
                         Example: --run-id 20260323_143000_ppo
  --models-dir PATH      Explicit path to directory containing trained models
                         Takes priority over --run-id
                         Example: --models-dir outputs/rl_training/20260323_143000/checkpoints

Walk-Forward Configuration:
  --folds N              Maximum number of folds (default: all)
  --train-window N       Training window in days (default: 252)
  --val-window N         Validation window in days (default: 63)
  --test-window N        Test window in days (default: 63)
  --step-size N          Step size between folds (default: 63)
  --expanding            Use expanding window (default: rolling)
"""

import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.base_runner import PhaseRunner
from src.evaluation.audit import (
    build_periodic_allocation_rows,
    compress_daily_weight_history,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Walk-forward windows
    train_window: int = 252      # 1 year
    val_window: int = 63         # 1 quarter
    test_window: int = 63        # 1 quarter
    step_size: int = 63          # quarterly step
    expanding: bool = False      # rolling vs expanding
    max_folds: Optional[int] = None

    # Environment
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "weekly"

    # Constraints (same as training)
    max_weight: float = 0.40
    min_weight: float = 0.00
    max_turnover: float = 0.30
    max_exposure: float = 1.0

    # Costs (same as training)
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    impact_coefficient: float = 0.1

    # Reward (for consistency)
    reward_scale: float = 100.0

    # Seed
    seed: int = 42


# =============================================================================
# Phase 6 Runner
# =============================================================================

class Phase6Runner(PhaseRunner):
    """Phase 6: Walk-Forward Evaluation."""

    phase_name = "Phase 6: Walk-Forward Evaluation"
    phase_number = 6

    def _run_tag(self) -> str:
        if not self.args:
            return ""
        agent = getattr(self.args, 'agent', 'ppo')
        source = getattr(self.args, 'run_id', None) or "latest"
        tag = f"{agent}_{source}"
        if getattr(self.args, 'quick', False):
            tag += "_quick"
        return tag

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add Phase 6 specific arguments."""
        parser.add_argument(
            '--agent', type=str, default='ppo',
            choices=['ppo', 'sac', 'td3', 'all'],
            help='Agent to evaluate (default: ppo)'
        )
        parser.add_argument(
            '--quick', '-q', action='store_true',
            help='Quick test mode (shorter windows)'
        )
        parser.add_argument(
            '--include-baselines', '-b', action='store_true',
            help='Include classical baselines in comparison'
        )
        parser.add_argument(
            '--folds', type=int, default=None,
            help='Maximum number of folds'
        )
        parser.add_argument(
            '--train-window', type=int, default=252,
            help='Training window in days (default: 252)'
        )
        parser.add_argument(
            '--val-window', type=int, default=63,
            help='Validation window in days (default: 63)'
        )
        parser.add_argument(
            '--test-window', type=int, default=63,
            help='Test window in days (default: 63)'
        )
        parser.add_argument(
            '--step-size', type=int, default=63,
            help='Step size between folds (default: 63)'
        )
        parser.add_argument(
            '--expanding', action='store_true',
            help='Use expanding window instead of rolling'
        )
        parser.add_argument(
            '--models-dir', type=str, default=None,
            help=(
                'Explicit path to directory containing trained models. '
                'Takes priority over --run-id. '
                'Example: --models-dir outputs/rl_training/20260323_143000/checkpoints'
            )
        )
        parser.add_argument(
            '--run-id', type=str, default=None,
            help=(
                'Evaluate models from a specific training run. '
                'Default: reads outputs/rl_training/latest_run.txt (the most recent run). '
                'Example: --run-id 20260323_143000_ppo'
            )
        )
        parser.add_argument(
            '--list-runs', action='store_true',
            help='List all available training runs with their key parameters and exit.'
        )
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Use only N algorithms (must match training)'
        )
        parser.add_argument(
            '--input-dir', type=str, default=None,
            help='Override processed dataset root used for evaluation universe'
        )
        parser.add_argument(
            '--seed', type=int, default=42,
            help='Random seed'
        )

    def _resolve_models_dir(self, args) -> Path:
        """
        Resolve the directory containing trained models, in priority order:
          1. --models-dir (explicit path, highest priority)
          2. --run-id (specific run under outputs/rl_training/)
          3. latest_run.txt (pointer written by Phase 5 after each training run)
          4. Error with helpful message listing available runs
        """
        import json as _json

        rl_root = self.op.rl_training.root

        # 1. Explicit override
        if args.models_dir:
            p = Path(args.models_dir)
            if not p.exists():
                raise FileNotFoundError(f"--models-dir not found: {p}")
            self.logger.info(f"Models dir (explicit): {p}")
            return p

        # 2. Specific run-id
        if args.run_id:
            p = rl_root / args.run_id / "checkpoints"
            if not p.exists():
                raise FileNotFoundError(
                    f"Run '{args.run_id}' not found at {p}. "
                    f"Run with --list-runs to see available runs."
                )
            self.logger.info(f"Models dir (run-id '{args.run_id}'): {p}")
            return p

        # 3. latest_run.txt
        latest_txt = rl_root / "latest_run.txt"
        if latest_txt.exists():
            run_id = latest_txt.read_text().strip()
            p = rl_root / run_id / "checkpoints"
            if p.exists():
                # Show run_info if available for traceability
                info_path = rl_root / run_id / "run_info.json"
                if info_path.exists():
                    info = _json.loads(info_path.read_text())
                    self.logger.info(
                        f"Evaluating latest run: '{run_id}'  "
                        f"(agents={info.get('agents')}, "
                        f"timesteps={info.get('total_timesteps')}, "
                        f"encoder={info.get('use_encoder')})"
                    )
                else:
                    self.logger.info(f"Evaluating latest run: '{run_id}' → {p}")
                return p
            self.logger.warning(
                f"latest_run.txt points to '{run_id}' but checkpoints not found at {p}"
            )

        # 4. Fallback: legacy flat structure (before run-id was introduced)
        legacy = rl_root / "checkpoints"
        if legacy.exists():
            self.logger.warning(
                f"No latest_run.txt found. Falling back to legacy path: {legacy}. "
                f"Re-run Phase 5 to generate a tracked run."
            )
            return legacy

        # Nothing found — list what exists and abort
        runs = sorted(
            [d.name for d in rl_root.iterdir() if d.is_dir() and d.name != "checkpoints"]
        ) if rl_root.exists() else []
        raise FileNotFoundError(
            f"No trained models found. "
            + (f"Available runs: {runs}" if runs else "Run Phase 5 first.")
            + " Use --run-id to select a specific run, or --list-runs to see all."
        )

    def _list_runs(self) -> None:
        """Print a table of all available training runs."""
        import json as _json

        rl_root = self.op.rl_training.root
        if not rl_root.exists():
            print("No training runs found. Run Phase 5 first.")
            return

        latest_txt = rl_root / "latest_run.txt"
        latest_id = latest_txt.read_text().strip() if latest_txt.exists() else None

        runs = sorted(
            [d for d in rl_root.iterdir() if d.is_dir() and (d / "run_info.json").exists()],
            key=lambda d: d.name,
        )

        if not runs:
            print("No runs with run_info.json found. Legacy runs may exist under checkpoints/.")
            return

        print(f"\n{'Run ID':<30} {'Agents':<15} {'Steps':>8} {'Encoder':>8} {'LR':>8}  {'Notes'}")
        print("-" * 90)
        for run_dir in runs:
            try:
                info = _json.loads((run_dir / "run_info.json").read_text())
            except Exception:
                continue
            marker = " ← latest" if run_dir.name == latest_id else ""
            agents = ",".join(info.get("agents", []))
            steps = info.get("total_timesteps", "?")
            enc = "yes" if info.get("use_encoder") else "no"
            lr = info.get("training", {}).get("learning_rate", "?")
            print(f"{run_dir.name:<30} {agents:<15} {steps:>8} {enc:>8} {lr:>8}{marker}")
        print()

    def run(self, args: argparse.Namespace) -> dict:
        """Execute Phase 6 pipeline."""
        # --list-runs: just print runs and exit (no evaluation)
        if hasattr(args, 'list_runs') and args.list_runs:
            self._list_runs()
            return {}

        # Quick mode overrides
        if args.quick:
            args.train_window = 126  # 6 months
            args.val_window = 21     # 1 month
            args.test_window = 21    # 1 month
            args.step_size = 21      # monthly
            args.folds = args.folds or 3
            args.sample = args.sample or 20  # Match Phase 5 quick mode

        # Build config
        config = EvalConfig(
            train_window=args.train_window,
            val_window=args.val_window,
            test_window=args.test_window,
            step_size=args.step_size,
            expanding=args.expanding,
            max_folds=args.folds,
            seed=args.seed,
        )

        results = {
            'mode': 'quick' if args.quick else 'standard',
            'config': asdict(config),
            'agents': {},
            'baselines': {},
            'comparison': {},
        }

        # Output dir and run_id are managed by PhaseRunner.execute() via _run_tag()
        models_dir = self._resolve_models_dir(args)
        output_dir = self.get_output_dir()
        self.logger.info(f"Evaluation run ID: {self._run_id or output_dir.name}")
        eval_start, eval_end = self._resolve_eval_bounds(models_dir)
        if eval_start is not None and eval_end is not None:
            self.logger.info(
                f"Restricting Phase 6 evaluation to Phase 5 holdout: {eval_start.date()} -> {eval_end.date()}"
            )

        # ==================================================================
        # STEP 6.1: LOAD DATA
        # ==================================================================
        input_dir = self._resolve_input_dir(args, models_dir)

        with self.step("6.1 Load Data"):
            algo_returns, benchmark_weights, benchmark_returns = self._load_data(input_dir)
            selected_algos_path = models_dir.parent / "cluster_selection" / "selected_algos.csv"
            if selected_algos_path.exists():
                selected_algos = pd.read_csv(selected_algos_path)["algo_id"].dropna().astype(str).tolist()
                selected_cols = [c for c in algo_returns.columns if c in selected_algos]
                algo_returns = algo_returns[selected_cols]
                if benchmark_weights is not None:
                    benchmark_weights = benchmark_weights.reindex(columns=selected_cols, fill_value=0.0)
                self.logger.info(
                    f"Applied Phase 5 selected universe from {selected_algos_path.name}: {len(selected_cols)} algos"
                )

            # Sample if requested - must match training configuration
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
                self.logger.info(f"Sampled {len(sampled_cols)} algorithms (matching training)")

            self.logger.info(f"Returns: {algo_returns.shape[0]} days x {algo_returns.shape[1]} algos")
            results['n_algos'] = algo_returns.shape[1]
            results['n_days'] = algo_returns.shape[0]

        # ==================================================================
        # STEP 6.2: GENERATE WALK-FORWARD FOLDS
        # ==================================================================
        with self.step("6.2 Generate Walk-Forward Folds"):
            from src.evaluation.walk_forward import WalkForwardValidator

            validator = WalkForwardValidator(
                train_window=config.train_window,
                val_window=config.val_window,
                test_window=config.test_window,
                step_size=config.step_size,
                expanding=config.expanding,
            )

            folds = validator.generate_folds(algo_returns.index)
            if eval_start is not None and eval_end is not None:
                folds = [
                    fold for fold in folds
                    if fold['test_start'] >= eval_start and fold['test_end'] <= eval_end
                ]
                self.logger.info(
                    f"Filtered folds to holdout window: {len(folds)} folds remain inside "
                    f"{eval_start.date()} -> {eval_end.date()}"
                )
                if not folds:
                    raise ValueError(
                        "No walk-forward folds remain after applying the Phase 5 holdout window. "
                        "Adjust the Phase 6 windows or rerun Phase 5 with a compatible holdout."
                    )

            if config.max_folds is not None:
                folds = folds[:config.max_folds]

            self.logger.info(f"Generated {len(folds)} walk-forward folds")
            for i, fold in enumerate(folds):
                self.logger.info(
                    f"  Fold {i}: train {fold['train_start'].date()} - {fold['train_end'].date()}, "
                    f"test {fold['test_start'].date()} - {fold['test_end'].date()}"
                )

            results['n_folds'] = len(folds)
            if eval_start is not None and eval_end is not None:
                results['evaluation_window'] = {
                    'start': str(eval_start.date()),
                    'end': str(eval_end.date()),
                    'source': 'phase5_test_split',
                }
            results['folds'] = [
                {k: str(v.date()) if hasattr(v, 'date') else v for k, v in fold.items()}
                for fold in folds
            ]

        # ==================================================================
        # STEP 6.3: EVALUATE RL AGENTS
        # ==================================================================
        agents_to_eval = ['ppo', 'sac', 'td3'] if args.agent == 'all' else [args.agent]

        for agent_name in agents_to_eval:
            model_path = models_dir / agent_name / "final_model"

            if not model_path.exists() and not (model_path.parent / "final_model.zip").exists():
                self.logger.warning(f"Model not found for {agent_name}: {model_path}")
                continue

            with self.step(f"6.3 Evaluate {agent_name.upper()} Agent"):
                agent_results = self._evaluate_agent(
                    agent_name=agent_name,
                    model_path=model_path,
                    folds=folds,
                    algo_returns=algo_returns,
                    benchmark_weights=benchmark_weights,
                    benchmark_returns=benchmark_returns,
                    config=config,
                    output_dir=output_dir,
                )
                results['agents'][agent_name] = agent_results

            gc.collect()

        # ==================================================================
        # STEP 6.4: EVALUATE BASELINES (optional)
        # ==================================================================
        if args.include_baselines:
            baselines = ['equal_weight', 'risk_parity', 'min_variance', 'max_sharpe', 'momentum', 'vol_targeting']

            for baseline_name in baselines:
                with self.step(f"6.4 Evaluate {baseline_name.replace('_', ' ').title()} Baseline"):
                    baseline_results = self._evaluate_baseline(
                        baseline_name=baseline_name,
                        folds=folds,
                        algo_returns=algo_returns,
                        benchmark_weights=benchmark_weights,
                        benchmark_returns=benchmark_returns,
                        config=config,
                        output_dir=output_dir,
                    )
                    results['baselines'][baseline_name] = baseline_results

                gc.collect()

        # ==================================================================
        # STEP 6.5: COMPARE ALL STRATEGIES
        # ==================================================================
        with self.step("6.5 Compare Strategies"):
            comparison = self._compare_strategies(results, benchmark_returns)
            results['comparison'] = comparison

            # Print summary table
            self._print_comparison_table(comparison)

        # ==================================================================
        # STEP 6.6: GENERATE REPORT
        # ==================================================================
        with self.step("6.6 Generate Report"):
            self._generate_report(results, output_dir)

        return results

    def _resolve_input_dir(self, args, models_dir: Path) -> Path:
        """Resolve evaluation dataset root."""
        if args.input_dir:
            return Path(args.input_dir)

        run_info_path = models_dir.parent / "run_info.json"
        if run_info_path.exists():
            try:
                run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
                input_dir = run_info.get("input_dir")
                if input_dir:
                    return Path(input_dir)
            except Exception:
                pass

        return self.dp.processed.root

    def _resolve_eval_bounds(self, models_dir: Path) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Read the Phase 5 test split and use it as the default out-of-sample evaluation window."""
        phase5_results_path = models_dir.parent / "phase5_results.json"
        if not phase5_results_path.exists():
            return None, None
        try:
            payload = json.loads(phase5_results_path.read_text(encoding="utf-8"))
            splits = payload.get("splits", {})
            test_split = splits.get("test")
            if isinstance(test_split, list) and len(test_split) == 2:
                return pd.Timestamp(test_split[0]), pd.Timestamp(test_split[1])
        except Exception:
            return None, None
        return None, None

    def _load_data(self, input_dir: Path):
        """Load Phase 1 data."""
        # Load returns matrix
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
                benchmark_returns = bench_df['return'].dropna()
            else:
                benchmark_returns = bench_df.iloc[:, 0].dropna()
            self.logger.info(f"Loaded benchmark returns: {len(benchmark_returns)} days")

        return algo_returns, benchmark_weights, benchmark_returns

    def _evaluate_agent(
        self,
        agent_name: str,
        model_path: Path,
        folds: List[dict],
        algo_returns: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        config: EvalConfig,
        output_dir: Path,
    ) -> dict:
        """Evaluate an RL agent using walk-forward."""
        from src.agents.ppo_agent import PPOAllocator
        from src.agents.sac_agent import SACAllocator
        from src.agents.td3_agent import TD3Allocator
        from src.environment.trading_env import TradingEnvironment, EpisodeConfig
        from src.environment.cost_model import CostModel
        from src.environment.constraints import PortfolioConstraints
        from src.environment.reward import RewardFunction, RewardType
        from src.evaluation.metrics import compute_full_metrics

        # Create components
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
            reward_type=RewardType.RISK_CALIBRATED_RETURNS,
        )

        # Load agent
        if agent_name == 'ppo':
            AgentClass = PPOAllocator
        elif agent_name == 'sac':
            AgentClass = SACAllocator
        elif agent_name == 'td3':
            AgentClass = TD3Allocator
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Check for .zip extension
        actual_path = model_path
        if not model_path.exists() and (model_path.parent / "final_model.zip").exists():
            actual_path = model_path.parent / "final_model.zip"

        # Load encoder if saved alongside the model (trained with AlgoUniverseEncoder)
        import pickle
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        encoder = None
        encoder_path = actual_path.parent / "universe_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
            self.logger.info(
                f"  Loaded encoder: obs_dim={encoder.obs_dim}, action_dim={encoder.action_dim}"
            )
        else:
            self.logger.info("  No encoder found — using raw observation/action space")

        # VecNormalize stats — MUST be reloaded for inference.
        # The model was trained on VecNormalize-normalized observations (clip_obs=10).
        # Evaluating without these stats feeds out-of-distribution inputs to the network.
        vec_norm_path = actual_path.parent / "vecnormalize.pkl"
        has_vec_norm = vec_norm_path.exists()
        if has_vec_norm:
            self.logger.info("  Found vecnormalize.pkl — will apply obs normalization at inference")
        else:
            self.logger.warning(
                "  vecnormalize.pkl not found — obs will NOT be normalized. "
                "Results may be poor if model was trained with VecNormalize."
            )

        fold_results = []
        all_test_returns = []
        allocation_rows = []

        for fold in folds:
            self.logger.info(f"  Processing fold {fold['fold_id']}...")

            # Create test environment
            episode_config = EpisodeConfig(
                random_start=False,
                episode_length=None,  # Run full period
            )

            test_env = TradingEnvironment(
                algo_returns=algo_returns,
                benchmark_weights=benchmark_weights,
                train_start=fold['test_start'],
                train_end=fold['test_end'],
                initial_capital=config.initial_capital,
                rebalance_frequency=config.rebalance_frequency,
                cost_model=cost_model,
                constraints=constraints,
                reward_function=reward_fn,
                episode_config=episode_config,
                reward_scale=config.reward_scale,
                encoder=encoder,
            )

            # Wrap in VecEnv (required for VecNormalize and SB3 agent interface)
            test_vec_env = DummyVecEnv([lambda: test_env])

            # Restore VecNormalize obs statistics from training — critical for correct inference.
            # training=False: stats are frozen (no updates during eval).
            # norm_reward=False: we want raw rewards for metric computation.
            if has_vec_norm:
                test_vec_env = VecNormalize.load(str(vec_norm_path), test_vec_env)
                test_vec_env.training = False
                test_vec_env.norm_reward = False

            # Load and evaluate agent (spaces must match — VecNormalize keeps same Box shape)
            agent = AgentClass.from_pretrained(str(actual_path), env=test_vec_env)

            # Run episode using VecEnv interface so observations pass through VecNormalize
            obs = test_vec_env.reset()
            done = False
            episode_returns = []
            episode_dates = []
            episode_weights = []
            total_reward = 0

            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, rewards, dones, infos = test_vec_env.step(action)

                info = infos[0] if infos else {}
                total_reward += float(rewards[0])
                if 'portfolio_return' in info:
                    episode_returns.append(info['portfolio_return'])
                if 'date' in info:
                    episode_dates.append(info['date'])
                if 'weights' in info:
                    episode_weights.append(np.asarray(info['weights'], dtype=np.float32))

                done = bool(dones[0])

            # Compute metrics for this fold
            if episode_returns:
                returns_series = pd.Series(episode_returns, index=episode_dates[:len(episode_returns)])

                # Get benchmark returns for this period
                bench_fold = None
                if benchmark_returns is not None:
                    bench_fold = benchmark_returns.loc[fold['test_start']:fold['test_end']]

                metrics = compute_full_metrics(returns_series, bench_fold)

                fold_results.append({
                    'fold_id': fold['fold_id'],
                    'test_start': str(fold['test_start'].date()),
                    'test_end': str(fold['test_end'].date()),
                    'total_reward': total_reward,
                    'n_steps': len(episode_returns),
                    **metrics,
                })

                all_test_returns.extend(episode_returns)
                snapshot_dates, snapshot_weights = compress_daily_weight_history(
                    episode_dates,
                    episode_weights,
                )
                allocation_rows.extend(
                    build_periodic_allocation_rows(
                        snapshot_dates,
                        snapshot_weights,
                        algo_returns.columns.tolist(),
                        final_period_end=fold['test_end'],
                        metadata={
                            'strategy': agent_name,
                            'strategy_type': 'rl_agent',
                            'fold_id': int(fold['fold_id']),
                            'split': 'test',
                        },
                    )
                )

        # Aggregate metrics across folds
        aggregated = self._aggregate_fold_metrics(fold_results)

        # Save fold results
        fold_df = pd.DataFrame(fold_results)
        fold_path = output_dir / "walk_forward" / f"{agent_name}_folds.csv"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(fold_path, index=False)
        if allocation_rows:
            allocations_path = output_dir / "walk_forward" / f"{agent_name}_allocations.csv"
            pd.DataFrame(allocation_rows).to_csv(allocations_path, index=False)

        self.logger.info(f"  Aggregated Sharpe: {aggregated.get('sharpe_ratio_mean', 0):.3f}")
        self.logger.info(f"  Aggregated Return: {aggregated.get('annualized_return_mean', 0):.2%}")
        self.logger.info(f"  Max Drawdown: {aggregated.get('max_drawdown_mean', 0):.2%}")

        return {
            'model_path': str(actual_path),
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'aggregated': aggregated,
            'allocations_path': str(output_dir / "walk_forward" / f"{agent_name}_allocations.csv") if allocation_rows else None,
        }

    def _evaluate_baseline(
        self,
        baseline_name: str,
        folds: List[dict],
        algo_returns: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        config: EvalConfig,
        output_dir: Path,
    ) -> dict:
        """Evaluate a classical baseline using walk-forward."""
        from src.baselines.equal_weight import EqualWeightAllocator
        from src.baselines.risk_parity import RiskParityAllocator
        from src.baselines.min_variance import MinVarianceAllocator
        from src.baselines.max_sharpe import MaxSharpeAllocator
        from src.baselines.momentum_allocator import MomentumAllocator
        from src.baselines.vol_targeting import VolTargetingAllocator
        from src.evaluation.metrics import compute_full_metrics

        # Select baseline
        baseline_map = {
            'equal_weight': EqualWeightAllocator,
            'risk_parity': RiskParityAllocator,
            'min_variance': MinVarianceAllocator,
            'max_sharpe': MaxSharpeAllocator,
            'momentum': MomentumAllocator,
            'vol_targeting': VolTargetingAllocator,
        }

        AllocatorClass = baseline_map.get(baseline_name)
        if AllocatorClass is None:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        fold_results = []
        allocation_rows = []

        for fold in folds:
            self.logger.info(f"  Processing fold {fold['fold_id']}...")

            # Get data for this fold
            train_data = algo_returns.loc[fold['train_start']:fold['train_end']]
            test_data = algo_returns.loc[fold['test_start']:fold['test_end']]

            # Create and fit allocator
            allocator = AllocatorClass(
                max_weight=config.max_weight,
                min_weight=config.min_weight,
            )

            # Compute weights using training data
            n_algos = train_data.shape[1]
            current_weights = np.zeros(n_algos)  # Start with no position
            calc_date = fold['train_end']
            weights = allocator.compute_weights(calc_date, train_data, current_weights)

            # Apply to test period
            test_returns = (test_data * weights).sum(axis=1)

            # Get benchmark returns for this period
            bench_fold = None
            if benchmark_returns is not None:
                bench_fold = benchmark_returns.loc[fold['test_start']:fold['test_end']]

            metrics = compute_full_metrics(test_returns, bench_fold)

            fold_results.append({
                'fold_id': fold['fold_id'],
                'test_start': str(fold['test_start'].date()),
                'test_end': str(fold['test_end'].date()),
                **metrics,
            })
            allocation_rows.extend(
                build_periodic_allocation_rows(
                    [fold['test_start']],
                    [weights],
                    algo_returns.columns.tolist(),
                    final_period_end=fold['test_end'],
                    metadata={
                        'strategy': baseline_name,
                        'strategy_type': 'baseline',
                        'fold_id': int(fold['fold_id']),
                        'split': 'test',
                    },
                )
            )

        # Aggregate metrics across folds
        aggregated = self._aggregate_fold_metrics(fold_results)

        if allocation_rows:
            allocations_dir = output_dir / "walk_forward"
            allocations_dir.mkdir(parents=True, exist_ok=True)
            allocations_path = allocations_dir / f"{baseline_name}_allocations.csv"
            pd.DataFrame(allocation_rows).to_csv(allocations_path, index=False)

        self.logger.info(f"  Aggregated Sharpe: {aggregated.get('sharpe_ratio_mean', 0):.3f}")
        self.logger.info(f"  Aggregated Return: {aggregated.get('annualized_return_mean', 0):.2%}")

        return {
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'aggregated': aggregated,
            'allocations_path': str(output_dir / "walk_forward" / f"{baseline_name}_allocations.csv") if allocation_rows else None,
        }

    def _aggregate_fold_metrics(self, fold_results: List[dict]) -> dict:
        """Aggregate metrics across folds."""
        if not fold_results:
            return {}

        aggregated = {}

        # Get all metric keys (excluding non-numeric)
        exclude_keys = {'fold_id', 'test_start', 'test_end', 'model_path'}
        metric_keys = [k for k in fold_results[0].keys() if k not in exclude_keys]

        for key in metric_keys:
            values = [f.get(key) for f in fold_results if f.get(key) is not None]
            values = [v for v in values if not np.isnan(v)]

            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)

        return aggregated

    def _compare_strategies(self, results: dict, benchmark_returns: Optional[pd.Series]) -> dict:
        """Compare all strategies."""
        comparison = {
            'strategies': [],
            'ranking': {},
        }

        # Add benchmark
        if benchmark_returns is not None:
            from src.evaluation.metrics import compute_full_metrics
            bench_metrics = compute_full_metrics(benchmark_returns)
            comparison['strategies'].append({
                'name': 'Benchmark',
                'type': 'benchmark',
                'sharpe_ratio': bench_metrics.get('sharpe_ratio', 0),
                'annualized_return': bench_metrics.get('annualized_return', 0),
                'max_drawdown': bench_metrics.get('max_drawdown', 0),
                'volatility': bench_metrics.get('annualized_volatility', 0),
            })

        # Add RL agents
        for agent_name, agent_results in results.get('agents', {}).items():
            agg = agent_results.get('aggregated', {})
            comparison['strategies'].append({
                'name': agent_name.upper(),
                'type': 'rl_agent',
                'sharpe_ratio': agg.get('sharpe_ratio_mean', 0),
                'annualized_return': agg.get('annualized_return_mean', 0),
                'max_drawdown': agg.get('max_drawdown_mean', 0),
                'volatility': agg.get('annualized_volatility_mean', 0),
                'information_ratio': agg.get('information_ratio_mean', 0),
            })

        # Add baselines
        for baseline_name, baseline_results in results.get('baselines', {}).items():
            agg = baseline_results.get('aggregated', {})
            comparison['strategies'].append({
                'name': baseline_name.replace('_', ' ').title(),
                'type': 'baseline',
                'sharpe_ratio': agg.get('sharpe_ratio_mean', 0),
                'annualized_return': agg.get('annualized_return_mean', 0),
                'max_drawdown': agg.get('max_drawdown_mean', 0),
                'volatility': agg.get('annualized_volatility_mean', 0),
            })

        # Rank by Sharpe ratio
        strategies = comparison['strategies']
        ranked = sorted(strategies, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        comparison['ranking']['by_sharpe'] = [s['name'] for s in ranked]

        # Rank by return
        ranked = sorted(strategies, key=lambda x: x.get('annualized_return', 0), reverse=True)
        comparison['ranking']['by_return'] = [s['name'] for s in ranked]

        return comparison

    def _print_comparison_table(self, comparison: dict):
        """Print comparison table to console."""
        strategies = comparison.get('strategies', [])
        if not strategies:
            return

        self.logger.info("\n" + "=" * 80)
        self.logger.info("STRATEGY COMPARISON")
        self.logger.info("=" * 80)
        self.logger.info(f"{'Strategy':<20} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'Vol':>10} {'IR':>10}")
        self.logger.info("-" * 80)

        for s in strategies:
            ir = s.get('information_ratio', 0) or 0
            self.logger.info(
                f"{s['name']:<20} "
                f"{s.get('sharpe_ratio', 0):>10.3f} "
                f"{s.get('annualized_return', 0):>11.2%} "
                f"{s.get('max_drawdown', 0):>10.2%} "
                f"{s.get('volatility', 0):>10.2%} "
                f"{ir:>10.3f}"
            )

        self.logger.info("=" * 80)

        # Print ranking
        ranking = comparison.get('ranking', {})
        if ranking.get('by_sharpe'):
            self.logger.info(f"\nRanking by Sharpe: {' > '.join(ranking['by_sharpe'])}")

    def _generate_report(self, results: dict, output_dir: Path):
        """Generate markdown report."""
        config = results['config']

        report = f"""# Phase 6: Walk-Forward Evaluation Report

**Generated**: {datetime.now().isoformat()}
**Mode**: {results['mode']}

## Configuration

| Parameter | Value |
|-----------|-------|
| Train Window | {config['train_window']} days |
| Validation Window | {config['val_window']} days |
| Test Window | {config['test_window']} days |
| Step Size | {config['step_size']} days |
| Window Type | {'Expanding' if config['expanding'] else 'Rolling'} |
| Number of Folds | {results['n_folds']} |

## Data Summary

| Metric | Value |
|--------|-------|
| Algorithms | {results['n_algos']} |
| Trading Days | {results['n_days']} |

## Walk-Forward Folds

| Fold | Train Period | Test Period |
|------|--------------|-------------|
"""
        for fold in results.get('folds', []):
            report += f"| {fold['fold_id']} | {fold['train_start']} - {fold['train_end']} | {fold['test_start']} - {fold['test_end']} |\n"

        report += "\n## RL Agents\n\n"

        for agent_name, agent_results in results.get('agents', {}).items():
            agg = agent_results.get('aggregated', {})
            report += f"""### {agent_name.upper()}

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Sharpe Ratio | {agg.get('sharpe_ratio_mean', 0):.3f} | {agg.get('sharpe_ratio_std', 0):.3f} | {agg.get('sharpe_ratio_min', 0):.3f} | {agg.get('sharpe_ratio_max', 0):.3f} |
| Ann. Return | {agg.get('annualized_return_mean', 0):.2%} | {agg.get('annualized_return_std', 0):.2%} | {agg.get('annualized_return_min', 0):.2%} | {agg.get('annualized_return_max', 0):.2%} |
| Max Drawdown | {agg.get('max_drawdown_mean', 0):.2%} | {agg.get('max_drawdown_std', 0):.2%} | {agg.get('max_drawdown_min', 0):.2%} | {agg.get('max_drawdown_max', 0):.2%} |
| Info Ratio | {agg.get('information_ratio_mean', 0):.3f} | {agg.get('information_ratio_std', 0):.3f} | {agg.get('information_ratio_min', 0):.3f} | {agg.get('information_ratio_max', 0):.3f} |

"""

        if results.get('baselines'):
            report += "\n## Classical Baselines\n\n"

            for baseline_name, baseline_results in results.get('baselines', {}).items():
                agg = baseline_results.get('aggregated', {})
                report += f"""### {baseline_name.replace('_', ' ').title()}

| Metric | Mean | Std |
|--------|------|-----|
| Sharpe Ratio | {agg.get('sharpe_ratio_mean', 0):.3f} | {agg.get('sharpe_ratio_std', 0):.3f} |
| Ann. Return | {agg.get('annualized_return_mean', 0):.2%} | {agg.get('annualized_return_std', 0):.2%} |
| Max Drawdown | {agg.get('max_drawdown_mean', 0):.2%} | {agg.get('max_drawdown_std', 0):.2%} |

"""

        # Comparison table
        comparison = results.get('comparison', {})
        strategies = comparison.get('strategies', [])

        if strategies:
            report += """## Strategy Comparison

| Strategy | Type | Sharpe | Return | Max DD | Vol |
|----------|------|--------|--------|--------|-----|
"""
            for s in strategies:
                report += f"| {s['name']} | {s['type']} | {s.get('sharpe_ratio', 0):.3f} | {s.get('annualized_return', 0):.2%} | {s.get('max_drawdown', 0):.2%} | {s.get('volatility', 0):.2%} |\n"

        ranking = comparison.get('ranking', {})
        if ranking.get('by_sharpe'):
            report += f"\n**Ranking by Sharpe Ratio**: {' > '.join(ranking['by_sharpe'])}\n"

        report += """
## Files Generated

- `walk_forward/*_folds.csv` - Per-fold metrics for each agent
- `PHASE6_SUMMARY.md` - This report
- `results.json` - Full results in JSON format
"""

        # Save report
        report_path = output_dir / "PHASE6_SUMMARY.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        self.logger.info(f"Report saved to: {report_path}")

        # Save JSON results
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    runner = Phase6Runner()
    runner.execute()


if __name__ == "__main__":
    main()
