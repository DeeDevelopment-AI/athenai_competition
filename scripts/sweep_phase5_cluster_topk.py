#!/usr/bin/env python3
"""
Sweep Phase 5 training runs across multiple top-k Phase 2 cluster selections.

Typical usage:
  python scripts/sweep_phase5_cluster_topk.py ^
    --agent ppo ^
    --phase2-analysis-dir data/processed/analysis ^
    --k-values 1 2 3 4 ^
    --reward-type pure_returns ^
    --no-hybrid ^
    --gpu-env ^
    --max-resources ^
    --evaluate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a sweep of Phase 5 runs varying top-k Phase 2 clusters."
    )
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "sac", "td3"])
    parser.add_argument("--timesteps", type=str, default="1M")
    parser.add_argument("--phase2-analysis-dir", type=str, required=True)
    parser.add_argument(
        "--phase2-cluster-source",
        type=str,
        default="temporal_cumulative",
        choices=["behavioral_family", "temporal_cumulative", "temporal_weekly", "temporal_monthly"],
    )
    parser.add_argument(
        "--phase2-cluster-score-mode",
        type=str,
        default="return_low_vol",
        choices=["return_low_vol", "return", "sharpe", "sortino"],
    )
    parser.add_argument("--k-values", type=int, nargs="+", required=True, help="List of top-k values to test")
    parser.add_argument("--phase2-cluster-min-size", type=int, default=20)
    parser.add_argument("--phase2-cluster-min-return", type=float, default=0.01)
    parser.add_argument("--phase2-cluster-max-vol", type=float, default=0.12)
    parser.add_argument("--reward-type", type=str, default="pure_returns")
    parser.add_argument("--rebalance-freq", type=str, default="weekly", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--run-prefix", type=str, default=None, help="Optional prefix for generated run ids")
    parser.add_argument("--gpu-env", action="store_true")
    parser.add_argument("--max-resources", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-encoder", action="store_true")
    parser.add_argument("--phase2-cluster-full-history", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run Phase 6 after each Phase 5 run")
    parser.add_argument("--include-baselines", action="store_true", help="Use with --evaluate")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("\n" + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = build_parser().parse_args()
    python_exe = sys.executable
    phase5_script = str(PROJECT_ROOT / "scripts" / "run_phase5.py")
    phase6_script = str(PROJECT_ROOT / "scripts" / "run_phase6.py")

    for k in args.k_values:
        run_prefix = args.run_prefix or f"{args.agent}_{args.phase2_cluster_source}_{args.phase2_cluster_score_mode}"
        run_id = f"{run_prefix}_top{k}"

        phase5_cmd = [
            python_exe,
            phase5_script,
            "--agent", args.agent,
            "--timesteps", args.timesteps,
            "--reward-type", args.reward_type,
            "--rebalance-freq", args.rebalance_freq,
            "--seed", str(args.seed),
            "--run-id", run_id,
            "--phase2-cluster-filter",
            "--phase2-analysis-dir", args.phase2_analysis_dir,
            "--phase2-cluster-source", args.phase2_cluster_source,
            "--phase2-cluster-score-mode", args.phase2_cluster_score_mode,
            "--phase2-cluster-top-k", str(k),
            "--phase2-cluster-min-size", str(args.phase2_cluster_min_size),
            "--phase2-cluster-min-return", str(args.phase2_cluster_min_return),
        ]

        if args.phase2_cluster_max_vol is not None:
            phase5_cmd.extend(["--phase2-cluster-max-vol", str(args.phase2_cluster_max_vol)])
        if args.sample is not None:
            phase5_cmd.extend(["--sample", str(args.sample)])
        if args.gpu_env:
            phase5_cmd.append("--gpu-env")
        if args.max_resources:
            phase5_cmd.append("--max-resources")
        if args.no_hybrid:
            phase5_cmd.append("--no-hybrid")
        if args.no_encoder:
            phase5_cmd.append("--no-encoder")
        if args.phase2_cluster_full_history:
            phase5_cmd.append("--phase2-cluster-full-history")
        if args.quick:
            phase5_cmd.append("--quick")
        if args.verbose:
            phase5_cmd.append("--verbose")

        run_command(phase5_cmd, dry_run=args.dry_run)

        if args.evaluate:
            phase6_cmd = [
                python_exe,
                phase6_script,
                "--run-id", run_id,
            ]
            if args.include_baselines:
                phase6_cmd.append("--include-baselines")
            if args.verbose:
                phase6_cmd.append("--verbose")
            run_command(phase6_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
