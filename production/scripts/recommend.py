#!/usr/bin/env python
"""
Generate Portfolio Recommendation
=================================
Generates weekly portfolio weight recommendations using trained RL agent.

Usage:
    python production/scripts/recommend.py
    python production/scripts/recommend.py --model-dir models/latest --hybrid
"""

import argparse
import json
import logging
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

from production.src.inference import (
    ProductionInferenceEngine,
    HybridInferenceEngine,
    save_recommendation,
    load_latest_recommendation,
)

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def update_returns_with_latest(
    returns_path: Path,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Update returns with any new data from raw CSVs.

    This allows incremental updates without full reprocessing.
    """
    # Load existing returns
    returns = pd.read_parquet(returns_path)
    last_date = returns.index.max()

    logger.info(f"Existing data ends at {last_date.date()}")

    # Check for new data in raw files
    # (This is a simplified version - could be extended to handle new CSVs)

    return returns


def format_recommendation_table(recommendation: dict) -> str:
    """Format recommendation as ASCII table for display."""
    weights = recommendation["weights"]
    summary = recommendation["summary"]
    n_assets = summary.get("n_assets", len(weights))

    # Sort by weight descending
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    lines = []
    lines.append("=" * 60)
    lines.append(f"PORTFOLIO RECOMMENDATION - {recommendation['date']}")
    lines.append("=" * 60)
    lines.append(f"Generated: {recommendation['generated_at']}")
    lines.append(f"Agent: {recommendation['agent_type']}")
    if "base_allocator" in recommendation:
        lines.append(f"Base: {recommendation['base_allocator']} + RL tilts")
    lines.append("-" * 60)

    # For large universes, show top/bottom positions
    equal_weight_ref = summary.get("equal_weight_ref", 1.0 / n_assets)
    show_threshold = equal_weight_ref * 1.5  # Show positions with >1.5x equal weight

    lines.append(f"{'Asset':<20} {'Weight':>10} {'Pct':>8} {'vs Equal':>12}")
    lines.append("-" * 60)

    # Show top 15 overweight positions
    shown = 0
    for asset, weight in sorted_weights:
        if weight >= show_threshold and shown < 15:
            pct = weight * 100
            multiple = weight / equal_weight_ref
            lines.append(f"{asset:<20} {weight:>10.6f} {pct:>7.4f}%  {multiple:>10.2f}x")
            shown += 1

    if shown == 0:
        lines.append("(All positions near equal weight)")

    lines.append("-" * 60)

    # Summary statistics
    lines.append(f"Universe: {n_assets:,} assets")
    lines.append(f"Equal weight ref: {equal_weight_ref*100:.4f}%")
    lines.append(f"Max weight: {summary['max_weight']*100:.4f}% ({summary['max_weight']/equal_weight_ref:.2f}x equal)")

    if "n_overweight" in summary:
        lines.append(f"Overweight (>1.5x): {summary['n_overweight']:,} positions")
    if "n_underweight" in summary:
        lines.append(f"Underweight (<0.5x): {summary['n_underweight']:,} positions")

    lines.append(f"HHI: {summary['hhi']:.6f}")

    if "base_weights" in recommendation:
        lines.append(f"Max tilt: {summary.get('max_tilt_applied', 0)*100:.4f}%")

    lines.append("=" * 60)

    return "\n".join(lines)


def compare_with_previous(
    current: dict,
    previous: dict,
) -> str:
    """Compare current recommendation with previous one."""
    if previous is None:
        return "No previous recommendation to compare."

    current_weights = current["weights"]
    previous_weights = previous["weights"]

    lines = []
    lines.append("\nCHANGES FROM PREVIOUS RECOMMENDATION")
    lines.append("-" * 50)

    all_assets = set(current_weights.keys()) | set(previous_weights.keys())

    changes = []
    for asset in all_assets:
        curr = current_weights.get(asset, 0)
        prev = previous_weights.get(asset, 0)
        diff = curr - prev

        if abs(diff) >= 0.01:  # Only show changes >= 1%
            changes.append((asset, prev, curr, diff))

    if not changes:
        lines.append("No significant changes (all < 1%)")
    else:
        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        lines.append(f"{'Asset':<25} {'Previous':>10} {'Current':>10} {'Change':>10}")
        lines.append("-" * 55)

        for asset, prev, curr, diff in changes[:10]:  # Top 10 changes
            sign = "+" if diff > 0 else ""
            lines.append(f"{asset:<25} {prev:>10.2%} {curr:>10.2%} {sign}{diff:>9.2%}")

    # Calculate turnover
    turnover = sum(
        abs(current_weights.get(a, 0) - previous_weights.get(a, 0))
        for a in all_assets
    ) / 2

    lines.append("-" * 55)
    lines.append(f"Total turnover: {turnover:.2%}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate portfolio recommendation")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory (default: models/latest)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="Agent type",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid mode (base allocator + RL tilts)",
    )
    parser.add_argument(
        "--base-allocator",
        type=str,
        default="risk_parity",
        choices=["risk_parity", "equal_weight"],
        help="Base allocator for hybrid mode",
    )
    parser.add_argument(
        "--max-tilt",
        type=float,
        default=0.15,
        help="Maximum tilt from base allocation",
    )
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
        default=PRODUCTION_DIR / "outputs" / "recommendations",
        help="Directory to save recommendations",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save recommendation to file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of table",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve model directory
    if args.model_dir is None:
        latest_link = PRODUCTION_DIR / "models" / "latest"
        if latest_link.exists():
            args.model_dir = latest_link.resolve()
        else:
            # Find most recent model directory
            model_dirs = sorted(
                (PRODUCTION_DIR / "models").glob("2*_*"),
                reverse=True,
            )
            if not model_dirs:
                logger.error("No trained models found in production/models/")
                logger.info("Run 'python production/scripts/train.py' first")
                sys.exit(1)
            args.model_dir = model_dirs[0]

    logger.info(f"Using model from: {args.model_dir}")

    # Load config
    config = load_config(args.config)

    # Load returns
    returns_path = args.data_dir / "returns.parquet"
    if not returns_path.exists():
        logger.error(f"Returns not found at {returns_path}")
        logger.info("Run 'python production/scripts/prepare_data.py' first")
        sys.exit(1)

    returns = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {returns.shape[0]} days x {returns.shape[1]} assets")
    logger.info(f"Data range: {returns.index.min().date()} to {returns.index.max().date()}")

    # Create inference engine
    try:
        if args.hybrid:
            logger.info(f"Using hybrid mode: {args.base_allocator} + {args.agent} tilts")
            engine = HybridInferenceEngine(
                model_dir=args.model_dir,
                agent_type=args.agent,
                config=config,
                base_allocator=args.base_allocator,
                max_tilt=args.max_tilt,
            )
        else:
            engine = ProductionInferenceEngine(
                model_dir=args.model_dir,
                agent_type=args.agent,
                config=config,
            )
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        sys.exit(1)

    # Generate recommendation
    logger.info("Generating recommendation...")
    recommendation = engine.get_recommendation(returns, deterministic=True)

    # Output
    if args.json:
        print(json.dumps(recommendation, indent=2))
    else:
        print(format_recommendation_table(recommendation))

        # Compare with previous
        previous = load_latest_recommendation(args.output_dir)
        if previous and previous["date"] != recommendation["date"]:
            print(compare_with_previous(recommendation, previous))

    # Save
    if not args.no_save:
        output_path = save_recommendation(recommendation, args.output_dir)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
