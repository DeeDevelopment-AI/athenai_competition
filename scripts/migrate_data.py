#!/usr/bin/env python
"""
Migrate data files from flat structure to organized hierarchy.

This script moves files from the old flat data/processed/ layout to the new
hierarchical structure defined in src/utils/paths.py.

Usage:
    python scripts/migrate_data.py              # Dry-run (show what would be moved)
    python scripts/migrate_data.py --execute    # Actually move files
    python scripts/migrate_data.py --cleanup    # Remove empty directories after migration
"""

import argparse
import shutil
import sys
from pathlib import Path


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import (
    PROJECT_ROOT,
    data_paths,
    output_paths,
    ensure_dir,
)


# Migration mapping: old_name -> new_path
def get_migration_map():
    """Returns mapping of old file names to new paths."""
    dp = data_paths()

    return {
        # Algorithm data
        "algo_returns.parquet": dp.algorithms.returns,
        "algo_features.parquet": dp.algorithms.features,
        "algo_stats.csv": dp.algorithms.stats,
        "asset_inference.csv": dp.algorithms.asset_inference,

        # Benchmark data
        "benchmark_weights.parquet": dp.benchmark.weights,
        "benchmark_positions.parquet": dp.benchmark.positions,
        "benchmark_daily_returns.csv": dp.benchmark.daily_returns,
        "benchmark_turnover.csv": dp.benchmark.turnover,
        "benchmark_concentration.csv": dp.benchmark.concentration,
        "benchmark_algo_equity.parquet": dp.benchmark.algo_equity,
        "benchmark_algo_features.parquet": dp.benchmark.algo_features,

        # Reports
        "PHASE1_SUMMARY.md": dp.processed.reports.phase1_summary,
        "PHASE1_DATA_LOADING_FINDINGS.md": dp.processed.reports.root / "phase1" / "DATA_LOADING_FINDINGS.md",
        "phase1_results.json": dp.processed.reports.phase1_results,
    }


# Directory migration mapping
def get_directory_migration_map():
    """Returns mapping of old directories to new paths."""
    dp = data_paths()

    return {
        "temporal_clusters": dp.processed.analysis.root / "clustering" / "temporal",
        "phase2": dp.processed.analysis.root,
    }


def migrate_file(old_path: Path, new_path: Path, execute: bool = False) -> bool:
    """
    Migrate a single file.

    Returns True if migration was successful or would be successful (dry-run).
    """
    if not old_path.exists():
        return False

    if new_path.exists():
        print(f"  SKIP (exists): {old_path.name} -> {new_path.relative_to(PROJECT_ROOT)}")
        return False

    if execute:
        ensure_dir(new_path.parent)
        shutil.move(str(old_path), str(new_path))
        print(f"  MOVED: {old_path.name} -> {new_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  WOULD MOVE: {old_path.name} -> {new_path.relative_to(PROJECT_ROOT)}")

    return True


def migrate_directory(old_path: Path, new_path: Path, execute: bool = False) -> bool:
    """
    Migrate a directory and all its contents.

    Returns True if migration was successful.
    """
    if not old_path.exists():
        return False

    if new_path.exists():
        print(f"  SKIP (exists): {old_path.name}/ -> {new_path.relative_to(PROJECT_ROOT)}/")
        return False

    if execute:
        ensure_dir(new_path.parent)
        shutil.move(str(old_path), str(new_path))
        print(f"  MOVED DIR: {old_path.name}/ -> {new_path.relative_to(PROJECT_ROOT)}/")
    else:
        # Count files in directory
        file_count = sum(1 for _ in old_path.rglob("*") if _.is_file())
        print(f"  WOULD MOVE DIR: {old_path.name}/ ({file_count} files) -> {new_path.relative_to(PROJECT_ROOT)}/")

    return True


def cleanup_empty_dirs(root: Path, execute: bool = False):
    """Remove empty directories after migration."""
    empty_dirs = []

    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            empty_dirs.append(path)

    if empty_dirs:
        print(f"\nEmpty directories found: {len(empty_dirs)}")
        for d in empty_dirs:
            if execute:
                d.rmdir()
                print(f"  REMOVED: {d.relative_to(PROJECT_ROOT)}/")
            else:
                print(f"  WOULD REMOVE: {d.relative_to(PROJECT_ROOT)}/")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate data files to new organized structure"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry-run)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove empty directories after migration"
    )

    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed"

    if not processed_dir.exists():
        print(f"Processed directory not found: {processed_dir}")
        return 1

    print("=" * 60)
    if args.execute:
        print("DATA MIGRATION - EXECUTING")
    else:
        print("DATA MIGRATION - DRY RUN (use --execute to actually move files)")
    print("=" * 60)

    # Migrate files
    print("\n[1] Migrating files...")
    file_map = get_migration_map()
    files_migrated = 0

    for old_name, new_path in file_map.items():
        old_path = processed_dir / old_name
        if migrate_file(old_path, new_path, execute=args.execute):
            files_migrated += 1

    if files_migrated == 0:
        print("  No files to migrate (all already in new location or not found)")

    # Migrate directories
    print("\n[2] Migrating directories...")
    dir_map = get_directory_migration_map()
    dirs_migrated = 0

    for old_name, new_path in dir_map.items():
        old_path = processed_dir / old_name
        if migrate_directory(old_path, new_path, execute=args.execute):
            dirs_migrated += 1

    if dirs_migrated == 0:
        print("  No directories to migrate")

    # Cleanup
    if args.cleanup:
        print("\n[3] Cleaning up empty directories...")
        cleanup_empty_dirs(processed_dir, execute=args.execute)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # List what's in each target directory
    dp = data_paths()

    target_dirs = [
        ("algorithms/", dp.processed.algorithms.root),
        ("benchmark/", dp.processed.benchmark.root),
        ("analysis/", dp.processed.analysis.root),
        ("features/", dp.processed.features.root),
        ("reports/", dp.processed.reports.root),
    ]

    print("\nTarget directory contents:")
    for name, path in target_dirs:
        if path.exists():
            files = list(path.rglob("*"))
            file_count = sum(1 for f in files if f.is_file())
            print(f"  {name}: {file_count} files")
        else:
            print(f"  {name}: (not created yet)")

    # List remaining files in flat structure
    print("\nRemaining in data/processed/ (flat):")
    remaining = [
        f for f in processed_dir.iterdir()
        if f.is_file() and f.name not in [".gitkeep", ".DS_Store"]
    ]

    if remaining:
        for f in remaining:
            print(f"  {f.name}")
    else:
        print("  (none)")

    if not args.execute:
        print("\n>>> Run with --execute to actually move files <<<")

    return 0


if __name__ == "__main__":
    sys.exit(main())
