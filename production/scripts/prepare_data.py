#!/usr/bin/env python
"""
Prepare Production Data
=======================
Loads data from DuckDB or OHLCV CSVs, prepares for training.

Usage:
    # From DuckDB (recommended for large datasets)
    python production/scripts/prepare_data.py --duckdb /path/to/data.duckdb

    # From CSV files
    python production/scripts/prepare_data.py --raw-dir /path/to/csvs
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup paths - use resolve() to get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PRODUCTION_DIR.parent

# Add to path if not already present
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from production.src (not src, to avoid conflict with project root's src)
from production.src.data_loader import prepare_production_data
from production.src.feature_builder import build_features, save_features, print_family_summary


def prepare_from_duckdb(args, logger):
    """Prepare data from DuckDB database."""
    from production.src.duckdb_loader import DuckDBLoader

    logger.info(f"Loading from DuckDB: {args.duckdb}")

    with DuckDBLoader(args.duckdb) as loader:
        # Print database info
        info = loader.get_info()
        print(f"\nDatabase info:")
        print(f"  Rows: {info['n_rows']:,}")
        print(f"  Symbols: {info['n_symbols']:,}")
        print(f"  Dates: {info['n_dates']:,}")
        print(f"  Range: {info['min_date']} to {info['max_date']}")

        # Prepare training data
        returns, features = loader.prepare_training_data(
            output_dir=args.output_dir,
            min_observations=args.min_history,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    return returns, features


def prepare_from_csv(args, logger):
    """Prepare data from CSV files."""
    # Validate input directory
    if not args.raw_dir.exists():
        logger.error(f"Raw data directory not found: {args.raw_dir}")
        logger.info(f"Please place your OHLCV CSV files in: {args.raw_dir}")
        sys.exit(1)

    csv_files = list(args.raw_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in: {args.raw_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} CSV files")

    # Load and process
    logger.info("Loading and processing OHLCV data...")
    returns, asset_names = prepare_production_data(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        min_history_days=args.min_history,
        min_coverage=args.min_coverage,
    )

    # Build features (including Sortino families)
    logger.info("Building features...")
    features = build_features(returns)
    save_features(features, args.output_dir / "features.parquet")

    return returns, features


def main():
    parser = argparse.ArgumentParser(
        description="Prepare production data from DuckDB or CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From DuckDB database
    python prepare_data.py --duckdb /path/to/prices.duckdb

    # From DuckDB with date filter
    python prepare_data.py --duckdb data.duckdb --start-date 2020-01-01

    # From CSV files
    python prepare_data.py --raw-dir /path/to/csvs
        """,
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--duckdb",
        type=Path,
        help="Path to DuckDB database file",
    )
    input_group.add_argument(
        "--raw-dir",
        type=Path,
        default=PRODUCTION_DIR / "data" / "raw",
        help="Directory containing OHLCV CSV files",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PRODUCTION_DIR / "data" / "processed",
        help="Output directory for processed data",
    )

    # Filtering options
    parser.add_argument(
        "--min-history",
        type=int,
        default=252,
        help="Minimum trading days required per asset (default: 252)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Minimum fraction of assets required per date (CSV only)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date filter (YYYY-MM-DD, DuckDB only)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date filter (YYYY-MM-DD, DuckDB only)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Process data
    try:
        if args.duckdb:
            returns, features = prepare_from_duckdb(args, logger)
        else:
            returns, features = prepare_from_csv(args, logger)

        # Summary
        n_assets = returns.shape[1]
        n_dates = returns.shape[0]

        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETE")
        print("=" * 60)
        print(f"Assets: {n_assets:,}")
        print(f"Trading days: {n_dates:,}")
        print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")
        print(f"\nOutput: {args.output_dir}")
        print("Files created:")
        print(f"  - returns.parquet ({n_dates} x {n_assets})")
        print(f"  - features.parquet ({features.shape[0]} x {features.shape[1]})")
        print(f"  - assets.txt")
        print(f"  - asset_stats.csv")
        if args.duckdb:
            print(f"  - data_info.json")

        # Print family distribution
        print_family_summary(features)

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == "__main__":
    main()
