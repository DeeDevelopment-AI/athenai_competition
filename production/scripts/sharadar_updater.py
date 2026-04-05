#!/usr/bin/env python
"""
Sharadar Data Updater
=====================
Downloads incremental updates from Nasdaq Data Link (Sharadar) and updates DuckDB.

Supports the full Sharadar Equity Bundle:
- SEP: Stock prices (equity_prices_daily)
- SFP: Fund prices (sharadar_sfp)
- SF1: Fundamentals (equity_fundamentals)
- SF2: Insider transactions (sharadar_sf2)
- SF3: Institutional holdings (sharadar_sf3)
- TICKERS: Ticker metadata (equity_universe)
- ACTIONS: Corporate actions (sharadar_actions)
- EVENTS: Events (sharadar_events)
- SP500: S&P 500 changes (equity_sp500)
- DAILY: Daily metrics (equity_daily_metrics)

Usage:
    # Update all tables from last stored date
    python production/scripts/sharadar_updater.py

    # Update specific tables only
    python production/scripts/sharadar_updater.py --tables SEP SFP SF1

    # Full refresh of a table
    python production/scripts/sharadar_updater.py --tables SEP --full-refresh

    # Check status only
    python production/scripts/sharadar_updater.py --status

Environment:
    Set NASDAQ_DATA_LINK_API_KEY or QUANDL_API_KEY environment variable
    Or create production/config/api_keys.yaml with:
        nasdaq_data_link:
            api_key: YOUR_API_KEY
"""

import argparse
import json
import logging
import os
import ssl
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import yaml

# Disable SSL warnings (for corporate proxies with self-signed certs)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PRODUCTION_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


_SSL_DISABLED = False

def disable_ssl_verification():
    """Disable SSL certificate verification for requests library."""
    global _SSL_DISABLED
    if _SSL_DISABLED:
        return  # Already disabled

    import requests
    from requests.adapters import HTTPAdapter

    # Method 1: Monkey-patch requests.Session
    _original_request = requests.Session.request

    def _patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return _original_request(self, *args, **kwargs)

    requests.Session.request = _patched_request

    # Method 2: Patch requests.get/post directly
    _original_get = requests.get
    _original_post = requests.post

    def _patched_get(*args, **kwargs):
        kwargs['verify'] = False
        return _original_get(*args, **kwargs)

    def _patched_post(*args, **kwargs):
        kwargs['verify'] = False
        return _original_post(*args, **kwargs)

    requests.get = _patched_get
    requests.post = _patched_post

    # Method 3: Set environment variables
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''

    # Method 4: Disable SSL context verification globally
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    _SSL_DISABLED = True
    logger.info("SSL verification disabled (corporate proxy workaround)")

# Default paths
DEFAULT_DUCKDB_PATH = PRODUCTION_DIR / "data" / "raw" / "quant.duckdb"
API_KEYS_FILE = PRODUCTION_DIR / "config" / "api_keys.yaml"

# Sharadar table mappings: API table -> (DuckDB table, date column, primary key columns)
SHARADAR_TABLES = {
    "SEP": {
        "duckdb_table": "equity_prices_daily",
        "date_column": "date",
        "primary_keys": ["symbol", "date"],
        "rename_columns": {"ticker": "symbol"},
        "description": "Daily stock prices (adjusted)",
    },
    "SFP": {
        "duckdb_table": "sharadar_sfp",
        "date_column": "date",
        "primary_keys": ["symbol", "date"],
        "rename_columns": {"ticker": "symbol"},
        "description": "Daily fund/ETF prices",
    },
    "SF1": {
        "duckdb_table": "equity_fundamentals",
        "date_column": "date",  # Using datekey
        "primary_keys": ["symbol", "date", "dimension"],
        "rename_columns": {"ticker": "symbol", "datekey": "date"},
        "description": "Quarterly/annual fundamentals",
    },
    "SF2": {
        "duckdb_table": "sharadar_sf2",
        "date_column": "filingdate",
        "primary_keys": ["ticker", "filingdate", "rownum"],
        "rename_columns": {},
        "description": "Insider transactions",
    },
    "SF3": {
        "duckdb_table": "sharadar_sf3",
        "date_column": "calendardate",
        "primary_keys": ["ticker", "investorname", "securitytype", "calendardate"],
        "rename_columns": {},
        "description": "Institutional holdings",
    },
    "TICKERS": {
        "duckdb_table": "equity_universe",
        "date_column": None,  # No date column
        "primary_keys": ["symbol"],
        "rename_columns": {"ticker": "symbol", "isdelisted": "is_delisted"},
        "description": "Ticker metadata",
    },
    "ACTIONS": {
        "duckdb_table": "sharadar_actions",
        "date_column": "date",
        "primary_keys": ["date", "action", "ticker"],
        "rename_columns": {},
        "description": "Corporate actions",
    },
    "EVENTS": {
        "duckdb_table": "sharadar_events",
        "date_column": "date",
        "primary_keys": ["ticker", "date"],
        "rename_columns": {},
        "description": "Events calendar",
    },
    "SP500": {
        "duckdb_table": "equity_sp500",
        "date_column": "date",
        "primary_keys": ["date", "action", "ticker"],
        "rename_columns": {},
        "description": "S&P 500 constituent changes",
    },
    "DAILY": {
        "duckdb_table": "equity_daily_metrics",
        "date_column": "date",
        "primary_keys": ["symbol", "date"],
        "rename_columns": {"ticker": "symbol"},
        "description": "Daily valuation metrics",
    },
}


def get_api_key() -> str:
    """Get Nasdaq Data Link API key from environment or config file."""
    # Check environment variables
    api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
    if api_key:
        return api_key

    api_key = os.environ.get("QUANDL_API_KEY")
    if api_key:
        return api_key

    # Check config file
    if API_KEYS_FILE.exists():
        with open(API_KEYS_FILE) as f:
            config = yaml.safe_load(f)
            if config and "nasdaq_data_link" in config:
                api_key = config["nasdaq_data_link"].get("api_key")
                if api_key:
                    return api_key

    raise ValueError(
        "API key not found. Set NASDAQ_DATA_LINK_API_KEY environment variable "
        f"or create {API_KEYS_FILE}"
    )


def get_table_status(
    conn: duckdb.DuckDBPyConnection,
    table_config: dict,
) -> dict:
    """Get status of a DuckDB table."""
    table_name = table_config["duckdb_table"]
    date_col = table_config["date_column"]

    status = {
        "table": table_name,
        "exists": False,
        "rows": 0,
        "min_date": None,
        "max_date": None,
    }

    try:
        # Check if table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        if table_name not in [t[0] for t in tables]:
            return status

        status["exists"] = True

        # Get row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        status["rows"] = count

        # Get date range if date column exists
        if date_col:
            try:
                result = conn.execute(f"""
                    SELECT MIN({date_col}), MAX({date_col})
                    FROM {table_name}
                """).fetchone()
                status["min_date"] = str(result[0]) if result[0] else None
                status["max_date"] = str(result[1]) if result[1] else None
            except:
                pass

    except Exception as e:
        logger.warning(f"Error getting status for {table_name}: {e}")

    return status


def show_status(db_path: Path) -> None:
    """Show status of all Sharadar tables."""
    print("\n" + "=" * 80)
    print("SHARADAR DATA STATUS")
    print("=" * 80)
    print(f"Database: {db_path}")
    print()

    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        print(f"{'API Table':<10} {'DuckDB Table':<25} {'Rows':>12} {'Min Date':<12} {'Max Date':<12}")
        print("-" * 80)

        for api_table, config in SHARADAR_TABLES.items():
            status = get_table_status(conn, config)

            if status["exists"]:
                rows = f"{status['rows']:,}"
                min_date = status["min_date"][:10] if status["min_date"] else "N/A"
                max_date = status["max_date"][:10] if status["max_date"] else "N/A"
            else:
                rows = "NOT FOUND"
                min_date = "-"
                max_date = "-"

            print(f"{api_table:<10} {config['duckdb_table']:<25} {rows:>12} {min_date:<12} {max_date:<12}")

        print("-" * 80)

    finally:
        conn.close()


def download_sharadar_table(
    api_key: str,
    table_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download data from Sharadar table via Nasdaq Data Link API.

    Parameters
    ----------
    api_key : str
        Nasdaq Data Link API key
    table_name : str
        Sharadar table name (SEP, SFP, SF1, etc.)
    start_date : str, optional
        Start date for incremental download (YYYY-MM-DD)
    end_date : str, optional
        End date for download

    Returns
    -------
    pd.DataFrame
        Downloaded data
    """
    # Disable SSL verification (for corporate proxies)
    disable_ssl_verification()

    try:
        import nasdaqdatalink
    except ImportError:
        raise ImportError("nasdaqdatalink package not installed. Run: pip install nasdaq-data-link")

    nasdaqdatalink.ApiConfig.api_key = api_key
    nasdaqdatalink.ApiConfig.verify_ssl = False  # Disable SSL verification

    logger.info(f"Downloading SHARADAR/{table_name}...")

    # Build query parameters
    params = {}

    # Date filtering based on table type
    config = SHARADAR_TABLES.get(table_name, {})
    date_col = config.get("date_column")

    if date_col and start_date:
        # Use the appropriate date column for filtering
        if table_name in ["SEP", "SFP", "DAILY", "ACTIONS", "EVENTS", "SP500"]:
            params["date"] = {"gte": start_date}
        elif table_name == "SF1":
            params["datekey"] = {"gte": start_date}
        elif table_name == "SF2":
            params["filingdate"] = {"gte": start_date}
        elif table_name == "SF3":
            params["calendardate"] = {"gte": start_date}

    if end_date:
        if "date" in params:
            params["date"]["lte"] = end_date
        elif "datekey" in params:
            params["datekey"]["lte"] = end_date

    # Download with pagination
    all_data = []
    cursor = None
    page = 0

    while True:
        page += 1
        logger.info(f"  Fetching page {page}...")

        try:
            if cursor:
                result = nasdaqdatalink.get_table(
                    f"SHARADAR/{table_name}",
                    paginate=True,
                    qopts={"cursor_id": cursor},
                    **params,
                )
            else:
                result = nasdaqdatalink.get_table(
                    f"SHARADAR/{table_name}",
                    paginate=True,
                    **params,
                )

            if isinstance(result, tuple):
                df, cursor_info = result
                cursor = cursor_info.get("next_cursor_id") if cursor_info else None
            else:
                df = result
                cursor = None

            if df is None or len(df) == 0:
                break

            all_data.append(df)
            logger.info(f"    Got {len(df)} rows")

            if cursor is None:
                break

        except Exception as e:
            if "cursor" in str(e).lower():
                # Try without pagination for small tables
                logger.info("  Trying without pagination...")
                df = nasdaqdatalink.get_table(f"SHARADAR/{table_name}", **params)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                break
            else:
                raise

    if not all_data:
        logger.warning(f"No data downloaded for {table_name}")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Downloaded {len(combined):,} total rows for {table_name}")

    return combined


def upsert_data(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_config: dict,
) -> int:
    """
    Upsert data into DuckDB table.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection
    df : pd.DataFrame
        Data to insert
    table_config : dict
        Table configuration from SHARADAR_TABLES

    Returns
    -------
    int
        Number of rows affected
    """
    if df.empty:
        return 0

    table_name = table_config["duckdb_table"]
    primary_keys = table_config["primary_keys"]
    rename_cols = table_config.get("rename_columns", {})

    # Rename columns
    df = df.rename(columns=rename_cols)

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    logger.info(f"Upserting {len(df):,} rows into {table_name}...")

    # Check if table exists
    tables = conn.execute("SHOW TABLES").fetchall()
    table_exists = table_name in [t[0] for t in tables]

    if not table_exists:
        # Create table from DataFrame
        logger.info(f"Creating table {table_name}...")
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        return len(df)

    # Get existing columns
    schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
    existing_cols = [row[0].lower() for row in schema]

    # Filter DataFrame to only include existing columns
    df_cols = [c for c in df.columns if c.lower() in existing_cols]
    df = df[df_cols]

    # Delete existing records that will be replaced (based on primary keys)
    pk_cols_lower = [pk.lower() for pk in primary_keys]
    pk_cols_exist = [pk for pk in pk_cols_lower if pk in df.columns]

    if pk_cols_exist:
        # Build delete condition
        # First, register the DataFrame
        conn.register("new_data", df)

        # Build the WHERE clause for deletion
        where_conditions = " AND ".join(
            f"{table_name}.{pk} = new_data.{pk}" for pk in pk_cols_exist
        )

        delete_query = f"""
            DELETE FROM {table_name}
            WHERE EXISTS (
                SELECT 1 FROM new_data
                WHERE {where_conditions}
            )
        """

        try:
            conn.execute(delete_query)
        except Exception as e:
            logger.warning(f"Delete failed (table may be empty): {e}")

        conn.unregister("new_data")

    # Insert new data
    conn.register("new_data", df)

    cols_str = ", ".join(df.columns)
    conn.execute(f"INSERT INTO {table_name} ({cols_str}) SELECT {cols_str} FROM new_data")

    conn.unregister("new_data")

    return len(df)


def update_table(
    api_key: str,
    conn: duckdb.DuckDBPyConnection,
    api_table: str,
    full_refresh: bool = False,
    days_overlap: int = 3,
) -> dict:
    """
    Update a single Sharadar table.

    Parameters
    ----------
    api_key : str
        Nasdaq Data Link API key
    conn : duckdb.DuckDBPyConnection
        DuckDB connection
    api_table : str
        Sharadar API table name
    full_refresh : bool
        If True, download all data; if False, incremental from last date
    days_overlap : int
        Days to overlap when doing incremental update (for late data)

    Returns
    -------
    dict
        Update results
    """
    if api_table not in SHARADAR_TABLES:
        raise ValueError(f"Unknown table: {api_table}. Valid: {list(SHARADAR_TABLES.keys())}")

    config = SHARADAR_TABLES[api_table]
    result = {
        "table": api_table,
        "duckdb_table": config["duckdb_table"],
        "success": False,
        "rows_before": 0,
        "rows_after": 0,
        "rows_added": 0,
        "error": None,
    }

    try:
        # Get current status
        status = get_table_status(conn, config)
        result["rows_before"] = status["rows"]

        # Determine start date
        start_date = None
        if not full_refresh and status["max_date"] and config["date_column"]:
            # Go back a few days for overlap (late data corrections)
            max_date = pd.to_datetime(status["max_date"])
            start_date = (max_date - timedelta(days=days_overlap)).strftime("%Y-%m-%d")
            logger.info(f"Incremental update from {start_date}")
        else:
            logger.info("Full refresh")

        # Download data
        df = download_sharadar_table(api_key, api_table, start_date=start_date)

        if df.empty:
            logger.info(f"No new data for {api_table}")
            result["success"] = True
            result["rows_after"] = status["rows"]
            return result

        # Upsert data
        rows_affected = upsert_data(conn, df, config)

        # Get new status
        new_status = get_table_status(conn, config)
        result["rows_after"] = new_status["rows"]
        result["rows_added"] = new_status["rows"] - status["rows"]
        result["success"] = True

        logger.info(f"Updated {api_table}: {result['rows_before']:,} -> {result['rows_after']:,} rows")

    except Exception as e:
        logger.error(f"Error updating {api_table}: {e}")
        result["error"] = str(e)

    return result


def recalculate_metrics(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Recalculate daily_metrics table after price updates.

    This updates the rolling metrics (returns, volatility, Sharpe, Sortino, etc.)
    based on the latest equity_prices_daily data.
    """
    logger.info("Recalculating daily_metrics...")

    # Check if we have the source table
    tables = conn.execute("SHOW TABLES").fetchall()
    if "equity_prices_daily" not in [t[0] for t in tables]:
        logger.warning("equity_prices_daily not found, skipping metrics calculation")
        return

    # Get date range to recalculate (last 126 days to ensure rolling windows are complete)
    max_date = conn.execute("SELECT MAX(date) FROM equity_prices_daily").fetchone()[0]
    if not max_date:
        return

    start_date = pd.to_datetime(max_date) - timedelta(days=126)
    start_date_str = start_date.strftime("%Y-%m-%d")

    logger.info(f"Recalculating metrics from {start_date_str} to {max_date}")

    # The calculation is complex - we'll do it in Python for clarity
    # Load recent price data
    prices_df = conn.execute(f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume
        FROM equity_prices_daily
        WHERE date >= '{start_date_str}'
        ORDER BY symbol, date
    """).fetchdf()

    if prices_df.empty:
        return

    prices_df["date"] = pd.to_datetime(prices_df["date"])

    # Calculate returns
    prices_df["daily_return"] = prices_df.groupby("symbol")["close"].pct_change()

    # Calculate rolling metrics for each window
    windows = [5, 21, 63]
    metrics_list = []

    for symbol in prices_df["symbol"].unique():
        sym_data = prices_df[prices_df["symbol"] == symbol].copy()

        if len(sym_data) < 5:
            continue

        for w in windows:
            if len(sym_data) < w:
                continue

            # Rolling return (annualized)
            sym_data[f"return_rolling_{w}d"] = sym_data["daily_return"].rolling(w).mean() * 252

            # Rolling volatility (annualized)
            sym_data[f"volatility_rolling_{w}d"] = sym_data["daily_return"].rolling(w).std() * (252 ** 0.5)

            # Drawdown
            cumret = (1 + sym_data["daily_return"].fillna(0)).cumprod()
            roll_max = cumret.rolling(w, min_periods=1).max()
            sym_data[f"drawdown_rolling_{w}d"] = (cumret - roll_max) / roll_max

            # Downside deviation
            downside = sym_data["daily_return"].copy()
            downside[downside > 0] = 0
            sym_data[f"downside_dev_{w}d"] = downside.rolling(w).apply(
                lambda x: (x**2).mean() ** 0.5 * (252**0.5), raw=True
            )

            # Sharpe
            sharpe = sym_data[f"return_rolling_{w}d"] / sym_data[f"volatility_rolling_{w}d"].replace(0, float("nan"))
            sym_data[f"sharpe_{w}d"] = sharpe.clip(-10, 10)
            sym_data[f"sharpe_{w}d_valid"] = sym_data[f"sharpe_{w}d"].notna()

            # Sortino
            sortino = sym_data[f"return_rolling_{w}d"] / sym_data[f"downside_dev_{w}d"].replace(0, float("nan"))
            sym_data[f"sortino_{w}d"] = sortino.clip(-10, 10)
            sym_data[f"sortino_{w}d_valid"] = sym_data[f"sortino_{w}d"].notna()

            # Calmar (return / max drawdown)
            max_dd = sym_data[f"drawdown_rolling_{w}d"].rolling(w).min().abs()
            calmar = sym_data[f"return_rolling_{w}d"] / max_dd.replace(0, float("nan"))
            sym_data[f"calmar_{w}d"] = calmar.clip(-10, 10)
            sym_data[f"calmar_{w}d_valid"] = sym_data[f"calmar_{w}d"].notna()

        metrics_list.append(sym_data)

    if not metrics_list:
        return

    metrics_df = pd.concat(metrics_list, ignore_index=True)

    # Select columns matching daily_metrics schema
    metric_cols = [
        "symbol", "date", "open", "high", "low", "close", "adj_close", "volume",
        "return_rolling_5d", "return_rolling_21d", "return_rolling_63d",
        "volatility_rolling_5d", "volatility_rolling_21d", "volatility_rolling_63d",
        "drawdown_rolling_5d", "drawdown_rolling_21d", "drawdown_rolling_63d",
        "downside_dev_5d", "downside_dev_21d", "downside_dev_63d",
        "sharpe_5d", "sharpe_21d", "sharpe_63d",
        "sortino_5d", "sortino_21d", "sortino_63d",
        "calmar_5d", "calmar_21d", "calmar_63d",
        "sharpe_5d_valid", "sharpe_21d_valid", "sharpe_63d_valid",
        "sortino_5d_valid", "sortino_21d_valid", "sortino_63d_valid",
        "calmar_5d_valid", "calmar_21d_valid", "calmar_63d_valid",
    ]

    # Keep only columns that exist
    existing_cols = [c for c in metric_cols if c in metrics_df.columns]
    metrics_df = metrics_df[existing_cols]

    # Delete old records for the date range we're updating
    conn.execute(f"""
        DELETE FROM daily_metrics
        WHERE date >= '{start_date_str}'
    """)

    # Insert new metrics
    conn.register("new_metrics", metrics_df)
    cols_str = ", ".join(existing_cols)
    conn.execute(f"INSERT INTO daily_metrics ({cols_str}) SELECT {cols_str} FROM new_metrics")
    conn.unregister("new_metrics")

    logger.info(f"Updated {len(metrics_df):,} rows in daily_metrics")


def update_all_tables(
    api_key: str,
    db_path: Path,
    tables: Optional[list[str]] = None,
    full_refresh: bool = False,
    recalculate: bool = True,
) -> dict:
    """
    Update all (or selected) Sharadar tables.

    Parameters
    ----------
    api_key : str
        Nasdaq Data Link API key
    db_path : Path
        Path to DuckDB database
    tables : list[str], optional
        List of tables to update (default: all)
    full_refresh : bool
        If True, download all data
    recalculate : bool
        If True, recalculate daily_metrics after SEP update

    Returns
    -------
    dict
        Update results for all tables
    """
    if tables is None:
        tables = list(SHARADAR_TABLES.keys())

    # Validate table names
    invalid = [t for t in tables if t not in SHARADAR_TABLES]
    if invalid:
        raise ValueError(f"Invalid tables: {invalid}. Valid: {list(SHARADAR_TABLES.keys())}")

    print("\n" + "=" * 60)
    print(f"SHARADAR DATA UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Tables: {', '.join(tables)}")
    print(f"Mode: {'Full refresh' if full_refresh else 'Incremental'}")
    print()

    conn = duckdb.connect(str(db_path))

    results = {
        "timestamp": datetime.now().isoformat(),
        "tables": {},
        "success": True,
    }

    try:
        for api_table in tables:
            print(f"\n[{tables.index(api_table) + 1}/{len(tables)}] Updating {api_table}...")
            result = update_table(api_key, conn, api_table, full_refresh=full_refresh)
            results["tables"][api_table] = result

            if not result["success"]:
                results["success"] = False
                print(f"  ✗ Error: {result['error']}")
            else:
                print(f"  ✓ {result['rows_before']:,} -> {result['rows_after']:,} rows (+{result['rows_added']:,})")

        # Recalculate metrics if SEP was updated
        if recalculate and "SEP" in tables:
            print("\nRecalculating daily metrics...")
            try:
                recalculate_metrics(conn)
                print("  ✓ Metrics updated")
            except Exception as e:
                logger.error(f"Error recalculating metrics: {e}")
                print(f"  ✗ Error: {e}")

    finally:
        conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)

    total_added = sum(r.get("rows_added", 0) for r in results["tables"].values())
    failed = [t for t, r in results["tables"].items() if not r["success"]]

    print(f"Total rows added: {total_added:,}")
    print(f"Success: {len(results['tables']) - len(failed)}/{len(results['tables'])} tables")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    print("=" * 60)

    # Save results
    results_file = PRODUCTION_DIR / "outputs" / "logs" / f"sharadar_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Update Sharadar data from Nasdaq Data Link",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Update all tables from last stored date
    python sharadar_updater.py

    # Update specific tables
    python sharadar_updater.py --tables SEP SFP SF1

    # Full refresh
    python sharadar_updater.py --tables SEP --full-refresh

    # Check status
    python sharadar_updater.py --status

Environment:
    Set NASDAQ_DATA_LINK_API_KEY or create production/config/api_keys.yaml
        """,
    )

    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(SHARADAR_TABLES.keys()),
        help="Tables to update (default: all)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status only, don't update",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Download all data instead of incremental",
    )
    parser.add_argument(
        "--no-recalculate",
        action="store_true",
        help="Skip recalculating daily_metrics after update",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DUCKDB_PATH,
        help=f"Path to DuckDB database (default: {DEFAULT_DUCKDB_PATH})",
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

    # Status only
    if args.status:
        if not args.db.exists():
            print(f"Database not found: {args.db}")
            sys.exit(1)
        show_status(args.db)
        return

    # Get API key
    try:
        api_key = get_api_key()
        logger.info("API key found")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Run update
    try:
        results = update_all_tables(
            api_key=api_key,
            db_path=args.db,
            tables=args.tables,
            full_refresh=args.full_refresh,
            recalculate=not args.no_recalculate,
        )

        if not results["success"]:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise


if __name__ == "__main__":
    main()
