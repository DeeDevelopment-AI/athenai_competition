"""
DuckDB Data Loader
==================
Loads training data from DuckDB database with pre-computed features.
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBLoader:
    """
    Loader for DuckDB database with OHLCV and pre-computed features.

    Expected schema:
        symbol, date, open, high, low, close, adj_close, volume,
        return_rolling_5d, return_rolling_21d, return_rolling_63d,
        volatility_rolling_5d, volatility_rolling_21d, volatility_rolling_63d,
        sharpe_5d, sharpe_21d, sharpe_63d,
        sortino_5d, sortino_21d, sortino_63d,
        calmar_5d, calmar_21d, calmar_63d,
        + validity flags
    """

    def __init__(self, db_path: str | Path, table_name: str = None):
        """
        Initialize loader with database path.

        Parameters
        ----------
        db_path : str or Path
            Path to DuckDB database file
        table_name : str, optional
            Table name to use (auto-detected if None)
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        self._table_name = table_name or self._detect_table_name()

        # Detect column names
        self._detect_columns()

        logger.info(f"Connected to DuckDB: {self.db_path}")
        logger.info(f"Using table: {self._table_name}")
        logger.info(f"Symbol column: {self._symbol_col}, Date column: {self._date_col}")

    def _detect_columns(self) -> None:
        """Detect symbol and date column names."""
        schema = self.conn.execute(f"DESCRIBE {self._table_name}").fetchall()
        col_names = [row[0].lower() for row in schema]

        # Detect symbol column
        for candidate in ["symbol", "ticker", "asset", "name", "ts"]:
            if candidate in col_names:
                self._symbol_col = schema[col_names.index(candidate)][0]
                break
        else:
            raise ValueError(f"Could not detect symbol column. Columns: {[r[0] for r in schema]}")

        # Detect date column
        for candidate in ["date", "datetime", "timestamp", "time", "ts"]:
            if candidate in col_names:
                self._date_col = schema[col_names.index(candidate)][0]
                break
        else:
            raise ValueError(f"Could not detect date column. Columns: {[r[0] for r in schema]}")

    def _detect_table_name(self) -> str:
        """Detect the main data table name."""
        tables = self.conn.execute("SHOW TABLES").fetchall()
        if not tables:
            raise ValueError("No tables found in database")

        # Return first table or look for common names (in priority order)
        table_names = [t[0] for t in tables]

        # Prefer tables with pre-computed features
        for candidate in ["daily_metrics", "features", "prices_daily", "equity_prices_daily"]:
            if candidate in table_names:
                return candidate

        # Then try basic price tables
        for candidate in ["prices", "ohlcv", "data", "stocks"]:
            if candidate in table_names:
                return candidate

        return table_names[0]

    def get_info(self) -> dict:
        """Get database information."""
        query = f"""
        SELECT
            COUNT(*) as n_rows,
            COUNT(DISTINCT {self._symbol_col}) as n_symbols,
            COUNT(DISTINCT {self._date_col}) as n_dates,
            MIN({self._date_col}) as min_date,
            MAX({self._date_col}) as max_date
        FROM {self._table_name}
        """
        result = self.conn.execute(query).fetchone()

        return {
            "n_rows": result[0],
            "n_symbols": result[1],
            "n_dates": result[2],
            "min_date": result[3],
            "max_date": result[4],
            "table_name": self._table_name,
            "symbol_column": self._symbol_col,
            "date_column": self._date_col,
        }

    def get_symbols(self, min_observations: int = 252) -> list[str]:
        """
        Get list of symbols with sufficient history.

        Parameters
        ----------
        min_observations : int
            Minimum number of observations required

        Returns
        -------
        list[str]
            List of symbol names
        """
        query = f"""
        SELECT {self._symbol_col}, COUNT(*) as n_obs
        FROM {self._table_name}
        WHERE close IS NOT NULL
        GROUP BY {self._symbol_col}
        HAVING n_obs >= {min_observations}
        ORDER BY {self._symbol_col}
        """
        result = self.conn.execute(query).fetchall()
        return [r[0] for r in result]

    def load_returns(
        self,
        symbols: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_observations: int = 252,
    ) -> pd.DataFrame:
        """
        Load daily returns matrix [dates x symbols].

        Parameters
        ----------
        symbols : list[str], optional
            Symbols to load (default: all with sufficient history)
        start_date : str, optional
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (YYYY-MM-DD)
        min_observations : int
            Minimum observations per symbol

        Returns
        -------
        pd.DataFrame
            Returns matrix [dates x symbols]
        """
        if symbols is None:
            symbols = self.get_symbols(min_observations)

        logger.info(f"Loading returns for {len(symbols)} symbols...")

        # Build query with filters
        where_clauses = ["close IS NOT NULL"]
        if start_date:
            where_clauses.append(f"{self._date_col} >= '{start_date}'")
        if end_date:
            where_clauses.append(f"{self._date_col} <= '{end_date}'")
        if symbols:
            symbols_str = ", ".join(f"'{s}'" for s in symbols)
            where_clauses.append(f"{self._symbol_col} IN ({symbols_str})")

        where_sql = " AND ".join(where_clauses)

        # Compute returns from close prices
        query = f"""
        WITH prices AS (
            SELECT {self._symbol_col} as symbol, {self._date_col} as date, close
            FROM {self._table_name}
            WHERE {where_sql}
        ),
        returns AS (
            SELECT
                symbol,
                date,
                (close / LAG(close) OVER (PARTITION BY symbol ORDER BY date)) - 1 AS daily_return
            FROM prices
        )
        SELECT symbol, date, daily_return
        FROM returns
        WHERE daily_return IS NOT NULL
        ORDER BY date, symbol
        """

        df = self.conn.execute(query).fetchdf()

        # Pivot to wide format
        returns = df.pivot(index="date", columns="symbol", values="daily_return")
        returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index()

        logger.info(f"Loaded returns: {returns.shape[0]} dates x {returns.shape[1]} symbols")

        return returns

    def load_features(
        self,
        symbols: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load pre-computed features from database.

        Parameters
        ----------
        symbols : list[str], optional
            Symbols to load
        start_date : str, optional
            Start date
        end_date : str, optional
            End date
        feature_columns : list[str], optional
            Specific feature columns to load (default: all rolling features)

        Returns
        -------
        pd.DataFrame
            Features with MultiIndex columns [(symbol, feature)]
        """
        if feature_columns is None:
            # Default feature columns from your schema
            feature_columns = [
                "return_rolling_5d", "return_rolling_21d", "return_rolling_63d",
                "volatility_rolling_5d", "volatility_rolling_21d", "volatility_rolling_63d",
                "sharpe_5d", "sharpe_21d", "sharpe_63d",
                "sortino_5d", "sortino_21d", "sortino_63d",
                "calmar_5d", "calmar_21d", "calmar_63d",
            ]

        if symbols is None:
            symbols = self.get_symbols()

        logger.info(f"Loading {len(feature_columns)} features for {len(symbols)} symbols...")

        # Build query
        where_clauses = []
        if start_date:
            where_clauses.append(f"{self._date_col} >= '{start_date}'")
        if end_date:
            where_clauses.append(f"{self._date_col} <= '{end_date}'")
        if symbols:
            symbols_str = ", ".join(f"'{s}'" for s in symbols)
            where_clauses.append(f"{self._symbol_col} IN ({symbols_str})")

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        cols_sql = ", ".join([f"{self._symbol_col} as symbol", f"{self._date_col} as date"] + feature_columns)

        query = f"""
        SELECT {cols_sql}
        FROM {self._table_name}
        {where_sql}
        ORDER BY date, symbol
        """

        df = self.conn.execute(query).fetchdf()
        df["date"] = pd.to_datetime(df["date"])

        # Pivot each feature column
        features = {}
        for feat in feature_columns:
            if feat in df.columns:
                pivoted = df.pivot(index="date", columns="symbol", values=feat)
                for symbol in pivoted.columns:
                    features[(symbol, feat)] = pivoted[symbol]

        feature_df = pd.DataFrame(features)
        feature_df.columns = pd.MultiIndex.from_tuples(
            feature_df.columns, names=["symbol", "feature"]
        )
        feature_df = feature_df.sort_index()

        logger.info(f"Loaded features: {feature_df.shape[0]} dates x {feature_df.shape[1]} columns")

        return feature_df

    def load_sortino_families(
        self,
        symbols: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load Sortino ratios and compute family classifications.

        Returns
        -------
        pd.DataFrame
            Sortino values and family labels for each window
        """
        sortino_cols = ["sortino_5d", "sortino_21d", "sortino_63d"]

        features = self.load_features(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            feature_columns=sortino_cols,
        )

        # Add family classifications
        def classify_family(sortino: float) -> int:
            if pd.isna(sortino):
                return -1
            if sortino > 2.0:
                return 0
            elif sortino > 1.0:
                return 1
            elif sortino > 0.0:
                return 2
            else:
                return 3

        symbols = list(set(c[0] for c in features.columns))

        for window in ["5d", "21d", "63d"]:
            for symbol in symbols:
                sortino_col = (symbol, f"sortino_{window}")
                if sortino_col in features.columns:
                    family_col = (symbol, f"family_{window}")
                    features[family_col] = features[sortino_col].apply(classify_family)

        return features

    def prepare_training_data(
        self,
        output_dir: Path,
        min_observations: int = 252,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare all training data and save to output directory.

        Parameters
        ----------
        output_dir : Path
            Directory to save processed data
        min_observations : int
            Minimum observations per symbol
        start_date : str, optional
            Start date filter
        end_date : str, optional
            End date filter

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (returns, features) DataFrames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get valid symbols
        symbols = self.get_symbols(min_observations)
        logger.info(f"Found {len(symbols)} symbols with >= {min_observations} observations")

        # Load returns
        returns = self.load_returns(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        # Load features (including Sortino families)
        features = self.load_sortino_families(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        # Align dates
        common_dates = returns.index.intersection(features.index)
        returns = returns.loc[common_dates]
        features = features.loc[common_dates]

        # Save
        returns.to_parquet(output_dir / "returns.parquet")
        features.to_parquet(output_dir / "features.parquet")

        # Save asset list
        (output_dir / "assets.txt").write_text("\n".join(returns.columns))

        # Save summary stats
        stats = pd.DataFrame({
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "sharpe": returns.mean() / returns.std() * np.sqrt(252),
            "n_observations": returns.notna().sum(),
        })
        stats.to_csv(output_dir / "asset_stats.csv")

        # Save info
        info = self.get_info()
        info["n_symbols_selected"] = len(symbols)
        info["n_dates_selected"] = len(common_dates)
        info["date_range"] = f"{common_dates.min()} to {common_dates.max()}"

        import json
        with open(output_dir / "data_info.json", "w") as f:
            json.dump(info, f, indent=2, default=str)

        logger.info(f"Saved training data to {output_dir}")
        logger.info(f"  Returns: {returns.shape}")
        logger.info(f"  Features: {features.shape}")

        return returns, features

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_from_duckdb(
    db_path: str | Path,
    output_dir: Optional[Path] = None,
    min_observations: int = 252,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load data from DuckDB.

    Parameters
    ----------
    db_path : str or Path
        Path to DuckDB database
    output_dir : Path, optional
        Directory to save processed data (if None, doesn't save)
    min_observations : int
        Minimum observations per symbol
    start_date : str, optional
        Start date (YYYY-MM-DD)
    end_date : str, optional
        End date (YYYY-MM-DD)

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (returns, features) DataFrames
    """
    with DuckDBLoader(db_path) as loader:
        info = loader.get_info()
        print(f"Database: {info['n_rows']:,} rows, {info['n_symbols']:,} symbols")
        print(f"Date range: {info['min_date']} to {info['max_date']}")

        if output_dir:
            return loader.prepare_training_data(
                output_dir=output_dir,
                min_observations=min_observations,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            symbols = loader.get_symbols(min_observations)
            returns = loader.load_returns(symbols, start_date, end_date)
            features = loader.load_sortino_families(symbols, start_date, end_date)
            return returns, features


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python duckdb_loader.py <path_to_database.duckdb>")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    production_dir = Path(__file__).parent.parent
    output_dir = production_dir / "data" / "processed"

    returns, features = load_from_duckdb(
        db_path=db_path,
        output_dir=output_dir,
        min_observations=252,
    )

    print(f"\nReturns: {returns.shape}")
    print(f"Features: {features.shape}")
