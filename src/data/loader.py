"""
Carga de datos de algoritmos (OHLC) y benchmark.

Formato real de datos:
- Algoritmos: CSVs con datetime, open, high, low, close (nombres de archivo = productname)
- Benchmark:
  - benchmark_monthly_returns.csv: month, start_equity, end_equity, monthly_return
  - benchmark_yearly_returns.csv: year, start_equity, end_equity, yearly_return
  - trades_benchmark.csv: volume, dateOpen, dateClose, total_invested_amount_EOD,
                          equity_EOD, AUM, equity_normalized, productname
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Standard date formats to try when parsing
_DATE_FORMATS = [
    '%Y%m%d %H%M%S',
    '%d/%m/%Y %H:%M',
    '%m/%d/%Y %H:%M',
    '%d-%m-%Y %H:%M:%S',
    '%Y/%m/%d %H:%M:%S',
    '%d.%m.%Y %H:%M:%S',
    '%Y%m%d%H%M%S',
    '%Y-%m-%d',
    '%d/%m/%Y',
    '%m/%d/%Y',
]


@dataclass
class AlgorithmData:
    """Contenedor de datos OHLC de un producto/algoritmo."""

    algo_id: str  # Nombre del producto (del filename)
    ohlc: pd.DataFrame  # datetime, open, high, low, close
    raw_path: Path


@dataclass
class BenchmarkData:
    """Contenedor de datos del benchmark."""

    monthly_returns: pd.DataFrame  # month, start_equity, end_equity, monthly_return
    yearly_returns: pd.DataFrame  # year, start_equity, end_equity, yearly_return
    trades: pd.DataFrame  # Trades con productname, fechas, montos


def _parse_datetime_robust(series: pd.Series) -> pd.Series:
    """
    Parse datetime with multiple fallback strategies.

    First tries pandas auto-detection, then falls back to common formats
    if more than 50% of values fail to parse.

    Args:
        series: Series of date strings to parse.

    Returns:
        Series of parsed datetimes (may contain NaT for failures).
    """
    # First try pandas auto-detection
    parsed = pd.to_datetime(series, errors='coerce')

    # If most values failed, try common non-standard formats
    if parsed.isna().sum() > len(parsed) * 0.5:
        for fmt in _DATE_FORMATS:
            attempt = pd.to_datetime(series, format=fmt, errors='coerce')
            if attempt.isna().sum() < len(attempt) * 0.5:
                return attempt

    return parsed


def _detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the datetime column from common naming conventions.

    Args:
        df: DataFrame to search.

    Returns:
        Name of datetime column or None if not found.
    """
    candidates = [
        'datetime', 'Datetime', 'date', 'Date', 'time', 'Time',
        'timestamp', 'Timestamp', 'Gmt time', 'gmt time'
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _detect_ohlc_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Detect OHLC columns using case-insensitive matching.

    Args:
        df: DataFrame to search.

    Returns:
        Dict mapping standard names to actual column names.
        E.g., {'open': 'Open', 'high': 'HIGH', 'low': 'low', 'close': 'Close'}
    """
    col_map = {}
    for target in ['open', 'high', 'low', 'close']:
        for col in df.columns:
            if col.lower() == target:
                col_map[target] = col
                break
    return col_map


class DataLoader:
    """
    Carga datos crudos de algoritmos (OHLC) y benchmark.

    Estructura esperada:
    data/raw/
    ├── algorithms/          # Miles de CSVs con OHLC por producto
    │   ├── fpJbh.csv
    │   ├── HcI9f.csv
    │   └── ...
    └── benchmark/
        ├── benchmark_monthly_returns.csv
        ├── benchmark_yearly_returns.csv
        └── trades_benchmark.csv
    """

    # Minimum file size in bytes to be considered valid (header + at least some data)
    MIN_FILE_SIZE = 50

    # Minimum rows required after parsing
    MIN_ROWS = 10

    def __init__(self, raw_path: str | Path = "data/raw/"):
        self.raw_path = Path(raw_path)
        # Support both "algoritmos" (actual) and "algorithms" (english) folder names
        if (self.raw_path / "algoritmos").exists():
            self.algos_path = self.raw_path / "algoritmos"
        else:
            self.algos_path = self.raw_path / "algorithms"
        self.benchmark_path = self.raw_path / "benchmark"

    def list_algorithms(self) -> list[str]:
        """
        Lista IDs de algoritmos/productos disponibles.

        Returns:
            Lista de nombres de producto (sin extensión .csv).
        """
        if not self.algos_path.exists():
            logger.warning(f"Algorithms path not found: {self.algos_path}")
            return []

        algo_files = list(self.algos_path.glob("*.csv"))
        return [f.stem for f in algo_files]

    def load_algorithm(
        self,
        algo_id: str,
        resample_to_daily: bool = True,
        validate: bool = True,
    ) -> Optional[AlgorithmData]:
        """
        Carga datos OHLC de un algoritmo/producto.

        Args:
            algo_id: Nombre del producto (stem del filename).
            resample_to_daily: If True, resample intraday data to daily OHLC.
            validate: If True, validate the data and return None for invalid files.

        Returns:
            AlgorithmData with OHLC, or None if validation fails.

        Raises:
            FileNotFoundError: Si no existe el archivo.
        """
        file_path = self.algos_path / f"{algo_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Algorithm file not found: {file_path}")

        # Check file size (skip empty/tiny files)
        if validate:
            file_size = os.path.getsize(file_path)
            if file_size < self.MIN_FILE_SIZE:
                logger.debug(f"Skipping {algo_id}: file too small ({file_size} bytes)")
                return None

        # Load CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.warning(f"Failed to read {algo_id}: {e}")
            return None

        # Check minimum rows
        if validate and len(df) < 2:
            logger.debug(f"Skipping {algo_id}: only {len(df)} rows")
            return None

        # Detect and parse datetime column
        dt_col = _detect_datetime_column(df)
        if dt_col is None:
            # Fall back to first column
            dt_col = df.columns[0]

        df['_dt'] = _parse_datetime_robust(df[dt_col])

        # Drop rows where date parsing failed
        df = df.dropna(subset=['_dt'])

        if validate and len(df) < self.MIN_ROWS:
            logger.debug(f"Skipping {algo_id}: only {len(df)} valid rows after date parsing")
            return None

        df = df.sort_values('_dt').set_index('_dt')
        df.index.name = 'datetime'

        # Detect OHLC columns (case-insensitive)
        col_map = _detect_ohlc_columns(df)

        if 'close' not in col_map:
            if validate:
                logger.debug(f"Skipping {algo_id}: no 'close' column found")
                return None
            raise ValueError(f"No 'close' column in {algo_id}")

        # Rename to standard names
        rename_map = {v: k for k, v in col_map.items()}
        df = df.rename(columns=rename_map)

        # Ensure numeric
        for c in ['open', 'high', 'low', 'close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Resample to daily if requested
        if resample_to_daily:
            agg = {}
            if 'open' in df.columns:
                agg['open'] = 'first'
            if 'high' in df.columns:
                agg['high'] = 'max'
            if 'low' in df.columns:
                agg['low'] = 'min'
            agg['close'] = 'last'

            df = df.resample('D').agg(agg).dropna(subset=['close'])

            # Fill missing OHLC from close if needed
            for c in ['open', 'high', 'low']:
                if c not in df.columns:
                    df[c] = df['close']

        # Final validation
        if validate and len(df) < self.MIN_ROWS:
            logger.debug(f"Skipping {algo_id}: only {len(df)} rows after resampling")
            return None

        logger.debug(f"Loaded {len(df)} rows for algorithm {algo_id}")

        return AlgorithmData(
            algo_id=algo_id,
            ohlc=df,
            raw_path=file_path,
        )

    def load_all_algorithms(
        self,
        algo_ids: Optional[list[str]] = None,
        show_progress: bool = True,
        resample_to_daily: bool = True,
        skip_invalid: bool = True,
    ) -> dict[str, AlgorithmData]:
        """
        Carga todos (o un subset de) los algoritmos disponibles.

        Args:
            algo_ids: Lista de IDs a cargar (None = todos).
            show_progress: Si mostrar progreso.
            resample_to_daily: If True, resample intraday data to daily.
            skip_invalid: If True, skip invalid files instead of raising errors.

        Returns:
            Dict {algo_id: AlgorithmData}.
        """
        if algo_ids is None:
            algo_ids = self.list_algorithms()

        if not algo_ids:
            logger.warning("No algorithms found")
            return {}

        algorithms = {}
        skipped = 0
        total = len(algo_ids)

        for i, algo_id in enumerate(algo_ids):
            try:
                result = self.load_algorithm(
                    algo_id,
                    resample_to_daily=resample_to_daily,
                    validate=skip_invalid,
                )
                if result is not None:
                    algorithms[algo_id] = result
                else:
                    skipped += 1

                if show_progress and (i + 1) % 100 == 0:
                    logger.info(f"Loaded {i + 1}/{total} algorithms ({skipped} skipped)")
            except Exception as e:
                if skip_invalid:
                    skipped += 1
                    logger.debug(f"Error loading {algo_id}: {e}")
                else:
                    logger.error(f"Error loading {algo_id}: {e}")

        logger.info(f"Loaded {len(algorithms)} algorithms ({skipped} skipped)")
        return algorithms

    def load_benchmark(self) -> BenchmarkData:
        """
        Carga datos del benchmark.

        Returns:
            BenchmarkData con retornos mensuales, anuales y trades.

        Raises:
            FileNotFoundError: Si no existen los archivos requeridos.
        """
        monthly_path = self.benchmark_path / "benchmark_monthly_returns.csv"
        yearly_path = self.benchmark_path / "benchmark_yearly_returns.csv"
        trades_path = self.benchmark_path / "trades_benchmark.csv"

        # Cargar retornos mensuales
        if monthly_path.exists():
            monthly = pd.read_csv(monthly_path)
            monthly["month"] = pd.to_datetime(monthly["month"])
            monthly = monthly.set_index("month").sort_index()
            logger.info(f"Loaded {len(monthly)} monthly benchmark returns")
        else:
            logger.warning(f"Monthly returns not found: {monthly_path}")
            monthly = pd.DataFrame()

        # Cargar retornos anuales
        if yearly_path.exists():
            yearly = pd.read_csv(yearly_path)
            yearly = yearly.set_index("year").sort_index()
            logger.info(f"Loaded {len(yearly)} yearly benchmark returns")
        else:
            logger.warning(f"Yearly returns not found: {yearly_path}")
            yearly = pd.DataFrame()

        # Cargar trades (crítico para entender qué productos usa el benchmark)
        if trades_path.exists():
            trades = pd.read_csv(trades_path)
            # Parsear fechas
            trades["dateOpen"] = pd.to_datetime(trades["dateOpen"], format='mixed')
            trades["dateClose"] = pd.to_datetime(trades["dateClose"], format='mixed')
            logger.info(f"Loaded {len(trades)} benchmark trades")

            # Info sobre productos únicos
            unique_products = trades["productname"].nunique()
            logger.info(f"Benchmark trades cover {unique_products} unique products")
        else:
            raise FileNotFoundError(f"Benchmark trades not found: {trades_path}")

        return BenchmarkData(
            monthly_returns=monthly,
            yearly_returns=yearly,
            trades=trades,
        )

    def get_benchmark_products(self) -> list[str]:
        """
        Obtiene lista de productos en los que invierte el benchmark.

        Returns:
            Lista de productnames únicos.
        """
        benchmark = self.load_benchmark()
        return benchmark.trades["productname"].unique().tolist()

    def load_algorithms_in_benchmark(
        self,
        resample_to_daily: bool = True,
    ) -> dict[str, AlgorithmData]:
        """
        Carga solo los algoritmos que aparecen en el benchmark.

        Args:
            resample_to_daily: If True, resample intraday data to daily.

        Returns:
            Dict {algo_id: AlgorithmData} solo para productos del benchmark.
        """
        products = self.get_benchmark_products()
        available = set(self.list_algorithms())

        # Filtrar productos que existen
        to_load = [p for p in products if p in available]
        missing = [p for p in products if p not in available]

        if missing:
            logger.warning(f"{len(missing)} benchmark products not found in algorithms: {missing[:5]}...")

        return self.load_all_algorithms(
            algo_ids=to_load,
            resample_to_daily=resample_to_daily,
        )

    def inspect_data_format(self) -> dict:
        """
        Inspecciona formato de datos disponibles.

        Returns:
            Dict con información de columnas, tipos y muestras.
        """
        info = {"algorithms": {}, "benchmark": {}}

        # Inspeccionar primer algoritmo
        algo_ids = self.list_algorithms()
        if algo_ids:
            algo = self.load_algorithm(algo_ids[0], validate=False)
            if algo is not None:
                info["algorithms"]["columns"] = list(algo.ohlc.columns)
                info["algorithms"]["dtypes"] = algo.ohlc.dtypes.astype(str).to_dict()
                info["algorithms"]["sample"] = algo.ohlc.head(3).to_dict()
                info["algorithms"]["date_range"] = {
                    "start": str(algo.ohlc.index.min()),
                    "end": str(algo.ohlc.index.max()),
                }
            info["algorithms"]["n_files"] = len(algo_ids)

        # Inspeccionar benchmark
        try:
            benchmark = self.load_benchmark()
            info["benchmark"]["monthly_columns"] = list(benchmark.monthly_returns.columns)
            info["benchmark"]["trades_columns"] = list(benchmark.trades.columns)
            info["benchmark"]["trades_sample"] = benchmark.trades.head(3).to_dict()
            info["benchmark"]["n_trades"] = len(benchmark.trades)
            info["benchmark"]["unique_products"] = benchmark.trades["productname"].nunique()
            info["benchmark"]["date_range"] = {
                "start": str(benchmark.trades["dateOpen"].min()),
                "end": str(benchmark.trades["dateClose"].max()),
            }
        except FileNotFoundError as e:
            info["benchmark"]["error"] = str(e)

        return info
