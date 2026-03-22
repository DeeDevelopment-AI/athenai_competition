"""
Preprocesamiento: reconstrucción de retornos desde OHLC y análisis de trades del benchmark.

Flujo:
1. Algoritmos: OHLC → retornos diarios desde close
2. Benchmark: trades → pesos por producto en cada fecha, equity curve
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .loader import AlgorithmData, BenchmarkData

logger = logging.getLogger(__name__)


@dataclass
class TrimInfo:
    """Information about dead tail trimming."""

    was_trimmed: bool
    original_days: int
    alive_days: int
    dead_days: int
    death_date: Optional[str]
    death_price: Optional[float]


@dataclass
class ProcessedAlgoData:
    """Datos procesados de un algoritmo."""

    algo_id: str
    ohlc: pd.DataFrame  # OHLC original
    returns: pd.Series  # Retornos diarios (desde close)
    equity_curve: pd.Series  # Equity normalizada (base 100)
    trim_info: Optional[TrimInfo] = None  # Info about trimming if applied


@dataclass
class ProcessedBenchmarkData:
    """Datos procesados del benchmark."""

    monthly_returns: pd.Series  # Retornos mensuales
    yearly_returns: pd.Series  # Retornos anuales
    equity_curve: pd.Series  # Equity mensual reconstruida
    trades: pd.DataFrame  # Trades originales
    positions: pd.DataFrame  # Posiciones activas por fecha [fecha x producto]
    weights: pd.DataFrame  # Pesos por producto en cada fecha [fecha x producto]
    products: list[str]  # Lista de productos únicos


def trim_dead_tail(
    daily_closes: pd.Series,
    max_flat_pct: float = 0.10,
) -> tuple[pd.Series, TrimInfo]:
    """
    Detect and trim the 'dead tail' of an algorithm that stopped trading.

    If the last N% of the series has zero returns (price flatlined),
    trim it back to the last movement.

    Args:
        daily_closes: Series of daily close prices.
        max_flat_pct: If the trailing flat segment is more than this fraction
                      of total data, trim it.

    Returns:
        Tuple of (trimmed_closes, TrimInfo).
    """
    if len(daily_closes) < 5:
        return daily_closes, TrimInfo(
            was_trimmed=False,
            original_days=len(daily_closes),
            alive_days=len(daily_closes),
            dead_days=0,
            death_date=None,
            death_price=None,
        )

    returns = daily_closes.pct_change()

    # Find last non-zero return
    nonzero_mask = returns.abs() > 1e-12
    if not nonzero_mask.any():
        # Completely dead - price never moved
        return daily_closes, TrimInfo(
            was_trimmed=False,
            original_days=len(daily_closes),
            alive_days=0,
            dead_days=len(daily_closes),
            death_date=str(daily_closes.index[0].date()) if hasattr(daily_closes.index[0], 'date') else str(daily_closes.index[0]),
            death_price=float(daily_closes.iloc[0]),
        )

    last_move_pos = nonzero_mask.values.nonzero()[0][-1]
    dead_days = len(daily_closes) - last_move_pos - 1
    total_days = len(daily_closes)

    # Only trim if the dead tail is significant
    if dead_days > 5 and (dead_days / total_days) > max_flat_pct:
        # Keep up to 1 day after last movement (to capture final price)
        trim_end = min(last_move_pos + 2, total_days)
        trimmed = daily_closes.iloc[:trim_end]

        death_idx = daily_closes.index[last_move_pos]
        death_date = str(death_idx.date()) if hasattr(death_idx, 'date') else str(death_idx)
        death_price = float(daily_closes.iloc[last_move_pos])

        return trimmed, TrimInfo(
            was_trimmed=True,
            original_days=total_days,
            alive_days=len(trimmed),
            dead_days=dead_days,
            death_date=death_date,
            death_price=death_price,
        )

    return daily_closes, TrimInfo(
        was_trimmed=False,
        original_days=total_days,
        alive_days=total_days,
        dead_days=0,
        death_date=None,
        death_price=None,
    )


class DataPreprocessor:
    """
    Procesa datos crudos para análisis y entrenamiento.

    Para algoritmos:
    - Calcula retornos desde precios close
    - Resamplea a frecuencia diaria si es intradía
    - Opcionalmente recorta dead tails

    Para benchmark:
    - Reconstruye posiciones desde trades
    - Calcula pesos por producto
    - Genera equity curve
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        resample_freq: str = "D",  # Frecuencia de resampleo
    ):
        self.initial_capital = initial_capital
        self.resample_freq = resample_freq

    def process_algorithm(
        self,
        algo_data: AlgorithmData,
        trim_dead: bool = False,
        max_flat_pct: float = 0.10,
    ) -> ProcessedAlgoData:
        """
        Procesa datos OHLC de un algoritmo.

        Args:
            algo_data: Datos crudos del algoritmo.
            trim_dead: If True, trim dead tails where price stopped moving.
            max_flat_pct: Threshold for trimming (only trim if dead portion > this).

        Returns:
            ProcessedAlgoData con retornos y equity.
        """
        ohlc = algo_data.ohlc.copy()

        # Asegurar que tenemos columna close
        if "close" not in ohlc.columns:
            raise ValueError(f"No 'close' column in {algo_data.algo_id}")

        # Resamplear a diario (tomar último close del día)
        if self.resample_freq:
            ohlc_daily = ohlc.resample(self.resample_freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }).dropna()
        else:
            ohlc_daily = ohlc

        # Optionally trim dead tail
        trim_info = None
        close_series = ohlc_daily["close"]

        if trim_dead:
            close_series, trim_info = trim_dead_tail(close_series, max_flat_pct)
            # Trim the OHLC data to match
            ohlc_daily = ohlc_daily.loc[close_series.index]

        # Calcular retornos desde close
        returns = close_series.pct_change().dropna()
        returns.name = algo_data.algo_id

        # Verificar que hay datos suficientes
        if len(returns) == 0:
            raise ValueError(f"No valid returns for {algo_data.algo_id} (empty after processing)")

        # Calcular equity curve normalizada
        equity_curve = self.initial_capital * (1 + returns).cumprod()
        equity_curve = pd.concat([
            pd.Series([self.initial_capital], index=[returns.index[0] - pd.Timedelta(days=1)]),
            equity_curve
        ])
        equity_curve.name = algo_data.algo_id

        return ProcessedAlgoData(
            algo_id=algo_data.algo_id,
            ohlc=ohlc_daily,
            returns=returns,
            equity_curve=equity_curve,
            trim_info=trim_info,
        )

    def process_all_algorithms(
        self,
        algorithms: dict[str, AlgorithmData],
        show_progress: bool = True,
        trim_dead: bool = False,
        max_flat_pct: float = 0.10,
    ) -> dict[str, ProcessedAlgoData]:
        """
        Procesa todos los algoritmos.

        Args:
            algorithms: Dict {algo_id: AlgorithmData}.
            show_progress: Si mostrar progreso.
            trim_dead: If True, trim dead tails.
            max_flat_pct: Threshold for trimming.

        Returns:
            Dict {algo_id: ProcessedAlgoData}.
        """
        processed = {}
        total = len(algorithms)

        for i, (algo_id, algo_data) in enumerate(algorithms.items()):
            try:
                processed[algo_id] = self.process_algorithm(
                    algo_data,
                    trim_dead=trim_dead,
                    max_flat_pct=max_flat_pct,
                )
                if show_progress and (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{total} algorithms")
            except Exception as e:
                logger.error(f"Error processing {algo_id}: {e}")

        logger.info(f"Processed {len(processed)} algorithms")
        return processed

    def process_benchmark(self, benchmark_data: BenchmarkData) -> ProcessedBenchmarkData:
        """
        Procesa datos del benchmark.

        Args:
            benchmark_data: Datos crudos del benchmark.

        Returns:
            ProcessedBenchmarkData con posiciones y pesos.
        """
        trades = self._clean_trades_dates(benchmark_data.trades)

        # Extraer retornos
        monthly_returns = benchmark_data.monthly_returns["monthly_return"]
        yearly_returns = benchmark_data.yearly_returns["yearly_return"]

        # Reconstruir equity desde monthly returns
        equity_curve = self.initial_capital * (1 + monthly_returns).cumprod()

        # Obtener productos únicos
        products = trades["productname"].unique().tolist()

        # Construir matriz de posiciones activas por día
        positions = self._build_positions_matrix(trades)

        # Calcular pesos desde posiciones
        weights = self._calculate_weights_from_trades(trades, positions)

        return ProcessedBenchmarkData(
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
            equity_curve=equity_curve,
            trades=trades,
            positions=positions,
            weights=weights,
            products=products,
        )

    def _build_positions_matrix(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Construye matriz de posiciones activas [fecha x producto].

        Una posición está activa entre dateOpen y dateClose.

        Returns:
            DataFrame con 1 si hay posición, 0 si no.
        """
        if trades.empty:
            raise ValueError("Trades are empty after date cleaning")

        # Obtener rango de fechas
        min_date = trades["dateOpen"].min().normalize()
        max_date = trades["dateClose"].max().normalize()
        all_dates = pd.date_range(min_date, max_date, freq="D")

        products = trades["productname"].unique()

        # Inicializar matriz
        positions = pd.DataFrame(0, index=all_dates, columns=products)

        # Marcar posiciones activas
        for _, trade in trades.iterrows():
            product = trade["productname"]
            start = trade["dateOpen"].normalize()
            end = trade["dateClose"].normalize()

            # Marcar días con posición activa
            mask = (positions.index >= start) & (positions.index <= end)
            positions.loc[mask, product] += 1  # Puede haber múltiples trades

        return positions

    def _clean_trades_dates(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza y valida dateOpen/dateClose para evitar NaT en operaciones.

        - Convierte a datetime (coerce si hay valores invalidos)
        - Elimina trades sin dateOpen
        - Rellena dateClose faltante con la ultima fecha disponible
        - Corrige rangos invertidos (dateClose < dateOpen)
        """
        trades = trades.copy()

        trades["dateOpen"] = pd.to_datetime(trades["dateOpen"], errors="coerce")
        trades["dateClose"] = pd.to_datetime(trades["dateClose"], errors="coerce")

        self._log_trades_quality(trades)

        missing_open = trades["dateOpen"].isna()
        if missing_open.any():
            missing_count = int(missing_open.sum())
            logger.warning("Dropping %d trades with missing dateOpen", missing_count)
            sample_missing = trades.loc[missing_open, ["productname", "dateClose"]].head(5)
            if not sample_missing.empty:
                logger.warning(
                    "Sample trades missing dateOpen (productname, dateClose): %s",
                    sample_missing.to_dict(orient="records"),
                )
            trades = trades.loc[~missing_open].copy()

        date_candidates = pd.concat([trades["dateOpen"], trades["dateClose"]], axis=0).dropna()
        if date_candidates.empty:
            raise ValueError("Trades contain no valid dateOpen/dateClose values")

        max_date = date_candidates.max().normalize()

        missing_close = trades["dateClose"].isna()
        if missing_close.any():
            logger.info(
                "Filling %d missing dateClose values with %s",
                missing_close.sum(),
                max_date.date(),
            )
            trades.loc[missing_close, "dateClose"] = max_date

        inverted = trades["dateClose"] < trades["dateOpen"]
        if inverted.any():
            logger.warning(
                "Found %d trades with dateClose before dateOpen; setting dateClose = dateOpen",
                inverted.sum(),
            )
            trades.loc[inverted, "dateClose"] = trades.loc[inverted, "dateOpen"]

        return trades

    def _log_trades_quality(self, trades: pd.DataFrame) -> None:
        """
        Log basic data quality signals in trades.
        """
        expected_cols = [
            "productname",
            "dateOpen",
            "dateClose",
            "total_invested_amount_EOD",
        ]
        missing_cols = [col for col in expected_cols if col not in trades.columns]
        if missing_cols:
            logger.warning("Trades missing expected columns: %s", missing_cols)

        cols_present = [col for col in expected_cols if col in trades.columns]
        if cols_present:
            null_counts = trades[cols_present].isna().sum()
            null_counts = null_counts[null_counts > 0]
            if not null_counts.empty:
                logger.warning("Null counts in trades: %s", null_counts.to_dict())

        if "productname" in trades.columns:
            series = trades["productname"]
            blank_mask = series.notna() & series.astype(str).str.strip().eq("")
            if blank_mask.any():
                logger.warning(
                    "Found %d trades with blank productname",
                    int(blank_mask.sum()),
                )

        numeric_cols = [
            "total_invested_amount_EOD",
            "equity_EOD",
            "AUM",
            "equity_normalized",
            "volume",
        ]
        for col in numeric_cols:
            if col not in trades.columns:
                continue
            numeric = pd.to_numeric(trades[col], errors="coerce")
            invalid_numeric = numeric.isna() & trades[col].notna()
            if invalid_numeric.any():
                logger.warning(
                    "Non-numeric values in %s: %d",
                    col,
                    int(invalid_numeric.sum()),
                )
            negative = numeric < 0
            if negative.any():
                logger.warning(
                    "Negative values in %s: %d",
                    col,
                    int(negative.sum()),
                )

    def _calculate_weights_from_trades(
        self,
        trades: pd.DataFrame,
        positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcula pesos por producto basado en capital invertido.

        Usa total_invested_amount_EOD para calcular peso relativo.

        Returns:
            DataFrame [fecha x producto] con pesos (suman ~1).
        """
        products = positions.columns.tolist()
        weights = pd.DataFrame(0.0, index=positions.index, columns=products)

        # Para cada día, calcular peso basado en capital invertido
        for date in positions.index:
            # Encontrar trades activos en esta fecha
            active_mask = (
                (trades["dateOpen"].dt.normalize() <= date) &
                (trades["dateClose"].dt.normalize() >= date)
            )
            active_trades = trades[active_mask]

            if len(active_trades) == 0:
                continue

            # Sumar capital por producto
            capital_by_product = active_trades.groupby("productname")["total_invested_amount_EOD"].sum()

            # Normalizar a pesos
            total_capital = capital_by_product.sum()
            if total_capital > 0:
                for product, capital in capital_by_product.items():
                    if product in weights.columns:
                        weights.loc[date, product] = capital / total_capital

        return weights

    def create_returns_matrix(
        self,
        processed_algos: dict[str, ProcessedAlgoData],
        fillna: bool = True,
    ) -> pd.DataFrame:
        """
        Crea matriz de retornos [fecha x algo_id].

        Args:
            processed_algos: Dict de algoritmos procesados.
            fillna: Si rellenar NaN con 0.

        Returns:
            DataFrame con retornos alineados.
        """
        returns_dict = {
            algo_id: data.returns for algo_id, data in processed_algos.items()
        }
        returns_matrix = pd.DataFrame(returns_dict)
        returns_matrix = returns_matrix.sort_index()

        if fillna:
            returns_matrix = returns_matrix.fillna(0)

        logger.info(
            f"Returns matrix: {returns_matrix.shape[0]} days, "
            f"{returns_matrix.shape[1]} products"
        )

        return returns_matrix

    def create_ohlc_panel(
        self,
        processed_algos: dict[str, ProcessedAlgoData],
        field: str = "close",
    ) -> pd.DataFrame:
        """
        Crea panel de un campo OHLC [fecha x algo_id].

        Args:
            processed_algos: Dict de algoritmos procesados.
            field: Campo a extraer ("open", "high", "low", "close").

        Returns:
            DataFrame con valores alineados.
        """
        data_dict = {
            algo_id: data.ohlc[field] for algo_id, data in processed_algos.items()
        }
        panel = pd.DataFrame(data_dict)
        return panel.sort_index()

    def align_with_benchmark(
        self,
        returns_matrix: pd.DataFrame,
        benchmark: ProcessedBenchmarkData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Alinea retornos de algoritmos con pesos del benchmark.

        Args:
            returns_matrix: Matriz de retornos de algoritmos.
            benchmark: Datos procesados del benchmark.

        Returns:
            Tuple (returns_aligned, weights_aligned) con índices comunes.
        """
        # Encontrar productos y fechas comunes
        common_products = list(
            set(returns_matrix.columns) & set(benchmark.weights.columns)
        )

        if not common_products:
            raise ValueError("No common products between algorithms and benchmark")

        # Alinear
        returns_aligned = returns_matrix[common_products]
        weights_aligned = benchmark.weights[common_products]

        # Encontrar fechas comunes
        common_dates = returns_aligned.index.intersection(weights_aligned.index)
        returns_aligned = returns_aligned.loc[common_dates]
        weights_aligned = weights_aligned.loc[common_dates]

        logger.info(
            f"Aligned {len(common_products)} products, "
            f"{len(common_dates)} common dates"
        )

        return returns_aligned, weights_aligned

    def calculate_benchmark_daily_returns(
        self,
        returns_matrix: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Calcula retornos diarios del benchmark desde pesos y retornos.

        Args:
            returns_matrix: Retornos diarios de productos.
            weights: Pesos del benchmark por producto.

        Returns:
            Serie de retornos diarios del benchmark.
        """
        # Normalize timezones to avoid join errors
        returns_copy = returns_matrix.copy()
        weights_copy = weights.copy()

        # Remove timezone info if present
        if hasattr(returns_copy.index, 'tz') and returns_copy.index.tz is not None:
            returns_copy.index = returns_copy.index.tz_localize(None)
        if hasattr(weights_copy.index, 'tz') and weights_copy.index.tz is not None:
            weights_copy.index = weights_copy.index.tz_localize(None)

        # Alinear
        common_dates = returns_copy.index.intersection(weights_copy.index)
        common_products = list(
            set(returns_copy.columns) & set(weights_copy.columns)
        )

        if not common_products or len(common_dates) == 0:
            logger.warning("No common products or dates between returns and weights")
            return pd.Series(dtype=float, name="benchmark")

        returns_aligned = returns_copy.loc[common_dates, common_products]
        weights_aligned = weights_copy.loc[common_dates, common_products]

        # Usar pesos del día anterior (decisión antes de observar retorno)
        weights_lagged = weights_aligned.shift(1).bfill()

        # Retorno del portfolio = suma ponderada de retornos
        benchmark_returns = (returns_aligned * weights_lagged).sum(axis=1)
        benchmark_returns.name = "benchmark"

        return benchmark_returns

    def get_summary_stats(
        self,
        processed_algos: dict[str, ProcessedAlgoData],
    ) -> pd.DataFrame:
        """
        Genera estadísticas resumen de todos los algoritmos (vectorized).

        Args:
            processed_algos: Dict de algoritmos procesados.

        Returns:
            DataFrame con estadísticas por algoritmo.
        """
        if not processed_algos:
            return pd.DataFrame()

        # Build returns matrix for vectorized computation
        algo_ids = list(processed_algos.keys())
        returns_dict = {aid: data.returns for aid, data in processed_algos.items()}
        returns_matrix = pd.DataFrame(returns_dict)

        # Vectorized statistics computation
        stats_df = pd.DataFrame(index=algo_ids)
        stats_df.index.name = 'algo_id'

        # Basic stats using pandas vectorized operations
        stats_df['n_days'] = returns_matrix.count()
        stats_df['start_date'] = returns_matrix.apply(lambda x: x.dropna().index.min() if x.dropna().size > 0 else None)
        stats_df['end_date'] = returns_matrix.apply(lambda x: x.dropna().index.max() if x.dropna().size > 0 else None)
        stats_df['mean_return_daily'] = returns_matrix.mean()
        stats_df['std_return_daily'] = returns_matrix.std()
        stats_df['annualized_return'] = stats_df['mean_return_daily'] * 252
        stats_df['annualized_volatility'] = stats_df['std_return_daily'] * np.sqrt(252)

        # Sharpe ratio (vectorized with safe division)
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe = (stats_df['mean_return_daily'] / stats_df['std_return_daily']) * np.sqrt(252)
            stats_df['sharpe'] = sharpe.fillna(0).replace([np.inf, -np.inf], 0)

        stats_df['min_return'] = returns_matrix.min()
        stats_df['max_return'] = returns_matrix.max()
        stats_df['skew'] = returns_matrix.skew()
        stats_df['kurtosis'] = returns_matrix.kurtosis()

        # Add trim info (must iterate for this)
        trim_info_data = []
        for algo_id in algo_ids:
            data = processed_algos[algo_id]
            if data.trim_info is not None:
                trim_info_data.append({
                    'algo_id': algo_id,
                    'was_trimmed': data.trim_info.was_trimmed,
                    'alive_days': data.trim_info.alive_days,
                    'dead_days': data.trim_info.dead_days,
                    'death_date': data.trim_info.death_date,
                })

        if trim_info_data:
            trim_df = pd.DataFrame(trim_info_data).set_index('algo_id')
            stats_df = stats_df.join(trim_df, how='left')

        return stats_df

    def calculate_benchmark_turnover(
        self,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Calcula turnover diario del benchmark.

        Turnover = sum(|w_t - w_{t-1}|) / 2

        Args:
            weights: DataFrame [fecha x producto] con pesos.

        Returns:
            Serie de turnover diario.
        """
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2
        turnover.name = "turnover"
        return turnover

    def calculate_benchmark_concentration(
        self,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcula métricas de concentración del benchmark.

        Args:
            weights: DataFrame [fecha x producto] con pesos.

        Returns:
            DataFrame con HHI, n_positions, top5_weight.
        """
        concentration = pd.DataFrame(index=weights.index)

        # HHI (Herfindahl-Hirschman Index)
        concentration["hhi"] = (weights ** 2).sum(axis=1)

        # Número de posiciones activas
        concentration["n_positions"] = (weights > 0.001).sum(axis=1)

        # Peso de top 5 posiciones
        def top5_weight(row):
            return row.nlargest(5).sum()
        concentration["top5_weight"] = weights.apply(top5_weight, axis=1)

        # Peso máximo
        concentration["max_weight"] = weights.max(axis=1)

        return concentration

    def get_benchmark_summary(
        self,
        benchmark: "ProcessedBenchmarkData",
        algo_returns: pd.DataFrame = None,
    ) -> dict:
        """
        Genera resumen completo del benchmark.

        Args:
            benchmark: Datos procesados del benchmark.
            algo_returns: Matriz de retornos de algoritmos (opcional).

        Returns:
            Dict con métricas del benchmark.
        """
        summary = {
            "n_trades": len(benchmark.trades),
            "n_products": len(benchmark.products),
            "date_range": {
                "start": str(benchmark.trades["dateOpen"].min().date()),
                "end": str(benchmark.trades["dateClose"].max().date()),
            },
        }

        # Turnover
        turnover = self.calculate_benchmark_turnover(benchmark.weights)
        summary["turnover"] = {
            "mean_daily": float(turnover.mean()),
            "median_daily": float(turnover.median()),
            "annualized": float(turnover.mean() * 252),
        }

        # Concentración
        concentration = self.calculate_benchmark_concentration(benchmark.weights)
        summary["concentration"] = {
            "mean_hhi": float(concentration["hhi"].mean()),
            "mean_n_positions": float(concentration["n_positions"].mean()),
            "mean_top5_weight": float(concentration["top5_weight"].mean()),
            "mean_max_weight": float(concentration["max_weight"].mean()),
        }

        # Retornos mensuales
        if len(benchmark.monthly_returns) > 0:
            summary["monthly_returns"] = {
                "mean": float(benchmark.monthly_returns.mean()),
                "std": float(benchmark.monthly_returns.std()),
                "min": float(benchmark.monthly_returns.min()),
                "max": float(benchmark.monthly_returns.max()),
                "sharpe": float(benchmark.monthly_returns.mean() / benchmark.monthly_returns.std() * np.sqrt(12))
                    if benchmark.monthly_returns.std() > 0 else 0,
            }

        # Retornos reconstruidos
        if algo_returns is not None:
            bench_returns = self.calculate_benchmark_daily_returns(algo_returns, benchmark.weights)
            if len(bench_returns) > 0:
                summary["reconstructed"] = {
                    "n_days": len(bench_returns),
                    "mean_daily": float(bench_returns.mean()),
                    "annualized_return": float(bench_returns.mean() * 252),
                    "annualized_vol": float(bench_returns.std() * np.sqrt(252)),
                    "sharpe": float(bench_returns.mean() / bench_returns.std() * np.sqrt(252))
                        if bench_returns.std() > 0 else 0,
                }

        return summary
