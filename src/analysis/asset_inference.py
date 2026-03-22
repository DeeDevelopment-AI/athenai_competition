"""
Two-Stage Asset Inference Engine.

Infers the underlying asset class of algorithms by comparing their behavior
to known benchmarks.

Stage 1: Fast Pearson correlation screen against ALL benchmarks.
Stage 2: Deep 6-signal analysis on top N candidates.

Signals:
1. Correlation (Pearson + Spearman) - direction and strength
2. Beta - sensitivity to benchmark
3. Drawdown co-movement - do they sink together?
4. Volatility regime match - share vol patterns?
5. Directional agreement - move in same direction?
6. Trading pattern analysis - activity patterns
"""

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class AssetExposure:
    """Single asset exposure detected."""

    name: str
    asset_class: str
    composite: float
    pearson_r: float
    beta: float
    direction: str  # 'long', 'short', 'neutral'
    is_residual: bool = False  # True if detected via residual cascade


@dataclass
class AssetInference:
    """Result of asset inference for an algorithm."""

    predicted_asset: str
    asset_class: str  # forex, equity, commodities, indices, mixed, unknown
    asset_label: str  # Human-readable label
    direction: str  # long, short/inverse, uncorrelated
    confidence: int  # 0-100
    best_composite: float
    significant_exposures: list[AssetExposure] = field(default_factory=list)
    top_matches: list[dict] = field(default_factory=list)
    n_benchmarks_screened: int = 0
    n_candidates_deep: int = 0
    trading_pattern: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)


# =============================================================================
# Constants
# =============================================================================

# Ticker to display name mapping
_TICKER_NAMES = {
    'SPXUSD': 'SP500', 'NASUSD': 'Nasdaq', 'NDXUSD': 'Nasdaq100',
    'NSXUSD': 'Nasdaq', 'DJIUSD': 'DowJones', 'RUTUSD': 'Russell2000',
    'XAUUSD': 'Gold', 'XAGUSD': 'Silver', 'WTIUSD': 'WTI_Oil',
    'BRENTUSD': 'Brent_Oil', 'NATGASUSD': 'NatGas',
    'EURUSD': 'EURUSD', 'GBPUSD': 'GBPUSD', 'USDJPY': 'USDJPY',
    'USDCHF': 'USDCHF', 'AUDUSD': 'AUDUSD', 'USDCAD': 'USDCAD',
    'NZDUSD': 'NZDUSD', 'EURGBP': 'EURGBP', 'EURJPY': 'EURJPY',
    'GBPJPY': 'GBPJPY',
    'BTCUSD': 'Bitcoin', 'ETHUSD': 'Ethereum',
}

# ETFs that replicate indices (skip to avoid double-counting)
_ETF_INDEX_DUPES = {
    'SPY', 'IVV', 'VOO',  # S&P 500
    'QQQ', 'QQQM',  # Nasdaq 100
    'IWM', 'VTWO',  # Russell 2000
    'DIA',  # Dow Jones
    'GLD', 'IAU', 'SGOL',  # Gold
    'SLV', 'SIVR',  # Silver
    'USO', 'BNO',  # Oil
    'UNG',  # Natural Gas
    'FXE', 'FXB', 'FXY', 'FXA', 'FXC',  # Currency ETFs
}

# Known sector/thematic ETFs
_KNOWN_ETFS = {
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLB', 'XLP', 'XLU', 'XLY', 'XLRE', 'XLC',
    'VGT', 'VHT', 'VFH', 'VIS', 'VNQ', 'VDE', 'VAW', 'VCR', 'VOX', 'VPU',
    'SMH', 'SOXX', 'IBB', 'XBI', 'ARKK', 'ARKG', 'ARKF', 'ARKW',
    'IYR', 'ITB', 'XHB', 'KRE', 'KBE', 'OIH', 'XOP', 'GDX', 'GDXJ',
    'TLT', 'TLH', 'IEF', 'SHY', 'BND', 'AGG', 'HYG', 'JNK', 'LQD',
    'EEM', 'EFA', 'VWO', 'VEA', 'IEMG', 'INDA', 'FXI', 'EWZ', 'EWJ',
    'VTI', 'ITOT', 'SCHB',
}

# CME futures root symbol -> (display_name, asset_class)
_FUTURES_META = {
    'ES': ('SP500_Fut', 'indices'),
    'NQ': ('Nasdaq100_Fut', 'indices'),
    'RTY': ('Russell2000_Fut', 'indices'),
    'YM': ('DowJones_Fut', 'indices'),
    'MBT': ('MicroBitcoin_Fut', 'crypto'),
    '6A': ('AUDUSD_Fut', 'forex'), '6B': ('GBPUSD_Fut', 'forex'),
    '6C': ('CADUSD_Fut', 'forex'), '6E': ('EURUSD_Fut', 'forex'),
    '6J': ('JPYUSD_Fut', 'forex'), '6N': ('NZDUSD_Fut', 'forex'),
    '6S': ('CHFUSD_Fut', 'forex'),
    'CL': ('CrudeOil_Fut', 'commodities'), 'BZ': ('Brent_Fut', 'commodities'),
    'NG': ('NatGas_Fut', 'commodities'), 'HO': ('HeatingOil_Fut', 'commodities'),
    'RB': ('Gasoline_Fut', 'commodities'),
    'GC': ('Gold_Fut', 'commodities'), 'SI': ('Silver_Fut', 'commodities'),
    'HG': ('Copper_Fut', 'commodities'), 'PL': ('Platinum_Fut', 'commodities'),
    'ZC': ('Corn_Fut', 'commodities'), 'ZS': ('Soybeans_Fut', 'commodities'),
    'ZL': ('SoybeanOil_Fut', 'commodities'), 'ZM': ('SoybeanMeal_Fut', 'commodities'),
    'ZN': ('10YTNote_Fut', 'rates'), 'KE': ('KCWheat_Fut', 'commodities'),
    'HE': ('LeanHogs_Fut', 'commodities'), 'LE': ('LiveCattle_Fut', 'commodities'),
}

_FUTURES_OUTRIGHT_PATTERN = re.compile(r'^([A-Z0-9]{2,4})([FGHJKMNQUVXZ])(\d{1,2})$')


# =============================================================================
# Helper functions
# =============================================================================

def _safe_float(val, default=0.0):
    """Convert value to float, returning default for NaN/inf."""
    if val is None:
        return default
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return default
    return float(val)


def _guess_asset_class(ticker: str) -> str:
    """Guess asset class from ticker when no subfolder is provided."""
    ticker_up = ticker.upper()
    forex_ccys = ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD', 'SEK', 'NOK']
    if any(ticker_up.startswith(c) or ticker_up.endswith(c) for c in forex_ccys):
        if len(ticker_up) == 6:
            return 'forex'
    commodities = ['XAU', 'XAG', 'WTI', 'BRENT', 'NATGAS', 'COPPER', 'PLAT', 'PALLAD']
    if any(ticker_up.startswith(c) for c in commodities):
        return 'commodities'
    indices = ['SPX', 'NAS', 'NDX', 'NSX', 'DJI', 'RUT', 'DAX', 'FTSE', 'NIK', 'CAC', 'STOXX']
    if any(ticker_up.startswith(c) for c in indices):
        return 'indices'
    crypto = ['BTC', 'ETH', 'XRP', 'SOL', 'BNB']
    if any(ticker_up.startswith(c) for c in crypto):
        return 'crypto'
    return 'other'


def _guess_asset_class_from_name(name: str) -> str:
    """Fallback asset class guess from benchmark display name."""
    n = name.upper()
    if n.startswith('EQ_'):
        return 'equity'
    if n.startswith('FUT_'):
        if any(x in n for x in ['SP500', 'NASDAQ', 'DOW', 'RUSSELL']):
            return 'indices'
        if any(x in n for x in ['CRUDE', 'BRENT', 'NATGAS', 'HEATING', 'GASOLINE']):
            return 'commodities'
        if any(x in n for x in ['GOLD', 'SILVER', 'COPPER', 'PLATINUM']):
            return 'commodities'
        if any(x in n for x in ['CORN', 'SOYBEAN', 'WHEAT']):
            return 'commodities'
        if any(x in n for x in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
            return 'forex'
        return 'futures'
    if any(x in n for x in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
        if not any(x in n for x in ['XAU', 'XAG', 'WTI', 'SPX', 'NAS', 'DJI']):
            return 'forex'
    if any(x in n for x in ['GOLD', 'XAU', 'XAG', 'SILVER', 'WTI', 'OIL', 'BRENT', 'GAS']):
        return 'commodities'
    if any(x in n for x in ['SP500', 'NASDAQ', 'DOW', 'RUSSELL', 'DAX', 'FTSE']):
        return 'indices'
    return 'other'


def _ticker_to_name(ticker: str) -> str:
    """Convert ticker to display name."""
    return _TICKER_NAMES.get(ticker.upper(), ticker.upper())


def _analyze_trading_pattern(algo_daily: pd.DataFrame) -> dict:
    """
    Analyze when the algorithm trades (hours, days of week).

    Different asset classes have different activity patterns:
    - Equities: Mon-Fri, ~14:30-21:00 UTC
    - FX: Mon-Fri, 24h with peaks at London/NY overlap
    - Crypto: 24/7
    """
    idx = algo_daily.index
    if len(idx) < 10:
        return {'pattern': 'unknown'}

    hours = idx.hour
    weekdays = idx.weekday

    has_weekend = (weekdays >= 5).any()
    weekend_pct = round((weekdays >= 5).mean() * 100, 1)

    hour_counts = pd.Series(hours).value_counts().sort_index()
    peak_hours = hour_counts.nlargest(3).index.tolist()

    close = algo_daily['close']
    changes = close.diff().abs()
    active_days_pct = round((changes > 0).mean() * 100, 1)

    daily_ret = close.pct_change().dropna()
    avg_daily_move = round(daily_ret.abs().mean() * 100, 4)

    if weekend_pct > 10:
        pattern = '24/7 (possibly crypto-like)'
    elif active_days_pct < 20:
        pattern = 'Infrequent trader (< 20% active days)'
    elif active_days_pct < 50:
        pattern = 'Moderate frequency'
    else:
        pattern = 'Active daily trader'

    return {
        'pattern': pattern,
        'weekend_activity_pct': weekend_pct,
        'active_days_pct': active_days_pct,
        'avg_daily_move_pct': avg_daily_move,
        'peak_hours': peak_hours,
    }


# =============================================================================
# Benchmark loading functions
# =============================================================================

class BenchmarkLoader:
    """
    Loads benchmarks from structured directory.

    Supported structures:
      benchmarks/                       <- flat: all DAT_ASCII files together
      benchmarks/forex/                 <- subfolders by asset class
      benchmarks/commodities/
      benchmarks/indices/
      benchmarks/sharadar/              <- Sharadar CSV (equities/ETFs)
      benchmarks/futures/               <- CME futures data
    """

    def __init__(self, bench_dir: str | Path):
        self.bench_dir = Path(bench_dir)

    def load_all(self) -> tuple[dict, dict, dict]:
        """
        Load all benchmarks.

        Returns:
            Tuple of (benchmarks, bench_returns, bench_meta) where:
            - benchmarks: {name: daily_df}
            - bench_returns: {name: returns_series}
            - bench_meta: {name: meta_dict}
        """
        benchmarks = {}
        bench_returns = {}
        bench_meta = {}

        # Load DAT_ASCII files
        dat_result = self._load_dat_ascii_benchmarks()
        benchmarks.update(dat_result[0])
        bench_returns.update(dat_result[1])
        bench_meta.update(dat_result[2])

        # Load Sharadar
        sharadar_dir = self.bench_dir / 'sharadar'
        if sharadar_dir.is_dir():
            sha_result = self._load_sharadar_benchmarks(sharadar_dir)
            benchmarks.update(sha_result[0])
            bench_returns.update(sha_result[1])
            bench_meta.update(sha_result[2])

        # Load Futures
        futures_dir = self.bench_dir / 'futures'
        if futures_dir.is_dir():
            fut_result = self._load_futures_benchmarks(futures_dir)
            benchmarks.update(fut_result[0])
            bench_returns.update(fut_result[1])
            bench_meta.update(fut_result[2])

        # Summary
        by_class = {}
        for name, meta in bench_meta.items():
            ac = meta.get('asset_class', 'other')
            by_class[ac] = by_class.get(ac, 0) + 1

        logger.info(f"Total benchmarks loaded: {len(benchmarks)}")
        for ac, cnt in sorted(by_class.items()):
            logger.info(f"  {ac}: {cnt}")

        return benchmarks, bench_returns, bench_meta

    def _load_dat_ascii_benchmarks(self) -> tuple[dict, dict, dict]:
        """Load DAT_ASCII format benchmarks."""
        benchmarks = {}
        bench_returns = {}
        bench_meta = {}

        # Find DAT_ASCII files (flat or subfolders)
        dat_files = sorted(glob.glob(str(self.bench_dir / 'DAT_ASCII_*_M1_*.csv')))
        dat_files += sorted(glob.glob(str(self.bench_dir / '*' / 'DAT_ASCII_*_M1_*.csv')))
        dat_files = sorted(set(dat_files))

        # Group by ticker
        ticker_files = {}
        for f in dat_files:
            basename = os.path.basename(f)
            parts = basename.replace('DAT_ASCII_', '').replace('.csv', '').split('_M1_')
            if len(parts) != 2:
                continue
            ticker = parts[0]

            parent = os.path.basename(os.path.dirname(f))
            if parent == self.bench_dir.name:
                asset_class = _guess_asset_class(ticker)
            else:
                asset_class = parent.lower()

            if ticker not in ticker_files:
                ticker_files[ticker] = {'files': [], 'asset_class': asset_class}
            ticker_files[ticker]['files'].append(f)

        # Load each ticker
        for ticker, info in ticker_files.items():
            files = sorted(info['files'])
            frames = []
            for f in files:
                try:
                    df = pd.read_csv(f, sep=';')
                    df.columns = [c.strip().lower() for c in df.columns]

                    dt_col = None
                    for candidate in ['datetime', 'gmt time', 'date', 'time', 'timestamp']:
                        if candidate in df.columns:
                            dt_col = candidate
                            break
                    if dt_col is None:
                        dt_col = df.columns[0]

                    df = df.rename(columns={dt_col: 'datetime'})
                    parsed = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S', errors='coerce')
                    if parsed.isna().sum() > len(parsed) * 0.5:
                        parsed = pd.to_datetime(df['datetime'], errors='coerce')
                    df['datetime'] = parsed

                    if df['datetime'].isna().all():
                        continue

                    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()

                    for c in ['open', 'high', 'low', 'close']:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors='coerce')

                    if 'close' not in df.columns:
                        continue

                    for c in ['open', 'high', 'low']:
                        if c not in df.columns:
                            df[c] = df['close']

                    frames.append(df)
                except Exception as e:
                    logger.debug(f"Error loading benchmark {f}: {e}")

            if frames:
                combined = pd.concat(frames).sort_index()
                combined = combined[~combined.index.duplicated(keep='first')]
                daily = combined.resample('D').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
                }).dropna()

                name = _ticker_to_name(ticker)
                benchmarks[name] = daily
                bench_returns[name] = daily['close'].pct_change().dropna()
                bench_meta[name] = {
                    'asset_class': info['asset_class'],
                    'ticker': ticker,
                    'source': 'dat_ascii',
                    'start': str(daily.index[0].date()),
                    'end': str(daily.index[-1].date()),
                    'n_days': len(daily),
                }

        return benchmarks, bench_returns, bench_meta

    def _load_sharadar_benchmarks(
        self,
        sharadar_dir: Path,
        top_n: int = 200,
    ) -> tuple[dict, dict, dict]:
        """Load Sharadar equity/ETF benchmarks."""
        benchmarks = {}
        bench_returns = {}
        bench_meta = {}

        csv_files = sorted(glob.glob(str(sharadar_dir / '*.csv')))
        if not csv_files:
            return benchmarks, bench_returns, bench_meta

        for csv_file in csv_files:
            fname = os.path.basename(csv_file)
            try:
                peek = pd.read_csv(csv_file, nrows=5)
            except Exception as e:
                logger.debug(f"Sharadar: ERROR reading {fname}: {e}")
                continue

            cols_lower = [c.lower().strip() for c in peek.columns]

            # Multi-ticker format
            if any(c in cols_lower for c in ['ticker', 'symbol', 'sym', 'name']):
                result = self._load_sharadar_big_csv(csv_file, top_n)
                for name, (daily, meta) in result.items():
                    benchmarks[name] = daily
                    bench_returns[name] = daily['close'].pct_change().dropna()
                    bench_meta[name] = meta
            elif 'close' in cols_lower or 'adj_close' in cols_lower:
                # Single-ticker format
                ticker = os.path.splitext(fname)[0].upper()
                if ticker in _ETF_INDEX_DUPES:
                    continue

                try:
                    df = pd.read_csv(csv_file)
                    df.columns = [c.lower().strip() for c in df.columns]
                    date_col = next((c for c in ['date', 'datetime', 'timestamp']
                                     if c in df.columns), df.columns[0])
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
                    close_col = next((c for c in ['adj_close', 'adjclose', 'close']
                                      if c in df.columns), None)
                    if close_col:
                        daily = pd.DataFrame({'close': pd.to_numeric(df[close_col], errors='coerce')}).dropna()
                        for t in ['open', 'high', 'low']:
                            daily[t] = pd.to_numeric(df[t], errors='coerce') if t in df.columns else daily['close']
                        if len(daily) >= 30:
                            ac = 'etf' if ticker in _KNOWN_ETFS else 'equity'
                            name = f"{'ETF' if ac == 'etf' else 'EQ'}_{ticker}"
                            benchmarks[name] = daily
                            bench_returns[name] = daily['close'].pct_change().dropna()
                            bench_meta[name] = {
                                'asset_class': ac, 'ticker': ticker, 'source': 'sharadar',
                                'start': str(daily.index[0].date()), 'end': str(daily.index[-1].date()),
                                'n_days': len(daily),
                            }
                except Exception:
                    pass

        return benchmarks, bench_returns, bench_meta

    def _load_sharadar_big_csv(
        self,
        filepath: str,
        top_n: int = 200,
    ) -> dict:
        """Load a single big Sharadar CSV, select top N by avg volume."""
        result = {}
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df.columns = [c.lower().strip() for c in df.columns]

            sym_col = next((c for c in ['symbol', 'ticker', 'sym', 'name', 'security',
                                        'fund', 'etf', 'stock', 'code', 'isin']
                            if c in df.columns), None)
            if not sym_col:
                return result

            date_col = next((c for c in ['date', 'datetime', 'timestamp', 'trade_date',
                                         'pricedate', 'price_date']
                             if c in df.columns), None)
            if not date_col:
                return result

            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            close_col = next((c for c in ['adj_close', 'adjclose', 'closeadj', 'close']
                              if c in df.columns), None)
            if not close_col:
                return result

            df = df[~df[sym_col].str.upper().isin(_ETF_INDEX_DUPES)]

            vol_col = next((c for c in ['volume', 'vol'] if c in df.columns), None)
            if vol_col:
                avg_vol = df.groupby(sym_col)[vol_col].mean().sort_values(ascending=False)
                top_tickers = avg_vol.head(top_n).index.tolist()
            else:
                counts = df.groupby(sym_col).size().sort_values(ascending=False)
                top_tickers = counts.head(top_n).index.tolist()

            for ticker in top_tickers:
                tdf = df[df[sym_col] == ticker].set_index(date_col).sort_index()
                daily = pd.DataFrame({'close': pd.to_numeric(tdf[close_col], errors='coerce')}).dropna()
                for t in ['open', 'high', 'low']:
                    daily[t] = pd.to_numeric(tdf[t], errors='coerce') if t in tdf.columns else daily['close']

                if len(daily) >= 60:
                    ticker_up = str(ticker).upper()
                    is_etf = ticker_up in _KNOWN_ETFS
                    ac = 'etf' if is_etf else 'equity'
                    name = f"{'ETF' if is_etf else 'EQ'}_{ticker_up}"
                    result[name] = (daily, {
                        'asset_class': ac, 'ticker': ticker_up, 'source': 'sharadar',
                        'start': str(daily.index[0].date()), 'end': str(daily.index[-1].date()),
                        'n_days': len(daily),
                    })
        except Exception as e:
            logger.debug(f"Sharadar ERROR: {e}")
        return result

    def _load_futures_benchmarks(self, futures_dir: Path) -> tuple[dict, dict, dict]:
        """Load CME/GLBX futures data."""
        benchmarks = {}
        bench_returns = {}
        bench_meta = {}

        csv_files = sorted(glob.glob(str(futures_dir / '*.csv')))
        if not csv_files:
            return benchmarks, bench_returns, bench_meta

        all_frames = []
        for csv_file in csv_files:
            fname = os.path.basename(csv_file)
            if fname == 'metadata.json':
                continue
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                all_frames.append(df)
            except Exception as e:
                logger.debug(f"Futures: ERROR reading {fname}: {e}")

        if not all_frames:
            return benchmarks, bench_returns, bench_meta

        df = pd.concat(all_frames, ignore_index=True)
        df.columns = [c.lower().strip() for c in df.columns]

        if 'ts_event' in df.columns:
            df['datetime'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            df['date'] = df['datetime'].dt.date
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['date'] = df['datetime'].dt.date
        else:
            return benchmarks, bench_returns, bench_meta

        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Filter out spreads
        if 'symbol' in df.columns:
            df = df[~df['symbol'].str.contains(r'[-:]', na=True, regex=True)]

        def extract_root(symbol):
            if pd.isna(symbol):
                return None
            match = _FUTURES_OUTRIGHT_PATTERN.match(str(symbol).upper())
            if match:
                return match.group(1)
            return None

        if 'symbol' in df.columns:
            df['root'] = df['symbol'].apply(extract_root)
            df = df.dropna(subset=['root', 'close'])

            for root in df['root'].unique():
                if root not in _FUTURES_META:
                    continue

                root_df = df[df['root'] == root].copy()

                daily_data = []
                for date, day_df in root_df.groupby('date'):
                    if day_df.empty:
                        continue
                    if 'volume' in day_df.columns and day_df['volume'].notna().any():
                        idx = day_df['volume'].idxmax()
                    else:
                        idx = day_df.index[0]

                    row = day_df.loc[idx]
                    daily_data.append({
                        'date': pd.Timestamp(date),
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                    })

                if len(daily_data) < 30:
                    continue

                daily = pd.DataFrame(daily_data).set_index('date').sort_index()

                if daily['close'].isna().mean() > 0.5:
                    continue
                if (daily['close'] <= 0).any():
                    daily = daily[daily['close'] > 0]
                if len(daily) < 30:
                    continue

                display_name, asset_class = _FUTURES_META[root]
                name = f"FUT_{display_name}"

                benchmarks[name] = daily[['open', 'high', 'low', 'close']]
                bench_returns[name] = daily['close'].pct_change().dropna()
                bench_meta[name] = {
                    'asset_class': asset_class,
                    'ticker': root,
                    'source': 'futures_cme',
                    'start': str(daily.index[0].date()),
                    'end': str(daily.index[-1].date()),
                    'n_days': len(daily),
                }

        return benchmarks, bench_returns, bench_meta


# =============================================================================
# Asset Inference Engine
# =============================================================================

class AssetInferenceEngine:
    """
    Two-stage asset inference engine.

    Stage 1: Fast Pearson correlation screen against ALL benchmarks.
    Stage 2: Deep 6-signal analysis on top N candidates.
    """

    def __init__(
        self,
        benchmarks: Optional[dict] = None,
        bench_returns: Optional[dict] = None,
        bench_meta: Optional[dict] = None,
        top_n_candidates: int = 10,
    ):
        """
        Initialize inference engine.

        Args:
            benchmarks: {name: daily_df} - benchmark OHLC data
            bench_returns: {name: returns_series} - benchmark returns
            bench_meta: {name: meta_dict} - benchmark metadata
            top_n_candidates: Number of candidates to analyze in Stage 2
        """
        self.benchmarks = benchmarks or {}
        self.bench_returns = bench_returns or {}
        self.bench_meta = bench_meta or {}
        self.top_n_candidates = top_n_candidates

    @classmethod
    def from_directory(cls, bench_dir: str | Path, **kwargs) -> "AssetInferenceEngine":
        """
        Create engine by loading benchmarks from directory.

        Args:
            bench_dir: Path to benchmarks directory
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured AssetInferenceEngine
        """
        loader = BenchmarkLoader(bench_dir)
        benchmarks, bench_returns, bench_meta = loader.load_all()
        return cls(
            benchmarks=benchmarks,
            bench_returns=bench_returns,
            bench_meta=bench_meta,
            **kwargs,
        )

    def infer(
        self,
        algo_daily: pd.DataFrame,
        algo_returns: Optional[pd.Series] = None,
    ) -> AssetInference:
        """
        Infer underlying asset for an algorithm.

        Args:
            algo_daily: Daily OHLC DataFrame with 'close' column
            algo_returns: Pre-computed returns (optional, computed if not provided)

        Returns:
            AssetInference with prediction and confidence
        """
        if algo_returns is None:
            algo_returns = algo_daily['close'].pct_change().dropna()

        if len(algo_returns) < 15:
            return self._empty_inference('too_few_data')

        # Stage 1: Fast correlation screen
        fast_scores = self._stage1_fast_screen(algo_returns)

        if not fast_scores:
            result = self._empty_inference('no_benchmark_overlap')
            result.trading_pattern = _analyze_trading_pattern(algo_daily)
            return result

        # Get top candidates
        candidates = sorted(fast_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [name for name, _ in candidates[:self.top_n_candidates]]

        # Stage 2: Deep analysis
        signals = self._stage2_deep_analysis(algo_returns, candidates)

        # Trading pattern
        trading_pattern = _analyze_trading_pattern(algo_daily)

        # Aggregate results
        return self._aggregate_results(
            signals=signals,
            fast_scores=fast_scores,
            candidates=candidates,
            algo_returns=algo_returns,
            trading_pattern=trading_pattern,
        )

    def _stage1_fast_screen(self, algo_returns: pd.Series) -> dict[str, float]:
        """
        Stage 1: Fast Pearson correlation screen.

        Uses vectorized correlation for performance with per-benchmark overlap validation.
        """
        import warnings

        # Build benchmark returns DataFrame for vectorized correlation (cached)
        if not hasattr(self, '_bench_returns_df'):
            self._bench_returns_df = pd.DataFrame(self.bench_returns)

        # Align algo returns with all benchmarks at once
        combined = self._bench_returns_df.join(
            algo_returns.rename('_algo_'), how='inner'
        )

        if len(combined) < 20:
            return {}

        algo_col = combined['_algo_']
        bench_cols = combined.drop(columns=['_algo_'])

        # Check algo variance
        if algo_col.std() < 1e-10:
            return {}

        # Vectorized correlation with all benchmarks (suppress warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = bench_cols.corrwith(algo_col)

        # Filter valid results - check each benchmark has sufficient pairwise overlap
        fast_scores = {}
        for bname, corr in correlations.items():
            if np.isnan(corr):
                continue

            # Get actual pairwise overlap (both algo and bench have values)
            bench_series = bench_cols[bname]
            pairwise_valid = (~bench_series.isna()).sum()

            if pairwise_valid >= 20:
                fast_scores[bname] = abs(corr)

        return fast_scores

    def _stage2_deep_analysis(
        self,
        algo_returns: pd.Series,
        candidates: list[str],
    ) -> dict[str, dict]:
        """
        Stage 2: Deep 6-signal analysis on candidates.

        Matches notebook implementation exactly for consistent results.
        """
        import warnings
        signals = {}

        # NaN-safe helper (matches notebook's _s function)
        def _s(v, default=0):
            """Convert NaN/inf to default."""
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return default
            return float(v)

        for bname in candidates:
            bench_ret = self.bench_returns.get(bname)
            if bench_ret is None:
                continue

            aligned = pd.DataFrame({'algo': algo_returns, 'bench': bench_ret}).dropna()
            n_overlap = len(aligned)
            if n_overlap < 20:
                signals[bname] = {'n_overlap': n_overlap, 'skip': True}
                continue

            a = aligned['algo'].values
            b = aligned['bench'].values

            # Suppress all warnings for numerical operations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Signal 1: Correlation
                    try:
                        pearson_r_val, pearson_p = pearsonr(a, b)
                        pearson_r_val, pearson_p = _s(pearson_r_val), _s(pearson_p, 1.0)
                    except Exception:
                        pearson_r_val, pearson_p = 0, 1.0

                    try:
                        spearman_r_val, _ = spearmanr(a, b)
                        spearman_r_val = _s(spearman_r_val)
                    except Exception:
                        spearman_r_val = 0

                    # Signal 2: Beta
                    cov = np.cov(a, b)[0, 1]
                    var_b = np.var(b)
                    beta = _s(cov / var_b) if var_b > 1e-10 else 0

                    # Signal 3: Drawdown co-movement
                    algo_cum = (1 + aligned['algo']).cumprod()
                    bench_cum = (1 + aligned['bench']).cumprod()
                    algo_dd = (algo_cum - algo_cum.cummax()) / algo_cum.cummax()
                    bench_dd = (bench_cum - bench_cum.cummax()) / bench_cum.cummax()
                    dd_corr = _s(algo_dd.corr(bench_dd)) if len(algo_dd) > 5 else 0

                    # Signal 4: Volatility regime match
                    window = min(20, n_overlap // 3)
                    if window >= 5:
                        algo_rvol = aligned['algo'].rolling(window).std()
                        bench_rvol = aligned['bench'].rolling(window).std()
                        vol_regime_corr = _s(algo_rvol.corr(bench_rvol))
                    else:
                        vol_regime_corr = 0

                    # Signal 5: Directional agreement
                    same_dir = _s(np.mean(np.sign(a) == np.sign(b)))

            # Composite score (matches notebook exactly)
            composite = (
                abs(pearson_r_val) * 0.25 +
                abs(spearman_r_val) * 0.15 +
                min(abs(beta), 2) / 2 * 0.20 +
                max(dd_corr, 0) * 0.15 +
                max(vol_regime_corr, 0) * 0.15 +
                same_dir * 0.10
            )

            signals[bname] = {
                'n_overlap': n_overlap,
                'skip': False,
                'pearson_r': round(pearson_r_val, 4),
                'pearson_p': round(pearson_p, 4),
                'spearman_r': round(spearman_r_val, 4),
                'beta': round(beta, 4),
                'dd_corr': round(dd_corr, 4),
                'vol_regime_corr': round(vol_regime_corr, 4),
                'directional_agreement': round(same_dir, 4),
                'composite_score': round(composite, 4),
            }

        return signals

    def _aggregate_results(
        self,
        signals: dict,
        fast_scores: dict,
        candidates: list[str],
        algo_returns: pd.Series,
        trading_pattern: dict,
    ) -> AssetInference:
        """Aggregate Stage 2 results into final inference."""
        valid_signals = {k: v for k, v in signals.items() if not v.get('skip', False)}

        if not valid_signals:
            result = self._empty_inference('no_valid_signals')
            result.trading_pattern = trading_pattern
            result.n_benchmarks_screened = len(fast_scores)
            return result

        # Find best match
        best_bench = max(valid_signals, key=lambda k: valid_signals[k]['composite_score'])
        best_score = _safe_float(valid_signals[best_bench]['composite_score'])
        best_sig = valid_signals[best_bench]

        # Confidence
        is_significant = best_sig.get('pearson_p', 1) < 0.05
        if best_score > 0.5 and is_significant:
            confidence = min(95, int(best_score * 100 + 15))
        elif best_score > 0.3 and is_significant:
            confidence = int(best_score * 100)
        elif best_score > 0.2:
            confidence = max(10, int(best_score * 80))
        else:
            confidence = max(5, int(best_score * 50))

        # Asset class
        meta = self.bench_meta.get(best_bench, {})
        asset_class = meta.get('asset_class', _guess_asset_class_from_name(best_bench))
        asset_label = f"{best_bench} ({asset_class})"

        # Build exposures list
        significant_exposures = []
        best_beta = best_sig.get('beta', 0)
        best_corr = best_sig.get('pearson_r', 0)

        significant_exposures.append(AssetExposure(
            name=best_bench,
            asset_class=meta.get('asset_class', '?'),
            composite=best_score,
            pearson_r=round(best_corr, 4),
            beta=round(best_beta, 4),
            direction='short' if best_corr < -0.1 else 'long' if best_corr > 0.1 else 'neutral',
            is_residual=False,
        ))

        # Residual cascade for multi-asset detection
        best_bench_ret = self.bench_returns.get(best_bench)
        if best_bench_ret is not None and abs(best_beta) > 0.01:
            residual_exposures = self._compute_residual_exposures(
                algo_returns, best_bench_ret, best_beta, candidates, best_bench, valid_signals
            )
            significant_exposures.extend(residual_exposures)

        # Multi-asset classification
        unique_classes = set(e.asset_class for e in significant_exposures)
        n_exposures = len(significant_exposures)

        if n_exposures >= 2 and len(unique_classes) >= 2:
            asset_class = 'mixed'
            top2 = significant_exposures[:2]
            asset_label = f"Mixed: {top2[0].name} ({top2[0].asset_class}) + {top2[1].name} ({top2[1].asset_class})"
            confidence = max(confidence - 10, 10)
        elif n_exposures >= 2 and significant_exposures[0].asset_class in ('equity', 'etf'):
            top_names = [e.name for e in significant_exposures[:4]]
            asset_label = f"Multi-equity: {', '.join(top_names)}"

        # Opaque?
        if best_score < 0.15 and n_exposures <= 1:
            asset_class = 'unknown'
            asset_label = 'Opaque / Unidentified strategy'
            confidence = max(5, confidence)

        # Direction
        direction = 'long'
        if best_sig.get('pearson_r', 0) < -0.2 and best_sig.get('beta', 0) < -0.1:
            direction = 'short/inverse'
        elif abs(best_sig.get('pearson_r', 0)) < 0.1:
            direction = 'uncorrelated'

        # Top matches
        second_scores = sorted(
            [(k, v['composite_score']) for k, v in valid_signals.items()],
            key=lambda x: x[1], reverse=True
        )
        top_matches = []
        for bname, score in second_scores[:5]:
            m2 = self.bench_meta.get(bname, {})
            top_matches.append({
                'name': bname,
                'asset_class': m2.get('asset_class', '?'),
                'composite': valid_signals[bname]['composite_score'],
                'pearson_r': valid_signals[bname]['pearson_r'],
                'beta': valid_signals[bname].get('beta', 0),
            })

        return AssetInference(
            predicted_asset=best_bench,
            asset_class=asset_class,
            asset_label=asset_label,
            direction=direction,
            confidence=confidence,
            best_composite=best_score,
            significant_exposures=significant_exposures,
            top_matches=top_matches,
            n_benchmarks_screened=len(fast_scores),
            n_candidates_deep=len(candidates),
            trading_pattern=trading_pattern,
            signals=signals,
        )

    def _compute_residual_exposures(
        self,
        algo_returns: pd.Series,
        best_bench_ret: pd.Series,
        best_beta: float,
        candidates: list[str],
        best_bench: str,
        valid_signals: dict,
    ) -> list[AssetExposure]:
        """
        Compute residual exposures for multi-asset detection.

        Matches notebook implementation for residual cascade analysis.
        """
        import warnings
        exposures = []

        residual_df = pd.DataFrame({'algo': algo_returns, 'best': best_bench_ret}).dropna()
        if len(residual_df) < 20:
            return exposures

        residuals = residual_df['algo'].values - best_beta * residual_df['best'].values
        residual_series = pd.Series(residuals, index=residual_df.index)

        for bname in candidates:
            if bname == best_bench:
                continue
            other_ret = self.bench_returns.get(bname)
            if other_ret is None:
                continue

            res_aligned = pd.DataFrame({'res': residual_series, 'other': other_ret}).dropna()
            if len(res_aligned) < 20:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with np.errstate(divide='ignore', invalid='ignore'):
                        r_corr, r_p = pearsonr(res_aligned['res'].values, res_aligned['other'].values)
                        r_cov = np.cov(res_aligned['res'].values, res_aligned['other'].values)[0, 1]
                        r_var = np.var(res_aligned['other'].values)
                        r_beta = r_cov / r_var if r_var > 1e-10 else 0

                # Check for valid correlation
                if np.isnan(r_corr) or np.isnan(r_p):
                    continue

                # Significant residual exposure: |corr| > 0.1 AND p < 0.10
                if abs(r_corr) > 0.1 and r_p < 0.10:
                    m2 = self.bench_meta.get(bname, {})
                    exposures.append(AssetExposure(
                        name=bname,
                        asset_class=m2.get('asset_class', _guess_asset_class_from_name(bname)),
                        composite=valid_signals.get(bname, {}).get('composite_score', 0),
                        pearson_r=round(float(r_corr), 4),
                        beta=round(float(r_beta), 4),
                        direction='short' if r_corr < -0.1 else 'long' if r_corr > 0.1 else 'neutral',
                        is_residual=True,
                    ))
            except Exception:
                pass

        return exposures

    def _empty_inference(self, reason: str) -> AssetInference:
        """Create empty inference result."""
        return AssetInference(
            predicted_asset='unknown',
            asset_class='unknown',
            asset_label=f'Unknown ({reason})',
            direction='unknown',
            confidence=0,
            best_composite=0,
        )
