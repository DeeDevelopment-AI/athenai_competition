#!/usr/bin/env python3
"""
=================================================================
ALGORITHM ANALYZER - Pipeline escalable para 14,000+ algoritmos
=================================================================

Uso:
  python3 algo_pipeline.py --algos ./algorithms/ --benchmarks ./benchmarks/ --output results/

Estructura esperada:
  ./algorithms/     -> Un CSV por algoritmo (formato: datetime,open,high,low,close)
  ./benchmarks/     -> CSVs de benchmarks (formato: Gmt time;Open;High;Low;Close;Volume)
                       Nombrados como: DAT_ASCII_SPXUSD_M1_2020.csv, etc.
  ./results/        -> Directorio de salida (se crea automaticamente)

Salida:
  - results/metrics_all.csv          -> Tabla con metricas de todos los algoritmos
  - results/style_analysis_all.csv   -> Exposicion a benchmarks por algoritmo (Sharpe style)
  - results/asset_inference_all.csv  -> Inferencia de activo subyacente (6 senales)
  - results/classification_all.csv   -> Tags de clasificacion
  - results/ranking.csv              -> Ranking final ponderado con asset inference

Motor de inferencia de activo subyacente (6 senales):
  1. Correlacion (Pearson + Spearman) - direccion y fuerza
  2. Beta - sensibilidad al benchmark
  3. Co-movimiento de drawdowns - se hunden juntos?
  4. Regimen de volatilidad - comparten patron de vol?
  5. Acuerdo direccional - se mueven en la misma direccion?
  6. Patron de trading - frecuencia, horas, dias activos
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
import json
import os
import glob
import argparse
import sys
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# DATA LOADING
# ============================================================

def load_algorithm_csv(filepath):
    """Load algorithm CSV -> daily OHLC. Handles multiple date formats robustly."""
    try:
        # --- Skip empty files (header only or truly empty) ---
        file_size = os.path.getsize(filepath)
        if file_size < 50:  # less than 50 bytes = no real data
            return None

        df = pd.read_csv(filepath)

        if len(df) < 2:  # header only or single row
            return None

        # --- Identify the datetime column ---
        dt_col = None
        for candidate in ['datetime', 'Datetime', 'date', 'Date', 'time', 'Time',
                          'timestamp', 'Timestamp', 'Gmt time', 'gmt time']:
            if candidate in df.columns:
                dt_col = candidate
                break
        if dt_col is None:
            # Fall back to first column
            dt_col = df.columns[0]

        # --- Parse dates with multiple fallback strategies ---
        parsed = pd.to_datetime(df[dt_col], errors='coerce')

        # If most values failed, try common non-standard formats
        if parsed.isna().sum() > len(parsed) * 0.5:
            for fmt in ['%Y%m%d %H%M%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M',
                        '%d-%m-%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                        '%d.%m.%Y %H:%M:%S', '%Y%m%d%H%M%S',
                        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                attempt = pd.to_datetime(df[dt_col], format=fmt, errors='coerce')
                if attempt.isna().sum() < len(attempt) * 0.5:
                    parsed = attempt
                    break

        # Drop rows where date parsing still failed
        df['_dt'] = parsed
        df = df.dropna(subset=['_dt'])

        if len(df) < 10:
            return None

        df = df.sort_values('_dt').set_index('_dt')
        df.index.name = 'datetime'

        # --- Identify OHLC columns (case-insensitive) ---
        col_map = {}
        for target in ['open', 'high', 'low', 'close']:
            for col in df.columns:
                if col.lower() == target:
                    col_map[target] = col
                    break
        if 'close' not in col_map:
            return None

        # Rename to standard names
        rename = {v: k for k, v in col_map.items()}
        df = df.rename(columns=rename)

        # Ensure numeric
        for c in ['open', 'high', 'low', 'close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Resample to daily
        agg = {}
        if 'open' in df.columns:
            agg['open'] = 'first'
        if 'high' in df.columns:
            agg['high'] = 'max'
        if 'low' in df.columns:
            agg['low'] = 'min'
        agg['close'] = 'last'

        daily = df.resample('D').agg(agg).dropna(subset=['close'])

        # Fill missing OHLC from close if needed
        for c in ['open', 'high', 'low']:
            if c not in daily.columns:
                daily[c] = daily['close']

        return daily

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None


def load_all_benchmarks(bench_dir):
    """
    Auto-discover and load benchmarks from a structured directory.

    Supported structures:
      benchmarks/                       <- flat: all DAT_ASCII files together
      benchmarks/forex/                 <- subfolders by asset class
      benchmarks/commodities/
      benchmarks/indices/
      benchmarks/etfs/
      benchmarks/sharadar/              <- Sharadar CSV (special loader)

    DAT_ASCII files are auto-named from the ticker in the filename:
      DAT_ASCII_EURUSD_M1_2020.csv -> ticker=EURUSD, asset_class from folder name

    Returns:
      benchmarks:    dict {name: daily_df}
      bench_returns: dict {name: returns_series}
      bench_meta:    dict {name: {asset_class, ticker, source}}
    """
    benchmarks = {}
    bench_returns = {}
    bench_meta = {}

    # --- Discover DAT_ASCII files (flat or subfolders) ---
    dat_files = sorted(glob.glob(os.path.join(bench_dir, 'DAT_ASCII_*_M1_*.csv')))
    dat_files += sorted(glob.glob(os.path.join(bench_dir, '*', 'DAT_ASCII_*_M1_*.csv')))
    # Deduplicate
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
        if parent == os.path.basename(bench_dir):
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
                # Normalize column names to avoid case/whitespace mismatches.
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
                    raise ValueError("No valid datetimes after parsing")

                df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()

                # Ensure OHLC columns exist and are numeric.
                for c in ['open', 'high', 'low', 'close']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')

                if 'close' not in df.columns:
                    raise ValueError("Missing close column after normalization")

                for c in ['open', 'high', 'low']:
                    if c not in df.columns:
                        df[c] = df['close']

                frames.append(df)
            except Exception as e:
                print(f"  ERROR loading benchmark {f}: {e}")

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

    # --- Discover Sharadar files ---
    sharadar_dir = os.path.join(bench_dir, 'sharadar')
    if os.path.isdir(sharadar_dir):
        sha_loaded = _load_sharadar_benchmarks(sharadar_dir)
        for name, (daily, meta) in sha_loaded.items():
            benchmarks[name] = daily
            bench_returns[name] = daily['close'].pct_change().dropna()
            bench_meta[name] = meta

    # --- Discover Futures files ---
    futures_dir = os.path.join(bench_dir, 'futures')
    if os.path.isdir(futures_dir):
        fut_loaded = _load_futures_benchmarks(futures_dir)
        for name, (daily, meta) in fut_loaded.items():
            benchmarks[name] = daily
            bench_returns[name] = daily['close'].pct_change().dropna()
            bench_meta[name] = meta

    # Summary
    by_class = {}
    for name, meta in bench_meta.items():
        ac = meta['asset_class']
        by_class[ac] = by_class.get(ac, 0) + 1

    print(f"  Total benchmarks loaded: {len(benchmarks)}")
    for ac, cnt in sorted(by_class.items()):
        names = [n for n, m in bench_meta.items() if m['asset_class'] == ac]
        sample = ', '.join(names[:5])
        if len(names) > 5:
            sample += f', ... (+{len(names) - 5} more)'
        print(f"    {ac:<15} {cnt:>4}  [{sample}]")

    return benchmarks, bench_returns, bench_meta


def _guess_asset_class(ticker):
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


_TICKER_NAMES = {
    'SPXUSD': 'SP500', 'NASUSD': 'Nasdaq', 'NDXUSD': 'Nasdaq100',
    'NSXUSD': 'Nasdaq',
    'DJIUSD': 'DowJones', 'RUTUSD': 'Russell2000',
    'XAUUSD': 'Gold', 'XAGUSD': 'Silver', 'WTIUSD': 'WTI_Oil',
    'BRENTUSD': 'Brent_Oil', 'NATGASUSD': 'NatGas',
    'EURUSD': 'EURUSD', 'GBPUSD': 'GBPUSD', 'USDJPY': 'USDJPY',
    'USDCHF': 'USDCHF', 'AUDUSD': 'AUDUSD', 'USDCAD': 'USDCAD',
    'NZDUSD': 'NZDUSD', 'EURGBP': 'EURGBP', 'EURJPY': 'EURJPY',
    'GBPJPY': 'GBPJPY',
    'BTCUSD': 'Bitcoin', 'ETHUSD': 'Ethereum',
}


def _ticker_to_name(ticker):
    return _TICKER_NAMES.get(ticker.upper(), ticker.upper())


def _load_sharadar_benchmarks(sharadar_dir):
    """
    Load Sharadar daily price data. Supports:
      Format A - One big CSV with 'ticker'/'symbol' column -> top N by volume
      Format B - One CSV per ticker (AAPL.csv, etc.)

    Uses adj_close when available (adjusts for splits/dividends).
    Deduplicates ETFs that replicate indices already in the benchmark universe.
    Returns: dict {name: (daily_df, meta_dict)}
    """
    result = {}
    big_csvs = sorted(glob.glob(os.path.join(sharadar_dir, '*.csv')))
    if not big_csvs:
        return result

    # Try each CSV individually — they may have different column names
    for csv_file in big_csvs:
        fname = os.path.basename(csv_file)
        try:
            peek = pd.read_csv(csv_file, nrows=5)
        except Exception as e:
            print(f"  Sharadar: ERROR reading {fname}: {e}")
            continue

        cols_lower = [c.lower().strip() for c in peek.columns]
        print(f"  Sharadar: {fname} columns: {list(peek.columns)}")

        # Format A: big file with ticker/symbol column
        if any(c in cols_lower for c in ['ticker', 'symbol', 'sym', 'name']):
            print(f"  Sharadar: {fname} -> multi-ticker format")
            partial = _load_sharadar_big_csv(csv_file)
            result.update(partial)
        elif 'close' in cols_lower or 'adj_close' in cols_lower:
            # Format B: single-ticker file
            ticker = os.path.splitext(fname)[0].upper()
            if ticker in _ETF_INDEX_DUPES:
                continue
            print(f"  Sharadar: {fname} -> single-ticker format ({ticker})")
            # Load as individual ticker...
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
                        result[name] = (daily, {
                            'asset_class': ac, 'ticker': ticker, 'source': 'sharadar',
                            'start': str(daily.index[0].date()), 'end': str(daily.index[-1].date()),
                            'n_days': len(daily),
                        })
            except:
                pass
        else:
            print(f"  Sharadar: {fname} -> UNKNOWN format, skipping (columns: {list(peek.columns)[:8]})")

    print(f"  Sharadar: total loaded: {len(result)} tickers")
    return result


# ETFs that directly replicate indices we already have via DAT_ASCII
# These get skipped to avoid double-counting
_ETF_INDEX_DUPES = {
    'SPY', 'IVV', 'VOO',  # S&P 500
    'QQQ', 'QQQM',  # Nasdaq 100
    'IWM', 'VTWO',  # Russell 2000
    'DIA',  # Dow Jones
    'GLD', 'IAU', 'SGOL',  # Gold
    'SLV', 'SIVR',  # Silver
    'USO', 'BNO',  # Oil
    'UNG',  # Natural Gas
    'FXE',  # EUR/USD
    'FXB',  # GBP/USD
    'FXY',  # JPY
    'FXA',  # AUD
    'FXC',  # CAD
}

# Known sector/thematic ETFs (classified as 'etf' not 'equity')
_KNOWN_ETFS = {
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLB', 'XLP', 'XLU', 'XLY', 'XLRE',
    'XLC',  # sector SPDRs
    'VGT', 'VHT', 'VFH', 'VIS', 'VNQ', 'VDE', 'VAW', 'VCR', 'VOX', 'VPU',
    'SMH', 'SOXX', 'IBB', 'XBI', 'ARKK', 'ARKG', 'ARKF', 'ARKW',
    'IYR', 'ITB', 'XHB', 'KRE', 'KBE', 'OIH', 'XOP', 'GDX', 'GDXJ',
    'TLT', 'TLH', 'IEF', 'SHY', 'BND', 'AGG', 'HYG', 'JNK', 'LQD',
    'EEM', 'EFA', 'VWO', 'VEA', 'IEMG', 'INDA', 'FXI', 'EWZ', 'EWJ',
    'VTI', 'ITOT', 'SCHB',  # total market
}


# ============================================================
# FUTURES BENCHMARKS (CME/GLBX)
# ============================================================

# CME futures root symbol -> (display_name, asset_class)
_FUTURES_META = {
    # Equity indices
    'ES': ('SP500_Fut', 'indices'),
    'NQ': ('Nasdaq100_Fut', 'indices'),
    'RTY': ('Russell2000_Fut', 'indices'),
    'YM': ('DowJones_Fut', 'indices'),
    'MBT': ('MicroBitcoin_Fut', 'crypto'),
    # FX
    '6A': ('AUDUSD_Fut', 'forex'),
    '6B': ('GBPUSD_Fut', 'forex'),
    '6C': ('CADUSD_Fut', 'forex'),
    '6E': ('EURUSD_Fut', 'forex'),
    '6J': ('JPYUSD_Fut', 'forex'),
    '6N': ('NZDUSD_Fut', 'forex'),
    '6S': ('CHFUSD_Fut', 'forex'),
    # Energy
    'CL': ('CrudeOil_Fut', 'commodities'),
    'BZ': ('Brent_Fut', 'commodities'),
    'NG': ('NatGas_Fut', 'commodities'),
    'HO': ('HeatingOil_Fut', 'commodities'),
    'RB': ('Gasoline_Fut', 'commodities'),
    # Metals
    'GC': ('Gold_Fut', 'commodities'),
    'SI': ('Silver_Fut', 'commodities'),
    'HG': ('Copper_Fut', 'commodities'),
    'PL': ('Platinum_Fut', 'commodities'),
    # Grains
    'ZC': ('Corn_Fut', 'commodities'),
    'ZS': ('Soybeans_Fut', 'commodities'),
    'ZL': ('SoybeanOil_Fut', 'commodities'),
    'ZM': ('SoybeanMeal_Fut', 'commodities'),
    'ZN': ('10YTNote_Fut', 'rates'),
    'KE': ('KCWheat_Fut', 'commodities'),
    # Livestock
    'HE': ('LeanHogs_Fut', 'commodities'),
    'LE': ('LiveCattle_Fut', 'commodities'),
}

# Regex for outright futures contracts (e.g., ESH0, CLG0, GCZ4)
# Pattern: ROOT + MONTH_CODE + YEAR_DIGIT(s)
_FUTURES_OUTRIGHT_PATTERN = re.compile(
    r'^([A-Z0-9]{2,4})([FGHJKMNQUVXZ])(\d{1,2})$'
)


def _load_futures_benchmarks(futures_dir):
    """
    Load CME/GLBX futures data from OHLCV-1D CSV files.

    Expects:
      - A CSV file with columns: ts_event, rtype, publisher_id, instrument_id,
        open, high, low, close, volume, symbol
      - ts_event is nanoseconds since epoch
      - symbol contains contract identifiers like 'ESH0', 'CLG0', etc.
      - Optional metadata.json with query info

    Returns: dict {name: (daily_df, meta_dict)}
    """
    result = {}
    import re

    # Find CSV files in futures directory
    csv_files = sorted(glob.glob(os.path.join(futures_dir, '*.csv')))
    if not csv_files:
        return result

    # Try to load metadata
    meta_file = os.path.join(futures_dir, 'metadata.json')
    query_meta = {}
    if os.path.isfile(meta_file):
        try:
            with open(meta_file, 'r') as f:
                query_meta = json.load(f)
            print(f"  Futures: loaded metadata from {os.path.basename(meta_file)}")
        except Exception as e:
            print(f"  Futures: failed to load metadata: {e}")

    # Load all CSV files
    all_frames = []
    for csv_file in csv_files:
        fname = os.path.basename(csv_file)
        if fname == 'metadata.json':
            continue
        print(f"  Futures: reading {fname}...")
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            all_frames.append(df)
        except Exception as e:
            print(f"  Futures: ERROR reading {fname}: {e}")

    if not all_frames:
        return result

    # Combine all frames
    df = pd.concat(all_frames, ignore_index=True)
    print(f"  Futures: total rows loaded: {len(df):,}")

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse timestamps (nanoseconds -> datetime)
    if 'ts_event' in df.columns:
        df['datetime'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)
        df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone
        df['date'] = df['datetime'].dt.date
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['date'] = df['datetime'].dt.date
    else:
        print(f"  Futures: no timestamp column found")
        return result

    # Ensure OHLCV columns are numeric
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Filter for outright contracts only (no spreads, butterflies, etc.)
    # Outrights have simple symbols like ESH0, CLG0, GCZ4
    # Spreads contain '-' or ':' in the symbol
    df = df[~df['symbol'].str.contains(r'[-:]', na=True, regex=True)]
    print(f"  Futures: rows after filtering spreads: {len(df):,}")

    # Extract root symbol from contract symbol
    def extract_root(symbol):
        if pd.isna(symbol):
            return None
        match = _FUTURES_OUTRIGHT_PATTERN.match(str(symbol).upper())
        if match:
            return match.group(1)
        return None

    df['root'] = df['symbol'].apply(extract_root)
    df = df.dropna(subset=['root', 'close'])

    # Get unique roots
    roots = df['root'].unique()
    print(f"  Futures: found {len(roots)} unique root symbols")

    # For each root, create a continuous series using the most liquid contract per day
    for root in sorted(roots):
        if root not in _FUTURES_META:
            # Skip unknown roots
            continue

        root_df = df[df['root'] == root].copy()

        # Group by date and select the contract with highest volume per day
        # This approximates a continuous front-month series
        daily_data = []
        for date, day_df in root_df.groupby('date'):
            if day_df.empty:
                continue
            # Select contract with highest volume
            if 'volume' in day_df.columns and day_df['volume'].notna().any():
                idx = day_df['volume'].idxmax()
            else:
                # Fallback: first contract alphabetically (usually front month)
                idx = day_df.index[0]

            row = day_df.loc[idx]
            daily_data.append({
                'date': pd.Timestamp(date),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row.get('volume', 0),
                'contract': row['symbol'],
            })

        if len(daily_data) < 30:
            continue

        daily = pd.DataFrame(daily_data).set_index('date').sort_index()

        # Skip if price data looks invalid
        if daily['close'].isna().mean() > 0.5:
            continue
        if (daily['close'] <= 0).any():
            daily = daily[daily['close'] > 0]
        if len(daily) < 30:
            continue

        display_name, asset_class = _FUTURES_META[root]
        name = f"FUT_{display_name}"

        result[name] = (daily[['open', 'high', 'low', 'close', 'volume']], {
            'asset_class': asset_class,
            'ticker': root,
            'source': 'futures_cme',
            'start': str(daily.index[0].date()),
            'end': str(daily.index[-1].date()),
            'n_days': len(daily),
            'n_contracts_rolled': daily['contract'].nunique() if 'contract' in daily.columns else 0,
        })

    print(f"  Futures: loaded {len(result)} continuous benchmarks")
    return result


def _load_sharadar_big_csv(filepath, top_n=200):
    """Load a single big Sharadar CSV, select top N by avg volume."""
    result = {}
    try:
        fname = os.path.basename(filepath)
        print(f"  Sharadar: reading {fname}...")
        df = pd.read_csv(filepath, low_memory=False)
        original_cols = list(df.columns)
        df.columns = [c.lower().strip() for c in df.columns]

        # Handle many possible symbol/ticker column names
        sym_col = next((c for c in ['symbol', 'ticker', 'sym', 'name', 'security',
                                    'fund', 'etf', 'stock', 'code', 'isin']
                        if c in df.columns), None)
        if not sym_col:
            print(f"  Sharadar: no symbol/ticker column found in {fname}")
            print(f"  Sharadar: available columns: {original_cols}")
            return result

        date_col = next((c for c in ['date', 'datetime', 'timestamp', 'trade_date',
                                     'pricedate', 'price_date']
                         if c in df.columns), None)
        if not date_col:
            print(f"  Sharadar: no date column found in {fname}")
            print(f"  Sharadar: available columns: {original_cols}")
            return result
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        # Prefer adj_close for equities (adjusts splits + dividends)
        close_col = next((c for c in ['adj_close', 'adjclose', 'closeadj', 'close']
                          if c in df.columns), None)
        if not close_col:
            print(f"  Sharadar: no close/adj_close column found")
            return result

        # Remove ETFs that duplicate indices
        df = df[~df[sym_col].str.upper().isin(_ETF_INDEX_DUPES)]

        # Volume column for liquidity filter
        vol_col = next((c for c in ['volume', 'vol'] if c in df.columns), None)
        if vol_col:
            avg_vol = df.groupby(sym_col)[vol_col].mean().sort_values(ascending=False)
            top_tickers = avg_vol.head(top_n).index.tolist()
        else:
            counts = df.groupby(sym_col).size().sort_values(ascending=False)
            top_tickers = counts.head(top_n).index.tolist()

        print(f"  Sharadar: selecting top {len(top_tickers)} from "
              f"{df[sym_col].nunique()} tickers (after dedup)")

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
                    'asset_class': ac, 'ticker': ticker_up,
                    'source': 'sharadar',
                    'start': str(daily.index[0].date()),
                    'end': str(daily.index[-1].date()),
                    'n_days': len(daily),
                })

        print(f"  Sharadar: loaded {len(result)} benchmarks "
              f"({sum(1 for _, m in result.values() if m['asset_class'] == 'equity')} equities, "
              f"{sum(1 for _, m in result.values() if m['asset_class'] == 'etf')} ETFs)")
    except Exception as e:
        print(f"  Sharadar ERROR: {e}")
    return result


# ============================================================
# METRICS
# ============================================================

def trim_dead_tail(daily_closes, max_flat_pct=0.10):
    """
    Detect and trim the 'dead tail' of an algorithm that stopped trading.

    If the last N% of the series has zero returns (price flatlined),
    trim it back to the last movement. Returns the trimmed series
    and metadata about what was trimmed.

    Args:
        daily_closes: pd.Series of daily close prices
        max_flat_pct: if the trailing flat segment is more than this fraction
                      of total data, trim it

    Returns:
        (trimmed_closes, trim_info) where trim_info is a dict with:
          - was_trimmed: bool
          - original_days: int
          - alive_days: int
          - dead_days: int
          - death_date: str or None
          - death_price: float or None
    """
    if len(daily_closes) < 5:
        return daily_closes, {'was_trimmed': False, 'original_days': len(daily_closes),
                              'alive_days': len(daily_closes), 'dead_days': 0,
                              'death_date': None, 'death_price': None}

    returns = daily_closes.pct_change()

    # Find last non-zero return
    nonzero_mask = returns.abs() > 1e-12
    if not nonzero_mask.any():
        # Completely dead — never moved
        return daily_closes, {'was_trimmed': False, 'original_days': len(daily_closes),
                              'alive_days': 0, 'dead_days': len(daily_closes),
                              'death_date': str(daily_closes.index[0].date()),
                              'death_price': float(daily_closes.iloc[0])}

    last_move_pos = nonzero_mask.values.nonzero()[0][-1]
    dead_days = len(daily_closes) - last_move_pos - 1
    total_days = len(daily_closes)

    # Only trim if the dead tail is significant
    if dead_days > 5 and (dead_days / total_days) > max_flat_pct:
        # Keep up to 1 day after last movement (to capture final price)
        trim_end = min(last_move_pos + 2, total_days)
        trimmed = daily_closes.iloc[:trim_end]
        death_date = str(daily_closes.index[last_move_pos].date())
        death_price = float(daily_closes.iloc[last_move_pos])

        return trimmed, {
            'was_trimmed': True,
            'original_days': total_days,
            'alive_days': len(trimmed),
            'dead_days': dead_days,
            'death_date': death_date,
            'death_price': death_price,
        }

    return daily_closes, {'was_trimmed': False, 'original_days': total_days,
                          'alive_days': total_days, 'dead_days': 0,
                          'death_date': None, 'death_price': None}


def compute_metrics(daily_closes, name="algo"):
    """Compute full risk/return metrics. Auto-trims dead tails."""

    # Trim dead tail first
    trimmed_closes, trim_info = trim_dead_tail(daily_closes)

    returns = trimmed_closes.pct_change().dropna()

    if len(returns) < 10:
        return None, trim_info

    # Skip fully dead algorithms (price never moves even after trim attempt)
    if returns.std() < 1e-10 or (returns == 0).mean() > 0.95:
        return None, trim_info

    n_days = len(returns)
    total_return = (trimmed_closes.iloc[-1] / trimmed_closes.iloc[0]) - 1
    date_range_years = (trimmed_closes.index[-1] - trimmed_closes.index[0]).days / 365.25

    if date_range_years < 0.02:
        return None, trim_info

    ann_ret = (1 + total_return) ** (1 / date_range_years) - 1
    ann_vol = returns.std() * np.sqrt(252)

    rf = 0.03
    rf_daily = (1 + rf) ** (1 / 252) - 1
    excess = returns - rf_daily
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    downside = returns[returns < rf_daily]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (ann_ret - rf) / down_vol if down_vol > 1e-6 else 0

    cum = (1 + returns).cumprod()
    dd_series = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd_series.min() if not dd_series.isna().all() else 0
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-4 else 0

    # Guard against NaN in higher-order stats
    skew = returns.skew()
    kurt = returns.kurtosis()
    win_rate = (returns > 0).mean()

    def safe_round(val, decimals=3):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return round(float(val), decimals)

    result = {
        'name': name,
        'start_date': trimmed_closes.index[0].strftime('%Y-%m-%d'),
        'end_date': trimmed_closes.index[-1].strftime('%Y-%m-%d'),
        'n_trading_days': n_days,
        'duration_years': round(date_range_years, 2),
        'total_return_pct': safe_round(total_return * 100, 2),
        'annualized_return_pct': safe_round(ann_ret * 100, 2),
        'annualized_volatility_pct': safe_round(ann_vol * 100, 2),
        'sharpe_ratio': safe_round(sharpe),
        'sortino_ratio': safe_round(sortino),
        'max_drawdown_pct': safe_round(max_dd * 100, 2),
        'calmar_ratio': safe_round(calmar),
        'win_rate_pct': safe_round(win_rate * 100, 2),
        'skewness': safe_round(skew),
        'kurtosis': safe_round(kurt),
        'var_95_pct': safe_round(np.percentile(returns, 5) * 100, 4),
        'current_value': safe_round(trimmed_closes.iloc[-1], 2),
        # Trim info
        'was_trimmed': trim_info['was_trimmed'],
        'alive_days': trim_info['alive_days'],
        'dead_days': trim_info['dead_days'],
        'death_date': trim_info['death_date'],
    }

    return result, trim_info


# ============================================================
# STYLE ANALYSIS (Sharpe Returns-Based)
# ============================================================

def sharpe_style_analysis(algo_returns, bench_returns_dict):
    """Returns-based style analysis (Sharpe method)."""
    aligned = pd.DataFrame({'algo': algo_returns})
    for name, br in bench_returns_dict.items():
        aligned[name] = br
    aligned = aligned.dropna()

    if len(aligned) < 30:
        return {'error': 'insufficient_overlap', 'n_overlap': len(aligned)}

    y = aligned['algo'].values
    bench_names = [c for c in aligned.columns if c != 'algo']
    X = aligned[bench_names].values
    n = len(bench_names)

    def obj(w):
        return np.sum((y - X @ w) ** 2)

    res = minimize(obj, np.ones(n) / (n + 1), method='SLSQP',
                   bounds=[(0, 1)] * n,
                   constraints=[{'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)}])

    if not res.success:
        return {'error': 'optimization_failed'}

    w = res.x
    resid = y - X @ w
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    exposures = {name: round(float(w[i]) * 100, 2) for i, name in enumerate(bench_names)}

    return {
        'exposures': exposures,
        'unexplained_pct': round((1 - sum(w)) * 100, 2),
        'r_squared': round(r2, 4),
        'alpha_annual_pct': round(np.mean(resid) * 252 * 100, 2),
        'tracking_error_pct': round(np.std(resid) * np.sqrt(252) * 100, 2),
        'n_overlap': len(aligned),
    }


# ============================================================
# TWO-STAGE ASSET INFERENCE ENGINE
# Stage 1: Fast Pearson screen against ALL benchmarks
# Stage 2: Deep 6-signal analysis on top N candidates
# ============================================================

def infer_underlying_asset(algo_daily, bench_daily_dict, bench_returns_dict,
                           bench_meta=None, top_n_candidates=10):
    """
    Two-stage inference of underlying asset class.

    Stage 1 (fast): Pearson correlation of returns against ALL benchmarks.
                    Keep top N candidates by |correlation|.
    Stage 2 (deep): Full 6-signal analysis only on candidates.

    Returns a dict with predicted_asset, confidence, signals, etc.
    """
    algo_close = algo_daily['close']
    algo_ret = algo_close.pct_change().dropna()

    if len(algo_ret) < 15:
        return _empty_inference('too_few_data')

    bench_meta = bench_meta or {}

    # ---- STAGE 1: Fast screen ----
    fast_scores = {}
    for bname, bench_ret in bench_returns_dict.items():
        aligned = pd.DataFrame({'a': algo_ret, 'b': bench_ret}).dropna()
        if len(aligned) < 20:
            continue
        try:
            r, _ = pearsonr(aligned['a'].values, aligned['b'].values)
            if not np.isnan(r):
                fast_scores[bname] = abs(r)
        except:
            pass

    if not fast_scores:
        result = _empty_inference('no_benchmark_overlap')
        result['trading_pattern'] = _analyze_trading_pattern(algo_daily)
        return result

    # Keep top N by |correlation|
    candidates = sorted(fast_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [name for name, _ in candidates[:top_n_candidates]]

    # ---- STAGE 2: Deep analysis on candidates only ----
    signals = {}
    for bname in candidates:
        bench_ret = bench_returns_dict[bname]
        bench_daily = bench_daily_dict.get(bname)

        aligned = pd.DataFrame({'algo': algo_ret, 'bench': bench_ret}).dropna()
        n_overlap = len(aligned)
        if n_overlap < 20:
            signals[bname] = {'n_overlap': n_overlap, 'skip': True}
            continue

        a = aligned['algo'].values
        b = aligned['bench'].values

        # NaN-safe helper
        def _s(v, default=0):
            """Convert NaN/inf to default."""
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return default
            return float(v)

        # Signal 1: Correlation
        pearson_r_val, pearson_p = pearsonr(a, b)
        pearson_r_val, pearson_p = _s(pearson_r_val), _s(pearson_p, 1.0)
        try:
            spearman_r_val, spearman_p = spearmanr(a, b)
            spearman_r_val = _s(spearman_r_val)
        except:
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

        # Signal 5: Directional co-movement
        same_dir = _s(np.mean(np.sign(a) == np.sign(b)))

        # Composite score (all inputs guaranteed non-NaN)
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

    # Signal 6: Trading pattern fingerprint
    trading_pattern = _analyze_trading_pattern(algo_daily)

    # ---- Aggregate: predict asset ----
    valid_signals = {k: v for k, v in signals.items() if not v.get('skip', False)}

    if not valid_signals:
        result = _empty_inference('no_valid_signals')
        result['trading_pattern'] = trading_pattern
        result['n_benchmarks_screened'] = len(fast_scores)
        return result

    # Find best matching benchmark
    best_bench = max(valid_signals, key=lambda k: valid_signals[k]['composite_score'])
    best_score = _s(valid_signals[best_bench]['composite_score'])
    best_sig = valid_signals[best_bench]

    # Confidence (all int() calls are NaN-safe via _s above)
    is_significant = best_sig.get('pearson_p', 1) < 0.05
    if best_score > 0.5 and is_significant:
        confidence = min(95, int(best_score * 100 + 15))
    elif best_score > 0.3 and is_significant:
        confidence = int(best_score * 100)
    elif best_score > 0.2:
        confidence = max(10, int(best_score * 80))
    else:
        confidence = max(5, int(best_score * 50))

    # Map benchmark to asset class
    meta = bench_meta.get(best_bench, {})
    asset_class = meta.get('asset_class', _guess_asset_class_from_name(best_bench))
    asset_label = f"{best_bench} ({asset_class})"

    second_scores = sorted(
        [(k, v['composite_score']) for k, v in valid_signals.items()],
        key=lambda x: x[1], reverse=True
    )

    # ---- RESIDUAL CASCADE: strip primary, find secondary exposures ----
    # After fitting the best benchmark, do residuals correlate with others?
    significant_exposures = []
    best_beta = best_sig.get('beta', 0)
    best_corr = best_sig.get('pearson_r', 0)

    # Primary exposure (always first)
    significant_exposures.append({
        'name': best_bench,
        'asset_class': meta.get('asset_class', '?'),
        'composite': best_score,
        'pearson_r': round(best_corr, 4),
        'beta': round(best_beta, 4),
        'direction': 'short' if best_corr < -0.1 else 'long' if best_corr > 0.1 else 'neutral',
        'is_residual': False,
    })

    # Compute residuals = algo_ret - beta * best_bench_ret
    best_bench_ret = bench_returns_dict.get(best_bench)
    if best_bench_ret is not None and abs(best_beta) > 0.01:
        residual_df = pd.DataFrame({'algo': algo_ret, 'best': best_bench_ret}).dropna()
        if len(residual_df) >= 20:
            residuals = residual_df['algo'].values - best_beta * residual_df['best'].values
            residual_series = pd.Series(residuals, index=residual_df.index)

            # Screen residuals against other candidates
            for bname in candidates:
                if bname == best_bench:
                    continue
                other_ret = bench_returns_dict.get(bname)
                if other_ret is None:
                    continue
                res_aligned = pd.DataFrame({'res': residual_series, 'other': other_ret}).dropna()
                if len(res_aligned) < 20:
                    continue
                try:
                    r_corr, r_p = pearsonr(res_aligned['res'].values, res_aligned['other'].values)
                    r_cov = np.cov(res_aligned['res'].values, res_aligned['other'].values)[0, 1]
                    r_var = np.var(res_aligned['other'].values)
                    r_beta = r_cov / r_var if r_var > 1e-10 else 0

                    # Significant residual exposure: |corr| > 0.1 AND p < 0.10
                    if abs(r_corr) > 0.1 and r_p < 0.10:
                        m2 = bench_meta.get(bname, {})
                        significant_exposures.append({
                            'name': bname,
                            'asset_class': m2.get('asset_class', _guess_asset_class_from_name(bname)),
                            'composite': valid_signals.get(bname, {}).get('composite_score', 0),
                            'pearson_r': round(float(r_corr), 4),
                            'beta': round(float(r_beta), 4),
                            'direction': 'short' if r_corr < -0.1 else 'long' if r_corr > 0.1 else 'neutral',
                            'is_residual': True,  # detected via residual cascade
                        })
                except:
                    pass

    # ---- Determine multi-asset classification ----
    unique_classes = set(e['asset_class'] for e in significant_exposures)
    n_exposures = len(significant_exposures)

    if n_exposures >= 2 and len(unique_classes) >= 2:
        asset_class = 'mixed'
        top2 = significant_exposures[:2]
        asset_label = f"Mixed: {top2[0]['name']} ({top2[0]['asset_class']}) + {top2[1]['name']} ({top2[1]['asset_class']})"
        # Mixed is harder to be confident about
        confidence = max(confidence - 10, 10)
    elif n_exposures >= 2 and significant_exposures[0]['asset_class'] in ('equity', 'etf'):
        top_names = [e['name'] for e in significant_exposures[:4]]
        asset_label = f"Multi-equity: {', '.join(top_names)}"
    elif n_exposures >= 2:
        top_names = [e['name'] for e in significant_exposures[:3]]
        asset_label = f"Multi-exposure: {', '.join(top_names)}"

    # Opaque?
    if best_score < 0.15 and n_exposures <= 1:
        asset_class = 'unknown'
        asset_label = 'Opaque / Unidentified strategy'
        confidence = max(5, confidence)

    # Direction (of primary exposure)
    direction = 'long'
    if best_sig.get('pearson_r', 0) < -0.2 and best_sig.get('beta', 0) < -0.1:
        direction = 'short/inverse'
    elif abs(best_sig.get('pearson_r', 0)) < 0.1:
        direction = 'uncorrelated'

    # Top 5 matches for output
    top_matches = []
    for bname, score in second_scores[:5]:
        m2 = bench_meta.get(bname, {})
        top_matches.append({
            'name': bname,
            'asset_class': m2.get('asset_class', '?'),
            'composite': valid_signals[bname]['composite_score'],
            'pearson_r': valid_signals[bname]['pearson_r'],
            'beta': valid_signals[bname].get('beta', 0),
        })

    return {
        'predicted_asset': best_bench,
        'asset_class': asset_class,
        'asset_label': asset_label,
        'direction': direction,
        'confidence': confidence,
        'best_composite': best_score,
        'n_significant_exposures': n_exposures,
        'significant_exposures': significant_exposures,
        'top_matches': top_matches,
        'signals': signals,
        'all_scores': {k: v.get('composite_score', 0) for k, v in valid_signals.items()},
        'trading_pattern': trading_pattern,
        'n_benchmarks_screened': len(fast_scores),
        'n_candidates_deep': len(candidates),
    }


def _guess_asset_class_from_name(name):
    """Fallback asset class guess from benchmark display name."""
    n = name.upper()
    if n.startswith('EQ_'):
        return 'equity'
    if n.startswith('FUT_'):
        # Parse futures display names
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


def _analyze_trading_pattern(algo_daily):
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

    # Check if the original data has intraday timestamps
    hours = idx.hour
    weekdays = idx.weekday  # 0=Mon, 6=Sun

    has_weekend = (weekdays >= 5).any()
    weekend_pct = round((weekdays >= 5).mean() * 100, 1)

    # Activity distribution by hour
    hour_counts = pd.Series(hours).value_counts().sort_index()
    peak_hours = hour_counts.nlargest(3).index.tolist()

    # Activity by day of week
    day_counts = pd.Series(weekdays).value_counts().sort_index()

    # Price change frequency (how often does the price actually move?)
    close = algo_daily['close']
    changes = close.diff().abs()
    active_days_pct = round((changes > 0).mean() * 100, 1)

    # Volatility of returns
    daily_ret = close.pct_change().dropna()
    avg_daily_move = round(daily_ret.abs().mean() * 100, 4)

    # Infer pattern
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


def _empty_inference(reason):
    return {
        'predicted_asset': 'unknown',
        'asset_class': 'unknown',
        'asset_label': 'Unknown (insufficient data)',
        'direction': 'unknown',
        'confidence': 0,
        'best_composite': 0,
        'signals': {},
        'all_scores': {},
        'reason': reason,
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(metrics, style, inference=None):
    """Generate classification tags."""
    tags = []
    ar = metrics['annualized_return_pct']
    if ar > 20:
        tags.append('high-return')
    elif ar > 5:
        tags.append('moderate-return')
    elif ar > 0:
        tags.append('low-return')
    else:
        tags.append('negative-return')

    v = metrics['annualized_volatility_pct']
    if v > 30:
        tags.append('high-vol')
    elif v > 15:
        tags.append('moderate-vol')
    else:
        tags.append('low-vol')

    mdd = abs(metrics['max_drawdown_pct'])
    if mdd > 30:
        tags.append('high-risk')
    elif mdd > 15:
        tags.append('moderate-risk')
    else:
        tags.append('low-risk')

    s = metrics['sharpe_ratio']
    if s > 1.5:
        tags.append('excellent-sharpe')
    elif s > 0.5:
        tags.append('good-sharpe')
    elif s > 0:
        tags.append('poor-sharpe')
    else:
        tags.append('negative-sharpe')

    # Style analysis tags
    if 'exposures' in style:
        exp = style['exposures']
        dominant = max(exp, key=exp.get) if exp else None
        if dominant and exp[dominant] > 30:
            tags.append(f'exposed:{dominant}')
        if style.get('unexplained_pct', 0) > 50:
            tags.append('opaque-strategy')
        if style.get('r_squared', 0) > 0.7:
            tags.append('benchmark-follower')
        elif style.get('r_squared', 0) < 0.3:
            tags.append('independent')

    # Enhanced inference tags
    if inference and inference.get('asset_class') != 'unknown':
        tags.append(f"asset:{inference['asset_class']}")
        tags.append(f"inferred:{inference['predicted_asset']}")
        conf = inference.get('confidence', 0)
        if conf >= 60:
            tags.append('inference:high-confidence')
        elif conf >= 30:
            tags.append('inference:moderate-confidence')
        else:
            tags.append('inference:low-confidence')
        direction = inference.get('direction', '')
        if direction == 'short/inverse':
            tags.append('direction:inverse')
        elif direction == 'long':
            tags.append('direction:long')

    # Trading pattern tags
    if inference and 'trading_pattern' in inference:
        tp = inference['trading_pattern']
        if isinstance(tp, dict):
            active = tp.get('active_days_pct', 50)
            if active < 15:
                tags.append('trading:very-infrequent')
            elif active < 40:
                tags.append('trading:infrequent')
            else:
                tags.append('trading:active')

    return tags


# ============================================================
# RANKING
# ============================================================

def compute_ranking(metrics_list):
    """
    Compute a composite ranking score.
    Higher is better. Weights can be adjusted.
    """
    if not metrics_list:
        return []

    df = pd.DataFrame(metrics_list)

    # Normalize each metric to [0, 1] range
    def norm(series, higher_is_better=True):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        n = (series - mn) / (mx - mn)
        return n if higher_is_better else 1 - n

    weights = {
        'annualized_return_pct': 0.25,
        'sharpe_ratio': 0.25,
        'sortino_ratio': 0.15,
        'max_drawdown_pct': 0.15,  # lower (less negative) is better
        'calmar_ratio': 0.10,
        'annualized_volatility_pct': 0.10,  # lower is better
    }

    score = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        hib = col not in ('max_drawdown_pct', 'annualized_volatility_pct')
        # For max_drawdown, more negative is worse, so higher_is_better=True
        # (less negative = better)
        if col == 'max_drawdown_pct':
            hib = True  # -5% > -30%, higher is better
        score += norm(df[col], higher_is_better=hib) * w

    df['composite_score'] = score.round(4)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


# ============================================================
# SINGLE ALGO WORKER (for parallel processing)
# ============================================================

def process_single_algo(args):
    """Process a single algorithm file. Used in parallel execution."""
    filepath, bench_returns_dict, bench_daily_dict, bench_meta = args
    name = Path(filepath).stem

    try:
        # --- Verbose loading: capture specific reason for failure ---
        file_size = os.path.getsize(filepath)
        if file_size < 50:
            return {'_status': 'skipped', '_name': name, '_reason': f'file_too_small ({file_size} bytes)'}

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return {'_status': 'skipped', '_name': name,
                    '_reason': f'csv_read_failed: {type(e).__name__}: {str(e)[:100]}'}

        if len(df) < 2:
            return {'_status': 'skipped', '_name': name, '_reason': f'only_{len(df)}_rows (header only)'}

        daily = load_algorithm_csv(filepath)
        if daily is None:
            # Try to figure out why
            cols = list(df.columns[:6])
            n_rows = len(df)
            return {'_status': 'skipped', '_name': name,
                    '_reason': f'load_failed (rows={n_rows}, cols={cols})'}

        if len(daily) < 10:
            return {'_status': 'skipped', '_name': name, '_reason': f'too_few_days ({len(daily)})'}

        metrics, trim_info = compute_metrics(daily['close'], name)
        if metrics is None:
            close_min = daily['close'].min()
            close_max = daily['close'].max()
            dead_days = trim_info.get('dead_days', 0)
            alive_days = trim_info.get('alive_days', 0)
            if close_min == close_max or alive_days == 0:
                reason = f'dead_algo (price={close_min:.2f}, never moved, {len(daily)} days)'
            elif alive_days < 10:
                reason = f'too_short_after_trim (alive={alive_days}, dead={dead_days}, total={len(daily)} days)'
            else:
                reason = f'metrics_failed (alive={alive_days}, dead={dead_days}, total={len(daily)} days)'
            return {'_status': 'skipped', '_name': name, '_reason': reason}

        # Use trimmed data for inference (only the alive period)
        trimmed_closes, _ = trim_dead_tail(daily['close'])
        trimmed_daily = daily.loc[trimmed_closes.index]
        algo_ret = trimmed_closes.pct_change().dropna()

        # Enhanced asset inference (two-stage funnel) — on alive period only
        inference = infer_underlying_asset(trimmed_daily, bench_daily_dict, bench_returns_dict,
                                           bench_meta=bench_meta, top_n_candidates=10)

        # Style analysis: run only against top matches from inference (max 8)
        top_for_style = list(inference.get('all_scores', {}).keys())[:8]
        if not top_for_style:
            top_for_style = list(bench_returns_dict.keys())[:8]
        style_bench = {k: bench_returns_dict[k] for k in top_for_style if k in bench_returns_dict}
        style = sharpe_style_analysis(algo_ret, style_bench)

        tags = classify(metrics, style, inference)

        return {
            '_status': 'ok',
            'metrics': metrics,
            'style': style,
            'inference': inference,
            'tags': tags,
        }
    except Exception as e:
        return {'_status': 'error', '_name': name, '_reason': f'{type(e).__name__}: {str(e)[:200]}'}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze trading algorithms at scale')
    parser.add_argument('--algos', required=True, help='Directory with algorithm CSVs')
    parser.add_argument('--benchmarks', required=True, help='Directory with benchmark CSVs')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of algos (for testing)')
    parser.add_argument('--debug', action='store_true', help='Print errors as they happen + save error log')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Load benchmarks ---
    print("\n[1/4] Loading benchmarks...")
    benchmarks, bench_returns, bench_meta = load_all_benchmarks(args.benchmarks)

    if not bench_returns:
        print("ERROR: No benchmarks loaded. Check benchmark directory.")
        sys.exit(1)

    # --- Find algorithm files ---
    print("\n[2/4] Scanning algorithm files...")
    algo_files = sorted(glob.glob(os.path.join(args.algos, '*.csv')))

    if args.limit:
        algo_files = algo_files[:args.limit]

    print(f"  Found {len(algo_files)} algorithm files")

    # --- Process all algorithms ---
    print(f"\n[3/4] Analyzing algorithms ({args.workers} workers)...")

    all_metrics = []
    all_styles = []
    all_tags = []
    all_inferences = []

    tasks = [(f, bench_returns, benchmarks, bench_meta) for f in algo_files]
    completed = 0
    skipped = 0
    errors = 0
    ok = 0
    error_log = []  # ALL errors: (name, reason)
    skip_reasons = {}  # reason -> count

    # Error log file (written live)
    error_log_path = os.path.join(args.output, 'error_log.txt')
    error_fh = open(error_log_path, 'w')
    error_fh.write("name\tstatus\treason\n")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_algo, t): t[0] for t in tasks}

        for future in as_completed(futures):
            filepath = futures[future]
            algo_name = Path(filepath).stem
            completed += 1
            if completed % 500 == 0 or completed == len(algo_files):
                print(f"  Progress: {completed}/{len(algo_files)} "
                      f"(ok={ok}, skipped={skipped}, errors={errors})")

            try:
                result = future.result()
            except Exception as e:
                # Worker process died (OOM, segfault, etc.)
                errors += 1
                reason = f'WORKER_CRASH: {type(e).__name__}: {str(e)[:200]}'
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                error_fh.write(f"{algo_name}\tcrash\t{reason}\n")
                if args.debug and errors <= 10:
                    print(f"  DEBUG [{algo_name}]: {reason}")
                continue

            if result is None:
                errors += 1
                reason = 'returned_none'
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                error_fh.write(f"{algo_name}\tnone\t{reason}\n")
                continue

            status = result.get('_status', 'ok')

            if status == 'skipped':
                skipped += 1
                reason = result.get('_reason', 'unknown')
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                error_fh.write(f"{algo_name}\tskipped\t{reason}\n")
                if args.debug and skipped <= 5:
                    print(f"  DEBUG SKIP [{algo_name}]: {reason}")
                continue

            if status == 'error':
                errors += 1
                reason = result.get('_reason', 'unknown')
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                error_fh.write(f"{algo_name}\terror\t{reason}\n")
                if args.debug and errors <= 10:
                    print(f"  DEBUG ERROR [{algo_name}]: {reason}")
                continue

            # --- Process successful result ---
            try:
                ok += 1
                all_metrics.append(result['metrics'])

                style_row = {'name': result['metrics']['name']}
                if 'exposures' in result['style']:
                    style_row.update(result['style']['exposures'])
                    style_row['r_squared'] = result['style']['r_squared']
                    style_row['alpha_annual_pct'] = result['style']['alpha_annual_pct']
                    style_row['unexplained_pct'] = result['style']['unexplained_pct']
                else:
                    style_row['error'] = result['style'].get('error', 'unknown')
                all_styles.append(style_row)

                all_tags.append({
                    'name': result['metrics']['name'],
                    'tags': ', '.join(result['tags'])
                })

                # Inference results
                inf = result.get('inference', {})
                inf_row = {
                    'name': result['metrics']['name'],
                    'predicted_asset': inf.get('predicted_asset', 'unknown'),
                    'asset_class': inf.get('asset_class', 'unknown'),
                    'asset_label': inf.get('asset_label', ''),
                    'direction': inf.get('direction', 'unknown'),
                    'confidence': inf.get('confidence', 0),
                    'best_composite': inf.get('best_composite', 0),
                    'n_significant_exposures': inf.get('n_significant_exposures', 0),
                    'n_benchmarks_screened': inf.get('n_benchmarks_screened', 0),
                }
                # Significant exposures (up to 5) as flat columns
                for i, exp in enumerate(inf.get('significant_exposures', [])[:5]):
                    inf_row[f'exposure_{i + 1}_name'] = exp.get('name', '')
                    inf_row[f'exposure_{i + 1}_class'] = exp.get('asset_class', '')
                    inf_row[f'exposure_{i + 1}_corr'] = exp.get('pearson_r', 0)
                    inf_row[f'exposure_{i + 1}_beta'] = exp.get('beta', 0)
                    inf_row[f'exposure_{i + 1}_score'] = exp.get('composite', 0)
                    inf_row[f'exposure_{i + 1}_dir'] = exp.get('direction', '')
                # Trading pattern
                tp = inf.get('trading_pattern', {})
                if isinstance(tp, dict):
                    inf_row['active_days_pct'] = tp.get('active_days_pct', 0)
                    inf_row['avg_daily_move_pct'] = tp.get('avg_daily_move_pct', 0)
                    inf_row['trading_pattern'] = tp.get('pattern', '')
                all_inferences.append(inf_row)

            except Exception as e:
                ok -= 1  # undo the premature ok increment
                errors += 1
                reason = f'PROCESS_RESULT: {type(e).__name__}: {str(e)[:200]}'
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                error_fh.write(f"{algo_name}\tprocess_error\t{reason}\n")
                if args.debug and errors <= 10:
                    print(f"  DEBUG PROCESS ERROR [{algo_name}]: {reason}")

    error_fh.close()

    print(f"\n  Completed: {ok} analyzed, {skipped} skipped, {errors} errors")

    # Debug summary — always print
    if skip_reasons:
        print(f"\n  SKIP/ERROR BREAKDOWN:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:>6}x  {reason[:150]}")

    print(f"\n  Full error log: {error_log_path}")

    # --- Save results ---
    print("\n[4/4] Saving results...")

    # Metrics CSV
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(args.output, 'metrics_all.csv'), index=False)
    print(f"  Saved metrics_all.csv ({len(df_metrics)} rows)")

    # Style analysis CSV
    df_styles = pd.DataFrame(all_styles)
    df_styles.to_csv(os.path.join(args.output, 'style_analysis_all.csv'), index=False)
    print(f"  Saved style_analysis_all.csv")

    # Classification CSV
    df_tags = pd.DataFrame(all_tags)
    df_tags.to_csv(os.path.join(args.output, 'classification_all.csv'), index=False)
    print(f"  Saved classification_all.csv")

    # Asset inference CSV
    df_inf = pd.DataFrame(all_inferences)
    df_inf.to_csv(os.path.join(args.output, 'asset_inference_all.csv'), index=False)
    print(f"  Saved asset_inference_all.csv ({len(df_inf)} rows)")

    # Print inference summary
    if len(df_inf) > 0 and 'asset_class' in df_inf.columns:
        print(f"\n  ASSET CLASS DISTRIBUTION:")
        counts = df_inf['asset_class'].value_counts()
        for cls_name, cnt in counts.items():
            pct = cnt / len(df_inf) * 100
            print(f"    {cls_name:<15} {cnt:>6} ({pct:.1f}%)")

        print(f"\n  CONFIDENCE DISTRIBUTION:")
        df_inf['conf_bucket'] = pd.cut(df_inf['confidence'],
                                       bins=[0, 20, 40, 60, 80, 100],
                                       labels=['Very low', 'Low', 'Moderate', 'High', 'Very high'],
                                       include_lowest=True)
        conf_counts = df_inf['conf_bucket'].value_counts().sort_index()
        for bucket, cnt in conf_counts.items():
            print(f"    {bucket:<15} {cnt:>6}")

    # Ranking
    df_ranked = compute_ranking(all_metrics)
    if isinstance(df_ranked, pd.DataFrame) and len(df_ranked) > 0:
        # Merge inference into ranking
        if len(df_inf) > 0:
            df_ranked = df_ranked.merge(
                df_inf[['name', 'predicted_asset', 'asset_class', 'asset_label',
                        'direction', 'confidence']],
                on='name', how='left'
            )
        df_ranked.to_csv(os.path.join(args.output, 'ranking.csv'), index=False)
        print(f"  Saved ranking.csv")
        print(f"\n  TOP 10 ALGORITHMS:")
        print(f"  {'Rank':<5}{'Name':<12}{'Ret%':<8}{'Sharpe':<8}{'Asset':<15}{'Conf':<6}{'Score':<8}")
        print(f"  {'-' * 62}")
        for _, r in df_ranked.head(10).iterrows():
            asset = r.get('asset_label', '?')
            if isinstance(asset, str) and len(asset) > 14:
                asset = asset[:12] + '..'
            conf = r.get('confidence', 0)
            conf_str = f"{int(conf)}%" if pd.notna(conf) else '?'
            print(f"  {int(r['rank']):<5}{r['name']:<12}{r['annualized_return_pct']:<8.2f}"
                  f"{r['sharpe_ratio']:<8.3f}{str(asset):<15}{conf_str:<6}"
                  f"{r['composite_score']:<8.4f}")

    print(f"\nAll results saved to: {args.output}/")
    print("Done!")


if __name__ == '__main__':
    main()
