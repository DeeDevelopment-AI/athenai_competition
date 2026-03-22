"""
Trade analyzer for computing trade-based features with lookahead bias protection.

Key constraint: Cutoff date 2024-12-31
- Trades closed before cutoff: use realized P&L
- Trades open at cutoff: use latent P&L (mark-to-market)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cutoff date to avoid lookahead bias
CUTOFF_DATE = pd.Timestamp("2024-12-31", tz="UTC")


@dataclass
class TradeRecord:
    """Processed trade with P&L information."""
    algo_id: str
    date_open: pd.Timestamp
    date_close: Optional[pd.Timestamp]  # None if still open at cutoff
    volume: float
    price_open: float
    price_close: float  # Actual close or MtM at cutoff
    pnl: float
    pnl_pct: float
    duration_days: float
    is_realized: bool  # True if closed before cutoff, False if latent


class TradeAnalyzer:
    """
    Analyzes trades from benchmark with lookahead bias protection.

    Loads trades from trades_benchmark.csv, matches with algo quotes,
    and computes P&L respecting the cutoff date.
    """

    def __init__(
        self,
        trades_path: str | Path,
        algos_dir: str | Path,
        cutoff_date: pd.Timestamp = CUTOFF_DATE,
    ):
        self.trades_path = Path(trades_path)
        self.algos_dir = Path(algos_dir)
        self.cutoff_date = cutoff_date

        self._raw_trades: Optional[pd.DataFrame] = None
        self._processed_trades: Optional[pd.DataFrame] = None
        self._algo_quotes_cache: dict[str, pd.DataFrame] = {}

    def load_raw_trades(self) -> pd.DataFrame:
        """Load raw trades from CSV."""
        if self._raw_trades is not None:
            return self._raw_trades

        logger.info(f"Loading trades from {self.trades_path}")
        df = pd.read_csv(self.trades_path)

        # Parse datetime columns (mixed formats: some have microseconds, some don't)
        df["dateOpen"] = pd.to_datetime(df["dateOpen"], format="mixed", utc=True)
        df["dateClose"] = pd.to_datetime(df["dateClose"], format="mixed", utc=True)

        # Rename for consistency
        df = df.rename(columns={"productname": "algo_id"})

        self._raw_trades = df
        logger.info(f"Loaded {len(df)} trades for {df['algo_id'].nunique()} algos")

        return df

    def load_algo_quotes(self, algo_id: str) -> Optional[pd.DataFrame]:
        """Load quote data for a specific algo with caching."""
        if algo_id in self._algo_quotes_cache:
            return self._algo_quotes_cache[algo_id]

        quote_path = self.algos_dir / f"{algo_id}.csv"
        if not quote_path.exists():
            logger.warning(f"Quote file not found for {algo_id}")
            return None

        try:
            df = pd.read_csv(quote_path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.set_index("datetime").sort_index()
            self._algo_quotes_cache[algo_id] = df
            return df
        except Exception as e:
            logger.warning(f"Error loading quotes for {algo_id}: {e}")
            return None

    def get_price_at_time(
        self,
        quotes: pd.DataFrame,
        timestamp: pd.Timestamp,
        method: str = "ffill",
    ) -> Optional[float]:
        """
        Get price at a specific timestamp.

        Uses forward-fill to get the last known price before timestamp.
        """
        if quotes is None or quotes.empty:
            return None

        # Find the closest price at or before the timestamp
        mask = quotes.index <= timestamp
        if not mask.any():
            # No data before timestamp, use first available
            return quotes["close"].iloc[0]

        return quotes.loc[mask, "close"].iloc[-1]

    def process_trades(self, show_progress: bool = True) -> pd.DataFrame:
        """
        Process all trades with P&L calculation and lookahead protection.

        Returns DataFrame with processed trade records.
        """
        if self._processed_trades is not None:
            return self._processed_trades

        raw_trades = self.load_raw_trades()

        # Filter to trades that opened before cutoff
        trades_before_cutoff = raw_trades[raw_trades["dateOpen"] <= self.cutoff_date].copy()
        logger.info(f"Processing {len(trades_before_cutoff)} trades opened before {self.cutoff_date.date()}")

        processed_records = []
        n_trades = len(trades_before_cutoff)
        progress_step = max(1, n_trades // 20)

        skipped_no_quotes = 0
        skipped_no_price = 0

        for idx, (_, trade) in enumerate(trades_before_cutoff.iterrows()):
            algo_id = trade["algo_id"]
            date_open = trade["dateOpen"]
            date_close_raw = trade["dateClose"]
            volume = trade["volume"]

            # Load quotes for this algo
            quotes = self.load_algo_quotes(algo_id)
            if quotes is None:
                skipped_no_quotes += 1
                continue

            # Get open price
            price_open = self.get_price_at_time(quotes, date_open)
            if price_open is None:
                skipped_no_price += 1
                continue

            # Determine if trade is realized or latent
            if date_close_raw <= self.cutoff_date:
                # Realized trade - use actual close
                is_realized = True
                date_close = date_close_raw
                price_close = self.get_price_at_time(quotes, date_close)
            else:
                # Latent trade - mark to market at cutoff
                is_realized = False
                date_close = None
                price_close = self.get_price_at_time(quotes, self.cutoff_date)

            if price_close is None:
                skipped_no_price += 1
                continue

            # Calculate P&L
            # Assuming long-only (buy at open, sell at close)
            pnl = (price_close - price_open) * volume
            pnl_pct = (price_close - price_open) / price_open if price_open != 0 else 0

            # Duration
            effective_close = date_close if is_realized else self.cutoff_date
            duration_days = (effective_close - date_open).total_seconds() / 86400

            processed_records.append({
                "algo_id": algo_id,
                "date_open": date_open,
                "date_close": date_close,
                "date_close_effective": effective_close,
                "volume": volume,
                "price_open": price_open,
                "price_close": price_close,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "duration_days": duration_days,
                "is_realized": is_realized,
            })

            if show_progress and (idx + 1) % progress_step == 0:
                pct = (idx + 1) / n_trades * 100
                logger.info(f"Trade processing: {idx+1}/{n_trades} ({pct:.0f}%)")

        self._processed_trades = pd.DataFrame(processed_records)

        logger.info(
            f"Processed {len(self._processed_trades)} trades. "
            f"Skipped: {skipped_no_quotes} (no quotes), {skipped_no_price} (no price)"
        )

        n_realized = self._processed_trades["is_realized"].sum()
        n_latent = len(self._processed_trades) - n_realized
        logger.info(f"Realized trades: {n_realized}, Latent (open at cutoff): {n_latent}")

        return self._processed_trades

    def get_trades_by_algo(self, algo_id: str) -> pd.DataFrame:
        """Get processed trades for a specific algo."""
        trades = self.process_trades(show_progress=False)
        return trades[trades["algo_id"] == algo_id].copy()

    def compute_rolling_trade_features(
        self,
        algo_id: str,
        dates: pd.DatetimeIndex,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Compute rolling trade-based features for an algo.

        Features computed:
        - hit_ratio: % of profitable trades
        - profit_factor: gross profit / gross loss
        - avg_trade_duration: mean duration in days
        - n_trades: number of closed trades
        - avg_pnl_pct: average P&L percentage

        Only realized (closed) trades are used for these metrics.
        Latent trades contribute to separate latent_exposure feature.

        Args:
            algo_id: Algorithm identifier
            dates: DatetimeIndex to compute features for
            window: Rolling window in days

        Returns:
            DataFrame with features indexed by date
        """
        algo_trades = self.get_trades_by_algo(algo_id)

        # Only use realized trades for win/loss metrics
        realized_trades = algo_trades[algo_trades["is_realized"]].copy()

        features = pd.DataFrame(index=dates)
        features["hit_ratio"] = np.nan
        features["profit_factor"] = np.nan
        features["avg_trade_duration"] = np.nan
        features["n_trades"] = 0
        features["avg_pnl_pct"] = np.nan
        features["latent_exposure"] = 0.0

        if realized_trades.empty:
            return features

        # Convert dates to UTC for comparison
        dates_utc = dates.tz_localize("UTC") if dates.tz is None else dates.tz_convert("UTC")

        for i, date in enumerate(dates_utc):
            window_start = date - pd.Timedelta(days=window)

            # Trades that CLOSED within the window (not opened)
            # This avoids lookahead: we only know about trades after they close
            mask = (
                (realized_trades["date_close_effective"] > window_start) &
                (realized_trades["date_close_effective"] <= date)
            )
            window_trades = realized_trades[mask]

            n_trades = len(window_trades)
            features.iloc[i, features.columns.get_loc("n_trades")] = n_trades

            if n_trades == 0:
                continue

            # Hit ratio: % profitable
            n_winners = (window_trades["pnl"] > 0).sum()
            hit_ratio = n_winners / n_trades
            features.iloc[i, features.columns.get_loc("hit_ratio")] = hit_ratio

            # Profit factor: gross profit / gross loss
            gross_profit = window_trades.loc[window_trades["pnl"] > 0, "pnl"].sum()
            gross_loss = abs(window_trades.loc[window_trades["pnl"] < 0, "pnl"].sum())
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = np.inf if gross_profit > 0 else np.nan
            features.iloc[i, features.columns.get_loc("profit_factor")] = profit_factor

            # Average trade duration
            avg_duration = window_trades["duration_days"].mean()
            features.iloc[i, features.columns.get_loc("avg_trade_duration")] = avg_duration

            # Average P&L %
            avg_pnl_pct = window_trades["pnl_pct"].mean()
            features.iloc[i, features.columns.get_loc("avg_pnl_pct")] = avg_pnl_pct

            # Latent exposure: sum of volume in open (unrealized) trades
            latent_mask = (
                (~algo_trades["is_realized"]) &
                (algo_trades["date_open"] <= date)
            )
            latent_vol = algo_trades.loc[latent_mask, "volume"].sum()
            features.iloc[i, features.columns.get_loc("latent_exposure")] = latent_vol

        return features

    def compute_turnover(
        self,
        algo_id: str,
        dates: pd.DatetimeIndex,
        window: int = 21,
    ) -> pd.Series:
        """
        Compute rolling turnover for an algo.

        Turnover = total volume traded / average volume
        """
        algo_trades = self.get_trades_by_algo(algo_id)

        if algo_trades.empty:
            return pd.Series(np.nan, index=dates, name="turnover")

        dates_utc = dates.tz_localize("UTC") if dates.tz is None else dates.tz_convert("UTC")

        turnover = []
        for date in dates_utc:
            window_start = date - pd.Timedelta(days=window)

            # Trades opened in window (represents activity)
            mask = (
                (algo_trades["date_open"] > window_start) &
                (algo_trades["date_open"] <= date)
            )
            window_volume = algo_trades.loc[mask, "volume"].sum()
            turnover.append(window_volume)

        return pd.Series(turnover, index=dates, name="turnover")

    def get_algo_trade_summary(self, algo_id: str) -> dict:
        """Get summary statistics for an algo's trades."""
        trades = self.get_trades_by_algo(algo_id)
        realized = trades[trades["is_realized"]]

        if realized.empty:
            return {
                "n_trades": 0,
                "n_realized": 0,
                "n_latent": len(trades) - len(realized),
                "total_pnl": 0,
                "hit_ratio": np.nan,
                "profit_factor": np.nan,
                "avg_duration_days": np.nan,
            }

        n_winners = (realized["pnl"] > 0).sum()
        gross_profit = realized.loc[realized["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(realized.loc[realized["pnl"] < 0, "pnl"].sum())

        return {
            "n_trades": len(trades),
            "n_realized": len(realized),
            "n_latent": len(trades) - len(realized),
            "total_pnl": realized["pnl"].sum(),
            "hit_ratio": n_winners / len(realized) if len(realized) > 0 else np.nan,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
            "avg_duration_days": realized["duration_days"].mean(),
            "avg_pnl_pct": realized["pnl_pct"].mean(),
        }

    def clear_cache(self):
        """Clear the quote cache to free memory."""
        self._algo_quotes_cache.clear()
        logger.info("Cleared quote cache")
