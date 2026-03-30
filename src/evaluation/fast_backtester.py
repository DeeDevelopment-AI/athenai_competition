"""
Fast Backtester with Pre-computation and Caching.

Optimizations:
1. Pre-filter active algorithms per period
2. Pre-extract factor values as numpy arrays
3. Pre-compute rolling covariance matrices
4. Batch allocation processing
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PeriodCache:
    """Pre-computed data for a backtest period."""
    # Basic data
    returns_matrix: np.ndarray  # [n_days x n_active_algos]
    dates: np.ndarray  # [n_days] datetime index
    algo_ids: list[str]  # [n_active_algos] algorithm IDs
    algo_id_to_idx: dict[str, int]  # Map algo_id -> column index

    # Factor data (pre-extracted)
    factor_matrix: np.ndarray  # [n_days x n_active_algos] factor values

    # Rebalance schedule
    rebalance_day_indices: np.ndarray  # [n_rebalances] day indices
    rebalance_dates: list  # [n_rebalances] actual dates

    # Pre-computed covariance matrices (for min_variance/max_sharpe)
    cov_matrices: Optional[dict] = None  # {day_idx: cov_matrix}

    # Pre-computed volatilities (for risk_parity)
    vol_matrices: Optional[np.ndarray] = None  # [n_rebalances x n_active_algos]


class FastBacktester:
    """
    High-performance backtester with pre-computation.

    Usage:
        backtester = FastBacktester(
            algo_returns=returns_df,
            features=features_df,
            factor_name='rolling_sharpe_21d',
            rebalance_frequency='weekly',
        )

        # Pre-compute for a period
        cache = backtester.prepare_period('2023-01-01', '2023-12-31')

        # Run multiple allocation strategies quickly
        for allocator in allocators:
            result = backtester.run_with_cache(cache, allocator)
    """

    def __init__(
        self,
        algo_returns: pd.DataFrame,
        features: pd.DataFrame,
        factor_name: str = 'rolling_sharpe_21d',
        rebalance_frequency: str = 'weekly',
        lookback_window: int = 63,
        min_observations: int = 20,
        precompute_cov: bool = True,
    ):
        self.algo_returns = algo_returns
        self.features = features
        self.factor_name = factor_name
        self.rebalance_frequency = rebalance_frequency
        self.lookback_window = lookback_window
        self.min_observations = min_observations
        self.precompute_cov = precompute_cov

        # All algorithm IDs
        self.all_algo_ids = algo_returns.columns.tolist()

    def prepare_period(
        self,
        start_date: str,
        end_date: str,
        hist_start: Optional[str] = None,
    ) -> PeriodCache:
        """
        Pre-compute everything needed for backtesting a period.

        Args:
            start_date: Period start
            end_date: Period end
            hist_start: Start of historical data (for lookback). Defaults to beginning.

        Returns:
            PeriodCache with all pre-computed data
        """
        logger.debug(f"Preparing period {start_date} to {end_date}")

        # 1. Filter to period
        mask = (self.algo_returns.index >= start_date) & (self.algo_returns.index <= end_date)
        period_returns = self.algo_returns.loc[mask]

        if len(period_returns) < 5:
            raise ValueError(f"Period too short: {len(period_returns)} days")

        # 2. Find active algorithms (have data in period)
        # Check that algo has at least some observations in period
        has_period_data = period_returns.notna().sum() >= 1

        # For history check, use data up to period END (not start)
        # This ensures algos have enough history by the time we need it
        hist_up_to_end = self.algo_returns.loc[:end_date]
        has_enough_history = hist_up_to_end.notna().sum() >= self.min_observations

        active_mask = has_period_data & has_enough_history
        active_algos = period_returns.columns[active_mask].tolist()
        n_active = len(active_algos)

        logger.debug(f"Active algorithms: {n_active} / {len(self.all_algo_ids)}")

        # 3. Build returns matrix (fill NaN with 0)
        returns_matrix = period_returns[active_algos].fillna(0).values.astype(np.float64)
        dates = period_returns.index.values

        # 4. Build algo ID mapping
        algo_id_to_idx = {algo: i for i, algo in enumerate(active_algos)}

        # 5. Pre-extract factor values for all dates
        factor_matrix = self._extract_factor_matrix(dates, active_algos)

        # 6. Compute rebalance schedule
        rebalance_day_indices, rebalance_dates = self._compute_rebalance_schedule(
            period_returns.index
        )

        # 7. Covariance matrices - computed lazily on demand, not upfront
        # This avoids computing ~50 large cov matrices that might not be used
        cov_matrices = {}  # Will be filled lazily

        # 8. Pre-compute volatilities for risk parity
        vol_matrices = self._precompute_volatilities(
            start_date, active_algos, rebalance_dates
        )

        return PeriodCache(
            returns_matrix=returns_matrix,
            dates=dates,
            algo_ids=active_algos,
            algo_id_to_idx=algo_id_to_idx,
            factor_matrix=factor_matrix,
            rebalance_day_indices=rebalance_day_indices,
            rebalance_dates=rebalance_dates,
            cov_matrices=cov_matrices,
            vol_matrices=vol_matrices,
        )

    def _extract_factor_matrix(
        self,
        dates: np.ndarray,
        algo_ids: list[str],
    ) -> np.ndarray:
        """Extract factor values as numpy matrix - optimized version."""
        n_days = len(dates)
        n_algos = len(algo_ids)
        factor_matrix = np.full((n_days, n_algos), np.nan, dtype=np.float64)

        # Build column names
        factor_cols = [f"{algo}_{self.factor_name}" for algo in algo_ids]

        # Check which columns exist (do this once)
        existing_mask = np.array([c in self.features.columns for c in factor_cols])
        existing_cols = [c for c, exists in zip(factor_cols, existing_mask) if exists]
        existing_indices = np.where(existing_mask)[0]

        if len(existing_cols) == 0:
            logger.warning(f"No factor columns found for {self.factor_name}")
            return factor_matrix

        # Convert dates to timestamps for faster lookup
        date_timestamps = pd.DatetimeIndex(dates)

        # Find which dates exist in features
        common_dates = date_timestamps.intersection(self.features.index)

        if len(common_dates) == 0:
            return factor_matrix

        # Extract all data at once using loc with lists
        try:
            # Batch extract for all common dates
            extracted = self.features.loc[common_dates, existing_cols].values

            # Map back to our date indices
            date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
            for i, date in enumerate(common_dates):
                day_idx = date_to_idx.get(date)
                if day_idx is not None:
                    factor_matrix[day_idx, existing_indices] = extracted[i]
        except Exception as e:
            logger.warning(f"Batch extraction failed, falling back to row-by-row: {e}")
            # Fallback to row-by-row
            for day_idx, date in enumerate(dates):
                date_ts = pd.Timestamp(date)
                if date_ts in self.features.index:
                    values = self.features.loc[date_ts, existing_cols].values
                    factor_matrix[day_idx, existing_indices] = values

        return factor_matrix

    def _compute_rebalance_schedule(
        self,
        dates: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, list]:
        """Compute which days are rebalance days."""
        rebalance_indices = []
        rebalance_dates = []
        last_rebalance = None

        for i, date in enumerate(dates):
            should_rebalance = False

            if last_rebalance is None:
                should_rebalance = True
            elif self.rebalance_frequency == 'daily':
                should_rebalance = True
            elif self.rebalance_frequency == 'weekly':
                should_rebalance = (date - last_rebalance).days >= 5
            elif self.rebalance_frequency == 'monthly':
                should_rebalance = (date - last_rebalance).days >= 21

            if should_rebalance:
                rebalance_indices.append(i)
                rebalance_dates.append(date)
                last_rebalance = date

        return np.array(rebalance_indices, dtype=np.int64), rebalance_dates

    def _precompute_covariances(
        self,
        period_start: str,
        algo_ids: list[str],
        rebalance_dates: list,
    ) -> dict:
        """Pre-compute covariance matrices for each rebalance date."""
        cov_matrices = {}

        for date in rebalance_dates:
            # Get historical returns up to this date
            hist = self.algo_returns.loc[:date, algo_ids].tail(self.lookback_window)

            if len(hist) < self.min_observations:
                cov_matrices[date] = None
                continue

            # Compute covariance matrix
            cov = hist.cov().values.astype(np.float64)

            # Handle NaN and make positive semi-definite
            cov = np.nan_to_num(cov, nan=0.0)
            cov = (cov + cov.T) / 2

            # Add small diagonal for numerical stability
            min_eig = np.min(np.linalg.eigvalsh(cov))
            if min_eig < 1e-6:
                cov += (1e-6 - min_eig) * np.eye(len(algo_ids))

            cov_matrices[date] = cov

        return cov_matrices

    def _precompute_volatilities(
        self,
        period_start: str,
        algo_ids: list[str],
        rebalance_dates: list,
    ) -> np.ndarray:
        """Pre-compute volatilities for each rebalance date."""
        n_rebalances = len(rebalance_dates)
        n_algos = len(algo_ids)
        vol_matrix = np.full((n_rebalances, n_algos), 1.0, dtype=np.float64)

        for i, date in enumerate(rebalance_dates):
            hist = self.algo_returns.loc[:date, algo_ids].tail(self.lookback_window)

            if len(hist) >= self.min_observations:
                # Use ddof=1, skipna is default True for pandas std()
                vols = hist.std(ddof=1).values.astype(np.float64)
                # Replace NaN with median vol, then clip
                median_vol = np.nanmedian(vols)
                if np.isnan(median_vol):
                    median_vol = 0.01
                vols = np.nan_to_num(vols, nan=median_vol)
                vols = np.maximum(vols, 1e-8)  # Avoid division by zero
                vol_matrix[i] = vols

        return vol_matrix

    def run_with_cache(
        self,
        cache: PeriodCache,
        optimizer: str = 'equal_weight',
        selection_method: str = 'top_n',
        selection_param: int = 100,
        max_weight: float = 0.40,
        shrinkage: float = 0.1,
    ) -> dict:
        """
        Run backtest using pre-computed cache.

        Args:
            cache: Pre-computed PeriodCache
            optimizer: Optimization method
            selection_method: 'top_n', 'bottom_n', 'top_percentile'
            selection_param: Number to select or percentile
            max_weight: Maximum weight per algorithm
            shrinkage: Shrinkage for covariance estimation

        Returns:
            Dict with portfolio_returns, sharpe, n_selected_avg
        """
        n_days, n_algos = cache.returns_matrix.shape
        n_rebalances = len(cache.rebalance_day_indices)

        # Allocate weights for each rebalance
        weights_list = []
        n_selected_list = []

        for reb_idx in range(n_rebalances):
            day_idx = cache.rebalance_day_indices[reb_idx]
            date = cache.rebalance_dates[reb_idx]

            # Get factor values at this date
            factor_values = cache.factor_matrix[day_idx]

            # Select algorithms
            selected_mask, selected_indices = self._select_algorithms(
                factor_values, selection_method, selection_param
            )
            n_selected = len(selected_indices)
            n_selected_list.append(n_selected)

            if n_selected == 0:
                weights_list.append(np.zeros(n_algos, dtype=np.float64))
                continue

            # Compute weights for selected algorithms
            selected_weights = self._optimize_weights(
                optimizer=optimizer,
                selected_indices=selected_indices,
                cache=cache,
                reb_idx=reb_idx,
                date=date,
                shrinkage=shrinkage,
            )

            # Expand to full weight vector
            full_weights = np.zeros(n_algos, dtype=np.float64)
            full_weights[selected_indices] = selected_weights

            # Apply max weight constraint
            full_weights = np.clip(full_weights, 0, max_weight)
            if full_weights.sum() > 1.0:
                full_weights /= full_weights.sum()

            weights_list.append(full_weights)

        # Compute portfolio returns
        portfolio_returns = self._compute_portfolio_returns(
            cache.returns_matrix,
            np.array(weights_list),
            cache.rebalance_day_indices,
        )

        # Compute Sharpe
        sharpe = self._compute_sharpe(portfolio_returns)

        return {
            'portfolio_returns': pd.Series(portfolio_returns, index=cache.dates),
            'sharpe': sharpe,
            'n_selected_avg': np.mean(n_selected_list) if n_selected_list else 0,
        }

    def _select_algorithms(
        self,
        factor_values: np.ndarray,
        method: str,
        param: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select algorithms based on factor values."""
        valid_mask = ~np.isnan(factor_values)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            return np.zeros(len(factor_values), dtype=bool), np.array([], dtype=np.int64)

        n_select = min(int(param), n_valid)

        if method == 'top_n':
            # Select highest values
            scores = np.copy(factor_values)
            scores[~valid_mask] = -np.inf
            top_indices = np.argsort(scores)[-n_select:]
            selected_mask = np.zeros(len(factor_values), dtype=bool)
            selected_mask[top_indices] = True

        elif method == 'bottom_n':
            # Select lowest values
            scores = np.copy(factor_values)
            scores[~valid_mask] = np.inf
            bottom_indices = np.argsort(scores)[:n_select]
            selected_mask = np.zeros(len(factor_values), dtype=bool)
            selected_mask[bottom_indices] = True

        else:
            # Default: select all valid
            selected_mask = valid_mask

        selected_indices = np.where(selected_mask)[0]
        return selected_mask, selected_indices

    def _optimize_weights(
        self,
        optimizer: str,
        selected_indices: np.ndarray,
        cache: PeriodCache,
        reb_idx: int,
        date,
        shrinkage: float,
    ) -> np.ndarray:
        """Compute optimal weights for selected algorithms."""
        n_selected = len(selected_indices)

        if n_selected == 0:
            return np.array([])

        if optimizer == 'equal_weight':
            return np.ones(n_selected, dtype=np.float64) / n_selected

        elif optimizer == 'risk_parity':
            # Inverse volatility weighting
            vols = cache.vol_matrices[reb_idx, selected_indices]
            inv_vols = 1.0 / np.maximum(vols, 1e-8)
            return inv_vols / inv_vols.sum()

        elif optimizer == 'momentum':
            # Weight by factor score
            factor_values = cache.factor_matrix[cache.rebalance_day_indices[reb_idx], selected_indices]
            scores = np.maximum(factor_values, 0)
            if scores.sum() < 1e-10:
                return np.ones(n_selected, dtype=np.float64) / n_selected
            return scores / scores.sum()

        elif optimizer == 'min_variance':
            return self._min_variance_weights(
                selected_indices, cache, date, shrinkage
            )

        elif optimizer == 'max_sharpe':
            return self._max_sharpe_weights(
                selected_indices, cache, reb_idx, date, shrinkage
            )

        elif optimizer == 'vol_targeting':
            # Equal weight with vol scaling (simplified)
            return np.ones(n_selected, dtype=np.float64) / n_selected

        else:
            return np.ones(n_selected, dtype=np.float64) / n_selected

    def _get_cov_matrix(
        self,
        selected_indices: np.ndarray,
        cache: PeriodCache,
        date,
        shrinkage: float,
    ) -> Optional[np.ndarray]:
        """Get covariance matrix for selected algorithms (lazy computation)."""
        n = len(selected_indices)
        selected_algos = [cache.algo_ids[i] for i in selected_indices]

        # Get historical returns
        hist = self.algo_returns.loc[:date, selected_algos].tail(self.lookback_window)

        if len(hist) < self.min_observations:
            return None

        # Compute covariance
        cov = hist.cov().values.astype(np.float64)
        cov = np.nan_to_num(cov, nan=0.0)
        cov = (cov + cov.T) / 2

        # Apply shrinkage
        if shrinkage > 0:
            diag = np.diag(np.diag(cov))
            cov = (1 - shrinkage) * cov + shrinkage * diag

        # Ensure positive semi-definite
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 1e-6:
            cov += (1e-6 - min_eig) * np.eye(n)

        return cov

    def _min_variance_weights(
        self,
        selected_indices: np.ndarray,
        cache: PeriodCache,
        date,
        shrinkage: float,
    ) -> np.ndarray:
        """Compute minimum variance weights."""
        n = len(selected_indices)

        cov = self._get_cov_matrix(selected_indices, cache, date, shrinkage)
        if cov is None:
            return np.ones(n, dtype=np.float64) / n

        # Solve for minimum variance
        try:
            ones = np.ones(n)
            cov_inv = np.linalg.inv(cov)
            weights = cov_inv @ ones
            weights = weights / weights.sum()
            weights = np.maximum(weights, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n) / n
        except np.linalg.LinAlgError:
            weights = np.ones(n) / n

        return weights

    def _max_sharpe_weights(
        self,
        selected_indices: np.ndarray,
        cache: PeriodCache,
        reb_idx: int,
        date,
        shrinkage: float,
    ) -> np.ndarray:
        """Compute maximum Sharpe ratio weights."""
        n = len(selected_indices)

        cov = self._get_cov_matrix(selected_indices, cache, date, shrinkage)
        if cov is None:
            return np.ones(n, dtype=np.float64) / n

        # Get expected returns (use factor values as proxy)
        day_idx = cache.rebalance_day_indices[reb_idx]
        mu = cache.factor_matrix[day_idx, selected_indices]
        mu = np.nan_to_num(mu, nan=0.0)

        # Solve for max Sharpe
        try:
            cov_inv = np.linalg.inv(cov)
            weights = cov_inv @ mu

            # Handle negative weights
            if weights.sum() <= 0:
                weights = np.ones(n) / n
            else:
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
        except np.linalg.LinAlgError:
            weights = np.ones(n) / n

        return weights

    def _compute_portfolio_returns(
        self,
        returns_matrix: np.ndarray,
        weights_matrix: np.ndarray,
        rebalance_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute portfolio returns with rebalancing (lookahead-free).

        Timing convention (matches MarketSimulator):
        - On rebalance day T, we observe data up to T and compute new weights
        - Those weights apply to returns starting from day T+1, NOT day T
        - This avoids lookahead bias: decision at T cannot use T's return

        Example:
        - Day 0: rebalance, observe factors, compute weights → weights apply day 1+
        - Day 0 return: 0 (no position yet, or use prior weights)
        - Day 1 return: dot(new_weights, returns[1])
        """
        n_days, n_algos = returns_matrix.shape
        portfolio_returns = np.zeros(n_days, dtype=np.float64)

        if len(weights_matrix) == 0:
            return portfolio_returns

        current_weights = np.zeros(n_algos, dtype=np.float64)
        reb_ptr = 0

        for day in range(n_days):
            # FIRST: compute this day's return with CURRENT weights (before update)
            # This ensures rebalance day T uses old weights, new weights apply T+1
            portfolio_returns[day] = np.dot(current_weights, returns_matrix[day])

            # THEN: update weights if this is a rebalance day
            # New weights will take effect starting TOMORROW
            if reb_ptr < len(rebalance_indices) and day == rebalance_indices[reb_ptr]:
                current_weights = weights_matrix[reb_ptr]
                reb_ptr += 1

        return portfolio_returns

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret < 1e-10:
            return 0.0

        return (mean_ret / std_ret) * np.sqrt(252)


def run_fast_backtest(
    algo_returns: pd.DataFrame,
    features: pd.DataFrame,
    factor_name: str,
    optimizer: str,
    selection_method: str,
    selection_param: int,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    **kwargs,
) -> dict:
    """
    Run fast backtest for train/val/test periods.

    Returns dict with results for each period.
    """
    backtester = FastBacktester(
        algo_returns=algo_returns,
        features=features,
        factor_name=factor_name,
        **kwargs,
    )

    results = {}

    for period_name, start, end in [
        ('train', train_start, train_end),
        ('val', val_start, val_end),
        ('test', test_start, test_end),
    ]:
        try:
            cache = backtester.prepare_period(start, end)
            result = backtester.run_with_cache(
                cache,
                optimizer=optimizer,
                selection_method=selection_method,
                selection_param=selection_param,
                **kwargs,
            )
            results[period_name] = result
        except Exception as e:
            logger.warning(f"Period {period_name} failed: {e}")
            results[period_name] = {
                'portfolio_returns': pd.Series(dtype=float),
                'sharpe': np.nan,
                'n_selected_avg': 0,
            }

    return results
