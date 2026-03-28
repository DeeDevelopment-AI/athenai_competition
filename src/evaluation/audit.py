"""Audit helpers for exporting rebalance-period allocations."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def build_periodic_allocation_rows(
    allocation_dates: Iterable,
    weights_iterable: Iterable,
    algo_names: list[str],
    *,
    active_threshold: float = 1e-6,
    final_period_end: Optional[pd.Timestamp] = None,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Convert dated weight snapshots into long-form audit rows."""
    dates = [pd.Timestamp(d) for d in allocation_dates]
    weights_list = [np.asarray(weights, dtype=np.float32) for weights in weights_iterable]
    if not dates or not weights_list:
        return []

    if len(dates) != len(weights_list):
        raise ValueError("allocation_dates and weights_iterable must have the same length")

    metadata = metadata or {}
    rows: list[dict] = []
    for idx, (period_start, weights) in enumerate(zip(dates, weights_list)):
        if idx + 1 < len(dates):
            period_end = dates[idx + 1] - pd.Timedelta(days=1)
            if period_end < period_start:
                period_end = period_start
        else:
            period_end = pd.Timestamp(final_period_end) if final_period_end is not None else period_start
            if period_end < period_start:
                period_end = period_start

        active_mask = weights > float(active_threshold)
        active_indices = np.flatnonzero(active_mask)
        n_active = int(active_indices.size)
        gross_exposure = float(weights.sum())

        if n_active == 0:
            rows.append(
                {
                    **metadata,
                    "rebalance_date": str(period_start.date()),
                    "period_start": str(period_start.date()),
                    "period_end": str(period_end.date()),
                    "algorithm": "",
                    "weight": 0.0,
                    "n_active": 0,
                    "gross_exposure": gross_exposure,
                }
            )
            continue

        for algo_idx in active_indices:
            rows.append(
                {
                    **metadata,
                    "rebalance_date": str(period_start.date()),
                    "period_start": str(period_start.date()),
                    "period_end": str(period_end.date()),
                    "algorithm": str(algo_names[algo_idx]),
                    "weight": float(weights[algo_idx]),
                    "n_active": n_active,
                    "gross_exposure": gross_exposure,
                }
            )
    return rows


def compress_daily_weight_history(
    dates: Iterable,
    weights_iterable: Iterable,
    *,
    atol: float = 1e-6,
) -> tuple[list[pd.Timestamp], list[np.ndarray]]:
    """Compress daily weight history into piecewise-constant snapshots."""
    snapshot_dates: list[pd.Timestamp] = []
    snapshot_weights: list[np.ndarray] = []

    for date, weights in zip(dates, weights_iterable):
        ts = pd.Timestamp(date)
        current = np.asarray(weights, dtype=np.float32)
        if not snapshot_weights or not np.allclose(current, snapshot_weights[-1], atol=atol, rtol=0.0):
            snapshot_dates.append(ts)
            snapshot_weights.append(current.copy())

    return snapshot_dates, snapshot_weights


def split_periodic_allocation_rows(
    allocation_dates: Iterable,
    weights_iterable: Iterable,
    algo_names: list[str],
    *,
    split_windows: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    active_threshold: float = 1e-6,
    final_period_end: Optional[pd.Timestamp] = None,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Export long-form rows, splitting each rebalance period across named windows."""
    dates = [pd.Timestamp(d) for d in allocation_dates]
    weights_list = [np.asarray(weights, dtype=np.float32) for weights in weights_iterable]
    metadata = metadata or {}
    rows: list[dict] = []

    for idx, (period_start, weights) in enumerate(zip(dates, weights_list)):
        if idx + 1 < len(dates):
            raw_period_end = dates[idx + 1] - pd.Timedelta(days=1)
        else:
            raw_period_end = pd.Timestamp(final_period_end) if final_period_end is not None else dates[idx]
        if raw_period_end < period_start:
            raw_period_end = period_start

        active_mask = weights > float(active_threshold)
        active_indices = np.flatnonzero(active_mask)
        n_active = int(active_indices.size)
        gross_exposure = float(weights.sum())

        overlaps = []
        for split_name, split_start, split_end in split_windows:
            start = max(period_start, pd.Timestamp(split_start))
            end = min(raw_period_end, pd.Timestamp(split_end))
            if start <= end:
                overlaps.append((split_name, start, end))

        if not overlaps:
            overlaps.append(("unspecified", period_start, raw_period_end))

        for split_name, segment_start, segment_end in overlaps:
            base_row = {
                **metadata,
                "split": split_name,
                "rebalance_date": str(period_start.date()),
                "period_start": str(segment_start.date()),
                "period_end": str(segment_end.date()),
                "n_active": n_active,
                "gross_exposure": gross_exposure,
            }
            if n_active == 0:
                rows.append({**base_row, "algorithm": "", "weight": 0.0})
                continue
            for algo_idx in active_indices:
                rows.append(
                    {
                        **base_row,
                        "algorithm": str(algo_names[algo_idx]),
                        "weight": float(weights[algo_idx]),
                    }
                )

    return rows
