"""
Memory-efficient dtype utilities for DataFrames and arrays.

This module provides functions to optimize memory usage by:
- Downcasting numeric columns to smaller dtypes
- Converting float64 to float32 where precision allows
- Optimizing integer columns
- Reporting memory usage

Usage:
    from src.utils.dtypes import optimize_dtypes, downcast_floats, get_memory_usage

    # Optimize a DataFrame
    df_optimized = optimize_dtypes(df)

    # Downcast floats to float32
    df = downcast_floats(df)

    # Get memory report
    print(get_memory_usage(df))
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_dtypes(
    df: pd.DataFrame,
    float_precision: str = "float32",
    int_downcast: bool = True,
    inplace: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes for memory efficiency.

    Args:
        df: DataFrame to optimize
        float_precision: Target float type ('float32' or 'float16')
        int_downcast: Whether to downcast integer columns
        inplace: Modify DataFrame in place
        verbose: Print memory reduction report

    Returns:
        Optimized DataFrame
    """
    if not inplace:
        df = df.copy()

    initial_mem = df.memory_usage(deep=True).sum()

    for col in df.columns:
        col_type = df[col].dtype

        # Handle float columns
        if col_type in [np.float64, np.float32, np.float16]:
            target_dtype = np.float32 if float_precision == "float32" else np.float16

            # Check if downcast is safe (no precision loss for typical financial data)
            if col_type == np.float64 and float_precision in ("float32", "float16"):
                # Check value range
                col_min = df[col].min()
                col_max = df[col].max()

                if float_precision == "float32":
                    if (np.isnan(col_min) or abs(col_min) < 3.4e38) and \
                       (np.isnan(col_max) or abs(col_max) < 3.4e38):
                        df[col] = df[col].astype(target_dtype)
                else:  # float16
                    if (np.isnan(col_min) or abs(col_min) < 6.5e4) and \
                       (np.isnan(col_max) or abs(col_max) < 6.5e4):
                        df[col] = df[col].astype(target_dtype)

        # Handle integer columns
        elif col_type in [np.int64, np.int32, np.int16, np.int8] and int_downcast:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= 0:
                # Unsigned integers
                if col_max <= 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max <= 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max <= 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                # Signed integers
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype(np.int32)

    final_mem = df.memory_usage(deep=True).sum()

    if verbose:
        reduction = (1 - final_mem / initial_mem) * 100
        logger.info(
            f"Memory optimization: {initial_mem / 1e6:.2f} MB -> "
            f"{final_mem / 1e6:.2f} MB ({reduction:.1f}% reduction)"
        )

    return df


def downcast_floats(
    df: pd.DataFrame,
    target_dtype: np.dtype = np.float32,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Convert all float columns to target dtype.

    Args:
        df: DataFrame to convert
        target_dtype: Target float dtype (np.float32 or np.float16)
        inplace: Modify in place

    Returns:
        DataFrame with converted floats
    """
    if not inplace:
        df = df.copy()

    float_cols = df.select_dtypes(include=[np.float64, np.float32]).columns
    for col in float_cols:
        df[col] = df[col].astype(target_dtype)

    return df


def upcast_floats(
    df: pd.DataFrame,
    target_dtype: np.dtype = np.float64,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Convert all float columns to higher precision dtype.

    Useful before operations that require high precision.

    Args:
        df: DataFrame to convert
        target_dtype: Target float dtype (np.float64)
        inplace: Modify in place

    Returns:
        DataFrame with converted floats
    """
    if not inplace:
        df = df.copy()

    float_cols = df.select_dtypes(include=[np.float16, np.float32]).columns
    for col in float_cols:
        df[col] = df[col].astype(target_dtype)

    return df


def get_memory_usage(
    df: pd.DataFrame,
    detailed: bool = True,
) -> Union[float, dict]:
    """
    Get memory usage of a DataFrame.

    Args:
        df: DataFrame to analyze
        detailed: Return per-column breakdown

    Returns:
        Total memory in MB, or dict with per-column breakdown
    """
    memory = df.memory_usage(deep=True)

    if detailed:
        return {
            "total_mb": memory.sum() / 1e6,
            "per_column_mb": (memory / 1e6).to_dict(),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
        }
    else:
        return memory.sum() / 1e6


def memory_report(df: pd.DataFrame, name: str = "DataFrame") -> str:
    """
    Generate a memory usage report for a DataFrame.

    Args:
        df: DataFrame to analyze
        name: Name for the report

    Returns:
        Formatted report string
    """
    memory = df.memory_usage(deep=True)
    total_mb = memory.sum() / 1e6

    lines = [
        f"\nMemory Report: {name}",
        "=" * 50,
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns",
        f"Total memory: {total_mb:.2f} MB",
        "",
        "Top columns by memory:",
    ]

    # Sort by memory usage
    sorted_cols = memory.drop("Index").sort_values(ascending=False)
    for col in sorted_cols.head(10).index:
        col_mb = memory[col] / 1e6
        col_dtype = df[col].dtype
        lines.append(f"  {col}: {col_mb:.3f} MB ({col_dtype})")

    # Dtype summary
    dtype_summary = df.dtypes.value_counts()
    lines.append("")
    lines.append("Dtype summary:")
    for dtype, count in dtype_summary.items():
        lines.append(f"  {dtype}: {count} columns")

    return "\n".join(lines)


def optimize_parquet_compression(
    df: pd.DataFrame,
    path: str,
    compression: str = "zstd",
    compression_level: int = 3,
    row_group_size: int = 100000,
) -> dict:
    """
    Save DataFrame to Parquet with optimized compression settings.

    Args:
        df: DataFrame to save
        path: Output path
        compression: Compression algorithm ('zstd', 'snappy', 'gzip', 'lz4')
        compression_level: Compression level (higher = smaller but slower)
        row_group_size: Number of rows per row group

    Returns:
        Dict with file size and compression ratio
    """
    import os

    # Optimize dtypes before saving
    df_opt = optimize_dtypes(df, inplace=False)

    # Calculate uncompressed size
    uncompressed_size = df_opt.memory_usage(deep=True).sum()

    # Save with specified compression
    df_opt.to_parquet(
        path,
        compression=compression,
        compression_level=compression_level if compression in ("zstd", "gzip") else None,
        row_group_size=row_group_size,
        index=True,
    )

    # Get compressed file size
    compressed_size = os.path.getsize(path)

    ratio = uncompressed_size / compressed_size if compressed_size > 0 else 0

    logger.info(
        f"Saved to {path}: {uncompressed_size / 1e6:.2f} MB -> "
        f"{compressed_size / 1e6:.2f} MB ({ratio:.1f}x compression)"
    )

    return {
        "path": path,
        "uncompressed_mb": uncompressed_size / 1e6,
        "compressed_mb": compressed_size / 1e6,
        "compression_ratio": ratio,
        "compression": compression,
    }


def convert_array_dtype(
    arr: np.ndarray,
    target_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Convert numpy array to target dtype with safety checks.

    Args:
        arr: Input array
        target_dtype: Target dtype

    Returns:
        Converted array
    """
    if arr.dtype == target_dtype:
        return arr

    # Check for potential overflow/precision issues
    if target_dtype in (np.float32, np.float16):
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            max_val = np.abs(arr[finite_mask]).max()
            if target_dtype == np.float32 and max_val > 3.4e38:
                logger.warning("Values may overflow in float32, keeping original dtype")
                return arr
            elif target_dtype == np.float16 and max_val > 6.5e4:
                logger.warning("Values may overflow in float16, keeping original dtype")
                return arr

    return arr.astype(target_dtype)


def estimate_memory_for_features(
    n_dates: int,
    n_algos: int,
    n_features_per_algo: int,
    n_regime_features: int,
    dtype: np.dtype = np.float32,
) -> dict:
    """
    Estimate memory required for a feature matrix.

    Args:
        n_dates: Number of dates
        n_algos: Number of algorithms
        n_features_per_algo: Features per algorithm
        n_regime_features: Number of regime features
        dtype: Data type

    Returns:
        Dict with memory estimates
    """
    total_columns = n_algos * n_features_per_algo + n_regime_features
    bytes_per_value = np.dtype(dtype).itemsize
    total_bytes = n_dates * total_columns * bytes_per_value

    return {
        "rows": n_dates,
        "columns": total_columns,
        "dtype": str(dtype),
        "bytes_per_value": bytes_per_value,
        "total_mb": total_bytes / 1e6,
        "total_gb": total_bytes / 1e9,
    }


class MemoryTracker:
    """
    Context manager to track memory usage during operations.

    Usage:
        with MemoryTracker("Feature computation") as tracker:
            features = compute_features(data)
        print(f"Peak memory: {tracker.peak_mb:.2f} MB")
    """

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_mb = 0.0
        self.peak_mb = 0.0
        self.end_mb = 0.0

    def __enter__(self):
        import tracemalloc
        tracemalloc.start()
        self.start_mb = tracemalloc.get_traced_memory()[0] / 1e6
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.end_mb = current / 1e6
        self.peak_mb = peak / 1e6

        logger.info(
            f"{self.name}: Start={self.start_mb:.2f} MB, "
            f"End={self.end_mb:.2f} MB, Peak={self.peak_mb:.2f} MB"
        )

        return False  # Don't suppress exceptions


def efficient_concat(
    dfs: list[pd.DataFrame],
    axis: int = 0,
    dtype: Optional[np.dtype] = np.float32,
) -> pd.DataFrame:
    """
    Memory-efficient DataFrame concatenation.

    Pre-converts dtypes before concatenation to avoid memory spikes.

    Args:
        dfs: List of DataFrames to concatenate
        axis: Concatenation axis
        dtype: Target dtype for float columns

    Returns:
        Concatenated DataFrame
    """
    import gc

    if not dfs:
        return pd.DataFrame()

    # Convert each DataFrame to target dtype
    if dtype is not None:
        converted = []
        for df in dfs:
            converted.append(downcast_floats(df, target_dtype=dtype))
            gc.collect()
        dfs = converted

    # Concatenate
    result = pd.concat(dfs, axis=axis, copy=False)

    # Force garbage collection
    gc.collect()

    return result


def sparse_to_dense_efficient(
    sparse_df: pd.DataFrame,
    fill_value: float = 0.0,
    dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """
    Convert sparse DataFrame to dense with memory efficiency.

    Args:
        sparse_df: DataFrame with sparse columns
        fill_value: Value to fill for missing entries
        dtype: Target dtype

    Returns:
        Dense DataFrame
    """
    result = pd.DataFrame(index=sparse_df.index)

    for col in sparse_df.columns:
        if hasattr(sparse_df[col], "sparse"):
            # Sparse column
            result[col] = sparse_df[col].sparse.to_dense().astype(dtype)
        else:
            result[col] = sparse_df[col].astype(dtype)

        result[col] = result[col].fillna(fill_value)

    return result
