#!/usr/bin/env python3
"""
=================================================================
ALGO-VS-ALGO ANALYSIS: Correlation, Cointegration & Relationships
=================================================================

Analyzes relationships BETWEEN algorithms (not algo vs benchmark).
Reads algorithm CSVs directly and produces:

  - algo_correlation_matrix.csv   -> Full pairwise correlation matrix
  - algo_pairs.csv                -> Top correlated/anticorrelated pairs
  - algo_cointegration.csv        -> Cointegration tests for top pairs
  - algo_network.json             -> Network graph data for visualization
  - algo_diversification.csv      -> Best diversifiers for each algo
  - algo_overlap_stats.csv        -> Which algos overlap in time

Uso:
  python3 algo_relationships.py --algos ./algorithms/ --output results/ --top-pairs 500

For 10K algos, the correlation matrix is computed efficiently via pandas.
Cointegration (expensive) runs only on the top N most correlated pairs.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import json
import os
import glob
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import loader from pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_paths import default_output_dir, raw_algos_path
try:
    from algo_pipeline import load_algorithm_csv
except ImportError:
    # Fallback minimal loader
    def load_algorithm_csv(filepath):
        try:
            df = pd.read_csv(filepath)
            if len(df) < 2:
                return None
            df.columns = [c.lower().strip() for c in df.columns]
            date_col = next((c for c in ['datetime','date','timestamp'] if c in df.columns), df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            if 'close' not in df.columns:
                return None
            daily = df[['close']].resample('D').last().dropna()
            daily['open'] = daily['high'] = daily['low'] = daily['close']
            return daily
        except:
            return None


# ============================================================
# 1. LOAD ALL ALGOS INTO ALIGNED RETURNS MATRIX
# ============================================================

def build_returns_matrix(algo_dir, limit=None):
    """
    Load all algorithms and build a date-aligned returns matrix.
    Columns = algo names, rows = dates, values = daily returns.
    """
    algo_files = sorted(glob.glob(os.path.join(algo_dir, '*.csv')))
    if limit:
        algo_files = algo_files[:limit]

    print(f"  Loading {len(algo_files)} algorithm files...")

    all_closes = {}
    algo_meta = {}
    loaded = 0
    skipped = 0

    for f in algo_files:
        name = Path(f).stem
        daily = load_algorithm_csv(f)
        if daily is None or len(daily) < 15:
            skipped += 1
            continue

        closes = daily['close']
        all_closes[name] = closes
        algo_meta[name] = {
            'start': str(closes.index[0].date()),
            'end': str(closes.index[-1].date()),
            'n_days': len(closes),
        }
        loaded += 1
        if loaded % 1000 == 0:
            print(f"    Loaded {loaded} algos...")

    print(f"  Loaded: {loaded}, Skipped: {skipped}")

    # Build combined DataFrame (dates aligned, NaN where no data)
    df_closes = pd.DataFrame(all_closes)
    df_returns = df_closes.pct_change()

    # Drop dates where ALL algos are NaN
    df_returns = df_returns.dropna(how='all')

    print(f"  Returns matrix: {df_returns.shape[0]} dates x {df_returns.shape[1]} algos")
    return df_returns, df_closes, algo_meta


# ============================================================
# 2. OVERLAP ANALYSIS
# ============================================================

def compute_overlap_stats(df_returns, algo_meta):
    """
    For each algo pair, compute how many overlapping trading days they have.
    Only store pairs with >= 20 overlap days.
    Returns a DataFrame with algo1, algo2, overlap_days, overlap_start, overlap_end.
    """
    algos = df_returns.columns.tolist()
    n = len(algos)

    # Compute which dates each algo has data
    has_data = df_returns.notna()

    # Pairwise overlap count via matrix multiplication (efficient)
    # has_data.T @ has_data gives overlap count for all pairs at once
    overlap_matrix = has_data.astype(int).T.dot(has_data.astype(int))

    return overlap_matrix


# ============================================================
# 3. CORRELATION MATRIX
# ============================================================

def compute_correlation_matrix(df_returns, min_overlap=30):
    """
    Compute pairwise Pearson correlation on overlapping daily returns.
    pandas .corr() automatically handles NaN alignment (pairwise complete).
    min_periods ensures we only get correlations where enough overlap exists.
    """
    print(f"  Computing correlation matrix (min_overlap={min_overlap})...")
    corr_matrix = df_returns.corr(min_periods=min_overlap)
    print(f"  Correlation matrix: {corr_matrix.shape}")

    # Count how many valid correlations each algo has
    valid_counts = corr_matrix.notna().sum() - 1  # subtract self
    print(f"  Avg valid correlations per algo: {valid_counts.mean():.0f}")

    return corr_matrix


# ============================================================
# 4. EXTRACT TOP PAIRS
# ============================================================

def extract_top_pairs(corr_matrix, overlap_matrix, top_n=500, min_overlap=30):
    """
    Extract the most interesting pairs:
      - Top N most positively correlated
      - Top N most negatively correlated (anticorrelated)
      - Exclude self-correlations
    """
    algos = corr_matrix.columns.tolist()
    pairs = []

    # Iterate upper triangle only
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            r = corr_matrix.iloc[i, j]
            if pd.isna(r):
                continue
            overlap = int(overlap_matrix.iloc[i, j])
            if overlap < min_overlap:
                continue
            pairs.append({
                'algo1': algos[i],
                'algo2': algos[j],
                'pearson_r': round(float(r), 4),
                'abs_r': round(abs(float(r)), 4),
                'overlap_days': overlap,
            })

    df_pairs = pd.DataFrame(pairs)
    if len(df_pairs) == 0:
        return df_pairs

    # Sort by absolute correlation
    df_pairs = df_pairs.sort_values('abs_r', ascending=False).reset_index(drop=True)

    # Keep top N most correlated + top N most anticorrelated
    top_pos = df_pairs.nlargest(top_n, 'pearson_r')
    top_neg = df_pairs.nsmallest(top_n, 'pearson_r')
    top_abs = df_pairs.nlargest(top_n, 'abs_r')

    # Combine and deduplicate
    combined = pd.concat([top_pos, top_neg, top_abs]).drop_duplicates(
        subset=['algo1', 'algo2']).sort_values('abs_r', ascending=False).reset_index(drop=True)

    print(f"  Extracted {len(combined)} interesting pairs")
    print(f"    Highest correlation:  {combined['pearson_r'].max():.4f}")
    print(f"    Lowest correlation:   {combined['pearson_r'].min():.4f}")
    print(f"    Median |correlation|: {combined['abs_r'].median():.4f}")

    return combined


# ============================================================
# 5. COINTEGRATION TESTS
# ============================================================

def test_cointegration_pair(args):
    """Test Engle-Granger cointegration for a single pair."""
    name1, name2, closes1, closes2 = args
    try:
        # Align
        aligned = pd.DataFrame({'a': closes1, 'b': closes2}).dropna()
        if len(aligned) < 60:
            return None

        a = aligned['a'].values
        b = aligned['b'].values

        # Engle-Granger: regress a on b, test residuals for stationarity (ADF)
        from numpy.polynomial import polynomial as P
        # Simple OLS: a = alpha + beta * b + epsilon
        beta = np.cov(a, b)[0, 1] / np.var(b) if np.var(b) > 0 else 0
        alpha = np.mean(a) - beta * np.mean(b)
        residuals = a - (alpha + beta * b)

        # ADF test on residuals (simplified: check if residuals are mean-reverting)
        # Use Dickey-Fuller: delta_r_t = phi * r_{t-1} + error
        r_lag = residuals[:-1]
        r_diff = np.diff(residuals)
        if len(r_lag) < 10 or np.std(r_lag) < 1e-10:
            return None

        phi = np.cov(r_diff, r_lag)[0, 1] / np.var(r_lag) if np.var(r_lag) > 0 else 0
        se_phi = np.std(r_diff - phi * r_lag) / (np.std(r_lag) * np.sqrt(len(r_lag)))
        adf_stat = phi / se_phi if se_phi > 0 else 0

        # Critical values (approximate for Engle-Granger with 2 variables):
        # 1%: -3.90, 5%: -3.34, 10%: -3.04
        is_cointegrated_5pct = adf_stat < -3.34
        is_cointegrated_10pct = adf_stat < -3.04

        # Half-life of mean reversion
        half_life = -np.log(2) / phi if phi < 0 else float('inf')
        half_life = min(half_life, 9999)

        # Spread stats
        spread_mean = np.mean(residuals)
        spread_std = np.std(residuals)
        current_z = (residuals[-1] - spread_mean) / spread_std if spread_std > 0 else 0

        return {
            'algo1': name1,
            'algo2': name2,
            'adf_stat': round(float(adf_stat), 4),
            'is_cointegrated_5pct': bool(is_cointegrated_5pct),
            'is_cointegrated_10pct': bool(is_cointegrated_10pct),
            'hedge_ratio': round(float(beta), 4),
            'half_life_days': round(float(half_life), 1),
            'spread_mean': round(float(spread_mean), 4),
            'spread_std': round(float(spread_std), 4),
            'current_z_score': round(float(current_z), 4),
            'n_days': len(aligned),
        }
    except Exception:
        return None


def run_cointegration_tests(df_pairs, df_closes, max_tests=500, workers=4):
    """Run cointegration tests on the top correlated pairs."""
    # Test pairs with highest absolute correlation
    test_pairs = df_pairs.head(max_tests)
    print(f"  Running cointegration on {len(test_pairs)} pairs ({workers} workers)...")

    tasks = []
    for _, row in test_pairs.iterrows():
        a1, a2 = row['algo1'], row['algo2']
        if a1 in df_closes.columns and a2 in df_closes.columns:
            tasks.append((a1, a2, df_closes[a1], df_closes[a2]))

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(test_cointegration_pair, t): t for t in tasks}
        for future in as_completed(futures):
            completed += 1
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
            except:
                pass

    df_coint = pd.DataFrame(results)
    if len(df_coint) > 0:
        df_coint = df_coint.sort_values('adf_stat').reset_index(drop=True)
        n_coint = df_coint['is_cointegrated_5pct'].sum()
        print(f"  Cointegrated pairs (5%): {n_coint} / {len(df_coint)}")

    return df_coint


# ============================================================
# 6. DIVERSIFICATION ANALYSIS
# ============================================================

def find_best_diversifiers(corr_matrix, algo_meta, top_n=10):
    """
    For each algorithm, find the N best diversifiers (lowest correlation)
    among algos with sufficient overlap.
    """
    algos = corr_matrix.columns.tolist()
    results = []

    for algo in algos:
        row = corr_matrix[algo].drop(algo).dropna()
        if len(row) == 0:
            continue

        # Best diversifiers = lowest correlation (ideally negative)
        best_div = row.nsmallest(top_n)
        for other, r in best_div.items():
            results.append({
                'algo': algo,
                'diversifier': other,
                'correlation': round(float(r), 4),
            })

    return pd.DataFrame(results)


# ============================================================
# 7. NETWORK GRAPH DATA
# ============================================================

def build_network_data(corr_matrix, df_pairs, algo_meta, corr_threshold=0.5):
    """
    Build a network graph where:
      - Nodes = algorithms
      - Edges = strong correlations (|r| > threshold)
    """
    if len(df_pairs) == 0 or 'abs_r' not in df_pairs.columns:
        return {'nodes': [], 'edges': [], 'threshold': corr_threshold, 'n_nodes': 0, 'n_edges': 0}

    strong_pairs = df_pairs[df_pairs['abs_r'] >= corr_threshold]

    if len(strong_pairs) == 0:
        # Lower threshold
        corr_threshold = df_pairs['abs_r'].quantile(0.9) if len(df_pairs) > 0 else 0.3
        strong_pairs = df_pairs[df_pairs['abs_r'] >= corr_threshold]

    involved_algos = set(strong_pairs['algo1'].tolist() + strong_pairs['algo2'].tolist())

    nodes = []
    for algo in involved_algos:
        meta = algo_meta.get(algo, {})
        n_edges = len(strong_pairs[(strong_pairs['algo1'] == algo) |
                                    (strong_pairs['algo2'] == algo)])
        nodes.append({
            'id': algo,
            'n_connections': n_edges,
            'start': meta.get('start', ''),
            'end': meta.get('end', ''),
            'n_days': meta.get('n_days', 0),
        })

    edges = []
    for _, row in strong_pairs.iterrows():
        edges.append({
            'source': row['algo1'],
            'target': row['algo2'],
            'weight': round(float(row['pearson_r']), 4),
            'abs_weight': round(float(row['abs_r']), 4),
        })

    return {
        'nodes': nodes,
        'edges': edges,
        'threshold': corr_threshold,
        'n_nodes': len(nodes),
        'n_edges': len(edges),
    }


# ============================================================
# 8. AGGREGATE STATS
# ============================================================

def compute_aggregate_stats(corr_matrix, df_pairs, df_coint):
    """Compute summary statistics about the algo universe."""
    # Flatten upper triangle
    n = corr_matrix.shape[0]
    upper = corr_matrix.values[np.triu_indices(n, k=1)]
    valid = upper[~np.isnan(upper)]

    stats = {
        'n_algorithms': n,
        'n_valid_pairs': len(valid),
        'mean_correlation': round(float(np.mean(valid)), 4) if len(valid) > 0 else 0,
        'median_correlation': round(float(np.median(valid)), 4) if len(valid) > 0 else 0,
        'std_correlation': round(float(np.std(valid)), 4) if len(valid) > 0 else 0,
        'pct_positive': round(float((valid > 0).mean() * 100), 1) if len(valid) > 0 else 0,
        'pct_strong_pos': round(float((valid > 0.5).mean() * 100), 1) if len(valid) > 0 else 0,
        'pct_strong_neg': round(float((valid < -0.5).mean() * 100), 1) if len(valid) > 0 else 0,
    }

    if len(df_coint) > 0:
        stats['n_cointegrated_5pct'] = int(df_coint['is_cointegrated_5pct'].sum())
        stats['n_cointegrated_10pct'] = int(df_coint['is_cointegrated_10pct'].sum())
        stats['avg_half_life'] = round(float(
            df_coint[df_coint['is_cointegrated_5pct']]['half_life_days'].mean()
        ), 1) if stats['n_cointegrated_5pct'] > 0 else 0

    # Correlation distribution
    if len(valid) > 0:
        hist, edges = np.histogram(valid, bins=50)
        stats['corr_distribution'] = {
            'counts': hist.tolist(),
            'edges': [round(e, 3) for e in edges.tolist()],
        }
    else:
        stats['corr_distribution'] = {'counts': [], 'edges': []}

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze relationships between algorithms')
    parser.add_argument('--algos', default=str(raw_algos_path()), help='Directory with algorithm CSVs')
    parser.add_argument('--output', default=str(default_output_dir('relationships')), help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit algos (for testing)')
    parser.add_argument('--top-pairs', type=int, default=500, help='Number of top pairs to extract')
    parser.add_argument('--coint-tests', type=int, default=300, help='Max cointegration tests')
    parser.add_argument('--min-overlap', type=int, default=30, help='Min overlapping days for correlation')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers for cointegration')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Load returns matrix
    print("\n[1/7] Building returns matrix...")
    df_returns, df_closes, algo_meta = build_returns_matrix(args.algos, args.limit)

    if df_returns.shape[1] < 2:
        print("ERROR: Need at least 2 algorithms")
        sys.exit(1)

    # 2. Overlap analysis
    print("\n[2/7] Computing overlap stats...")
    overlap_matrix = compute_overlap_stats(df_returns, algo_meta)

    # Save overlap as compressed stats (not full N×N matrix)
    overlap_vals = overlap_matrix.values[np.triu_indices(len(overlap_matrix), k=1)]
    valid_overlaps = overlap_vals[overlap_vals > 0]
    print(f"  Pairs with any overlap: {len(valid_overlaps)}")
    print(f"  Avg overlap: {valid_overlaps.mean():.0f} days") if len(valid_overlaps) > 0 else None
    print(f"  Median overlap: {np.median(valid_overlaps):.0f} days") if len(valid_overlaps) > 0 else None

    # 3. Correlation matrix
    print("\n[3/7] Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(df_returns, min_overlap=args.min_overlap)

    # 4. Extract interesting pairs
    print("\n[4/7] Extracting top pairs...")
    df_pairs = extract_top_pairs(corr_matrix, overlap_matrix,
                                  top_n=args.top_pairs, min_overlap=args.min_overlap)

    # 5. Cointegration tests
    print("\n[5/7] Cointegration analysis...")
    df_coint = run_cointegration_tests(df_pairs, df_closes,
                                        max_tests=args.coint_tests, workers=args.workers)

    # 6. Diversification
    print("\n[6/7] Finding best diversifiers...")
    df_diversifiers = find_best_diversifiers(corr_matrix, algo_meta, top_n=5)

    # 7. Network graph
    print("\n[7/7] Building network data...")
    network = build_network_data(corr_matrix, df_pairs, algo_meta)
    stats = compute_aggregate_stats(corr_matrix, df_pairs, df_coint)

    # --- Save everything ---
    print("\n  Saving results...")

    # Correlation matrix (save sparse: only non-NaN upper triangle above threshold)
    # Full matrix would be huge for 10K algos, so save top pairs instead
    df_pairs.to_csv(os.path.join(args.output, 'algo_pairs.csv'), index=False)
    print(f"  algo_pairs.csv ({len(df_pairs)} pairs)")

    if len(df_coint) > 0:
        df_coint.to_csv(os.path.join(args.output, 'algo_cointegration.csv'), index=False)
        print(f"  algo_cointegration.csv ({len(df_coint)} pairs)")

    df_diversifiers.to_csv(os.path.join(args.output, 'algo_diversifiers.csv'), index=False)
    print(f"  algo_diversifiers.csv ({len(df_diversifiers)} rows)")

    # Network + stats as JSON
    output_json = {
        'stats': stats,
        'network': network,
    }
    with open(os.path.join(args.output, 'algo_relationships.json'), 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"  algo_relationships.json")

    # Save full correlation matrix for smaller datasets
    if corr_matrix.shape[0] <= 2000:
        corr_matrix.to_csv(os.path.join(args.output, 'algo_correlation_matrix.csv'))
        print(f"  algo_correlation_matrix.csv ({corr_matrix.shape[0]}x{corr_matrix.shape[1]})")
    else:
        print(f"  Skipping full corr matrix (too large: {corr_matrix.shape[0]}x{corr_matrix.shape[1]})")
        print(f"  Use algo_pairs.csv for top pairs instead")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ALGO UNIVERSE SUMMARY")
    print(f"{'='*60}")
    print(f"  Algorithms analyzed:   {stats['n_algorithms']}")
    print(f"  Valid pairs:           {stats['n_valid_pairs']}")
    print(f"  Mean correlation:      {stats['mean_correlation']}")
    print(f"  Median correlation:    {stats['median_correlation']}")
    print(f"  Strong positive (>0.5): {stats['pct_strong_pos']}%")
    print(f"  Strong negative (<-0.5): {stats['pct_strong_neg']}%")
    if 'n_cointegrated_5pct' in stats:
        print(f"  Cointegrated pairs:    {stats['n_cointegrated_5pct']} (5% level)")
        if stats['avg_half_life'] > 0:
            print(f"  Avg half-life:         {stats['avg_half_life']} days")
    print(f"  Network: {network['n_nodes']} nodes, {network['n_edges']} edges "
          f"(threshold: |r|>{network['threshold']:.2f})")

    # Top 10 most correlated pairs
    if len(df_pairs) > 0 and 'pearson_r' in df_pairs.columns:
        print(f"\n  TOP 10 MOST CORRELATED PAIRS:")
        print(f"  {'Algo1':<12}{'Algo2':<12}{'Pearson r':<12}{'Overlap':<10}")
        print(f"  {'-'*46}")
        for _, row in df_pairs.head(10).iterrows():
            print(f"  {row['algo1']:<12}{row['algo2']:<12}"
                  f"{row['pearson_r']:<12.4f}{row['overlap_days']:<10}")

    # Top cointegrated pairs
    if len(df_coint) > 0 and df_coint['is_cointegrated_5pct'].any():
        top_coint = df_coint[df_coint['is_cointegrated_5pct']].head(10)
        print(f"\n  TOP COINTEGRATED PAIRS (5% level):")
        print(f"  {'Algo1':<12}{'Algo2':<12}{'ADF stat':<10}{'Hedge β':<10}{'Half-life':<12}")
        print(f"  {'-'*56}")
        for _, row in top_coint.iterrows():
            print(f"  {row['algo1']:<12}{row['algo2']:<12}"
                  f"{row['adf_stat']:<10.3f}{row['hedge_ratio']:<10.4f}"
                  f"{row['half_life_days']:<12.1f}d")

    print(f"\nAll results saved to: {args.output}/")
    print("Done!")


if __name__ == '__main__':
    main()
