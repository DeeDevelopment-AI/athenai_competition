#!/usr/bin/env python3
"""
=================================================================
BENCHMARK ASSET ANALYSIS + SP500 REGIME DETECTION
=================================================================
Part 1: Analyze what kind of assets the benchmark selects
  - Cross-reference trades with asset inference
  - Time-varying asset class composition
  - Selection bias analysis (does it pick better algos?)

Part 2: SP500 regime detection model (Investment Clock)
  - Isolate SP500-correlated algos
  - Detect market regimes using 4-state Investment Clock:
      * Expansion  - Growth phase, positive trend, moderate volatility
      * Peak       - Late cycle, momentum fading, volatility rising
      * Contraction - Correction/recession, negative trend, high volatility
      * Recovery   - Early cycle rebound, improving momentum from lows
  - Analyze how the benchmark rotates into/out of SP500 algos
  - Build a predictive regime model

Usage:
  python3 benchmark_asset_regime.py \
    --trades trades_benchmark.csv \
    --inference results/asset_inference_all.csv \
    --metrics results/metrics_all.csv \
    --clusters results/clusters.csv \
    --benchmarks ./benchmarks/ \
    --algos ./algorithms/ \
    --output results/regime/
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import sys
import glob
import argparse
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_paths import default_output_dir, notebook_data_path, raw_benchmark_path, raw_path
try:
    from algo_pipeline import load_algorithm_csv, trim_dead_tail, load_all_benchmarks
except:
    pass


# ============================================================
# PART 1: BENCHMARK ASSET COMPOSITION ANALYSIS
# ============================================================

def analyze_benchmark_composition(trades_path, inference_path, metrics_path, clusters_path=None):
    """
    Cross-reference benchmark trades with asset inference to understand
    what the benchmark invests in and how its composition evolves.
    """
    print("\n  Loading data...")
    trades = pd.read_csv(trades_path)
    trades['dateOpen'] = pd.to_datetime(trades['dateOpen'], format='mixed', utc=True).dt.tz_localize(None)
    trades['dateClose'] = pd.to_datetime(trades['dateClose'], format='mixed', utc=True).dt.tz_localize(None)
    trades = trades.rename(columns={'productname': 'algo'})

    inference = pd.read_csv(inference_path)
    metrics = pd.read_csv(metrics_path)

    clusters = None
    if clusters_path and os.path.exists(clusters_path):
        clusters = pd.read_csv(clusters_path)

    # --- Which algos does the benchmark use? ---
    bench_algos = set(trades['algo'].unique())
    print(f"  Benchmark uses {len(bench_algos)} unique algos")

    # Merge inference data
    bench_inf = inference[inference['name'].isin(bench_algos)].copy()
    all_inf = inference.copy()

    print(f"  Inference available for {len(bench_inf)}/{len(bench_algos)} benchmark algos")

    # --- 1. Asset class distribution in benchmark ---
    print("\n  === BENCHMARK ASSET CLASS DISTRIBUTION ===")
    if 'asset_class' in bench_inf.columns:
        bench_classes = bench_inf['asset_class'].value_counts()
        total = len(bench_inf)
        for cls_name, cnt in bench_classes.items():
            print(f"    {cls_name:<15} {cnt:>4} algos ({cnt / total * 100:.1f}%)")

    # Compare with overall universe
    print("\n  === VS OVERALL UNIVERSE ===")
    if 'asset_class' in all_inf.columns:
        print(f"  {'Asset Class':<15} {'Benchmark':>12} {'Universe':>12} {'Over/Under':>12}")
        print(f"  {'-' * 54}")
        bench_total = len(bench_inf)
        uni_total = len(all_inf)
        for cls_name in all_inf['asset_class'].unique():
            b_pct = (bench_inf['asset_class'] == cls_name).sum() / bench_total * 100 if bench_total > 0 else 0
            u_pct = (all_inf['asset_class'] == cls_name).sum() / uni_total * 100 if uni_total > 0 else 0
            diff = b_pct - u_pct
            arrow = '↑' if diff > 2 else '↓' if diff < -2 else '~'
            print(f"  {cls_name:<15} {b_pct:>10.1f}% {u_pct:>10.1f}% {diff:>+10.1f}% {arrow}")

    # --- 2. Volume-weighted asset allocation ---
    print("\n  === VOLUME-WEIGHTED ALLOCATION ===")
    trades_with_inf = trades.merge(bench_inf[['name', 'asset_class', 'predicted_asset', 'confidence']],
                                   left_on='algo', right_on='name', how='left')

    vol_by_class = trades_with_inf.groupby('asset_class')['volume'].sum()
    vol_total = vol_by_class.sum()
    for cls_name, vol in vol_by_class.sort_values(ascending=False).items():
        print(f"    {cls_name:<15} {vol:>12,.0f} units ({vol / vol_total * 100:.1f}%)")

    # --- 3. Top predicted assets by volume ---
    print("\n  === TOP PREDICTED ASSETS (by volume) ===")
    if 'predicted_asset' in trades_with_inf.columns:
        vol_by_asset = trades_with_inf.groupby('predicted_asset')['volume'].sum()
        for asset, vol in vol_by_asset.sort_values(ascending=False).head(15).items():
            pct = vol / vol_total * 100
            print(f"    {asset:<25} {vol:>12,.0f} ({pct:.1f}%)")

    # --- 4. Metrics: benchmark vs universe ---
    print("\n  === BENCHMARK ALGO QUALITY VS UNIVERSE ===")
    bench_metrics = metrics[metrics['name'].isin(bench_algos)]

    compare_cols = ['annualized_return_pct', 'annualized_volatility_pct', 'sharpe_ratio',
                    'max_drawdown_pct', 'win_rate_pct']
    print(f"  {'Metric':<25} {'Bench Mean':>12} {'Bench Med':>12} {'Univ Mean':>12} {'Univ Med':>12}")
    print(f"  {'-' * 74}")
    for col in compare_cols:
        if col in bench_metrics.columns and col in metrics.columns:
            bm = bench_metrics[col].mean()
            bmed = bench_metrics[col].median()
            um = metrics[col].mean()
            umed = metrics[col].median()
            print(f"  {col:<25} {bm:>12.2f} {bmed:>12.2f} {um:>12.2f} {umed:>12.2f}")

    # --- 5. Time-varying composition ---
    print("\n  Computing time-varying asset allocation...")
    trades_with_inf['open_month'] = trades_with_inf['dateOpen'].dt.to_period('M')
    monthly_alloc = trades_with_inf.groupby(['open_month', 'asset_class'])['volume'].sum().unstack(fill_value=0)
    monthly_pct = monthly_alloc.div(monthly_alloc.sum(axis=1), axis=0) * 100

    # --- 6. Confidence distribution ---
    print("\n  === INFERENCE CONFIDENCE FOR BENCHMARK ALGOS ===")
    if 'confidence' in bench_inf.columns:
        conf = bench_inf['confidence']
        print(f"    Mean confidence: {conf.mean():.1f}%")
        print(f"    Median confidence: {conf.median():.1f}%")
        print(f"    High confidence (>60%): {(conf > 60).sum()} algos")
        print(f"    Moderate (30-60%): {((conf >= 30) & (conf <= 60)).sum()} algos")
        print(f"    Low (<30%): {(conf < 30).sum()} algos")

    # --- 7. Cluster distribution ---
    cluster_info = None
    if clusters is not None and 'cluster' in clusters.columns:
        print("\n  === CLUSTER DISTRIBUTION IN BENCHMARK ===")
        bench_clusters = clusters[clusters['name'].isin(bench_algos)]
        if len(bench_clusters) > 0:
            bc = bench_clusters['cluster'].value_counts().sort_index()
            for cl, cnt in bc.items():
                pct = cnt / len(bench_clusters) * 100
                print(f"    Cluster {cl}: {cnt} algos ({pct:.1f}%)")
            cluster_info = bc.to_dict()

    return {
        'bench_algos': list(bench_algos),
        'asset_distribution': bench_classes.to_dict() if 'asset_class' in bench_inf.columns else {},
        'volume_by_class': vol_by_class.to_dict(),
        'monthly_allocation_pct': monthly_pct.to_dict() if len(monthly_pct) > 0 else {},
        'cluster_distribution': cluster_info,
    }


# ============================================================
# PART 2: SP500 REGIME DETECTION
# ============================================================

def load_sp500_daily(bench_dir):
    """Load SP500 daily data from benchmarks directory.

    Tries multiple sources in order:
      1. DAT_ASCII spot data (indices/DAT_ASCII_SPXUSD_M1_*.csv)
      2. ES futures continuous series (futures/*.csv)
    """
    # --- Try DAT_ASCII spot data first ---
    patterns = [
        os.path.join(bench_dir, 'indices', 'DAT_ASCII_SPXUSD_M1_*.csv'),
        os.path.join(bench_dir, 'DAT_ASCII_SPXUSD_M1_*.csv'),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=';')
            # Normalize column names to lowercase
            df.columns = [c.strip().lower() for c in df.columns]

            # Find datetime column
            dt_col = None
            for candidate in ['gmt time', 'datetime', 'date', 'time', 'timestamp']:
                if candidate in df.columns:
                    dt_col = candidate
                    break
            if dt_col is None:
                dt_col = df.columns[0]

            # Parse datetime
            parsed = pd.to_datetime(df[dt_col], format='%Y%m%d %H%M%S', errors='coerce')
            if parsed.isna().sum() > len(parsed) * 0.5:
                parsed = pd.to_datetime(df[dt_col], errors='coerce')
            df['datetime'] = parsed
            df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()

            # Ensure OHLC columns exist
            for c in ['open', 'high', 'low', 'close']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            if 'close' in df.columns:
                frames.append(df[['open', 'high', 'low', 'close']])
        except Exception as e:
            print(f"  WARNING: Failed to load {f}: {e}")

    if frames:
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        daily = combined.resample('D').agg({'open': 'first', 'high': 'max',
                                            'low': 'min', 'close': 'last'}).dropna()
        print(f"  SP500 (spot): {daily.index[0].date()} -> {daily.index[-1].date()}, {len(daily)} days")
        return daily

    # --- Fallback: Try ES futures from futures folder ---
    futures_dir = os.path.join(bench_dir, 'futures')
    if os.path.isdir(futures_dir):
        print("  No SP500 spot data, trying ES futures...")
        try:
            # Import the futures loader from algo_pipeline
            from algo_pipeline import _load_futures_benchmarks
            fut_loaded = _load_futures_benchmarks(futures_dir)
            if 'FUT_SP500_Fut' in fut_loaded:
                daily, meta = fut_loaded['FUT_SP500_Fut']
                print(f"  SP500 (ES futures): {daily.index[0].date()} -> {daily.index[-1].date()}, {len(daily)} days")
                return daily[['open', 'high', 'low', 'close']]
        except Exception as e:
            print(f"  WARNING: Failed to load ES futures: {e}")

    print("  WARNING: No SP500 data found")
    return None


def build_sp500_features(sp500_daily):
    """
    Build feature set from SP500 price data for regime detection.

    Features:
      - Returns at multiple horizons (1d, 5d, 20d, 60d)
      - Volatility at multiple horizons
      - Momentum indicators
      - Trend indicators
      - Mean reversion indicators
      - Volatility regime indicators
    """
    close = sp500_daily['close']
    high = sp500_daily['high']
    low = sp500_daily['low']

    features = pd.DataFrame(index=sp500_daily.index)

    # --- Returns ---
    features['ret_1d'] = np.log(close / close.shift(1))
    features['ret_5d'] = np.log(close / close.shift(5))
    features['ret_20d'] = np.log(close / close.shift(20))
    features['ret_60d'] = np.log(close / close.shift(60))

    # --- Volatility ---
    for w in [5, 10, 20, 60]:
        features[f'vol_{w}d'] = features['ret_1d'].rolling(w).std() * np.sqrt(252)

    # Volatility ratio (short/long)
    features['vol_ratio_5_20'] = features['vol_5d'] / features['vol_20d'].replace(0, np.nan)
    features['vol_ratio_10_60'] = features['vol_10d'] / features['vol_60d'].replace(0, np.nan)

    # --- Momentum ---
    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        features[f'price_vs_sma_{w}'] = (close - sma) / sma

    # SMA crossovers
    features['sma_5_20_cross'] = (close.rolling(5).mean() - close.rolling(20).mean()) / close
    features['sma_20_50_cross'] = (close.rolling(20).mean() - close.rolling(50).mean()) / close

    # --- RSI (14-day) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # --- Bollinger Band position ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['bb_position'] = (close - sma20) / (2 * std20).replace(0, np.nan)

    # --- Average True Range (normalized) ---
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    features['atr_14_norm'] = tr.rolling(14).mean() / close

    # --- Drawdown from peak ---
    rolling_max = close.rolling(252, min_periods=1).max()
    features['drawdown_from_peak'] = (close - rolling_max) / rolling_max

    # --- Distance from 52-week high/low ---
    features['dist_52w_high'] = (close - close.rolling(252, min_periods=60).max()) / close
    features['dist_52w_low'] = (close - close.rolling(252, min_periods=60).min()) / close

    # --- Returns skewness (rolling) ---
    features['skew_20d'] = features['ret_1d'].rolling(20).skew()

    return features.dropna()


def define_regimes(sp500_daily, features):
    """
    Define market regimes based on SP500 behavior using Investment Clock model.

    Regimes (4-state Investment Clock):
      1 = Expansion  - Economy/market growing, positive trend, moderate volatility
      2 = Peak       - Late cycle, trend slowing, volatility rising, near highs
      3 = Contraction - Recession/correction, negative trend, high volatility
      4 = Recovery   - Early cycle, rebounding from lows, improving momentum

    The cycle typically flows: Expansion -> Peak -> Contraction -> Recovery -> Expansion
    """
    close = sp500_daily['close'].reindex(features.index)

    # Compute signals
    ret_5d = np.log(close / close.shift(5))
    ret_20d = np.log(close / close.shift(20))
    ret_60d = np.log(close / close.shift(60))
    vol_20d = features['vol_20d']
    vol_60d = features['vol_60d']
    dd = features['drawdown_from_peak']

    # Momentum acceleration/deceleration
    momentum_accel = ret_20d - ret_60d / 3  # Is short-term momentum > long-term?

    # Volatility regime
    vol_median = vol_20d.median()
    vol_high = vol_20d.quantile(0.75)
    vol_low = vol_20d.quantile(0.25)

    # Volatility trend (is vol rising or falling?)
    vol_change = vol_20d - vol_20d.shift(10)

    # Distance from 52-week high
    dist_from_high = features['dist_52w_high']

    # Initialize all as expansion (default state)
    regimes = pd.Series('expansion', index=features.index)

    # === CONTRACTION ===
    # Negative trend + significant drawdown OR high volatility with negative returns
    contraction_mask = (
        ((ret_20d < -0.02) & (ret_60d < 0)) |  # Negative trend
        ((dd < -0.10) & (ret_5d < 0)) |         # Deep drawdown, still falling
        ((vol_20d > vol_high) & (ret_20d < -0.03))  # High vol + negative returns
    )
    regimes[contraction_mask] = 'contraction'

    # === RECOVERY ===
    # Coming up from drawdown, positive short-term momentum, vol normalizing
    recovery_mask = (
        (dd < -0.05) &                    # Still in drawdown territory
        (ret_5d > 0.005) &                # Positive short-term momentum
        (ret_20d > ret_60d / 3) &         # Momentum improving (less negative or positive)
        (~contraction_mask)               # Not in active contraction
    )
    regimes[recovery_mask] = 'recovery'

    # === PEAK ===
    # Near highs but momentum fading, volatility rising
    peak_mask = (
        (dist_from_high > -0.05) &        # Within 5% of 52-week high
        (
            (momentum_accel < -0.01) |    # Momentum decelerating
            (vol_change > 0.02) |         # Volatility rising significantly
            ((ret_5d < 0) & (ret_60d > 0.05))  # Short-term weakness in uptrend
        ) &
        (~contraction_mask) &
        (~recovery_mask)
    )
    regimes[peak_mask] = 'peak'

    # === EXPANSION ===
    # Positive trend, moderate volatility, not in other states
    # (Already the default, but let's be explicit for remaining cases)
    expansion_mask = (
        (ret_20d > 0.01) &                # Positive trend
        (ret_60d > 0) &                   # Longer-term also positive
        (vol_20d <= vol_high) &           # Volatility not extreme
        (~contraction_mask) &
        (~recovery_mask) &
        (~peak_mask)
    )
    regimes[expansion_mask] = 'expansion'

    # Handle any remaining edge cases (assign to nearest logical state)
    remaining_mask = ~(contraction_mask | recovery_mask | peak_mask | expansion_mask)
    # If vol is high, lean toward peak/contraction based on trend
    regimes.loc[remaining_mask & (vol_20d > vol_median) & (ret_20d < 0)] = 'contraction'
    regimes.loc[remaining_mask & (vol_20d > vol_median) & (ret_20d >= 0)] = 'peak'
    regimes.loc[remaining_mask & (vol_20d <= vol_median) & (ret_20d < 0)] = 'recovery'
    # Remaining positives stay as expansion

    # Stats
    print(f"\n  Investment Clock Regime Distribution:")
    regime_order = ['expansion', 'peak', 'contraction', 'recovery']
    for regime in regime_order:
        cnt = (regimes == regime).sum()
        pct = cnt / len(regimes) * 100
        # Avg return in this regime
        regime_ret = features.loc[regimes == regime, 'ret_1d'].mean() * 252
        regime_vol = features.loc[regimes == regime, 'vol_20d'].mean()
        regime_dd = features.loc[regimes == regime, 'drawdown_from_peak'].mean()
        print(f"    {regime:<12} {cnt:>5} days ({pct:>5.1f}%) | "
              f"ann_ret={regime_ret:>+7.1f}% | avg_vol={regime_vol:.1%} | avg_dd={regime_dd:>+.1%}")

    # Transition analysis
    print(f"\n  Regime Transitions:")
    regime_shifted = regimes.shift(1)
    for from_regime in regime_order:
        transitions = []
        from_mask = regime_shifted == from_regime
        for to_regime in regime_order:
            if from_regime == to_regime:
                continue
            cnt = ((from_mask) & (regimes == to_regime)).sum()
            if cnt > 0:
                transitions.append(f"{to_regime}({cnt})")
        if transitions:
            print(f"    {from_regime:<12} -> {', '.join(transitions)}")

    return regimes


def analyze_benchmark_sp500_rotation(trades_path, inference_path, sp500_regimes):
    """
    Analyze how the benchmark rotates its SP500-correlated algos
    across different market regimes.
    """
    trades = pd.read_csv(trades_path)
    trades['dateOpen'] = pd.to_datetime(trades['dateOpen'], format='mixed', utc=True).dt.tz_localize(None)
    trades['dateClose'] = pd.to_datetime(trades['dateClose'], format='mixed', utc=True).dt.tz_localize(None)
    trades['open_day'] = trades['dateOpen'].dt.normalize()
    trades = trades.rename(columns={'productname': 'algo'})

    inference = pd.read_csv(inference_path)

    # Identify SP500-related algos
    sp500_algos = set()
    if 'predicted_asset' in inference.columns:
        sp500_mask = inference['predicted_asset'].str.contains('SP500|EQ_', na=False, case=False)
        sp500_algos = set(inference.loc[sp500_mask, 'name'])

    if 'asset_class' in inference.columns:
        idx_mask = inference['asset_class'].isin(['indices', 'equity', 'etf'])
        sp500_algos |= set(inference.loc[idx_mask, 'name'])

    print(f"\n  SP500-related algos in benchmark: {len(sp500_algos & set(trades['algo']))}")

    # Tag trades by regime at open
    trades['regime'] = trades['open_day'].map(sp500_regimes)
    trades['is_sp500'] = trades['algo'].isin(sp500_algos)

    # Allocation to SP500 algos by regime
    print("\n  === SP500 ALLOCATION BY REGIME ===")
    regime_alloc = trades.groupby('regime').apply(
        lambda g: pd.Series({
            'total_trades': len(g),
            'sp500_trades': g['is_sp500'].sum(),
            'sp500_volume': g.loc[g['is_sp500'], 'volume'].sum(),
            'total_volume': g['volume'].sum(),
            'sp500_pct_trades': g['is_sp500'].mean() * 100,
            'sp500_pct_volume': (g.loc[g['is_sp500'], 'volume'].sum() / g['volume'].sum() * 100
                                 if g['volume'].sum() > 0 else 0),
        })
    )

    if len(regime_alloc) > 0:
        for regime, row in regime_alloc.iterrows():
            print(f"    {regime:<12} trades={int(row['total_trades']):>5} | "
                  f"SP500 trades={row['sp500_pct_trades']:.1f}% | "
                  f"SP500 volume={row['sp500_pct_volume']:.1f}%")

    return regime_alloc


def train_regime_model(features, regimes, sp500_daily):
    """
    Train a regime detection model using SP500 features.
    Uses time-series cross-validation to avoid lookahead bias.
    """
    print("\n  === TRAINING REGIME DETECTION MODEL ===")

    # Align
    common = features.index.intersection(regimes.index)
    X = features.loc[common]
    y = regimes.loc[common]

    # Remove any remaining NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    print(f"  Training data: {len(X)} samples, {X.shape[1]} features")
    print(f"  Regime counts: {y.value_counts().to_dict()}")

    # Encode labels
    regime_labels = sorted(y.unique())
    label_map = {r: i for i, r in enumerate(regime_labels)}
    y_encoded = y.map(label_map)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # --- Model 1: Random Forest ---
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                class_weight='balanced', random_state=42, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=5)
    rf_scores = cross_val_score(rf, X_scaled, y_encoded, cv=tscv, scoring='f1_weighted')
    print(f"    CV F1 (weighted): {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")

    # --- Model 2: Gradient Boosting ---
    print("\n  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=20, random_state=42)
    gb_scores = cross_val_score(gb, X_scaled, y_encoded, cv=tscv, scoring='f1_weighted')
    print(f"    CV F1 (weighted): {gb_scores.mean():.4f} +/- {gb_scores.std():.4f}")

    # Pick best model
    if rf_scores.mean() >= gb_scores.mean():
        best_model = rf
        best_name = 'RandomForest'
        best_scores = rf_scores
    else:
        best_model = gb
        best_name = 'GradientBoosting'
        best_scores = gb_scores

    print(f"\n  Best model: {best_name} (F1={best_scores.mean():.4f})")

    # Fit on full data for feature importance and prediction
    best_model.fit(X_scaled, y_encoded)

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print(f"\n  Top 15 features:")
    for feat, imp in importances.head(15).items():
        print(f"    {feat:<25} {imp:.4f}")

    # Full dataset prediction for analysis
    y_pred = best_model.predict(X_scaled)
    y_pred_labels = pd.Series([regime_labels[i] for i in y_pred], index=X.index)

    # Classification report
    print(f"\n  Classification report (full dataset, for diagnostics):")
    print(classification_report(y_encoded, y_pred, target_names=regime_labels))

    # Confusion matrix
    cm = confusion_matrix(y_encoded, y_pred)
    cm_df = pd.DataFrame(cm, index=regime_labels, columns=regime_labels)

    # Current regime prediction
    if len(X_scaled) > 0:
        latest_features = X_scaled.iloc[-1:]
        current_pred = best_model.predict(latest_features)[0]
        current_regime = regime_labels[current_pred]
        current_proba = best_model.predict_proba(latest_features)[0]

        print(f"\n  CURRENT REGIME PREDICTION ({X.index[-1].date()}):")
        print(f"    Predicted: {current_regime}")
        for i, r in enumerate(regime_labels):
            print(f"    P({r}): {current_proba[i]:.3f}")

    # Regime transition matrix
    transitions = pd.DataFrame(0, index=regime_labels, columns=regime_labels)
    for i in range(1, len(y)):
        prev = y.iloc[i - 1]
        curr = y.iloc[i]
        transitions.loc[prev, curr] += 1

    # Normalize rows
    trans_pct = transitions.div(transitions.sum(axis=1), axis=0) * 100

    print(f"\n  Regime transition probabilities (%):")
    print(f"  {'From \\ To':<12}", end='')
    for r in regime_labels:
        print(f" {r:>10}", end='')
    print()
    for r_from in regime_labels:
        print(f"  {r_from:<12}", end='')
        for r_to in regime_labels:
            print(f" {trans_pct.loc[r_from, r_to]:>9.1f}%", end='')
        print()

    # Average regime duration
    print(f"\n  Average regime duration:")
    regime_changes = y != y.shift(1)
    regime_runs = []
    current_run = 1
    for i in range(1, len(y)):
        if y.iloc[i] == y.iloc[i - 1]:
            current_run += 1
        else:
            regime_runs.append((y.iloc[i - 1], current_run))
            current_run = 1
    regime_runs.append((y.iloc[-1], current_run))

    run_df = pd.DataFrame(regime_runs, columns=['regime', 'duration'])
    avg_duration = run_df.groupby('regime')['duration'].agg(['mean', 'median', 'max'])
    for regime, row in avg_duration.iterrows():
        print(f"    {regime:<12} avg={row['mean']:.0f}d  med={row['median']:.0f}d  max={row['max']}d")

    return {
        'model_name': best_name,
        'cv_f1_mean': round(float(best_scores.mean()), 4),
        'cv_f1_std': round(float(best_scores.std()), 4),
        'feature_importances': {k: round(float(v), 4) for k, v in importances.head(20).items()},
        'regime_labels': regime_labels,
        'confusion_matrix': cm_df.to_dict(),
        'transition_matrix_pct': trans_pct.to_dict(),
        'avg_duration': avg_duration.to_dict(),
        'predictions': y_pred_labels,
        'model': best_model,
        'scaler': scaler,
        'feature_columns': list(X.columns),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark asset analysis + SP500 regime detection')
    parser.add_argument('--trades', default=str(raw_benchmark_path('trades_benchmark.csv')))
    parser.add_argument('--inference', default=str(notebook_data_path('pipeline', 'asset_inference_all.csv')),
                        help='asset_inference_all.csv')
    parser.add_argument('--metrics', default=str(notebook_data_path('pipeline', 'metrics_all.csv')),
                        help='metrics_all.csv')
    parser.add_argument('--clusters', default=str(notebook_data_path('clusters', 'clusters.csv')),
                        help='clusters.csv (optional)')
    parser.add_argument('--benchmarks', default=str(raw_path()), help='benchmarks directory')
    parser.add_argument('--output', default=str(default_output_dir('regime')))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- PART 1: Asset composition ----
    print("\n" + "=" * 60)
    print("  PART 1: BENCHMARK ASSET COMPOSITION")
    print("=" * 60)

    composition = analyze_benchmark_composition(
        args.trades, args.inference, args.metrics, args.clusters)

    # ---- PART 2: SP500 regime detection ----
    print("\n" + "=" * 60)
    print("  PART 2: SP500 REGIME DETECTION")
    print("=" * 60)

    print("\n  Loading SP500 data...")
    sp500 = load_sp500_daily(args.benchmarks)

    if sp500 is None:
        print("  ERROR: Cannot load SP500 data. Skipping regime analysis.")
        return

    print("\n  Building features...")
    features = build_sp500_features(sp500)
    print(f"  Features: {features.shape[1]} columns, {len(features)} days")

    print("\n  Defining regimes...")
    regimes = define_regimes(sp500, features)

    print("\n  Analyzing benchmark rotation during regimes...")
    rotation = analyze_benchmark_sp500_rotation(args.trades, args.inference, regimes)

    print("\n  Training regime model...")
    model_results = train_regime_model(features, regimes, sp500)

    # ---- Save everything ----
    print("\n  Saving results...")

    # Regime predictions
    pred_df = pd.DataFrame({
        'date': model_results['predictions'].index,
        'actual_regime': regimes.reindex(model_results['predictions'].index),
        'predicted_regime': model_results['predictions'].values,
    })
    pred_df.to_csv(os.path.join(args.output, 'regime_predictions.csv'), index=False)

    # Features + regimes (for further analysis)
    feat_regime = features.copy()
    feat_regime['regime'] = regimes
    feat_regime.to_csv(os.path.join(args.output, 'sp500_features_regimes.csv'))

    # Rotation analysis
    if isinstance(rotation, pd.DataFrame):
        rotation.to_csv(os.path.join(args.output, 'benchmark_rotation_by_regime.csv'))

    # Full report
    report = {
        'composition': {
            'asset_distribution': composition['asset_distribution'],
            'volume_by_class': composition['volume_by_class'],
        },
        'regime_model': {
            'model': model_results['model_name'],
            'cv_f1': model_results['cv_f1_mean'],
            'feature_importances': model_results['feature_importances'],
            'regime_labels': model_results['regime_labels'],
            'transition_matrix': model_results['transition_matrix_pct'],
        },
    }
    with open(os.path.join(args.output, 'regime_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  All saved to: {args.output}/")
    print("Done!")


if __name__ == '__main__':
    main()
