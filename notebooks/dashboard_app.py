#!/usr/bin/env python3
"""
=================================================================
ALGO ANALYZER — Streamlit Dashboard
=================================================================
Interactive dashboard for exploring algorithm analysis results.

Run:
  pip install streamlit plotly
  streamlit run dashboard_app.py -- --results ./results/

Reads all output CSVs/JSONs from the pipeline scripts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import argparse

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Algo Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Parse results directory from command line
DEFAULT_DIR = "./results"
for i, arg in enumerate(sys.argv):
    if arg == "--results" and i + 1 < len(sys.argv):
        DEFAULT_DIR = sys.argv[i + 1]


def find_file(filename, search_dirs):
    """Search for a file in multiple directories."""
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def discover_results(base_dir):
    """
    Auto-discover result files by scanning base_dir, its parent,
    and ALL subdirectories up to 3 levels deep.
    Also scans sibling directories (other results_* folders).
    """
    known_files = [
        'metrics_all.csv', 'asset_inference_all.csv', 'classification_all.csv',
        'ranking.csv', 'style_analysis_all.csv', 'error_log.txt',
        'clusters.csv', 'cluster_profiles.csv', 'cluster_analysis.json', 'period_stats.csv',
        'algo_pairs.csv', 'algo_cointegration.csv', 'algo_diversifiers.csv',
        'algo_correlation_matrix.csv', 'algo_relationships.json',
        'reconstruction_daily.csv', 'reconstruction_results.csv', 'reconstruction_report.json',
        'frequency_comparison.csv', 'drawdown_comparison.csv', 'drawdown_curves.csv',
        'regime_analysis.csv', 'monthly_returns_comparison.csv', 'comparison_report.json',
        'regime_predictions.csv', 'sp500_features_regimes.csv',
        'benchmark_rotation_by_regime.csv', 'regime_report.json',
        'daily_fund_timeline.csv', 'monthly_fund_evolution.csv',
        'cashflow_events.csv', 'cashflow_report.json',
        'dietz_reconstruction_daily.csv', 'dietz_reconstruction_results.csv',
        'dietz_reconstruction_report.json', 'dietz_monthly_returns.csv',
        'dietz_comparison_results.csv', 'dietz_phase_analysis.csv',
        'dietz_monthly_comparison.csv', 'dietz_comparison_report.json',
        'phase_composition.csv', 'phase_top_algos.csv',
        'phase_cluster_allocation.csv', 'phase_quality_comparison.csv',
        'composition_report.json',
    ]

    # Collect all directories to search
    search_dirs = set()

    # The specified base_dir
    search_dirs.add(os.path.abspath(base_dir))

    # Parent of base_dir (catches sibling folders like results_recon/)
    parent = os.path.dirname(os.path.abspath(base_dir))
    search_dirs.add(parent)

    # All subdirs of parent (sibling result folders)
    if os.path.isdir(parent):
        for entry in os.listdir(parent):
            full = os.path.join(parent, entry)
            if os.path.isdir(full):
                search_dirs.add(full)
                # 1 level deeper
                try:
                    for sub in os.listdir(full):
                        subfull = os.path.join(full, sub)
                        if os.path.isdir(subfull):
                            search_dirs.add(subfull)
                            # 2 levels deeper
                            try:
                                for sub2 in os.listdir(subfull):
                                    sub2full = os.path.join(subfull, sub2)
                                    if os.path.isdir(sub2full):
                                        search_dirs.add(sub2full)
                            except PermissionError:
                                pass
                except PermissionError:
                    pass

    # Filter to only dirs that look like result dirs (avoid scanning huge dirs)
    # Keep dirs that contain at least one .csv or .json file
    valid_dirs = []
    for d in search_dirs:
        try:
            entries = os.listdir(d)
            if any(e.endswith('.csv') or e.endswith('.json') for e in entries):
                valid_dirs.append(d)
        except (PermissionError, FileNotFoundError):
            pass

    found = {}
    for filename in known_files:
        path = find_file(filename, valid_dirs)
        if path:
            found[filename] = path

    return found


# Run discovery
FILE_MAP = discover_results(DEFAULT_DIR)


# ============================================================
# DATA LOADING (cached)
# ============================================================

@st.cache_data
def load_csv(path, **kwargs):
    if os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    return None


@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def safe_load(filename, **kwargs):
    """Load a CSV by filename, auto-discovering its path."""
    path = FILE_MAP.get(filename)
    if path:
        return load_csv(path, **kwargs)
    return None


def safe_load_json(filename):
    path = FILE_MAP.get(filename)
    if path:
        return load_json(path)
    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📊 Algo Analyzer")
    st.caption("Fund-of-Algorithms Dashboard")

    page = st.radio("Navigation", [
        "🏠 Overview",
        "🔍 Algorithm Explorer",
        "🎯 Asset Inference",
        "🧩 Clustering",
        "🔗 Algo Relationships",
        "📈 Benchmark Reconstruction",
        "📊 Benchmark Comparison",
        "📐 Dietz Reconstruction",
        "💰 Cashflow Analysis",
        "🧬 Benchmark Composition",
        "🌡️ Regime Detection",
        "📖 Methodology",
    ])

    st.divider()
    st.caption(f"Scanning: `{DEFAULT_DIR}`")

    # Check which key files were found
    key_files = ['metrics_all.csv', 'asset_inference_all.csv', 'clusters.csv',
                 'algo_pairs.csv', 'reconstruction_daily.csv', 'regime_report.json',
                 'frequency_comparison.csv', 'drawdown_curves.csv']

    files_status = {f: f in FILE_MAP for f in key_files}
    n_ready = sum(files_status.values())
    st.progress(n_ready / len(files_status), text=f"{n_ready}/{len(files_status)} datasets found")

    with st.expander("File discovery", expanded=False):
        for f, found in files_status.items():
            icon = "✅" if found else "❌"
            path = FILE_MAP.get(f, "not found")
            st.caption(f"{icon} {f}")
            if found:
                st.caption(f"   → `{path}`")

# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "🏠 Overview":
    st.header("Overview")

    metrics = safe_load('metrics_all.csv')
    inference = safe_load('asset_inference_all.csv')
    ranking = safe_load('ranking.csv')

    if metrics is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Algorithms", f"{len(metrics):,}")
        c2.metric("Avg Return", f"{metrics['annualized_return_pct'].mean():.2f}%")
        c3.metric("Avg Sharpe", f"{metrics['sharpe_ratio'].mean():.3f}")
        c4.metric("Avg Max DD", f"{metrics['max_drawdown_pct'].mean():.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Return Distribution")
            fig = px.histogram(metrics, x='annualized_return_pct', nbins=50,
                               title="Annualized Return (%)",
                               color_discrete_sequence=['#534AB7'])
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Risk-Return Scatter")
            fig = px.scatter(metrics, x='annualized_volatility_pct', y='annualized_return_pct',
                             size=metrics['sharpe_ratio'].clip(0, 3).fillna(0.1) + 0.1,
                             hover_name='name',
                             title="Return vs Volatility",
                             color='sharpe_ratio',
                             color_continuous_scale='RdYlGn')
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')

        # Asset class distribution
        if inference is not None and 'asset_class' in inference.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Asset Class Distribution")
                ac_counts = inference['asset_class'].value_counts()
                fig = px.pie(values=ac_counts.values, names=ac_counts.index,
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

            with col2:
                st.subheader("Confidence Distribution")
                if 'confidence' in inference.columns:
                    fig = px.histogram(inference, x='confidence', nbins=30,
                                       title="Inference Confidence (%)",
                                       color_discrete_sequence=['#1D9E75'])
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, width='stretch')

        # Top 10 ranking
        if ranking is not None:
            st.subheader("Top 20 Algorithms")
            display_cols = ['rank', 'name', 'annualized_return_pct', 'sharpe_ratio',
                            'max_drawdown_pct', 'composite_score']
            if 'asset_label' in ranking.columns:
                display_cols.insert(3, 'asset_label')
            if 'confidence' in ranking.columns:
                display_cols.insert(4, 'confidence')
            top = ranking.head(20)[display_cols]
            st.dataframe(top, width='stretch', hide_index=True)
    else:
        st.warning("No metrics_all.csv found. Run `algo_pipeline.py` first.")


# ============================================================
# PAGE: ALGORITHM EXPLORER
# ============================================================

elif page == "🔍 Algorithm Explorer":
    st.header("Algorithm Explorer")

    metrics = safe_load('metrics_all.csv')
    inference = safe_load('asset_inference_all.csv')
    ranking = safe_load('ranking.csv')

    if metrics is None:
        st.warning("No metrics_all.csv found.")
    else:
        # Merge all data
        df = metrics.copy()
        if inference is not None:
            merge_cols = ['name'] + [c for c in inference.columns if c not in df.columns]
            df = df.merge(inference[merge_cols], on='name', how='left')
        if ranking is not None and 'composite_score' in ranking.columns:
            df = df.merge(ranking[['name', 'composite_score', 'rank']], on='name', how='left')

        # Filters
        st.subheader("Filters")
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            min_ret = st.number_input("Min Return %", value=-100.0, step=5.0)
        with fc2:
            max_vol = st.number_input("Max Volatility %", value=200.0, step=5.0)
        with fc3:
            min_sharpe = st.number_input("Min Sharpe", value=-10.0, step=0.1)
        with fc4:
            if 'asset_class' in df.columns:
                asset_filter = st.multiselect("Asset Class", df['asset_class'].dropna().unique())
            else:
                asset_filter = []

        # Apply filters
        mask = (df['annualized_return_pct'] >= min_ret) & \
               (df['annualized_volatility_pct'] <= max_vol) & \
               (df['sharpe_ratio'] >= min_sharpe)

        if asset_filter:
            mask &= df['asset_class'].isin(asset_filter)

        filtered = df[mask].sort_values('sharpe_ratio', ascending=False)

        st.caption(f"Showing {len(filtered):,} / {len(df):,} algorithms")

        # Display table
        display_cols = ['name', 'annualized_return_pct', 'annualized_volatility_pct',
                        'sharpe_ratio', 'max_drawdown_pct', 'duration_years']
        if 'asset_class' in filtered.columns:
            display_cols.append('asset_class')
        if 'confidence' in filtered.columns:
            display_cols.append('confidence')
        if 'composite_score' in filtered.columns:
            display_cols.append('composite_score')
        if 'was_trimmed' in filtered.columns:
            display_cols.append('was_trimmed')

        st.dataframe(filtered[display_cols].reset_index(drop=True),
                     width='stretch', hide_index=True, height=500)

        # Scatter plot of filtered
        st.subheader("Filtered Universe")
        color_col = 'asset_class' if 'asset_class' in filtered.columns else None
        fig = px.scatter(filtered, x='annualized_volatility_pct', y='annualized_return_pct',
                         hover_name='name', color=color_col,
                         size=filtered['sharpe_ratio'].clip(0, 3).fillna(0.1) + 0.1,
                         title=f"Return vs Volatility ({len(filtered)} algos)")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')


# ============================================================
# PAGE: ASSET INFERENCE
# ============================================================

elif page == "🎯 Asset Inference":
    st.header("Asset Inference")

    inference = safe_load('asset_inference_all.csv')

    if inference is None:
        st.warning("No asset_inference_all.csv found.")
    else:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed", len(inference))
        c2.metric("Identified", f"{(inference['asset_class'] != 'unknown').sum()}")
        c3.metric("Avg Confidence", f"{inference['confidence'].mean():.1f}%")
        c4.metric("High Conf (>60%)", f"{(inference['confidence'] > 60).sum()}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Asset Class Distribution")
            ac = inference['asset_class'].value_counts()
            fig = px.bar(x=ac.index, y=ac.values, color=ac.index,
                         title="Algorithms by Asset Class",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, height=400, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Confidence by Asset Class")
            if 'confidence' in inference.columns:
                fig = px.box(inference[inference['asset_class'] != 'unknown'],
                             x='asset_class', y='confidence',
                             color='asset_class',
                             title="Confidence Distribution per Class",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, width='stretch')

        # Direction breakdown
        if 'direction' in inference.columns:
            st.subheader("Direction Analysis")
            col1, col2 = st.columns(2)
            with col1:
                dir_counts = inference['direction'].value_counts()
                fig = px.pie(values=dir_counts.values, names=dir_counts.index,
                             title="Trade Direction")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

            with col2:
                if 'trading_pattern' in inference.columns:
                    tp = inference['trading_pattern'].value_counts()
                    fig = px.bar(x=tp.index, y=tp.values,
                                 title="Trading Patterns",
                                 color_discrete_sequence=['#D85A30'])
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, width='stretch')

        # Multi-exposure analysis
        exp_cols = [c for c in inference.columns if c.startswith('exposure_1_')]
        if exp_cols:
            st.subheader("Primary Exposures (Top Match per Algo)")
            if 'exposure_1_name' in inference.columns:
                top_exp = inference['exposure_1_name'].dropna().value_counts().head(20)
                fig = px.bar(x=top_exp.values, y=top_exp.index, orientation='h',
                             title="Most Common Primary Exposures",
                             color_discrete_sequence=['#534AB7'])
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, width='stretch')

        # Detailed table
        st.subheader("Full Inference Table")
        display_cols = ['name', 'asset_class', 'predicted_asset', 'direction',
                        'confidence', 'best_composite', 'n_significant_exposures',
                        'active_days_pct', 'trading_pattern']
        display_cols = [c for c in display_cols if c in inference.columns]
        st.dataframe(inference[display_cols].sort_values('confidence', ascending=False),
                     width='stretch', hide_index=True, height=500)


# ============================================================
# PAGE: CLUSTERING
# ============================================================

elif page == "🧩 Clustering":
    st.header("Clustering Analysis")

    clusters = safe_load('clusters.csv')
    profiles = safe_load('cluster_profiles.csv')
    cluster_json = safe_load_json('cluster_analysis.json')

    if clusters is None:
        st.warning("No clusters.csv found. Run `algo_cluster.py` first.")
    else:
        n_clusters = clusters['cluster'].nunique()
        st.metric("Number of Clusters", n_clusters)

        # PCA scatter
        if 'pca_x' in clusters.columns and 'pca_y' in clusters.columns:
            st.subheader("Cluster Map (PCA)")
            clusters['cluster_str'] = clusters['cluster'].astype(str)
            fig = px.scatter(clusters, x='pca_x', y='pca_y',
                             color='cluster_str', hover_name='name',
                             title="Algorithms in PCA Space (colored by cluster)",
                             color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(height=550)
            st.plotly_chart(fig, width='stretch')

        # Cluster profiles
        if profiles is not None:
            st.subheader("Cluster Profiles")

            metric_cols = [c for c in profiles.columns if c.endswith('_mean')
                           and not c.startswith('SP500') and not c.startswith('Gold')
                           and not c.startswith('EURUSD')]

            if 'label' in profiles.columns:
                for _, row in profiles.iterrows():
                    with st.expander(f"Cluster {int(row['cluster'])}: {row['label']} "
                                     f"({int(row['n_algorithms'])} algos, {row['pct_of_total']:.1f}%)"):
                        cols = st.columns(4)
                        for i, mc in enumerate(metric_cols[:8]):
                            display_name = mc.replace('_mean', '').replace('_', ' ').title()
                            cols[i % 4].metric(display_name, f"{row[mc]:.3f}")

            # Radar chart
            if cluster_json and 'radar' in cluster_json:
                st.subheader("Cluster Radar Comparison")
                radar = cluster_json['radar']
                fig = go.Figure()
                for cid, values in radar.get('data', {}).items():
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=radar['labels'] + [radar['labels'][0]],
                        name=f"Cluster {cid}",
                        fill='toself', opacity=0.3
                    ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                  height=500)
                st.plotly_chart(fig, width='stretch')


# ============================================================
# PAGE: ALGO RELATIONSHIPS
# ============================================================

elif page == "🔗 Algo Relationships":
    st.header("Algorithm Relationships")

    pairs = safe_load('algo_pairs.csv')
    coint = safe_load('algo_cointegration.csv')
    diversifiers = safe_load('algo_diversifiers.csv')
    rel_json = safe_load_json('algo_relationships.json')

    if pairs is None:
        st.warning("No algo_pairs.csv found. Run `algo_relationships.py` first.")
    else:
        # Summary stats
        if rel_json and 'stats' in rel_json:
            stats = rel_json['stats']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Algorithms", stats.get('n_algorithms', 0))
            c2.metric("Valid Pairs", f"{stats.get('n_valid_pairs', 0):,}")
            c3.metric("Mean Correlation", f"{stats.get('mean_correlation', 0):.4f}")
            c4.metric("Strong Positive (>0.5)", f"{stats.get('pct_strong_pos', 0):.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Correlation Distribution")
            if rel_json and 'stats' in rel_json:
                dist = rel_json['stats'].get('corr_distribution', {})
                if dist.get('counts') and dist.get('edges'):
                    edges = dist['edges']
                    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
                    fig = px.bar(x=mids, y=dist['counts'],
                                 title="Pairwise Correlation Distribution",
                                 color_discrete_sequence=['#534AB7'])
                    fig.update_layout(xaxis_title="Pearson r", yaxis_title="Count",
                                      showlegend=False, height=400)
                    st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Top Correlated Pairs")
            if len(pairs) > 0:
                top_pairs = pairs.head(15)
                fig = px.bar(top_pairs, x='pearson_r', y=top_pairs['algo1'] + ' ↔ ' + top_pairs['algo2'],
                             orientation='h', title="Highest Correlations",
                             color='pearson_r', color_continuous_scale='RdYlGn')
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, width='stretch')

        # Cointegration
        if coint is not None and len(coint) > 0:
            st.subheader("Cointegrated Pairs")
            coint_5 = coint[coint.get('is_cointegrated_5pct', pd.Series(dtype=bool))] \
                if 'is_cointegrated_5pct' in coint.columns else pd.DataFrame()

            if len(coint_5) > 0:
                st.caption(f"{len(coint_5)} cointegrated pairs found at 5% significance")
                display_cols = ['algo1', 'algo2', 'adf_stat', 'hedge_ratio',
                                'half_life_days', 'current_z_score']
                display_cols = [c for c in display_cols if c in coint_5.columns]
                st.dataframe(coint_5[display_cols].head(20), width='stretch', hide_index=True)
            else:
                st.info("No cointegrated pairs at 5% significance.")

        # Diversifiers
        if diversifiers is not None and len(diversifiers) > 0:
            st.subheader("Best Diversifiers")
            algo_search = st.text_input("Search algo for diversifiers:", "")
            if algo_search:
                div_results = diversifiers[diversifiers['algo'].str.contains(algo_search, case=False)]
                if len(div_results) > 0:
                    st.dataframe(div_results, width='stretch', hide_index=True)
                else:
                    st.info(f"No diversifiers found for '{algo_search}'")


# ============================================================
# PAGE: BENCHMARK RECONSTRUCTION
# ============================================================

elif page == "📈 Benchmark Reconstruction":
    st.header("Benchmark Reconstruction")

    daily = safe_load('reconstruction_daily.csv', index_col=0, parse_dates=True)
    results = safe_load('reconstruction_results.csv')
    report = safe_load_json('reconstruction_report.json')

    if daily is None:
        st.warning("No reconstruction_daily.csv found. Run `benchmark_reconstruction.py` first.")
    else:
        methods = [c for c in daily.columns if c != 'actual']

        # Comparison table
        if results is not None:
            st.subheader("Method Comparison")
            st.dataframe(results, width='stretch', hide_index=True)

        # Index chart
        st.subheader("Index Reconstruction")
        selected_methods = st.multiselect("Methods to display",
                                          methods, default=methods[:2])

        fig = go.Figure()
        if 'actual' in daily.columns:
            actual = daily['actual'].dropna()
            fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                                     name='Actual', line=dict(color='black', width=2.5)))

        colors = ['#534AB7', '#1D9E75', '#D85A30', '#378ADD']
        for i, method in enumerate(selected_methods):
            series = daily[method].dropna()
            fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                     name=method, line=dict(color=colors[i % len(colors)], width=1.5)))

        fig.update_layout(title="Actual vs Reconstructed Index",
                          height=500, hovermode='x unified',
                          yaxis_title="Index Value")
        st.plotly_chart(fig, width='stretch')

        # Deviation chart
        st.subheader("Deviation from Actual")
        fig2 = go.Figure()
        actual = daily['actual'].dropna()
        for i, method in enumerate(selected_methods):
            series = daily[method].dropna()
            common = actual.index.intersection(series.index)
            a_norm = actual.loc[common] / actual.loc[common].iloc[0] * 100
            r_norm = series.loc[common] / series.loc[common].iloc[0] * 100
            dev = r_norm - a_norm
            fig2.add_trace(go.Scatter(x=common, y=dev,
                                      name=method, line=dict(color=colors[i % len(colors)])))

        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Index Level Deviation (reconstruction - actual)",
                           height=400, yaxis_title="Deviation (index points)")
        st.plotly_chart(fig2, width='stretch')

        # Report summary
        if report:
            best = report.get('best_method', {})
            if best.get('method'):
                st.success(f"**Best method:** {best['method']} "
                           f"(Tracking Error: {best.get('tracking_error_annual', '?'):.4f}, "
                           f"Monthly Corr: {best.get('monthly_correlation', '?')})")


# ============================================================
# PAGE: BENCHMARK COMPARISON
# ============================================================

elif page == "📊 Benchmark Comparison":
    st.header("Benchmark Deep Comparison")

    daily = safe_load('reconstruction_daily.csv', index_col=0, parse_dates=True)
    freq_comp = safe_load('frequency_comparison.csv')
    dd_comp = safe_load('drawdown_comparison.csv')
    dd_curves = safe_load('drawdown_curves.csv', index_col=0, parse_dates=True)
    regime = safe_load('regime_analysis.csv')

    if daily is None:
        st.warning("No reconstruction_daily.csv found. Run `benchmark_reconstruction.py` first.")
    else:
        methods = [c for c in daily.columns if c != 'actual']

        # --- Method selector ---
        selected_method = st.selectbox("Select reconstruction method to compare:",
                                       methods, index=0)

        actual = daily['actual'].dropna()
        recon = daily[selected_method].dropna()

        # Align and normalize
        common = actual.index.intersection(recon.index)
        common = common[common.notna()]
        if len(common) < 10:
            st.error("Not enough overlapping data between actual and selected method.")
        else:
            a = actual.loc[common]
            r = recon.loc[common]
            a_norm = a / a.iloc[0] * 100
            r_norm = r / r.iloc[0] * 100

            # --- KPIs ---
            a_logret = np.log(a_norm / a_norm.shift(1)).dropna()
            r_logret = np.log(r_norm / r_norm.shift(1)).dropna()
            ret_aligned = pd.DataFrame({'actual': a_logret, 'recon': r_logret}).dropna()

            daily_corr = ret_aligned['actual'].corr(ret_aligned['recon'])
            tracking_err = (ret_aligned['actual'] - ret_aligned['recon']).std() * np.sqrt(252)
            final_diff = r_norm.iloc[-1] - a_norm.iloc[-1]

            # Monthly correlation
            a_m = a_norm.resample('ME').last()
            r_m = r_norm.resample('ME').last()
            monthly_corr = a_m.corr(r_m)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Daily Correlation", f"{daily_corr:.4f}")
            c2.metric("Monthly Correlation", f"{monthly_corr:.4f}")
            c3.metric("Tracking Error (ann.)", f"{tracking_err:.4f}")
            c4.metric("Final Difference", f"{final_diff:+.2f} pts")

            # --- Index overlay ---
            st.subheader("Index: Actual vs " + selected_method)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=a_norm.index, y=a_norm.values,
                                     name='Actual', line=dict(color='black', width=2.5)))
            fig.add_trace(go.Scatter(x=r_norm.index, y=r_norm.values,
                                     name=selected_method, line=dict(color='#534AB7', width=1.8)))
            fig.update_layout(height=450, hovermode='x unified', yaxis_title="Index (base 100)")
            st.plotly_chart(fig, width='stretch')

            # --- Deviation ---
            dev = r_norm - a_norm
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dev.index, y=dev.values,
                                      fill='tozeroy', line=dict(color='#534AB7', width=1),
                                      fillcolor='rgba(83,74,183,0.15)'))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(title="Deviation (reconstruction − actual)",
                               height=350, yaxis_title="Index points")
            st.plotly_chart(fig2, width='stretch')

            # --- Monthly returns comparison (computed dynamically) ---
            st.subheader("Monthly Returns Comparison")
            a_monthly_ret = a_norm.resample('ME').last().pct_change().dropna() * 100
            r_monthly_ret = r_norm.resample('ME').last().pct_change().dropna() * 100
            monthly_df = pd.DataFrame({
                'month': a_monthly_ret.index.strftime('%Y-%m'),
                'actual_pct': a_monthly_ret.values,
                'recon_pct': r_monthly_ret.reindex(a_monthly_ret.index).values,
            }).dropna()
            monthly_df['diff_pct'] = monthly_df['recon_pct'] - monthly_df['actual_pct']

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=monthly_df['month'], y=monthly_df['actual_pct'],
                                  name='Actual', marker_color='#534AB7'))
            fig3.add_trace(go.Bar(x=monthly_df['month'], y=monthly_df['recon_pct'],
                                  name=selected_method, marker_color='#1D9E75'))
            fig3.update_layout(barmode='group', height=400,
                               title=f"Monthly Returns: Actual vs {selected_method} (%)",
                               yaxis_title="Return %")
            st.plotly_chart(fig3, width='stretch')

            # Diff chart
            fig4 = go.Figure()
            colors_diff = ['#1D9E75' if x >= 0 else '#E24B4A' for x in monthly_df['diff_pct']]
            fig4.add_trace(go.Bar(x=monthly_df['month'], y=monthly_df['diff_pct'],
                                  marker_color=colors_diff))
            fig4.add_hline(y=0, line_dash="dash", line_color="gray")
            fig4.update_layout(title=f"Monthly Difference ({selected_method} − actual)",
                               height=300, yaxis_title="Diff %", showlegend=False)
            st.plotly_chart(fig4, width='stretch')

            # Stats
            avg_diff = monthly_df['diff_pct'].abs().mean()
            max_diff = monthly_df['diff_pct'].abs().max()
            within_05 = (monthly_df['diff_pct'].abs() < 0.5).sum()
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Avg |monthly diff|", f"{avg_diff:.4f}%")
            sc2.metric("Max |monthly diff|", f"{max_diff:.4f}%")
            sc3.metric("Months within ±0.5%", f"{within_05}/{len(monthly_df)}")

        st.divider()

        # --- Cross-method comparison (frequency, drawdown, regime) ---
        st.subheader("Cross-Method Analysis")

        if freq_comp is not None:
            st.caption("Correlation by Frequency (all methods)")
            fig = px.bar(freq_comp, x='frequency', y='correlation', color='method',
                         barmode='group', color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=380)
            st.plotly_chart(fig, width='stretch')

        if dd_curves is not None:
            st.caption("Drawdown Curves (all methods)")
            fig = go.Figure()
            for col in dd_curves.columns:
                w = 2.5 if col == 'actual' else 1.2
                c = 'black' if col == 'actual' else None
                fig.add_trace(go.Scatter(x=dd_curves.index, y=dd_curves[col] * 100,
                                         name=col, line=dict(width=w, color=c)))
            fig.update_layout(height=400, yaxis_title="Drawdown %")
            st.plotly_chart(fig, width='stretch')

        if dd_comp is not None:
            st.dataframe(dd_comp, width='stretch', hide_index=True)

        if regime is not None and len(regime) > 0:
            st.caption("Tracking Correlation by Market Regime (all methods)")
            fig = px.bar(regime, x='regime', y='correlation', color='method',
                         barmode='group')
            fig.update_layout(height=380)
            st.plotly_chart(fig, width='stretch')


# ============================================================
# PAGE: DIETZ RECONSTRUCTION (WITH CASHFLOWS)
# ============================================================

elif page == "📐 Dietz Reconstruction":
    st.header("Dietz Reconstruction (With Cashflows)")
    st.caption("Portfolio returns calculated using Modified Dietz / TWR methods from "
               "Christopherson et al. Chapter 5")

    dietz_daily = safe_load('dietz_reconstruction_daily.csv', index_col=0, parse_dates=True)
    dietz_results = safe_load('dietz_reconstruction_results.csv')
    dietz_report = safe_load_json('dietz_reconstruction_report.json')
    dietz_monthly = safe_load('dietz_monthly_returns.csv')
    phase_analysis = safe_load('dietz_phase_analysis.csv')
    nocf_daily = safe_load('reconstruction_daily.csv', index_col=0, parse_dates=True)

    if dietz_daily is None:
        st.warning("No Dietz reconstruction data. Run `benchmark_reconstruction_dietz.py` first.")
    else:
        dietz_methods = [c for c in dietz_daily.columns if c != 'actual']

        # Results table
        if dietz_results is not None:
            st.subheader("Method Comparison")
            st.dataframe(dietz_results, width='stretch', hide_index=True)

        # Method selector
        selected = st.selectbox("Select Dietz method:", dietz_methods, index=0)

        # Also allow comparison with no-cashflow
        nocf_method = None
        if nocf_daily is not None:
            nocf_methods = [c for c in nocf_daily.columns if c != 'actual']
            nocf_method = st.selectbox("Compare with no-cashflow method:",
                                       ['(none)'] + nocf_methods, index=0)
            if nocf_method == '(none)':
                nocf_method = None

        # Index chart
        st.subheader("Index: Actual vs Dietz Reconstruction")
        actual = dietz_daily['actual'].dropna()
        recon = dietz_daily[selected].dropna()

        if len(actual) > 0 and len(recon) > 0:
            common = actual.index.intersection(recon.index)
            a_norm = actual.loc[common] / actual.loc[common].iloc[0] * 100
            r_norm = recon.loc[common] / recon.loc[common].iloc[0] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=a_norm.index, y=a_norm.values,
                                     name='Actual', line=dict(color='black', width=2.5)))
            fig.add_trace(go.Scatter(x=r_norm.index, y=r_norm.values,
                                     name=f'[Dietz] {selected}',
                                     line=dict(color='#D85A30', width=1.8)))

            if nocf_method and nocf_daily is not None:
                nocf_series = nocf_daily[nocf_method].dropna()
                nocf_common = a_norm.index.intersection(nocf_series.index)
                if len(nocf_common) > 0:
                    n_norm = nocf_series.loc[nocf_common] / nocf_series.loc[nocf_common].iloc[0] * 100
                    fig.add_trace(go.Scatter(x=n_norm.index, y=n_norm.values,
                                             name=f'[NoCF] {nocf_method}',
                                             line=dict(color='#534AB7', width=1.2, dash='dash')))

            fig.update_layout(height=500, hovermode='x unified', yaxis_title="Index (base 100)")
            st.plotly_chart(fig, width='stretch')

            # Deviation
            dev_dietz = r_norm - a_norm
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dev_dietz.index, y=dev_dietz.values,
                                      name=f'[Dietz] {selected}', fill='tozeroy',
                                      line=dict(color='#D85A30'),
                                      fillcolor='rgba(216,90,48,0.15)'))

            if nocf_method and nocf_daily is not None:
                nocf_series = nocf_daily[nocf_method].dropna()
                nocf_common = a_norm.index.intersection(nocf_series.index)
                if len(nocf_common) > 0:
                    n_norm = nocf_series.loc[nocf_common] / nocf_series.loc[nocf_common].iloc[0] * 100
                    dev_nocf = n_norm - a_norm.loc[nocf_common]
                    fig2.add_trace(go.Scatter(x=dev_nocf.index, y=dev_nocf.values,
                                              name=f'[NoCF] {nocf_method}',
                                              line=dict(color='#534AB7', dash='dash')))

            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(title="Deviation from Actual (Dietz vs No-Cashflow)",
                               height=400, yaxis_title="Index points")
            st.plotly_chart(fig2, width='stretch')

        # Monthly Dietz returns detail
        if dietz_monthly is not None and len(dietz_monthly) > 0:
            st.subheader("Monthly Modified Dietz Returns")
            display_cols = [c for c in ['month', 'return', 'BV', 'EV', 'sum_cf', 'ABV', 'method']
                            if c in dietz_monthly.columns]
            st.dataframe(dietz_monthly[display_cols], width='stretch', hide_index=True)

        # Phase analysis
        if phase_analysis is not None and len(phase_analysis) > 0:
            st.subheader("Accuracy by Fund Phase")
            phase_display = phase_analysis[phase_analysis['method'].str.contains('Dietz', case=False)]
            if len(phase_display) > 0:
                cols_show = [c for c in ['method', 'phase', 'correlation_daily',
                                         'tracking_error_annual', 'final_diff_pct'] if c in phase_display.columns]
                st.dataframe(phase_display[cols_show], width='stretch', hide_index=True)

        # Reference
        if dietz_report and 'reference' in dietz_report:
            with st.expander("📚 Formulas Reference (Chapter 5)"):
                ref = dietz_report['reference']
                for key, val in ref.items():
                    st.markdown(f"**{key}**: `{val}`")


# ============================================================
# PAGE: BENCHMARK COMPOSITION
# ============================================================

elif page == "🧬 Benchmark Composition":
    st.header("Benchmark Composition by Phase")
    st.caption("What does the benchmark invest in? How does allocation change across Seed → Transition → Scale?")

    phase_comp = safe_load('phase_composition.csv')
    top_algos = safe_load('phase_top_algos.csv')
    cluster_alloc = safe_load('phase_cluster_allocation.csv')
    quality_comp = safe_load('phase_quality_comparison.csv')
    comp_report = safe_load_json('composition_report.json')

    if phase_comp is None and comp_report is None:
        st.warning("No composition data. Run `benchmark_composition.py` first.")
    else:
        phases_list = phase_comp['phase'].unique().tolist() if phase_comp is not None else []

        # --- Asset class allocation per phase ---
        if phase_comp is not None and len(phase_comp) > 0:
            st.subheader("Asset Class Allocation by Phase")

            fig = px.bar(phase_comp, x='phase', y='volume_pct', color='asset_class',
                         title="Volume-Weighted Asset Class Allocation (%)",
                         barmode='stack', color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=450, yaxis_title="% of Portfolio Volume")
            st.plotly_chart(fig, width='stretch')

            # Side-by-side comparison
            pivot = phase_comp.pivot_table(index='asset_class', columns='phase',
                                           values='volume_pct', fill_value=0)
            fig2 = px.bar(phase_comp, x='asset_class', y='volume_pct', color='phase',
                          barmode='group', title="Asset Class Weight: Phase Comparison",
                          color_discrete_sequence=['#888780', '#D85A30', '#534AB7'])
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, width='stretch')

            # Number of algos per class per phase
            fig3 = px.bar(phase_comp, x='asset_class', y='n_algos', color='phase',
                          barmode='group', title="Number of Algos per Asset Class",
                          color_discrete_sequence=['#888780', '#D85A30', '#534AB7'])
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, width='stretch')

        # --- Concentration ---
        if comp_report and 'phases' in comp_report:
            st.subheader("Portfolio Concentration")
            conc_rows = []
            for pname, pdata in comp_report['phases'].items():
                c = pdata.get('concentration', {})
                if c:
                    c['phase'] = pname
                    conc_rows.append(c)

            if conc_rows:
                conc_df = pd.DataFrame(conc_rows)
                cols = st.columns(len(conc_rows))
                for i, row in enumerate(conc_rows):
                    with cols[i]:
                        st.markdown(f"**{row['phase']}**")
                        st.metric("Total algos", row.get('total_algos', 0))
                        st.metric("Effective N", row.get('effective_n', 0))
                        st.metric("Top 5 share", f"{row.get('top_5_pct', 0)}%")
                        st.metric("Top 10 share", f"{row.get('top_10_pct', 0)}%")
                        st.metric("HHI", f"{row.get('hhi', 0):,.0f}")

        # --- Holding periods ---
        if comp_report and 'phases' in comp_report:
            st.subheader("Holding Periods by Phase")
            hp_rows = []
            for pname, pdata in comp_report['phases'].items():
                hp = pdata.get('holding_periods', {})
                if hp:
                    hp['phase'] = pname
                    hp_rows.append(hp)

            if hp_rows:
                hp_df = pd.DataFrame(hp_rows).set_index('phase')

                bucket_cols = ['pct_intraday', 'pct_under_7d', 'pct_7_30d', 'pct_30_90d', 'pct_over_90d']
                bucket_cols = [c for c in bucket_cols if c in hp_df.columns]
                if bucket_cols:
                    hp_melt = hp_df[bucket_cols].reset_index().melt(
                        id_vars='phase', var_name='bucket', value_name='pct')
                    hp_melt['bucket'] = hp_melt['bucket'].str.replace('pct_', '').str.replace('_', ' ')
                    fig = px.bar(hp_melt, x='phase', y='pct', color='bucket',
                                 barmode='stack', title="Holding Period Distribution (%)",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_layout(height=400, yaxis_title="%")
                    st.plotly_chart(fig, width='stretch')

        # --- Top algos per phase ---
        if top_algos is not None and len(top_algos) > 0:
            st.subheader("Top Algos per Phase")
            selected_phase = st.selectbox("Phase:", phases_list)

            phase_data = top_algos[top_algos['phase'] == selected_phase] if 'phase' in top_algos.columns else top_algos

            display = ['total_volume', 'pct_of_portfolio', 'n_trades',
                       'avg_holding_days', 'asset_class', 'predicted_asset',
                       'direction', 'confidence', 'sharpe_ratio',
                       'annualized_return_pct', 'max_drawdown_pct']
            display = [c for c in display if c in phase_data.columns]

            st.dataframe(phase_data[display].head(20), width='stretch')

        # --- Cluster allocation ---
        if cluster_alloc is not None and len(cluster_alloc) > 0:
            st.subheader("Cluster Allocation by Phase")
            if 'phase' in cluster_alloc.columns and 'volume_pct' in cluster_alloc.columns:
                cluster_alloc['cluster_str'] = cluster_alloc.index.astype(str) \
                    if 'cluster' not in cluster_alloc.columns else cluster_alloc['cluster'].astype(str)
                fig = px.bar(cluster_alloc, x='cluster_str', y='volume_pct', color='phase',
                             barmode='group', title="Cluster Volume Allocation (%)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

        # --- Quality comparison ---
        if quality_comp is not None and len(quality_comp) > 0:
            st.subheader("Algo Quality: Benchmark vs Universe")

            for phase in quality_comp['phase'].unique() if 'phase' in quality_comp.columns else []:
                pq = quality_comp[quality_comp['phase'] == phase]
                with st.expander(f"{phase}"):
                    if 'metric' in pq.columns and 'bench_mean' in pq.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=pq['metric'], y=pq['bench_mean'],
                                             name='Benchmark', marker_color='#534AB7'))
                        fig.add_trace(go.Bar(x=pq['metric'], y=pq['universe_mean'],
                                             name='Universe', marker_color='#888780'))
                        fig.update_layout(barmode='group', height=350,
                                          title=f"{phase}: Benchmark vs Universe Avg")
                        st.plotly_chart(fig, width='stretch')
                    st.dataframe(pq, width='stretch', hide_index=True)

        # --- Cross-phase algo overlap ---
        if comp_report and 'algo_overlap' in comp_report:
            st.subheader("Algo Overlap Between Phases")
            overlap = comp_report['algo_overlap']
            for pair, count in overlap.items():
                st.caption(f"{pair.replace('_and_', ' ∩ ')}: **{count}** common algos")


# ============================================================
# PAGE: REGIME DETECTION
# ============================================================

elif page == "🌡️ Regime Detection":
    st.header("SP500 Regime Detection")

    regime_pred = safe_load('regime_predictions.csv')
    features = safe_load('sp500_features_regimes.csv',
                         index_col=0, parse_dates=True)
    rotation = safe_load('benchmark_rotation_by_regime.csv')
    regime_report = safe_load_json('regime_report.json')

    if regime_pred is None and features is None:
        st.warning("No regime data found. Run `benchmark_asset_regime.py` first.")
    else:
        # Regime report summary
        if regime_report:
            rm = regime_report.get('regime_model', {})
            st.metric("Model", f"{rm.get('model', '?')} (F1={rm.get('cv_f1', 0):.4f})")

            # Feature importance
            if 'feature_importances' in rm:
                st.subheader("Feature Importance")
                fi = rm['feature_importances']
                fi_df = pd.DataFrame({'feature': list(fi.keys()), 'importance': list(fi.values())})
                fi_df = fi_df.sort_values('importance', ascending=True).tail(15)
                fig = px.bar(fi_df, x='importance', y='feature', orientation='h',
                             title="Top 15 Features for Regime Detection",
                             color_discrete_sequence=['#534AB7'])
                fig.update_layout(height=450)
                st.plotly_chart(fig, width='stretch')

            # Transition matrix
            if 'transition_matrix' in rm:
                st.subheader("Regime Transition Probabilities")
                trans = pd.DataFrame(rm['transition_matrix'])
                fig = px.imshow(trans, text_auto='.1f',
                                title="Transition Matrix (%)",
                                color_continuous_scale='Blues')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

            # Composition
            comp = regime_report.get('composition', {})
            if 'asset_distribution' in comp:
                st.subheader("Benchmark Asset Distribution")
                ad = comp['asset_distribution']
                fig = px.pie(values=list(ad.values()), names=list(ad.keys()),
                             title="Benchmark Algo Asset Classes")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

        # Regime timeline
        if features is not None and 'regime' in features.columns:
            st.subheader("Regime Timeline")

            regime_colors = {
                'bull': '#1D9E75', 'bear': '#E24B4A', 'high_vol': '#D85A30',
                'recovery': '#534AB7', 'calm': '#888780'
            }

            fig = go.Figure()
            for regime, color in regime_colors.items():
                mask = features['regime'] == regime
                if mask.any():
                    subset = features[mask]
                    if 'ret_20d' in subset.columns:
                        fig.add_trace(go.Scatter(
                            x=subset.index, y=subset['ret_20d'] * 100,
                            mode='markers', name=regime,
                            marker=dict(color=color, size=4, opacity=0.6)
                        ))

            fig.update_layout(title="SP500 20-day Return colored by Regime",
                              height=450, yaxis_title="20d Return %",
                              hovermode='closest')
            st.plotly_chart(fig, width='stretch')

            # Regime distribution
            regime_counts = features['regime'].value_counts()
            fig2 = px.bar(x=regime_counts.index, y=regime_counts.values,
                          color=regime_counts.index,
                          color_discrete_map=regime_colors,
                          title="Regime Distribution (days)")
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, width='stretch')

        # Rotation analysis
        if rotation is not None and len(rotation) > 0:
            st.subheader("Benchmark SP500 Allocation by Regime")
            if 'sp500_pct_volume' in rotation.columns:
                fig = px.bar(rotation, y='sp500_pct_volume',
                             x=rotation.index if rotation.index.name else rotation.columns[0],
                             title="% Volume allocated to SP500-related algos by regime",
                             color_discrete_sequence=['#378ADD'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            st.dataframe(rotation, width='stretch')


# ============================================================
# PAGE: CASHFLOW ANALYSIS
# ============================================================

elif page == "💰 Cashflow Analysis":
    st.header("Benchmark Cashflow Analysis")
    st.caption("GIPS Unit Price Method — comparing equity_normalized (pure performance) vs equity_EOD (real equity)")

    timeline = safe_load('daily_fund_timeline.csv', index_col=0, parse_dates=True)
    monthly_evo = safe_load('monthly_fund_evolution.csv')
    cf_events = safe_load('cashflow_events.csv', index_col=0, parse_dates=True)
    cf_report = safe_load_json('cashflow_report.json')

    if timeline is None and cf_report is None:
        st.warning("No cashflow data found. Run `benchmark_cashflows.py` first.")
    else:
        # Totals — compute live from timeline CSV
        _cf_col = 'cashflow' if (timeline is not None and 'cashflow' in timeline.columns) else \
            'cashflow_est' if (timeline is not None and 'cashflow_est' in timeline.columns) else None

        total_in, total_out, n_in, n_out = 0, 0, 0, 0
        if _cf_col and timeline is not None:
            _cf = timeline[_cf_col].dropna()
            total_in = float(_cf[_cf > 100].sum())
            total_out = float(_cf[_cf < -100].sum())
            n_in = int((_cf > 100).sum())
            n_out = int((_cf < -100).sum())
        elif cf_report and 'totals' in cf_report:
            t = cf_report['totals']
            total_in = t.get('total_inflows', 0) or 0
            total_out = t.get('total_outflows', 0) or 0
            n_in = t.get('n_inflow_events', 0)
            n_out = t.get('n_outflow_events', 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inflows", f"${total_in:,.0f}", delta=f"{n_in} events")
        c2.metric("Total Outflows", f"${total_out:,.0f}", delta=f"{n_out} events")
        c3.metric("Net Cashflow", f"${total_in + total_out:,.0f}")
        c4.metric("Method", "GIPS Unit Price")

        # Phase summary
        # Phase summary — compute inflows/outflows live from timeline
        if cf_report and 'phases' in cf_report:
            st.subheader("Fund Phases")
            phases = cf_report['phases']

            # Get cashflow column from timeline for live computation
            cf_col = None
            if timeline is not None:
                cf_col = 'cashflow' if 'cashflow' in timeline.columns else \
                    'cashflow_est' if 'cashflow_est' in timeline.columns else None

            cols = st.columns(len(phases))
            for i, (phase_name, p) in enumerate(phases.items()):
                with cols[i]:
                    short_name = phase_name.split(': ')[1] if ': ' in phase_name else phase_name
                    st.markdown(f"**{short_name}**")
                    st.caption(p.get('date_range', ''))
                    st.metric("Avg AUM", f"${p.get('avg_aum', 0):,.0f}")

                    # Try JSON first, then compute from timeline
                    inflows = p.get('total_inflows', 0) or 0
                    outflows = p.get('total_outflows', 0) or 0

                    if (inflows == 0 and outflows == 0) and cf_col and timeline is not None:
                        # Compute live from timeline
                        date_range = p.get('date_range', '')
                        if '→' in date_range:
                            parts = date_range.split('→')
                            d_start = pd.Timestamp(parts[0].strip())
                            d_end = pd.Timestamp(parts[1].strip())
                            mask = (timeline.index >= d_start) & (timeline.index <= d_end)
                            phase_cf = timeline.loc[mask, cf_col].dropna()
                            inflows = float(phase_cf[phase_cf > 100].sum())
                            outflows = float(phase_cf[phase_cf < -100].sum())

                    st.metric("Inflows", f"${inflows:,.0f}")
                    st.metric("Outflows", f"${outflows:,.0f}")
                    st.metric("Trades/day", f"{p.get('trades_per_day', 0)}")
                    st.metric("Algos", f"{p.get('n_unique_algos', 0)}")
                    st.metric("Leverage", f"{p.get('avg_leverage', 0)}x")

        # AUM + Equity timeline
        if timeline is not None:
            st.subheader("AUM & Equity Evolution")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("AUM", "Equity EOD vs Equity Normalized"),
                                vertical_spacing=0.08, row_heights=[0.5, 0.5])

            aum_col = 'AUM' if 'AUM' in timeline.columns else 'aum' if 'aum' in timeline.columns else None
            if aum_col:
                fig.add_trace(go.Scatter(x=timeline.index, y=timeline[aum_col],
                                         name='AUM', line=dict(color='#534AB7', width=1.5)), row=1, col=1)
            eq_col = 'equity_EOD' if 'equity_EOD' in timeline.columns else \
                'equity_eod' if 'equity_eod' in timeline.columns else None
            norm_col = 'equity_normalized' if 'equity_normalized' in timeline.columns else \
                'equity_norm' if 'equity_norm' in timeline.columns else None
            if eq_col:
                fig.add_trace(go.Scatter(x=timeline.index, y=timeline[eq_col],
                                         name='Equity EOD (real)', line=dict(color='#D85A30', width=1.5)), row=2, col=1)
            if norm_col:
                fig.add_trace(go.Scatter(x=timeline.index, y=timeline[norm_col],
                                         name='Equity Normalized (perf.)', line=dict(color='#1D9E75', width=1.5)),
                              row=2, col=1)

            fig.update_layout(height=550, hovermode='x unified')
            st.plotly_chart(fig, width='stretch')

        # Inflows and Outflows SEPARATE
        if timeline is not None:
            # Handle both old (cashflow_est) and new (cashflow) column names
            cf_col = 'cashflow' if 'cashflow' in timeline.columns else \
                'cashflow_est' if 'cashflow_est' in timeline.columns else None

            if cf_col:
                st.subheader("Cashflows (Inflows & Outflows)")

                cf = timeline[cf_col].dropna()
                inflows = cf.clip(lower=0)
                outflows = cf.clip(upper=0)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=inflows.index, y=inflows.values,
                                     name='Inflows', marker_color='#1D9E75'))
                fig.add_trace(go.Bar(x=outflows.index, y=outflows.values,
                                     name='Outflows', marker_color='#E24B4A'))
                fig.update_layout(barmode='relative', height=400,
                                  title="Daily Inflows (green) & Outflows (red)",
                                  yaxis_title="Cashflow ($)")
                st.plotly_chart(fig, width='stretch')

                # Cumulative: separate inflows and outflows
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=inflows.cumsum().index, y=inflows.cumsum().values,
                                          name='Cumulative Inflows', fill='tozeroy',
                                          line=dict(color='#1D9E75'), fillcolor='rgba(29,158,117,0.15)'))
                fig2.add_trace(go.Scatter(x=outflows.cumsum().index, y=outflows.cumsum().values,
                                          name='Cumulative Outflows', fill='tozeroy',
                                          line=dict(color='#E24B4A'), fillcolor='rgba(226,75,74,0.15)'))
                fig2.add_trace(go.Scatter(x=cf.cumsum().index, y=cf.cumsum().values,
                                          name='Net Cumulative', line=dict(color='black', width=2)))
                fig2.update_layout(height=400, title="Cumulative Cashflows",
                                   yaxis_title="Cumulative ($)")
                st.plotly_chart(fig2, width='stretch')

        # Monthly table with inflows/outflows
        if monthly_evo is not None:
            st.subheader("Monthly Cashflow Breakdown")

            display_cols = ['inflows', 'outflows', 'net_cashflow', 'n_inflows', 'n_outflows',
                            'strategy_return_pct', 'equity_change_pct', 'cf_impact_pct']
            display_cols = [c for c in display_cols if c in monthly_evo.columns]

            if display_cols:
                st.dataframe(monthly_evo[display_cols], width='stretch')

            # Bar chart: monthly inflows vs outflows
            if 'inflows' in monthly_evo.columns and 'outflows' in monthly_evo.columns:
                month_col = monthly_evo.index if 'month' not in monthly_evo.columns else monthly_evo['month']
                fig = go.Figure()
                fig.add_trace(go.Bar(x=month_col, y=monthly_evo['inflows'],
                                     name='Inflows', marker_color='#1D9E75'))
                fig.add_trace(go.Bar(x=month_col, y=monthly_evo['outflows'],
                                     name='Outflows', marker_color='#E24B4A'))
                fig.update_layout(barmode='relative', height=400,
                                  title="Monthly Inflows & Outflows",
                                  yaxis_title="$", xaxis_tickangle=-45)
                st.plotly_chart(fig, width='stretch')

            # Strategy return vs equity change (shows cashflow impact)
            if 'strategy_return_pct' in monthly_evo.columns and 'equity_change_pct' in monthly_evo.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=month_col, y=monthly_evo['strategy_return_pct'],
                                     name='Strategy Return %', marker_color='#534AB7'))
                fig.add_trace(go.Bar(x=month_col, y=monthly_evo['equity_change_pct'],
                                     name='Equity Change %', marker_color='#D85A30'))
                fig.update_layout(barmode='group', height=400,
                                  title="Strategy Return vs Equity Change (gap = cashflow impact)",
                                  yaxis_title="%", xaxis_tickangle=-45)
                st.plotly_chart(fig, width='stretch')

        # Algo rotation
        if cf_report and 'algo_rotation' in cf_report:
            st.subheader("Algo Rotation: Seed → Scale")
            rot = cf_report['algo_rotation']
            c1, c2, c3 = st.columns(3)
            c1.metric("Seed-only algos", rot.get('seed_only', rot.get('seed_only_algos', 0)))
            c2.metric("Common algos", rot.get('common', rot.get('common_algos', 0)))
            c3.metric("New in Scale", rot.get('scale_only', rot.get('scale_only_algos', 0)))
            ret_pct = rot.get('retention_pct', rot.get('pct_seed_retained', 0))
            new_pct = rot.get('new_in_scale_pct', rot.get('pct_scale_new', 0))
            st.caption(f"Retention: {ret_pct}% | New: {new_pct}%")

        # Cashflow events table
        if cf_events is not None and len(cf_events) > 0:
            st.subheader("Major Cashflow Events")
            st.dataframe(cf_events.sort_index(ascending=False).head(50),
                         width='stretch')


# ============================================================
# PAGE: METHODOLOGY
# ============================================================

elif page == "📖 Methodology":
    st.header("Methodology & Explanation")

    st.markdown(r"""
    ## Detecting Cashflows: GIPS Unit Price Method

    Following **GIPS (Global Investment Performance Standards)** and the **Unit Price Method**,
    we use two columns from the trades data that behave differently:

    ### 1. `equity_normalized` — Pure Performance

    This column measures **only strategy performance**. If the strategy gains 2% in the market,
    this value rises 2%. It is **blind to cash movements** — deposits and withdrawals don't affect it.
    It's the equivalent of a NAV per unit/share.

    ### 2. `equity_EOD` — Real Equity

    This is the **actual money** in the account at end of day. It reflects both:
    - Strategy performance (organic growth/decline)
    - External cash movements (deposits/withdrawals)

    ### The Golden Rule

    If there are **no cashflows**, both columns must move by the same percentage:

    $$\Delta\%\ Equity\_EOD = \Delta\%\ Equity\_Normalized$$

    When `equity_EOD` changes disproportionately relative to `equity_normalized`,
    the difference is a cashflow.

    ### Calculation

    **Step 1** — Strategy return from the normalized (pure performance) column:

    $$Retorno = \frac{Equity\_Normalized_t}{Equity\_Normalized_{t-1}} - 1$$

    **Step 2** — Expected equity if no deposit/withdrawal occurred:

    $$Patrimonio\_Esperado_t = Equity\_EOD_{t-1} \times (1 + Retorno)$$

    **Step 3** — Cashflow = actual equity minus expected equity:

    $$Flujo\_de\_Caja = Equity\_EOD_t - Patrimonio\_Esperado_t$$

    - **Positive** → Inflow (deposit — fresh money entered the fund)
    - **Negative** → Outflow (withdrawal — money was taken out)

    ---

    ## Benchmark Reconstruction

    ### The Holdings Matrix (Cumulative Units)

    For each day and algorithm, we track accumulated invested units:

    ```
    holdings[day, algo] = holdings[day-1, algo]
                        + Σ(volume where algo OPENS on this day)
                        - Σ(volume where algo CLOSES on this day)
    ```

    ### Why Volume-Weighted Works Best (0.989 monthly correlation)

    **Volume-weighted**: `weight_A = units_A / Σ(all_units)`

    This works best because:

    1. **Matches the fund's allocation decision.** Volume represents the fund manager's
       explicit choice — 8,000 units in algo X and 2,000 in algo Y means 80/20 allocation.

    2. **No price circularity.** Value-weighted uses `units × price` but the algorithm's price
       IS what we're replicating — using it for weights creates a feedback loop.

    3. **Cashflow-agnostic.** The relative volume across algos reflects intended portfolio
       composition regardless of total AUM.

    ### Why Log Returns

    All returns are computed as $\ln(P_t / P_{t-1})$:

    - **Additive across time**: $\ln(P_3/P_1) = \ln(P_2/P_1) + \ln(P_3/P_2)$
    - **Additive across portfolio**: $r_{port} \approx \sum w_i \cdot r_i$
    - **Symmetric**: +10% and -10% have equal magnitude
    - **No compounding errors** over 1,674 days of daily aggregation

    ### Low Daily Correlation (0.08) vs High Monthly (0.989)

    Daily correlation is low because many algos trade sparsely (20-50% of days active).
    At monthly aggregation, noise cancels and the structural signal emerges.
    This is **timing noise, not methodology error**.

    ---

    ## Dietz Reconstruction (With Cashflows)

    Reference: *Christopherson, Carino & Ferson — Portfolio Performance Measurement 
    and Benchmarking*, Chapter 5: Returns in the Presence of Cash Flows.

    ### Modified Dietz Formula (eq 5.5)

    For a period with beginning value $BV$, ending value $EV$, and cashflows $C_k$:

    $$r = \frac{EV - BV - \sum_{k=1}^{K} C_k}{BV + \sum_{k=1}^{K} W_k \cdot C_k}$$

    where $W_k$ is the day-weight fraction (eq 5.4):

    $$W_k = \frac{TD - D_k}{TD}$$

    $TD$ = total days in period, $D_k$ = days since period start when cashflow $k$ occurs.

    This effectively creates an **Adjusted Beginning Value** (eq 5.6):

    $$ABV = BV + \sum_{k=1}^{K} W_k \cdot C_k$$

    and an **Adjusted Ending Value** (eq 5.7):

    $$AEV = EV - \sum_{k=1}^{K} (1 - W_k) \cdot C_k$$

    so that $r = AEV / ABV - 1$, analogous to the simple holding period return.

    ### True Time-Weighted Return (eq 5.2)

    Creates subperiods at each cashflow event, computes returns within each subperiod,
    and links them:

    $$r = (1 + r_1)(1 + r_2) \cdots (1 + r_T) - 1$$

    where $r_t = EV_t / BV_t - 1$ for each subperiod. This is the gold standard
    when valuations at cashflow times are available. It completely neutralizes 
    cashflow effects because each subperiod starts AFTER the cashflow is absorbed.

    ### Four Dietz Methods Implemented

    | Method | Description |
    |--------|-------------|
    | **TWR** | True Time-Weighted Return — subperiods at each cashflow, linked returns |
    | **Linked Modified Dietz** | Monthly subperiods, Modified Dietz (eq 5.5) within each month, linked |
    | **Daily Modified Dietz** | Midpoint Dietz applied daily on fund-level equity |
    | **Vol-Weighted + CF Adj** | No-cashflow volume-weighted but dampened on cashflow days |

    ---

    ## Three Fund Phases Detected

    | | Phase 1: Seed | Phase 2: Transition | Phase 3: Scale |
    |---|---|---|---|
    | **Period** | Jun 2020 → Dec 2023 | Dec 26-31, 2023 | Feb 2024 → Dec 2024 |
    | **AUM** | ~143K | ~2.5M | ~6.9M |
    | **Leverage** | 10x | 10x | 10x |
    | **Trades/day** | 2.2 | 17.9 | 18.2 |
    | **Unique algos** | 78 | 82 | 231 |
    | **Median holding** | 50 days | 17 days | 24 days |
    | **Cashflows** | ~neutral | Large inflows | Active in/out |

    The fund maintains consistent **~10x leverage** across all phases (AUM / equity_EOD ≈ 10).

    In the **Seed** phase, cashflows are minimal — the fund operates with stable capital.
    The no-cashflow assumption holds well here, which is why reconstruction accuracy is high
    for this period.

    The **Transition** (Dec 26, 2023) marks a massive capital injection: AUM jumps from ~120K
    to ~2M in 5 days. From this point, the fund operates at a fundamentally different scale
    with active daily cashflows of ±$1-2M.

    In **Scale**, the fund rotates aggressively: 81.8% of algos are new (not in Seed),
    holding periods drop from 50 to 24 days, and the fund manages 231 unique algos
    simultaneously vs 78 in Seed.
    """)

    # Show reconstruction results if available
    results = safe_load('reconstruction_results.csv')
    if results is not None:
        st.subheader("Reconstruction Method Comparison")
        st.dataframe(results, width='stretch', hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Algo Analyzer Dashboard | Built with Streamlit + Plotly")