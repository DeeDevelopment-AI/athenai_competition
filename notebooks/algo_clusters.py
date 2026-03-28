#!/usr/bin/env python3
"""
=================================================================
ALGORITHM CLUSTERER + BENCHMARK COMPARATOR
=================================================================

Paso 2 del pipeline. Lee el metrics_all.csv y style_analysis_all.csv
generados por algo_pipeline.py y produce:

  - clusters.csv                -> Cada algo con su cluster asignado
  - cluster_profiles.csv        -> Perfil medio de cada cluster
  - cluster_summary.json        -> Resumen para dashboard
  - benchmark_comparison.csv    -> Rendimiento de benchmarks en cada periodo
  - full_analysis.json          -> JSON completo para el dashboard interactivo

Uso:
  python3 algo_cluster.py --input results/ --output results/ --n-clusters 6

Si no especificas --n-clusters, lo calcula automaticamente con Silhouette.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import json
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from notebook_paths import default_output_dir, notebook_data_path


# ============================================================
# 1. LOAD PIPELINE RESULTS
# ============================================================

def load_pipeline_results(input_dir):
    """Load metrics and style analysis from pipeline output."""
    metrics_path = os.path.join(input_dir, 'metrics_all.csv')
    style_path = os.path.join(input_dir, 'style_analysis_all.csv')

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No se encuentra {metrics_path}. Ejecuta primero algo_pipeline.py")

    df_metrics = pd.read_csv(metrics_path)
    print(f"  Loaded {len(df_metrics)} algorithms from metrics_all.csv")

    df_style = None
    if os.path.exists(style_path):
        df_style = pd.read_csv(style_path)
        print(f"  Loaded style analysis from style_analysis_all.csv")

    return df_metrics, df_style


# ============================================================
# 2. FEATURE ENGINEERING FOR CLUSTERING
# ============================================================

def prepare_features(df_metrics, df_style=None):
    """
    Construye la matriz de features para clustering.

    Features principales (de metrics):
      - annualized_return_pct     -> Eje rendimiento
      - annualized_volatility_pct -> Eje riesgo/volatilidad
      - sharpe_ratio              -> Eficiencia riesgo-retorno
      - sortino_ratio             -> Penaliza solo downside
      - max_drawdown_pct          -> Riesgo de cola
      - calmar_ratio              -> Retorno / drawdown
      - win_rate_pct              -> Consistencia
      - skewness                  -> Asimetria de retornos
      - kurtosis                  -> Colas gordas

    Features de estilo (si hay style_analysis):
      - SP500 exposure            -> Correlacion con equities
      - Gold exposure             -> Correlacion con commodities
      - EURUSD exposure           -> Correlacion con forex
      - r_squared                 -> Cuanto explican los benchmarks
      - unexplained_pct           -> Independencia de la estrategia
    """
    feature_cols = [
        'annualized_return_pct',
        'annualized_volatility_pct',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown_pct',
        'calmar_ratio',
        'win_rate_pct',
        'skewness',
        'kurtosis',
    ]

    # Verificar que existen
    available = [c for c in feature_cols if c in df_metrics.columns]
    df_feat = df_metrics[['name'] + available].copy()

    # Merge style si existe
    if df_style is not None and 'name' in df_style.columns:
        style_cols = []
        for c in ['SP500', 'Gold', 'EURUSD', 'r_squared', 'unexplained_pct']:
            if c in df_style.columns:
                style_cols.append(c)
        if style_cols:
            df_style_clean = df_style[['name'] + style_cols].copy()
            # Rellenar NaN con 0 (sin solapamiento = sin exposicion)
            df_style_clean[style_cols] = df_style_clean[style_cols].fillna(0)
            df_feat = df_feat.merge(df_style_clean, on='name', how='left')
            df_feat[style_cols] = df_feat[style_cols].fillna(0)
            available += style_cols

    # Limpiar infinitos y NaN
    df_feat[available] = df_feat[available].replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(subset=available)

    # Clip outliers extremos (percentiles 1-99) para que no distorsionen clusters
    for col in available:
        p1, p99 = df_feat[col].quantile(0.01), df_feat[col].quantile(0.99)
        df_feat[col] = df_feat[col].clip(p1, p99)

    print(f"  Features: {len(available)} columnas, {len(df_feat)} algoritmos validos")
    print(f"  Columnas: {available}")

    return df_feat, available


# ============================================================
# 3. OPTIMAL K SELECTION
# ============================================================

def find_optimal_k(X_scaled, k_range=None):
    """
    Encuentra el K optimo usando Silhouette Score.
    Mayor silhouette = clusters mas coherentes y separados.
    """
    n_samples = X_scaled.shape[0]
    max_k = min(11, n_samples - 1)  # k must be < n_samples
    if k_range is None:
        k_range = range(2, max_k + 1)
    else:
        k_range = range(max(2, k_range.start), min(k_range.stop, max_k + 1))

    if len(k_range) == 0:
        print(f"  Solo {n_samples} muestras, asignando K=2")
        return 2, {}

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        scores[k] = round(sil, 4)

    best_k = max(scores, key=scores.get)
    print(f"\n  Silhouette scores: {scores}")
    print(f"  Optimo: K={best_k} (silhouette={scores[best_k]})")
    return best_k, scores


# ============================================================
# 4. CLUSTERING
# ============================================================

def run_clustering(df_feat, feature_cols, n_clusters=None):
    """
    Ejecuta K-Means + Hierarchical clustering.
    Devuelve el dataframe con labels asignados.
    """
    X = df_feat[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Determinar K ---
    n_samples = X_scaled.shape[0]
    if n_clusters is not None:
        n_clusters = min(n_clusters, n_samples - 1)
    if n_clusters is None or n_clusters < 2:
        n_clusters, sil_scores = find_optimal_k(X_scaled)
    else:
        sil_scores = {}
        print(f"\n  Usando K={n_clusters} (especificado por usuario)")

    # --- K-Means ---
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    df_feat = df_feat.copy()
    df_feat['cluster_kmeans'] = km.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, df_feat['cluster_kmeans'])

    # --- Hierarchical (Ward) ---
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df_feat['cluster_hierarchical'] = hc.fit_predict(X_scaled)
    hc_sil = silhouette_score(X_scaled, df_feat['cluster_hierarchical'])

    print(f"\n  K-Means silhouette:      {km_sil:.4f}")
    print(f"  Hierarchical silhouette: {hc_sil:.4f}")

    # Usar el mejor
    if km_sil >= hc_sil:
        df_feat['cluster'] = df_feat['cluster_kmeans']
        best_method = 'K-Means'
    else:
        df_feat['cluster'] = df_feat['cluster_hierarchical']
        best_method = 'Hierarchical'
    print(f"  Metodo seleccionado: {best_method}")

    # --- PCA para visualizacion 2D ---
    n_components = min(2, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_feat['pca_x'] = coords[:, 0].round(4)
    df_feat['pca_y'] = coords[:, 1].round(4) if n_components >= 2 else 0.0

    explained = pca.explained_variance_ratio_

    col_names = [f'PC{i+1}' for i in range(n_components)]
    if n_components < 2:
        col_names.append('PC2')
        pad = np.zeros((len(feature_cols), 1))
        components = np.hstack([pca.components_.T, pad])
    else:
        components = pca.components_.T

    print(f"  PCA varianza explicada: {', '.join(f'PC{i+1}={v:.2%}' for i, v in enumerate(explained))}")

    # Feature importance (PCA loadings)
    loadings = pd.DataFrame(
        components,
        columns=['PC1', 'PC2'],
        index=feature_cols
    ).round(4)

    return df_feat, n_clusters, sil_scores, best_method, loadings, list(explained) + [0.0] * (2 - len(explained))


# ============================================================
# 5. CLUSTER PROFILING
# ============================================================

def profile_clusters(df_feat, feature_cols, n_clusters):
    """
    Genera un perfil descriptivo de cada cluster:
    media, mediana, y un "nombre" interpretable.
    """
    profiles = []

    for c in range(n_clusters):
        mask = df_feat['cluster'] == c
        group = df_feat[mask]
        n = len(group)

        profile = {'cluster': c, 'n_algorithms': n, 'pct_of_total': round(n / len(df_feat) * 100, 1)}

        for col in feature_cols:
            profile[f'{col}_mean'] = round(group[col].mean(), 3)
            profile[f'{col}_median'] = round(group[col].median(), 3)
            profile[f'{col}_std'] = round(group[col].std(), 3)

        # --- Auto-naming heuristic ---
        ret = profile.get('annualized_return_pct_mean', 0)
        vol = profile.get('annualized_volatility_pct_mean', 0)
        sharpe = profile.get('sharpe_ratio_mean', 0)
        mdd = profile.get('max_drawdown_pct_mean', 0)

        parts = []
        # Retorno
        if ret > 15:
            parts.append('Alto retorno')
        elif ret > 3:
            parts.append('Retorno moderado')
        elif ret > -3:
            parts.append('Retorno neutro')
        else:
            parts.append('Retorno negativo')

        # Volatilidad
        if vol > 25:
            parts.append('alta volatilidad')
        elif vol > 12:
            parts.append('volatilidad media')
        else:
            parts.append('baja volatilidad')

        # Eficiencia
        if sharpe > 1.0:
            parts.append('(muy eficiente)')
        elif sharpe > 0.3:
            parts.append('(eficiente)')
        elif sharpe < -0.5:
            parts.append('(ineficiente)')

        profile['label'] = ' · '.join(parts)
        profiles.append(profile)

    return pd.DataFrame(profiles)


# ============================================================
# 6. BENCHMARK COMPARISON (CONTEXTUAL)
# ============================================================

def compute_benchmark_comparison(df_metrics, bench_dir=None):
    """
    Para cada algoritmo, calcula el rendimiento de los benchmarks
    en el MISMO periodo temporal. Asi se puede comparar en contexto.

    Si no tiene bench_dir, genera una tabla de periodos unicos
    para que el usuario pueda calcular manualmente.
    """
    # Generar periodos unicos
    df = df_metrics.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Agrupar por ventana temporal (quarter/year) para contextualizar
    df['start_year'] = df['start_date'].dt.year
    df['end_year'] = df['end_date'].dt.year
    df['period_label'] = df['start_date'].dt.strftime('%Y-%m') + ' → ' + df['end_date'].dt.strftime('%Y-%m')

    # Estadisticas por periodo temporal
    period_stats = df.groupby('start_year').agg(
        n_algos=('name', 'count'),
        avg_return=('annualized_return_pct', 'mean'),
        avg_vol=('annualized_volatility_pct', 'mean'),
        avg_sharpe=('sharpe_ratio', 'mean'),
        avg_mdd=('max_drawdown_pct', 'mean'),
        best_return=('annualized_return_pct', 'max'),
        worst_return=('annualized_return_pct', 'min'),
    ).round(3)

    return df, period_stats


# ============================================================
# 7. EXPORT FOR DASHBOARD
# ============================================================

def export_dashboard_json(df_feat, profiles, feature_cols, loadings,
                          pca_variance, sil_scores, n_clusters, output_path):
    """Export all results as JSON for the interactive dashboard."""

    # Cluster scatter data (PCA)
    scatter = []
    for _, row in df_feat.iterrows():
        scatter.append({
            'name': row['name'],
            'x': float(row['pca_x']),
            'y': float(row['pca_y']),
            'cluster': int(row['cluster']),
            'ret': float(row.get('annualized_return_pct', 0)),
            'vol': float(row.get('annualized_volatility_pct', 0)),
            'sharpe': float(row.get('sharpe_ratio', 0)),
            'mdd': float(row.get('max_drawdown_pct', 0)),
        })

    # Cluster profiles
    cluster_data = []
    for _, p in profiles.iterrows():
        entry = {
            'id': int(p['cluster']),
            'label': p['label'],
            'n': int(p['n_algorithms']),
            'pct': float(p['pct_of_total']),
        }
        for col in feature_cols:
            entry[f'{col}_mean'] = float(p.get(f'{col}_mean', 0))
        cluster_data.append(entry)

    # PCA loadings for feature importance
    loading_data = []
    for feat in feature_cols:
        loading_data.append({
            'feature': feat,
            'pc1': float(loadings.loc[feat, 'PC1']),
            'pc2': float(loadings.loc[feat, 'PC2']),
            'importance': float(abs(loadings.loc[feat, 'PC1']) + abs(loadings.loc[feat, 'PC2'])),
        })
    loading_data.sort(key=lambda x: x['importance'], reverse=True)

    # Distribution data (histograms) for key metrics
    distributions = {}
    for col in ['annualized_return_pct', 'annualized_volatility_pct', 'sharpe_ratio', 'max_drawdown_pct']:
        if col in df_feat.columns:
            values = df_feat[col].dropna()
            hist, edges = np.histogram(values, bins=30)
            distributions[col] = {
                'counts': hist.tolist(),
                'edges': [round(e, 3) for e in edges.tolist()],
            }

    # Cluster radar data (normalized means for radar chart)
    radar_cols = ['annualized_return_pct', 'annualized_volatility_pct', 'sharpe_ratio',
                  'sortino_ratio', 'max_drawdown_pct', 'calmar_ratio', 'win_rate_pct']
    radar_cols = [c for c in radar_cols if c in feature_cols]
    radar_data = {}
    for _, p in profiles.iterrows():
        cid = int(p['cluster'])
        vals = []
        for col in radar_cols:
            v = p.get(f'{col}_mean', 0)
            # Normalize to 0-100 for radar
            all_vals = df_feat[col]
            mn, mx = all_vals.min(), all_vals.max()
            if mx > mn:
                norm_v = (v - mn) / (mx - mn) * 100
            else:
                norm_v = 50
            vals.append(round(norm_v, 1))
        radar_data[cid] = vals

    output = {
        'scatter': scatter,
        'clusters': cluster_data,
        'loadings': loading_data,
        'distributions': distributions,
        'radar': {'labels': [c.replace('_pct','').replace('_',' ').title() for c in radar_cols],
                  'data': radar_data},
        'pca_variance': [round(v, 4) for v in pca_variance],
        'silhouette_scores': sil_scores,
        'n_clusters': n_clusters,
        'total_algorithms': len(df_feat),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    print(f"  Dashboard JSON: {output_path} ({os.path.getsize(output_path)//1024}KB)")
    return output


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Cluster and compare trading algorithms')
    parser.add_argument('--input', default=str(notebook_data_path('pipeline')),
                        help='Directory with pipeline output (metrics_all.csv, etc.)')
    parser.add_argument('--output', default=str(default_output_dir('clusters')), help='Output directory')
    parser.add_argument('--n-clusters', type=int, default=None, help='Number of clusters (auto if omitted)')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    print("\n[1/5] Loading pipeline results...")
    df_metrics, df_style = load_pipeline_results(args.input)

    # 2. Features
    print("\n[2/5] Preparing features...")
    df_feat, feature_cols = prepare_features(df_metrics, df_style)

    # 3. Clustering
    print("\n[3/5] Clustering...")
    df_feat, n_clusters, sil_scores, method, loadings, pca_var = run_clustering(
        df_feat, feature_cols, args.n_clusters
    )

    # 4. Profiling
    print("\n[4/5] Profiling clusters...")
    profiles = profile_clusters(df_feat, feature_cols, n_clusters)

    for _, p in profiles.iterrows():
        print(f"\n  Cluster {int(p['cluster'])}: {p['label']}")
        print(f"    N={int(p['n_algorithms'])} ({p['pct_of_total']}%)")
        print(f"    Ret={p.get('annualized_return_pct_mean',0):.2f}% | "
              f"Vol={p.get('annualized_volatility_pct_mean',0):.2f}% | "
              f"Sharpe={p.get('sharpe_ratio_mean',0):.3f} | "
              f"MDD={p.get('max_drawdown_pct_mean',0):.2f}%")

    # 5. Benchmark context
    print("\n[5/5] Benchmark comparison...")
    df_ctx, period_stats = compute_benchmark_comparison(df_metrics)
    print("\n  Rendimiento medio de los algos por ano de inicio:")
    print(period_stats.to_string())

    # --- Save ---
    print("\n  Saving files...")

    # Clusters CSV
    cluster_cols = ['name', 'cluster', 'cluster_kmeans', 'cluster_hierarchical', 'pca_x', 'pca_y'] + feature_cols
    df_feat[[c for c in cluster_cols if c in df_feat.columns]].to_csv(
        os.path.join(output_dir, 'clusters.csv'), index=False)
    print(f"  clusters.csv")

    # Profiles CSV
    profiles.to_csv(os.path.join(output_dir, 'cluster_profiles.csv'), index=False)
    print(f"  cluster_profiles.csv")

    # Period stats
    period_stats.to_csv(os.path.join(output_dir, 'period_stats.csv'))
    print(f"  period_stats.csv")

    # Dashboard JSON
    dashboard_data = export_dashboard_json(
        df_feat, profiles, feature_cols, loadings, pca_var,
        sil_scores, n_clusters,
        os.path.join(output_dir, 'cluster_analysis.json')
    )

    print(f"\nAll clustering results saved to: {output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
