# CLAUDE.md — RL Meta-Allocator sobre Algoritmos Caja Negra

## Contexto del Proyecto

Caso práctico de entrevista para un laboratorio de IA aplicada a finanzas:

1. Recibimos **transacciones de N algoritmos caja negra** (no sabemos qué hacen internamente).
2. Recibimos un **benchmark** que invierte en esos algoritmos (transacciones + performance).
3. Diseñamos un **sistema de RL meta-allocator**: asigna capital entre los algoritmos para batir al benchmark.
4. **Restricción crítica**: analizar el benchmark (frecuencia, duración, capital, riesgo) para **comparar peras con peras**.
5. Concepto de **investment clock** como herramienta de ingeniería inversa por regímenes.
6. **No usamos Gymnasium** como entorno — backtesting realista, simulador propio.

Los datos de entrenamiento provienen del ejercicio del año anterior.

## Objetivo Final

- Análisis exhaustivo del benchmark y algoritmos (reverse engineering).
- Baselines clásicos (MPT: equal weight, risk parity, min variance, max Sharpe).
- Entorno RL custom (sin Gymnasium) con simulador financiero realista.
- Agente RL entrenado (PPO baseline + comparación SAC/TD3).
- Evaluación walk-forward con métricas comparables al benchmark.

---

## Principios Fundamentales

Estas reglas son **invariantes del proyecto**. Aplicar en TODA decisión de diseño e implementación, sin excepción.

### 1. Benchmark: referencia comparable, NO objetivo a replicar
El benchmark se estudia para entender sus características (frecuencia operativa, perfil de riesgo, duración de posiciones, concentración). Pero el meta-allocator **NO tiene que copiar su comportamiento**: no tiene que invertir en los mismos activos, no tiene que tener el mismo turnover, no tiene que tener los mismos pesos. El objetivo es **batirlo en métricas comparables** (Sharpe, drawdown, información ratio), no clonarlo. Comparar peras con peras no significa ser idéntico — significa operar bajo restricciones y fricciones equivalentes para que la comparación sea justa.

### 2. Performance computacional: numba + vectorización siempre
Todo cálculo numérico intensivo debe usar `@numba.njit` con fallback a numpy vectorizado. Prohibidos los loops Python sobre filas/fechas/activos cuando exista alternativa vectorizada. Antes de implementar cualquier función de cálculo, considerar: ¿se puede expresar como operación matricial? ¿Se puede compilar con numba?

### 3. Eficiencia de memoria: diseñar para máquinas limitadas
No se conoce la máquina destino. Todo el pipeline debe ser memory-efficient:
- Procesar datos en **batches** cuando el volumen lo requiera.
- Liberar memoria explícitamente con `del` + `gc.collect()` tras operaciones grandes.
- Usar `float32` en lugar de `float64` donde la precisión no sea crítica.
- Preferir formatos compactos (parquet con compresión) sobre CSV.
- No mantener DataFrames completos en memoria si solo se necesita una ventana temporal.
- Monitorizar uso de memoria en operaciones críticas.

### 4. Reutilización: no recalcular lo que ya existe
Antes de ejecutar cualquier cálculo, **verificar si el output ya existe** en `data/processed/` o en outputs de fases anteriores. Cargar y reutilizar datos preprocesados siempre que sea posible. Si un módulo necesita features que ya se calcularon en Phase 1, leerlos de disco — no recalcularlos. Esto aplica tanto a datos como a código: si una función ya existe en otro módulo, importarla en lugar de duplicarla.

### 5. Testing obligatorio y regresión continua
Toda funcionalidad nueva debe incluir su propio conjunto de tests. Tras añadir cualquier funcionalidad, ejecutar **toda la suite de tests existente** para verificar que nada se rompe. No se considera completa una feature hasta que sus tests pasan Y los tests previos siguen pasando. Comando de regresión completa:
```bash
pytest tests/ -v --tb=short
```

---

## Estructura del Proyecto

```
rl-meta-allocator/
├── data/
│   ├── raw/                           # Datos crudos de entrada
│   │   ├── algorithms/                # *.csv de cada algoritmo (OHLC)
│   │   ├── benchmark/                 # trades.csv, cashflows.csv
│   │   ├── forex/                     # Benchmarks forex (DAT_ASCII)
│   │   ├── indices/                   # Benchmarks índices
│   │   ├── commodities/               # Benchmarks commodities
│   │   ├── futures/                   # Benchmarks futuros
│   │   └── sharadar/                  # Acciones US
│   ├── processed/                     # Outputs procesados (ver Data Organization)
│   │   ├── algorithms/                # Phase 1: returns, features, stats
│   │   ├── benchmark/                 # Phase 1: weights, positions, metrics
│   │   ├── analysis/                  # Phase 2: profiles, clustering, regimes
│   │   ├── features/                  # Derived features (cross-sectional, regime)
│   │   └── reports/                   # Phase reports (SUMMARY.md, results.json)
│   └── synthetic/                     # Datos sintéticos para testing
├── src/
│   ├── data/                          # loader, preprocessor, feature_engineering
│   ├── analysis/                      # profilers, clusterers, regime inference
│   ├── baselines/                     # MPT allocators + factor-based
│   ├── environment/                   # RL simulator, costs, constraints, rewards
│   ├── agents/                        # PPO, SAC, TD3, offline RL
│   ├── evaluation/                    # walk_forward, metrics, fast_backtester
│   └── utils/                         # paths, config, logging, plotting, numba_utils
├── configs/                           # YAML configs (default, ppo, sac, td3)
├── notebooks/                         # 01-06: EDA → benchmark → regimes → baselines → RL → eval
├── scripts/                           # run_phase1.py, run_phase3.py, etc.
├── tests/
└── outputs/                           # ALL phase reports + experiment outputs (see Output Organization)
    ├── data_pipeline/                 # Phase 1: PHASE1_SUMMARY.md, results.json, metrics.json
    ├── analysis/                      # Phase 2: PHASE2_SUMMARY.md, results.json, metrics.json
    ├── baselines/                     # Phase 3: trials/, figures/, PHASE3_SUMMARY.md
    ├── environment/                   # Phase 4: PHASE4_SUMMARY.md
    ├── rl_training/                   # Phase 5: {run_id}/checkpoints/, latest_run.txt
    ├── evaluation/                    # Phase 6: {run_id}/walk_forward/, latest_run.txt
    ├── swarm_pso/                     # Phase 7:  {run_id}/weights/, backtests/, latest_run.txt
    └── swarm_aco/                     # Phase 7A: {run_id}/weights/, backtests/, latest_run.txt
```

---

## Data Organization (IMPORTANTE)

Usar el módulo `src/utils/paths.py` para acceso centralizado a todas las rutas. **NUNCA hardcodear rutas**.

### Uso Básico

```python
from src.utils.paths import data_paths, output_paths, ensure_parent_dir

dp = data_paths()
op = output_paths()

# Leer datos procesados
returns = pd.read_parquet(dp.algorithms.returns)
features = pd.read_parquet(dp.algorithms.features)
weights = pd.read_parquet(dp.benchmark.weights)

# Guardar outputs de baselines
ensure_parent_dir(op.baselines.trials_csv)
results.to_csv(op.baselines.trials_csv)
```

### Estructura de `data/processed/`

| Path | Contenido | Generado por |
|------|-----------|--------------|
| `algorithms/returns.parquet` | Matriz retornos [dates x algos] | Phase 1 |
| `algorithms/features.parquet` | Features rolling para todos los algos | Phase 1 |
| `algorithms/stats.csv` | Estadísticas por algoritmo | Phase 1 |
| `algorithms/asset_inference.csv` | Activo inferido por algo | Phase 1 |
| `benchmark/weights.parquet` | Pesos diarios [dates x products] | Phase 1 |
| `benchmark/positions.parquet` | Posiciones diarias | Phase 1 |
| `benchmark/daily_returns.csv` | Retornos reconstruidos | Phase 1 |
| `benchmark/turnover.csv` | Turnover diario | Phase 1 |
| `benchmark/concentration.csv` | HHI diario | Phase 1 |
| `benchmark/algo_equity.parquet` | Equity solo productos benchmark | Phase 1 |
| `benchmark/algo_features.parquet` | Features solo productos benchmark | Phase 1 |
| `analysis/profiles/summary.csv` | Resumen perfiles de algos | Phase 2 |
| `analysis/profiles/full.json` | Perfiles completos | Phase 2 |
| `analysis/clustering/temporal/` | Cluster history, method comparison | Phase 1/2 |
| `analysis/clustering/behavioral/` | Familias comportamentales | Phase 2 |
| `analysis/clustering/correlation/` | Clusters por correlación | Phase 2 |
| `analysis/regimes/labels.csv` | Etiquetas de régimen | Phase 2 |
| `analysis/regimes/probabilities.parquet` | Probabilidades HMM | Phase 2 |
| `features/cross_sectional.parquet` | Features cross-sectional | Phase 2 |
| `features/regime.parquet` | Features de régimen | Phase 2 |

### Estructura de `outputs/`

| Path | Contenido | Generado por |
|------|-----------|--------------|
| `data_pipeline/PHASE1_SUMMARY.md` | Reporte Phase 1 | Phase 1 |
| `data_pipeline/phase1_results.json` | Resultados Phase 1 | Phase 1 |
| `data_pipeline/phase1_metrics.json` | Métricas de performance Phase 1 | Phase 1 |
| `analysis/PHASE2_SUMMARY.md` | Reporte Phase 2 | Phase 2 |
| `analysis/phase2_results.json` | Resultados Phase 2 | Phase 2 |
| `analysis/phase2_metrics.json` | Métricas de performance Phase 2 | Phase 2 |
| `baselines/trials/results.csv` | Resultados de todos los trials | Phase 3 |
| `baselines/figures/` | Visualizaciones de baselines | Phase 3 |
| `baselines/PHASE3_SUMMARY.md` | Reporte de baselines | Phase 3 |
| `environment/PHASE4_SUMMARY.md` | Reporte validación entorno | Phase 4 |
| `rl_training/{run_id}/checkpoints/{agent}/` | Model checkpoints (un dir por run) | Phase 5 |
| `rl_training/{run_id}/logs/{agent}/` | Training logs (un dir por run) | Phase 5 |
| `rl_training/{run_id}/run_info.json` | Metadatos del run | Phase 5 |
| `rl_training/{run_id}/PHASE5_SUMMARY.md` | Reporte del run | Phase 5 |
| `rl_training/latest_run.txt` | Puntero al run más reciente | Phase 5 |
| `evaluation/{run_id}/walk_forward/` | Resultados walk-forward (un dir por run) | Phase 6 |
| `evaluation/{run_id}/PHASE6_SUMMARY.md` | Reporte del run | Phase 6 |
| `evaluation/latest_run.txt` | Puntero al run más reciente | Phase 6 |
| `swarm_pso/{run_id}/weights/` | Pesos PSO (un dir por run) | Phase 7 |
| `swarm_pso/{run_id}/backtests/` | Retornos del backtest | Phase 7 |
| `swarm_pso/{run_id}/reports/` | Summary y comparison JSON | Phase 7 |
| `swarm_pso/{run_id}/PHASE7_SUMMARY.md` | Reporte del run | Phase 7 |
| `swarm_pso/latest_run.txt` | Puntero al run más reciente | Phase 7 |
| `swarm_aco/{run_id}/weights/` | Pesos ACO (un dir por run) | Phase 7A |
| `swarm_aco/{run_id}/backtests/` | Retornos del backtest | Phase 7A |
| `swarm_aco/{run_id}/reports/` | Summary y comparison JSON | Phase 7A |
| `swarm_aco/{run_id}/PHASE7A_SUMMARY.md` | Reporte del run | Phase 7A |
| `swarm_aco/latest_run.txt` | Puntero al run más reciente | Phase 7A |

### Best Practices de Data

1. **Usar paths.py**: Importar `from src.utils.paths import data_paths, output_paths`
2. **Crear directorios**: Usar `ensure_parent_dir(path)` antes de escribir
3. **Formato**: Parquet para DataFrames grandes, CSV para tablas pequeñas, JSON para configs/metadata
4. **Compresión**: Parquet usa snappy por defecto, suficiente para este proyecto
5. **Verificar existencia**: Antes de recalcular, verificar si el archivo ya existe
6. **No duplicar**: Si un dato se genera en Phase N, fases posteriores lo leen, no lo regeneran

---

## Estado Actual — Phase 1 COMPLETADA ✅, Phase 3 EN PROGRESO

**Módulos implementados**:
- `src/data/` completo (loader, preprocessor, feature_engineering)
- `src/analysis/asset_inference.py`, `src/analysis/algo_clusterer.py`
- `src/baselines/` completo (6 allocators + base.py con factor selection)
- `src/evaluation/fast_backtester.py` (backtesting optimizado con caching)
- `src/utils/paths.py` (gestión centralizada de rutas)
- `src/utils/numba_utils.py` (funciones numba optimizadas)

**Scripts**:
- `python scripts/run_phase1.py` — Pipeline Phase 1
- `python scripts/run_phase3.py` — Baseline backtesting (351 trials)

**146 tests passing** across 5 test files (loader: 26, preprocessor: 18, features: 34, asset_inference: 35, temporal_clustering: 33).

### Outputs actuales en `data/processed/` (pendiente migración):
Los archivos existentes están en estructura flat. Ver `scripts/migrate_data.py` para migrar a nueva estructura.

- `algo_returns.parquet` → `algorithms/returns.parquet`
- `algo_features.parquet` → `algorithms/features.parquet`
- `algo_stats.csv` → `algorithms/stats.csv`
- `asset_inference.csv` → `algorithms/asset_inference.csv`
- `benchmark_*.parquet/csv` → `benchmark/*.parquet/csv`
- `phase1_results.json`, `PHASE1_SUMMARY.md` → `reports/phase1/`
- `temporal_clusters/` → `analysis/clustering/temporal/`

### Notas clave de Phase 1:
- **Loader**: Detección automática de formato de fecha y columnas OHLC (case-insensitive). Soporta resample a diario. Carga benchmarks desde `forex/`, `indices/`, `commodities/`, `futures/`, `sharadar/`.
- **Preprocessor**: Dead tail trimming automático. Reconstruye equity curves, retornos, pesos/posiciones del benchmark.
- **Features**: Cumulative + rolling (ventanas 5/21/63) + cross-sectional regime features. Usa `@numba.njit` con fallback numpy.
- **Asset Inference**: Two-stage (fast Pearson screen → deep 6-signal analysis). Multi-asset detection.
- **Clustering**: Semanal, tres horizontes (cumulative/weekly/monthly). Métodos: KMeans, GMM, Hierarchical, DBSCAN, HDBSCAN.

---

## Fases Pendientes

### FASE 2: Análisis y Reverse Engineering
- `algo_profiler.py`: Perfil completo por algoritmo (performance, operativa, riesgo, comportamiento por régimen).
- `benchmark_profiler.py`: Inferir política implícita — sizing, temporal, riesgo, régimen.
- `latent_regime_inference.py`: Inferencia de régimen latente (NO clasificación macro). Arquitectura de dos capas: (A) régimen de mercado global, (B) comportamiento del benchmark. Pipeline: máscara actividad → taxonomía familias → features temporales → HMM/Fuzzy/GMM → análisis condicional. **Usar máscara de actividad explícita, NUNCA fillna(9999999)**.
- `correlation_analyzer.py`: Correlaciones rolling/estáticas, estabilidad por régimen, diversification ratio.
- Módulo legacy `regime_detector.py` se mantiene para compatibilidad.

### FASE 3: Baselines Clásicos
- 6 baselines: Equal Weight, Risk Parity, Min Variance, Max Sharpe, Momentum, Vol Targeting.
- Todos implementan `BaseAllocator.compute_weights()` y `apply_constraints()`.
- Mismas restricciones, fricciones y evaluación walk-forward que el agente RL.
- Restricciones se calibran al benchmark (max_weight ~0.40, max_turnover ~0.30, rebalanceo semanal).

### FASE 4: Entorno RL Custom
- `market_simulator.py`: Event-driven, simula retornos de algos + costes + restricciones. NO simula mercados directamente.
- `trading_env.py`: Wrapper compatible con SB3 (usa `gymnasium.spaces` solo como utilidad técnica, NO hereda de `gym.Env`).
- `cost_model.py`: Comisión fija + spread + slippage + impacto de mercado. Calibrar conservadoramente.
- `constraints.py`: max/min peso, max turnover, max exposure, vol targeting opcional.
- `reward.py`: Alpha vs benchmark penalizado por costes, turnover, drawdown, riesgo. Variantes: alpha penalizado, info ratio, risk-adjusted, bonus diversificación.

### FASE 5: Agentes RL
- PPO (baseline principal), SAC (off-policy continuo), TD3 (alternativa determinista).
- Usar `stable-baselines3`. Red MLP [256, 256] por defecto.
- Placeholder offline RL (CQL/IQL) con documentación de por qué tiene sentido.
- **Enfoque híbrido preferido**: cartera base clásica (risk parity) + RL como tilts dinámicos con restricciones de magnitud.

### FASE 6: Evaluación
- Walk-forward: train 252d / val 63d / test 63d / step 63d. NUNCA train/test aleatorio.
- Métricas: performance absoluta + relativa (excess return, tracking error, information ratio, alpha/beta) + operativa (turnover, holding period, HHI) + riesgo (VaR, CVaR).
- Tabla comparativa: Benchmark vs EW vs RP vs MinVar vs MaxSharpe vs PPO vs SAC vs TD3.
- **Criterio de éxito**: batir benchmark Y baselines a igualdad de riesgo y fricciones.
- Sanity checks: EW constante = baseline EW, costes=0 → más rotación, turnover penalty alta → buy-and-hold, replicar benchmark → reward ≈ 0.

---

## Decisiones Técnicas Clave

- **¿Por qué no Gymnasium?** Simulador event-driven propio con costes, restricciones y datos reales. Solo `gymnasium.spaces` para compatibilidad SB3.
- **Algoritmos**: PPO siempre como baseline. SAC si queremos eficiencia muestral. TD3 si reward ruidoso. Offline RL si solo hay logs históricos.
- **Comparación justa**: restricciones y fricciones equivalentes, walk-forward, métricas de retorno Y riesgo. Comparable, no idéntico (ver Principio 1).
- **Híbrido MPT+RL**: `target_weights = risk_parity_base + rl_tilts`, con constraints sobre el tilt.

---

## Convenciones de Código

_(Además de los Principios Fundamentales, que tienen prioridad absoluta)_

- Type hints en todo el código.
- `logging` estándar, no prints.
- Parquet con compresión para almacenamiento intermedio.
- Config centralizada en YAML, no hardcodear parámetros.
- Fijar seeds (numpy, torch, entorno) para reproducibilidad.
- Cada módulo testeable de forma independiente.
- Ser conservador si hay dudas sobre si un feature o restricción es realista.
- **No saltar fases. Cada fase depende de la anterior.**

## Testing

```bash
# Regresión completa (OBLIGATORIO tras cada cambio — ver Principio 5)
pytest tests/ -v --tb=short

# Phase 1 (146 tests implementados)
pytest tests/test_loader.py tests/test_preprocessor.py tests/test_feature_engineering.py tests/test_asset_inference.py tests/test_temporal_clustering.py -v

# Pendientes: test_cost_model, test_environment, test_metrics, test_walk_forward, test_baselines
```