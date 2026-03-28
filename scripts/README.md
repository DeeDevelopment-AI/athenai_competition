# Scripts

Esta carpeta contiene los puntos de entrada ejecutables del proyecto. La idea general es:

1. `Phase 1` construye el dataset procesado.
2. `Phase 2` analiza el universo y genera artefactos de clustering, perfiles y regímenes.
3. `Phase 3` calcula baselines clásicos.
4. `Phase 4` valida el entorno de entrenamiento.
5. `Phase 5` entrena agentes RL.
6. `Phase 6` evalúa esos agentes con walk-forward.
7. `Phase 7` y `Phase 7A` optimizan meta-asignación con PSO y ACO.
8. Los scripts auxiliares sirven para migración, tuning, benchmarking o construcción de subconjuntos de universo.

## Si quieres X, ejecuta Y

| Si quieres... | Ejecuta... |
|---|---|
| regenerar el dataset procesado | `python scripts/run_phase1.py` |
| recalcular perfiles, regímenes y clusters de `Phase 2` | `python scripts/run_phase2.py` |
| comparar estrategias clásicas sin RL | `python scripts/run_phase3.py --full` |
| validar que el entorno RL funciona antes de entrenar | `python scripts/run_phase4.py --full` |
| entrenar un PPO estándar | `python scripts/run_phase5.py --agent ppo --timesteps 1M --gpu-env --max-resources` |
| entrenar RL solo sobre los mejores clusters temporales de `Phase 2` | `python scripts/run_phase5.py --phase2-cluster-filter --phase2-cluster-source temporal_cumulative ...` |
| entrenar RL a retorno puro | `python scripts/run_phase5.py --reward-type pure_returns --no-hybrid ...` |
| evaluar un run RL con walk-forward y exportar rebalanceos auditables | `python scripts/run_phase6.py --run-id <run_id> --include-baselines` |
| optimizar cartera con PSO | `python scripts/run_phase7.py ...` |
| evaluar PSO con walk-forward sobre datos no vistos y exportar pesos por rebalanceo | `python scripts/run_phase7_walk_forward.py ...` |
| optimizar cartera con ACO | `python scripts/run_phase7_aco.py ...` |
| evaluar ACO con walk-forward sobre datos no vistos y exportar pesos por rebalanceo | `python scripts/run_phase7_aco_walk_forward.py ...` |
| barrer `top-k` de clusters en `Phase 5` | `python scripts/sweep_phase5_cluster_topk.py ...` |
| resumir varios runs de `Phase 6` en una tabla | `python scripts/summarize_phase6_runs.py ...` |
| hacer tuning de PSO antes del run final | `python scripts/tune_phase7.py ...` |
| hacer tuning de ACO y luego walk-forward | `python scripts/tune_phase7_aco.py ...` y luego `python scripts/run_phase7_aco_walk_forward.py ...` |
| construir un dataset alternativo restringido a un cluster temporal | `python scripts/build_temporal_cluster_dataset.py --output-dir ...` |
| migrar datos desde estructura legacy | `python scripts/migrate_data.py --execute` |

## Convenciones

- Casi todos los scripts de fase heredan de [base_runner.py](/C:/Users/Admin/DataspellProjects/athenai_competition/scripts/base_runner.py).
- Opciones comunes en fases:
  - `--verbose`, `-v`
  - `--dry-run`
  - `--output-dir`, `-o`
  - `--log-file`
  - `--no-report`
- Artefactos típicos por run:
  - `phase{N}_metrics.json`: métricas de ejecución, tiempos y memoria.
  - `phase{N}_results.json`: resultados funcionales del script.
  - `PHASE{N}_SUMMARY.md`: resumen legible.

## Vista rápida

| Script | Rol | Entrada principal | Salida principal |
|---|---|---|---|
| `run_phase1.py` | construir datos procesados | `data/raw` | `data/processed` |
| `run_phase2.py` | análisis, clustering y regímenes | `data/processed` | `data/processed/analysis` + snapshot en `outputs/analysis/<run_id>` |
| `run_phase3.py` | backtests clásicos | `data/processed` | `outputs/baselines/<run_id>` |
| `run_phase4.py` | validación del entorno RL | `data/processed` | `outputs/environment/<run_id>` |
| `run_phase5.py` | entrenamiento RL | `data/processed` | `outputs/rl_training/<run_id>` |
| `run_phase6.py` | evaluación walk-forward RL | modelos de Phase 5 | `outputs/evaluation/<run_id>` |
| `run_phase7.py` | meta-allocator PSO | `data/processed` + `analysis` | `outputs/swarm_pso/<run_id>` |
| `run_phase7_walk_forward.py` | walk-forward de PSO | `data/processed` + `analysis` | `outputs/swarm_pso` |
| `run_phase7_aco.py` | meta-allocator ACO | `data/processed` + `analysis` | `outputs/swarm_aco/<run_id>` |
| `run_phase7_aco_walk_forward.py` | walk-forward de ACO | mejor config de tuning ACO | `outputs/swarm_aco` |
| `tune_phase7.py` | tuning temporal de PSO | `data/processed` + `analysis` | `outputs/swarm_pso/<run_id>` |
| `tune_phase7_aco.py` | tuning temporal de ACO | `data/processed` + `analysis` | `outputs/swarm_aco/<run_id>` |
| `run_temporal_clustering.py` | clustering temporal standalone | datos raw/procesados según uso | carpeta de salida elegida |
| `build_temporal_cluster_dataset.py` | construir universo filtrado por cluster temporal | `data/processed` | dataset procesado alternativo |
| `sweep_phase5_cluster_topk.py` | barrido de `top-k` en RL | `Phase 2 analysis` + `Phase 5` | múltiples runs en `outputs/rl_training` |
| `summarize_phase6_runs.py` | resumen de runs evaluados | `outputs/evaluation` | tabla comparativa en consola o CSV |
| `migrate_data.py` | migración de estructura de datos | árbol legacy | árbol organizado |
| `benchmark_optimizations.py` | benchmark de utilidades numéricas | código interno | salida en consola |
| `base_runner.py` | infraestructura común | n/a | n/a |

## Scripts de fase

### `run_phase1.py`

Qué hace:
- carga los algoritmos y benchmark desde `data/raw`,
- limpia series,
- genera matriz de retornos,
- genera features,
- calcula stats por algoritmo,
- infiere asset class y dirección cuando aplica.

Cómo se usa:
```powershell
python scripts/run_phase1.py
python scripts/run_phase1.py --sample 100
python scripts/run_phase1.py --benchmark-only
python scripts/run_phase1.py --skip-features --no-inference
```

Argumentos principales:
- `--sample`: procesa solo `N` algoritmos.
- `--benchmark-only`: procesa solo productos benchmark.
- `--skip-features`: omite ingeniería de features.
- `--no-trim`: desactiva trimming de colas muertas.
- `--no-inference`: desactiva asset inference.
- `--no-trade-features`: omite features basadas en actividad/trading.
- `--data-path`: ruta alternativa a los datos raw.

Artefactos que genera:
- `data/processed/algorithms/returns.parquet`
- `data/processed/algorithms/features.parquet`
- `data/processed/algorithms/stats.csv`
- `data/processed/algorithms/asset_inference.csv`
- `data/processed/benchmark/weights.parquet`
- `data/processed/benchmark/positions.parquet`
- `data/processed/benchmark/daily_returns.csv`
- `outputs/data_pipeline/<run_id>/phase1_metrics.json`
- `outputs/data_pipeline/<run_id>/phase1_results.json`

Qué consigue al final:
- deja un dataset procesado y consistente que sirve como entrada única del resto del pipeline.

### `run_phase2.py`

Qué hace:
- perfila algoritmos,
- reverse-engineer del benchmark,
- infiere familias/regímenes,
- analiza correlaciones,
- ejecuta clustering temporal,
- guarda una snapshot del análisis para reutilizarlo de forma reproducible en fases posteriores.

Cómo se usa:
```powershell
python scripts/run_phase2.py
python scripts/run_phase2.py --sample 500 --skip-temporal
python scripts/run_phase2.py --family-clustering-method hdbscan --family-refinement-strategy direct
python scripts/run_phase2.py --clustering-method hdbscan --n-clusters 7
```

Argumentos principales:
- `--sample`: submuestra de algoritmos.
- `--n-regimes`: número de regímenes latentes.
- `--n-families`: número de familias.
- `--family-clustering-method`: `kmeans`, `gmm`, `hierarchical`, `dbscan`, `hdbscan`.
- `--family-refinement-strategy`: `none`, `direct`, `self_training`, `confidence_refinement`, `anomaly`.
- `--family-confidence-threshold`
- `--family-refinement-max-iter`
- `--skip-inference`
- `--skip-correlations`
- `--skip-temporal`
- `--n-clusters`: número de clusters temporales.
- `--clustering-method`: método del clustering temporal.
- `--input-dir`: dataset procesado alternativo.

Artefactos que genera:
- `data/processed/analysis/profiles/summary.csv`
- `data/processed/analysis/profiles/full.json`
- `data/processed/analysis/benchmark_profile/metrics.json`
- `data/processed/analysis/benchmark_profile/report.txt`
- `data/processed/analysis/clustering/behavioral/*`
- `data/processed/analysis/clustering/correlation/*`
- `data/processed/analysis/clustering/temporal/*`
- `data/processed/analysis/regimes/*`
- `outputs/analysis/<run_id>/phase2_metrics.json`
- `outputs/analysis/<run_id>/phase2_results.json`
- `outputs/analysis/<run_id>/analysis_snapshot/*`

Qué consigue al final:
- produce la capa semántica del universo: perfiles, familias, clusters temporales y regímenes, lista para selección de universo, reporting y swarm optimization.

### `run_phase3.py`

Qué hace:
- ejecuta y compara baselines clásicos de asignación,
- opcionalmente aplica walk-forward,
- guarda trials y resúmenes de rendimiento.

Cómo se usa:
```powershell
python scripts/run_phase3.py
python scripts/run_phase3.py --quick
python scripts/run_phase3.py --full
python scripts/run_phase3.py --baseline risk_parity --walk-forward
```

Argumentos principales:
- `--quick`: modo rápido.
- `--full`: modo completo con más configuraciones.
- `--baseline`: `equal_weight`, `risk_parity`, `min_variance`, `max_sharpe`, `momentum`, `vol_targeting`.
- `--fresh`: limpia resultados previos del run.
- `--walk-forward`: añade validación walk-forward.
- `--n-top`: número de configuraciones top para walk-forward.
- `--input-dir`: dataset procesado alternativo.

Artefactos que genera:
- `outputs/baselines/<run_id>/trials/results.csv`
- `outputs/baselines/<run_id>/figures/*`
- `outputs/baselines/<run_id>/phase3_metrics.json`
- `outputs/baselines/<run_id>/phase3_results.json`

Qué consigue al final:
- establece un benchmark interno serio contra el que comparar RL o swarm allocators.

### `run_phase4.py`

Qué hace:
- valida que el entorno de trading y la función de reward se comporten como esperas,
- ejecuta episodios de prueba,
- opcionalmente lanza validación más dura con stress tests.

Cómo se usa:
```powershell
python scripts/run_phase4.py
python scripts/run_phase4.py --sample 50 --episodes 10
python scripts/run_phase4.py --full --rebalance-freq weekly
```

Argumentos principales:
- `--sample`
- `--episodes`
- `--full`
- `--rebalance-freq`: `daily`, `weekly`, `monthly`, `quarterly`
- `--input-dir`

Artefactos que genera:
- `outputs/environment/<run_id>/phase4_metrics.json`
- `outputs/environment/<run_id>/phase4_results.json`

Qué consigue al final:
- asegura que antes de entrenar RL el entorno no tenga errores conceptuales o mecánicos graves.

### `run_phase5.py`

Qué hace:
- entrena agentes RL `PPO`, `SAC`, `TD3`,
- construye train/val/test,
- crea el entorno vectorizado,
- opcionalmente usa encoder,
- opcionalmente usa hybrid mode,
- opcionalmente hace warm-start con behavioral cloning,
- opcionalmente restringe el universo con clusters de `Phase 2`.

Cómo se usa:
```powershell
python scripts/run_phase5.py
python scripts/run_phase5.py --agent ppo --timesteps 1M --gpu-env --max-resources
python scripts/run_phase5.py --agent sac --reward-type pure_returns --no-hybrid
python scripts/run_phase5.py --phase2-cluster-filter --phase2-analysis-dir data/processed/analysis --phase2-cluster-source temporal_cumulative
```

Argumentos principales:
- agente y entrenamiento:
  - `--agent {ppo,sac,td3,all}`
  - `--quick`
  - `--timesteps`
  - `--eval-freq`
  - `--n-eval-episodes`
  - `--learning-rate`
  - `--resume`
- recursos:
  - `--max-resources`
  - `--n-envs`
  - `--use-subproc`
  - `--gpu-env`
- universo y representación:
  - `--input-dir`
  - `--sample`
  - `--no-encoder`
  - `--pca-encoder`
  - `--pca-components`
  - `--min-days-active`
- modo híbrido:
  - `--no-hybrid`
  - `--base-allocator`
  - `--max-tilt`
- behavioral cloning:
  - `--pretrain-bc`
  - `--bc-strategy`
  - `--bc-epochs`
  - `--bc-lr`
  - `--bc-batch-size`
  - `--bc-lookback`
- reward:
  - `--reward-type`
- filtros basados en clusters:
  - `--cluster-filter` y sus opciones legacy por familias
  - `--phase2-cluster-filter`
  - `--phase2-analysis-dir`
  - `--phase2-cluster-source`
  - `--phase2-cluster-score-mode`
  - `--phase2-cluster-top-k`
  - `--phase2-cluster-min-size`
  - `--phase2-cluster-min-return`
  - `--phase2-cluster-max-vol`
  - `--phase2-cluster-full-history`

Artefactos que genera:
- `outputs/rl_training/<run_id>/checkpoints/<agent>/final_model.zip`
- `outputs/rl_training/<run_id>/checkpoints/<agent>/best_model.zip`
- `outputs/rl_training/<run_id>/checkpoints/<agent>/vecnormalize.pkl`
- `outputs/rl_training/<run_id>/checkpoints/<agent>/universe_encoder.pkl`
- `outputs/rl_training/<run_id>/logs/<agent>/metrics.csv`
- `outputs/rl_training/<run_id>/run_info.json`
- `outputs/rl_training/<run_id>/cluster_selection/selected_algos.csv`
- `outputs/rl_training/<run_id>/cluster_selection/cluster_ranking.csv`
- `outputs/rl_training/<run_id>/phase5_metrics.json`
- `outputs/rl_training/<run_id>/phase5_results.json`

Qué consigue al final:
- produce modelos entrenados y empaquetados, con trazabilidad suficiente para evaluación y reproducción.

### `run_phase6.py`

Qué hace:
- carga un run concreto de `Phase 5`,
- genera folds walk-forward,
- evalúa el agente en cada fold,
- opcionalmente evalúa baselines en el mismo protocolo,
- genera comparación final.

Cómo se usa:
```powershell
python scripts/run_phase6.py
python scripts/run_phase6.py --run-id 20260327_181451_ppo
python scripts/run_phase6.py --include-baselines --folds 5
python scripts/run_phase6.py --models-dir outputs/rl_training/<run_id>/checkpoints
```

Argumentos principales:
- `--agent`
- `--quick`
- `--include-baselines`
- `--folds`
- `--train-window`
- `--val-window`
- `--test-window`
- `--step-size`
- `--expanding`
- `--models-dir`
- `--run-id`
- `--list-runs`
- `--sample`
- `--input-dir`

Artefactos que genera:
- `outputs/evaluation/<run_id>/walk_forward/*`
- `outputs/evaluation/<run_id>/walk_forward/<agent>_allocations.csv`
- `outputs/evaluation/<run_id>/walk_forward/<baseline>_allocations.csv`
- `outputs/evaluation/<run_id>/phase6_metrics.json`
- `outputs/evaluation/<run_id>/phase6_results.json`
- `outputs/evaluation/<run_id>/results.json`

Qué consigue al final:
- convierte un run de entrenamiento en evidencia out-of-sample comparable y presentable, con trazabilidad de qué algoritmos y pesos estuvieron activos en cada periodo entre rebalanceos.

### `run_phase7.py`

Qué hace:
- ejecuta meta-asignación con `PSO`,
- selecciona un subconjunto candidato de algoritmos,
- optimiza pesos bajo una función objetivo configurable,
- puede generar resumen temporal train/validation/test,
- y ahora también puede restringir el universo a los mejores clusters de `Phase 2`.

Cómo se usa:
```powershell
python scripts/run_phase7.py
python scripts/run_phase7.py --top-k 24 --particles 96 --iterations 70
python scripts/run_phase7.py --analysis-dir data/processed/analysis --selection-factor rolling_sharpe_21d
python scripts/run_phase7.py --analysis-dir data/processed/analysis --phase2-cluster-filter --phase2-cluster-source temporal_cumulative --phase2-cluster-top-k 2
```

Argumentos principales:
- universo:
  - `--sample`
  - `--top-k`
  - `--lookback-window`
  - `--min-history`
  - `--selection-factor`
  - `--input-dir`
  - `--analysis-dir`
  - `--phase2-cluster-filter`
  - `--phase2-cluster-source`
  - `--phase2-cluster-score-mode`
  - `--phase2-cluster-top-k`
  - `--phase2-cluster-min-size`
  - `--phase2-cluster-min-return`
  - `--phase2-cluster-max-vol`
  - `--phase2-cluster-full-history`
- PSO:
  - `--particles`
  - `--iterations`
  - `--inertia`
  - `--cognitive-weight`
  - `--social-weight`
- restricciones:
  - `--max-weight`
  - `--max-family-exposure`
  - `--min-active-weight`
  - `--min-gross-exposure`
- función objetivo:
  - `--expected-return-weight`
  - `--volatility-weight`
  - `--tracking-error-weight`
  - `--turnover-weight`
  - `--concentration-weight`
  - `--diversification-weight`
  - `--family-penalty-weight`
  - `--risk-budget-weight`
  - `--sparsity-penalty-weight`
  - `--objective-name`
  - `--normalize-objective-metrics`
- evaluación temporal:
  - `--temporal-split`
  - `--train-ratio`
  - `--validation-ratio`
- ejecución:
  - `--cpu-only`
  - `--seed`

Artefactos que genera:
- `outputs/swarm_pso/<run_id>/summary.json`
- `outputs/swarm_pso/<run_id>/comparison.json`
- `outputs/swarm_pso/<run_id>/comparison.csv`
- `outputs/swarm_pso/<run_id>/portfolio_returns.csv`
- `outputs/swarm_pso/<run_id>/weights.csv`
- `outputs/swarm_pso/<run_id>/phase7_metrics.json`
- `outputs/swarm_pso/<run_id>/phase7_results.json`

Qué consigue al final:
- encuentra una cartera meta-allocada por PSO y la evalúa con métricas de portfolio y comparación.

### `run_phase7_walk_forward.py`

Qué hace:
- evalúa la estrategia PSO con folds walk-forward sobre datos no vistos,
- compara el comportamiento out-of-sample frente al benchmark,
- soporta el mismo filtro por mejores clusters de `Phase 2` que `run_phase7.py`.

Cómo se usa:
```powershell
python scripts/run_phase7_walk_forward.py
python scripts/run_phase7_walk_forward.py --analysis-dir data/processed/analysis --phase2-cluster-filter --phase2-cluster-source temporal_cumulative --phase2-cluster-top-k 2
python scripts/run_phase7_walk_forward.py --max-folds 5 --expanding
```

Argumentos principales:
- `--sample`
- `--input-dir`
- `--analysis-dir`
- `--phase2-cluster-filter`
- `--phase2-cluster-source`
- `--phase2-cluster-score-mode`
- `--phase2-cluster-top-k`
- `--phase2-cluster-min-size`
- `--phase2-cluster-min-return`
- `--phase2-cluster-max-vol`
- `--phase2-cluster-full-history`
- `--selection-factor`
- `--lookback-window`
- `--min-history`
- `--particles`
- `--iterations`
- `--rebalance-freq`
- `--selection-mode`
- `--cpu-only`
- `--start-date`
- `--end-date`
- `--train-window`
- `--validation-window`
- `--test-window`
- `--step-size`
- `--expanding`
- `--max-folds`

Artefactos que genera:
- `outputs/swarm_pso/<run_id>/walk_forward/folds.csv`
- `outputs/swarm_pso/<run_id>/walk_forward/rebalance_allocations.csv`
- `outputs/swarm_pso/<run_id>/walk_forward/portfolio_test_returns.csv`
- `outputs/swarm_pso/<run_id>/walk_forward/benchmark_test_returns.csv`
- `outputs/swarm_pso/<run_id>/walk_forward/comparison.csv`
- `outputs/swarm_pso/<run_id>/walk_forward/summary.json`

Qué consigue al final:
- convierte PSO en una comparación robusta contra benchmark sobre datos no vistos y deja un log auditable de algoritmos/pesos por tramo de rebalanceo.

### `run_phase7_aco.py`

Qué hace:
- equivalente conceptual a `run_phase7.py`, pero usando `ACO`,
- soporta GPU salvo que se fuerce CPU,
- permite búsqueda sobre factores de selección y objetivo configurable,
- y también puede restringir el universo a los mejores clusters de `Phase 2`.

Cómo se usa:
```powershell
python scripts/run_phase7_aco.py
python scripts/run_phase7_aco.py --top-k 24 --ants 96 --iterations 60
python scripts/run_phase7_aco.py --cpu-only
python scripts/run_phase7_aco.py --analysis-dir data/processed/analysis --phase2-cluster-filter --phase2-cluster-source temporal_cumulative --phase2-cluster-top-k 2
```

Argumentos principales:
- universo:
  - `--sample`
  - `--top-k`
  - `--lookback-window`
  - `--min-history`
  - `--selection-factor`
  - `--input-dir`
  - `--analysis-dir`
  - `--phase2-cluster-filter`
  - `--phase2-cluster-source`
  - `--phase2-cluster-score-mode`
  - `--phase2-cluster-top-k`
  - `--phase2-cluster-min-size`
  - `--phase2-cluster-min-return`
  - `--phase2-cluster-max-vol`
  - `--phase2-cluster-full-history`
- ACO:
  - `--ants`
  - `--iterations`
  - `--weight-buckets`
  - `--pheromone-power`
  - `--heuristic-power`
  - `--evaporation-rate`
  - `--pheromone-deposit-scale`
  - `--elite-ants`
- restricciones y objetivo:
  - `--max-weight`
  - `--max-family-exposure`
  - `--expected-return-weight`
  - `--volatility-weight`
  - `--turnover-weight`
  - `--concentration-weight`
  - `--diversification-weight`
  - `--family-penalty-weight`
  - `--family-alpha-reward-weight`
  - `--risk-budget-weight`
  - `--sparsity-penalty-weight`
  - `--entropy-reward-weight`
  - `--sharpe-weight`
  - `--objective-name`
  - `--normalize-objective-metrics`
- evaluación temporal:
  - `--temporal-split`
  - `--train-ratio`
  - `--validation-ratio`
- ejecución:
  - `--cpu-only`
  - `--seed`

Artefactos que genera:
- `outputs/swarm_aco/<run_id>/summary.json`
- `outputs/swarm_aco/<run_id>/comparison.json`
- `outputs/swarm_aco/<run_id>/comparison.csv`
- `outputs/swarm_aco/<run_id>/phase8_metrics.json` o métricas equivalentes del run
- `outputs/swarm_aco/<run_id>/phase8_results.json` o resultados equivalentes del run

Qué consigue al final:
- obtiene una cartera meta-allocada mediante colonia de hormigas y deja trazabilidad completa del experimento.

### `run_phase7_aco_walk_forward.py`

Qué hace:
- carga la mejor configuración encontrada por `tune_phase7_aco.py`,
- la evalúa en múltiples folds walk-forward,
- genera resumen stitched de retornos y comparación contra benchmark,
- y ahora también acepta filtro por mejores clusters de `Phase 2`.

Cómo se usa:
```powershell
python scripts/run_phase7_aco_walk_forward.py
python scripts/run_phase7_aco_walk_forward.py --config-path outputs/swarm_aco/<run_id>/best_config.json
python scripts/run_phase7_aco_walk_forward.py --max-folds 5 --expanding
python scripts/run_phase7_aco_walk_forward.py --analysis-dir data/processed/analysis --phase2-cluster-filter --phase2-cluster-source temporal_cumulative --phase2-cluster-top-k 2
```

Argumentos principales:
- `--config-path`
- `--sample`
- `--input-dir`
- `--analysis-dir`
- `--phase2-cluster-filter`
- `--phase2-cluster-source`
- `--phase2-cluster-score-mode`
- `--phase2-cluster-top-k`
- `--phase2-cluster-min-size`
- `--phase2-cluster-min-return`
- `--phase2-cluster-max-vol`
- `--phase2-cluster-full-history`
- `--start-date`
- `--end-date`
- `--train-window`
- `--validation-window`
- `--test-window`
- `--step-size`
- `--expanding`
- `--max-folds`
- `--cpu-only`

Artefactos que genera:
- `walk_forward/folds.csv`
- `walk_forward/rebalance_allocations.csv`
- `walk_forward/portfolio_test_returns.csv`
- `walk_forward/benchmark_test_returns.csv`
- `walk_forward/summary.json`
- `walk_forward/comparison.json`
- `walk_forward/comparison.csv`
- `walk_forward/SUMMARY.md`

Qué consigue al final:
- convierte la mejor configuración ACO en una evaluación temporal robusta y deja un log auditable de algoritmos/pesos por tramo de rebalanceo.

## Scripts de tuning

### `tune_phase7.py`

Qué hace:
- prueba múltiples configuraciones de `Phase 7`,
- usa split temporal o validación simple,
- elige la mejor según retorno o Sharpe de validación o blend.

Cómo se usa:
```powershell
python scripts/tune_phase7.py --max-trials 8
python scripts/tune_phase7.py --selection-objective blended_sharpe
```

Argumentos principales:
- `--sample`
- `--max-trials`
- `--cpu-only`
- `--input-dir`
- `--analysis-dir`
- `--start-date`
- `--end-date`
- `--train-ratio`
- `--validation-ratio`
- `--max-validation-drawdown`
- `--max-test-drawdown`
- `--validation-weight`
- `--test-weight`
- `--selection-objective`

Artefactos que genera:
- ranking de trials,
- mejor configuración seleccionada,
- resumen del proceso de tuning en la carpeta del run.

Qué consigue al final:
- reduce el espacio de búsqueda de PSO a una configuración defendible antes del run final.

### `tune_phase7_aco.py`

Qué hace:
- busca combinaciones de factores y configuraciones para `ACO`,
- puede evaluar por `walk_forward` o `single_split`,
- selecciona la mejor combinación con restricciones de drawdown y objetivo configurable.

Cómo se usa:
```powershell
python scripts/tune_phase7_aco.py --quick
python scripts/tune_phase7_aco.py --max-trials 20 --selection-scheme walk_forward
```

Argumentos principales:
- `--sample`
- `--quick`
- `--max-trials`
- `--factor`
- `--min-combination-size`
- `--max-combination-size`
- `--input-dir`
- `--analysis-dir`
- `--start-date`
- `--end-date`
- `--selection-scheme`
- `--train-ratio`
- `--validation-ratio`
- `--train-window`
- `--validation-window`
- `--test-window`
- `--step-size`
- `--expanding`
- `--max-folds`
- `--max-validation-drawdown`
- `--max-test-drawdown`
- `--validation-weight`
- `--test-weight`
- `--selection-objective`
- `--allow-test-in-selection`

Artefactos que genera:
- lista de trials de factores,
- mejor configuración encontrada,
- artefactos de scoring temporal.

Qué consigue al final:
- produce el `best_config.json` que normalmente se pasa a `run_phase7_aco_walk_forward.py`.

## Scripts auxiliares

### `run_temporal_clustering.py`

Qué hace:
- ejecuta clustering temporal standalone fuera de `Phase 2`,
- útil para experimentación rápida con métodos alternativos.

Cómo se usa:
```powershell
python scripts/run_temporal_clustering.py --data-path data/raw --output outputs/temp_clusters
python scripts/run_temporal_clustering.py --method hdbscan --n-clusters 7 --compare-methods
```

Argumentos principales:
- `--data-path`
- `--output`
- `--start-date`
- `--n-clusters`
- `--sample`
- `--compare-methods`
- `--method`

Artefactos que genera:
- historial de clusters,
- comparativas entre métodos,
- ficheros de salida en la carpeta indicada por `--output`.

Qué consigue al final:
- te deja aislar y estudiar el clustering temporal sin recorrer toda la fase 2.

### `build_temporal_cluster_dataset.py`

Qué hace:
- crea un dataset procesado alternativo restringido a un `cluster_cumulative`,
- si no le das `cluster-id`, elige automáticamente el mejor cluster temporal por retorno alto y volatilidad baja.

Cómo se usa:
```powershell
python scripts/build_temporal_cluster_dataset.py --output-dir data/processed/cluster_highret_lowvol
python scripts/build_temporal_cluster_dataset.py --processed-root data/processed --cluster-id 6 --output-dir data/processed/cluster_6
```

Argumentos principales:
- `--processed-root`
- `--output-dir`
- `--cluster-id`

Artefactos que genera:
- `<output>/algorithms/returns.parquet`
- `<output>/benchmark/weights.parquet`
- `<output>/benchmark/daily_returns.csv`
- `<output>/meta/selected_algos.csv`
- `<output>/meta/cluster_summary.csv`
- `<output>/meta/manifest.json`

Qué consigue al final:
- produce un universo autocontenido que puede pasarse directamente a `Phase 5` mediante `--input-dir`.

### `sweep_phase5_cluster_topk.py`

Qué hace:
- lanza varios runs de `Phase 5` cambiando `top-k` de clusters de `Phase 2`,
- opcionalmente ejecuta `Phase 6` después de cada entrenamiento,
- sirve para comprobar si usar más clusters mejora o empeora el resultado.

Cómo se usa:
```powershell
python scripts/sweep_phase5_cluster_topk.py --agent ppo --phase2-analysis-dir data/processed/analysis --k-values 1 2 3 4 --reward-type pure_returns --no-hybrid --gpu-env --max-resources --evaluate --include-baselines
python scripts/sweep_phase5_cluster_topk.py --agent sac --phase2-analysis-dir data/processed/analysis --k-values 1 2 3 --dry-run
```

Argumentos principales:
- `--agent`
- `--timesteps`
- `--phase2-analysis-dir`
- `--phase2-cluster-source`
- `--phase2-cluster-score-mode`
- `--k-values`
- `--phase2-cluster-min-size`
- `--phase2-cluster-min-return`
- `--phase2-cluster-max-vol`
- `--reward-type`
- `--rebalance-freq`
- `--sample`
- `--run-prefix`
- `--gpu-env`
- `--max-resources`
- `--no-hybrid`
- `--no-encoder`
- `--phase2-cluster-full-history`
- `--evaluate`
- `--include-baselines`
- `--dry-run`

Artefactos que genera:
- no genera un artefacto propio central,
- dispara múltiples runs en `outputs/rl_training/<run_id>` y opcionalmente en `outputs/evaluation/<run_id>`.

Qué consigue al final:
- te deja comparar sistemáticamente si `top-k = 1, 2, 3, ...` mejora el entrenamiento RL.

### `summarize_phase6_runs.py`

Qué hace:
- lee varios runs de `Phase 6`,
- extrae métricas clave,
- construye una tabla de comparación rápida.

Cómo se usa:
```powershell
python scripts/summarize_phase6_runs.py --prefix ppo_temporal_return_only
python scripts/summarize_phase6_runs.py --run-id run_a run_b run_c --output-csv outputs/evaluation/summary.csv
```

Argumentos principales:
- `--run-id`
- `--prefix`
- `--output-csv`

Artefactos que genera:
- salida en consola,
- opcionalmente un CSV resumen.

Qué consigue al final:
- evita abrir `phase6_results.json` uno a uno para comparar barridos.

### `migrate_data.py`

Qué hace:
- migra ficheros desde una estructura legacy a la estructura organizada del proyecto.

Cómo se usa:
```powershell
python scripts/migrate_data.py
python scripts/migrate_data.py --execute
python scripts/migrate_data.py --execute --cleanup
```

Argumentos principales:
- `--execute`: ejecuta la migración real.
- `--cleanup`: limpia directorios vacíos al final.

Artefactos que genera:
- no genera artefactos analíticos; mueve o reorganiza datos.

Qué consigue al final:
- deja el árbol de datos en la convención actual del proyecto.

### `benchmark_optimizations.py`

Qué hace:
- compara funciones optimizadas vs implementaciones de referencia,
- mide rolling calculations, métricas financieras, correlación, costes y memoria.

Cómo se usa:
```powershell
python scripts/benchmark_optimizations.py
```

Notas:
- no expone CLI útil más allá de ejecución directa,
- depende de imports internos del proyecto y de ejecutarse desde la raíz.

Artefactos que genera:
- salida en consola con tiempos y speedups.

Qué consigue al final:
- sirve para validar que las optimizaciones de bajo nivel merecen la pena.

### `base_runner.py`

Qué hace:
- define la clase `PhaseRunner`,
- centraliza logging, monitorización, output dirs, parseo común y persistencia de resultados.

Cuándo tocarlo:
- cuando quieras añadir una fase nueva o cambiar el comportamiento común de todas las fases.

Qué consigue al final:
- evita duplicación y da consistencia operativa a todos los scripts de fase.

## Recomendaciones de uso

- Si vas a fijar un análisis concreto de `Phase 2`, usa un run nuevo que ya contenga `analysis_snapshot`.
- Si vas a entrenar RL sobre un subconjunto del universo, deja rastro de:
  - `phase2_analysis_dir`
  - `phase2_cluster_source`
  - `phase2_cluster_score_mode`
  - `top-k`
  - filtros de tamaño, retorno y volatilidad
- Para runs largos:
  - `Phase 5`: usa `--gpu-env --max-resources`
  - `Phase 7A`: ACO ya usa GPU salvo `--cpu-only`
- Para reproducibilidad:
  - fija `--run-id`
  - fija `--seed`
  - fija `--phase2-analysis-dir` cuando dependas de clusters de `Phase 2`

## Recetas completas

### 1. Pipeline estándar de punta a punta

Úsalo cuando quieras regenerar datos, análisis, baselines, entrenar PPO y evaluar.

```powershell
python scripts/run_phase1.py
python scripts/run_phase2.py
python scripts/run_phase3.py --full
python scripts/run_phase4.py --full
python scripts/run_phase5.py --agent ppo --timesteps 1M --gpu-env --max-resources --run-id ppo_standard
python scripts/run_phase6.py --run-id ppo_standard --include-baselines
```

Resultado:
- dejas un baseline completo del proyecto,
- un run PPO reproducible,
- y una comparación walk-forward contra baselines.

### 2. PPO con clusters temporales de `Phase 2`

Úsalo cuando quieras que `Phase 5` solo vea los mejores clusters temporales detectados por `Phase 2`.

Primero genera el análisis:
```powershell
python scripts/run_phase2.py --clustering-method hdbscan --n-clusters 7
```

Luego entrena con ese análisis concreto:
```powershell
python scripts/run_phase5.py --agent ppo --timesteps 1M --gpu-env --max-resources --phase2-cluster-filter --phase2-analysis-dir data/processed/analysis --phase2-cluster-source temporal_cumulative --phase2-cluster-score-mode return_low_vol --phase2-cluster-top-k 2 --phase2-cluster-min-size 20 --phase2-cluster-min-return 0.01 --phase2-cluster-max-vol 0.12 --run-id ppo_temporal_clusters
python scripts/run_phase6.py --run-id ppo_temporal_clusters --include-baselines
```

Resultado:
- el agente se entrena solo sobre los clusters temporales mejor puntuados,
- y `Phase 6` reutiliza automáticamente el mismo subconjunto.

### 3. RL agresivo a retorno puro

Úsalo cuando quieras maximizar retorno puro y eliminar híbrido y calibración de riesgo del reward.

```powershell
python scripts/run_phase5.py --agent ppo --timesteps 1M --gpu-env --max-resources --reward-type pure_returns --no-hybrid --phase2-cluster-filter --phase2-analysis-dir data/processed/analysis --phase2-cluster-source temporal_cumulative --phase2-cluster-score-mode return_low_vol --phase2-cluster-top-k 2 --phase2-cluster-min-size 20 --phase2-cluster-min-return 0.01 --phase2-cluster-max-vol 0.12 --run-id ppo_return_only
python scripts/run_phase6.py --run-id ppo_return_only --include-baselines
```

Variantes directas:

```powershell
python scripts/run_phase5.py --agent sac --timesteps 1M --gpu-env --max-resources --reward-type pure_returns --no-hybrid --phase2-cluster-filter --phase2-analysis-dir data/processed/analysis --phase2-cluster-source temporal_cumulative --phase2-cluster-score-mode return_low_vol --phase2-cluster-top-k 2 --phase2-cluster-min-size 20 --phase2-cluster-min-return 0.01 --phase2-cluster-max-vol 0.12 --run-id sac_return_only
```

```powershell
python scripts/run_phase5.py --agent td3 --timesteps 1M --gpu-env --max-resources --reward-type pure_returns --no-hybrid --phase2-cluster-filter --phase2-analysis-dir data/processed/analysis --phase2-cluster-source temporal_cumulative --phase2-cluster-score-mode return_low_vol --phase2-cluster-top-k 2 --phase2-cluster-min-size 20 --phase2-cluster-min-return 0.01 --phase2-cluster-max-vol 0.12 --run-id td3_return_only
```

Resultado:
- entrenas al agente para retorno puro,
- restringido al universo de clusters temporales mejor puntuados.

### 4. Universo filtrado en dataset alternativo

Úsalo cuando prefieras construir un dataset procesado nuevo en vez de filtrar dentro de `Phase 5`.

```powershell
python scripts/build_temporal_cluster_dataset.py --output-dir data/processed/cluster_highret_lowvol
python scripts/run_phase5.py --agent ppo --timesteps 1M --gpu-env --max-resources --input-dir data/processed/cluster_highret_lowvol --reward-type pure_returns --no-hybrid --run-id ppo_cluster_dataset
python scripts/run_phase6.py --run-id ppo_cluster_dataset --input-dir data/processed/cluster_highret_lowvol --include-baselines
```

Resultado:
- trabajas con un dataset autocontenido,
- útil si quieres aislar un universo y reutilizarlo en varios experimentos.

### 5. Meta-allocator PSO

Úsalo para optimización directa de cartera con swarm search.

```powershell
python scripts/run_phase7.py --top-k 24 --lookback-window 63 --min-history 42 --particles 96 --iterations 70 --selection-factor rolling_sharpe_21d --analysis-dir data/processed/analysis
```

Versión rápida:
```powershell
python scripts/run_phase7.py --sample 32 --top-k 16 --particles 32 --iterations 10 --selection-factor rolling_sharpe_21d --analysis-dir data/processed/analysis
```

Resultado:
- obtienes una cartera optimizada por PSO,
- con resumen de pesos, retornos y comparación.

### 5B. PSO walk-forward con `k=2` mejores clusters de `Phase 2`

Úsalo para medir PSO contra benchmark sobre datos no vistos, restringiendo el universo a los 2 mejores clusters temporales de `Phase 2`.

```powershell
python scripts/run_phase7_walk_forward.py --analysis-dir data/processed/analysis --phase2-cluster-filter --phase2-cluster-source temporal_cumulative --phase2-cluster-score-mode return_low_vol --phase2-cluster-top-k 2 --phase2-cluster-min-size 20 --phase2-cluster-min-return 0.01 --phase2-cluster-max-vol 0.12 --selection-factor rolling_sharpe_21d --lookback-window 63 --min-history 42 --particles 96 --iterations 70 --train-window 252 --validation-window 63 --test-window 63 --step-size 63 --max-folds 5
```

Resultado:
- obtienes métricas fold a fold,
- retornos stitched de test,
- y comparación contra benchmark sobre datos no vistos.

### 6. Meta-allocator ACO

Úsalo para optimización discreta con colonia de hormigas.

```powershell
python scripts/run_phase7_aco.py --top-k 24 --lookback-window 63 --min-history 42 --ants 96 --iterations 60 --selection-factor rolling_sharpe_21d --analysis-dir data/processed/analysis
```

Forzar CPU:
```powershell
python scripts/run_phase7_aco.py --cpu-only --top-k 24 --lookback-window 63 --min-history 42 --ants 96 --iterations 60 --selection-factor rolling_sharpe_21d --analysis-dir data/processed/analysis
```

Resultado:
- obtienes una cartera optimizada por ACO,
- con soporte GPU si CUDA está disponible.

### 6B. ACO walk-forward con `k=2` mejores clusters de `Phase 2`

Úsalo para medir ACO contra benchmark sobre datos no vistos, restringiendo el universo a los 2 mejores clusters temporales de `Phase 2`.

```powershell
python scripts/run_phase7_aco_walk_forward.py --analysis-dir data/processed/analysis --phase2-cluster-filter --phase2-cluster-source temporal_cumulative --phase2-cluster-score-mode return_low_vol --phase2-cluster-top-k 2 --phase2-cluster-min-size 20 --phase2-cluster-min-return 0.01 --phase2-cluster-max-vol 0.12 --train-window 252 --validation-window 63 --test-window 63 --step-size 63 --max-folds 5
```

Resultado:
- obtienes evaluación walk-forward de ACO,
- con comparación contra benchmark en periodos no vistos.

### 7. Tuning PSO antes del run final

Útil para elegir una configuración PSO con validación temporal antes de lanzar un run grande.

```powershell
python scripts/tune_phase7.py --max-trials 8 --analysis-dir data/processed/analysis
```

Más estricto:
```powershell
python scripts/tune_phase7.py --max-trials 12 --selection-objective blended_sharpe --max-validation-drawdown 0.15 --max-test-drawdown 0.20 --analysis-dir data/processed/analysis
```

Resultado:
- reduces el espacio de búsqueda a una configuración PSO mejor justificada.

### 8. Tuning ACO + walk-forward final

Úsalo cuando quieras seleccionar factores/configuración ACO y luego validarlos en walk-forward.

```powershell
python scripts/tune_phase7_aco.py --quick --analysis-dir data/processed/analysis
python scripts/run_phase7_aco_walk_forward.py --analysis-dir data/processed/analysis
```

Versión más completa:
```powershell
python scripts/tune_phase7_aco.py --max-trials 20 --selection-scheme walk_forward --analysis-dir data/processed/analysis
python scripts/run_phase7_aco_walk_forward.py --config-path outputs/swarm_aco/<run_id>/best_config.json --analysis-dir data/processed/analysis --max-folds 5 --expanding
```

Resultado:
- eliges la mejor configuración ACO,
- y la validas con una estimación temporal más robusta.
