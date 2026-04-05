# Production RL Meta-Allocator

Sistema de producción para entrenar un agente RL de asignación de cartera y generar recomendaciones semanales de pesos.

---

## Tabla de Contenidos

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Paso 1: Preparar Datos](#paso-1-preparar-datos)
5. [Paso 2: Entrenar Agente](#paso-2-entrenar-agente)
6. [Paso 3: Generar Recomendaciones](#paso-3-generar-recomendaciones)
7. [Clasificación por Familias (Sortino)](#clasificación-por-familias-sortino)
8. [Configuración Avanzada](#configuración-avanzada)
9. [Troubleshooting](#troubleshooting)

---

## Requisitos

### Hardware Recomendado

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| CPU | 8 cores | 32+ cores |
| RAM | 32 GB | 64+ GB |
| GPU | Opcional | 1x RTX 3060+ |
| Disco | 50 GB SSD | 100 GB SSD |

> **Nota**: El entrenamiento RL es CPU-bound. Más cores = más entornos paralelos = entrenamiento más rápido. La GPU se usa poco.

### Software

- Python 3.10+
- CUDA 11.8+ (opcional, para GPU)

---

## Instalación

```bash
# 1. Clonar/acceder al proyecto
cd /path/to/athenai_competition

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate   # Windows

# 3. Instalar dependencias de producción
pip install -r production/requirements.txt

# 4. Verificar instalación
python -c "import duckdb, torch, stable_baselines3; print('OK')"
```

---

## Estructura del Proyecto

```
production/
├── config/
│   └── production.yaml       # Configuración (costes, restricciones, training)
├── data/
│   ├── raw/                  # CSVs OHLCV (alternativa a DuckDB)
│   └── processed/            # Datos procesados (auto-generados)
│       ├── returns.parquet   # Matriz de retornos [fechas × activos]
│       ├── features.parquet  # Features + familias Sortino
│       ├── assets.txt        # Lista de activos
│       └── asset_stats.csv   # Estadísticas por activo
├── models/
│   ├── {run_id}/             # Directorio por entrenamiento
│   │   ├── checkpoints/ppo/
│   │   │   ├── best_model.zip
│   │   │   ├── final_model.zip
│   │   │   ├── vecnormalize.pkl
│   │   │   └── universe_encoder.pkl
│   │   └── run_info.json
│   └── latest -> {run_id}    # Symlink al último modelo
├── outputs/
│   └── recommendations/      # Histórico de recomendaciones
│       └── recommendation_YYYY-MM-DD.json
├── scripts/
│   ├── prepare_data.py       # Paso 1: Preparar datos
│   ├── train.py              # Paso 2: Entrenar agente
│   └── recommend.py          # Paso 3: Generar recomendaciones
├── src/
│   ├── data_loader.py        # Carga desde CSV
│   ├── duckdb_loader.py      # Carga desde DuckDB
│   ├── feature_builder.py    # Construcción de features + familias
│   └── inference.py          # Motor de inferencia
└── requirements.txt
```

---

## Paso 1: Preparar Datos

### Desde DuckDB (Recomendado)

```bash
# Básico
python production/scripts/prepare_data.py --duckdb /path/to/prices.duckdb

# Con filtros
python production/scripts/prepare_data.py \
    --duckdb /path/to/prices.duckdb \
    --start-date 2018-01-01 \
    --end-date 2024-12-31 \
    --min-history 504
```

**Esquema esperado en DuckDB:**

| Columna | Tipo | Requerido | Descripción |
|---------|------|-----------|-------------|
| symbol | VARCHAR | ✓ | Identificador del activo |
| date | DATE | ✓ | Fecha |
| close | DOUBLE | ✓ | Precio de cierre |
| open, high, low | DOUBLE | | Precios OHLC |
| volume | DOUBLE | | Volumen |
| sortino_5d/21d/63d | FLOAT | | Sortino pre-calculado |
| sharpe_5d/21d/63d | FLOAT | | Sharpe pre-calculado |
| return_rolling_5d/21d/63d | DOUBLE | | Retornos rolling |
| volatility_rolling_5d/21d/63d | DOUBLE | | Volatilidad rolling |

### Desde CSVs (Alternativa)

```bash
# Colocar CSVs en production/data/raw/
# Un archivo por activo: AAPL.csv, MSFT.csv, etc.

python production/scripts/prepare_data.py --raw-dir production/data/raw
```

**Formato CSV esperado:**

```csv
date,open,high,low,close,volume
2020-01-02,100.0,102.5,99.5,101.2,1000000
2020-01-03,101.2,103.0,100.8,102.5,1200000
```

### Output

```
production/data/processed/
├── returns.parquet      # Matriz [fechas × activos]
├── features.parquet     # Features con familias Sortino
├── assets.txt           # Lista de símbolos
├── asset_stats.csv      # Estadísticas
└── data_info.json       # Metadatos (solo DuckDB)
```

### Verificar Datos

```python
import pandas as pd

returns = pd.read_parquet("production/data/processed/returns.parquet")
print(f"Activos: {returns.shape[1]}")
print(f"Fechas: {returns.shape[0]}")
print(f"Rango: {returns.index.min()} - {returns.index.max()}")
```

---

## Paso 2: Entrenar Agente

### Entrenamiento Básico

```bash
python production/scripts/train.py
```

### Entrenamiento Optimizado (Recomendado)

```bash
# Más entornos paralelos = entrenamiento más rápido
# Usar ~75% de los cores disponibles

python production/scripts/train.py \
    --agent ppo \
    --timesteps 500000 \
    --n-envs 24
```

### Opciones de Entrenamiento

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--agent` | ppo | Tipo de agente: `ppo`, `sac`, `td3` |
| `--timesteps` | 100000 | Pasos de entrenamiento |
| `--n-envs` | 4 | Entornos paralelos (más = más rápido) |
| `--config` | config/production.yaml | Archivo de configuración |

### Entrenamiento con GPU (Recomendado)

Para aprovechar la GPU con el ambiente vectorizado:

```bash
python production/scripts/train.py \
    --agent ppo \
    --timesteps 500000 \
    --n-envs 24 \
    --gpu-env
```

El flag `--gpu-env` activa el `GPUVecTradingEnv` con `FamilyEncoder`, que:
- Clasifica los ~12K activos en 4 familias basadas en Sortino ratio
- Reduce la dimensionalidad del espacio de observación a 24 (6 features × 4 familias)
- Mantiene el espacio de acción en 4 dimensiones (pesos por familia)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--agent` | ppo | Tipo de agente: `ppo`, `sac`, `td3` |
| `--timesteps` | 100000 | Pasos de entrenamiento |
| `--n-envs` | 4 | Entornos paralelos (más = más rápido) |
| `--gpu-env` | False | Usar GPU-accelerated vectorized environment |
| `--config` | config/production.yaml | Archivo de configuración |

### Tiempo Estimado de Entrenamiento

**Con GPU Environment (`--gpu-env`):**

| Timesteps | n-envs | Tiempo Aprox. | FPS |
|-----------|--------|---------------|-----|
| 100K | 24 | 2-3 min | ~500 |
| 500K | 24 | 15-20 min | ~450 |
| 1M | 24 | 35-45 min | ~450 |

**CPU Environment (SubprocVecEnv):**

| Timesteps | n-envs | Tiempo Aprox. |
|-----------|--------|---------------|
| 100K | 8 | 1-2 horas |
| 100K | 24 | 30-45 min |
| 500K | 24 | 2-4 horas |
| 1M | 32 | 4-8 horas |

### Output del Entrenamiento

```
production/models/{run_id}/
├── checkpoints/ppo/
│   ├── best_model.zip       # Mejor modelo (por validación)
│   ├── final_model.zip      # Modelo final
│   ├── vecnormalize.pkl     # Estadísticas de normalización
│   └── universe_encoder.pkl # Encoder PCA
├── logs/ppo/
│   └── evaluations.npz      # Métricas de evaluación
├── tensorboard/             # Logs de TensorBoard
└── run_info.json            # Metadatos del entrenamiento
```

### Monitorear Entrenamiento (Opcional)

```bash
# En otra terminal
tensorboard --logdir production/models/{run_id}/tensorboard
```

---

## Paso 3: Generar Recomendaciones

### Recomendación Básica

```bash
python production/scripts/recommend.py
```

### Modo Híbrido (Recomendado)

Combina asignación base (Risk Parity) + ajustes del agente RL:

```bash
python production/scripts/recommend.py --hybrid
```

### Todas las Opciones

```bash
python production/scripts/recommend.py \
    --hybrid \
    --base-allocator risk_parity \
    --max-tilt 0.15 \
    --model-dir production/models/latest
```

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--hybrid` | False | Usar modo híbrido (base + RL tilts) |
| `--base-allocator` | risk_parity | Base: `risk_parity` o `equal_weight` |
| `--max-tilt` | 0.15 | Máxima desviación de la base (±15%) |
| `--model-dir` | models/latest | Directorio del modelo |
| `--json` | False | Output en JSON (para integración) |
| `--no-save` | False | No guardar a archivo |

### Output de Ejemplo

```
==================================================
PORTFOLIO RECOMMENDATION - 2024-03-29
==================================================
Generated: 2024-03-29T18:30:00
Agent: ppo_hybrid
--------------------------------------------------
Asset                          Weight        Pct
--------------------------------------------------
AAPL                           0.1523     15.23%  #######
MSFT                           0.1245     12.45%  ######
GOOGL                          0.0987      9.87%  ####
NVDA                           0.0876      8.76%  ####
AMZN                           0.0654      6.54%  ###
...
--------------------------------------------------
Active positions: 12
Max weight: 15.23%
HHI (concentration): 0.0892
==================================================

CHANGES FROM PREVIOUS RECOMMENDATION
--------------------------------------------------
Asset                    Previous    Current     Change
--------------------------------------------------
NVDA                        5.20%      8.76%     +3.56%
AAPL                       18.00%     15.23%     -2.77%
--------------------------------------------------
Total turnover: 12.45%

Saved to: production/outputs/recommendations/recommendation_2024-03-29.json
```

### Integración con Sistemas Externos

```bash
# Output JSON para APIs
python production/scripts/recommend.py --hybrid --json > recommendation.json
```

```python
# Uso programático
from production.src import HybridInferenceEngine
import pandas as pd

returns = pd.read_parquet("production/data/processed/returns.parquet")
engine = HybridInferenceEngine(
    model_dir="production/models/latest",
    agent_type="ppo",
    base_allocator="risk_parity",
    max_tilt=0.15,
)

recommendation = engine.get_recommendation(returns)
print(recommendation["weights"])
```

---

## Clasificación por Familias (Sortino)

Los activos se clasifican automáticamente en 4 familias basadas en el Sortino ratio:

| Familia | Sortino Ratio | Descripción |
|---------|---------------|-------------|
| 0 | > 2.0 | Excelente retorno ajustado por riesgo |
| 1 | 1.0 - 2.0 | Buen retorno ajustado por riesgo |
| 2 | 0.0 - 1.0 | Retorno moderado |
| 3 | < 0.0 | Retorno negativo ajustado por riesgo |

### Ventanas Temporales

- `family_cumulative`: Desde inicio del histórico
- `family_5d`: Últimos 5 días
- `family_21d`: Últimos 21 días (~1 mes)
- `family_63d`: Últimos 63 días (~3 meses)

### Ver Distribución de Familias

```python
from production.src.feature_builder import print_family_summary
import pandas as pd

features = pd.read_parquet("production/data/processed/features.parquet")
print_family_summary(features)
```

Output:
```
============================================================
SORTINO FAMILY DISTRIBUTION - 2024-03-29
============================================================
Total assets: 500

  CUMULATIVE      | excellent (>2): 12.4%  good (1-2): 23.6%  moderate (0-1): 41.2%  poor (<0): 22.8%
  5D              | excellent (>2):  8.2%  good (1-2): 18.4%  moderate (0-1): 38.6%  poor (<0): 34.8%
  21D             | excellent (>2): 10.6%  good (1-2): 21.2%  moderate (0-1): 40.0%  poor (<0): 28.2%
  63D             | excellent (>2): 11.8%  good (1-2): 22.8%  moderate (0-1): 39.4%  poor (<0): 26.0%
============================================================
```

---

## Configuración Avanzada

Editar `production/config/production.yaml`:

```yaml
# Restricciones de cartera
constraints:
  max_weight: 0.40        # Máximo 40% en un solo activo
  min_weight: 0.00        # Sin ventas en corto
  max_turnover: 0.30      # Máximo 30% rotación por rebalanceo
  max_exposure: 1.0       # 100% invertido

# Costes de transacción
costs:
  spread_bps: 5           # 5 puntos básicos de spread
  slippage_bps: 2         # 2 puntos básicos de slippage
  market_impact_coef: 0.1 # Coeficiente de impacto de mercado

# Entrenamiento
training:
  agent: "ppo"
  total_timesteps: 100000
  learning_rate: 0.0001
  net_arch: [256, 256]    # Arquitectura de la red neuronal
  n_envs: 4

  # Modo híbrido
  hybrid_mode: true
  base_allocator: "risk_parity"
  max_tilt: 0.15          # ±15% desviación del base

  # Encoder
  use_encoder: true
  n_pca_components: 20    # Componentes PCA para reducción dimensional

# Reward
reward:
  scale: 100.0
  cost_penalty: 1.0
  turnover_penalty: 0.005
  drawdown_penalty: 0.1
```

---

## Troubleshooting

### "No trained models found"

```bash
# Solución: Entrenar un modelo primero
python production/scripts/train.py --timesteps 100000
```

### "Returns not found" / "Database not found"

```bash
# Solución: Preparar datos primero
python production/scripts/prepare_data.py --duckdb /path/to/data.duckdb
```

### "Insufficient data" / Pocos activos cargados

```bash
# Reducir el mínimo de observaciones requeridas
python production/scripts/prepare_data.py --duckdb data.duckdb --min-history 126
```

### Entrenamiento muy lento

```bash
# Aumentar entornos paralelos (usar ~75% de cores)
python production/scripts/train.py --n-envs 24
```

### Out of Memory durante entrenamiento

```bash
# Reducir entornos paralelos
python production/scripts/train.py --n-envs 8

# O reducir componentes PCA en config
# training.n_pca_components: 10
```

### Weights dimension mismatch

El encoder PCA se entrena para un número específico de activos. Si cambias el universo de activos, re-entrena el modelo:

```bash
python production/scripts/prepare_data.py --duckdb data.duckdb
python production/scripts/train.py --timesteps 100000
```

---

## Flujo de Trabajo Semanal

```
┌─────────────────────────────────────────────────────────────┐
│                    VIERNES / FIN DE SEMANA                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Actualizar datos (si hay nuevos)                        │
│     python scripts/prepare_data.py --duckdb data.duckdb     │
│                                                             │
│  2. Generar recomendación                                   │
│     python scripts/recommend.py --hybrid                    │
│                                                             │
│  3. Revisar output y comparar con semana anterior           │
│                                                             │
│  4. Implementar cambios en cartera real                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Re-entrenamiento Periódico

Se recomienda re-entrenar el modelo:
- Cada 3-6 meses
- Cuando cambie significativamente el universo de activos
- Cuando el rendimiento del modelo degrade

```bash
# Re-entrenar con datos actualizados
python production/scripts/prepare_data.py --duckdb data.duckdb
python production/scripts/train.py --timesteps 500000 --n-envs 24
```

---

## Licencia

Uso interno. No redistribuir.
