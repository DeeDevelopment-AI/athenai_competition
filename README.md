# RL Meta-Allocator for Black-Box Algorithms

A reinforcement learning system that acts as a meta-allocator, assigning capital across black-box trading algorithms to beat a benchmark.

## Project Overview

This project prepares a practical interview case for an AI finance lab. The exercise:

1. Receive transactions from N black-box algorithms (unknown internal logic)
2. Receive a benchmark that invests in these algorithms
3. Design and implement an RL system as meta-allocator to beat the benchmark
4. Analyze benchmark behavior (frequency, duration, capital, risk profile) for fair comparison

## Project Structure

```
athenai/
├── data/
│   ├── raw/                    # Raw input data
│   │   ├── algoritmos/         # 14,761 algorithm OHLC files
│   │   ├── benchmark/          # Benchmark transactions and returns
│   │   ├── commodities/        # Market data (commodities)
│   │   ├── forex/              # Market data (FX pairs)
│   │   ├── indices/            # Market data (SP500, Nasdaq)
│   │   ├── futures/            # CME futures data
│   │   └── sharadar/           # US equities and ETFs
│   └── processed/              # Processed data outputs
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── analysis/               # Analysis and profiling
│   ├── baselines/              # Classical allocation strategies
│   ├── environment/            # RL environment
│   ├── agents/                 # RL agents (PPO, SAC, TD3)
│   └── evaluation/             # Backtesting and metrics
├── scripts/                    # Execution scripts
├── notebooks/                  # Jupyter notebooks
├── configs/                    # Configuration files
└── outputs/                    # Results and reports
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Execution Commands by Phase

### Phase 1: Data Loading and Reconstruction

Load raw data, reconstruct equity curves and returns, compute rolling features.

```bash
# Full Phase 1 (all 14,761 algorithms) - takes ~15 minutes
python scripts/run_phase1.py --data-path data/raw --output data/processed

# Benchmark products only (271 algorithms) - faster, ~2 minutes
python scripts/run_phase1.py --data-path data/raw --output data/processed --benchmark-only

# Quick test with sample
python scripts/run_phase1.py --data-path data/raw --output data/processed --sample 500

# Skip feature engineering (even faster)
python scripts/run_phase1.py --data-path data/raw --output data/processed --benchmark-only --skip-features
```

**Outputs** (`data/processed/`):
| File | Description |
|------|-------------|
| `algo_stats.csv` | Per-algorithm statistics |
| `algo_returns.parquet` | Daily returns matrix |
| `benchmark_algo_equity.parquet` | Equity curves (benchmark products) |
| `benchmark_weights.parquet` | Daily benchmark allocation weights |
| `benchmark_positions.parquet` | Daily position counts |
| `benchmark_turnover.csv` | Daily turnover |
| `benchmark_concentration.csv` | Daily concentration (HHI) |
| `benchmark_daily_returns.csv` | Reconstructed benchmark returns |
| `algo_features.parquet` | Rolling features per algorithm |
| `cross_features.parquet` | Cross-sectional features |
| `regime_features.parquet` | Market regime features |

---

### Phase 2: Analysis and Reverse Engineering

Profile algorithms, reverse-engineer benchmark behavior, detect market regimes.

```bash
# Algorithm pipeline (detailed asset inference)
python notebooks/algo_pipeline.py \
    --algos data/raw/algorithms \
    --benchmarks notebooks/benchmarks \
    --output outputs/results

# Benchmark asset regime analysis
python notebooks/benchmark_asset_regime.py \
    --trades data/raw/benchmark/trades_benchmark.csv \
    --inference outputs/results/asset_inference_all.csv \
    --metrics outputs/results/metrics_all.csv \
    --benchmarks notebooks/benchmarks \
    --output outputs/regime
```

**SP500 Regime Detection** (4-state Investment Clock):
- `expansion` - Growth phase, positive trend, moderate volatility
- `peak` - Late cycle, momentum fading, volatility rising
- `contraction` - Correction/recession, negative trend, high volatility
- `recovery` - Early cycle rebound, improving momentum

---

### Phase 3: Baseline Strategies

Implement classical allocation baselines for comparison.

```bash
# Run all baselines
python scripts/run_baselines.py --data-path data/processed --output outputs/baselines
```

**Baselines implemented** (`src/baselines/`):
| Strategy | File | Description |
|----------|------|-------------|
| Equal Weight | `equal_weight.py` | 1/N allocation |
| Risk Parity | `risk_parity.py` | Inverse volatility weighting |
| Minimum Variance | `min_variance.py` | Quadratic optimization |
| Maximum Sharpe | `max_sharpe.py` | Mean-variance optimization |
| Momentum | `momentum_allocator.py` | Cross-sectional momentum |
| Vol Targeting | `vol_targeting.py` | Scale exposure to target vol |

---

### Phase 4: RL Environment

Build custom trading environment (without Gymnasium).

```bash
# Test environment
python -m pytest tests/test_environment.py -v

# Test cost model
python -m pytest tests/test_cost_model.py -v
```

**Environment components** (`src/environment/`):
- `market_simulator.py` - Event-driven simulation
- `trading_env.py` - RL interface
- `cost_model.py` - Transaction costs
- `constraints.py` - Portfolio constraints
- `reward.py` - Reward functions

---

### Phase 5: RL Training

Train RL agents (PPO, SAC, TD3).

```bash
# Train PPO agent (baseline)
python scripts/train_agent.py --agent ppo --config configs/ppo.yaml --output outputs/models

# Train SAC agent
python scripts/train_agent.py --agent sac --config configs/sac.yaml --output outputs/models

# Train TD3 agent
python scripts/train_agent.py --agent td3 --config configs/td3.yaml --output outputs/models
```

---

### Phase 6: Evaluation

Walk-forward validation and comparison.

```bash
# Full evaluation
python scripts/evaluate.py \
    --models outputs/models \
    --baselines outputs/baselines \
    --output outputs/reports
```

**Outputs** (`outputs/reports/`):
- `comparison_table.csv` - Agent vs benchmark vs baselines
- `equity_curves.png` - Visual comparison
- `regime_analysis.png` - Performance by regime
- `final_report.md` - Comprehensive report

---

## Quick Start

```bash
# 1. Run Phase 1 (benchmark products only for speed)
python scripts/run_phase1.py --benchmark-only

# 2. Check the summary
cat data/processed/PHASE1_SUMMARY.md

# 3. View algorithm statistics
python -c "import pandas as pd; print(pd.read_csv('data/processed/algo_stats.csv').describe())"
```

---

## Data Formats

### Algorithm Data (OHLC)
```csv
datetime,open,high,low,close
2020-06-01 13:00:00+00:00,100.0,101.5,99.5,101.0
```

### Benchmark Trades
```csv
volume,dateOpen,dateClose,total_invested_amount_EOD,equity_EOD,AUM,equity_normalized,productname
8000,2020-06-03 14:00:00+00:00,2020-09-15 16:00:00+00:00,120000,12000,120000,100.5,fpJbh
```

---

## Key Findings (Phase 1)

| Metric | Value |
|--------|-------|
| Algorithm Universe | 14,761 algorithms |
| Selection Rate | 1.8% (271/14,761) |
| Avg Algorithm Sharpe | -0.38 |
| Benchmark Sharpe | ~0.95 |
| Data Coverage | 45.5% |
| Benchmark CAGR | ~5.24% |
| Max Drawdown | < 2% |

---

## Configuration

Edit `configs/default.yaml`:

```yaml
data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  frequency: "daily"

constraints:
  max_weight_per_algo: 0.40
  min_weight_per_algo: 0.00
  max_turnover_per_rebalance: 0.30
  max_total_exposure: 1.0

costs:
  spread_bps: 5.0
  slippage_bps: 2.0
  impact_coefficient: 0.1

reward:
  type: "alpha_penalized"
  cost_penalty: 1.0
  turnover_penalty: 0.1
  drawdown_penalty: 0.5
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# Specific test modules
pytest tests/test_data_pipeline.py -v
pytest tests/test_environment.py -v
pytest tests/test_baselines.py -v
```

---

## Dependencies

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- stable-baselines3, torch
- pyarrow (parquet)
- cvxpy (optimization)
- hmmlearn (regimes)

---

## License

Private - Interview case study
