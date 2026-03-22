# Scripts

This folder contains executable scripts for running the RL Meta-Allocator pipeline.

## Phase Scripts

Each phase has a single consolidated script with built-in performance monitoring:

| Script | Description | Output Directory |
|--------|-------------|------------------|
| `run_phase1.py` | Data Loading & Feature Engineering | `data/processed/` |
| `run_phase2.py` | Analysis & Reverse Engineering | `data/processed/analysis/` |
| `run_phase3.py` | Baseline Backtesting | `outputs/baselines/` |

### Usage

```bash
# Phase 1: Load and process data
python scripts/run_phase1.py
python scripts/run_phase1.py --sample 100 --skip-features  # Quick test
python scripts/run_phase1.py --benchmark-only              # Only benchmark products

# Phase 2: Analysis and regime inference
python scripts/run_phase2.py
python scripts/run_phase2.py --sample 100 --skip-inference  # Quick test
python scripts/run_phase2.py --n-regimes 6 --n-families 10  # Custom params

# Phase 3: Baseline backtesting
python scripts/run_phase3.py
python scripts/run_phase3.py --quick                        # Few configs
python scripts/run_phase3.py --full                         # All + walk-forward
python scripts/run_phase3.py --baseline risk_parity         # Single baseline
```

### Common Options

All phase scripts support:

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Enable verbose logging |
| `--dry-run` | Show what would be done without executing |
| `--output-dir`, `-o` | Override output directory |
| `--log-file` | Write logs to file |
| `--no-report` | Skip generating performance report |

### Performance Monitoring

Each script automatically tracks and reports:

- **Timing**: Wall time, CPU time per step
- **Memory**: Start, end, peak, and delta memory usage
- **GPU**: Memory allocation (if CUDA available)
- **Checkpoints**: Per-step timing and memory changes

Reports are saved as:
- `phase{N}_metrics.json` - Detailed performance metrics
- `phase{N}_results.json` - Execution results
- `PHASE{N}_SUMMARY.md` - Human-readable summary

## Utility Scripts

| Script | Description |
|--------|-------------|
| `migrate_data.py` | Migrate data files to new organized structure |
| `benchmark_optimizations.py` | Benchmark numba and optimization performance |
| `run_temporal_clustering.py` | Run standalone temporal clustering (also included in Phase 2) |

### migrate_data.py

```bash
python scripts/migrate_data.py              # Dry-run
python scripts/migrate_data.py --execute    # Actually move files
python scripts/migrate_data.py --cleanup    # Remove empty directories
```

### benchmark_optimizations.py

```bash
python scripts/benchmark_optimizations.py   # Run all benchmarks
```

## Base Runner

The `base_runner.py` module provides the `PhaseRunner` class that all phase scripts inherit from. It handles:

- Argument parsing with common options
- Logging configuration
- Performance monitoring integration
- Report generation
- Output directory management

To create a new phase script:

```python
from scripts.base_runner import PhaseRunner

class Phase4Runner(PhaseRunner):
    phase_name = "Phase 4: RL Environment"
    phase_number = 4

    def add_arguments(self, parser):
        parser.add_argument('--custom', type=int)

    def run(self, args):
        with self.step("Step 1"):
            # Do work
            pass
        return {'status': 'completed'}

if __name__ == '__main__':
    runner = Phase4Runner()
    runner.execute()
```

## Output Organization

### Data Outputs (`data/processed/`)

```
data/processed/
├── algorithms/          # Algorithm data (Phase 1)
│   ├── returns.parquet
│   ├── features.parquet
│   ├── stats.csv
│   └── asset_inference.csv
├── benchmark/           # Benchmark data (Phase 1)
│   ├── weights.parquet
│   ├── positions.parquet
│   └── daily_returns.csv
├── analysis/            # Analysis outputs (Phase 2)
│   ├── profiles/
│   ├── clustering/
│   └── regimes/
└── reports/             # Phase reports
```

### Experiment Outputs (`outputs/`)

```
outputs/
├── baselines/           # Phase 3 outputs
│   ├── trials/
│   └── figures/
├── rl_training/         # Phase 5 outputs
│   ├── checkpoints/
│   └── logs/
└── evaluation/          # Phase 6 outputs
    ├── walk_forward/
    └── comparison/
```
