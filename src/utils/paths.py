"""
Centralized path management for the project.

This module defines all data and output paths to ensure consistent
organization across all scripts and modules.

Usage:
    from src.utils.paths import DataPaths, OutputPaths

    # Get processed data paths
    returns_path = DataPaths.algorithms.returns
    weights_path = DataPaths.benchmark.weights

    # Get output paths
    trials_path = OutputPaths.baselines.trials
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os


# =============================================================================
# Project Root
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for CLAUDE.md or pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "CLAUDE.md").exists() or (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we're in src/utils/
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()


# =============================================================================
# Data Paths
# =============================================================================

@dataclass(frozen=True)
class RawDataPaths:
    """Paths to raw data directories."""
    root: Path = PROJECT_ROOT / "data" / "raw"

    @property
    def algorithms(self) -> Path:
        return self.root / "algorithms"

    @property
    def benchmark(self) -> Path:
        return self.root / "benchmark"

    @property
    def forex(self) -> Path:
        return self.root / "forex"

    @property
    def indices(self) -> Path:
        return self.root / "indices"

    @property
    def commodities(self) -> Path:
        return self.root / "commodities"

    @property
    def futures(self) -> Path:
        return self.root / "futures"

    @property
    def sharadar(self) -> Path:
        return self.root / "sharadar"


@dataclass(frozen=True)
class AlgorithmDataPaths:
    """Paths to processed algorithm data (Phase 1 outputs)."""
    root: Path = PROJECT_ROOT / "data" / "processed" / "algorithms"

    @property
    def returns(self) -> Path:
        """Daily returns matrix [dates x algos]."""
        return self.root / "returns.parquet"

    @property
    def features(self) -> Path:
        """Rolling features for all algorithms."""
        return self.root / "features.parquet"

    @property
    def stats(self) -> Path:
        """Per-algorithm statistics."""
        return self.root / "stats.csv"

    @property
    def asset_inference(self) -> Path:
        """Predicted underlying assets."""
        return self.root / "asset_inference.csv"


@dataclass(frozen=True)
class BenchmarkDataPaths:
    """Paths to processed benchmark data (Phase 1 outputs)."""
    root: Path = PROJECT_ROOT / "data" / "processed" / "benchmark"

    @property
    def weights(self) -> Path:
        """Daily benchmark weights [dates x products]."""
        return self.root / "weights.parquet"

    @property
    def positions(self) -> Path:
        """Daily benchmark positions."""
        return self.root / "positions.parquet"

    @property
    def daily_returns(self) -> Path:
        """Reconstructed benchmark daily returns."""
        return self.root / "daily_returns.csv"

    @property
    def turnover(self) -> Path:
        """Daily turnover metrics."""
        return self.root / "turnover.csv"

    @property
    def concentration(self) -> Path:
        """Daily HHI concentration."""
        return self.root / "concentration.csv"

    @property
    def algo_equity(self) -> Path:
        """Equity curves for benchmark products only."""
        return self.root / "algo_equity.parquet"

    @property
    def algo_features(self) -> Path:
        """Features for benchmark products only."""
        return self.root / "algo_features.parquet"


@dataclass(frozen=True)
class AnalysisDataPaths:
    """Paths to analysis data (Phase 2 outputs)."""
    root: Path = PROJECT_ROOT / "data" / "processed" / "analysis"

    # Profiles
    @property
    def profiles_summary(self) -> Path:
        return self.root / "profiles" / "summary.csv"

    @property
    def profiles_full(self) -> Path:
        return self.root / "profiles" / "full.json"

    # Clustering - Temporal
    @property
    def temporal_cluster_history(self) -> Path:
        return self.root / "clustering" / "temporal" / "history.parquet"

    @property
    def temporal_cluster_comparison(self) -> Path:
        return self.root / "clustering" / "temporal" / "method_comparison.csv"

    @property
    def temporal_cluster_report(self) -> Path:
        return self.root / "clustering" / "temporal" / "REPORT.md"

    # Clustering - Behavioral
    @property
    def behavioral_features(self) -> Path:
        return self.root / "clustering" / "behavioral" / "features.csv"

    @property
    def family_labels(self) -> Path:
        return self.root / "clustering" / "behavioral" / "family_labels.csv"

    # Clustering - Correlation
    @property
    def correlation_matrix(self) -> Path:
        return self.root / "clustering" / "correlation" / "matrix.parquet"

    @property
    def correlation_clusters(self) -> Path:
        return self.root / "clustering" / "correlation" / "clusters.csv"

    # Regimes
    @property
    def regime_labels(self) -> Path:
        return self.root / "regimes" / "labels.csv"

    @property
    def regime_probabilities(self) -> Path:
        return self.root / "regimes" / "probabilities.parquet"

    @property
    def regime_report(self) -> Path:
        return self.root / "regimes" / "REPORT.md"

    # Benchmark Profile
    @property
    def benchmark_metrics(self) -> Path:
        return self.root / "benchmark_profile" / "metrics.json"

    @property
    def benchmark_report(self) -> Path:
        return self.root / "benchmark_profile" / "REPORT.txt"


@dataclass(frozen=True)
class FeatureDataPaths:
    """Paths to derived features."""
    root: Path = PROJECT_ROOT / "data" / "processed" / "features"

    @property
    def cross_sectional(self) -> Path:
        return self.root / "cross_sectional.parquet"

    @property
    def regime(self) -> Path:
        return self.root / "regime.parquet"


@dataclass(frozen=True)
class ReportDataPaths:
    """Paths to phase reports in data folder."""
    root: Path = PROJECT_ROOT / "data" / "processed" / "reports"

    @property
    def phase1_summary(self) -> Path:
        return self.root / "phase1" / "SUMMARY.md"

    @property
    def phase1_results(self) -> Path:
        return self.root / "phase1" / "results.json"

    @property
    def phase1_metrics(self) -> Path:
        return self.root / "phase1" / "metrics.json"

    @property
    def phase2_results(self) -> Path:
        return self.root / "phase2" / "results.json"

    @property
    def phase2_metrics(self) -> Path:
        return self.root / "phase2" / "metrics.json"


@dataclass(frozen=True)
class ProcessedDataPaths:
    """All processed data paths."""
    root: Path = PROJECT_ROOT / "data" / "processed"

    algorithms: AlgorithmDataPaths = None
    benchmark: BenchmarkDataPaths = None
    analysis: AnalysisDataPaths = None
    features: FeatureDataPaths = None

    def __post_init__(self):
        object.__setattr__(self, 'algorithms', AlgorithmDataPaths())
        object.__setattr__(self, 'benchmark', BenchmarkDataPaths())
        object.__setattr__(self, 'analysis', AnalysisDataPaths())
        object.__setattr__(self, 'features', FeatureDataPaths())


@dataclass(frozen=True)
class DataPaths:
    """All data paths (raw + processed)."""
    root: Path = PROJECT_ROOT / "data"
    raw: RawDataPaths = None
    processed: ProcessedDataPaths = None

    def __post_init__(self):
        object.__setattr__(self, 'raw', RawDataPaths())
        object.__setattr__(self, 'processed', ProcessedDataPaths())

    # Shortcuts for backward compatibility
    @property
    def algorithms(self) -> AlgorithmDataPaths:
        return self.processed.algorithms

    @property
    def benchmark(self) -> BenchmarkDataPaths:
        return self.processed.benchmark


# =============================================================================
# Output Paths
# =============================================================================

@dataclass(frozen=True)
class BaselineOutputPaths:
    """Paths for Phase 3 baseline outputs."""
    root: Path = PROJECT_ROOT / "outputs" / "baselines"

    @property
    def trials_csv(self) -> Path:
        return self.root / "trials" / "results.csv"

    @property
    def trials_dir(self) -> Path:
        return self.root / "trials"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def summary(self) -> Path:
        return self.root / "SUMMARY.md"


@dataclass(frozen=True)
class RLTrainingOutputPaths:
    """Paths for Phase 5 RL training outputs."""
    root: Path = PROJECT_ROOT / "outputs" / "rl_training"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"


@dataclass(frozen=True)
class EvaluationOutputPaths:
    """Paths for Phase 6 evaluation outputs."""
    root: Path = PROJECT_ROOT / "outputs" / "evaluation"

    @property
    def walk_forward_dir(self) -> Path:
        return self.root / "walk_forward"

    @property
    def comparison_dir(self) -> Path:
        return self.root / "comparison"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"


@dataclass(frozen=True)
class DataPipelineOutputPaths:
    """Paths for Phase 1 data pipeline reports."""
    root: Path = PROJECT_ROOT / "outputs" / "data_pipeline"


@dataclass(frozen=True)
class AnalysisOutputPaths:
    """Paths for Phase 2 analysis reports."""
    root: Path = PROJECT_ROOT / "outputs" / "analysis"


@dataclass(frozen=True)
class EnvironmentOutputPaths:
    """Paths for Phase 4 environment validation outputs."""
    root: Path = PROJECT_ROOT / "outputs" / "environment"


@dataclass(frozen=True)
class SwarmPSOOutputPaths:
    """Paths for Phase 7 PSO swarm meta-allocator outputs."""
    root: Path = PROJECT_ROOT / "outputs" / "swarm_pso"

    def run_dir(self, run_id: str) -> Path:
        """Get the directory for a specific run."""
        return self.root / run_id

    @property
    def latest_run_file(self) -> Path:
        return self.root / "latest_run.txt"


@dataclass(frozen=True)
class SwarmACOOutputPaths:
    """Paths for Phase 7A ACO swarm meta-allocator outputs."""
    root: Path = PROJECT_ROOT / "outputs" / "swarm_aco"

    def run_dir(self, run_id: str) -> Path:
        """Get the directory for a specific run."""
        return self.root / run_id

    @property
    def latest_run_file(self) -> Path:
        return self.root / "latest_run.txt"


@dataclass(frozen=True)
class OutputPaths:
    """All output paths."""
    root: Path = PROJECT_ROOT / "outputs"
    data_pipeline: DataPipelineOutputPaths = None
    analysis: AnalysisOutputPaths = None
    baselines: BaselineOutputPaths = None
    environment: EnvironmentOutputPaths = None
    rl_training: RLTrainingOutputPaths = None
    evaluation: EvaluationOutputPaths = None
    swarm_pso: SwarmPSOOutputPaths = None
    swarm_aco: SwarmACOOutputPaths = None

    def __post_init__(self):
        object.__setattr__(self, 'data_pipeline', DataPipelineOutputPaths())
        object.__setattr__(self, 'analysis', AnalysisOutputPaths())
        object.__setattr__(self, 'baselines', BaselineOutputPaths())
        object.__setattr__(self, 'environment', EnvironmentOutputPaths())
        object.__setattr__(self, 'rl_training', RLTrainingOutputPaths())
        object.__setattr__(self, 'evaluation', EvaluationOutputPaths())
        object.__setattr__(self, 'swarm_pso', SwarmPSOOutputPaths())
        object.__setattr__(self, 'swarm_aco', SwarmACOOutputPaths())

    @property
    def swarm(self) -> SwarmPSOOutputPaths:
        """Backward-compatibility alias for swarm_pso."""
        return self.swarm_pso

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    """Ensure parent directory of file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_data_paths() -> DataPaths:
    """Get DataPaths instance."""
    return DataPaths()


def get_output_paths() -> OutputPaths:
    """Get OutputPaths instance."""
    return OutputPaths()


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Old path constants (deprecated, use DataPaths/OutputPaths instead)
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# =============================================================================
# Migration Helper
# =============================================================================

def get_legacy_to_new_mapping() -> dict:
    """
    Returns mapping of old file paths to new paths for migration.

    Use this to update existing code or migrate files.
    """
    dp = DataPaths()
    op = OutputPaths()
    return {
        # Algorithm data
        "algo_returns.parquet": dp.algorithms.returns,
        "algo_features.parquet": dp.algorithms.features,
        "algo_stats.csv": dp.algorithms.stats,
        "asset_inference.csv": dp.algorithms.asset_inference,

        # Benchmark data
        "benchmark_weights.parquet": dp.benchmark.weights,
        "benchmark_positions.parquet": dp.benchmark.positions,
        "benchmark_daily_returns.csv": dp.benchmark.daily_returns,
        "benchmark_turnover.csv": dp.benchmark.turnover,
        "benchmark_concentration.csv": dp.benchmark.concentration,
        "benchmark_algo_equity.parquet": dp.benchmark.algo_equity,
        "benchmark_algo_features.parquet": dp.benchmark.algo_features,

        # Reports (now in outputs/)
        "PHASE1_SUMMARY.md": op.data_pipeline.root / "PHASE1_SUMMARY.md",
        "phase1_results.json": op.data_pipeline.root / "phase1_results.json",

        # Analysis
        "regime_features.parquet": dp.features.regime,
        "cross_features.parquet": dp.features.cross_sectional,
    }


# =============================================================================
# Module-level instances (for convenience)
# =============================================================================

# Singleton instances
_data_paths: Optional[DataPaths] = None
_output_paths: Optional[OutputPaths] = None


def data_paths() -> DataPaths:
    """Get or create DataPaths singleton."""
    global _data_paths
    if _data_paths is None:
        _data_paths = DataPaths()
    return _data_paths


def output_paths() -> OutputPaths:
    """Get or create OutputPaths singleton."""
    global _output_paths
    if _output_paths is None:
        _output_paths = OutputPaths()
    return _output_paths
