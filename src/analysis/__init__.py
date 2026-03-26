"""
Módulo de análisis: perfiles de algoritmos, benchmark y regímenes.

Incluye:
- Profilers: AlgoProfiler, BenchmarkProfiler, RegimeDetector
- Clustering: AlgoClusterer con soporte de dos capas
- Features: AlgoFeatureExtractor para extracción de características
- Latent Regime Inference: Inferencia de régimen latente basada en
  comportamiento del benchmark (NO predicción del ciclo macro)
"""

from .algo_profiler import AlgoProfiler
from .benchmark_profiler import BenchmarkProfiler
from .regime_detector import RegimeDetector
from .correlation_analyzer import CorrelationAnalyzer
from .algo_clusterer import (
    AlgoClusterer,
    ClusterMethod,
    ScalerType,
    ClusterResult,
    TwoLayerClusterResult,
    TemporalAlgoClusterer,
    TemporalClusterResult,
    TemporalClusteringOutput,
    name_clusters,
    name_life_clusters,
    name_behavior_clusters,
)
from .pseudo_label_clusterer import (
    PseudoLabelClusterer,
    PseudoLabelStrategy,
    PseudoLabelClusteringResult,
)
from .algo_features import (
    AlgoFeatureExtractor,
    AlgoFeatureConfig,
    ACTIVITY_FEATURES,
    PERFORMANCE_FEATURES,
    TRANSITION_FEATURES,
    BENCHMARK_FEATURES,
    LIFE_PROFILE_FEATURES,
    FINANCIAL_BEHAVIOR_FEATURES,
)
from .latent_regime_inference import (
    LatentRegimeInference,
    InferenceMethod,
    ActivityMask,
    FamilyAggregates,
    UniverseFeatures,
    BenchmarkFeatures,
    LatentRegimeState,
    LatentRegimeResult,
    BenchmarkConditionalAnalysis,
)
from .asset_inference import (
    AssetInferenceEngine,
    AssetInference,
    AssetExposure,
    BenchmarkLoader,
)

__all__ = [
    # Profilers
    "AlgoProfiler",
    "BenchmarkProfiler",
    "RegimeDetector",
    "CorrelationAnalyzer",
    # Clustering
    "AlgoClusterer",
    "ClusterMethod",
    "ScalerType",
    "ClusterResult",
    "TwoLayerClusterResult",
    "TemporalAlgoClusterer",
    "TemporalClusterResult",
    "TemporalClusteringOutput",
    "name_clusters",
    "name_life_clusters",
    "name_behavior_clusters",
    "PseudoLabelClusterer",
    "PseudoLabelStrategy",
    "PseudoLabelClusteringResult",
    # Feature extraction
    "AlgoFeatureExtractor",
    "AlgoFeatureConfig",
    "ACTIVITY_FEATURES",
    "PERFORMANCE_FEATURES",
    "TRANSITION_FEATURES",
    "BENCHMARK_FEATURES",
    "LIFE_PROFILE_FEATURES",
    "FINANCIAL_BEHAVIOR_FEATURES",
    # Latent Regime Inference (new approach)
    "LatentRegimeInference",
    "InferenceMethod",
    "ActivityMask",
    "FamilyAggregates",
    "UniverseFeatures",
    "BenchmarkFeatures",
    "LatentRegimeState",
    "LatentRegimeResult",
    "BenchmarkConditionalAnalysis",
    # Asset Inference
    "AssetInferenceEngine",
    "AssetInference",
    "AssetExposure",
    "BenchmarkLoader",
]
