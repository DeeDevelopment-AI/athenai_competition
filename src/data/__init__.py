"""
Módulo de datos: carga, preprocesamiento y feature engineering.
"""

from .loader import DataLoader, AlgorithmData, BenchmarkData
from .preprocessor import DataPreprocessor, ProcessedAlgoData, ProcessedBenchmarkData, TrimInfo, trim_dead_tail
from .feature_engineering import FeatureEngineer
from .trade_analyzer import TradeAnalyzer, CUTOFF_DATE

__all__ = [
    "DataLoader",
    "AlgorithmData",
    "BenchmarkData",
    "DataPreprocessor",
    "ProcessedAlgoData",
    "ProcessedBenchmarkData",
    "TrimInfo",
    "trim_dead_tail",
    "FeatureEngineer",
    "TradeAnalyzer",
    "CUTOFF_DATE",
]
