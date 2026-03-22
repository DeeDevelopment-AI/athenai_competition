"""
Modulo de agentes RL: PPO, SAC, TD3 y offline RL.
"""

from .base import BaseAgent, TrainingMetrics, compute_sharpe_from_rewards
from .callbacks import (
    CSVLoggerCallback,
    EarlyStoppingCallback,
    FinancialMetricsCallback,
    ProgressCallback,
    create_eval_callback,
)
from .offline_rl import CQLAllocatorPlaceholder, OfflineDataset, OfflineDatasetBuilder
from .ppo_agent import PPOAllocator
from .sac_agent import SACAllocator
from .td3_agent import TD3Allocator

__all__ = [
    # Base classes
    "BaseAgent",
    "TrainingMetrics",
    "compute_sharpe_from_rewards",
    # Agents
    "PPOAllocator",
    "SACAllocator",
    "TD3Allocator",
    # Offline RL
    "CQLAllocatorPlaceholder",
    "OfflineDataset",
    "OfflineDatasetBuilder",
    # Callbacks
    "FinancialMetricsCallback",
    "EarlyStoppingCallback",
    "CSVLoggerCallback",
    "ProgressCallback",
    "create_eval_callback",
]
