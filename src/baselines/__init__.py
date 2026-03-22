"""
Baselines module: Factor-based allocation strategies.

Architecture:
1. Factor Model (selection) → 2. Portfolio Optimizer (weighting) → 3. Constraints (benchmark matching)
"""

from .base import BaseAllocator, FactorBasedAllocator, AllocationResult
from .equal_weight import EqualWeightAllocator
from .risk_parity import RiskParityAllocator, RiskParityERC
from .min_variance import MinVarianceAllocator
from .max_sharpe import MaxSharpeAllocator
from .momentum_allocator import MomentumAllocator, MomentumVolAdjusted
from .vol_targeting import VolTargetingAllocator, AdaptiveVolTargeting

__all__ = [
    # Base classes
    "BaseAllocator",
    "FactorBasedAllocator",
    "AllocationResult",
    # Allocators
    "EqualWeightAllocator",
    "RiskParityAllocator",
    "RiskParityERC",
    "MinVarianceAllocator",
    "MaxSharpeAllocator",
    "MomentumAllocator",
    "MomentumVolAdjusted",
    "VolTargetingAllocator",
    "AdaptiveVolTargeting",
]
