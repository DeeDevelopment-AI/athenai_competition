"""
Módulo de entorno RL: simulador financiero custom (sin Gymnasium).
"""

from .market_simulator import MarketSimulator, SimulatorState, StepResult
from .trading_env import TradingEnvironment, EpisodeConfig, VecTradingEnv
from .cost_model import CostModel
from .constraints import PortfolioConstraints, ConstraintViolation
from .reward import RewardFunction, RewardType, RewardComponents

__all__ = [
    "MarketSimulator",
    "SimulatorState",
    "StepResult",
    "TradingEnvironment",
    "EpisodeConfig",
    "VecTradingEnv",
    "CostModel",
    "PortfolioConstraints",
    "ConstraintViolation",
    "RewardFunction",
    "RewardType",
    "RewardComponents",
]
