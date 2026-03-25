"""
Ant-colony meta allocator for portfolio weights.

The implementation follows the ACO flow described in the reference paper:
- initialize a pheromone matrix with uniform values
- let each ant sample discrete portfolio-weight buckets using
  `pheromone^alpha * heuristic^beta`
- normalize sampled bucket values into portfolio weights
- evaluate fitness with a Sharpe-oriented portfolio objective
- evaporate and reinforce pheromones using the best ants

It reuses the existing Phase 7 selection, regime weighting, and reporting
machinery by subclassing the current swarm allocator/backtester classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .meta_allocator import (
    SwarmAllocatorBacktester,
    SwarmConfig,
    SwarmMetaAllocator,
    SwarmOptimizationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ACOConfig(SwarmConfig):
    """Configuration for the ant-colony portfolio allocator."""

    n_iterations: int = 60
    n_ants: int = 96
    pheromone_power: float = 1.0
    heuristic_power: float = 2.0
    evaporation_rate: float = 0.30
    pheromone_deposit_scale: float = 1.0
    elite_ants: int = 8
    weight_buckets: int = 21
    heuristic_floor: float = 1e-3
    sharpe_weight: float = 1.75
    entropy_reward_weight: float = 0.05
    use_gpu: bool = False
    objective_name: str = "aco_sharpe_balanced"


class ACOMetaAllocator(SwarmMetaAllocator):
    """Ant colony optimizer for meta allocation."""

    config: ACOConfig

    def optimize(
        self,
        returns_window: pd.DataFrame,
        previous_weights: np.ndarray,
        regime_weights: Optional[np.ndarray] = None,
    ) -> SwarmOptimizationResult:
        algo_ids = returns_window.columns.tolist()
        returns_np = returns_window.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        family_matrix = self._build_family_matrix(algo_ids)
        return self._optimize_aco(
            returns_np=returns_np,
            previous_weights=previous_weights.astype(np.float32, copy=False),
            algo_ids=algo_ids,
            family_matrix=family_matrix,
            regime_weights=regime_weights,
        )

    def _optimize_aco(
        self,
        returns_np: np.ndarray,
        previous_weights: np.ndarray,
        algo_ids: list[str],
        family_matrix: Optional[np.ndarray],
        regime_weights: Optional[np.ndarray],
    ) -> SwarmOptimizationResult:
        n_assets = returns_np.shape[1]
        if n_assets == 0:
            raise ValueError("ACO optimizer received an empty universe")

        n_buckets = max(int(self.config.weight_buckets), 2)
        pheromone = np.ones((n_assets, n_buckets), dtype=np.float32)
        bucket_values = np.linspace(0.0, 1.0, n_buckets, dtype=np.float32)
        heuristic_matrix, asset_strength = self._build_heuristic_matrix(returns_np, algo_ids)
        family_caps = self._build_family_caps(algo_ids)
        family_reward_vector = self._build_family_reward_vector(algo_ids)

        seed_buckets = self._build_seed_buckets(previous_weights, asset_strength, n_buckets)
        best_score = -np.inf
        best_weights = np.full(n_assets, 1.0 / max(n_assets, 1), dtype=np.float32)
        best_buckets = seed_buckets[0].copy()
        best_diag: dict[str, float] = {}
        best_iteration = -1

        n_ants = max(int(self.config.n_ants), 1)
        elite_ants = max(1, min(int(self.config.elite_ants), n_ants))

        for iteration in range(max(int(self.config.n_iterations), 1)):
            ant_buckets = np.zeros((n_ants, n_assets), dtype=np.int32)
            ant_weights = np.zeros((n_ants, n_assets), dtype=np.float32)

            for ant_idx in range(n_ants):
                sampled_buckets = self._sample_weight_buckets(pheromone, heuristic_matrix)
                ant_buckets[ant_idx] = sampled_buckets
                ant_weights[ant_idx] = self._buckets_to_weights(
                    sampled_buckets,
                    bucket_values=bucket_values,
                    fallback_order=np.argsort(asset_strength)[::-1],
                )

            for seed_idx, seed in enumerate(seed_buckets[: min(len(seed_buckets), n_ants)]):
                ant_buckets[seed_idx] = seed
                ant_weights[seed_idx] = self._buckets_to_weights(
                    seed,
                    bucket_values=bucket_values,
                    fallback_order=np.argsort(asset_strength)[::-1],
                )

            scores, diagnostics = self._evaluate_aco_numpy(
                weights=ant_weights,
                returns_np=returns_np,
                previous_weights=previous_weights,
                family_matrix=family_matrix,
                family_caps=family_caps,
                family_reward_vector=family_reward_vector,
                regime_weights=regime_weights,
            )

            iteration_best_idx = int(np.argmax(scores))
            iteration_best_score = float(scores[iteration_best_idx])
            if iteration_best_score > best_score:
                best_score = iteration_best_score
                best_weights, _ = self._finalize_weights_numpy(
                    ant_weights[iteration_best_idx : iteration_best_idx + 1],
                    returns_np=returns_np,
                    regime_weights=regime_weights,
                )
                best_weights = best_weights[0]
                best_buckets = ant_buckets[iteration_best_idx].astype(np.int32, copy=True)
                best_diag = {key: float(value[iteration_best_idx]) for key, value in diagnostics.items()}
                best_iteration = iteration

            pheromone *= max(1e-6, 1.0 - float(self.config.evaporation_rate))
            elite_indices = np.argsort(scores)[-elite_ants:]
            elite_scores = scores[elite_indices].astype(np.float32, copy=False)
            score_floor = float(np.min(elite_scores))
            shifted = elite_scores - score_floor
            scale = float(np.max(shifted))
            if scale < 1e-8:
                scaled_deposits = np.full(elite_ants, self.config.pheromone_deposit_scale, dtype=np.float32)
            else:
                scaled_deposits = (
                    (shifted / scale) + self.config.heuristic_floor
                ).astype(np.float32, copy=False) * float(self.config.pheromone_deposit_scale)

            repeated_assets = np.tile(np.arange(n_assets, dtype=np.int32), elite_ants)
            repeated_buckets = ant_buckets[elite_indices].reshape(-1)
            repeated_weights = ant_weights[elite_indices].reshape(-1)
            repeated_deposits = np.repeat(scaled_deposits, n_assets) * np.maximum(
                repeated_weights,
                1.0 / max(n_assets * 4, 1),
            )
            np.add.at(pheromone, (repeated_assets, repeated_buckets), repeated_deposits.astype(np.float32, copy=False))

            np.add.at(
                pheromone,
                (np.arange(n_assets, dtype=np.int32), best_buckets),
                np.full(n_assets, float(self.config.pheromone_deposit_scale), dtype=np.float32),
            )
            pheromone = np.clip(pheromone, self.config.heuristic_floor, 1e6)

        best_diag = {
            **best_diag,
            "best_iteration": float(best_iteration),
            "n_ants": float(n_ants),
            "weight_buckets": float(n_buckets),
            "pheromone_mean": float(pheromone.mean()),
            "pheromone_std": float(pheromone.std()),
        }
        return SwarmOptimizationResult(
            weights=best_weights.astype(np.float32, copy=False),
            score=float(best_score),
            selected_algorithms=algo_ids,
            diagnostics=best_diag,
            active_count=int(best_diag.get("active_count", 0.0)),
        )

    def _build_seed_buckets(
        self,
        previous_weights: np.ndarray,
        asset_strength: np.ndarray,
        n_buckets: int,
    ) -> list[np.ndarray]:
        equal_bucket = np.full(len(previous_weights), max(1, (n_buckets - 1) // 2), dtype=np.int32)
        previous_max = float(np.max(previous_weights)) if previous_weights.size else 0.0
        if previous_max > 1e-8:
            previous_bucket = np.clip(
                np.rint(previous_weights / previous_max * (n_buckets - 1)),
                0,
                n_buckets - 1,
            ).astype(np.int32, copy=False)
        else:
            previous_bucket = equal_bucket.copy()

        momentum_bucket = np.zeros(len(previous_weights), dtype=np.int32)
        top_count = max(1, min(len(previous_weights), max(2, len(previous_weights) // 4)))
        top_indices = np.argsort(asset_strength)[-top_count:]
        momentum_bucket[top_indices] = n_buckets - 1
        return [equal_bucket, previous_bucket, momentum_bucket]

    def _build_heuristic_matrix(
        self,
        returns_np: np.ndarray,
        algo_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_returns = returns_np.mean(axis=0).astype(np.float32, copy=False)
        volatility = returns_np.std(axis=0).astype(np.float32, copy=False)
        volatility = np.maximum(volatility, 1e-6)
        sharpe_proxy = mean_returns / volatility
        upside = np.clip(returns_np, 0.0, None).mean(axis=0).astype(np.float32, copy=False)
        downside = np.abs(np.clip(returns_np, None, 0.0)).mean(axis=0).astype(np.float32, copy=False)
        reward = self._build_family_reward_vector(algo_ids)
        if reward is None:
            reward = np.zeros(len(algo_ids), dtype=np.float32)

        asset_signal = (
            1.25 * sharpe_proxy
            + 0.60 * (mean_returns / np.maximum(downside, 1e-6))
            + 0.30 * (upside / volatility)
            + 0.20 * reward
        ).astype(np.float32, copy=False)
        normalized_signal = self._normalize_to_unit_interval(asset_signal)
        bucket_scale = np.linspace(0.0, 1.0, max(int(self.config.weight_buckets), 2), dtype=np.float32)

        strength = normalized_signal[:, None]
        heuristic = 1.0 + 2.0 * (
            strength * bucket_scale[None, :] + (1.0 - strength) * (1.0 - bucket_scale[None, :])
        )
        heuristic = heuristic.astype(np.float32, copy=False)
        heuristic += float(self.config.heuristic_floor)
        return heuristic, normalized_signal

    def _normalize_to_unit_interval(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return np.full(values.shape, 0.5, dtype=np.float32)

        clean = values.copy()
        floor = float(np.nanmin(clean[finite_mask]))
        clean[~finite_mask] = floor
        min_value = float(clean.min())
        span = float(clean.max() - min_value)
        if span < 1e-8:
            return np.full(values.shape, 0.5, dtype=np.float32)
        return ((clean - min_value) / span).astype(np.float32, copy=False)

    def _sample_weight_buckets(self, pheromone: np.ndarray, heuristic_matrix: np.ndarray) -> np.ndarray:
        desirability = np.power(pheromone, float(self.config.pheromone_power)) * np.power(
            heuristic_matrix,
            float(self.config.heuristic_power),
        )
        row_sums = desirability.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 1e-8] = 1.0
        probabilities = desirability / row_sums
        cumulative = probabilities.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        draws = self._rng.random(pheromone.shape[0], dtype=np.float32)
        sampled = (draws[:, None] > cumulative).sum(axis=1)
        return np.clip(sampled, 0, pheromone.shape[1] - 1).astype(np.int32, copy=False)

    def _buckets_to_weights(
        self,
        bucket_indices: np.ndarray,
        bucket_values: np.ndarray,
        fallback_order: np.ndarray,
    ) -> np.ndarray:
        raw = bucket_values[bucket_indices].astype(np.float32, copy=False)
        if float(raw.sum()) <= 1e-8:
            raw = np.zeros_like(raw)
            raw[int(fallback_order[0])] = float(bucket_values[-1])
        normalized = raw / np.maximum(raw.sum(), 1e-8)
        return self._project_weights_numpy(normalized[None, :])[0]

    def _evaluate_aco_numpy(
        self,
        weights: np.ndarray,
        returns_np: np.ndarray,
        previous_weights: np.ndarray,
        family_matrix: Optional[np.ndarray],
        family_caps: Optional[np.ndarray],
        family_reward_vector: Optional[np.ndarray],
        regime_weights: Optional[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        weights, exposure_scale = self._finalize_weights_numpy(
            weights=weights,
            returns_np=returns_np,
            regime_weights=regime_weights,
        )

        port_returns = weights @ returns_np.T
        if regime_weights is not None:
            weighted_returns = port_returns * regime_weights[None, :]
            denom = max(float(regime_weights.sum()), 1e-8)
            mean_returns = weighted_returns.sum(axis=1) / denom
            centered = port_returns - mean_returns[:, None]
            variance = (centered**2 * regime_weights[None, :]).sum(axis=1) / denom
        else:
            mean_returns = port_returns.mean(axis=1)
            variance = port_returns.var(axis=1)

        ann_return = mean_returns * 252.0
        ann_vol = np.sqrt(np.maximum(variance, 1e-8)) * np.sqrt(252.0)
        sharpe_ratio = ann_return / np.maximum(ann_vol, 1e-8)

        turnover = np.abs(weights - previous_weights[None, :]).sum(axis=1) / 2.0
        concentration = (weights**2).sum(axis=1)

        cov = np.cov(returns_np, rowvar=False)
        cov = np.nan_to_num(cov, nan=0.0)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float32)
        port_var = np.einsum("bi,ij,bj->b", weights, cov, weights)
        diversification = 1.0 - np.sqrt(np.maximum(port_var, 0.0))

        family_penalty = np.zeros(weights.shape[0], dtype=np.float32)
        if family_matrix is not None and family_matrix.shape[1] > 0:
            family_exposure = weights @ family_matrix
            caps = family_caps[None, :] if family_caps is not None else self.config.max_family_exposure
            family_penalty = np.maximum(family_exposure - caps, 0.0).sum(axis=1)

        family_alpha_reward = np.zeros(weights.shape[0], dtype=np.float32)
        if family_reward_vector is not None:
            family_alpha_reward = (weights * family_reward_vector[None, :]).sum(axis=1)

        risk_budget_penalty = np.abs(ann_vol - self.config.target_portfolio_vol)
        active_count = (weights >= self.config.min_active_weight).sum(axis=1).astype(np.float32)
        sparsity_penalty = active_count / max(weights.shape[1], 1)
        gross_exposure = weights.sum(axis=1)
        under_investment_penalty = np.maximum(self.config.min_gross_exposure - gross_exposure, 0.0)

        if weights.shape[1] > 1:
            entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-8)), axis=1) / np.log(weights.shape[1])
        else:
            entropy = np.ones(weights.shape[0], dtype=np.float32)

        score = (
            self.config.sharpe_weight * sharpe_ratio
            + self.config.expected_return_weight * ann_return
            - self.config.volatility_weight * ann_vol
            - self.config.turnover_weight * turnover
            - self.config.concentration_weight * concentration
            + self.config.diversification_weight * diversification
            + self.config.entropy_reward_weight * entropy
            - self.config.family_penalty_weight * family_penalty
            + self.config.family_alpha_reward_weight * family_alpha_reward
            - self.config.risk_budget_weight * risk_budget_penalty
            - self.config.sparsity_penalty_weight * sparsity_penalty
            - self.config.under_investment_penalty_weight * under_investment_penalty
        ).astype(np.float32, copy=False)

        diagnostics = {
            "ann_return": ann_return.astype(np.float32, copy=False),
            "ann_vol": ann_vol.astype(np.float32, copy=False),
            "sharpe_ratio": sharpe_ratio.astype(np.float32, copy=False),
            "turnover": turnover.astype(np.float32, copy=False),
            "concentration": concentration.astype(np.float32, copy=False),
            "diversification": diversification.astype(np.float32, copy=False),
            "entropy": entropy.astype(np.float32, copy=False),
            "family_penalty": family_penalty.astype(np.float32, copy=False),
            "family_alpha_reward": family_alpha_reward.astype(np.float32, copy=False),
            "risk_budget_penalty": risk_budget_penalty.astype(np.float32, copy=False),
            "active_count": active_count.astype(np.float32, copy=False),
            "gross_exposure": gross_exposure.astype(np.float32, copy=False),
            "exposure_scale": exposure_scale.astype(np.float32, copy=False),
            "under_investment_penalty": under_investment_penalty.astype(np.float32, copy=False),
        }
        return score, diagnostics

    def _finalize_weights_numpy(
        self,
        weights: np.ndarray,
        returns_np: np.ndarray,
        regime_weights: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = self._project_weights_numpy(weights)
        port_returns = weights @ returns_np.T
        if regime_weights is not None:
            weighted_returns = port_returns * regime_weights[None, :]
            denom = max(float(regime_weights.sum()), 1e-8)
            mean_returns = weighted_returns.sum(axis=1) / denom
            centered = port_returns - mean_returns[:, None]
            variance = (centered**2 * regime_weights[None, :]).sum(axis=1) / denom
        else:
            mean_returns = port_returns.mean(axis=1)
            variance = port_returns.var(axis=1)

        ann_vol = np.sqrt(np.maximum(variance, 1e-8)) * np.sqrt(252.0)
        finalized_weights, exposure_scale = self._apply_risk_budget_numpy(weights, ann_vol)
        gross = np.maximum(finalized_weights.sum(axis=1, keepdims=True), 1e-8)
        normalized = finalized_weights / gross
        finalized_weights = self._project_weights_numpy(normalized) * gross
        return finalized_weights, exposure_scale


class ACOAllocatorBacktester(SwarmAllocatorBacktester):
    """Phase-7-style backtester backed by the ACO optimizer."""

    config: ACOConfig

    def __init__(
        self,
        algo_returns: pd.DataFrame,
        features: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        benchmark_weights: Optional[pd.DataFrame],
        config: ACOConfig,
        family_labels: Optional[pd.Series] = None,
        family_alpha_scores: Optional[dict[str, float]] = None,
        cluster_history: Optional[pd.DataFrame] = None,
        cluster_stability: Optional[pd.DataFrame] = None,
        cluster_alpha_scores: Optional[dict[str, dict[str, float]]] = None,
        regime_labels: Optional[pd.Series] = None,
        selection_factor: str | list[str] | tuple[str, ...] = "rolling_sharpe_21d",
    ):
        super().__init__(
            algo_returns=algo_returns,
            features=features,
            benchmark_returns=benchmark_returns,
            benchmark_weights=benchmark_weights,
            config=config,
            family_labels=family_labels,
            family_alpha_scores=family_alpha_scores,
            cluster_history=cluster_history,
            cluster_stability=cluster_stability,
            cluster_alpha_scores=cluster_alpha_scores,
            regime_labels=regime_labels,
            selection_factor=selection_factor,
        )
        self.allocator = ACOMetaAllocator(
            config=config,
            family_labels=family_labels,
            family_alpha_scores=self.family_alpha_scores,
        )


__all__ = [
    "ACOAllocatorBacktester",
    "ACOConfig",
]
