"""
Swarm-based meta allocator using particle swarm optimization (PSO).

Design goals:
- Reuse Phase 1 processed returns/features and optional Phase 2 family/regime outputs
- Keep core math vectorized
- Prefer GPU execution through torch when available
- Fall back cleanly to CPU without changing call sites
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_full_metrics
from src.utils.device import HAS_TORCH, get_device
from src.utils.numba_utils import backtest_portfolio_returns

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

if HAS_TORCH:
    import torch
else:
    torch = None


def _normalize_cluster_key(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            numeric = float(stripped)
            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            return stripped
        return stripped
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    return str(value).strip()


@njit(cache=True)
def _combine_cluster_scores_numba(
    cumulative_scores: np.ndarray,
    weekly_scores: np.ndarray,
    monthly_scores: np.ndarray,
    stability_scores: np.ndarray,
) -> np.ndarray:
    out = np.empty(cumulative_scores.shape[0], dtype=np.float32)
    for i in range(cumulative_scores.shape[0]):
        consensus = 0.0
        if cumulative_scores[i] > 0.0:
            consensus += 1.0
        if weekly_scores[i] > 0.0:
            consensus += 1.0
        if monthly_scores[i] > 0.0:
            consensus += 1.0
        out[i] = (
            0.20 * cumulative_scores[i]
            + 0.45 * weekly_scores[i]
            + 0.35 * monthly_scores[i]
            + 0.35 * stability_scores[i]
            + 0.20 * consensus
        )
    return out


@njit(cache=True)
def _compute_gate_mask_numba(
    feature_pass_ratio: np.ndarray,
    cluster_signal: np.ndarray,
    stability_scores: np.ndarray,
    composite_scores: np.ndarray,
    min_feature_pass_ratio: float,
    min_cluster_signal: float,
    min_stability: float,
    min_composite_score: float,
) -> np.ndarray:
    out = np.zeros(feature_pass_ratio.shape[0], dtype=np.uint8)
    for i in range(feature_pass_ratio.shape[0]):
        if (
            feature_pass_ratio[i] >= min_feature_pass_ratio
            and cluster_signal[i] >= min_cluster_signal
            and stability_scores[i] >= min_stability
            and composite_scores[i] >= min_composite_score
        ):
            out[i] = 1
    return out


@njit(cache=True)
def _compute_consensus_gate_mask_numba(
    feature_pass_ratio: np.ndarray,
    weekly_scores: np.ndarray,
    monthly_scores: np.ndarray,
    cumulative_scores: np.ndarray,
    stability_scores: np.ndarray,
    composite_scores: np.ndarray,
    min_feature_pass_ratio: float,
    min_weekly_score: float,
    min_monthly_score: float,
    min_cumulative_score: float,
    min_stability: float,
    min_composite_score: float,
) -> np.ndarray:
    out = np.zeros(feature_pass_ratio.shape[0], dtype=np.uint8)
    for i in range(feature_pass_ratio.shape[0]):
        positive_weekly = weekly_scores[i] >= min_weekly_score
        positive_monthly = monthly_scores[i] >= min_monthly_score
        positive_cumulative = cumulative_scores[i] >= min_cumulative_score
        consensus_count = 0
        if positive_weekly:
            consensus_count += 1
        if positive_monthly:
            consensus_count += 1
        if positive_cumulative:
            consensus_count += 1

        primary_consensus = positive_weekly and positive_monthly
        secondary_consensus = positive_weekly and positive_cumulative and stability_scores[i] >= (min_stability + 0.05)

        if (
            feature_pass_ratio[i] >= min_feature_pass_ratio
            and stability_scores[i] >= min_stability
            and composite_scores[i] >= min_composite_score
            and consensus_count >= 2
            and (primary_consensus or secondary_consensus)
        ):
            out[i] = 1
    return out


@dataclass
class SwarmConfig:
    """Configuration for the PSO meta allocator."""

    lookback_window: int = 126
    rebalance_frequency: str = "weekly"
    top_k: int = 256
    min_history: int = 63
    n_particles: int = 128
    n_iterations: int = 80
    inertia: float = 0.70
    cognitive_weight: float = 1.40
    social_weight: float = 1.30
    max_weight: float = 0.40
    max_family_exposure: float = 0.30
    expected_return_weight: float = 1.00
    excess_return_weight: float = 0.0
    information_ratio_weight: float = 0.0
    benchmark_hit_rate_weight: float = 0.0
    downside_excess_weight: float = 0.0
    volatility_weight: float = 0.50
    tracking_error_weight: float = 0.35
    turnover_weight: float = 0.15
    concentration_weight: float = 0.10
    diversification_weight: float = 0.10
    family_penalty_weight: float = 0.20
    family_alpha_reward_weight: float = 0.15
    risk_budget_weight: float = 0.30
    sparsity_penalty_weight: float = 0.01
    regime_focus: float = 1.50
    target_portfolio_vol: float = 0.16
    min_active_weight: float = 0.0025
    min_gross_exposure: float = 0.85
    under_investment_penalty_weight: float = 0.35
    seed: int = 42
    use_gpu: bool = True
    objective_name: str = "risk_calibrated_return"
    selection_mode: str = "legacy"
    normalize_objective_metrics: bool = False
    min_selection_sharpe_21d: float = 0.10
    min_selection_calmar_21d: float = 0.00
    min_selection_profit_factor_21d: float = 1.00
    max_selection_drawdown_63d: float = 0.12
    min_selection_pass_ratio: float = 0.75
    cluster_alpha_weight: float = 1.25
    cluster_stability_weight: float = 0.35
    cluster_gate_min_signal: float = 0.05
    cluster_gate_min_stability: float = 0.20
    cluster_gate_min_candidates: int = 24
    cluster_gate_score_quantile: float = 0.55
    cluster_gate_min_consensus: int = 2


@dataclass
class SwarmOptimizationResult:
    """Optimization output for one rebalance date."""

    weights: np.ndarray
    score: float
    selected_algorithms: list[str]
    diagnostics: dict[str, float] = field(default_factory=dict)
    active_count: int = 0


@dataclass
class SwarmBacktestResult:
    """Backtest artifacts for the swarm allocator."""

    weights: pd.DataFrame
    portfolio_returns: pd.Series
    benchmark_returns: Optional[pd.Series]
    diagnostics: pd.DataFrame
    summary: dict
    comparison: dict


class SwarmMetaAllocator:
    """Particle swarm optimizer for meta allocation."""

    def __init__(
        self,
        config: SwarmConfig,
        family_labels: Optional[pd.Series] = None,
        family_alpha_scores: Optional[dict[str, float]] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.family_labels = family_labels
        self.family_alpha_scores = family_alpha_scores or {}
        self.device = self._resolve_device(device)
        self._torch_device = torch.device(self.device) if HAS_TORCH else None
        self._rng = np.random.default_rng(config.seed)
        if HAS_TORCH:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device
        if self.config.use_gpu and HAS_TORCH:
            return get_device()
        return "cpu"

    def optimize(
        self,
        returns_window: pd.DataFrame,
        previous_weights: np.ndarray,
        regime_weights: Optional[np.ndarray] = None,
    ) -> SwarmOptimizationResult:
        """Optimize weights over the selected universe."""
        algo_ids = returns_window.columns.tolist()
        returns_np = returns_window.fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        family_matrix = self._build_family_matrix(algo_ids)
        if HAS_TORCH:
            return self._optimize_torch(
                returns_np=returns_np,
                previous_weights=previous_weights.astype(np.float32, copy=False),
                algo_ids=algo_ids,
                family_matrix=family_matrix,
                regime_weights=regime_weights,
            )
        return self._optimize_numpy(
            returns_np=returns_np,
            previous_weights=previous_weights.astype(np.float32, copy=False),
            algo_ids=algo_ids,
            family_matrix=family_matrix,
            regime_weights=regime_weights,
        )

    def _build_family_matrix(self, algo_ids: list[str]) -> Optional[np.ndarray]:
        if self.family_labels is None or len(self.family_labels) == 0:
            return None

        aligned = self.family_labels.reindex(algo_ids)
        if aligned.isna().all():
            return None

        families = sorted({str(v) for v in aligned.dropna().tolist()})
        family_to_idx = {fam: idx for idx, fam in enumerate(families)}
        matrix = np.zeros((len(algo_ids), len(families)), dtype=np.float32)
        for row_idx, value in enumerate(aligned.tolist()):
            if pd.isna(value):
                continue
            matrix[row_idx, family_to_idx[str(value)]] = 1.0
        return matrix

    def _build_family_caps(self, algo_ids: list[str]) -> Optional[np.ndarray]:
        if self.family_labels is None or len(self.family_labels) == 0:
            return None

        aligned = self.family_labels.reindex(algo_ids)
        if aligned.isna().all():
            return None

        families = sorted({str(v) for v in aligned.dropna().tolist()})
        ranked_families = [
            family for family, _ in sorted(
                self.family_alpha_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        caps = np.full(len(families), min(self.config.max_family_exposure, 0.15), dtype=np.float32)
        if ranked_families:
            top_family = ranked_families[0]
            if top_family in families:
                caps[families.index(top_family)] = 0.90
        if len(ranked_families) > 1:
            second_family = ranked_families[1]
            if second_family in families:
                caps[families.index(second_family)] = 0.55
        return caps

    def _build_family_reward_vector(self, algo_ids: list[str]) -> Optional[np.ndarray]:
        if self.family_labels is None or len(self.family_labels) == 0 or not self.family_alpha_scores:
            return None
        aligned = self.family_labels.reindex(algo_ids)
        rewards = np.zeros(len(algo_ids), dtype=np.float32)
        for idx, value in enumerate(aligned.tolist()):
            if pd.isna(value):
                continue
            rewards[idx] = float(self.family_alpha_scores.get(str(value), 0.0))
        return rewards

    def _init_particles(self, n_assets: int, previous_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        particles = self._rng.random((self.config.n_particles, n_assets), dtype=np.float32)
        velocities = self._rng.normal(
            loc=0.0,
            scale=0.05,
            size=(self.config.n_particles, n_assets),
        ).astype(np.float32)

        if n_assets > 0:
            equal_weight = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
            particles[0] = equal_weight
            particles[1] = np.maximum(previous_weights, 0.0)

        return particles, velocities

    def _project_weights_torch(self, positions):
        weights = torch.clamp(positions, min=0.0)
        row_sums = weights.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums > 1e-8, row_sums, torch.ones_like(row_sums))
        weights = weights / row_sums

        for _ in range(10):
            over = weights > self.config.max_weight
            if not torch.any(over):
                break

            capped = torch.where(over, torch.full_like(weights, self.config.max_weight), weights)
            active = ~over
            remaining_mass = torch.clamp(1.0 - capped.sum(dim=1, keepdim=True), min=0.0)
            active_weights = torch.where(active, capped, torch.zeros_like(capped))
            active_sums = active_weights.sum(dim=1, keepdim=True)
            active_counts = active.sum(dim=1, keepdim=True).to(weights.dtype)

            redistributed = torch.where(
                active_sums > 1e-8,
                active_weights / active_sums * remaining_mass,
                torch.where(active, remaining_mass / torch.clamp(active_counts, min=1.0), torch.zeros_like(weights)),
            )
            weights = torch.where(over, torch.full_like(weights, self.config.max_weight), redistributed)

        return weights

    def _project_weights_numpy(self, positions: np.ndarray) -> np.ndarray:
        weights = np.clip(positions, 0.0, None).astype(np.float32, copy=True)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)

        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 1e-8] = 1.0
        weights = weights / row_sums

        for _ in range(10):
            over = weights > self.config.max_weight
            if not over.any():
                break

            capped = np.where(over, self.config.max_weight, weights)
            active = ~over
            remaining_mass = np.clip(1.0 - capped.sum(axis=1, keepdims=True), 0.0, None)
            active_weights = np.where(active, capped, 0.0)
            active_sums = active_weights.sum(axis=1, keepdims=True)
            active_counts = active.sum(axis=1, keepdims=True)

            redistributed = np.divide(
                active_weights,
                np.where(active_sums > 1e-8, active_sums, 1.0),
            ) * remaining_mass
            equal_fill = np.where(
                active,
                remaining_mass / np.maximum(active_counts, 1),
                0.0,
            )
            redistributed = np.where(active_sums > 1e-8, redistributed, equal_fill)
            weights = np.where(over, self.config.max_weight, redistributed)

        return weights

    def _apply_risk_budget_torch(self, weights, ann_vol):
        scale = torch.clamp(
            torch.full_like(ann_vol, self.config.target_portfolio_vol) / torch.clamp(ann_vol, min=1e-8),
            max=1.0,
        )
        scaled = weights * scale.unsqueeze(1)
        thresholded = torch.where(scaled >= self.config.min_active_weight, scaled, torch.zeros_like(scaled))
        gross = thresholded.sum(dim=1, keepdim=True)
        gross = torch.clamp(gross, min=1e-8)
        normalized = thresholded / gross
        return normalized * scale.unsqueeze(1), scale

    def _apply_risk_budget_numpy(self, weights: np.ndarray, ann_vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = np.clip(self.config.target_portfolio_vol / np.maximum(ann_vol, 1e-8), 0.0, 1.0).astype(np.float32)
        scaled = weights * scale[:, None]
        thresholded = np.where(scaled >= self.config.min_active_weight, scaled, 0.0)
        gross = np.maximum(thresholded.sum(axis=1, keepdims=True), 1e-8)
        normalized = thresholded / gross
        return (normalized * scale[:, None]).astype(np.float32, copy=False), scale

    def _zscore_torch(self, values):
        if values.numel() <= 1:
            return torch.zeros_like(values)
        centered = values - values.mean()
        scale = centered.std(unbiased=False)
        if torch.isnan(scale) or scale < 1e-8:
            return torch.zeros_like(values)
        return centered / scale

    def _zscore_numpy(self, values: np.ndarray) -> np.ndarray:
        if values.size <= 1:
            return np.zeros_like(values, dtype=np.float32)
        centered = values - values.mean()
        scale = centered.std()
        if np.isnan(scale) or scale < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return centered / scale

    def _evaluate_torch(
        self,
        weights,
        returns_tensor,
        previous_weights,
        family_tensor,
        family_caps_tensor,
        family_reward_tensor,
        regime_tensor,
    ):
        port_returns = weights @ returns_tensor.T
        if regime_tensor is not None:
            weighted_port_returns = port_returns * regime_tensor
            denom = torch.clamp(regime_tensor.sum(), min=1e-8)
            mean_returns = weighted_port_returns.sum(dim=1) / denom
            centered = port_returns - mean_returns.unsqueeze(1)
            variance = ((centered**2) * regime_tensor).sum(dim=1) / denom
        else:
            mean_returns = port_returns.mean(dim=1)
            variance = port_returns.var(dim=1, unbiased=False)

        volatility = torch.sqrt(torch.clamp(variance, min=1e-8))
        ann_return = mean_returns * 252.0
        ann_vol = volatility * np.sqrt(252.0)
        weights, exposure_scale = self._apply_risk_budget_torch(weights, ann_vol)
        port_returns = weights @ returns_tensor.T
        if regime_tensor is not None:
            weighted_port_returns = port_returns * regime_tensor
            denom = torch.clamp(regime_tensor.sum(), min=1e-8)
            mean_returns = weighted_port_returns.sum(dim=1) / denom
            centered = port_returns - mean_returns.unsqueeze(1)
            variance = ((centered**2) * regime_tensor).sum(dim=1) / denom
        else:
            mean_returns = port_returns.mean(dim=1)
            variance = port_returns.var(dim=1, unbiased=False)
        volatility = torch.sqrt(torch.clamp(variance, min=1e-8))
        ann_return = mean_returns * 252.0
        ann_vol = volatility * np.sqrt(252.0)

        turnover = torch.abs(weights - previous_weights.unsqueeze(0)).sum(dim=1) / 2.0
        concentration = (weights**2).sum(dim=1)

        cov = torch.cov(returns_tensor.T) if returns_tensor.shape[1] > 1 else torch.zeros((returns_tensor.shape[1], returns_tensor.shape[1]), device=weights.device)
        cov = torch.nan_to_num(cov, nan=0.0)
        port_var = torch.einsum("bi,ij,bj->b", weights, cov, weights)
        diversification = 1.0 - torch.sqrt(torch.clamp(port_var, min=0.0))

        family_penalty = torch.zeros(weights.shape[0], device=weights.device)
        if family_tensor is not None and family_tensor.shape[1] > 0:
            family_exposure = weights @ family_tensor
            family_caps = (
                family_caps_tensor.unsqueeze(0)
                if family_caps_tensor is not None
                else torch.full_like(family_exposure, self.config.max_family_exposure)
            )
            over = torch.relu(family_exposure - family_caps)
            family_penalty = over.sum(dim=1)
        family_alpha_reward = torch.zeros(weights.shape[0], device=weights.device)
        if family_reward_tensor is not None:
            family_alpha_reward = (weights * family_reward_tensor.unsqueeze(0)).sum(dim=1)
        risk_budget_penalty = torch.abs(ann_vol - self.config.target_portfolio_vol)
        active_count = (weights >= self.config.min_active_weight).to(weights.dtype).sum(dim=1)
        sparsity_penalty = active_count / max(weights.shape[1], 1)
        gross_exposure = weights.sum(dim=1)
        under_investment_penalty = torch.relu(self.config.min_gross_exposure - gross_exposure)

        score = (
            self.config.expected_return_weight * ann_return
            - self.config.volatility_weight * ann_vol
            - self.config.turnover_weight * turnover
            - self.config.concentration_weight * concentration
            + self.config.diversification_weight * diversification
            - self.config.family_penalty_weight * family_penalty
            + self.config.family_alpha_reward_weight * family_alpha_reward
            - self.config.risk_budget_weight * risk_budget_penalty
            - self.config.sparsity_penalty_weight * sparsity_penalty
            - self.config.under_investment_penalty_weight * under_investment_penalty
        )
        diagnostics = {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "turnover": turnover,
            "concentration": concentration,
            "family_penalty": family_penalty,
            "family_alpha_reward": family_alpha_reward,
            "risk_budget_penalty": risk_budget_penalty,
            "active_count": active_count,
            "gross_exposure": gross_exposure,
            "exposure_scale": exposure_scale,
            "under_investment_penalty": under_investment_penalty,
        }
        return score, diagnostics

    def _optimize_torch(
        self,
        returns_np: np.ndarray,
        previous_weights: np.ndarray,
        algo_ids: list[str],
        family_matrix: Optional[np.ndarray],
        regime_weights: Optional[np.ndarray],
    ) -> SwarmOptimizationResult:
        n_assets = returns_np.shape[1]
        particles_np, velocities_np = self._init_particles(n_assets, previous_weights)

        returns_tensor = torch.as_tensor(returns_np, device=self._torch_device)
        previous_tensor = torch.as_tensor(previous_weights, device=self._torch_device)
        family_tensor = (
            torch.as_tensor(family_matrix, device=self._torch_device)
            if family_matrix is not None
            else None
        )
        family_caps = self._build_family_caps(algo_ids)
        family_caps_tensor = (
            torch.as_tensor(family_caps, device=self._torch_device)
            if family_caps is not None
            else None
        )
        family_reward_vector = self._build_family_reward_vector(algo_ids)
        family_reward_tensor = (
            torch.as_tensor(family_reward_vector, device=self._torch_device)
            if family_reward_vector is not None
            else None
        )
        regime_tensor = (
            torch.as_tensor(regime_weights.reshape(1, -1), device=self._torch_device)
            if regime_weights is not None
            else None
        )
        if regime_tensor is not None:
            regime_tensor = regime_tensor.squeeze(0)

        particles = torch.as_tensor(particles_np, device=self._torch_device)
        velocities = torch.as_tensor(velocities_np, device=self._torch_device)
        personal_best = particles.clone()
        projected = self._project_weights_torch(particles)
        personal_scores, _ = self._evaluate_torch(
            projected,
            returns_tensor,
            previous_tensor,
            family_tensor,
            family_caps_tensor,
            family_reward_tensor,
            regime_tensor,
        )
        best_idx = int(torch.argmax(personal_scores).item())
        global_best = particles[best_idx].clone()

        for _ in range(self.config.n_iterations):
            r1 = torch.rand_like(particles)
            r2 = torch.rand_like(particles)
            velocities = (
                self.config.inertia * velocities
                + self.config.cognitive_weight * r1 * (personal_best - particles)
                + self.config.social_weight * r2 * (global_best.unsqueeze(0) - particles)
            )
            particles = particles + velocities

            projected = self._project_weights_torch(particles)
            scores, _ = self._evaluate_torch(
                projected,
                returns_tensor,
                previous_tensor,
                family_tensor,
                family_caps_tensor,
                family_reward_tensor,
                regime_tensor,
            )
            improved = scores > personal_scores
            personal_best = torch.where(improved.unsqueeze(1), particles, personal_best)
            personal_scores = torch.where(improved, scores, personal_scores)
            best_idx = int(torch.argmax(personal_scores).item())
            global_best = personal_best[best_idx].clone()

        best_weights = self._project_weights_torch(global_best.unsqueeze(0)).squeeze(0)
        final_score, final_diag = self._evaluate_torch(
            best_weights.unsqueeze(0),
            returns_tensor,
            previous_tensor,
            family_tensor,
            family_caps_tensor,
            family_reward_tensor,
            regime_tensor,
        )
        diagnostics = {key: float(value.squeeze(0).detach().cpu().item()) for key, value in final_diag.items()}
        return SwarmOptimizationResult(
            weights=best_weights.detach().cpu().numpy(),
            score=float(final_score.squeeze(0).detach().cpu().item()),
            selected_algorithms=algo_ids,
            diagnostics=diagnostics,
            active_count=int(diagnostics.get("active_count", 0.0)),
        )

    def _evaluate_numpy(
        self,
        weights: np.ndarray,
        returns_np: np.ndarray,
        previous_weights: np.ndarray,
        family_matrix: Optional[np.ndarray],
        family_caps: Optional[np.ndarray],
        family_reward_vector: Optional[np.ndarray],
        regime_weights: Optional[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        weights = self._project_weights_numpy(weights)

        port_returns = weights @ returns_np.T
        if regime_weights is not None:
            weighted_returns = port_returns * regime_weights[None, :]
            denom = max(regime_weights.sum(), 1e-8)
            mean_returns = weighted_returns.sum(axis=1) / denom
            centered = port_returns - mean_returns[:, None]
            variance = (centered**2 * regime_weights[None, :]).sum(axis=1) / denom
        else:
            mean_returns = port_returns.mean(axis=1)
            variance = port_returns.var(axis=1)

        ann_return = mean_returns * 252.0
        ann_vol = np.sqrt(np.maximum(variance, 1e-8)) * np.sqrt(252.0)
        weights, exposure_scale = self._apply_risk_budget_numpy(weights, ann_vol)
        port_returns = weights @ returns_np.T
        if regime_weights is not None:
            weighted_returns = port_returns * regime_weights[None, :]
            denom = max(regime_weights.sum(), 1e-8)
            mean_returns = weighted_returns.sum(axis=1) / denom
            centered = port_returns - mean_returns[:, None]
            variance = (centered**2 * regime_weights[None, :]).sum(axis=1) / denom
        else:
            mean_returns = port_returns.mean(axis=1)
            variance = port_returns.var(axis=1)
        ann_return = mean_returns * 252.0
        ann_vol = np.sqrt(np.maximum(variance, 1e-8)) * np.sqrt(252.0)

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

        score = (
            self.config.expected_return_weight * ann_return
            - self.config.volatility_weight * ann_vol
            - self.config.turnover_weight * turnover
            - self.config.concentration_weight * concentration
            + self.config.diversification_weight * diversification
            - self.config.family_penalty_weight * family_penalty
            + self.config.family_alpha_reward_weight * family_alpha_reward
            - self.config.risk_budget_weight * risk_budget_penalty
            - self.config.sparsity_penalty_weight * sparsity_penalty
            - self.config.under_investment_penalty_weight * under_investment_penalty
        )
        diagnostics = {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "turnover": turnover,
            "concentration": concentration,
            "family_penalty": family_penalty,
            "family_alpha_reward": family_alpha_reward,
            "risk_budget_penalty": risk_budget_penalty,
            "active_count": active_count,
            "gross_exposure": gross_exposure,
            "exposure_scale": exposure_scale,
            "under_investment_penalty": under_investment_penalty,
        }
        return score, diagnostics

    def _optimize_numpy(
        self,
        returns_np: np.ndarray,
        previous_weights: np.ndarray,
        algo_ids: list[str],
        family_matrix: Optional[np.ndarray],
        regime_weights: Optional[np.ndarray],
    ) -> SwarmOptimizationResult:
        n_assets = returns_np.shape[1]
        particles, velocities = self._init_particles(n_assets, previous_weights)
        family_caps = self._build_family_caps(algo_ids)
        family_reward_vector = self._build_family_reward_vector(algo_ids)
        personal_best = particles.copy()
        personal_scores, _ = self._evaluate_numpy(
            particles,
            returns_np,
            previous_weights,
            family_matrix,
            family_caps,
            family_reward_vector,
            regime_weights,
        )
        global_best = personal_best[int(np.argmax(personal_scores))].copy()

        for _ in range(self.config.n_iterations):
            r1 = self._rng.random(size=particles.shape, dtype=np.float32)
            r2 = self._rng.random(size=particles.shape, dtype=np.float32)
            velocities = (
                self.config.inertia * velocities
                + self.config.cognitive_weight * r1 * (personal_best - particles)
                + self.config.social_weight * r2 * (global_best[None, :] - particles)
            )
            particles = particles + velocities
            scores, _ = self._evaluate_numpy(
                particles,
                returns_np,
                previous_weights,
                family_matrix,
                family_caps,
                family_reward_vector,
                regime_weights,
            )
            improved = scores > personal_scores
            personal_best[improved] = particles[improved]
            personal_scores[improved] = scores[improved]
            global_best = personal_best[int(np.argmax(personal_scores))].copy()

        best_weights = self._project_weights_numpy(global_best[None, :])[0]
        final_score, final_diag = self._evaluate_numpy(
            best_weights[None, :],
            returns_np,
            previous_weights,
            family_matrix,
            family_caps,
            family_reward_vector,
            regime_weights,
        )
        diagnostics = {key: float(value[0]) for key, value in final_diag.items()}
        return SwarmOptimizationResult(
            weights=best_weights.astype(np.float32, copy=False),
            score=float(final_score[0]),
            selected_algorithms=algo_ids,
            diagnostics=diagnostics,
            active_count=int(diagnostics.get("active_count", 0.0)),
        )


class SwarmAllocatorBacktester:
    """Backtest wrapper for the swarm meta allocator."""

    def __init__(
        self,
        algo_returns: pd.DataFrame,
        features: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        benchmark_weights: Optional[pd.DataFrame],
        config: SwarmConfig,
        family_labels: Optional[pd.Series] = None,
        family_alpha_scores: Optional[dict[str, float]] = None,
        cluster_history: Optional[pd.DataFrame] = None,
        cluster_stability: Optional[pd.DataFrame] = None,
        cluster_alpha_scores: Optional[dict[str, dict[str, float]]] = None,
        regime_labels: Optional[pd.Series] = None,
        selection_factor: str | list[str] | tuple[str, ...] = "rolling_sharpe_21d",
    ):
        self.algo_returns = algo_returns.sort_index()
        self.features = features
        self.benchmark_returns = benchmark_returns
        self.benchmark_weights = benchmark_weights
        self.config = config
        self.family_labels = family_labels
        self.family_alpha_scores = family_alpha_scores or {}
        self.cluster_history = cluster_history
        self.cluster_stability = cluster_stability
        self.cluster_alpha_scores = cluster_alpha_scores or {}
        self.regime_labels = regime_labels
        self.selection_factors = self._normalize_selection_factors(selection_factor)
        self.selection_factor = (
            self.selection_factors[0] if len(self.selection_factors) == 1 else list(self.selection_factors)
        )
        self._last_selection_diagnostics: dict[str, float] = {}
        self.allocator = SwarmMetaAllocator(
            config=config,
            family_labels=family_labels,
            family_alpha_scores=self.family_alpha_scores,
        )

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> SwarmBacktestResult:
        start_ts = pd.Timestamp(start_date) if start_date is not None else self.algo_returns.index[0]
        end_ts = pd.Timestamp(end_date) if end_date is not None else self.algo_returns.index[-1]

        returns = self.algo_returns.loc[(self.algo_returns.index >= start_ts) & (self.algo_returns.index <= end_ts)]
        rebalance_dates = self._compute_rebalance_schedule(returns.index)
        if not rebalance_dates:
            raise ValueError("No rebalance dates available for Phase 7")

        n_assets = returns.shape[1]
        weights_matrix = []
        rebalance_indices = []
        diagnostics_rows = []
        previous_weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

        has_started = False
        for rebalance_date in rebalance_dates:
            selected_algos = self._select_algorithms(rebalance_date)
            if not selected_algos:
                if has_started:
                    logger.warning("No algorithms selected on %s; keeping previous weights", rebalance_date)
                    weights_matrix.append(previous_weights.copy())
                    rebalance_indices.append(int(returns.index.get_loc(rebalance_date)))
                continue

            hist = self.algo_returns.loc[:rebalance_date, selected_algos].tail(self.config.lookback_window)
            if len(hist) < self.config.min_history:
                continue

            regime_weights = self._compute_regime_weights(hist.index, rebalance_date)
            selected_previous = previous_weights[[self.algo_returns.columns.get_loc(a) for a in selected_algos]]

            optimization = self.allocator.optimize(
                returns_window=hist,
                previous_weights=selected_previous,
                regime_weights=regime_weights,
            )

            full_weights = np.zeros(n_assets, dtype=np.float32)
            selected_indices = [self.algo_returns.columns.get_loc(a) for a in optimization.selected_algorithms]
            full_weights[selected_indices] = optimization.weights
            weights_matrix.append(full_weights)
            rebalance_indices.append(int(returns.index.get_loc(rebalance_date)))
            previous_weights = full_weights
            has_started = True

            diagnostics_rows.append(
                {
                    "date": rebalance_date,
                    "score": optimization.score,
                    "n_candidates": len(selected_algos),
                    "n_active": optimization.active_count,
                    **self._last_selection_diagnostics,
                    **optimization.diagnostics,
                }
            )

        if not weights_matrix:
            raise ValueError("Swarm allocator could not produce any weights")

        active_rebalance_dates = returns.index[np.asarray(rebalance_indices, dtype=np.int64)]
        weights_df = pd.DataFrame(
            weights_matrix,
            index=pd.DatetimeIndex(active_rebalance_dates),
            columns=self.algo_returns.columns,
        )
        portfolio_returns_np = backtest_portfolio_returns(
            returns.fillna(0.0).to_numpy(dtype=np.float64, copy=False),
            weights_df.to_numpy(dtype=np.float64, copy=False),
            np.asarray(rebalance_indices, dtype=np.int64),
        )
        portfolio_returns = pd.Series(portfolio_returns_np, index=returns.index, name="swarm_meta_allocator").fillna(0.0)
        benchmark_returns = None
        if self.benchmark_returns is not None:
            benchmark_returns = self.benchmark_returns.reindex(returns.index).fillna(0.0)

        evaluation_start = weights_df.index.min()
        eval_portfolio_returns = portfolio_returns.loc[evaluation_start:]
        eval_benchmark_returns = benchmark_returns.loc[evaluation_start:] if benchmark_returns is not None else None

        summary = compute_full_metrics(eval_portfolio_returns, eval_benchmark_returns)
        summary["n_rebalances"] = len(weights_df)
        if diagnostics_rows:
            diagnostics_summary_df = pd.DataFrame(diagnostics_rows)
            summary["mean_n_candidates"] = float(diagnostics_summary_df["n_candidates"].mean())
            summary["mean_n_active"] = float(diagnostics_summary_df["n_active"].mean())
        else:
            summary["mean_n_candidates"] = 0.0
            summary["mean_n_active"] = 0.0
        summary["objective_name"] = self.config.objective_name
        summary["device"] = self.allocator.device
        summary["valid_return_observations"] = int(eval_portfolio_returns.notna().sum())
        summary["evaluation_start_date"] = str(evaluation_start.date())
        comparison = self._build_benchmark_comparison(eval_portfolio_returns, eval_benchmark_returns)
        summary.update(comparison)

        diagnostics_df = pd.DataFrame(diagnostics_rows)
        if not diagnostics_df.empty:
            diagnostics_df = diagnostics_df.set_index("date")

        return SwarmBacktestResult(
            weights=weights_df,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            diagnostics=diagnostics_df,
            summary=summary,
            comparison=comparison,
        )

    def _compute_rebalance_schedule(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        schedule = []
        last_rebalance = None
        for date in dates:
            if not self._has_eligible_universe(date):
                continue
            if last_rebalance is None:
                schedule.append(date)
                last_rebalance = date
                continue
            days_since = (date - last_rebalance).days
            if self.config.rebalance_frequency == "daily":
                should_rebalance = True
            elif self.config.rebalance_frequency == "weekly":
                should_rebalance = days_since >= 5
            elif self.config.rebalance_frequency == "monthly":
                should_rebalance = days_since >= 21
            else:
                should_rebalance = days_since >= 5
            if should_rebalance:
                schedule.append(date)
                last_rebalance = date
        return schedule

    def _has_eligible_universe(self, date: pd.Timestamp) -> bool:
        hist = self.algo_returns.loc[:date].tail(self.config.lookback_window)
        if len(hist) < self.config.min_history:
            return False
        valid = hist.notna().sum() >= self.config.min_history
        return bool(valid.any())

    def _select_algorithms(self, date: pd.Timestamp) -> list[str]:
        hist = self.algo_returns.loc[:date].tail(self.config.lookback_window)
        valid = hist.notna().sum() >= self.config.min_history
        candidates = hist.columns[valid].tolist()
        self._last_selection_diagnostics = {
            "n_universe_eligible": float(len(candidates)),
            "n_post_gate": 0.0,
            "n_positive_cluster_signal": 0.0,
            "n_positive_weekly_cluster": 0.0,
            "n_positive_monthly_cluster": 0.0,
            "n_positive_cumulative_cluster": 0.0,
        }
        if not candidates:
            return []

        feature_row = self._latest_feature_row(date)
        cluster_signal_scores = self._compute_cluster_signal_scores(date, candidates)
        composite_scores = self._compute_candidate_scores(
            hist=hist[candidates],
            date=date,
            candidates=candidates,
            cluster_signal_scores=cluster_signal_scores,
        )
        filtered_candidates = self._apply_selection_filters(
            date=date,
            feature_row=feature_row,
            candidates=candidates,
            cluster_signal_scores=cluster_signal_scores,
            composite_scores=composite_scores,
        )
        self._last_selection_diagnostics["n_post_gate"] = float(len(filtered_candidates))
        if cluster_signal_scores is not None and not cluster_signal_scores.empty:
            self._last_selection_diagnostics["n_positive_cluster_signal"] = float(
                (cluster_signal_scores.reindex(candidates).fillna(0.0) > 0.0).sum()
            )
        if not filtered_candidates:
            filtered_candidates = candidates
        if composite_scores:
            ranked = sorted(composite_scores.items(), key=lambda item: item[1], reverse=True)
            filtered_set = set(filtered_candidates)
            ranked = [algo for algo, _ in ranked if algo in filtered_set]
            if ranked:
                return ranked[: self.config.top_k]

        sharpe_proxy = hist[filtered_candidates].mean() / hist[filtered_candidates].std().replace(0, np.nan)
        ranked = sharpe_proxy.sort_values(ascending=False).dropna()
        if not ranked.empty:
            return ranked.index[: self.config.top_k].tolist()
        return filtered_candidates[: self.config.top_k]

    def _apply_selection_filters(
        self,
        date: pd.Timestamp,
        feature_row: Optional[pd.Series],
        candidates: list[str],
        cluster_signal_scores: Optional[pd.Series],
        composite_scores: dict[str, float],
    ) -> list[str]:
        if not candidates:
            return []

        feature_pass_ratio = self._compute_feature_pass_ratios(feature_row, candidates)
        if cluster_signal_scores is not None and not cluster_signal_scores.empty:
            cluster_signal = cluster_signal_scores.reindex(candidates).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        else:
            cluster_signal = np.zeros(len(candidates), dtype=np.float32)
        stability_scores = self._compute_stability_scores(candidates)
        cluster_gate_frame = self._compute_cluster_gate_frame(date, candidates)
        weekly_scores = cluster_gate_frame["weekly_score"].to_numpy(dtype=np.float32, copy=False)
        monthly_scores = cluster_gate_frame["monthly_score"].to_numpy(dtype=np.float32, copy=False)
        cumulative_scores = cluster_gate_frame["cumulative_score"].to_numpy(dtype=np.float32, copy=False)
        self._last_selection_diagnostics["n_positive_weekly_cluster"] = float((weekly_scores > 0.0).sum())
        self._last_selection_diagnostics["n_positive_monthly_cluster"] = float((monthly_scores > 0.0).sum())
        self._last_selection_diagnostics["n_positive_cumulative_cluster"] = float((cumulative_scores > 0.0).sum())

        composite_array = np.asarray(
            [float(composite_scores.get(algo, np.nan)) for algo in candidates],
            dtype=np.float32,
        )
        finite_mask = np.isfinite(composite_array)
        if finite_mask.any():
            fallback_floor = float(np.nanmin(composite_array[finite_mask]))
            composite_array = np.where(finite_mask, composite_array, fallback_floor).astype(np.float32, copy=False)
        else:
            composite_array = np.zeros(len(candidates), dtype=np.float32)

        positive_cluster_mask = cluster_signal > 0.0
        if positive_cluster_mask.any():
            min_cluster_signal = max(
                self.config.cluster_gate_min_signal,
                float(np.nanpercentile(cluster_signal[positive_cluster_mask], 25)),
            )
            min_composite_score = float(
                np.nanquantile(
                    composite_array[positive_cluster_mask],
                    self.config.cluster_gate_score_quantile,
                )
            )
        else:
            min_cluster_signal = -np.inf
            min_composite_score = float(np.nanquantile(composite_array, self.config.cluster_gate_score_quantile))

        positive_stability_mask = stability_scores > 0.0
        if positive_stability_mask.any():
            min_stability = max(
                self.config.cluster_gate_min_stability,
                float(np.nanpercentile(stability_scores[positive_stability_mask], 25)),
            )
        else:
            min_stability = 0.0

        positive_weekly_mask = weekly_scores > 0.0
        positive_monthly_mask = monthly_scores > 0.0
        positive_cumulative_mask = cumulative_scores > 0.0
        min_weekly_score = (
            max(self.config.cluster_gate_min_signal, float(np.nanpercentile(weekly_scores[positive_weekly_mask], 25)))
            if positive_weekly_mask.any()
            else -np.inf
        )
        min_monthly_score = (
            max(self.config.cluster_gate_min_signal, float(np.nanpercentile(monthly_scores[positive_monthly_mask], 25)))
            if positive_monthly_mask.any()
            else -np.inf
        )
        min_cumulative_score = (
            max(0.0, float(np.nanpercentile(cumulative_scores[positive_cumulative_mask], 20)))
            if positive_cumulative_mask.any()
            else -np.inf
        )

        gate_mask = _compute_consensus_gate_mask_numba(
            feature_pass_ratio.astype(np.float32, copy=False),
            weekly_scores.astype(np.float32, copy=False),
            monthly_scores.astype(np.float32, copy=False),
            cumulative_scores.astype(np.float32, copy=False),
            stability_scores.astype(np.float32, copy=False),
            composite_array.astype(np.float32, copy=False),
            float(self.config.min_selection_pass_ratio),
            float(min_weekly_score),
            float(min_monthly_score),
            float(min_cumulative_score),
            float(min_stability),
            float(min_composite_score),
        )
        selected = [algo for algo, keep in zip(candidates, gate_mask.tolist()) if keep]
        if len(selected) >= min(self.config.top_k, self.config.cluster_gate_min_candidates):
            return selected

        relaxed_mask = (
            (feature_pass_ratio >= max(0.50, self.config.min_selection_pass_ratio - 0.25))
            & (stability_scores >= max(0.10, min_stability - 0.10))
            & (composite_array >= float(np.nanmedian(composite_array)))
        )
        if np.isfinite(min_weekly_score):
            relaxed_mask &= weekly_scores >= max(0.0, min_weekly_score - 0.20)
        if np.isfinite(min_monthly_score):
            relaxed_mask &= monthly_scores >= max(0.0, min_monthly_score - 0.20)
        relaxed = [algo for algo, keep in zip(candidates, relaxed_mask.tolist()) if keep]
        if relaxed:
            return relaxed

        ranked = sorted(composite_scores.items(), key=lambda item: item[1], reverse=True)
        if ranked:
            return [algo for algo, _ in ranked[: min(self.config.top_k, self.config.cluster_gate_min_candidates)]]
        return []

    def _compute_feature_pass_ratios(
        self,
        feature_row: Optional[pd.Series],
        candidates: list[str],
    ) -> np.ndarray:
        if feature_row is None or feature_row.empty:
            return np.ones(len(candidates), dtype=np.float32)

        metrics = [
            ("rolling_sharpe_21d", self.config.min_selection_sharpe_21d, "ge"),
            ("rolling_calmar_21d", self.config.min_selection_calmar_21d, "ge"),
            ("rolling_profit_factor_21d", self.config.min_selection_profit_factor_21d, "ge"),
            ("rolling_drawdown_63d", self.config.max_selection_drawdown_63d, "le_abs"),
        ]
        pass_counts = np.zeros(len(candidates), dtype=np.float32)
        available_counts = np.zeros(len(candidates), dtype=np.float32)

        for suffix, threshold, mode in metrics:
            values = pd.to_numeric(
                pd.Series([feature_row.get(f"{algo}_{suffix}", np.nan) for algo in candidates], index=candidates),
                errors="coerce",
            )
            mask = values.notna().to_numpy()
            if not mask.any():
                continue
            raw_values = values.to_numpy(dtype=np.float32, copy=False)
            if mode == "ge":
                passed = raw_values >= float(threshold)
            else:
                passed = np.abs(raw_values) <= float(threshold)
            available_counts += mask.astype(np.float32)
            pass_counts += (passed & mask).astype(np.float32)

        ratios = np.divide(
            pass_counts,
            np.where(available_counts > 0.0, available_counts, 1.0),
        ).astype(np.float32, copy=False)
        ratios[available_counts <= 0.0] = 1.0
        return ratios

    def _compute_stability_scores(self, candidates: list[str]) -> np.ndarray:
        if self.cluster_stability is None or self.cluster_stability.empty:
            return np.ones(len(candidates), dtype=np.float32)
        aligned = self.cluster_stability.reindex(candidates)
        if "stability_ratio" not in aligned.columns:
            return np.ones(len(candidates), dtype=np.float32)
        return aligned["stability_ratio"].fillna(0.0).to_numpy(dtype=np.float32, copy=False)

    def _compute_cluster_gate_frame(self, date: pd.Timestamp, candidates: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(
            0.0,
            index=candidates,
            columns=[
                "cumulative_score",
                "weekly_score",
                "monthly_score",
                "weekly_monthly_agreement",
                "consensus_count",
            ],
            dtype=np.float32,
        )
        if self.cluster_history is None or self.cluster_history.empty or not self.cluster_alpha_scores:
            return frame

        valid_rows = self.cluster_history[self.cluster_history["week_end"] <= date]
        if valid_rows.empty:
            return frame

        latest_week = valid_rows["week_end"].max()
        latest = valid_rows[valid_rows["week_end"] == latest_week].set_index("algo_id").reindex(candidates)
        cumulative_scores = self._map_cluster_scores(
            latest["cluster_cumulative"],
            self.cluster_alpha_scores.get("cluster_cumulative", {}),
        )
        weekly_scores = self._map_cluster_scores(
            latest["cluster_weekly"],
            self.cluster_alpha_scores.get("cluster_weekly", {}),
        )
        monthly_scores = self._map_cluster_scores(
            latest["cluster_monthly"],
            self.cluster_alpha_scores.get("cluster_monthly", {}),
        )

        weekly_cluster = latest["cluster_weekly"].map(_normalize_cluster_key)
        monthly_cluster = latest["cluster_monthly"].map(_normalize_cluster_key)
        agreement = ((weekly_cluster == monthly_cluster) & (weekly_scores > 0.0) & (monthly_scores > 0.0)).astype(np.float32)
        consensus_count = (
            (cumulative_scores > 0.0).astype(np.float32)
            + (weekly_scores > 0.0).astype(np.float32)
            + (monthly_scores > 0.0).astype(np.float32)
        )

        frame["cumulative_score"] = cumulative_scores
        frame["weekly_score"] = weekly_scores
        frame["monthly_score"] = monthly_scores
        frame["weekly_monthly_agreement"] = agreement.to_numpy(dtype=np.float32, copy=False)
        frame["consensus_count"] = consensus_count.astype(np.float32, copy=False)
        return frame

    def _compute_candidate_scores(
        self,
        hist: pd.DataFrame,
        date: pd.Timestamp,
        candidates: list[str],
        cluster_signal_scores: Optional[pd.Series] = None,
    ) -> dict[str, float]:
        if hist.empty:
            return {}

        stats = pd.DataFrame(index=candidates)
        stats["mean_return"] = hist.mean()
        stats["volatility"] = hist.std().replace(0, np.nan)
        stats["sharpe_proxy"] = stats["mean_return"] / stats["volatility"]
        stats["positive_rate"] = (hist > 0).mean()
        stats["downside_mean"] = hist.clip(upper=0.0).abs().mean()
        stats["rolling_calmar_proxy"] = hist.mean() / hist.clip(upper=0.0).abs().mean().replace(0, np.nan)

        feature_row = self._latest_feature_row(date)
        if feature_row is not None:
            stats["feature_score"] = [self._score_feature_row(feature_row, algo) for algo in candidates]
            selected_signal = self._compute_selected_signal(feature_row, candidates)
            stats["selected_signal"] = selected_signal.reindex(candidates)
        else:
            stats["feature_score"] = 0.0
            stats["selected_signal"] = np.nan

        if cluster_signal_scores is not None and not cluster_signal_scores.empty:
            stats["cluster_signal"] = cluster_signal_scores.reindex(candidates)
        else:
            stats["cluster_signal"] = 0.0
        cluster_gate_frame = self._compute_cluster_gate_frame(date, candidates)
        stats["cluster_weekly"] = cluster_gate_frame["weekly_score"]
        stats["cluster_monthly"] = cluster_gate_frame["monthly_score"]
        stats["cluster_cumulative"] = cluster_gate_frame["cumulative_score"]
        stats["cluster_consensus"] = cluster_gate_frame["consensus_count"]
        stats["cluster_agreement"] = cluster_gate_frame["weekly_monthly_agreement"]

        component_weights = {
            "cluster_signal": self.config.cluster_alpha_weight,
            "cluster_weekly": 0.80,
            "cluster_monthly": 0.70,
            "cluster_cumulative": 0.25,
            "cluster_consensus": 0.65,
            "cluster_agreement": 0.55,
            "selected_signal": 1.10,
            "sharpe_proxy": 1.00,
            "feature_score": 0.90,
            "rolling_calmar_proxy": 0.55,
            "positive_rate": 0.25,
            "volatility": -0.55,
            "downside_mean": -0.85,
        }
        score = pd.Series(0.0, index=stats.index, dtype=np.float64)
        for column, weight in component_weights.items():
            score = score + weight * self._cross_sectional_zscore(stats[column])

        score = score.replace([np.inf, -np.inf], np.nan).dropna()
        return {algo: float(value) for algo, value in score.items()}

    def _compute_cluster_signal_scores(self, date: pd.Timestamp, candidates: list[str]) -> pd.Series:
        if (
            self.cluster_history is None
            or self.cluster_history.empty
            or not self.cluster_alpha_scores
        ):
            return pd.Series(0.0, index=candidates, dtype=np.float32)

        valid_rows = self.cluster_history[self.cluster_history["week_end"] <= date]
        if valid_rows.empty:
            return pd.Series(0.0, index=candidates, dtype=np.float32)

        latest_week = valid_rows["week_end"].max()
        latest = valid_rows[valid_rows["week_end"] == latest_week].set_index("algo_id")
        latest = latest.reindex(candidates)

        cumulative_scores = self._map_cluster_scores(latest["cluster_cumulative"], self.cluster_alpha_scores.get("cluster_cumulative", {}))
        weekly_scores = self._map_cluster_scores(latest["cluster_weekly"], self.cluster_alpha_scores.get("cluster_weekly", {}))
        monthly_scores = self._map_cluster_scores(latest["cluster_monthly"], self.cluster_alpha_scores.get("cluster_monthly", {}))

        if self.cluster_stability is not None and not self.cluster_stability.empty:
            stability = self.cluster_stability.reindex(candidates)["stability_ratio"].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        else:
            stability = np.zeros(len(candidates), dtype=np.float32)

        combined = _combine_cluster_scores_numba(
            cumulative_scores,
            weekly_scores,
            monthly_scores,
            stability,
        )
        return pd.Series(combined, index=candidates, dtype=np.float32)

    def _map_cluster_scores(self, series: pd.Series, score_map: dict[str, float]) -> np.ndarray:
        mapped = series.map(_normalize_cluster_key).map(score_map).fillna(0.0)
        return mapped.to_numpy(dtype=np.float32, copy=False)

    def _score_feature_row(self, feature_row: pd.Series, algo: str) -> float:
        preferred_suffixes = {
            "rolling_sharpe_5d": 0.35,
            "rolling_sharpe_21d": 0.90,
            "rolling_sharpe_63d": 1.00,
            "rolling_return_21d": 0.80,
            "rolling_return_63d": 0.90,
            "rolling_profit_factor_21d": 0.55,
            "rolling_profit_factor_63d": 0.65,
            "rolling_calmar_21d": 0.70,
            "rolling_calmar_63d": 0.80,
            "volatility": -0.30,
            "max_drawdown": -0.55,
            "drawdown": -0.45,
        }
        score = 0.0
        found = False
        for suffix, weight in preferred_suffixes.items():
            column = f"{algo}_{suffix}"
            if column not in feature_row.index:
                continue
            value = feature_row.get(column)
            if pd.isna(value):
                continue
            score += weight * float(value)
            found = True
        return score if found else np.nan

    def _normalize_selection_factors(
        self,
        selection_factor: str | list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        if isinstance(selection_factor, str):
            factors = [selection_factor]
        else:
            factors = [str(factor) for factor in selection_factor if str(factor)]
        normalized = tuple(dict.fromkeys(factors))
        return normalized or ("rolling_sharpe_21d",)

    def _selection_factor_direction(self, suffix: str) -> float:
        suffix_lower = suffix.lower()
        if "drawdown" in suffix_lower or "volatility" in suffix_lower:
            return -1.0
        return 1.0

    def _compute_selected_signal(
        self,
        feature_row: pd.Series,
        candidates: list[str],
    ) -> pd.Series:
        components = []
        for suffix in self.selection_factors:
            raw = pd.to_numeric(
                pd.Series([feature_row.get(f"{algo}_{suffix}", np.nan) for algo in candidates], index=candidates),
                errors="coerce",
            )
            if raw.notna().sum() == 0:
                continue
            oriented = raw * self._selection_factor_direction(suffix)
            components.append(self._cross_sectional_zscore(oriented))
        if not components:
            return pd.Series(np.nan, index=candidates, dtype=np.float64)
        return pd.concat(components, axis=1).mean(axis=1, skipna=True).replace([np.inf, -np.inf], np.nan)

    def _cross_sectional_zscore(self, series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() <= 1:
            return pd.Series(0.0, index=series.index)
        centered = series - series.mean()
        scale = series.std(ddof=0)
        if pd.isna(scale) or scale < 1e-8:
            return pd.Series(0.0, index=series.index)
        return (centered / scale).fillna(0.0)

    def _latest_feature_row(self, date: pd.Timestamp) -> Optional[pd.Series]:
        if self.features is None or self.features.empty:
            return None
        valid_dates = self.features.index[self.features.index <= date]
        if len(valid_dates) == 0:
            return None
        return self.features.loc[valid_dates[-1]]

    def _compute_regime_weights(self, index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[np.ndarray]:
        if self.regime_labels is None or self.regime_labels.empty:
            return None
        valid_dates = self.regime_labels.index[self.regime_labels.index <= date]
        if len(valid_dates) == 0:
            return None
        current_regime = self.regime_labels.loc[valid_dates[-1]]
        aligned = self.regime_labels.reindex(index)
        weights = np.ones(len(index), dtype=np.float32)
        weights[np.where(aligned.values == current_regime)[0]] = self.config.regime_focus
        return weights / np.maximum(weights.sum(), 1e-8)

    def _build_benchmark_comparison(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
    ) -> dict:
        comparison = {
            "portfolio_total_return": float((1.0 + portfolio_returns).prod() - 1.0),
        }
        if benchmark_returns is None:
            return comparison

        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").fillna(0.0)
        aligned.columns = ["portfolio", "benchmark"]
        benchmark_metrics = compute_full_metrics(aligned["benchmark"])
        portfolio_total = float((1.0 + aligned["portfolio"]).prod() - 1.0)
        benchmark_total = float((1.0 + aligned["benchmark"]).prod() - 1.0)

        comparison.update(
            {
                "benchmark_total_return": benchmark_total,
                "benchmark_annualized_volatility": float(benchmark_metrics.get("annualized_volatility", 0.0)),
                "benchmark_annualized_return": float(benchmark_metrics.get("annualized_return", 0.0)),
                "benchmark_sharpe_ratio": float(benchmark_metrics.get("sharpe_ratio", 0.0)),
                "benchmark_max_drawdown": float(benchmark_metrics.get("max_drawdown", 0.0)),
            }
        )
        portfolio_metrics = compute_full_metrics(aligned["portfolio"])
        volatility_gap = float(
            portfolio_metrics.get("annualized_volatility", 0.0)
            - benchmark_metrics.get("annualized_volatility", 0.0)
        )
        drawdown_gap = float(
            abs(portfolio_metrics.get("max_drawdown", 0.0))
            - abs(benchmark_metrics.get("max_drawdown", 0.0))
        )
        comparison.update(
            {
                "volatility_gap": volatility_gap,
                "drawdown_gap": drawdown_gap,
                "abs_volatility_gap": abs(volatility_gap),
                "abs_drawdown_gap": abs(drawdown_gap),
                "risk_alignment_score": float(1.0 / (1.0 + abs(volatility_gap) + abs(drawdown_gap))),
            }
        )
        return comparison
