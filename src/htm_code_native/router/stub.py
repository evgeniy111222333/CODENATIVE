from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn

from htm_code_native.data.types import RouterDecision, RouterFeatures, RouterRuntimeState, TrainingPhase


PRE_LANE_NAMES = ("lm", "semantic_hot", "erm", "semantic_cold", "eem", "graph")
POST_LANE_NAMES = ("lm", "semantic", "erm", "eem", "graph")
EXPENSIVE_LANE_NAMES = ("semantic_cold", "eem", "graph")


class RouterAdapter(Protocol):
    def reset(self) -> None:
        ...

    def route_pre(self, features: RouterFeatures, warmup_active: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def route_post(self, features: RouterFeatures, pre_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def apply_warmup(
        self,
        *,
        post_logits: torch.Tensor,
        learned_weights: torch.Tensor,
        post_mask: torch.Tensor,
        oracle_availability: torch.Tensor,
        phase: TrainingPhase,
        global_step: int,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool, bool, bool, float, float, int]:
        ...


class TwoStageRouter(nn.Module):
    def __init__(
        self,
        pre_feature_dim: int,
        post_feature_dim: int,
        pre_hidden_dim: int,
        post_hidden_dim: int,
        temperature: float,
        route_top_k: int,
        thresholds: tuple[float, float, float],
        lane_costs: tuple[float, float, float, float, float, float],
        warmup_steps: int,
        oracle_sharpness: float,
        oracle_biases: tuple[float, float, float, float, float],
        lane_dropout_prob: float,
        collapse_mass_threshold: float,
        collapse_window: int,
        recovery_steps: int,
    ) -> None:
        super().__init__()
        self.temperature = max(temperature, 1e-6)
        self.route_top_k = max(route_top_k, 0)
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)
        self.lane_costs = torch.tensor(lane_costs, dtype=torch.float32)
        self.warmup_steps = max(warmup_steps, 1)
        self.oracle_sharpness = oracle_sharpness
        self.oracle_biases = torch.tensor(oracle_biases, dtype=torch.float32)
        self.lane_dropout_prob = float(max(lane_dropout_prob, 0.0))
        self.collapse_mass_threshold = float(collapse_mass_threshold)
        self.collapse_window = max(collapse_window, 1)
        self.recovery_steps = max(recovery_steps, 0)
        self._dominant_mass_history: list[float] = []
        self._recovery_steps_remaining = 0

        self.pre_router = nn.Sequential(
            nn.LayerNorm(pre_feature_dim),
            nn.Linear(pre_feature_dim, pre_hidden_dim),
            nn.GELU(),
            nn.Linear(pre_hidden_dim, len(EXPENSIVE_LANE_NAMES)),
        )
        self.post_router = nn.Sequential(
            nn.LayerNorm(post_feature_dim),
            nn.Linear(post_feature_dim, post_hidden_dim),
            nn.GELU(),
            nn.Linear(post_hidden_dim, len(POST_LANE_NAMES)),
        )

    def init_state(self) -> RouterRuntimeState:
        return RouterRuntimeState(
            dominant_mass_history=(),
            recovery_steps_remaining=0,
        )

    def reset(self) -> None:
        self.load_state(self.init_state())

    def load_state(self, state: RouterRuntimeState) -> None:
        self._dominant_mass_history = list(state.dominant_mass_history)
        self._recovery_steps_remaining = state.recovery_steps_remaining

    def export_state(self) -> RouterRuntimeState:
        return RouterRuntimeState(
            dominant_mass_history=tuple(self._dominant_mass_history),
            recovery_steps_remaining=self._recovery_steps_remaining,
        )

    def route(self, features: RouterFeatures) -> RouterDecision:
        pre_logits, expensive_probs, pre_mask, energy_proxy, always_on_energy = self.route_pre(features)
        post_logits, weights, post_mask = self.route_post(features, pre_mask)
        oracle_availability = (
            features.oracle_availability
            if features.oracle_availability is not None
            else torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool, device=features.pre_features.device)
        )
        (
            oracle_weights,
            effective_weights,
            warmup_beta,
            warmup_active,
            dominant_lane_dropped,
            collapse_detected,
            router_entropy,
            dominant_lane_mass,
            warmup_steps_remaining,
        ) = self.apply_warmup(
            post_logits=post_logits,
            learned_weights=weights,
            post_mask=post_mask,
            oracle_availability=oracle_availability,
            phase=features.phase,
            global_step=features.step_index,
            training=self.training,
        )
        return RouterDecision(
            pre_logits=pre_logits,
            expensive_probs=expensive_probs,
            pre_mask=pre_mask,
            post_logits=post_logits,
            weights=weights,
            post_mask=post_mask,
            energy_proxy=energy_proxy,
            always_on_energy=always_on_energy,
            oracle_weights=oracle_weights,
            effective_weights=effective_weights,
            warmup_beta=warmup_beta,
            warmup_active=warmup_active,
            dominant_lane_dropped=dominant_lane_dropped,
            collapse_detected=collapse_detected,
            router_entropy=router_entropy,
            dominant_lane_mass=dominant_lane_mass,
            warmup_steps_remaining=warmup_steps_remaining,
        )

    def route_pre(
        self,
        features: RouterFeatures,
        warmup_active: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = features.pre_features.device
        thresholds = self.thresholds.to(device)
        lane_costs = self.lane_costs.to(device)
        availability_mask = features.availability_mask.to(device=device, dtype=torch.bool)
        always_on_pre_mask = (
            features.always_on_pre_mask.to(device=device, dtype=torch.bool)
            if features.always_on_pre_mask is not None
            else torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool, device=device)
        )

        pre_logits = self.pre_router(features.pre_features)
        expensive_probs = torch.sigmoid(pre_logits / self.temperature)
        threshold_mask = expensive_probs >= thresholds
        top_k = max(self.route_top_k, 2) if warmup_active else self.route_top_k
        topk_mask = self._topk_mask(expensive_probs, availability_mask, top_k)
        expensive_mask = availability_mask & (threshold_mask | topk_mask)

        pre_mask = always_on_pre_mask.clone()
        pre_mask[3:] = pre_mask[3:] | expensive_mask
        energy_proxy = (pre_mask.to(dtype=torch.float32) * lane_costs).sum()
        always_on_energy = (always_on_pre_mask.to(dtype=torch.float32) * lane_costs).sum()
        return pre_logits, expensive_probs, pre_mask, energy_proxy, always_on_energy

    def route_post(
        self,
        features: RouterFeatures,
        pre_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = features.post_features.device
        post_logits = self.post_router(features.post_features)
        allowed_post_mask = (
            features.allowed_post_mask.to(device=device, dtype=torch.bool)
            if features.allowed_post_mask is not None
            else torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool, device=device)
        )
        post_mask = torch.tensor(
            [
                bool(pre_mask[0].item()),
                True,
                bool(pre_mask[2].item()),
                bool(pre_mask[4].item()),
                bool(pre_mask[5].item()),
            ],
            dtype=torch.bool,
            device=device,
        ) & allowed_post_mask
        if not bool(post_mask.any().item()):
            post_mask[1] = True
        masked_logits = post_logits.clone()
        masked_logits[~post_mask] = -1e9
        learned_weights = torch.softmax(masked_logits / self.temperature, dim=-1)
        learned_weights = torch.where(post_mask, learned_weights, torch.zeros_like(learned_weights))
        learned_weights = learned_weights / learned_weights.sum().clamp_min(1e-8)
        return post_logits, learned_weights, post_mask

    def apply_warmup(
        self,
        *,
        post_logits: torch.Tensor,
        learned_weights: torch.Tensor,
        post_mask: torch.Tensor,
        oracle_availability: torch.Tensor,
        phase: TrainingPhase,
        global_step: int,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool, bool, bool, float, float, int]:
        device = learned_weights.device
        warmup_active = training and phase in {
            TrainingPhase.PHASE_B,
            TrainingPhase.PHASE_C,
            TrainingPhase.PHASE_D,
        } and global_step < self.warmup_steps
        beta = min(float(global_step) / float(self.warmup_steps), 1.0) if warmup_active else 1.0

        oracle_mask = oracle_availability.to(device=device, dtype=torch.bool) & post_mask
        if not bool(oracle_mask.any().item()):
            oracle_mask = post_mask.clone()
        oracle_logits = self.oracle_sharpness * oracle_availability.to(device=device, dtype=torch.float32)
        oracle_logits = oracle_logits + self.oracle_biases.to(device)
        oracle_logits = oracle_logits.masked_fill(~oracle_mask, -1e9)
        oracle_weights = torch.softmax(oracle_logits, dim=-1)
        oracle_weights = torch.where(oracle_mask, oracle_weights, torch.zeros_like(oracle_weights))
        oracle_weights = oracle_weights / oracle_weights.sum().clamp_min(1e-8)

        collapse_detected = self._update_collapse_state(learned_weights)
        if collapse_detected and self.recovery_steps > 0:
            self._recovery_steps_remaining = self.recovery_steps
        if self._recovery_steps_remaining > 0:
            beta = min(beta, 0.5)
            self._recovery_steps_remaining -= 1

        if warmup_active:
            effective_weights = (1.0 - beta) * oracle_weights + beta * learned_weights
        else:
            effective_weights = learned_weights
        effective_weights = torch.where(post_mask, effective_weights, torch.zeros_like(effective_weights))
        effective_weights = effective_weights / effective_weights.sum().clamp_min(1e-8)

        dominant_lane_dropped = False
        dropout_prob = self.lane_dropout_prob * (2.0 if collapse_detected else 1.0)
        if warmup_active and dropout_prob > 0.0 and bool((post_mask.sum() > 1).item()):
            if float(torch.rand((), device=device).item()) < dropout_prob:
                dominant_index = int(torch.argmax(effective_weights).item())
                if bool(post_mask[dominant_index].item()) and bool((post_mask.sum() > 1).item()):
                    effective_weights = effective_weights.clone()
                    effective_weights[dominant_index] = 0.0
                    if bool(effective_weights.sum().gt(0).item()):
                        effective_weights = effective_weights / effective_weights.sum().clamp_min(1e-8)
                        dominant_lane_dropped = True

        normalized = learned_weights / learned_weights.sum().clamp_min(1e-8)
        router_entropy = float(
            (-normalized * torch.log(normalized.clamp_min(1e-8))).sum().item()
        )
        dominant_lane_mass = float(normalized.max().item())
        return (
            oracle_weights,
            effective_weights,
            beta,
            warmup_active,
            dominant_lane_dropped,
            collapse_detected,
            router_entropy,
            dominant_lane_mass,
            self._recovery_steps_remaining,
        )

    def _update_collapse_state(self, learned_weights: torch.Tensor) -> bool:
        dominant_mass = float(learned_weights.max().item())
        self._dominant_mass_history.append(dominant_mass)
        if len(self._dominant_mass_history) > self.collapse_window:
            self._dominant_mass_history.pop(0)
        if len(self._dominant_mass_history) < self.collapse_window:
            return False
        mean_mass = sum(self._dominant_mass_history) / len(self._dominant_mass_history)
        return mean_mass >= self.collapse_mass_threshold

    def _topk_mask(
        self,
        scores: torch.Tensor,
        availability_mask: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        if top_k <= 0 or not bool(availability_mask.any().item()):
            return torch.zeros_like(availability_mask, dtype=torch.bool)
        masked_scores = scores.masked_fill(~availability_mask, -math.inf)
        k = min(top_k, int(availability_mask.sum().item()))
        values, indices = torch.topk(masked_scores, k=k)
        topk_mask = torch.zeros_like(availability_mask, dtype=torch.bool)
        valid = torch.isfinite(values)
        topk_mask[indices[valid]] = True
        return topk_mask


class NoOpRouter:
    def reset(self) -> None:
        return None

    def route_pre(self, features: RouterFeatures, warmup_active: bool = False):
        device = features.pre_features.device
        pre_mask = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool, device=device)
        return (
            torch.zeros(3, device=device),
            torch.ones(3, device=device),
            pre_mask,
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        )

    def route_post(self, features: RouterFeatures, pre_mask: torch.Tensor):
        device = features.post_features.device
        post_mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool, device=device)
        weights = torch.tensor([0.4, 0.2, 0.15, 0.125, 0.125], dtype=torch.float32, device=device)
        return torch.zeros(5, device=device), weights, post_mask

    def apply_warmup(
        self,
        *,
        post_logits: torch.Tensor,
        learned_weights: torch.Tensor,
        post_mask: torch.Tensor,
        oracle_availability: torch.Tensor,
        phase: TrainingPhase,
        global_step: int,
        training: bool,
    ):
        normalized = learned_weights / learned_weights.sum().clamp_min(1e-8)
        return (
            normalized,
            normalized,
            1.0,
            False,
            False,
            False,
            float((-normalized * torch.log(normalized.clamp_min(1e-8))).sum().item()),
            float(normalized.max().item()),
            0,
        )

    def route(self, features: RouterFeatures) -> RouterDecision:
        pre_logits, expensive_probs, pre_mask, energy_proxy, always_on_energy = self.route_pre(features)
        post_logits, weights, post_mask = self.route_post(features, pre_mask)
        oracle_weights, effective_weights, warmup_beta, warmup_active, dominant_lane_dropped, collapse_detected, router_entropy, dominant_lane_mass, warmup_steps_remaining = self.apply_warmup(
            post_logits=post_logits,
            learned_weights=weights,
            post_mask=post_mask,
            oracle_availability=torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool, device=weights.device),
            phase=features.phase,
            global_step=features.step_index,
            training=False,
        )
        return RouterDecision(
            pre_logits=pre_logits,
            expensive_probs=expensive_probs,
            pre_mask=pre_mask,
            post_logits=post_logits,
            weights=weights,
            post_mask=post_mask,
            energy_proxy=energy_proxy,
            always_on_energy=always_on_energy,
            oracle_weights=oracle_weights,
            effective_weights=effective_weights,
            warmup_beta=warmup_beta,
            warmup_active=warmup_active,
            dominant_lane_dropped=dominant_lane_dropped,
            collapse_detected=collapse_detected,
            router_entropy=router_entropy,
            dominant_lane_mass=dominant_lane_mass,
            warmup_steps_remaining=warmup_steps_remaining,
        )
