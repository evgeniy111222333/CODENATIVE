from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn

from htm_code_native.data.types import RouterDecision, RouterFeatures


PRE_LANE_NAMES = ("lm", "semantic_hot", "erm", "semantic_cold", "eem", "graph")
POST_LANE_NAMES = ("lm", "semantic", "erm", "eem", "graph")
EXPENSIVE_LANE_NAMES = ("semantic_cold", "eem", "graph")


class RouterAdapter(Protocol):
    def reset(self) -> None:
        ...

    def route(self, features: RouterFeatures) -> RouterDecision:
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
    ) -> None:
        super().__init__()
        self.temperature = max(temperature, 1e-6)
        self.route_top_k = max(route_top_k, 0)
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)
        self.lane_costs = torch.tensor(lane_costs, dtype=torch.float32)
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

    def reset(self) -> None:
        return None

    def route(self, features: RouterFeatures) -> RouterDecision:
        device = features.pre_features.device
        thresholds = self.thresholds.to(device)
        lane_costs = self.lane_costs.to(device)

        pre_logits = self.pre_router(features.pre_features)
        expensive_probs = torch.sigmoid(pre_logits / self.temperature)
        availability_mask = features.availability_mask.to(device=device, dtype=torch.bool)
        threshold_mask = expensive_probs >= thresholds
        topk_mask = self._topk_mask(expensive_probs, availability_mask)
        expensive_mask = availability_mask & (threshold_mask | topk_mask)

        pre_mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool, device=device)
        pre_mask[3:] = expensive_mask

        post_logits = self.post_router(features.post_features)
        post_mask = torch.tensor(
            [True, True, True, bool(expensive_mask[1].item()), bool(expensive_mask[2].item())],
            dtype=torch.bool,
            device=device,
        )
        masked_logits = post_logits.clone()
        masked_logits[~post_mask] = -1e9
        weights = torch.softmax(masked_logits / self.temperature, dim=-1)
        weights = torch.where(post_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum().clamp_min(1e-8)

        energy_proxy = (
            pre_mask.to(dtype=torch.float32) * lane_costs
        ).sum()
        always_on_energy = lane_costs[:3].sum()

        return RouterDecision(
            pre_logits=pre_logits,
            expensive_probs=expensive_probs,
            pre_mask=pre_mask,
            post_logits=post_logits,
            weights=weights,
            post_mask=post_mask,
            energy_proxy=energy_proxy,
            always_on_energy=always_on_energy,
        )

    def _topk_mask(self, scores: torch.Tensor, availability_mask: torch.Tensor) -> torch.Tensor:
        if self.route_top_k <= 0 or not bool(availability_mask.any().item()):
            return torch.zeros_like(availability_mask, dtype=torch.bool)
        masked_scores = scores.masked_fill(~availability_mask, -math.inf)
        k = min(self.route_top_k, int(availability_mask.sum().item()))
        values, indices = torch.topk(masked_scores, k=k)
        topk_mask = torch.zeros_like(availability_mask, dtype=torch.bool)
        valid = torch.isfinite(values)
        topk_mask[indices[valid]] = True
        return topk_mask


class NoOpRouter:
    def reset(self) -> None:
        return None

    def route(self, features: RouterFeatures) -> RouterDecision:
        device = features.pre_features.device
        pre_mask = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool, device=device)
        post_mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool, device=device)
        weights = torch.tensor([0.4, 0.2, 0.15, 0.125, 0.125], dtype=torch.float32, device=device)
        return RouterDecision(
            pre_logits=torch.zeros(3, device=device),
            expensive_probs=torch.ones(3, device=device),
            pre_mask=pre_mask,
            post_logits=torch.zeros(5, device=device),
            weights=weights,
            post_mask=post_mask,
            energy_proxy=torch.tensor(0.0, device=device),
            always_on_energy=torch.tensor(0.0, device=device),
        )
