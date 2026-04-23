from __future__ import annotations

import math

import torch
from torch import nn

from htm_code_native.config.settings import HSSMConfig, SemanticMemoryConfig
from htm_code_native.data.types import ColdCluster, HSSMState, SemanticReadResult, SemanticSlot


class SemanticMemory(nn.Module):
    def __init__(self, hidden_size: int, hssm_config: HSSMConfig, config: SemanticMemoryConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hssm_config = hssm_config
        self.config = config
        self.master_dim = hidden_size * hssm_config.num_levels
        self.query_projections = nn.ModuleList(
            [nn.Linear(self.master_dim, config.key_dim) for _ in range(hssm_config.num_levels)]
        )
        self.key_projections = nn.ModuleList(
            [nn.Linear(hidden_size, config.key_dim) for _ in range(hssm_config.num_levels)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hssm_config.num_levels)]
        )
        self.output_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hssm_config.num_levels)]
        )
        self.reset()

    def reset(self) -> None:
        self.hot_slots: dict[int, list[SemanticSlot]] = {
            level: [] for level in range(self.hssm_config.num_levels)
        }
        self.cold_clusters: dict[int, list[ColdCluster]] = {
            level: [] for level in range(self.hssm_config.num_levels)
        }

    def read_write(self, step_state: HSSMState, budget: float) -> SemanticReadResult:
        device = step_state.master_state.device
        per_level_outputs: list[torch.Tensor] = []
        entropies: dict[int, float] = {}
        hot_reads = 0
        cold_reads = 0
        maintenance_invocations = 0

        for level, level_state in enumerate(step_state.level_states):
            query = self.query_projections[level](step_state.master_state)
            hot_output, hot_entropy, hot_count = self._read_hot(level, query, device)
            cold_output, cold_entropy, cold_count = self._read_cold(level, query, device)
            combined = self.output_projections[level](hot_output + cold_output)
            per_level_outputs.append(combined)
            entropies[level] = hot_entropy + cold_entropy
            hot_reads += hot_count
            cold_reads += cold_count

            self._write_hot(level, level_state.detach(), step_state.step_index)
            if self._should_consolidate(level, budget):
                self._consolidate(level, step_state.step_index)
                maintenance_invocations += 1

        return SemanticReadResult(
            per_level_outputs=per_level_outputs,
            entropies=entropies,
            maintenance_invocations=maintenance_invocations,
            hot_reads=hot_reads,
            cold_reads=cold_reads,
        )

    def _read_hot(
        self,
        level: int,
        query: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, int]:
        slots = self.hot_slots[level]
        if not slots:
            return torch.zeros(self.hidden_size, device=device), 0.0, 0

        keys = torch.stack([slot.key.to(device) for slot in slots], dim=0)
        values = torch.stack([slot.value.to(device) for slot in slots], dim=0)
        scores = (keys @ query) / math.sqrt(keys.shape[-1])
        weights = torch.softmax(scores, dim=0)
        for slot, weight in zip(slots, weights.tolist(), strict=False):
            slot.access_score += float(weight)
        entropy = float((-weights * torch.log(weights.clamp_min(1e-8))).sum().item())
        return weights @ values, entropy, len(slots)

    def _read_cold(
        self,
        level: int,
        query: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, int]:
        clusters = self.cold_clusters[level]
        if not clusters:
            return torch.zeros(self.hidden_size, device=device), 0.0, 0

        centroids = torch.stack([cluster.centroid.to(device) for cluster in clusters], dim=0)
        values = torch.stack([cluster.value.to(device) for cluster in clusters], dim=0)
        query_norm = torch.linalg.norm(query).clamp_min(1e-6)
        centroid_norm = torch.linalg.norm(centroids, dim=-1).clamp_min(1e-6)
        scores = (centroids @ query) / (query_norm * centroid_norm)
        topk = min(self.config.beam_width, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k=topk)
        selected_values = values[top_indices]
        weights = torch.softmax(top_scores, dim=0)
        entropy = float((-weights * torch.log(weights.clamp_min(1e-8))).sum().item())
        return weights @ selected_values, entropy, topk

    def _write_hot(self, level: int, state: torch.Tensor, timestamp: int) -> None:
        key = self.key_projections[level](state)
        value = self.value_projections[level](state)
        slots = self.hot_slots[level]
        slots.append(
            SemanticSlot(
                level=level,
                key=key,
                value=value,
                access_score=0.0,
                timestamp=timestamp,
            )
        )
        if len(slots) > self.config.hot_slots:
            evict_index = min(
                range(len(slots)),
                key=lambda idx: (slots[idx].access_score, slots[idx].timestamp),
            )
            del slots[evict_index]

    def _should_consolidate(self, level: int, budget: float) -> bool:
        slots = self.hot_slots[level]
        fill_ratio = len(slots) / max(self.config.hot_slots, 1)
        return (
            fill_ratio >= self.config.consolidation_fill_threshold
            and budget >= self.config.maintenance_budget
            and len(slots) >= self.config.min_slots_for_consolidation
        )

    def _consolidate(self, level: int, timestamp: int) -> None:
        slots = sorted(self.hot_slots[level], key=lambda slot: (slot.access_score, slot.timestamp))
        group = slots[: self.config.min_slots_for_consolidation]
        if not group:
            return

        weights = torch.tensor([slot.access_score + 1.0 for slot in group], dtype=torch.float32)
        keys = torch.stack([slot.key for slot in group], dim=0)
        values = torch.stack([slot.value for slot in group], dim=0)
        centroid = (weights.unsqueeze(-1) * keys).sum(dim=0) / weights.sum()
        aggregate_value = (weights.unsqueeze(-1) * values).sum(dim=0) / weights.sum()
        clusters = self.cold_clusters[level]
        clusters.append(
            ColdCluster(
                level=level,
                centroid=centroid.detach(),
                value=aggregate_value.detach(),
                member_count=len(group),
                last_updated=timestamp,
            )
        )
        if len(clusters) > self.config.cold_slots:
            clusters.pop(0)

        group_ids = {id(slot) for slot in group}
        self.hot_slots[level] = [slot for slot in self.hot_slots[level] if id(slot) not in group_ids]
