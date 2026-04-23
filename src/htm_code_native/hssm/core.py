from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from htm_code_native.config.settings import HSSMConfig


@dataclass(slots=True)
class HSSMRunOutput:
    level_states: torch.Tensor
    master_states: torch.Tensor
    update_mask: torch.Tensor
    lower_aggregates: torch.Tensor
    last_update_indices: list[int]


class HSSMLevelCell(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lower_norm = nn.LayerNorm(hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)
        self.update_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.reset_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.candidate = nn.Linear(hidden_size * 3, hidden_size)

    def forward(
        self,
        lower_input: torch.Tensor,
        upper_input: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_lower = self.up_proj(self.lower_norm(lower_input))
        projected_upper = self.down_proj(upper_input)
        fused = torch.cat([projected_lower, projected_upper, prev_state], dim=-1)
        update = torch.sigmoid(self.update_gate(fused))
        reset = torch.sigmoid(self.reset_gate(fused))
        candidate_input = torch.cat(
            [projected_lower, projected_upper, reset * prev_state],
            dim=-1,
        )
        candidate = torch.tanh(self.candidate(candidate_input))
        return projected_lower, (1.0 - update) * prev_state + update * candidate


class HSSMCore(nn.Module):
    def __init__(self, config: HSSMConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.cells = nn.ModuleList(
            [HSSMLevelCell(config.hidden_size) for _ in range(config.num_levels)]
        )

    def forward(self, embeddings: torch.Tensor, boundaries: dict[int, torch.Tensor]) -> HSSMRunOutput:
        seq_len, hidden_size = embeddings.shape
        device = embeddings.device
        num_levels = self.config.num_levels

        histories: list[list[torch.Tensor]] = [[] for _ in range(num_levels)]
        prev_states = [torch.zeros(hidden_size, device=device) for _ in range(num_levels)]
        last_update_indices = [-1 for _ in range(num_levels)]
        segment_starts = [0 for _ in range(num_levels)]

        level_states = torch.zeros(seq_len, num_levels, hidden_size, device=device)
        master_states = torch.zeros(seq_len, num_levels * hidden_size, device=device)
        update_mask = torch.zeros(seq_len, num_levels, dtype=torch.bool, device=device)
        lower_aggregates = torch.zeros(seq_len, num_levels, hidden_size, device=device)

        for step in range(seq_len):
            new_states: list[torch.Tensor] = []
            for level in range(num_levels):
                lower_input = (
                    embeddings[step]
                    if level == 0
                    else self._aggregate_lower(
                        histories[level - 1],
                        new_states[level - 1],
                        segment_starts[level],
                        step,
                        self._stride(level),
                    )
                )
                lower_aggregates[step, level] = lower_input
                upper_state = (
                    prev_states[level + 1]
                    if level < num_levels - 1
                    else torch.zeros(hidden_size, device=device)
                )
                _, proposed_state = self.cells[level](lower_input, upper_state, prev_states[level])
                should_update = level == 0 or bool(boundaries[level][step].item()) or (
                    step % self._stride(level) == 0
                )
                if should_update:
                    updated_state = self._project_norm(proposed_state)
                    prev_states[level] = updated_state
                    last_update_indices[level] = step
                    update_mask[step, level] = True
                new_states.append(prev_states[level])
                level_states[step, level] = prev_states[level]

            for level in range(num_levels):
                histories[level].append(new_states[level])
                if bool(boundaries[level][step].item()):
                    segment_starts[level] = step + 1

            master_states[step] = torch.cat(new_states, dim=-1)

        return HSSMRunOutput(
            level_states=level_states,
            master_states=master_states,
            update_mask=update_mask,
            lower_aggregates=lower_aggregates,
            last_update_indices=last_update_indices,
        )

    def _stride(self, level: int) -> int:
        return self.config.stride_base**level

    def _aggregate_lower(
        self,
        history: list[torch.Tensor],
        current_state: torch.Tensor,
        segment_start: int,
        step: int,
        stride: int,
    ) -> torch.Tensor:
        history_with_current = [*history, current_state]
        fallback_start = max(0, step - stride + 1)
        start = min(segment_start, step)
        effective_start = start if start <= step else fallback_start
        effective_start = max(fallback_start, effective_start)
        segment = history_with_current[effective_start : step + 1]
        if not segment:
            return current_state
        return torch.stack(segment, dim=0).mean(dim=0)

    def _project_norm(self, state: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(state, ord=2)
        if norm > self.config.norm_clip:
            return self.config.norm_clip * state / norm
        return state
