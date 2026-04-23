from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn

from htm_code_native.data.types import ExactRecentMemoryState, ExactRecentReadResult, ExactRecentSlot


class ExactRecentMemoryAdapter(Protocol):
    def reset(self) -> None:
        ...

    def write(
        self,
        step_state: torch.Tensor,
        token_id: int,
        span: tuple[int, int],
        payload: bytes,
        timestamp: int,
    ) -> bool:
        ...

    def read(self, hidden_state: torch.Tensor) -> ExactRecentReadResult:
        ...


class ExactRecentMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        window_size: int,
        vocab_size: int,
        max_byte_payload: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.max_byte_payload = max_byte_payload
        self.write_projection = nn.Linear(hidden_size, key_dim)
        self.query_projection = nn.Linear(hidden_size, key_dim)
        self.reset()

    def init_state(self) -> ExactRecentMemoryState:
        return ExactRecentMemoryState(
            slots=[None] * self.window_size,
            write_pointer=0,
            filled=0,
            total_writes=0,
            total_overwrites=0,
        )

    def reset(self) -> None:
        self.load_state(self.init_state())

    def load_state(self, state: ExactRecentMemoryState) -> None:
        self.slots = [
            None if slot is None else self._clone_slot(slot)
            for slot in state.slots[: self.window_size]
        ]
        if len(self.slots) < self.window_size:
            self.slots.extend([None] * (self.window_size - len(self.slots)))
        self.write_pointer = state.write_pointer % self.window_size
        self.filled = min(state.filled, self.window_size)
        self.total_writes = state.total_writes
        self.total_overwrites = state.total_overwrites

    def export_state(self) -> ExactRecentMemoryState:
        return ExactRecentMemoryState(
            slots=[None if slot is None else self._clone_slot(slot) for slot in self.slots],
            write_pointer=self.write_pointer,
            filled=self.filled,
            total_writes=self.total_writes,
            total_overwrites=self.total_overwrites,
        )

    def write(
        self,
        step_state: torch.Tensor,
        token_id: int,
        span: tuple[int, int],
        payload: bytes,
        timestamp: int,
    ) -> bool:
        key = self.write_projection(step_state)
        overwrite = self.slots[self.write_pointer] is not None
        self.slots[self.write_pointer] = ExactRecentSlot(
            token_id=token_id,
            start_byte=span[0],
            end_byte=span[1],
            byte_payload=payload[: self.max_byte_payload],
            key=key,
            timestamp=timestamp,
        )
        self.write_pointer = (self.write_pointer + 1) % self.window_size
        self.filled = min(self.filled + 1, self.window_size)
        self.total_writes += 1
        if overwrite:
            self.total_overwrites += 1
        return overwrite

    def read(self, hidden_state: torch.Tensor) -> ExactRecentReadResult:
        device = hidden_state.device
        distribution = torch.zeros(self.vocab_size, device=device)
        log_distribution = torch.full((self.vocab_size,), fill_value=math.log(1e-8), device=device)
        attention = torch.zeros(self.window_size, device=device)
        slot_token_ids = torch.full((self.window_size,), fill_value=-1, dtype=torch.long, device=device)

        ordered_slots = self._ordered_slots()
        if not ordered_slots:
            return ExactRecentReadResult(
                distribution=distribution,
                log_distribution=log_distribution,
                attention=attention,
                slot_token_ids=slot_token_ids,
                filled_size=0,
                read_count=0,
                write_count=self.total_writes,
                overwrite_count=self.total_overwrites,
            )

        query = self.query_projection(hidden_state)
        keys = torch.stack([slot.key.to(device) for slot in ordered_slots], dim=0)
        token_ids = torch.tensor([slot.token_id for slot in ordered_slots], dtype=torch.long, device=device)
        scores = (keys @ query) / math.sqrt(self.key_dim)
        weights = torch.softmax(scores, dim=0)
        distribution.index_add_(0, token_ids, weights)
        log_distribution = torch.log(distribution.clamp_min(1e-8))
        attention[: len(ordered_slots)] = weights
        slot_token_ids[: len(ordered_slots)] = token_ids

        return ExactRecentReadResult(
            distribution=distribution,
            log_distribution=log_distribution,
            attention=attention,
            slot_token_ids=slot_token_ids,
            filled_size=len(ordered_slots),
            read_count=len(ordered_slots),
            write_count=self.total_writes,
            overwrite_count=self.total_overwrites,
        )

    def _ordered_slots(self) -> list[ExactRecentSlot]:
        if self.filled == 0:
            return []
        if self.filled < self.window_size:
            return [slot for slot in self.slots[: self.filled] if slot is not None]
        return [
            slot
            for slot in [*self.slots[self.write_pointer :], *self.slots[: self.write_pointer]]
            if slot is not None
        ]

    def _clone_slot(self, slot: ExactRecentSlot) -> ExactRecentSlot:
        return ExactRecentSlot(
            token_id=slot.token_id,
            start_byte=slot.start_byte,
            end_byte=slot.end_byte,
            byte_payload=slot.byte_payload,
            key=slot.key.detach().clone(),
            timestamp=slot.timestamp,
        )


class NoOpExactRecentMemory:
    def reset(self) -> None:
        return None

    def write(
        self,
        step_state: torch.Tensor,
        token_id: int,
        span: tuple[int, int],
        payload: bytes,
        timestamp: int,
    ) -> bool:
        return False

    def read(self, hidden_state: torch.Tensor) -> ExactRecentReadResult:
        device = hidden_state.device
        return ExactRecentReadResult(
            distribution=torch.zeros(1, device=device),
            log_distribution=torch.zeros(1, device=device),
            attention=torch.zeros(1, device=device),
            slot_token_ids=torch.full((1,), fill_value=-1, dtype=torch.long, device=device),
            filled_size=0,
            read_count=0,
            write_count=0,
            overwrite_count=0,
        )
