from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn

from htm_code_native.data.types import (
    AlignedDocument,
    EpisodicChunk,
    EpisodicChunkMetadata,
    ExactEpisodicMemoryState,
    ExactEpisodicReadResult,
    PhaseABatch,
)


class ExactEpisodicMemoryAdapter(Protocol):
    def reset(self) -> None:
        ...

    def maybe_finalize_chunk(
        self,
        document: AlignedDocument,
        batch: PhaseABatch,
        level0_states: torch.Tensor,
        start_index: int,
        end_index: int,
        chunk_type: str,
        timestamp: int,
    ) -> EpisodicChunk | None:
        ...

    def retrieve(self, hidden: torch.Tensor) -> ExactEpisodicReadResult:
        ...


class ExactEpisodicMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        pointer_key_dim: int,
        vocab_size: int,
        top_k: int,
        max_chunk_tokens: int,
        max_chunks: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.pointer_key_dim = pointer_key_dim
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.max_chunk_tokens = max_chunk_tokens
        self.max_chunks = max_chunks
        self.chunk_write_projection = nn.Linear(hidden_size, key_dim)
        self.chunk_query_projection = nn.Linear(hidden_size, key_dim)
        self.pointer_write_projection = nn.Linear(hidden_size, pointer_key_dim)
        self.pointer_query_projection = nn.Linear(hidden_size, pointer_key_dim)
        self.reset()

    def init_state(self) -> ExactEpisodicMemoryState:
        return ExactEpisodicMemoryState(
            chunks=[],
            next_chunk_id=0,
            total_chunks_finalized=0,
        )

    def reset(self) -> None:
        self.load_state(self.init_state())

    def load_state(self, state: ExactEpisodicMemoryState) -> None:
        self.chunks = [self._clone_chunk(chunk) for chunk in state.chunks[: self.max_chunks]]
        self.next_chunk_id = state.next_chunk_id
        self.total_chunks_finalized = state.total_chunks_finalized

    def export_state(self) -> ExactEpisodicMemoryState:
        return ExactEpisodicMemoryState(
            chunks=[self._clone_chunk(chunk) for chunk in self.chunks],
            next_chunk_id=self.next_chunk_id,
            total_chunks_finalized=self.total_chunks_finalized,
        )

    def maybe_finalize_chunk(
        self,
        document: AlignedDocument,
        batch: PhaseABatch,
        level0_states: torch.Tensor,
        start_index: int,
        end_index: int,
        chunk_type: str,
        timestamp: int,
    ) -> EpisodicChunk | None:
        if start_index > end_index:
            return None

        token_spans_slice = batch.token_spans[start_index : end_index + 1]
        if token_spans_slice.numel() == 0:
            return None
        raw_start = int(token_spans_slice[0, 0].item())
        raw_end = int(token_spans_slice[-1, 1].item())
        state_slice = level0_states[start_index : end_index + 1].detach()
        token_ids = tuple(int(token_id) for token_id in batch.token_ids[start_index : end_index + 1].tolist())
        token_spans = tuple(
            (int(span[0]), int(span[1])) for span in token_spans_slice.detach().cpu().tolist()
        )
        pooled_state = state_slice.mean(dim=0)
        chunk_key = self.chunk_write_projection(pooled_state).detach()
        pointer_keys = self.pointer_write_projection(state_slice).detach()
        metadata = self._build_metadata(document, batch, start_index, end_index, chunk_type, timestamp)
        chunk = EpisodicChunk(
            chunk_id=self.next_chunk_id,
            raw_bytes=document.raw_bytes[raw_start:raw_end],
            token_ids=token_ids,
            token_spans=token_spans,
            key=chunk_key,
            pointer_keys=pointer_keys,
            metadata=metadata,
        )
        self.next_chunk_id += 1
        self.total_chunks_finalized += 1
        self.chunks.append(chunk)
        if len(self.chunks) > self.max_chunks:
            self.chunks.pop(0)
        return chunk

    def retrieve(self, hidden: torch.Tensor) -> ExactEpisodicReadResult:
        device = hidden.device
        distribution = torch.zeros(self.vocab_size, device=device)
        log_distribution = torch.full((self.vocab_size,), fill_value=math.log(1e-8), device=device)
        chunk_attention = torch.zeros(self.top_k, device=device)
        pointer_attention = torch.zeros(self.top_k * self.max_chunk_tokens, device=device)
        retrieved_chunk_ids = torch.full((self.top_k,), fill_value=-1, dtype=torch.long, device=device)
        pointer_token_ids = torch.full(
            (self.top_k * self.max_chunk_tokens,),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        if not self.chunks:
            return ExactEpisodicReadResult(
                distribution=distribution,
                log_distribution=log_distribution,
                chunk_attention=chunk_attention,
                pointer_attention=pointer_attention,
                retrieved_chunk_ids=retrieved_chunk_ids,
                pointer_token_ids=pointer_token_ids,
                retrieved_chunk_count=0,
                read_count=0,
                chunks_finalized=self.total_chunks_finalized,
                chunk_overhead=0.0,
                stored_chunks=0,
            )

        chunk_query = self.chunk_query_projection(hidden)
        all_chunk_keys = torch.stack([chunk.key.to(device) for chunk in self.chunks], dim=0)
        chunk_scores = (all_chunk_keys @ chunk_query) / math.sqrt(self.key_dim)
        top_k = min(self.top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(chunk_scores, k=top_k)
        top_weights = torch.softmax(top_scores, dim=0)
        selected_chunks = [self.chunks[int(index.item())] for index in top_indices]
        chunk_attention[:top_k] = top_weights
        retrieved_chunk_ids[:top_k] = torch.tensor(
            [chunk.chunk_id for chunk in selected_chunks],
            dtype=torch.long,
            device=device,
        )

        pointer_query = self.pointer_query_projection(hidden)
        pointer_scores_list: list[torch.Tensor] = []
        pointer_token_id_list: list[int] = []
        fill_index = 0
        for chunk_weight, chunk in zip(top_weights, selected_chunks, strict=False):
            keys = chunk.pointer_keys.to(device)
            scores = (keys @ pointer_query) / math.sqrt(self.pointer_key_dim)
            scores = scores + torch.log(chunk_weight.clamp_min(1e-8))
            pointer_scores_list.append(scores)
            pointer_token_id_list.extend(list(chunk.token_ids))
            usable = min(len(chunk.token_ids), self.max_chunk_tokens)
            pointer_token_ids[fill_index : fill_index + usable] = torch.tensor(
                list(chunk.token_ids[:usable]),
                dtype=torch.long,
                device=device,
            )
            fill_index += self.max_chunk_tokens

        flat_scores = torch.cat(pointer_scores_list, dim=0)
        flat_weights = torch.softmax(flat_scores, dim=0)
        distribution.index_add_(
            0,
            torch.tensor(pointer_token_id_list, dtype=torch.long, device=device),
            flat_weights,
        )
        log_distribution = torch.log(distribution.clamp_min(1e-8))
        usable_pointer_count = min(pointer_attention.shape[0], flat_weights.shape[0])
        pointer_attention[:usable_pointer_count] = flat_weights[:usable_pointer_count]

        return ExactEpisodicReadResult(
            distribution=distribution,
            log_distribution=log_distribution,
            chunk_attention=chunk_attention,
            pointer_attention=pointer_attention,
            retrieved_chunk_ids=retrieved_chunk_ids,
            pointer_token_ids=pointer_token_ids,
            retrieved_chunk_count=top_k,
            read_count=int(flat_weights.shape[0]),
            chunks_finalized=self.total_chunks_finalized,
            chunk_overhead=float(flat_weights.shape[0]),
            stored_chunks=len(self.chunks),
        )

    def _build_metadata(
        self,
        document: AlignedDocument,
        batch: PhaseABatch,
        start_index: int,
        end_index: int,
        chunk_type: str,
        timestamp: int,
    ) -> EpisodicChunkMetadata:
        symbol_id: str | None = None
        scope_path: tuple[str, ...] = ()
        for info in document.token_structures[start_index : end_index + 1]:
            if info.symbol_id is not None:
                symbol_id = info.symbol_id
            if info.scope_path:
                scope_path = info.scope_path
        return EpisodicChunkMetadata(
            chunk_id=self.next_chunk_id,
            file_id=document.file_path,
            symbol_id=symbol_id,
            language=document.language,
            chunk_type=chunk_type,
            line_range=(document.tokens[start_index].line, document.tokens[end_index].line),
            scope_path=scope_path,
            timestamp_start=start_index,
            timestamp_end=timestamp,
            token_start_index=start_index,
            token_end_index=end_index,
        )

    def _clone_chunk(self, chunk: EpisodicChunk) -> EpisodicChunk:
        metadata = EpisodicChunkMetadata(
            chunk_id=chunk.metadata.chunk_id,
            file_id=chunk.metadata.file_id,
            symbol_id=chunk.metadata.symbol_id,
            language=chunk.metadata.language,
            chunk_type=chunk.metadata.chunk_type,
            line_range=chunk.metadata.line_range,
            scope_path=chunk.metadata.scope_path,
            timestamp_start=chunk.metadata.timestamp_start,
            timestamp_end=chunk.metadata.timestamp_end,
            token_start_index=chunk.metadata.token_start_index,
            token_end_index=chunk.metadata.token_end_index,
        )
        return EpisodicChunk(
            chunk_id=chunk.chunk_id,
            raw_bytes=chunk.raw_bytes,
            token_ids=chunk.token_ids,
            token_spans=chunk.token_spans,
            key=chunk.key.detach().clone(),
            pointer_keys=chunk.pointer_keys.detach().clone(),
            metadata=metadata,
        )


class NoOpExactEpisodicMemory:
    def reset(self) -> None:
        return None

    def maybe_finalize_chunk(
        self,
        document: AlignedDocument,
        batch: PhaseABatch,
        level0_states: torch.Tensor,
        start_index: int,
        end_index: int,
        chunk_type: str,
        timestamp: int,
    ) -> EpisodicChunk | None:
        return None

    def retrieve(self, hidden: torch.Tensor) -> ExactEpisodicReadResult:
        device = hidden.device
        return ExactEpisodicReadResult(
            distribution=torch.zeros(1, device=device),
            log_distribution=torch.zeros(1, device=device),
            chunk_attention=torch.zeros(1, device=device),
            pointer_attention=torch.zeros(1, device=device),
            retrieved_chunk_ids=torch.full((1,), fill_value=-1, dtype=torch.long, device=device),
            pointer_token_ids=torch.full((1,), fill_value=-1, dtype=torch.long, device=device),
            retrieved_chunk_count=0,
            read_count=0,
            chunks_finalized=0,
            chunk_overhead=0.0,
            stored_chunks=0,
        )
