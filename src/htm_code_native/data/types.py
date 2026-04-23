from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from htm_code_native.data.vocabulary import VocabularySnapshot


class TokenClass(str, Enum):
    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    OPERATOR = "operator"
    DELIMITER = "delimiter"
    STRING = "string"
    NUMBER = "number"
    NEWLINE = "newline"
    INDENT = "indent"
    DEDENT = "dedent"
    COMMENT = "comment"
    WHITESPACE_CONTROL = "whitespace-control"
    FALLBACK_BYTE_PIECE = "fallback-byte-piece"


@dataclass(slots=True)
class CodeToken:
    index: int
    token_class: TokenClass
    token_type: str
    value: str
    start_byte: int
    end_byte: int
    language: str
    structural_tags: tuple[str, ...] = ()
    line: int = 0
    column: int = 0


@dataclass(slots=True)
class ASTNodeSpan:
    node_id: str
    node_type: str
    start_byte: int
    end_byte: int
    depth: int
    parent_id: str | None = None


@dataclass(slots=True)
class SymbolSpan:
    symbol_id: str
    name: str
    kind: str
    start_byte: int
    end_byte: int
    scope_path: tuple[str, ...]


@dataclass(slots=True)
class TokenStructureInfo:
    token_index: int
    ast_path: tuple[str, ...]
    ast_node_ids: tuple[str, ...]
    symbol_id: str | None
    symbol_name: str | None
    scope_path: tuple[str, ...]
    file_id: str


@dataclass(slots=True)
class AlignedDocument:
    file_path: str
    language: str
    source_text: str
    raw_bytes: bytes
    tokens: list[CodeToken]
    byte_to_token_index: list[int]
    ast_nodes: list[ASTNodeSpan] = field(default_factory=list)
    symbols: list[SymbolSpan] = field(default_factory=list)
    token_structures: list[TokenStructureInfo] = field(default_factory=list)

    def token_bytes(self, token_index: int) -> bytes:
        token = self.tokens[token_index]
        return self.raw_bytes[token.start_byte : token.end_byte]

    def to_summary(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "token_count": len(self.tokens),
            "ast_nodes": len(self.ast_nodes),
            "symbols": len(self.symbols),
        }


@dataclass(slots=True)
class BoundaryEvents:
    level_events: dict[int, list[bool]]

    def mask_for_level(self, level: int) -> list[bool]:
        return self.level_events.get(level, [])


@dataclass(slots=True)
class PhaseABatch:
    token_ids: torch.Tensor
    token_class_ids: torch.Tensor
    language_ids: torch.Tensor
    scope_ids: torch.Tensor
    positions: torch.Tensor
    byte_values: torch.Tensor
    byte_mask: torch.Tensor
    ast_type_ids: torch.Tensor
    ast_depth_ids: torch.Tensor
    ast_mask: torch.Tensor
    symbol_ids: torch.Tensor
    file_ids: torch.Tensor
    token_spans: torch.Tensor
    token_payload_lengths: torch.Tensor
    boundaries: dict[int, torch.Tensor]
    targets: torch.Tensor
    registry_size: int
    vocabulary_snapshot: VocabularySnapshot
    document: AlignedDocument


@dataclass(slots=True)
class HSSMState:
    level_states: list[torch.Tensor]
    last_update_indices: list[int]
    master_state: torch.Tensor
    step_index: int


@dataclass(slots=True)
class SemanticSlot:
    level: int
    key: torch.Tensor
    value: torch.Tensor
    access_score: float
    timestamp: int


@dataclass(slots=True)
class ColdCluster:
    level: int
    centroid: torch.Tensor
    value: torch.Tensor
    member_count: int
    last_updated: int


@dataclass(slots=True)
class SemanticReadResult:
    per_level_outputs: list[torch.Tensor]
    entropies: dict[int, float]
    maintenance_invocations: int
    hot_reads: int
    cold_reads: int


@dataclass(slots=True)
class ExactRecentSlot:
    token_id: int
    start_byte: int
    end_byte: int
    byte_payload: bytes
    key: torch.Tensor
    timestamp: int


@dataclass(slots=True)
class ExactRecentReadResult:
    distribution: torch.Tensor
    log_distribution: torch.Tensor
    attention: torch.Tensor
    slot_token_ids: torch.Tensor
    filled_size: int
    read_count: int
    write_count: int
    overwrite_count: int


@dataclass(slots=True)
class EpisodicChunkMetadata:
    chunk_id: int
    file_id: str
    symbol_id: str | None
    language: str
    chunk_type: str
    line_range: tuple[int, int]
    scope_path: tuple[str, ...]
    timestamp_start: int
    timestamp_end: int
    token_start_index: int
    token_end_index: int


@dataclass(slots=True)
class EpisodicChunk:
    chunk_id: int
    raw_bytes: bytes
    token_ids: tuple[int, ...]
    token_spans: tuple[tuple[int, int], ...]
    key: torch.Tensor
    pointer_keys: torch.Tensor
    metadata: EpisodicChunkMetadata


@dataclass(slots=True)
class ExactEpisodicReadResult:
    distribution: torch.Tensor
    log_distribution: torch.Tensor
    chunk_attention: torch.Tensor
    pointer_attention: torch.Tensor
    retrieved_chunk_ids: torch.Tensor
    pointer_token_ids: torch.Tensor
    retrieved_chunk_count: int
    read_count: int
    chunks_finalized: int
    chunk_overhead: float
    stored_chunks: int


@dataclass(slots=True)
class PhaseAOutput:
    logits: torch.Tensor
    lm_logits: torch.Tensor
    semantic_logits: torch.Tensor | None
    erm_logits: torch.Tensor | None
    erm_attention: torch.Tensor | None
    copy_target_mask: torch.Tensor | None
    eem_logits: torch.Tensor | None
    eem_attention: torch.Tensor | None
    pointer_attention: torch.Tensor | None
    episodic_target_mask: torch.Tensor | None
    hidden_states: torch.Tensor
    semantic_contexts: torch.Tensor
    diagnostics: dict[str, float]
    memory_stats: dict[str, float]
    auxiliary: dict[str, Any] = field(default_factory=dict)
