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


class TrainingPhase(str, Enum):
    PHASE_A = "phase_a"
    PHASE_B = "phase_b"
    PHASE_C = "phase_c"
    PHASE_D = "phase_d"
    PHASE_E = "phase_e"


class TaskLabel(str, Enum):
    AR = "ar"
    INFILL = "infill"
    RECENT_COPY = "recent_copy"
    EPISODIC_RECALL = "episodic_recall"
    REPO_GRAPH = "repo_graph"
    EDIT_FIX = "edit_fix"


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
    symbol_line: int | None = None
    syntax_node_type: str | None = None


@dataclass(slots=True)
class ParseNode:
    node_id: str
    node_type: str
    start_byte: int
    end_byte: int
    depth: int
    language: str
    is_named: bool
    parent_id: str | None = None
    field_name: str | None = None


@dataclass(slots=True)
class ParseDocument:
    language: str
    parser_backend: str
    root_type: str
    nodes: list[ParseNode]
    error_count: int
    error_messages: tuple[str, ...] = ()


@dataclass(slots=True)
class SyntaxStateFeatures:
    token_index: int
    node_type: str
    parent_type: str | None
    depth: int
    inside_call: bool
    inside_literal: bool
    inside_comment: bool
    block_depth: int
    parser_language: str


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
    parse_document: ParseDocument | None = None
    syntax_features: list[SyntaxStateFeatures] = field(default_factory=list)

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
            "parser_backend": self.parse_document.parser_backend if self.parse_document else None,
            "parse_errors": self.parse_document.error_count if self.parse_document else 0,
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
    task_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HSSMState:
    level_states: list[torch.Tensor]
    last_update_indices: list[int]
    master_state: torch.Tensor
    step_index: int


@dataclass(slots=True)
class HSSMRuntimeState:
    prev_states: list[torch.Tensor]
    last_update_indices: list[int]
    segment_start_indices: list[int]
    history_tails: list[list[torch.Tensor]]


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
class SemanticMemoryState:
    hot_slots: dict[int, list[SemanticSlot]]
    cold_clusters: dict[int, list[ColdCluster]]


@dataclass(slots=True)
class SemanticReadResult:
    per_level_outputs: list[torch.Tensor]
    per_level_hot_outputs: list[torch.Tensor]
    per_level_cold_outputs: list[torch.Tensor]
    entropies: dict[int, float]
    hot_entropies: dict[int, float]
    cold_entropies: dict[int, float]
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
class ExactRecentMemoryState:
    slots: list[ExactRecentSlot | None]
    write_pointer: int
    filled: int
    total_writes: int
    total_overwrites: int


@dataclass(slots=True)
class ExactPayloadCandidate:
    source: str
    token_id: int
    start_byte: int
    end_byte: int
    byte_payload: bytes
    score: float
    slot_index: int | None = None
    chunk_id: int | None = None
    chunk_token_index: int | None = None


@dataclass(slots=True)
class ExactRecentReadResult:
    distribution: torch.Tensor
    log_distribution: torch.Tensor
    attention: torch.Tensor
    slot_token_ids: torch.Tensor
    payload_candidates: tuple[ExactPayloadCandidate, ...]
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
class ExactEpisodicMemoryState:
    chunks: list[EpisodicChunk]
    next_chunk_id: int
    total_chunks_finalized: int


@dataclass(slots=True)
class ExactEpisodicReadResult:
    distribution: torch.Tensor
    log_distribution: torch.Tensor
    chunk_attention: torch.Tensor
    pointer_attention: torch.Tensor
    retrieved_chunk_ids: torch.Tensor
    pointer_token_ids: torch.Tensor
    payload_candidates: tuple[ExactPayloadCandidate, ...]
    retrieved_chunk_count: int
    read_count: int
    chunks_finalized: int
    chunk_overhead: float
    stored_chunks: int


@dataclass(slots=True)
class RepositoryGraphNode:
    node_id: str
    kind: str
    name: str
    file_path: str | None
    copy_terms: tuple[str, ...]
    key: torch.Tensor
    value: torch.Tensor
    heuristic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphNodeFeatures:
    node_id: str
    kind: str
    language: str
    terms: tuple[str, ...]
    scope_path: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    metadata_flags: tuple[str, ...]


@dataclass(slots=True)
class NeuralGraphCache:
    node_ids: tuple[str, ...]
    feature_terms: tuple[tuple[str, ...], ...]
    scope_paths: tuple[tuple[str, ...], ...]
    edge_kinds: tuple[tuple[str, ...], ...]
    metadata_flags: tuple[tuple[str, ...], ...]


@dataclass(slots=True)
class RepositoryGraphEdge:
    source_id: str
    target_id: str
    kind: str
    heuristic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepositoryGraphIndex:
    root_path: str
    nodes: list[RepositoryGraphNode]
    edges: list[RepositoryGraphEdge]
    nodes_by_id: dict[str, RepositoryGraphNode]
    node_ids_by_file: dict[str, tuple[str, ...]]
    import_closure_by_file: dict[str, tuple[str, ...]]
    test_files_by_source: dict[str, tuple[str, ...]]
    diagnostic_files_by_source: dict[str, tuple[str, ...]]

    def to_summary(self) -> dict[str, Any]:
        node_kind_counts: dict[str, int] = {}
        edge_kind_counts: dict[str, int] = {}
        for node in self.nodes:
            node_kind_counts[node.kind] = node_kind_counts.get(node.kind, 0) + 1
        for edge in self.edges:
            edge_kind_counts[edge.kind] = edge_kind_counts.get(edge.kind, 0) + 1
        return {
            "root_path": self.root_path,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_kinds": node_kind_counts,
            "edge_kinds": edge_kind_counts,
        }


@dataclass(slots=True)
class RepoGraphQueryContext:
    file_path: str
    current_symbol_id: str | None
    current_symbol_name: str | None
    scope_path: tuple[str, ...]
    token_value: str
    token_class: str
    probe_kind: str | None = None
    target_symbol_name: str | None = None
    target_token_value: str | None = None
    target_copy_value: str | None = None


@dataclass(slots=True)
class RepoGraphReadResult:
    graph_context: torch.Tensor
    distribution: torch.Tensor
    log_distribution: torch.Tensor
    attention: torch.Tensor
    copy_token_ids: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_node_ids: tuple[str, ...]
    candidate_kinds: tuple[str, ...]
    candidate_names: tuple[str, ...]
    retrieved_count: int
    read_count: int
    candidate_count: int
    copy_supported_count: int
    samefile_hits: int
    import_hits: int
    symbol_hits: int
    test_hits: int
    diagnostic_hits: int
    target_node_id: str | None = None


@dataclass(slots=True)
class RouterFeatures:
    pre_features: torch.Tensor
    post_features: torch.Tensor
    availability_mask: torch.Tensor
    phase: TrainingPhase
    task_label: TaskLabel
    step_index: int
    oracle_availability: torch.Tensor | None = None
    always_on_pre_mask: torch.Tensor | None = None
    allowed_post_mask: torch.Tensor | None = None


@dataclass(slots=True)
class RouterWarmupState:
    step_index: int
    beta: float
    active: bool
    collapse_detected: bool
    recovery_steps_remaining: int


@dataclass(slots=True)
class RouterRuntimeState:
    dominant_mass_history: tuple[float, ...]
    recovery_steps_remaining: int


@dataclass(slots=True)
class RouterDecision:
    pre_logits: torch.Tensor
    expensive_probs: torch.Tensor
    pre_mask: torch.Tensor
    post_logits: torch.Tensor
    weights: torch.Tensor
    post_mask: torch.Tensor
    energy_proxy: torch.Tensor
    always_on_energy: torch.Tensor
    oracle_weights: torch.Tensor
    effective_weights: torch.Tensor
    warmup_beta: float
    warmup_active: bool
    dominant_lane_dropped: bool
    collapse_detected: bool
    router_entropy: float
    dominant_lane_mass: float
    warmup_steps_remaining: int


@dataclass(slots=True)
class PhaseASessionState:
    hssm: HSSMRuntimeState
    semantic_memory: SemanticMemoryState
    exact_recent: ExactRecentMemoryState
    exact_episodic: ExactEpisodicMemoryState
    router: RouterRuntimeState
    stream_token_index: int
    position_offset: int
    current_chunk_start: int
    previous_lane_stats: torch.Tensor


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
    graph_logits: torch.Tensor | None
    graph_attention: torch.Tensor | None
    graph_copy_target_mask: torch.Tensor | None
    graph_copy_target_ids: torch.Tensor | None
    exact_payload_target_mask: torch.Tensor | None
    exact_span_target_mask: torch.Tensor | None
    base_hidden_states: torch.Tensor | None
    graph_contexts: torch.Tensor | None
    router_weights: torch.Tensor | None
    effective_router_weights: torch.Tensor | None
    oracle_router_weights: torch.Tensor | None
    oracle_availability: torch.Tensor | None
    router_pre_mask: torch.Tensor | None
    router_post_mask: torch.Tensor | None
    lane_entropies: torch.Tensor | None
    invoked_lanes: torch.Tensor | None
    energy_proxy: torch.Tensor | None
    warmup_beta: torch.Tensor | None
    collapse_detected: torch.Tensor | None
    phase_name: str | None
    task_label: str | None
    hidden_states: torch.Tensor
    semantic_contexts: torch.Tensor
    diagnostics: dict[str, float]
    memory_stats: dict[str, float]
    auxiliary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskExample:
    file_path: str
    task_label: TaskLabel
    repo_root: str | None = None
    report_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskBatch:
    example: TaskExample
    batch: PhaseABatch
    supervision_mask: torch.Tensor
    infill_span: tuple[int, int] | None = None
    edit_target_span: tuple[int, int] | None = None
    replacement_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaintenanceDecision:
    should_consolidate: bool
    hot_occupancy: float
    ar_ema: float
    ar_delta: float
    cadence_hit: bool
    under_warmup: bool
    reason: str
    maintenance_invocations: int = 0


@dataclass(slots=True)
class TrainingStepResult:
    example: TaskExample
    output: PhaseAOutput
    losses: dict[str, float]
    gradient_norms: dict[str, float]
    maintenance_decision: MaintenanceDecision


@dataclass(slots=True)
class PhaseExitReport:
    phase: str
    probe_set: str
    passed: bool
    metrics: dict[str, float]
    failing_checks: tuple[str, ...] = ()
    example_count: int = 0


@dataclass(slots=True)
class EditRequest:
    file_path: str
    instruction: str
    repo_root: str | None = None
    report_paths: tuple[str, ...] = ()
    target_symbol: str | None = None
    phase: str | None = None
    max_candidates: int = 3


@dataclass(slots=True)
class EditTargetSpan:
    start_byte: int
    end_byte: int
    token_start: int
    token_end: int
    node_type: str | None
    symbol_name: str | None
    score: float
    reasons: tuple[str, ...]
    source_text: str


@dataclass(slots=True)
class PatchCandidate:
    span: EditTargetSpan
    replacement_text: str
    patched_source: str
    diff_preview: str
    valid: bool
    validation_errors: tuple[str, ...]
    score: float
    support_terms: tuple[str, ...] = ()


@dataclass(slots=True)
class PatchApplyResult:
    candidate_index: int
    span: EditTargetSpan
    replacement_text: str
    patched_source_hash: str
    patched_source_length: int
    diff_preview: str
    applied: bool
    valid: bool
    validation_errors: tuple[str, ...]
    syntax_error_count: int


@dataclass(slots=True)
class PatchVerificationSummary:
    candidate_count: int
    apply_success_rate: float
    syntax_valid_rate: float
    best_candidate_apply_valid: bool


@dataclass(slots=True)
class PatchPlan:
    file_path: str
    original_source: str
    patch_candidates: tuple[PatchCandidate, ...]
    best_candidate: PatchCandidate | None
    validation_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EditRunOutput:
    request: EditRequest
    selected_context: dict[str, Any]
    router_summary: dict[str, Any]
    span_candidates: tuple[EditTargetSpan, ...]
    patch_plan: PatchPlan
    diff_preview: str
    apply_results: tuple[PatchApplyResult, ...] = ()
    best_apply_result: PatchApplyResult | None = None
    verification_summary: PatchVerificationSummary | None = None
    validation_summary: dict[str, Any] = field(default_factory=dict)
