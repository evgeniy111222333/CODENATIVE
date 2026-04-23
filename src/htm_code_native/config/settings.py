from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class HSSMConfig:
    max_level: int = 5
    hidden_size: int = 128
    stride_base: int = 2
    norm_clip: float = 10.0

    @property
    def num_levels(self) -> int:
        return self.max_level + 1


@dataclass(slots=True)
class SemanticMemoryConfig:
    key_dim: int = 128
    hot_slots: int = 32
    cold_slots: int = 128
    beam_width: int = 8
    consolidation_fill_threshold: float = 0.75
    maintenance_budget: float = 1.0
    min_slots_for_consolidation: int = 8


@dataclass(slots=True)
class PhaseAModelConfig:
    vocabulary_size: int = 4096
    model_dim: int = 128
    byte_embedding_dim: int = 32
    max_position_embeddings: int = 4096
    max_byte_window: int = 32
    max_ast_depth: int = 16
    semantic_blend: float = 0.2
    use_semantic_head: bool = True
    recent_window: int = 128
    erm_key_dim: int = 128
    erm_blend: float = 0.15
    copy_recent_weight: float = 0.2
    max_recent_byte_payload: int = 32
    eem_key_dim: int = 128
    pointer_key_dim: int = 128
    eem_blend: float = 0.15
    copy_episodic_weight: float = 0.2
    min_chunk_tokens: int = 16
    max_chunk_tokens: int = 128
    eem_top_k: int = 2
    max_episodic_chunks: int = 64
    graph_key_dim: int = 128
    graph_top_k: int = 4
    graph_blend: float = 0.15
    graph_copy_weight: float = 0.5
    graph_samefile_bias: float = 2.0
    graph_import_bias: float = 1.5
    graph_symbol_bias: float = 1.25
    graph_test_bias: float = 1.0
    graph_diagnostic_bias: float = 1.0
    repo_max_files: int = 256
    parser_backend: str = "tree-sitter"
    supported_languages: list[str] = field(
        default_factory=lambda: ["python", "javascript", "typescript", "json", "yaml", "toml", "ini"]
    )
    max_parse_errors: int = 32
    graph_value_dim: int = 128
    graph_out_blend: float = 1.0
    symbol_link_weight: float = 0.1
    pre_router_hidden_dim: int = 128
    post_router_hidden_dim: int = 128
    route_temperature: float = 1.0
    route_top_k: int = 2
    route_threshold_cold_semantic: float = 0.35
    route_threshold_eem: float = 0.35
    route_threshold_graph: float = 0.35
    route_weight: float = 0.1
    energy_weight: float = 0.01
    route_consistency_weight: float = 0.02
    training_phase: str = "phase_d"
    router_warmup_steps: int = 256
    router_oracle_sharpness: float = 4.0
    router_oracle_bias_lm: float = 0.0
    router_oracle_bias_semantic: float = 0.0
    router_oracle_bias_erm: float = 0.0
    router_oracle_bias_eem: float = 0.0
    router_oracle_bias_graph: float = 0.0
    router_lane_dropout_prob: float = 0.05
    router_entropy_floor_min: float = 1.10
    router_entropy_floor_weight: float = 0.02
    router_collapse_mass_threshold: float = 0.95
    router_collapse_window: int = 32
    router_recovery_steps: int = 64
    lane_cost_lm: float = 1.0
    lane_cost_semantic_hot: float = 1.0
    lane_cost_erm: float = 1.0
    lane_cost_semantic_cold: float = 2.0
    lane_cost_eem: float = 2.5
    lane_cost_graph: float = 2.5
    maintenance_cost: float = 0.5
    optimizer_base_lr: float = 1e-3
    auxiliary_cap_ratio: float = 0.5
    maintenance_cadence: int = 16
    maintenance_ema_decay: float = 0.9
    maintenance_loss_spike_delta: float = 0.05
    semantic_session_chunk_size: int = 64
    semantic_maintenance_warmup_steps: int = 0
    probe_min_tokens_per_sec: float = 5.0
    probe_max_energy_proxy: float = 20.0
    probe_min_cold_read_rate: float = 0.01
    probe_min_recent_copy_hit_rate: float = 0.01
    probe_min_episodic_hit_rate: float = 0.01
    probe_min_symbol_link_hit_rate: float = 0.01
    probe_min_graph_copy_hit_rate: float = 0.01
    probe_min_route_entropy: float = 0.5
    probe_min_patch_candidate_valid_rate: float = 0.25
    probe_min_best_patch_hit_rate: float = 0.25
    probe_min_diagnostic_to_span_recall: float = 0.25
    probe_min_patch_apply_success_rate: float = 0.25
    probe_min_patch_syntax_valid_rate: float = 0.25
    edit_span_weight: float = 0.1
    edit_patch_weight: float = 0.2
    diagnostic_alignment_weight: float = 0.1
    edit_max_candidates: int = 3


@dataclass(slots=True)
class HTMCodeNativeConfig:
    model: PhaseAModelConfig
    hssm: HSSMConfig
    semantic_memory: SemanticMemoryConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HTMCodeNativeConfig":
        return cls(
            model=PhaseAModelConfig(**payload.get("model", {})),
            hssm=HSSMConfig(**payload.get("hssm", {})),
            semantic_memory=SemanticMemoryConfig(**payload.get("semantic_memory", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HTMCodeNativeConfig":
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(loaded)

    @classmethod
    def default(cls) -> "HTMCodeNativeConfig":
        return cls(
            model=PhaseAModelConfig(),
            hssm=HSSMConfig(),
            semantic_memory=SemanticMemoryConfig(),
        )

    @property
    def recent_window(self) -> int:
        return self.model.recent_window

    @property
    def erm_key_dim(self) -> int:
        return self.model.erm_key_dim

    @property
    def erm_blend(self) -> float:
        return self.model.erm_blend

    @property
    def copy_recent_weight(self) -> float:
        return self.model.copy_recent_weight

    @property
    def max_recent_byte_payload(self) -> int:
        return self.model.max_recent_byte_payload

    @property
    def eem_key_dim(self) -> int:
        return self.model.eem_key_dim

    @property
    def pointer_key_dim(self) -> int:
        return self.model.pointer_key_dim

    @property
    def eem_blend(self) -> float:
        return self.model.eem_blend

    @property
    def copy_episodic_weight(self) -> float:
        return self.model.copy_episodic_weight

    @property
    def min_chunk_tokens(self) -> int:
        return self.model.min_chunk_tokens

    @property
    def max_chunk_tokens(self) -> int:
        return self.model.max_chunk_tokens

    @property
    def eem_top_k(self) -> int:
        return self.model.eem_top_k

    @property
    def max_episodic_chunks(self) -> int:
        return self.model.max_episodic_chunks

    @property
    def graph_key_dim(self) -> int:
        return self.model.graph_key_dim

    @property
    def graph_top_k(self) -> int:
        return self.model.graph_top_k

    @property
    def graph_blend(self) -> float:
        return self.model.graph_blend

    @property
    def graph_copy_weight(self) -> float:
        return self.model.graph_copy_weight

    @property
    def graph_samefile_bias(self) -> float:
        return self.model.graph_samefile_bias

    @property
    def graph_import_bias(self) -> float:
        return self.model.graph_import_bias

    @property
    def graph_symbol_bias(self) -> float:
        return self.model.graph_symbol_bias

    @property
    def graph_test_bias(self) -> int | float:
        return self.model.graph_test_bias

    @property
    def graph_diagnostic_bias(self) -> int | float:
        return self.model.graph_diagnostic_bias

    @property
    def repo_max_files(self) -> int:
        return self.model.repo_max_files

    @property
    def parser_backend(self) -> str:
        return self.model.parser_backend

    @property
    def supported_languages(self) -> list[str]:
        return self.model.supported_languages

    @property
    def max_parse_errors(self) -> int:
        return self.model.max_parse_errors

    @property
    def graph_value_dim(self) -> int:
        return self.model.graph_value_dim

    @property
    def graph_out_blend(self) -> float:
        return self.model.graph_out_blend

    @property
    def symbol_link_weight(self) -> float:
        return self.model.symbol_link_weight

    @property
    def pre_router_hidden_dim(self) -> int:
        return self.model.pre_router_hidden_dim

    @property
    def post_router_hidden_dim(self) -> int:
        return self.model.post_router_hidden_dim

    @property
    def route_temperature(self) -> float:
        return self.model.route_temperature

    @property
    def route_top_k(self) -> int:
        return self.model.route_top_k

    @property
    def route_threshold_cold_semantic(self) -> float:
        return self.model.route_threshold_cold_semantic

    @property
    def route_threshold_eem(self) -> float:
        return self.model.route_threshold_eem

    @property
    def route_threshold_graph(self) -> float:
        return self.model.route_threshold_graph

    @property
    def route_weight(self) -> float:
        return self.model.route_weight

    @property
    def energy_weight(self) -> float:
        return self.model.energy_weight

    @property
    def route_consistency_weight(self) -> float:
        return self.model.route_consistency_weight

    @property
    def training_phase(self) -> str:
        return self.model.training_phase

    @property
    def router_warmup_steps(self) -> int:
        return self.model.router_warmup_steps

    @property
    def router_oracle_sharpness(self) -> float:
        return self.model.router_oracle_sharpness

    @property
    def router_oracle_bias_lm(self) -> float:
        return self.model.router_oracle_bias_lm

    @property
    def router_oracle_bias_semantic(self) -> float:
        return self.model.router_oracle_bias_semantic

    @property
    def router_oracle_bias_erm(self) -> float:
        return self.model.router_oracle_bias_erm

    @property
    def router_oracle_bias_eem(self) -> float:
        return self.model.router_oracle_bias_eem

    @property
    def router_oracle_bias_graph(self) -> float:
        return self.model.router_oracle_bias_graph

    @property
    def router_lane_dropout_prob(self) -> float:
        return self.model.router_lane_dropout_prob

    @property
    def router_entropy_floor_min(self) -> float:
        return self.model.router_entropy_floor_min

    @property
    def router_entropy_floor_weight(self) -> float:
        return self.model.router_entropy_floor_weight

    @property
    def router_collapse_mass_threshold(self) -> float:
        return self.model.router_collapse_mass_threshold

    @property
    def router_collapse_window(self) -> int:
        return self.model.router_collapse_window

    @property
    def router_recovery_steps(self) -> int:
        return self.model.router_recovery_steps

    @property
    def lane_cost_lm(self) -> float:
        return self.model.lane_cost_lm

    @property
    def lane_cost_semantic_hot(self) -> float:
        return self.model.lane_cost_semantic_hot

    @property
    def lane_cost_erm(self) -> float:
        return self.model.lane_cost_erm

    @property
    def lane_cost_semantic_cold(self) -> float:
        return self.model.lane_cost_semantic_cold

    @property
    def lane_cost_eem(self) -> float:
        return self.model.lane_cost_eem

    @property
    def lane_cost_graph(self) -> float:
        return self.model.lane_cost_graph

    @property
    def maintenance_cost(self) -> float:
        return self.model.maintenance_cost
