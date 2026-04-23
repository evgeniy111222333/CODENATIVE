from __future__ import annotations

from dataclasses import dataclass
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
