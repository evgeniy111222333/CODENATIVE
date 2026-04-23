from __future__ import annotations

import torch
from torch import nn

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import PhaseABatch, TokenClass


class CodeAwareEmbedding(nn.Module):
    def __init__(self, config: HTMCodeNativeConfig) -> None:
        super().__init__()
        vocab_size = config.model.vocabulary_size
        model_dim = config.model.model_dim
        byte_dim = config.model.byte_embedding_dim
        max_positions = config.model.max_position_embeddings
        max_ast_depth = config.model.max_ast_depth

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.class_embedding = nn.Embedding(len(TokenClass), model_dim)
        self.language_embedding = nn.Embedding(8, model_dim)
        self.scope_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_positions, model_dim)

        self.byte_embedding = nn.Embedding(257, byte_dim, padding_idx=256)
        self.byte_proj = nn.Linear(byte_dim, model_dim)

        self.ast_type_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.ast_depth_embedding = nn.Embedding(max_ast_depth + 1, model_dim, padding_idx=0)
        self.symbol_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.file_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)

        self.tok_proj = nn.Linear(model_dim, model_dim)
        self.struct_proj = nn.Linear(model_dim, model_dim)
        self.output_bias = nn.Parameter(torch.zeros(model_dim))

    def forward(
        self,
        batch: PhaseABatch,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shifted_positions = (batch.positions + position_offset).clamp(
            max=self.position_embedding.num_embeddings - 1
        )
        token_component = (
            self.token_embedding(batch.token_ids)
            + self.class_embedding(batch.token_class_ids)
            + self.language_embedding(batch.language_ids)
            + self.scope_embedding(batch.scope_ids)
            + self.position_embedding(shifted_positions)
        )

        byte_emb = self.byte_embedding(batch.byte_values)
        byte_mask = batch.byte_mask.unsqueeze(-1)
        byte_sum = (byte_emb * byte_mask).sum(dim=1)
        byte_den = byte_mask.sum(dim=1).clamp(min=1)
        byte_component = self.byte_proj(byte_sum / byte_den)

        ast_component = self.ast_type_embedding(batch.ast_type_ids) + self.ast_depth_embedding(
            batch.ast_depth_ids
        )
        ast_component = (ast_component * batch.ast_mask.unsqueeze(-1)).sum(dim=1)
        struct_component = (
            ast_component
            + self.symbol_embedding(batch.symbol_ids)
            + self.file_embedding(batch.file_ids)
        )

        embeddings = (
            self.tok_proj(token_component)
            + byte_component
            + self.struct_proj(struct_component)
            + self.output_bias
        )

        return embeddings, {
            "token_component": token_component,
            "byte_component": byte_component,
            "struct_component": struct_component,
        }
