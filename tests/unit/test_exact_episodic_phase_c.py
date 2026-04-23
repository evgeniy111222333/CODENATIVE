from __future__ import annotations

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import (
    AlignedDocument,
    BoundaryEvents,
    CodeToken,
    PhaseABatch,
    TokenClass,
    TokenStructureInfo,
)
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.losses.core import episodic_pointer_loss
from htm_code_native.memory.exact_episodic import ExactEpisodicMemory


def _build_minimal_batch() -> tuple[AlignedDocument, PhaseABatch]:
    raw = b"alpha beta alpha"
    tokens = [
        CodeToken(0, TokenClass.IDENTIFIER, "NAME", "alpha", 0, 5, "python", line=1, column=0),
        CodeToken(1, TokenClass.IDENTIFIER, "NAME", "beta", 6, 10, "python", line=1, column=6),
        CodeToken(2, TokenClass.IDENTIFIER, "NAME", "alpha", 11, 16, "python", line=1, column=11),
    ]
    structures = [
        TokenStructureInfo(
            token_index=i,
            ast_path=("Module",),
            ast_node_ids=(f"node-{i}",),
            symbol_id="sym-alpha" if token.value == "alpha" else None,
            symbol_name="alpha" if token.value == "alpha" else None,
            scope_path=("module",),
            file_id="fixture.py",
        )
        for i, token in enumerate(tokens)
    ]
    document = AlignedDocument(
        file_path="fixture.py",
        language="python",
        source_text="alpha beta alpha",
        raw_bytes=raw,
        tokens=tokens,
        byte_to_token_index=[0, 0, 0, 0, 0, -1, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2],
        token_structures=structures,
    )
    registry = VocabularyRegistry(capacity=32)
    alpha_id = registry.encode_token("alpha")
    beta_id = registry.encode_token("beta")
    token_ids = torch.tensor([alpha_id, beta_id, alpha_id], dtype=torch.long)
    batch = PhaseABatch(
        token_ids=token_ids,
        token_class_ids=torch.zeros(3, dtype=torch.long),
        language_ids=torch.zeros(3, dtype=torch.long),
        scope_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3, dtype=torch.long),
        byte_values=torch.zeros((3, 4), dtype=torch.long),
        byte_mask=torch.zeros((3, 4), dtype=torch.bool),
        ast_type_ids=torch.zeros((3, 2), dtype=torch.long),
        ast_depth_ids=torch.zeros((3, 2), dtype=torch.long),
        ast_mask=torch.zeros((3, 2), dtype=torch.bool),
        symbol_ids=torch.zeros(3, dtype=torch.long),
        file_ids=torch.zeros(3, dtype=torch.long),
        token_spans=torch.tensor([[0, 5], [6, 10], [11, 16]], dtype=torch.long),
        token_payload_lengths=torch.tensor([5, 4, 5], dtype=torch.long),
        boundaries={0: torch.tensor([True, True, True])},
        targets=torch.tensor([beta_id, alpha_id, alpha_id], dtype=torch.long),
        registry_size=registry.size,
        vocabulary_snapshot=registry.snapshot(),
        document=document,
    )
    return document, batch


def test_eem_chunk_creation_is_immutable_and_wraps() -> None:
    document, batch = _build_minimal_batch()
    memory = ExactEpisodicMemory(
        hidden_size=4,
        key_dim=4,
        pointer_key_dim=4,
        vocab_size=32,
        top_k=2,
        max_chunk_tokens=4,
        max_chunks=1,
    )
    with torch.no_grad():
        memory.chunk_write_projection.weight.copy_(torch.eye(4))
        memory.chunk_write_projection.bias.zero_()
        memory.pointer_write_projection.weight.copy_(torch.eye(4))
        memory.pointer_write_projection.bias.zero_()
        memory.chunk_query_projection.weight.copy_(torch.eye(4))
        memory.chunk_query_projection.bias.zero_()
        memory.pointer_query_projection.weight.copy_(torch.eye(4))
        memory.pointer_query_projection.bias.zero_()

    states = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    first_chunk = memory.maybe_finalize_chunk(document, batch, states, 0, 1, "block", 1)
    assert first_chunk is not None
    assert first_chunk.raw_bytes == b"alpha beta"
    second_chunk = memory.maybe_finalize_chunk(document, batch, states, 2, 2, "file", 2)
    assert second_chunk is not None
    assert len(memory.chunks) == 1
    assert memory.chunks[0].chunk_id == second_chunk.chunk_id
    assert first_chunk.raw_bytes == b"alpha beta"


def test_eem_retrieval_returns_pointer_mass_for_matching_token() -> None:
    document, batch = _build_minimal_batch()
    memory = ExactEpisodicMemory(
        hidden_size=4,
        key_dim=4,
        pointer_key_dim=4,
        vocab_size=32,
        top_k=2,
        max_chunk_tokens=4,
        max_chunks=4,
    )
    with torch.no_grad():
        memory.chunk_write_projection.weight.copy_(torch.eye(4))
        memory.chunk_write_projection.bias.zero_()
        memory.pointer_write_projection.weight.copy_(torch.eye(4))
        memory.pointer_write_projection.bias.zero_()
        memory.chunk_query_projection.weight.copy_(torch.eye(4))
        memory.chunk_query_projection.bias.zero_()
        memory.pointer_query_projection.weight.copy_(torch.eye(4))
        memory.pointer_query_projection.bias.zero_()

    states = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    memory.maybe_finalize_chunk(document, batch, states, 0, 2, "file", 2)
    result = memory.retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    alpha_id = int(batch.token_ids[0].item())
    beta_id = int(batch.token_ids[1].item())
    alpha_candidates = [
        candidate
        for candidate in result.payload_candidates
        if candidate.token_id == alpha_id and candidate.byte_payload == b"alpha"
    ]
    assert result.retrieved_chunk_count == 1
    assert result.distribution[alpha_id].item() > result.distribution[beta_id].item()
    assert alpha_candidates
    assert all(candidate.chunk_id == 0 for candidate in alpha_candidates)
    assert any(candidate.start_byte == 0 and candidate.end_byte == 5 for candidate in alpha_candidates)
    loss = episodic_pointer_loss(
        result.log_distribution.unsqueeze(0),
        torch.tensor([alpha_id]),
        torch.tensor([True]),
    )
    assert loss.item() > 0.0
