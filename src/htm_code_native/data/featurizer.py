from __future__ import annotations

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import AlignedDocument, BoundaryEvents, CodeToken, PhaseABatch, TokenClass
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.utils.hashing import stable_int_hash


TOKEN_CLASS_TO_ID = {token_class: index for index, token_class in enumerate(TokenClass)}


def build_batch_from_document(
    document: AlignedDocument,
    boundaries: BoundaryEvents,
    config: HTMCodeNativeConfig,
    registry: VocabularyRegistry | None = None,
) -> PhaseABatch:
    seq_len = len(document.tokens)
    max_byte_window = config.model.max_byte_window
    max_ast_depth = config.model.max_ast_depth
    vocab_size = config.model.vocabulary_size
    registry = registry or VocabularyRegistry(capacity=vocab_size)

    token_ids = torch.tensor(
        [_encode_token_for_registry(token, registry) for token in document.tokens],
        dtype=torch.long,
    )
    token_class_ids = torch.tensor(
        [TOKEN_CLASS_TO_ID[token.token_class] for token in document.tokens],
        dtype=torch.long,
    )
    language_ids = torch.zeros(seq_len, dtype=torch.long)
    scope_ids = torch.tensor(
        [
            stable_int_hash("/".join(info.scope_path) or "module", vocab_size)
            for info in document.token_structures
        ],
        dtype=torch.long,
    )
    positions = torch.arange(seq_len, dtype=torch.long)
    byte_values = torch.full((seq_len, max_byte_window), 256, dtype=torch.long)
    byte_mask = torch.zeros((seq_len, max_byte_window), dtype=torch.bool)

    for token in document.tokens:
        token_bytes = list(document.token_bytes(token.index)[:max_byte_window])
        if token_bytes:
            byte_values[token.index, : len(token_bytes)] = torch.tensor(token_bytes, dtype=torch.long)
            byte_mask[token.index, : len(token_bytes)] = True

    ast_type_ids = torch.zeros((seq_len, max_ast_depth), dtype=torch.long)
    ast_depth_ids = torch.zeros((seq_len, max_ast_depth), dtype=torch.long)
    ast_mask = torch.zeros((seq_len, max_ast_depth), dtype=torch.bool)
    symbol_ids = torch.zeros(seq_len, dtype=torch.long)
    token_spans = torch.tensor(
        [[token.start_byte, token.end_byte] for token in document.tokens],
        dtype=torch.long,
    )
    token_payload_lengths = torch.tensor(
        [
            min(len(document.token_bytes(token.index)), config.model.max_recent_byte_payload)
            for token in document.tokens
        ],
        dtype=torch.long,
    )
    file_ids = torch.full(
        (seq_len,),
        stable_int_hash(document.file_path, vocab_size),
        dtype=torch.long,
    )

    for info in document.token_structures:
        for depth, node_type in enumerate(info.ast_path[:max_ast_depth], start=1):
            ast_type_ids[info.token_index, depth - 1] = stable_int_hash(node_type, vocab_size - 1) + 1
            ast_depth_ids[info.token_index, depth - 1] = depth
            ast_mask[info.token_index, depth - 1] = True
        if info.symbol_id is not None:
            symbol_ids[info.token_index] = stable_int_hash(info.symbol_id, vocab_size - 1) + 1

    boundaries_tensor = {
        level: torch.tensor(mask, dtype=torch.bool) for level, mask in boundaries.level_events.items()
    }
    targets = torch.roll(token_ids, shifts=-1)
    if seq_len:
        targets[-1] = token_ids[-1]

    return PhaseABatch(
        token_ids=token_ids,
        token_class_ids=token_class_ids,
        language_ids=language_ids,
        scope_ids=scope_ids,
        positions=positions,
        byte_values=byte_values,
        byte_mask=byte_mask,
        ast_type_ids=ast_type_ids,
        ast_depth_ids=ast_depth_ids,
        ast_mask=ast_mask,
        symbol_ids=symbol_ids,
        file_ids=file_ids,
        token_spans=token_spans,
        token_payload_lengths=token_payload_lengths,
        boundaries=boundaries_tensor,
        targets=targets,
        registry_size=registry.size,
        vocabulary_snapshot=registry.snapshot(),
        document=document,
    )


def _encode_token_for_registry(token: CodeToken, registry: VocabularyRegistry) -> int:
    token_key = token.value if token.value else f"<{token.token_type}>"
    return registry.encode_token(token_key)
