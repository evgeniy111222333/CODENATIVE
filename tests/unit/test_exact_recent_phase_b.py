from __future__ import annotations

import torch

from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.memory.exact_recent import ExactRecentMemory


def test_vocabulary_registry_stable_ids_and_unk() -> None:
    registry = VocabularyRegistry(capacity=5)
    alpha_id = registry.encode_token("alpha")
    alpha_repeat_id = registry.encode_token("alpha")
    beta_id = registry.encode_token("beta")
    gamma_id = registry.encode_token("gamma")

    assert alpha_id == alpha_repeat_id
    assert beta_id != registry.unk_id
    assert gamma_id == registry.unk_id


def test_exact_recent_ring_wraparound_and_payload_truncation() -> None:
    memory = ExactRecentMemory(
        hidden_size=4,
        key_dim=4,
        window_size=2,
        vocab_size=32,
        max_byte_payload=4,
    )
    with torch.no_grad():
        memory.write_projection.weight.copy_(torch.eye(4))
        memory.write_projection.bias.zero_()
        memory.query_projection.weight.copy_(torch.eye(4))
        memory.query_projection.bias.zero_()

    state = torch.tensor([1.0, 0.0, 0.0, 0.0])
    memory.write(state, 7, (0, 6), b"abcdef", 0)
    memory.write(state, 8, (6, 10), b"ghij", 1)
    overwritten = memory.write(state, 9, (10, 16), b"klmnop", 2)
    result = memory.read(state)

    assert overwritten is True
    assert result.filled_size == 2
    assert result.write_count == 3
    assert result.overwrite_count == 1
    assert result.slot_token_ids.tolist()[:2] == [8, 9]
    assert memory.slots[(memory.write_pointer - 1) % memory.window_size].byte_payload == b"klmn"


def test_exact_recent_repeated_tokens_accumulate_copy_mass() -> None:
    memory = ExactRecentMemory(
        hidden_size=4,
        key_dim=4,
        window_size=4,
        vocab_size=32,
        max_byte_payload=8,
    )
    with torch.no_grad():
        memory.write_projection.weight.copy_(torch.eye(4))
        memory.write_projection.bias.zero_()
        memory.query_projection.weight.copy_(torch.eye(4))
        memory.query_projection.bias.zero_()

    state = torch.tensor([1.0, 0.0, 0.0, 0.0])
    memory.write(state, 7, (0, 1), b"a", 0)
    memory.write(state, 3, (1, 2), b"b", 1)
    memory.write(state, 7, (2, 3), b"a", 2)
    result = memory.read(state)

    assert torch.isclose(result.attention[:3].sum(), torch.tensor(1.0))
    assert result.distribution[7].item() > result.distribution[3].item()
