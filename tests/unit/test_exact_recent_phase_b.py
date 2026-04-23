from __future__ import annotations

import torch

from htm_code_native.data.types import ExactPayloadCandidate
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.losses.core import exact_emission_loss
from htm_code_native.memory.exact_recent import ExactRecentMemory
from htm_code_native.model.phase_a import PhaseACodeModel


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
    assert result.payload_candidates[1].token_id == 9
    assert result.payload_candidates[1].byte_payload == b"klmn"
    assert result.payload_candidates[1].start_byte == 10
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
    alpha_payloads = [
        candidate.byte_payload
        for candidate in result.payload_candidates
        if candidate.token_id == 7
    ]
    assert alpha_payloads == [b"a", b"a"]


def test_exact_emission_loss_handles_empty_missing_and_valid_targets() -> None:
    assert exact_emission_loss([], []).item() == 0.0

    missing_scores = torch.tensor([0.2, 0.8], requires_grad=True)
    missing_loss = exact_emission_loss([missing_scores], [None])
    assert missing_loss.device == missing_scores.device
    assert missing_loss.item() == 0.0

    scores = torch.tensor([0.2, 1.1], requires_grad=True)
    loss = exact_emission_loss([scores], [1])
    loss.backward()

    assert loss.item() > 0.0
    assert scores.grad is not None


def test_exact_emission_target_index_requires_exact_payload_and_span(config) -> None:
    model = PhaseACodeModel(config)
    candidates = (
        ExactPayloadCandidate(
            source="exact_recent",
            token_id=7,
            start_byte=0,
            end_byte=5,
            byte_payload=b"alpha",
            score=0.9,
        ),
        ExactPayloadCandidate(
            source="exact_recent",
            token_id=7,
            start_byte=10,
            end_byte=15,
            byte_payload=b"alpha",
            score=0.1,
        ),
        ExactPayloadCandidate(
            source="exact_episodic",
            token_id=7,
            start_byte=10,
            end_byte=15,
            byte_payload=b"beta",
            score=1.0,
        ),
    )

    assert (
        model._exact_emission_target_index(
            candidates=candidates,
            target_token_id=7,
            target_payload=b"alpha",
            target_span=(10, 15),
        )
        == 1
    )
    assert (
        model._exact_emission_target_index(
            candidates=candidates,
            target_token_id=7,
            target_payload=b"gamma",
            target_span=(20, 25),
        )
        is None
    )
