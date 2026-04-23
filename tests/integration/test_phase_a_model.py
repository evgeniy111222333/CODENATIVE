from __future__ import annotations

from pathlib import Path

from htm_code_native.cli import main
from htm_code_native.losses.core import autoregressive_loss, recent_copy_loss
from htm_code_native.model.phase_a import PhaseACodeModel


def test_end_to_end_forward_returns_logits(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/sample_module.py"))
    model = PhaseACodeModel(config)
    output = model(batch)
    assert output.logits.shape[0] == len(batch.document.tokens)
    assert output.logits.shape[1] == config.model.vocabulary_size
    assert output.memory_stats["hot_reads"] >= 0
    assert output.lm_logits.shape == output.logits.shape
    assert output.erm_logits.shape == output.logits.shape
    assert output.erm_attention.shape[1] == config.model.recent_window
    assert output.eem_logits.shape == output.logits.shape
    assert output.eem_attention.shape[1] == config.model.eem_top_k
    assert output.pointer_attention.shape[1] == config.model.eem_top_k * config.model.max_chunk_tokens
    assert output.memory_stats["erm_writes"] == len(batch.document.tokens)
    assert output.memory_stats["erm_fill"] <= config.model.recent_window


def test_smoke_train_step_backward(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/episodic_copy_module.py"))
    model = PhaseACodeModel(config)
    output = model(batch)
    loss = autoregressive_loss(output.logits, batch.targets) + recent_copy_loss(
        output.erm_logits,
        batch.targets,
        output.copy_target_mask,
    )
    loss = loss + episodic_pointer_loss(
        output.eem_logits,
        batch.targets,
        output.episodic_target_mask,
    )
    loss.backward()
    assert loss.item() >= 0.0


def test_recent_copy_mask_and_logits_are_emitted(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/recent_copy_module.py"))
    model = PhaseACodeModel(config)
    output = model(batch)
    assert bool(output.copy_target_mask.any().item()) is True
    matching_steps = output.copy_target_mask.nonzero(as_tuple=False).flatten()
    first_step = int(matching_steps[0].item())
    target_id = int(batch.targets[first_step].item())
    assert output.erm_logits[first_step, target_id].exp().item() > 0.0


def test_eem_outputs_and_chunk_stats_are_emitted(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/episodic_copy_module.py"))
    model = PhaseACodeModel(config)
    output = model(batch)
    assert output.memory_stats["chunks_finalized"] > 0
    assert output.memory_stats["stored_chunks"] > 0
    assert output.memory_stats["eem_reads"] >= 0
    assert output.eem_logits is not None
    assert output.pointer_attention is not None


def test_cli_commands_execute() -> None:
    sample_path = "tests/fixtures/sample_module.py"
    assert main(["tokenize", sample_path, "--limit", "5"]) == 0
    assert main(["inspect-structure", sample_path]) == 0
    assert main(["run-forward", sample_path]) == 0
    assert main(["smoke-train", "--steps", "1", "tests/fixtures/episodic_copy_module.py"]) == 0
