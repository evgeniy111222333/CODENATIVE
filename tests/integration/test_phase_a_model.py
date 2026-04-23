from __future__ import annotations

from pathlib import Path

from htm_code_native.cli import main
from htm_code_native.data.types import TaskLabel, TrainingPhase
from htm_code_native.losses.core import (
    autoregressive_loss,
    energy_penalty,
    episodic_pointer_loss,
    graph_copy_loss,
    recent_copy_loss,
    route_consistency_loss,
    router_entropy_floor_loss,
    router_oracle_loss,
    routing_loss,
    symbol_link_loss,
)
from htm_code_native.memory.repo_graph import RepositoryGraphIndexer
from htm_code_native.model.phase_a import PhaseACodeModel


REPO_GRAPH_ROOT = Path("tests/fixtures/repo_graph_workspace")
REPO_GRAPH_REPORTS = [
    str(REPO_GRAPH_ROOT / "reports" / "junit.xml"),
    str(REPO_GRAPH_ROOT / "reports" / "eslint.json"),
]


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
    model.train()
    output = model(batch, phase=TrainingPhase.PHASE_C, task_label=TaskLabel.EPISODIC_RECALL, global_step=0)
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
    loss = loss + graph_copy_loss(
        output.graph_logits,
        batch.targets,
        output.graph_copy_target_mask,
    )
    loss = loss + symbol_link_loss(
        output.auxiliary["graph_candidate_scores"],
        output.auxiliary["graph_candidate_ids"],
        output.auxiliary["graph_target_node_ids"],
    )
    loss = loss + routing_loss(
        output.auxiliary["router_post_logits"],
        output.auxiliary["route_teacher_indices"],
    )
    loss = loss + route_consistency_loss(
        output.auxiliary["router_pre_logits"],
        output.auxiliary["route_teacher_expensive"],
    )
    loss = loss + router_oracle_loss(
        output.auxiliary["router_post_logits"],
        output.auxiliary["oracle_router_weights"],
        output.auxiliary["router_post_masks"],
    )
    loss = loss + router_entropy_floor_loss(
        output.router_weights,
        config.model.router_entropy_floor_min,
    )
    loss = loss + energy_penalty(
        output.energy_proxy,
        output.memory_stats["always_on_energy"],
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


def test_graph_outputs_and_stats_are_emitted(build_batch, config) -> None:
    _, _, batch = build_batch(REPO_GRAPH_ROOT / "app" / "core.py")
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    model = PhaseACodeModel(config)
    model.set_repo_graph_index(indexer.build(REPO_GRAPH_ROOT, report_paths=REPO_GRAPH_REPORTS))
    model.train()
    output = model(batch, phase=TrainingPhase.PHASE_D, task_label=TaskLabel.REPO_GRAPH, global_step=0)
    assert output.graph_logits is not None
    assert output.graph_logits.shape == output.logits.shape
    assert output.graph_attention is not None
    assert output.graph_attention.shape[1] == config.model.graph_top_k
    assert output.base_hidden_states is not None
    assert output.base_hidden_states.shape == output.hidden_states.shape
    assert output.graph_contexts is not None
    assert output.graph_contexts.shape[1] == config.model.graph_value_dim
    assert output.router_weights is not None
    assert output.router_weights.shape[1] == 5
    assert output.effective_router_weights is not None
    assert output.oracle_router_weights is not None
    assert output.oracle_availability is not None
    assert output.router_pre_mask is not None
    assert output.router_pre_mask.shape[1] == 6
    assert output.router_post_mask is not None
    assert output.router_post_mask.shape[1] == 5
    assert output.lane_entropies is not None
    assert output.lane_entropies.shape[1] == 5
    assert output.energy_proxy is not None
    assert bool(output.router_pre_mask[:, :3].all().item()) is True
    assert not output.base_hidden_states.equal(output.hidden_states)
    assert not output.effective_router_weights.equal(output.router_weights)
    assert output.memory_stats["graph_reads"] > 0
    assert output.memory_stats["graph_candidates"] > 0
    assert output.memory_stats["graph_fusion_steps"] > 0
    assert output.memory_stats["avg_energy_proxy"] >= output.memory_stats["always_on_energy"]
    assert output.memory_stats["graph_task_eligible_steps"] < len(batch.document.tokens)
    assert output.memory_stats["graph_invocations"] <= output.memory_stats["graph_task_eligible_steps"]
    assert output.memory_stats["hard_gated_energy_savings"] > 0.0
    assert output.memory_stats["warmup_active_steps"] > 0.0
    assert "graph_candidate_ids" in output.auxiliary
    assert "graph_candidate_kinds" in output.auxiliary


def test_cli_commands_execute() -> None:
    sample_path = "tests/fixtures/sample_module.py"
    repo_root = "tests/fixtures"
    assert main(["tokenize", sample_path, "--limit", "5"]) == 0
    assert main(["inspect-structure", sample_path, "--repo-root", repo_root]) == 0
    assert main(["run-forward", sample_path, "--repo-root", repo_root]) == 0
    assert main(["smoke-train", "--steps", "1", "--repo-root", repo_root, "tests/fixtures/episodic_copy_module.py"]) == 0


def test_cli_graph_commands_execute() -> None:
    graph_path = str(REPO_GRAPH_ROOT / "app" / "core.py")
    repo_root = str(REPO_GRAPH_ROOT)
    junit_report = str(REPO_GRAPH_ROOT / "reports" / "junit.xml")
    eslint_report = str(REPO_GRAPH_ROOT / "reports" / "eslint.json")
    assert main(["inspect", graph_path, "--phase", "phase_d", "--repo-root", repo_root, "--report-path", junit_report, "--report-path", eslint_report]) == 0
    assert main(["run-forward", graph_path, "--phase", "phase_d", "--repo-root", repo_root, "--report-path", junit_report, "--report-path", eslint_report]) == 0
    assert main(["smoke-train", "--steps", "1", "--phase", "phase_d", "--repo-root", repo_root, "--report-path", junit_report, "--report-path", eslint_report, graph_path]) == 0
