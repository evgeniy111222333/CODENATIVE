from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from htm_code_native.cli import main
from htm_code_native.data.types import TaskLabel, TrainingPhase
from htm_code_native.losses.core import (
    autoregressive_loss,
    energy_penalty,
    episodic_pointer_loss,
    exact_emission_loss,
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
from htm_code_native.training.tasks import _slice_task_batch_window, build_task_batch, build_task_example


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
    loss = loss + exact_emission_loss(
        output.exact_emission_candidate_scores,
        output.exact_emission_target_indices,
    )
    loss = loss + graph_copy_loss(
        output.graph_logits,
        batch.targets,
        output.graph_copy_target_mask,
        output.graph_copy_target_ids,
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
    assert output.exact_payload_target_mask is not None
    assert output.exact_span_target_mask is not None
    assert output.exact_emission_target_mask is not None
    assert output.exact_emission_candidate_scores is not None
    assert output.exact_emission_target_indices is not None
    assert output.exact_emission_predictions is not None
    assert bool(output.exact_payload_target_mask.any().item()) is True
    assert bool(output.exact_emission_target_mask.any().item()) is True
    assert output.memory_stats["exact_byte_candidate_hits"] > 0
    assert output.memory_stats["exact_span_candidate_hits"] > 0
    assert output.memory_stats["exact_emission_supervision_steps"] > 0
    assert output.auxiliary["phase_exit_probe_metrics"]["exact_payload_recall"] > 0.0
    assert output.auxiliary["phase_exit_probe_metrics"]["exact_emission_candidate_coverage"] > 0.0
    assert output.auxiliary["phase_exit_probe_metrics"]["avg_exact_emission_candidates"] > 0.0
    matching_steps = output.copy_target_mask.nonzero(as_tuple=False).flatten()
    first_step = int(matching_steps[0].item())
    target_id = int(batch.targets[first_step].item())
    assert output.erm_logits[first_step, target_id].exp().item() > 0.0
    assert output.auxiliary["exact_recent_payload_candidates"][first_step]


def test_eem_outputs_and_chunk_stats_are_emitted(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/episodic_copy_module.py"))
    model = PhaseACodeModel(config)
    output = model(batch)
    assert output.memory_stats["chunks_finalized"] > 0
    assert output.memory_stats["stored_chunks"] > 0
    assert output.memory_stats["eem_reads"] >= 0
    assert output.eem_logits is not None
    assert output.pointer_attention is not None
    assert output.memory_stats["exact_episodic_payload_candidates"] > 0
    assert any(output.auxiliary["exact_episodic_payload_candidates"])
    assert "exact_byte_emission_hit_rate" in output.auxiliary["phase_exit_probe_metrics"]


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
    assert output.graph_copy_target_ids is not None
    assert output.graph_copy_target_ids.shape[0] == output.logits.shape[0]
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
    assert output.memory_stats["graph_candidate_pool_size"] > 0
    assert output.memory_stats["graph_total_nodes_considered"] >= output.memory_stats["graph_candidate_pool_size"]
    assert output.memory_stats["graph_pruned_nodes"] > 0
    assert output.memory_stats["graph_prune_rate"] > 0.0
    assert output.memory_stats["graph_fusion_steps"] > 0
    assert output.memory_stats["avg_energy_proxy"] >= output.memory_stats["always_on_energy"]
    assert output.memory_stats["graph_task_eligible_steps"] < len(batch.document.tokens)
    assert output.memory_stats["graph_invocations"] <= output.memory_stats["graph_task_eligible_steps"]
    assert output.memory_stats["hard_gated_energy_savings"] > 0.0
    assert output.memory_stats["warmup_active_steps"] > 0.0
    assert "graph_candidate_ids" in output.auxiliary
    assert "graph_candidate_kinds" in output.auxiliary


def test_graph_supervision_alignment_tracks_only_supervised_steps(config) -> None:
    example = build_task_example(
        str(REPO_GRAPH_ROOT / "app" / "core.py"),
        TaskLabel.REPO_GRAPH,
        repo_root=str(REPO_GRAPH_ROOT),
        report_paths=REPO_GRAPH_REPORTS,
        metadata={
            "probe_kind": "definition_use",
            "target_token_value": "GRAPH_SHARED_NAME",
            "target_symbol": "GRAPH_SHARED_NAME",
        },
    )
    task_batch = build_task_batch(example, config)
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    model = PhaseACodeModel(config)
    model.set_repo_graph_index(indexer.build(REPO_GRAPH_ROOT, report_paths=REPO_GRAPH_REPORTS))
    output = model(
        task_batch.batch,
        phase=TrainingPhase.PHASE_D,
        task_label=TaskLabel.REPO_GRAPH,
        global_step=0,
    )

    supervision_mask = output.auxiliary["graph_supervision_mask"]
    supervision_count = int(output.auxiliary["graph_supervision_count"])
    target_step = max(
        index
        for index, token in enumerate(task_batch.batch.document.tokens)
        if token.value == "GRAPH_SHARED_NAME"
    )
    import_step = min(
        index
        for index, token in enumerate(task_batch.batch.document.tokens)
        if token.value == "GRAPH_SHARED_NAME"
    )
    expected_rate = output.memory_stats["symbol_link_hits"] / max(output.memory_stats["graph_supervision_steps"], 1.0)
    expected_copy_rate = output.memory_stats["graph_copy_hits"] / max(
        output.memory_stats["graph_copy_supervision_steps"],
        1.0,
    )

    assert supervision_count > 0
    assert supervision_count == int(supervision_mask.sum().item())
    assert bool(supervision_mask[import_step].item()) is False
    assert bool(supervision_mask[target_step].item()) is True
    assert output.auxiliary["graph_target_node_ids"][target_step] is not None
    assert output.graph_copy_target_ids[target_step].item() == task_batch.batch.token_ids[target_step].item()
    assert bool(output.graph_copy_target_mask[target_step].item()) is True
    assert "definition_use" in output.auxiliary["graph_supervision_mode"]
    assert abs(output.auxiliary["phase_exit_probe_metrics"]["symbol_link_hit_rate"] - expected_rate) < 1e-6
    assert abs(output.auxiliary["phase_exit_probe_metrics"]["graph_copy_hit_rate"] - expected_copy_rate) < 1e-6


def test_edit_fix_graph_supervision_uses_masked_edit_anchor(config) -> None:
    example = build_task_example(
        str(REPO_GRAPH_ROOT / "app" / "core.py"),
        TaskLabel.EDIT_FIX,
        repo_root=str(REPO_GRAPH_ROOT),
        report_paths=REPO_GRAPH_REPORTS,
        metadata={
            "probe_kind": "edit_fix",
            "target_token_value": "GRAPH_SHARED_NAME",
            "replacement_text": '"shared_graph_token"',
            "target_symbol": "GRAPH_SHARED_NAME",
        },
    )
    task_batch = build_task_batch(example, config)
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    model = PhaseACodeModel(config)
    model.set_repo_graph_index(indexer.build(REPO_GRAPH_ROOT, report_paths=REPO_GRAPH_REPORTS))
    output = model(
        task_batch.batch,
        phase=TrainingPhase.PHASE_E,
        task_label=TaskLabel.EDIT_FIX,
        global_step=0,
    )

    edit_start, edit_end = task_batch.edit_target_span
    import_step = next(
        index
        for index, token in enumerate(task_batch.batch.document.tokens)
        if token.value == "GRAPH_SHARED_NAME"
    )
    replacement_id = task_batch.batch.vocabulary_snapshot.lookup_token(task_batch.replacement_text)

    assert bool(output.auxiliary["graph_supervision_mask"][import_step].item()) is False
    assert output.auxiliary["graph_supervision_mode"][edit_start] == "edit_fix"
    assert bool(output.auxiliary["graph_supervision_mask"][edit_start].item()) is True
    assert output.auxiliary["graph_target_node_ids"][edit_start] is not None
    assert bool(output.graph_copy_target_mask[edit_start].item()) is True
    assert output.graph_copy_target_ids[edit_start].item() == replacement_id
    assert int(output.auxiliary["graph_supervision_count"]) == edit_end - edit_start


def test_forward_with_session_preserves_recent_copy_memory(config) -> None:
    example = build_task_example("tests/fixtures/recent_copy_module.py", TaskLabel.RECENT_COPY)
    task_batch = build_task_batch(example, config)
    first_window = _slice_task_batch_window(task_batch, 0, 24)
    second_window = _slice_task_batch_window(task_batch, 24, len(task_batch.batch.document.tokens))

    model = PhaseACodeModel(config)
    session_state = model.init_session_state()
    _, session_state = model.forward_with_session(
        first_window.batch,
        session_state=session_state,
        phase=TrainingPhase.PHASE_B,
        task_label=TaskLabel.RECENT_COPY,
        global_step=0,
    )
    resumed_output, resumed_session = model.forward_with_session(
        second_window.batch,
        session_state=session_state,
        phase=TrainingPhase.PHASE_B,
        task_label=TaskLabel.RECENT_COPY,
        global_step=1,
    )
    fresh_output, _ = model.forward_with_session(
        second_window.batch,
        session_state=model.init_session_state(),
        phase=TrainingPhase.PHASE_B,
        task_label=TaskLabel.RECENT_COPY,
        global_step=1,
    )

    target_step = 2
    target_id = int(second_window.batch.targets[target_step].item())
    assert bool(resumed_output.copy_target_mask[target_step].item()) is True
    assert bool(fresh_output.copy_target_mask[target_step].item()) is False
    assert resumed_output.erm_logits[target_step, target_id].item() > fresh_output.erm_logits[target_step, target_id].item()
    assert resumed_output.memory_stats["erm_fill"] >= first_window.batch.token_ids.shape[0]
    assert resumed_session.stream_token_index == len(task_batch.batch.document.tokens)


def test_forward_with_session_preserves_episodic_chunks_across_windows(config) -> None:
    example = build_task_example("tests/fixtures/episodic_copy_module.py", TaskLabel.EPISODIC_RECALL)
    task_batch = build_task_batch(example, config)
    first_window = _slice_task_batch_window(task_batch, 0, 128)
    second_window = _slice_task_batch_window(task_batch, 128, 200)

    model = PhaseACodeModel(config)
    session_state = model.init_session_state()
    _, session_state = model.forward_with_session(
        first_window.batch,
        session_state=session_state,
        phase=TrainingPhase.PHASE_C,
        task_label=TaskLabel.EPISODIC_RECALL,
        global_step=0,
    )
    resumed_output, resumed_session = model.forward_with_session(
        second_window.batch,
        session_state=session_state,
        phase=TrainingPhase.PHASE_C,
        task_label=TaskLabel.EPISODIC_RECALL,
        global_step=1,
    )
    fresh_output, _ = model.forward_with_session(
        second_window.batch,
        session_state=model.init_session_state(),
        phase=TrainingPhase.PHASE_C,
        task_label=TaskLabel.EPISODIC_RECALL,
        global_step=1,
    )

    assert len(session_state.exact_episodic.chunks) > 0
    assert resumed_output.memory_stats["stored_chunks"] > 0
    assert fresh_output.memory_stats["stored_chunks"] == 0
    assert resumed_session.stream_token_index == 200
    assert len(resumed_session.router.dominant_mass_history) >= len(session_state.router.dominant_mass_history)


def test_forward_with_session_preserves_cold_semantic_and_router_runtime(config) -> None:
    session_config = deepcopy(config)
    session_config.semantic_memory.hot_slots = 8
    session_config.semantic_memory.cold_slots = 8
    session_config.semantic_memory.beam_width = 2
    session_config.semantic_memory.consolidation_fill_threshold = 0.75
    session_config.semantic_memory.min_slots_for_consolidation = 4
    session_config.semantic_memory.maintenance_budget = 1.0
    session_config.model.router_collapse_window = 2
    session_config.model.router_collapse_mass_threshold = 0.8
    session_config.model.router_recovery_steps = 4

    example = build_task_example("tests/fixtures/sample_module.py", TaskLabel.AR)
    task_batch = build_task_batch(example, session_config)
    first_window = _slice_task_batch_window(task_batch, 0, 32)
    second_window = _slice_task_batch_window(task_batch, 32, 33)

    model = PhaseACodeModel(session_config)
    with torch.no_grad():
        model.router.pre_router[-1].weight.zero_()
        model.router.pre_router[-1].bias.copy_(torch.tensor([12.0, -12.0, -12.0]))
        model.router.post_router[-1].weight.zero_()
        model.router.post_router[-1].bias.copy_(torch.tensor([12.0, -12.0, -12.0, -12.0, -12.0]))

    session_state = model.init_session_state()
    _, session_state = model.forward_with_session(
        first_window.batch,
        session_state=session_state,
        phase=TrainingPhase.PHASE_B,
        task_label=TaskLabel.AR,
        global_step=0,
        maintenance_budget=1.0,
    )
    resumed_output, resumed_session = model.forward_with_session(
        second_window.batch,
        session_state=session_state,
        phase=TrainingPhase.PHASE_B,
        task_label=TaskLabel.AR,
        global_step=1,
        maintenance_budget=1.0,
    )
    fresh_output, fresh_session = model.forward_with_session(
        second_window.batch,
        session_state=model.init_session_state(),
        phase=TrainingPhase.PHASE_B,
        task_label=TaskLabel.AR,
        global_step=1,
        maintenance_budget=1.0,
    )

    cold_clusters = sum(len(clusters) for clusters in session_state.semantic_memory.cold_clusters.values())
    assert cold_clusters > 0
    assert resumed_output.memory_stats["cold_reads"] > 0
    assert fresh_output.memory_stats["cold_reads"] == 0
    assert resumed_output.memory_stats["warmup_steps_remaining"] > 0
    assert fresh_output.memory_stats["warmup_steps_remaining"] == 0
    assert resumed_session.router.recovery_steps_remaining > fresh_session.router.recovery_steps_remaining


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
