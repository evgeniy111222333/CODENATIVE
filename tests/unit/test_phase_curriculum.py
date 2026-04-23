from __future__ import annotations

from pathlib import Path

from htm_code_native.data.types import TaskLabel, TrainingPhase
from htm_code_native.memory.repo_graph import RepositoryGraphIndexer
from htm_code_native.model.phase_a import PhaseACodeModel


REPO_GRAPH_ROOT = Path("tests/fixtures/repo_graph_workspace")
REPO_GRAPH_REPORTS = [
    str(REPO_GRAPH_ROOT / "reports" / "junit.xml"),
    str(REPO_GRAPH_ROOT / "reports" / "eslint.json"),
]


def test_phase_a_forces_semantic_only(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/sample_module.py"))
    model = PhaseACodeModel(config)
    output = model(batch, phase=TrainingPhase.PHASE_A, task_label=TaskLabel.AR)
    assert float(output.effective_router_weights[:, 1].min().item()) == 1.0
    assert float(output.effective_router_weights[:, [0, 2, 3, 4]].sum().item()) == 0.0
    assert output.memory_stats["erm_writes"] == 0.0
    assert output.memory_stats["eem_invocations"] == 0.0
    assert output.memory_stats["graph_invocations"] == 0.0


def test_phase_b_enables_only_recent_exact(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/recent_copy_module.py"))
    model = PhaseACodeModel(config)
    model.train()
    output = model(batch, phase=TrainingPhase.PHASE_B, task_label=TaskLabel.RECENT_COPY, global_step=0)
    assert bool(output.router_post_mask[:, 3].any().item()) is False
    assert bool(output.router_post_mask[:, 4].any().item()) is False
    assert output.memory_stats["erm_writes"] == len(batch.document.tokens)
    assert output.memory_stats["warmup_active_steps"] > 0.0


def test_phase_c_enables_eem_but_keeps_graph_off(build_batch, config) -> None:
    _, _, batch = build_batch(Path("tests/fixtures/episodic_copy_module.py"))
    model = PhaseACodeModel(config)
    model.train()
    output = model(batch, phase=TrainingPhase.PHASE_C, task_label=TaskLabel.EPISODIC_RECALL, global_step=0)
    assert output.memory_stats["chunks_finalized"] > 0.0
    assert output.memory_stats["graph_invocations"] == 0.0
    assert bool(output.router_post_mask[:, 4].any().item()) is False


def test_phase_d_gates_graph_to_repo_tasks(build_batch, config) -> None:
    _, _, batch = build_batch(REPO_GRAPH_ROOT / "app" / "core.py")
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    graph_index = indexer.build(REPO_GRAPH_ROOT, report_paths=REPO_GRAPH_REPORTS)

    model = PhaseACodeModel(config)
    model.set_repo_graph_index(graph_index)
    output_ar = model(batch, phase=TrainingPhase.PHASE_D, task_label=TaskLabel.AR)
    assert output_ar.memory_stats["graph_invocations"] == 0.0

    model = PhaseACodeModel(config)
    model.set_repo_graph_index(graph_index)
    output_repo = model(batch, phase=TrainingPhase.PHASE_D, task_label=TaskLabel.REPO_GRAPH)
    assert output_repo.memory_stats["graph_task_eligible_steps"] > 0.0
    assert output_repo.memory_stats["graph_invocations"] > 0.0


def test_phase_e_allows_full_graph_without_warmup(build_batch, config) -> None:
    _, _, batch = build_batch(REPO_GRAPH_ROOT / "app" / "core.py")
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    model = PhaseACodeModel(config)
    model.set_repo_graph_index(indexer.build(REPO_GRAPH_ROOT, report_paths=REPO_GRAPH_REPORTS))
    output = model(batch, phase=TrainingPhase.PHASE_E, task_label=TaskLabel.AR)
    assert output.memory_stats["graph_invocations"] > 0.0
    assert float(output.warmup_beta.min().item()) == 1.0
