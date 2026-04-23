from __future__ import annotations

import json
from pathlib import Path

from htm_code_native.cli import main
from htm_code_native.data.types import TaskLabel, TrainingPhase
from htm_code_native.losses.core import masked_autoregressive_loss
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.training.probes import build_probe_examples, run_phase_exit_probes
from htm_code_native.training.session import TaskSessionRunConfig, run_task_batch_with_session
from htm_code_native.training.tasks import build_task_batch, build_task_example


REPO_GRAPH_ROOT = Path("tests/fixtures/repo_graph_workspace")
REPO_GRAPH_REPORTS = [
    str(REPO_GRAPH_ROOT / "reports" / "junit.xml"),
    str(REPO_GRAPH_ROOT / "reports" / "eslint.json"),
]


def test_infill_training_path(config) -> None:
    example = build_task_example("tests/fixtures/sample_module.py", TaskLabel.INFILL)
    task_batch = build_task_batch(example, config)
    model = PhaseACodeModel(config)
    output = model(
        task_batch.batch,
        phase=TrainingPhase.PHASE_C,
        task_label=TaskLabel.INFILL,
        global_step=0,
    )
    loss = masked_autoregressive_loss(
        output.logits,
        task_batch.batch.targets,
        task_batch.supervision_mask,
    )
    assert task_batch.infill_span is not None
    assert int(task_batch.supervision_mask.sum().item()) > 0
    assert loss.item() >= 0.0


def test_phase_exit_probes_report(config) -> None:
    model = PhaseACodeModel(config)
    report = run_phase_exit_probes(
        model,
        build_probe_examples(
            "default",
            repo_root=str(REPO_GRAPH_ROOT),
            report_paths=REPO_GRAPH_REPORTS,
        ),
        config,
        TrainingPhase.PHASE_D,
        probe_set="default",
        max_steps=2,
    )
    assert report.phase == TrainingPhase.PHASE_D.value
    assert report.probe_set == "default"
    assert "tokens_per_sec" in report.metrics
    assert "graph_copy_hit_rate" in report.metrics
    assert "definition_use_graph_copy_hit_rate" in report.metrics
    assert "cold_read_rate" in report.metrics
    assert "semantic_cold_clusters" in report.metrics
    assert "exact_payload_recall" in report.metrics
    assert "exact_span_recall" in report.metrics
    assert report.metrics["cold_read_rate"] > 0.0
    assert report.metrics["semantic_cold_clusters"] > 0.0
    assert report.metrics["exact_payload_recall"] >= 0.0
    assert report.example_count == 2


def test_session_runner_keeps_examples_isolated(config) -> None:
    example = build_task_example("tests/fixtures/sample_module.py", TaskLabel.AR)
    task_batch = build_task_batch(example, config)
    model = PhaseACodeModel(config)
    run_config = TaskSessionRunConfig(
        chunk_size=32,
        maintenance_budget=config.semantic_memory.maintenance_budget,
        phase=TrainingPhase.PHASE_B,
        task_label=TaskLabel.AR,
    )

    first_result = run_task_batch_with_session(model, task_batch, config, run_config)
    second_result = run_task_batch_with_session(
        model,
        task_batch,
        config,
        TaskSessionRunConfig(
            chunk_size=0,
            maintenance_budget=0.0,
            phase=TrainingPhase.PHASE_B,
            task_label=TaskLabel.AR,
        ),
    )

    assert first_result.aggregate_metrics["cold_reads"] > 0.0
    assert first_result.aggregate_metrics["semantic_cold_clusters"] > 0.0
    assert "exact_payload_recall" in first_result.aggregate_metrics
    assert first_result.aggregate_metrics["exact_payload_candidate_steps"] > 0.0
    assert second_result.aggregate_metrics["cold_reads"] == 0.0
    assert second_result.aggregate_metrics["semantic_cold_clusters"] == 0.0


def test_phase_d_probes_prioritize_graph_examples_for_short_runs(config) -> None:
    model = PhaseACodeModel(config)
    report = run_phase_exit_probes(
        model,
        build_probe_examples(
            "default",
            repo_root=str(REPO_GRAPH_ROOT),
            report_paths=REPO_GRAPH_REPORTS,
        ),
        config,
        TrainingPhase.PHASE_D,
        probe_set="default",
        max_steps=3,
    )
    assert report.metrics["graph_supervision_count"] > 0.0
    assert report.metrics["graph_copy_hit_rate"] > 0.0
    assert report.metrics["cold_read_rate"] > 0.0


def test_phase_e_probes_include_planner_metrics(config) -> None:
    model = PhaseACodeModel(config)
    report = run_phase_exit_probes(
        model,
        build_probe_examples(
            "default",
            repo_root=str(REPO_GRAPH_ROOT),
            report_paths=REPO_GRAPH_REPORTS,
        ),
        config,
        TrainingPhase.PHASE_E,
        probe_set="default",
        max_steps=3,
    )
    assert "patch_candidate_valid_rate" in report.metrics
    assert "best_patch_hit_rate" in report.metrics
    assert "diagnostic_to_span_recall" in report.metrics
    assert "edit_fix_copy_hit_rate" in report.metrics
    assert "cold_semantic_invocation_rate" in report.metrics
    assert "patch_apply_success_rate" in report.metrics
    assert "patch_syntax_valid_rate" in report.metrics
    assert "best_patch_apply_valid_rate" in report.metrics
    assert report.metrics["patch_apply_success_rate"] > 0.0
    assert report.metrics["patch_syntax_valid_rate"] > 0.0


def test_cli_edit_plan_is_dry_run_and_reports_apply_results(capsys) -> None:
    file_path = REPO_GRAPH_ROOT / "app" / "core.py"
    original_source = file_path.read_text(encoding="utf-8")

    assert (
        main(
            [
                "edit-plan",
                str(file_path),
                "--phase",
                "phase_e",
                "--repo-root",
                str(REPO_GRAPH_ROOT),
                "--report-path",
                REPO_GRAPH_REPORTS[0],
                "--report-path",
                REPO_GRAPH_REPORTS[1],
                "--target-symbol",
                "GRAPH_SHARED_NAME",
                "--instruction",
                'Replace GRAPH_SHARED_NAME with "shared_graph_token"',
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert file_path.read_text(encoding="utf-8") == original_source
    assert payload["apply_results"]
    assert payload["verification_summary"]["candidate_count"] == len(payload["apply_results"])
    assert payload["validation_summary"]["patch_apply_success_rate"] > 0.0
    assert "--apply" not in payload


def test_cli_eval_only_phase_report() -> None:
    assert (
        main(
            [
                "smoke-train",
                "--eval-only",
                "--phase",
                "phase_d",
                "--probe-set",
                "default",
                "--max-steps",
                "2",
                "--repo-root",
                str(REPO_GRAPH_ROOT),
                "--report-path",
                REPO_GRAPH_REPORTS[0],
                "--report-path",
                REPO_GRAPH_REPORTS[1],
            ]
        )
        == 0
    )
