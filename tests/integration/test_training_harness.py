from __future__ import annotations

import json
from pathlib import Path

import htm_code_native.cli as cli_module
from htm_code_native.cli import main, phase_loss_scales
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


def _repo_graph_edit_example():
    return build_task_example(
        str(REPO_GRAPH_ROOT / "app" / "core.py"),
        TaskLabel.EDIT_FIX,
        repo_root=str(REPO_GRAPH_ROOT),
        report_paths=REPO_GRAPH_REPORTS,
        metadata={
            "probe_kind": "edit_fix",
            "target_token_value": "GRAPH_SHARED_NAME",
            "replacement_text": "\"shared_graph_token\"",
            "instruction": "Inline shared_graph_token expected by diagnostics in app/core.py",
            "target_symbol": "GRAPH_SHARED_NAME",
        },
    )


def _run_single_smoke_train(capsys, monkeypatch, example, phase: TrainingPhase) -> dict[str, object]:
    def default_examples(**_kwargs):
        return {example.task_label: [example]}

    monkeypatch.setattr(cli_module, "default_task_examples", default_examples)
    assert (
        main(
            [
                "smoke-train",
                "--phase",
                phase.value,
                "--max-steps",
                "1",
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
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    return json.loads(lines[-1])


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


def test_edit_loss_scales_are_phase_e_only() -> None:
    for phase in (TrainingPhase.PHASE_A, TrainingPhase.PHASE_B, TrainingPhase.PHASE_C, TrainingPhase.PHASE_D):
        scales = phase_loss_scales(phase)
        assert scales["edit_span"] == 0.0
        assert scales["edit_patch"] == 0.0
        assert scales["diagnostic_alignment"] == 0.0
    assert phase_loss_scales(TrainingPhase.PHASE_A)["exact_emission"] == 0.0
    for phase in (TrainingPhase.PHASE_B, TrainingPhase.PHASE_C, TrainingPhase.PHASE_D, TrainingPhase.PHASE_E):
        assert phase_loss_scales(phase)["exact_emission"] == 1.0

    phase_e_scales = phase_loss_scales(TrainingPhase.PHASE_E)
    assert phase_e_scales["edit_span"] == 1.0
    assert phase_e_scales["edit_patch"] == 1.0
    assert phase_e_scales["diagnostic_alignment"] == 1.0


def test_phase_e_smoke_train_logs_edit_losses(capsys, monkeypatch) -> None:
    payload = _run_single_smoke_train(capsys, monkeypatch, _repo_graph_edit_example(), TrainingPhase.PHASE_E)

    assert payload["task_label"] == TaskLabel.EDIT_FIX.value
    assert payload["edit_span_loss"] > 0.0
    assert payload["edit_patch_loss"] > 0.0
    assert payload["diagnostic_alignment_loss"] > 0.0
    assert payload["edit_aux_loss"] > 0.0


def test_phase_d_smoke_train_logs_zero_weighted_edit_aux_loss(capsys, monkeypatch) -> None:
    payload = _run_single_smoke_train(capsys, monkeypatch, _repo_graph_edit_example(), TrainingPhase.PHASE_D)

    assert payload["task_label"] == TaskLabel.EDIT_FIX.value
    assert payload["edit_span_loss"] > 0.0
    assert payload["edit_patch_loss"] > 0.0
    assert payload["diagnostic_alignment_loss"] > 0.0
    assert payload["edit_aux_loss"] == 0.0


def test_phase_b_smoke_train_logs_exact_emission_loss(capsys, monkeypatch) -> None:
    example = build_task_example("tests/fixtures/recent_copy_module.py", TaskLabel.RECENT_COPY)
    payload = _run_single_smoke_train(capsys, monkeypatch, example, TrainingPhase.PHASE_B)

    assert payload["task_label"] == TaskLabel.RECENT_COPY.value
    assert payload["exact_emission_loss"] > 0.0
    assert payload["exact_emission_aux_loss"] > 0.0
    assert payload["phase_exit_probe_metrics"]["exact_emission_candidate_coverage"] > 0.0


def test_phase_a_smoke_train_logs_zero_weighted_exact_emission_aux_loss(capsys, monkeypatch) -> None:
    example = build_task_example("tests/fixtures/recent_copy_module.py", TaskLabel.RECENT_COPY)
    payload = _run_single_smoke_train(capsys, monkeypatch, example, TrainingPhase.PHASE_A)

    assert payload["exact_emission_loss"] == 0.0
    assert payload["exact_emission_aux_loss"] == 0.0


def test_non_edit_smoke_train_keeps_edit_losses_zero(capsys, monkeypatch) -> None:
    example = build_task_example("tests/fixtures/sample_module.py", TaskLabel.AR)
    payload = _run_single_smoke_train(capsys, monkeypatch, example, TrainingPhase.PHASE_E)

    assert payload["task_label"] == TaskLabel.AR.value
    assert payload["edit_span_loss"] == 0.0
    assert payload["edit_patch_loss"] == 0.0
    assert payload["diagnostic_alignment_loss"] == 0.0
    assert payload["edit_aux_loss"] == 0.0


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
    assert "graph_prune_rate" in report.metrics
    assert "cold_read_rate" in report.metrics
    assert "semantic_cold_clusters" in report.metrics
    assert "exact_payload_recall" in report.metrics
    assert "exact_span_recall" in report.metrics
    assert "exact_emission_candidate_coverage" in report.metrics
    assert "exact_byte_emission_hit_rate" in report.metrics
    assert "exact_span_emission_hit_rate" in report.metrics
    assert "avg_exact_emission_candidates" in report.metrics
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
    assert "exact_byte_emission_hit_rate" in first_result.aggregate_metrics
    assert first_result.aggregate_metrics["avg_exact_emission_candidates"] > 0.0
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
    assert report.metrics["graph_prune_rate"] > 0.0
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
