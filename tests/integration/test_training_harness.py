from __future__ import annotations

from pathlib import Path

from htm_code_native.cli import main
from htm_code_native.data.types import TaskLabel, TrainingPhase
from htm_code_native.losses.core import masked_autoregressive_loss
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.training.probes import build_probe_examples, run_phase_exit_probes
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
    assert report.example_count == 2


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
