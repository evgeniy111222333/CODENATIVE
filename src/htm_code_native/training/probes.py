from __future__ import annotations

import time

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import PhaseExitReport, TaskBatch, TaskExample, TaskLabel, TrainingPhase
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.editing.planner import build_edit_request, run_edit_plan
from htm_code_native.losses.core import autoregressive_loss, masked_autoregressive_loss
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.training.metrics import has_invalid_number, safe_delta, safe_mean
from htm_code_native.training.tasks import (
    build_repo_graph_index,
    build_task_batch,
    default_task_examples,
    flatten_examples,
)


def build_probe_examples(
    probe_set: str,
    *,
    repo_root: str | None = None,
    report_paths: list[str] | tuple[str, ...] | None = None,
) -> list[TaskExample]:
    del probe_set
    return flatten_examples(default_task_examples(repo_root=repo_root, report_paths=report_paths))


def run_phase_exit_probes(
    model: PhaseACodeModel,
    probe_examples: list[TaskExample],
    config: HTMCodeNativeConfig,
    phase: TrainingPhase,
    *,
    probe_set: str = "default",
    max_steps: int | None = None,
) -> PhaseExitReport:
    selected_examples = _select_probe_examples_for_phase(probe_examples, phase, max_steps)
    examples = selected_examples
    if not examples:
        return PhaseExitReport(
            phase=phase.value,
            probe_set=probe_set,
            passed=False,
            metrics={"example_count": 0.0},
            failing_checks=("no_examples",),
            example_count=0,
        )

    registry = VocabularyRegistry(config.model.vocabulary_size)
    baseline = PhaseACodeModel(config)
    baseline.load_state_dict(model.state_dict())
    baseline.eval()

    graph_cache: dict[tuple[str, tuple[str, ...]], object] = {}
    was_training = model.training
    model.eval()

    metrics_accumulator: dict[str, list[float]] = {
        "ar_loss_mean": [],
        "recent_copy_hit_rate": [],
        "episodic_hit_rate": [],
        "graph_copy_hit_rate": [],
        "symbol_link_hit_rate": [],
        "route_entropy": [],
        "energy_proxy": [],
        "graph_supervision_count": [],
        "definition_use_hit_rate": [],
        "diagnostic_link_hit_rate": [],
        "edit_fix_graph_hit_rate": [],
        "definition_use_graph_copy_hit_rate": [],
        "diagnostic_graph_copy_hit_rate": [],
        "edit_fix_copy_hit_rate": [],
        "recent_copy_delta_vs_semantic": [],
        "episodic_delta_vs_semantic": [],
        "symbol_link_delta_vs_semantic": [],
        "patch_candidate_valid_rate": [],
        "best_patch_hit_rate": [],
        "diagnostic_to_span_recall": [],
    }

    start = time.perf_counter()
    with torch.no_grad():
        for step_index, example in enumerate(examples):
            task_batch = build_task_batch(example, config, registry=registry)
            graph_index = None
            if example.repo_root is not None or example.report_paths:
                cache_key = (str(example.repo_root), tuple(example.report_paths))
                if cache_key not in graph_cache:
                    graph_cache[cache_key] = build_repo_graph_index(
                        example.file_path,
                        config,
                        repo_root=example.repo_root,
                        report_paths=example.report_paths,
                    )
                graph_index = graph_cache[cache_key]

            model.set_repo_graph_index(graph_index)
            baseline.set_repo_graph_index(graph_index)
            output = model(
                task_batch.batch,
                reset_eem=True,
                phase=phase,
                task_label=example.task_label,
                global_step=step_index,
            )
            baseline_output = baseline(
                task_batch.batch,
                reset_eem=True,
                phase=TrainingPhase.PHASE_A,
                task_label=TaskLabel.AR,
                global_step=step_index,
            )

            if example.task_label == TaskLabel.INFILL:
                ar_value = float(
                    masked_autoregressive_loss(
                        output.logits,
                        task_batch.batch.targets,
                        task_batch.supervision_mask,
                    ).item()
                )
            else:
                ar_value = float(autoregressive_loss(output.logits, task_batch.batch.targets).item())
            metrics_accumulator["ar_loss_mean"].append(ar_value)
            metrics_accumulator["recent_copy_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["recent_copy_hit_rate"])
            )
            metrics_accumulator["episodic_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["episodic_hit_rate"])
            )
            metrics_accumulator["graph_copy_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["graph_copy_hit_rate"])
            )
            metrics_accumulator["symbol_link_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["symbol_link_hit_rate"])
            )
            metrics_accumulator["graph_supervision_count"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["graph_supervision_count"])
            )
            metrics_accumulator["definition_use_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["definition_use_hit_rate"])
            )
            metrics_accumulator["diagnostic_link_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["diagnostic_link_hit_rate"])
            )
            metrics_accumulator["edit_fix_graph_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["edit_fix_graph_hit_rate"])
            )
            metrics_accumulator["definition_use_graph_copy_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["definition_use_graph_copy_hit_rate"])
            )
            metrics_accumulator["diagnostic_graph_copy_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["diagnostic_graph_copy_hit_rate"])
            )
            metrics_accumulator["edit_fix_copy_hit_rate"].append(
                float(output.auxiliary["phase_exit_probe_metrics"]["edit_fix_copy_hit_rate"])
            )
            metrics_accumulator["route_entropy"].append(float(output.memory_stats["router_entropy"]))
            metrics_accumulator["energy_proxy"].append(float(output.memory_stats["avg_energy_proxy"]))
            metrics_accumulator["recent_copy_delta_vs_semantic"].append(
                safe_delta(
                    float(output.auxiliary["phase_exit_probe_metrics"]["recent_copy_hit_rate"]),
                    float(baseline_output.auxiliary["phase_exit_probe_metrics"]["recent_copy_hit_rate"]),
                )
            )
            metrics_accumulator["episodic_delta_vs_semantic"].append(
                safe_delta(
                    float(output.auxiliary["phase_exit_probe_metrics"]["episodic_hit_rate"]),
                    float(baseline_output.auxiliary["phase_exit_probe_metrics"]["episodic_hit_rate"]),
                )
            )
            metrics_accumulator["symbol_link_delta_vs_semantic"].append(
                safe_delta(
                    float(output.auxiliary["phase_exit_probe_metrics"]["symbol_link_hit_rate"]),
                    float(baseline_output.auxiliary["phase_exit_probe_metrics"]["symbol_link_hit_rate"]),
                )
            )
            if phase == TrainingPhase.PHASE_E and _example_uses_edit_planner(example):
                edit_output = run_edit_plan(
                    model,
                    _build_probe_edit_request(example, config),
                    config,
                )
                planner_metrics = _planner_probe_metrics(edit_output, task_batch)
                metrics_accumulator["patch_candidate_valid_rate"].append(planner_metrics["patch_candidate_valid_rate"])
                metrics_accumulator["best_patch_hit_rate"].append(planner_metrics["best_patch_hit_rate"])
                metrics_accumulator["diagnostic_to_span_recall"].append(planner_metrics["diagnostic_to_span_recall"])

    elapsed = time.perf_counter() - start
    token_count = sum(len(build_task_batch(example, config).batch.document.tokens) for example in examples)
    aggregate_metrics = {name: safe_mean(values) for name, values in metrics_accumulator.items()}
    aggregate_metrics["tokens_per_sec"] = float(token_count / max(elapsed, 1e-6))
    aggregate_metrics["example_count"] = float(len(examples))

    failing_checks: list[str] = []
    if has_invalid_number(aggregate_metrics):
        failing_checks.append("nan_detected")
    if aggregate_metrics["tokens_per_sec"] < config.model.probe_min_tokens_per_sec:
        failing_checks.append("throughput_below_threshold")
    if aggregate_metrics["energy_proxy"] > config.model.probe_max_energy_proxy:
        failing_checks.append("energy_above_threshold")
    if phase in {TrainingPhase.PHASE_B, TrainingPhase.PHASE_C, TrainingPhase.PHASE_D, TrainingPhase.PHASE_E}:
        if aggregate_metrics["recent_copy_hit_rate"] < config.model.probe_min_recent_copy_hit_rate:
            failing_checks.append("recent_copy_below_threshold")
    if phase in {TrainingPhase.PHASE_C, TrainingPhase.PHASE_D, TrainingPhase.PHASE_E}:
        if aggregate_metrics["episodic_hit_rate"] < config.model.probe_min_episodic_hit_rate:
            failing_checks.append("episodic_below_threshold")
    if phase in {TrainingPhase.PHASE_D, TrainingPhase.PHASE_E}:
        if aggregate_metrics["symbol_link_hit_rate"] < config.model.probe_min_symbol_link_hit_rate:
            failing_checks.append("symbol_link_below_threshold")
        if aggregate_metrics["graph_copy_hit_rate"] < config.model.probe_min_graph_copy_hit_rate:
            failing_checks.append("graph_copy_below_threshold")
    if phase == TrainingPhase.PHASE_E:
        if aggregate_metrics["route_entropy"] < config.model.probe_min_route_entropy:
            failing_checks.append("route_entropy_below_threshold")
        if aggregate_metrics["patch_candidate_valid_rate"] < config.model.probe_min_patch_candidate_valid_rate:
            failing_checks.append("patch_candidate_valid_below_threshold")
        if aggregate_metrics["best_patch_hit_rate"] < config.model.probe_min_best_patch_hit_rate:
            failing_checks.append("best_patch_below_threshold")
        if aggregate_metrics["diagnostic_to_span_recall"] < config.model.probe_min_diagnostic_to_span_recall:
            failing_checks.append("diagnostic_to_span_below_threshold")

    if was_training:
        model.train()
    return PhaseExitReport(
        phase=phase.value,
        probe_set=probe_set,
        passed=not failing_checks,
        metrics=aggregate_metrics,
        failing_checks=tuple(failing_checks),
        example_count=len(examples),
    )


def _select_probe_examples_for_phase(
    probe_examples: list[TaskExample],
    phase: TrainingPhase,
    max_steps: int | None,
) -> list[TaskExample]:
    ranked = sorted(
        enumerate(probe_examples),
        key=lambda item: (_probe_priority(item[1], phase), item[0]),
    )
    ordered = [example for _, example in ranked]
    return ordered[:max_steps] if max_steps is not None else ordered


def _probe_priority(example: TaskExample, phase: TrainingPhase) -> tuple[int, int]:
    probe_kind = str(example.metadata.get("probe_kind", "")).strip()
    if phase == TrainingPhase.PHASE_D:
        if example.task_label == TaskLabel.RECENT_COPY:
            return 0, 0
        if example.task_label == TaskLabel.EPISODIC_RECALL:
            return 1, 0
        if example.task_label == TaskLabel.REPO_GRAPH:
            graph_rank = {"definition_use": 0, "diagnostic_to_symbol": 1, "edit_fix": 2}.get(probe_kind, 3)
            return 2, graph_rank
        if example.task_label == TaskLabel.EDIT_FIX:
            return 3, 0
        if example.task_label == TaskLabel.AR:
            return 4, 0
        return 5, 0
    if phase == TrainingPhase.PHASE_E:
        if example.task_label == TaskLabel.EDIT_FIX:
            return 0, 0
        if example.task_label == TaskLabel.RECENT_COPY:
            return 1, 0
        if example.task_label == TaskLabel.EPISODIC_RECALL:
            return 2, 0
        if example.task_label == TaskLabel.REPO_GRAPH and probe_kind == "edit_fix":
            return 3, 0
        if example.task_label == TaskLabel.REPO_GRAPH and probe_kind == "diagnostic_to_symbol":
            return 4, 0
        if example.task_label == TaskLabel.REPO_GRAPH and probe_kind == "definition_use":
            return 5, 0
        if example.task_label == TaskLabel.AR:
            return 6, 0
        return 7, 0
    if phase == TrainingPhase.PHASE_C:
        if example.task_label == TaskLabel.EPISODIC_RECALL:
            return 0, 0
        if example.task_label == TaskLabel.RECENT_COPY:
            return 1, 0
        if example.task_label == TaskLabel.AR:
            return 2, 0
        return 3, 0
    if phase == TrainingPhase.PHASE_B:
        if example.task_label == TaskLabel.RECENT_COPY:
            return 0, 0
        if example.task_label == TaskLabel.AR:
            return 1, 0
        return 2, 0
    if example.task_label == TaskLabel.AR:
        return 0, 0
    if example.task_label == TaskLabel.INFILL:
        return 1, 0
    return 2, 0


def _example_uses_edit_planner(example: TaskExample) -> bool:
    return example.task_label == TaskLabel.EDIT_FIX or str(example.metadata.get("probe_kind", "")).strip() == "edit_fix"


def _build_probe_edit_request(example: TaskExample, config: HTMCodeNativeConfig):
    instruction = str(
        example.metadata.get("instruction")
        or "Inline shared_graph_token expected by diagnostics in app/core.py"
    )
    return build_edit_request(
        file_path=example.file_path,
        instruction=instruction,
        repo_root=example.repo_root,
        report_paths=example.report_paths,
        target_symbol=str(example.metadata.get("target_symbol", "")).strip() or None,
        phase=TrainingPhase.PHASE_E,
        max_candidates=config.model.edit_max_candidates,
    )


def _planner_probe_metrics(edit_output, task_batch: TaskBatch) -> dict[str, float]:
    valid_rate = float(edit_output.validation_summary.get("patch_candidate_valid_rate", 0.0))
    target_span = task_batch.edit_target_span
    overlapping_spans = [
        span
        for span in edit_output.span_candidates
        if target_span is not None and _span_overlaps_target(span.token_start, span.token_end, target_span)
    ]
    best_candidate = edit_output.patch_plan.best_candidate
    expected_replacement = str(
        task_batch.metadata.get("replacement_text")
        or task_batch.replacement_text
        or ""
    ).strip()
    best_patch_hit = 0.0
    if (
        best_candidate is not None
        and best_candidate.valid
        and target_span is not None
        and _span_overlaps_target(best_candidate.span.token_start, best_candidate.span.token_end, target_span)
    ):
        if not expected_replacement or best_candidate.replacement_text == expected_replacement:
            best_patch_hit = 1.0
    return {
        "patch_candidate_valid_rate": valid_rate,
        "best_patch_hit_rate": best_patch_hit,
        "diagnostic_to_span_recall": float(bool(overlapping_spans)),
    }


def _span_overlaps_target(token_start: int, token_end: int, target_span: tuple[int, int]) -> bool:
    return token_start < target_span[1] and token_end > target_span[0]
