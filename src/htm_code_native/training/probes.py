from __future__ import annotations

import time

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import PhaseExitReport, TaskExample, TaskLabel, TrainingPhase
from htm_code_native.data.vocabulary import VocabularyRegistry
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
    examples = probe_examples[:max_steps] if max_steps is not None else probe_examples
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
        "recent_copy_delta_vs_semantic": [],
        "episodic_delta_vs_semantic": [],
        "symbol_link_delta_vs_semantic": [],
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
    if phase == TrainingPhase.PHASE_E:
        if aggregate_metrics["route_entropy"] < config.model.probe_min_route_entropy:
            failing_checks.append("route_entropy_below_threshold")

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
