from __future__ import annotations

from dataclasses import dataclass, field

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import PhaseAOutput, PhaseASessionState, TaskBatch, TaskLabel, TrainingPhase
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.training.tasks import _iter_contiguous_task_windows


@dataclass(slots=True)
class TaskSessionRunConfig:
    chunk_size: int
    maintenance_budget: float
    phase: TrainingPhase
    task_label: TaskLabel
    global_step_base: int = 0


@dataclass(slots=True)
class TaskSessionRunResult:
    outputs: list[PhaseAOutput]
    windows: list[TaskBatch]
    final_session_state: PhaseASessionState
    aggregate_metrics: dict[str, float] = field(default_factory=dict)


def run_task_batch_with_session(
    model: PhaseACodeModel,
    task_batch: TaskBatch,
    config: HTMCodeNativeConfig,
    run_config: TaskSessionRunConfig,
) -> TaskSessionRunResult:
    del config
    windows = (
        _iter_contiguous_task_windows(task_batch, run_config.chunk_size)
        if run_config.chunk_size > 0
        else [task_batch]
    )
    session_state = model.init_session_state(device=task_batch.batch.token_ids.device)
    outputs: list[PhaseAOutput] = []
    for window_index, window in enumerate(windows):
        output, session_state = model.forward_with_session(
            window.batch,
            session_state=session_state,
            phase=run_config.phase,
            task_label=run_config.task_label,
            global_step=run_config.global_step_base + window_index,
            maintenance_budget=run_config.maintenance_budget,
        )
        outputs.append(output)
    return TaskSessionRunResult(
        outputs=outputs,
        windows=windows,
        final_session_state=session_state,
        aggregate_metrics=_aggregate_session_metrics(outputs, windows, session_state),
    )


def _aggregate_session_metrics(
    outputs: list[PhaseAOutput],
    windows: list[TaskBatch],
    session_state: PhaseASessionState,
) -> dict[str, float]:
    token_count = float(sum(len(window.batch.document.tokens) for window in windows))
    cold_reads = _sum_memory_stat(outputs, "cold_reads")
    cold_invocations = _sum_memory_stat(outputs, "cold_semantic_invocations")
    cold_enabled_steps = _sum_memory_stat(outputs, "cold_read_enabled_steps")
    exact_payload_steps = _sum_memory_stat(outputs, "exact_payload_candidate_steps")
    exact_byte_hits = _sum_memory_stat(outputs, "exact_byte_candidate_hits")
    exact_span_hits = _sum_memory_stat(outputs, "exact_span_candidate_hits")
    semantic_cold_clusters = float(
        sum(len(clusters) for clusters in session_state.semantic_memory.cold_clusters.values())
    )
    return {
        "session_window_count": float(len(windows)),
        "session_token_count": token_count,
        "hot_reads": _sum_memory_stat(outputs, "hot_reads"),
        "cold_reads": cold_reads,
        "cold_read_rate": cold_reads / max(token_count, 1.0),
        "cold_semantic_invocations": cold_invocations,
        "cold_semantic_invocation_rate": cold_invocations / max(token_count, 1.0),
        "cold_read_enabled_steps": cold_enabled_steps,
        "maintenance_invocations": _sum_memory_stat(outputs, "maintenance_invocations"),
        "maintenance_budgeted_steps": _sum_memory_stat(outputs, "maintenance_budgeted_steps"),
        "maintenance_effective_steps": _sum_memory_stat(outputs, "maintenance_effective_steps"),
        "cold_clusters_created": _sum_memory_stat(outputs, "cold_clusters_created"),
        "semantic_cold_clusters": semantic_cold_clusters,
        "exact_payload_candidate_steps": exact_payload_steps,
        "exact_payload_recall": exact_byte_hits / max(exact_payload_steps, 1.0),
        "exact_span_recall": exact_span_hits / max(exact_payload_steps, 1.0),
        "exact_recent_payload_hits": _sum_memory_stat(outputs, "exact_recent_payload_hits"),
        "exact_episodic_payload_hits": _sum_memory_stat(outputs, "exact_episodic_payload_hits"),
        "exact_recent_payload_candidates": _sum_memory_stat(outputs, "exact_recent_payload_candidates"),
        "exact_episodic_payload_candidates": _sum_memory_stat(outputs, "exact_episodic_payload_candidates"),
        "erm_carryover_hits": sum(
            float(output.memory_stats.get("copy_target_hits", 0.0))
            for window_index, output in enumerate(outputs)
            if window_index > 0
        ),
        "router_continuity_windows": sum(
            1.0
            for window_index, output in enumerate(outputs)
            if window_index > 0
            and (
                output.memory_stats.get("warmup_steps_remaining", 0.0) > 0.0
                or output.memory_stats.get("router_collapse_steps", 0.0) > 0.0
                or output.auxiliary.get("session_stats", {}).get("router_history_length", 0.0) > 0.0
            )
        ),
    }


def _sum_memory_stat(outputs: list[PhaseAOutput], name: str) -> float:
    return float(sum(float(output.memory_stats.get(name, 0.0)) for output in outputs))
