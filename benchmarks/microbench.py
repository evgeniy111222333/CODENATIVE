from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from htm_code_native.cli import build_repo_graph_index, load_config
from htm_code_native.data.types import PhaseABatch, RepositoryGraphIndex, TaskLabel, TrainingPhase
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.training import (
    TaskSessionRunConfig,
    build_task_batch,
    build_task_example,
    run_task_batch_with_session,
)


def _mean_target_probability(output, batch, token_class: str) -> float:
    values: list[float] = []
    seq_len = len(batch.document.tokens)
    for step in range(seq_len - 1):
        if not bool(output.copy_target_mask[step].item()):
            continue
        target_token = batch.document.tokens[step + 1]
        if target_token.token_class.value != token_class:
            continue
        target_id = int(batch.targets[step].item())
        values.append(float(output.erm_logits[step, target_id].exp().item()))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_episodic_target_probability(output, batch, token_class: str) -> float:
    values: list[float] = []
    seq_len = len(batch.document.tokens)
    for step in range(seq_len - 1):
        if not bool(output.episodic_target_mask[step].item()):
            continue
        target_token = batch.document.tokens[step + 1]
        if target_token.token_class.value != token_class:
            continue
        target_id = int(batch.targets[step].item())
        values.append(float(output.eem_logits[step, target_id].exp().item()))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_graph_target_probability(
    output,
    batch,
    token_class: str | None = None,
    required_kinds: set[str] | None = None,
) -> float:
    values: list[float] = []
    candidate_kinds = output.auxiliary.get("graph_candidate_kinds", [])
    graph_copy_target_ids = output.graph_copy_target_ids
    seq_len = len(batch.document.tokens)
    for step in range(seq_len - 1):
        if output.graph_logits is None or not bool(output.graph_copy_target_mask[step].item()):
            continue
        step_kinds = set(candidate_kinds[step]) if step < len(candidate_kinds) else set()
        if required_kinds is not None and not step_kinds.intersection(required_kinds):
            continue
        target_id = (
            int(graph_copy_target_ids[step].item())
            if graph_copy_target_ids is not None
            else int(batch.targets[step].item())
        )
        if target_id < 0:
            continue
        target_text = batch.vocabulary_snapshot.id_to_token.get(target_id, "")
        if token_class is not None and _token_class_for_graph_copy_target(target_text) != token_class:
            continue
        values.append(float(output.graph_logits[step, target_id].exp().item()))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _token_class_for_graph_copy_target(value: str) -> str:
    stripped = value.strip("\"'")
    if stripped.isidentifier():
        return "identifier"
    try:
        float(stripped)
    except ValueError:
        return "string"
    return "number"


def _infer_graph_probe_metadata(batch: PhaseABatch, graph_index: RepositoryGraphIndex | None) -> dict[str, str]:
    if graph_index is None:
        return {}
    copy_terms = {
        term
        for node in graph_index.nodes
        if node.kind in {"symbol", "function", "class"}
        for term in node.copy_terms
    }
    candidates: list[tuple[float, str]] = []
    for token in batch.document.tokens:
        if token.token_class.value != "identifier" or token.value not in copy_terms:
            continue
        structure = batch.document.token_structures[token.index]
        score = 0.0
        if structure.scope_path:
            score += 3.0
        if not {"import_statement", "import_from_statement"}.intersection(structure.ast_path):
            score += 2.0
        candidates.append((score, token.value))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0], reverse=True)
    target = candidates[0][1]
    return {
        "probe_kind": "definition_use",
        "target_token_value": target,
        "target_symbol": target,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", default="tests/fixtures/repo_graph_workspace/app/core.py")
    parser.add_argument("--repo-root")
    parser.add_argument("--report-path", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    config = load_config(None)
    phase = TrainingPhase(config.model.training_phase)
    example = build_task_example(
        args.file_path,
        repo_root=args.repo_root,
        report_paths=args.report_path,
    )
    task_label = example.task_label
    task_batch = build_task_batch(example, config)
    batch = task_batch.batch
    graph_index = build_repo_graph_index(
        args.file_path,
        config,
        repo_root=args.repo_root,
        report_paths=args.report_path,
    )
    if task_label == TaskLabel.REPO_GRAPH and not example.metadata.get("target_token_value"):
        inferred_metadata = _infer_graph_probe_metadata(batch, graph_index)
        if inferred_metadata:
            example = build_task_example(
                args.file_path,
                task_label,
                repo_root=args.repo_root,
                report_paths=args.report_path,
                metadata=inferred_metadata,
            )
            task_batch = build_task_batch(example, config)
            batch = task_batch.batch
    effective_chunk_size = args.chunk_size if args.chunk_size > 0 else config.model.semantic_session_chunk_size
    model = PhaseACodeModel(config)
    model.set_repo_graph_index(graph_index)
    model.eval()

    warmup = max(args.warmup, 0)
    iterations = max(args.iterations, 1)
    for _ in range(warmup):
        with torch.no_grad():
            run_task_batch_with_session(
                model,
                task_batch,
                config,
                TaskSessionRunConfig(
                    chunk_size=effective_chunk_size,
                    maintenance_budget=config.semantic_memory.maintenance_budget,
                    phase=phase,
                    task_label=task_label,
                ),
            )

    start = time.perf_counter()
    hot_reads = 0.0
    cold_reads = 0.0
    maintenance = 0.0
    erm_reads = 0.0
    erm_writes = 0.0
    erm_overwrites = 0.0
    copy_hits = 0.0
    exact_payload_recall = 0.0
    exact_span_recall = 0.0
    exact_recent_payload_hits = 0.0
    exact_episodic_payload_hits = 0.0
    exact_emission_candidate_coverage = 0.0
    exact_byte_emission_hit_rate = 0.0
    exact_span_emission_hit_rate = 0.0
    exact_emission_candidates = 0.0
    episodic_hits = 0.0
    identifier_recall = 0.0
    string_recall = 0.0
    number_recall = 0.0
    eem_reads = 0.0
    chunks_finalized = 0.0
    chunk_overhead = 0.0
    long_identifier_recall = 0.0
    long_string_recall = 0.0
    long_number_recall = 0.0
    graph_reads = 0.0
    graph_candidates = 0.0
    graph_candidate_pool_size = 0.0
    graph_total_nodes = 0.0
    graph_pruned_nodes = 0.0
    graph_symbol_recall = 0.0
    graph_test_recall = 0.0
    graph_diagnostic_recall = 0.0
    route_entropy = 0.0
    energy_proxy = 0.0
    skipped_expensive = 0.0
    graph_invocation_rate = 0.0
    eem_invocation_rate = 0.0
    cold_semantic_invocation_rate = 0.0
    erm_carryover_hits = 0.0
    router_continuity_windows = 0.0
    total_windows = 0.0
    tokens = 0
    for _ in range(iterations):
        with torch.no_grad():
            session_result = run_task_batch_with_session(
                model,
                task_batch,
                config,
                TaskSessionRunConfig(
                    chunk_size=effective_chunk_size,
                    maintenance_budget=config.semantic_memory.maintenance_budget,
                    phase=phase,
                    task_label=task_label,
                ),
            )
            outputs = session_result.outputs
            windows = session_result.windows
        for window_index, (window, output) in enumerate(zip(windows, outputs, strict=False)):
            window_batch = window.batch
            hot_reads += output.memory_stats["hot_reads"]
            cold_reads += output.memory_stats["cold_reads"]
            maintenance += output.memory_stats["maintenance_invocations"]
            erm_reads += output.memory_stats["erm_reads"]
            erm_writes += output.memory_stats["erm_writes"]
            erm_overwrites += output.memory_stats["erm_overwrites"]
            copy_hits += output.memory_stats["copy_target_hits"]
            exact_payload_recall += float(
                output.auxiliary["phase_exit_probe_metrics"]["exact_payload_recall"]
            )
            exact_span_recall += float(
                output.auxiliary["phase_exit_probe_metrics"]["exact_span_recall"]
            )
            exact_recent_payload_hits += float(output.memory_stats["exact_recent_payload_hits"])
            exact_episodic_payload_hits += float(output.memory_stats["exact_episodic_payload_hits"])
            exact_emission_candidate_coverage += float(
                output.auxiliary["phase_exit_probe_metrics"]["exact_emission_candidate_coverage"]
            )
            exact_byte_emission_hit_rate += float(
                output.auxiliary["phase_exit_probe_metrics"]["exact_byte_emission_hit_rate"]
            )
            exact_span_emission_hit_rate += float(
                output.auxiliary["phase_exit_probe_metrics"]["exact_span_emission_hit_rate"]
            )
            exact_emission_candidates += float(
                output.auxiliary["phase_exit_probe_metrics"]["avg_exact_emission_candidates"]
            )
            episodic_hits += output.memory_stats["episodic_target_hits"]
            identifier_recall += _mean_target_probability(output, window_batch, "identifier")
            string_recall += _mean_target_probability(output, window_batch, "string")
            number_recall += _mean_target_probability(output, window_batch, "number")
            eem_reads += output.memory_stats["eem_reads"]
            chunks_finalized += output.memory_stats["chunks_finalized"]
            chunk_overhead += output.memory_stats["avg_chunk_overhead"]
            long_identifier_recall += _mean_episodic_target_probability(output, window_batch, "identifier")
            long_string_recall += _mean_episodic_target_probability(output, window_batch, "string")
            long_number_recall += _mean_episodic_target_probability(output, window_batch, "number")
            graph_reads += output.memory_stats["graph_reads"]
            graph_candidates += output.memory_stats["graph_candidates"]
            graph_candidate_pool_size += output.memory_stats["graph_candidate_pool_size"]
            graph_total_nodes += output.memory_stats["graph_total_nodes_considered"]
            graph_pruned_nodes += output.memory_stats["graph_pruned_nodes"]
            graph_symbol_recall += _mean_graph_target_probability(
                output,
                window_batch,
                token_class="identifier",
                required_kinds={"symbol", "function", "class"},
            )
            graph_test_recall += _mean_graph_target_probability(
                output,
                window_batch,
                required_kinds={"test"},
            )
            graph_diagnostic_recall += _mean_graph_target_probability(
                output,
                window_batch,
                required_kinds={"diagnostic"},
            )
            route_entropy += float(output.diagnostics["route_entropy"])
            energy_proxy += float(output.memory_stats["avg_energy_proxy"])
            skipped_expensive += float(output.memory_stats["avg_skipped_expensive_reads"])
            graph_invocation_rate += float(output.memory_stats["graph_invocations"]) / max(len(window_batch.document.tokens), 1)
            eem_invocation_rate += float(output.memory_stats["eem_invocations"]) / max(len(window_batch.document.tokens), 1)
            cold_semantic_invocation_rate += float(output.memory_stats["cold_semantic_invocations"]) / max(len(window_batch.document.tokens), 1)
            if window_index > 0:
                erm_carryover_hits += float(output.memory_stats["copy_target_hits"])
                if (
                    output.memory_stats["warmup_steps_remaining"] > 0
                    or output.memory_stats["router_collapse_steps"] > 0
                    or output.auxiliary["session_stats"]["router_history_length"] > 0
                ):
                    router_continuity_windows += 1.0
            total_windows += 1.0
            tokens += len(window_batch.document.tokens)
    elapsed = time.perf_counter() - start
    last_output = outputs[-1]

    print(
        {
            "file": args.file_path,
            "chunk_size": effective_chunk_size,
            "session_window_count": len(windows),
            "tokens_per_sec": tokens / max(elapsed, 1e-6),
            "avg_hot_reads": hot_reads / iterations,
            "avg_cold_reads": cold_reads / iterations,
            "avg_maintenance_invocations": maintenance / iterations,
            "avg_erm_reads": erm_reads / iterations,
            "avg_erm_writes": erm_writes / iterations,
            "avg_erm_overwrites": erm_overwrites / iterations,
            "avg_copy_target_hits": copy_hits / iterations,
            "exact_payload_recall": exact_payload_recall / max(total_windows, 1.0),
            "exact_span_recall": exact_span_recall / max(total_windows, 1.0),
            "avg_exact_recent_payload_hits": exact_recent_payload_hits / iterations,
            "avg_exact_episodic_payload_hits": exact_episodic_payload_hits / iterations,
            "exact_emission_candidate_coverage": exact_emission_candidate_coverage / max(total_windows, 1.0),
            "exact_byte_emission_hit_rate": exact_byte_emission_hit_rate / max(total_windows, 1.0),
            "exact_span_emission_hit_rate": exact_span_emission_hit_rate / max(total_windows, 1.0),
            "avg_exact_emission_candidates": exact_emission_candidates / max(total_windows, 1.0),
            "avg_episodic_target_hits": episodic_hits / iterations,
            "repeated_identifier_recall": identifier_recall / iterations,
            "repeated_string_recall": string_recall / iterations,
            "repeated_number_recall": number_recall / iterations,
            "avg_eem_reads": eem_reads / iterations,
            "avg_chunks_finalized": chunks_finalized / iterations,
            "avg_chunk_overhead": chunk_overhead / iterations,
            "long_range_identifier_recall": long_identifier_recall / iterations,
            "long_range_string_recall": long_string_recall / iterations,
            "long_range_number_recall": long_number_recall / iterations,
            "avg_graph_reads": graph_reads / iterations,
            "avg_graph_candidates": graph_candidates / iterations,
            "avg_graph_candidate_pool_size": graph_candidate_pool_size / iterations,
            "avg_graph_prune_rate": graph_pruned_nodes / max(graph_total_nodes, 1.0),
            "graph_symbol_recall": graph_symbol_recall / iterations,
            "graph_test_recall": graph_test_recall / iterations,
            "graph_diagnostic_recall": graph_diagnostic_recall / iterations,
            "avg_route_entropy": route_entropy / iterations,
            "avg_energy_proxy": energy_proxy / iterations,
            "avg_warmup_beta": float(last_output.warmup_beta.mean().item()),
            "collapse_detected": bool(last_output.collapse_detected.any().item()),
            "avg_skipped_expensive_reads": skipped_expensive / iterations,
            "graph_invocation_rate": graph_invocation_rate / iterations,
            "eem_invocation_rate": eem_invocation_rate / iterations,
            "cold_semantic_invocation_rate": cold_semantic_invocation_rate / iterations,
            "avg_erm_carryover_hits": erm_carryover_hits / iterations,
            "avg_router_continuity_windows": router_continuity_windows / iterations,
            "avg_windows_per_iteration": total_windows / iterations,
            "always_on_energy": last_output.memory_stats["always_on_energy"],
            "full_enabled_energy": last_output.memory_stats["full_enabled_energy"],
            "hard_gated_energy_savings": last_output.memory_stats["hard_gated_energy_savings"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
