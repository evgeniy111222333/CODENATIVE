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

from htm_code_native.cli import build_batch, build_repo_graph_index, load_config
from htm_code_native.model.phase_a import PhaseACodeModel


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
    seq_len = len(batch.document.tokens)
    for step in range(seq_len - 1):
        if output.graph_logits is None or not bool(output.graph_copy_target_mask[step].item()):
            continue
        target_token = batch.document.tokens[step + 1]
        if token_class is not None and target_token.token_class.value != token_class:
            continue
        step_kinds = set(candidate_kinds[step]) if step < len(candidate_kinds) else set()
        if required_kinds is not None and not step_kinds.intersection(required_kinds):
            continue
        target_id = int(batch.targets[step].item())
        values.append(float(output.graph_logits[step, target_id].exp().item()))
    if not values:
        return 0.0
    return sum(values) / len(values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", default="tests/fixtures/repo_graph_workspace/app/core.py")
    parser.add_argument("--repo-root")
    parser.add_argument("--report-path", action="append", default=[])
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    config = load_config(None)
    _, _, batch = build_batch(args.file_path, config)
    graph_index = build_repo_graph_index(
        args.file_path,
        config,
        repo_root=args.repo_root,
        report_paths=args.report_path,
    )
    model = PhaseACodeModel(config)
    model.set_repo_graph_index(graph_index)
    model.eval()

    warmup = 2
    iterations = 5
    for _ in range(warmup):
        with torch.no_grad():
            model(batch)

    start = time.perf_counter()
    hot_reads = 0.0
    cold_reads = 0.0
    maintenance = 0.0
    erm_reads = 0.0
    erm_writes = 0.0
    erm_overwrites = 0.0
    copy_hits = 0.0
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
    graph_symbol_recall = 0.0
    graph_test_recall = 0.0
    graph_diagnostic_recall = 0.0
    for _ in range(iterations):
        with torch.no_grad():
            output = model(batch)
        hot_reads += output.memory_stats["hot_reads"]
        cold_reads += output.memory_stats["cold_reads"]
        maintenance += output.memory_stats["maintenance_invocations"]
        erm_reads += output.memory_stats["erm_reads"]
        erm_writes += output.memory_stats["erm_writes"]
        erm_overwrites += output.memory_stats["erm_overwrites"]
        copy_hits += output.memory_stats["copy_target_hits"]
        episodic_hits += output.memory_stats["episodic_target_hits"]
        identifier_recall += _mean_target_probability(output, batch, "identifier")
        string_recall += _mean_target_probability(output, batch, "string")
        number_recall += _mean_target_probability(output, batch, "number")
        eem_reads += output.memory_stats["eem_reads"]
        chunks_finalized += output.memory_stats["chunks_finalized"]
        chunk_overhead += output.memory_stats["avg_chunk_overhead"]
        long_identifier_recall += _mean_episodic_target_probability(output, batch, "identifier")
        long_string_recall += _mean_episodic_target_probability(output, batch, "string")
        long_number_recall += _mean_episodic_target_probability(output, batch, "number")
        graph_reads += output.memory_stats["graph_reads"]
        graph_candidates += output.memory_stats["graph_candidates"]
        graph_symbol_recall += _mean_graph_target_probability(
            output,
            batch,
            token_class="identifier",
            required_kinds={"symbol", "function", "class"},
        )
        graph_test_recall += _mean_graph_target_probability(
            output,
            batch,
            required_kinds={"test"},
        )
        graph_diagnostic_recall += _mean_graph_target_probability(
            output,
            batch,
            required_kinds={"diagnostic"},
        )
    elapsed = time.perf_counter() - start
    tokens = len(batch.document.tokens) * iterations

    print(
        {
            "file": args.file_path,
            "tokens_per_sec": tokens / max(elapsed, 1e-6),
            "avg_hot_reads": hot_reads / iterations,
            "avg_cold_reads": cold_reads / iterations,
            "avg_maintenance_invocations": maintenance / iterations,
            "avg_erm_reads": erm_reads / iterations,
            "avg_erm_writes": erm_writes / iterations,
            "avg_erm_overwrites": erm_overwrites / iterations,
            "avg_copy_target_hits": copy_hits / iterations,
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
            "graph_symbol_recall": graph_symbol_recall / iterations,
            "graph_test_recall": graph_test_recall / iterations,
            "graph_diagnostic_recall": graph_diagnostic_recall / iterations,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
