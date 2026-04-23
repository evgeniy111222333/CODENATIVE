from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.featurizer import build_batch_from_document
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.losses.core import (
    autoregressive_loss,
    energy_penalty,
    episodic_pointer_loss,
    graph_copy_loss,
    hierarchical_consistency_loss,
    recent_copy_loss,
    route_consistency_loss,
    routing_loss,
    sparse_retrieval_entropy_loss,
    symbol_link_loss,
)
from htm_code_native.memory.repo_graph import RepositoryGraphIndexer
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.tokenizer.boundary import BoundaryScheduler
from htm_code_native.tokenizer.tree_sitter_backend import detect_language, parse_source_document


def load_config(config_path: str | None) -> HTMCodeNativeConfig:
    if config_path:
        return HTMCodeNativeConfig.from_yaml(config_path)
    default_path = Path("configs/phase_a.yaml")
    if default_path.exists():
        return HTMCodeNativeConfig.from_yaml(default_path)
    return HTMCodeNativeConfig.default()


def resolve_repo_root(file_path: str, repo_root: str | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    current = Path(file_path).resolve()
    start = current if current.is_dir() else current.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def default_report_paths(repo_root: Path) -> list[str]:
    report_dir = repo_root / "reports"
    if not report_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".xml", ".json", ".txt", ".log"}
    ]


def build_repo_graph_index(
    file_path: str,
    config: HTMCodeNativeConfig,
    repo_root: str | None = None,
    report_paths: list[str] | None = None,
):
    resolved_root = resolve_repo_root(file_path, repo_root)
    effective_reports = report_paths if report_paths else default_report_paths(resolved_root)
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    return indexer.build(resolved_root, report_paths=effective_reports)


def build_batch(
    file_path: str,
    config: HTMCodeNativeConfig,
    registry: VocabularyRegistry | None = None,
):
    source = Path(file_path).read_text(encoding="utf-8")
    scheduler = BoundaryScheduler(max_level=config.hssm.max_level)
    document = parse_source_document(source, file_path, language=detect_language(file_path))
    boundaries = scheduler.build(document)
    batch = build_batch_from_document(document, boundaries, config, registry=registry)
    return document, boundaries, batch


def command_tokenize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    document, _, batch = build_batch(args.file_path, config)
    tokens = [
        {
            "index": token.index,
            "registry_id": int(batch.token_ids[token.index].item()),
            "class": token.token_class.value,
            "type": token.token_type,
            "value": token.value,
            "start_byte": token.start_byte,
            "end_byte": token.end_byte,
            "tags": list(token.structural_tags),
        }
        for token in document.tokens[: args.limit]
    ]
    print(
        json.dumps(
            {
                "summary": document.to_summary(),
                "registry_size": batch.registry_size,
                "tokens": tokens,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    document, boundaries, _ = build_batch(args.file_path, config)
    graph_index = build_repo_graph_index(
        args.file_path,
        config,
        repo_root=args.repo_root,
        report_paths=args.report_path,
    )
    payload = {
        "summary": document.to_summary(),
        "parser_backend": document.parse_document.parser_backend if document.parse_document else None,
        "parse_errors": list(document.parse_document.error_messages) if document.parse_document else [],
        "symbols": [
            {
                "symbol_id": symbol.symbol_id,
                "name": symbol.name,
                "kind": symbol.kind,
                "scope": list(symbol.scope_path),
            }
            for symbol in document.symbols
        ],
        "boundary_counts": {
            str(level): int(sum(mask)) for level, mask in boundaries.level_events.items()
        },
        "graph_summary": graph_index.to_summary(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_run_forward(args: argparse.Namespace) -> int:
    config = load_config(args.config)
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
    with torch.no_grad():
        output = model(batch)
        top_values, top_indices = torch.topk(output.logits[-1], k=min(5, output.logits.shape[-1]))
        copy_values, copy_indices = torch.topk(
            output.erm_logits[-1].exp(),
            k=min(5, output.erm_logits.shape[-1]),
        )
        eem_values, eem_indices = torch.topk(
            output.eem_logits[-1].exp(),
            k=min(5, output.eem_logits.shape[-1]),
        )
        graph_values, graph_indices = torch.topk(
            output.graph_logits[-1].exp(),
            k=min(5, output.graph_logits.shape[-1]),
        )
    payload = {
        "logits_shape": list(output.logits.shape),
        "registry_size": batch.registry_size,
        "diagnostics": output.diagnostics,
        "memory_stats": output.memory_stats,
        "router_last_weights": [float(value) for value in output.router_weights[-1].tolist()],
        "router_last_pre_mask": [bool(value) for value in output.router_pre_mask[-1].tolist()],
        "router_last_post_mask": [bool(value) for value in output.router_post_mask[-1].tolist()],
        "router_last_lane_entropies": [float(value) for value in output.lane_entropies[-1].tolist()],
        "energy_proxy_mean": float(output.energy_proxy.mean().item()),
        "last_step_top_ids": top_indices.tolist(),
        "last_step_top_scores": [float(value) for value in top_values.tolist()],
        "last_step_top_copy_ids": copy_indices.tolist(),
        "last_step_top_copy_scores": [float(value) for value in copy_values.tolist()],
        "last_step_top_eem_ids": eem_indices.tolist(),
        "last_step_top_eem_scores": [float(value) for value in eem_values.tolist()],
        "last_step_top_graph_ids": graph_indices.tolist(),
        "last_step_top_graph_scores": [float(value) for value in graph_values.tolist()],
        "last_graph_candidate_ids": list(output.auxiliary["graph_candidate_ids"][-1]),
        "last_graph_candidate_kinds": list(output.auxiliary["graph_candidate_kinds"][-1]),
        "last_graph_candidate_names": list(output.auxiliary["graph_candidate_names"][-1]),
        "graph_fusion_norm": float(output.graph_contexts[-1].norm().item()),
        "graph_summary": graph_index.to_summary(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_smoke_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    input_files = args.files or sorted(str(path) for path in Path("tests/fixtures").glob("*.py"))
    if not input_files:
        raise FileNotFoundError("No fixture files found for smoke training.")

    registry = VocabularyRegistry(config.model.vocabulary_size)
    model = PhaseACodeModel(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    graph_cache: dict[tuple[str, tuple[str, ...]], object] = {}

    for step in range(args.steps):
        file_path = input_files[step % len(input_files)]
        _, _, batch = build_batch(file_path, config, registry=registry)
        resolved_root = str(resolve_repo_root(file_path, args.repo_root))
        report_paths = tuple(sorted(args.report_path or default_report_paths(Path(resolved_root))))
        cache_key = (resolved_root, report_paths)
        if cache_key not in graph_cache:
            graph_cache[cache_key] = build_repo_graph_index(
                file_path,
                config,
                repo_root=resolved_root,
                report_paths=list(report_paths),
            )
        model.set_repo_graph_index(graph_cache[cache_key])
        optimizer.zero_grad()
        output = model(batch, reset_eem=(step == 0))
        ar_loss = autoregressive_loss(output.logits, batch.targets)
        hier_loss = hierarchical_consistency_loss(
            output.auxiliary["level_states"],
            output.auxiliary["lower_aggregates"],
            output.auxiliary["update_mask"],
        )
        sparse_loss = sparse_retrieval_entropy_loss(output.auxiliary["entropy_tensor"])
        copy_r_loss = recent_copy_loss(output.erm_logits, batch.targets, output.copy_target_mask)
        ptr_loss = episodic_pointer_loss(
            output.eem_logits,
            batch.targets,
            output.episodic_target_mask,
        )
        graph_loss = graph_copy_loss(
            output.graph_logits,
            batch.targets,
            output.graph_copy_target_mask,
        )
        sym_loss = symbol_link_loss(
            output.auxiliary["graph_candidate_scores"],
            output.auxiliary["graph_candidate_ids"],
            output.auxiliary["graph_target_node_ids"],
        )
        route_loss = routing_loss(
            output.auxiliary["router_post_logits"],
            output.auxiliary["route_teacher_indices"],
        )
        consistency_loss = route_consistency_loss(
            output.auxiliary["router_pre_logits"],
            output.auxiliary["route_teacher_expensive"],
        )
        energy_loss = energy_penalty(
            output.energy_proxy,
            output.memory_stats["always_on_energy"],
        )
        total_loss = (
            ar_loss
            + 0.2 * hier_loss
            + 0.01 * sparse_loss
            + config.model.copy_recent_weight * copy_r_loss
            + config.model.copy_episodic_weight * ptr_loss
            + config.model.graph_blend * graph_loss
            + config.model.symbol_link_weight * sym_loss
            + config.model.route_weight * route_loss
            + config.model.route_consistency_weight * consistency_loss
            + config.model.energy_weight * energy_loss
        )
        total_loss.backward()
        optimizer.step()
        print(
            json.dumps(
                {
                    "step": step,
                    "file": input_files[step % len(input_files)],
                    "loss": float(total_loss.item()),
                    "ar_loss": float(ar_loss.item()),
                    "hier_loss": float(hier_loss.item()),
                    "sparse_loss": float(sparse_loss.item()),
                    "copy_r_loss": float(copy_r_loss.item()),
                    "ptr_loss": float(ptr_loss.item()),
                    "graph_loss": float(graph_loss.item()),
                    "sym_loss": float(sym_loss.item()),
                    "route_loss": float(route_loss.item()),
                    "route_consistency_loss": float(consistency_loss.item()),
                    "energy_loss": float(energy_loss.item()),
                    "copy_target_hits": float(output.memory_stats["copy_target_hits"]),
                    "episodic_target_hits": float(output.memory_stats["episodic_target_hits"]),
                    "graph_copy_hits": float(output.memory_stats["graph_copy_hits"]),
                    "graph_reads": float(output.memory_stats["graph_reads"]),
                    "symbol_link_hits": float(output.memory_stats["symbol_link_hits"]),
                    "avg_energy_proxy": float(output.memory_stats["avg_energy_proxy"]),
                    "registry_size": batch.registry_size,
                }
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="htm_code_native")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenize_parser = subparsers.add_parser("tokenize")
    tokenize_parser.add_argument("file_path")
    tokenize_parser.add_argument("--config")
    tokenize_parser.add_argument("--limit", type=int, default=30)
    tokenize_parser.set_defaults(func=command_tokenize)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("file_path")
    inspect_parser.add_argument("--config")
    inspect_parser.add_argument("--repo-root")
    inspect_parser.add_argument("--report-path", action="append", default=[])
    inspect_parser.set_defaults(func=command_inspect)

    inspect_structure_parser = subparsers.add_parser("inspect-structure")
    inspect_structure_parser.add_argument("file_path")
    inspect_structure_parser.add_argument("--config")
    inspect_structure_parser.add_argument("--repo-root")
    inspect_structure_parser.add_argument("--report-path", action="append", default=[])
    inspect_structure_parser.set_defaults(func=command_inspect)

    forward_parser = subparsers.add_parser("run-forward")
    forward_parser.add_argument("file_path")
    forward_parser.add_argument("--config")
    forward_parser.add_argument("--repo-root")
    forward_parser.add_argument("--report-path", action="append", default=[])
    forward_parser.set_defaults(func=command_run_forward)

    smoke_parser = subparsers.add_parser("smoke-train")
    smoke_parser.add_argument("--config")
    smoke_parser.add_argument("--steps", type=int, default=2)
    smoke_parser.add_argument("--repo-root")
    smoke_parser.add_argument("--report-path", action="append", default=[])
    smoke_parser.add_argument("files", nargs="*")
    smoke_parser.set_defaults(func=command_smoke_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
