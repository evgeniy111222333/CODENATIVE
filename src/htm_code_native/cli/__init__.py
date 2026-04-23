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
    episodic_pointer_loss,
    hierarchical_consistency_loss,
    recent_copy_loss,
    sparse_retrieval_entropy_loss,
)
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.tokenizer.boundary import BoundaryScheduler
from htm_code_native.tokenizer.python_tokenizer import PythonTokenizer
from htm_code_native.tokenizer.structure import PythonStructureExtractor


def load_config(config_path: str | None) -> HTMCodeNativeConfig:
    if config_path:
        return HTMCodeNativeConfig.from_yaml(config_path)
    default_path = Path("configs/phase_a.yaml")
    if default_path.exists():
        return HTMCodeNativeConfig.from_yaml(default_path)
    return HTMCodeNativeConfig.default()


def build_batch(
    file_path: str,
    config: HTMCodeNativeConfig,
    registry: VocabularyRegistry | None = None,
):
    source = Path(file_path).read_text(encoding="utf-8")
    tokenizer = PythonTokenizer()
    structure = PythonStructureExtractor()
    scheduler = BoundaryScheduler(max_level=config.hssm.max_level)
    document = structure.enrich(tokenizer.encode(source, file_path))
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
    payload = {
        "summary": document.to_summary(),
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
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_run_forward(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _, _, batch = build_batch(args.file_path, config)
    model = PhaseACodeModel(config)
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
    payload = {
        "logits_shape": list(output.logits.shape),
        "registry_size": batch.registry_size,
        "diagnostics": output.diagnostics,
        "memory_stats": output.memory_stats,
        "last_step_top_ids": top_indices.tolist(),
        "last_step_top_scores": [float(value) for value in top_values.tolist()],
        "last_step_top_copy_ids": copy_indices.tolist(),
        "last_step_top_copy_scores": [float(value) for value in copy_values.tolist()],
        "last_step_top_eem_ids": eem_indices.tolist(),
        "last_step_top_eem_scores": [float(value) for value in eem_values.tolist()],
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

    for step in range(args.steps):
        _, _, batch = build_batch(input_files[step % len(input_files)], config, registry=registry)
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
        total_loss = (
            ar_loss
            + 0.2 * hier_loss
            + 0.01 * sparse_loss
            + config.model.copy_recent_weight * copy_r_loss
            + config.model.copy_episodic_weight * ptr_loss
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
                    "copy_target_hits": float(output.memory_stats["copy_target_hits"]),
                    "episodic_target_hits": float(output.memory_stats["episodic_target_hits"]),
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
    inspect_parser.set_defaults(func=command_inspect)

    inspect_structure_parser = subparsers.add_parser("inspect-structure")
    inspect_structure_parser.add_argument("file_path")
    inspect_structure_parser.add_argument("--config")
    inspect_structure_parser.set_defaults(func=command_inspect)

    forward_parser = subparsers.add_parser("run-forward")
    forward_parser.add_argument("file_path")
    forward_parser.add_argument("--config")
    forward_parser.set_defaults(func=command_run_forward)

    smoke_parser = subparsers.add_parser("smoke-train")
    smoke_parser.add_argument("--config")
    smoke_parser.add_argument("--steps", type=int, default=2)
    smoke_parser.add_argument("files", nargs="*")
    smoke_parser.set_defaults(func=command_smoke_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
