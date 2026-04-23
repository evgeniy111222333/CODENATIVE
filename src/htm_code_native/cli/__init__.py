from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.featurizer import build_batch_from_document
from htm_code_native.data.types import TaskExample, TaskLabel, TrainingPhase
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.editing.planner import build_edit_request, run_edit_plan
from htm_code_native.losses.core import (
    autoregressive_loss,
    cap_auxiliary_losses,
    definition_use_loss,
    diagnostic_probe_loss,
    energy_penalty,
    episodic_pointer_loss,
    graph_copy_loss,
    hierarchical_consistency_loss,
    masked_autoregressive_loss,
    recent_copy_loss,
    route_consistency_loss,
    router_entropy_floor_loss,
    router_oracle_loss,
    routing_loss,
    sparse_retrieval_entropy_loss,
    symbol_link_loss,
)
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.tokenizer.boundary import BoundaryScheduler
from htm_code_native.tokenizer.tree_sitter_backend import detect_language, parse_source_document
from htm_code_native.training import (
    build_optimizer,
    build_probe_examples,
    build_repo_graph_index,
    build_task_batch,
    build_task_example,
    build_task_schedule,
    clip_grad_groups,
    default_task_examples,
    infer_task_label,
    resolve_repo_root,
    run_phase_exit_probes,
    schedule_maintenance,
    update_ar_ema,
)


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
    document = parse_source_document(source, file_path, language=detect_language(file_path))
    boundaries = BoundaryScheduler(max_level=config.hssm.max_level).build(document)
    batch = build_batch_from_document(document, boundaries, config, registry=registry)
    return document, boundaries, batch


def resolve_phase(args: argparse.Namespace, config: HTMCodeNativeConfig) -> TrainingPhase:
    return TrainingPhase(args.phase or config.model.training_phase)


def phase_loss_scales(phase: TrainingPhase) -> dict[str, float]:
    if phase == TrainingPhase.PHASE_A:
        return {
            "copy_r": 0.0,
            "ptr": 0.0,
            "graph": 0.0,
            "sym": 0.0,
            "definition_use": 0.0,
            "diagnostic": 0.0,
            "route": 0.0,
            "consistency": 0.0,
            "oracle": 0.0,
            "entropy": 0.0,
            "energy": 0.0,
        }
    if phase == TrainingPhase.PHASE_B:
        return {
            "copy_r": 1.0,
            "ptr": 0.0,
            "graph": 0.0,
            "sym": 0.0,
            "definition_use": 0.0,
            "diagnostic": 0.0,
            "route": 1.0,
            "consistency": 1.0,
            "oracle": 1.0,
            "entropy": 1.0,
            "energy": 1.0,
        }
    if phase == TrainingPhase.PHASE_C:
        return {
            "copy_r": 1.0,
            "ptr": 1.0,
            "graph": 0.0,
            "sym": 0.0,
            "definition_use": 0.0,
            "diagnostic": 0.0,
            "route": 1.0,
            "consistency": 1.0,
            "oracle": 1.0,
            "entropy": 1.0,
            "energy": 1.0,
        }
    if phase == TrainingPhase.PHASE_D:
        return {
            "copy_r": 1.0,
            "ptr": 1.0,
            "graph": 1.0,
            "sym": 1.0,
            "definition_use": 1.0,
            "diagnostic": 1.0,
            "route": 1.0,
            "consistency": 1.0,
            "oracle": 1.0,
            "entropy": 1.0,
            "energy": 1.0,
        }
    return {
        "copy_r": 1.0,
        "ptr": 1.0,
        "graph": 1.0,
        "sym": 1.0,
        "definition_use": 1.0,
        "diagnostic": 1.0,
        "route": 1.0,
        "consistency": 1.0,
        "oracle": 0.0,
        "entropy": 0.0,
        "energy": 1.0,
    }


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
    phase = resolve_phase(args, config)
    document, boundaries, _ = build_batch(args.file_path, config)
    graph_index = build_repo_graph_index(
        args.file_path,
        config,
        repo_root=args.repo_root,
        report_paths=args.report_path,
    )
    payload = {
        "summary": document.to_summary(),
        "phase": phase.value,
        "task_label": infer_task_label(args.file_path).value,
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
        "route_eligible_lanes": ["lm", "semantic_hot", "erm", "semantic_cold", "eem", "graph"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_run_forward(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    phase = resolve_phase(args, config)
    task_label = infer_task_label(args.file_path)
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
        output = model(batch, phase=phase, task_label=task_label)
        top_values, top_indices = torch.topk(output.logits[-1], k=min(5, output.logits.shape[-1]))
        copy_values, copy_indices = torch.topk(output.erm_logits[-1].exp(), k=min(5, output.erm_logits.shape[-1]))
        eem_values, eem_indices = torch.topk(output.eem_logits[-1].exp(), k=min(5, output.eem_logits.shape[-1]))
        graph_values, graph_indices = torch.topk(
            output.graph_logits[-1].exp(),
            k=min(5, output.graph_logits.shape[-1]),
        )
    payload = {
        "logits_shape": list(output.logits.shape),
        "registry_size": batch.registry_size,
        "phase": phase.value,
        "task_label": task_label.value,
        "diagnostics": output.diagnostics,
        "memory_stats": output.memory_stats,
        "router_last_weights": [float(value) for value in output.router_weights[-1].tolist()],
        "router_last_effective_weights": [float(value) for value in output.effective_router_weights[-1].tolist()],
        "router_last_oracle_weights": [float(value) for value in output.oracle_router_weights[-1].tolist()],
        "router_last_oracle_availability": [bool(value) for value in output.oracle_availability[-1].tolist()],
        "router_last_pre_mask": [bool(value) for value in output.router_pre_mask[-1].tolist()],
        "router_last_post_mask": [bool(value) for value in output.router_post_mask[-1].tolist()],
        "router_last_lane_entropies": [float(value) for value in output.lane_entropies[-1].tolist()],
        "warmup_beta_mean": float(output.warmup_beta.mean().item()),
        "collapse_detected": bool(output.collapse_detected.any().item()),
        "energy_proxy_mean": float(output.energy_proxy.mean().item()),
        "hard_gated_energy_savings": float(output.memory_stats["hard_gated_energy_savings"]),
        "graph_task_eligible_steps": float(output.memory_stats["graph_task_eligible_steps"]),
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


def command_edit_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    phase = resolve_phase(args, config)
    request = build_edit_request(
        file_path=args.file_path,
        instruction=args.instruction,
        repo_root=args.repo_root,
        report_paths=args.report_path,
        target_symbol=args.target_symbol,
        phase=phase,
        max_candidates=args.max_candidates or config.model.edit_max_candidates,
    )
    model = PhaseACodeModel(config)
    output = run_edit_plan(model, request, config)
    payload = {
        "request": {
            "file_path": output.request.file_path,
            "instruction": output.request.instruction,
            "repo_root": output.request.repo_root,
            "report_paths": list(output.request.report_paths),
            "target_symbol": output.request.target_symbol,
            "phase": output.request.phase,
            "max_candidates": output.request.max_candidates,
        },
        "selected_context": output.selected_context,
        "router_summary": output.router_summary,
        "span_candidates": [_edit_span_to_json(span) for span in output.span_candidates],
        "patch_candidates": [
            {
                "candidate_index": index,
                "span": _edit_span_to_json(candidate.span),
                "replacement_text": candidate.replacement_text,
                "valid": candidate.valid,
                "validation_errors": list(candidate.validation_errors),
                "score": candidate.score,
                "support_terms": list(candidate.support_terms),
                "diff_preview": candidate.diff_preview,
            }
            for index, candidate in enumerate(output.patch_plan.patch_candidates)
        ],
        "apply_results": [
            {
                "candidate_index": result.candidate_index,
                "span": _edit_span_to_json(result.span),
                "replacement_text": result.replacement_text,
                "patched_source_hash": result.patched_source_hash,
                "patched_source_length": result.patched_source_length,
                "diff_preview": result.diff_preview,
                "applied": result.applied,
                "valid": result.valid,
                "validation_errors": list(result.validation_errors),
                "syntax_error_count": result.syntax_error_count,
            }
            for result in output.apply_results
        ],
        "best_candidate_index": (
            output.best_apply_result.candidate_index
            if output.best_apply_result is not None
            else None
        ),
        "best_diff_preview": output.diff_preview,
        "validation_summary": output.validation_summary,
        "verification_summary": (
            {
                "candidate_count": output.verification_summary.candidate_count,
                "apply_success_rate": output.verification_summary.apply_success_rate,
                "syntax_valid_rate": output.verification_summary.syntax_valid_rate,
                "best_candidate_apply_valid": output.verification_summary.best_candidate_apply_valid,
            }
            if output.verification_summary is not None
            else None
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _edit_span_to_json(span) -> dict[str, object]:
    return {
        "start_byte": span.start_byte,
        "end_byte": span.end_byte,
        "token_start": span.token_start,
        "token_end": span.token_end,
        "node_type": span.node_type,
        "symbol_name": span.symbol_name,
        "score": span.score,
        "reasons": list(span.reasons),
        "source_text": span.source_text,
    }


def command_smoke_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    phase = resolve_phase(args, config)
    explicit_examples = _explicit_examples_from_args(args)

    if args.eval_only:
        probe_examples = explicit_examples or build_probe_examples(
            args.probe_set,
            repo_root=args.repo_root,
            report_paths=args.report_path,
        )
        model = PhaseACodeModel(config)
        report = run_phase_exit_probes(
            model,
            probe_examples,
            config,
            phase,
            probe_set=args.probe_set,
            max_steps=args.max_steps,
        )
        print(
            json.dumps(
                {
                    "phase": report.phase,
                    "probe_set": report.probe_set,
                    "passed": report.passed,
                    "metrics": report.metrics,
                    "failing_checks": list(report.failing_checks),
                    "example_count": report.example_count,
                },
                indent=2,
            )
        )
        return 0

    task_buckets = _task_buckets(explicit_examples, args)
    if not task_buckets:
        raise FileNotFoundError("No fixture files found for smoke training.")

    task_schedule = build_task_schedule(phase, task_buckets)
    registry = VocabularyRegistry(config.model.vocabulary_size)
    model = PhaseACodeModel(config)
    model.train()
    graph_cache: dict[tuple[str, tuple[str, ...]], object] = {}
    task_offsets = {label: 0 for label in task_buckets}
    loss_scales = phase_loss_scales(phase)
    ar_ema: float | None = None
    previous_hot_occupancy = 0.0
    previous_ar_loss = 0.0
    optimizer: torch.optim.Optimizer | None = None
    optimizer_warmup_active: bool | None = None
    step_count = args.max_steps or args.steps

    for step in range(step_count):
        task_label = task_schedule[step % len(task_schedule)]
        example_bucket = task_buckets[task_label]
        example = example_bucket[task_offsets[task_label] % len(example_bucket)]
        task_offsets[task_label] += 1
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

        warmup_active = (
            phase in {TrainingPhase.PHASE_B, TrainingPhase.PHASE_C, TrainingPhase.PHASE_D}
            and step < config.model.router_warmup_steps
        )
        if optimizer is None or optimizer_warmup_active != warmup_active:
            optimizer = build_optimizer(model, config, phase, warmup_active)
            optimizer_warmup_active = warmup_active

        maintenance_decision = schedule_maintenance(
            step,
            {
                "hot_occupancy": previous_hot_occupancy,
                "ar_loss": previous_ar_loss,
                "ar_ema": ar_ema if ar_ema is not None else previous_ar_loss,
            },
            config,
            phase,
        )
        maintenance_budget = (
            config.semantic_memory.maintenance_budget if maintenance_decision.should_consolidate else 0.0
        )
        if (
            maintenance_budget <= 0.0
            and phase in {TrainingPhase.PHASE_B, TrainingPhase.PHASE_C, TrainingPhase.PHASE_D, TrainingPhase.PHASE_E}
            and config.model.semantic_session_chunk_size > 0
        ):
            maintenance_budget = config.semantic_memory.maintenance_budget

        optimizer.zero_grad()
        output = model(
            task_batch.batch,
            reset_eem=(step == 0),
            phase=phase,
            task_label=example.task_label,
            global_step=step,
            maintenance_budget=maintenance_budget,
        )
        maintenance_decision.maintenance_invocations = int(output.memory_stats["maintenance_invocations"])
        output.auxiliary["task_supervision_mask"] = task_batch.supervision_mask
        output.auxiliary["infill_span"] = task_batch.infill_span
        output.auxiliary["maintenance_decision"] = maintenance_decision

        if example.task_label == TaskLabel.INFILL:
            ar_loss = masked_autoregressive_loss(
                output.logits,
                task_batch.batch.targets,
                task_batch.supervision_mask,
            )
        else:
            ar_loss = autoregressive_loss(output.logits, task_batch.batch.targets)

        hier_loss = hierarchical_consistency_loss(
            output.auxiliary["level_states"],
            output.auxiliary["lower_aggregates"],
            output.auxiliary["update_mask"],
        )
        sparse_loss = sparse_retrieval_entropy_loss(output.auxiliary["entropy_tensor"])
        copy_r_loss = recent_copy_loss(output.erm_logits, task_batch.batch.targets, output.copy_target_mask)
        ptr_loss = episodic_pointer_loss(output.eem_logits, task_batch.batch.targets, output.episodic_target_mask)
        graph_loss = graph_copy_loss(
            output.graph_logits,
            task_batch.batch.targets,
            output.graph_copy_target_mask,
            output.graph_copy_target_ids,
        )
        sym_loss = symbol_link_loss(
            output.auxiliary["graph_candidate_scores"],
            output.auxiliary["graph_candidate_ids"],
            output.auxiliary["graph_target_node_ids"],
        )
        def_use_loss = definition_use_loss(
            output.auxiliary["graph_candidate_scores"],
            output.auxiliary["graph_candidate_ids"],
            output.auxiliary["graph_candidate_kinds"],
            output.auxiliary["graph_target_node_ids"],
        )
        diag_loss = diagnostic_probe_loss(
            output.auxiliary["graph_candidate_scores"],
            output.auxiliary["graph_candidate_ids"],
            output.auxiliary["graph_candidate_kinds"],
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
        oracle_loss = router_oracle_loss(
            output.auxiliary["router_post_logits"],
            output.auxiliary["oracle_router_weights"],
            output.auxiliary["router_post_masks"],
        )
        entropy_floor_loss = router_entropy_floor_loss(
            output.router_weights,
            config.model.router_entropy_floor_min,
        )
        energy_loss = energy_penalty(
            output.energy_proxy,
            output.memory_stats["always_on_energy"],
        )

        probe_kind = example.metadata.get("probe_kind")
        weighted_aux_losses = {
            "copy_r": loss_scales["copy_r"] * config.model.copy_recent_weight * copy_r_loss,
            "ptr": loss_scales["ptr"] * config.model.copy_episodic_weight * ptr_loss,
            "graph": loss_scales["graph"] * config.model.graph_blend * graph_loss,
            "sym": loss_scales["sym"] * config.model.symbol_link_weight * sym_loss,
            "definition_use": loss_scales["definition_use"] * config.model.symbol_link_weight * def_use_loss,
            "diagnostic": loss_scales["diagnostic"] * config.model.graph_blend * diag_loss,
            "route": loss_scales["route"] * config.model.route_weight * route_loss,
            "consistency": loss_scales["consistency"] * config.model.route_consistency_weight * consistency_loss,
            "oracle": loss_scales["oracle"] * config.model.route_weight * oracle_loss,
            "entropy": loss_scales["entropy"] * config.model.router_entropy_floor_weight * entropy_floor_loss,
            "energy": loss_scales["energy"] * config.model.energy_weight * energy_loss,
        }
        if probe_kind == "definition_use":
            weighted_aux_losses["diagnostic"] = weighted_aux_losses["diagnostic"] * 0.0
        elif probe_kind in {"diagnostic_to_symbol", "edit_fix"}:
            weighted_aux_losses["definition_use"] = weighted_aux_losses["definition_use"] * 0.0
        capped_aux_losses = cap_auxiliary_losses(
            ar_loss,
            weighted_aux_losses,
            config.model.auxiliary_cap_ratio,
        )

        total_loss = ar_loss + (0.2 * hier_loss) + (0.01 * sparse_loss) + sum(capped_aux_losses.values())
        total_loss.backward()
        gradient_norms = clip_grad_groups(optimizer)
        optimizer.step()

        previous_ar_loss = float(ar_loss.item())
        ar_ema = update_ar_ema(ar_ema, previous_ar_loss, config.model.maintenance_ema_decay)
        previous_hot_occupancy = float(output.memory_stats["hot_occupancy"])

        print(
            json.dumps(
                {
                    "step": step,
                    "file": example.file_path,
                    "phase": phase.value,
                    "task_label": example.task_label.value,
                    "probe_kind": probe_kind,
                    "loss": float(total_loss.item()),
                    "ar_loss": float(ar_loss.item()),
                    "hier_loss": float(hier_loss.item()),
                    "sparse_loss": float(sparse_loss.item()),
                    "copy_r_loss": float(copy_r_loss.item()),
                    "ptr_loss": float(ptr_loss.item()),
                    "graph_loss": float(graph_loss.item()),
                    "sym_loss": float(sym_loss.item()),
                    "definition_use_loss": float(def_use_loss.item()),
                    "diagnostic_probe_loss": float(diag_loss.item()),
                    "route_loss": float(route_loss.item()),
                    "route_consistency_loss": float(consistency_loss.item()),
                    "router_oracle_loss": float(oracle_loss.item()),
                    "router_entropy_floor_loss": float(entropy_floor_loss.item()),
                    "energy_loss": float(energy_loss.item()),
                    "maintenance_decision": {
                        "should_consolidate": maintenance_decision.should_consolidate,
                        "reason": maintenance_decision.reason,
                        "hot_occupancy": maintenance_decision.hot_occupancy,
                        "ar_ema": maintenance_decision.ar_ema,
                        "ar_delta": maintenance_decision.ar_delta,
                        "maintenance_invocations": maintenance_decision.maintenance_invocations,
                    },
                    "gradient_norms": gradient_norms,
                    "copy_target_hits": float(output.memory_stats["copy_target_hits"]),
                    "episodic_target_hits": float(output.memory_stats["episodic_target_hits"]),
                    "graph_copy_hits": float(output.memory_stats["graph_copy_hits"]),
                    "graph_reads": float(output.memory_stats["graph_reads"]),
                    "symbol_link_hits": float(output.memory_stats["symbol_link_hits"]),
                    "avg_energy_proxy": float(output.memory_stats["avg_energy_proxy"]),
                    "warmup_beta_mean": float(output.warmup_beta.mean().item()),
                    "collapse_detected": bool(output.collapse_detected.any().item()),
                    "registry_size": task_batch.batch.registry_size,
                    "task_supervision_count": int(task_batch.supervision_mask.sum().item()),
                    "infill_span": task_batch.infill_span,
                    "phase_exit_probe_metrics": output.auxiliary["phase_exit_probe_metrics"],
                }
            )
        )
    return 0


def _explicit_examples_from_args(args: argparse.Namespace) -> list[TaskExample]:
    return [
        build_task_example(
            file_path,
            infer_task_label(file_path),
            repo_root=args.repo_root,
            report_paths=args.report_path,
        )
        for file_path in (args.files or [])
    ]


def _task_buckets(
    explicit_examples: list[TaskExample],
    args: argparse.Namespace,
) -> dict[TaskLabel, list[TaskExample]]:
    if explicit_examples:
        buckets: dict[TaskLabel, list[TaskExample]] = {}
        for example in explicit_examples:
            buckets.setdefault(example.task_label, []).append(example)
        return buckets
    return default_task_examples(repo_root=args.repo_root, report_paths=args.report_path)


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
    inspect_parser.add_argument("--phase")
    inspect_parser.set_defaults(func=command_inspect)

    inspect_structure_parser = subparsers.add_parser("inspect-structure")
    inspect_structure_parser.add_argument("file_path")
    inspect_structure_parser.add_argument("--config")
    inspect_structure_parser.add_argument("--repo-root")
    inspect_structure_parser.add_argument("--report-path", action="append", default=[])
    inspect_structure_parser.add_argument("--phase")
    inspect_structure_parser.set_defaults(func=command_inspect)

    forward_parser = subparsers.add_parser("run-forward")
    forward_parser.add_argument("file_path")
    forward_parser.add_argument("--config")
    forward_parser.add_argument("--repo-root")
    forward_parser.add_argument("--report-path", action="append", default=[])
    forward_parser.add_argument("--phase")
    forward_parser.set_defaults(func=command_run_forward)

    edit_parser = subparsers.add_parser("edit-plan")
    edit_parser.add_argument("file_path")
    edit_parser.add_argument("--config")
    edit_parser.add_argument("--repo-root")
    edit_parser.add_argument("--report-path", action="append", default=[])
    edit_parser.add_argument("--phase", default=TrainingPhase.PHASE_E.value)
    edit_parser.add_argument("--instruction", required=True)
    edit_parser.add_argument("--target-symbol")
    edit_parser.add_argument("--max-candidates", type=int)
    edit_parser.set_defaults(func=command_edit_plan)

    smoke_parser = subparsers.add_parser("smoke-train")
    smoke_parser.add_argument("--config")
    smoke_parser.add_argument("--steps", type=int, default=2)
    smoke_parser.add_argument("--max-steps", type=int)
    smoke_parser.add_argument("--eval-only", action="store_true")
    smoke_parser.add_argument("--probe-set", default="default")
    smoke_parser.add_argument("--repo-root")
    smoke_parser.add_argument("--report-path", action="append", default=[])
    smoke_parser.add_argument("--phase")
    smoke_parser.add_argument("files", nargs="*")
    smoke_parser.set_defaults(func=command_smoke_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
