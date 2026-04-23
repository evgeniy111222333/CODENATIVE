from __future__ import annotations

import ast
import difflib
import hashlib
import re
import time
from pathlib import Path

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.featurizer import build_batch_from_document
from htm_code_native.data.types import (
    AlignedDocument,
    EditRequest,
    EditRunOutput,
    EditTargetSpan,
    PatchApplyResult,
    PatchCandidate,
    PatchPlan,
    PatchVerificationSummary,
    RepositoryGraphIndex,
    TaskLabel,
    TrainingPhase,
)
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.tokenizer.boundary import BoundaryScheduler
from htm_code_native.tokenizer.tree_sitter_backend import detect_language, parse_source_document


INSTRUCTION_TERM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
QUOTED_TERM_RE = re.compile(r"""["']([^"'\n]{1,128})["']""")


def build_edit_request(
    file_path: str,
    instruction: str,
    repo_root: str | None = None,
    report_paths: list[str] | tuple[str, ...] | None = None,
    target_symbol: str | None = None,
    phase: TrainingPhase | str | None = None,
    max_candidates: int = 3,
) -> EditRequest:
    from htm_code_native.training.tasks import resolve_repo_root

    resolved_root = str(resolve_repo_root(file_path, repo_root)) if repo_root or Path(file_path).exists() else repo_root
    phase_name = phase.value if isinstance(phase, TrainingPhase) else phase
    return EditRequest(
        file_path=file_path,
        instruction=instruction,
        repo_root=resolved_root,
        report_paths=tuple(report_paths or ()),
        target_symbol=target_symbol,
        phase=phase_name,
        max_candidates=max(1, max_candidates),
    )


def run_edit_plan(
    model: PhaseACodeModel,
    request: EditRequest,
    config: HTMCodeNativeConfig,
) -> EditRunOutput:
    from htm_code_native.training.tasks import build_repo_graph_index

    document = _parse_document(request.file_path)
    boundaries = BoundaryScheduler(max_level=config.hssm.max_level).build(document)
    registry = VocabularyRegistry(config.model.vocabulary_size)
    batch = build_batch_from_document(document, boundaries, config, registry=registry)
    graph_index = build_repo_graph_index(
        request.file_path,
        config,
        repo_root=request.repo_root,
        report_paths=request.report_paths,
    )
    model.set_repo_graph_index(graph_index)
    requested_phase = TrainingPhase(request.phase) if request.phase is not None else TrainingPhase(config.model.training_phase)

    previous_mode = model.training
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        output = model(
            batch,
            phase=requested_phase,
            task_label=TaskLabel.EDIT_FIX,
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if previous_mode:
        model.train()

    relative_path = _relative_graph_path(request.file_path, graph_index)
    span_candidates = _rank_edit_spans(
        document=document,
        output=output,
        graph_index=graph_index,
        request=request,
        relative_path=relative_path,
    )[: request.max_candidates]
    patch_candidates = tuple(
        _build_patch_candidate(
            document=document,
            graph_index=graph_index,
            request=request,
            output=output,
            span=span,
        )
        for span in span_candidates
    )
    apply_results = tuple(
        dry_run_apply_patch_candidate(
            document,
            candidate,
            request.file_path,
            candidate_index=index,
        )
        for index, candidate in enumerate(patch_candidates)
    )
    best_candidate, best_apply_result = _select_best_patch_candidate(patch_candidates, apply_results)
    verification_summary = _summarize_apply_results(apply_results, best_apply_result)
    patch_plan = PatchPlan(
        file_path=request.file_path,
        original_source=document.source_text,
        patch_candidates=patch_candidates,
        best_candidate=best_candidate,
        validation_summary={
            "valid_candidates": sum(1 for candidate in patch_candidates if candidate.valid),
            "candidate_count": len(patch_candidates),
            "apply_success_rate": verification_summary.apply_success_rate,
            "syntax_valid_rate": verification_summary.syntax_valid_rate,
            "best_candidate_apply_valid": verification_summary.best_candidate_apply_valid,
        },
    )
    validation_summary = {
        "patch_candidate_valid_rate": (
            sum(1 for candidate in patch_candidates if candidate.valid) / max(len(patch_candidates), 1)
        ),
        "best_candidate_valid": bool(best_candidate.valid) if best_candidate is not None else False,
        "patch_apply_success_rate": verification_summary.apply_success_rate,
        "patch_syntax_valid_rate": verification_summary.syntax_valid_rate,
        "best_patch_apply_valid_rate": float(verification_summary.best_candidate_apply_valid),
        "latency_ms": latency_ms,
    }
    selected_context = {
        "file_summary": document.to_summary(),
        "graph_summary": graph_index.to_summary(),
        "diagnostic_terms": _diagnostic_terms_for_file(graph_index, relative_path),
        "target_symbol": request.target_symbol,
        "instruction_terms": _instruction_terms(request.instruction),
    }
    router_summary = {
        "phase": requested_phase.value,
        "task_label": TaskLabel.EDIT_FIX.value,
        "router_mean_weights": [float(value) for value in output.effective_router_weights.mean(dim=0).tolist()],
        "graph_invocations": float(output.memory_stats["graph_invocations"]),
        "eem_invocations": float(output.memory_stats["eem_invocations"]),
        "avg_energy_proxy": float(output.memory_stats["avg_energy_proxy"]),
        "latency_ms": latency_ms,
    }
    return EditRunOutput(
        request=request,
        selected_context=selected_context,
        router_summary=router_summary,
        span_candidates=tuple(span_candidates),
        patch_plan=patch_plan,
        diff_preview=best_candidate.diff_preview if best_candidate is not None else "",
        apply_results=apply_results,
        best_apply_result=best_apply_result,
        verification_summary=verification_summary,
        validation_summary=validation_summary,
    )


def render_unified_diff(plan: PatchPlan) -> str:
    if plan.best_candidate is None:
        return ""
    return plan.best_candidate.diff_preview


def dry_run_apply_patch_candidate(
    document: AlignedDocument,
    candidate: PatchCandidate,
    file_path: str,
    *,
    candidate_index: int = 0,
) -> PatchApplyResult:
    applied, patched_source, apply_errors = _apply_replacement_to_document(
        document=document,
        span=candidate.span,
        replacement_text=candidate.replacement_text,
    )
    validation_errors: tuple[str, ...] = apply_errors
    syntax_error_count = 0
    if applied:
        valid, validation_errors, syntax_error_count = _validate_patched_source(
            original_document=document,
            patched_source=patched_source,
            span=candidate.span,
            file_path=file_path,
            existing_errors=apply_errors,
        )
    else:
        valid = False
    return PatchApplyResult(
        candidate_index=candidate_index,
        span=candidate.span,
        replacement_text=candidate.replacement_text,
        patched_source_hash=_source_hash(patched_source),
        patched_source_length=len(patched_source.encode("utf-8")),
        diff_preview=_render_diff(document.source_text, patched_source, file_path) if applied else "",
        applied=applied,
        valid=valid,
        validation_errors=validation_errors,
        syntax_error_count=syntax_error_count,
    )


def dry_run_apply_patch_plan(plan: PatchPlan, file_path: str) -> tuple[PatchApplyResult, ...]:
    document = parse_source_document(plan.original_source, file_path, language=detect_language(file_path))
    return tuple(
        dry_run_apply_patch_candidate(
            document,
            candidate,
            file_path,
            candidate_index=index,
        )
        for index, candidate in enumerate(plan.patch_candidates)
    )


def _parse_document(file_path: str) -> AlignedDocument:
    source = Path(file_path).read_text(encoding="utf-8")
    return parse_source_document(source, file_path, language=detect_language(file_path))


def _relative_graph_path(file_path: str, graph_index: RepositoryGraphIndex) -> str:
    path = Path(file_path).resolve()
    root = Path(graph_index.root_path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _rank_edit_spans(
    *,
    document: AlignedDocument,
    output,
    graph_index: RepositoryGraphIndex,
    request: EditRequest,
    relative_path: str,
) -> list[EditTargetSpan]:
    edit_scores = torch.sigmoid(output.auxiliary["edit_token_scores"]).detach().cpu()
    graph_weights = output.effective_router_weights[:, 4].detach().cpu()
    eem_weights = output.effective_router_weights[:, 3].detach().cpu()
    graph_candidate_scores = output.auxiliary["graph_candidate_scores"]
    graph_candidate_ids = output.auxiliary["graph_candidate_ids"]
    graph_candidate_kinds = output.auxiliary["graph_candidate_kinds"]
    graph_candidate_names = output.auxiliary["graph_candidate_names"]
    diagnostic_terms = set(_diagnostic_terms_for_file(graph_index, relative_path))
    instruction_terms = set(_instruction_terms(request.instruction))
    target_symbol = request.target_symbol

    span_map: dict[tuple[int, int], EditTargetSpan] = {}
    for symbol in document.symbols:
        token_start, token_end = _byte_span_to_token_range(document, symbol.start_byte, symbol.end_byte)
        if token_start is None or token_end is None:
            continue
        reasons = ["symbol_boundary"]
        if target_symbol is not None and symbol.name == target_symbol:
            reasons.append("target_symbol")
        span = _make_span_candidate(
            document=document,
            token_start=token_start,
            token_end=token_end,
            node_type=symbol.kind,
            symbol_name=symbol.name,
            edit_scores=edit_scores,
            graph_weights=graph_weights,
            eem_weights=eem_weights,
            graph_candidate_scores=graph_candidate_scores,
            graph_candidate_ids=graph_candidate_ids,
            graph_candidate_kinds=graph_candidate_kinds,
            graph_candidate_names=graph_candidate_names,
            instruction_terms=instruction_terms,
            diagnostic_terms=diagnostic_terms,
            reasons=reasons,
            graph_index=graph_index,
        )
        _update_best_span(span_map, span)

    for node in document.ast_nodes:
        if node.node_type not in {"import_statement", "import_from_statement", "lexical_declaration"}:
            continue
        token_start, token_end = _byte_span_to_token_range(document, node.start_byte, node.end_byte)
        if token_start is None or token_end is None:
            continue
        span = _make_span_candidate(
            document=document,
            token_start=token_start,
            token_end=token_end,
            node_type=node.node_type,
            symbol_name=None,
            edit_scores=edit_scores,
            graph_weights=graph_weights,
            eem_weights=eem_weights,
            graph_candidate_scores=graph_candidate_scores,
            graph_candidate_ids=graph_candidate_ids,
            graph_candidate_kinds=graph_candidate_kinds,
            graph_candidate_names=graph_candidate_names,
            instruction_terms=instruction_terms,
            diagnostic_terms=diagnostic_terms,
            reasons=["ast_boundary"],
            graph_index=graph_index,
        )
        _update_best_span(span_map, span)

    for token in document.tokens:
        token_value = token.value.strip("\"'")
        if token.token_class.value not in {"identifier", "string", "number"}:
            continue
        if not (
            token_value in instruction_terms
            or token_value in diagnostic_terms
            or (target_symbol is not None and token.value == target_symbol)
            or _token_has_graph_match(token.index, graph_candidate_names, graph_candidate_ids, graph_index, target_symbol)
        ):
            continue
        reasons = ["token_match"]
        if target_symbol is not None and token_value == target_symbol.strip("\"'"):
            reasons.append("task_target")
        if {"import_statement", "import_from_statement"}.intersection(
            document.token_structures[token.index].ast_path
        ):
            reasons.append("import_context")
        span = _make_span_candidate(
            document=document,
            token_start=token.index,
            token_end=token.index + 1,
            node_type=document.token_structures[token.index].syntax_node_type,
            symbol_name=document.token_structures[token.index].symbol_name,
            edit_scores=edit_scores,
            graph_weights=graph_weights,
            eem_weights=eem_weights,
            graph_candidate_scores=graph_candidate_scores,
            graph_candidate_ids=graph_candidate_ids,
            graph_candidate_kinds=graph_candidate_kinds,
            graph_candidate_names=graph_candidate_names,
            instruction_terms=instruction_terms,
            diagnostic_terms=diagnostic_terms,
            reasons=reasons,
            graph_index=graph_index,
        )
        _update_best_span(span_map, span)

    ranked = sorted(span_map.values(), key=lambda span: span.score, reverse=True)
    return ranked


def _make_span_candidate(
    *,
    document: AlignedDocument,
    token_start: int,
    token_end: int,
    node_type: str | None,
    symbol_name: str | None,
    edit_scores: torch.Tensor,
    graph_weights: torch.Tensor,
    eem_weights: torch.Tensor,
    graph_candidate_scores: list[torch.Tensor],
    graph_candidate_ids: list[tuple[str, ...]],
    graph_candidate_kinds: list[tuple[str, ...]],
    graph_candidate_names: list[tuple[str, ...]],
    instruction_terms: set[str],
    diagnostic_terms: set[str],
    reasons: list[str],
    graph_index: RepositoryGraphIndex,
) -> EditTargetSpan:
    token_end = max(token_end, token_start + 1)
    span_tokens = document.tokens[token_start:token_end]
    start_byte = span_tokens[0].start_byte
    end_byte = span_tokens[-1].end_byte
    source_text = document.raw_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")
    span_edit_score = float(edit_scores[token_start:token_end].mean().item())
    span_graph_weight = float(graph_weights[token_start:token_end].mean().item())
    span_eem_weight = float(eem_weights[token_start:token_end].mean().item())
    span_graph_support = 0.0
    span_reasons = set(reasons)

    for step in range(token_start, token_end):
        if graph_candidate_scores[step].numel() > 0:
            span_graph_support += float(torch.softmax(graph_candidate_scores[step], dim=0).max().item())
        candidate_terms = (
            [
                term
                for candidate_id in graph_candidate_ids[step]
                for term in graph_index.nodes_by_id.get(candidate_id, graph_index.nodes[0]).copy_terms
            ]
            if graph_index.nodes
            else []
        )
        step_terms = {
            *graph_candidate_names[step],
            *candidate_terms,
        }
        if diagnostic_terms.intersection(step_terms):
            span_reasons.add("diagnostic_support")
        if instruction_terms.intersection(step_terms):
            span_reasons.add("instruction_support")
        if "diagnostic" in graph_candidate_kinds[step]:
            span_reasons.add("diagnostic_candidate")
        if any(kind in {"symbol", "function", "class"} for kind in graph_candidate_kinds[step]):
            span_reasons.add("symbol_candidate")

    score = span_edit_score
    score += 0.35 * span_graph_weight
    score += 0.25 * span_eem_weight
    score += 0.2 * span_graph_support / max(token_end - token_start, 1)
    if symbol_name is not None:
        score += 0.2
    if "task_target" in span_reasons:
        score += 1.25
    if "import_context" in span_reasons:
        score -= 0.5
    if "diagnostic_support" in span_reasons:
        score += 0.6
    if "instruction_support" in span_reasons:
        score += 0.4

    return EditTargetSpan(
        start_byte=start_byte,
        end_byte=end_byte,
        token_start=token_start,
        token_end=token_end,
        node_type=node_type,
        symbol_name=symbol_name,
        score=score,
        reasons=tuple(sorted(span_reasons)),
        source_text=source_text,
    )


def _build_patch_candidate(
    *,
    document: AlignedDocument,
    graph_index: RepositoryGraphIndex,
    request: EditRequest,
    output,
    span: EditTargetSpan,
) -> PatchCandidate:
    replacement_text, support_terms = _propose_replacement(
        document=document,
        graph_index=graph_index,
        request=request,
        output=output,
        span=span,
    )
    applied, patched_source, apply_errors = _apply_replacement_to_document(
        document=document,
        span=span,
        replacement_text=replacement_text,
    )
    if applied:
        valid, validation_errors, _syntax_errors = _validate_patched_source(
            original_document=document,
            patched_source=patched_source,
            span=span,
            file_path=request.file_path,
            existing_errors=apply_errors,
        )
    else:
        valid = False
        validation_errors = apply_errors
    diff_preview = _render_diff(document.source_text, patched_source, request.file_path)
    score = span.score + (0.5 if valid else -0.25) + (0.05 * len(support_terms))
    return PatchCandidate(
        span=span,
        replacement_text=replacement_text,
        patched_source=patched_source,
        diff_preview=diff_preview,
        valid=valid,
        validation_errors=validation_errors,
        score=score,
        support_terms=support_terms,
    )


def _propose_replacement(
    *,
    document: AlignedDocument,
    graph_index: RepositoryGraphIndex,
    request: EditRequest,
    output,
    span: EditTargetSpan,
) -> tuple[str, tuple[str, ...]]:
    span_tokens = document.tokens[span.token_start:span.token_end]
    original_text = span.source_text
    support_terms: list[str] = []
    instruction_literals = QUOTED_TERM_RE.findall(request.instruction)
    instruction_terms = _instruction_terms(request.instruction)
    support_terms.extend(instruction_literals)
    support_terms.extend([term for term in instruction_terms if term != original_text.strip("\"'")])

    for step in range(span.token_start, span.token_end):
        for candidate_id in output.auxiliary["graph_candidate_ids"][step]:
            node = graph_index.nodes_by_id.get(candidate_id)
            if node is None:
                continue
            support_terms.extend(node.copy_terms)
            support_terms.append(node.name)

    relative_path = _relative_graph_path(request.file_path, graph_index)
    support_terms.extend(_diagnostic_terms_for_file(graph_index, relative_path))

    logits_index = max(0, span.token_start - 1)
    top_ids = torch.topk(output.logits[logits_index], k=min(8, output.logits.shape[-1])).indices.tolist()
    vocab_snapshot = output.auxiliary["vocabulary_snapshot"]
    support_terms.extend(vocab_snapshot.token_for_id(token_id) for token_id in top_ids)
    unique_terms = tuple(_unique_terms(term for term in support_terms if term))

    if span_tokens and len(span_tokens) == 1:
        token = span_tokens[0]
        stripped = token.value.strip("\"'")
        if token.token_class.value == "identifier":
            string_terms = [term for term in unique_terms if _looks_like_literal(term) and term != stripped]
            if string_terms:
                return _wrap_string(string_terms[0], token.value), unique_terms
            identifier_terms = [term for term in unique_terms if _looks_like_identifier(term) and term != stripped]
            if identifier_terms:
                return identifier_terms[0], unique_terms
        if token.token_class.value == "string":
            string_terms = [term for term in unique_terms if term != stripped]
            if string_terms:
                return _wrap_string(string_terms[0], token.value), unique_terms

    string_terms = [term for term in unique_terms if _looks_like_literal(term)]
    if string_terms:
        return _wrap_string(string_terms[0], original_text), unique_terms
    identifier_terms = [term for term in unique_terms if _looks_like_identifier(term)]
    if identifier_terms:
        return identifier_terms[0], unique_terms
    return original_text, unique_terms


def _validate_patch(
    *,
    original_document: AlignedDocument,
    patched_source: str,
    span: EditTargetSpan,
    file_path: str,
) -> tuple[bool, tuple[str, ...]]:
    valid, errors, _syntax_errors = _validate_patched_source(
        original_document=original_document,
        patched_source=patched_source,
        span=span,
        file_path=file_path,
    )
    return valid, errors


def _apply_replacement_to_document(
    *,
    document: AlignedDocument,
    span: EditTargetSpan,
    replacement_text: str,
) -> tuple[bool, str, tuple[str, ...]]:
    errors: list[str] = []
    raw = document.raw_bytes
    if span.start_byte < 0 or span.end_byte > len(raw) or span.start_byte >= span.end_byte:
        errors.append("span_bounds_invalid")
        return False, document.source_text, tuple(errors)
    if replacement_text == "":
        errors.append("empty_replacement")
        return False, document.source_text, tuple(errors)
    patched_raw = (
        raw[: span.start_byte]
        + replacement_text.encode("utf-8")
        + raw[span.end_byte :]
    )
    try:
        patched_source = patched_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"utf8_decode:{exc.reason}")
        return False, document.source_text, tuple(errors)
    if patched_source == document.source_text:
        errors.append("unchanged_patch")
        return False, patched_source, tuple(errors)
    return True, patched_source, ()


def _validate_patched_source(
    *,
    original_document: AlignedDocument,
    patched_source: str,
    span: EditTargetSpan,
    file_path: str,
    existing_errors: tuple[str, ...] = (),
) -> tuple[bool, tuple[str, ...], int]:
    errors: list[str] = list(existing_errors)
    syntax_error_count = 0
    language = detect_language(file_path)
    patched_document = parse_source_document(patched_source, file_path, language=language)
    if patched_document.parse_document is not None and patched_document.parse_document.error_count > 0:
        syntax_error_count += patched_document.parse_document.error_count
        errors.append(f"tree_sitter_errors:{patched_document.parse_document.error_count}")
    if language == "python":
        try:
            ast.parse(patched_source)
        except SyntaxError as exc:
            syntax_error_count += 1
            errors.append(f"python_ast:{exc.msg}")

    enclosing_symbol = None
    for symbol in original_document.symbols:
        if symbol.start_byte <= span.start_byte and symbol.end_byte >= span.end_byte:
            enclosing_symbol = symbol
            break
    if enclosing_symbol is not None:
        if not any(symbol.name == enclosing_symbol.name and symbol.kind == enclosing_symbol.kind for symbol in patched_document.symbols):
            errors.append("enclosing_symbol_missing")
    return not errors, tuple(errors), syntax_error_count


def _select_best_patch_candidate(
    patch_candidates: tuple[PatchCandidate, ...],
    apply_results: tuple[PatchApplyResult, ...],
) -> tuple[PatchCandidate | None, PatchApplyResult | None]:
    if not patch_candidates:
        return None, None
    pairs = list(zip(patch_candidates, apply_results, strict=False))
    valid_pairs = [
        (candidate, result)
        for candidate, result in pairs
        if candidate.valid and result.valid
    ]
    if valid_pairs:
        return max(valid_pairs, key=lambda item: item[0].score)
    fallback = max(
        pairs,
        key=lambda item: item[0].score + (0.25 if item[1].valid else -0.25),
    )
    return fallback


def _summarize_apply_results(
    apply_results: tuple[PatchApplyResult, ...],
    best_apply_result: PatchApplyResult | None,
) -> PatchVerificationSummary:
    candidate_count = len(apply_results)
    return PatchVerificationSummary(
        candidate_count=candidate_count,
        apply_success_rate=sum(1 for result in apply_results if result.applied) / max(candidate_count, 1),
        syntax_valid_rate=(
            sum(1 for result in apply_results if result.applied and result.syntax_error_count == 0)
            / max(candidate_count, 1)
        ),
        best_candidate_apply_valid=bool(best_apply_result.valid) if best_apply_result is not None else False,
    )


def _source_hash(source: str) -> str:
    return hashlib.blake2b(source.encode("utf-8"), digest_size=16).hexdigest()


def _render_diff(original_source: str, patched_source: str, file_path: str) -> str:
    original_lines = original_source.splitlines(keepends=True)
    patched_lines = patched_source.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile=file_path,
            tofile=file_path,
            lineterm="",
        )
    )


def _byte_span_to_token_range(
    document: AlignedDocument,
    start_byte: int,
    end_byte: int,
) -> tuple[int | None, int | None]:
    token_indices = [
        token.index
        for token in document.tokens
        if token.end_byte > start_byte and token.start_byte < end_byte
    ]
    if not token_indices:
        return None, None
    return min(token_indices), max(token_indices) + 1


def _update_best_span(span_map: dict[tuple[int, int], EditTargetSpan], span: EditTargetSpan) -> None:
    key = (span.start_byte, span.end_byte)
    previous = span_map.get(key)
    if previous is None or span.score > previous.score:
        span_map[key] = span


def _instruction_terms(instruction: str) -> list[str]:
    terms = [match.group(0) for match in INSTRUCTION_TERM_RE.finditer(instruction)]
    return _unique_terms(term for term in terms)


def _diagnostic_terms_for_file(graph_index: RepositoryGraphIndex, relative_path: str) -> tuple[str, ...]:
    node_ids = graph_index.node_ids_by_file.get(relative_path, ())
    terms: list[str] = []
    for node_id in node_ids:
        node = graph_index.nodes_by_id[node_id]
        if node.kind != "diagnostic":
            continue
        terms.extend(node.copy_terms)
        terms.append(node.name)
    return tuple(_unique_terms(term for term in terms if term))


def _token_has_graph_match(
    token_index: int,
    graph_candidate_names: list[tuple[str, ...]],
    graph_candidate_ids: list[tuple[str, ...]],
    graph_index: RepositoryGraphIndex,
    target_symbol: str | None,
) -> bool:
    if target_symbol is not None and target_symbol in graph_candidate_names[token_index]:
        return True
    for node_id in graph_candidate_ids[token_index]:
        node = graph_index.nodes_by_id.get(node_id)
        if node is None:
            continue
        if target_symbol is not None and target_symbol in node.copy_terms:
            return True
    return False


def _looks_like_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))


def _looks_like_literal(value: str) -> bool:
    return (not _looks_like_identifier(value)) or ("_" in value and value.lower() == value)


def _wrap_string(value: str, original_text: str) -> str:
    if original_text.startswith("'") and original_text.endswith("'"):
        return f"'{value}'"
    return f"\"{value}\""


def _unique_terms(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip().strip("\"'")
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered
