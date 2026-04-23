from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.featurizer import build_batch_from_document
from htm_code_native.data.types import (
    AlignedDocument,
    BoundaryEvents,
    CodeToken,
    PhaseABatch,
    SyntaxStateFeatures,
    TaskBatch,
    TaskExample,
    TaskLabel,
    TokenStructureInfo,
    TrainingPhase,
)
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.memory.repo_graph import RepositoryGraphIndexer
from htm_code_native.tokenizer.boundary import BoundaryScheduler
from htm_code_native.tokenizer.tree_sitter_backend import detect_language, parse_source_document


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
    report_paths: list[str] | tuple[str, ...] | None = None,
):
    resolved_root = resolve_repo_root(file_path, repo_root)
    effective_reports = list(report_paths) if report_paths else default_report_paths(resolved_root)
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    return indexer.build(resolved_root, report_paths=effective_reports)


def parse_task_document(file_path: str) -> tuple[AlignedDocument, object]:
    source = Path(file_path).read_text(encoding="utf-8")
    scheduler = BoundaryScheduler()
    document = parse_source_document(source, file_path, language=detect_language(file_path))
    boundaries = scheduler.build(document)
    return document, boundaries


def infer_task_label(file_path: str) -> TaskLabel:
    lowered = str(file_path).replace("\\", "/").lower()
    if "edit" in lowered or "patch" in lowered:
        return TaskLabel.EDIT_FIX
    if "repo_graph_workspace" in lowered:
        return TaskLabel.REPO_GRAPH
    if "recent_copy" in lowered:
        return TaskLabel.RECENT_COPY
    if "episodic" in lowered:
        return TaskLabel.EPISODIC_RECALL
    if "infill" in lowered:
        return TaskLabel.INFILL
    return TaskLabel.AR


def build_task_example(
    file_path: str,
    task_label: TaskLabel | str | None = None,
    *,
    repo_root: str | None = None,
    report_paths: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, object] | None = None,
) -> TaskExample:
    label = TaskLabel(task_label) if isinstance(task_label, str) else (task_label or infer_task_label(file_path))
    resolved_root = (
        str(resolve_repo_root(file_path, repo_root))
        if label in {TaskLabel.REPO_GRAPH, TaskLabel.EDIT_FIX}
        else repo_root
    )
    effective_reports = tuple(report_paths) if report_paths else tuple(default_report_paths(Path(resolved_root))) if resolved_root else ()
    return TaskExample(
        file_path=file_path,
        task_label=label,
        repo_root=resolved_root,
        report_paths=effective_reports,
        metadata=dict(metadata or {}),
    )


def build_task_batch(
    example: TaskExample,
    config: HTMCodeNativeConfig,
    registry: VocabularyRegistry | None = None,
) -> TaskBatch:
    active_registry = registry or VocabularyRegistry(capacity=config.model.vocabulary_size)
    document, boundaries = parse_task_document(example.file_path)
    if example.task_label == TaskLabel.INFILL:
        infill_start, infill_end = _select_infill_span(document)
        masked_document = _mask_document_span(document, infill_start, infill_end)
        batch = build_batch_from_document(masked_document, boundaries, config, registry=active_registry)
        supervision_mask = torch.zeros(len(masked_document.tokens), dtype=torch.bool)
        for token_index in range(infill_start, infill_end):
            supervision_index = token_index - 1
            if supervision_index >= 0:
                supervision_mask[supervision_index] = True
        metadata = {
            "task_label": example.task_label.value,
            "probe_kind": example.metadata.get("probe_kind"),
            "masked_token_range": (infill_start, infill_end),
        }
        batch.task_metadata.update(metadata)
        return TaskBatch(
            example=example,
            batch=batch,
            supervision_mask=supervision_mask,
            infill_span=(infill_start, infill_end),
            edit_target_span=None,
            replacement_text=None,
            metadata=metadata,
        )
    if example.task_label == TaskLabel.EDIT_FIX:
        edit_start, edit_end, replacement_text = _select_edit_target(document, example)
        masked_document = _mask_document_span(document, edit_start, edit_end)
        batch = build_batch_from_document(masked_document, boundaries, config, registry=active_registry)
        supervision_mask = torch.zeros(len(masked_document.tokens), dtype=torch.bool)
        target_token_mask = torch.zeros(len(masked_document.tokens), dtype=torch.bool)
        diagnostic_token_mask = torch.zeros(len(masked_document.tokens), dtype=torch.bool)
        replacement_token_id = active_registry.encode_token(replacement_text)
        for token_index in range(edit_start, edit_end):
            target_token_mask[token_index] = True
            left = max(0, token_index - 1)
            right = min(len(masked_document.tokens), token_index + 2)
            diagnostic_token_mask[left:right] = True
            supervision_index = token_index - 1
            if supervision_index >= 0:
                supervision_mask[supervision_index] = True
                batch.targets[supervision_index] = replacement_token_id
        _sync_batch_vocabulary(batch, active_registry)
        batch.task_metadata.update(
            {
                "task_label": example.task_label.value,
                "probe_kind": example.metadata.get("probe_kind", "edit_fix"),
                "edit_target_span": (edit_start, edit_end),
                "target_token_value": example.metadata.get("target_token_value"),
                "replacement_text": replacement_text,
                "replacement_token_id": replacement_token_id,
                "instruction": example.metadata.get("instruction"),
                "target_symbol": example.metadata.get("target_symbol"),
            }
        )
        return TaskBatch(
            example=example,
            batch=batch,
            supervision_mask=supervision_mask,
            infill_span=None,
            edit_target_span=(edit_start, edit_end),
            replacement_text=replacement_text,
            metadata={
                "task_label": example.task_label.value,
                "probe_kind": example.metadata.get("probe_kind", "edit_fix"),
                "edit_target_token_mask": target_token_mask,
                "diagnostic_token_mask": diagnostic_token_mask,
                "replacement_text": replacement_text,
            },
        )

    batch = build_batch_from_document(document, boundaries, config, registry=active_registry)
    supervision_mask = torch.ones(len(document.tokens), dtype=torch.bool)
    replacement_text = example.metadata.get("replacement_text")
    if replacement_text is not None:
        active_registry.encode_token(str(replacement_text))
        _sync_batch_vocabulary(batch, active_registry)
    batch.task_metadata.update(
        {
            "task_label": example.task_label.value,
            "probe_kind": example.metadata.get("probe_kind"),
            "target_token_value": example.metadata.get("target_token_value"),
            "replacement_text": replacement_text,
            "instruction": example.metadata.get("instruction"),
            "target_symbol": example.metadata.get("target_symbol"),
        }
    )
    return TaskBatch(
        example=example,
        batch=batch,
        supervision_mask=supervision_mask,
        infill_span=None,
        edit_target_span=None,
        replacement_text=None,
        metadata={
            "task_label": example.task_label.value,
            "probe_kind": example.metadata.get("probe_kind"),
        },
    )


def _sync_batch_vocabulary(batch: PhaseABatch, registry: VocabularyRegistry) -> None:
    batch.registry_size = registry.size
    batch.vocabulary_snapshot = registry.snapshot()


def default_task_examples(
    *,
    repo_root: str | None = None,
    report_paths: list[str] | tuple[str, ...] | None = None,
) -> dict[TaskLabel, list[TaskExample]]:
    fixture_root = Path("tests/fixtures")
    examples: dict[TaskLabel, list[TaskExample]] = {}

    for path in [fixture_root / "sample_module.py", fixture_root / "unicode_module.py"]:
        if path.exists():
            examples.setdefault(TaskLabel.AR, []).append(build_task_example(str(path), TaskLabel.AR))
            examples.setdefault(TaskLabel.INFILL, []).append(build_task_example(str(path), TaskLabel.INFILL))

    recent_path = fixture_root / "recent_copy_module.py"
    if recent_path.exists():
        examples.setdefault(TaskLabel.RECENT_COPY, []).append(
            build_task_example(str(recent_path), TaskLabel.RECENT_COPY)
        )

    episodic_path = fixture_root / "episodic_copy_module.py"
    if episodic_path.exists():
        examples.setdefault(TaskLabel.EPISODIC_RECALL, []).append(
            build_task_example(str(episodic_path), TaskLabel.EPISODIC_RECALL)
        )

    repo_graph_path = fixture_root / "repo_graph_workspace" / "app" / "core.py"
    if repo_graph_path.exists():
        resolved_repo_root = repo_root or str((fixture_root / "repo_graph_workspace").resolve())
        repo_examples = [
            build_task_example(
                str(repo_graph_path),
                TaskLabel.REPO_GRAPH,
                repo_root=resolved_repo_root,
                report_paths=report_paths,
                metadata={
                    "probe_kind": "definition_use",
                    "target_token_value": "GRAPH_SHARED_NAME",
                    "target_symbol": "GRAPH_SHARED_NAME",
                },
            ),
            build_task_example(
                str(repo_graph_path),
                TaskLabel.REPO_GRAPH,
                repo_root=resolved_repo_root,
                report_paths=report_paths,
                metadata={
                    "probe_kind": "diagnostic_to_symbol",
                    "target_token_value": "GRAPH_SHARED_NAME",
                    "target_symbol": "GRAPH_SHARED_NAME",
                },
            ),
            build_task_example(
                str(repo_graph_path),
                TaskLabel.REPO_GRAPH,
                repo_root=resolved_repo_root,
                report_paths=report_paths,
                metadata={
                    "probe_kind": "edit_fix",
                    "target_token_value": "GRAPH_SHARED_NAME",
                    "replacement_text": "\"shared_graph_token\"",
                    "instruction": "Inline shared_graph_token expected by diagnostics in app/core.py",
                    "target_symbol": "GRAPH_SHARED_NAME",
                },
            ),
        ]
        examples.setdefault(TaskLabel.REPO_GRAPH, []).extend(repo_examples)
        examples.setdefault(TaskLabel.EDIT_FIX, []).append(
            build_task_example(
                str(repo_graph_path),
                TaskLabel.EDIT_FIX,
                repo_root=resolved_repo_root,
                report_paths=report_paths,
                metadata={
                    "probe_kind": "edit_fix",
                    "target_token_value": "GRAPH_SHARED_NAME",
                    "replacement_text": "\"shared_graph_token\"",
                    "instruction": "Inline shared_graph_token expected by diagnostics in app/core.py",
                    "target_symbol": "GRAPH_SHARED_NAME",
                },
            )
        )

    return {label: bucket for label, bucket in examples.items() if bucket}


def phase_task_weights(phase: TrainingPhase) -> dict[TaskLabel, int]:
    if phase == TrainingPhase.PHASE_A:
        return {TaskLabel.AR: 70, TaskLabel.INFILL: 30}
    if phase == TrainingPhase.PHASE_B:
        return {TaskLabel.AR: 55, TaskLabel.INFILL: 20, TaskLabel.RECENT_COPY: 25}
    if phase == TrainingPhase.PHASE_C:
        return {
            TaskLabel.AR: 45,
            TaskLabel.INFILL: 15,
            TaskLabel.RECENT_COPY: 20,
            TaskLabel.EPISODIC_RECALL: 20,
        }
    return {
        TaskLabel.AR: 35,
        TaskLabel.INFILL: 15,
        TaskLabel.RECENT_COPY: 15,
        TaskLabel.EPISODIC_RECALL: 15,
        TaskLabel.REPO_GRAPH: 10,
        TaskLabel.EDIT_FIX: 10,
    }


def build_task_schedule(
    phase: TrainingPhase,
    task_buckets: dict[TaskLabel, list[TaskExample]],
) -> list[TaskLabel]:
    weights = phase_task_weights(phase)
    available = {label: weight for label, weight in weights.items() if task_buckets.get(label)}
    if not available:
        return [next(iter(task_buckets))]
    slot_count = 20
    total = sum(available.values())
    counts = {
        label: max(1, int(round(slot_count * weight / total)))
        for label, weight in available.items()
    }
    while sum(counts.values()) > slot_count:
        label = max(counts, key=counts.get)
        if counts[label] > 1:
            counts[label] -= 1
        else:
            break
    while sum(counts.values()) < slot_count:
        label = max(available, key=available.get)
        counts[label] += 1
    schedule: list[TaskLabel] = []
    for label, count in counts.items():
        schedule.extend([label] * count)
    return schedule


def flatten_examples(task_buckets: dict[TaskLabel, list[TaskExample]]) -> list[TaskExample]:
    flattened: list[TaskExample] = []
    for bucket in task_buckets.values():
        flattened.extend(bucket)
    return flattened


def _slice_task_batch_window(task_batch: TaskBatch, start: int, end: int) -> TaskBatch:
    batch = _slice_phase_a_batch_window(task_batch.batch, start, end)
    supervision_mask = task_batch.supervision_mask[start:end].clone()
    target_span = _slice_optional_span(task_batch.edit_target_span, start, end)
    infill_span = _slice_optional_span(task_batch.infill_span, start, end)
    metadata = dict(task_batch.metadata)
    for key in ("edit_target_token_mask", "diagnostic_token_mask"):
        value = metadata.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] == task_batch.batch.token_ids.shape[0]:
            metadata[key] = value[start:end].clone()
    masked_range = metadata.get("masked_token_range")
    if isinstance(masked_range, tuple) and len(masked_range) == 2:
        metadata["masked_token_range"] = _slice_optional_span(masked_range, start, end)
    return TaskBatch(
        example=task_batch.example,
        batch=batch,
        supervision_mask=supervision_mask,
        infill_span=infill_span,
        edit_target_span=target_span,
        replacement_text=task_batch.replacement_text,
        metadata=metadata,
    )


def _iter_contiguous_task_windows(task_batch: TaskBatch, max_tokens: int) -> list[TaskBatch]:
    if max_tokens <= 0 or task_batch.batch.token_ids.shape[0] <= max_tokens:
        return [task_batch]
    boundaries = _window_end_indices(task_batch.batch, max_tokens)
    windows: list[TaskBatch] = []
    start = 0
    for end in boundaries:
        windows.append(_slice_task_batch_window(task_batch, start, end))
        start = end
    return windows


def _select_infill_span(document: AlignedDocument) -> tuple[int, int]:
    seq_len = len(document.tokens)
    if seq_len <= 2:
        return 0, seq_len
    span_length = max(1, min(8, seq_len // 6 or 1))
    start = max(1, (seq_len - span_length) // 3)
    end = min(seq_len, start + span_length)
    if end <= start:
        end = min(seq_len, start + 1)
    return start, end


def _mask_document_span(document: AlignedDocument, start: int, end: int) -> AlignedDocument:
    raw_bytes = bytearray(document.raw_bytes)
    masked_tokens: list[CodeToken] = list(document.tokens)
    for token_index in range(start, end):
        token = document.tokens[token_index]
        span_length = max(token.end_byte - token.start_byte, 1)
        raw_bytes[token.start_byte : token.end_byte] = b"#" * span_length
        masked_tokens[token_index] = replace(
            token,
            token_type="mask",
            value="<mask>",
            structural_tags=tuple(sorted(set(token.structural_tags) | {"masked-span"})),
        )
    masked_bytes = bytes(raw_bytes)
    return replace(
        document,
        raw_bytes=masked_bytes,
        source_text=masked_bytes.decode("utf-8", errors="ignore"),
        tokens=masked_tokens,
    )


def _select_edit_target(
    document: AlignedDocument,
    example: TaskExample,
) -> tuple[int, int, str]:
    target_value = str(example.metadata.get("target_token_value", "")).strip()
    replacement_text = str(example.metadata.get("replacement_text", "")).strip()
    if not replacement_text:
        replacement_text = "\"patched_value\""
    if target_value:
        matches: list[tuple[float, int]] = []
        for token in document.tokens:
            if token.value != target_value:
                continue
            structure = document.token_structures[token.index]
            score = 0.0
            if structure.scope_path:
                score += 2.0
            if structure.syntax_node_type == "identifier":
                score += 1.0
            if not {"import_statement", "import_from_statement"}.intersection(structure.ast_path):
                score += 3.0
            matches.append((score, token.index))
        if matches:
            matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_index = matches[0][1]
            return best_index, best_index + 1, replacement_text
    target_symbol = str(example.metadata.get("target_symbol", "")).strip()
    if target_symbol:
        for symbol in document.symbols:
            if symbol.name != target_symbol:
                continue
            for token in document.tokens:
                if token.start_byte >= symbol.start_byte and token.end_byte <= symbol.end_byte:
                    return token.index, token.index + 1, replacement_text
    for token in document.tokens:
        if token.token_class.value == "identifier":
            return token.index, token.index + 1, replacement_text
    return 0, min(1, len(document.tokens)), replacement_text


def _slice_phase_a_batch_window(batch: PhaseABatch, start: int, end: int) -> PhaseABatch:
    seq_len = batch.token_ids.shape[0]
    bounded_start = max(0, min(start, seq_len))
    bounded_end = max(bounded_start + 1, min(end, seq_len))
    sliced_document = _slice_aligned_document(batch.document, bounded_start, bounded_end)
    byte_offset = int(batch.token_spans[bounded_start, 0].item())
    return PhaseABatch(
        token_ids=batch.token_ids[bounded_start:bounded_end].clone(),
        token_class_ids=batch.token_class_ids[bounded_start:bounded_end].clone(),
        language_ids=batch.language_ids[bounded_start:bounded_end].clone(),
        scope_ids=batch.scope_ids[bounded_start:bounded_end].clone(),
        positions=torch.arange(bounded_end - bounded_start, dtype=batch.positions.dtype),
        byte_values=batch.byte_values[bounded_start:bounded_end].clone(),
        byte_mask=batch.byte_mask[bounded_start:bounded_end].clone(),
        ast_type_ids=batch.ast_type_ids[bounded_start:bounded_end].clone(),
        ast_depth_ids=batch.ast_depth_ids[bounded_start:bounded_end].clone(),
        ast_mask=batch.ast_mask[bounded_start:bounded_end].clone(),
        symbol_ids=batch.symbol_ids[bounded_start:bounded_end].clone(),
        file_ids=batch.file_ids[bounded_start:bounded_end].clone(),
        token_spans=(batch.token_spans[bounded_start:bounded_end] - byte_offset).clone(),
        token_payload_lengths=batch.token_payload_lengths[bounded_start:bounded_end].clone(),
        boundaries={
            level: values[bounded_start:bounded_end].clone()
            for level, values in batch.boundaries.items()
        },
        targets=batch.targets[bounded_start:bounded_end].clone(),
        registry_size=batch.registry_size,
        vocabulary_snapshot=batch.vocabulary_snapshot,
        document=sliced_document,
        task_metadata=dict(batch.task_metadata),
    )


def _slice_aligned_document(document: AlignedDocument, start: int, end: int) -> AlignedDocument:
    tokens = document.tokens[start:end]
    if not tokens:
        raise ValueError("Cannot slice an empty document window.")
    raw_start = tokens[0].start_byte
    raw_end = tokens[-1].end_byte
    raw_bytes = document.raw_bytes[raw_start:raw_end]
    shifted_tokens = [
        replace(
            token,
            index=local_index,
            start_byte=token.start_byte - raw_start,
            end_byte=token.end_byte - raw_start,
        )
        for local_index, token in enumerate(tokens)
    ]
    shifted_structures = [
        replace(structure, token_index=local_index)
        for local_index, structure in enumerate(document.token_structures[start:end])
    ]
    shifted_syntax = [
        replace(syntax, token_index=local_index)
        for local_index, syntax in enumerate(document.syntax_features[start:end])
    ]
    byte_to_token_index = [-1] * len(raw_bytes)
    for token in shifted_tokens:
        for byte_index in range(token.start_byte, token.end_byte):
            if 0 <= byte_index < len(byte_to_token_index):
                byte_to_token_index[byte_index] = token.index
    return replace(
        document,
        source_text=raw_bytes.decode("utf-8", errors="ignore"),
        raw_bytes=raw_bytes,
        tokens=shifted_tokens,
        byte_to_token_index=byte_to_token_index,
        token_structures=shifted_structures,
        syntax_features=shifted_syntax,
    )


def _slice_optional_span(
    span: tuple[int, int] | None,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    if span is None:
        return None
    left = max(span[0], start)
    right = min(span[1], end)
    if left >= right:
        return None
    return left - start, right - start


def _window_end_indices(batch: PhaseABatch, max_tokens: int) -> list[int]:
    seq_len = batch.token_ids.shape[0]
    if seq_len <= max_tokens:
        return [seq_len]
    candidate_levels = [4, 3, 2]
    boundaries: list[int] = []
    start = 0
    while start < seq_len:
        target = min(start + max_tokens, seq_len)
        boundary_end = target
        for level in candidate_levels:
            mask = batch.boundaries.get(level)
            if mask is None or target >= seq_len:
                continue
            for step in range(target - 1, start, -1):
                if bool(mask[step].item()):
                    boundary_end = step + 1
                    break
            if boundary_end != target:
                break
        if boundary_end <= start:
            boundary_end = target
        boundaries.append(boundary_end)
        start = boundary_end
    return boundaries
