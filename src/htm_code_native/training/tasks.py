from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.featurizer import build_batch_from_document
from htm_code_native.data.types import (
    AlignedDocument,
    CodeToken,
    TaskBatch,
    TaskExample,
    TaskLabel,
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
    resolved_root = str(resolve_repo_root(file_path, repo_root)) if label == TaskLabel.REPO_GRAPH else repo_root
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
    document, boundaries = parse_task_document(example.file_path)
    if example.task_label == TaskLabel.INFILL:
        infill_start, infill_end = _select_infill_span(document)
        masked_document = _mask_document_span(document, infill_start, infill_end)
        batch = build_batch_from_document(masked_document, boundaries, config, registry=registry)
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
        return TaskBatch(
            example=example,
            batch=batch,
            supervision_mask=supervision_mask,
            infill_span=(infill_start, infill_end),
            metadata=metadata,
        )

    batch = build_batch_from_document(document, boundaries, config, registry=registry)
    supervision_mask = torch.ones(len(document.tokens), dtype=torch.bool)
    return TaskBatch(
        example=example,
        batch=batch,
        supervision_mask=supervision_mask,
        infill_span=None,
        metadata={
            "task_label": example.task_label.value,
            "probe_kind": example.metadata.get("probe_kind"),
        },
    )


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
                metadata={"probe_kind": "definition_use"},
            ),
            build_task_example(
                str(repo_graph_path),
                TaskLabel.REPO_GRAPH,
                repo_root=resolved_repo_root,
                report_paths=report_paths,
                metadata={"probe_kind": "diagnostic_to_symbol"},
            ),
            build_task_example(
                str(repo_graph_path),
                TaskLabel.REPO_GRAPH,
                repo_root=resolved_repo_root,
                report_paths=report_paths,
                metadata={"probe_kind": "edit_fix"},
            ),
        ]
        examples.setdefault(TaskLabel.REPO_GRAPH, []).extend(repo_examples)

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
        TaskLabel.REPO_GRAPH: 20,
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
