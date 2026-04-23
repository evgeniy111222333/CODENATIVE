from __future__ import annotations

from pathlib import Path

from htm_code_native.data.types import TaskLabel
from htm_code_native.training.tasks import (
    build_task_batch,
    build_task_example,
    default_task_examples,
    infer_task_label,
)


def test_infill_task_builds_contiguous_supervision_mask(config) -> None:
    example = build_task_example("tests/fixtures/sample_module.py", TaskLabel.INFILL)
    task_batch = build_task_batch(example, config)
    indices = task_batch.supervision_mask.nonzero(as_tuple=False).flatten().tolist()
    assert task_batch.infill_span is not None
    assert indices
    assert indices == list(range(indices[0], indices[-1] + 1))
    start, end = task_batch.infill_span
    assert start < end
    assert task_batch.batch.document.tokens[start].value == "<mask>"


def test_repo_graph_examples_include_probe_kinds() -> None:
    examples = default_task_examples()
    repo_examples = examples[TaskLabel.REPO_GRAPH]
    probe_kinds = {str(example.metadata["probe_kind"]) for example in repo_examples}
    assert {"definition_use", "diagnostic_to_symbol", "edit_fix"} <= probe_kinds


def test_task_label_fallback_heuristics_still_work() -> None:
    assert infer_task_label("tests/fixtures/recent_copy_module.py") == TaskLabel.RECENT_COPY
    assert infer_task_label("tests/fixtures/episodic_copy_module.py") == TaskLabel.EPISODIC_RECALL
    assert infer_task_label(str(Path("tests/fixtures/repo_graph_workspace/app/core.py"))) == TaskLabel.REPO_GRAPH
