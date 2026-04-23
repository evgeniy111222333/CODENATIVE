from __future__ import annotations

import hashlib

from htm_code_native.data.types import EditTargetSpan, PatchCandidate, PatchPlan
from htm_code_native.editing.planner import (
    dry_run_apply_patch_candidate,
    dry_run_apply_patch_plan,
)
from htm_code_native.tokenizer.tree_sitter_backend import parse_source_document


def _document(source: str = 'def fn():\n    value = "old"\n    return value\n'):
    return parse_source_document(source, "fixture.py", language="python")


def _candidate(source: str, replacement: str, *, start: int | None = None, end: int | None = None) -> PatchCandidate:
    span_start = source.index('"old"') if start is None else start
    span_end = span_start + len('"old"') if end is None else end
    span = EditTargetSpan(
        start_byte=span_start,
        end_byte=span_end,
        token_start=0,
        token_end=1,
        node_type="string",
        symbol_name=None,
        score=1.0,
        reasons=("test",),
        source_text=source[span_start:span_end] if 0 <= span_start < span_end <= len(source) else "",
    )
    patched_source = (
        source[:span_start] + replacement + source[span_end:]
        if 0 <= span_start < span_end <= len(source)
        else source
    )
    return PatchCandidate(
        span=span,
        replacement_text=replacement,
        patched_source=patched_source,
        diff_preview="",
        valid=True,
        validation_errors=(),
        score=1.0,
    )


def test_dry_run_apply_candidate_returns_diff_and_hash_without_mutating_document() -> None:
    source = 'def fn():\n    value = "old"\n    return value\n'
    document = _document(source)
    candidate = _candidate(source, '"new"')
    expected_source = source.replace('"old"', '"new"', 1)

    result = dry_run_apply_patch_candidate(document, candidate, "fixture.py", candidate_index=2)

    assert document.source_text == source
    assert result.candidate_index == 2
    assert result.applied is True
    assert result.valid is True
    assert result.validation_errors == ()
    assert result.patched_source_length == len(expected_source.encode("utf-8"))
    assert result.patched_source_hash == hashlib.blake2b(
        expected_source.encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    assert '"new"' in result.diff_preview


def test_dry_run_apply_plan_indexes_candidates() -> None:
    source = 'def fn():\n    value = "old"\n    return value\n'
    plan = PatchPlan(
        file_path="fixture.py",
        original_source=source,
        patch_candidates=(
            _candidate(source, '"new"'),
            _candidate(source, '"newer"'),
        ),
        best_candidate=None,
    )

    results = dry_run_apply_patch_plan(plan, "fixture.py")

    assert [result.candidate_index for result in results] == [0, 1]
    assert all(result.applied for result in results)


def test_dry_run_apply_rejects_invalid_span() -> None:
    source = 'def fn():\n    value = "old"\n'
    result = dry_run_apply_patch_candidate(
        _document(source),
        _candidate(source, '"new"', start=-1, end=4),
        "fixture.py",
    )

    assert result.applied is False
    assert result.valid is False
    assert "span_bounds_invalid" in result.validation_errors


def test_dry_run_apply_rejects_unchanged_patch() -> None:
    source = 'def fn():\n    value = "old"\n'
    result = dry_run_apply_patch_candidate(
        _document(source),
        _candidate(source, '"old"'),
        "fixture.py",
    )

    assert result.applied is False
    assert result.valid is False
    assert "unchanged_patch" in result.validation_errors


def test_dry_run_apply_surfaces_python_syntax_errors() -> None:
    source = 'def fn():\n    value = "old"\n'
    result = dry_run_apply_patch_candidate(
        _document(source),
        _candidate(source, '"unterminated'),
        "fixture.py",
    )

    assert result.applied is True
    assert result.valid is False
    assert result.syntax_error_count > 0
    assert any(error.startswith("python_ast:") for error in result.validation_errors)
