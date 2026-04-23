from __future__ import annotations

from pathlib import Path

from htm_code_native.tokenizer.boundary import BoundaryScheduler


def test_token_byte_alignment_with_unicode_and_comments(build_document) -> None:
    path = Path("tests/fixtures/unicode_module.py")
    document = build_document(path)
    assert any("😀".encode("utf-8") in document.token_bytes(token.index) for token in document.tokens)
    comment_tokens = [token for token in document.tokens if token.token_type == "COMMENT"]
    assert comment_tokens
    for token in document.tokens:
        token_bytes = document.token_bytes(token.index)
        if token.start_byte != token.end_byte:
            assert token_bytes == document.raw_bytes[token.start_byte : token.end_byte]


def test_ast_and_symbol_spans_are_mapped(build_document) -> None:
    path = Path("tests/fixtures/sample_module.py")
    document = build_document(path)
    assert document.ast_nodes
    assert document.symbols
    token_structures = [info for info in document.token_structures if info.ast_path]
    assert token_structures
    assert any(info.symbol_name == "Accumulator" for info in document.token_structures)


def test_boundary_scheduler_detects_control_and_callable_nodes(build_document) -> None:
    path = Path("tests/fixtures/sample_module.py")
    document = build_document(path)
    boundaries = BoundaryScheduler().build(document)
    assert sum(boundaries.level_events[1]) > 0
    assert sum(boundaries.level_events[2]) > 0
    assert sum(boundaries.level_events[3]) >= 2
    assert boundaries.level_events[5][-1] is True
