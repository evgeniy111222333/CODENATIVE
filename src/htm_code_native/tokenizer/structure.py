from __future__ import annotations

from htm_code_native.data.types import AlignedDocument
from htm_code_native.tokenizer.tree_sitter_backend import parse_source_document


class PythonStructureExtractor:
    """Compatibility wrapper over the canonical tree-sitter backend."""

    def enrich(self, document: AlignedDocument) -> AlignedDocument:
        if document.parse_document is not None:
            return document
        return parse_source_document(document.source_text, document.file_path, language=document.language)
