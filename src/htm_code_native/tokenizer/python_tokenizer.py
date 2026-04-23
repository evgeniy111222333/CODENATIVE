from __future__ import annotations

from htm_code_native.data.types import AlignedDocument
from htm_code_native.tokenizer.tree_sitter_backend import parse_source_document


class PythonTokenizer:
    """Compatibility wrapper over the canonical tree-sitter backend."""

    language = "python"

    def encode(self, source: str, file_path: str) -> AlignedDocument:
        return parse_source_document(source, file_path, language=self.language)
