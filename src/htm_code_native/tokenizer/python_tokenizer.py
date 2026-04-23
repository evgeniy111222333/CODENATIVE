from __future__ import annotations

import io
import keyword
import token as token_mod
import tokenize
from collections.abc import Iterable

from htm_code_native.data.types import AlignedDocument, CodeToken, TokenClass, TokenStructureInfo
from htm_code_native.utils.text import linecol_to_byte_offset


DELIMITERS = {"(", ")", "[", "]", "{", "}", ",", ".", ":", ";"}


class PythonTokenizer:
    """Lexer-aware Python tokenizer with UTF-8 byte alignment."""

    language = "python"

    def encode(self, source: str, file_path: str) -> AlignedDocument:
        raw_bytes = source.encode("utf-8")
        tokens: list[CodeToken] = []
        byte_to_token_index = [-1] * len(raw_bytes)

        for index, info in enumerate(self._iter_tokens(source)):
            start_byte = linecol_to_byte_offset(source, info.start[0], info.start[1])
            end_byte = linecol_to_byte_offset(source, info.end[0], info.end[1])
            token_class = self._map_token_class(info)
            code_token = CodeToken(
                index=index,
                token_class=token_class,
                token_type=token_mod.tok_name[info.type],
                value=info.string,
                start_byte=start_byte,
                end_byte=end_byte,
                language=self.language,
                structural_tags=("unstructured",),
                line=info.start[0],
                column=info.start[1],
            )
            tokens.append(code_token)
            for byte_index in range(start_byte, end_byte):
                byte_to_token_index[byte_index] = index

        token_structures = [
            TokenStructureInfo(
                token_index=token.index,
                ast_path=(),
                ast_node_ids=(),
                symbol_id=None,
                symbol_name=None,
                scope_path=(),
                file_id=file_path,
            )
            for token in tokens
        ]

        return AlignedDocument(
            file_path=file_path,
            language=self.language,
            source_text=source,
            raw_bytes=raw_bytes,
            tokens=tokens,
            byte_to_token_index=byte_to_token_index,
            token_structures=token_structures,
        )

    def _iter_tokens(self, source: str) -> Iterable[tokenize.TokenInfo]:
        for info in tokenize.generate_tokens(io.StringIO(source).readline):
            if info.type == tokenize.ENDMARKER:
                continue
            yield info

    def _map_token_class(self, info: tokenize.TokenInfo) -> TokenClass:
        if info.type == tokenize.NAME:
            if keyword.iskeyword(info.string):
                return TokenClass.KEYWORD
            return TokenClass.IDENTIFIER
        if info.type == tokenize.NUMBER:
            return TokenClass.NUMBER
        if info.type == tokenize.STRING:
            return TokenClass.STRING
        if info.type in (tokenize.NEWLINE, tokenize.NL):
            return TokenClass.NEWLINE
        if info.type == tokenize.INDENT:
            return TokenClass.INDENT
        if info.type == tokenize.DEDENT:
            return TokenClass.DEDENT
        if info.type == tokenize.COMMENT:
            return TokenClass.COMMENT
        if info.type == tokenize.OP:
            return TokenClass.DELIMITER if info.string in DELIMITERS else TokenClass.OPERATOR
        return TokenClass.FALLBACK_BYTE_PIECE
