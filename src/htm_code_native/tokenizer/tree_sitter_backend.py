from __future__ import annotations

import configparser
import ctypes
import keyword
import warnings
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Parser

from htm_code_native.data.types import (
    ASTNodeSpan,
    AlignedDocument,
    CodeToken,
    ParseDocument,
    ParseNode,
    SymbolSpan,
    SyntaxStateFeatures,
    TokenClass,
    TokenStructureInfo,
)
from htm_code_native.utils.text import linecol_to_byte_offset


DELIMITERS = {"(", ")", "[", "]", "{", "}", ",", ".", ":", ";", "=>"}
OPERATOR_CHARS = set("=+-*/%<>!&|^~?")
LANGUAGE_SYMBOLS = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "json": "tree_sitter_json",
    "yaml": "tree_sitter_yaml",
    "toml": "tree_sitter_toml",
}
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
}
STRING_NODE_TYPES = {
    "string",
    "string_literal",
    "template_string",
    "template_literal_type",
    "quoted_string",
}
NUMBER_NODE_TYPES = {
    "integer",
    "float",
    "number",
    "number_literal",
}
COMMENT_NODE_TYPES = {
    "comment",
    "line_comment",
    "block_comment",
}
IDENTIFIER_NODE_TYPES = {
    "identifier",
    "property_identifier",
    "type_identifier",
    "bare_key",
}
PYTHON_SYMBOL_NODES = {"function_definition", "class_definition"}
JS_TS_SYMBOL_NODES = {
    "function_declaration",
    "method_definition",
    "class_declaration",
    "variable_declarator",
}
BLOCK_NODE_TYPES = {
    "block",
    "statement_block",
    "class_body",
    "body",
    "object",
    "array",
}
CALL_NODE_TYPES = {"call", "call_expression"}
PYTHON_KEYWORDS = set(keyword.kwlist)
JS_TS_KEYWORDS = {
    "import",
    "from",
    "export",
    "class",
    "function",
    "return",
    "const",
    "let",
    "var",
    "if",
    "else",
    "for",
    "while",
    "switch",
    "case",
    "break",
    "continue",
    "try",
    "catch",
    "finally",
    "new",
    "extends",
    "implements",
    "async",
    "await",
    "true",
    "false",
    "null",
    "undefined",
    "type",
}
STATEMENT_NODE_TYPES = {
    "python": {
        "expression_statement",
        "assignment",
        "augmented_assignment",
        "return_statement",
        "import_statement",
        "import_from_statement",
        "function_definition",
        "class_definition",
        "if_statement",
        "for_statement",
        "while_statement",
        "with_statement",
        "try_statement",
        "raise_statement",
        "assert_statement",
        "pass_statement",
        "break_statement",
        "continue_statement",
    },
    "javascript": {
        "expression_statement",
        "lexical_declaration",
        "import_statement",
        "export_statement",
        "function_declaration",
        "class_declaration",
        "return_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
    },
    "typescript": {
        "expression_statement",
        "lexical_declaration",
        "import_statement",
        "export_statement",
        "function_declaration",
        "class_declaration",
        "return_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
    },
}
CONTROL_NODE_TYPES = {
    "python": {"if_statement", "for_statement", "while_statement", "with_statement", "try_statement", "match_statement"},
    "javascript": {"if_statement", "for_statement", "while_statement", "switch_statement", "try_statement", "statement_block"},
    "typescript": {"if_statement", "for_statement", "while_statement", "switch_statement", "try_statement", "statement_block"},
}
CALLABLE_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "class_declaration", "method_definition", "arrow_function"},
}


@dataclass(slots=True)
class _TokenDraft:
    sort_key: tuple[int, int, int]
    token_class: TokenClass
    token_type: str
    value: str
    start_byte: int
    end_byte: int
    language: str
    line: int
    column: int


class TreeSitterParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}
        self._dll = self._load_languages_dll()

    def detect_language(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(suffix, "python")

    def parse(self, source: str, file_path: str, language: str | None = None) -> AlignedDocument:
        resolved_language = language or self.detect_language(file_path)
        if resolved_language == "ini":
            return self._parse_ini(source, file_path)
        parser = self._get_parser(resolved_language)
        tree = parser.parse(source.encode("utf-8"))
        return self._build_document_from_tree(source, file_path, resolved_language, tree.root_node)

    def _get_parser(self, language: str) -> Parser:
        if language not in self._parsers:
            parser = Parser()
            parser.language = self._get_language(language)
            self._parsers[language] = parser
        return self._parsers[language]

    def _get_language(self, language: str) -> Language:
        if language not in self._languages:
            symbol = LANGUAGE_SYMBOLS[language]
            getattr(self._dll, symbol).restype = ctypes.c_void_p
            pointer = getattr(self._dll, symbol)()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._languages[language] = Language(pointer)
        return self._languages[language]

    def _build_document_from_tree(self, source: str, file_path: str, language: str, root_node) -> AlignedDocument:
        raw_bytes = source.encode("utf-8")
        real_tokens = self._collect_tokens(language, root_node, source)
        synthetic_tokens = self._build_synthetic_tokens(language, source)
        drafts = sorted(real_tokens + synthetic_tokens, key=lambda item: item.sort_key)

        tokens: list[CodeToken] = []
        byte_to_token_index = [-1] * len(raw_bytes)
        for index, draft in enumerate(drafts):
            token = CodeToken(
                index=index,
                token_class=draft.token_class,
                token_type=draft.token_type,
                value=draft.value,
                start_byte=draft.start_byte,
                end_byte=draft.end_byte,
                language=draft.language,
                structural_tags=("unstructured",),
                line=draft.line,
                column=draft.column,
            )
            tokens.append(token)
            for byte_index in range(token.start_byte, token.end_byte):
                if 0 <= byte_index < len(byte_to_token_index):
                    byte_to_token_index[byte_index] = index

        parse_nodes, ast_nodes, token_structures, symbols, syntax_features, parse_document = self._build_structure_views(
            root_node=root_node,
            tokens=tokens,
            source=source,
            file_path=file_path,
            language=language,
        )

        enriched_tokens = [
            CodeToken(
                index=token.index,
                token_class=token.token_class,
                token_type=token.token_type,
                value=token.value,
                start_byte=token.start_byte,
                end_byte=token.end_byte,
                language=token.language,
                structural_tags=self._build_tags(token_structures[token.index]),
                line=token.line,
                column=token.column,
            )
            for token in tokens
        ]

        return AlignedDocument(
            file_path=file_path,
            language=language,
            source_text=source,
            raw_bytes=raw_bytes,
            tokens=enriched_tokens,
            byte_to_token_index=byte_to_token_index,
            ast_nodes=ast_nodes,
            symbols=symbols,
            token_structures=token_structures,
            parse_document=parse_document,
            syntax_features=syntax_features,
        )

    def _parse_ini(self, source: str, file_path: str) -> AlignedDocument:
        raw_bytes = source.encode("utf-8")
        tokens: list[CodeToken] = []
        token_structures: list[TokenStructureInfo] = []
        syntax_features: list[SyntaxStateFeatures] = []
        ast_nodes: list[ASTNodeSpan] = []
        parse_nodes: list[ParseNode] = []
        parser = configparser.ConfigParser()
        try:
            parser.read_string(source)
            error_count = 0
            errors: tuple[str, ...] = ()
        except configparser.Error as exc:
            error_count = 1
            errors = (str(exc),)
        byte_to_token_index = [-1] * len(raw_bytes)
        for index, line in enumerate(source.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            start_byte = linecol_to_byte_offset(source, index + 1, 0)
            end_byte = linecol_to_byte_offset(source, index + 1, len(line))
            token_class = TokenClass.DELIMITER if stripped.startswith("[") else TokenClass.IDENTIFIER
            token = CodeToken(index=len(tokens), token_class=token_class, token_type="INI_LINE", value=stripped, start_byte=start_byte, end_byte=end_byte, language="ini", structural_tags=("unstructured",), line=index + 1, column=0)
            tokens.append(token)
            token_structures.append(
                TokenStructureInfo(
                    token_index=token.index,
                    ast_path=("ini_document",),
                    ast_node_ids=(f"ini:{token.index}",),
                    symbol_id=None,
                    symbol_name=None,
                    scope_path=(),
                    file_id=file_path,
                    syntax_node_type="ini_line",
                )
            )
            syntax_features.append(
                SyntaxStateFeatures(
                    token_index=token.index,
                    node_type="ini_line",
                    parent_type="ini_document",
                    depth=1,
                    inside_call=False,
                    inside_literal=False,
                    inside_comment=False,
                    block_depth=0,
                    parser_language="ini",
                )
            )
            ast_nodes.append(ASTNodeSpan(node_id=f"ini:{token.index}", node_type="ini_line", start_byte=start_byte, end_byte=end_byte, depth=1, parent_id=None))
            parse_nodes.append(ParseNode(node_id=f"ini:{token.index}", node_type="ini_line", start_byte=start_byte, end_byte=end_byte, depth=1, language="ini", is_named=True))
            for byte_index in range(start_byte, end_byte):
                if 0 <= byte_index < len(byte_to_token_index):
                    byte_to_token_index[byte_index] = token.index
        parse_document = ParseDocument(language="ini", parser_backend="ini-light", root_type="ini_document", nodes=parse_nodes, error_count=error_count, error_messages=errors)
        return AlignedDocument(
            file_path=file_path,
            language="ini",
            source_text=source,
            raw_bytes=raw_bytes,
            tokens=tokens,
            byte_to_token_index=byte_to_token_index,
            ast_nodes=ast_nodes,
            symbols=[],
            token_structures=token_structures,
            parse_document=parse_document,
            syntax_features=syntax_features,
        )

    def _collect_tokens(self, language: str, root_node, source: str) -> list[_TokenDraft]:
        drafts: list[_TokenDraft] = []

        def visit(node) -> None:
            if self._should_emit_named_node(node):
                text = source.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
                token_class = self._map_token_class(language, node.type, text)
                token_type = "COMMENT" if token_class == TokenClass.COMMENT else node.type
                drafts.append(
                    _TokenDraft(
                        sort_key=(node.start_byte, node.end_byte, 0),
                        token_class=token_class,
                        token_type=token_type,
                        value=text,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        language=language,
                        line=node.start_point.row + 1,
                        column=node.start_point.column,
                    )
                )
                return
            if node.child_count == 0:
                text = source.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
                if text:
                    token_class = self._map_token_class(language, node.type, text)
                    token_type = "COMMENT" if token_class == TokenClass.COMMENT else node.type
                    drafts.append(
                        _TokenDraft(
                            sort_key=(node.start_byte, node.end_byte, 0),
                            token_class=token_class,
                            token_type=token_type,
                            value=text,
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            language=language,
                            line=node.start_point.row + 1,
                            column=node.start_point.column,
                        )
                    )
                return
            for child in node.children:
                visit(child)

        visit(root_node)
        return drafts

    def _build_synthetic_tokens(self, language: str, source: str) -> list[_TokenDraft]:
        if language != "python":
            return []
        drafts: list[_TokenDraft] = []
        indent_stack = [0]
        for line_no, line in enumerate(source.splitlines(True), start=1):
            stripped = line.strip()
            newline_col = len(line.rstrip("\r\n"))
            newline_byte = linecol_to_byte_offset(source, line_no, newline_col)
            if stripped:
                content_col = len(line) - len(line.lstrip(" "))
                indent = content_col
                start_byte = linecol_to_byte_offset(source, line_no, content_col)
                while indent > indent_stack[-1]:
                    indent_stack.append(indent)
                    drafts.append(
                        _TokenDraft(
                            sort_key=(start_byte, start_byte, -1),
                            token_class=TokenClass.INDENT,
                            token_type="INDENT",
                            value="<INDENT>",
                            start_byte=start_byte,
                            end_byte=start_byte,
                            language=language,
                            line=line_no,
                            column=content_col,
                        )
                    )
                while indent < indent_stack[-1]:
                    indent_stack.pop()
                    drafts.append(
                        _TokenDraft(
                            sort_key=(start_byte, start_byte, -1),
                            token_class=TokenClass.DEDENT,
                            token_type="DEDENT",
                            value="<DEDENT>",
                            start_byte=start_byte,
                            end_byte=start_byte,
                            language=language,
                            line=line_no,
                            column=content_col,
                        )
                    )
            drafts.append(
                _TokenDraft(
                    sort_key=(newline_byte, newline_byte, 2),
                    token_class=TokenClass.NEWLINE,
                    token_type="NEWLINE",
                    value="\n",
                    start_byte=newline_byte,
                    end_byte=newline_byte,
                    language=language,
                    line=line_no,
                    column=newline_col,
                )
            )
        eof_line = max(len(source.splitlines()), 1)
        eof_byte = len(source.encode("utf-8"))
        while len(indent_stack) > 1:
            indent_stack.pop()
            drafts.append(
                _TokenDraft(
                    sort_key=(eof_byte, eof_byte, 1),
                    token_class=TokenClass.DEDENT,
                    token_type="DEDENT",
                    value="<DEDENT>",
                    start_byte=eof_byte,
                    end_byte=eof_byte,
                    language=language,
                    line=eof_line,
                    column=0,
                )
            )
        return drafts

    def _build_structure_views(self, root_node, tokens: list[CodeToken], source: str, file_path: str, language: str) -> tuple[list[ParseNode], list[ASTNodeSpan], list[TokenStructureInfo], list[SymbolSpan], list[SyntaxStateFeatures], ParseDocument]:
        parse_nodes: list[ParseNode] = []
        ast_nodes: list[ASTNodeSpan] = []
        symbols: list[SymbolSpan] = []
        best_paths: list[tuple[str, ...]] = [() for _ in tokens]
        best_node_ids: list[tuple[str, ...]] = [() for _ in tokens]
        best_symbols: list[tuple[str | None, str | None, tuple[str, ...], int | None]] = [(None, None, (), None) for _ in tokens]
        syntax_payloads: list[tuple[str, str | None, int, bool, bool, bool, int]] = [("module", None, 0, False, False, False, 0) for _ in tokens]
        errors: list[str] = []

        def visit(node, depth: int, parent_id: str | None, field_name: str | None, path: tuple[str, ...], scope_path: tuple[str, ...], active_symbol: tuple[str | None, str | None, tuple[str, ...], int | None]) -> None:
            node_id = f"{language}:{len(parse_nodes)}:{node.type}:{node.start_byte}:{node.end_byte}"
            parse_nodes.append(ParseNode(node_id=node_id, node_type=node.type, start_byte=node.start_byte, end_byte=node.end_byte, depth=depth, language=language, is_named=node.is_named, parent_id=parent_id, field_name=field_name))
            if node.is_named:
                ast_nodes.append(ASTNodeSpan(node_id=node_id, node_type=node.type, start_byte=node.start_byte, end_byte=node.end_byte, depth=depth, parent_id=parent_id))
            if node.type == "ERROR" or node.has_error:
                errors.append(f"{node.type}@{node.start_point.row + 1}:{node.start_point.column}")

            next_scope = scope_path
            next_symbol = active_symbol
            symbol_info = self._symbol_info(language, node, source.encode("utf-8"))
            if symbol_info is not None:
                symbol_name, symbol_kind, symbol_line = symbol_info
                symbol_id = self._make_symbol_id(file_path, symbol_kind, symbol_name, symbol_line)
                next_scope = (*scope_path, symbol_name)
                next_symbol = (symbol_id, symbol_name, next_scope, symbol_line)
                symbols.append(
                    SymbolSpan(
                        symbol_id=symbol_id,
                        name=symbol_name,
                        kind=symbol_kind,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        scope_path=next_scope,
                    )
                )

            if node.is_named:
                candidate_path = (*path, node.type)
                block_depth = sum(1 for item in candidate_path if item in BLOCK_NODE_TYPES)
                inside_call = any(item in CALL_NODE_TYPES for item in candidate_path)
                inside_literal = any(item in STRING_NODE_TYPES or item in NUMBER_NODE_TYPES for item in candidate_path)
                for token in tokens:
                    if self._token_in_span(token.start_byte, token.end_byte, node.start_byte, node.end_byte):
                        if len(candidate_path) >= len(best_paths[token.index]):
                            best_paths[token.index] = candidate_path
                            best_node_ids[token.index] = (*best_node_ids[token.index], node_id)
                            syntax_payloads[token.index] = (
                                node.type,
                                path[-1] if path else None,
                                depth,
                                inside_call,
                                inside_literal,
                                token.token_class == TokenClass.COMMENT,
                                block_depth,
                            )
                        if next_symbol[0] is not None:
                            best_symbols[token.index] = next_symbol

            for child in node.children:
                child_field_name = None
                try:
                    child_field_name = node.field_name_for_child(child.child_index)
                except Exception:
                    child_field_name = None
                visit(child, depth + 1, node_id, child_field_name, (*path, node.type) if node.is_named else path, next_scope, next_symbol)

        visit(root_node, 0, None, None, (), (), (None, None, (), None))

        token_structures = [
            TokenStructureInfo(
                token_index=token.index,
                ast_path=best_paths[token.index],
                ast_node_ids=best_node_ids[token.index],
                symbol_id=best_symbols[token.index][0],
                symbol_name=best_symbols[token.index][1],
                scope_path=best_symbols[token.index][2],
                file_id=file_path,
                symbol_line=best_symbols[token.index][3],
                syntax_node_type=syntax_payloads[token.index][0],
            )
            for token in tokens
        ]
        syntax_features = [
            SyntaxStateFeatures(
                token_index=token.index,
                node_type=syntax_payloads[token.index][0],
                parent_type=syntax_payloads[token.index][1],
                depth=syntax_payloads[token.index][2],
                inside_call=syntax_payloads[token.index][3],
                inside_literal=syntax_payloads[token.index][4],
                inside_comment=syntax_payloads[token.index][5],
                block_depth=syntax_payloads[token.index][6],
                parser_language=language,
            )
            for token in tokens
        ]
        parse_document = ParseDocument(language=language, parser_backend="tree-sitter", root_type=root_node.type, nodes=parse_nodes, error_count=len(errors), error_messages=tuple(errors[:32]))
        return parse_nodes, ast_nodes, token_structures, symbols, syntax_features, parse_document

    def _should_emit_named_node(self, node) -> bool:
        if not node.is_named:
            return False
        if node.type in COMMENT_NODE_TYPES or node.type in STRING_NODE_TYPES or node.type in NUMBER_NODE_TYPES or node.type in IDENTIFIER_NODE_TYPES:
            return True
        return node.named_child_count == 0

    def _map_token_class(self, language: str, node_type: str, text: str) -> TokenClass:
        if node_type in COMMENT_NODE_TYPES:
            return TokenClass.COMMENT
        if node_type in STRING_NODE_TYPES:
            return TokenClass.STRING
        if node_type in NUMBER_NODE_TYPES:
            return TokenClass.NUMBER
        if node_type in IDENTIFIER_NODE_TYPES:
            return TokenClass.IDENTIFIER
        if text in DELIMITERS:
            return TokenClass.DELIMITER
        if text in PYTHON_KEYWORDS or text in JS_TS_KEYWORDS:
            return TokenClass.KEYWORD
        if text and all(char in OPERATOR_CHARS for char in text):
            return TokenClass.OPERATOR
        if node_type in {"true", "false", "null"}:
            return TokenClass.KEYWORD
        return TokenClass.FALLBACK_BYTE_PIECE

    def _token_in_span(self, token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
        if token_start == token_end:
            return span_start <= token_start <= span_end
        return token_start >= span_start and token_end <= span_end

    def _symbol_info(self, language: str, node, raw_bytes: bytes) -> tuple[str, str, int] | None:
        if language == "python" and node.type in PYTHON_SYMBOL_NODES:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return None
            name = raw_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
            kind = "class" if node.type == "class_definition" else "function"
            return name, kind, name_node.start_point.row + 1
        if language in {"javascript", "typescript"}:
            if node.type in {"function_declaration", "class_declaration", "method_definition"}:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    return None
                name = raw_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
                kind = "class" if node.type == "class_declaration" else "function"
                return name, kind, name_node.start_point.row + 1
            if node.type == "variable_declarator":
                value_node = node.child_by_field_name("value")
                name_node = node.child_by_field_name("name")
                if value_node is None or name_node is None:
                    return None
                if value_node.type not in {"arrow_function", "function"}:
                    return None
                name = raw_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
                return name, "function", name_node.start_point.row + 1
        return None

    def _make_symbol_id(self, file_path: str, kind: str, name: str, line: int) -> str:
        return f"{kind}:{file_path}:{name}:{line}"

    def _build_tags(self, structure: TokenStructureInfo) -> tuple[str, ...]:
        tags: list[str] = []
        if structure.ast_path:
            tags.append(f"ast:{structure.ast_path[-1]}")
            tags.append(f"depth:{len(structure.ast_path)}")
        if structure.symbol_name is not None:
            tags.append(f"symbol:{structure.symbol_name}")
        if structure.scope_path:
            tags.append(f"scope:{'/'.join(structure.scope_path)}")
        if structure.syntax_node_type is not None:
            tags.append(f"syntax:{structure.syntax_node_type}")
        return tuple(tags or ["unstructured"])

    def _load_languages_dll(self):
        import tree_sitter_languages

        dll_path = Path(tree_sitter_languages.__file__).resolve().parent / "languages.dll"
        return ctypes.CDLL(str(dll_path))


_DEFAULT_REGISTRY = TreeSitterParserRegistry()


def parse_source_document(source: str, file_path: str, language: str | None = None) -> AlignedDocument:
    return _DEFAULT_REGISTRY.parse(source, file_path, language=language)


def detect_language(file_path: str) -> str:
    return _DEFAULT_REGISTRY.detect_language(file_path)
