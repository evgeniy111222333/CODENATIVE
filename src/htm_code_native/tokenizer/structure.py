from __future__ import annotations

import ast
from dataclasses import replace

from htm_code_native.data.types import (
    ASTNodeSpan,
    AlignedDocument,
    SymbolSpan,
    TokenStructureInfo,
)
from htm_code_native.utils.text import linecol_to_byte_offset


SYMBOL_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class PythonStructureExtractor:
    """Enriches aligned documents with AST paths and symbol metadata."""

    def enrich(self, document: AlignedDocument) -> AlignedDocument:
        tree = ast.parse(document.source_text, filename=document.file_path)
        best_ast_paths: list[tuple[str, ...]] = [() for _ in document.tokens]
        best_node_ids: list[tuple[str, ...]] = [() for _ in document.tokens]
        best_symbols: list[tuple[str | None, str | None, tuple[str, ...]]] = [
            (None, None, ()) for _ in document.tokens
        ]
        token_tags: list[tuple[str, ...]] = [("unstructured",) for _ in document.tokens]
        ast_nodes: list[ASTNodeSpan] = []
        symbols: list[SymbolSpan] = []
        counters = {"ast": 0, "sym": 0}

        def visit(
            node: ast.AST,
            path: tuple[str, ...],
            parent_id: str | None,
            scope_path: tuple[str, ...],
            active_symbol: tuple[str | None, str | None, tuple[str, ...]],
        ) -> None:
            node_id = self._make_node_id(node, counters)
            span = self._node_span(document, node)
            next_scope = scope_path
            next_symbol = active_symbol

            if span is not None:
                start_byte, end_byte = span
                ast_nodes.append(
                    ASTNodeSpan(
                        node_id=node_id,
                        node_type=type(node).__name__,
                        start_byte=start_byte,
                        end_byte=end_byte,
                        depth=len(path),
                        parent_id=parent_id,
                    )
                )

                if isinstance(node, SYMBOL_NODE_TYPES) and getattr(node, "name", None):
                    symbol_id = self._make_symbol_id(node, counters, document.file_path)
                    next_scope = (*scope_path, str(getattr(node, "name")))
                    next_symbol = (symbol_id, str(getattr(node, "name")), next_scope)
                    symbols.append(
                        SymbolSpan(
                            symbol_id=symbol_id,
                            name=str(getattr(node, "name")),
                            kind=type(node).__name__,
                            start_byte=start_byte,
                            end_byte=end_byte,
                            scope_path=next_scope,
                        )
                    )

                for token in document.tokens:
                    if self._token_in_span(token.start_byte, token.end_byte, start_byte, end_byte):
                        candidate_path = (*path, type(node).__name__)
                        if len(candidate_path) >= len(best_ast_paths[token.index]):
                            best_ast_paths[token.index] = candidate_path
                            best_node_ids[token.index] = (*best_node_ids[token.index], node_id)
                        if next_symbol[0] is not None:
                            best_symbols[token.index] = next_symbol
                        token_tags[token.index] = self._build_tags(candidate_path, best_symbols[token.index])

            for child in ast.iter_child_nodes(node):
                visit(child, (*path, type(node).__name__), node_id, next_scope, next_symbol)

        visit(tree, (), None, (), (None, None, ()))

        token_structures = [
            TokenStructureInfo(
                token_index=token.index,
                ast_path=best_ast_paths[token.index],
                ast_node_ids=best_node_ids[token.index],
                symbol_id=best_symbols[token.index][0],
                symbol_name=best_symbols[token.index][1],
                scope_path=best_symbols[token.index][2],
                file_id=document.file_path,
            )
            for token in document.tokens
        ]

        enriched_tokens = [
            replace(token, structural_tags=token_tags[token.index]) for token in document.tokens
        ]

        return AlignedDocument(
            file_path=document.file_path,
            language=document.language,
            source_text=document.source_text,
            raw_bytes=document.raw_bytes,
            tokens=enriched_tokens,
            byte_to_token_index=document.byte_to_token_index,
            ast_nodes=ast_nodes,
            symbols=symbols,
            token_structures=token_structures,
        )

    def _node_span(self, document: AlignedDocument, node: ast.AST) -> tuple[int, int] | None:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return None
        start_byte = linecol_to_byte_offset(
            document.source_text,
            int(getattr(node, "lineno")),
            int(getattr(node, "col_offset", 0)),
        )
        end_byte = linecol_to_byte_offset(
            document.source_text,
            int(getattr(node, "end_lineno")),
            int(getattr(node, "end_col_offset", 0)),
        )
        return start_byte, end_byte

    def _token_in_span(
        self,
        token_start: int,
        token_end: int,
        span_start: int,
        span_end: int,
    ) -> bool:
        if token_start == token_end:
            return span_start <= token_start <= span_end
        return token_start >= span_start and token_end <= span_end

    def _make_node_id(self, node: ast.AST, counters: dict[str, int]) -> str:
        counters["ast"] += 1
        return f"{type(node).__name__}:{counters['ast']}"

    def _make_symbol_id(self, node: ast.AST, counters: dict[str, int], file_path: str) -> str:
        counters["sym"] += 1
        name = str(getattr(node, "name", "symbol"))
        lineno = int(getattr(node, "lineno", 0))
        return f"{file_path}:{name}:{lineno}:{counters['sym']}"

    def _build_tags(
        self,
        ast_path: tuple[str, ...],
        symbol_info: tuple[str | None, str | None, tuple[str, ...]],
    ) -> tuple[str, ...]:
        tags: list[str] = []
        if ast_path:
            tags.append(f"ast:{ast_path[-1]}")
            tags.append(f"depth:{len(ast_path)}")
        if symbol_info[1] is not None:
            tags.append(f"symbol:{symbol_info[1]}")
        if symbol_info[2]:
            tags.append(f"scope:{'/'.join(symbol_info[2])}")
        return tuple(tags or ["unstructured"])
