from __future__ import annotations

from htm_code_native.data.types import AlignedDocument, BoundaryEvents, TokenClass


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


class BoundaryScheduler:
    def __init__(self, file_chunk_size: int = 128, max_level: int = 5) -> None:
        self.file_chunk_size = file_chunk_size
        self.max_level = max_level

    def build(self, document: AlignedDocument) -> BoundaryEvents:
        token_count = len(document.tokens)
        level_events = {level: [False] * token_count for level in range(self.max_level + 1)}
        statement_ends = self._node_ends(document, STATEMENT_NODE_TYPES.get(document.language, set()))
        control_ends = self._node_ends(document, CONTROL_NODE_TYPES.get(document.language, set()))
        callable_ends = self._node_ends(document, CALLABLE_NODE_TYPES.get(document.language, set()))

        for index, token in enumerate(document.tokens):
            level_events[0][index] = True
            level_events[1][index] = (
                token.token_class == TokenClass.NEWLINE
                or token.value == ";"
                or token.end_byte in statement_ends
            )
            level_events[2][index] = (
                token.token_class == TokenClass.DEDENT
                or token.end_byte in control_ends
            )
            level_events[3][index] = token.end_byte in callable_ends
            level_events[4][index] = ((index + 1) % self.file_chunk_size == 0) or (
                index == token_count - 1
            )
            level_events[5][index] = index == token_count - 1

        return BoundaryEvents(level_events=level_events)

    def _node_ends(self, document: AlignedDocument, node_types: set[str]) -> set[int]:
        return {node.end_byte for node in document.ast_nodes if node.node_type in node_types}
