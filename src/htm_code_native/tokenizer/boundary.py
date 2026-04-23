from __future__ import annotations

from htm_code_native.data.types import AlignedDocument, BoundaryEvents, TokenClass


STATEMENT_TYPES = {
    "AnnAssign",
    "Assign",
    "Assert",
    "AsyncFor",
    "AsyncFunctionDef",
    "AsyncWith",
    "AugAssign",
    "Break",
    "ClassDef",
    "Continue",
    "Delete",
    "Expr",
    "For",
    "FunctionDef",
    "Global",
    "If",
    "Import",
    "ImportFrom",
    "Nonlocal",
    "Pass",
    "Raise",
    "Return",
    "Try",
    "While",
    "With",
}
CONTROL_FLOW_TYPES = {"If", "For", "AsyncFor", "While", "Try", "With", "AsyncWith", "Match"}
CALLABLE_TYPES = {"FunctionDef", "AsyncFunctionDef", "ClassDef"}


class BoundaryScheduler:
    def __init__(self, file_chunk_size: int = 128, max_level: int = 5) -> None:
        self.file_chunk_size = file_chunk_size
        self.max_level = max_level

    def build(self, document: AlignedDocument) -> BoundaryEvents:
        token_count = len(document.tokens)
        level_events = {level: [False] * token_count for level in range(self.max_level + 1)}
        statement_ends = self._node_ends(document, STATEMENT_TYPES)
        control_ends = self._node_ends(document, CONTROL_FLOW_TYPES)
        callable_ends = self._node_ends(document, CALLABLE_TYPES)

        for index, token in enumerate(document.tokens):
            level_events[0][index] = True
            level_events[1][index] = (
                token.token_class == TokenClass.NEWLINE
                or token.value == ";"
                or token.end_byte in statement_ends
            )
            level_events[2][index] = (
                token.token_class == TokenClass.DEDENT or token.end_byte in control_ends
            )
            level_events[3][index] = token.end_byte in callable_ends
            level_events[4][index] = ((index + 1) % self.file_chunk_size == 0) or (
                index == token_count - 1
            )
            level_events[5][index] = index == token_count - 1

        return BoundaryEvents(level_events=level_events)

    def _node_ends(self, document: AlignedDocument, node_types: set[str]) -> set[int]:
        return {node.end_byte for node in document.ast_nodes if node.node_type in node_types}
