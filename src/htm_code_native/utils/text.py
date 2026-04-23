from __future__ import annotations


def build_line_start_offsets(source: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in source.splitlines(keepends=True):
        running += len(line.encode("utf-8"))
        offsets.append(running)
    if not source.endswith(("\n", "\r")):
        offsets.append(running)
    return offsets


def linecol_to_byte_offset(source: str, line: int, column: int) -> int:
    if line <= 0:
        return 0
    lines = source.splitlines(keepends=True)
    prefix = b"".join(part.encode("utf-8") for part in lines[: line - 1])
    line_text = lines[line - 1] if line - 1 < len(lines) else ""
    return len(prefix) + len(line_text[:column].encode("utf-8"))
