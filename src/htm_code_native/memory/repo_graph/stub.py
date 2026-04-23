from __future__ import annotations

import ast
import configparser
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence
from xml.etree import ElementTree

import torch
import yaml
from torch import nn

from htm_code_native.data.types import (
    AlignedDocument,
    RepoGraphQueryContext,
    RepoGraphReadResult,
    RepositoryGraphEdge,
    RepositoryGraphIndex,
    RepositoryGraphNode,
)
from htm_code_native.data.vocabulary import VocabularySnapshot
from htm_code_native.tokenizer.tree_sitter_backend import parse_source_document
from htm_code_native.utils.hashing import stable_int_hash

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None


PYTHON_EXTENSIONS = {".py"}
JS_TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
SUPPORTED_EXTENSIONS = PYTHON_EXTENSIONS | JS_TS_EXTENSIONS | CONFIG_EXTENSIONS
SKIP_DIRECTORIES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}
TEST_PATH_MARKERS = ("tests", "test")
JS_TEST_MARKERS = ("describe(", "it(", "test(")
JS_CALL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "function",
    "return",
    "typeof",
    "console",
}
TS_IMPORT_RE = re.compile(r"""import\s+(?:type\s+)?(?:.+?)\s+from\s+["'](.+?)["']""")
TS_REQUIRE_RE = re.compile(r"""require\(\s*["'](.+?)["']\s*\)""")
TS_FUNCTION_RE = re.compile(r"""(?:export\s+)?function\s+([A-Za-z_]\w*)\s*\(""")
TS_CLASS_RE = re.compile(r"""(?:export\s+)?class\s+([A-Za-z_]\w*)\b""")
TS_ARROW_RE = re.compile(r"""(?:export\s+)?const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>""")
TS_CALL_RE = re.compile(r"""\b([A-Za-z_]\w*)\s*\(""")
STRING_RE = re.compile(r"""["']([^"'\n]{1,64})["']""")
NUMBER_RE = re.compile(r"""\b\d+(?:\.\d+)?\b""")
TSC_DIAGNOSTIC_RE = re.compile(
    r"""^(?P<path>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s+error\s+(?P<code>TS\d+):\s+(?P<message>.+)$"""
)


@dataclass(slots=True)
class _SymbolRecord:
    node_id: str
    name: str
    file_path: str
    kind: str


@dataclass(slots=True)
class _FileRecord:
    path: Path
    relative_path: str
    file_node_id: str
    language: str
    import_specs: list[str]
    resolved_imports: list[str]
    exported_names: list[str]
    call_records: list[tuple[str, str]]
    source_copy_terms: list[str]
    test_node_ids: list[str]


class RepoGraphAdapter(Protocol):
    def set_index(self, index: RepositoryGraphIndex | None) -> None:
        ...

    def reset(self) -> None:
        ...

    def query(
        self,
        hidden: torch.Tensor,
        context: RepoGraphQueryContext,
        vocabulary_snapshot: VocabularySnapshot,
    ) -> RepoGraphReadResult:
        ...


class RepositoryGraphIndexer:
    def __init__(self, key_dim: int, value_dim: int, max_files: int = 256) -> None:
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_files = max_files

    def build(
        self,
        repo_root: str | Path,
        report_paths: Sequence[str] | None = None,
    ) -> RepositoryGraphIndex:
        root = Path(repo_root).resolve()
        nodes: list[RepositoryGraphNode] = []
        edges: list[RepositoryGraphEdge] = []
        nodes_by_id: dict[str, RepositoryGraphNode] = {}
        node_ids_by_file: dict[str, list[str]] = {}
        file_records: dict[str, _FileRecord] = {}
        symbol_records_by_name: dict[str, list[_SymbolRecord]] = {}
        source_files = self._collect_source_files(root)

        for source_path in source_files:
            relative_path = self._relative_path(source_path, root)
            file_node = self._make_file_node(relative_path, source_path)
            self._register_node(nodes, nodes_by_id, node_ids_by_file, file_node)
            record = _FileRecord(
                path=source_path,
                relative_path=relative_path,
                file_node_id=file_node.node_id,
                language=self._language_for_path(source_path),
                import_specs=[],
                resolved_imports=[],
                exported_names=[],
                call_records=[],
                source_copy_terms=list(file_node.copy_terms),
                test_node_ids=[],
            )
            file_records[relative_path] = record
            parsed_nodes, parsed_edges, symbol_records = self._parse_source_file(root, source_path, relative_path)
            for node in parsed_nodes:
                self._register_node(nodes, nodes_by_id, node_ids_by_file, node)
                record.source_copy_terms.extend(node.copy_terms)
                if node.kind == "test":
                    record.test_node_ids.append(node.node_id)
            edges.extend(parsed_edges)
            record.import_specs.extend(self._import_specs_from_nodes(parsed_nodes))
            record.exported_names.extend(self._exported_names_from_nodes(parsed_nodes))
            record.call_records.extend(self._call_records_from_nodes(parsed_nodes))
            for symbol_record in symbol_records:
                symbol_records_by_name.setdefault(symbol_record.name, []).append(symbol_record)

        diagnostics_by_source: dict[str, set[str]] = {}
        test_files_by_source: dict[str, set[str]] = {}

        for relative_path, record in file_records.items():
            for import_spec in record.import_specs:
                resolved = self._resolve_import(record.path, root, import_spec)
                if resolved is None:
                    continue
                resolved_relative = self._relative_path(resolved, root)
                if resolved_relative not in file_records:
                    continue
                record.resolved_imports.append(resolved_relative)
                edges.append(
                    RepositoryGraphEdge(
                        source_id=record.file_node_id,
                        target_id=file_records[resolved_relative].file_node_id,
                        kind="imports",
                    )
                )

            for source_node_id, callee_name in record.call_records:
                resolved_target = self._resolve_call_target(
                    file_path=record.relative_path,
                    callee_name=callee_name,
                    symbol_records_by_name=symbol_records_by_name,
                    imported_files=set(record.resolved_imports),
                )
                if resolved_target is None:
                    continue
                edges.append(
                    RepositoryGraphEdge(
                        source_id=source_node_id,
                        target_id=resolved_target.node_id,
                        kind="calls",
                        heuristic=resolved_target.file_path != record.relative_path,
                    )
                )

            if record.test_node_ids:
                for imported_file in record.resolved_imports:
                    test_files_by_source.setdefault(imported_file, set()).add(record.relative_path)
                    for test_node_id in record.test_node_ids:
                        edges.append(
                            RepositoryGraphEdge(
                                source_id=file_records[imported_file].file_node_id,
                                target_id=test_node_id,
                                kind="tested_by",
                            )
                        )

        report_nodes, report_edges, report_targets = self._ingest_reports(root, report_paths or (), nodes_by_id)
        for node in report_nodes:
            self._register_node(nodes, nodes_by_id, node_ids_by_file, node)
        edges.extend(report_edges)
        for source_file, target_files in report_targets.items():
            diagnostics_by_source.setdefault(source_file, set()).update(target_files)

        import_closure = self._build_import_closure(file_records)

        return RepositoryGraphIndex(
            root_path=str(root),
            nodes=nodes,
            edges=edges,
            nodes_by_id=nodes_by_id,
            node_ids_by_file={key: tuple(value) for key, value in node_ids_by_file.items()},
            import_closure_by_file={key: tuple(sorted(value)) for key, value in import_closure.items()},
            test_files_by_source={key: tuple(sorted(value)) for key, value in test_files_by_source.items()},
            diagnostic_files_by_source={key: tuple(sorted(value)) for key, value in diagnostics_by_source.items()},
        )

    def _collect_source_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        for path in sorted(root.rglob("*")):
            if len(files) >= self.max_files:
                break
            if path.is_dir():
                continue
            if any(part in SKIP_DIRECTORIES for part in path.parts):
                continue
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
        return files

    def _parse_source_file(
        self,
        root: Path,
        source_path: Path,
        relative_path: str,
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], list[_SymbolRecord]]:
        suffix = source_path.suffix.lower()
        if suffix in PYTHON_EXTENSIONS:
            return self._parse_python_file(root, source_path, relative_path)
        if suffix in JS_TS_EXTENSIONS:
            return self._parse_ts_js_file(root, source_path, relative_path)
        return self._parse_config_file(root, source_path, relative_path)

    def _parse_python_file(
        self,
        root: Path,
        source_path: Path,
        relative_path: str,
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], list[_SymbolRecord]]:
        content = source_path.read_text(encoding="utf-8")
        document = parse_source_document(content, str(source_path), language="python")
        return self._parse_code_document(relative_path, source_path.stem, document, heuristic=False)

    def _parse_ts_js_file(
        self,
        root: Path,
        source_path: Path,
        relative_path: str,
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], list[_SymbolRecord]]:
        content = source_path.read_text(encoding="utf-8")
        language = "typescript" if source_path.suffix.lower() in {".ts", ".tsx"} else "javascript"
        document = parse_source_document(content, str(source_path), language=language)
        return self._parse_code_document(relative_path, source_path.stem, document, heuristic=True)

    def _parse_config_file(
        self,
        root: Path,
        source_path: Path,
        relative_path: str,
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], list[_SymbolRecord]]:
        content = source_path.read_text(encoding="utf-8")
        document = parse_source_document(content, str(source_path))
        config_terms = self._config_terms_from_document(source_path, document)
        config_node = self._make_config_node(relative_path, source_path.stem, tuple(config_terms))
        return [
            config_node
        ], [
            RepositoryGraphEdge(
                source_id=self._file_node_id(relative_path),
                target_id=config_node.node_id,
                kind="contains",
            )
        ], []

    def _parse_code_document(
        self,
        relative_path: str,
        stem: str,
        document: AlignedDocument,
        heuristic: bool,
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], list[_SymbolRecord]]:
        nodes: list[RepositoryGraphNode] = []
        edges: list[RepositoryGraphEdge] = []
        symbols: list[_SymbolRecord] = []
        file_node_id = self._file_node_id(relative_path)
        literals = self._literal_terms_from_text(document.source_text)

        for symbol in document.symbols:
            line = self._line_for_byte_offset(document, symbol.start_byte)
            kind = symbol.kind if symbol.kind in {"function", "class"} else "symbol"
            symbol_node = self._make_symbol_node(
                kind=kind,
                relative_path=relative_path,
                name=symbol.name,
                line=line,
                copy_terms=self._symbol_copy_terms(document, symbol),
                heuristic=heuristic,
            )
            nodes.append(symbol_node)
            edges.append(
                RepositoryGraphEdge(
                    source_id=file_node_id,
                    target_id=symbol_node.node_id,
                    kind="defines",
                    heuristic=heuristic,
                )
            )
            symbols.append(
                _SymbolRecord(
                    node_id=symbol_node.node_id,
                    name=symbol.name,
                    file_path=relative_path,
                    kind=kind,
                )
            )

        import_specs = self._import_specs_from_document(document)
        for index, import_spec in enumerate(import_specs, start=1):
            import_node = self._make_import_node(relative_path, import_spec, index, heuristic=heuristic)
            nodes.append(import_node)
            edges.append(
                RepositoryGraphEdge(
                    source_id=file_node_id,
                    target_id=import_node.node_id,
                    kind="imports",
                    heuristic=heuristic,
                )
            )

        if self._is_test_file(relative_path, document.source_text):
            test_node = self._make_test_node(relative_path, stem, tuple(literals), heuristic=heuristic)
            nodes.append(test_node)
            edges.append(
                RepositoryGraphEdge(
                    source_id=file_node_id,
                    target_id=test_node.node_id,
                    kind="contains",
                    heuristic=heuristic,
                )
            )

        call_records = self._call_records_from_document(document, relative_path)
        for source_node_id, callee_name in call_records:
            call_node = self._make_reference_node(
                relative_path,
                source_node_id,
                callee_name,
                heuristic=heuristic,
            )
            nodes.append(call_node)
            edges.append(
                RepositoryGraphEdge(
                    source_id=source_node_id,
                    target_id=call_node.node_id,
                    kind="references",
                    heuristic=heuristic,
                )
            )

        file_copy_terms = tuple(self._code_copy_terms_from_document(document, literals))
        if file_copy_terms:
            nodes.append(
                self._make_file_overlay_node(
                    relative_path,
                    "symbol",
                    f"{stem}_symbols",
                    file_copy_terms,
                    heuristic=heuristic,
                )
            )

        return nodes, edges, symbols

    def _import_specs_from_document(self, document: AlignedDocument) -> list[str]:
        specs: list[str] = []
        if document.parse_document is None:
            return specs
        import_node_types = {"import_statement", "import_from_statement"}
        for node in document.parse_document.nodes:
            if node.node_type not in import_node_types:
                continue
            span_text = document.source_text.encode("utf-8")[node.start_byte : node.end_byte].decode(
                "utf-8",
                errors="ignore",
            )
            spec = self._extract_import_spec(span_text, document.language)
            if spec:
                specs.append(spec)
        return self._unique_terms(specs)

    def _call_records_from_document(
        self,
        document: AlignedDocument,
        relative_path: str,
    ) -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        if document.parse_document is None:
            return records
        current_scope = self._file_node_id(relative_path)
        for node in document.parse_document.nodes:
            if node.node_type not in {"call", "call_expression"}:
                continue
            span_text = document.source_text.encode("utf-8")[node.start_byte : node.end_byte].decode(
                "utf-8",
                errors="ignore",
            )
            callee_name = self._extract_call_name(span_text)
            if not callee_name or callee_name in JS_CALL_KEYWORDS:
                continue
            source_node_id = current_scope
            for info in document.token_structures:
                token = document.tokens[info.token_index]
                if token.start_byte >= node.start_byte and token.end_byte <= node.end_byte and info.symbol_id:
                    source_node_id = info.symbol_id
                    break
            records.append((source_node_id, callee_name))
        return records

    def _config_terms_from_document(self, source_path: Path, document: AlignedDocument) -> list[str]:
        terms: list[str] = [source_path.stem]
        for token in document.tokens:
            if token.token_class.value in {"identifier", "string", "number"}:
                terms.append(token.value.strip("\"'"))
        if len(terms) <= 1:
            terms.extend(self._config_terms(source_path))
        return self._unique_terms(terms)

    def _symbol_copy_terms(self, document: AlignedDocument, symbol) -> tuple[str, ...]:
        terms: list[str] = [symbol.name]
        for token in document.tokens:
            if token.start_byte < symbol.start_byte or token.end_byte > symbol.end_byte:
                continue
            if token.token_class.value in {"identifier", "string", "number"}:
                terms.append(token.value.strip("\"'"))
        return tuple(self._unique_terms(terms))

    def _code_copy_terms_from_document(
        self,
        document: AlignedDocument,
        literals: Sequence[str],
    ) -> list[str]:
        terms: list[str] = []
        for token in document.tokens:
            if token.token_class.value in {"identifier", "string", "number"}:
                terms.append(token.value.strip("\"'"))
        terms.extend(literals)
        return self._unique_terms(terms)

    def _extract_import_spec(self, span_text: str, language: str) -> str | None:
        stripped = span_text.strip()
        if language == "python":
            match = re.search(r"from\s+([A-Za-z0-9_\.]+)\s+import", stripped)
            if match:
                return match.group(1)
            match = re.search(r"import\s+([A-Za-z0-9_\.]+)", stripped)
            if match:
                return match.group(1)
            return None
        match = re.search(r"""from\s+["'](.+?)["']""", stripped)
        if match:
            return match.group(1)
        match = re.search(r"""require\(\s*["'](.+?)["']\s*\)""", stripped)
        if match:
            return match.group(1)
        return None

    def _extract_call_name(self, span_text: str) -> str | None:
        prefix = span_text.split("(", 1)[0].strip()
        if not prefix:
            return None
        if "." in prefix:
            prefix = prefix.split(".")[-1]
        prefix = prefix.split()[-1]
        return prefix or None

    def _line_for_byte_offset(self, document: AlignedDocument, byte_offset: int) -> int:
        for token in document.tokens:
            if token.start_byte <= byte_offset <= token.end_byte:
                return token.line
        if document.tokens:
            return document.tokens[-1].line
        return 1

    def _import_specs_from_nodes(self, nodes: Sequence[RepositoryGraphNode]) -> list[str]:
        specs: list[str] = []
        for node in nodes:
            if node.kind == "import":
                specs.append(node.name)
        return specs

    def _exported_names_from_nodes(self, nodes: Sequence[RepositoryGraphNode]) -> list[str]:
        exports: list[str] = []
        for node in nodes:
            if node.kind in {"function", "class", "symbol"}:
                exports.append(node.name)
        return exports

    def _call_records_from_nodes(self, nodes: Sequence[RepositoryGraphNode]) -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        for node in nodes:
            if node.kind == "symbol" and "reference_target" in node.metadata:
                records.append((str(node.metadata["source_node_id"]), str(node.metadata["reference_target"])))
        return records

    def _python_call_records(self, tree: ast.AST, relative_path: str) -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        current_scope = self._file_node_id(relative_path)

        class Visitor(ast.NodeVisitor):
            def __init__(self, outer: "RepositoryGraphIndexer") -> None:
                self.outer = outer
                self.scope_stack: list[str] = [current_scope]

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                scope_id = self.outer._symbol_node_id(
                    relative_path,
                    "function",
                    node.name,
                    int(getattr(node, "lineno", 0)),
                )
                self.scope_stack.append(scope_id)
                self.generic_visit(node)
                self.scope_stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                scope_id = self.outer._symbol_node_id(
                    relative_path,
                    "class",
                    node.name,
                    int(getattr(node, "lineno", 0)),
                )
                self.scope_stack.append(scope_id)
                self.generic_visit(node)
                self.scope_stack.pop()

            def visit_Call(self, node: ast.Call) -> Any:
                callee_name = ""
                if isinstance(node.func, ast.Name):
                    callee_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    callee_name = node.func.attr
                if callee_name:
                    records.append((self.scope_stack[-1], callee_name))
                self.generic_visit(node)

        Visitor(self).visit(tree)
        return records

    def _config_terms(self, source_path: Path) -> list[str]:
        suffix = source_path.suffix.lower()
        content = source_path.read_text(encoding="utf-8")
        terms: list[str] = [source_path.stem]
        try:
            if suffix == ".json":
                payload = json.loads(content)
                terms.extend(self._flatten_scalar_terms(payload))
            elif suffix in {".yaml", ".yml"}:
                payload = yaml.safe_load(content)
                terms.extend(self._flatten_scalar_terms(payload))
            elif suffix in {".ini", ".cfg"}:
                parser = configparser.ConfigParser()
                parser.read_string(content)
                for section in parser.sections():
                    terms.append(section)
                    for key, value in parser.items(section):
                        terms.extend([key, str(value)])
            elif suffix == ".toml":
                if tomllib is not None:
                    payload = tomllib.loads(content)
                    terms.extend(self._flatten_scalar_terms(payload))
                else:
                    for line in content.splitlines():
                        if "=" in line and not line.strip().startswith("#"):
                            left, right = line.split("=", 1)
                            terms.extend([left.strip(), right.strip().strip("\"'")])
        except Exception:
            terms.extend(self._literal_terms_from_text(content))
        return self._unique_terms(terms)

    def _ingest_reports(
        self,
        root: Path,
        report_paths: Sequence[str],
        nodes_by_id: dict[str, RepositoryGraphNode],
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], dict[str, set[str]]]:
        nodes: list[RepositoryGraphNode] = []
        edges: list[RepositoryGraphEdge] = []
        targets_by_source: dict[str, set[str]] = {}
        for report_path in report_paths:
            path = Path(report_path).resolve()
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            if suffix == ".xml":
                parsed_nodes, parsed_edges, targets = self._parse_junit_report(root, path, nodes_by_id)
            elif suffix == ".json":
                parsed_nodes, parsed_edges, targets = self._parse_eslint_report(root, path, nodes_by_id)
            else:
                parsed_nodes, parsed_edges, targets = self._parse_tsc_report(root, path, nodes_by_id)
            nodes.extend(parsed_nodes)
            edges.extend(parsed_edges)
            for source_file, target_files in targets.items():
                targets_by_source.setdefault(source_file, set()).update(target_files)
        return nodes, edges, targets_by_source

    def _parse_junit_report(
        self,
        root: Path,
        path: Path,
        nodes_by_id: dict[str, RepositoryGraphNode],
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], dict[str, set[str]]]:
        nodes: list[RepositoryGraphNode] = []
        edges: list[RepositoryGraphEdge] = []
        targets: dict[str, set[str]] = {}
        tree = ElementTree.parse(path)
        counter = 0
        for testcase in tree.findall(".//testcase"):
            failure = testcase.find("failure")
            if failure is None:
                failure = testcase.find("error")
            if failure is None:
                continue
            counter += 1
            file_attr = testcase.attrib.get("file", "")
            class_name = testcase.attrib.get("classname", "")
            target_file = self._resolve_report_target(root, file_attr or class_name)
            diagnostic_node = self._make_diagnostic_node(
                target_file=target_file,
                name=testcase.attrib.get("name", f"diagnostic_{counter}"),
                copy_terms=(
                    testcase.attrib.get("name", ""),
                    class_name,
                    failure.attrib.get("message", ""),
                ),
                report_path=str(path),
                sequence=counter,
            )
            nodes.append(diagnostic_node)
            if target_file is not None:
                targets.setdefault(target_file, set()).add(target_file)
                file_node_id = self._file_node_id(target_file)
                if file_node_id in nodes_by_id:
                    edges.append(
                        RepositoryGraphEdge(
                            source_id=file_node_id,
                            target_id=diagnostic_node.node_id,
                            kind="fails_with",
                        )
                    )
        return nodes, edges, targets

    def _parse_eslint_report(
        self,
        root: Path,
        path: Path,
        nodes_by_id: dict[str, RepositoryGraphNode],
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], dict[str, set[str]]]:
        nodes: list[RepositoryGraphNode] = []
        edges: list[RepositoryGraphEdge] = []
        targets: dict[str, set[str]] = {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        counter = 0
        for entry in payload:
            target_file = self._resolve_path_to_relative(root, entry.get("filePath", ""))
            for message in entry.get("messages", []):
                counter += 1
                diagnostic_node = self._make_diagnostic_node(
                    target_file=target_file,
                    name=message.get("ruleId") or f"eslint_{counter}",
                    copy_terms=(message.get("ruleId", ""), message.get("message", "")),
                    report_path=str(path),
                    sequence=counter,
                    heuristic=True,
                )
                nodes.append(diagnostic_node)
                if target_file is not None:
                    targets.setdefault(target_file, set()).add(target_file)
                    file_node_id = self._file_node_id(target_file)
                    if file_node_id in nodes_by_id:
                        edges.append(
                            RepositoryGraphEdge(
                                source_id=file_node_id,
                                target_id=diagnostic_node.node_id,
                                kind="fails_with",
                                heuristic=True,
                            )
                        )
        return nodes, edges, targets

    def _parse_tsc_report(
        self,
        root: Path,
        path: Path,
        nodes_by_id: dict[str, RepositoryGraphNode],
    ) -> tuple[list[RepositoryGraphNode], list[RepositoryGraphEdge], dict[str, set[str]]]:
        nodes: list[RepositoryGraphNode] = []
        edges: list[RepositoryGraphEdge] = []
        targets: dict[str, set[str]] = {}
        counter = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            match = TSC_DIAGNOSTIC_RE.match(line.strip())
            if match is None:
                continue
            counter += 1
            target_file = self._resolve_path_to_relative(root, match.group("path"))
            diagnostic_node = self._make_diagnostic_node(
                target_file=target_file,
                name=match.group("code"),
                copy_terms=(match.group("code"), match.group("message")),
                report_path=str(path),
                sequence=counter,
                heuristic=True,
            )
            nodes.append(diagnostic_node)
            if target_file is not None:
                targets.setdefault(target_file, set()).add(target_file)
                file_node_id = self._file_node_id(target_file)
                if file_node_id in nodes_by_id:
                    edges.append(
                        RepositoryGraphEdge(
                            source_id=file_node_id,
                            target_id=diagnostic_node.node_id,
                            kind="fails_with",
                            heuristic=True,
                        )
                    )
        return nodes, edges, targets

    def _resolve_call_target(
        self,
        file_path: str,
        callee_name: str,
        symbol_records_by_name: dict[str, list[_SymbolRecord]],
        imported_files: set[str],
    ) -> _SymbolRecord | None:
        candidates = symbol_records_by_name.get(callee_name, [])
        if not candidates:
            return None
        for candidate in candidates:
            if candidate.file_path == file_path:
                return candidate
        for candidate in candidates:
            if candidate.file_path in imported_files:
                return candidate
        return candidates[0]

    def _build_import_closure(self, file_records: dict[str, _FileRecord]) -> dict[str, set[str]]:
        closure: dict[str, set[str]] = {path: set(record.resolved_imports) for path, record in file_records.items()}
        changed = True
        while changed:
            changed = False
            for path, imported in closure.items():
                current = set(imported)
                for imported_path in tuple(imported):
                    current.update(closure.get(imported_path, set()))
                if current != imported:
                    closure[path] = current
                    changed = True
        return closure

    def _resolve_import(self, source_path: Path, root: Path, import_spec: str) -> Path | None:
        spec = import_spec.strip()
        if not spec:
            return None
        if source_path.suffix.lower() in PYTHON_EXTENSIONS:
            return self._resolve_python_import(source_path, root, spec)
        return self._resolve_js_import(source_path, root, spec)

    def _resolve_python_import(self, source_path: Path, root: Path, import_spec: str) -> Path | None:
        level = len(import_spec) - len(import_spec.lstrip("."))
        module = import_spec.lstrip(".")
        base_path = source_path.parent
        if level > 0:
            for _ in range(max(level - 1, 0)):
                base_path = base_path.parent
        else:
            base_path = root
        module_path = base_path / Path(*module.split(".")) if module else base_path
        for candidate in (module_path.with_suffix(".py"), module_path / "__init__.py"):
            if candidate.exists():
                return candidate.resolve()
        return None

    def _resolve_js_import(self, source_path: Path, root: Path, import_spec: str) -> Path | None:
        base = source_path.parent if import_spec.startswith(".") else root
        candidate = (base / import_spec).resolve()
        candidates = [
            candidate,
            candidate.with_suffix(".ts"),
            candidate.with_suffix(".tsx"),
            candidate.with_suffix(".js"),
            candidate.with_suffix(".jsx"),
            candidate / "index.ts",
            candidate / "index.js",
        ]
        for resolved in candidates:
            if resolved.exists():
                return resolved
        return None

    def _resolve_report_target(self, root: Path, raw_value: str) -> str | None:
        if not raw_value:
            return None
        normalized = raw_value if "/" in raw_value or "\\" in raw_value else raw_value.replace(".", "/")
        candidate = self._resolve_path_to_relative(root, normalized)
        if candidate is not None:
            return candidate
        for suffix in (".py", ".ts", ".js"):
            candidate = self._resolve_path_to_relative(root, normalized + suffix)
            if candidate is not None:
                return candidate
        return None

    def _resolve_path_to_relative(self, root: Path, raw_path: str) -> str | None:
        cleaned = raw_path.replace("\\", "/")
        path = Path(cleaned)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(root).as_posix()
            except ValueError:
                return None
        candidate = (root / path).resolve()
        if candidate.exists():
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:
                return None
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.as_posix().endswith(cleaned):
                return file_path.relative_to(root).as_posix()
        return None

    def _make_file_node(self, relative_path: str, source_path: Path) -> RepositoryGraphNode:
        copy_terms = self._unique_terms([source_path.stem, *source_path.parts[-3:]])
        return RepositoryGraphNode(
            node_id=self._file_node_id(relative_path),
            kind="file",
            name=relative_path,
            file_path=relative_path,
            copy_terms=tuple(copy_terms),
            key=self._encode_terms((relative_path, source_path.stem, "file"), self.key_dim),
            value=self._encode_terms(tuple(copy_terms), self.value_dim),
            metadata={"language": self._language_for_path(source_path)},
        )

    def _make_symbol_node(
        self,
        kind: str,
        relative_path: str,
        name: str,
        line: int,
        copy_terms: Sequence[str],
        heuristic: bool = False,
    ) -> RepositoryGraphNode:
        node_id = self._symbol_node_id(relative_path, kind, name, line)
        unique_terms = self._unique_terms([name, *copy_terms])
        return RepositoryGraphNode(
            node_id=node_id,
            kind=kind,
            name=name,
            file_path=relative_path,
            copy_terms=tuple(unique_terms),
            key=self._encode_terms((kind, name, relative_path), self.key_dim),
            value=self._encode_terms(tuple(unique_terms), self.value_dim),
            heuristic=heuristic,
            metadata={"line": line, "symbol_name": name},
        )

    def _make_file_overlay_node(
        self,
        relative_path: str,
        kind: str,
        name: str,
        copy_terms: Sequence[str],
        heuristic: bool = False,
    ) -> RepositoryGraphNode:
        unique_terms = self._unique_terms([name, *copy_terms])
        return RepositoryGraphNode(
            node_id=f"overlay:{relative_path}:{name}",
            kind=kind,
            name=name,
            file_path=relative_path,
            copy_terms=tuple(unique_terms),
            key=self._encode_terms((relative_path, name, kind), self.key_dim),
            value=self._encode_terms(tuple(unique_terms), self.value_dim),
            heuristic=heuristic,
            metadata={"symbol_name": name},
        )

    def _make_import_node(
        self,
        relative_path: str,
        import_spec: str,
        sequence: int,
        heuristic: bool = False,
    ) -> RepositoryGraphNode:
        copy_terms = tuple(self._unique_terms([import_spec, *import_spec.split(".")]))
        return RepositoryGraphNode(
            node_id=f"import:{relative_path}:{sequence}",
            kind="import",
            name=import_spec,
            file_path=relative_path,
            copy_terms=copy_terms,
            key=self._encode_terms(("import", import_spec, relative_path), self.key_dim),
            value=self._encode_terms(copy_terms, self.value_dim),
            heuristic=heuristic,
            metadata={"import_spec": import_spec},
        )

    def _make_test_node(
        self,
        relative_path: str,
        name: str,
        copy_terms: Sequence[str],
        heuristic: bool = False,
    ) -> RepositoryGraphNode:
        unique_terms = self._unique_terms([name, *copy_terms, "test"])
        return RepositoryGraphNode(
            node_id=f"test:{relative_path}:{name}",
            kind="test",
            name=name,
            file_path=relative_path,
            copy_terms=tuple(unique_terms),
            key=self._encode_terms(("test", relative_path, name), self.key_dim),
            value=self._encode_terms(tuple(unique_terms), self.value_dim),
            heuristic=heuristic,
            metadata={"symbol_name": name},
        )

    def _make_config_node(self, relative_path: str, name: str, copy_terms: Sequence[str]) -> RepositoryGraphNode:
        unique_terms = self._unique_terms([name, *copy_terms, "config"])
        return RepositoryGraphNode(
            node_id=f"config:{relative_path}",
            kind="config",
            name=name,
            file_path=relative_path,
            copy_terms=tuple(unique_terms),
            key=self._encode_terms(("config", relative_path, name), self.key_dim),
            value=self._encode_terms(tuple(unique_terms), self.value_dim),
            metadata={},
        )

    def _make_diagnostic_node(
        self,
        target_file: str | None,
        name: str,
        copy_terms: Sequence[str],
        report_path: str,
        sequence: int,
        heuristic: bool = False,
    ) -> RepositoryGraphNode:
        unique_terms = self._unique_terms([name, *copy_terms, "diagnostic"])
        file_path = target_file or Path(report_path).name
        return RepositoryGraphNode(
            node_id=f"diagnostic:{file_path}:{sequence}",
            kind="diagnostic",
            name=name,
            file_path=target_file,
            copy_terms=tuple(unique_terms),
            key=self._encode_terms(("diagnostic", file_path, name), self.key_dim),
            value=self._encode_terms(tuple(unique_terms), self.value_dim),
            heuristic=heuristic,
            metadata={"report_path": report_path},
        )

    def _make_reference_node(
        self,
        relative_path: str,
        source_node_id: str,
        callee_name: str,
        heuristic: bool = False,
    ) -> RepositoryGraphNode:
        return RepositoryGraphNode(
            node_id=f"ref:{source_node_id}:{callee_name}",
            kind="symbol",
            name=callee_name,
            file_path=relative_path,
            copy_terms=(callee_name,),
            key=self._encode_terms(("reference", relative_path, callee_name), self.key_dim),
            value=self._encode_terms((callee_name,), self.value_dim),
            heuristic=heuristic,
            metadata={"source_node_id": source_node_id, "reference_target": callee_name, "symbol_name": callee_name},
        )

    def _register_node(
        self,
        nodes: list[RepositoryGraphNode],
        nodes_by_id: dict[str, RepositoryGraphNode],
        node_ids_by_file: dict[str, list[str]],
        node: RepositoryGraphNode,
    ) -> None:
        if node.node_id in nodes_by_id:
            return
        nodes.append(node)
        nodes_by_id[node.node_id] = node
        if node.file_path is not None:
            node_ids_by_file.setdefault(node.file_path, []).append(node.node_id)

    def _flatten_scalar_terms(self, payload: Any) -> list[str]:
        terms: list[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                terms.append(str(key))
                terms.extend(self._flatten_scalar_terms(value))
        elif isinstance(payload, list):
            for item in payload:
                terms.extend(self._flatten_scalar_terms(item))
        elif isinstance(payload, (str, int, float, bool)):
            terms.append(str(payload))
        return self._unique_terms(terms)

    def _python_file_copy_terms(self, tree: ast.AST, literals: Sequence[str]) -> list[str]:
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
        return self._unique_terms([*names, *literals])

    def _literal_terms_from_ast(self, node: ast.AST) -> list[str]:
        terms: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, (str, int, float)):
                terms.append(str(child.value))
        return self._unique_terms(terms)

    def _literal_terms_from_text(self, content: str) -> list[str]:
        return self._unique_terms([*STRING_RE.findall(content), *NUMBER_RE.findall(content)])

    def _is_test_file(self, relative_path: str, content: str) -> bool:
        path_lower = relative_path.lower()
        if any(part in path_lower for part in TEST_PATH_MARKERS):
            return True
        return any(marker in content for marker in JS_TEST_MARKERS)

    def _encode_terms(self, terms: Sequence[str], dim: int) -> torch.Tensor:
        vector = torch.zeros(dim, dtype=torch.float32)
        for term in terms:
            text = str(term).strip()
            if not text:
                continue
            index = stable_int_hash(text, dim)
            sign = -1.0 if stable_int_hash(f"sign:{text}", 2) == 0 else 1.0
            vector[index] += sign
        norm = vector.norm(p=2)
        if norm.item() > 0:
            vector = vector / norm
        return vector

    def _language_for_path(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in PYTHON_EXTENSIONS:
            return "python"
        if suffix in JS_TS_EXTENSIONS:
            return "typescript" if suffix in {".ts", ".tsx"} else "javascript"
        return "config"

    def _relative_path(self, path: Path, root: Path) -> str:
        return path.resolve().relative_to(root).as_posix()

    def _file_node_id(self, relative_path: str) -> str:
        return f"file:{relative_path}"

    def _symbol_node_id(self, relative_path: str, kind: str, name: str, line: int) -> str:
        return f"{kind}:{relative_path}:{name}:{line}"

    def _line_for_offset(self, content: str, offset: int) -> int:
        return content.count("\n", 0, offset) + 1

    def _unique_terms(self, values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(text)
        return unique


class RepositoryGraphMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        vocab_size: int,
        top_k: int,
        graph_copy_weight: float,
        samefile_bias: float,
        import_bias: float,
        symbol_bias: float,
        test_bias: float,
        diagnostic_bias: float,
        value_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.value_dim = value_dim or hidden_size
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.graph_copy_weight = graph_copy_weight
        self.samefile_bias = samefile_bias
        self.import_bias = import_bias
        self.symbol_bias = symbol_bias
        self.test_bias = test_bias
        self.diagnostic_bias = diagnostic_bias
        self.term_vocab_size = 2048
        self.kind_to_id = {
            "file": 0,
            "symbol": 1,
            "function": 2,
            "class": 3,
            "import": 4,
            "test": 5,
            "diagnostic": 6,
            "config": 7,
        }
        self.edge_kind_to_id = {
            "contains": 0,
            "defines": 1,
            "imports": 2,
            "calls": 3,
            "references": 4,
            "tested_by": 5,
            "fails_with": 6,
        }
        self.flag_to_id = {
            "heuristic": 0,
            "samefile": 1,
            "symbol_name": 2,
            "import_spec": 3,
            "report_path": 4,
            "test": 5,
            "diagnostic": 6,
        }
        self.term_embedding = nn.Embedding(self.term_vocab_size, key_dim)
        self.kind_embedding = nn.Embedding(len(self.kind_to_id), key_dim)
        self.edge_kind_embedding = nn.Embedding(len(self.edge_kind_to_id), key_dim)
        self.flag_embedding = nn.Embedding(len(self.flag_to_id), key_dim)
        self.node_feature_norm = nn.LayerNorm(key_dim)
        self.node_key_projection = nn.Linear(key_dim, key_dim)
        self.node_value_projection = nn.Linear(key_dim, self.value_dim)
        self.context_projection = nn.Linear(key_dim, key_dim)
        self.query_projection = nn.Linear(hidden_size + key_dim, key_dim)
        self.prior_head = nn.Linear(self.value_dim, vocab_size)
        self.index: RepositoryGraphIndex | None = None
        self.node_edge_kinds: dict[str, tuple[str, ...]] = {}

    def set_index(self, index: RepositoryGraphIndex | None) -> None:
        self.index = index
        if index is None:
            self.node_edge_kinds = {}
            return
        edge_kinds: dict[str, list[str]] = {}
        for edge in index.edges:
            edge_kinds.setdefault(edge.source_id, []).append(edge.kind)
            edge_kinds.setdefault(edge.target_id, []).append(edge.kind)
        self.node_edge_kinds = {
            node_id: tuple(kinds) for node_id, kinds in edge_kinds.items()
        }

    def reset(self) -> None:
        return None

    def query(
        self,
        hidden: torch.Tensor,
        context: RepoGraphQueryContext,
        vocabulary_snapshot: VocabularySnapshot,
    ) -> RepoGraphReadResult:
        device = hidden.device
        distribution = torch.zeros(self.vocab_size, device=device)
        log_distribution = torch.full((self.vocab_size,), fill_value=math.log(1e-8), device=device)
        attention = torch.zeros(self.top_k, device=device)
        copy_token_ids = torch.full((self.vocab_size,), fill_value=-1, dtype=torch.long, device=device)
        graph_context = torch.zeros(self.value_dim, device=device)

        if self.index is None or not self.index.nodes:
            return RepoGraphReadResult(
                graph_context=graph_context,
                distribution=distribution,
                log_distribution=log_distribution,
                attention=attention,
                copy_token_ids=copy_token_ids[:0],
                candidate_scores=torch.zeros(0, device=device),
                candidate_node_ids=(),
                candidate_kinds=(),
                candidate_names=(),
                retrieved_count=0,
                read_count=0,
                candidate_count=0,
                copy_supported_count=0,
                samefile_hits=0,
                import_hits=0,
                symbol_hits=0,
                test_hits=0,
                diagnostic_hits=0,
                target_node_id=None,
            )

        context_features = self._encode_context(context, device)
        query = self.query_projection(torch.cat([hidden, context_features], dim=-1))
        scored: list[tuple[float, torch.Tensor, RepositoryGraphNode, dict[str, bool], torch.Tensor]] = []
        import_closure = set(self.index.import_closure_by_file.get(context.file_path, ()))
        test_files = set(self.index.test_files_by_source.get(context.file_path, ()))
        diagnostic_files = set(self.index.diagnostic_files_by_source.get(context.file_path, ()))

        for node in self.index.nodes:
            key, value = self._encode_node(node, device)
            score_tensor = (query @ key) / math.sqrt(self.key_dim)
            score = float(score_tensor.detach().item())
            flags = {
                "samefile": node.file_path == context.file_path,
                "import": node.file_path in import_closure if node.file_path is not None else False,
                "symbol": self._in_symbol_closure(node, context),
                "test": self._in_test_relation(node, test_files),
                "diagnostic": self._in_diagnostic_relation(node, diagnostic_files),
            }
            if flags["samefile"]:
                score += self.samefile_bias
            if flags["import"]:
                score += self.import_bias
            if flags["symbol"]:
                score += self.symbol_bias
            if flags["test"]:
                score += self.test_bias
            if flags["diagnostic"]:
                score += self.diagnostic_bias
            priority_bonus = self._priority_bonus(node=node, context=context, flags=flags)
            score += priority_bonus
            scored.append(
                (
                    score,
                    score_tensor + self._bias_tensor(flags, device) + torch.tensor(priority_bonus, dtype=torch.float32, device=device),
                    node,
                    flags,
                    value,
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: self.top_k]
        if not selected:
            return RepoGraphReadResult(
                graph_context=graph_context,
                distribution=distribution,
                log_distribution=log_distribution,
                attention=attention,
                copy_token_ids=copy_token_ids[:0],
                candidate_scores=torch.zeros(0, device=device),
                candidate_node_ids=(),
                candidate_kinds=(),
                candidate_names=(),
                retrieved_count=0,
                read_count=0,
                candidate_count=len(scored),
                copy_supported_count=0,
                samefile_hits=0,
                import_hits=0,
                symbol_hits=0,
                test_hits=0,
                diagnostic_hits=0,
                target_node_id=None,
            )

        score_tensor = torch.stack([item[1] for item in selected], dim=0)
        weights = torch.softmax(score_tensor, dim=0)
        attention[: len(selected)] = weights
        candidate_node_ids = tuple(item[2].node_id for item in selected)
        candidate_kinds = tuple(item[2].kind for item in selected)
        candidate_names = tuple(item[2].name for item in selected)
        samefile_hits = sum(1 for _, _, _, flags, _ in selected if flags["samefile"])
        import_hits = sum(1 for _, _, _, flags, _ in selected if flags["import"])
        symbol_hits = sum(1 for _, _, _, flags, _ in selected if flags["symbol"])
        test_hits = sum(1 for _, _, _, flags, _ in selected if flags["test"])
        diagnostic_hits = sum(1 for _, _, _, flags, _ in selected if flags["diagnostic"])
        target_node_id = self._select_target_node_id(selected, context)

        context_vectors = torch.stack([item[4] for item in selected], dim=0)
        graph_context = torch.sum(weights.unsqueeze(-1) * context_vectors, dim=0)
        prior_logits = self.prior_head(graph_context)
        prior_distribution = torch.softmax(prior_logits, dim=-1)

        copy_distribution = torch.zeros(self.vocab_size, device=device)
        supported_token_ids: list[int] = []
        for weight, (_, _, node, _, _) in zip(weights, selected, strict=False):
            supported_ids = self._copy_token_ids_for_terms(node.copy_terms, vocabulary_snapshot)
            if not supported_ids:
                continue
            mass = weight / len(supported_ids)
            supported_token_ids.extend(supported_ids)
            copy_distribution.index_add_(
                0,
                torch.tensor(supported_ids, dtype=torch.long, device=device),
                torch.full((len(supported_ids),), fill_value=float(mass.item()), device=device),
            )

        if copy_distribution.sum().item() > 0:
            copy_distribution = copy_distribution / copy_distribution.sum().clamp_min(1e-8)
            graph_distribution = (
                (1.0 - self.graph_copy_weight) * prior_distribution
                + self.graph_copy_weight * copy_distribution
            )
        else:
            graph_distribution = prior_distribution

        log_distribution = torch.log(graph_distribution.clamp_min(1e-8))
        unique_supported = sorted(set(supported_token_ids))
        copy_token_ids = torch.tensor(unique_supported, dtype=torch.long, device=device)

        return RepoGraphReadResult(
            graph_context=graph_context,
            distribution=graph_distribution,
            log_distribution=log_distribution,
            attention=attention,
            copy_token_ids=copy_token_ids,
            candidate_scores=score_tensor,
            candidate_node_ids=candidate_node_ids,
            candidate_kinds=candidate_kinds,
            candidate_names=candidate_names,
            retrieved_count=len(selected),
            read_count=len(selected),
            candidate_count=len(scored),
            copy_supported_count=len(unique_supported),
            samefile_hits=samefile_hits,
            import_hits=import_hits,
            symbol_hits=symbol_hits,
            test_hits=test_hits,
            diagnostic_hits=diagnostic_hits,
            target_node_id=target_node_id,
        )

    def _copy_token_ids_for_terms(
        self,
        terms: Sequence[str],
        vocabulary_snapshot: VocabularySnapshot,
    ) -> list[int]:
        token_ids: list[int] = []
        seen: set[int] = set()
        for term in terms:
            for alias in self._copy_term_aliases(term):
                token_id = vocabulary_snapshot.lookup_token(alias)
                if token_id is None or token_id in seen:
                    continue
                seen.add(token_id)
                token_ids.append(token_id)
        return token_ids

    def _copy_term_aliases(self, term: str) -> tuple[str, ...]:
        raw = str(term).strip()
        if not raw:
            return ()
        stripped = raw.strip("\"'")
        aliases = [raw]
        if stripped and stripped != raw:
            aliases.append(stripped)
        if stripped:
            aliases.append(f"\"{stripped}\"")
            aliases.append(f"'{stripped}'")
        unique: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            if alias and alias not in seen:
                seen.add(alias)
                unique.append(alias)
        return tuple(unique)

    def _encode_context(self, context: RepoGraphQueryContext, device: torch.device) -> torch.Tensor:
        terms = [
            context.file_path,
            context.current_symbol_name or "",
            *context.scope_path,
            context.token_value,
            context.token_class,
            context.target_copy_value or "",
        ]
        embedded = self._embed_terms(terms, device)
        return self.context_projection(embedded)

    def _bias_tensor(self, flags: dict[str, bool], device: torch.device) -> torch.Tensor:
        value = 0.0
        if flags["samefile"]:
            value += self.samefile_bias
        if flags["import"]:
            value += self.import_bias
        if flags["symbol"]:
            value += self.symbol_bias
        if flags["test"]:
            value += self.test_bias
        if flags["diagnostic"]:
            value += self.diagnostic_bias
        return torch.tensor(value, dtype=torch.float32, device=device)

    def _encode_node(self, node: RepositoryGraphNode, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        term_features = node.key.to(device)
        kind_id = self.kind_to_id.get(node.kind, self.kind_to_id["symbol"])
        kind_features = self.kind_embedding(
            torch.tensor(kind_id, dtype=torch.long, device=device)
        )
        edge_features = self._embed_edge_kinds(self.node_edge_kinds.get(node.node_id, ()), device)
        flags = []
        if node.heuristic:
            flags.append("heuristic")
        if node.kind == "test":
            flags.append("test")
        if node.kind == "diagnostic":
            flags.append("diagnostic")
        if "symbol_name" in node.metadata:
            flags.append("symbol_name")
        if "import_spec" in node.metadata:
            flags.append("import_spec")
        if "report_path" in node.metadata:
            flags.append("report_path")
        flag_features = self._embed_flags(flags, device)
        feature = self.node_feature_norm(term_features + kind_features + edge_features + flag_features)
        value_input = node.value.to(device) + feature
        return self.node_key_projection(feature), self.node_value_projection(value_input)

    def _embed_terms(self, terms: Sequence[str], device: torch.device) -> torch.Tensor:
        indices = [
            stable_int_hash(text, self.term_vocab_size)
            for text in (str(term).strip() for term in terms)
            if text
        ]
        if not indices:
            return torch.zeros(self.key_dim, device=device)
        tensor = torch.tensor(indices, dtype=torch.long, device=device)
        return self.term_embedding(tensor).mean(dim=0)

    def _embed_edge_kinds(self, edge_kinds: Sequence[str], device: torch.device) -> torch.Tensor:
        indices = [
            self.edge_kind_to_id[kind]
            for kind in edge_kinds
            if kind in self.edge_kind_to_id
        ]
        if not indices:
            return torch.zeros(self.key_dim, device=device)
        tensor = torch.tensor(indices, dtype=torch.long, device=device)
        return self.edge_kind_embedding(tensor).mean(dim=0)

    def _embed_flags(self, flags: Sequence[str], device: torch.device) -> torch.Tensor:
        indices = [
            self.flag_to_id[flag]
            for flag in flags
            if flag in self.flag_to_id
        ]
        if not indices:
            return torch.zeros(self.key_dim, device=device)
        tensor = torch.tensor(indices, dtype=torch.long, device=device)
        return self.flag_embedding(tensor).mean(dim=0)

    def _in_symbol_closure(self, node: RepositoryGraphNode, context: RepoGraphQueryContext) -> bool:
        if context.current_symbol_name is None:
            return False
        symbol_name = str(node.metadata.get("symbol_name", ""))
        return symbol_name == context.current_symbol_name or context.current_symbol_name in node.copy_terms

    def _in_test_relation(self, node: RepositoryGraphNode, test_files: set[str]) -> bool:
        if not test_files:
            return False
        if node.kind == "test":
            return node.file_path in test_files
        return node.file_path in test_files if node.file_path is not None else False

    def _in_diagnostic_relation(self, node: RepositoryGraphNode, diagnostic_files: set[str]) -> bool:
        if not diagnostic_files:
            return False
        if node.kind == "diagnostic":
            return node.file_path in diagnostic_files
        return node.file_path in diagnostic_files if node.file_path is not None else False

    def _priority_bonus(
        self,
        *,
        node: RepositoryGraphNode,
        context: RepoGraphQueryContext,
        flags: dict[str, bool],
    ) -> float:
        token_term = (context.target_token_value or context.token_value or "").strip("\"'")
        copy_term = (context.target_copy_value or "").strip("\"'")
        bonus = 0.0
        if node.kind in {"symbol", "function", "class"}:
            if context.target_symbol_name and node.name == context.target_symbol_name:
                bonus += 1.25
            elif context.current_symbol_name and node.name == context.current_symbol_name:
                bonus += 0.85
            if token_term:
                if node.name == token_term:
                    bonus += 0.75
                elif token_term in node.copy_terms:
                    bonus += 0.45 if flags["samefile"] else 0.35 if flags["import"] else 0.2
            if copy_term:
                if node.name == copy_term:
                    bonus += 5.0
                elif copy_term in node.copy_terms:
                    bonus += 4.0 if flags["import"] else 3.0 if flags["samefile"] else 2.0
            if flags["samefile"]:
                bonus += 0.15
            if flags["import"]:
                bonus += 0.1
        elif node.kind == "diagnostic" and context.probe_kind in {"diagnostic_to_symbol", "edit_fix"} and flags["diagnostic"]:
            bonus += 0.3
        elif node.kind == "test" and context.probe_kind in {"diagnostic_to_symbol", "edit_fix"} and flags["test"]:
            bonus += 0.15
        return bonus

    def _select_target_node_id(
        self,
        selected: list[tuple[float, torch.Tensor, RepositoryGraphNode, dict[str, bool], torch.Tensor]],
        context: RepoGraphQueryContext,
    ) -> str | None:
        token_term = (context.target_token_value or context.token_value or "").strip("\"'")
        probe_kind = context.probe_kind or ""
        predicates = [
            lambda node, flags: context.target_symbol_name is not None
            and node.kind in {"symbol", "function", "class"}
            and node.name == context.target_symbol_name,
            lambda node, flags: token_term
            and node.kind in {"symbol", "function", "class"}
            and node.name == token_term
            and flags["samefile"],
            lambda node, flags: token_term
            and node.kind in {"symbol", "function", "class"}
            and token_term in node.copy_terms
            and flags["samefile"],
            lambda node, flags: token_term
            and node.kind in {"symbol", "function", "class"}
            and node.name == token_term
            and flags["import"],
            lambda node, flags: token_term
            and node.kind in {"symbol", "function", "class"}
            and token_term in node.copy_terms
            and flags["import"],
            lambda node, flags: probe_kind in {"diagnostic_to_symbol", "edit_fix"}
            and node.kind == "diagnostic"
            and flags["diagnostic"],
            lambda node, flags: probe_kind in {"diagnostic_to_symbol", "edit_fix"}
            and node.kind == "test"
            and flags["test"],
            lambda node, flags: token_term and token_term in node.copy_terms,
        ]
        for predicate in predicates:
            for _score, _score_tensor, node, flags, _value in selected:
                if predicate(node, flags):
                    return node.node_id
        return None


class NoOpRepoGraph:
    def set_index(self, index: RepositoryGraphIndex | None) -> None:
        return None

    def reset(self) -> None:
        return None

    def query(
        self,
        hidden: torch.Tensor,
        context: RepoGraphQueryContext,
        vocabulary_snapshot: VocabularySnapshot,
    ) -> RepoGraphReadResult:
        device = hidden.device
        return RepoGraphReadResult(
            graph_context=torch.zeros(hidden.shape[0], device=device) if hidden.dim() > 1 else torch.zeros_like(hidden),
            distribution=torch.zeros(1, device=device),
            log_distribution=torch.zeros(1, device=device),
            attention=torch.zeros(1, device=device),
            copy_token_ids=torch.full((0,), fill_value=-1, dtype=torch.long, device=device),
            candidate_scores=torch.zeros(0, device=device),
            candidate_node_ids=(),
            candidate_kinds=(),
            candidate_names=(),
            retrieved_count=0,
            read_count=0,
            candidate_count=0,
            copy_supported_count=0,
            samefile_hits=0,
            import_hits=0,
            symbol_hits=0,
            test_hits=0,
            diagnostic_hits=0,
            target_node_id=None,
        )
