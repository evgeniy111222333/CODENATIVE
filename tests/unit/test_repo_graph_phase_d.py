from __future__ import annotations

from pathlib import Path

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import RepoGraphQueryContext
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.memory.repo_graph import RepositoryGraphIndexer, RepositoryGraphMemory


WORKSPACE_ROOT = Path("tests/fixtures/repo_graph_workspace")
REPORT_PATHS = [
    str(WORKSPACE_ROOT / "reports" / "junit.xml"),
    str(WORKSPACE_ROOT / "reports" / "eslint.json"),
]


def test_repo_graph_indexer_builds_multilanguage_graph() -> None:
    config = HTMCodeNativeConfig.from_yaml(Path("configs/phase_a.yaml"))
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    index = indexer.build(WORKSPACE_ROOT, report_paths=REPORT_PATHS)
    summary = index.to_summary()

    assert summary["node_kinds"]["file"] >= 4
    assert summary["node_kinds"]["function"] >= 2
    assert summary["node_kinds"]["class"] >= 1
    assert summary["node_kinds"]["import"] >= 2
    assert summary["node_kinds"]["test"] >= 1
    assert summary["node_kinds"]["diagnostic"] >= 2
    assert summary["node_kinds"]["config"] >= 1
    assert summary["edge_kinds"]["imports"] >= 2
    assert summary["edge_kinds"]["references"] >= 2
    assert summary["edge_kinds"]["tested_by"] >= 1
    assert summary["edge_kinds"]["fails_with"] >= 2

    heuristic_nodes = [node for node in index.nodes if node.file_path == "frontend/widget.ts" and node.heuristic]
    assert heuristic_nodes


def test_repo_graph_query_returns_bias_hits_and_copy_support() -> None:
    config = HTMCodeNativeConfig.from_yaml(Path("configs/phase_a.yaml"))
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    index = indexer.build(WORKSPACE_ROOT, report_paths=REPORT_PATHS)
    memory = RepositoryGraphMemory(
        hidden_size=config.model.model_dim,
        key_dim=config.model.graph_key_dim,
        vocab_size=config.model.vocabulary_size,
        top_k=16,
        graph_copy_weight=config.model.graph_copy_weight,
        samefile_bias=config.model.graph_samefile_bias,
        import_bias=config.model.graph_import_bias,
        symbol_bias=config.model.graph_symbol_bias,
        test_bias=config.model.graph_test_bias,
        diagnostic_bias=config.model.graph_diagnostic_bias,
    )
    memory.set_index(index)
    with torch.no_grad():
        memory.query_projection.weight.zero_()
        memory.query_projection.bias.zero_()
        memory.prior_head.weight.zero_()
        memory.prior_head.bias.zero_()

    registry = VocabularyRegistry(config.model.vocabulary_size)
    target_identifier = registry.encode_token("GRAPH_SHARED_NAME")
    quoted_shared_token = registry.encode_token('"shared_graph_token"')
    registry.encode_token("shared_graph_token")
    registry.encode_token("repo_graph_service")

    result = memory.query(
        hidden=torch.zeros(config.model.model_dim),
        context=RepoGraphQueryContext(
            file_path="app/core.py",
            current_symbol_id="function:app/core.py:build_payload:7",
            current_symbol_name="build_payload",
            scope_path=("build_payload",),
            token_value="GRAPH_SHARED_NAME",
            token_class="identifier",
        ),
        vocabulary_snapshot=registry.snapshot(),
    )

    assert result.retrieved_count > 0
    assert result.import_hits > 0
    assert result.test_hits > 0
    assert result.diagnostic_hits > 0
    assert "function" in result.candidate_kinds or "symbol" in result.candidate_kinds
    assert target_identifier in result.copy_token_ids.tolist()
    assert quoted_shared_token in result.copy_token_ids.tolist()
    assert result.distribution[target_identifier].item() > 0.0
    assert result.distribution[quoted_shared_token].item() > 0.0
    assert result.candidate_scores.numel() == result.retrieved_count
    assert result.candidate_scores.requires_grad is True
    assert result.target_node_id is not None


def test_repo_graph_query_prefers_explicit_symbol_target() -> None:
    config = HTMCodeNativeConfig.from_yaml(Path("configs/phase_a.yaml"))
    indexer = RepositoryGraphIndexer(
        key_dim=config.model.graph_key_dim,
        value_dim=config.model.graph_value_dim,
        max_files=config.model.repo_max_files,
    )
    index = indexer.build(WORKSPACE_ROOT, report_paths=REPORT_PATHS)
    memory = RepositoryGraphMemory(
        hidden_size=config.model.model_dim,
        key_dim=config.model.graph_key_dim,
        vocab_size=config.model.vocabulary_size,
        top_k=16,
        graph_copy_weight=config.model.graph_copy_weight,
        samefile_bias=config.model.graph_samefile_bias,
        import_bias=config.model.graph_import_bias,
        symbol_bias=config.model.graph_symbol_bias,
        test_bias=config.model.graph_test_bias,
        diagnostic_bias=config.model.graph_diagnostic_bias,
    )
    memory.set_index(index)

    expected_target_ids = {
        node.node_id
        for node in index.nodes
        if node.kind in {"symbol", "function", "class"} and "GRAPH_SHARED_NAME" in node.copy_terms
    }

    result = memory.query(
        hidden=torch.zeros(config.model.model_dim),
        context=RepoGraphQueryContext(
            file_path="app/core.py",
            current_symbol_id="function:app/core.py:build_payload:7",
            current_symbol_name="build_payload",
            scope_path=("build_payload",),
            token_value="GRAPH_SHARED_NAME",
            token_class="identifier",
            probe_kind="definition_use",
            target_symbol_name="GRAPH_SHARED_NAME",
            target_token_value="GRAPH_SHARED_NAME",
        ),
        vocabulary_snapshot=VocabularyRegistry(config.model.vocabulary_size).snapshot(),
    )

    assert expected_target_ids.intersection(result.candidate_node_ids)
    assert result.target_node_id in expected_target_ids
