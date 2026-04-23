from __future__ import annotations

from pathlib import Path

import pytest

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.featurizer import build_batch_from_document
from htm_code_native.data.vocabulary import VocabularyRegistry
from htm_code_native.tokenizer.boundary import BoundaryScheduler
from htm_code_native.tokenizer.tree_sitter_backend import detect_language, parse_source_document


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def config() -> HTMCodeNativeConfig:
    return HTMCodeNativeConfig.from_yaml(Path("configs/phase_a.yaml"))


@pytest.fixture()
def build_document():
    def _build(path: Path):
        source = path.read_text(encoding="utf-8")
        return parse_source_document(source, str(path), language=detect_language(str(path)))

    return _build


@pytest.fixture()
def build_batch(config: HTMCodeNativeConfig, build_document):
    def _build(path: Path, registry: VocabularyRegistry | None = None):
        document = build_document(path)
        boundaries = BoundaryScheduler(max_level=config.hssm.max_level).build(document)
        return document, boundaries, build_batch_from_document(
            document,
            boundaries,
            config,
            registry=registry,
        )

    return _build
