from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
src_package_path = Path(__file__).resolve().parent.parent / "src" / "htm_code_native"
if src_package_path.exists():
    __path__.append(str(src_package_path))  # type: ignore[attr-defined]

from .config.settings import HTMCodeNativeConfig  # noqa: E402
from .data.vocabulary import VocabularyRegistry  # noqa: E402
from .memory.exact_recent import ExactRecentMemory  # noqa: E402
from .model.phase_a import PhaseACodeModel  # noqa: E402
from .tokenizer.python_tokenizer import PythonTokenizer  # noqa: E402
from .tokenizer.structure import PythonStructureExtractor  # noqa: E402

__all__ = [
    "HTMCodeNativeConfig",
    "VocabularyRegistry",
    "ExactRecentMemory",
    "PhaseACodeModel",
    "PythonTokenizer",
    "PythonStructureExtractor",
]
