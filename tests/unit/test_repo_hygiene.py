from __future__ import annotations

from pathlib import Path


def test_gitignore_covers_python_cache_artifacts() -> None:
    gitignore = Path(".gitignore")
    contents = gitignore.read_text(encoding="utf-8")
    assert gitignore.exists()
    assert "__pycache__/" in contents
    assert "pytest-cache-files-*" in contents
    assert ".pytest_cache/" in contents
