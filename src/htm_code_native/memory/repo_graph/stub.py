from __future__ import annotations

from typing import Protocol

import torch


class RepoGraphAdapter(Protocol):
    def reset(self) -> None:
        ...

    def query(self, hidden: torch.Tensor) -> torch.Tensor | None:
        ...


class NoOpRepoGraph:
    def reset(self) -> None:
        return None

    def query(self, hidden: torch.Tensor) -> torch.Tensor | None:
        return None
