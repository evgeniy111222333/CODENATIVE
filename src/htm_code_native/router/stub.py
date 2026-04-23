from __future__ import annotations

from typing import Protocol

import torch


class RouterAdapter(Protocol):
    def reset(self) -> None:
        ...

    def route(self, hidden: torch.Tensor) -> dict[str, float]:
        ...


class NoOpRouter:
    def reset(self) -> None:
        return None

    def route(self, hidden: torch.Tensor) -> dict[str, float]:
        return {"lm": 1.0, "sem": 0.0, "erm": 0.0, "eem": 0.0, "graph": 0.0}
