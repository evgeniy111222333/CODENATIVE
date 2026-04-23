from __future__ import annotations

import math


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_delta(current: float, baseline: float) -> float:
    return float(current - baseline)


def has_invalid_number(values: dict[str, float]) -> bool:
    return any(math.isnan(value) or math.isinf(value) for value in values.values())
