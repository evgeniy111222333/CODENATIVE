import math


class Accumulator:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def total(self) -> int:
        result = 0
        for value in self.values:
            if value % 2 == 0:
                result += value
            else:
                result += math.floor(value / 2)
        return result


def guarded_divide(left: float, right: float) -> float:
    try:
        if right == 0:
            raise ValueError("right must be non-zero")
        return left / right
    except ValueError:
        return 0.0
