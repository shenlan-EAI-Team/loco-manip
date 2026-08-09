from __future__ import annotations

import math
from typing import Iterable


def percent_to_raw(value: float) -> int:
    """Map training 0..100 to O6 0..255 using the deployed driver's rule."""
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(f"O6 percentage must be finite and in [0, 100], got {value!r}")
    raw = int(round(value * 255.0 / 100.0))
    if not 0 <= raw <= 255:
        raise AssertionError(f"mapped O6 integer outside [0, 255]: {raw}")
    return raw


def percentages_to_raw(values: Iterable[float]) -> list[int]:
    result = [percent_to_raw(value) for value in values]
    if len(result) != 6:
        raise ValueError(f"O6 command requires six values, got {len(result)}")
    return result


def raw_to_percent(value: int | float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 255.0:
        raise ValueError(f"O6 raw value must be finite and in [0, 255], got {value!r}")
    return value * 100.0 / 255.0
