from __future__ import annotations

import pytest

from deployment.real_bridge.mapping import percent_to_raw, percentages_to_raw, raw_to_percent


@pytest.mark.parametrize(
    ("percent", "expected"),
    [
        (0.0, 0),
        (100.0, 255),
        (50.0, 128),  # 127.5 uses Python's round-to-even rule.
        (25.0, 64),
        (75.0, 191),
        (100.0 * 10.5 / 255.0, 10),
        (100.0 * 11.5 / 255.0, 12),
    ],
)
def test_deployed_python_rounding_rule(percent: float, expected: int) -> None:
    assert percent_to_raw(percent) == expected


def test_six_dimension_mapping_and_round_trip_bound() -> None:
    raw = percentages_to_raw([0, 20, 40, 60, 80, 100])
    assert raw == [0, 51, 102, 153, 204, 255]
    for value in raw:
        assert abs(percent_to_raw(raw_to_percent(value)) - value) == 0


@pytest.mark.parametrize("bad", [-0.001, 100.001, float("nan"), float("inf")])
def test_invalid_percent_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        percent_to_raw(bad)
