from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from deployment.safety_filter import SafetyFilter


def _state() -> dict[str, np.ndarray]:
    return {
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
        "left_o6": np.full(6, 100.0),
        "right_o6": np.full(6, 100.0),
    }


def test_isolated_left_o6_100_to_0_spike_is_rejected() -> None:
    config = yaml.safe_load(Path("deployment/config/adapter.yaml").read_text())
    guard = SafetyFilter(config)
    guard.reset(_state())
    target = _state()
    target["left_o6"] = np.zeros(6)
    target["right_o6"] = np.zeros(6)
    filtered = guard.filter_step(target, dt=1.0 / 30.0)
    np.testing.assert_allclose(filtered["left_o6"], 100.0)
    np.testing.assert_allclose(filtered["right_o6"], 100.0)
    assert guard.counters["left_o6"].spike_rejected == 6
    assert guard.counters["right_o6"].feedback_only_substitution == 6


def test_confirmed_left_o6_step_is_still_limited_to_15_points_per_second() -> None:
    config = yaml.safe_load(Path("deployment/config/adapter.yaml").read_text())
    guard = SafetyFilter(config)
    guard.reset(_state())
    target = _state()
    target["left_o6"] = np.zeros(6)
    outputs = [guard.filter_step(target, dt=1.0 / 30.0)["left_o6"] for _ in range(4)]
    np.testing.assert_allclose(outputs[:3], 100.0)
    assert np.all(outputs[3] >= 99.5 - 1e-6)
    assert np.all(outputs[3] < 100.0)

