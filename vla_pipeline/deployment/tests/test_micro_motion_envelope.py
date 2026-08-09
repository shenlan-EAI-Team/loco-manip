from __future__ import annotations

import numpy as np
import pytest

from deployment.real_bridge.envelope import MicroMotionEnvelope


def state() -> dict[str, np.ndarray]:
    return {
        "left_arm": np.zeros(7),
        "right_arm": np.ones(7),
        "left_o6": np.full(6, 50.0),
        "right_o6": np.full(6, 99.0),
    }


def test_arm_envelope_limits_acceleration_velocity_and_excursion() -> None:
    envelope = MicroMotionEnvelope()
    envelope.reset(state())
    target = {
        "left_arm": np.full(7, 10.0),
        "right_arm": np.full(7, -10.0),
        "left_o6": np.full(6, 100.0),
        "right_o6": np.zeros(6),
    }
    first = envelope.step(target, dt=0.1)
    np.testing.assert_allclose(first["left_arm"], 0.004, atol=1e-12)
    np.testing.assert_allclose(first["right_arm"], 0.996, atol=1e-12)
    np.testing.assert_allclose(first["left_o6"], 51.5, atol=1e-12)
    np.testing.assert_allclose(first["right_o6"], 97.5, atol=1e-12)
    for _ in range(100):
        last = envelope.step(target, dt=0.1)
    assert np.max(np.abs(last["left_arm"] - state()["left_arm"])) <= 0.01 + 1e-12
    assert np.max(np.abs(last["right_arm"] - state()["right_arm"])) <= 0.01 + 1e-12
    assert np.max(np.abs(last["left_o6"] - state()["left_o6"])) <= 5.0 + 1e-12
    assert np.max(np.abs(last["right_o6"] - state()["right_o6"])) <= 5.0 + 1e-12


def test_feedback_outside_arming_envelope_faults() -> None:
    envelope = MicroMotionEnvelope()
    anchor = state()
    envelope.reset(anchor)
    bad = {key: value.copy() for key, value in anchor.items()}
    bad["right_o6"][2] -= 5.01
    with pytest.raises(RuntimeError, match="right_o6"):
        envelope.assert_feedback_within_envelope(bad)


def test_nonfinite_and_wrong_shape_are_rejected() -> None:
    envelope = MicroMotionEnvelope()
    envelope.reset(state())
    target = state()
    target["left_arm"] = np.full(7, np.nan)
    with pytest.raises(ValueError, match="NaN or Inf"):
        envelope.step(target, dt=1 / 30)
