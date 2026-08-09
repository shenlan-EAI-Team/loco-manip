from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import ACTION_DIMS, validate_groups


@dataclass
class EnvelopeCounters:
    excursion: dict[str, int] = field(default_factory=lambda: {key: 0 for key in ACTION_DIMS})
    velocity: dict[str, int] = field(default_factory=lambda: {key: 0 for key in ACTION_DIMS})
    acceleration: dict[str, int] = field(default_factory=lambda: {key: 0 for key in ACTION_DIMS})


class MicroMotionEnvelope:
    """Final stateful limiter immediately in front of real transports."""

    def __init__(
        self,
        *,
        arm_excursion_rad: float = 0.01,
        arm_velocity_rad_s: float = 0.12,
        arm_acceleration_rad_s2: float = 0.4,
        o6_excursion_points: float = 5.0,
        o6_velocity_points_s: float = 15.0,
    ) -> None:
        self.arm_excursion = float(arm_excursion_rad)
        self.arm_velocity = float(arm_velocity_rad_s)
        self.arm_acceleration = float(arm_acceleration_rad_s2)
        self.o6_excursion = float(o6_excursion_points)
        self.o6_velocity = float(o6_velocity_points_s)
        self.anchor: dict[str, np.ndarray] = {}
        self.previous: dict[str, np.ndarray] = {}
        self.previous_velocity: dict[str, np.ndarray] = {}
        self.counters = EnvelopeCounters()

    def reset(self, feedback: dict[str, Any]) -> None:
        current = validate_groups(feedback, label="arming_feedback")
        self.anchor = {key: value.copy() for key, value in current.items()}
        self.previous = {key: value.copy() for key, value in current.items()}
        self.previous_velocity = {key: np.zeros_like(value) for key, value in current.items()}
        self.counters = EnvelopeCounters()

    def step(self, targets: dict[str, Any], *, dt: float) -> dict[str, np.ndarray]:
        if not self.anchor:
            raise RuntimeError("MicroMotionEnvelope.reset(feedback) must be called first")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and > 0")
        requested = validate_groups(targets, label="micro_target")
        output: dict[str, np.ndarray] = {}
        for key, target in requested.items():
            is_arm = key.endswith("arm")
            excursion = self.arm_excursion if is_arm else self.o6_excursion
            max_velocity = self.arm_velocity if is_arm else self.o6_velocity
            lower = self.anchor[key] - excursion
            upper = self.anchor[key] + excursion
            if not is_arm:
                lower = np.maximum(lower, 0.0)
                upper = np.minimum(upper, 100.0)
            bounded = np.clip(target, lower, upper)
            self.counters.excursion[key] += int(np.count_nonzero(~np.isclose(target, bounded)))

            requested_velocity = (bounded - self.previous[key]) / dt
            velocity = np.clip(requested_velocity, -max_velocity, max_velocity)
            self.counters.velocity[key] += int(
                np.count_nonzero(~np.isclose(requested_velocity, velocity))
            )
            if is_arm:
                requested_acceleration = (velocity - self.previous_velocity[key]) / dt
                acceleration = np.clip(
                    requested_acceleration,
                    -self.arm_acceleration,
                    self.arm_acceleration,
                )
                self.counters.acceleration[key] += int(
                    np.count_nonzero(~np.isclose(requested_acceleration, acceleration))
                )
                velocity = self.previous_velocity[key] + acceleration * dt

            value = np.clip(self.previous[key] + velocity * dt, lower, upper)
            actual_velocity = (value - self.previous[key]) / dt
            output[key] = value.astype(np.float64)
            self.previous[key] = value
            self.previous_velocity[key] = actual_velocity
        return output

    def assert_feedback_within_envelope(self, feedback: dict[str, Any], *, tolerance: float = 1e-6) -> None:
        values = validate_groups(feedback, label="runtime_feedback")
        for key, value in values.items():
            limit = self.arm_excursion if key.endswith("arm") else self.o6_excursion
            if np.any(np.abs(value - self.anchor[key]) > limit + tolerance):
                raise RuntimeError(f"{key} feedback exceeded arming envelope")
