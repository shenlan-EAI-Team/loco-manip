"""Stateful physical-unit safety filtering for dry-run action targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FilterCounters:
    nonfinite: int = 0
    position_limit: int = 0
    velocity_limit: int = 0
    acceleration_limit: int = 0
    o6_delta_limit: int = 0
    spike_rejected: int = 0
    spike_hysteresis_pending: int = 0
    feedback_only_substitution: int = 0

    def as_dict(self) -> dict[str, int]:
        return vars(self).copy()


@dataclass
class SafetyFilter:
    config: dict[str, Any]
    previous: dict[str, np.ndarray] = field(default_factory=dict)
    previous_velocity: dict[str, np.ndarray] = field(default_factory=dict)
    counters: dict[str, FilterCounters] = field(default_factory=dict)
    spike_candidate: np.ndarray | None = None
    spike_candidate_count: np.ndarray | None = None

    def reset(self, state: dict[str, np.ndarray]) -> None:
        self.previous = {
            key: np.asarray(state[key], dtype=np.float64).reshape(-1).copy()
            for key in ("left_arm", "right_arm", "left_o6", "right_o6")
        }
        self.previous_velocity = {key: np.zeros_like(value) for key, value in self.previous.items()}
        self.counters = {key: FilterCounters() for key in self.previous}
        self.spike_candidate = self.previous["left_o6"].copy()
        self.spike_candidate_count = np.zeros_like(self.previous["left_o6"], dtype=np.int64)

    def _guard_left_o6(self, target: np.ndarray) -> np.ndarray:
        limits = self.config["left_o6"]
        threshold = float(limits.get("spike_rejection_threshold_points", 30.0))
        confirmations = int(limits.get("spike_confirmation_steps", 4))
        hysteresis = float(limits.get("spike_hysteresis_points", 5.0))
        if confirmations < 1:
            raise ValueError("left_o6.spike_confirmation_steps must be >= 1")
        assert self.spike_candidate is not None
        assert self.spike_candidate_count is not None
        previous = self.previous["left_o6"]
        guarded = target.copy()
        large = np.abs(target - previous) > threshold
        close_to_candidate = np.abs(target - self.spike_candidate) <= hysteresis
        continuing = large & close_to_candidate
        new_candidate = large & ~close_to_candidate
        self.spike_candidate[new_candidate] = target[new_candidate]
        self.spike_candidate_count[new_candidate] = 1
        self.spike_candidate_count[continuing] += 1
        confirmed = large & (self.spike_candidate_count >= confirmations)
        rejected = large & ~confirmed
        guarded[rejected] = previous[rejected]
        count = self.counters["left_o6"]
        count.spike_rejected += int(np.count_nonzero(rejected))
        count.spike_hysteresis_pending += int(np.count_nonzero(rejected))
        cleared = ~large | confirmed
        self.spike_candidate_count[cleared] = 0
        self.spike_candidate[cleared] = guarded[cleared]
        return guarded

    def filter_step(
        self,
        targets: dict[str, np.ndarray],
        *,
        dt: float,
    ) -> dict[str, np.ndarray]:
        if not self.previous:
            raise RuntimeError("SafetyFilter.reset(state) must be called first")
        safe: dict[str, np.ndarray] = {}
        for key, target_value in targets.items():
            target = np.asarray(target_value, dtype=np.float64).reshape(-1).copy()
            previous = self.previous[key]
            previous_velocity = self.previous_velocity[key]
            limits = self.config[key]
            count = self.counters[key]

            finite = np.isfinite(target)
            count.nonfinite += int(np.count_nonzero(~finite))
            target[~finite] = previous[~finite]

            if key == "left_o6":
                target = self._guard_left_o6(target)
            elif key == "right_o6" and self.config.get("right_o6_feedback_only", False):
                count.feedback_only_substitution += int(
                    np.count_nonzero(~np.isclose(target, previous))
                )
                target = previous.copy()

            lower = np.asarray(limits["lower"], dtype=np.float64)
            upper = np.asarray(limits["upper"], dtype=np.float64)
            clipped = np.clip(target, lower, upper)
            count.position_limit += int(np.count_nonzero(~np.isclose(clipped, target)))
            target = clipped

            delta = target - previous
            if key.endswith("o6"):
                max_delta = float(limits["max_delta_per_30hz_step"]) * dt * float(
                    self.config["control_timeline_hz"]
                )
                clipped_delta = np.clip(delta, -max_delta, max_delta)
                count.o6_delta_limit += int(np.count_nonzero(~np.isclose(clipped_delta, delta)))
                delta = clipped_delta

            requested_velocity = delta / dt
            max_velocity = np.asarray(limits["max_velocity"], dtype=np.float64)
            if key == "left_o6" and "final_max_rate_points_s" in limits:
                max_velocity = np.minimum(
                    max_velocity,
                    float(limits["final_max_rate_points_s"]),
                )
            velocity = np.clip(requested_velocity, -max_velocity, max_velocity)
            count.velocity_limit += int(
                np.count_nonzero(~np.isclose(velocity, requested_velocity))
            )

            max_accel = np.asarray(limits["max_acceleration"], dtype=np.float64)
            # Discrete braking envelope: begin slowing before a position bound,
            # instead of clipping at the bound and creating an acceleration spike.
            distance_upper = np.maximum(upper - previous, 0.0)
            distance_lower = np.maximum(previous - lower, 0.0)
            upper_brake_velocity = -max_accel * dt + np.sqrt(
                (max_accel * dt) ** 2 + 2.0 * max_accel * distance_upper
            )
            lower_brake_velocity = -max_accel * dt + np.sqrt(
                (max_accel * dt) ** 2 + 2.0 * max_accel * distance_lower
            )
            brake_limited = np.clip(velocity, -lower_brake_velocity, upper_brake_velocity)
            count.velocity_limit += int(
                np.count_nonzero(~np.isclose(brake_limited, velocity))
            )
            velocity = brake_limited

            requested_accel = (velocity - previous_velocity) / dt
            accel = np.clip(requested_accel, -max_accel, max_accel)
            count.acceleration_limit += int(np.count_nonzero(~np.isclose(accel, requested_accel)))
            velocity = previous_velocity + accel * dt
            filtered = np.clip(previous + velocity * dt, lower, upper)
            actual_velocity = (filtered - previous) / dt

            safe[key] = filtered.astype(np.float32)
            self.previous[key] = filtered
            self.previous_velocity[key] = actual_velocity
        return safe
