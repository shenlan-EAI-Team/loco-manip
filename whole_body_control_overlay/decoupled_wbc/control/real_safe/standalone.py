"""Hardware-independent core for a fail-closed standalone G1 standing sequence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


BODY_DOF = 29
LOWER_BODY = slice(0, 15)
ARMS = slice(15, 29)


class StandaloneState(str, Enum):
    READ_ONLY = "READ_ONLY"
    ARM_CONTROL = "ARM_CONTROL"
    HOLD = "HOLD"
    ENGAGE_WBC = "ENGAGE_WBC"
    STAND = "STAND"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


class SafetyFault(RuntimeError):
    pass


@dataclass(frozen=True)
class RobotSnapshot:
    q: np.ndarray
    dq: np.ndarray
    base_quat_wxyz: np.ndarray
    base_angular_velocity: np.ndarray
    secondary_quat_wxyz: np.ndarray
    secondary_angular_velocity: np.ndarray
    lowstate_monotonic: float
    imu_monotonic: float


@dataclass(frozen=True)
class StandaloneSafetyLimits:
    q_lower: np.ndarray
    q_upper: np.ndarray
    measured_dq_abs_limit: np.ndarray
    lower_target_rate_abs_limit: np.ndarray
    lower_target_step_abs_limit: np.ndarray
    lowstate_stale_s: float
    imu_stale_s: float
    max_control_interval_s: float
    hold_min_duration_s: float
    engage_duration_s: float
    hold_feedback_delta_abs_limit: np.ndarray

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "StandaloneSafetyLimits":
        arrays = {
            "q_lower": (BODY_DOF, values["q_lower"]),
            "q_upper": (BODY_DOF, values["q_upper"]),
            "measured_dq_abs_limit": (BODY_DOF, values["measured_dq_abs_limit"]),
            "lower_target_rate_abs_limit": (15, values["lower_target_rate_abs_limit"]),
            "lower_target_step_abs_limit": (15, values["lower_target_step_abs_limit"]),
            "hold_feedback_delta_abs_limit": (BODY_DOF, values["hold_feedback_delta_abs_limit"]),
        }
        parsed: dict[str, np.ndarray] = {}
        for name, (dimension, raw) in arrays.items():
            array = np.asarray(raw, dtype=np.float64)
            if array.shape != (dimension,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite shape ({dimension},), got {array.shape}")
            parsed[name] = array
        if np.any(parsed["q_lower"] >= parsed["q_upper"]):
            raise ValueError("q_lower must be strictly less than q_upper")
        for name in (
            "measured_dq_abs_limit",
            "lower_target_rate_abs_limit",
            "lower_target_step_abs_limit",
            "hold_feedback_delta_abs_limit",
        ):
            if np.any(parsed[name] <= 0):
                raise ValueError(f"{name} must be positive")

        scalars = {
            name: float(values[name])
            for name in (
                "lowstate_stale_s",
                "imu_stale_s",
                "max_control_interval_s",
                "hold_min_duration_s",
                "engage_duration_s",
            )
        }
        if not all(np.isfinite(value) and value > 0 for value in scalars.values()):
            raise ValueError("all timing limits must be finite and positive")
        return cls(**parsed, **scalars)


class StandaloneSafetyGate:
    def __init__(self, limits: StandaloneSafetyLimits):
        self.limits = limits

    def validate_snapshot(self, snapshot: RobotSnapshot, now: float) -> None:
        timestamps = np.asarray(
            [now, snapshot.lowstate_monotonic, snapshot.imu_monotonic],
            dtype=np.float64,
        )
        if not np.isfinite(timestamps).all():
            raise SafetyFault("non-finite control or state timestamp")
        q = np.asarray(snapshot.q, dtype=np.float64)
        dq = np.asarray(snapshot.dq, dtype=np.float64)
        quat = np.asarray(snapshot.base_quat_wxyz, dtype=np.float64)
        omega = np.asarray(snapshot.base_angular_velocity, dtype=np.float64)
        secondary_quat = np.asarray(snapshot.secondary_quat_wxyz, dtype=np.float64)
        secondary_omega = np.asarray(snapshot.secondary_angular_velocity, dtype=np.float64)
        if q.shape != (BODY_DOF,) or dq.shape != (BODY_DOF,):
            raise SafetyFault(f"body state must be shape ({BODY_DOF},)")
        if (
            quat.shape != (4,)
            or omega.shape != (3,)
            or secondary_quat.shape != (4,)
            or secondary_omega.shape != (3,)
        ):
            raise SafetyFault("IMU quaternion/angular velocity shape is invalid")
        if not all(
            np.isfinite(value).all()
            for value in (q, dq, quat, omega, secondary_quat, secondary_omega)
        ):
            raise SafetyFault("non-finite lowstate or IMU value")
        quat_norm = float(np.linalg.norm(quat))
        if not 0.95 <= quat_norm <= 1.05:
            raise SafetyFault(f"IMU quaternion norm is invalid: {quat_norm:.6f}")
        secondary_quat_norm = float(np.linalg.norm(secondary_quat))
        if not 0.95 <= secondary_quat_norm <= 1.05:
            raise SafetyFault(
                f"secondary IMU quaternion norm is invalid: {secondary_quat_norm:.6f}"
            )
        if np.any(q < self.limits.q_lower) or np.any(q > self.limits.q_upper):
            indices = np.flatnonzero((q < self.limits.q_lower) | (q > self.limits.q_upper))
            raise SafetyFault(f"measured q outside hard limits at motors {indices.tolist()}")
        if np.any(np.abs(dq) > self.limits.measured_dq_abs_limit):
            indices = np.flatnonzero(np.abs(dq) > self.limits.measured_dq_abs_limit)
            raise SafetyFault(f"measured dq outside limits at motors {indices.tolist()}")
        if now - float(snapshot.lowstate_monotonic) > self.limits.lowstate_stale_s:
            raise SafetyFault("rt/lowstate is stale")
        if now - float(snapshot.imu_monotonic) > self.limits.imu_stale_s:
            raise SafetyFault("secondary IMU is stale")
        if snapshot.lowstate_monotonic > now or snapshot.imu_monotonic > now:
            raise SafetyFault("state timestamp is in the future")

    def validate_command(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (BODY_DOF,) or not np.isfinite(command).all():
            raise SafetyFault(f"whole-body command must be finite shape ({BODY_DOF},)")
        if np.any(command < self.limits.q_lower) or np.any(command > self.limits.q_upper):
            indices = np.flatnonzero(
                (command < self.limits.q_lower) | (command > self.limits.q_upper)
            )
            raise SafetyFault(f"target q outside hard limits at motors {indices.tolist()}")


class StandaloneRealSafeCore:
    """Pure state machine; it never creates DDS or MotionSwitcher objects."""

    def __init__(
        self,
        limits: StandaloneSafetyLimits,
        *,
        one_time_arm_token: str,
    ) -> None:
        if not one_time_arm_token:
            raise ValueError("a non-empty one-time arm token is required")
        self.limits = limits
        self.safety = StandaloneSafetyGate(limits)
        self.state = StandaloneState.READ_ONLY
        self._arm_token = one_time_arm_token
        self._token_consumed = False
        self.arming_q: np.ndarray | None = None
        self.last_command: np.ndarray | None = None
        self.last_heartbeat: float | None = None
        self.hold_started_at: float | None = None
        self.engage_started_at: float | None = None
        self.fault_reason: str | None = None
        self.lower_slew_limited_count = 0

    @property
    def command_allowed(self) -> bool:
        return self.state in {
            StandaloneState.HOLD,
            StandaloneState.ENGAGE_WBC,
            StandaloneState.STAND,
        }

    def _enter_fault(self, reason: str) -> None:
        self.state = StandaloneState.FAULT
        self.fault_reason = reason

    def _guard(self, operation):
        try:
            return operation()
        except SafetyFault as exc:
            self._enter_fault(str(exc))
            raise

    def read_only_tick(self, snapshot: RobotSnapshot, now: float) -> None:
        if self.state != StandaloneState.READ_ONLY:
            raise RuntimeError(f"read_only_tick invalid in {self.state.value}")
        self.safety.validate_snapshot(snapshot, now)
        self.last_heartbeat = now

    def request_arm(self, token: str, snapshot: RobotSnapshot, now: float) -> np.ndarray:
        if self.state != StandaloneState.READ_ONLY:
            raise RuntimeError(f"request_arm invalid in {self.state.value}")
        if self._token_consumed or token != self._arm_token:
            raise PermissionError("invalid or already-consumed arm token")
        self.safety.validate_snapshot(snapshot, now)
        self._token_consumed = True
        self._arm_token = ""
        self.arming_q = np.asarray(snapshot.q, dtype=np.float64).copy()
        self.last_command = self.arming_q.copy()
        self.last_heartbeat = now
        self.state = StandaloneState.ARM_CONTROL
        return self.last_command.copy()

    def mark_takeover_complete(self, now: float) -> None:
        if self.state != StandaloneState.ARM_CONTROL:
            raise RuntimeError(f"mark_takeover_complete invalid in {self.state.value}")
        self.hold_started_at = now
        self.last_heartbeat = now
        self.state = StandaloneState.HOLD

    def _active_tick(self, snapshot: RobotSnapshot, now: float) -> float:
        def validate() -> None:
            self.safety.validate_snapshot(snapshot, now)
            if self.last_heartbeat is not None:
                interval = now - self.last_heartbeat
                if interval <= 0.0:
                    raise SafetyFault(
                        f"control clock is not strictly monotonic: interval={interval:.6f}s"
                    )
                if interval > self.limits.max_control_interval_s:
                    raise SafetyFault(f"50Hz control watchdog missed: interval={interval:.6f}s")

        self._guard(validate)
        dt = 0.02 if self.last_heartbeat is None else max(now - self.last_heartbeat, 1e-6)
        self.last_heartbeat = now
        return dt

    def hold_command(self, snapshot: RobotSnapshot, now: float) -> np.ndarray:
        if self.state != StandaloneState.HOLD:
            raise RuntimeError(f"hold_command invalid in {self.state.value}")
        self._active_tick(snapshot, now)
        assert self.arming_q is not None and self.last_command is not None
        feedback_delta = np.abs(np.asarray(snapshot.q) - self.arming_q)
        if np.any(feedback_delta > self.limits.hold_feedback_delta_abs_limit):
            indices = np.flatnonzero(feedback_delta > self.limits.hold_feedback_delta_abs_limit)
            reason = f"current-q HOLD feedback drift at motors {indices.tolist()}"
            self._enter_fault(reason)
            raise SafetyFault(reason)
        self.safety.validate_command(self.last_command)
        return self.last_command.copy()

    def begin_wbc_engage(self, now: float) -> None:
        if self.state != StandaloneState.HOLD or self.hold_started_at is None:
            raise RuntimeError(f"begin_wbc_engage invalid in {self.state.value}")
        if now - self.hold_started_at < self.limits.hold_min_duration_s:
            raise RuntimeError("current-q HOLD has not met its minimum stable duration")
        if self.last_heartbeat is None or now - self.last_heartbeat > self.limits.max_control_interval_s:
            reason = "current-q HOLD was not continuously serviced through WBC engage"
            self._enter_fault(reason)
            raise SafetyFault(reason)
        if now < self.last_heartbeat:
            reason = "control clock moved backwards before WBC engage"
            self._enter_fault(reason)
            raise SafetyFault(reason)
        self.engage_started_at = now
        self.state = StandaloneState.ENGAGE_WBC

    def wbc_command(
        self,
        snapshot: RobotSnapshot,
        lower_target: np.ndarray,
        now: float,
    ) -> np.ndarray:
        if self.state not in {StandaloneState.ENGAGE_WBC, StandaloneState.STAND}:
            raise RuntimeError(f"wbc_command invalid in {self.state.value}")
        dt = self._active_tick(snapshot, now)
        assert self.arming_q is not None and self.last_command is not None
        target = np.asarray(lower_target, dtype=np.float64)
        if target.shape != (15,) or not np.isfinite(target).all():
            reason = "Gear WBC lower target must be finite shape (15,)"
            self._enter_fault(reason)
            raise SafetyFault(reason)
        if np.any(target < self.limits.q_lower[LOWER_BODY]) or np.any(
            target > self.limits.q_upper[LOWER_BODY]
        ):
            reason = "Gear WBC lower target is outside hard joint limits"
            self._enter_fault(reason)
            raise SafetyFault(reason)

        desired_lower = target
        if self.state == StandaloneState.ENGAGE_WBC:
            assert self.engage_started_at is not None
            phase = np.clip(
                (now - self.engage_started_at) / self.limits.engage_duration_s,
                0.0,
                1.0,
            )
            alpha = phase * phase * (3.0 - 2.0 * phase)
            desired_lower = self.arming_q[LOWER_BODY] + alpha * (
                target - self.arming_q[LOWER_BODY]
            )
            if phase >= 1.0:
                self.state = StandaloneState.STAND

        previous_lower = self.last_command[LOWER_BODY]
        requested_delta = desired_lower - previous_lower
        allowed_delta = np.minimum(
            self.limits.lower_target_step_abs_limit,
            self.limits.lower_target_rate_abs_limit * dt,
        )
        safe_delta = np.clip(requested_delta, -allowed_delta, allowed_delta)
        self.lower_slew_limited_count += int(np.count_nonzero(~np.isclose(safe_delta, requested_delta)))

        command = self.last_command.copy()
        command[LOWER_BODY] = previous_lower + safe_delta
        command[ARMS] = self.arming_q[ARMS]
        self.safety.validate_command(command)
        self.last_command = command
        return command.copy()

    def watchdog_expired(self, now: float) -> bool:
        if not self.command_allowed or self.last_heartbeat is None:
            return False
        if now - self.last_heartbeat <= self.limits.max_control_interval_s:
            return False
        self._enter_fault(
            f"independent 50Hz watchdog expired after {now - self.last_heartbeat:.6f}s"
        )
        return True

    def mark_exit_complete(self, *, verified_platform_exit: bool) -> None:
        if self.state not in {StandaloneState.FAULT, StandaloneState.HOLD, StandaloneState.STAND}:
            raise RuntimeError(f"mark_exit_complete invalid in {self.state.value}")
        if not verified_platform_exit:
            raise PermissionError("platform-specific supported/damped exit must be verified")
        self.state = StandaloneState.STOPPED
