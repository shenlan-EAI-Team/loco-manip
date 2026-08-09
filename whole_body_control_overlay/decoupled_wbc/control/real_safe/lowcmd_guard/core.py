"""Hardware-independent state and validation for the G1 LowCmd guard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any

import numpy as np

from ..standalone import BODY_DOF, RobotSnapshot, SafetyFault, StandaloneSafetyGate


class GuardState(str, Enum):
    READ_ONLY = "READ_ONLY"
    READY = "READY"
    TAKEOVER_PENDING = "TAKEOVER_PENDING"
    HOLD = "HOLD"
    RECOVERY_PENDING = "RECOVERY_PENDING"
    VERIFY_OWNER = "VERIFY_OWNER"
    FAULT_BLOCKED = "FAULT_BLOCKED"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class GuardSnapshot:
    robot: RobotSnapshot
    mode_machine: int
    motor_modes: np.ndarray
    motor_errors: np.ndarray
    motor_tau_est: np.ndarray


@dataclass(frozen=True)
class GuardCommand:
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    motor_mode: np.ndarray
    mode_pr: int
    mode_machine: int
    prepared_monotonic: float


@dataclass(frozen=True)
class GuardConfig:
    transport_frequency_hz: float
    policy_target_frequency_hz: float
    commissioning_wbc_preflight_samples: int
    measured_minimum_transport_frequency_hz: float | None
    required_initial_owner: str
    hold_duration_s: float
    post_recovery_monitor_s: float
    first_write_deadline_s: float
    release_verify_timeout_s: float
    recovery_verify_timeout_s: float
    pc_heartbeat_timeout_s: float
    command_prepare_max_age_s: float
    lower_body_mailbox_stale_s: float
    commissioning_stand_duration_s: float
    target_step_abs_limit: np.ndarray
    target_rate_abs_limit: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    motor_mode: np.ndarray
    mode_pr: int
    real_execution_enabled: bool
    recovery_handoff_verified: bool
    commissioning_execution_enabled: bool
    official_reference_transport_frequency_hz: float

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "GuardConfig":
        arrays = {}
        for name in (
            "target_step_abs_limit",
            "target_rate_abs_limit",
            "kp",
            "kd",
            "motor_mode",
        ):
            dtype = np.int64 if name == "motor_mode" else np.float64
            value = np.asarray(values[name], dtype=dtype)
            if value.shape != (BODY_DOF,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite shape ({BODY_DOF},)")
            arrays[name] = value
        for name in ("target_step_abs_limit", "target_rate_abs_limit"):
            if np.any(arrays[name] <= 0):
                raise ValueError(f"{name} must be strictly positive")
        for name in ("kp", "kd"):
            if np.any(arrays[name] < 0):
                raise ValueError(f"{name} must be nonnegative")
        if np.any(arrays["motor_mode"] != 1):
            raise ValueError("all 29 controlled motors must explicitly use mode=1")

        positive_scalars = {
            name: float(values[name])
            for name in (
                "transport_frequency_hz",
                "policy_target_frequency_hz",
                "hold_duration_s",
                "post_recovery_monitor_s",
                "first_write_deadline_s",
                "release_verify_timeout_s",
                "recovery_verify_timeout_s",
                "pc_heartbeat_timeout_s",
                "command_prepare_max_age_s",
                "lower_body_mailbox_stale_s",
                "commissioning_stand_duration_s",
                "official_reference_transport_frequency_hz",
            )
        }
        if not all(np.isfinite(value) and value > 0 for value in positive_scalars.values()):
            raise ValueError("guard timing and frequency values must be finite and positive")
        if positive_scalars["transport_frequency_hz"] < positive_scalars[
            "official_reference_transport_frequency_hz"
        ]:
            raise ValueError("initial transport frequency cannot be below the official reference")
        if positive_scalars["policy_target_frequency_hz"] >= positive_scalars[
            "transport_frequency_hz"
        ]:
            raise ValueError("policy target loop must be slower than LowCmd transport")

        measured = values.get("measured_minimum_transport_frequency_hz")
        measured_value = None if measured is None else float(measured)
        if measured_value is not None and (not np.isfinite(measured_value) or measured_value <= 0):
            raise ValueError("measured minimum transport frequency must be positive or null")
        required_owner = str(values["required_initial_owner"])
        if not required_owner:
            raise ValueError("required_initial_owner cannot be empty")
        preflight_samples = int(values["commissioning_wbc_preflight_samples"])
        if preflight_samples < 6:
            raise ValueError("commissioning WBC preflight must fill the six-frame history")

        return cls(
            **arrays,
            **positive_scalars,
            measured_minimum_transport_frequency_hz=measured_value,
            required_initial_owner=required_owner,
            commissioning_wbc_preflight_samples=preflight_samples,
            mode_pr=int(values["mode_pr"]),
            real_execution_enabled=bool(values["real_execution_enabled"]),
            recovery_handoff_verified=bool(values["recovery_handoff_verified"]),
            commissioning_execution_enabled=bool(
                values["commissioning_execution_enabled"]
            ),
        )


class PcTargetMailbox:
    """Thread-safe 50 Hz target/heartbeat inbox; it never owns a writer."""

    def __init__(self, config: GuardConfig, safety: StandaloneSafetyGate, session_id: str):
        if not session_id:
            raise ValueError("session_id is required")
        self.config = config
        self.safety = safety
        self.session_id = session_id
        self._lock = Lock()
        self._sequence = -1
        self._q: np.ndarray | None = None
        self._received_monotonic: float | None = None

    def accept(self, payload: dict[str, Any], received_monotonic: float) -> None:
        if payload.get("schema_version") != 1 or payload.get("kind") != "validated_target":
            raise ValueError("unsupported PC target schema or kind")
        if payload.get("session_id") != self.session_id:
            raise PermissionError("PC target session does not match the armed guard session")
        sequence = int(payload["sequence"])
        q = np.asarray(payload["q_rad"], dtype=np.float64)
        self.safety.validate_command(q)
        if not np.isfinite(received_monotonic):
            raise ValueError("target receive timestamp is not finite")

        with self._lock:
            if sequence <= self._sequence:
                raise ValueError("target sequence must increase strictly")
            if self._q is not None and self._received_monotonic is not None:
                dt = received_monotonic - self._received_monotonic
                if dt <= 0:
                    raise ValueError("target receive clock must increase strictly")
                delta = np.abs(q - self._q)
                allowed = np.minimum(
                    self.config.target_step_abs_limit,
                    self.config.target_rate_abs_limit * dt,
                )
                if np.any(delta > allowed):
                    indices = np.flatnonzero(delta > allowed)
                    raise SafetyFault(f"PC target step/rate violation at motors {indices.tolist()}")
            self._sequence = sequence
            self._q = q.copy()
            self._received_monotonic = received_monotonic

    def latest(self, now: float) -> np.ndarray:
        with self._lock:
            if self._q is None or self._received_monotonic is None:
                raise SafetyFault("no validated PC target/heartbeat received")
            age = now - self._received_monotonic
            if age < 0 or age > self.config.pc_heartbeat_timeout_s:
                raise SafetyFault(f"PC target/heartbeat stale: age={age:.6f}s")
            return self._q.copy()


class LowCmdGuardCore:
    """Pure lifecycle core. Release/select/write operations live in the runtime."""

    def __init__(
        self,
        config: GuardConfig,
        safety: StandaloneSafetyGate,
        *,
        one_time_token: str,
    ) -> None:
        if not one_time_token:
            raise ValueError("one-time lifecycle token is required")
        self.config = config
        self.safety = safety
        self.state = GuardState.READ_ONLY
        self._token = one_time_token
        self._token_consumed = False
        self.latest_snapshot: GuardSnapshot | None = None
        self.latest_owner: str | None = None
        self.prepared_command: GuardCommand | None = None
        self.original_owner: str | None = None
        self.release_return_monotonic: float | None = None
        self.first_write_monotonic: float | None = None
        self.fault_reason: str | None = None

    def _validate_guard_snapshot(self, snapshot: GuardSnapshot, now: float) -> None:
        self.safety.validate_snapshot(snapshot.robot, now)
        modes = np.asarray(snapshot.motor_modes)
        errors = np.asarray(snapshot.motor_errors)
        tau_est = np.asarray(snapshot.motor_tau_est, dtype=np.float64)
        if (
            modes.shape != (BODY_DOF,)
            or errors.shape != (BODY_DOF,)
            or tau_est.shape != (BODY_DOF,)
        ):
            raise SafetyFault("motor mode/error/tau_est state must be shape (29,)")
        if not np.isfinite(tau_est).all():
            raise SafetyFault("motor tau_est contains non-finite values")
        if np.any(errors != 0):
            indices = np.flatnonzero(errors != 0)
            raise SafetyFault(f"motor error state is nonzero at motors {indices.tolist()}")

    def observe(self, snapshot: GuardSnapshot, owner: str, now: float) -> None:
        self._validate_guard_snapshot(snapshot, now)
        if not owner:
            raise SafetyFault("motion owner is empty during READ_ONLY preparation")
        self.latest_snapshot = snapshot
        self.latest_owner = owner

    def prepare_current_q_hold(self, now: float) -> GuardCommand:
        if self.state != GuardState.READ_ONLY:
            raise RuntimeError(f"prepare_current_q_hold invalid in {self.state.value}")
        if self.latest_snapshot is None or self.latest_owner != self.config.required_initial_owner:
            raise SafetyFault(
                f"expected owner {self.config.required_initial_owner!r}, got {self.latest_owner!r}"
            )
        self.safety.validate_snapshot(self.latest_snapshot.robot, now)
        q = np.asarray(self.latest_snapshot.robot.q, dtype=np.float64).copy()
        command = GuardCommand(
            q=q,
            dq=np.zeros(BODY_DOF, dtype=np.float64),
            kp=self.config.kp.copy(),
            kd=self.config.kd.copy(),
            tau=np.zeros(BODY_DOF, dtype=np.float64),
            motor_mode=self.config.motor_mode.copy(),
            mode_pr=self.config.mode_pr,
            mode_machine=int(self.latest_snapshot.mode_machine),
            prepared_monotonic=now,
        )
        self.safety.validate_command(command.q)
        self.prepared_command = command
        self.original_owner = self.latest_owner
        self.state = GuardState.READY
        return command

    def refresh_current_q_hold_after_transport(
        self,
        snapshot: GuardSnapshot,
        owner: str,
        now: float,
    ) -> GuardCommand:
        """Refresh the immutable HOLD after silent writer discovery, before release."""
        if self.state != GuardState.TAKEOVER_PENDING:
            raise RuntimeError(
                f"refresh_current_q_hold_after_transport invalid in {self.state.value}"
            )
        self._validate_guard_snapshot(snapshot, now)
        if owner != self.config.required_initial_owner or owner != self.original_owner:
            raise SafetyFault(
                f"owner changed during silent transport preparation: {owner!r}"
            )
        self.latest_snapshot = snapshot
        self.latest_owner = owner
        command = GuardCommand(
            q=np.asarray(snapshot.robot.q, dtype=np.float64).copy(),
            dq=np.zeros(BODY_DOF, dtype=np.float64),
            kp=self.config.kp.copy(),
            kd=self.config.kd.copy(),
            tau=np.zeros(BODY_DOF, dtype=np.float64),
            motor_mode=self.config.motor_mode.copy(),
            mode_pr=self.config.mode_pr,
            mode_machine=int(snapshot.mode_machine),
            prepared_monotonic=now,
        )
        self.safety.validate_command(command.q)
        self.prepared_command = command
        return command

    def authorize_takeover(
        self,
        *,
        token: str,
        hardware_transport_enabled: bool,
        command_publication_enabled: bool,
        lifecycle_armed: bool,
        now: float,
        commissioning_mode: bool = False,
    ) -> None:
        if self.state != GuardState.READY or self.prepared_command is None:
            raise RuntimeError(f"authorize_takeover invalid in {self.state.value}")
        if not self.config.real_execution_enabled:
            raise PermissionError("real execution is disabled in guard configuration")
        if commissioning_mode and not self.config.commissioning_execution_enabled:
            raise PermissionError("supported commissioning execution is disabled")
        if not commissioning_mode and not self.config.recovery_handoff_verified:
            raise PermissionError("ai recovery handoff semantics are not verified")
        if not all((hardware_transport_enabled, command_publication_enabled, lifecycle_armed)):
            raise PermissionError("all three runtime enable gates must be true")
        if self._token_consumed or token != self._token:
            raise PermissionError("invalid or already-consumed lifecycle token")
        age = now - self.prepared_command.prepared_monotonic
        if age < 0 or age > self.config.command_prepare_max_age_s:
            raise SafetyFault(f"prepared current-q command is stale: age={age:.6f}s")
        self._token_consumed = True
        self._token = ""
        self.state = GuardState.TAKEOVER_PENDING

    def mark_release_returned(self, now: float) -> None:
        if self.state != GuardState.TAKEOVER_PENDING:
            raise RuntimeError(f"mark_release_returned invalid in {self.state.value}")
        self.release_return_monotonic = now

    def mark_first_write(self, now: float) -> None:
        if self.state != GuardState.TAKEOVER_PENDING or self.release_return_monotonic is None:
            raise RuntimeError(f"mark_first_write invalid in {self.state.value}")
        delay = now - self.release_return_monotonic
        if delay < 0 or delay > self.config.first_write_deadline_s:
            self.enter_fault(f"ReleaseMode-to-first-HOLD write missed: delay={delay:.6f}s")
            raise SafetyFault(self.fault_reason)
        self.first_write_monotonic = now

    def mark_hold_verified(self, owner: str) -> None:
        if self.state != GuardState.TAKEOVER_PENDING or self.first_write_monotonic is None:
            raise RuntimeError(f"mark_hold_verified invalid in {self.state.value}")
        if owner:
            raise SafetyFault(f"motion owner did not release: {owner!r}")
        self.state = GuardState.HOLD

    def validate_hold_feedback(self, snapshot: GuardSnapshot, now: float) -> None:
        if self.state not in {
            GuardState.TAKEOVER_PENDING,
            GuardState.HOLD,
            GuardState.RECOVERY_PENDING,
            GuardState.VERIFY_OWNER,
        }:
            raise RuntimeError(f"validate_hold_feedback invalid in {self.state.value}")
        if self.prepared_command is None:
            raise RuntimeError("current-q HOLD command is missing")
        self._validate_guard_snapshot(snapshot, now)
        if snapshot.mode_machine != self.prepared_command.mode_machine:
            raise SafetyFault(
                "mode_machine changed during lifecycle: "
                f"{self.prepared_command.mode_machine} -> {snapshot.mode_machine}"
            )
        drift = np.abs(np.asarray(snapshot.robot.q) - self.prepared_command.q)
        limits = self.safety.limits.hold_feedback_delta_abs_limit
        if np.any(drift > limits):
            indices = np.flatnonzero(drift > limits)
            raise SafetyFault(f"current-q HOLD drift at motors {indices.tolist()}")

    def begin_recovery(self, reason: str) -> None:
        if self.state not in {GuardState.TAKEOVER_PENDING, GuardState.HOLD, GuardState.FAULT_BLOCKED}:
            raise RuntimeError(f"begin_recovery invalid in {self.state.value}")
        self.fault_reason = reason
        self.state = GuardState.RECOVERY_PENDING

    def mark_owner_selection_requested(self) -> None:
        if self.state != GuardState.RECOVERY_PENDING:
            raise RuntimeError(f"mark_owner_selection_requested invalid in {self.state.value}")
        self.state = GuardState.VERIFY_OWNER

    def mark_recovered(self, owner: str) -> None:
        if self.state != GuardState.VERIFY_OWNER:
            raise RuntimeError(f"mark_recovered invalid in {self.state.value}")
        if owner != self.original_owner or owner != self.config.required_initial_owner:
            raise SafetyFault(f"owner recovery mismatch: expected {self.original_owner!r}, got {owner!r}")
        self.state = GuardState.STOPPED

    def enter_fault(self, reason: str) -> None:
        self.fault_reason = reason
        self.state = GuardState.FAULT_BLOCKED
