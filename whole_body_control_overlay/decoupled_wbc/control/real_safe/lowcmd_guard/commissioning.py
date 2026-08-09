"""Thread-safe lower-body target composition for supported G1 commissioning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import Lock
from typing import Callable

import numpy as np

from .core import GuardCommand
from ..standalone import LOWER_BODY, ARMS, SafetyFault, StandaloneSafetyGate


@dataclass(frozen=True)
class LowerBodyTarget:
    q: np.ndarray
    timestamp: float
    sequence: int


class LowerBodyMailbox:
    """Single-producer mailbox carrying validated 15D Gear WBC targets."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sample: LowerBodyTarget | None = None

    def publish(self, target: np.ndarray, *, timestamp: float, sequence: int) -> None:
        q = np.asarray(target, dtype=np.float64)
        if q.shape != (15,) or not np.isfinite(q).all():
            raise SafetyFault("lower-body mailbox target must be finite shape (15,)")
        if not np.isfinite(timestamp):
            raise SafetyFault("lower-body mailbox timestamp must be finite")
        seq = int(sequence)
        with self._lock:
            if self._sample is not None:
                if seq <= self._sample.sequence:
                    raise SafetyFault("lower-body mailbox sequence must increase strictly")
                if timestamp <= self._sample.timestamp:
                    raise SafetyFault("lower-body mailbox timestamp must increase strictly")
            self._sample = LowerBodyTarget(q.copy(), float(timestamp), seq)

    def latest(self, *, now: float, max_age_s: float) -> LowerBodyTarget:
        if not np.isfinite(now) or not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("mailbox time and max age must be finite and positive")
        with self._lock:
            sample = self._sample
        if sample is None:
            raise SafetyFault("lower-body mailbox has no target")
        age = now - sample.timestamp
        if age < 0 or age > max_age_s:
            raise SafetyFault(f"lower-body mailbox target stale: age={age:.6f}s")
        return LowerBodyTarget(sample.q.copy(), sample.timestamp, sample.sequence)


class CommissioningPhase(str, Enum):
    HOLD = "HOLD"
    ENGAGE = "ENGAGE"
    STAND = "STAND"
    FROZEN = "FROZEN"


class WbcGuardCommandComposer:
    """Composes 15D WBC targets into the guard's sole 29DoF LowCmd stream."""

    def __init__(
        self,
        mailbox: LowerBodyMailbox,
        safety: StandaloneSafetyGate,
        *,
        mailbox_stale_s: float,
        engage_duration_s: float,
        lower_rate_limit: np.ndarray,
        lower_step_limit: np.ndarray,
    ) -> None:
        self.mailbox = mailbox
        self.safety = safety
        self.mailbox_stale_s = float(mailbox_stale_s)
        self.engage_duration_s = float(engage_duration_s)
        self.lower_rate_limit = np.asarray(lower_rate_limit, dtype=np.float64)
        self.lower_step_limit = np.asarray(lower_step_limit, dtype=np.float64)
        if (
            not np.isfinite(self.mailbox_stale_s)
            or self.mailbox_stale_s <= 0
            or not np.isfinite(self.engage_duration_s)
            or self.engage_duration_s <= 0
        ):
            raise ValueError("composer timing must be finite and positive")
        for name, value in (
            ("lower_rate_limit", self.lower_rate_limit),
            ("lower_step_limit", self.lower_step_limit),
        ):
            if value.shape != (15,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(f"{name} must be positive finite shape (15,)")
        self._lock = Lock()
        self._template: GuardCommand | None = None
        self._arming_q: np.ndarray | None = None
        self._last_command: GuardCommand | None = None
        self._last_compose_time: float | None = None
        self._engage_started: float | None = None
        self.phase = CommissioningPhase.HOLD
        self.fault_reason: str | None = None

    def arm_current_q(self, command: GuardCommand, *, now: float) -> None:
        self.safety.validate_command(command.q)
        if command.q.shape != (29,) or not np.isfinite(now):
            raise SafetyFault("commissioning arming command/time is invalid")
        with self._lock:
            self._template = command
            self._arming_q = command.q.copy()
            self._last_command = command
            self._last_compose_time = float(now)
            self._engage_started = None
            self.phase = CommissioningPhase.HOLD
            self.fault_reason = None

    def begin_engage(self, *, now: float) -> None:
        self.mailbox.latest(now=now, max_age_s=self.mailbox_stale_s)
        with self._lock:
            if self._template is None or self._arming_q is None:
                raise RuntimeError("composer is not armed")
            if self.phase != CommissioningPhase.HOLD:
                raise RuntimeError(f"begin_engage invalid in {self.phase.value}")
            self._engage_started = float(now)
            self._last_compose_time = float(now)
            self.phase = CommissioningPhase.ENGAGE

    def freeze(self, reason: str, *, fault: bool = True) -> None:
        with self._lock:
            self.phase = CommissioningPhase.FROZEN
            if fault:
                self.fault_reason = reason

    def command(self, *, now: float) -> GuardCommand:
        with self._lock:
            template = self._template
            arming_q = None if self._arming_q is None else self._arming_q.copy()
            previous = self._last_command
            previous_time = self._last_compose_time
            phase = self.phase
            engage_started = self._engage_started
        if template is None or arming_q is None or previous is None or previous_time is None:
            raise RuntimeError("composer is not armed")
        if phase in {CommissioningPhase.HOLD, CommissioningPhase.FROZEN}:
            return previous

        sample = self.mailbox.latest(now=now, max_age_s=self.mailbox_stale_s)
        target = sample.q
        lower_limits = self.safety.limits
        if np.any(target < lower_limits.q_lower[LOWER_BODY]) or np.any(
            target > lower_limits.q_upper[LOWER_BODY]
        ):
            raise SafetyFault("Gear WBC lower target is outside hard limits")
        dt = now - previous_time
        if not np.isfinite(dt) or dt <= 0:
            raise SafetyFault("500Hz command composer clock must increase strictly")

        desired = target
        next_phase = phase
        if phase == CommissioningPhase.ENGAGE:
            assert engage_started is not None
            progress = float(np.clip((now - engage_started) / self.engage_duration_s, 0.0, 1.0))
            alpha = progress * progress * (3.0 - 2.0 * progress)
            desired = arming_q[LOWER_BODY] + alpha * (target - arming_q[LOWER_BODY])
            if progress >= 1.0:
                next_phase = CommissioningPhase.STAND

        requested = desired - previous.q[LOWER_BODY]
        allowed = np.minimum(self.lower_step_limit, self.lower_rate_limit * dt)
        q = previous.q.copy()
        q[LOWER_BODY] = previous.q[LOWER_BODY] + np.clip(requested, -allowed, allowed)
        q[ARMS] = arming_q[ARMS]
        self.safety.validate_command(q)
        result = replace(template, q=q, prepared_monotonic=float(now))
        with self._lock:
            self._last_command = result
            self._last_compose_time = float(now)
            self.phase = next_phase
        return result

    def command_or_freeze(
        self,
        *,
        now: float,
        on_fault: Callable[[str], None],
    ) -> GuardCommand:
        try:
            return self.command(now=now)
        except Exception as exc:
            reason = f"WBC command composition failed: {type(exc).__name__}: {exc}"
            self.freeze(reason)
            on_fault(reason)
            with self._lock:
                if self._last_command is None:
                    raise
                return self._last_command
