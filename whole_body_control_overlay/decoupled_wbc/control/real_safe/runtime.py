"""Dependency-injected runtime ordering for standalone real-safe G1 control."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .standalone import RobotSnapshot, SafetyFault, StandaloneRealSafeCore, StandaloneState


class SnapshotReader(Protocol):
    def read(self, now: float) -> RobotSnapshot: ...


class MotionModeTakeover(Protocol):
    def release_mode(self, *, takeover_confirmed: bool) -> None: ...


class WholeBodyCommandTransport(Protocol):
    def prepare(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> None: ...

    def arm_writes(self, *, takeover_confirmed: bool) -> None: ...

    def disarm_writes(self) -> None: ...

    def write_prepared(self) -> None: ...

    def send(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> None: ...


class FaultExitStrategy(Protocol):
    @property
    def verified_for_supported_robot(self) -> bool: ...

    def execute(self, reason: str, last_safe_q: np.ndarray | None) -> bool: ...


class StandaloneRealSafeRuntime:
    """Enforce READ_ONLY -> takeover -> HOLD -> ENGAGE_WBC ordering."""

    def __init__(
        self,
        core: StandaloneRealSafeCore,
        reader: SnapshotReader,
        mode_takeover: MotionModeTakeover,
        transport: WholeBodyCommandTransport,
        fault_exit: FaultExitStrategy,
    ) -> None:
        self.core = core
        self.reader = reader
        self.mode_takeover = mode_takeover
        self.transport = transport
        self.fault_exit = fault_exit
        self.release_count = 0
        self.write_count = 0
        self.fault_exit_count = 0

    def read_only_step(self, now: float) -> RobotSnapshot:
        snapshot = self.reader.read(now)
        self.core.read_only_tick(snapshot, now)
        return snapshot

    @staticmethod
    def _zero_dynamics() -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(29, dtype=np.float64), np.zeros(29, dtype=np.float64)

    def arm_control(self, token: str, now: float) -> np.ndarray:
        if not self.fault_exit.verified_for_supported_robot:
            raise PermissionError(
                "ARM_CONTROL blocked: supported-platform FAULT exit strategy is not verified"
            )
        snapshot = self.reader.read(now)
        hold = self.core.request_arm(token, snapshot, now)
        dq, tau = self._zero_dynamics()
        self.transport.prepare(hold, dq, tau)
        try:
            self.mode_takeover.release_mode(takeover_confirmed=True)
            self.release_count += 1
            self.transport.arm_writes(takeover_confirmed=True)
            self.core.mark_takeover_complete(now)
            self.transport.write_prepared()
            self.write_count += 1
        except Exception as exc:
            self._fault(f"ARM_CONTROL transition failed: {type(exc).__name__}: {exc}")
            raise
        return hold

    def hold_step(self, now: float) -> np.ndarray:
        snapshot = self.reader.read(now)
        try:
            command = self.core.hold_command(snapshot, now)
            dq, tau = self._zero_dynamics()
            self.transport.send(command, dq, tau)
            self.write_count += 1
            return command
        except Exception as exc:
            self._fault(f"HOLD failed: {type(exc).__name__}: {exc}")
            raise

    def begin_wbc_engage(self, now: float) -> None:
        self.core.begin_wbc_engage(now)

    def wbc_step(self, lower_target: np.ndarray, now: float) -> np.ndarray:
        snapshot = self.reader.read(now)
        try:
            command = self.core.wbc_command(snapshot, lower_target, now)
            dq, tau = self._zero_dynamics()
            self.transport.send(command, dq, tau)
            self.write_count += 1
            return command
        except Exception as exc:
            self._fault(f"WBC step failed: {type(exc).__name__}: {exc}")
            raise

    def watchdog_poll(self, now: float) -> bool:
        if not self.core.watchdog_expired(now):
            return False
        self._fault(self.core.fault_reason or "50Hz watchdog expired")
        return True

    def _fault(self, reason: str) -> None:
        if self.core.state != StandaloneState.FAULT:
            self.core._enter_fault(reason)
        # Revoke the normal control loop's authority before handing execution to
        # the separately verified, platform-specific exit strategy.
        try:
            self.transport.disarm_writes()
        except Exception as exc:
            reason = (
                f"{reason}; normal write-gate disarm failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self.core.fault_reason = reason
        self.fault_exit_count += 1
        completed = self.fault_exit.execute(reason, self.core.last_command)
        if completed:
            self.core.mark_exit_complete(verified_platform_exit=True)
        elif self.core.state != StandaloneState.FAULT:
            raise SafetyFault("fault exit returned incomplete but state left FAULT")
