"""Dependency-injected local takeover/hold/recovery runtime."""

from __future__ import annotations

from threading import Event
import time
from typing import Callable, Protocol, TYPE_CHECKING

import numpy as np

from .core import GuardCommand, GuardSnapshot, GuardState, LowCmdGuardCore
from .scheduler import NoCatchUpScheduler
from .commissioning import CommissioningPhase, WbcGuardCommandComposer
from ..standalone import SafetyFault

if TYPE_CHECKING:
    from ..gear_wbc_producer import GearWbcReadOnlyProducer


class SnapshotSource(Protocol):
    def latest(self, now: float) -> GuardSnapshot: ...


class MotionModeClient(Protocol):
    def check_mode(self) -> tuple[int, str, str]: ...

    def release_mode(self) -> int: ...

    def select_mode(self, owner: str) -> int: ...


class LowCmdWriter(Protocol):
    def write(self, command: GuardCommand) -> None: ...

    def close(self) -> None: ...


class LowCmdGuardRuntime:
    """Runs exactly one current-q lifecycle; it never retries automatically."""

    def __init__(
        self,
        core: LowCmdGuardCore,
        source: SnapshotSource,
        mode: MotionModeClient,
        writer_factory: Callable[[], LowCmdWriter],
        *,
        authorization_commit: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.core = core
        self.source = source
        self.mode = mode
        self.writer_factory = writer_factory
        self.authorization_commit = authorization_commit or (lambda: None)
        self.clock = clock
        self.sleep = sleep
        self.writer: LowCmdWriter | None = None
        self.scheduler: NoCatchUpScheduler | None = None
        self.local_fault = Event()
        self.local_fault_reason: str | None = None
        self.release_calls = 0
        self.release_succeeded = False
        self.select_calls = 0
        self.write_calls = 0
        self.recovery_attempted = False
        self.feedback_trace: list[dict[str, object]] = []
        self.owner_trace: list[dict[str, object]] = []
        self.command_provider: Callable[[float], GuardCommand] | None = None

    def _check_owner(self) -> str:
        status, _form, owner = self.mode.check_mode()
        if status != 0:
            raise SafetyFault(f"MotionSwitcher CheckMode failed: status={status}")
        self.owner_trace.append(
            {"monotonic_s": self.clock(), "form": _form, "owner": owner}
        )
        return owner

    def _record_feedback(self, snapshot: GuardSnapshot, phase: str) -> None:
        robot = snapshot.robot
        self.feedback_trace.append(
            {
                "monotonic_s": self.clock(),
                "phase": phase,
                "state": self.core.state.value,
                "mode_machine": snapshot.mode_machine,
                "q_rad": np.asarray(robot.q).tolist(),
                "dq_rad_s": np.asarray(robot.dq).tolist(),
                "tau_est_nm": np.asarray(snapshot.motor_tau_est).tolist(),
                "base_quat_wxyz": np.asarray(robot.base_quat_wxyz).tolist(),
                "base_angular_velocity_rad_s": np.asarray(
                    robot.base_angular_velocity
                ).tolist(),
                "secondary_quat_wxyz": np.asarray(robot.secondary_quat_wxyz).tolist(),
                "secondary_angular_velocity_rad_s": np.asarray(
                    robot.secondary_angular_velocity
                ).tolist(),
                "lowstate_monotonic_s": robot.lowstate_monotonic,
                "imu_monotonic_s": robot.imu_monotonic,
            }
        )

    def read_only_step(self) -> GuardSnapshot:
        before = self.clock()
        snapshot = self.source.latest(before)
        now = self.clock()
        self.core.observe(snapshot, self._check_owner(), now)
        self._record_feedback(snapshot, "prepare_read_only")
        return snapshot

    def prepare_current_q_hold(self) -> GuardCommand:
        self.read_only_step()
        return self.core.prepare_current_q_hold(self.clock())

    def _write(self) -> None:
        if self.writer is None or self.core.prepared_command is None:
            raise RuntimeError("writer or prepared current-q command is missing")
        command = (
            self.core.prepared_command
            if self.command_provider is None
            else self.command_provider(self.clock())
        )
        self.writer.write(command)
        self.write_calls += 1

    def _writer_fault(self, exc: Exception) -> None:
        self.local_fault_reason = f"local 500Hz writer failed: {type(exc).__name__}: {exc}"
        self.local_fault.set()

    def signal_local_fault(self, reason: str) -> None:
        """Signal-safe request observed by the local lifecycle loop."""
        if not self.local_fault.is_set():
            self.local_fault_reason = reason
        self.local_fault.set()

    def _raise_if_local_fault(self) -> None:
        if self.local_fault.is_set():
            raise SafetyFault(self.local_fault_reason or "local lifecycle stop requested")

    def _validate_live_hold(self) -> None:
        now_before = self.clock()
        snapshot = self.source.latest(now_before)
        now = self.clock()
        self.core.validate_hold_feedback(snapshot, now)
        self._record_feedback(snapshot, "active_hold_or_recovery")

    def _validate_live_commissioning(
        self,
        composer: WbcGuardCommandComposer,
    ) -> None:
        before = self.clock()
        snapshot = self.source.latest(before)
        now = self.clock()
        self.core._validate_guard_snapshot(snapshot, now)
        prepared = self.core.prepared_command
        if prepared is None:
            raise RuntimeError("commissioning current-q command is missing")
        if snapshot.mode_machine != prepared.mode_machine:
            raise SafetyFault("mode_machine changed during WBC commissioning")
        arm_delta = np.abs(snapshot.robot.q[15:29] - prepared.q[15:29])
        arm_limits = self.core.safety.limits.hold_feedback_delta_abs_limit[15:29]
        if np.any(arm_delta > arm_limits):
            indices = (np.flatnonzero(arm_delta > arm_limits) + 15).tolist()
            raise SafetyFault(f"frozen-arm feedback drift at motors {indices}")
        self._record_feedback(snapshot, f"commissioning_{composer.phase.value.lower()}")

    def _verify_owner(
        self,
        expected: str,
        timeout_s: float,
        *,
        abort_on_local_fault: bool,
    ) -> None:
        deadline = self.clock() + timeout_s
        while self.clock() < deadline:
            if abort_on_local_fault:
                self._raise_if_local_fault()
            owner = self._check_owner()
            if owner == expected:
                return
            self._validate_live_hold()
            self.sleep(0.002)
        raise SafetyFault(f"owner verification timeout: expected {expected!r}")

    def execute_current_q_lifecycle(
        self,
        *,
        token: str,
        hardware_transport_enabled: bool,
        command_publication_enabled: bool,
        lifecycle_armed: bool,
    ) -> dict[str, object]:
        command = self.prepare_current_q_hold()
        self.core.authorize_takeover(
            token=token,
            hardware_transport_enabled=hardware_transport_enabled,
            command_publication_enabled=command_publication_enabled,
            lifecycle_armed=lifecycle_armed,
            now=self.clock(),
        )
        self.authorization_commit()

        # The real writer can only be constructed after every fail-closed gate
        # has passed. It remains silent until ReleaseMode succeeds.
        try:
            self.writer = self.writer_factory()
            refresh_before = self.clock()
            refresh_snapshot = self.source.latest(refresh_before)
            refresh_now = self.clock()
            refresh_owner = self._check_owner()
            command = self.core.refresh_current_q_hold_after_transport(
                refresh_snapshot,
                refresh_owner,
                refresh_now,
            )
            self._record_feedback(refresh_snapshot, "silent_transport_refresh")
            age = self.clock() - command.prepared_monotonic
            if age < 0 or age > self.core.config.command_prepare_max_age_s:
                self.core.enter_fault(
                    f"current-q command became stale before ReleaseMode: age={age:.6f}s"
                )
                raise SafetyFault(self.core.fault_reason)
            self._raise_if_local_fault()
            owner_before_release = self._check_owner()
            if owner_before_release != self.core.config.required_initial_owner:
                self.core.enter_fault(
                    f"owner changed before ReleaseMode: {owner_before_release!r}"
                )
                raise SafetyFault(self.core.fault_reason)
            self._raise_if_local_fault()
            release_status = self.mode.release_mode()
            self.release_calls += 1
            if release_status != 0:
                self.core.enter_fault(f"ReleaseMode failed: status={release_status}")
                raise SafetyFault(self.core.fault_reason)
            self.release_succeeded = True
            release_return = self.clock()
            self.core.mark_release_returned(release_return)

            # First HOLD is synchronous in the same local thread. No owner poll,
            # PC RPC, or policy operation is allowed in this interval.
            self._write()
            first_write = self.clock()
            self.core.mark_first_write(first_write)

            self.scheduler = NoCatchUpScheduler(
                self.core.config.transport_frequency_hz,
                self._write,
                on_error=self._writer_fault,
                clock=self.clock,
            )
            self.scheduler.start()
            self._verify_owner(
                "",
                self.core.config.release_verify_timeout_s,
                abort_on_local_fault=True,
            )
            self.core.mark_hold_verified("")

            hold_deadline = self.clock() + self.core.config.hold_duration_s
            while self.clock() < hold_deadline:
                self._raise_if_local_fault()
                self._validate_live_hold()
                self.sleep(1.0 / self.core.config.policy_target_frequency_hz)

            self._recover_to_original_owner("planned current-q HOLD complete")
            self._post_recovery_read_only_monitor()
        except BaseException as exc:
            if not self.release_succeeded and self.writer is not None:
                self.writer.close()
            elif self.release_succeeded and self.core.state in {
                GuardState.TAKEOVER_PENDING,
                GuardState.HOLD,
            }:
                self._recover_to_original_owner(f"fault: {type(exc).__name__}: {exc}")
            raise

        assert self.scheduler is not None
        return {
            "state": self.core.state.value,
            "release_calls": self.release_calls,
            "select_calls": self.select_calls,
            "write_calls": self.write_calls,
            "writer_transport_alive": self._writer_transport_alive(),
            "release_to_first_write_s": (
                self.core.first_write_monotonic - self.core.release_return_monotonic
            ),
            "scheduler": self.scheduler.metrics.summary(),
            "original_owner": self.core.original_owner,
            **self.diagnostic_summary(),
        }

    def _recover_to_original_owner(
        self,
        reason: str,
        *,
        allow_supported_commissioning: bool = False,
    ) -> None:
        if self.writer is None:
            raise SafetyFault("cannot recover after release without a local LowCmd writer object")
        if (
            not self.core.config.recovery_handoff_verified
            and not allow_supported_commissioning
        ):
            self.core.enter_fault("recovery handoff is not verified")
            raise SafetyFault(self.core.fault_reason)
        if self.recovery_attempted:
            self.core.enter_fault("automatic recovery retry is forbidden")
            raise SafetyFault(self.core.fault_reason)
        self.recovery_attempted = True
        if self.core.state != GuardState.RECOVERY_PENDING:
            self.core.begin_recovery(reason)
        owner = self.core.original_owner
        if not owner:
            self.core.enter_fault("original motion owner was not recorded")
            raise SafetyFault(self.core.fault_reason)

        # Candidate sequence: keep the local current-q HOLD alive while asking
        # MotionSwitcher to restore the recorded owner, then stop LowCmd only
        # after CheckMode confirms ownership. This ordering is hard-disabled
        # until it is validated under physical support on this exact G1.
        status = self.mode.select_mode(owner)
        self.select_calls += 1
        if status != 0:
            self.core.enter_fault(f"SelectMode({owner!r}) failed: status={status}")
            raise SafetyFault(self.core.fault_reason)
        self.core.mark_owner_selection_requested()
        self._verify_owner(
            owner,
            self.core.config.recovery_verify_timeout_s,
            abort_on_local_fault=False,
        )
        if self.scheduler is not None:
            self.scheduler.stop()
        self.writer.close()
        self.core.mark_recovered(owner)

    def execute_wbc_commissioning(
        self,
        *,
        token: str,
        hardware_transport_enabled: bool,
        command_publication_enabled: bool,
        lifecycle_armed: bool,
        producer: "GearWbcReadOnlyProducer",
        composer: WbcGuardCommandComposer,
    ) -> dict[str, object]:
        """Run one supported HOLD -> smooth engage -> short stand lifecycle."""
        # Warm and validate the complete six-frame ONNX history before consuming
        # authorization or constructing the silent LowCmd writer.
        for _ in range(self.core.config.commissioning_wbc_preflight_samples):
            producer.tick()
            self.sleep(1.0 / self.core.config.policy_target_frequency_hz)
        command = self.prepare_current_q_hold()
        self.core.authorize_takeover(
            token=token,
            hardware_transport_enabled=hardware_transport_enabled,
            command_publication_enabled=command_publication_enabled,
            lifecycle_armed=lifecycle_armed,
            now=self.clock(),
            commissioning_mode=True,
        )
        self.authorization_commit()
        producer_scheduler: NoCatchUpScheduler | None = None

        try:
            self.writer = self.writer_factory()
            refresh_before = self.clock()
            refresh_snapshot = self.source.latest(refresh_before)
            refresh_now = self.clock()
            command = self.core.refresh_current_q_hold_after_transport(
                refresh_snapshot,
                self._check_owner(),
                refresh_now,
            )
            self._record_feedback(refresh_snapshot, "silent_transport_refresh")
            composer.arm_current_q(command, now=refresh_now)
            self.command_provider = lambda now: composer.command_or_freeze(
                now=now,
                on_fault=self.signal_local_fault,
            )
            if self.clock() - command.prepared_monotonic > self.core.config.command_prepare_max_age_s:
                raise SafetyFault("current-q command became stale before commissioning ReleaseMode")
            if self._check_owner() != self.core.config.required_initial_owner:
                raise SafetyFault("owner changed before commissioning ReleaseMode")
            # Writer discovery can take seconds. Re-run one read-only inference
            # against the refreshed state immediately before ownership changes.
            producer.tick()
            self._raise_if_local_fault()

            release_status = self.mode.release_mode()
            self.release_calls += 1
            if release_status != 0:
                self.core.enter_fault(f"ReleaseMode failed: status={release_status}")
                raise SafetyFault(self.core.fault_reason)
            self.release_succeeded = True
            release_return = self.clock()
            self.core.mark_release_returned(release_return)
            self._write()
            self.core.mark_first_write(self.clock())

            self.scheduler = NoCatchUpScheduler(
                self.core.config.transport_frequency_hz,
                self._write,
                on_error=self._writer_fault,
                clock=self.clock,
            )
            self.scheduler.start()
            self._verify_owner(
                "",
                self.core.config.release_verify_timeout_s,
                abort_on_local_fault=True,
            )
            self.core.mark_hold_verified("")

            producer_scheduler = NoCatchUpScheduler(
                self.core.config.policy_target_frequency_hz,
                producer.tick,
                on_error=lambda exc: self.signal_local_fault(
                    f"50Hz Gear WBC producer failed: {type(exc).__name__}: {exc}"
                ),
                clock=self.clock,
            )
            producer_scheduler.start()
            hold_deadline = self.clock() + self.core.config.hold_duration_s
            while self.clock() < hold_deadline:
                self._raise_if_local_fault()
                self._validate_live_hold()
                self.sleep(1.0 / self.core.config.policy_target_frequency_hz)

            composer.begin_engage(now=self.clock())
            active_deadline = (
                self.clock()
                + composer.engage_duration_s
                + self.core.config.commissioning_stand_duration_s
            )
            while self.clock() < active_deadline:
                self._raise_if_local_fault()
                self._validate_live_commissioning(composer)
                self.sleep(1.0 / self.core.config.policy_target_frequency_hz)
            if composer.phase != CommissioningPhase.STAND:
                raise SafetyFault(
                    f"commissioning engage did not reach STAND: {composer.phase.value}"
                )

            composer.freeze("planned commissioning stand complete", fault=False)
            producer_scheduler.stop()
            self._recover_to_original_owner(
                "planned commissioning stand complete",
                allow_supported_commissioning=True,
            )
            self._post_recovery_read_only_monitor()
        except BaseException as exc:
            composer.freeze(f"commissioning fault: {type(exc).__name__}: {exc}")
            if producer_scheduler is not None:
                producer_scheduler.stop()
            if not self.release_succeeded and self.writer is not None:
                self.writer.close()
            elif self.release_succeeded and self.core.state in {
                GuardState.TAKEOVER_PENDING,
                GuardState.HOLD,
            }:
                self._recover_to_original_owner(
                    f"commissioning fault: {type(exc).__name__}: {exc}",
                    allow_supported_commissioning=True,
                )
            raise

        assert self.scheduler is not None and producer_scheduler is not None
        return {
            "state": self.core.state.value,
            "commissioning": True,
            "release_calls": self.release_calls,
            "select_calls": self.select_calls,
            "write_calls": self.write_calls,
            "producer_inferences": producer.inference_count,
            "producer_scheduler": producer_scheduler.metrics.summary(),
            "transport_scheduler": self.scheduler.metrics.summary(),
            "composer_phase": composer.phase.value,
            "composer_fault": composer.fault_reason,
            **self.diagnostic_summary(),
        }

    def _post_recovery_read_only_monitor(self) -> None:
        deadline = self.clock() + self.core.config.post_recovery_monitor_s
        while self.clock() < deadline:
            before = self.clock()
            snapshot = self.source.latest(before)
            now = self.clock()
            self.core.safety.validate_snapshot(snapshot.robot, now)
            owner = self._check_owner()
            if owner != self.core.config.required_initial_owner:
                raise SafetyFault("recovered owner changed during post-recovery monitor")
            self._record_feedback(snapshot, "post_recovery_read_only")
            self.sleep(1.0 / self.core.config.policy_target_frequency_hz)

    def _writer_transport_alive(self) -> bool:
        if self.scheduler is None or self.scheduler._thread is None:
            return False
        return self.scheduler._thread.is_alive()

    def diagnostic_summary(self) -> dict[str, object]:
        result: dict[str, object] = {
            "feedback_trace": self.feedback_trace,
            "owner_trace": self.owner_trace,
        }
        if self.core.prepared_command is not None:
            command = self.core.prepared_command
            result["prepared_command"] = {
                "q_rad": command.q.tolist(),
                "dq_rad_s": command.dq.tolist(),
                "kp": command.kp.tolist(),
                "kd": command.kd.tolist(),
                "tau_nm": command.tau.tolist(),
                "motor_mode": command.motor_mode.tolist(),
                "mode_pr": command.mode_pr,
                "mode_machine": command.mode_machine,
            }
        if self.feedback_trace and self.core.prepared_command is not None:
            q = np.asarray([sample["q_rad"] for sample in self.feedback_trace])
            dq = np.asarray([sample["dq_rad_s"] for sample in self.feedback_trace])
            tau = np.asarray([sample["tau_est_nm"] for sample in self.feedback_trace])
            result["max_abs_feedback_delta_rad_by_motor"] = np.max(
                np.abs(q - self.core.prepared_command.q), axis=0
            ).tolist()
            result["max_abs_dq_rad_s_by_motor"] = np.max(np.abs(dq), axis=0).tolist()
            result["max_abs_tau_est_nm_by_motor"] = np.max(np.abs(tau), axis=0).tolist()
        return result

    def request_local_recovery(self, reason: str) -> None:
        """Operator/watchdog entry that never depends on the PC target path."""
        if not self.release_succeeded:
            self.core.enter_fault(reason)
            return
        if self.core.state == GuardState.STOPPED:
            return
        self._recover_to_original_owner(reason)
