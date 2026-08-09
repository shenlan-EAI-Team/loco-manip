"""Isolated deterministic wrist proof layered on the audited WBC guard."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import json

import numpy as np

from .commissioning import CommissioningPhase, WbcGuardCommandComposer
from .core import GuardCommand
from .runtime import LowCmdGuardRuntime
from ..standalone import ARMS, SafetyFault

if TYPE_CHECKING:
    from ..gear_wbc_producer import GearWbcReadOnlyProducer


PROOF_JOINT_NAME = "left_wrist_yaw_joint"
PROOF_MOTOR_INDEX = 21
PROOF_OFFSET_RAD = 0.01


class WristProofPhase(str, Enum):
    WAITING = "WAITING"
    RAMP = "RAMP"
    HOLD = "HOLD"
    COMPLETE = "COMPLETE"
    FROZEN = "FROZEN"


@dataclass(frozen=True)
class SingleWristProofConfig:
    joint_name: str
    motor_cmd_index: int
    motor_state_index: int
    offset_rad: float
    ramp_duration_s: float
    hold_duration_s: float
    preview_current_q_rad: float
    max_initial_q_deviation_rad: float
    feedback_envelope_rad: float
    reverse_fault_threshold_rad: float
    minimum_success_response_rad: float
    max_abs_feedback_dq_rad_s: float

    @classmethod
    def from_json(cls, path: Path) -> "SingleWristProofConfig":
        values = json.loads(path.read_text())
        if values.get("schema_version") != 1:
            raise ValueError("unsupported single-wrist proof config schema")
        proof = values["single_wrist_proof"]
        config = cls(
            joint_name=str(proof["joint_name"]),
            motor_cmd_index=int(proof["motor_cmd_index"]),
            motor_state_index=int(proof["motor_state_index"]),
            offset_rad=float(proof["offset_rad"]),
            ramp_duration_s=float(proof["ramp_duration_s"]),
            hold_duration_s=float(proof["hold_duration_s"]),
            preview_current_q_rad=float(proof["preview_current_q_rad"]),
            max_initial_q_deviation_rad=float(proof["max_initial_q_deviation_rad"]),
            feedback_envelope_rad=float(proof["feedback_envelope_rad"]),
            reverse_fault_threshold_rad=float(proof["reverse_fault_threshold_rad"]),
            minimum_success_response_rad=float(proof["minimum_success_response_rad"]),
            max_abs_feedback_dq_rad_s=float(proof["max_abs_feedback_dq_rad_s"]),
        )
        if config.joint_name != PROOF_JOINT_NAME:
            raise ValueError("first proof is hard-locked to left_wrist_yaw_joint")
        if config.motor_cmd_index != PROOF_MOTOR_INDEX or config.motor_state_index != PROOF_MOTOR_INDEX:
            raise ValueError("first proof is hard-locked to motor command/state index 21")
        if config.offset_rad != PROOF_OFFSET_RAD:
            raise ValueError("first proof offset must be exactly +0.01 rad")
        scalars = np.asarray(
            [
                config.ramp_duration_s,
                config.hold_duration_s,
                config.preview_current_q_rad,
                config.max_initial_q_deviation_rad,
                config.feedback_envelope_rad,
                config.reverse_fault_threshold_rad,
                config.minimum_success_response_rad,
                config.max_abs_feedback_dq_rad_s,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("single-wrist proof values must be finite")
        if config.ramp_duration_s < 0.5 or config.hold_duration_s <= 0:
            raise ValueError("single-wrist proof timing is too aggressive")
        if not 0 < config.minimum_success_response_rad < config.offset_rad:
            raise ValueError("minimum response must be positive and below the command offset")
        if config.feedback_envelope_rad < config.offset_rad:
            raise ValueError("feedback envelope cannot be smaller than the command offset")
        if min(
            config.max_initial_q_deviation_rad,
            config.reverse_fault_threshold_rad,
            config.max_abs_feedback_dq_rad_s,
        ) <= 0:
            raise ValueError("single-wrist proof safety thresholds must be positive")
        return config


class SingleWristProofComposer(WbcGuardCommandComposer):
    """Adds one rate-limited distal wrist reference after WBC reaches STAND."""

    def __init__(
        self,
        *args,
        proof: SingleWristProofConfig,
        arm_rate_limit: float,
        arm_step_limit: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proof = proof
        self.arm_rate_limit = float(arm_rate_limit)
        self.arm_step_limit = float(arm_step_limit)
        if not np.isfinite([self.arm_rate_limit, self.arm_step_limit]).all():
            raise ValueError("wrist command limits must be finite")
        if self.arm_rate_limit <= 0 or self.arm_step_limit <= 0:
            raise ValueError("wrist command limits must be positive")
        self.proof_phase = WristProofPhase.WAITING
        self.proof_started_at: float | None = None
        self.proof_initial_q: float | None = None
        self.proof_target_q: float | None = None

    def arm_current_q(self, command: GuardCommand, *, now: float) -> None:
        super().arm_current_q(command, now=now)
        initial = float(command.q[self.proof.motor_state_index])
        if abs(initial - self.proof.preview_current_q_rad) > self.proof.max_initial_q_deviation_rad:
            raise SafetyFault(
                "left wrist yaw moved too far from the read-only preview before ReleaseMode"
            )
        target = initial + self.proof.offset_rad
        limits = self.safety.limits
        if target < limits.q_lower[PROOF_MOTOR_INDEX] or target > limits.q_upper[PROOF_MOTOR_INDEX]:
            raise SafetyFault("single-wrist proof target is outside the hard limit")
        self.proof_initial_q = initial
        self.proof_target_q = target

    def begin_proof(self, *, now: float) -> None:
        if self.phase != CommissioningPhase.STAND:
            raise SafetyFault("single-wrist proof requires Gear WBC STAND")
        if self.proof_phase != WristProofPhase.WAITING:
            raise RuntimeError("single-wrist proof cannot be started twice")
        if self.proof_initial_q is None or self.proof_target_q is None:
            raise RuntimeError("single-wrist proof has no arming reference")
        self.proof_started_at = float(now)
        self.proof_phase = WristProofPhase.RAMP

    def command(self, *, now: float) -> GuardCommand:
        with self._lock:
            prior = self._last_command
            prior_time = self._last_compose_time
        command = super().command(now=now)
        if self.proof_phase in {WristProofPhase.WAITING, WristProofPhase.FROZEN}:
            return command
        if self.proof_started_at is None or self.proof_initial_q is None:
            raise RuntimeError("single-wrist proof timing is not initialized")
        if self.phase != CommissioningPhase.STAND:
            raise SafetyFault("Gear WBC left STAND during single-wrist proof")
        if prior is None or prior_time is None:
            raise RuntimeError("single-wrist proof has no previous command")

        elapsed = max(0.0, now - self.proof_started_at)
        progress = float(np.clip(elapsed / self.proof.ramp_duration_s, 0.0, 1.0))
        alpha = progress * progress * (3.0 - 2.0 * progress)
        desired = self.proof_initial_q + alpha * self.proof.offset_rad
        dt = now - prior_time
        if not np.isfinite(dt) or dt <= 0:
            raise SafetyFault("single-wrist command clock must increase strictly")
        allowed = min(self.arm_step_limit, self.arm_rate_limit * dt)
        prior_q = float(prior.q[PROOF_MOTOR_INDEX])
        next_q = prior_q + float(np.clip(desired - prior_q, -allowed, allowed))
        q = command.q.copy()
        q[PROOF_MOTOR_INDEX] = next_q
        self.safety.validate_command(q)
        result = replace(command, q=q, prepared_monotonic=float(now))
        with self._lock:
            self._last_command = result
            self._last_compose_time = float(now)

        if progress < 1.0:
            self.proof_phase = WristProofPhase.RAMP
        elif elapsed < self.proof.ramp_duration_s + self.proof.hold_duration_s:
            self.proof_phase = WristProofPhase.HOLD
        else:
            self.proof_phase = WristProofPhase.COMPLETE
        return result

    def latest_command(self) -> GuardCommand:
        with self._lock:
            if self._last_command is None:
                raise RuntimeError("single-wrist proof composer is not armed")
            return self._last_command

    def freeze(self, reason: str, *, fault: bool = True) -> None:
        super().freeze(reason, fault=fault)
        self.proof_phase = WristProofPhase.FROZEN


class SingleWristProofRuntime(LowCmdGuardRuntime):
    """One supported WBC + wrist proof; handback is intentionally not attempted."""

    def __init__(self, *args, proof: SingleWristProofConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proof = proof
        self.proof_trace: list[dict[str, object]] = []
        self.max_positive_response_rad = 0.0
        self.max_other_arm_drift_rad = 0.0

    def _validate_live_proof(self, composer: SingleWristProofComposer) -> None:
        before = self.clock()
        snapshot = self.source.latest(before)
        now = self.clock()
        self.core._validate_guard_snapshot(snapshot, now)
        prepared = self.core.prepared_command
        if prepared is None:
            raise RuntimeError("single-wrist proof has no arming command")
        if snapshot.mode_machine != prepared.mode_machine:
            raise SafetyFault("mode_machine changed during single-wrist proof")

        q = np.asarray(snapshot.robot.q, dtype=np.float64)
        dq = np.asarray(snapshot.robot.dq, dtype=np.float64)
        arm_delta = q[ARMS] - prepared.q[ARMS]
        joint_arm_offset = PROOF_MOTOR_INDEX - ARMS.start
        other = np.delete(np.abs(arm_delta), joint_arm_offset)
        other_limit = np.delete(
            self.core.safety.limits.hold_feedback_delta_abs_limit[ARMS],
            joint_arm_offset,
        )
        if np.any(other > other_limit):
            arm_indices = np.delete(np.arange(ARMS.start, ARMS.stop), joint_arm_offset)
            indices = arm_indices[np.flatnonzero(other > other_limit)].tolist()
            raise SafetyFault(f"non-target arm feedback drift at motors {indices}")
        self.max_other_arm_drift_rad = max(
            self.max_other_arm_drift_rad,
            float(np.max(other)) if other.size else 0.0,
        )

        actual_delta = float(q[PROOF_MOTOR_INDEX] - prepared.q[PROOF_MOTOR_INDEX])
        if actual_delta < -self.proof.reverse_fault_threshold_rad:
            raise SafetyFault("left wrist yaw feedback moved opposite to the positive command")
        if actual_delta > self.proof.feedback_envelope_rad:
            raise SafetyFault("left wrist yaw feedback exceeded the proof envelope")
        if abs(float(dq[PROOF_MOTOR_INDEX])) > self.proof.max_abs_feedback_dq_rad_s:
            raise SafetyFault("left wrist yaw feedback velocity exceeded the proof limit")
        self.max_positive_response_rad = max(self.max_positive_response_rad, actual_delta)

        command = composer.latest_command()
        command_delta = float(
            command.q[PROOF_MOTOR_INDEX] - prepared.q[PROOF_MOTOR_INDEX]
        )
        if command_delta < -1e-12 or command_delta > self.proof.offset_rad + 1e-12:
            raise SafetyFault("single-wrist command escaped the [0, +0.01] envelope")
        self.proof_trace.append(
            {
                "monotonic_s": now,
                "wbc_phase": composer.phase.value,
                "proof_phase": composer.proof_phase.value,
                "motor_cmd_index": PROOF_MOTOR_INDEX,
                "motor_state_index": PROOF_MOTOR_INDEX,
                "q_initial_rad": float(prepared.q[PROOF_MOTOR_INDEX]),
                "q_target_rad": float(command.q[PROOF_MOTOR_INDEX]),
                "q_actual_rad": float(q[PROOF_MOTOR_INDEX]),
                "command_delta_rad": command_delta,
                "feedback_delta_rad": actual_delta,
                "direction_consistent": bool(actual_delta >= 0.0),
                "dq_rad_s": float(dq[PROOF_MOTOR_INDEX]),
                "tau_est_nm": float(snapshot.motor_tau_est[PROOF_MOTOR_INDEX]),
                "max_other_arm_drift_rad": self.max_other_arm_drift_rad,
            }
        )
        self._record_feedback(snapshot, f"single_wrist_{composer.proof_phase.value.lower()}")

    def proof_summary(self, composer: SingleWristProofComposer) -> dict[str, object]:
        prepared = self.core.prepared_command
        if prepared is None:
            raise RuntimeError("single-wrist proof has no prepared command")
        target = float(prepared.q[PROOF_MOTOR_INDEX] + self.proof.offset_rad)
        return {
            "state": self.core.state.value,
            "commissioning": True,
            "single_wrist_proof": True,
            "proof_joint_name": self.proof.joint_name,
            "motor_cmd_index": PROOF_MOTOR_INDEX,
            "motor_state_index": PROOF_MOTOR_INDEX,
            "q_initial_rad": float(prepared.q[PROOF_MOTOR_INDEX]),
            "q_target_rad": target,
            "offset_rad": self.proof.offset_rad,
            "max_positive_response_rad": self.max_positive_response_rad,
            "max_other_arm_drift_rad": self.max_other_arm_drift_rad,
            "direction_response_pass": bool(
                self.max_positive_response_rad >= self.proof.minimum_success_response_rad
            ),
            "proof_phase": composer.proof_phase.value,
            "proof_trace": self.proof_trace,
            "release_calls": self.release_calls,
            "select_calls": self.select_calls,
            "write_calls": self.write_calls,
            "planned_end": "local_500hz_last_valid_hold_until_confirmed_physical_estop",
            **self.diagnostic_summary(),
        }

    def execute(
        self,
        *,
        token: str,
        hardware_transport_enabled: bool,
        command_publication_enabled: bool,
        lifecycle_armed: bool,
        producer: "GearWbcReadOnlyProducer",
        composer: SingleWristProofComposer,
        completion_hold: Callable[[dict[str, object]], None],
    ) -> None:
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
        producer_scheduler = None

        try:
            self.writer = self.writer_factory()
            before = self.clock()
            snapshot = self.source.latest(before)
            now = self.clock()
            command = self.core.refresh_current_q_hold_after_transport(
                snapshot,
                self._check_owner(),
                now,
            )
            self._record_feedback(snapshot, "silent_transport_refresh")
            composer.arm_current_q(command, now=now)
            self.command_provider = lambda tick: composer.command_or_freeze(
                now=tick,
                on_fault=self.signal_local_fault,
            )
            if self.clock() - command.prepared_monotonic > self.core.config.command_prepare_max_age_s:
                raise SafetyFault("current-q command became stale before proof ReleaseMode")
            if self._check_owner() != self.core.config.required_initial_owner:
                raise SafetyFault("owner changed before proof ReleaseMode")
            producer.tick()
            self._raise_if_local_fault()

            status = self.mode.release_mode()
            self.release_calls += 1
            if status != 0:
                self.core.enter_fault(f"ReleaseMode failed: status={status}")
                raise SafetyFault(self.core.fault_reason)
            self.release_succeeded = True
            self.core.mark_release_returned(self.clock())
            self._write()
            self.core.mark_first_write(self.clock())

            from .scheduler import NoCatchUpScheduler

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
            stand_deadline = (
                self.clock()
                + composer.engage_duration_s
                + self.core.config.commissioning_stand_duration_s
            )
            while self.clock() < stand_deadline:
                self._raise_if_local_fault()
                self._validate_live_commissioning(composer)
                self.sleep(1.0 / self.core.config.policy_target_frequency_hz)
            if composer.phase != CommissioningPhase.STAND:
                raise SafetyFault("Gear WBC did not reach STAND before wrist proof")

            composer.begin_proof(now=self.clock())
            proof_deadline = (
                self.clock()
                + self.proof.ramp_duration_s
                + self.proof.hold_duration_s
                + 1.0 / self.core.config.policy_target_frequency_hz
            )
            while self.clock() < proof_deadline:
                self._raise_if_local_fault()
                self._validate_live_proof(composer)
                self.sleep(1.0 / self.core.config.policy_target_frequency_hz)
            if composer.proof_phase != WristProofPhase.COMPLETE:
                raise SafetyFault("single-wrist proof did not complete its one-shot window")
            if self.max_positive_response_rad < self.proof.minimum_success_response_rad:
                raise SafetyFault("single-wrist command produced no measurable positive response")

            composer.freeze("planned single-wrist proof complete", fault=False)
            producer_scheduler.stop()
            completion_hold(self.proof_summary(composer))
            while True:
                self._raise_if_local_fault()
                self._validate_live_proof(composer)
                self.sleep(1.0 / self.core.config.policy_target_frequency_hz)
        except BaseException as exc:
            composer.freeze(f"single-wrist proof stopped: {type(exc).__name__}: {exc}")
            if producer_scheduler is not None:
                producer_scheduler.stop()
            if not self.release_succeeded and self.writer is not None:
                self.writer.close()
            raise
