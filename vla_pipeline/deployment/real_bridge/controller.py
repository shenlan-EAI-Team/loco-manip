from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .envelope import MicroMotionEnvelope
from .logging import JsonlBridgeLogger
from .mapping import percentages_to_raw
from .models import BridgeState, FeedbackSnapshot, validate_groups
from .transports import G1Transport, O6Transport


GROUP_NAMES = {
    "left_arm": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
    "right_arm": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"],
    "left_o6": ["thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"],
    "right_o6": ["thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"],
}
ARM_MOTOR_INDICES = {
    "left_arm": tuple(range(15, 22)),
    "right_arm": tuple(range(22, 29)),
}


class RealBridgeSession:
    def __init__(
        self,
        g1: G1Transport,
        o6: O6Transport,
        logger: JsonlBridgeLogger,
        *,
        arm_publish_hz: float = 50.0,
        o6_publish_hz: float = 30.0,
        activation_ramp_s: float = 1.0,
        release_ramp_s: float = 2.0,
        arm_excursion_rad: float = 0.01,
        arm_velocity_rad_s: float = 0.12,
        arm_acceleration_rad_s2: float = 0.4,
        o6_excursion_points: float = 5.0,
        o6_velocity_points_s: float = 15.0,
        release_max_arm_rebound_rad: float = 0.01,
        post_release_monitor_s: float = 0.5,
        o6_feedback_stale_timeout_s: float = 0.2,
        scheduler_max_lateness_s: float = 0.04,
        o6_position_commands_enabled: bool = False,
    ) -> None:
        self.g1 = g1
        self.o6 = o6
        self.logger = logger
        self.arm_publish_hz = arm_publish_hz
        self.o6_publish_hz = o6_publish_hz
        self.activation_ramp_s = activation_ramp_s
        self.release_ramp_s = release_ramp_s
        self.release_max_arm_rebound_rad = release_max_arm_rebound_rad
        self.post_release_monitor_s = post_release_monitor_s
        self.o6_feedback_stale_timeout_s = o6_feedback_stale_timeout_s
        self.scheduler_max_lateness_s = scheduler_max_lateness_s
        self.o6_position_commands_enabled = bool(o6_position_commands_enabled)
        self.state = BridgeState.READY
        self.envelope = MicroMotionEnvelope(
            arm_excursion_rad=arm_excursion_rad,
            arm_velocity_rad_s=arm_velocity_rad_s,
            arm_acceleration_rad_s2=arm_acceleration_rad_s2,
            o6_excursion_points=o6_excursion_points,
            o6_velocity_points_s=o6_velocity_points_s,
        )
        self.anchor: FeedbackSnapshot | None = None
        self.last_feedback: FeedbackSnapshot | None = None
        self.last_target: dict[str, np.ndarray] | None = None
        self.last_policy_raw: dict[str, np.ndarray] | None = None
        self.stop_reason: str | None = None
        self.current_weight = 0.0
        self._last_arm_publish_actual: float | None = None
        self.release_error: str | None = None
        self._right_zero_clamp_inferences: set[int] = set()
        self.o6_position_command_count = 0
        self.waist_leg_command_count = 0
        self._o6_lock = threading.Lock()
        self._o6_stop = threading.Event()
        self._o6_thread: threading.Thread | None = None
        self._o6_latest: dict[str, np.ndarray] | None = None
        self._o6_latest_monotonic_ns = 0
        self._o6_error: Exception | None = None
        self._o6_command_active = False
        self._o6_command_target: dict[str, np.ndarray] | None = None
        self._o6_command_generation = 0
        self.logger.write("state", state=self.state.value, ownership="arm_sdk weight=0")

    @staticmethod
    def _validate_o6(hands: dict[str, Any]) -> dict[str, np.ndarray]:
        if set(hands) != {"left_o6", "right_o6"}:
            raise ValueError("O6 feedback must contain exactly left_o6 and right_o6")
        parsed = {}
        for key in ("left_o6", "right_o6"):
            value = np.asarray(hands[key], dtype=np.float64).reshape(-1)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError(f"{key} feedback must contain six finite values")
            if np.any(value < 0) or np.any(value > 100):
                raise RuntimeError(f"{key} feedback outside training 0..100 scale")
            parsed[key] = value.copy()
        return parsed

    @staticmethod
    def _compose_feedback(
        g1: FeedbackSnapshot,
        hands: dict[str, np.ndarray],
    ) -> FeedbackSnapshot:
        groups = dict(g1.groups)
        groups.update(hands)
        return FeedbackSnapshot.create(
            groups,
            monotonic_ns=g1.monotonic_ns,
            g1_mode_machine=g1.g1_mode_machine,
            g1_mode_pr=g1.g1_mode_pr,
            waist=g1.waist,
        )

    def feedback(self) -> FeedbackSnapshot:
        g1 = self.g1.feedback()
        hands = self._validate_o6(self.o6.feedback())
        return self._compose_feedback(g1, hands)

    def _start_o6_monitor(self, initial: dict[str, np.ndarray]) -> None:
        if self._o6_thread is not None:
            raise RuntimeError("O6 feedback monitor already started")
        with self._o6_lock:
            self._o6_latest = self._validate_o6(initial)
            self._o6_latest_monotonic_ns = time.monotonic_ns()
            self._o6_error = None
        self._o6_stop.clear()
        self._o6_thread = threading.Thread(
            target=self._o6_monitor_loop,
            name="o6-independent-io-worker",
            daemon=True,
        )
        self._o6_thread.start()

    def _o6_monitor_loop(self) -> None:
        period = 1.0 / self.o6_publish_hz
        next_deadline = time.monotonic()
        while not self._o6_stop.is_set():
            remaining = next_deadline - time.monotonic()
            if remaining > 0 and self._o6_stop.wait(remaining):
                return
            started = time.monotonic()
            try:
                with self._o6_lock:
                    command_active = self._o6_command_active
                    command_target = (
                        None
                        if self._o6_command_target is None
                        else {
                            key: value.copy()
                            for key, value in self._o6_command_target.items()
                        }
                    )
                    command_generation = self._o6_command_generation
                if command_active:
                    if command_target is None:
                        raise RuntimeError("O6 command active without a target")
                    self._send_left_hand(
                        command_target,
                        command_generation=command_generation,
                        io_started=started,
                    )
                hands = self._validate_o6(self.o6.feedback())
            except Exception as exc:
                with self._o6_lock:
                    self._o6_error = exc
                self.logger.write(
                    "o6_worker_error",
                    state=self.state.value,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                    arm_release_required=self.current_weight > 0.0,
                )
                return
            with self._o6_lock:
                self._o6_latest = hands
                self._o6_latest_monotonic_ns = time.monotonic_ns()
            # Rebase after an overrun; never issue catch-up CAN operations.
            next_deadline = max(next_deadline + period, time.monotonic())

    def _set_o6_command_active(self, active: bool) -> None:
        with self._o6_lock:
            if active and not self.o6_position_commands_enabled:
                raise PermissionError("O6 position commands are disabled for this session")
            self._o6_command_active = bool(active)
            if not active:
                self._o6_command_target = None

    def _queue_o6_target(self, target: dict[str, np.ndarray]) -> int:
        if not self.o6_position_commands_enabled:
            raise PermissionError("O6 position commands are disabled for this session")
        parsed = validate_groups(target, label="o6_worker_target")
        with self._o6_lock:
            self._o6_command_target = {
                key: parsed[key].copy() for key in ("left_o6", "right_o6")
            }
            self._o6_command_generation += 1
            return self._o6_command_generation

    def _cached_o6(self) -> tuple[dict[str, np.ndarray], float]:
        with self._o6_lock:
            error = self._o6_error
            values = self._o6_latest
            timestamp_ns = self._o6_latest_monotonic_ns
        if error is not None:
            raise RuntimeError(f"O6 feedback monitor failed: {type(error).__name__}: {error}")
        if values is None or timestamp_ns == 0:
            raise TimeoutError("O6 feedback monitor has no sample")
        age_s = (time.monotonic_ns() - timestamp_ns) / 1e9
        if age_s > self.o6_feedback_stale_timeout_s:
            raise TimeoutError(f"O6 feedback stale: {age_s * 1000.0:.1f} ms")
        return ({key: value.copy() for key, value in values.items()}, age_s)

    def arm_hold(self) -> FeedbackSnapshot:
        if self.state is not BridgeState.READY:
            raise RuntimeError(f"cannot arm hold from {self.state.value}")
        first = self.feedback()
        time.sleep(0.1)
        self.anchor = self.feedback()
        if (
            first.g1_mode_machine != self.anchor.g1_mode_machine
            or first.g1_mode_pr != self.anchor.g1_mode_pr
        ):
            raise RuntimeError("G1 mode fields were not stable before arming")
        for key in ("left_arm", "right_arm"):
            if np.any(np.abs(first.groups[key] - self.anchor.groups[key]) > 0.01):
                raise RuntimeError(f"{key} moved more than 0.01 rad during pre-arming check")
        self.last_feedback = self.anchor
        self.last_target = {key: value.copy() for key, value in self.anchor.groups.items()}
        self.envelope.reset(self.anchor.groups)
        self._start_o6_monitor(
            {key: self.anchor.groups[key] for key in ("left_o6", "right_o6")}
        )
        self.state = BridgeState.ARMED_HOLD
        self.logger.write("arming_feedback", state=self.state.value, feedback=self.anchor.as_dict())
        return self.anchor

    def _send_arms(
        self,
        target: dict[str, np.ndarray],
        weight: float,
        *,
        schedule: dict[str, Any] | None = None,
    ) -> None:
        if self.anchor is None or self.anchor.g1_mode_machine is None or self.anchor.g1_mode_pr is None:
            raise RuntimeError("arming mode fields are unavailable")
        actual = self.g1.send_arms(
            target["left_arm"],
            target["right_arm"],
            weight=weight,
            mode_machine=self.anchor.g1_mode_machine,
            mode_pr=self.anchor.g1_mode_pr,
        )
        self.current_weight = float(weight)
        self.logger.write(
            "command",
            state=self.state.value,
            transport="g1_arm_sdk",
            final_target={key: target[key] for key in ("left_arm", "right_arm")},
            actual_command=actual,
            ownership={"requested": False, "arm_sdk_weight": weight},
            waist_leg_command_count=self.waist_leg_command_count,
            scheduler=schedule,
        )

    def _send_left_hand(
        self,
        target: dict[str, np.ndarray],
        *,
        command_generation: int,
        io_started: float,
    ) -> None:
        if not self.o6_position_commands_enabled:
            raise PermissionError("O6 position commands are disabled for this session")
        left_raw = percentages_to_raw(target["left_o6"])
        actual = self.o6.send_left_hand(left_raw)
        self.o6_position_command_count += 1
        self.logger.write(
            "command",
            state=self.state.value,
            transport="o6_can_position",
            final_target={"left_o6": target["left_o6"]},
            actual_command={
                "left_raw_255": left_raw,
                "right_o6_command": None,
                "transport_response": actual,
            },
            right_o6_feedback_only=True,
            right_o6_command_count=0,
            o6_position_command_count=self.o6_position_command_count,
            ownership={"requested": False, "can_exclusive_lock": True},
            o6_worker={
                "thread": threading.current_thread().name,
                "target_generation": command_generation,
                "io_started_monotonic_ns": int(io_started * 1e9),
                "setter_completed_monotonic_ns": time.monotonic_ns(),
            },
        )

    def _check_feedback(self, *, hold: bool) -> FeedbackSnapshot:
        if self.anchor is None:
            raise RuntimeError("session has no arming anchor")
        current, o6_age_s = self._read_cached_feedback()
        if hold:
            for key, threshold in (
                ("left_arm", 0.01),
                ("right_arm", 0.01),
                ("left_o6", 2.0),
                ("right_o6", 2.0),
            ):
                if np.any(np.abs(current.groups[key] - self.anchor.groups[key]) > threshold):
                    raise RuntimeError(f"current-position hold jump exceeded threshold for {key}")
        else:
            self.envelope.assert_feedback_within_envelope(current.groups)
            self._check_direction_and_speed(current)
        self.last_feedback = current
        self.logger.write(
            "feedback",
            state=self.state.value,
            feedback=current.as_dict(),
            o6_cached_age_ms=o6_age_s * 1000.0,
        )
        self._log_joint_response(current, phase="hold" if hold else "micro")
        return current

    def _read_cached_feedback(self) -> tuple[FeedbackSnapshot, float]:
        if self.anchor is None:
            raise RuntimeError("session has no arming anchor")
        g1 = self.g1.feedback()
        hands, o6_age_s = self._cached_o6()
        current = self._compose_feedback(g1, hands)
        if (
            current.g1_mode_machine != self.anchor.g1_mode_machine
            or current.g1_mode_pr != self.anchor.g1_mode_pr
        ):
            raise RuntimeError("G1 control mode changed during upper-body micro-motion")
        return current, o6_age_s

    def _read_release_feedback(self) -> tuple[FeedbackSnapshot, dict[str, Any]]:
        """Read G1 directly while treating O6 status as non-blocking metadata."""
        if self.anchor is None:
            raise RuntimeError("session has no arming anchor")
        g1 = self.g1.feedback()
        if (
            g1.g1_mode_machine != self.anchor.g1_mode_machine
            or g1.g1_mode_pr != self.anchor.g1_mode_pr
        ):
            raise RuntimeError("G1 control mode changed during arm release")
        with self._o6_lock:
            values = self._o6_latest
            timestamp_ns = self._o6_latest_monotonic_ns
            error = self._o6_error
        hands = (
            {key: self.anchor.groups[key].copy() for key in ("left_o6", "right_o6")}
            if values is None
            else {key: value.copy() for key, value in values.items()}
        )
        age_ms = None if timestamp_ns == 0 else (time.monotonic_ns() - timestamp_ns) / 1e6
        return self._compose_feedback(g1, hands), {
            "required_for_release": False,
            "age_ms": age_ms,
            "stale": age_ms is None or age_ms > self.o6_feedback_stale_timeout_s * 1000.0,
            "error": None if error is None else f"{type(error).__name__}: {error}",
        }

    def _log_joint_response(
        self,
        current: FeedbackSnapshot,
        *,
        phase: str,
        group_keys: tuple[str, ...] | None = None,
    ) -> None:
        if self.anchor is None or self.last_target is None:
            return
        groups = {}
        keys = tuple(GROUP_NAMES) if group_keys is None else group_keys
        for key in keys:
            names = GROUP_NAMES[key]
            command_delta = self.last_target[key] - self.anchor.groups[key]
            feedback_delta = current.groups[key] - self.anchor.groups[key]
            raw_policy = (
                self.anchor.groups[key]
                if self.last_policy_raw is None
                else self.last_policy_raw[key]
            )
            sign_consistent = (command_delta * feedback_delta >= 0.0)
            command_index = int(np.argmax(np.abs(command_delta)))
            response_index = int(np.argmax(np.abs(feedback_delta)))
            groups[key] = {
                "joints": [
                    {
                        "index": index,
                        "name": name,
                        "motor_cmd_index": (
                            ARM_MOTOR_INDICES[key][index] if key in ARM_MOTOR_INDICES else None
                        ),
                        "motor_state_index": (
                            ARM_MOTOR_INDICES[key][index] if key in ARM_MOTOR_INDICES else None
                        ),
                        "initial_q": float(self.anchor.groups[key][index]),
                        "raw_policy_target": float(raw_policy[index]),
                        "final_command_q": float(self.last_target[key][index]),
                        "command_delta": float(command_delta[index]),
                        "feedback_delta": float(feedback_delta[index]),
                        "sign_consistent": bool(sign_consistent[index]),
                        "sign_evaluable": bool(
                            abs(command_delta[index]) > (1e-5 if key.endswith("arm") else 0.01)
                            and abs(feedback_delta[index]) > (1e-4 if key.endswith("arm") else 0.05)
                        ),
                    }
                    for index, name in enumerate(names)
                ],
                "max_command_joint_index": command_index,
                "max_command_joint_name": names[command_index],
                "max_response_joint_index": response_index,
                "max_response_joint_name": names[response_index],
                "max_response_matches_max_command": response_index == command_index,
            }
        self.logger.write("joint_response", state=self.state.value, phase=phase, groups=groups)

    def _check_direction_and_speed(self, current: FeedbackSnapshot) -> None:
        if self.anchor is None or self.last_feedback is None or self.last_target is None:
            return
        dt = max((current.monotonic_ns - self.last_feedback.monotonic_ns) / 1e9, 1e-3)
        for key in self.anchor.groups:
            feedback_delta = current.groups[key] - self.last_feedback.groups[key]
            speed_limit = 0.24 if key.endswith("arm") else 30.0
            if np.any(np.abs(feedback_delta) / dt > speed_limit):
                raise RuntimeError(f"{key} feedback speed exceeded conservative runtime threshold")
            command_from_anchor = self.last_target[key] - self.anchor.groups[key]
            feedback_from_anchor = current.groups[key] - self.anchor.groups[key]
            cmd_min = 0.004 if key.endswith("arm") else 0.75
            feedback_min = 0.003 if key.endswith("arm") else 0.5
            opposite = (
                (np.abs(command_from_anchor) > cmd_min)
                & (np.abs(feedback_from_anchor) > feedback_min)
                & (command_from_anchor * feedback_from_anchor < 0)
            )
            if np.any(opposite):
                raise RuntimeError(f"{key} feedback moved clearly opposite to command")

    def _wait_for_deadline(self, deadline: float) -> tuple[float, float]:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        actual = time.monotonic()
        lateness = max(0.0, actual - deadline)
        if lateness >= self.scheduler_max_lateness_s:
            raise RuntimeError(
                f"arm scheduler severe deadline miss: {lateness * 1000.0:.3f} ms"
            )
        return actual, lateness

    def _scheduled_arm_tick(
        self,
        *,
        target: dict[str, np.ndarray],
        weight: float,
        phase: str,
        tick: int,
        deadline: float,
        previous_actual: float | None,
    ) -> float:
        prior = self._last_arm_publish_actual if previous_actual is None else previous_actual
        effective_deadline = (
            deadline
            if prior is None
            else max(deadline, prior + 1.0 / self.arm_publish_hz)
        )
        actual, effective_lateness = self._wait_for_deadline(effective_deadline)
        lateness = max(0.0, actual - deadline)
        if lateness >= self.scheduler_max_lateness_s:
            raise RuntimeError(
                f"arm scheduler cumulative deadline miss: {lateness * 1000.0:.3f} ms"
            )
        interval = None if previous_actual is None else actual - previous_actual
        schedule = {
            "phase": phase,
            "tick": tick,
            "planned_deadline_monotonic_ns": int(deadline * 1e9),
            "effective_deadline_monotonic_ns": int(effective_deadline * 1e9),
            "actual_publish_start_monotonic_ns": int(actual * 1e9),
            "lateness_ms": lateness * 1000.0,
            "effective_lateness_ms": effective_lateness * 1000.0,
            "interval_ms": None if interval is None else interval * 1000.0,
            "period_ms": 1000.0 / self.arm_publish_hz,
        }
        self._send_arms(target, weight, schedule=schedule)
        self._last_arm_publish_actual = actual
        self._check_feedback(hold=self.state is BridgeState.ARMED_HOLD)
        return actual

    def execute_hold(self, full_weight_duration_s: float = 2.0) -> None:
        if self.state is not BridgeState.ARMED_HOLD or self.last_target is None:
            raise RuntimeError("hold is not armed")
        if full_weight_duration_s <= 0:
            raise ValueError("full-weight hold duration must be positive")
        period = 1.0 / self.arm_publish_hz
        start = time.monotonic()
        ramp_ticks = max(1, int(round(self.activation_ramp_s * self.arm_publish_hz)))
        full_ticks = max(1, int(round(full_weight_duration_s * self.arm_publish_hz)))
        previous_actual = None
        for tick in range(ramp_ticks + full_ticks + 1):
            weight = min(1.0, tick / ramp_ticks)
            previous_actual = self._scheduled_arm_tick(
                target=self.last_target,
                weight=weight,
                phase="activation" if tick <= ramp_ticks else "full_weight_hold",
                tick=tick,
                deadline=start + tick * period,
                previous_actual=previous_actual,
            )
        self.logger.write(
            "hold_complete",
            activation_ramp_s=self.activation_ramp_s,
            full_weight_hold_s=full_weight_duration_s,
            final_weight=self.current_weight,
            command_count=ramp_ticks + full_ticks + 1,
        )

    def execute_micro(
        self,
        plan: dict[str, Any],
        *,
        publish_left_o6: bool = True,
    ) -> None:
        if self.state is not BridgeState.ARMED_HOLD:
            raise RuntimeError("micro-motion requires a passed current-position hold")
        if publish_left_o6 and not self.o6_position_commands_enabled:
            raise PermissionError("model micro-motion requires explicitly enabled left O6 commands")
        frames = plan.get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 15:
            raise ValueError("model micro plan must contain 1..15 frames")
        self.state = BridgeState.ARMED_MICRO
        self.logger.write("state", state=self.state.value, ownership="arm_sdk weight=1")
        period = 1.0 / self.arm_publish_hz
        start = time.monotonic()
        last_control_index = -1
        previous_actual = None
        self._set_o6_command_active(publish_left_o6)
        try:
            for arm_tick in range(int(round(0.5 * self.arm_publish_hz))):
                elapsed = arm_tick * period
                control_index = min(int(elapsed * 30.0), len(frames) - 1)
                frame = frames[control_index]
                self.last_policy_raw = validate_groups(
                    frame["policy_raw_absolute"], label="policy_raw_absolute"
                )
                safety_target = validate_groups(
                    frame["ordinary_safety"], label="ordinary_safety"
                )
                self.last_target = self.envelope.step(
                    safety_target, dt=1.0 / self.arm_publish_hz
                )
                if control_index != last_control_index:
                    self._track_right_zero_clamp(frame, self.last_target)
                    last_control_index = control_index
                generation = (
                    self._queue_o6_target(self.last_target)
                    if publish_left_o6
                    else None
                )
                self.logger.write(
                    "policy_bridge_trace",
                    state=self.state.value,
                    inference_index=frame["inference_index"],
                    policy_raw=frame["policy_raw_absolute"],
                    adapter_target=frame["adapter_absolute"],
                    ordinary_safety=frame["ordinary_safety"],
                    micro_envelope=self.last_target,
                    o6_target_generation=generation,
                    right_o6_feedback_only=True,
                    right_o6_command_count=0,
                    left_o6_feedback_only=not publish_left_o6,
                    left_o6_command_count=(
                        self.o6_position_command_count if publish_left_o6 else 0
                    ),
                )
                previous_actual = self._scheduled_arm_tick(
                    target=self.last_target,
                    weight=1.0,
                    phase="model_micro",
                    tick=arm_tick,
                    deadline=start + arm_tick * period,
                    previous_actual=previous_actual,
                )
        finally:
            self._set_o6_command_active(False)

    def _track_right_zero_clamp(self, frame: dict[str, Any], target: dict[str, np.ndarray]) -> None:
        if self.anchor is None:
            return
        raw = np.asarray(frame["policy_raw_absolute"]["right_o6"], dtype=np.float64)
        clamped = target["right_o6"]
        if np.all(raw <= 1e-6) and np.any(clamped > raw + 1e-6):
            self._right_zero_clamp_inferences.add(int(frame["inference_index"]))
        if len(self._right_zero_clamp_inferences) > 5:
            raise RuntimeError("right O6 persistently requested zero beyond the one-shot window")

    def stop(self, reason: str, *, fault: bool = False) -> str | None:
        if self.state is BridgeState.STOPPED:
            return self.release_error
        self.stop_reason = reason
        self._set_o6_command_active(False)
        if fault:
            self.state = BridgeState.FAULT
            self.logger.write("watchdog", state=self.state.value, reason=reason)
        release_error = None
        try:
            if self.last_target is not None and self.current_weight > 0.0:
                release_start = self.g1.feedback()
                release_start_arms = {
                    key: release_start.groups[key].copy()
                    for key in ("left_arm", "right_arm")
                }
                start_weight = self.current_weight
                steps = max(1, int(round(self.release_ramp_s * self.arm_publish_hz)))
                period = 1.0 / self.arm_publish_hz
                release_deadline_origin = time.monotonic()
                previous_actual = release_deadline_origin
                for index in range(1, steps + 1):
                    planned_deadline = release_deadline_origin + index * period
                    # Never burst to catch up during release. A delayed release extends safely.
                    deadline = max(planned_deadline, previous_actual + period)
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    actual = time.monotonic()
                    lateness = max(0.0, actual - planned_deadline)
                    interval = actual - previous_actual
                    if lateness >= self.scheduler_max_lateness_s and release_error is None:
                        release_error = (
                            "release scheduler severe deadline miss: "
                            f"{lateness * 1000.0:.3f} ms"
                        )
                        self.logger.write(
                            "watchdog",
                            state=self.state.value,
                            reason=release_error,
                        )
                    weight = max(0.0, start_weight * (1.0 - index / steps))
                    self._send_arms(
                        self.last_target,
                        weight,
                        schedule={
                            "phase": "release",
                            "tick": index,
                            "planned_deadline_monotonic_ns": int(planned_deadline * 1e9),
                            "effective_deadline_monotonic_ns": int(deadline * 1e9),
                            "actual_publish_start_monotonic_ns": int(actual * 1e9),
                            "lateness_ms": lateness * 1000.0,
                            "interval_ms": interval * 1000.0,
                            "period_ms": period * 1000.0,
                        },
                    )
                    feedback, o6_status = self._read_release_feedback()
                    rebound = {
                        key: feedback.groups[key] - release_start_arms[key]
                        for key in ("left_arm", "right_arm")
                    }
                    self.logger.write(
                        "release_feedback",
                        state=self.state.value,
                        weight=weight,
                        feedback=feedback.as_dict(),
                        rebound_from_release_start=rebound,
                        max_allowed_rebound_rad=self.release_max_arm_rebound_rad,
                        o6_monitor_status=o6_status,
                    )
                    self._log_joint_response(
                        feedback,
                        phase="release",
                        group_keys=("left_arm", "right_arm"),
                    )
                    if any(
                        np.any(np.abs(value) > self.release_max_arm_rebound_rad)
                        for value in rebound.values()
                    ):
                        release_error = (
                            "arm feedback rebound exceeded "
                            f"{self.release_max_arm_rebound_rad:.6f} rad during release"
                        )
                        self.logger.write(
                            "watchdog",
                            state=self.state.value,
                            reason=release_error,
                        )
                    previous_actual = actual
                release_end = time.monotonic()
                monitor_steps = max(1, int(round(self.post_release_monitor_s * self.arm_publish_hz)))
                for index in range(1, monitor_steps + 1):
                    deadline = release_end + index * period
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    feedback, o6_status = self._read_release_feedback()
                    rebound = {
                        key: feedback.groups[key] - release_start_arms[key]
                        for key in ("left_arm", "right_arm")
                    }
                    self.logger.write(
                        "release_feedback",
                        state=self.state.value,
                        phase="weight_zero_post_release_monitor",
                        monitor_tick=index,
                        weight=0.0,
                        command_publication_active=False,
                        feedback=feedback.as_dict(),
                        rebound_from_release_start=rebound,
                        max_allowed_rebound_rad=self.release_max_arm_rebound_rad,
                        o6_monitor_status=o6_status,
                    )
                    self._log_joint_response(
                        feedback,
                        phase="weight_zero_post_release_monitor",
                        group_keys=("left_arm", "right_arm"),
                    )
                    if any(
                        np.any(np.abs(value) > self.release_max_arm_rebound_rad)
                        for value in rebound.values()
                    ):
                        release_error = (
                            "arm feedback rebound exceeded "
                            f"{self.release_max_arm_rebound_rad:.6f} rad after weight reached zero"
                        )
                        self.logger.write(
                            "watchdog",
                            state=self.state.value,
                            reason=release_error,
                        )
        except Exception as exc:
            release_error = f"{type(exc).__name__}: {exc}"
            self.logger.write("watchdog", state=self.state.value, reason="G1 release failed", detail=release_error)
        finally:
            self._o6_stop.set()
            try:
                self.o6.close()
            finally:
                if self._o6_thread is not None:
                    self._o6_thread.join(timeout=2.0)
                self.g1.close()
        self.state = BridgeState.STOPPED
        self.release_error = release_error
        self.logger.write(
            "state",
            state=self.state.value,
            stop_reason=reason,
            o6_stop="publication stopped; no zero/hold command; CAN interface left up",
            g1_stop="last safe q retained while arm_sdk weight ramped to zero",
            release_error=release_error,
            o6_position_command_count=self.o6_position_command_count,
            waist_leg_command_count=self.waist_leg_command_count,
        )
        return release_error
