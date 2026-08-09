from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .message_preview import (
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    WEIGHT_INDEX,
    _preview_crc,
)
from .models import FeedbackSnapshot


class G1ArmSdkTransport:
    """Dedicated upper-body DDS transport; construction is itself safety-gated."""

    def __init__(self, interface: str, *, feedback_timeout_s: float = 2.0) -> None:
        # Keep every Unitree import inside the constructor. Preview/tests never import SDK code.
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        self._message_factory = unitree_hg_msg_dds__LowCmd_
        self._lock = threading.Lock()
        self._latest: Any = None
        self._latest_monotonic_ns = 0
        self._closed = False

        ChannelFactoryInitialize(0, interface)
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_state, 10)
        deadline = time.monotonic() + feedback_timeout_s
        while self._latest is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._latest is None:
            raise TimeoutError("no rt/lowstate received before G1 arm transport startup deadline")

    def _on_state(self, message: Any) -> None:
        with self._lock:
            self._latest = message
            self._latest_monotonic_ns = time.monotonic_ns()

    def feedback(self) -> FeedbackSnapshot:
        with self._lock:
            message = self._latest
            monotonic_ns = self._latest_monotonic_ns
        if message is None:
            raise TimeoutError("G1 feedback unavailable")
        age_ms = (time.monotonic_ns() - monotonic_ns) / 1e6
        if age_ms > 100.0:
            raise TimeoutError(f"G1 feedback stale: {age_ms:.1f} ms")
        groups = {
            "left_arm": [message.motor_state[index].q for index in LEFT_ARM_INDICES],
            "right_arm": [message.motor_state[index].q for index in RIGHT_ARM_INDICES],
            # O6 values are filled by the composite bridge before validation.
            "left_o6": np.zeros(6),
            "right_o6": np.zeros(6),
        }
        return FeedbackSnapshot.create(
            groups,
            monotonic_ns=monotonic_ns,
            g1_mode_machine=int(message.mode_machine),
            g1_mode_pr=int(message.mode_pr),
            waist=[message.motor_state[index].q for index in (12, 13, 14)],
        )

    def send_arms(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        weight: float,
        mode_machine: int,
        mode_pr: int,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("G1 transport is closed")
        left = np.asarray(left, dtype=np.float64).reshape(-1)
        right = np.asarray(right, dtype=np.float64).reshape(-1)
        if left.shape != (7,) or right.shape != (7,):
            raise ValueError("G1 arm command must be left=(7,), right=(7,)")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("G1 arm command contains NaN or Inf")
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("G1 arm_sdk weight must be in [0, 1]")
        if not 0 <= int(mode_machine) <= 255 or not 0 <= int(mode_pr) <= 255:
            raise ValueError("G1 mode fields must be uint8 values")
        live = self.feedback()
        if live.g1_mode_machine != int(mode_machine) or live.g1_mode_pr != int(mode_pr):
            raise RuntimeError(
                "G1 mode changed before arm_sdk publish: "
                f"live=({live.g1_mode_machine},{live.g1_mode_pr}) "
                f"armed=({mode_machine},{mode_pr})"
            )

        message = self._message_factory()
        message.mode_pr = int(mode_pr)
        message.mode_machine = int(mode_machine)
        for index, value in zip(LEFT_ARM_INDICES, left):
            motor = message.motor_cmd[index]
            motor.mode = 0
            motor.q = float(value)
            motor.dq = 0.0
            motor.kp = 60.0
            motor.kd = 1.5
            motor.tau = 0.0
            motor.reserve = 0
        for index, value in zip(RIGHT_ARM_INDICES, right):
            motor = message.motor_cmd[index]
            motor.mode = 0
            motor.q = float(value)
            motor.dq = 0.0
            motor.kp = 60.0
            motor.kd = 1.5
            motor.tau = 0.0
            motor.reserve = 0
        message.motor_cmd[WEIGHT_INDEX].q = float(weight)
        serialized = [
            {
                "mode": int(motor.mode),
                "q": float(motor.q),
                "dq": float(motor.dq),
                "tau": float(motor.tau),
                "kp": float(motor.kp),
                "kd": float(motor.kd),
                "reserve": int(motor.reserve),
            }
            for motor in message.motor_cmd
        ]
        message.crc = _preview_crc(int(message.mode_pr), int(message.mode_machine), serialized)
        self._publisher.Write(message)
        return {
            "topic": "rt/arm_sdk",
            "mode_pr": int(message.mode_pr),
            "mode_machine": int(message.mode_machine),
            "motor_cmd_serialized": [
                {"index": index, **motor} for index, motor in enumerate(serialized)
            ],
            "low_cmd_reserve": [int(value) for value in message.reserve],
            "crc": int(message.crc),
        }

    def close(self) -> None:
        # Releasing arm_sdk weight is an explicit controller operation, never implicit here.
        self._closed = True
