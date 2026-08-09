"""Unitree SDK backend. Importing this module does not create DDS writers."""

from __future__ import annotations

from threading import Lock
import time

import numpy as np

from .core import GuardCommand, GuardSnapshot
from ..standalone import RobotSnapshot, SafetyFault


class UnitreeGuardStateSource:
    def __init__(self) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._lock = Lock()
        self._lowstate = None
        self._secondary_imu = None
        self._lowstate_time = None
        self._imu_time = None
        self._crc = CRC()
        self.crc_errors = 0
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.secondary_imu_subscriber = ChannelSubscriber("rt/secondary_imu", IMUState_)
        self.lowstate_subscriber.Init(self._on_lowstate, 1)
        self.secondary_imu_subscriber.Init(self._on_secondary_imu, 1)

    def _on_lowstate(self, message) -> None:
        if int(message.crc) != int(self._crc.Crc(message)):
            self.crc_errors += 1
            return
        with self._lock:
            self._lowstate = message
            self._lowstate_time = time.monotonic()

    def _on_secondary_imu(self, message) -> None:
        with self._lock:
            self._secondary_imu = message
            self._imu_time = time.monotonic()

    def latest(self, now: float) -> GuardSnapshot:
        del now
        with self._lock:
            lowstate = self._lowstate
            imu = self._secondary_imu
            lowstate_time = self._lowstate_time
            imu_time = self._imu_time
        if lowstate is None or imu is None or lowstate_time is None or imu_time is None:
            raise SafetyFault("lowstate/secondary IMU is not ready")
        states = lowstate.motor_state
        return GuardSnapshot(
            robot=RobotSnapshot(
                q=np.asarray([states[i].q for i in range(29)], dtype=np.float64),
                dq=np.asarray([states[i].dq for i in range(29)], dtype=np.float64),
                base_quat_wxyz=np.asarray(lowstate.imu_state.quaternion, dtype=np.float64),
                base_angular_velocity=np.asarray(lowstate.imu_state.gyroscope, dtype=np.float64),
                secondary_quat_wxyz=np.asarray(imu.quaternion, dtype=np.float64),
                secondary_angular_velocity=np.asarray(imu.gyroscope, dtype=np.float64),
                lowstate_monotonic=float(lowstate_time),
                imu_monotonic=float(imu_time),
            ),
            mode_machine=int(lowstate.mode_machine),
            motor_modes=np.asarray([states[i].mode for i in range(29)], dtype=np.int64),
            motor_errors=np.asarray([states[i].motorstate for i in range(29)], dtype=np.int64),
            motor_tau_est=np.asarray([states[i].tau_est for i in range(29)], dtype=np.float64),
        )


class UnitreeMotionModeClient:
    def __init__(self) -> None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )

        self._client = MotionSwitcherClient()
        self._client.SetTimeout(3.0)
        self._client.Init()

    def check_mode(self) -> tuple[int, str, str]:
        status, result = self._client.CheckMode()
        if status != 0 or result is None:
            return int(status), "", ""
        return int(status), str(result.get("form", "")), str(result.get("name", ""))

    def release_mode(self) -> int:
        status, _ = self._client.ReleaseMode()
        return int(status)

    def select_mode(self, owner: str) -> int:
        status, _ = self._client.SelectMode(owner)
        return int(status)


class UnitreeLowCmdWriter:
    """The only class in real_safe permitted to construct rt/lowcmd."""

    def __init__(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self._message = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self.write_count = 0
        self.closed = False

    def write(self, command: GuardCommand) -> None:
        if self.closed:
            raise RuntimeError("LowCmd writer is closed")
        self._message.mode_pr = int(command.mode_pr)
        self._message.mode_machine = int(command.mode_machine)
        for index in range(29):
            motor = self._message.motor_cmd[index]
            motor.mode = int(command.motor_mode[index])
            motor.q = float(command.q[index])
            motor.dq = float(command.dq[index])
            motor.kp = float(command.kp[index])
            motor.kd = float(command.kd[index])
            motor.tau = float(command.tau[index])
        # The hg IDL contains six non-G1 slots. Serialize them explicitly as
        # disabled zeros instead of relying on allocator defaults.
        for index in range(29, len(self._message.motor_cmd)):
            motor = self._message.motor_cmd[index]
            motor.mode = 0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        self._message.crc = self._crc.Crc(self._message)
        self._publisher.Write(self._message)
        self.write_count += 1

    def close(self) -> None:
        self._publisher.Close()
        self.closed = True
