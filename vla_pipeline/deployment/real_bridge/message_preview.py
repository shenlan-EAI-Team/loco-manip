from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .mapping import percentages_to_raw
from .models import validate_groups


LEFT_ARM_INDICES = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM_INDICES = (22, 23, 24, 25, 26, 27, 28)
WEIGHT_INDEX = 29
FORBIDDEN_INDICES = tuple(range(15))
MOTOR_COUNT = 35


def _crc32_core(words: list[int]) -> int:
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for current in words:
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if current & bit:
                crc ^= polynomial
            bit >>= 1
    return crc


def _preview_crc(mode_pr: int, mode_machine: int, motors: list[dict[str, Any]]) -> int:
    values: list[Any] = [mode_pr, mode_machine]
    for motor in motors:
        values.extend(
            [
                motor["mode"],
                motor["q"],
                motor["dq"],
                motor["tau"],
                motor["kp"],
                motor["kd"],
                motor["reserve"],
            ]
        )
    values.extend([0, 0, 0, 0, 0])
    payload = struct.pack("<2B2x" + "B3x5fI" * MOTOR_COUNT + "5I", *values)
    words = [
        int.from_bytes(payload[offset : offset + 4], "little")
        for offset in range(0, len(payload) - 4, 4)
    ]
    return _crc32_core(words)


def preview_g1_arm_message(
    left: Any,
    right: Any,
    *,
    weight: float,
    kp: float = 60.0,
    kd: float = 1.5,
    mode_machine: int = 0,
    mode_pr: int = 0,
) -> dict[str, Any]:
    groups = validate_groups(
        {
            "left_arm": left,
            "right_arm": right,
            "left_o6": np.zeros(6),
            "right_o6": np.zeros(6),
        },
        label="g1_message",
    )
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("arm_sdk weight must be finite and in [0, 1]")
    if not 0 <= int(mode_machine) <= 255 or not 0 <= int(mode_pr) <= 255:
        raise ValueError("mode fields must be uint8 values")
    serialized = [
        {"mode": 0, "q": 0.0, "dq": 0.0, "tau": 0.0, "kp": 0.0, "kd": 0.0, "reserve": 0}
        for _ in range(MOTOR_COUNT)
    ]
    for index, value in zip(LEFT_ARM_INDICES, groups["left_arm"]):
        serialized[index].update({"q": float(value), "kp": kp, "kd": kd})
    for index, value in zip(RIGHT_ARM_INDICES, groups["right_arm"]):
        serialized[index].update({"q": float(value), "kp": kp, "kd": kd})
    serialized[WEIGHT_INDEX]["q"] = float(weight)
    crc = _preview_crc(int(mode_pr), int(mode_machine), serialized)
    return {
        "dds_topic": "rt/arm_sdk",
        "message_type": "unitree_hg.msg.dds_.LowCmd_",
        "mode_pr": int(mode_pr),
        "mode_machine": int(mode_machine),
        "motor_cmd_serialized": serialized,
        "low_cmd_reserve": [0, 0, 0, 0],
        "crc": crc,
        "crc_algorithm": "Unitree CRC32 over packed HG LowCmd excluding final crc word",
    }


def preview_o6_messages(left_percent: Any, right_percent: Any) -> dict[str, Any]:
    groups = validate_groups(
        {
            "left_arm": np.zeros(7),
            "right_arm": np.zeros(7),
            "left_o6": left_percent,
            "right_o6": right_percent,
        },
        label="o6_message",
    )
    return {
        "left": {
            "can": "can2",
            "arbitration_id": "0x28",
            "data": [0x01, *percentages_to_raw(groups["left_o6"])],
        },
        "right": {
            "can": "can1",
            "feedback_only": True,
            "command": None,
            "command_count": 0,
            "feedback_preview_0_100": groups["right_o6"].tolist(),
        },
        "position_call": "LinkerHandO6Can.try_set_joint_positions",
    }
