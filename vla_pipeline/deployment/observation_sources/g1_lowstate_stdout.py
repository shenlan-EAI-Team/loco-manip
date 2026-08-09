#!/usr/bin/env python3
"""Read-only G1 LowState DDS subscriber emitting JSON lines on stdout.

This process deliberately imports neither ChannelPublisher nor any motion/control
client. It is intended to run in the existing Python 3.8 Unitree SDK environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


LEFT_ARM = (15, 16, 17, 18, 19, 20, 21)
RIGHT_ARM = (22, 23, 24, 25, 26, 27, 28)
WAIST = (12, 13, 14)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--topic", default="rt/lowstate")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)

    def callback(message: LowState_) -> None:
        receive_wall_ns = time.time_ns()
        receive_monotonic_ns = time.monotonic_ns()
        motors = message.motor_state
        record = {
            "schema": "g1_lowstate_readonly_v1",
            "topic": args.topic,
            "tick": int(message.tick),
            "receive_wall_ns": receive_wall_ns,
            "receive_monotonic_ns": receive_monotonic_ns,
            "left_arm": [float(motors[index].q) for index in LEFT_ARM],
            "right_arm": [float(motors[index].q) for index in RIGHT_ARM],
            "waist": [float(motors[index].q) for index in WAIST],
            "base_quat_wxyz": [float(value) for value in message.imu_state.quaternion],
            "mode_machine": int(message.mode_machine),
            "mode_pr": int(message.mode_pr),
        }
        sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    subscriber = ChannelSubscriber(args.topic, LowState_)
    subscriber.Init(callback, 1)
    print(
        json.dumps(
            {
                "event": "reader_started",
                "topic": args.topic,
                "interface": args.interface,
                "publisher_created": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
