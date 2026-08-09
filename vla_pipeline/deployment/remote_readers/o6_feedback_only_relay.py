#!/usr/bin/env python3
"""G1-side dual O6 feedback-only ZMQ relay for Live Shadow.

There is deliberately no command socket and no position/speed/torque method
call. The SDK's status getter emits only the O6 0x01 read request. CAN must
already be configured and UP; this process refuses to change link state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time
from typing import Any

import msgpack
import zmq

from LinkerHand.linker_hand_api import LinkerHandApi


JOINT_ORDER = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


def require_can_already_up(name: str) -> None:
    operstate = Path(f"/sys/class/net/{name}/operstate")
    if not operstate.is_file() or operstate.read_text().strip() != "up":
        raise RuntimeError(
            f"{name} is not already UP; feedback relay refuses to configure or modify CAN"
        )


def close_reader(api: Any) -> None:
    hand = getattr(api, "hand", None)
    close = getattr(hand, "close_can_interface", None)
    if callable(close):
        close()


def percent_from_raw(raw: Any) -> list[float]:
    values = [int(value) for value in raw]
    if len(values) != 6 or any(value < 0 or value > 255 for value in values):
        raise ValueError(f"invalid O6 raw feedback: {values}")
    return [value * 100.0 / 255.0 for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-can", default="can2")
    parser.add_argument("--right-can", default="can1")
    parser.add_argument("--left-sn", default="WG1JA06260615023")
    parser.add_argument("--right-sn", default="WG1KA06260618541")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--state-port", type=int, default=5558)
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args()
    if args.left_can == args.right_can:
        raise SystemExit("left and right CAN must differ")
    require_can_already_up(args.left_can)
    require_can_already_up(args.right_can)

    # Constructor and get_state() issue version/SN/status read requests only.
    # No setter is referenced anywhere in this process.
    left = LinkerHandApi(hand_type="left", hand_joint="O6", can=args.left_can)
    right = LinkerHandApi(hand_type="right", hand_joint="O6", can=args.right_can)
    if str(left.get_serial_number()) != args.left_sn:
        raise RuntimeError("left O6 serial mismatch")
    if str(right.get_serial_number()) != args.right_sn:
        raise RuntimeError("right O6 serial mismatch")

    context = zmq.Context()
    state = context.socket(zmq.PUB)
    state.setsockopt(zmq.LINGER, 0)
    state.setsockopt(zmq.SNDHWM, 1)
    state.setsockopt(zmq.CONFLATE, 1)
    state.bind(f"tcp://{args.bind_host}:{args.state_port}")
    running = True

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print("O6 FEEDBACK-ONLY RELAY", flush=True)
    print("NO O6 CONTROL COMMANDS WILL BE SENT", flush=True)
    print("NO COMMAND SOCKET EXISTS", flush=True)
    print(json.dumps({"joint_order": JOINT_ORDER, "native_range": [0, 255]}), flush=True)

    period = 1.0 / args.rate
    sequence = 0
    last_valid: dict[str, list[float] | None] = {"left": None, "right": None}
    try:
        while running:
            tick = time.monotonic()
            side_data = {}
            for side, api in (("left", left), ("right", right)):
                start_ns = time.monotonic_ns()
                valid = True
                error = None
                try:
                    actual = percent_from_raw(api.get_state())
                    last_valid[side] = actual
                except Exception as exc:
                    valid = False
                    error = f"{type(exc).__name__}: {exc}"
                    actual = last_valid[side]
                if actual is None:
                    side_data = {}
                    break
                side_data[side] = {
                    "side": side,
                    "actual_q": actual,
                    "feedback_valid": valid,
                    "feedback_age_ms": (time.monotonic_ns() - start_ns) / 1e6,
                    "timestamp_ns": time.monotonic_ns(),
                    "error": error,
                }
            if len(side_data) == 2:
                sequence += 1
                now_ns = time.monotonic_ns()
                message = {
                    "schema_version": 2,
                    "feedback_only": True,
                    "timestamp_ns": now_ns,
                    "sequence": sequence,
                    "left": side_data["left"],
                    "right": side_data["right"],
                    "control_command_calls": 0,
                    "command_socket_created": False,
                }
                try:
                    state.send(msgpack.packb(message, use_bin_type=True), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        state.close(linger=0)
        context.term()
        close_reader(left)
        close_reader(right)


if __name__ == "__main__":
    main()
