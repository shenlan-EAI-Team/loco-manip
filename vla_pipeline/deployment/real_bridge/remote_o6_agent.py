#!/usr/bin/env python3
"""Remote stdin/stdout O6 transport. Starting this process is a real-command action."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


PREFIX = "REAL_BRIDGE_JSON "


def emit(value: dict[str, Any]) -> None:
    print(PREFIX + json.dumps(value, separators=(",", ":")), flush=True)


def validate_raw(values: Any) -> list[int]:
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("O6 target must contain six integers")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError("O6 target values must be integers in [0, 255]")
        result.append(value)
    return result


class Hand:
    def __init__(self, side: str, can_name: str, sdk_root: Path) -> None:
        self.side = side
        self.can_name = can_name
        self.lock = open(f"/tmp/g1_o6_real_bridge_{can_name}.lock", "a+", encoding="utf-8")
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.seek(0)
        self.lock.truncate()
        self.lock.write(f"pid={os.getpid()} side={side}\n")
        self.lock.flush()
        sys.path.insert(0, str(sdk_root / "LinkerHand"))
        from core.can.linker_hand_o6_can import LinkerHandO6Can

        can_id = 0x28 if side == "left" else 0x27
        self.hand = LinkerHandO6Can(can_id=can_id, can_channel=can_name, baudrate=1_000_000)

    def feedback(self, timeout_s: float = 0.1) -> list[float]:
        before = int(self.hand.x01_monotonic_ns)
        self.hand.try_request_current_status()
        deadline = time.monotonic() + timeout_s
        while int(self.hand.x01_monotonic_ns) <= before and time.monotonic() < deadline:
            time.sleep(0.001)
        if int(self.hand.x01_monotonic_ns) <= before:
            raise TimeoutError(f"{self.side} O6 feedback timeout")
        values = [float(value) for value in self.hand.get_current_pub_status()]
        if len(values) != 6 or not all(math.isfinite(value) and 0 <= value <= 255 for value in values):
            raise ValueError(f"invalid {self.side} O6 feedback: {values!r}")
        return values

    def send(self, values: Any) -> None:
        self.hand.try_set_joint_positions(validate_raw(values))

    def close(self) -> None:
        # LinkerHandApi.close_can() is intentionally forbidden because it sets canX down.
        self.hand.close_can_interface()
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
        self.lock.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--left-can", default="can2")
    parser.add_argument("--right-can", default="can1")
    parser.add_argument("--sdk-root", default="/home/unitree/linker_hand_python_sdk")
    args = parser.parse_args()
    if args.left_can != "can2" or args.right_can != "can1":
        raise SystemExit("first micro-motion mapping is fixed to left=can2, right=can1")

    hands: dict[str, Hand] = {}
    initialized = False
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request.get("operation")
                if operation == "init":
                    if initialized:
                        raise RuntimeError("agent already initialized")
                    if request.get("protocol") != "g1_o6_real_bridge_v1":
                        raise PermissionError("invalid bridge protocol marker")
                    hands = {
                        "left": Hand("left", args.left_can, Path(args.sdk_root)),
                        "right": Hand("right", args.right_can, Path(args.sdk_root)),
                    }
                    initialized = True
                    emit({"ok": True, "operation": "init", "pid": os.getpid()})
                elif not initialized:
                    raise PermissionError("init is required before any hardware operation")
                elif operation == "feedback":
                    emit(
                        {
                            "ok": True,
                            "operation": "feedback",
                            "left_raw_255": hands["left"].feedback(),
                            "right_raw_255": hands["right"].feedback(),
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                elif operation == "command_left":
                    hands["left"].send(request.get("left_raw_255"))
                    emit({"ok": True, "operation": "command_left", "monotonic_ns": time.monotonic_ns()})
                elif operation == "close":
                    emit({"ok": True, "operation": "close"})
                    break
                else:
                    raise ValueError(f"unsupported operation: {operation!r}")
            except Exception as exc:
                emit({"ok": False, "error": type(exc).__name__, "detail": str(exc)})
    finally:
        for hand in hands.values():
            try:
                hand.close()
            except Exception as exc:
                print(f"O6 close warning: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
