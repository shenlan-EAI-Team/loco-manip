#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import msgpack
import numpy as np
import zmq


PREFIX = "O6_FEEDBACK_ONLY_JSON "


def side(value: object, expected: str) -> list[float]:
    if not isinstance(value, dict) or value.get("side", expected) != expected:
        raise ValueError(f"invalid {expected} O6 object")
    actual = np.asarray(value.get("actual_q"), dtype=np.float64)
    if actual.shape != (6,) or not np.isfinite(actual).all():
        raise ValueError(f"invalid {expected} O6 feedback shape/value")
    if np.any(actual < 0) or np.any(actual > 100):
        raise ValueError(f"{expected} O6 feedback outside 0..100")
    if not bool(value.get("feedback_valid", value.get("valid", False))):
        raise ValueError(f"{expected} O6 feedback is invalid")
    age = float(value.get("feedback_age_ms", value.get("age_ms", -1)))
    if not np.isfinite(age) or not 0 <= age <= 100.0:
        raise ValueError(f"{expected} O6 feedback is stale")
    return actual.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.endpoint)
    try:
        while True:
            raw = socket.recv()
            message = msgpack.unpackb(raw, raw=False)
            if int(message.get("schema_version", 0)) != 2:
                continue
            output = {
                "left_o6": side(message.get("left"), "left"),
                "right_o6": side(message.get("right"), "right"),
                "receive_monotonic_ns": time.monotonic_ns(),
            }
            print(PREFIX + json.dumps(output, separators=(",", ":")), flush=True)
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
