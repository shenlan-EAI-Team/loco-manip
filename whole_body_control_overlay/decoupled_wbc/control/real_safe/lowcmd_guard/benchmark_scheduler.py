#!/usr/bin/env python3
"""Wall-clock benchmark for the guard scheduler; contains no DDS code."""

from __future__ import annotations

import argparse
import json
import time

from .scheduler import NoCatchUpScheduler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", type=float, default=500.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--block-every", type=int, default=0)
    parser.add_argument("--block-ms", type=float, default=0.0)
    args = parser.parse_args()
    if args.duration <= 0 or args.block_every < 0 or args.block_ms < 0:
        raise SystemExit("duration must be positive and block parameters nonnegative")

    callback_count = 0
    errors = []

    def callback() -> None:
        nonlocal callback_count
        callback_count += 1
        if args.block_every and callback_count % args.block_every == 0:
            time.sleep(args.block_ms / 1000.0)

    scheduler = NoCatchUpScheduler(args.frequency, callback, on_error=errors.append)
    started = time.monotonic()
    scheduler.start()
    time.sleep(args.duration)
    scheduler.stop()
    elapsed = time.monotonic() - started
    summary = scheduler.metrics.summary()
    summary.update(
        {
            "requested_frequency_hz": args.frequency,
            "duration_s": elapsed,
            "callback_count": callback_count,
            "effective_frequency_hz": callback_count / elapsed,
            "block_every": args.block_every,
            "block_ms": args.block_ms,
            "errors": [str(error) for error in errors],
            "dds_objects_created": 0,
            "command_writes": 0,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
