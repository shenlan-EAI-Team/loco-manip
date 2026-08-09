#!/usr/bin/env python3
"""Read-only DCPSPublication audit for the G1 rt/lowcmd topic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decoupled_wbc.control.real_safe.lowcmd_guard.dds_exclusivity import (
    LOWCMD_TOPIC,
    LowcmdExclusivityPolicy,
    create_monitor_for_interface,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--discovery", type=float, default=5.0)
    parser.add_argument("--stable", type=float, default=1.0)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    monitor = create_monitor_for_interface(args.interface)
    publications = monitor.observe_stable(
        LOWCMD_TOPIC,
        discovery_s=args.discovery,
        stable_s=args.stable,
    )
    policy = LowcmdExclusivityPolicy()
    result = "PASS"
    reason = None
    try:
        policy.capture_ai_baseline(publications)
    except Exception as exc:
        result = "FAIL"
        reason = f"{type(exc).__name__}: {exc}"

    summary = {
        "schema_version": 1,
        "read_only": True,
        "dds_business_writers_created": 0,
        "lowcmd_writes": 0,
        "release_calls": 0,
        "select_calls": 0,
        "topic": LOWCMD_TOPIC,
        "result": result,
        "reason": reason,
        "publications": [value.to_mapping() for value in sorted(publications)],
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(payload + "\n")
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
