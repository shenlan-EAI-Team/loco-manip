#!/usr/bin/env python3
"""Prove the dedicated O6 relay references no control setter."""

from __future__ import annotations

import ast
from pathlib import Path

from deployment.common import PROJECT_ROOT, json_dump


def main() -> None:
    path = PROJECT_ROOT / "deployment/remote_readers/o6_feedback_only_relay.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.append({"line": node.lineno, "method": node.func.attr})
    banned = {
        "finger_move",
        "set_joint_positions",
        "set_speed",
        "set_joint_speed",
        "set_torque",
        "set_current",
        "set_target",
        "write",
    }
    findings = [item for item in calls if item["method"] in banned]
    text = path.read_text()
    assertions = {
        "no_control_setter_calls": not findings,
        "no_command_zmq_socket": "zmq.PUSH" not in text and "zmq.PULL" not in text,
        "state_pub_only": "zmq.PUB" in text,
        "can_must_already_be_up": "require_can_already_up" in text,
        "command_counter_literal_zero": '"control_command_calls": 0' in text,
    }
    result = {
        "gate_passed": all(assertions.values()),
        "file": str(path),
        "assertions": assertions,
        "findings": findings,
        "allowed_feedback_calls": [item for item in calls if item["method"].startswith("get_")],
    }
    json_dump(PROJECT_ROOT / "deployment/o6_feedback_reader_safety.json", result)
    if not result["gate_passed"]:
        raise SystemExit("O6 feedback reader safety gate failed")
    print(PROJECT_ROOT / "deployment/o6_feedback_reader_safety.json")


if __name__ == "__main__":
    main()
