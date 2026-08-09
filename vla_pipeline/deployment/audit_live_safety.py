#!/usr/bin/env python3
"""Static safety gate for the executable Live Shadow code path."""

from __future__ import annotations

import ast
from pathlib import Path

from deployment.common import PROJECT_ROOT, json_dump


ROOT = PROJECT_ROOT / "deployment"
FILES = [
    ROOT / "run_live_shadow.py",
    ROOT / "null_sink.py",
    ROOT / "observation_sources/g1_live.py",
    ROOT / "observation_sources/g1_lowstate_stdout.py",
]
BANNED_SYMBOLS = {
    "ChannelPublisher",
    "MotionSwitcherClient",
    "LowCmd_",
    "unitree_hg_msg_dds__LowCmd_",
    "LinkerHandApi",
    "SportClient",
}
BANNED_METHODS = {
    "send",
    "write",
    "set_target",
    "set_joint_positions",
    "finger_move",
    "set_speed",
    "set_torque",
    "ReleaseMode",
    "request_control",
}


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def main() -> None:
    findings = []
    inventory = []
    for path in FILES:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                inventory.append({"file": str(path), "line": node.lineno, "imports": names})
                for name in names:
                    if any(symbol in name for symbol in BANNED_SYMBOLS):
                        findings.append(
                            {"file": str(path), "line": node.lineno, "kind": "banned_import", "value": name}
                        )
            elif isinstance(node, ast.Call):
                name = dotted(node.func)
                leaf = name.rsplit(".", 1)[-1]
                # File/stdout logging is not a hardware call.
                logging_write = leaf == "write" and (
                    name.endswith("stdout.write") or name in {"log.write"}
                )
                if leaf in BANNED_METHODS and not logging_write:
                    findings.append(
                        {"file": str(path), "line": node.lineno, "kind": "banned_call", "value": name}
                    )
            elif isinstance(node, ast.Attribute) and node.attr in {"PUSH", "PUB"}:
                findings.append(
                    {
                        "file": str(path),
                        "line": node.lineno,
                        "kind": "command_capable_zmq_socket",
                        "value": dotted(node),
                    }
                )

    live_text = (ROOT / "run_live_shadow.py").read_text()
    assertions = {
        "runner_never_calls_adapter_drain_to_mock": ".drain_to_mock(" not in live_text,
        "runner_never_calls_emergency_stop": ".emergency_stop(" not in live_text,
        "runner_uses_null_sink": "NullActionSink" in live_text,
        "lowstate_subscriber_only": "ChannelSubscriber" in FILES[-1].read_text(),
        "no_static_banned_findings": not findings,
    }
    result = {
        "gate_passed": all(assertions.values()),
        "files": [str(path) for path in FILES],
        "assertions": assertions,
        "findings": findings,
        "runtime_required_counters": {
            "command_publish_attempts": 0,
            "control_ownership_requests": 0,
            "real_sdk_objects_created": 0,
        },
    }
    json_dump(ROOT / "live_safety_audit.json", result)
    if not result["gate_passed"]:
        raise SystemExit("Live Shadow static safety gate failed")
    print(ROOT / "live_safety_audit.json")


if __name__ == "__main__":
    main()
