from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
BRIDGE = ROOT / "real_bridge"
CONFIG = ROOT / "config/micro_motion.yaml"
JSON_CONFIG = ROOT / "config/micro_motion.json"


def called_attributes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def run_audit() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    json_config = json.loads(JSON_CONFIG.read_text(encoding="utf-8"))
    remote = BRIDGE / "remote_o6_agent.py"
    remote_calls = called_attributes(remote)
    forbidden_o6_calls = {
        "set_speed",
        "set_torque",
        "set_enable",
        "reset",
        "home",
        "homing",
        "calibrate",
        "close_can",
    }
    all_text = "\n".join(path.read_text(encoding="utf-8") for path in BRIDGE.glob("*.py"))
    cli_text = (BRIDGE / "cli.py").read_text(encoding="utf-8")
    g1_text = (BRIDGE / "real_g1.py").read_text(encoding="utf-8")
    controller_text = (BRIDGE / "controller.py").read_text(encoding="utf-8")
    execute_micro_text = controller_text.split("def execute_micro", 1)[1].split(
        "def _track_right_zero_clamp", 1
    )[0]
    o6_worker_text = controller_text.split("def _o6_monitor_loop", 1)[1].split(
        "def _set_o6_command_active", 1
    )[0]
    stop_text = controller_text.split("def stop", 1)[1]
    preview_text = (BRIDGE / "message_preview.py").read_text(encoding="utf-8")
    plan_text = (BRIDGE / "preview.py").read_text(encoding="utf-8")
    checks = {
        "default_hardware_transport_disabled": config["hardware_transport_enabled"] is False,
        "default_command_publication_disabled": config["command_publication_enabled"] is False,
        "default_micro_motion_disarmed": config["micro_motion_armed"] is False,
        "runtime_json_exactly_matches_audited_yaml": json_config == config,
        "g1_uses_dedicated_arm_sdk_topic": '"rt/arm_sdk"' in all_text,
        "no_whole_body_lowcmd_topic": "rt/lowcmd" not in all_text.lower(),
        "no_motion_switcher_client": "MotionSwitcherClient(" not in all_text
        and "comm.motion_switcher" not in all_text,
        "remote_o6_has_no_forbidden_calls": not bool(remote_calls & forbidden_o6_calls),
        "remote_o6_only_position_setter": "try_set_joint_positions" in remote_calls,
        "remote_o6_does_not_create_command_socket": "socket" not in remote_calls,
        "gates_checked_before_token": cli_text.index("gates.require_all()")
        < cli_text.index("OneTimeToken.consume("),
        "token_consumed_before_real_factory": cli_text.index("OneTimeToken.consume(")
        < cli_text.index("session = _create_real_session("),
        "g1_forbidden_indices_declared": config["g1"]["forbidden_indices"] == list(range(15)),
        "g1_official_arm_and_weight_indices": config["g1"]["left_arm_indices"] == list(range(15, 22))
        and config["g1"]["right_arm_indices"] == list(range(22, 29))
        and config["g1"]["weight_index"] == 29,
        "g1_exact_35_slot_preview": "MOTOR_COUNT = 35" in preview_text
        and "motor_cmd_serialized" in g1_text,
        "g1_mode_fields_match_arming_feedback": "message.mode_pr = int(mode_pr)" in g1_text
        and "message.mode_machine = int(mode_machine)" in g1_text
        and "live.g1_mode_machine != int(mode_machine)" in g1_text,
        "hold_o6_transport_has_no_setter": "O6FeedbackOnlySubprocessTransport" in cli_text
        and "o6_position_commands_enabled=not hold_only" in cli_text,
        "first_arm_excursion_tightened_to_0p01": config["micro_motion"]["arm_max_excursion_rad"] == 0.01,
        "release_rebound_limit_0p01": config["micro_motion"]["release_max_arm_rebound_rad"] == 0.01,
        "weight_zero_post_release_monitor_enabled": config["micro_motion"]["post_release_monitor_s"] >= 0.5
        and "weight_zero_post_release_monitor" in controller_text,
        "per_joint_response_fields_present": all(
            field in controller_text
            for field in (
                "command_delta",
                "feedback_delta",
                "sign_consistent",
                "max_response_matches_max_command",
                "motor_cmd_index",
                "motor_state_index",
                "initial_q",
                "raw_policy_target",
                "final_command_q",
            )
        ),
        "o6_left_can2_right_can1": config["o6"]["left_can"] == "can2"
        and config["o6"]["right_can"] == "can1",
        "remote_o6_agent_hash_pinned": len(config["network"]["o6_remote_agent_sha256"]) == 64,
        "no_zero_stop": config["stop"]["sends_zero_posture"] is False,
        "no_mode_change": config["stop"]["changes_motion_mode"] is False,
        "micro_arm_loop_has_no_o6_transport_io": "self.o6." not in execute_micro_text
        and "_send_left_hand(" not in execute_micro_text,
        "o6_worker_owns_getter_and_setter": "self.o6.feedback()" in o6_worker_text
        and "_send_left_hand(" in o6_worker_text,
        "release_does_not_require_o6_feedback": "_read_release_feedback()" in stop_text
        and "_read_cached_feedback()" not in stop_text,
        "right_o6_has_no_command_method": "send_right_hand" not in controller_text,
        "arms_only_uses_feedback_only_o6_transport": (
            'feedback_only_o6 = args.phase in ("hold-only", "arms-only-micro")'
            in cli_text
        ),
        "arms_only_plan_requires_zero_hand_and_waist_leg_commands": all(
            item in plan_text
            for item in (
                '"left_o6_command_count": 0',
                '"right_o6_command_count": 0',
                '"waist_leg_command_count": 0',
                '"automatic_repeat": False',
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "remote_o6_called_attributes": sorted(remote_calls),
        "forbidden_o6_calls": sorted(forbidden_o6_calls),
    }


def main() -> int:
    result = run_audit()
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
