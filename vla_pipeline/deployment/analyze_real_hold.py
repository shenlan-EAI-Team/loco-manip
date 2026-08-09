from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ARM_GROUPS = ("left_arm", "right_arm")
ARM_NAMES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
ARM_INDICES = {"left_arm": tuple(range(15, 22)), "right_arm": tuple(range(22, 29))}


def _percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
        "min": float(data.min()),
    }


def analyze(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    anchor = next(row["feedback"] for row in rows if row["event"] == "arming_feedback")
    commands = [
        row for row in rows if row["event"] == "command" and row.get("transport") == "g1_arm_sdk"
    ]
    hold_feedback = [row["feedback"] for row in rows if row["event"] == "feedback"]
    release_rows = [
        row
        for row in rows
        if row["event"] == "release_feedback" and row.get("phase") is None
    ]
    post_rows = [
        row
        for row in rows
        if row["event"] == "release_feedback"
        and row.get("phase") == "weight_zero_post_release_monitor"
    ]
    all_feedback = hold_feedback + [row["feedback"] for row in release_rows + post_rows]
    activation_feedback = hold_feedback[:51]

    per_joint = []
    for group in ARM_GROUPS:
        initial = np.asarray(anchor["groups"][group], dtype=np.float64)
        command = np.asarray(
            [
                commands[0]["actual_command"]["motor_cmd_serialized"][index]["q"]
                for index in ARM_INDICES[group]
            ],
            dtype=np.float64,
        )
        all_values = np.asarray([row["groups"][group] for row in all_feedback])
        activation_values = np.asarray([row["groups"][group] for row in activation_feedback])
        release_rebounds = np.asarray(
            [row["rebound_from_release_start"][group] for row in release_rows]
        )
        post_rebounds = np.asarray(
            [row["rebound_from_release_start"][group] for row in post_rows]
        )
        for index, name in enumerate(ARM_NAMES):
            command_delta = float(command[index] - initial[index])
            maximum = float(np.max(np.abs(all_values[:, index] - initial[index])))
            takeover = float(
                np.max(np.abs(activation_values[:, index] - initial[index]))
            )
            release = float(np.max(np.abs(release_rebounds[:, index])))
            post = float(np.max(np.abs(post_rebounds[:, index])))
            per_joint.append(
                {
                    "group": group,
                    "joint_index": index,
                    "joint_name": name,
                    "motor_cmd_index": ARM_INDICES[group][index],
                    "motor_state_index": ARM_INDICES[group][index],
                    "initial_q_rad": float(initial[index]),
                    "command_q_rad": float(command[index]),
                    "command_delta_rad": command_delta,
                    "max_feedback_offset_rad": maximum,
                    "takeover_max_offset_rad": takeover,
                    "release_max_rebound_rad": release,
                    "post_release_max_rebound_rad": post,
                    "sign_evaluable": False,
                    "sign_consistent": None,
                    "erroneous_response": maximum > 0.01,
                }
            )

    schedules = [row["scheduler"] for row in commands]
    phases = {
        phase: [row for row in commands if row["scheduler"]["phase"] == phase]
        for phase in ("activation", "full_weight_hold", "release")
    }

    def publish_time(row: dict[str, Any]) -> float:
        return row["scheduler"]["actual_publish_start_monotonic_ns"] / 1e9

    activation_start = publish_time(phases["activation"][0])
    reached_one = publish_time(phases["activation"][-1])
    full_weight_end = publish_time(phases["full_weight_hold"][-1])
    release_end = publish_time(phases["release"][-1])
    intervals = [float(row["interval_ms"]) for row in schedules if row["interval_ms"] is not None]
    modes = {
        (feedback["g1_mode_machine"], feedback["g1_mode_pr"])
        for feedback in all_feedback
    }
    first_serialized = commands[0]["actual_command"]["motor_cmd_serialized"]
    forbidden_indices = (*range(15), *range(30, 35))
    forbidden_zero = all(
        all(
            command["actual_command"]["motor_cmd_serialized"][index][field] == 0
            for field in ("mode", "q", "dq", "tau", "kp", "kd", "reserve")
        )
        for command in commands
        for index in forbidden_indices
    )
    weight_aux_zero = all(
        all(
            command["actual_command"]["motor_cmd_serialized"][29][field] == 0
            for field in ("mode", "dq", "tau", "kp", "kd", "reserve")
        )
        for command in commands
    )
    waist_initial = np.asarray(anchor["waist"], dtype=np.float64)
    waist_values = np.asarray([feedback["waist"] for feedback in all_feedback])
    o6_max_delta = {
        group: float(
            np.max(
                np.abs(
                    np.asarray([feedback["groups"][group] for feedback in all_feedback])
                    - np.asarray(anchor["groups"][group])
                )
            )
        )
        for group in ("left_o6", "right_o6")
    }
    watchdogs = [row for row in rows if row["event"] == "watchdog"]
    stopped = [row for row in rows if row["event"] == "state" and row["state"] == "STOPPED"][-1]
    checks = {
        "weight_reached_exactly_one": phases["activation"][-1]["ownership"]["arm_sdk_weight"] == 1.0,
        "full_weight_held_at_least_2s": full_weight_end - reached_one >= 2.0,
        "weight_released_exactly_zero": phases["release"][-1]["ownership"]["arm_sdk_weight"] == 0.0,
        "activation_approximately_1s": abs((reached_one - activation_start) - 1.0) <= 0.04,
        "release_approximately_2s": abs((release_end - full_weight_end) - 2.0) <= 0.06,
        "no_interval_below_20ms_tolerance": min(intervals) >= 19.9,
        "max_interval_below_40ms": max(intervals) < 40.0,
        "all_arm_offsets_below_0p01rad": all(
            row["max_feedback_offset_rad"] < 0.01 for row in per_joint
        ),
        "mode_stable_5_0": modes == {(5, 0)},
        "forbidden_slots_explicit_zero": forbidden_zero,
        "weight_aux_fields_zero": weight_aux_zero,
        "waist_leg_command_count_zero": stopped["waist_leg_command_count"] == 0,
        "o6_position_command_count_zero": stopped["o6_position_command_count"] == 0,
        "watchdog_and_fault_count_zero": len(watchdogs) == 0,
        "release_error_none": stopped["release_error"] is None,
        "post_release_samples_25": len(post_rows) == 25,
    }
    return {
        "schema_version": 2,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model_micro_motion_executed": False,
        "model_micro_motion_allowed_now": False,
        "model_micro_motion_blocker": (
            "left O6 position publication is still synchronous in execute_micro; "
            "decouple and wall-clock validate it before a new plan/confirmation"
        ),
        "log": str(path),
        "plan_sha256": next(row["plan_sha256"] for row in rows if row["event"] == "gates"),
        "mode": {"observed": sorted([list(item) for item in modes]), "stable": len(modes) == 1},
        "counts": {
            "arm_sdk_messages": len(commands),
            "activation_messages": len(phases["activation"]),
            "full_weight_messages_after_boundary": len(phases["full_weight_hold"]),
            "release_messages": len(phases["release"]),
            "post_release_read_only_samples": len(post_rows),
            "o6_position_commands": stopped["o6_position_command_count"],
            "waist_leg_commands": stopped["waist_leg_command_count"],
            "watchdogs_faults": len(watchdogs),
        },
        "timing": {
            "activation_0_to_1_s": reached_one - activation_start,
            "full_weight_1_to_last_1_s": full_weight_end - reached_one,
            "release_last_1_to_0_s": release_end - full_weight_end,
            "arm_interval_ms": _percentiles(intervals),
            "scheduler_lateness_ms": _percentiles(
                [float(row["lateness_ms"]) for row in schedules]
            ),
        },
        "maxima": {
            "arm_feedback_offset_rad": max(row["max_feedback_offset_rad"] for row in per_joint),
            "takeover_offset_rad": max(row["takeover_max_offset_rad"] for row in per_joint),
            "release_rebound_rad": max(row["release_max_rebound_rad"] for row in per_joint),
            "post_release_rebound_rad": max(
                row["post_release_max_rebound_rad"] for row in per_joint
            ),
            "waist_feedback_change_rad": float(
                np.max(np.abs(waist_values - waist_initial))
            ),
            "o6_feedback_change_points": o6_max_delta,
            "o6_cached_age_ms": max(
                float(row.get("o6_cached_age_ms", 0.0))
                for row in rows
                if row["event"] in ("feedback", "release_feedback")
            ),
        },
        "serialization": {
            "forbidden_indices": list(forbidden_indices),
            "forbidden_slots_explicit_zero_all_messages": forbidden_zero,
            "weight_index": 29,
            "weight_aux_fields_zero_all_messages": weight_aux_zero,
            "first_message_mode_machine": commands[0]["actual_command"]["mode_machine"],
            "first_message_mode_pr": commands[0]["actual_command"]["mode_pr"],
            "first_message_crc": commands[0]["actual_command"]["crc"],
            "first_message_slot_count": len(first_serialized),
        },
        "per_joint": per_joint,
        "faults": watchdogs,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = analyze(Path(args.log))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
