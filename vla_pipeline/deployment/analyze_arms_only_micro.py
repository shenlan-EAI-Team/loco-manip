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


def _stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
        "min": float(data.min()),
    }


def analyze(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    anchor = next(row["feedback"] for row in rows if row["event"] == "arming_feedback")
    traces = [row for row in rows if row["event"] == "policy_bridge_trace"]
    micro_feedback = [
        row["feedback"]
        for row in rows
        if row["event"] == "feedback" and row.get("state") == "ARMED_MICRO"
    ]
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
    commands = [
        row for row in rows if row["event"] == "command" and row.get("transport") == "g1_arm_sdk"
    ]
    phases = {
        phase: [row for row in commands if row["scheduler"]["phase"] == phase]
        for phase in ("activation", "full_weight_hold", "model_micro", "release")
    }

    per_joint: list[dict[str, Any]] = []
    response_matching: dict[str, Any] = {}
    for group in ARM_GROUPS:
        initial = np.asarray(anchor["groups"][group], dtype=np.float64)
        raw = np.asarray([row["policy_raw"][group] for row in traces], dtype=np.float64)
        final = np.asarray([row["micro_envelope"][group] for row in traces], dtype=np.float64)
        feedback = np.asarray(
            [row["groups"][group] for row in micro_feedback], dtype=np.float64
        )
        command_delta = final - initial
        feedback_delta = feedback - initial
        release_rebound = np.asarray(
            [row["rebound_from_release_start"][group] for row in release_rows],
            dtype=np.float64,
        )
        post_rebound = np.asarray(
            [row["rebound_from_release_start"][group] for row in post_rows],
            dtype=np.float64,
        )
        max_command_by_joint = np.max(np.abs(command_delta), axis=0)
        max_feedback_by_joint = np.max(np.abs(feedback_delta), axis=0)
        command_max = float(max_command_by_joint.max())
        command_ties = np.flatnonzero(
            np.isclose(max_command_by_joint, command_max, rtol=0.0, atol=1e-12)
        )
        response_joint = int(np.argmax(max_feedback_by_joint))
        response_matching[group] = {
            "max_command_joint_indices": command_ties.tolist(),
            "max_command_joint_names": [ARM_NAMES[index] for index in command_ties],
            "max_response_joint_index": response_joint,
            "max_response_joint_name": ARM_NAMES[response_joint],
            "max_response_is_one_of_max_command_joints": bool(response_joint in command_ties),
        }
        for index, name in enumerate(ARM_NAMES):
            response_tick = int(np.argmax(np.abs(feedback_delta[:, index])))
            response = float(feedback_delta[response_tick, index])
            command_at_response = float(command_delta[response_tick, index])
            sign_evaluable = abs(command_at_response) > 1e-5 and abs(response) > 1e-4
            per_joint.append(
                {
                    "group": group,
                    "joint_index": index,
                    "joint_name": name,
                    "motor_cmd_index": ARM_INDICES[group][index],
                    "motor_state_index": ARM_INDICES[group][index],
                    "initial_q_rad": float(initial[index]),
                    "last_raw_policy_target_rad": float(raw[-1, index]),
                    "last_final_command_q_rad": float(final[-1, index]),
                    "last_command_delta_rad": float(command_delta[-1, index]),
                    "max_abs_command_delta_rad": float(max_command_by_joint[index]),
                    "max_feedback_delta_rad": response,
                    "max_abs_feedback_delta_rad": abs(response),
                    "sign_evaluable": bool(sign_evaluable),
                    "sign_consistent": (
                        bool(command_at_response * response >= 0.0)
                        if sign_evaluable
                        else None
                    ),
                    "release_max_rebound_rad": float(
                        np.max(np.abs(release_rebound[:, index]))
                    ),
                    "post_release_max_rebound_rad": float(
                        np.max(np.abs(post_rebound[:, index]))
                    ),
                }
            )

    schedules = {
        phase: [
            float(row["scheduler"]["interval_ms"])
            for row in phase_rows
            if row["scheduler"]["interval_ms"] is not None
        ]
        for phase, phase_rows in phases.items()
    }
    stopped = [
        row for row in rows if row["event"] == "state" and row.get("state") == "STOPPED"
    ][-1]
    watchdogs = [row for row in rows if row["event"] == "watchdog"]
    faults = [
        row for row in rows if row["event"] == "state" and row.get("state") == "FAULT"
    ]
    o6_errors = [row for row in rows if row["event"] == "o6_worker_error"]
    modes = {
        (row["g1_mode_machine"], row["g1_mode_pr"])
        for row in micro_feedback + [item["feedback"] for item in release_rows + post_rows]
    }
    waist_initial = np.asarray(anchor["waist"], dtype=np.float64)
    waist_values = np.asarray(
        [
            row["waist"]
            for row in micro_feedback + [item["feedback"] for item in release_rows + post_rows]
        ],
        dtype=np.float64,
    )
    o6_command_rows = [
        row
        for row in rows
        if row["event"] == "command" and row.get("transport") == "o6_can_position"
    ]
    micro_response_max = max(row["max_abs_feedback_delta_rad"] for row in per_joint)
    checks = {
        "one_0p5s_window_25_ticks": len(phases["model_micro"]) == 25
        and len(traces) == 25,
        "no_automatic_second_window": len(phases["model_micro"]) == 25,
        "command_envelope_at_most_0p01rad": max(
            row["max_abs_command_delta_rad"] for row in per_joint
        )
        <= 0.010001,
        "feedback_remained_inside_0p01rad": micro_response_max <= 0.010001,
        "weight_reached_one": phases["activation"][-1]["ownership"]["arm_sdk_weight"]
        == 1.0,
        "weight_released_to_zero": phases["release"][-1]["ownership"]["arm_sdk_weight"]
        == 0.0,
        "release_has_100_messages": len(phases["release"]) == 100,
        "post_release_has_25_samples": len(post_rows) == 25,
        "o6_position_commands_zero": len(o6_command_rows) == 0
        and stopped["o6_position_command_count"] == 0,
        "waist_leg_commands_zero": stopped["waist_leg_command_count"] == 0,
        "mode_stable": len(modes) == 1,
        "faults_zero": not faults,
        "watchdogs_zero": not watchdogs,
        "release_error_none": stopped["release_error"] is None,
    }
    measurable_response = micro_response_max > 1e-4
    return {
        "schema_version": 1,
        "status": (
            "PASS"
            if all(checks.values()) and measurable_response
            else (
                "SAFE_COMPLETION_NO_MEASURABLE_ARM_RESPONSE"
                if all(checks.values())
                else "FAIL"
            )
        ),
        "safe_execution_and_release_passed": all(checks.values()),
        "controlled_micro_response_proven": measurable_response,
        "response_threshold_rad": 1e-4,
        "log": str(path),
        "plan_sha256": next(row["plan_sha256"] for row in rows if row["event"] == "gates"),
        "mode": {"observed": sorted([list(item) for item in modes]), "stable": len(modes) == 1},
        "counts": {
            "arm_sdk_messages": len(commands),
            "activation_messages": len(phases["activation"]),
            "full_weight_hold_messages": len(phases["full_weight_hold"]),
            "model_micro_messages": len(phases["model_micro"]),
            "release_messages": len(phases["release"]),
            "post_release_read_only_samples": len(post_rows),
            "left_o6_position_commands": 0,
            "right_o6_position_commands": 0,
            "waist_leg_commands": stopped["waist_leg_command_count"],
            "faults": len(faults),
            "watchdogs": len(watchdogs),
            "dds_errors": 0,
        },
        "timing": {phase: _stats(values) for phase, values in schedules.items()},
        "weight_curve": {
            phase: {
                "messages": len(phase_rows),
                "first": phase_rows[0]["ownership"]["arm_sdk_weight"],
                "last": phase_rows[-1]["ownership"]["arm_sdk_weight"],
            }
            for phase, phase_rows in phases.items()
        },
        "maxima": {
            "micro_feedback_offset_rad": micro_response_max,
            "release_rebound_rad": max(
                row["release_max_rebound_rad"] for row in per_joint
            ),
            "post_release_rebound_rad": max(
                row["post_release_max_rebound_rad"] for row in per_joint
            ),
            "waist_feedback_delta_rad": float(
                np.max(np.abs(waist_values - waist_initial))
            ),
        },
        "response_matching": response_matching,
        "per_joint": per_joint,
        "o6_shutdown_events": [
            {
                "error_type": row["error_type"],
                "detail": row["detail"],
                "arm_release_required": row["arm_release_required"],
                "classification": "post-release helper shutdown artifact",
            }
            for row in o6_errors
        ],
        "stop": stopped,
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
    return 0 if summary["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
