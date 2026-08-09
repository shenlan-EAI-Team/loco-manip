from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any

import yaml

from deployment.safety_filter import SafetyFilter

from .envelope import MicroMotionEnvelope
from .message_preview import preview_g1_arm_message, preview_o6_messages
from .models import FeedbackSnapshot, validate_groups


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    required = (
        "model.safetensors.index.json",
        "config.json",
        "processor_config.json",
        "statistics.json",
    )
    files = {name: sha256_file(root / name) for name in required}
    manifest = hashlib.sha256(
        "".join(f"{name}:{files[name]}\n" for name in sorted(files)).encode("ascii")
    ).hexdigest()
    return {"path": str(root), "manifest_sha256": manifest, "files": files}


def _load_feedback(path: Path) -> FeedbackSnapshot:
    value = json.loads(path.read_text(encoding="utf-8"))
    if "groups" in value:
        return FeedbackSnapshot.create(
            value["groups"],
            monotonic_ns=value.get("monotonic_ns"),
            wall_ns=value.get("wall_ns"),
            g1_mode_machine=value.get("g1_mode_machine"),
            g1_mode_pr=value.get("g1_mode_pr"),
            waist=value.get("waist"),
        )
    if "feedback" in value:
        return FeedbackSnapshot.create(value["feedback"])
    return FeedbackSnapshot.create(value)


def build_plan_from_live_log(
    feedback_path: str | Path,
    live_jsonl_path: str | Path,
    adapter_config_path: str | Path,
    micro_config_path: str | Path = "deployment/config/micro_motion.yaml",
    checkpoint_path: str | Path = "outputs/formal_train_26_corrected_v1/checkpoint-3000",
    *,
    inference_count: int = 5,
) -> dict[str, Any]:
    feedback_path = Path(feedback_path)
    live_path = Path(live_jsonl_path)
    feedback = _load_feedback(feedback_path)
    rows = [json.loads(line) for line in live_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < inference_count:
        raise ValueError(f"need {inference_count} real policy records, found {len(rows)}")
    selected = rows[-inference_count:]

    adapter_config = yaml.safe_load(Path(adapter_config_path).read_text(encoding="utf-8"))
    micro_config = yaml.safe_load(Path(micro_config_path).read_text(encoding="utf-8"))
    ordinary = SafetyFilter(adapter_config)
    ordinary.reset(feedback.groups)
    frames = []
    dt = 1.0 / 30.0
    for row in selected:
        chunk = row["raw_action_chunk"]
        for chunk_index in range(3):
            policy_raw = {key: chunk[key][chunk_index] for key in feedback.groups}
            adapter_target = validate_groups(policy_raw, label="adapter_target")
            ordinary_safe = ordinary.filter_step(adapter_target, dt=dt)
            frames.append(
                {
                    "inference_index": int(row["inference_index"]),
                    "chunk_index": chunk_index,
                    "policy_raw_absolute": {
                        key: value.tolist() for key, value in adapter_target.items()
                    },
                    "adapter_absolute": {
                        key: value.tolist() for key, value in adapter_target.items()
                    },
                    "ordinary_safety": {
                        key: value.tolist() for key, value in ordinary_safe.items()
                    },
                }
            )
    if len(frames) != 15:
        raise AssertionError(f"0.5 second plan must contain 15 frames, got {len(frames)}")
    current = feedback.groups
    limits = micro_config["micro_motion"]
    g1_config = micro_config["g1"]
    micro = MicroMotionEnvelope(
        arm_excursion_rad=float(limits["arm_max_excursion_rad"]),
        arm_velocity_rad_s=float(limits["arm_max_velocity_rad_s"]),
        arm_acceleration_rad_s2=float(limits["arm_max_acceleration_rad_s2"]),
        o6_excursion_points=float(limits["o6_max_excursion_points"]),
        o6_velocity_points_s=float(limits["o6_max_velocity_points_s"]),
    )
    micro.reset(current)
    arm_ticks = []
    arm_targets = []
    for tick in range(25):
        elapsed = tick / 50.0
        control_index = min(int(elapsed * 30.0), len(frames) - 1)
        frame = frames[control_index]
        final_target = micro.step(frame["ordinary_safety"], dt=1.0 / 50.0)
        arm_targets.append({key: value.copy() for key, value in final_target.items()})
        arm_ticks.append(
            {
                "tick": tick,
                "elapsed_s": elapsed,
                "control_frame_index": control_index,
                "inference_index": frame["inference_index"],
                "micro_envelope": {key: value.tolist() for key, value in final_target.items()},
                "g1_message": preview_g1_arm_message(
                    final_target["left_arm"],
                    final_target["right_arm"],
                    weight=1.0,
                    mode_machine=feedback.g1_mode_machine or 0,
                    mode_pr=feedback.g1_mode_pr or 0,
                ),
            }
        )
    o6_ticks = []
    for tick in range(15):
        elapsed = tick / 30.0
        arm_tick = min(int(elapsed * 50.0), len(arm_targets) - 1)
        target = arm_targets[arm_tick]
        o6_ticks.append(
            {
                "tick": tick,
                "elapsed_s": elapsed,
                "source_arm_tick": arm_tick,
                "micro_envelope": {
                    key: target[key].tolist() for key in ("left_o6", "right_o6")
                },
                "o6_messages": preview_o6_messages(target["left_o6"], target["right_o6"]),
            }
        )
    arm_publish_hz = float(g1_config["sdk_publish_hz"])
    full_weight_hold_ticks = int(
        round(float(limits["full_weight_hold_s"]) * arm_publish_hz)
    )
    activation_ticks = int(round(float(g1_config["activation_ramp_s"]) * arm_publish_hz))
    release_ticks = int(round(float(g1_config["release_ramp_s"]) * arm_publish_hz))
    post_release_monitor_ticks = int(
        round(float(limits["post_release_monitor_s"]) * arm_publish_hz)
    )
    return {
        "schema_version": 1,
        "mode": "one_shot_micro_motion",
        "duration_s": 0.5,
        "policy_hz": 10.0,
        "control_timeline_hz": 30.0,
        "source": {
            "checkpoint": checkpoint_identity(checkpoint_path),
            "feedback_path": str(feedback_path),
            "feedback_sha256": sha256_file(feedback_path),
            "live_jsonl_path": str(live_path),
            "live_jsonl_sha256": sha256_file(live_path),
            "selected_inference_indices": [int(row["inference_index"]) for row in selected],
        },
        "arming_feedback_preview": feedback.as_dict(),
        "hold_preview": {
            "activation_ramp_s": float(g1_config["activation_ramp_s"]),
            "full_weight_hold_s": float(limits["full_weight_hold_s"]),
            "absolute_target": {key: value.tolist() for key, value in current.items()},
            "g1_first_message_weight_0": preview_g1_arm_message(
                current["left_arm"],
                current["right_arm"],
                weight=0.0,
                mode_machine=feedback.g1_mode_machine or 0,
                mode_pr=feedback.g1_mode_pr or 0,
            ),
            "g1_full_weight_message": preview_g1_arm_message(
                current["left_arm"],
                current["right_arm"],
                weight=1.0,
                mode_machine=feedback.g1_mode_machine or 0,
                mode_pr=feedback.g1_mode_pr or 0,
            ),
            "o6_messages": preview_o6_messages(current["left_o6"], current["right_o6"]),
            "right_o6_command_count": 0,
            "arm_sdk_weight_curve": {
                "publish_hz": arm_publish_hz,
                "activation_publish_messages": activation_ticks + 1,
                "activation_intervals": activation_ticks,
                "activation_values": [tick / activation_ticks for tick in range(activation_ticks + 1)],
                "full_weight_hold_intervals": full_weight_hold_ticks,
                "full_weight_hold_messages_including_boundary": full_weight_hold_ticks + 1,
                "release_ticks": release_ticks,
                "release_values": [
                    1.0 - tick / release_ticks for tick in range(1, release_ticks + 1)
                ],
                "post_release_monitor_ticks": post_release_monitor_ticks,
                "post_release_monitor_value": 0.0,
                "post_release_monitor_publishes": 0,
            },
        },
        "frames": frames,
        "runtime_preview": {
            "g1_arm_sdk_50hz_ticks": arm_ticks,
            "o6_can_30hz_ticks": o6_ticks,
        },
        "scheduler_contract": {
            "arm_publish_hz": arm_publish_hz,
            "arm_loop_direct_o6_getter_calls": 0,
            "arm_loop_direct_o6_setter_calls": 0,
            "o6_io_owner": "o6-independent-io-worker",
            "o6_target_queue": "latest-only",
            "catch_up_bursts_allowed": False,
            "o6_failure_blocks_arm_release": False,
            "release_feedback_dependency": "G1-only; O6 cache is optional metadata",
        },
        "right_o6_feedback_only": True,
        "right_o6_command_count": 0,
        "confirmation_token": None,
        "real_commands_sent": 0,
        "ordinary_safety_counters": {
            key: value.as_dict() for key, value in ordinary.counters.items()
        },
        "micro_envelope_counters": {
            "excursion": micro.counters.excursion,
            "velocity": micro.counters.velocity,
            "acceleration": micro.counters.acceleration,
        },
    }


def build_hold_only_plan(micro_plan: dict[str, Any]) -> dict[str, Any]:
    hold_preview = copy.deepcopy(micro_plan["hold_preview"])
    hold_preview.pop("o6_messages", None)
    feedback = micro_plan["arming_feedback_preview"]["groups"]
    hold_preview["o6_feedback_only"] = {
        "left": {"feedback_0_100": feedback["left_o6"], "position_command": None},
        "right": {"feedback_0_100": feedback["right_o6"], "position_command": None},
        "position_command_count": 0,
    }
    return {
        "schema_version": 2,
        "mode": "hold_only_waiting_for_user_confirmation",
        "source": micro_plan["source"],
        "arming_feedback_preview": micro_plan["arming_feedback_preview"],
        "hold_preview": hold_preview,
        "right_o6_feedback_only": True,
        "right_o6_command_count": 0,
        "real_commands_sent": 0,
        "real_sdk_command_objects_created": 0,
        "confirmation_token": None,
    }


def build_arms_only_micro_plan(micro_plan: dict[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(micro_plan)
    feedback = plan["arming_feedback_preview"]["groups"]
    for frame in plan["frames"]:
        frame["suppressed_o6_policy_target"] = {
            key: copy.deepcopy(frame["policy_raw_absolute"][key])
            for key in ("left_o6", "right_o6")
        }
        for stage in ("adapter_absolute", "ordinary_safety"):
            for key in ("left_o6", "right_o6"):
                frame[stage][key] = copy.deepcopy(feedback[key])
    for tick in plan["runtime_preview"]["g1_arm_sdk_50hz_ticks"]:
        for key in ("left_o6", "right_o6"):
            tick["micro_envelope"][key] = copy.deepcopy(feedback[key])
    plan["runtime_preview"]["o6_can_30hz_ticks"] = [
        {
            "tick": tick,
            "elapsed_s": tick / 30.0,
            "left_o6": {"feedback_only": True, "position_command": None},
            "right_o6": {"feedback_only": True, "position_command": None},
            "position_command_count": 0,
        }
        for tick in range(15)
    ]
    for counter in plan["micro_envelope_counters"].values():
        counter["left_o6"] = 0
        counter["right_o6"] = 0
    plan.update(
        {
            "schema_version": 3,
            "mode": "arms_only_micro_motion_waiting_for_user_confirmation",
            "duration_s": 0.5,
            "left_o6_feedback_only": True,
            "left_o6_command_count": 0,
            "right_o6_feedback_only": True,
            "right_o6_command_count": 0,
            "waist_leg_command_count": 0,
            "single_window_only": True,
            "automatic_repeat": False,
            "outer_startup_envelope_rad": 0.03,
            "effective_first_run_arm_envelope_rad": 0.01,
            "confirmation_token": None,
            "real_commands_sent": 0,
        }
    )
    plan["scheduler_contract"].update(
        {
            "o6_io_mode": "bilateral-feedback-only",
            "left_o6_position_command_count": 0,
            "right_o6_position_command_count": 0,
        }
    )
    return plan


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
