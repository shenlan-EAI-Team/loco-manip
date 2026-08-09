#!/usr/bin/env python3
"""Read-only audit for the 30-episode G1 + O6 teleoperation dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FPS = 30.0
EXPECTED_DT = 1.0 / EXPECTED_FPS

ARM_JOINTS = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw",
]
O6_JOINTS = [
    "thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch",
    "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch",
]
WAIST_JOINTS = ["waist_yaw", "waist_roll", "waist_pitch"]


def stack_column(df: pd.DataFrame, key: str, dim: int) -> np.ndarray:
    if key not in df:
        raise KeyError(f"missing required column: {key}")
    arr = np.stack(df[key].to_numpy()).astype(np.float64, copy=False)
    if arr.shape != (len(df), dim):
        raise ValueError(f"{key}: expected {(len(df), dim)}, got {arr.shape}")
    return arr


def body29_from_full43(full: np.ndarray) -> np.ndarray:
    """Recover the 29-value hardware-ordered vector inserted into full q."""
    return np.concatenate([full[:, :22], full[:, 29:36]], axis=1)


def reconstruct_g1(df: pd.DataFrame) -> dict[str, np.ndarray]:
    # The C++ g1_debug publisher performs IsaacLab->MuJoCo reordering and
    # offset/scale conversion before run_data_exporter.py builds these 43-D
    # Pinocchio containers. Applying those transforms here again corrupts the
    # joint identity and units.
    feedback_abs_hw = body29_from_full43(stack_column(df, "observation.state", 43))
    command_abs_hw = body29_from_full43(stack_column(df, "action.wbc", 43))
    return {
        "feedback_abs_hw": feedback_abs_hw,
        "command_abs_hw": command_abs_hw,
        "instant_delta_hw": command_abs_hw - feedback_abs_hw,
        "command_published_abs_hw": command_abs_hw,
    }


def video_probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    stream = json.loads(result.stdout)["streams"][0]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    return {
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": num / den,
        "frames": int(stream["nb_read_frames"]),
    }


def scalar_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def vector_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "std": np.std(x, axis=0).tolist(),
        "range": np.ptp(x, axis=0).tolist(),
    }


def audit_episode(source: Path, episode_id: int) -> tuple[dict, dict[str, np.ndarray]]:
    parquet = source / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
    video = source / "videos/chunk-000/observation.images.ego_view" / f"episode_{episode_id:06d}.mp4"
    df = pd.read_parquet(parquet)
    g1 = reconstruct_g1(df)
    left_q = stack_column(df, "observation.left_hand_q", 6)
    right_q = stack_column(df, "observation.right_hand_q", 6)
    left_cmd = stack_column(df, "action.left_hand", 6)
    right_cmd = stack_column(df, "action.right_hand", 6)
    projected_gravity = stack_column(df, "observation.projected_gravity", 3)
    action76 = stack_column(df, "action", 76)
    motion64 = stack_column(df, "action.motion_token", 64)

    timestamp = df["timestamp"].to_numpy(dtype=np.float64)
    frame_index = df["frame_index"].to_numpy(dtype=np.int64)
    dt = np.diff(timestamp)
    video_info = video_probe(video)

    selected = np.concatenate(
        [g1["feedback_abs_hw"], g1["command_abs_hw"], left_q, right_q,
         left_cmd, right_cmd, projected_gravity], axis=1,
    )
    finite = bool(np.isfinite(selected).all())
    left_valid = df["left_hand_valid"].to_numpy(dtype=np.float64)
    right_valid = df["right_hand_valid"].to_numpy(dtype=np.float64)
    left_age = df["left_hand_age_ms"].to_numpy(dtype=np.float64)
    right_age = df["right_hand_age_ms"].to_numpy(dtype=np.float64)

    arm_fb = g1["feedback_abs_hw"][:, 15:29]
    arm_cmd = g1["command_abs_hw"][:, 15:29]
    waist = g1["feedback_abs_hw"][:, 12:15]
    arm_cmd_jump = np.abs(np.diff(arm_cmd, axis=0))
    arm_fb_jump = np.abs(np.diff(arm_fb, axis=0))
    waist_jump = np.abs(np.diff(waist, axis=0))
    left_jump = np.abs(np.diff(left_cmd, axis=0))
    right_jump = np.abs(np.diff(right_cmd, axis=0))

    canonical_action_match = bool(
        np.array_equal(action76[:, :64], motion64)
        and np.array_equal(action76[:, 64:70], left_cmd)
        and np.array_equal(action76[:, 70:76], right_cmd)
    )
    metrics = {
        "source_episode": episode_id,
        "frames": len(df),
        "duration_s": float(timestamp[-1] - timestamp[0]) if len(df) else 0.0,
        "timestamp_start": float(timestamp[0]),
        "timestamp_end": float(timestamp[-1]),
        "dt_median_s": float(np.median(dt)),
        "dt_max_abs_error_s": float(np.max(np.abs(dt - EXPECTED_DT))),
        "timestamp_monotonic": bool(np.all(dt > 0)),
        "frame_index_contiguous": bool(np.array_equal(frame_index, np.arange(len(df)))),
        "video": video_info,
        "video_frame_match": video_info["frames"] == len(df),
        "all_selected_values_finite": finite,
        "canonical_action76_matches_component_columns": canonical_action_match,
        "left_valid_rate": float(np.mean(left_valid > 0.5)),
        "right_valid_rate": float(np.mean(right_valid > 0.5)),
        "left_age_ms": scalar_stats(left_age),
        "right_age_ms": scalar_stats(right_age),
        "left_age_rate_gt_50ms": float(np.mean(left_age > 50.0)),
        "right_age_rate_gt_50ms": float(np.mean(right_age > 50.0)),
        "left_feedback_range": np.ptp(left_q, axis=0).tolist(),
        "right_feedback_range": np.ptp(right_q, axis=0).tolist(),
        "left_command_range": np.ptp(left_cmd, axis=0).tolist(),
        "right_command_range": np.ptp(right_cmd, axis=0).tolist(),
        "left_command_saturation_rate_le1_or_ge99": float(np.mean((left_cmd <= 1.0) | (left_cmd >= 99.0))),
        "right_command_saturation_rate_le1_or_ge99": float(np.mean((right_cmd <= 1.0) | (right_cmd >= 99.0))),
        "left_command_saturation_rate_by_joint": np.mean((left_cmd <= 1.0) | (left_cmd >= 99.0), axis=0).tolist(),
        "right_command_saturation_rate_by_joint": np.mean((right_cmd <= 1.0) | (right_cmd >= 99.0), axis=0).tolist(),
        "max_arm_command_jump_rad_per_frame": float(np.max(arm_cmd_jump)),
        "max_arm_feedback_jump_rad_per_frame": float(np.max(arm_fb_jump)),
        "max_waist_feedback_jump_rad_per_frame": float(np.max(waist_jump)),
        "max_left_o6_command_jump_pct_per_frame": float(np.max(left_jump)),
        "max_right_o6_command_jump_pct_per_frame": float(np.max(right_jump)),
        "arm_command_jump_count_gt_0_25_rad": int(np.sum(arm_cmd_jump > 0.25)),
        "arm_feedback_jump_count_gt_0_25_rad": int(np.sum(arm_fb_jump > 0.25)),
        "waist_jump_count_gt_0_15_rad": int(np.sum(waist_jump > 0.15)),
        "left_o6_jump_count_gt_30_pct": int(np.sum(left_jump > 30.0)),
        "right_o6_jump_count_gt_30_pct": int(np.sum(right_jump > 30.0)),
        "projected_gravity_norm": scalar_stats(np.linalg.norm(projected_gravity, axis=1)),
        "arm_abs_tracking_error_rad": scalar_stats(np.abs(arm_cmd - arm_fb)),
    }
    arrays = {
        "left_q": left_q, "right_q": right_q,
        "left_cmd": left_cmd, "right_cmd": right_cmd,
        "feedback_abs_hw": g1["feedback_abs_hw"],
        "command_abs_hw": g1["command_abs_hw"],
        "projected_gravity": projected_gravity,
    }
    return metrics, arrays


def audit(source: Path) -> dict:
    info = json.loads((source / "meta/info.json").read_text())
    episode_files = sorted((source / "data/chunk-000").glob("episode_*.parquet"))
    episode_ids = [int(p.stem.split("_")[-1]) for p in episode_files]
    episodes = []
    all_arrays: dict[str, list[np.ndarray]] = {}
    for episode_id in episode_ids:
        metrics, arrays = audit_episode(source, episode_id)
        episodes.append(metrics)
        for key, arr in arrays.items():
            all_arrays.setdefault(key, []).append(arr)
    merged = {key: np.concatenate(parts, axis=0) for key, parts in all_arrays.items()}

    left_valid_frames = sum(round(e["left_valid_rate"] * e["frames"]) for e in episodes)
    right_valid_frames = sum(round(e["right_valid_rate"] * e["frames"]) for e in episodes)
    total_frames = sum(e["frames"] for e in episodes)
    return {
        "source": str(source.resolve()),
        "source_codebase_version": info.get("codebase_version"),
        "episode_ids": episode_ids,
        "episode_count": len(episodes),
        "total_frames": total_frames,
        "declared_fps": info.get("fps"),
        "observed_transport_and_topics": {
            "camera": "ZMQ/TCP ComposedCameraClientSensor on port 5555; no topic prefix",
            "pico_smpl": "ZMQ SUB topics pose, planner, manager_state on port 5556",
            "g1_feedback_and_policy_command": "ZMQ SUB topic g1_debug on port 5557",
            "o6_record_stream": "MessagePack two-sided schema on port 5558; no ROS topic on this hop",
            "wuji_left_ros2": "/cb_left_hand_control_cmd and /cb_left_hand_state",
            "wuji_right_ros2": "/cb_right_hand_control_cmd and /cb_right_hand_state",
        },
        "source_field_semantics": {
            "observation.state": "43-D Pinocchio full-q container; embedded 29-D body_q was already published in hardware/MuJoCo order as absolute feedback; legacy Dex3 slots are not O6",
            "action.wbc": "43-D Pinocchio container; embedded 29-D last_action was already published in hardware/MuJoCo order as the scaled absolute q_target",
            "observation.left_hand_q/right_hand_q": "O6 actual feedback, SDK order, percentage 0..100",
            "action.left_hand/right_hand": "actual command sent toward O6 driver, SDK order, percentage 0..100; not raw Wuji landmarks",
            "action": "64-D SONIC motion token plus 6-D left O6 command plus 6-D right O6 command",
            "timestamp": "episode-relative seconds at 30 Hz",
        },
        "all_video_frame_counts_match": all(e["video_frame_match"] for e in episodes),
        "all_timestamps_monotonic": all(e["timestamp_monotonic"] for e in episodes),
        "all_frame_indices_contiguous": all(e["frame_index_contiguous"] for e in episodes),
        "all_selected_values_finite": all(e["all_selected_values_finite"] for e in episodes),
        "all_action76_component_checks_pass": all(e["canonical_action76_matches_component_columns"] for e in episodes),
        "left_valid_rate": left_valid_frames / total_frames,
        "right_valid_rate": right_valid_frames / total_frames,
        "global": {
            "left_o6_feedback_pct": vector_stats(merged["left_q"]),
            "right_o6_feedback_pct": vector_stats(merged["right_q"]),
            "left_o6_command_pct": vector_stats(merged["left_cmd"]),
            "right_o6_command_pct": vector_stats(merged["right_cmd"]),
            "g1_feedback_abs_rad_hardware_order": vector_stats(merged["feedback_abs_hw"]),
            "g1_command_abs_rad_hardware_order": vector_stats(merged["command_abs_hw"]),
            "projected_gravity": vector_stats(merged["projected_gravity"]),
        },
        "episodes": episodes,
        "thresholds_are_diagnostics_not_hard_safety_limits": {
            "arm_jump_rad_per_frame": 0.25,
            "waist_jump_rad_per_frame": 0.15,
            "o6_jump_percentage_points_per_frame": 30.0,
            "o6_saturation": "value <= 1 or >= 99 percent",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = audit(args.source)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
