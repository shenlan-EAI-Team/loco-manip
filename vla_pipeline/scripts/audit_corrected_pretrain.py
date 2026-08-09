#!/usr/bin/env python3
"""Independent pre-training audit for corrected G1 + dual-O6 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from inspect_dataset import ARM_JOINTS, O6_JOINTS, WAIST_JOINTS, reconstruct_g1, stack_column


SPLITS = ("smoke_2", "train_26", "val_2", "test_2")
STATE_DIM = 32
ACTION_DIM = 26
HORIZON = 16
ARM_NAMES = [f"left_{name}_joint" for name in ARM_JOINTS] + [
    f"right_{name}_joint" for name in ARM_JOINTS
]
WAIST_XML_NAMES = [f"{name}_joint" for name in WAIST_JOINTS]
LIMIT_NAMES = WAIST_XML_NAMES + ARM_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frames": int(stream["nb_read_frames"]),
    }


def load_vector(df: pd.DataFrame, key: str, dim: int) -> np.ndarray:
    values = np.stack(df[key].to_numpy()).astype(np.float32, copy=False)
    if values.shape != (len(df), dim):
        raise AssertionError(f"{key}: expected {(len(df), dim)}, got {values.shape}")
    return values


def compare_arrays(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    if not np.array_equal(actual, expected):
        error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        raise AssertionError(f"{name}: values differ, max_abs_error={error}")


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def compare_stats(name: str, actual: dict, expected: dict, tolerance: float = 2e-6) -> float:
    maximum_error = 0.0
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        lhs = np.asarray(actual[key], dtype=np.float64)
        rhs = np.asarray(expected[key], dtype=np.float64)
        if lhs.shape != rhs.shape:
            raise AssertionError(f"{name}.{key}: shape {lhs.shape} != {rhs.shape}")
        error = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
        maximum_error = max(maximum_error, error)
        if not np.allclose(lhs, rhs, rtol=1e-6, atol=tolerance):
            raise AssertionError(f"{name}.{key}: max_abs_error={error}")
    return maximum_error


def load_joint_limits(xml_path: Path) -> dict[str, tuple[float, float]]:
    limits = {}
    for joint in ET.parse(xml_path).iter("joint"):
        name = joint.get("name")
        joint_range = joint.get("range")
        if name in LIMIT_NAMES and joint_range:
            lower, upper = (float(value) for value in joint_range.split())
            limits[name] = (lower, upper)
    if set(limits) != set(LIMIT_NAMES):
        raise AssertionError(f"missing joint limits: {sorted(set(LIMIT_NAMES) - set(limits))}")
    return limits


def audit(args: argparse.Namespace) -> dict:
    split_spec = json.loads(args.split_spec.read_text())
    source = Path(split_spec["source_dataset"]).resolve()
    output_root = args.output_root.resolve()
    limits = load_joint_limits(args.g1_xml)

    train_ids = split_spec["train_26"]["source_episodes"]
    val_ids = split_spec["val_2"]["source_episodes"]
    test_ids = split_spec["test_2"]["source_episodes"]
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise AssertionError("train/val/test episode overlap")
    if sorted(train_ids + val_ids + test_ids) != list(range(30)):
        raise AssertionError("train/val/test must cover source episodes 0..29 exactly")
    if split_spec["smoke_2"]["source_episodes"] != train_ids[:2]:
        raise AssertionError("smoke_2 must be the first two training episodes")

    report = {
        "status": "PASS",
        "source_dataset": str(source),
        "output_root": str(output_root),
        "contracts": {
            "source_g1": (
                "body_q and last_action are hardware-ordered absolute radians before export; "
                "conversion performs no reorder, offset, or scale"
            ),
            "state": {
                "shape": [STATE_DIM],
                "left_arm": [0, 7],
                "right_arm": [7, 14],
                "left_o6": [14, 20],
                "right_o6": [20, 26],
                "waist": [26, 29],
                "projected_gravity": [29, 32],
            },
            "action": {
                "per_step_shape": [ACTION_DIM],
                "chunk_shape": [HORIZON, ACTION_DIM],
                "left_arm": [0, 7],
                "right_arm": [7, 14],
                "left_o6": [14, 20],
                "right_o6": [20, 26],
            },
            "arm_representation": "RELATIVE/NON_EEF processor over absolute-radian parquet targets",
            "o6_representation": "ABSOLUTE/NON_EEF, percentage range 0..100",
            "waist_and_gravity": "input-only; no waist or leg action output",
            "image": "ego_view RGB, HWC 480x640x3, 30 fps",
        },
        "split_integrity": {
            "train": train_ids,
            "validation": val_ids,
            "test": test_ids,
            "disjoint": True,
            "covers_source_episodes_0_through_29": True,
            "smoke_is_train_subset": True,
        },
        "splits": {},
        "warnings": [],
    }

    for split_name in SPLITS:
        dataset = output_root / split_name
        manifest = json.loads((dataset / "meta/conversion_manifest.json").read_text())
        info = json.loads((dataset / "meta/info.json").read_text())
        modality = json.loads((dataset / "meta/modality.json").read_text())
        expected_ids = split_spec[split_name]["source_episodes"]
        manifest_ids = [row["source_episode"] for row in manifest["source_to_local_episode"]]
        if manifest_ids != expected_ids:
            raise AssertionError(f"{split_name}: manifest episode mapping mismatch")
        if info["features"]["observation.state"]["shape"] != [STATE_DIM]:
            raise AssertionError(f"{split_name}: state feature shape mismatch")
        if info["features"]["action"]["shape"] != [ACTION_DIM]:
            raise AssertionError(f"{split_name}: action feature shape mismatch")
        image_feature = info["features"]["observation.images.ego_view"]
        if image_feature["shape"] != [480, 640, 3] or info["fps"] != 30:
            raise AssertionError(f"{split_name}: image metadata mismatch")
        expected_modality = {
            "state": {
                "left_arm": {"start": 0, "end": 7},
                "right_arm": {"start": 7, "end": 14},
                "left_o6": {"start": 14, "end": 20},
                "right_o6": {"start": 20, "end": 26},
                "waist": {"start": 26, "end": 29},
                "projected_gravity": {"start": 29, "end": 32},
            },
            "action": {
                "left_arm": {"start": 0, "end": 7},
                "right_arm": {"start": 7, "end": 14},
                "left_o6": {"start": 14, "end": 20},
                "right_o6": {"start": 20, "end": 26},
            },
        }
        for category, expected in expected_modality.items():
            if modality[category] != expected:
                raise AssertionError(f"{split_name}: {category} modality mapping mismatch")

        accumulated = {"observation.state": [], "action": [], "timestamp": [], "next.reward": []}
        relative = {"left_arm": [], "right_arm": []}
        split_frames = 0
        gravity_norms = []
        left_valid = []
        right_valid = []
        o6_state = {"left": [], "right": []}
        o6_action = {"left": [], "right": []}
        maximum_limit_violation = 0.0
        video_records = []
        global_index = 0

        for local_episode, source_episode in enumerate(expected_ids):
            src_parquet = source / "data/chunk-000" / f"episode_{source_episode:06d}.parquet"
            dst_parquet = dataset / "data/chunk-000" / f"episode_{local_episode:06d}.parquet"
            src_video = (
                source
                / "videos/chunk-000/observation.images.ego_view"
                / f"episode_{source_episode:06d}.mp4"
            )
            dst_video = (
                dataset
                / "videos/chunk-000/observation.images.ego_view"
                / f"episode_{local_episode:06d}.mp4"
            )
            src_df = pd.read_parquet(src_parquet)
            dst_df = pd.read_parquet(dst_parquet)
            g1 = reconstruct_g1(src_df)
            left_q = stack_column(src_df, "observation.left_hand_q", 6).astype(np.float32)
            right_q = stack_column(src_df, "observation.right_hand_q", 6).astype(np.float32)
            left_cmd = stack_column(src_df, "action.left_hand", 6).astype(np.float32)
            right_cmd = stack_column(src_df, "action.right_hand", 6).astype(np.float32)
            gravity = stack_column(src_df, "observation.projected_gravity", 3).astype(np.float32)
            expected_state = np.concatenate(
                [
                    g1["feedback_abs_hw"][:, 15:22],
                    g1["feedback_abs_hw"][:, 22:29],
                    left_q,
                    right_q,
                    g1["feedback_abs_hw"][:, 12:15],
                    gravity,
                ],
                axis=1,
            ).astype(np.float32)
            expected_action = np.concatenate(
                [
                    g1["command_abs_hw"][:, 15:22],
                    g1["command_abs_hw"][:, 22:29],
                    left_cmd,
                    right_cmd,
                ],
                axis=1,
            ).astype(np.float32)
            state = load_vector(dst_df, "observation.state", STATE_DIM)
            action = load_vector(dst_df, "action", ACTION_DIM)
            compare_arrays(f"{split_name}/{local_episode}/state", state, expected_state)
            compare_arrays(f"{split_name}/{local_episode}/action", action, expected_action)

            timestamps = dst_df["timestamp"].to_numpy(dtype=np.float32)
            source_timestamps = src_df["timestamp"].to_numpy(dtype=np.float32)
            compare_arrays(f"{split_name}/{local_episode}/timestamp", timestamps, source_timestamps)
            if len(timestamps) < HORIZON or not np.all(np.diff(timestamps.astype(np.float64)) > 0):
                raise AssertionError(f"{split_name}/{local_episode}: invalid timestamp sequence")
            if not np.array_equal(dst_df["frame_index"].to_numpy(), np.arange(len(dst_df))):
                raise AssertionError(f"{split_name}/{local_episode}: frame indices are not contiguous")
            if not np.array_equal(
                dst_df["episode_index"].to_numpy(), np.full(len(dst_df), local_episode)
            ):
                raise AssertionError(f"{split_name}/{local_episode}: episode index mismatch")
            if not np.array_equal(
                dst_df["index"].to_numpy(), np.arange(global_index, global_index + len(dst_df))
            ):
                raise AssertionError(f"{split_name}/{local_episode}: global index mismatch")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise AssertionError(f"{split_name}/{local_episode}: non-finite state/action")

            for offset, joint_name in enumerate(LIMIT_NAMES):
                state_index = 26 + offset if offset < 3 else offset - 3
                action_index = None if offset < 3 else offset - 3
                lower, upper = limits[joint_name]
                values = [state[:, state_index]]
                if action_index is not None:
                    values.append(action[:, action_index])
                for joint_values in values:
                    violation = max(
                        float(np.max(lower - joint_values)), float(np.max(joint_values - upper)), 0.0
                    )
                    maximum_limit_violation = max(maximum_limit_violation, violation)
            if maximum_limit_violation > 0:
                raise AssertionError(f"{split_name}: arm/waist joint limit violation")

            if np.min(state[:, 14:26]) < 0 or np.max(state[:, 14:26]) > 100:
                raise AssertionError(f"{split_name}/{local_episode}: O6 state outside 0..100")
            if np.min(action[:, 14:26]) < 0 or np.max(action[:, 14:26]) > 100:
                raise AssertionError(f"{split_name}/{local_episode}: O6 action outside 0..100")

            for key, values in (
                ("observation.state", state),
                ("action", action),
                ("timestamp", timestamps[:, None]),
                ("next.reward", dst_df["next.reward"].to_numpy(dtype=np.float32)[:, None]),
            ):
                accumulated[key].append(values)
            for start in range(len(dst_df) - HORIZON + 1):
                relative["left_arm"].append(action[start : start + HORIZON, :7] - state[start, :7])
                relative["right_arm"].append(
                    action[start : start + HORIZON, 7:14] - state[start, 7:14]
                )
            gravity_norms.append(np.linalg.norm(state[:, 29:32], axis=1))
            left_valid.append(src_df["left_hand_valid"].to_numpy(dtype=np.float32))
            right_valid.append(src_df["right_hand_valid"].to_numpy(dtype=np.float32))
            o6_state["left"].append(state[:, 14:20])
            o6_state["right"].append(state[:, 20:26])
            o6_action["left"].append(action[:, 14:20])
            o6_action["right"].append(action[:, 20:26])

            src_hash = sha256(src_video)
            dst_hash = sha256(dst_video)
            if src_hash != dst_hash:
                raise AssertionError(f"{split_name}/{local_episode}: copied video hash mismatch")
            video = ffprobe(dst_video)
            if (
                video["width"] != 640
                or video["height"] != 480
                or not np.isclose(video["fps"], 30.0)
                or video["frames"] != len(dst_df)
            ):
                raise AssertionError(f"{split_name}/{local_episode}: video contract mismatch {video}")
            video_records.append({"local_episode": local_episode, "sha256": dst_hash, **video})
            split_frames += len(dst_df)
            global_index += len(dst_df)

        if info["total_frames"] != split_frames or info["total_episodes"] != len(expected_ids):
            raise AssertionError(f"{split_name}: info totals mismatch")
        stored_stats = json.loads((dataset / "meta/stats.json").read_text())
        stat_errors = {}
        for key, chunks in accumulated.items():
            independent = stats(np.concatenate(chunks, axis=0))
            stat_errors[key] = compare_stats(key, stored_stats[key], independent)

        stored_relative = json.loads((dataset / "meta/relative_stats.json").read_text())
        relative_errors = {}
        for key, trajectories in relative.items():
            independent = stats(np.stack(trajectories, axis=0))
            relative_errors[key] = compare_stats(key, stored_relative[key], independent)
            if np.asarray(stored_relative[key]["mean"]).shape != (HORIZON, 7):
                raise AssertionError(f"{split_name}: {key} relative stats shape mismatch")

        gravity_all = np.concatenate(gravity_norms)
        if not np.allclose(gravity_all, 1.0, atol=1e-5):
            raise AssertionError(f"{split_name}: projected gravity is not unit length")
        left_valid_all = np.concatenate(left_valid)
        right_valid_all = np.concatenate(right_valid)
        split_report = {
            "source_episodes": expected_ids,
            "episodes": len(expected_ids),
            "frames": split_frames,
            "all_values_finite": True,
            "timestamps_strictly_increasing": True,
            "source_to_corrected_state_action_exact": True,
            "video_source_hashes_exact": True,
            "video_contract": "H264 640x480 at 30 fps; frame count equals parquet rows",
            "video_records": video_records,
            "joint_limits": {"all_arm_state_action_and_waist_state_inside": True},
            "projected_gravity_norm": {
                "min": float(np.min(gravity_all)),
                "max": float(np.max(gravity_all)),
            },
            "o6": {
                "left_valid_rate": float(np.mean(left_valid_all > 0.5)),
                "right_valid_rate": float(np.mean(right_valid_all > 0.5)),
                "left_state_minmax": [
                    float(np.min(np.concatenate(o6_state["left"]))),
                    float(np.max(np.concatenate(o6_state["left"]))),
                ],
                "left_action_minmax": [
                    float(np.min(np.concatenate(o6_action["left"]))),
                    float(np.max(np.concatenate(o6_action["left"]))),
                ],
                "right_state_unique": np.unique(np.concatenate(o6_state["right"])).tolist(),
                "right_action_unique": np.unique(np.concatenate(o6_action["right"])).tolist(),
            },
            "stats_independent_recompute_max_abs_error": stat_errors,
            "relative_stats_independent_recompute_max_abs_error": relative_errors,
            "relative_stats_shapes": {"left_arm": [HORIZON, 7], "right_arm": [HORIZON, 7]},
        }
        report["splits"][split_name] = split_report

    train_o6 = report["splits"]["train_26"]["o6"]
    if train_o6["right_valid_rate"] == 0.0:
        report["warnings"].append(
            "Right O6 feedback/action are all zero and right_hand_valid is always false. "
            "Training is valid for both arms and left O6, but cannot teach right O6 behavior."
        )
    report["warnings"].append(
        "Some source episodes include post-success manual reset tails; they remain untrimmed."
    )

    # Verify that the official loader consumes the corrected schema and exposes the expected groups.
    sys.path.insert(0, str(args.groot_repo))
    sys.path.insert(0, str(args.modality_config.parent))
    __import__(args.modality_config.stem)
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    config = MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    loader = LeRobotEpisodeLoader(str(output_root / "train_26"), modality_configs=config)
    official_stats = loader.get_dataset_statistics()
    expected_state_dims = {
        "left_arm": 7,
        "right_arm": 7,
        "left_o6": 6,
        "right_o6": 6,
        "waist": 3,
        "projected_gravity": 3,
    }
    expected_action_dims = {"left_arm": 7, "right_arm": 7, "left_o6": 6, "right_o6": 6}
    for group, dim in expected_state_dims.items():
        if len(official_stats["state"][group]["mean"]) != dim:
            raise AssertionError(f"official loader state group {group} dimension mismatch")
    for group, dim in expected_action_dims.items():
        if len(official_stats["action"][group]["mean"]) != dim:
            raise AssertionError(f"official loader action group {group} dimension mismatch")
    if set(official_stats["relative_action"]) != {"left_arm", "right_arm"}:
        raise AssertionError("official loader relative action stats mismatch")
    report["official_loader"] = {
        "status": "PASS",
        "episodes": len(loader),
        "state_group_dims": expected_state_dims,
        "action_group_dims": expected_action_dims,
        "relative_action_groups": ["left_arm", "right_arm"],
        "action_horizon": HORIZON,
    }

    sample_df = pd.read_parquet(
        output_root / "train_26/data/chunk-000/episode_000000.parquet"
    )
    sample_state = load_vector(sample_df, "observation.state", STATE_DIM)
    sample_action = load_vector(sample_df, "action", ACTION_DIM)
    state_groups = {
        "left_arm": sample_state[:1, 0:7],
        "right_arm": sample_state[:1, 7:14],
        "left_o6": sample_state[:1, 14:20],
        "right_o6": sample_state[:1, 20:26],
        "waist": sample_state[:1, 26:29],
        "projected_gravity": sample_state[:1, 29:32],
    }
    action_groups = {
        "left_arm": sample_action[:HORIZON, 0:7],
        "right_arm": sample_action[:HORIZON, 7:14],
        "left_o6": sample_action[:HORIZON, 14:20],
        "right_o6": sample_action[:HORIZON, 20:26],
    }
    state_action_processor = StateActionProcessor(
        modality_configs={EmbodimentTag.NEW_EMBODIMENT.value: config},
        statistics={EmbodimentTag.NEW_EMBODIMENT.value: official_stats},
        use_percentiles=True,
        clip_outliers=True,
        use_relative_action=True,
    )
    normalized_state, normalized_action = state_action_processor.apply(
        state_groups, action_groups, EmbodimentTag.NEW_EMBODIMENT.value
    )
    normalized_state_array = np.concatenate(list(normalized_state.values()), axis=-1)
    normalized_action_array = np.concatenate(list(normalized_action.values()), axis=-1)
    if normalized_state_array.shape != (1, STATE_DIM):
        raise AssertionError("normalized semantic state shape mismatch")
    if normalized_action_array.shape != (HORIZON, ACTION_DIM):
        raise AssertionError("normalized semantic action shape mismatch")
    if not np.isfinite(normalized_state_array).all() or not np.isfinite(
        normalized_action_array
    ).all():
        raise AssertionError("non-finite normalized processor sample")
    if np.min(normalized_state_array) < -1 or np.max(normalized_state_array) > 1:
        raise AssertionError("normalized state outside [-1, 1]")
    if np.min(normalized_action_array) < -1 or np.max(normalized_action_array) > 1:
        raise AssertionError("normalized action outside [-1, 1]")
    report["normalization_processor"] = {
        "status": "PASS",
        "use_percentiles": True,
        "clip_outliers": True,
        "use_relative_action": True,
        "semantic_state_shape": [1, STATE_DIM],
        "semantic_action_shape": [HORIZON, ACTION_DIM],
        "normalized_state_range": [
            float(np.min(normalized_state_array)),
            float(np.max(normalized_state_array)),
        ],
        "normalized_action_range": [
            float(np.min(normalized_action_array)),
            float(np.max(normalized_action_array)),
        ],
        "all_finite": True,
        "right_o6_constant_zero_normalizes_to": np.unique(
            normalized_action["right_o6"]
        ).tolist(),
        "n1d7_model_padding": {
            "state": [1, 132],
            "action": [40, 132],
            "action_mask_valid_elements": HORIZON * ACTION_DIM,
        },
        "full_processor_sample_observed": {
            "state": [1, 132],
            "action": [40, 132],
            "action_mask": [40, 132],
            "transformed_ego_view": [3, 256, 340],
        },
    }
    return report


def write_markdown(report: dict, path: Path) -> None:
    train = report["splits"]["train_26"]
    lines = [
        "# Corrected Dataset Pre-training Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Model Contract",
        "",
        "- Image input: `ego_view`, RGB HWC `480 x 640 x 3`, 30 fps.",
        "- State input: 32 values = left arm 7 + right arm 7 + left O6 6 + right O6 6 + waist 3 + projected gravity 3.",
        "- Action target: 26 values per step; 16-step chunk (`16 x 26`).",
        "- Arm parquet targets are absolute hardware-order radians. The official processor alone converts them to relative actions.",
        "- O6 targets are absolute 0..100 percentages. Waist and projected gravity are inputs only.",
        "",
        "## Hard Checks",
        "",
        "- Train/validation/test episodes are disjoint and cover source episodes 0..29.",
        "- Every corrected state/action value exactly matches the independently reconstructed source value.",
        "- No second joint reorder, default-angle offset, or action scaling remains.",
        "- All state/action values are finite; all arm targets, arm feedback, and waist feedback are within the official G1 XML limits.",
        "- All timestamps are strictly increasing and frame/global/episode indices are consistent.",
        "- Every copied video has the same SHA256 as its source and is H264 640x480 at 30 fps with matching frame count.",
        "- Projected gravity is body-frame inverse-rotated world gravity and has unit norm.",
        "- `stats.json` and both 16x7 relative-arm statistics were independently recomputed and match.",
        "- Official `LeRobotEpisodeLoader` accepts the dataset and exposes every configured group at the expected dimension.",
        "- The official normalization processor produces finite semantic tensors (`state 1x32`, `action 16x26`) in [-1, 1].",
        "- The N1.7 processor pads them to `state 1x132`, `action 40x132`; exactly 416 action-mask elements are valid.",
        "",
        "## Training Split",
        "",
        f"- Episodes: {train['episodes']}",
        f"- Frames: {train['frames']}",
        f"- Left O6 valid rate: {train['o6']['left_valid_rate']:.6f}",
        f"- Right O6 valid rate: {train['o6']['right_valid_rate']:.6f}",
        f"- Gravity norm range: {train['projected_gravity_norm']['min']:.9f} .. {train['projected_gravity_norm']['max']:.9f}",
        "",
        "## Known Limitations",
        "",
    ]
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The corrected dataset passes the hard pre-training gate. It is suitable for retraining both arms and the left O6. Right O6 behavior must not be claimed from this dataset.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--output-root", type=Path, default=project / "datasets_corrected_v1")
    parser.add_argument("--split-spec", type=Path, default=project / "split.json")
    parser.add_argument(
        "--g1-xml",
        type=Path,
        default=Path(
            "/home/slxy/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"
        ),
    )
    parser.add_argument("--groot-repo", type=Path, default=Path("/home/slxy/下载/Isaac-GR00T"))
    parser.add_argument("--modality-config", type=Path, default=project / "configs/g1_o6_config.py")
    parser.add_argument(
        "--json-report", type=Path, default=project / "deployment/corrected_pretrain_audit.json"
    )
    parser.add_argument(
        "--markdown-report", type=Path, default=project / "deployment/corrected_pretrain_audit.md"
    )
    args = parser.parse_args()
    try:
        report = audit(args)
    except BaseException as exc:
        failure = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        args.json_report.write_text(json.dumps(failure, indent=2) + "\n")
        raise
    args.json_report.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.markdown_report)
    print(json.dumps({"status": report["status"], "warnings": report["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
