#!/usr/bin/env python3
"""Convert selected source episodes to GR00T-flavored LeRobot v2 datasets.

The source is never opened for writing. Each destination is assembled in a
temporary sibling directory and atomically renamed only after completion.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from inspect_dataset import (
    ARM_JOINTS,
    O6_JOINTS,
    WAIST_JOINTS,
    reconstruct_g1,
    stack_column,
)


TASK = "Pick up the green cylinder from the table using the left O6 hand and place it into the white bin."
STATE_NAMES = (
    [f"left_{name}_joint" for name in ARM_JOINTS]
    + [f"right_{name}_joint" for name in ARM_JOINTS]
    + [f"left_o6_{name}" for name in O6_JOINTS]
    + [f"right_o6_{name}" for name in O6_JOINTS]
    + WAIST_JOINTS
    + ["projected_gravity_x", "projected_gravity_y", "projected_gravity_z"]
)
ACTION_NAMES = (
    [f"left_{name}_target" for name in ARM_JOINTS]
    + [f"right_{name}_target" for name in ARM_JOINTS]
    + [f"left_o6_{name}_target_pct" for name in O6_JOINTS]
    + [f"right_o6_{name}_target_pct" for name in O6_JOINTS]
)
STATE_DIM = 32
ACTION_DIM = 26

MODALITY = {
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
    "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def feature_info() -> dict:
    return {
        "observation.images.ego_view": {
            "dtype": "video", "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": 480, "video.width": 640, "video.codec": "h264",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "video.fps": 30, "video.channels": 3, "has_audio": False,
            },
        },
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ACTION_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }


ARROW_SCHEMA = pa.schema([
    pa.field("observation.state", pa.list_(pa.float32(), STATE_DIM)),
    pa.field("action", pa.list_(pa.float32(), ACTION_DIM)),
    pa.field("timestamp", pa.float32()),
    pa.field("frame_index", pa.int64()),
    pa.field("episode_index", pa.int64()),
    pa.field("index", pa.int64()),
    pa.field("task_index", pa.int64()),
    pa.field("annotation.human.task_description", pa.int64()),
    pa.field("next.reward", pa.float32()),
    pa.field("next.done", pa.bool_()),
])


def fixed_list_array(values: np.ndarray, dim: int) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"expected (*, {dim}), got {values.shape}")
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def transform_frame(df: pd.DataFrame, local_episode: int, global_start: int) -> pa.Table:
    g1 = reconstruct_g1(df)
    feedback_hw = g1["feedback_abs_hw"]
    command_hw = g1["command_abs_hw"]
    left_q = stack_column(df, "observation.left_hand_q", 6)
    right_q = stack_column(df, "observation.right_hand_q", 6)
    left_command = stack_column(df, "action.left_hand", 6)
    right_command = stack_column(df, "action.right_hand", 6)
    gravity = stack_column(df, "observation.projected_gravity", 3)

    state = np.concatenate(
        [feedback_hw[:, 15:22], feedback_hw[:, 22:29], left_q, right_q,
         feedback_hw[:, 12:15], gravity], axis=1,
    ).astype(np.float32)
    # Store absolute commands. g1_o6_config.py marks arm keys RELATIVE, so the
    # official N1.7 processor computes q_target(t+k) - q_feedback(t).
    action = np.concatenate(
        [command_hw[:, 15:22], command_hw[:, 22:29], left_command, right_command],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("non-finite converted state/action")

    n = len(df)
    arrays = [
        fixed_list_array(state, STATE_DIM),
        fixed_list_array(action, ACTION_DIM),
        pa.array(df["timestamp"].to_numpy(dtype=np.float32), type=pa.float32()),
        pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(n, local_episode, dtype=np.int64), type=pa.int64()),
        pa.array(np.arange(global_start, global_start + n, dtype=np.int64), type=pa.int64()),
        pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        pa.array(np.zeros(n, dtype=np.float32), type=pa.float32()),
        pa.array(np.arange(n) == n - 1, type=pa.bool_()),
    ]
    return pa.Table.from_arrays(arrays, schema=ARROW_SCHEMA)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def convert_subset(source: Path, target: Path, source_episode_ids: list[int], subset_name: str) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"stale temporary output exists: {tmp}")
    data_dir = tmp / "data/chunk-000"
    video_dir = tmp / "videos/chunk-000/observation.images.ego_view"
    meta_dir = tmp / "meta"
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    episodes_meta = []
    source_to_local = []
    global_start = 0
    try:
        for local_episode, source_episode in enumerate(source_episode_ids):
            src_parquet = source / "data/chunk-000" / f"episode_{source_episode:06d}.parquet"
            src_video = source / "videos/chunk-000/observation.images.ego_view" / f"episode_{source_episode:06d}.mp4"
            df = pd.read_parquet(src_parquet)
            table = transform_frame(df, local_episode, global_start)
            dst_parquet = data_dir / f"episode_{local_episode:06d}.parquet"
            pq.write_table(table, dst_parquet, compression="zstd")
            shutil.copy2(src_video, video_dir / f"episode_{local_episode:06d}.mp4")
            episodes_meta.append({"episode_index": local_episode, "tasks": [TASK], "length": len(df)})
            source_to_local.append({"source_episode": source_episode, "local_episode": local_episode, "frames": len(df)})
            global_start += len(df)

        info = {
            "codebase_version": "v2.1",
            "robot_type": "unitree_g1_o6_upper_body",
            "total_episodes": len(source_episode_ids),
            "total_frames": global_start,
            "total_tasks": 1,
            "total_videos": len(source_episode_ids),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 30,
            "splits": {"train": f"0:{len(source_episode_ids)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feature_info(),
        }
        write_json(meta_dir / "info.json", info)
        write_json(meta_dir / "modality.json", MODALITY)
        (meta_dir / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
        (meta_dir / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in episodes_meta)
        )
        write_json(meta_dir / "conversion_manifest.json", {
            "subset": subset_name,
            "source_dataset": str(source.resolve()),
            "source_opened_read_only": True,
            "source_to_local_episode": source_to_local,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "g1_source_transport_contract": (
                "g1_debug body_q and last_action are already hardware/MuJoCo ordered absolute "
                "radians; no second reorder, default-angle offset, or action scaling is applied"
            ),
            "arm_state_storage": "absolute hardware joint feedback in rad",
            "arm_action_storage": "absolute hardware joint target in rad; N1.7 config converts to relative",
            "o6_action_storage": "absolute SDK percentage target, 0..100",
        })
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    if source == output_root or output_root in source.parents:
        raise ValueError("output must not contain or equal source")
    split = json.loads(args.split.read_text())
    for subset in ("smoke_2", "train_26", "val_2", "test_2"):
        convert_subset(source, output_root / subset, split[subset]["source_episodes"], subset)


if __name__ == "__main__":
    main()
