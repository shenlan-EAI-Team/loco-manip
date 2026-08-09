#!/usr/bin/env python3
"""Validate converted G1/O6 datasets without changing them."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def video_frames(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(path)],
        check=True, text=True, capture_output=True,
    )
    return int(result.stdout.strip())


def validate(root: Path) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    parquets = sorted(root.glob("data/chunk-*/*.parquet"))
    errors = []
    total = 0
    global_next = 0
    for expected_episode, path in enumerate(parquets):
        schema = pq.read_schema(path)
        if str(schema.field("observation.state").type) != "fixed_size_list<element: float>[32]":
            errors.append(f"{path.name}: bad state Arrow type {schema.field('observation.state').type}")
        if str(schema.field("action").type) != "fixed_size_list<element: float>[26]":
            errors.append(f"{path.name}: bad action Arrow type {schema.field('action').type}")
        df = pd.read_parquet(path)
        state = np.stack(df["observation.state"]).astype(np.float32)
        action = np.stack(df["action"]).astype(np.float32)
        if state.shape != (len(df), 32) or action.shape != (len(df), 26):
            errors.append(f"{path.name}: bad array shape state={state.shape}, action={action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            errors.append(f"{path.name}: NaN/Inf")
        if not np.array_equal(df["frame_index"], np.arange(len(df))):
            errors.append(f"{path.name}: non-contiguous frame_index")
        if not np.all(df["episode_index"].to_numpy() == expected_episode):
            errors.append(f"{path.name}: wrong episode_index")
        expected_global = np.arange(global_next, global_next + len(df))
        if not np.array_equal(df["index"], expected_global):
            errors.append(f"{path.name}: wrong global index")
        if not bool(df["next.done"].iloc[-1]) or bool(df["next.done"].iloc[:-1].any()):
            errors.append(f"{path.name}: wrong next.done")
        video = root / "videos/chunk-000/observation.images.ego_view" / f"episode_{expected_episode:06d}.mp4"
        if video_frames(video) != len(df):
            errors.append(f"{path.name}: video/parquet frame mismatch")
        global_next += len(df)
        total += len(df)
    expected_slices = {
        "state": {
            "left_arm": [0, 7], "right_arm": [7, 14], "left_o6": [14, 20],
            "right_o6": [20, 26], "waist": [26, 29], "projected_gravity": [29, 32],
        },
        "action": {
            "left_arm": [0, 7], "right_arm": [7, 14],
            "left_o6": [14, 20], "right_o6": [20, 26],
        },
    }
    for group, keys in expected_slices.items():
        for key, (start, end) in keys.items():
            actual = modality[group].get(key)
            if actual != {"start": start, "end": end}:
                errors.append(f"modality {group}.{key}: {actual}")
    if len(parquets) != info["total_episodes"] or total != info["total_frames"]:
        errors.append("info.json totals disagree with files")
    return {
        "dataset": str(root.resolve()), "episodes": len(parquets), "frames": total,
        "state_dim": 32, "action_dim": 26,
        "waist_or_leg_in_action": False,
        "errors": errors, "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [validate(path) for path in args.datasets]
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    raise SystemExit(0 if all(r["ok"] for r in reports) else 1)


if __name__ == "__main__":
    main()
