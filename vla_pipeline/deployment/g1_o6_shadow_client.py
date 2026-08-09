#!/usr/bin/env python3
"""Offline Replay Shadow client. It records policy outputs and never publishes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from deployment.common import ACTION_KEYS, PROJECT_ROOT, TEST_DATASET, json_dump, load_policy
from deployment.observation_sources import LeRobotReplayObservationSource


def bounds_violation(key: str, action: np.ndarray) -> int:
    if key.endswith("o6"):
        return int(np.count_nonzero((action < 0) | (action > 100)))
    # Exact arm limits are enforced later by the Action Adapter; use a broad
    # finite-radian diagnostic here to avoid duplicating configuration.
    return int(np.count_nonzero((action < -3.2) | (action > 3.2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=TEST_DATASET)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--execution-horizon", type=int, default=3)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument(
        "--log-dir", type=Path, default=PROJECT_ROOT / "deployment/logs/replay_shadow"
    )
    args = parser.parse_args()
    if args.execution_horizon < 1:
        raise ValueError("execution_horizon must be positive")

    policy = load_policy(args.denoising_steps)
    source = LeRobotReplayObservationSource(
        args.dataset,
        policy.get_modality_config(),
        real_time=args.real_time,
        playback_speed=args.playback_speed,
    )
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "replay_shadow.jsonl"
    summary_path = args.log_dir / "summary.json"
    source.start()

    active_chunk: dict[str, np.ndarray] | None = None
    chunk_offset = 0
    previous_action: dict[str, np.ndarray] | None = None
    last_episode: int | None = None
    latencies = []
    counts = {
        "frames": 0,
        "inferences": 0,
        "nan_or_inf": 0,
        "out_of_bounds": 0,
        "buffer_underrun": 0,
        "missed_30hz_deadline": 0,
    }

    with log_path.open("w", encoding="utf-8") as handle:
        while True:
            sample = source.get_observation()
            if sample is None:
                break
            if sample.episode_index != last_episode:
                active_chunk = None
                chunk_offset = 0
                previous_action = None
                last_episode = sample.episode_index
            inference_start = None
            inference_end = None
            if active_chunk is None or chunk_offset >= args.execution_horizon:
                inference_start = time.monotonic()
                action, _ = policy.get_action(sample.observation)
                inference_end = time.monotonic()
                active_chunk = {key: np.asarray(action[key][0]) for key in ACTION_KEYS}
                chunk_offset = 0
                latency = inference_end - inference_start
                latencies.append(latency)
                counts["inferences"] += 1
                counts["missed_30hz_deadline"] += int(latency > 1.0 / 30.0)
            if active_chunk is None or chunk_offset >= min(
                len(active_chunk[key]) for key in ACTION_KEYS
            ):
                counts["buffer_underrun"] += 1
                continue

            executed = {key: active_chunk[key][chunk_offset] for key in ACTION_KEYS}
            finite = all(np.isfinite(value).all() for value in executed.values())
            violation = sum(bounds_violation(key, value) for key, value in executed.items())
            counts["nan_or_inf"] += int(not finite)
            counts["out_of_bounds"] += violation
            jumps = {
                key: (
                    float(np.max(np.abs(value - previous_action[key])))
                    if previous_action is not None
                    else 0.0
                )
                for key, value in executed.items()
            }
            record = {
                "episode": sample.episode_index,
                "frame": sample.frame_index,
                "observation_timestamp": sample.dataset_timestamp,
                "inference_start_monotonic": inference_start,
                "inference_end_monotonic": inference_end,
                "inference_latency_s": (
                    inference_end - inference_start if inference_start is not None else None
                ),
                "state": {
                    key: np.asarray(sample.flat_observation[f"state.{key}"]).tolist()
                    for key in (
                        "left_arm",
                        "right_arm",
                        "left_o6",
                        "right_o6",
                        "waist",
                        "projected_gravity",
                    )
                },
                "action_chunk": (
                    {key: value.tolist() for key, value in active_chunk.items()}
                    if chunk_offset == 0
                    else None
                ),
                "executed_action": {key: value.tolist() for key, value in executed.items()},
                "action_min": {key: float(np.min(value)) for key, value in executed.items()},
                "action_max": {key: float(np.max(value)) for key, value in executed.items()},
                "adjacent_jump": jumps,
                "finite": finite,
                "out_of_bounds_elements": violation,
                "buffer_underrun": False,
                "missed_30hz_deadline": (
                    inference_start is not None
                    and inference_end - inference_start > 1.0 / 30.0
                ),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            previous_action = {key: value.copy() for key, value in executed.items()}
            chunk_offset += 1
            counts["frames"] += 1
    source.stop()

    latency_array = np.asarray(latencies)
    summary = {
        "mode": "OFFLINE REPLAY SHADOW: NO COMMANDS SENT",
        "dry_run": True,
        "real_hardware_enabled": False,
        "dataset": str(args.dataset),
        "denoising_steps": args.denoising_steps,
        "execution_horizon": args.execution_horizon,
        "real_time": args.real_time,
        "playback_speed": args.playback_speed,
        "counts": counts,
        "latency_ms": {
            "mean": float(np.mean(latency_array) * 1000),
            "p50": float(np.percentile(latency_array, 50) * 1000),
            "p90": float(np.percentile(latency_array, 90) * 1000),
            "p99": float(np.percentile(latency_array, 99) * 1000),
            "max": float(np.max(latency_array) * 1000),
        },
    }
    json_dump(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
