#!/usr/bin/env python3
"""Real-observation Live Shadow runner with a transport-free Null Sink."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import time
from typing import Any

import cv2
import numpy as np
import yaml

from configs.g1_o6_config import g1_o6_config
from deployment.action_adapter import ActionAdapter
from deployment.common import ACTION_KEYS, CHECKPOINT, PROJECT_ROOT, json_dump, load_policy, seed_everything
from deployment.null_sink import NullActionSink
from deployment.observation_sources import G1LiveObservationSource


DEPLOYMENT_ROOT = PROJECT_ROOT / "deployment"


def percentiles_ms(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def vector_dict(value: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(item).tolist() for key, item in value.items()}


def image_feature(image: np.ndarray) -> list[float]:
    resized = cv2.resize(np.asarray(image), (16, 16), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(-1).tolist()


def analyze(records: list[dict[str, Any]], adapter: ActionAdapter) -> dict[str, Any]:
    result: dict[str, Any] = {"per_group": {}}
    for key in ACTION_KEYS:
        first_deltas = np.asarray([row["first_target_delta"][key] for row in records])
        first_targets = np.asarray([row["raw_first_target"][key] for row in records])
        feedback = np.asarray([row["feedback"][key] for row in records])
        filtered = np.asarray([row["filtered_first_target"][key] for row in records])
        target_drift = np.diff(first_targets, axis=0) if len(first_targets) > 1 else np.empty((0, 1))
        feedback_motion = np.diff(feedback, axis=0) if len(feedback) > 1 else np.empty((0, 1))
        result["per_group"][key] = {
            "first_target_minus_feedback_abs_mean": float(np.mean(np.abs(first_deltas))),
            "first_target_minus_feedback_abs_p95": float(np.percentile(np.abs(first_deltas), 95)),
            "first_target_minus_feedback_abs_max": float(np.max(np.abs(first_deltas))),
            "successive_first_target_drift_abs_mean": (
                float(np.mean(np.abs(target_drift))) if target_drift.size else 0.0
            ),
            "successive_first_target_drift_abs_p95": (
                float(np.percentile(np.abs(target_drift), 95)) if target_drift.size else 0.0
            ),
            "feedback_motion_abs_mean": (
                float(np.mean(np.abs(feedback_motion))) if feedback_motion.size else 0.0
            ),
            "feedback_motion_abs_max": (
                float(np.max(np.abs(feedback_motion))) if feedback_motion.size else 0.0
            ),
            "raw_target_min": float(first_targets.min()),
            "raw_target_max": float(first_targets.max()),
            "filtered_first_target_min": float(filtered.min()),
            "filtered_first_target_max": float(filtered.max()),
            "raw_all_zero_ratio": float(np.mean(np.isclose(first_targets, 0.0))),
            "filter_counters": adapter.filter.counters[key].as_dict(),
        }

    executed = int(adapter.config["execution_horizon"])
    arm_denominator = len(records) * executed * 7
    result["arm_filter_trigger_rates"] = {
        key: {
            "velocity": (
                adapter.filter.counters[key].velocity_limit / arm_denominator
                if arm_denominator else 0.0
            ),
            "acceleration": (
                adapter.filter.counters[key].acceleration_limit / arm_denominator
                if arm_denominator else 0.0
            ),
            "denominator_joint_steps": arm_denominator,
        }
        for key in ("left_arm", "right_arm")
    }
    if records:
        raw_left = np.concatenate(
            [np.asarray(row["raw_action_chunk"]["left_o6"][:executed]) for row in records],
            axis=0,
        )
        final_left = np.concatenate(
            [np.asarray(row["filtered_action_chunk"]["left_o6"]) for row in records],
            axis=0,
        )
        raw_jumps = np.abs(np.diff(raw_left, axis=0))
        final_jumps = np.abs(np.diff(final_left, axis=0))
        boundary_raw = np.asarray(
            [
                np.asarray(records[index]["raw_action_chunk"]["left_o6"])[0]
                - np.asarray(records[index - 1]["raw_action_chunk"]["left_o6"])[executed - 1]
                for index in range(1, len(records))
            ]
        )
        result["left_o6_spike_analysis"] = {
            "threshold_points": 30.0,
            "raw_adjacent_scalar_spike_ratio_gt_30": (
                float(np.mean(raw_jumps > 30.0)) if raw_jumps.size else 0.0
            ),
            "raw_adjacent_max_jump_points": float(raw_jumps.max()) if raw_jumps.size else 0.0,
            "raw_chunk_boundary_scalar_spike_ratio_gt_30": (
                float(np.mean(np.abs(boundary_raw) > 30.0)) if boundary_raw.size else 0.0
            ),
            "raw_chunk_boundary_max_jump_points": (
                float(np.max(np.abs(boundary_raw))) if boundary_raw.size else 0.0
            ),
            "final_scalar_spike_ratio_gt_30": (
                float(np.mean(final_jumps > 30.0)) if final_jumps.size else 0.0
            ),
            "final_max_jump_points": float(final_jumps.max()) if final_jumps.size else 0.0,
            "final_min": float(final_left.min()),
            "final_max": float(final_left.max()),
        }

    if len(records) > 1:
        images = np.asarray([row["image_feature"] for row in records])
        image_change = np.linalg.norm(np.diff(images, axis=0), axis=1)
        action_vector = np.concatenate(
            [np.asarray([row["raw_first_target"][key] for row in records]) for key in ACTION_KEYS],
            axis=1,
        )
        action_change = np.linalg.norm(np.diff(action_vector, axis=0), axis=1)
        if np.std(image_change) > 1e-9 and np.std(action_change) > 1e-9:
            correlation = float(np.corrcoef(image_change, action_change)[0, 1])
        else:
            correlation = 0.0
        result["image_action_response"] = {
            "image_change_mean": float(image_change.mean()),
            "action_change_mean": float(action_change.mean()),
            "pearson_change_correlation": correlation,
            "unique_image_features_rounded_3dp": int(
                len(np.unique(np.round(images, 3), axis=0))
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEPLOYMENT_ROOT / "config/live_shadow.yaml")
    parser.add_argument("--adapter-config", type=Path, default=DEPLOYMENT_ROOT / "config/adapter.yaml")
    parser.add_argument("--scenario", choices=("A", "B", "timing_7p5hz"), required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--replanning-hz", type=float, default=None)
    parser.add_argument("--log-root", type=Path, default=DEPLOYMENT_ROOT / "logs/live_shadow")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.replanning_hz is not None:
        config["replanning_hz"] = float(args.replanning_hz)
        config["execution_horizon"] = 4 if args.replanning_hz == 7.5 else config["execution_horizon"]

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{args.scenario}"
    run_dir = args.log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_dir = run_dir / "frames"
    frame_dir.mkdir()
    log_path = run_dir / "live_shadow.jsonl"
    summary_path = run_dir / "summary.json"

    print("LIVE SHADOW MODE", flush=True)
    print("NO COMMANDS WILL BE SENT", flush=True)
    print("NO CONTROL OWNERSHIP WILL BE REQUESTED", flush=True)
    print(f"SCENARIO {args.scenario}", flush=True)

    seed_everything(42)
    source = G1LiveObservationSource(config, g1_o6_config)
    adapter = ActionAdapter(args.adapter_config)
    null_sink = NullActionSink()
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    timings: dict[str, list[float]] = {
        "observation_construct": [],
        "policy": [],
        "adapter": [],
        "null_sink": [],
        "end_to_end": [],
    }
    records: list[dict[str, Any]] = []
    counts = Counter()
    warmups: list[float] = []
    started_wall = time.time()
    failure: str | None = None
    try:
        source.start()
        policy = load_policy(int(config["denoising_steps"]), args.checkpoint)
        warmup_count = int(config["warmup_inferences"])
        warmup_deadline = time.monotonic() + float(config["warmup_timeout_s"])
        while len(warmups) < warmup_count and not stop_requested:
            if time.monotonic() >= warmup_deadline:
                raise TimeoutError("no synchronized observations available for warm-up")
            sample = source.get_observation()
            if sample is None:
                time.sleep(0.001)
                continue
            start = time.perf_counter()
            policy.get_action(sample.observation)
            warmups.append((time.perf_counter() - start) * 1000)
        if len(warmups) != warmup_count:
            raise RuntimeError("warm-up interrupted")
        print(f"WARM-UP COMPLETE: {warmup_count} inferences", flush=True)

        duration_start = time.monotonic()
        next_plan = duration_start
        period = 1.0 / float(config["replanning_hz"])
        adapter_initialized = False
        with log_path.open("w", encoding="utf-8") as log:
            while not stop_requested and time.monotonic() - duration_start < args.duration:
                sample = source.get_observation()
                if sample is None:
                    time.sleep(0.001)
                    continue
                now = time.monotonic()
                if now < next_plan:
                    counts["observations_not_selected_for_replan"] += 1
                    continue
                pipeline_start = time.perf_counter()
                state = {
                    key: np.asarray(sample.flat_observation[f"state.{key}"], dtype=np.float32)
                    for key in ACTION_KEYS
                }
                if not adapter_initialized:
                    adapter.reset(state)
                    adapter_initialized = True

                policy_start = time.perf_counter()
                action_batched, _ = policy.get_action(sample.observation)
                policy_end = time.perf_counter()
                raw_chunk = {
                    key: np.asarray(action_batched[key][0], dtype=np.float32) for key in ACTION_KEYS
                }

                adapter_start = time.perf_counter()
                safe_chunk = adapter.prepare_chunk(
                    raw_chunk, timestamp=sample.monotonic_timestamp
                )
                adapter_end = time.perf_counter()

                sink_start = time.perf_counter()
                sink_count = 0
                while len(adapter.buffer):
                    item = adapter.buffer.pop()
                    if item is None:
                        raise RuntimeError("unexpected Null Sink buffer underrun")
                    null_sink.record(item.timestamp, item.values)
                    sink_count += 1
                sink_end = time.perf_counter()
                pipeline_end = sink_end

                timing = {
                    "observation_construct": float(
                        sample.source_metadata["observation_construct_ms"]
                    ),
                    "policy": (policy_end - policy_start) * 1000,
                    "adapter": (adapter_end - adapter_start) * 1000,
                    "null_sink": (sink_end - sink_start) * 1000,
                    "end_to_end": (pipeline_end - pipeline_start) * 1000
                    + float(sample.source_metadata["observation_construct_ms"]),
                }
                for key, value in timing.items():
                    timings[key].append(value)
                deadline_miss = timing["end_to_end"] > period * 1000
                counts["deadline_miss"] += int(deadline_miss)
                counts["inferences"] += 1

                raw_first = {key: raw_chunk[key][0] for key in ACTION_KEYS}
                filtered_first = safe_chunk[0]
                row = {
                    "scenario": args.scenario,
                    "inference_index": counts["inferences"] - 1,
                    "frame_index": sample.frame_index,
                    "wall_time": time.time(),
                    "monotonic_time": sample.monotonic_timestamp,
                    "source_metadata": sample.source_metadata,
                    "feedback": vector_dict(state),
                    "raw_first_target": vector_dict(raw_first),
                    "first_target_delta": vector_dict(
                        {key: raw_first[key] - state[key] for key in ACTION_KEYS}
                    ),
                    "filtered_first_target": vector_dict(filtered_first),
                    "filtered_action_chunk": {
                        key: [np.asarray(point[key]).tolist() for point in safe_chunk]
                        for key in ACTION_KEYS
                    },
                    "raw_action_chunk": vector_dict(raw_chunk),
                    "filter_counters": {
                        key: counter.as_dict() for key, counter in adapter.filter.counters.items()
                    },
                    "null_sink_points": sink_count,
                    "timing_ms": timing,
                    "deadline_miss": deadline_miss,
                    "image_feature": image_feature(sample.flat_observation["video.ego_view"]),
                }
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                records.append(row)

                if counts["inferences"] == 1 or counts["inferences"] % max(
                    1, round(float(config["replanning_hz"]) * 5)
                ) == 0:
                    image = np.asarray(sample.flat_observation["video.ego_view"])
                    cv2.imwrite(
                        str(frame_dir / f"inference_{counts['inferences'] - 1:06d}.jpg"),
                        image[..., ::-1],
                    )
                    print(
                        f"shadow inference={counts['inferences']} "
                        f"latency={timing['end_to_end']:.1f}ms "
                        f"skew={sample.source_metadata['cross_modal_skew_ms']:.1f}ms",
                        flush=True,
                    )
                next_plan += period
                if next_plan < now - period:
                    skipped = int((now - next_plan) / period)
                    counts["scheduler_periods_skipped"] += skipped
                    next_plan = now + period
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"LIVE SHADOW FAILED CLOSED: {failure}", flush=True)
    finally:
        source.stop()

    sink_metrics = null_sink.metrics()
    if sink_metrics["command_publish_attempts"] != 0:
        raise AssertionError("command publish attempt counter is nonzero")
    if sink_metrics["control_ownership_requests"] != 0:
        raise AssertionError("control ownership request counter is nonzero")
    summary = {
        "mode": "LIVE SHADOW MODE",
        "real_hardware_enabled": False,
        "publish_commands": False,
        "shadow_only": True,
        "dry_run": True,
        "scenario": args.scenario,
        "checkpoint": str(args.checkpoint.resolve()),
        "requested_duration_s": args.duration,
        "actual_wall_duration_s": time.time() - started_wall,
        "replanning_hz": float(config["replanning_hz"]),
        "execution_horizon": int(config["execution_horizon"]),
        "control_timeline_hz": float(config["control_timeline_hz"]),
        "null_sink_hz": float(config["null_sink_hz"]),
        "failure": failure,
        "warmup_ms": percentiles_ms(warmups),
        "counts": dict(counts),
        "deadline_miss_ratio": (
            float(counts["deadline_miss"] / counts["inferences"])
            if counts["inferences"]
            else 0.0
        ),
        "timing_ms": {key: percentiles_ms(value) for key, value in timings.items()},
        "source_diagnostics": source.diagnostics(),
        "null_sink": sink_metrics,
        "right_o6_command_count": 0,
        "adapter_metrics": adapter.metrics(),
        "analysis": analyze(records, adapter) if records else {},
        "log": str(log_path),
    }
    json_dump(summary_path, summary)
    print(summary_path, flush=True)
    if failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
