#!/usr/bin/env python3
"""Replay Policy API -> Adapter -> filters/buffer -> Mock SDK, never hardware."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from deployment.action_adapter import ActionAdapter
from deployment.common import (
    ACTION_KEYS,
    CHECKPOINT,
    PROJECT_ROOT,
    TEST_DATASET,
    build_policy_observation,
    json_dump,
    load_policy,
    make_loader,
)
from deployment.mock import MockG1SDK, MockO6SDK


def plot_before_after(raw: np.ndarray, safe: np.ndarray, title: str, output: Path) -> None:
    fig, axes = plt.subplots(raw.shape[1], 1, figsize=(12, 2.0 * raw.shape[1]), sharex=True)
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(raw[:, index], label="Policy API", alpha=0.7)
        axis.plot(safe[:, index], label="Adapter", lw=1.0)
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("30 Hz action point")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def kinematics(
    values: np.ndarray, episode_ranges: list[tuple[int, int]], hz: float = 30.0
) -> dict[str, float]:
    velocity = np.concatenate(
        [np.diff(values[start:end], axis=0) * hz for start, end in episode_ranges], axis=0
    )
    acceleration = np.concatenate(
        [
            np.diff(values[start:end], n=2, axis=0) * hz**2
            for start, end in episode_ranges
        ],
        axis=0,
    )
    return {
        "max_abs_velocity": float(np.max(np.abs(velocity))),
        "p95_abs_velocity": float(np.percentile(np.abs(velocity), 95)),
        "max_abs_acceleration": float(np.max(np.abs(acceleration))),
        "p95_abs_acceleration": float(np.percentile(np.abs(acceleration), 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "deployment/config/adapter.yaml"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "deployment")
    args = parser.parse_args()
    policy = load_policy(4)
    modality = policy.get_modality_config()
    loader = make_loader(TEST_DATASET, modality)
    g1 = MockG1SDK()
    o6 = MockO6SDK()
    adapter = ActionAdapter(args.config, g1_sdk=g1, o6_sdk=o6)

    raw_values: dict[str, list[np.ndarray]] = defaultdict(list)
    safe_values: dict[str, list[np.ndarray]] = defaultdict(list)
    boundary_jumps_before: dict[str, list[float]] = defaultdict(list)
    boundary_jumps_after: dict[str, list[float]] = defaultdict(list)
    previous_raw: dict[str, np.ndarray] | None = None
    previous_safe: dict[str, np.ndarray] | None = None
    cumulative_filter_counts: dict[str, dict[str, int]] = {
        key: defaultdict(int) for key in ACTION_KEYS
    }
    episode_ranges: list[tuple[int, int]] = []

    for episode_id in range(len(loader)):
        trajectory = loader[episode_id]
        episode_start = len(raw_values["left_arm"])
        previous_raw = None
        previous_safe = None
        initial = {
            key: np.asarray(trajectory[f"state.{key}"].iloc[0], dtype=np.float32)
            for key in ACTION_KEYS
        }
        adapter.reset(initial)
        for frame in range(0, len(trajectory), adapter.config["execution_horizon"]):
            observation, _, _ = build_policy_observation(trajectory, frame, modality)
            action, _ = policy.get_action(observation)
            raw_chunk = {key: np.asarray(action[key][0]) for key in ACTION_KEYS}
            safe_chunk = adapter.prepare_chunk(action, timestamp=float(frame) / 30.0)
            adapter.drain_to_mock()
            execute = len(safe_chunk)
            for offset in range(execute):
                raw_step = {key: raw_chunk[key][offset] for key in ACTION_KEYS}
                safe_step = safe_chunk[offset]
                for key in ACTION_KEYS:
                    raw_values[key].append(raw_step[key].copy())
                    safe_values[key].append(safe_step[key].copy())
                    if offset == 0 and previous_raw is not None:
                        boundary_jumps_before[key].append(
                            float(np.max(np.abs(raw_step[key] - previous_raw[key])))
                        )
                        boundary_jumps_after[key].append(
                            float(np.max(np.abs(safe_step[key] - previous_safe[key])))
                        )
                previous_raw = raw_step
                previous_safe = safe_step
        for key, counter in adapter.filter.counters.items():
            for counter_name, value in counter.as_dict().items():
                cumulative_filter_counts[key][counter_name] += value
        episode_ranges.append((episode_start, len(raw_values["left_arm"])))

    # Explicit failure injection tests use a fresh adapter and never leave mock memory.
    fault_g1 = MockG1SDK()
    fault_o6 = MockO6SDK()
    fault_adapter = ActionAdapter(args.config, g1_sdk=fault_g1, o6_sdk=fault_o6)
    zero_state = {
        "left_arm": np.zeros(7, dtype=np.float32),
        "right_arm": np.zeros(7, dtype=np.float32),
        "left_o6": np.zeros(6, dtype=np.float32),
        "right_o6": np.zeros(6, dtype=np.float32),
    }
    fault_adapter.reset(zero_state)
    safe_action = {
        key: np.zeros((1, 16, len(value)), dtype=np.float32)
        for key, value in zero_state.items()
    }
    nan_action = {key: value.copy() for key, value in safe_action.items()}
    nan_action["left_arm"][0, 0, 0] = np.nan
    nan_filtered = fault_adapter.prepare_chunk(nan_action, timestamp=0.0)
    nan_blocked = all(np.isfinite(step["left_arm"]).all() for step in nan_filtered)
    fault_adapter.drain_to_mock()

    fault_g1.inject_timeout = True
    fault_adapter.prepare_chunk(safe_action, timestamp=1.0)
    fault_adapter.drain_to_mock()
    timeout_recorded = any("timeout" in value for value in fault_adapter.sdk_errors)
    fault_g1.inject_timeout = False
    fault_o6.connected = False
    fault_adapter.prepare_chunk(safe_action, timestamp=2.0)
    fault_adapter.drain_to_mock()
    disconnect_recorded = any("disconnected" in value for value in fault_adapter.sdk_errors)
    stale_now = fault_adapter.last_policy_time + fault_adapter.config["watchdog_timeout_s"] + 1.0
    watchdog_target = fault_adapter.watchdog_target(now=stale_now)
    watchdog_holds_last = all(
        np.array_equal(watchdog_target[key], fault_adapter.last_safe[key]) for key in ACTION_KEYS
    )
    estop_target = fault_adapter.emergency_stop()
    estop_holds_last = all(
        np.array_equal(estop_target[key], fault_adapter.last_safe[key]) for key in ACTION_KEYS
    )
    fault_adapter.buffer.pop()

    plots = args.output_dir / "plots/adapter_dry_run"
    per_group = {}
    for key in ACTION_KEYS:
        raw = np.stack(raw_values[key])
        safe = np.stack(safe_values[key])
        plot_before_after(raw, safe, key, plots / f"{key}_before_after.png")
        per_group[key] = {
            "policy_api": kinematics(raw, episode_ranges),
            "adapter": kinematics(safe, episode_ranges),
            "max_chunk_boundary_jump_before": max(boundary_jumps_before[key], default=0.0),
            "max_chunk_boundary_jump_after": max(boundary_jumps_after[key], default=0.0),
        }

    adapter_metrics = adapter.metrics()
    adapter_metrics["filter_counters"] = {
        key: dict(values) for key, values in cumulative_filter_counts.items()
    }
    result = {
        "mode": "DRY RUN: MOCK SDKS ONLY; NO COMMANDS SENT",
        "dataset": str(TEST_DATASET),
        "checkpoint": str(CHECKPOINT),
        "pipeline": "Policy API -> Action Adapter -> Safety Filter -> Action Buffer -> Mock SDK",
        "per_group": per_group,
        "adapter_metrics": adapter_metrics,
        "fault_tests": {
            "nan_injection_filtered_to_finite": nan_blocked,
            "timeout_recorded_and_hold": timeout_recorded,
            "network_disconnect_recorded_and_hold": disconnect_recorded,
            "watchdog_holds_last_safe_target": watchdog_holds_last,
            "emergency_stop_holds_last_safe_target": estop_holds_last,
            "empty_buffer_underrun_recorded": fault_adapter.buffer.underruns > 0,
            "fault_adapter_metrics": fault_adapter.metrics(),
        },
    }
    json_dump(args.output_dir / "adapter_dry_run_metrics.json", result)
    print(args.output_dir / "adapter_dry_run_metrics.json")


if __name__ == "__main__":
    main()
