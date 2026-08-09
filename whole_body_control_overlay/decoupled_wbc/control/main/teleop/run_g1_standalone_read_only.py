#!/usr/bin/env python3
"""Read-only G1 standalone preflight. This module has no command transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import time

import numpy as np
import yaml

from decoupled_wbc.control.envs.g1.utils.state_processor import BodyStateProcessor
from decoupled_wbc.control.real_safe import StandaloneRealSafeCore, StandaloneSafetyLimits


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "g1_standalone_real_safe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("duration must be positive")
    values = yaml.safe_load(args.config.read_text())
    limits = StandaloneSafetyLimits.from_mapping(values)

    # Import the DDS factory only inside the read-only executable. No LowCmd or
    # MotionSwitcher module is imported by this entry point.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, args.interface)
    processor = BodyStateProcessor(
        {
            "ENV_TYPE": "real",
            "ROBOT_TYPE": "g1_29dof",
            "NUM_JOINTS": 29,
            "JOINT2MOTOR": list(range(29)),
        }
    )
    core = StandaloneRealSafeCore(
        limits,
        one_time_arm_token=secrets.token_urlsafe(24),
    )

    frequency = float(values["control_frequency_hz"])
    period = 1.0 / frequency
    start = time.monotonic()
    deadline = start + args.duration
    next_tick = start
    accepted = 0
    missing = 0
    max_interval = 0.0
    loop_intervals = []
    lowstate_ages = []
    imu_ages = []
    lowstate_sample_times = []
    imu_sample_times = []
    last_lowstate_time = None
    last_imu_time = None
    q_initial = None
    q_final = None
    dq_final = None
    base_quat_final = None
    base_angular_velocity_final = None
    secondary_quat_final = None
    secondary_angular_velocity_final = None
    q_min = None
    q_max = None
    dq_abs_max = None
    base_quat_norm_range = [float("inf"), float("-inf")]
    secondary_quat_norm_range = [float("inf"), float("-inf")]
    base_angular_velocity_abs_max = np.zeros(3, dtype=np.float64)
    secondary_angular_velocity_abs_max = np.zeros(3, dtype=np.float64)
    observed_mode_machine = set()
    last_tick = None
    while time.monotonic() < deadline:
        tick_started = time.monotonic()
        sample = processor.read_real_safe_snapshot()
        # State receive timestamps are assigned inside Read(), so validation time
        # must be captured afterwards to preserve monotonic causality.
        now = time.monotonic()
        if sample is None:
            missing += 1
        else:
            core.read_only_tick(sample, now)
            accepted += 1
            q = np.asarray(sample.q, dtype=np.float64)
            dq = np.asarray(sample.dq, dtype=np.float64)
            if q_initial is None:
                q_initial = q.copy()
                q_min = q.copy()
                q_max = q.copy()
                dq_abs_max = np.abs(dq)
            else:
                q_min = np.minimum(q_min, q)
                q_max = np.maximum(q_max, q)
                dq_abs_max = np.maximum(dq_abs_max, np.abs(dq))
            q_final = q.copy()
            dq_final = dq.copy()
            base_quat_final = np.asarray(sample.base_quat_wxyz, dtype=np.float64).copy()
            base_angular_velocity_final = np.asarray(
                sample.base_angular_velocity, dtype=np.float64
            ).copy()
            secondary_quat_final = np.asarray(
                sample.secondary_quat_wxyz, dtype=np.float64
            ).copy()
            secondary_angular_velocity_final = np.asarray(
                sample.secondary_angular_velocity, dtype=np.float64
            ).copy()

            lowstate_ages.append(now - sample.lowstate_monotonic)
            imu_ages.append(now - sample.imu_monotonic)
            if sample.lowstate_monotonic != last_lowstate_time:
                lowstate_sample_times.append(sample.lowstate_monotonic)
                last_lowstate_time = sample.lowstate_monotonic
            if sample.imu_monotonic != last_imu_time:
                imu_sample_times.append(sample.imu_monotonic)
                last_imu_time = sample.imu_monotonic

            base_norm = float(np.linalg.norm(sample.base_quat_wxyz))
            secondary_norm = float(np.linalg.norm(sample.secondary_quat_wxyz))
            base_quat_norm_range[0] = min(base_quat_norm_range[0], base_norm)
            base_quat_norm_range[1] = max(base_quat_norm_range[1], base_norm)
            secondary_quat_norm_range[0] = min(
                secondary_quat_norm_range[0], secondary_norm
            )
            secondary_quat_norm_range[1] = max(
                secondary_quat_norm_range[1], secondary_norm
            )
            base_angular_velocity_abs_max = np.maximum(
                base_angular_velocity_abs_max,
                np.abs(sample.base_angular_velocity),
            )
            secondary_angular_velocity_abs_max = np.maximum(
                secondary_angular_velocity_abs_max,
                np.abs(sample.secondary_angular_velocity),
            )
            mode_machine = getattr(processor.robot_low_state, "mode_machine", None)
            if mode_machine is not None:
                observed_mode_machine.add(int(mode_machine))
        if last_tick is not None:
            interval = tick_started - last_tick
            loop_intervals.append(interval)
            max_interval = max(max_interval, interval)
        last_tick = tick_started
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))

    def observed_rate(sample_times: list[float]) -> float | None:
        if len(sample_times) < 2:
            return None
        elapsed = sample_times[-1] - sample_times[0]
        return (len(sample_times) - 1) / elapsed if elapsed > 0 else None

    def percentile(values: list[float], percentile_value: float) -> float | None:
        return float(np.percentile(values, percentile_value)) if values else None

    summary = {
        "phase": core.state.value,
        "read_only": True,
        "motion_switcher_objects_created": 0,
        "lowcmd_publishers_created": 0,
        "lowcmd_write_attempts": 0,
        "dex3_senders_created": 0,
        "accepted_snapshots": accepted,
        "missing_snapshots": missing,
        "lowstate_new_samples": len(lowstate_sample_times),
        "secondary_imu_new_samples": len(imu_sample_times),
        "lowstate_observed_rate_hz": observed_rate(lowstate_sample_times),
        "secondary_imu_observed_rate_hz": observed_rate(imu_sample_times),
        "lowstate_age_s": {
            "p99": percentile(lowstate_ages, 99),
            "max": max(lowstate_ages) if lowstate_ages else None,
        },
        "secondary_imu_age_s": {
            "p99": percentile(imu_ages, 99),
            "max": max(imu_ages) if imu_ages else None,
        },
        "loop_interval_s": {
            "mean": float(np.mean(loop_intervals)) if loop_intervals else None,
            "p99": percentile(loop_intervals, 99),
            "max": max_interval,
        },
        "read_only_watchdog_pass": bool(
            loop_intervals
            and max_interval <= limits.max_control_interval_s
            and lowstate_ages
            and max(lowstate_ages) <= limits.lowstate_stale_s
            and imu_ages
            and max(imu_ages) <= limits.imu_stale_s
        ),
        "max_loop_interval_s": max_interval,
        "duration_s": time.monotonic() - start,
        "interface": args.interface,
        "observed_mode_machine": sorted(observed_mode_machine),
        "q_initial_rad": None if q_initial is None else q_initial.tolist(),
        "q_final_rad": None if q_final is None else q_final.tolist(),
        "dq_final_rad_s": None if dq_final is None else dq_final.tolist(),
        "base_quaternion_final_wxyz": None
        if base_quat_final is None
        else base_quat_final.tolist(),
        "base_angular_velocity_final_rad_s": None
        if base_angular_velocity_final is None
        else base_angular_velocity_final.tolist(),
        "secondary_quaternion_final_wxyz": None
        if secondary_quat_final is None
        else secondary_quat_final.tolist(),
        "secondary_angular_velocity_final_rad_s": None
        if secondary_angular_velocity_final is None
        else secondary_angular_velocity_final.tolist(),
        "q_min_rad": None if q_min is None else q_min.tolist(),
        "q_max_rad": None if q_max is None else q_max.tolist(),
        "q_peak_to_peak_rad": None
        if q_min is None
        else (q_max - q_min).tolist(),
        "dq_abs_max_rad_s": None if dq_abs_max is None else dq_abs_max.tolist(),
        "base_quaternion_norm_range": base_quat_norm_range
        if accepted
        else None,
        "secondary_quaternion_norm_range": secondary_quat_norm_range
        if accepted
        else None,
        "base_angular_velocity_abs_max_rad_s": base_angular_velocity_abs_max.tolist(),
        "secondary_angular_velocity_abs_max_rad_s": (
            secondary_angular_velocity_abs_max.tolist()
        ),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(payload + "\n")
    if accepted == 0:
        raise SystemExit("no valid synchronized lowstate/secondary-IMU snapshot received")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
