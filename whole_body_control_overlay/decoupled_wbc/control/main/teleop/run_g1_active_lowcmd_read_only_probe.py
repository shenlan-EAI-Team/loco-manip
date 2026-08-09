#!/usr/bin/env python3
"""Read-only probe of the active ai LowCmd stream and matching LowState."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Lock
import time

import numpy as np


def stats(values: np.ndarray) -> dict[str, object]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
    }


def interval_stats(timestamps: np.ndarray) -> dict[str, float | int | None]:
    if timestamps.size < 2:
        return {"samples": 0, "mean_s": None, "p99_s": None, "max_s": None}
    intervals = np.diff(timestamps)
    return {
        "samples": int(intervals.size),
        "mean_s": float(np.mean(intervals)),
        "p99_s": float(np.percentile(intervals, 99)),
        "max_s": float(np.max(intervals)),
        "effective_frequency_hz": float(1.0 / np.mean(intervals)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.duration <= 0:
        raise SystemExit("duration must be positive")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(0, args.interface)
    crc = CRC()
    lock = Lock()
    commands: list[dict[str, object]] = []
    states: list[dict[str, object]] = []
    command_crc_errors = 0
    state_crc_errors = 0

    def on_command(message) -> None:
        nonlocal command_crc_errors
        now = time.monotonic()
        if int(message.crc) != int(crc.Crc(message)):
            command_crc_errors += 1
            return
        motors = message.motor_cmd
        sample = {
            "t": now,
            "publication_handle": int(message.sample_info.publication_handle),
            "source_timestamp_ns": int(message.sample_info.source_timestamp),
            "mode_pr": int(message.mode_pr),
            "mode_machine": int(message.mode_machine),
            "mode": [int(motors[i].mode) for i in range(35)],
            "q": [float(motors[i].q) for i in range(35)],
            "dq": [float(motors[i].dq) for i in range(35)],
            "kp": [float(motors[i].kp) for i in range(35)],
            "kd": [float(motors[i].kd) for i in range(35)],
            "tau": [float(motors[i].tau) for i in range(35)],
        }
        with lock:
            commands.append(sample)

    def on_state(message) -> None:
        nonlocal state_crc_errors
        now = time.monotonic()
        if int(message.crc) != int(crc.Crc(message)):
            state_crc_errors += 1
            return
        motors = message.motor_state
        sample = {
            "t": now,
            "mode_machine": int(message.mode_machine),
            "q": [float(motors[i].q) for i in range(29)],
            "dq": [float(motors[i].dq) for i in range(29)],
            "tau_est": [float(motors[i].tau_est) for i in range(29)],
        }
        with lock:
            states.append(sample)

    command_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    command_subscriber.Init(on_command, 10)
    state_subscriber.Init(on_state, 10)
    time.sleep(args.duration)
    command_subscriber.Close()
    state_subscriber.Close()

    with lock:
        command_samples = list(commands)
        state_samples = list(states)
    if not command_samples or not state_samples:
        raise RuntimeError(
            f"missing read-only samples: lowcmd={len(command_samples)}, lowstate={len(state_samples)}"
        )

    command_t = np.asarray([sample["t"] for sample in command_samples])
    state_t = np.asarray([sample["t"] for sample in state_samples])
    command_fields = {
        name: np.asarray([sample[name] for sample in command_samples], dtype=np.float64)
        for name in ("q", "dq", "kp", "kd", "tau")
    }
    state_fields = {
        name: np.asarray([sample[name] for sample in state_samples], dtype=np.float64)
        for name in ("q", "dq", "tau_est")
    }
    nearest_state = np.searchsorted(state_t, command_t, side="left")
    nearest_state = np.clip(nearest_state, 0, len(state_t) - 1)
    previous = np.maximum(nearest_state - 1, 0)
    choose_previous = np.abs(state_t[previous] - command_t) < np.abs(
        state_t[nearest_state] - command_t
    )
    nearest_state[choose_previous] = previous[choose_previous]
    q_error = command_fields["q"][:, :29] - state_fields["q"][nearest_state]

    summary = {
        "schema_version": 1,
        "read_only": True,
        "lowcmd_publishers_created": 0,
        "lowcmd_writes": 0,
        "release_calls": 0,
        "select_calls": 0,
        "duration_s": args.duration,
        "command_crc_errors": command_crc_errors,
        "state_crc_errors": state_crc_errors,
        "lowcmd_timing": interval_stats(command_t),
        "lowstate_timing": interval_stats(state_t),
        "mode_pr_values": sorted({int(sample["mode_pr"]) for sample in command_samples}),
        "lowcmd_publication_handles": sorted(
            {int(sample["publication_handle"]) for sample in command_samples}
        ),
        "command_mode_machine_values": sorted(
            {int(sample["mode_machine"]) for sample in command_samples}
        ),
        "state_mode_machine_values": sorted(
            {int(sample["mode_machine"]) for sample in state_samples}
        ),
        "motor_mode_values_by_slot": [
            sorted({int(sample["mode"][i]) for sample in command_samples}) for i in range(35)
        ],
        "command": {name: stats(value) for name, value in command_fields.items()},
        "state": {name: stats(value) for name, value in state_fields.items()},
        "command_q_minus_nearest_state_q_rad": stats(q_error),
        "nearest_state_skew_s": stats(
            np.abs(command_t - state_t[nearest_state]).reshape(-1, 1)
        ),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
