#!/usr/bin/env python3
"""Offline Gear WBC input/decode diagnosis from a read-only G1 state report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from decoupled_wbc.control.real_safe.gear_wbc_producer import (
    GearWbcModelConfig,
    GearWbcStandingModel,
)
from decoupled_wbc.control.real_safe.lowcmd_guard.core import GuardSnapshot
from decoupled_wbc.control.real_safe.standalone import RobotSnapshot
from decoupled_wbc.control.utils.gear_wbc_utils import get_gravity_orientation


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = (
    ROOT
    / "decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx"
)
DEFAULT_MODEL_CONFIG = (
    ROOT / "decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml"
)
DEFAULT_SAFETY_CONFIG = Path(__file__).with_name("configs") / "g1_standalone_real_safe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--safety-config", type=Path, default=DEFAULT_SAFETY_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def array(report: dict[str, object], key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(report[key], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{key} must be finite shape {shape}")
    return value


def snapshot(
    q: np.ndarray,
    dq: np.ndarray,
    quat: np.ndarray,
    omega: np.ndarray,
) -> GuardSnapshot:
    robot = RobotSnapshot(
        q=q.copy(),
        dq=dq.copy(),
        base_quat_wxyz=quat.copy(),
        base_angular_velocity=omega.copy(),
        secondary_quat_wxyz=quat.copy(),
        secondary_angular_velocity=omega.copy(),
        lowstate_monotonic=1.0,
        imu_monotonic=1.0,
    )
    return GuardSnapshot(
        robot=robot,
        mode_machine=5,
        motor_modes=np.zeros(29, dtype=np.int64),
        motor_errors=np.zeros(29, dtype=np.int64),
        motor_tau_est=np.zeros(29),
    )


def rollout(
    config: GearWbcModelConfig,
    config_path: Path,
    model_path: Path,
    state: GuardSnapshot,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> dict[str, object]:
    model = GearWbcStandingModel.from_onnx(config_path=config_path, model_path=model_path)
    frames = []
    for index in range(10):
        target = model.target(state)
        raw = (target - config.default_angles.astype(np.float64)) / config.action_scale
        violations = np.flatnonzero((target < q_lower[:15]) | (target > q_upper[:15]))
        frames.append(
            {
                "history_frames": index + 1,
                "raw_action": raw.tolist(),
                "decoded_absolute_q_rad": target.tolist(),
                "delta_from_input_q_rad": (target - state.robot.q[:15]).tolist(),
                "hard_limit_violation_indices": violations.tolist(),
                "max_abs_delta_from_input_q_rad": float(
                    np.max(np.abs(target - state.robot.q[:15]))
                ),
                "max_abs_delta_motor_index": int(
                    np.argmax(np.abs(target - state.robot.q[:15]))
                ),
            }
        )
    return {
        "first_frame": frames[0],
        "full_history_frame": frames[5],
        "preflight_final_frame": frames[-1],
        "all_preflight_frames": frames,
    }


def main() -> int:
    args = parse_args()
    report = json.loads(args.state_report.read_text())
    config = GearWbcModelConfig.from_yaml(args.model_config)
    safety = yaml.safe_load(args.safety_config.read_text())
    q_lower = np.asarray(safety["q_lower"], dtype=np.float64)
    q_upper = np.asarray(safety["q_upper"], dtype=np.float64)

    live_q = array(report, "q_final_rad", (29,))
    live_dq = array(report, "dq_final_rad_s", (29,))
    live_quat = array(report, "base_quaternion_final_wxyz", (4,))
    live_omega = array(report, "base_angular_velocity_final_rad_s", (3,))
    live_quat = live_quat / np.linalg.norm(live_quat)

    default_q = np.zeros(29, dtype=np.float64)
    default_q[:15] = config.default_angles
    zeros_dq = np.zeros(29, dtype=np.float64)
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    zeros_omega = np.zeros(3, dtype=np.float64)

    variants: dict[str, GuardSnapshot] = {
        "policy_default_pose": snapshot(default_q, zeros_dq, identity_quat, zeros_omega),
        "policy_default_pose_live_imu": snapshot(
            default_q, zeros_dq, live_quat, live_omega
        ),
        "live_recorded": snapshot(live_q, live_dq, live_quat, live_omega),
        "live_pose_zero_velocity_identity_imu": snapshot(
            live_q, zeros_dq, identity_quat, zeros_omega
        ),
    }
    lower_default = live_q.copy()
    lower_default[:15] = config.default_angles
    variants["live_with_lower_body_default"] = snapshot(
        lower_default, zeros_dq, live_quat, live_omega
    )
    motor4_default = live_q.copy()
    motor4_default[4] = config.default_angles[4]
    variants["live_with_motor4_default"] = snapshot(
        motor4_default, zeros_dq, live_quat, live_omega
    )
    ankles_default = live_q.copy()
    ankles_default[[4, 10]] = config.default_angles[[4, 10]]
    variants["live_with_both_ankles_default"] = snapshot(
        ankles_default, zeros_dq, live_quat, live_omega
    )
    for fraction in (0.25, 0.5, 0.75, 0.9):
        blended = live_q.copy()
        blended[:15] += fraction * (config.default_angles - blended[:15])
        variants[f"live_lower_to_default_{int(fraction * 100)}pct"] = snapshot(
            blended, zeros_dq, live_quat, live_omega
        )

    results = {
        name: rollout(config, args.model_config, args.model, state, q_lower, q_upper)
        for name, state in variants.items()
    }
    payload = {
        "schema_version": 1,
        "read_only_offline": True,
        "state_report": str(args.state_report.resolve()),
        "model": str(args.model.resolve()),
        "model_config": str(args.model_config.resolve()),
        "contract": {
            "observation": "6x86=516, oldest-to-newest, startup left-zero-padded",
            "action": "raw15 * action_scale + default_angles exactly once",
            "action_scale": config.action_scale,
            "output_motor_indices": list(range(15)),
        },
        "live_input": {
            "q_rad": live_q.tolist(),
            "dq_rad_s": live_dq.tolist(),
            "base_quaternion_wxyz": live_quat.tolist(),
            "base_angular_velocity_rad_s": live_omega.tolist(),
            "projected_gravity": get_gravity_orientation(live_quat).tolist(),
        },
        "policy_default_lower_q_rad": config.default_angles.tolist(),
        "hard_limits_lower_rad": {
            "lower": q_lower[:15].tolist(),
            "upper": q_upper[:15].tolist(),
        },
        "variants": results,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
