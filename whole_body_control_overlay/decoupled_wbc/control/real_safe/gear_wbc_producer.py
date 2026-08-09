"""Read-only 50 Hz Gear WBC inference producer for G1 commissioning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
from typing import Callable, Protocol

import numpy as np
import yaml

from .lowcmd_guard.commissioning import LowerBodyMailbox
from .lowcmd_guard.core import GuardSnapshot
from .standalone import SafetyFault, StandaloneSafetyGate
from ..utils.gear_wbc_utils import get_gravity_orientation


class GuardSnapshotSource(Protocol):
    def latest(self, now: float) -> GuardSnapshot: ...


@dataclass(frozen=True)
class GearWbcModelConfig:
    default_angles: np.ndarray
    cmd: np.ndarray
    height: float
    rpy: np.ndarray
    cmd_scale: np.ndarray
    angular_velocity_scale: float
    position_scale: float
    velocity_scale: float
    action_scale: float
    history_length: int
    observation_size: int

    @classmethod
    def from_yaml(cls, path: Path) -> "GearWbcModelConfig":
        values = yaml.safe_load(path.read_text())
        config = cls(
            default_angles=np.asarray(values["default_angles"], dtype=np.float32),
            cmd=np.asarray(values["cmd_init"], dtype=np.float32),
            height=float(values["height_cmd"]),
            rpy=np.asarray(values["rpy_cmd"], dtype=np.float32),
            cmd_scale=np.asarray(values["cmd_scale"], dtype=np.float32),
            angular_velocity_scale=float(values["ang_vel_scale"]),
            position_scale=float(values["dof_pos_scale"]),
            velocity_scale=float(values["dof_vel_scale"]),
            action_scale=float(values["action_scale"]),
            history_length=int(values["obs_history_len"]),
            observation_size=int(values["num_obs"]),
        )
        if config.default_angles.shape != (15,):
            raise ValueError("Gear WBC default_angles must be shape (15,)")
        if config.cmd.shape != (3,) or config.rpy.shape != (3,) or config.cmd_scale.shape != (3,):
            raise ValueError("Gear WBC command/rpy arrays must be shape (3,)")
        if config.history_length != 6 or config.observation_size != 516:
            raise ValueError("commissioning requires the audited 6x86 = 516 observation contract")
        scalars = np.asarray(
            [
                config.height,
                config.angular_velocity_scale,
                config.position_scale,
                config.velocity_scale,
                config.action_scale,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("Gear WBC model configuration contains non-finite values")
        return config


class GearWbcStandingModel:
    """Exact audited 516D observation and 15D absolute-q target contract."""

    def __init__(
        self,
        config: GearWbcModelConfig,
        inference: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.config = config
        self.inference = inference
        self.previous_action = np.zeros(15, dtype=np.float32)
        self.history: deque[np.ndarray] = deque(maxlen=config.history_length)

    @classmethod
    def from_onnx(cls, config_path: Path, model_path: Path) -> "GearWbcStandingModel":
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape[-1] != 516:
            raise ValueError("Gear WBC Balance ONNX input must be 516D")
        if len(outputs) != 1 or outputs[0].shape[-1] != 15:
            raise ValueError("Gear WBC Balance ONNX output must be 15D")
        input_name = inputs[0].name

        def infer(value: np.ndarray) -> np.ndarray:
            return np.asarray(session.run(None, {input_name: value})[0])

        return cls(GearWbcModelConfig.from_yaml(config_path), infer)

    def target(self, snapshot: GuardSnapshot) -> np.ndarray:
        robot = snapshot.robot
        q = np.asarray(robot.q, dtype=np.float32)
        dq = np.asarray(robot.dq, dtype=np.float32)
        quat = np.asarray(robot.base_quat_wxyz, dtype=np.float32)
        omega = np.asarray(robot.base_angular_velocity, dtype=np.float32)
        if q.shape != (29,) or dq.shape != (29,) or quat.shape != (4,) or omega.shape != (3,):
            raise SafetyFault("Gear WBC state contract is not 29DoF plus base IMU")
        if not all(np.isfinite(value).all() for value in (q, dq, quat, omega)):
            raise SafetyFault("Gear WBC state contains non-finite values")

        defaults = np.zeros(29, dtype=np.float32)
        defaults[:15] = self.config.default_angles
        observation = np.zeros(86, dtype=np.float32)
        observation[0:3] = self.config.cmd * self.config.cmd_scale
        observation[3] = self.config.height
        observation[4:7] = self.config.rpy
        observation[7:10] = omega * self.config.angular_velocity_scale
        observation[10:13] = get_gravity_orientation(quat)
        observation[13:42] = (q - defaults) * self.config.position_scale
        observation[42:71] = dq * self.config.velocity_scale
        observation[71:86] = self.previous_action
        self.history.append(observation)
        padded = [np.zeros(86, dtype=np.float32)] * (
            self.config.history_length - len(self.history)
        ) + list(self.history)
        model_input = np.concatenate(padded).reshape(1, self.config.observation_size)
        action = np.asarray(self.inference(model_input), dtype=np.float32).reshape(-1)
        if action.shape != (15,) or not np.isfinite(action).all():
            raise SafetyFault("Gear WBC ONNX output must be finite shape (15,)")
        self.previous_action = action.copy()
        return (
            action.astype(np.float64) * self.config.action_scale
            + self.config.default_angles.astype(np.float64)
        )


class GearWbcReadOnlyProducer:
    """Reads cached state and publishes only to memory; it has no DDS writer imports."""

    def __init__(
        self,
        source: GuardSnapshotSource,
        safety: StandaloneSafetyGate,
        model: GearWbcStandingModel,
        mailbox: LowerBodyMailbox,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.source = source
        self.safety = safety
        self.model = model
        self.mailbox = mailbox
        self.clock = clock
        self.sequence = 0
        self.inference_count = 0

    def tick(self) -> None:
        before = self.clock()
        snapshot = self.source.latest(before)
        now = self.clock()
        self.safety.validate_snapshot(snapshot.robot, now)
        target = self.model.target(snapshot)
        limits = self.safety.limits
        if np.any(target < limits.q_lower[:15]) or np.any(target > limits.q_upper[:15]):
            raise SafetyFault("Gear WBC target is outside audited lower-body hard limits")
        self.sequence += 1
        self.mailbox.publish(target, timestamp=now, sequence=self.sequence)
        self.inference_count += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_manifest(path: Path, *, repository_root: Path) -> dict[str, str]:
    values = json.loads(path.read_text())
    if values.get("schema_version") != 1:
        raise ValueError("unsupported commissioning artifact manifest")
    locked = values.get("sha256")
    if not isinstance(locked, dict) or not locked:
        raise ValueError("commissioning artifact manifest has no hashes")
    for relative, expected in locked.items():
        actual = sha256_file(repository_root / relative)
        if actual != expected:
            raise RuntimeError(f"commissioning artifact hash mismatch: {relative}")
    return dict(locked)
