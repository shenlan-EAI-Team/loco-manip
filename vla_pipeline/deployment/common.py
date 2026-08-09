"""Shared, hardware-free helpers for GR00T replay and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy


PROJECT_ROOT = Path("/home/slxy/下载/g1_o6_gr00t")
CHECKPOINT = PROJECT_ROOT / "outputs/formal_train_26_corrected_v1/checkpoint-3000"
CORRECTED_CHECKPOINT = CHECKPOINT
VAL_DATASET = PROJECT_ROOT / "datasets_corrected_v1/val_2"
TEST_DATASET = PROJECT_ROOT / "datasets_corrected_v1/test_2"
ACTION_KEYS = ("left_arm", "right_arm", "left_o6", "right_o6")
STATE_KEYS = (
    "left_arm",
    "right_arm",
    "left_o6",
    "right_o6",
    "waist",
    "projected_gravity",
)
ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
O6_JOINTS = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(
    denoising_steps: int = 4,
    checkpoint: str | Path | None = None,
) -> Gr00tPolicy:
    model_path = CHECKPOINT if checkpoint is None else Path(checkpoint).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"policy checkpoint does not exist: {model_path}")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(model_path),
        device="cuda",
    )
    policy.model.action_head.num_inference_timesteps = denoising_steps
    return policy


def make_loader(dataset: Path, modality_configs: dict[str, Any]) -> LeRobotEpisodeLoader:
    return LeRobotEpisodeLoader(str(dataset), modality_configs=modality_configs)


def observation_modalities(modality_configs: dict[str, Any]) -> dict[str, Any]:
    result = dict(modality_configs)
    result.pop("action", None)
    return result


def build_flat_observation(
    trajectory: Any,
    frame_index: int,
    modality_configs: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    point = extract_step_data(
        trajectory,
        frame_index,
        observation_modalities(modality_configs),
        EmbodimentTag.NEW_EMBODIMENT,
    )
    flat: dict[str, Any] = {
        f"state.{key}": np.asarray(value, dtype=np.float32)
        for key, value in point.states.items()
    }
    flat.update({f"video.{key}": np.asarray(value) for key, value in point.images.items()})
    for language_key in modality_configs["language"].modality_keys:
        flat[language_key] = point.text
    return flat, point


def build_policy_observation(
    trajectory: Any,
    frame_index: int,
    modality_configs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    flat, point = build_flat_observation(trajectory, frame_index, modality_configs)
    return parse_observation_gr00t(flat, modality_configs), flat, point


def unbatch_action(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for key in ACTION_KEYS:
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f"Expected {key} shape (1,T,D), got {value.shape}")
        result[key] = value[0]
    return result


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=default) + "\n",
        encoding="utf-8",
    )


def array_stats(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "finite": bool(np.isfinite(arr).all()),
    }
