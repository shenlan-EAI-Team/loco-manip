#!/usr/bin/env python3
"""Offline grouped open-loop metrics for a GR00T N1.7 checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy


ACTION_KEYS = ("left_arm", "right_arm", "left_o6", "right_o6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    error = pred.astype(np.float64) - gt.astype(np.float64)
    return {
        "frames": int(gt.shape[0]),
        "dimensions": int(gt.shape[1]),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tag = EmbodimentTag.resolve("NEW_EMBODIMENT")
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=str(args.checkpoint),
        device="cuda",
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(str(args.dataset), modality_configs=modality)

    observation_modality = dict(modality)
    observation_modality.pop("action")
    aggregate_gt: dict[str, list[np.ndarray]] = defaultdict(list)
    aggregate_pred: dict[str, list[np.ndarray]] = defaultdict(list)
    episode_results: list[dict[str, object]] = []

    for episode_id in range(len(loader)):
        trajectory = loader[episode_id]
        frame_count = len(trajectory)
        predicted: dict[str, list[np.ndarray]] = defaultdict(list)

        for step in range(0, frame_count, args.execution_horizon):
            point = extract_step_data(trajectory, step, observation_modality, tag)
            observation = {f"state.{key}": value for key, value in point.states.items()}
            observation.update(
                {f"video.{key}": np.asarray(value) for key, value in point.images.items()}
            )
            for language_key in modality["language"].modality_keys:
                observation[language_key] = point.text
            parsed = parse_observation_gr00t(observation, modality)
            action, _ = policy.get_action(parsed)
            take = min(args.execution_horizon, frame_count - step)
            for key in ACTION_KEYS:
                predicted[key].append(np.asarray(action[key][0])[:take])

        per_key: dict[str, dict[str, float | int]] = {}
        save_arrays: dict[str, np.ndarray] = {}
        for key in ACTION_KEYS:
            gt = np.vstack(trajectory[f"action.{key}"].to_numpy()).astype(np.float32)
            pred = np.concatenate(predicted[key], axis=0).astype(np.float32)
            if gt.shape != pred.shape:
                raise RuntimeError(f"{key}: gt={gt.shape}, pred={pred.shape}")
            per_key[key] = metrics(gt, pred)
            aggregate_gt[key].append(gt)
            aggregate_pred[key].append(pred)
            save_arrays[f"gt_{key}"] = gt
            save_arrays[f"pred_{key}"] = pred

        np.savez_compressed(args.output_dir / f"episode_{episode_id}_predictions.npz", **save_arrays)
        episode_results.append(
            {"local_episode": episode_id, "frames": frame_count, "modalities": per_key}
        )

    aggregate = {
        key: metrics(np.concatenate(aggregate_gt[key]), np.concatenate(aggregate_pred[key]))
        for key in ACTION_KEYS
    }
    all_gt = np.concatenate(
        [np.concatenate(aggregate_gt[key], axis=0) for key in ACTION_KEYS], axis=1
    )
    all_pred = np.concatenate(
        [np.concatenate(aggregate_pred[key], axis=0) for key in ACTION_KEYS], axis=1
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "execution_horizon": args.execution_horizon,
        "denoising_steps": args.denoising_steps,
        "seed": args.seed,
        "episodes": episode_results,
        "aggregate_by_modality": aggregate,
        "aggregate_all_raw_units": metrics(all_gt, all_pred),
        "warning": "Raw-unit aggregate mixes arm radians and O6 percentages; compare per modality.",
    }
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate_by_modality"], indent=2))


if __name__ == "__main__":
    main()
