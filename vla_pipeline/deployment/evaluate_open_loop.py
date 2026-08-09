#!/usr/bin/env python3
"""Reproducible test_2/val_2 open-loop and continuity evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from deployment.common import (
    ACTION_KEYS,
    ARM_JOINTS,
    CHECKPOINT,
    O6_JOINTS,
    PROJECT_ROOT,
    TEST_DATASET,
    VAL_DATASET,
    build_policy_observation,
    json_dump,
    load_policy,
    make_loader,
    seed_everything,
    unbatch_action,
)


def error_metrics(gt: np.ndarray, pred: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    error = pred.astype(np.float64) - gt.astype(np.float64)
    result = {}
    for index, name in enumerate(names):
        joint_error = error[:, index]
        result[name] = {
            "mae": float(np.mean(np.abs(joint_error))),
            "rmse": float(np.sqrt(np.mean(joint_error**2))),
            "max_absolute_error": float(np.max(np.abs(joint_error))),
        }
    return result


def abs_distribution(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    if not absolute.size:
        return {"mean_abs": 0.0, "p95_abs": 0.0, "max_abs": 0.0}
    return {
        "mean_abs": float(np.mean(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def plot_group(
    gt: np.ndarray,
    pred: np.ndarray,
    names: tuple[str, ...],
    title: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.2 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for index, (axis, name) in enumerate(zip(axes, names)):
        axis.plot(gt[:, index], label="ground truth", lw=1.0)
        axis.plot(pred[:, index], label="prediction", lw=0.9, alpha=0.8)
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("30 Hz frame")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def evaluate_dataset(
    policy: Any,
    dataset: Path,
    *,
    denoising_steps: int,
    execution_horizon: int,
    plot_dir: Path | None,
    seed: int,
) -> dict[str, Any]:
    policy.model.action_head.num_inference_timesteps = denoising_steps
    modality = policy.get_modality_config()
    loader = make_loader(dataset, modality)
    episode_results = []
    aggregate_gt: dict[str, list[np.ndarray]] = defaultdict(list)
    aggregate_pred: dict[str, list[np.ndarray]] = defaultdict(list)
    boundary_jumps: dict[str, list[np.ndarray]] = defaultdict(list)
    adjacent_jumps: dict[str, list[np.ndarray]] = defaultdict(list)
    velocities: dict[str, list[np.ndarray]] = defaultdict(list)
    accelerations: dict[str, list[np.ndarray]] = defaultdict(list)

    seed_everything(seed)
    for episode_id in range(len(loader)):
        trajectory = loader[episode_id]
        predictions: dict[str, list[np.ndarray]] = defaultdict(list)
        boundaries: list[int] = []
        for frame in range(0, len(trajectory), execution_horizon):
            observation, _, _ = build_policy_observation(trajectory, frame, modality)
            action, _ = policy.get_action(observation)
            chunk = unbatch_action(action)
            take = min(execution_horizon, len(trajectory) - frame)
            if frame:
                boundaries.append(frame)
            for key in ACTION_KEYS:
                predictions[key].append(chunk[key][:take])

        episode_metric: dict[str, Any] = {"local_episode": episode_id, "frames": len(trajectory)}
        for key in ACTION_KEYS:
            gt = np.vstack(trajectory[f"action.{key}"].to_numpy()).astype(np.float32)
            pred = np.concatenate(predictions[key], axis=0).astype(np.float32)
            names = ARM_JOINTS if key.endswith("arm") else O6_JOINTS
            episode_metric[key] = error_metrics(gt, pred, names)
            aggregate_gt[key].append(gt)
            aggregate_pred[key].append(pred)
            delta = np.diff(pred, axis=0)
            adjacent_jumps[key].append(delta)
            velocities[key].append(delta * 30.0)
            accelerations[key].append(np.diff(pred, n=2, axis=0) * 30.0**2)
            if boundaries:
                boundary_jumps[key].append(
                    np.stack([pred[index] - pred[index - 1] for index in boundaries])
                )
            if plot_dir is not None:
                plot_group(
                    gt,
                    pred,
                    names,
                    f"{dataset.name} ep{episode_id} {key}; denoise={denoising_steps}, execute={execution_horizon}",
                    plot_dir / f"episode_{episode_id}_{key}.png",
                )
        episode_results.append(episode_metric)

    aggregate: dict[str, Any] = {}
    for key in ACTION_KEYS:
        gt = np.concatenate(aggregate_gt[key], axis=0)
        pred = np.concatenate(aggregate_pred[key], axis=0)
        names = ARM_JOINTS if key.endswith("arm") else O6_JOINTS
        group = {
            "per_dimension_error": error_metrics(gt, pred, names),
            "prediction": {
                "min": np.min(pred, axis=0).tolist(),
                "max": np.max(pred, axis=0).tolist(),
                "mean": np.mean(pred, axis=0).tolist(),
                "std": np.std(pred, axis=0).tolist(),
            },
            "adjacent_prediction_jump": abs_distribution(
                np.concatenate(adjacent_jumps[key], axis=0)
            ),
            "chunk_boundary_jump": abs_distribution(
                np.concatenate(boundary_jumps[key], axis=0)
                if boundary_jumps[key]
                else np.empty((0, pred.shape[1]))
            ),
            "velocity": abs_distribution(np.concatenate(velocities[key], axis=0)),
            "acceleration": abs_distribution(np.concatenate(accelerations[key], axis=0)),
        }
        if key == "right_o6":
            group["prediction"].update(
                {
                    "global_min": float(np.min(pred)),
                    "global_max": float(np.max(pred)),
                    "global_mean": float(np.mean(pred)),
                    "global_std": float(np.std(pred)),
                    "nonzero_prediction_frame_ratio": float(
                        np.mean(np.any(np.abs(pred) > 1e-6, axis=1))
                    ),
                }
            )
        aggregate[key] = group
    return {
        "dataset": str(dataset),
        "denoising_steps": denoising_steps,
        "execution_horizon": execution_horizon,
        "seed": seed,
        "episodes": episode_results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "deployment")
    args = parser.parse_args()
    plots = args.output_dir / "plots/test2"
    policy = load_policy(4)
    configurations = ((4, 16), (4, 1), (8, 1))
    test_results = {}
    for denoise, horizon in configurations:
        name = f"denoise_{denoise}_execute_{horizon}"
        test_results[name] = evaluate_dataset(
            policy,
            TEST_DATASET,
            denoising_steps=denoise,
            execution_horizon=horizon,
            plot_dir=plots / name,
            seed=42,
        )
    val_reference = evaluate_dataset(
        policy,
        VAL_DATASET,
        denoising_steps=4,
        execution_horizon=16,
        plot_dir=None,
        seed=42,
    )
    result = {
        "checkpoint": str(CHECKPOINT),
        "test_is_final_unseen_evaluation_only": True,
        "test_results": test_results,
        "val_reference_denoise_4_execute_16": val_reference,
    }
    json_dump(args.output_dir / "test2_metrics.json", result)
    print(args.output_dir / "test2_metrics.json")


if __name__ == "__main__":
    main()
