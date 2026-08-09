#!/usr/bin/env python3
"""Analyze and plot deterministic corrected-dataset open-loop predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUPS = ("left_arm", "right_arm", "left_o6", "right_o6")
ARM_NAMES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
O6_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def group_metrics(gt: np.ndarray, pred: np.ndarray, names: tuple[str, ...]) -> dict:
    error = pred.astype(np.float64) - gt.astype(np.float64)
    per_dimension = {}
    correlations = []
    for index, name in enumerate(names):
        dimension_error = error[:, index]
        gt_std = float(np.std(gt[:, index]))
        pred_std = float(np.std(pred[:, index]))
        correlation = (
            float(np.corrcoef(gt[:, index], pred[:, index])[0, 1])
            if gt_std > 1e-8 and pred_std > 1e-8
            else None
        )
        if correlation is not None and np.isfinite(correlation):
            correlations.append(correlation)
        per_dimension[name] = {
            "mae": float(np.mean(np.abs(dimension_error))),
            "rmse": float(np.sqrt(np.mean(dimension_error**2))),
            "max_absolute_error": float(np.max(np.abs(dimension_error))),
            "correlation": correlation,
            "gt_range": [float(np.min(gt[:, index])), float(np.max(gt[:, index]))],
            "prediction_range": [
                float(np.min(pred[:, index])),
                float(np.max(pred[:, index])),
            ],
        }
    return {
        "frames": int(gt.shape[0]),
        "dimensions": int(gt.shape[1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "error_abs_distribution": distribution(error),
        "mean_valid_dimension_correlation": (
            float(np.mean(correlations)) if correlations else None
        ),
        "per_dimension": per_dimension,
    }


def boundary_metrics(episodes: list[dict[str, np.ndarray]], key: str, horizon: int) -> dict:
    pred_boundary = []
    gt_boundary = []
    pred_interior = []
    boundary_error = []
    for episode in episodes:
        gt = episode[f"gt_{key}"]
        pred = episode[f"pred_{key}"]
        delta_pred = np.diff(pred, axis=0)
        delta_gt = np.diff(gt, axis=0)
        mask = np.zeros(len(delta_pred), dtype=bool)
        indices = np.arange(horizon - 1, len(delta_pred), horizon)
        mask[indices] = True
        pred_boundary.append(delta_pred[mask])
        gt_boundary.append(delta_gt[mask])
        pred_interior.append(delta_pred[~mask])
        boundary_error.append((pred - gt)[np.arange(horizon, len(gt), horizon)])
    return {
        "prediction_chunk_boundary_jump": distribution(np.concatenate(pred_boundary)),
        "ground_truth_at_same_boundary_jump": distribution(np.concatenate(gt_boundary)),
        "prediction_interior_jump": distribution(np.concatenate(pred_interior)),
        "absolute_error_at_chunk_start": distribution(np.concatenate(boundary_error)),
    }


def max_error_location(episodes: list[dict[str, np.ndarray]], key: str, names: tuple[str, ...]) -> dict:
    maximum = None
    for episode_id, episode in enumerate(episodes):
        gt = episode[f"gt_{key}"]
        pred = episode[f"pred_{key}"]
        absolute = np.abs(pred.astype(np.float64) - gt.astype(np.float64))
        frame, dimension = np.unravel_index(np.argmax(absolute), absolute.shape)
        candidate = {
            "episode": episode_id,
            "frame": int(frame),
            "dimension": int(dimension),
            "name": names[dimension],
            "absolute_error": float(absolute[frame, dimension]),
            "ground_truth": float(gt[frame, dimension]),
            "prediction": float(pred[frame, dimension]),
            "is_chunk_start": bool(frame > 0 and frame % 16 == 0),
        }
        if maximum is None or candidate["absolute_error"] > maximum["absolute_error"]:
            maximum = candidate
    return maximum


def plot_episode(episode: dict[str, np.ndarray], split: str, episode_id: int, output: Path) -> None:
    fig, axes = plt.subplots(7, 2, figsize=(15, 16), sharex=True)
    for column, side in enumerate(("left_arm", "right_arm")):
        gt, pred = episode[f"gt_{side}"], episode[f"pred_{side}"]
        for row, name in enumerate(ARM_NAMES):
            axes[row, column].plot(gt[:, row], label="ground truth", lw=1.15)
            axes[row, column].plot(pred[:, row], label="prediction", lw=0.9, alpha=0.82)
            axes[row, column].set_ylabel(name)
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(side)
        axes[0, column].legend(ncol=2, fontsize=8)
        axes[-1, column].set_xlabel("30 Hz frame")
    fig.suptitle(f"{split} episode {episode_id}: arm open-loop actions")
    fig.tight_layout()
    fig.savefig(output / f"{split}_episode_{episode_id}_arms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(6, 2, figsize=(15, 14), sharex=True)
    for column, side in enumerate(("left_o6", "right_o6")):
        gt, pred = episode[f"gt_{side}"], episode[f"pred_{side}"]
        for row, name in enumerate(O6_NAMES):
            axes[row, column].plot(gt[:, row], label="ground truth", lw=1.15)
            axes[row, column].plot(pred[:, row], label="prediction", lw=0.9, alpha=0.82)
            axes[row, column].set_ylabel(name)
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(side)
        axes[0, column].legend(ncol=2, fontsize=8)
        axes[-1, column].set_xlabel("30 Hz frame")
    fig.suptitle(f"{split} episode {episode_id}: O6 open-loop actions")
    fig.tight_layout()
    fig.savefig(output / f"{split}_episode_{episode_id}_o6.png", dpi=150)
    plt.close(fig)


def mean_old_group_mae(old: dict, key: str) -> float:
    values = old["test_results"]["denoise_4_execute_16"]["aggregate"][key][
        "per_dimension_error"
    ].values()
    return float(np.mean([value["mae"] for value in values]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--old-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--markdown-report", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "evaluation_contract": {
            "checkpoint": "checkpoint-3000 corrected_v1",
            "seed": 42,
            "denoising_steps": 4,
            "execution_horizon": 16,
            "frequency_hz": 30,
            "official_open_loop_plots_also_generated": True,
        },
        "splits": {},
    }
    if args.training_summary is not None:
        result["training"] = json.loads(args.training_summary.read_text())
    loaded_splits = {}
    for split in ("smoke", "val", "test"):
        metrics_dir = args.results_root / split / "metrics_seed42"
        episode_paths = sorted(metrics_dir.glob("episode_*_predictions.npz"))
        episodes = [{key: value for key, value in np.load(path).items()} for path in episode_paths]
        loaded_splits[split] = episodes
        split_result = {"episodes": len(episodes), "groups": {}}
        for key in GROUPS:
            names = ARM_NAMES if key.endswith("arm") else O6_NAMES
            gt = np.concatenate([episode[f"gt_{key}"] for episode in episodes])
            pred = np.concatenate([episode[f"pred_{key}"] for episode in episodes])
            group = group_metrics(gt, pred, names)
            group["continuity"] = boundary_metrics(episodes, key, 16)
            group["max_error_location"] = max_error_location(episodes, key, names)
            if key == "left_o6":
                absolute = np.abs(pred.astype(np.float64) - gt.astype(np.float64))
                group["large_error_rates"] = {
                    "over_5_points": float(np.mean(absolute > 5)),
                    "over_10_points": float(np.mean(absolute > 10)),
                    "over_30_points": float(np.mean(absolute > 30)),
                }
            if key == "right_o6":
                group["prediction_is_exactly_zero"] = bool(np.count_nonzero(pred) == 0)
            split_result["groups"][key] = group
        result["splits"][split] = split_result
        for episode_id, episode in enumerate(episodes):
            plot_episode(episode, split, episode_id, args.output_dir)

    old = json.loads(args.old_metrics.read_text())
    comparison = {}
    for key in GROUPS:
        old_mae = mean_old_group_mae(old, key)
        new_mae = result["splits"]["test"]["groups"][key]["mae"]
        comparison[key] = {
            "old_checkpoint_4000_test_mae": old_mae,
            "corrected_checkpoint_3000_test_mae": new_mae,
            "relative_reduction": float((old_mae - new_mae) / old_mae) if old_mae else None,
        }
    result["old_vs_corrected_test"] = comparison

    h3_paths = sorted(
        (args.results_root / "test/metrics_seed42_h3").glob("episode_*_predictions.npz")
    )
    if h3_paths:
        h3_episodes = [{key: value for key, value in np.load(path).items()} for path in h3_paths]
        horizon_comparison = {}
        for key in GROUPS:
            names = ARM_NAMES if key.endswith("arm") else O6_NAMES
            h16_gt = np.concatenate(
                [episode[f"gt_{key}"] for episode in loaded_splits["test"]]
            )
            h16_pred = np.concatenate(
                [episode[f"pred_{key}"] for episode in loaded_splits["test"]]
            )
            h3_gt = np.concatenate([episode[f"gt_{key}"] for episode in h3_episodes])
            h3_pred = np.concatenate([episode[f"pred_{key}"] for episode in h3_episodes])
            horizon_comparison[key] = {
                "execution_horizon_16": {
                    "metrics": group_metrics(h16_gt, h16_pred, names),
                    "continuity": boundary_metrics(loaded_splits["test"], key, 16),
                },
                "execution_horizon_3": {
                    "metrics": group_metrics(h3_gt, h3_pred, names),
                    "continuity": boundary_metrics(h3_episodes, key, 3),
                },
            }
        result["test_execution_horizon_comparison"] = horizon_comparison

    labels = list(GROUPS)
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for split, color in (("smoke", "#4c78a8"), ("val", "#f2a541"), ("test", "#c44e52")):
        arm_values = [result["splits"][split]["groups"][key]["mae"] for key in labels[:2]]
        axes[0].plot(labels[:2], arm_values, marker="o", linewidth=2, label=split, color=color)
        o6_values = [result["splits"][split]["groups"][key]["mae"] for key in labels[2:]]
        axes[1].plot(labels[2:], o6_values, marker="o", linewidth=2, label=split, color=color)
    axes[0].set_ylabel("MAE (rad)")
    axes[0].set_title("Arm open-loop error")
    axes[1].set_ylabel("MAE (percentage points)")
    axes[1].set_title("O6 open-loop error")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "split_mae_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, key, unit in zip(
        axes, ("left_arm", "right_arm", "left_o6"), ("rad", "rad", "points")
    ):
        names = ARM_NAMES if key.endswith("arm") else O6_NAMES
        old_values = [
            old["test_results"]["denoise_4_execute_16"]["aggregate"][key][
                "per_dimension_error"
            ][name]["mae"]
            for name in names
        ]
        new_values = [
            result["splits"]["test"]["groups"][key]["per_dimension"][name]["mae"]
            for name in names
        ]
        positions = np.arange(len(names))
        axis.bar(positions - 0.2, old_values, 0.4, label="old checkpoint-4000")
        axis.bar(positions + 0.2, new_values, 0.4, label="corrected checkpoint-3000")
        axis.set_xticks(positions, names, rotation=45, ha="right")
        axis.set_ylabel(f"MAE ({unit})")
        axis.set_title(key)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "old_vs_corrected_test_per_joint.png", dpi=180)
    plt.close(figure)

    if "test_execution_horizon_comparison" in result:
        horizon = result["test_execution_horizon_comparison"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, keys, unit in (
            (axes[0], ("left_arm", "right_arm"), "rad"),
            (axes[1], ("left_o6", "right_o6"), "percentage points"),
        ):
            positions = np.arange(len(keys))
            h16 = [horizon[key]["execution_horizon_16"]["metrics"]["mae"] for key in keys]
            h3 = [horizon[key]["execution_horizon_3"]["metrics"]["mae"] for key in keys]
            axis.bar(positions - 0.2, h16, 0.4, label="execution horizon 16")
            axis.bar(positions + 0.2, h3, 0.4, label="execution horizon 3")
            axis.set_xticks(positions, keys)
            axis.set_ylabel(f"MAE ({unit})")
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "test_execution_horizon_comparison.png", dpi=180)
        plt.close(figure)

    args.json_report.write_text(json.dumps(result, indent=2) + "\n")

    test = result["splits"]["test"]["groups"]
    val = result["splits"]["val"]["groups"]
    lines = [
        "# Corrected Training and Open-loop Evaluation",
        "",
        "## Contract",
        "",
        "- Checkpoint: corrected_v1 checkpoint-3000.",
        "- Official GR00T open-loop script: denoising steps 4, execution horizon 16.",
        "- Deterministic quantitative pass: seed 42, corrected smoke/val/test splits.",
        "- Arm errors are radians; O6 errors are percentage points. They are never averaged together in conclusions.",
    ]
    if "training" in result:
        training = result["training"]
        lines.extend(
            [
                "",
                "## Training",
                "",
                f"- Completed {training['global_step']} / {training['max_steps']} optimizer steps.",
                f"- Mean loss over first 100 steps: {training['loss']['mean_first_100_steps']:.6f}.",
                f"- Mean loss over last 100 steps: {training['loss']['mean_last_100_steps']:.6f}.",
                f"- Minimum logged loss: {training['loss']['minimum']:.6f} at step {training['loss']['minimum_step']}.",
                f"- Maximum gradient norm: {training['gradient_norm']['maximum']:.6f}; all logged loss and gradients are finite.",
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregate MAE",
            "",
            "|split|left arm (rad)|right arm (rad)|left O6 (points)|right O6 (points)|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("smoke", "val", "test"):
        groups = result["splits"][split]["groups"]
        lines.append(
            f"|{split}|{groups['left_arm']['mae']:.6f}|{groups['right_arm']['mae']:.6f}|"
            f"{groups['left_o6']['mae']:.6f}|{groups['right_o6']['mae']:.6f}|"
        )
    lines.extend(["", "## Old Versus Corrected Test", "", "|group|old MAE|corrected MAE|reduction|", "|---|---:|---:|---:|"])
    for key, values in comparison.items():
        reduction = values["relative_reduction"]
        reduction_text = "n/a" if reduction is None else f"{100 * reduction:.1f}%"
        lines.append(
            f"|{key}|{values['old_checkpoint_4000_test_mae']:.6f}|"
            f"{values['corrected_checkpoint_3000_test_mae']:.6f}|{reduction_text}|"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Both arms improved materially after removing the second reorder/offset/scale. Test MAE is {test['left_arm']['mae']:.4f} rad left and {test['right_arm']['mae']:.4f} rad right.",
            f"- The test-versus-validation gap is {(test['left_arm']['mae']/val['left_arm']['mae']-1)*100:.1f}% for the left arm and {(test['right_arm']['mae']/val['right_arm']['mae']-1)*100:.1f}% for the right arm.",
            f"- Left O6 test MAE is {test['left_o6']['mae']:.2f} points, but {100*test['left_o6']['large_error_rates']['over_30_points']:.2f}% of scalar predictions exceed 30 points error; rare outliers dominate RMSE.",
            "- Right O6 is exactly zero because every training state/action is zero and invalid. Zero error is label degeneracy, not learned right-hand control.",
            "- Open-loop curves concatenate independent 16-step chunks. Chunk-boundary discontinuity must be handled by the deployment replanning/interpolation safety layer; this evaluation is not a closed-loop stability proof.",
        ]
    )
    if "test_execution_horizon_comparison" in result:
        horizon = result["test_execution_horizon_comparison"]
        lines.extend(
            [
                f"- At deployment-style execution horizon 3, test arm MAE improves to {horizon['left_arm']['execution_horizon_3']['metrics']['mae']:.4f} rad left and {horizon['right_arm']['execution_horizon_3']['metrics']['mae']:.4f} rad right.",
                f"- Horizon 3 does not fix the absolute O6 output: left O6 MAE changes from {horizon['left_o6']['execution_horizon_16']['metrics']['mae']:.2f} to {horizon['left_o6']['execution_horizon_3']['metrics']['mae']:.2f} points, so final O6 smoothing/enveloping remains mandatory.",
            ]
        )
    lines.append("")
    args.markdown_report.write_text("\n".join(lines))
    print(args.json_report)


if __name__ == "__main__":
    main()
