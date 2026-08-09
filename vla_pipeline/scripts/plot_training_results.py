#!/usr/bin/env python3
"""Generate compact formal-training curves and a machine-readable summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    result = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), result])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.trainer_state.read_text())
    validation = json.loads(args.validation_metrics.read_text())
    logs = [entry for entry in state["log_history"] if "loss" in entry]
    steps = np.array([entry["step"] for entry in logs])
    losses = np.array([entry["loss"] for entry in logs])
    grad_norm = np.array([entry["grad_norm"] for entry in logs])
    learning_rate = np.array([entry["learning_rate"] for entry in logs])

    milestone_summary = {}
    for end in (1000, 2000, 3000, 4000):
        mask = (steps > end - 1000) & (steps <= end)
        milestone_summary[str(end)] = {
            "mean_loss_in_preceding_1000_steps": float(np.mean(losses[mask])),
            "min_logged_loss_in_preceding_1000_steps": float(np.min(losses[mask])),
            "last_logged_loss": float(losses[steps == end][-1]),
        }

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(steps, losses, color="#7aa6c2", alpha=0.35, label="10-step loss")
    axes[0].plot(steps, rolling_mean(losses, 20), color="#005f87", lw=2, label="200-step mean")
    axes[0].set_ylabel("training loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, grad_norm, color="#a05a2c", lw=1)
    axes[1].set_ylabel("gradient norm")
    axes[1].grid(alpha=0.25)
    axes[2].plot(steps, learning_rate, color="#4c956c", lw=2)
    axes[2].set_ylabel("learning rate")
    axes[2].set_xlabel("optimizer step")
    axes[2].grid(alpha=0.25)
    fig.suptitle("GR00T N1.7 G1+O6 formal finetune (26 episodes)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    grouped = validation["aggregate_by_modality"]
    names = list(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, [grouped[name]["mae"] for name in names], color="#3572a5")
    axes[0].set_title("Validation MAE (raw units)")
    axes[0].set_ylabel("rad for arms; percent for O6")
    axes[1].bar(names, [grouped[name]["rmse"] for name in names], color="#d17b49")
    axes[1].set_title("Validation RMSE (raw units)")
    axes[1].set_ylabel("rad for arms; percent for O6")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_metrics.png", dpi=180)
    plt.close(fig)

    best_milestone = min(
        milestone_summary,
        key=lambda key: milestone_summary[key]["mean_loss_in_preceding_1000_steps"],
    )
    summary = {
        "global_step": int(state["global_step"]),
        "train_loss_reported": next(
            (
                entry["train_loss"]
                for entry in state["log_history"]
                if "train_loss" in entry
            ),
            float(np.mean(losses)),
        ),
        "final_logged_loss": float(losses[-1]),
        "minimum_logged_loss": float(np.min(losses)),
        "milestones": milestone_summary,
        "best_saved_checkpoint": f"checkpoint-{best_milestone}",
        "selection_rule": "lowest mean training loss in each preceding 1000-step window; final checkpoint then validated on val_2",
        "validation": validation["aggregate_by_modality"],
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
