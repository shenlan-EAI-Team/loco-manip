#!/usr/bin/env python3
"""Plot loss, gradient norm, and learning rate from a Trainer state file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), smoothed])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    state = json.loads(args.trainer_state.read_text())
    rows = [
        row
        for row in state["log_history"]
        if all(key in row for key in ("step", "loss", "grad_norm", "learning_rate"))
    ]
    if not rows:
        raise RuntimeError("trainer state contains no loss records")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
    loss = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    grad = np.asarray([row["grad_norm"] for row in rows], dtype=np.float64)
    learning_rate = np.asarray([row["learning_rate"] for row in rows], dtype=np.float64)
    smoothed_loss = moving_average(loss, 10)

    with (args.output_dir / "training_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "loss", "loss_moving_average_10", "grad_norm", "learning_rate"])
        writer.writerows(zip(steps, loss, smoothed_loss, grad, learning_rate))

    figure, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(steps, loss, color="#8a8f98", linewidth=0.9, alpha=0.65, label="loss / 10 steps")
    axes[0].plot(steps, smoothed_loss, color="#1769aa", linewidth=2.0, label="10-record moving average")
    axes[0].set_ylabel("Training loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, grad, color="#b23a48", linewidth=1.1)
    axes[1].set_ylabel("Gradient norm")
    axes[1].grid(alpha=0.25)
    axes[2].plot(steps, learning_rate, color="#2f7d32", linewidth=1.4)
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Optimizer step")
    axes[2].grid(alpha=0.25)
    figure.suptitle("GR00T N1.7 corrected G1 + O6 training, 3000 steps")
    figure.tight_layout()
    figure.savefig(args.output_dir / "training_curves.png", dpi=180)
    plt.close(figure)

    first_100 = loss[steps <= 100]
    last_100 = loss[steps > steps[-1] - 100]
    minimum_index = int(np.argmin(loss))
    summary = {
        "trainer_state": str(args.trainer_state.resolve()),
        "global_step": int(state["global_step"]),
        "max_steps": int(state["max_steps"]),
        "logged_records": len(rows),
        "logging_interval_steps": int(np.median(np.diff(steps))),
        "loss": {
            "first": float(loss[0]),
            "last": float(loss[-1]),
            "mean_first_100_steps": float(np.mean(first_100)),
            "mean_last_100_steps": float(np.mean(last_100)),
            "minimum": float(loss[minimum_index]),
            "minimum_step": int(steps[minimum_index]),
            "maximum": float(np.max(loss)),
            "all_finite": bool(np.isfinite(loss).all()),
        },
        "gradient_norm": {
            "median": float(np.median(grad)),
            "maximum": float(np.max(grad)),
            "last": float(grad[-1]),
            "all_finite": bool(np.isfinite(grad).all()),
        },
        "learning_rate": {
            "maximum": float(np.max(learning_rate)),
            "last": float(learning_rate[-1]),
        },
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
