#!/usr/bin/env python3
"""Offline local inference benchmark using real test_2 observations."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import _rec_to_dtype

from deployment.common import (
    PROJECT_ROOT,
    TEST_DATASET,
    json_dump,
    load_policy,
    make_loader,
    observation_modalities,
    seed_everything,
)


def ms_stats(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def construct_observation(point: Any, modality: dict[str, Any]) -> dict[str, Any]:
    flat = {f"state.{key}": value for key, value in point.states.items()}
    flat.update({f"video.{key}": np.asarray(value) for key, value in point.images.items()})
    for language_key in modality["language"].modality_keys:
        flat[language_key] = point.text
    return parse_observation_gr00t(flat, modality)


def benchmark(policy: Any, denoising_steps: int, warmup: int, iterations: int) -> dict[str, Any]:
    policy.model.action_head.num_inference_timesteps = denoising_steps
    modality = policy.get_modality_config()
    loader = make_loader(TEST_DATASET, modality)
    trajectory = loader[0]
    obs_modality = observation_modalities(modality)

    observations = []
    data_times = []
    construction_times = []
    for index in range(iterations + warmup):
        frame = index % len(trajectory)
        start = time.perf_counter()
        point = extract_step_data(
            trajectory, frame, obs_modality, EmbodimentTag.NEW_EMBODIMENT
        )
        data_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        observations.append(construct_observation(point, modality))
        construction_times.append(time.perf_counter() - start)

    seed_everything(42)
    for observation in observations[:warmup]:
        policy.get_action(observation)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    public_times = []
    e2e_times = []
    for index, observation in enumerate(observations[warmup:]):
        start = time.perf_counter()
        policy.get_action(observation)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        public_times.append(elapsed)
        e2e_times.append(
            elapsed + data_times[warmup + index] + construction_times[warmup + index]
        )
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    # Exact internal pipeline timing: preprocessing, model, postprocessing.
    preprocessing = []
    model_times = []
    postprocessing = []
    for observation in observations[warmup:]:
        start = time.perf_counter()
        unbatched = policy._unbatch_observation(observation)
        processed_inputs = []
        states = []
        for obs in unbatched:
            step = policy._to_vla_step_data(obs)
            states.append(step.states)
            processed_inputs.append(
                policy.processor(
                    [{"type": MessageType.EPISODE_STEP.value, "content": step}]
                )
            )
        collated = _rec_to_dtype(policy.collate_fn(processed_inputs), torch.bfloat16)
        preprocessing.append(time.perf_counter() - start)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            model_pred = policy.model.get_action(**collated)
        torch.cuda.synchronize()
        model_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in modality["state"].modality_keys
        }
        policy.processor.decode_action(
            model_pred["action_pred"].float().cpu().numpy(),
            policy.embodiment_tag,
            batched_states,
        )
        postprocessing.append(time.perf_counter() - start)

    result = {
        "denoising_steps": denoising_steps,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "data_read": ms_stats(data_times[warmup:]),
        "observation_dict_construction": ms_stats(construction_times[warmup:]),
        "processor_preprocessing_including_image": ms_stats(preprocessing),
        "model_forward": ms_stats(model_times),
        "action_postprocessing": ms_stats(postprocessing),
        "policy_get_action": ms_stats(public_times),
        "complete_local_e2e": ms_stats(e2e_times),
        "gpu_peak_allocated_mib": peak_allocated / 1024**2,
        "gpu_peak_reserved_mib": peak_reserved / 1024**2,
        "theoretical_sustainable_hz_from_mean": 1.0 / np.mean(e2e_times),
        "scope_warning": "Offline local cached replay only; excludes camera, G1 network, SDK, synchronization, and actuator latency.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "deployment/inference_benchmark.json")
    args = parser.parse_args()
    if args.warmup < 10 or args.iterations < 100:
        raise ValueError("Requirements: warmup >=10 and iterations >=100")
    policy = load_policy(4)
    results = {
        "test_dataset": str(TEST_DATASET),
        "runs": {
            "denoising_4": benchmark(policy, 4, args.warmup, args.iterations),
            "denoising_8": benchmark(policy, 8, args.warmup, args.iterations),
        },
        "candidate_schedules": {
            "10_hz": {"execution_horizon": 3, "period_ms": 100.0},
            "7_5_hz": {"execution_horizon": 4, "period_ms": 133.333333},
            "5_hz": {"execution_horizon": 6, "period_ms": 200.0},
        },
    }
    json_dump(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
