#!/usr/bin/env python3
"""Numerically verify the local checkpoint's public Policy API action contract."""

from __future__ import annotations

import inspect
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from gr00t.data.types import MessageType
from gr00t.data.utils import unnormalize_values_meanstd, unnormalize_values_minmax
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from gr00t.policy.server_client import PolicyClient

from deployment.common import (
    ACTION_KEYS,
    CORRECTED_CHECKPOINT,
    PROJECT_ROOT,
    TEST_DATASET,
    array_stats,
    build_policy_observation,
    json_dump,
    load_policy,
    make_loader,
    seed_everything,
)


def code_location(obj: object) -> dict[str, object]:
    lines, start = inspect.getsourcelines(obj)
    return {
        "path": str(Path(inspect.getsourcefile(obj) or "").resolve()),
        "function": getattr(obj, "__qualname__", str(obj)),
        "start_line": start,
        "end_line": start + len(lines) - 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CORRECTED_CHECKPOINT)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets_corrected_v1/test_2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "deployment/policy_output_contract_corrected.json",
    )
    args = parser.parse_args()
    output = args.output
    policy = load_policy(4, args.checkpoint)
    modality = policy.get_modality_config()
    loader = make_loader(args.dataset, modality)
    trajectory = loader[0]
    observation, flat, _ = build_policy_observation(trajectory, 0, modality)

    # Reproduce Gr00tPolicy._get_action step by step and retain internal tensors.
    seed_everything(2026)
    unbatched = policy._unbatch_observation(observation)
    processed_inputs = []
    raw_states = []
    for obs in unbatched:
        step = policy._to_vla_step_data(obs)
        raw_states.append(step.states)
        processed_inputs.append(
            policy.processor(
                [{"type": MessageType.EPISODE_STEP.value, "content": step}]
            )
        )
    collated = _rec_to_dtype(policy.collate_fn(processed_inputs), torch.bfloat16)
    with torch.inference_mode():
        model_pred = policy.model.get_action(**collated)
    normalized_concat = model_pred["action_pred"].float().cpu().numpy()
    batched_states = {
        key: np.stack([state[key] for state in raw_states], axis=0)
        for key in modality["state"].modality_keys
    }
    decoded = policy.processor.decode_action(
        normalized_concat, policy.embodiment_tag, batched_states
    )

    # Same seed must make the public API match the manual decode exactly.
    seed_everything(2026)
    public_action, _ = policy.get_action(observation)

    sap = policy.processor.state_action_processor
    tag = policy.embodiment_tag.value
    start = 0
    normalized_groups = {}
    denormalized_before_relative = {}
    for key in ACTION_KEYS:
        dim = int(sap.norm_params[tag]["action"][key]["dim"].item())
        group = normalized_concat[..., :16, start : start + dim]
        normalized_groups[key] = group
        params = sap.norm_params[tag]["action"][key]
        mean_std = modality["action"].mean_std_embedding_keys or []
        if key in mean_std:
            denormalized = unnormalize_values_meanstd(group, params)
        else:
            denormalized = unnormalize_values_minmax(group, params)
        denormalized_before_relative[key] = denormalized
        start += dim

    groups = {}
    all_equal = True
    for key in ACTION_KEYS:
        public = np.asarray(public_action[key])
        manual = np.asarray(decoded[key])
        state_key = key if key.endswith("arm") else None
        result = {
            "representation": str(
                modality["action"].action_configs[ACTION_KEYS.index(key)].rep.value
            ),
            "type": str(modality["action"].action_configs[ACTION_KEYS.index(key)].type.value),
            "public_api": array_stats(public),
            "model_normalized": array_stats(normalized_groups[key]),
            "denormalized_before_relative": array_stats(denormalized_before_relative[key]),
            "manual_decode_equals_public": bool(np.allclose(public, manual, atol=1e-6)),
            "max_manual_public_difference": float(np.max(np.abs(public - manual))),
        }
        all_equal &= result["manual_decode_equals_public"]
        if state_key:
            current = batched_states[state_key][:, -1:, :]
            recovered_relative = public - current
            result.update(
                {
                    "current_state": array_stats(current),
                    "public_minus_current": array_stats(recovered_relative),
                    "public_minus_current_equals_denormalized_relative": bool(
                        np.allclose(
                            recovered_relative,
                            denormalized_before_relative[key],
                            atol=1e-5,
                        )
                    ),
                }
            )
        else:
            result["public_equals_denormalized_absolute"] = bool(
                np.allclose(public, denormalized_before_relative[key], atol=1e-6)
            )
        groups[key] = result

    # PolicyClient returns the server's get_action payload without action processing.
    client = PolicyClient(host="127.0.0.1", port=1, timeout_ms=1, strict=False)
    client.call_endpoint = lambda *args, **kwargs: [public_action, {"mocked_local_transport": True}]
    client_action, client_info = client._get_action(observation)
    client.close()
    client_equal = all(
        np.array_equal(np.asarray(client_action[key]), np.asarray(public_action[key]))
        for key in ACTION_KEYS
    )

    project_hits = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts or path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(errors="replace")
        for needle in ("q_current + action", "q_current+action", "denormalize(", "unnormalize("):
            if needle in text:
                project_hits.append({"path": str(path), "needle": needle})

    result = {
        "test_frame": {
            "dataset": str(args.dataset.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "local_episode": 0,
            "frame": 0,
            "state_shapes": {
                key: list(np.asarray(flat[f"state.{key}"]).shape)
                for key in modality["state"].modality_keys
            },
        },
        "groups": groups,
        "all_manual_decode_equals_public": all_equal,
        "policy_client_payload_unchanged": client_equal,
        "policy_client_info": client_info,
        "duplicate_transform_scan_hits": project_hits,
        "adapter_contract": {
            "input_is_physical_units": True,
            "adapter_must_denormalize": False,
            "adapter_must_add_q_current": False,
            "arm_unit": "rad absolute joint target",
            "o6_unit": "0-100 absolute command",
        },
        "code_locations": {
            "Gr00tPolicy._get_action": code_location(Gr00tPolicy._get_action),
            "processor.decode_action": code_location(type(policy.processor).decode_action),
            "StateActionProcessor.unapply_action": code_location(type(sap).unapply_action),
            "PolicyClient._get_action": code_location(PolicyClient._get_action),
        },
    }
    json_dump(output, result)
    if not all_equal or not client_equal:
        raise SystemExit("Policy output contract mismatch")
    print(output)


if __name__ == "__main__":
    main()
