#!/usr/bin/env python3
"""Render deployment validation JSON artifacts into concise Markdown reports."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/slxy/下载/g1_o6_gr00t")
DEPLOY = ROOT / "deployment"


def load(name: str):
    return json.loads((DEPLOY / name).read_text())


def write(name: str, text: str) -> None:
    (DEPLOY / name).write_text(text.strip() + "\n", encoding="utf-8")


def mean_joint(group: dict, metric: str) -> float:
    values = group["per_dimension_error"].values()
    return sum(item[metric] for item in values) / len(group["per_dimension_error"])


def render_open_loop() -> None:
    data = load("test2_metrics.json")
    sections = [
        "# test_2 Open-loop Evaluation",
        "",
        "纯离线评估；test_2 只用于最终未见数据报告，没有用于选择模型或继续调参。",
    ]
    for name, result in data["test_results"].items():
        sections += ["", f"## {name}", ""]
        for group_name, group in result["aggregate"].items():
            unit = "rad" if group_name.endswith("arm") else "percentage points"
            sections += [
                f"### {group_name} ({unit})",
                "",
                "|dimension|MAE|RMSE|max abs error|",
                "|---|---:|---:|---:|",
            ]
            for dimension, metric in group["per_dimension_error"].items():
                sections.append(
                    f"|{dimension}|{metric['mae']:.6f}|{metric['rmse']:.6f}|{metric['max_absolute_error']:.6f}|"
                )
            sections += [
                "",
                f"Adjacent prediction max jump: `{group['adjacent_prediction_jump']['max_abs']:.6f}`; "
                f"chunk-boundary max jump: `{group['chunk_boundary_jump']['max_abs']:.6f}`.",
                f"Velocity abs mean/p95/max: `{group['velocity']['mean_abs']:.4f}` / "
                f"`{group['velocity']['p95_abs']:.4f}` / `{group['velocity']['max_abs']:.4f}`.",
                f"Acceleration abs mean/p95/max: `{group['acceleration']['mean_abs']:.4f}` / "
                f"`{group['acceleration']['p95_abs']:.4f}` / `{group['acceleration']['max_abs']:.4f}`.",
            ]
            if group_name == "right_o6":
                pred = group["prediction"]
                sections += [
                    f"Right O6 global prediction min/max/mean/std: `{pred['global_min']}` / "
                    f"`{pred['global_max']}` / `{pred['global_mean']}` / `{pred['global_std']}`; "
                    f"nonzero frame ratio: `{pred['nonzero_prediction_frame_ratio']}`.",
                ]

    test_base = data["test_results"]["denoise_4_execute_16"]["aggregate"]
    val_base = data["val_reference_denoise_4_execute_16"]["aggregate"]
    sections += [
        "",
        "## test_2 versus val_2 (denoise=4, execute=16)",
        "",
        "|group|val mean per-dim MAE|test mean per-dim MAE|",
        "|---|---:|---:|",
    ]
    for key in test_base:
        sections.append(
            f"|{key}|{mean_joint(val_base[key], 'mae'):.6f}|{mean_joint(test_base[key], 'mae'):.6f}|"
        )
    sections += [
        "",
        "结论：execute=1 的 arm MAE 较 execute=16 低，但逐帧随机重规划仍会产生显著跳变；"
        "denoise=8 相对 4 没有实质收益。左 O6 出现最大 100 点跳变，必须经过安全过滤。"
        "右 O6 始终为零是零方差训练标签造成的退化输出，不代表右手能力。",
        "",
        "Plots: `deployment/plots/test2/<configuration>/`.",
    ]
    write("test2_open_loop_report.md", "\n".join(sections))


def render_contract() -> None:
    data = load("policy_output_contract.json")
    sections = [
        "# Policy Output Contract",
        "",
        "## Proven contract",
        "",
        "- left_arm/right_arm config: `RELATIVE + NON_EEF`.",
        "- left_o6/right_o6 config: `ABSOLUTE + NON_EEF`.",
        "- `Gr00tPolicy.get_action()` returns decoded physical-unit actions.",
        "- Arms returned by the standard API are absolute joint targets in rad.",
        "- O6 returned by the standard API is an absolute 0–100 command.",
        "- Action Adapter must not denormalize and must not add `q_current` again.",
        "- `PolicyClient` returns the server payload unchanged.",
        "",
        "## Numerical test on test_2 local episode 0 frame 0",
        "",
        "|group|normalized range|pre-relative physical range|public API range|manual decode == API|",
        "|---|---|---|---|---|",
    ]
    for key, group in data["groups"].items():
        n = group["model_normalized"]
        d = group["denormalized_before_relative"]
        p = group["public_api"]
        sections.append(
            f"|{key}|[{n['min']:.6f}, {n['max']:.6f}]|[{d['min']:.6f}, {d['max']:.6f}]|"
            f"[{p['min']:.6f}, {p['max']:.6f}]|{group['manual_decode_equals_public']}|"
        )
    sections += ["", "Arm identity tests:"]
    for key in ("left_arm", "right_arm"):
        sections.append(
            f"- `{key}`: `public_action - current_state == denormalized_relative` → "
            f"`{data['groups'][key]['public_minus_current_equals_denormalized_relative']}`."
        )
    sections += [
        "",
        f"Project duplicate-transform scan hits: `{len(data['duplicate_transform_scan_hits'])}`.",
        "",
        "## Local code path",
        "",
    ]
    for name, location in data["code_locations"].items():
        sections.append(
            f"- `{name}`: `{location['path']}:{location['start_line']}`–`{location['end_line']}`"
        )
    sections += [
        "",
        "`Gr00tPolicy._get_action` obtains normalized `action_pred`, then calls "
        "`Gr00tN1d7Processor.decode_action`; this calls "
        "`StateActionProcessor.unapply_action`, which denormalizes first and then converts "
        "relative arm chunks to absolute using the last raw state.",
    ]
    write("policy_output_contract.md", "\n".join(sections))


def render_benchmark() -> None:
    data = load("inference_benchmark.json")
    sections = [
        "# Offline Inference Benchmark",
        "",
        "test_2 cached replay on the local GPU; 10 warmups + 100 measured calls per setting.",
        "",
        "|setting|mean E2E ms|p50|p90|p99|mean policy ms|theoretical Hz|peak allocated MiB|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in data["runs"].items():
        e = result["complete_local_e2e"]
        p = result["policy_get_action"]
        sections.append(
            f"|{name}|{e['mean_ms']:.3f}|{e['p50_ms']:.3f}|{e['p90_ms']:.3f}|"
            f"{e['p99_ms']:.3f}|{p['mean_ms']:.3f}|{result['theoretical_sustainable_hz_from_mean']:.2f}|"
            f"{result['gpu_peak_allocated_mib']:.1f}|"
        )
    sections += ["", "## Stage breakdown (mean ms)", "", "|setting|data|dict|preprocess|model|postprocess|", "|---|---:|---:|---:|---:|---:|"]
    for name, result in data["runs"].items():
        sections.append(
            f"|{name}|{result['data_read']['mean_ms']:.3f}|"
            f"{result['observation_dict_construction']['mean_ms']:.3f}|"
            f"{result['processor_preprocessing_including_image']['mean_ms']:.3f}|"
            f"{result['model_forward']['mean_ms']:.3f}|"
            f"{result['action_postprocessing']['mean_ms']:.3f}|"
        )
    sections += [
        "",
        "## Scheduling recommendation",
        "",
        "Initial recommendation: `denoising_steps=4`, 10 Hz replanning, `execution_horizon=3`, "
        "with a continuous 30 Hz action timeline and 100 Hz SDK interpolation. The 4-step p99 "
        "fits a 100 ms replan budget locally; 8 denoising steps adds latency without useful test "
        "error improvement. If future Live Shadow p99 approaches 80–90 ms after camera/network "
        "costs, fall back to 7.5 Hz / horizon 4. 5 Hz / horizon 6 has more timing margin but reacts "
        "more slowly and executes longer stale chunks.",
        "",
        "This benchmark excludes real camera acquisition, image transport, timestamp alignment, "
        "G1/O6 network latency, SDK queueing, controller latency, and motor response. Those must be "
        "measured separately in Live Shadow.",
    ]
    write("inference_benchmark.md", "\n".join(sections))


def render_replay() -> None:
    data = load("logs/replay_shadow/summary.json")
    c = data["counts"]
    l = data["latency_ms"]
    text = f"""
# Offline Replay Shadow Report

Mode: **OFFLINE REPLAY SHADOW: NO COMMANDS SENT**.

- Dataset frames: {c['frames']}
- Policy calls: {c['inferences']}
- denoising steps: {data['denoising_steps']}
- execution horizon: {data['execution_horizon']}
- NaN/Inf: {c['nan_or_inf']}
- out-of-bounds elements: {c['out_of_bounds']}
- buffer underruns: {c['buffer_underrun']}
- calls exceeding a single 30 Hz period: {c['missed_30hz_deadline']}

Inference latency: mean `{l['mean']:.3f}` ms, p50 `{l['p50']:.3f}` ms, p90
`{l['p90']:.3f}` ms, p99 `{l['p99']:.3f}` ms, max `{l['max']:.3f}` ms.

All calls exceed 33.3 ms, so inference cannot be scheduled every camera frame. The recommended
10 Hz planner budget is 100 ms and is satisfied after warmup. The maximum includes first-call
cold start; a future live process must warm up before entering its timed loop.

Full per-frame log: `deployment/logs/replay_shadow/replay_shadow.jsonl`.
"""
    write("replay_shadow_report.md", text)


def render_adapter() -> None:
    data = load("adapter_dry_run_metrics.json")
    sections = [
        "# Action Adapter Dry-run Report",
        "",
        "Pipeline: `Policy API -> Action Adapter -> Safety Filter -> Action Buffer -> Mock SDK`.",
        "No real SDK, topic, network endpoint, or motor was used.",
        "",
        "|group|max policy velocity|max adapter velocity|max policy accel|max adapter accel|boundary before|boundary after|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in data["per_group"].items():
        raw = value["policy_api"]
        safe = value["adapter"]
        sections.append(
            f"|{key}|{raw['max_abs_velocity']:.4f}|{safe['max_abs_velocity']:.4f}|"
            f"{raw['max_abs_acceleration']:.4f}|{safe['max_abs_acceleration']:.4f}|"
            f"{value['max_chunk_boundary_jump_before']:.4f}|{value['max_chunk_boundary_jump_after']:.4f}|"
        )
    counters = data["adapter_metrics"]["filter_counters"]
    sections += [
        "",
        "## Filter triggers",
        "",
        "|group|position|velocity|acceleration|O6 8-point delta|nonfinite|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, count in counters.items():
        sections.append(
            f"|{key}|{count['position_limit']}|{count['velocity_limit']}|"
            f"{count['acceleration_limit']}|{count['o6_delta_limit']}|{count['nonfinite']}|"
        )
    fault = data["fault_tests"]
    sections += [
        "",
        "## Mock failure tests",
        "",
        f"- NaN filtered to finite: `{fault['nan_injection_filtered_to_finite']}`",
        f"- timeout recorded/hold: `{fault['timeout_recorded_and_hold']}`",
        f"- network disconnect recorded/hold: `{fault['network_disconnect_recorded_and_hold']}`",
        f"- watchdog holds last safe target: `{fault['watchdog_holds_last_safe_target']}`",
        f"- emergency stop holds last safe target: `{fault['emergency_stop_holds_last_safe_target']}`",
        f"- empty buffer underrun recorded: `{fault['empty_buffer_underrun_recorded']}`",
        "",
        f"Main run mock records: G1 `{data['adapter_metrics']['mock_g1_records']}`, "
        f"O6 `{data['adapter_metrics']['mock_o6_records']}`; main-run underruns "
        f"`{data['adapter_metrics']['buffer_underruns']}`.",
        "",
        "Plots: `deployment/plots/adapter_dry_run/`.",
    ]
    write("adapter_dry_run_report.md", "\n".join(sections))


def render_live() -> None:
    write(
        "live_shadow_requirements.md",
        """
# Future Live Shadow Requirements

The current `G1LiveObservationSource` is a read-only skeleton. It prints:

```text
SHADOW MODE: NO COMMANDS WILL BE SENT
```

Missing interfaces that must be provided explicitly, without guessing topic names or message types:

1. G1 left/right arm feedback reader.
2. G1 waist feedback reader.
3. projected gravity or synchronized IMU reader.
4. left/right O6 feedback readers.
5. D435i `ego_view` RGB reader.
6. cross-source timestamp synchronization.
7. stale data detection and age reporting.

Before any live run:

- `real_hardware_enabled: false`
- `publish_commands: false`
- `shadow_only: true`
- no arm control ownership request
- no G1 publisher and no O6 publisher
- camera/network/SDK latency added to the offline timing budget
- 10 model warmups before timed processing
- synchronization tolerance and stale timeout validated on real streams

The skeleton returns no observation when dependencies are absent and does not fail fatally.
It deliberately contains no command publisher implementation.
""",
    )


def render_readme() -> None:
    write(
        "README.md",
        """
# GR00T N1.7 G1+O6 Deployment Preflight

Everything in this directory defaults to offline shadow or mock mode. No script connects to G1,
claims robot control, or publishes G1/O6 commands.

## Required environment

```bash
cd /home/slxy/下载/g1_o6_gr00t
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/g1_o6_matplotlib
PY=/home/slxy/下载/Isaac-GR00T/.venv/bin/python
```

## Run order

1. test_2 evaluation

   ```bash
   $PY deployment/evaluate_open_loop.py
   ```

2. Policy output contract

   ```bash
   $PY deployment/tests/verify_policy_output_contract.py
   ```

3. Inference benchmark

   ```bash
   $PY deployment/benchmark_inference.py --warmup 10 --iterations 100
   ```

4. Replay Shadow

   ```bash
   $PY deployment/g1_o6_shadow_client.py --denoising-steps 4 --execution-horizon 3
   ```

5. Adapter dry-run

   ```bash
   $PY deployment/run_adapter_dry_run.py
   ```

6. Mock failure tests

   These are included in `run_adapter_dry_run.py`: NaN, timeout, network disconnect,
   watchdog, emergency stop, and empty buffer.

7. Future Live Shadow skeleton only

   ```bash
   $PY -c 'import yaml; from deployment.observation_sources.g1_live import G1LiveObservationSource; c=yaml.safe_load(open("deployment/config/live_shadow.yaml")); s=G1LiveObservationSource(c); s.start(); print(s.get_observation()); s.stop()'
   ```

Do not change `real_hardware_enabled`, `publish_commands`, or `shadow_only` until the read-only
interfaces in `live_shadow_requirements.md` are implemented and reviewed. The current Action
Adapter only accepts standard Policy API physical-unit outputs; it never denormalizes and never
adds `q_current`.
""",
    )


def main() -> None:
    render_open_loop()
    render_contract()
    render_benchmark()
    render_replay()
    render_adapter()
    render_live()
    render_readme()


if __name__ == "__main__":
    main()
