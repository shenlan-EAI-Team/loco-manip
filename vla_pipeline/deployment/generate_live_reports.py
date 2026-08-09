#!/usr/bin/env python3
"""Generate required Live Shadow reports from inventory and run summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deployment.common import PROJECT_ROOT


ROOT = PROJECT_ROOT / "deployment"


def load_summaries() -> list[dict[str, Any]]:
    result = []
    for path in sorted((ROOT / "logs/live_shadow").glob("*/summary.json")):
        value = json.loads(path.read_text())
        value["_path"] = str(path)
        result.append(value)
    return result


def write(name: str, content: str) -> None:
    (ROOT / name).write_text(content.strip() + "\n", encoding="utf-8")


def fmt_timing(summary: dict[str, Any]) -> str:
    rows = []
    for key, value in summary.get("timing_ms", {}).items():
        rows.append(
            f"|{key}|{value['mean']:.3f}|{value['p50']:.3f}|{value['p90']:.3f}|"
            f"{value['p99']:.3f}|{value['max']:.3f}|"
        )
    return "\n".join(rows)


def main() -> None:
    summaries = load_summaries()
    complete = [item for item in summaries if not item.get("failure") and item.get("counts", {}).get("inferences", 0)]
    safety_path = ROOT / "live_safety_audit.json"
    safety = json.loads(safety_path.read_text()) if safety_path.exists() else {}

    write(
        "live_interface_inventory.md",
        """
# Live interface inventory

## G1 status observed through read-only SSH

- Host: `unitree-g1-nx`, user `unitree`, IP `192.168.123.164`.
- Wired interface: `enP8p1s0=192.168.123.164/24`; Wi-Fi: `192.168.31.171/24`.
- At audit time there was no tmux runtime and no listener on 5555/5556/5557/5558/5560/5561/60061.
- `g1_deploy_onnx_ref` was not running.

## Selected read-only inputs

|Input|Verified interface|Shape/order|Unit|
|---|---|---|---|
|G1 state|DDS SUB `rt/lowstate`, `unitree_hg::msg::dds_::LowState_`|left arm motor 15:22; right 22:29; waist 12:15|rad|
|IMU|`LowState.imu_state.quaternion`|w,x,y,z|unit quaternion|
|Projected gravity|inverse(wxyz) applied to world `[0,0,-1]`|x,y,z|unit vector|
|Camera|ZMQ SUB `tcp://192.168.123.164:5555`, `ego_view`|480x640x3 RGB|uint8|
|O6|ZMQ SUB `tcp://192.168.123.164:5558`, atomic schema v2|thumb pitch/yaw,index,middle,ring,pinky|training scale 0..100|

SONIC 5557 was rejected as a Live Shadow source: its executable constructs `MotionSwitcherClient`, releases an active mode, creates a `LowCmd` publisher and starts a command writer. It is not read-only.
""",
    )

    write(
        "o6_interface_contract.md",
        """
# O6 interface contract

- Upstream LinkerHand O6 CAN SDK command and feedback registers are native integer `0..255`.
- `get_current_status()` sends a CAN read request (`0x01` with no target bytes); `set_joint_positions()` sends `0x01` plus six target bytes. These are distinct operations.
- The data-collection ZMQ client rejects anything outside `0..100`; converted train/test state and action are also `0..100`.
- Joint order is `thumb_cmc_pitch, thumb_cmc_yaw, index_mcp_pitch, middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch`.
- The original 30-episode parquet was checked numerically: all 249 unique left feedback values and all 255 unique left command values satisfy `percent * 255 / 100 = integer raw` with maximum floating-point residual `9.54e-6`. Thus the forward mapping used for collection is exactly `percent = raw_255 * 100 / 255`.
- A future command bridge must apply the inverse conversion exactly once: `raw_255 = percent * 255 / 100`, after clamping percentage to 0..100. The remote helper must still be inspected to copy its exact integer rounding rule rather than guessing floor vs round.
- Live Shadow creates only a ZMQ SUB socket. It creates no O6 command socket and never imports `LinkerHandApi`.
""",
    )

    if complete:
        latest = complete[-1]
        status = "COMPLETE"
        timing = fmt_timing(latest)
        deadline = f"{latest['deadline_miss_ratio']:.4%}"
        source_diag = json.dumps(latest.get("source_diagnostics", {}), indent=2, ensure_ascii=False)
        analysis = json.dumps(latest.get("analysis", {}), indent=2, ensure_ascii=False)
    else:
        latest = summaries[-1] if summaries else None
        status = "BLOCKED / NOT RUN"
        timing = "|No valid real-time run|0|0|0|0|0|"
        deadline = "not measured"
        source_diag = json.dumps(latest.get("source_diagnostics", {}) if latest else {}, indent=2, ensure_ascii=False)
        analysis = json.dumps(latest.get("analysis", {}) if latest else {}, indent=2, ensure_ascii=False)

    write(
        "live_sync_report.md",
        f"""
# Live synchronization report

Status: **{status}**

The synchronizer uses a new camera frame as the inference trigger and accepts it only when G1 and dual-O6 receive timestamps are within 20 ms. It rejects missing, stale, invalid, non-finite, wrong-shape and quaternion-norm failures. It never zero-fills or silently reuses a stale datum.

Machine diagnostics:

```json
{source_diag}
```
""",
    )

    write(
        "live_timing_report.md",
        f"""
# Live timing report

Status: **{status}**

|Stage|mean ms|P50|P90|P99|max|
|---|---:|---:|---:|---:|---:|
{timing}

10 Hz deadline miss ratio: **{deadline}**.

The 7.5 Hz / horizon 4 fallback may only be assessed after a complete 10 Hz real-input run.
""",
    )

    write(
        "live_arm_analysis.md",
        f"""
# Live arm and O6 analysis

Status: **{status}**

No claim about static drift, real right-arm motion, object-position response, O6 opening/closing, or filter-cause attribution is made without a complete real-input run.

Machine analysis when available:

```json
{analysis}
```
""",
    )

    safety_ok = bool(safety.get("gate_passed"))
    runtime_zero = all(
        item.get("null_sink", {}).get(key, 1) == 0
        for item in complete
        for key in ("command_publish_attempts", "control_ownership_requests", "real_sdk_objects_created")
    )
    gate = bool(complete and safety_ok and runtime_zero)
    write(
        "live_shadow_gate_report.md",
        f"""
# Live Shadow gate report

- Static safety gate: `{safety_ok}`
- Complete real-input run available: `{bool(complete)}`
- Runtime command/ownership/real-SDK counters all zero: `{runtime_zero if complete else 'not measured'}`
- Gate passed: **{gate}**

Current result does not authorize real command bridge operation. A future real-SDK bridge may be developed only after this gate passes, and it must still default to command publication disabled.
""",
    )

    write(
        "live_shadow_report.md",
        f"""
# G1 + O6 Live Shadow report

Status: **{status}**

The implementation is fixed to `real_hardware_enabled=false`, `publish_commands=false`, `shadow_only=true`, and `dry_run=true`. Policy outputs pass through the existing physical-unit Action Adapter and then an in-memory Null Sink. No real command client is constructed.

Completed runs: `{len(complete)}`. Discovered run summaries: `{len(summaries)}`.

Until a complete real-input run exists, the answers about 10 Hz stability, arm drift, filter causes and object-position response remain explicitly unmeasured.
""",
    )

    print("generated Live Shadow reports; complete runs:", len(complete))


if __name__ == "__main__":
    main()
