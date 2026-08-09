from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from deployment.real_bridge.controller import RealBridgeSession
from deployment.real_bridge.logging import JsonlBridgeLogger
from deployment.real_bridge.models import FeedbackSnapshot
from deployment.real_bridge.transports import MockG1Transport, MockO6Transport


class TimedMockG1Transport(MockG1Transport):
    def send_arms(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = super().send_arms(*args, **kwargs)
        self.records[-1]["sent_monotonic_s"] = time.monotonic()
        return result


class BlockingO6Transport(MockO6Transport):
    def __init__(
        self,
        values: dict[str, np.ndarray],
        *,
        setter_delay_s: float,
        setter_timeout: bool,
    ) -> None:
        super().__init__(values)
        self.setter_delay_s = setter_delay_s
        self.setter_timeout = setter_timeout
        self.setter_attempts = 0
        self.right_command_count = 0

    def send_left_hand(self, left_raw: list[int]) -> dict:
        self.setter_attempts += 1
        time.sleep(self.setter_delay_s)
        if self.setter_timeout:
            raise TimeoutError("injected O6 setter timeout")
        return super().send_left_hand(left_raw)


def _plan(groups: dict[str, np.ndarray]) -> dict[str, Any]:
    target = {key: value.copy() for key, value in groups.items()}
    target["left_arm"] += 0.005
    target["right_arm"] -= 0.005
    target["left_o6"] += 2.0
    frames = []
    for index in range(15):
        raw = {key: value.tolist() for key, value in target.items()}
        raw["right_o6"] = [0.0] * 6
        frames.append(
            {
                "inference_index": index // 3,
                "policy_raw_absolute": raw,
                "adapter_absolute": raw,
                "ordinary_safety": {key: value.tolist() for key, value in target.items()},
            }
        )
    return {"frames": frames}


def _stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
        "min": float(data.min()),
    }


def run_case(output_dir: Path, label: str, delay_s: float, timeout: bool) -> dict[str, Any]:
    groups = {
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
        "left_o6": np.full(6, 50.0),
        "right_o6": np.full(6, 50.0),
    }
    snapshot = FeedbackSnapshot.create(
        groups,
        g1_mode_machine=5,
        g1_mode_pr=0,
        waist=np.zeros(3),
    )
    g1 = TimedMockG1Transport(snapshot)
    o6 = BlockingO6Transport(
        {"left_o6": groups["left_o6"], "right_o6": groups["right_o6"]},
        setter_delay_s=delay_s,
        setter_timeout=timeout,
    )
    log_path = output_dir / f"{label}.jsonl"
    started = time.monotonic()
    micro_error = None
    with JsonlBridgeLogger(log_path) as logger:
        session = RealBridgeSession(
            g1,
            o6,
            logger,
            activation_ramp_s=1.0,
            release_ramp_s=2.0,
            post_release_monitor_s=0.5,
            o6_feedback_stale_timeout_s=0.2,
            scheduler_max_lateness_s=0.04,
            o6_position_commands_enabled=True,
        )
        session.arm_hold()
        session.execute_hold(2.0)
        try:
            session.execute_micro(_plan(groups))
        except Exception as exc:
            micro_error = f"{type(exc).__name__}: {exc}"
            release_error = session.stop(micro_error, fault=True)
        else:
            release_error = session.stop("mock micro complete")
    elapsed = time.monotonic() - started

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    arm_commands = [
        row for row in rows if row.get("event") == "command" and row.get("transport") == "g1_arm_sdk"
    ]
    micro_commands = [
        row for row in arm_commands if row.get("scheduler", {}).get("phase") == "model_micro"
    ]
    release_commands = [
        row for row in arm_commands if row.get("scheduler", {}).get("phase") == "release"
    ]
    arm_times = [
        row["scheduler"]["actual_publish_start_monotonic_ns"] / 1e9
        for row in arm_commands
        if row.get("scheduler") is not None
    ]
    intervals_ms = np.diff(np.asarray(arm_times)) * 1000.0
    o6_commands = [
        row for row in rows if row.get("event") == "command" and row.get("transport") == "o6_can_position"
    ]
    worker_errors = [row for row in rows if row.get("event") == "o6_worker_error"]
    watchdogs = [row for row in rows if row.get("event") == "watchdog"]
    post_rows = [
        row
        for row in rows
        if row.get("event") == "release_feedback"
        and row.get("phase") == "weight_zero_post_release_monitor"
    ]
    release_weights = [row["ownership"]["arm_sdk_weight"] for row in release_commands]
    expected_timeout = timeout and micro_error is not None and len(worker_errors) == 1
    checks = {
        "arm_interval_mean_near_20ms": abs(float(intervals_ms.mean()) - 20.0) < 1.0,
        "arm_interval_p99_below_30ms": float(np.percentile(intervals_ms, 99)) < 30.0,
        "arm_interval_max_below_40ms": float(intervals_ms.max()) < 40.0,
        "no_catch_up_burst": float(intervals_ms.min()) >= 19.9,
        "right_o6_command_count_zero": o6.right_command_count == 0,
        "worker_owns_all_left_o6_commands": all(
            row["o6_worker"]["thread"] == "o6-independent-io-worker"
            for row in o6_commands
        ),
        "release_has_100_messages": len(release_commands) == 100,
        "release_reaches_zero": len(release_weights) == 100 and release_weights[-1] == 0.0,
        "release_curve_exact": bool(
            np.allclose(release_weights, 1.0 - np.arange(1, 101) / 100.0, atol=1e-12)
        ),
        "release_error_none": release_error is None,
        "post_release_25_samples": len(post_rows) == 25,
        "normal_or_expected_timeout": (
            expected_timeout
            if timeout
            else micro_error is None and len(micro_commands) == 25 and not worker_errors
        ),
        "o6_success_count_matches_records": session.o6_position_command_count == len(o6.records),
    }
    return {
        "label": label,
        "setter_delay_ms": delay_s * 1000.0,
        "setter_timeout": timeout,
        "wall_clock_s": elapsed,
        "result": "PASS" if all(checks.values()) else "FAIL",
        "arm_interval_ms": _stats(intervals_ms.tolist()),
        "arm_command_count": len(arm_commands),
        "micro_arm_command_count": len(micro_commands),
        "release_arm_command_count": len(release_commands),
        "left_o6_setter_attempts": o6.setter_attempts,
        "left_o6_successful_commands": len(o6.records),
        "right_o6_command_count": o6.right_command_count,
        "micro_error": micro_error,
        "worker_error_count": len(worker_errors),
        "watchdog_count": len(watchdogs),
        "release_error": release_error,
        "final_weight": g1.records[-1]["weight"],
        "checks": checks,
        "log": str(log_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cases = [
        run_case(output_dir, "setter_block_20ms", 0.020, False),
        run_case(output_dir, "setter_block_50ms", 0.050, False),
        run_case(output_dir, "setter_block_100ms", 0.100, False),
        run_case(output_dir, "setter_timeout", 0.100, True),
    ]
    summary = {
        "schema_version": 1,
        "result": "PASS" if all(case["result"] == "PASS" for case in cases) else "FAIL",
        "wall_clock_s": time.monotonic() - started,
        "real_hardware_objects_created": 0,
        "cases": cases,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
