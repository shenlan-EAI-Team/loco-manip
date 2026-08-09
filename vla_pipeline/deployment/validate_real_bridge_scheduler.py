from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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


class SlowFeedbackOnlyO6Transport(MockO6Transport):
    def __init__(self, values: dict[str, np.ndarray], delay_s: float) -> None:
        super().__init__(values)
        self.delay_s = delay_s
        self.feedback_count = 0

    def feedback(self) -> dict[str, np.ndarray]:
        time.sleep(self.delay_s)
        self.feedback_count += 1
        return super().feedback()


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def run_validation(log_path: Path, *, o6_delay_s: float = 0.05) -> dict:
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
    o6 = SlowFeedbackOnlyO6Transport(
        {"left_o6": groups["left_o6"], "right_o6": groups["right_o6"]},
        delay_s=o6_delay_s,
    )
    started = time.monotonic()
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
            o6_position_commands_enabled=False,
        )
        session.arm_hold()
        session.execute_hold(2.0)
        release_error = session.stop("offline wall-clock scheduler validation")
    ended = time.monotonic()

    weights = np.asarray([record["weight"] for record in g1.records], dtype=np.float64)
    sent = np.asarray([record["sent_monotonic_s"] for record in g1.records], dtype=np.float64)
    intervals_ms = np.diff(sent) * 1000.0
    hold_count = 151
    release_count = 100
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    scheduler_rows = [
        row["scheduler"]
        for row in rows
        if row.get("event") == "command" and row.get("scheduler") is not None
    ]
    lateness_ms = np.asarray([row["lateness_ms"] for row in scheduler_rows], dtype=np.float64)
    watchdogs = [row for row in rows if row.get("event") == "watchdog"]
    post_rows = [
        row
        for row in rows
        if row.get("event") == "release_feedback"
        and row.get("phase") == "weight_zero_post_release_monitor"
    ]
    activation_elapsed_s = float(sent[50] - sent[0])
    full_weight_elapsed_s = float(sent[150] - sent[50])
    release_elapsed_s = float(sent[-1] - sent[150])
    checks = {
        "command_count_251": len(weights) == hold_count + release_count,
        "activation_curve_exact": bool(
            np.allclose(weights[:51], np.arange(51) / 50.0, atol=1e-12)
        ),
        "full_weight_exact": bool(np.all(weights[50:hold_count] == 1.0)),
        "release_curve_exact": bool(
            np.allclose(
                weights[hold_count:],
                1.0 - np.arange(1, release_count + 1) / release_count,
                atol=1e-12,
            )
        ),
        "activation_timing_1s": abs(activation_elapsed_s - 1.0) <= 0.04,
        "full_weight_timing_2s": abs(full_weight_elapsed_s - 2.0) <= 0.04,
        "release_timing_2s": abs(release_elapsed_s - 2.0) <= 0.06,
        "interval_p99_below_30ms": _percentile(intervals_ms, 99) < 30.0,
        "interval_max_below_40ms": float(intervals_ms.max()) < 40.0,
        "post_release_25_read_only_samples": len(post_rows) == 25,
        "weight_reached_zero": float(weights[-1]) == 0.0,
        "o6_position_command_count_zero": session.o6_position_command_count == 0,
        "waist_leg_command_count_zero": session.waist_leg_command_count == 0,
        "watchdog_count_zero": len(watchdogs) == 0,
        "release_error_none": release_error is None,
    }
    return {
        "schema_version": 1,
        "result": "PASS" if all(checks.values()) else "FAIL",
        "wall_clock_s": ended - started,
        "o6_feedback_delay_s": o6_delay_s,
        "o6_feedback_count": o6.feedback_count,
        "arm_command_count": len(weights),
        "hold_command_count": hold_count,
        "release_command_count": release_count,
        "activation_elapsed_s": activation_elapsed_s,
        "full_weight_elapsed_s": full_weight_elapsed_s,
        "release_elapsed_s": release_elapsed_s,
        "arm_interval_ms": {
            "mean": float(intervals_ms.mean()),
            "p95": _percentile(intervals_ms, 95),
            "p99": _percentile(intervals_ms, 99),
            "max": float(intervals_ms.max()),
        },
        "scheduler_lateness_ms": {
            "mean": float(lateness_ms.mean()),
            "p95": _percentile(lateness_ms, 95),
            "p99": _percentile(lateness_ms, 99),
            "max": float(lateness_ms.max()),
        },
        "deadline_misses_over_20ms": int(np.sum(lateness_ms >= 20.0)),
        "o6_position_command_count": session.o6_position_command_count,
        "waist_leg_command_count": session.waist_leg_command_count,
        "watchdog_count": len(watchdogs),
        "post_release_read_only_samples": len(post_rows),
        "checks": checks,
        "log_path": str(log_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    report = run_validation(Path(args.log))
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
