"""Physical-unit Action Adapter. Real hardware is deliberately unsupported here."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from deployment.action_buffer import ActionBuffer
from deployment.common import ACTION_KEYS
from deployment.mock import MockG1SDK, MockO6SDK
from deployment.safety_filter import SafetyFilter


class ActionAdapter:
    """Consume standard Policy API absolute outputs and drive mock SDKs only.

    No denormalization and no q_current addition occur in this class. The local
    Policy API contract test proves both transformations have already happened.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        g1_sdk: MockG1SDK | None = None,
        o6_sdk: MockO6SDK | None = None,
    ) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text())
        if not self.config.get("dry_run", False):
            raise RuntimeError("ActionAdapter requires dry_run=true")
        if self.config.get("real_hardware_enabled", False):
            raise RuntimeError("Offline ActionAdapter refuses real_hardware_enabled=true")
        self.g1_sdk = g1_sdk or MockG1SDK()
        self.o6_sdk = o6_sdk or MockO6SDK()
        self.filter = SafetyFilter(self.config)
        self.buffer = ActionBuffer(
            self.config["control_timeline_hz"], self.config["sdk_publish_hz"]
        )
        self.last_safe: dict[str, np.ndarray] = {}
        self.last_policy_time: float | None = None
        self.estopped = False
        self.watchdog_holds = 0
        self.sdk_errors: list[str] = []

    def reset(self, state: dict[str, np.ndarray]) -> None:
        initial = {
            key: np.asarray(state[key], dtype=np.float32).reshape(-1)
            for key in ACTION_KEYS
        }
        self.filter.reset(initial)
        self.last_safe = {key: value.copy() for key, value in initial.items()}
        self.last_policy_time = time.monotonic()
        self.estopped = False
        self.buffer.clear()

    def prepare_chunk(
        self,
        action: dict[str, np.ndarray],
        *,
        timestamp: float,
    ) -> list[dict[str, np.ndarray]]:
        if not self.last_safe:
            raise RuntimeError("reset(current_state) must be called first")
        chunks: dict[str, np.ndarray] = {}
        horizon = None
        for key in ACTION_KEYS:
            value = np.asarray(action[key], dtype=np.float32)
            if value.ndim == 3:
                if value.shape[0] != 1:
                    raise ValueError(f"{key}: expected batch size 1, got {value.shape}")
                value = value[0]
            if value.ndim != 2:
                raise ValueError(f"{key}: expected (T,D), got {value.shape}")
            horizon = value.shape[0] if horizon is None else horizon
            if value.shape[0] != horizon:
                raise ValueError("Action groups have different horizons")
            chunks[key] = value

        execute = min(int(self.config["execution_horizon"]), int(horizon))
        dt = 1.0 / float(self.config["control_timeline_hz"])
        start = {key: value.copy() for key, value in self.last_safe.items()}
        safe_chunk: list[dict[str, np.ndarray]] = []
        for index in range(execute):
            target = {key: chunks[key][index] for key in ACTION_KEYS}
            filtered = self.filter.filter_step(target, dt=dt)
            safe_chunk.append(filtered)
            self.last_safe = {key: value.copy() for key, value in filtered.items()}
        self.buffer.push_interpolated(start, safe_chunk, timestamp)
        self.last_policy_time = time.monotonic()
        return safe_chunk

    def drain_to_mock(self) -> int:
        sent = 0
        while len(self.buffer):
            item = self.buffer.pop()
            if item is None:
                break
            values = item.values
            if not all(np.isfinite(values[key]).all() for key in ACTION_KEYS):
                self.sdk_errors.append("adapter blocked non-finite target before mock SDK")
                continue
            try:
                self.g1_sdk.send_arm_targets(
                    values["left_arm"], values["right_arm"], item.timestamp
                )
                self.o6_sdk.send_hand_targets(
                    values["left_o6"], values["right_o6"], item.timestamp
                )
                sent += 1
            except (TimeoutError, ConnectionError) as exc:
                self.sdk_errors.append(str(exc))
                self.watchdog_holds += 1
                break
        return sent

    def watchdog_target(self, now: float | None = None) -> dict[str, np.ndarray]:
        if not self.last_safe:
            raise RuntimeError("Adapter has no safe target")
        now = time.monotonic() if now is None else now
        stale = self.last_policy_time is None or (
            now - self.last_policy_time > float(self.config["watchdog_timeout_s"])
        )
        if stale or self.estopped:
            self.watchdog_holds += 1
        return {key: value.copy() for key, value in self.last_safe.items()}

    def emergency_stop(self) -> dict[str, np.ndarray]:
        self.estopped = True
        self.buffer.clear()
        return self.watchdog_target()

    def metrics(self) -> dict[str, Any]:
        return {
            "dry_run": self.config["dry_run"],
            "real_hardware_enabled": self.config["real_hardware_enabled"],
            "filter_counters": {
                key: counter.as_dict() for key, counter in self.filter.counters.items()
            },
            "buffer_underruns": self.buffer.underruns,
            "watchdog_holds": self.watchdog_holds,
            "sdk_errors": list(self.sdk_errors),
            "mock_g1_records": len(self.g1_sdk.records),
            "mock_o6_records": len(self.o6_sdk.records),
        }
