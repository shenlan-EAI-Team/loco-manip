"""Read/write-free mock of the future G1 arm SDK boundary."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class MockG1SDK:
    def __init__(self, feedback_delay_s: float = 0.0) -> None:
        self.feedback_delay_s = feedback_delay_s
        self.connected = True
        self.inject_timeout = False
        self.inject_nan = False
        self.records: list[dict[str, Any]] = []

    def send_arm_targets(
        self, left_arm: np.ndarray, right_arm: np.ndarray, timestamp: float
    ) -> bool:
        if not self.connected:
            raise ConnectionError("mock G1 network disconnected")
        if self.inject_timeout:
            raise TimeoutError("mock G1 timeout")
        left = np.asarray(left_arm, dtype=np.float32).copy()
        right = np.asarray(right_arm, dtype=np.float32).copy()
        if self.inject_nan:
            left[0] = np.nan
        if self.feedback_delay_s:
            time.sleep(self.feedback_delay_s)
        self.records.append({"timestamp": timestamp, "left_arm": left, "right_arm": right})
        return True
