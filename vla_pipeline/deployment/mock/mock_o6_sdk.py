"""Read/write-free mock of the future dual O6 SDK boundary."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class MockO6SDK:
    def __init__(self, feedback_delay_s: float = 0.0) -> None:
        self.feedback_delay_s = feedback_delay_s
        self.connected = True
        self.inject_timeout = False
        self.inject_nan = False
        self.records: list[dict[str, Any]] = []

    def send_hand_targets(
        self, left_o6: np.ndarray, right_o6: np.ndarray, timestamp: float
    ) -> bool:
        if not self.connected:
            raise ConnectionError("mock O6 network disconnected")
        if self.inject_timeout:
            raise TimeoutError("mock O6 timeout")
        left = np.asarray(left_o6, dtype=np.float32).copy()
        right = np.asarray(right_o6, dtype=np.float32).copy()
        if self.inject_nan:
            left[0] = np.nan
        if self.feedback_delay_s:
            time.sleep(self.feedback_delay_s)
        self.records.append({"timestamp": timestamp, "left_o6": left, "right_o6": right})
        return True
