"""Action chunk buffering and interpolation without hardware I/O."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BufferedTarget:
    timestamp: float
    values: dict[str, np.ndarray]


class ActionBuffer:
    def __init__(self, control_hz: float, publish_hz: float) -> None:
        self.control_hz = float(control_hz)
        self.publish_hz = float(publish_hz)
        self._queue: deque[BufferedTarget] = deque()
        self.underruns = 0

    def clear(self) -> None:
        self._queue.clear()

    def push_interpolated(
        self,
        start: dict[str, np.ndarray],
        chunk: list[dict[str, np.ndarray]],
        start_timestamp: float,
    ) -> None:
        if not chunk:
            return
        publish_count = max(1, round(len(chunk) * self.publish_hz / self.control_hz))
        for publish_index in range(1, publish_count + 1):
            time_from_start = publish_index / self.publish_hz
            control_position = time_from_start * self.control_hz
            segment = min(int(control_position), len(chunk) - 1)
            alpha = min(1.0, control_position - segment)
            segment_start = start if segment == 0 else chunk[segment - 1]
            segment_end = chunk[segment]
            values = {
                key: (
                    (1.0 - alpha) * segment_start[key] + alpha * segment_end[key]
                ).astype(np.float32)
                for key in segment_end
            }
            self._queue.append(
                BufferedTarget(start_timestamp + time_from_start, values)
            )

    def pop(self) -> BufferedTarget | None:
        if not self._queue:
            self.underruns += 1
            return None
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)
