"""In-memory target recorder for Live Shadow; it has no transport or SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NullActionSink:
    """Record hypothetical targets without constructing a hardware client."""

    records: list[dict[str, Any]] = field(default_factory=list)
    command_publish_attempts: int = 0
    control_ownership_requests: int = 0
    real_sdk_objects_created: int = 0

    def record(self, timestamp: float, values: dict[str, np.ndarray]) -> None:
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("Null sink refuses non-finite hypothetical targets")
        self.records.append(
            {
                "timestamp": float(timestamp),
                "values": {
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in values.items()
                },
            }
        )

    def metrics(self) -> dict[str, int]:
        return {
            "records": len(self.records),
            "command_publish_attempts": self.command_publish_attempts,
            "control_ownership_requests": self.control_ownership_requests,
            "real_sdk_objects_created": self.real_sdk_objects_created,
        }
