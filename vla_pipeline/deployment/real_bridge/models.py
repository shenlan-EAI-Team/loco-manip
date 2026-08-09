from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any

import numpy as np


ACTION_DIMS = {"left_arm": 7, "right_arm": 7, "left_o6": 6, "right_o6": 6}


class BridgeState(str, Enum):
    DISABLED = "DISABLED"
    READY = "READY"
    ARMED_HOLD = "ARMED_HOLD"
    ARMED_MICRO = "ARMED_MICRO"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


def validate_groups(groups: dict[str, Any], *, label: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if set(groups) != set(ACTION_DIMS):
        raise ValueError(f"{label}: expected groups {sorted(ACTION_DIMS)}, got {sorted(groups)}")
    for key, dimension in ACTION_DIMS.items():
        value = np.asarray(groups[key], dtype=np.float64).reshape(-1)
        if value.shape != (dimension,):
            raise ValueError(f"{label}.{key}: expected ({dimension},), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{label}.{key}: contains NaN or Inf")
        result[key] = value.copy()
    return result


@dataclass(frozen=True)
class FeedbackSnapshot:
    groups: dict[str, np.ndarray]
    monotonic_ns: int
    wall_ns: int
    g1_mode_machine: int | None = None
    g1_mode_pr: int | None = None
    waist: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        groups: dict[str, Any],
        *,
        monotonic_ns: int | None = None,
        wall_ns: int | None = None,
        g1_mode_machine: int | None = None,
        g1_mode_pr: int | None = None,
        waist: Any | None = None,
    ) -> "FeedbackSnapshot":
        parsed = validate_groups(groups, label="feedback")
        parsed_waist = None if waist is None else np.asarray(waist, dtype=np.float64).reshape(-1)
        if parsed_waist is not None and (
            parsed_waist.shape != (3,) or not np.isfinite(parsed_waist).all()
        ):
            raise ValueError("feedback.waist must contain three finite radians")
        return cls(
            groups=parsed,
            monotonic_ns=time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns),
            wall_ns=time.time_ns() if wall_ns is None else int(wall_ns),
            g1_mode_machine=g1_mode_machine,
            g1_mode_pr=g1_mode_pr,
            waist=parsed_waist,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": {key: value.tolist() for key, value in self.groups.items()},
            "monotonic_ns": self.monotonic_ns,
            "wall_ns": self.wall_ns,
            "g1_mode_machine": self.g1_mode_machine,
            "g1_mode_pr": self.g1_mode_pr,
            "waist": None if self.waist is None else self.waist.tolist(),
        }
