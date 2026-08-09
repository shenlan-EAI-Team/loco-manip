from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .models import FeedbackSnapshot


class G1Transport(Protocol):
    def feedback(self) -> FeedbackSnapshot: ...
    def send_arms(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        weight: float,
        mode_machine: int,
        mode_pr: int,
    ) -> dict: ...
    def close(self) -> None: ...


class O6Transport(Protocol):
    def feedback(self) -> dict[str, np.ndarray]: ...
    def send_left_hand(self, left_raw: list[int]) -> dict: ...
    def close(self) -> None: ...


@dataclass
class MockG1Transport:
    snapshot: FeedbackSnapshot
    records: list[dict] = field(default_factory=list)

    def feedback(self) -> FeedbackSnapshot:
        return self.snapshot

    def send_arms(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        weight: float,
        mode_machine: int,
        mode_pr: int,
    ) -> dict:
        record = {
            "left": list(map(float, left)),
            "right": list(map(float, right)),
            "weight": float(weight),
            "mode_machine": int(mode_machine),
            "mode_pr": int(mode_pr),
        }
        self.records.append(record)
        return record

    def close(self) -> None:
        return None


@dataclass
class MockO6Transport:
    values: dict[str, np.ndarray]
    records: list[dict] = field(default_factory=list)

    def feedback(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self.values.items()}

    def send_left_hand(self, left_raw: list[int]) -> dict:
        record = {"left_raw_255": list(left_raw), "right_o6_command": None}
        self.records.append(record)
        return record

    def close(self) -> None:
        return None
