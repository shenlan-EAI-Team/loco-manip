"""Observation source contract shared by replay and future live shadow mode."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class ObservationSample:
    episode_index: int
    frame_index: int
    dataset_timestamp: float
    monotonic_timestamp: float
    observation: dict[str, Any]
    flat_observation: dict[str, Any]
    source_metadata: dict[str, Any] = field(default_factory=dict)


class ObservationSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def get_observation(self) -> ObservationSample | None: ...

    @abstractmethod
    def stop(self) -> None: ...
