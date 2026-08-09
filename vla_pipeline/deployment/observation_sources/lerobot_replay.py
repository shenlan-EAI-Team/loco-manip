"""Replay test episodes as a hardware-free 30 Hz observation stream."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from deployment.common import build_policy_observation, make_loader
from deployment.observation_sources.base import ObservationSample, ObservationSource


class LeRobotReplayObservationSource(ObservationSource):
    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, Any],
        *,
        real_time: bool = False,
        playback_speed: float = 1.0,
        fps: float = 30.0,
    ) -> None:
        if playback_speed not in (0.25, 0.5, 1.0):
            raise ValueError("playback_speed must be one of 0.25, 0.5, 1.0")
        self.loader = make_loader(Path(dataset_path), modality_configs)
        self.modality_configs = modality_configs
        self.real_time = real_time
        self.playback_speed = playback_speed
        self.fps = fps
        self._episode = 0
        self._frame = 0
        self._started = False
        self._next_deadline = 0.0
        self._trajectory = None

    def start(self) -> None:
        self._episode = 0
        self._frame = 0
        self._started = True
        self._next_deadline = time.monotonic()
        self._trajectory = None

    def get_observation(self) -> ObservationSample | None:
        if not self._started:
            raise RuntimeError("start() must be called first")
        while self._episode < len(self.loader):
            if self._trajectory is None:
                self._trajectory = self.loader[self._episode]
            trajectory = self._trajectory
            if self._frame >= len(trajectory):
                self._episode += 1
                self._frame = 0
                self._trajectory = None
                self._next_deadline = time.monotonic()
                continue
            if self.real_time:
                now = time.monotonic()
                if now < self._next_deadline:
                    time.sleep(self._next_deadline - now)
            policy_obs, flat_obs, _ = build_policy_observation(
                trajectory, self._frame, self.modality_configs
            )
            row = trajectory.iloc[self._frame]
            if "timestamp" in trajectory.columns:
                dataset_timestamp = float(row["timestamp"])
            else:
                # LeRobotEpisodeLoader intentionally projects only configured
                # modalities. The converted source is fixed-rate 30 Hz, so this
                # is the exact replay timeline when timestamp is not projected.
                dataset_timestamp = self._frame / self.fps
            sample = ObservationSample(
                episode_index=self._episode,
                frame_index=self._frame,
                dataset_timestamp=dataset_timestamp,
                monotonic_timestamp=time.monotonic(),
                observation=policy_obs,
                flat_observation=flat_obs,
            )
            self._frame += 1
            self._next_deadline += 1.0 / (self.fps * self.playback_speed)
            return sample
        return None

    def stop(self) -> None:
        self._started = False
        self._trajectory = None
