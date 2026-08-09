"""Monotonic periodic scheduler that never emits catch-up write bursts."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Thread
import time
from typing import Callable

import numpy as np


@dataclass
class SchedulerMetrics:
    intervals_s: list[float] = field(default_factory=list)
    missed_deadlines: int = 0
    callback_errors: int = 0

    def summary(self) -> dict[str, float | int | None]:
        if not self.intervals_s:
            return {
                "samples": 0,
                "mean_s": None,
                "p99_s": None,
                "max_s": None,
                "missed_deadlines": self.missed_deadlines,
                "callback_errors": self.callback_errors,
            }
        return {
            "samples": len(self.intervals_s),
            "mean_s": float(np.mean(self.intervals_s)),
            "p99_s": float(np.percentile(self.intervals_s, 99)),
            "max_s": max(self.intervals_s),
            "missed_deadlines": self.missed_deadlines,
            "callback_errors": self.callback_errors,
        }


class NoCatchUpScheduler:
    def __init__(
        self,
        frequency_hz: float,
        callback: Callable[[], None],
        *,
        on_error: Callable[[Exception], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        self.period_s = 1.0 / frequency_hz
        self.callback = callback
        self.on_error = on_error
        self.clock = clock
        self.metrics = SchedulerMetrics()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("scheduler already started")
        self._thread = Thread(target=self._run, name="g1-lowcmd-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # The takeover path performs the first write synchronously. The periodic
        # writer starts one full period later to avoid a zero-interval double write.
        next_deadline = self.clock() + self.period_s
        last_callback = None
        while not self._stop.is_set():
            remaining = next_deadline - self.clock()
            if remaining > 0 and self._stop.wait(remaining):
                break
            started = self.clock()
            if last_callback is not None:
                self.metrics.intervals_s.append(started - last_callback)
            last_callback = started
            try:
                self.callback()
            except Exception as exc:
                self.metrics.callback_errors += 1
                self.on_error(exc)
                break
            next_deadline += self.period_s
            now = self.clock()
            if next_deadline <= now:
                self.metrics.missed_deadlines += 1
                # Skip all expired slots. Never emit a catch-up burst.
                next_deadline = now + self.period_s

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("LowCmd scheduler did not stop within timeout")
