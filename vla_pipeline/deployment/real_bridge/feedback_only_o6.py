from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
import time

import numpy as np


class O6FeedbackOnlySubprocessTransport:
    """Read schema-v2 O6 feedback from a SUB-only helper; no setter exists."""

    PREFIX = "O6_FEEDBACK_ONLY_JSON "

    def __init__(
        self,
        python: str,
        script: str | Path,
        endpoint: str,
        *,
        timeout_s: float = 1.0,
    ) -> None:
        self.timeout_s = timeout_s
        self.process = subprocess.Popen(
            [python, "-u", str(script), "--endpoint", endpoint],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def feedback(self) -> dict[str, np.ndarray]:
        if self.process.poll() is not None:
            detail = "" if self.process.stderr is None else self.process.stderr.read()
            raise ConnectionError(f"O6 feedback-only reader exited: {detail}")
        assert self.process.stdout is not None
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            readable, _, _ = select.select(
                [self.process.stdout], [], [], deadline - time.monotonic()
            )
            if not readable:
                break
            line = self.process.stdout.readline().rstrip()
            if not line.startswith(self.PREFIX):
                continue
            value = json.loads(line[len(self.PREFIX) :])
            return {
                key: np.asarray(value[key], dtype=np.float64)
                for key in ("left_o6", "right_o6")
            }
        raise TimeoutError("fresh feedback-only O6 schema-v2 frame unavailable")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
