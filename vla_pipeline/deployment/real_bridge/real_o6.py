from __future__ import annotations

import json
import select
import subprocess
import time
from typing import Any

import numpy as np

from .mapping import raw_to_percent


class RemoteO6Transport:
    """Persistent SSH stdio transport; no TCP command socket is created."""

    PREFIX = "REAL_BRIDGE_JSON "

    def __init__(
        self,
        *,
        host: str,
        user: str,
        remote_python: str,
        remote_agent: str,
        timeout_s: float = 1.0,
    ) -> None:
        self.timeout_s = 8.0
        self.diagnostics: list[str] = []
        self.process = subprocess.Popen(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                f"{user}@{host}",
                remote_python,
                remote_agent,
                "--serve",
                "--left-can",
                "can2",
                "--right-can",
                "can1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._request({"operation": "init", "protocol": "g1_o6_real_bridge_v1"})
        self.timeout_s = timeout_s

    def _request(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise ConnectionError(f"remote O6 agent exited with {self.process.returncode}")
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.process.stdout], [], [], deadline - time.monotonic())
            if not readable:
                break
            line = self.process.stdout.readline()
            if not line:
                raise ConnectionError("remote O6 agent closed stdout")
            line = line.rstrip()
            if not line.startswith(self.PREFIX):
                self.diagnostics.append(line)
                continue
            response = json.loads(line[len(self.PREFIX) :])
            if not response.get("ok"):
                raise RuntimeError(f"remote O6 error: {response}")
            return response
        raise TimeoutError(f"remote O6 operation timed out: {value.get('operation')}")

    def feedback(self) -> dict[str, np.ndarray]:
        response = self._request({"operation": "feedback"})
        return {
            "left_o6": np.asarray(
                [raw_to_percent(value) for value in response["left_raw_255"]], dtype=np.float64
            ),
            "right_o6": np.asarray(
                [raw_to_percent(value) for value in response["right_raw_255"]], dtype=np.float64
            ),
        }

    def send_left_hand(self, left_raw: list[int]) -> dict[str, Any]:
        return self._request(
            {
                "operation": "command_left",
                "left_raw_255": left_raw,
            }
        )

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request({"operation": "close"})
            finally:
                try:
                    self.process.wait(timeout=4.0)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=4.0)
