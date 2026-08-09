from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np


class JsonlBridgeLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        def default(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()
            raise TypeError(type(value).__name__)

        record = {
            "event": event,
            "wall_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=default) + "\n"
        with self._lock:
            self._stream.write(line)

    def close(self) -> None:
        with self._lock:
            self._stream.close()

    def __enter__(self) -> "JsonlBridgeLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
