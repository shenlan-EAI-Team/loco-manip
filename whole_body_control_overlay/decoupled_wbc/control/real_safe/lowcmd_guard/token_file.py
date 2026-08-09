"""Persistent one-shot authorization token for a local guard lifecycle."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import time


class OneTimeTokenFile:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> str:
        info = self.path.lstat()
        if not stat.S_ISREG(info.st_mode) or self.path.is_symlink():
            raise PermissionError("lifecycle token must be a regular non-symlink file")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError("lifecycle token permissions must be 0600 or stricter")
        token = self.path.read_text().strip()
        if not token:
            raise PermissionError("lifecycle token file is empty")
        return token

    def consume(self, expected_token: str) -> Path:
        if self.load() != expected_token:
            raise PermissionError("lifecycle token changed before authorization commit")
        digest = hashlib.sha256(expected_token.encode("utf-8")).hexdigest()
        marker = self.path.with_name(f".{self.path.name}.consumed.{digest}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(marker, flags, 0o600)
        try:
            payload = f"pid={os.getpid()}\nconsumed_unix_ns={time.time_ns()}\n"
            os.write(descriptor, payload.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.path.unlink()
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return marker
