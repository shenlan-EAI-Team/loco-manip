from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time


@dataclass(frozen=True)
class GateSettings:
    hardware_transport_enabled: bool = False
    command_publication_enabled: bool = False
    micro_motion_armed: bool = False

    @property
    def all_enabled(self) -> bool:
        return (
            self.hardware_transport_enabled
            and self.command_publication_enabled
            and self.micro_motion_armed
        )

    def require_all(self) -> None:
        disabled = [
            name
            for name, enabled in (
                ("hardware_transport_enabled", self.hardware_transport_enabled),
                ("command_publication_enabled", self.command_publication_enabled),
                ("micro_motion_armed", self.micro_motion_armed),
            )
            if not enabled
        ]
        if disabled:
            raise PermissionError("real transport blocked; disabled gates: " + ", ".join(disabled))


class OneTimeToken:
    """A file-backed token whose successful verification is consumed atomically."""

    @staticmethod
    def issue(
        path: str | Path,
        *,
        ttl_s: float = 3600.0,
        bound_sha256: str | None = None,
    ) -> str:
        token_path = Path(path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        if token_path.exists():
            raise FileExistsError(f"refusing to replace unconsumed token: {token_path}")
        token = secrets.token_urlsafe(24)
        record = {
            "schema_version": 1,
            "sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
            "issued_wall_ns": time.time_ns(),
            "expires_wall_ns": time.time_ns() + int(ttl_s * 1e9),
            "consumed": False,
            "bound_sha256": bound_sha256,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(token_path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        return token

    @staticmethod
    def consume(
        path: str | Path,
        supplied: str,
        *,
        bound_sha256: str | None = None,
    ) -> None:
        token_path = Path(path)
        consumed_path = token_path.with_suffix(token_path.suffix + ".consumed")
        record = json.loads(token_path.read_text(encoding="utf-8"))
        if record.get("consumed"):
            raise PermissionError("confirmation token was already consumed")
        if time.time_ns() > int(record["expires_wall_ns"]):
            raise PermissionError("confirmation token expired")
        actual = hashlib.sha256(supplied.encode("ascii")).hexdigest()
        if not hmac.compare_digest(actual, str(record["sha256"])):
            raise PermissionError("confirmation token mismatch")
        recorded_binding = record.get("bound_sha256")
        if recorded_binding != bound_sha256:
            raise PermissionError("confirmation token is not bound to this command plan")
        record["consumed"] = True
        record["consumed_wall_ns"] = time.time_ns()
        fd = os.open(consumed_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        token_path.unlink()
