from pathlib import Path

import pytest

from decoupled_wbc.control.real_safe.lowcmd_guard import OneTimeTokenFile


def issue(path: Path, token: str) -> None:
    path.write_text(token + "\n")
    path.chmod(0o600)


def test_token_is_persistently_consumed_before_reuse(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.token"
    issue(path, "one-shot")
    token_file = OneTimeTokenFile(path)
    assert token_file.load() == "one-shot"
    marker = token_file.consume("one-shot")
    assert marker.exists()
    assert not path.exists()

    issue(path, "one-shot")
    with pytest.raises(FileExistsError):
        token_file.consume("one-shot")


def test_token_requires_private_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle.token"
    issue(path, "one-shot")
    path.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        OneTimeTokenFile(path).load()

    path.unlink()
    target = tmp_path / "target"
    issue(target, "one-shot")
    path.symlink_to(target)
    with pytest.raises(PermissionError, match="non-symlink"):
        OneTimeTokenFile(path).load()
