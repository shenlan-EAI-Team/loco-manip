from pathlib import Path

import pytest

from decoupled_wbc.control.real_safe.lowcmd_guard import ExclusiveGuardLock


def test_guard_process_lock_allows_exactly_one_local_instance(tmp_path: Path) -> None:
    path = tmp_path / "guard.lock"
    first = ExclusiveGuardLock(path)
    second = ExclusiveGuardLock(path)
    first.acquire()
    assert "pid=" in path.read_text()
    with pytest.raises(RuntimeError, match="another G1 LowCmd guard"):
        second.acquire()
    first.close()
    second.acquire()
    second.close()
