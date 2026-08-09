import ast
from pathlib import Path

import yaml


CONTROL = Path(__file__).resolve().parents[3] / "control"
CONFIG = CONTROL / "main/teleop/configs/g1_lowcmd_guard.yaml"


def test_only_unitree_writer_class_constructs_rt_lowcmd() -> None:
    matches = []
    for path in (CONTROL / "real_safe").rglob("*.py"):
        if 'ChannelPublisher("rt/lowcmd"' in path.read_text():
            matches.append(path.name)
    assert matches == ["unitree_backend.py"]

    source = (CONTROL / "real_safe/lowcmd_guard/unitree_backend.py").read_text()
    before_writer = source.split("class UnitreeLowCmdWriter", 1)[0]
    assert "ChannelPublisher" not in before_writer
    mode_constructor = source.split("class UnitreeMotionModeClient", 1)[1].split(
        "def check_mode", 1
    )[0]
    assert ".ReleaseMode(" not in mode_constructor
    assert ".SelectMode(" not in mode_constructor
    assert source.count("._client.ReleaseMode()") == 1
    assert source.count("._client.SelectMode(owner)") == 1


def test_read_only_cli_cannot_construct_writer_branch() -> None:
    source = (CONTROL / "main/teleop/run_g1_lowcmd_guard.py").read_text()
    tree = ast.parse(source)
    assert tree is not None
    read_only = source.split('if args.mode == "read-only":', 1)[1].split(
        "runtime = LowCmdGuardRuntime", 1
    )[0]
    assert "UnitreeLowCmdWriter(" not in read_only
    assert "release_mode(" not in read_only
    assert "select_mode(" not in read_only


def test_frequency_and_gains_match_source_backed_project_values() -> None:
    guard = yaml.safe_load(CONFIG.read_text())
    wbc = yaml.safe_load((CONFIG.parent / "g1_29dof_gear_wbc.yaml").read_text())
    assert guard["official_reference_transport_frequency_hz"] == 500.0
    assert guard["transport_frequency_hz"] == 500.0
    assert guard["measured_minimum_transport_frequency_hz"] is None
    assert guard["policy_target_frequency_hz"] == 50.0
    assert guard["kp"] == wbc["MOTOR_KP"]
    assert guard["kd"] == wbc["MOTOR_KD"]
    assert min(guard["kp"]) >= 0
    assert min(guard["kd"]) >= 0
    assert guard["real_execution_enabled"] is False
    assert guard["recovery_handoff_verified"] is False
    assert guard["commissioning_execution_enabled"] is False


def test_unused_hg_slots_are_explicitly_disabled() -> None:
    source = (CONTROL / "real_safe/lowcmd_guard/unitree_backend.py").read_text()
    assert "for index in range(29, len(self._message.motor_cmd))" in source
    block = source.split("for index in range(29, len(self._message.motor_cmd))", 1)[1]
    for field in ("mode", "q", "dq", "kp", "kd", "tau"):
        assert f"motor.{field} = 0" in block


def test_active_lowcmd_probe_is_subscriber_only() -> None:
    source = (CONTROL / "main/teleop/run_g1_active_lowcmd_read_only_probe.py").read_text()
    for forbidden in (
        "ChannelPublisher",
        "UnitreeLowCmdWriter",
        "ReleaseMode",
        "SelectMode",
        ".Write(",
    ):
        assert forbidden not in source
