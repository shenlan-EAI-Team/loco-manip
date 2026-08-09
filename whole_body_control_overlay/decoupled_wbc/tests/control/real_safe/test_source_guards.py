import ast
from pathlib import Path


CONTROL = Path(__file__).resolve().parents[3] / "control"


def test_real_state_constructor_does_not_release_motion_mode() -> None:
    source = (CONTROL / "envs/g1/utils/state_processor.py").read_text()
    constructor = source.split("def __init__", 1)[1].split("def _get_motion_switcher", 1)[0]
    assert "MotionSwitcherClient()" not in constructor
    assert "ReleaseMode()" not in constructor
    module_imports = source.split("class BodyStateProcessor", 1)[0]
    assert "motion_switcher" not in module_imports


def test_lowcmd_write_is_behind_explicit_gate() -> None:
    source = (CONTROL / "envs/g1/utils/command_sender.py").read_text()
    write = source.split("def write_prepared", 1)[1].split("def send_command", 1)[0]
    gate_position = write.index("if not self._writes_armed")
    write_position = write.index("self.lowcmd_publisher_.Write")
    assert gate_position < write_position


def test_real_g1_rejects_dex3_sender_before_body_construction() -> None:
    source = (CONTROL / "envs/g1/g1_env.py").read_text()
    constructor = source.split("def __init__", 1)[1].split("def start_simulator", 1)[0]
    reject_position = constructor.index('config.get("ENV_TYPE") == "real" and self.with_hands')
    body_position = constructor.index("self._body = G1Body")
    assert reject_position < body_position
    assert "Dex3 sender/calibration must not be constructed" in constructor

    sender_source = (CONTROL / "envs/g1/utils/command_sender.py").read_text()
    hand_constructor = sender_source.split("class HandCommandSender", 1)[1].split(
        "def send_command", 1
    )[0]
    environment_gate = hand_constructor.index('if env_type != "sim"')
    publisher_creation = hand_constructor.index("ChannelPublisher(")
    assert environment_gate < publisher_creation


def test_wbc_factory_receives_monotonic_init_time() -> None:
    source = (CONTROL / "main/teleop/run_g1_control_loop.py").read_text()
    assert 'init_time=time.monotonic()' in source
    assert 'wbc_config, config.upper_body_joint_speed)' not in source
    main = source.split("def main", 1)[1]
    real_guard = main.index('if config.env_type == "real"')
    env_construction = main.index("env = G1Env")
    assert real_guard < env_construction


def test_read_only_entry_has_no_command_transport() -> None:
    source = (CONTROL / "main/teleop/run_g1_standalone_read_only.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("command_sender" in name or "motion_switcher" in name for name in imported_modules)
    assert ".Write(" not in source
    assert ".ReleaseMode(" not in source
    assert '"lowcmd_write_attempts": 0' in source

    loop = source.split("while time.monotonic() < deadline:", 1)[1].split(
        "def observed_rate", 1
    )[0]
    read_position = loop.index("sample = processor.read_real_safe_snapshot()")
    validation_time_position = loop.index("now = time.monotonic()")
    validate_position = loop.index("core.read_only_tick(sample, now)")
    assert read_position < validation_time_position < validate_position
