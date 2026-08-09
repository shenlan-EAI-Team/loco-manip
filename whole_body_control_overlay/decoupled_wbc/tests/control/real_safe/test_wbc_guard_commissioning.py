from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from decoupled_wbc.control.real_safe.gear_wbc_producer import (
    GearWbcModelConfig,
    GearWbcStandingModel,
    verify_artifact_manifest,
)
from decoupled_wbc.control.real_safe.lowcmd_guard import (
    CommissioningPhase,
    LowerBodyMailbox,
    WbcGuardCommandComposer,
)
from decoupled_wbc.control.real_safe.standalone import SafetyFault

from .test_lowcmd_guard_core import guard_config, ready_core, safety_gate, snapshot


ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "decoupled_wbc/control/main/teleop/configs"


def composer(mailbox: LowerBodyMailbox, now: float = 0.0):
    safety = safety_gate()
    result = WbcGuardCommandComposer(
        mailbox,
        safety,
        mailbox_stale_s=0.1,
        engage_duration_s=3.0,
        lower_rate_limit=safety.limits.lower_target_rate_abs_limit,
        lower_step_limit=safety.limits.lower_target_step_abs_limit,
    )
    core = ready_core(guard_config(), now=now)
    result.arm_current_q(core.prepared_command, now=now)
    return result, core.prepared_command.q.copy()


def test_mailbox_rejects_stale_invalid_and_nonincreasing_sequence() -> None:
    mailbox = LowerBodyMailbox()
    mailbox.publish(np.zeros(15), timestamp=1.0, sequence=1)
    with pytest.raises(SafetyFault, match="sequence"):
        mailbox.publish(np.zeros(15), timestamp=1.02, sequence=1)
    with pytest.raises(SafetyFault, match="timestamp"):
        mailbox.publish(np.zeros(15), timestamp=1.0, sequence=2)
    with pytest.raises(SafetyFault, match="finite shape"):
        mailbox.publish(np.full(15, np.nan), timestamp=1.02, sequence=2)
    with pytest.raises(SafetyFault, match="stale"):
        mailbox.latest(now=1.2, max_age_s=0.1)


def test_50hz_mailbox_drives_500hz_smoothstep_and_freezes_both_arms() -> None:
    mailbox = LowerBodyMailbox()
    command_composer, arming = composer(mailbox)
    lower_target = arming[:15].copy() + 0.03
    mailbox.publish(lower_target, timestamp=0.0, sequence=1)
    command_composer.begin_engage(now=0.0)

    commands = []
    sequence = 1
    for tick in range(1, 1501):
        now = tick * 0.002
        if tick % 10 == 0:
            sequence += 1
            mailbox.publish(lower_target, timestamp=now, sequence=sequence)
        commands.append(command_composer.command(now=now))

    assert len(commands) == 1500
    assert sequence == 151
    np.testing.assert_allclose(commands[749].q[:15], arming[:15] + 0.015, atol=2e-4)
    np.testing.assert_allclose(commands[-1].q[:15], lower_target, atol=2e-4)
    for command in commands:
        np.testing.assert_array_equal(command.q[15:29], arming[15:29])
    assert command_composer.phase == CommissioningPhase.STAND


def test_stale_mailbox_freezes_last_valid_command_without_changing_arms() -> None:
    mailbox = LowerBodyMailbox()
    command_composer, arming = composer(mailbox)
    mailbox.publish(arming[:15] + 0.02, timestamp=0.0, sequence=1)
    command_composer.begin_engage(now=0.0)
    last = command_composer.command(now=0.02)
    faults = []
    frozen = command_composer.command_or_freeze(now=0.2, on_fault=faults.append)
    np.testing.assert_array_equal(frozen.q, last.q)
    np.testing.assert_array_equal(frozen.q[15:29], arming[15:29])
    assert command_composer.phase == CommissioningPhase.FROZEN
    assert len(faults) == 1


def test_read_only_producer_has_exact_516_to_15_contract_and_no_command_imports() -> None:
    values = yaml.safe_load(
        (ROOT / "decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml").read_text()
    )
    config = GearWbcModelConfig(
        default_angles=np.asarray(values["default_angles"], dtype=np.float32),
        cmd=np.asarray(values["cmd_init"], dtype=np.float32),
        height=float(values["height_cmd"]),
        rpy=np.asarray(values["rpy_cmd"], dtype=np.float32),
        cmd_scale=np.asarray(values["cmd_scale"], dtype=np.float32),
        angular_velocity_scale=float(values["ang_vel_scale"]),
        position_scale=float(values["dof_pos_scale"]),
        velocity_scale=float(values["dof_vel_scale"]),
        action_scale=float(values["action_scale"]),
        history_length=6,
        observation_size=516,
    )
    observed = []

    def infer(model_input):
        observed.append(model_input.copy())
        return np.zeros((1, 15), dtype=np.float32)

    model = GearWbcStandingModel(config, infer)
    target = model.target(snapshot(1.0))
    assert observed[0].shape == (1, 516)
    np.testing.assert_allclose(target, config.default_angles)

    source = (ROOT / "decoupled_wbc/control/real_safe/gear_wbc_producer.py").read_text()
    for forbidden in ("ChannelPublisher", "BodyCommandSender", '"rt/lowcmd"', ".Write("):
        assert forbidden not in source


def test_commissioning_gate_does_not_require_production_recovery_flag() -> None:
    config = replace(
        guard_config(enabled=True, recovery_verified=False),
        commissioning_execution_enabled=True,
    )
    core = ready_core(config)
    core.authorize_takeover(
        token="one-shot",
        hardware_transport_enabled=True,
        command_publication_enabled=True,
        lifecycle_armed=True,
        now=1.001,
        commissioning_mode=True,
    )


def test_audited_guard_gains_are_nonnegative_and_legacy_negative_override_is_gone() -> None:
    values = yaml.safe_load((CONFIG_DIR / "g1_lowcmd_guard.yaml").read_text())
    assert np.all(np.asarray(values["kp"]) >= 0)
    assert np.all(np.asarray(values["kd"]) >= 0)
    config_source = (CONFIG_DIR / "configs.py").read_text()
    assert 'MOTOR_KD"][14]' not in config_source


def test_commissioning_artifacts_and_identity_motor_mapping_are_locked() -> None:
    locked = verify_artifact_manifest(
        CONFIG_DIR / "g1_wbc_commissioning_manifest.json",
        repository_root=ROOT,
    )
    assert len(locked) == 13
    mapping = yaml.safe_load((CONFIG_DIR / "g1_29dof_gear_wbc.yaml").read_text())
    assert mapping["MOTOR2JOINT"] == list(range(29))
    assert mapping["JOINT2MOTOR"] == list(range(29))
    weak = mapping["WeakMotorJointIndex"]
    assert weak["left_hip_pitch_joint"] == 0
    assert weak["left_hip_roll_joint"] == 1
    assert weak["left_hip_yaw_joint"] == 2
    assert weak["left_ankle_pitch_joint"] == 4
    assert weak["right_hip_pitch_joint"] == 6
    assert weak["right_hip_roll_joint"] == 7
    assert weak["right_hip_yaw_joint"] == 8
    assert weak["right_ankle_pitch_joint"] == 10

    default_guard = yaml.safe_load((CONFIG_DIR / "g1_lowcmd_guard.yaml").read_text())
    commissioning_guard = yaml.safe_load(
        (CONFIG_DIR / "g1_wbc_commissioning_guard.yaml").read_text()
    )
    assert default_guard["real_execution_enabled"] is False
    assert default_guard["commissioning_execution_enabled"] is False
    assert commissioning_guard["real_execution_enabled"] is True
    assert commissioning_guard["commissioning_execution_enabled"] is True
    for key in ("real_execution_enabled", "commissioning_execution_enabled"):
        default_guard.pop(key)
        commissioning_guard.pop(key)
    assert default_guard == commissioning_guard
