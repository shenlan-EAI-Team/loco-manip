from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from decoupled_wbc.control.real_safe import (
    RobotSnapshot,
    SafetyFault,
    StandaloneSafetyLimits,
)
from decoupled_wbc.control.real_safe.lowcmd_guard import (
    GuardConfig,
    GuardSnapshot,
    GuardState,
    LowCmdGuardCore,
    PcTargetMailbox,
)
from decoupled_wbc.control.real_safe.standalone import StandaloneSafetyGate


CONFIG_DIR = Path(__file__).resolve().parents[3] / "control/main/teleop/configs"


def guard_config(*, enabled: bool = False, recovery_verified: bool = False) -> GuardConfig:
    config = GuardConfig.from_mapping(
        yaml.safe_load((CONFIG_DIR / "g1_lowcmd_guard.yaml").read_text())
    )
    return replace(
        config,
        real_execution_enabled=enabled,
        recovery_handoff_verified=recovery_verified,
    )


def safety_gate() -> StandaloneSafetyGate:
    limits = StandaloneSafetyLimits.from_mapping(
        yaml.safe_load((CONFIG_DIR / "g1_standalone_real_safe.yaml").read_text())
    )
    return StandaloneSafetyGate(limits)


def snapshot(now: float) -> GuardSnapshot:
    q = np.zeros(29)
    q[3] = 0.3
    q[4] = -0.2
    q[9] = 0.3
    q[10] = -0.2
    return GuardSnapshot(
        robot=RobotSnapshot(
            q=q,
            dq=np.zeros(29),
            base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            base_angular_velocity=np.zeros(3),
            secondary_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            secondary_angular_velocity=np.zeros(3),
            lowstate_monotonic=now,
            imu_monotonic=now,
        ),
        mode_machine=5,
        motor_modes=np.ones(29, dtype=np.int64),
        motor_errors=np.zeros(29, dtype=np.int64),
        motor_tau_est=np.zeros(29, dtype=np.float64),
    )


def ready_core(config: GuardConfig, now: float = 1.0) -> LowCmdGuardCore:
    core = LowCmdGuardCore(config, safety_gate(), one_time_token="one-shot")
    core.observe(snapshot(now), "ai", now)
    command = core.prepare_current_q_hold(now)
    np.testing.assert_array_equal(command.q, snapshot(now).robot.q)
    np.testing.assert_array_equal(command.dq, 0.0)
    np.testing.assert_array_equal(command.tau, 0.0)
    assert command.mode_machine == 5
    assert core.state == GuardState.READY
    return core


def test_default_configuration_is_hard_blocked_before_takeover() -> None:
    config = guard_config()
    assert config.transport_frequency_hz == 500.0
    assert config.official_reference_transport_frequency_hz == 500.0
    assert config.measured_minimum_transport_frequency_hz is None
    assert config.real_execution_enabled is False
    assert config.recovery_handoff_verified is False
    core = ready_core(config)
    with pytest.raises(PermissionError, match="real execution is disabled"):
        core.authorize_takeover(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
            now=1.001,
        )
    assert core.state == GuardState.READY


def test_recovery_semantics_and_all_runtime_gates_are_required() -> None:
    core = ready_core(guard_config(enabled=True, recovery_verified=False))
    with pytest.raises(PermissionError, match="recovery handoff"):
        core.authorize_takeover(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
            now=1.001,
        )

    core = ready_core(guard_config(enabled=True, recovery_verified=True))
    with pytest.raises(PermissionError, match="all three"):
        core.authorize_takeover(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=False,
            now=1.001,
        )
    core.authorize_takeover(
        token="one-shot",
        hardware_transport_enabled=True,
        command_publication_enabled=True,
        lifecycle_armed=True,
        now=1.001,
    )
    assert core.state == GuardState.TAKEOVER_PENDING


def test_current_q_command_must_be_fresh_and_first_write_has_deadline() -> None:
    core = ready_core(guard_config(enabled=True, recovery_verified=True))
    with pytest.raises(SafetyFault, match="prepared current-q command is stale"):
        core.authorize_takeover(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
            now=1.1,
        )

    core = ready_core(guard_config(enabled=True, recovery_verified=True))
    core.authorize_takeover(
        token="one-shot",
        hardware_transport_enabled=True,
        command_publication_enabled=True,
        lifecycle_armed=True,
        now=1.001,
    )
    core.mark_release_returned(1.002)
    with pytest.raises(SafetyFault, match="first-HOLD"):
        core.mark_first_write(1.008)
    assert core.state == GuardState.FAULT_BLOCKED


def test_current_q_is_refreshed_after_silent_transport_discovery() -> None:
    core = ready_core(guard_config(enabled=True, recovery_verified=True))
    core.authorize_takeover(
        token="one-shot",
        hardware_transport_enabled=True,
        command_publication_enabled=True,
        lifecycle_armed=True,
        now=1.001,
    )
    fresh = snapshot(2.0)
    fresh.robot.q[15] = 0.2
    command = core.refresh_current_q_hold_after_transport(fresh, "ai", 2.0)
    assert command.prepared_monotonic == 2.0
    assert command.q[15] == 0.2
    with pytest.raises(SafetyFault, match="owner changed"):
        core.refresh_current_q_hold_after_transport(snapshot(2.1), "", 2.1)


def test_motor_errors_and_wrong_owner_block_preparation() -> None:
    core = LowCmdGuardCore(guard_config(), safety_gate(), one_time_token="one-shot")
    bad = snapshot(1.0)
    bad.motor_errors[4] = 7
    with pytest.raises(SafetyFault, match="motor error"):
        core.observe(bad, "ai", 1.0)

    core.observe(snapshot(2.0), "normal", 2.0)
    with pytest.raises(SafetyFault, match="expected owner"):
        core.prepare_current_q_hold(2.0)


def test_pc_target_mailbox_enforces_session_sequence_rate_and_heartbeat() -> None:
    config = guard_config()
    mailbox = PcTargetMailbox(config, safety_gate(), session_id="session")
    initial = snapshot(1.0).robot.q
    mailbox.accept(
        {
            "schema_version": 1,
            "kind": "validated_target",
            "session_id": "session",
            "sequence": 1,
            "q_rad": initial.tolist(),
        },
        1.0,
    )
    np.testing.assert_allclose(mailbox.latest(1.05), initial)
    with pytest.raises(SafetyFault, match="stale"):
        mailbox.latest(1.2)

    jumped = initial.copy()
    jumped[15] += 0.02
    with pytest.raises(SafetyFault, match="step/rate"):
        mailbox.accept(
            {
                "schema_version": 1,
                "kind": "validated_target",
                "session_id": "session",
                "sequence": 2,
                "q_rad": jumped.tolist(),
            },
            1.02,
        )
