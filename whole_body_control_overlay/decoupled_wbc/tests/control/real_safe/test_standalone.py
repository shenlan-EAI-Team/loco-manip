from pathlib import Path

import numpy as np
import pytest
import yaml

from decoupled_wbc.control.real_safe import (
    RobotSnapshot,
    SafetyFault,
    StandaloneRealSafeCore,
    StandaloneSafetyLimits,
    StandaloneState,
)


CONFIG = (
    Path(__file__).resolve().parents[3]
    / "control/main/teleop/configs/g1_standalone_real_safe.yaml"
)


def limits() -> StandaloneSafetyLimits:
    return StandaloneSafetyLimits.from_mapping(yaml.safe_load(CONFIG.read_text()))


def snapshot(now: float, *, q: np.ndarray | None = None, dq: np.ndarray | None = None):
    q_value = np.zeros(29) if q is None else np.asarray(q, dtype=np.float64).copy()
    q_value[3] = 0.3
    q_value[4] = -0.2
    q_value[9] = 0.3
    q_value[10] = -0.2
    return RobotSnapshot(
        q=q_value,
        dq=np.zeros(29) if dq is None else np.asarray(dq, dtype=np.float64),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        secondary_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        secondary_angular_velocity=np.zeros(3),
        lowstate_monotonic=now,
        imu_monotonic=now,
    )


def armed_core(now: float = 10.0):
    core = StandaloneRealSafeCore(limits(), one_time_arm_token="one-shot")
    current = snapshot(now)
    core.read_only_tick(current, now)
    preview = core.request_arm("one-shot", current, now)
    np.testing.assert_allclose(preview, current.q)
    assert core.state == StandaloneState.ARM_CONTROL
    core.mark_takeover_complete(now + 0.001)
    return core, current


def test_read_only_never_allows_commands_and_token_is_one_time() -> None:
    now = 10.0
    core = StandaloneRealSafeCore(limits(), one_time_arm_token="one-shot")
    core.read_only_tick(snapshot(now), now)
    assert core.state == StandaloneState.READ_ONLY
    assert core.command_allowed is False

    with pytest.raises(PermissionError):
        core.request_arm("wrong", snapshot(now), now)
    core.request_arm("one-shot", snapshot(now), now)
    with pytest.raises(RuntimeError):
        core.request_arm("one-shot", snapshot(now), now)


def test_stale_or_nonfinite_read_only_input_is_rejected() -> None:
    core = StandaloneRealSafeCore(limits(), one_time_arm_token="one-shot")
    stale = snapshot(10.0)
    with pytest.raises(SafetyFault, match="lowstate is stale"):
        core.read_only_tick(stale, 10.2)

    bad_q = snapshot(20.0, q=np.full(29, np.nan))
    with pytest.raises(SafetyFault, match="non-finite"):
        core.read_only_tick(bad_q, 20.0)


def test_safety_limits_match_29dof_hard_limits_and_are_conservative() -> None:
    safe_config = yaml.safe_load(CONFIG.read_text())
    wbc_config = yaml.safe_load(
        (CONFIG.parent / "g1_29dof_gear_wbc.yaml").read_text()
    )
    np.testing.assert_allclose(
        safe_config["q_lower"], wbc_config["motor_pos_lower_limit_list"], atol=1e-6
    )
    np.testing.assert_allclose(
        safe_config["q_upper"], wbc_config["motor_pos_upper_limit_list"], atol=1e-6
    )
    assert np.all(
        np.asarray(safe_config["measured_dq_abs_limit"])
        <= np.asarray(wbc_config["motor_vel_limit_list"])
    )
    assert np.all(
        np.asarray(safe_config["lower_target_rate_abs_limit"]) / 50.0
        <= np.asarray(safe_config["lower_target_step_abs_limit"]) + 1e-12
    )


def test_hold_uses_arming_feedback_and_detects_drift() -> None:
    core, current = armed_core()
    command = core.hold_command(current, 10.02)
    np.testing.assert_allclose(command, current.q)

    drifted_q = current.q.copy()
    drifted_q[15] += 0.011
    with pytest.raises(SafetyFault, match="HOLD feedback drift"):
        core.hold_command(snapshot(10.04, q=drifted_q), 10.04)
    assert core.state == StandaloneState.FAULT
    assert core.command_allowed is False


def test_engage_is_smooth_rate_limited_and_holds_both_arms() -> None:
    core, current = armed_core()
    for index in range(1, 102):
        now = 10.001 + index * 0.02
        core.hold_command(snapshot(now, q=current.q.copy()), now)
    core.begin_wbc_engage(12.021)

    target = current.q[:15].copy()
    target[0] += 0.2
    previous = current.q.copy()
    for index in range(1, 152):
        now = 12.021 + index * 0.02
        command = core.wbc_command(snapshot(now, q=previous), target, now)
        assert np.max(np.abs(command[:15] - previous[:15])) <= 0.0100001
        np.testing.assert_allclose(command[15:29], current.q[15:29])
        previous = command

    assert core.state == StandaloneState.STAND
    np.testing.assert_allclose(previous[:15], target, atol=1e-9)


def test_engage_requires_continuously_serviced_hold() -> None:
    core, _ = armed_core()
    with pytest.raises(SafetyFault, match="continuously serviced"):
        core.begin_wbc_engage(12.01)
    assert core.state == StandaloneState.FAULT


def test_active_control_rejects_nonmonotonic_clock() -> None:
    core, current = armed_core()
    with pytest.raises(SafetyFault, match="strictly monotonic"):
        core.hold_command(snapshot(10.0, q=current.q), 10.0)
    assert core.state == StandaloneState.FAULT


def test_50hz_deadline_miss_faults_without_producing_a_command() -> None:
    core, current = armed_core()
    with pytest.raises(SafetyFault, match="50Hz control watchdog missed"):
        core.hold_command(snapshot(10.10, q=current.q), 10.10)
    assert core.state == StandaloneState.FAULT
    assert core.command_allowed is False


def test_independent_watchdog_and_exit_remain_fail_closed() -> None:
    core, _ = armed_core()
    assert core.watchdog_expired(10.02) is False
    assert core.watchdog_expired(10.10) is True
    assert core.state == StandaloneState.FAULT
    with pytest.raises(PermissionError, match="platform-specific"):
        core.mark_exit_complete(verified_platform_exit=False)
    core.mark_exit_complete(verified_platform_exit=True)
    assert core.state == StandaloneState.STOPPED
