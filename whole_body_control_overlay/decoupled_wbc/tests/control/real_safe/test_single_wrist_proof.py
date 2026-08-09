from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from decoupled_wbc.control.real_safe.lowcmd_guard import LowerBodyMailbox
from decoupled_wbc.control.real_safe.lowcmd_guard.single_wrist_proof import (
    PROOF_MOTOR_INDEX,
    SingleWristProofComposer,
    SingleWristProofConfig,
    SingleWristProofRuntime,
    WristProofPhase,
)
from decoupled_wbc.control.real_safe.standalone import SafetyFault

from .test_lowcmd_guard_core import guard_config, ready_core, safety_gate, snapshot


ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "decoupled_wbc/control/main/teleop/configs"


def proof_config(*, preview_q: float = 0.0) -> SingleWristProofConfig:
    loaded = SingleWristProofConfig.from_json(CONFIG_DIR / "g1_single_wrist_proof.json")
    return replace(loaded, preview_current_q_rad=preview_q)


def test_proof_config_is_hard_locked_to_left_wrist_yaw_plus_point_zero_one() -> None:
    proof = SingleWristProofConfig.from_json(CONFIG_DIR / "g1_single_wrist_proof.json")
    assert proof.joint_name == "left_wrist_yaw_joint"
    assert proof.motor_cmd_index == 21
    assert proof.motor_state_index == 21
    assert proof.offset_rad == 0.01
    assert proof.ramp_duration_s == 0.5
    assert proof.hold_duration_s == 1.0

    limits = yaml.safe_load((CONFIG_DIR / "g1_standalone_real_safe.yaml").read_text())
    guard = yaml.safe_load((CONFIG_DIR / "g1_wbc_commissioning_guard.yaml").read_text())
    target = proof.preview_current_q_rad + proof.offset_rad
    assert limits["q_lower"][21] < target < limits["q_upper"][21]
    assert guard["kp"][21] == 20
    assert guard["kd"][21] == 2
    assert guard["target_rate_abs_limit"][21] == 0.12
    assert guard["target_step_abs_limit"][21] == 0.0024


def test_wbc_stays_on_lower_body_and_only_left_wrist_yaw_moves() -> None:
    safety = safety_gate()
    guard = guard_config(enabled=True, recovery_verified=False)
    core = ready_core(guard, now=0.0)
    arming = core.prepared_command.q.copy()
    mailbox = LowerBodyMailbox()
    composer = SingleWristProofComposer(
        mailbox,
        safety,
        mailbox_stale_s=0.1,
        engage_duration_s=3.0,
        lower_rate_limit=safety.limits.lower_target_rate_abs_limit,
        lower_step_limit=safety.limits.lower_target_step_abs_limit,
        proof=proof_config(preview_q=arming[PROOF_MOTOR_INDEX]),
        arm_rate_limit=guard.target_rate_abs_limit[PROOF_MOTOR_INDEX],
        arm_step_limit=guard.target_step_abs_limit[PROOF_MOTOR_INDEX],
    )
    composer.arm_current_q(core.prepared_command, now=0.0)

    lower_target = arming[:15].copy() + 0.02
    mailbox.publish(lower_target, timestamp=0.0, sequence=1)
    composer.begin_engage(now=0.0)
    sequence = 1
    for tick in range(1, 1501):
        now = tick * 0.002
        if tick % 10 == 0:
            sequence += 1
            mailbox.publish(lower_target, timestamp=now, sequence=sequence)
        command = composer.command(now=now)
        np.testing.assert_array_equal(command.q[15:29], arming[15:29])
    np.testing.assert_allclose(command.q[:15], lower_target, atol=2e-4)

    composer.begin_proof(now=3.0)
    wrist_commands = []
    for tick in range(1, 761):
        now = 3.0 + tick * 0.002
        if tick % 10 == 0:
            sequence += 1
            mailbox.publish(lower_target, timestamp=now, sequence=sequence)
        command = composer.command(now=now)
        wrist_commands.append(float(command.q[PROOF_MOTOR_INDEX]))
        other_arm_indices = [index for index in range(15, 29) if index != PROOF_MOTOR_INDEX]
        np.testing.assert_array_equal(command.q[other_arm_indices], arming[other_arm_indices])

    wrist_delta = np.asarray(wrist_commands) - arming[PROOF_MOTOR_INDEX]
    assert np.min(wrist_delta) >= -1e-12
    assert np.max(wrist_delta) <= 0.01 + 1e-12
    assert np.all(np.diff(wrist_delta) >= -1e-12)
    np.testing.assert_allclose(wrist_delta[-1], 0.01, atol=1e-8)
    assert np.max(np.diff(wrist_commands)) <= 0.0024 + 1e-12
    assert np.max(np.diff(wrist_commands)) / 0.002 <= 0.12 + 1e-12
    assert composer.proof_phase == WristProofPhase.COMPLETE


def test_proof_runner_has_no_handback_or_other_control_stack() -> None:
    runtime_source = (
        ROOT
        / "decoupled_wbc/control/real_safe/lowcmd_guard/single_wrist_proof.py"
    ).read_text()
    runner_source = (
        ROOT
        / "decoupled_wbc/control/main/teleop/run_g1_single_wrist_proof.py"
    ).read_text()
    combined = runtime_source + runner_source
    assert ".select_mode(" not in combined
    assert "_recover_to_original_owner(" not in combined
    for forbidden in (
        "run_live_shadow",
        "groot_o6",
        "o6_feedback",
        "o6_command",
        "rt/arm_sdk",
        "ENABLE_SONIC",
        "g1_deploy_onnx_ref",
    ):
        assert forbidden not in combined


@pytest.mark.parametrize(
    ("feedback_delta", "message"),
    [
        (-0.0021, "opposite"),
        (0.0151, "envelope"),
    ],
)
def test_wrist_feedback_direction_and_envelope_faults(
    feedback_delta: float,
    message: str,
) -> None:
    safety = safety_gate()
    guard = guard_config(enabled=True, recovery_verified=False)
    core = ready_core(guard, now=1.0)
    mailbox = LowerBodyMailbox()
    composer = SingleWristProofComposer(
        mailbox,
        safety,
        mailbox_stale_s=0.1,
        engage_duration_s=3.0,
        lower_rate_limit=safety.limits.lower_target_rate_abs_limit,
        lower_step_limit=safety.limits.lower_target_step_abs_limit,
        proof=proof_config(preview_q=core.prepared_command.q[PROOF_MOTOR_INDEX]),
        arm_rate_limit=guard.target_rate_abs_limit[PROOF_MOTOR_INDEX],
        arm_step_limit=guard.target_step_abs_limit[PROOF_MOTOR_INDEX],
    )
    composer.arm_current_q(core.prepared_command, now=1.0)
    observed = snapshot(1.0)
    observed.robot.q[PROOF_MOTOR_INDEX] += feedback_delta

    class Source:
        @staticmethod
        def latest(_now):
            return observed

    runtime = SingleWristProofRuntime(
        core,
        Source(),
        object(),
        writer_factory=lambda: None,
        proof=proof_config(preview_q=core.prepared_command.q[PROOF_MOTOR_INDEX]),
        clock=lambda: 1.0,
    )
    with pytest.raises(SafetyFault, match=message):
        runtime._validate_live_proof(composer)


def test_small_positive_wrist_feedback_is_recorded_as_consistent() -> None:
    safety = safety_gate()
    guard = guard_config(enabled=True, recovery_verified=False)
    core = ready_core(guard, now=1.0)
    mailbox = LowerBodyMailbox()
    proof = proof_config(preview_q=core.prepared_command.q[PROOF_MOTOR_INDEX])
    composer = SingleWristProofComposer(
        mailbox,
        safety,
        mailbox_stale_s=0.1,
        engage_duration_s=3.0,
        lower_rate_limit=safety.limits.lower_target_rate_abs_limit,
        lower_step_limit=safety.limits.lower_target_step_abs_limit,
        proof=proof,
        arm_rate_limit=guard.target_rate_abs_limit[PROOF_MOTOR_INDEX],
        arm_step_limit=guard.target_step_abs_limit[PROOF_MOTOR_INDEX],
    )
    composer.arm_current_q(core.prepared_command, now=1.0)
    observed = snapshot(1.0)
    observed.robot.q[PROOF_MOTOR_INDEX] += 0.002

    class Source:
        @staticmethod
        def latest(_now):
            return observed

    runtime = SingleWristProofRuntime(
        core,
        Source(),
        object(),
        writer_factory=lambda: None,
        proof=proof,
        clock=lambda: 1.0,
    )
    runtime._validate_live_proof(composer)
    assert runtime.max_positive_response_rad == pytest.approx(0.002)
    assert runtime.proof_trace[-1]["direction_consistent"] is True


def test_proof_manifest_locks_only_additions_on_top_of_base_manifest() -> None:
    values = json.loads((CONFIG_DIR / "g1_single_wrist_proof_manifest.json").read_text())
    assert values["schema_version"] == 1
    assert values["base_manifest"] == "g1_wbc_commissioning_manifest.json"
    assert len(values["sha256"]) == 4
