from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from deployment.real_bridge import cli
from deployment.real_bridge.controller import RealBridgeSession
from deployment.real_bridge.gates import GateSettings, OneTimeToken
from deployment.real_bridge.logging import JsonlBridgeLogger
from deployment.real_bridge.message_preview import preview_g1_arm_message, preview_o6_messages
from deployment.real_bridge.models import BridgeState, FeedbackSnapshot
from deployment.real_bridge.preview import build_arms_only_micro_plan, build_hold_only_plan
from deployment.real_bridge.transports import MockG1Transport, MockO6Transport


def test_every_gate_defaults_false() -> None:
    gates = GateSettings()
    assert gates.all_enabled is False
    with pytest.raises(PermissionError):
        gates.require_all()


def test_missing_gate_blocks_before_token_or_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"frames": []}), encoding="utf-8")
    token_file = tmp_path / "token.json"
    token = OneTimeToken.issue(
        token_file,
        bound_sha256=cli.sha256_file(plan),
    )
    called = False

    def forbidden_factory(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal called
        called = True
        raise AssertionError("real transport factory must not run")

    monkeypatch.setattr(cli, "_create_real_session", forbidden_factory)
    with pytest.raises(PermissionError):
        cli.main(
            [
                "execute",
                "--plan",
                str(plan),
                "--log",
                str(tmp_path / "run.jsonl"),
                "--token-file",
                str(token_file),
                "--confirmation-token",
                token,
                "--phase",
                "hold-only",
                "--hardware-transport-enabled",
                "--command-publication-enabled",
                # micro_motion_armed deliberately absent
            ]
        )
    assert token_file.exists(), "gate failure must not consume the one-time token"
    assert called is False


def test_obsolete_plan_is_rejected() -> None:
    with pytest.raises(PermissionError, match="obsolete"):
        cli.reject_obsolete_plan(
            "c909fc15b7ce3c7699b18b9e066c271cf7d403696c8a7022412628279f05e814"
        )


def test_arm_message_preview_never_populates_waist_or_legs() -> None:
    preview = preview_g1_arm_message(np.arange(7), -np.arange(7), weight=1.0)
    assert preview["dds_topic"] == "rt/arm_sdk"
    assert len(preview["motor_cmd_serialized"]) == 35
    for index in (*range(15), *range(30, 35)):
        assert preview["motor_cmd_serialized"][index] == {
            "mode": 0,
            "q": 0.0,
            "dq": 0.0,
            "tau": 0.0,
            "kp": 0.0,
            "kd": 0.0,
            "reserve": 0,
        }
    assert isinstance(preview["crc"], int) and preview["crc"] != 0


def test_o6_preview_has_no_right_hand_command() -> None:
    preview = preview_o6_messages(np.full(6, 50.0), np.full(6, 100.0))
    assert preview["left"]["data"] == [1, 128, 128, 128, 128, 128, 128]
    assert preview["right"]["feedback_only"] is True
    assert preview["right"]["command"] is None
    assert preview["right"]["command_count"] == 0


def test_arm_message_preview_exact_arm_weight_and_top_level_fields() -> None:
    left = np.linspace(-0.03, 0.03, 7)
    right = np.linspace(0.04, -0.02, 7)
    preview = preview_g1_arm_message(left, right, weight=0.42)
    assert preview["mode_pr"] == 0
    assert preview["mode_machine"] == 0
    assert preview["low_cmd_reserve"] == [0, 0, 0, 0]
    for index, expected in zip(range(15, 22), left):
        assert preview["motor_cmd_serialized"][index] == {
            "mode": 0,
            "q": float(expected),
            "dq": 0.0,
            "tau": 0.0,
            "kp": 60.0,
            "kd": 1.5,
            "reserve": 0,
        }
    for index, expected in zip(range(22, 29), right):
        assert preview["motor_cmd_serialized"][index] == {
            "mode": 0,
            "q": float(expected),
            "dq": 0.0,
            "tau": 0.0,
            "kp": 60.0,
            "kd": 1.5,
            "reserve": 0,
        }
    assert preview["motor_cmd_serialized"][29] == {
        "mode": 0,
        "q": 0.42,
        "dq": 0.0,
        "tau": 0.0,
        "kp": 0.0,
        "kd": 0.0,
        "reserve": 0,
    }


def test_arm_message_preview_can_match_live_mode_fields() -> None:
    preview = preview_g1_arm_message(
        np.zeros(7), np.zeros(7), weight=0.0, mode_machine=5, mode_pr=0
    )
    assert preview["mode_machine"] == 5
    assert preview["mode_pr"] == 0


def test_mock_one_shot_runs_hold_micro_and_release_without_real_sdk(
    tmp_path: Path,
) -> None:
    groups = {
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
        "left_o6": np.full(6, 99.0),
        "right_o6": np.full(6, 99.0),
    }
    snapshot = FeedbackSnapshot.create(groups, g1_mode_machine=5, g1_mode_pr=0, waist=np.zeros(3))
    g1 = MockG1Transport(snapshot)
    o6 = MockO6Transport(
        {"left_o6": groups["left_o6"], "right_o6": groups["right_o6"]}
    )
    frames = []
    for index in range(15):
        inference = index // 3
        target = {key: value.tolist() for key, value in groups.items()}
        raw = {key: value.tolist() for key, value in groups.items()}
        raw["right_o6"] = [0.0] * 6
        frames.append(
            {
                "inference_index": inference,
                "policy_raw_absolute": raw,
                "adapter_absolute": raw,
                "ordinary_safety": target,
            }
        )
    with JsonlBridgeLogger(tmp_path / "mock.jsonl") as logger:
        session = RealBridgeSession(
            g1,
            o6,
            logger,
            activation_ramp_s=0.02,
            release_ramp_s=0.02,
            post_release_monitor_s=0.02,
            o6_position_commands_enabled=True,
        )
        session.arm_hold()
        session.execute_hold(0.02)
        assert len(o6.records) == 0
        session.execute_micro({"frames": frames})
        session.stop("mock complete")
    assert session.state is BridgeState.STOPPED
    assert len(g1.records) > 0
    assert len(o6.records) > 0
    assert g1.records[-1]["weight"] == 0.0


def test_exact_weight_curve_release_monitor_and_joint_response_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = 0.0

    def monotonic() -> float:
        return clock

    def sleep(duration: float) -> None:
        nonlocal clock
        clock += max(duration, 0.0001)

    monkeypatch.setattr("deployment.real_bridge.controller.time.monotonic", monotonic)
    monkeypatch.setattr("deployment.real_bridge.controller.time.sleep", sleep)
    groups = {
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
        "left_o6": np.full(6, 50.0),
        "right_o6": np.full(6, 50.0),
    }
    snapshot = FeedbackSnapshot.create(groups, g1_mode_machine=5, g1_mode_pr=0, waist=np.zeros(3))
    g1 = MockG1Transport(snapshot)
    o6 = MockO6Transport({"left_o6": groups["left_o6"], "right_o6": groups["right_o6"]})
    log_path = tmp_path / "weight.jsonl"
    with JsonlBridgeLogger(log_path) as logger:
        session = RealBridgeSession(g1, o6, logger)
        session.arm_hold()
        session.execute_hold(2.0)
        hold_weights = [record["weight"] for record in g1.records]
        session.stop("weight curve proof")

    assert len(hold_weights) == 151
    np.testing.assert_allclose(hold_weights[:51], np.arange(51) / 50.0, atol=1e-12)
    np.testing.assert_allclose(hold_weights[51:], 1.0, atol=1e-12)
    release_weights = [record["weight"] for record in g1.records[151:]]
    assert len(release_weights) == 100
    np.testing.assert_allclose(release_weights, 1.0 - np.arange(1, 101) / 100.0, atol=1e-12)

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    post_release = [
        row
        for row in rows
        if row.get("event") == "release_feedback"
        and row.get("phase") == "weight_zero_post_release_monitor"
    ]
    assert len(post_release) == 25
    assert all(row["weight"] == 0.0 and row["command_publication_active"] is False for row in post_release)
    response = next(row for row in rows if row.get("event") == "joint_response")
    for group in response["groups"].values():
        assert "max_response_matches_max_command" in group
        for joint in group["joints"]:
            assert {
                "motor_cmd_index",
                "motor_state_index",
                "initial_q",
                "raw_policy_target",
                "final_command_q",
                "command_delta",
                "feedback_delta",
                "sign_consistent",
                "sign_evaluable",
            } <= joint.keys()


def test_crc_preview_has_stable_known_vector() -> None:
    preview = preview_g1_arm_message(np.arange(7), -np.arange(7), weight=1.0)
    assert preview["crc"] == 708443392


def test_execute_micro_has_no_direct_o6_transport_call() -> None:
    source = inspect.getsource(RealBridgeSession.execute_micro)
    assert "self.o6." not in source
    assert "_send_left_hand" not in source
    assert "_queue_o6_target" in source
    worker = inspect.getsource(RealBridgeSession._o6_monitor_loop)
    assert "_send_left_hand" in worker
    assert "self.o6.feedback()" in worker


def test_release_feedback_does_not_require_o6_cache(tmp_path: Path) -> None:
    groups = {
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
        "left_o6": np.full(6, 50.0),
        "right_o6": np.full(6, 50.0),
    }
    snapshot = FeedbackSnapshot.create(
        groups, g1_mode_machine=5, g1_mode_pr=0, waist=np.zeros(3)
    )
    g1 = MockG1Transport(snapshot)
    o6 = MockO6Transport(
        {"left_o6": groups["left_o6"], "right_o6": groups["right_o6"]}
    )
    with JsonlBridgeLogger(tmp_path / "release_o6_fault.jsonl") as logger:
        session = RealBridgeSession(
            g1,
            o6,
            logger,
            activation_ramp_s=0.02,
            release_ramp_s=0.04,
            post_release_monitor_s=0.02,
        )
        session.arm_hold()
        session.execute_hold(0.02)
        with session._o6_lock:
            session._o6_error = TimeoutError("injected O6 timeout")
        release_error = session.stop("injected O6 timeout", fault=True)
    assert release_error is None
    assert g1.records[-1]["weight"] == 0.0
    assert session.state is BridgeState.STOPPED


def test_right_o6_has_no_command_path_in_worker() -> None:
    source = inspect.getsource(RealBridgeSession._send_left_hand)
    assert "send_right" not in source
    assert '"right_o6_command": None' in source
    assert "right_o6_command_count=0" in source


def test_arms_only_micro_never_queues_o6_command(tmp_path: Path) -> None:
    groups = {
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
        "left_o6": np.full(6, 50.0),
        "right_o6": np.full(6, 50.0),
    }
    snapshot = FeedbackSnapshot.create(
        groups, g1_mode_machine=5, g1_mode_pr=0, waist=np.zeros(3)
    )
    g1 = MockG1Transport(snapshot)
    o6 = MockO6Transport(
        {"left_o6": groups["left_o6"], "right_o6": groups["right_o6"]}
    )
    frames = []
    for index in range(15):
        target = {key: value.tolist() for key, value in groups.items()}
        frames.append(
            {
                "inference_index": index // 3,
                "policy_raw_absolute": target,
                "adapter_absolute": target,
                "ordinary_safety": target,
            }
        )
    with JsonlBridgeLogger(tmp_path / "arms_only.jsonl") as logger:
        session = RealBridgeSession(
            g1,
            o6,
            logger,
            activation_ramp_s=0.02,
            release_ramp_s=0.02,
            post_release_monitor_s=0.02,
            o6_position_commands_enabled=False,
        )
        session.arm_hold()
        session.execute_hold(0.02)
        session.execute_micro({"frames": frames}, publish_left_o6=False)
        session.stop("arms-only mock complete")
    assert len(o6.records) == 0
    assert session.o6_position_command_count == 0
    assert g1.records[-1]["weight"] == 0.0


def test_arms_only_plan_suppresses_both_o6_targets() -> None:
    feedback = {
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "left_o6": [50.0] * 6,
        "right_o6": [60.0] * 6,
    }
    stage = {key: list(value) for key, value in feedback.items()}
    stage["left_o6"] = [10.0] * 6
    source = {
        "schema_version": 1,
        "mode": "one_shot_micro_motion",
        "arming_feedback_preview": {"groups": feedback},
        "frames": [{
            "policy_raw_absolute": stage,
            "adapter_absolute": stage,
            "ordinary_safety": stage,
        }],
        "runtime_preview": {
            "g1_arm_sdk_50hz_ticks": [{"micro_envelope": stage}],
            "o6_can_30hz_ticks": [{"o6_messages": {}}],
        },
        "micro_envelope_counters": {
            key: {group: 1 for group in feedback}
            for key in ("excursion", "velocity", "acceleration")
        },
        "scheduler_contract": {},
    }
    plan = build_arms_only_micro_plan(source)
    assert plan["mode"] == "arms_only_micro_motion_waiting_for_user_confirmation"
    assert plan["left_o6_feedback_only"] is True
    assert plan["left_o6_command_count"] == 0
    assert plan["right_o6_feedback_only"] is True
    assert plan["right_o6_command_count"] == 0
    assert plan["waist_leg_command_count"] == 0
    assert plan["frames"][0]["ordinary_safety"]["left_o6"] == [50.0] * 6
    assert plan["frames"][0]["ordinary_safety"]["right_o6"] == [60.0] * 6


def test_hold_only_plan_remains_available_with_arms_only_preview() -> None:
    feedback = {
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "left_o6": [50.0] * 6,
        "right_o6": [60.0] * 6,
    }
    source = {
        "source": {"checkpoint": {"path": "corrected-checkpoint"}},
        "arming_feedback_preview": {"groups": feedback},
        "hold_preview": {"o6_messages": {"must_be_removed": True}},
    }
    plan = build_hold_only_plan(source)
    assert plan["mode"] == "hold_only_waiting_for_user_confirmation"
    assert plan["hold_preview"]["o6_feedback_only"]["position_command_count"] == 0
    assert "o6_messages" not in plan["hold_preview"]
    assert plan["real_commands_sent"] == 0
