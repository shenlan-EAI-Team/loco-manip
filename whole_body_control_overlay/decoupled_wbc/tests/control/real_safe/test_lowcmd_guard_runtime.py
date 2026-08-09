from dataclasses import replace
import time

import pytest

from decoupled_wbc.control.real_safe import SafetyFault
from decoupled_wbc.control.real_safe.lowcmd_guard import (
    GuardState,
    LowerBodyMailbox,
    LowCmdGuardCore,
    LowCmdGuardRuntime,
    WbcGuardCommandComposer,
)

from .test_lowcmd_guard_core import guard_config, safety_gate, snapshot


class Source:
    def latest(self, now):
        return snapshot(time.monotonic())


class Mode:
    def __init__(self, events, *, recover=True, release_status=0):
        self.events = events
        self.owner = "ai"
        self.recover = recover
        self.release_status = release_status

    def check_mode(self):
        return 0, "0", self.owner

    def release_mode(self):
        self.events.append("release")
        if self.release_status == 0:
            self.owner = ""
        return self.release_status

    def select_mode(self, owner):
        self.events.append("select")
        if self.recover:
            self.owner = owner
        return 0


class Writer:
    def __init__(self, events, *, fail_first_write=False):
        self.events = events
        self.closed = False
        self.times = []
        self.fail_first_write = fail_first_write

    def write(self, command):
        assert not self.closed
        assert command.q.shape == (29,)
        if self.fail_first_write and not self.times:
            self.events.append("write_failed")
            raise RuntimeError("injected first write failure")
        self.events.append("write")
        self.times.append(time.monotonic())

    def close(self):
        self.events.append("close")
        self.closed = True


def make_runtime(
    *,
    recover=True,
    recovery_timeout=0.03,
    release_status=0,
    fail_first_write=False,
):
    events = []
    config = replace(
        guard_config(enabled=True, recovery_verified=True),
        hold_duration_s=0.05,
        post_recovery_monitor_s=0.02,
        recovery_verify_timeout_s=recovery_timeout,
    )
    core = LowCmdGuardCore(config, safety_gate(), one_time_token="one-shot")
    mode = Mode(events, recover=recover, release_status=release_status)
    writer = Writer(events, fail_first_write=fail_first_write)
    runtime = LowCmdGuardRuntime(core, Source(), mode, lambda: writer)
    return runtime, core, mode, writer, events


def test_single_lifecycle_orders_release_first_write_hold_select_confirm_close() -> None:
    runtime, core, mode, writer, events = make_runtime()
    summary = runtime.execute_current_q_lifecycle(
        token="one-shot",
        hardware_transport_enabled=True,
        command_publication_enabled=True,
        lifecycle_armed=True,
    )
    assert core.state == GuardState.STOPPED
    assert summary["release_calls"] == 1
    assert summary["select_calls"] == 1
    assert summary["release_to_first_write_s"] <= core.config.first_write_deadline_s
    assert events[0:2] == ["release", "write"]
    assert events.index("select") > events.index("write")
    assert events.index("close") > events.index("select")
    assert writer.closed is True
    assert mode.owner == "ai"
    assert len(writer.times) >= 15
    assert len(summary["feedback_trace"]) > 0
    assert len(summary["owner_trace"]) > 0
    assert len(summary["max_abs_feedback_delta_rad_by_motor"]) == 29
    assert len(summary["max_abs_dq_rad_s_by_motor"]) == 29
    assert len(summary["max_abs_tau_est_nm_by_motor"]) == 29


def test_writer_factory_is_never_called_when_default_gates_block() -> None:
    called = []
    core = LowCmdGuardCore(guard_config(), safety_gate(), one_time_token="one-shot")
    runtime = LowCmdGuardRuntime(
        core,
        Source(),
        Mode([]),
        lambda: called.append(True),
    )
    with pytest.raises(PermissionError, match="real execution is disabled"):
        runtime.execute_current_q_lifecycle(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
        )
    assert called == []
    assert runtime.release_calls == 0
    assert runtime.write_calls == 0


def test_authorization_commit_precedes_writer_and_release() -> None:
    events = []
    config = replace(
        guard_config(enabled=True, recovery_verified=True),
        hold_duration_s=0.02,
        post_recovery_monitor_s=0.01,
    )
    core = LowCmdGuardCore(config, safety_gate(), one_time_token="one-shot")
    mode = Mode(events)

    def writer_factory():
        events.append("writer_factory")
        return Writer(events)

    runtime = LowCmdGuardRuntime(
        core,
        Source(),
        mode,
        writer_factory,
        authorization_commit=lambda: events.append("token_consumed"),
    )
    runtime.execute_current_q_lifecycle(
        token="one-shot",
        hardware_transport_enabled=True,
        command_publication_enabled=True,
        lifecycle_armed=True,
    )
    assert events[:3] == ["token_consumed", "writer_factory", "release"]


def test_local_signal_before_release_closes_silent_writer_and_never_releases() -> None:
    events = []
    config = guard_config(enabled=True, recovery_verified=True)
    core = LowCmdGuardCore(config, safety_gate(), one_time_token="one-shot")
    mode = Mode(events)
    writer = Writer(events)
    runtime = None

    def commit():
        assert runtime is not None
        runtime.signal_local_fault("SIGTERM before release")

    runtime = LowCmdGuardRuntime(
        core,
        Source(),
        mode,
        lambda: writer,
        authorization_commit=commit,
    )
    with pytest.raises(SafetyFault, match="SIGTERM before release"):
        runtime.execute_current_q_lifecycle(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
        )
    assert runtime.release_calls == 0
    assert runtime.write_calls == 0
    assert events == ["close"]
    assert writer.closed is True


def test_recovery_timeout_never_closes_the_active_hold_writer() -> None:
    runtime, core, _mode, writer, _events = make_runtime(recover=False)
    with pytest.raises(SafetyFault, match="owner verification timeout"):
        runtime.execute_current_q_lifecycle(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
        )
    assert writer.closed is False
    assert runtime.scheduler is not None
    assert runtime.scheduler._thread is not None
    assert runtime.scheduler._thread.is_alive()
    runtime.scheduler.stop()


def test_failed_release_closes_silent_writer_without_select_or_write() -> None:
    runtime, core, mode, writer, events = make_runtime(release_status=9)
    with pytest.raises(SafetyFault, match="ReleaseMode failed"):
        runtime.execute_current_q_lifecycle(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
        )
    assert runtime.release_succeeded is False
    assert runtime.write_calls == 0
    assert runtime.select_calls == 0
    assert events == ["release", "close"]
    assert writer.closed is True
    assert mode.owner == "ai"
    assert core.state == GuardState.FAULT_BLOCKED


def test_first_write_failure_still_requests_local_ai_recovery() -> None:
    runtime, core, mode, writer, events = make_runtime(fail_first_write=True)
    with pytest.raises(RuntimeError, match="first write failure"):
        runtime.execute_current_q_lifecycle(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
        )
    assert runtime.release_succeeded is True
    assert events == ["release", "write_failed", "select", "close"]
    assert runtime.select_calls == 1
    assert writer.closed is True
    assert mode.owner == "ai"
    assert core.state == GuardState.STOPPED


def test_local_signal_during_hold_recovers_without_pc_path() -> None:
    runtime, core, mode, writer, events = make_runtime()

    original_validate = runtime._validate_live_hold

    def request_stop_after_first_validation():
        original_validate()
        runtime.signal_local_fault("SIGTERM")

    runtime._validate_live_hold = request_stop_after_first_validation
    with pytest.raises(SafetyFault, match="SIGTERM"):
        runtime.execute_current_q_lifecycle(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
        )
    assert runtime.recovery_attempted is True
    assert runtime.select_calls == 1
    assert mode.owner == "ai"
    assert writer.closed is True
    assert core.state == GuardState.STOPPED


def test_failed_recovery_is_not_retried_automatically() -> None:
    runtime, core, _mode, writer, events = make_runtime(recover=False)
    with pytest.raises(SafetyFault, match="owner verification timeout"):
        runtime.execute_current_q_lifecycle(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
        )
    assert runtime.recovery_attempted is True
    assert runtime.select_calls == 1
    assert events.count("select") == 1
    assert writer.closed is False
    assert core.state == GuardState.VERIFY_OWNER
    assert runtime._writer_transport_alive() is True
    runtime.scheduler.stop()


def test_supported_wbc_commissioning_uses_one_writer_for_hold_engage_and_stand() -> None:
    events = []
    config = replace(
        guard_config(enabled=True, recovery_verified=False),
        commissioning_execution_enabled=True,
        hold_duration_s=0.05,
        commissioning_stand_duration_s=0.03,
        post_recovery_monitor_s=0.02,
    )
    safety = safety_gate()
    core = LowCmdGuardCore(config, safety, one_time_token="one-shot")
    mode = Mode(events)
    writer = Writer(events)
    runtime = LowCmdGuardRuntime(core, Source(), mode, lambda: writer)
    mailbox = LowerBodyMailbox()
    command_composer = WbcGuardCommandComposer(
        mailbox,
        safety,
        mailbox_stale_s=0.1,
        engage_duration_s=0.05,
        lower_rate_limit=safety.limits.lower_target_rate_abs_limit,
        lower_step_limit=safety.limits.lower_target_step_abs_limit,
    )

    class Producer:
        def __init__(self):
            self.inference_count = 0

        def tick(self):
            self.inference_count += 1
            mailbox.publish(
                snapshot(time.monotonic()).robot.q[:15],
                timestamp=time.monotonic(),
                sequence=self.inference_count,
            )

    producer = Producer()
    summary = runtime.execute_wbc_commissioning(
        token="one-shot",
        hardware_transport_enabled=True,
        command_publication_enabled=True,
        lifecycle_armed=True,
        producer=producer,
        composer=command_composer,
    )
    assert summary["state"] == GuardState.STOPPED.value
    assert summary["commissioning"] is True
    assert summary["producer_inferences"] >= 5
    assert summary["composer_fault"] is None
    assert runtime.release_calls == 1
    assert runtime.select_calls == 1
    assert writer.closed is True
    assert events.count("release") == 1
    assert events.count("select") == 1


def test_invalid_wbc_preflight_blocks_before_writer_token_and_release() -> None:
    events = []
    config = replace(
        guard_config(enabled=True, recovery_verified=False),
        commissioning_execution_enabled=True,
    )
    safety = safety_gate()
    core = LowCmdGuardCore(config, safety, one_time_token="one-shot")
    writer_factory_calls = []
    runtime = LowCmdGuardRuntime(
        core,
        Source(),
        Mode(events),
        lambda: writer_factory_calls.append(True),
        authorization_commit=lambda: events.append("token_consumed"),
    )
    mailbox = LowerBodyMailbox()
    command_composer = WbcGuardCommandComposer(
        mailbox,
        safety,
        mailbox_stale_s=0.1,
        engage_duration_s=3.0,
        lower_rate_limit=safety.limits.lower_target_rate_abs_limit,
        lower_step_limit=safety.limits.lower_target_step_abs_limit,
    )

    class InvalidProducer:
        inference_count = 0

        @staticmethod
        def tick():
            raise SafetyFault("target outside hard limits")

    with pytest.raises(SafetyFault, match="outside hard limits"):
        runtime.execute_wbc_commissioning(
            token="one-shot",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            lifecycle_armed=True,
            producer=InvalidProducer(),
            composer=command_composer,
        )
    assert writer_factory_calls == []
    assert events == []
    assert runtime.release_calls == 0
    assert runtime.write_calls == 0
    assert core.state == GuardState.READ_ONLY
