import numpy as np
import pytest

from decoupled_wbc.control.real_safe import (
    RobotSnapshot,
    SafetyFault,
    StandaloneRealSafeCore,
    StandaloneRealSafeRuntime,
    StandaloneSafetyLimits,
    StandaloneState,
)

from .test_standalone import limits, snapshot


class Reader:
    def __init__(self):
        self.q = snapshot(0.0).q

    def read(self, now: float) -> RobotSnapshot:
        return snapshot(now, q=self.q)


class Mode:
    def __init__(self):
        self.release_count = 0

    def release_mode(self, *, takeover_confirmed: bool) -> None:
        assert takeover_confirmed
        self.release_count += 1


class Transport:
    def __init__(self):
        self.armed = False
        self.disarm_count = 0
        self.prepared = None
        self.records = []

    def prepare(self, q, dq, tau) -> None:
        self.prepared = (q.copy(), dq.copy(), tau.copy())

    def arm_writes(self, *, takeover_confirmed: bool) -> None:
        assert takeover_confirmed
        self.armed = True

    def disarm_writes(self) -> None:
        self.armed = False
        self.disarm_count += 1

    def write_prepared(self) -> None:
        if not self.armed:
            raise PermissionError("disarmed")
        self.records.append(self.prepared)

    def send(self, q, dq, tau) -> None:
        self.prepare(q, dq, tau)
        self.write_prepared()


class Exit:
    def __init__(self, verified: bool, completes: bool = False):
        self.verified_for_supported_robot = verified
        self.completes = completes
        self.records = []

    def execute(self, reason, last_safe_q) -> bool:
        self.records.append((reason, None if last_safe_q is None else last_safe_q.copy()))
        return self.completes


def runtime(exit_strategy: Exit):
    core = StandaloneRealSafeCore(limits(), one_time_arm_token="one-shot")
    reader = Reader()
    mode = Mode()
    transport = Transport()
    return StandaloneRealSafeRuntime(core, reader, mode, transport, exit_strategy), mode, transport


def test_read_only_has_zero_release_and_zero_write() -> None:
    app, mode, transport = runtime(Exit(verified=False))
    for index in range(100):
        app.read_only_step(1.0 + index * 0.02)
    assert app.core.state == StandaloneState.READ_ONLY
    assert mode.release_count == 0
    assert transport.records == []


def test_arm_is_blocked_until_fault_exit_is_verified() -> None:
    app, mode, transport = runtime(Exit(verified=False))
    with pytest.raises(PermissionError, match="FAULT exit strategy"):
        app.arm_control("one-shot", 1.0)
    assert app.core.state == StandaloneState.READ_ONLY
    assert mode.release_count == 0
    assert transport.records == []


def test_verified_mock_orders_prepare_release_arm_write() -> None:
    events = []

    class OrderedMode(Mode):
        def release_mode(self, *, takeover_confirmed: bool) -> None:
            events.append("release")
            super().release_mode(takeover_confirmed=takeover_confirmed)

    class OrderedTransport(Transport):
        def prepare(self, q, dq, tau) -> None:
            events.append("prepare")
            super().prepare(q, dq, tau)

        def arm_writes(self, *, takeover_confirmed: bool) -> None:
            events.append("arm_writes")
            super().arm_writes(takeover_confirmed=takeover_confirmed)

        def write_prepared(self) -> None:
            events.append("write")
            super().write_prepared()

    core = StandaloneRealSafeCore(limits(), one_time_arm_token="one-shot")
    mode = OrderedMode()
    transport = OrderedTransport()
    app = StandaloneRealSafeRuntime(core, Reader(), mode, transport, Exit(verified=True))
    hold = app.arm_control("one-shot", 1.0)
    assert events == ["prepare", "release", "arm_writes", "write"]
    np.testing.assert_allclose(transport.records[0][0], hold)
    np.testing.assert_allclose(transport.records[0][1], 0.0)
    np.testing.assert_allclose(transport.records[0][2], 0.0)
    assert app.core.state == StandaloneState.HOLD


def test_fault_revokes_normal_write_gate_before_exit_strategy() -> None:
    events = []

    class OrderedTransport(Transport):
        def disarm_writes(self) -> None:
            events.append("disarm")
            super().disarm_writes()

    class OrderedExit(Exit):
        def execute(self, reason, last_safe_q) -> bool:
            events.append("exit")
            return super().execute(reason, last_safe_q)

    core = StandaloneRealSafeCore(limits(), one_time_arm_token="one-shot")
    transport = OrderedTransport()
    exit_strategy = OrderedExit(verified=True)
    app = StandaloneRealSafeRuntime(core, Reader(), Mode(), transport, exit_strategy)
    app.arm_control("one-shot", 1.0)

    with pytest.raises(SafetyFault, match="watchdog"):
        app.hold_step(1.1)

    assert events == ["disarm", "exit"]
    assert transport.armed is False
    assert core.state == StandaloneState.FAULT


def test_disarm_failure_does_not_skip_verified_fault_exit() -> None:
    class BrokenDisarmTransport(Transport):
        def disarm_writes(self) -> None:
            raise RuntimeError("gate hardware error")

    core = StandaloneRealSafeCore(limits(), one_time_arm_token="one-shot")
    transport = BrokenDisarmTransport()
    exit_strategy = Exit(verified=True)
    app = StandaloneRealSafeRuntime(core, Reader(), Mode(), transport, exit_strategy)
    app.arm_control("one-shot", 1.0)

    with pytest.raises(SafetyFault, match="watchdog"):
        app.hold_step(1.1)

    assert len(exit_strategy.records) == 1
    assert "write-gate disarm failed" in exit_strategy.records[0][0]
    assert core.state == StandaloneState.FAULT
