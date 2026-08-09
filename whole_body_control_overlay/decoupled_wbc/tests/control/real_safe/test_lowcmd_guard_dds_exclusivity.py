import pytest

from decoupled_wbc.control.real_safe.lowcmd_guard.dds_exclusivity import (
    LOWCMD_TOPIC,
    LOWCMD_TYPE,
    LowcmdExclusivityPolicy,
    PublicationIdentity,
    create_exclusivity_checked_writer,
)


def endpoint(name: str, participant: str = "participant") -> PublicationIdentity:
    return PublicationIdentity(name, participant, LOWCMD_TOPIC, LOWCMD_TYPE)


def test_baseline_allows_debug_mode_zero_or_one_known_firmware_endpoint() -> None:
    policy = LowcmdExclusivityPolicy()
    policy.capture_ai_baseline([])
    assert policy.ai_baseline == set()
    with pytest.raises(RuntimeError, match="zero or one firmware baseline"):
        policy.capture_ai_baseline([endpoint("ai-a"), endpoint("ai-b")])
    policy.capture_ai_baseline([endpoint("ai")])
    assert policy.ai_baseline == {endpoint("ai")}


def test_guard_claim_allows_baseline_plus_one_and_rejects_third_party() -> None:
    policy = LowcmdExclusivityPolicy()
    ai = endpoint("ai", "firmware")
    guard = endpoint("guard", "nx")
    policy.capture_ai_baseline([ai])
    assert policy.claim_guard_endpoint([ai, guard]) == guard
    policy.validate_active_guard([ai, guard])
    policy.validate_active_guard([guard])
    with pytest.raises(RuntimeError, match="unknown"):
        policy.validate_active_guard([ai, guard, endpoint("external")])
    with pytest.raises(RuntimeError, match="disappeared"):
        policy.validate_active_guard([ai])

    debug_policy = LowcmdExclusivityPolicy()
    debug_policy.capture_ai_baseline([])
    assert debug_policy.claim_guard_endpoint([guard]) == guard
    debug_policy.validate_active_guard([guard])
    with pytest.raises(RuntimeError, match="unknown"):
        debug_policy.validate_active_guard([guard, endpoint("external")])


def test_wrong_type_is_fail_closed() -> None:
    policy = LowcmdExclusivityPolicy()
    wrong = PublicationIdentity("x", "p", LOWCMD_TOPIC, "wrong::Type")
    with pytest.raises(RuntimeError, match="unexpected"):
        policy.capture_ai_baseline([wrong])


def test_checked_writer_claims_one_endpoint_and_checks_every_write() -> None:
    ai = endpoint("ai")
    guard = endpoint("guard")
    policy = LowcmdExclusivityPolicy()
    policy.capture_ai_baseline([ai])

    class Monitor:
        def observe_stable(self, *args, **kwargs):
            return frozenset({ai, guard})

        def publications(self, topic):
            return frozenset({ai, guard})

    class Writer:
        def __init__(self):
            self.writes = 0
            self.closed = False

        def write(self, command):
            self.writes += 1

        def close(self):
            self.closed = True

    delegate = Writer()
    writer = create_exclusivity_checked_writer(
        lambda: delegate,
        Monitor(),
        policy,
        discovery_s=1.0,
        stable_s=0.5,
    )
    writer.write(object())
    assert delegate.writes == 1
    writer.close()
    assert delegate.closed is True


def test_read_only_audit_source_contains_no_business_writer_or_control_api() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3]
        / "control/main/teleop/run_g1_lowcmd_publication_audit.py"
    ).read_text()
    for forbidden in (
        "ChannelPublisher",
        "UnitreeLowCmdWriter",
        "ReleaseMode",
        "SelectMode",
        ".Write(",
    ):
        assert forbidden not in source
