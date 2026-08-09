"""Read-only DDS publication discovery and LowCmd endpoint exclusivity policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time
from typing import Callable, Iterable


LOWCMD_TOPIC = "rt/lowcmd"
LOWCMD_TYPE = "unitree_hg::msg::dds_::LowCmd_"


@dataclass(frozen=True, order=True)
class PublicationIdentity:
    key: str
    participant_key: str
    topic_name: str
    type_name: str

    def to_mapping(self) -> dict[str, str]:
        return asdict(self)


class LowcmdExclusivityPolicy:
    """Tracks the observed firmware baseline and one future guard endpoint."""

    def __init__(self) -> None:
        self.ai_baseline: frozenset[PublicationIdentity] | None = None
        self.guard_endpoint: PublicationIdentity | None = None

    @staticmethod
    def _validate_types(publications: Iterable[PublicationIdentity]) -> frozenset[PublicationIdentity]:
        values = frozenset(publications)
        invalid = [value for value in values if value.topic_name != LOWCMD_TOPIC]
        if invalid:
            raise ValueError("exclusivity policy received a non-lowcmd publication")
        wrong_type = [value for value in values if value.type_name != LOWCMD_TYPE]
        if wrong_type:
            raise RuntimeError(f"unexpected rt/lowcmd DDS types: {wrong_type}")
        return values

    def capture_ai_baseline(self, publications: Iterable[PublicationIdentity]) -> None:
        """Capture zero debug-mode endpoints or one known ai firmware endpoint."""
        values = self._validate_types(publications)
        if len(values) > 1:
            raise RuntimeError(
                "expected zero or one firmware baseline rt/lowcmd endpoint, "
                f"observed {len(values)}"
            )
        self.ai_baseline = values

    def claim_guard_endpoint(self, publications: Iterable[PublicationIdentity]) -> PublicationIdentity:
        if self.ai_baseline is None:
            raise RuntimeError("ai baseline must be captured before guard endpoint claim")
        values = self._validate_types(publications)
        new_values = values - self.ai_baseline
        if len(new_values) != 1 or not self.ai_baseline.issubset(values):
            raise RuntimeError(
                "guard writer discovery must contain the ai baseline plus exactly one new endpoint"
            )
        self.guard_endpoint = next(iter(new_values))
        return self.guard_endpoint

    def validate_active_guard(self, publications: Iterable[PublicationIdentity]) -> None:
        if self.ai_baseline is None or self.guard_endpoint is None:
            raise RuntimeError("ai baseline and guard endpoint must both be known")
        values = self._validate_types(publications)
        allowed = self.ai_baseline | {self.guard_endpoint}
        unknown = values - allowed
        if unknown:
            raise RuntimeError(f"unknown rt/lowcmd publication endpoints: {sorted(unknown)}")
        if self.guard_endpoint not in values:
            raise RuntimeError("guard rt/lowcmd publication endpoint disappeared")


class DdsPublicationMonitor:
    """Maintains alive publication endpoints from CycloneDDS built-in topics."""

    def __init__(self, participant) -> None:
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication

        self._reader = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
        self._alive: dict[str, PublicationIdentity] = {}

    def poll(self) -> None:
        from cyclonedds.core import InstanceState

        for sample in self._reader.take(4096):
            key = str(sample.key)
            info = sample.sample_info
            if info.instance_state == InstanceState.Alive:
                self._alive[key] = PublicationIdentity(
                    key=key,
                    participant_key=str(sample.participant_key),
                    topic_name=str(sample.topic_name),
                    type_name=str(sample.type_name),
                )
            else:
                self._alive.pop(key, None)

    def publications(self, topic_name: str) -> frozenset[PublicationIdentity]:
        self.poll()
        return frozenset(
            value for value in self._alive.values() if value.topic_name == topic_name
        )

    def observe_stable(
        self,
        topic_name: str,
        *,
        discovery_s: float,
        stable_s: float,
        poll_s: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> frozenset[PublicationIdentity]:
        if discovery_s <= 0 or stable_s <= 0 or stable_s > discovery_s or poll_s <= 0:
            raise ValueError("invalid DDS discovery/stability timing")
        deadline = clock() + discovery_s
        last_values: frozenset[PublicationIdentity] | None = None
        stable_since = clock()
        while clock() < deadline:
            values = self.publications(topic_name)
            if values != last_values:
                last_values = values
                stable_since = clock()
            sleep(poll_s)
        if last_values is None or clock() - stable_since < stable_s:
            raise TimeoutError(f"DDS publications for {topic_name!r} did not stabilize")
        return last_values


class ExclusivityCheckedWriter:
    """Checks the discovered LowCmd endpoint set before every delegated write."""

    def __init__(
        self,
        delegate,
        monitor: DdsPublicationMonitor,
        policy: LowcmdExclusivityPolicy,
    ) -> None:
        self._delegate = delegate
        self._monitor = monitor
        self._policy = policy

    def write(self, command) -> None:
        self._policy.validate_active_guard(self._monitor.publications(LOWCMD_TOPIC))
        self._delegate.write(command)

    def close(self) -> None:
        self._delegate.close()


def create_exclusivity_checked_writer(
    writer_factory,
    monitor: DdsPublicationMonitor,
    policy: LowcmdExclusivityPolicy,
    *,
    discovery_s: float,
    stable_s: float,
):
    """Construct a silent writer, claim its one new endpoint, or close it."""
    delegate = writer_factory()
    try:
        publications = monitor.observe_stable(
            LOWCMD_TOPIC,
            discovery_s=discovery_s,
            stable_s=stable_s,
        )
        policy.claim_guard_endpoint(publications)
        return ExclusivityCheckedWriter(delegate, monitor, policy)
    except BaseException:
        delegate.close()
        raise


def create_monitor_for_interface(interface: str, domain_id: int = 0):
    """Create only a discovery participant/reader on the requested interface."""
    if not interface or re.fullmatch(r"[A-Za-z0-9_.:-]+", interface) is None:
        raise ValueError("network interface name is invalid")
    from cyclonedds.domain import Domain, DomainParticipant

    config = f'''<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS><Domain Id="any"><General><Interfaces>
<NetworkInterface name="{interface}" priority="default" multicast="default"/>
</Interfaces></General></Domain></CycloneDDS>'''
    domain = Domain(domain_id, config)
    participant = DomainParticipant(domain_id)
    monitor = DdsPublicationMonitor(participant)
    # Keep the explicit domain and participant alive as long as the monitor.
    monitor._domain = domain
    monitor._participant = participant
    return monitor
