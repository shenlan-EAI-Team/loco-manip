#!/usr/bin/env python3
"""G1-resident LowCmd lifecycle guard. Defaults to strictly read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import signal
import time

import yaml

from decoupled_wbc.control.real_safe import StandaloneSafetyLimits
from decoupled_wbc.control.real_safe.standalone import SafetyFault, StandaloneSafetyGate
from decoupled_wbc.control.real_safe.lowcmd_guard import (
    ExclusiveGuardLock,
    GuardConfig,
    LowCmdGuardCore,
    LowCmdGuardRuntime,
    LowcmdExclusivityPolicy,
    OneTimeTokenFile,
    create_exclusivity_checked_writer,
)


CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_GUARD_CONFIG = CONFIG_DIR / "g1_lowcmd_guard.yaml"
COMMISSIONING_GUARD_CONFIG = CONFIG_DIR / "g1_wbc_commissioning_guard.yaml"
DEFAULT_SAFETY_CONFIG = CONFIG_DIR / "g1_standalone_real_safe.yaml"
DEFAULT_COMMISSIONING_MANIFEST = CONFIG_DIR / "g1_wbc_commissioning_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument(
        "--mode",
        choices=("read-only", "lifecycle", "wbc-commissioning"),
        default="read-only",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--guard-config", type=Path, default=DEFAULT_GUARD_CONFIG)
    parser.add_argument("--safety-config", type=Path, default=DEFAULT_SAFETY_CONFIG)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=DEFAULT_COMMISSIONING_MANIFEST,
    )
    parser.add_argument("--hardware-transport-enabled", action="store_true")
    parser.add_argument("--command-publication-enabled", action="store_true")
    parser.add_argument("--lifecycle-armed", action="store_true")
    return parser.parse_args()


def load_core(args: argparse.Namespace):
    guard_values = yaml.safe_load(args.guard_config.read_text())
    safety_values = yaml.safe_load(args.safety_config.read_text())
    guard_config = GuardConfig.from_mapping(guard_values)
    safety_limits = StandaloneSafetyLimits.from_mapping(safety_values)
    safety = StandaloneSafetyGate(safety_limits)
    token = secrets.token_urlsafe(32)
    token_file = None
    if args.mode != "read-only":
        if args.token_file is None:
            raise PermissionError("control mode requires an explicit one-time token file")
        token_file = OneTimeTokenFile(args.token_file)
        token = token_file.load()
    return (
        guard_config,
        LowCmdGuardCore(guard_config, safety, one_time_token=token),
        token,
        token_file,
    )


def emit_summary(summary: dict[str, object], path: Path | None) -> None:
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n")


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("duration must be positive")
    guard_config, core, token, token_file = load_core(args)
    process_lock = ExclusiveGuardLock(Path("/tmp/g1_lowcmd_guard.lock"))
    process_lock.acquire()

    # Import and initialize the Unitree backend only after configuration has
    # parsed successfully. These constructors create subscribers and a query
    # client, never a LowCmd publisher or ownership change.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from decoupled_wbc.control.real_safe.lowcmd_guard.unitree_backend import (
        UnitreeGuardStateSource,
        UnitreeLowCmdWriter,
        UnitreeMotionModeClient,
    )

    ChannelFactoryInitialize(0, args.interface)
    source = UnitreeGuardStateSource()
    mode = UnitreeMotionModeClient()

    if args.mode == "read-only":
        start = time.monotonic()
        deadline = start + args.duration
        accepted = 0
        missing = 0
        owners = set()
        mode_machines = set()
        next_tick = start
        owner = None
        last_owner_query = float("-inf")
        while time.monotonic() < deadline:
            try:
                if owner is None or time.monotonic() - last_owner_query >= 1.0:
                    status, _form, owner = mode.check_mode()
                    last_owner_query = time.monotonic()
                    if status != 0:
                        raise SafetyFault(f"CheckMode failed: status={status}")
                before = time.monotonic()
                snapshot = source.latest(before)
                now = time.monotonic()
                core.observe(snapshot, owner, now)
                accepted += 1
                owners.add(owner)
                mode_machines.add(snapshot.mode_machine)
            except SafetyFault as exc:
                if "not ready" not in str(exc):
                    raise
                missing += 1
            next_tick += 1.0 / guard_config.policy_target_frequency_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))
        emit_summary(
            {
                "state": core.state.value,
                "read_only": True,
                "accepted_snapshots": accepted,
                "missing_snapshots": missing,
                "owners": sorted(owners),
                "mode_machines": sorted(mode_machines),
                "lowcmd_publishers_created": 0,
                "lowcmd_writes": 0,
                "release_calls": 0,
                "select_calls": 0,
                "transport_frequency_hz_configured": guard_config.transport_frequency_hz,
                "measured_minimum_transport_frequency_hz": (
                    guard_config.measured_minimum_transport_frequency_hz
                ),
            },
            args.summary,
        )
        process_lock.close()
        return 0 if accepted else 1

    from cyclonedds.domain import DomainParticipant
    from decoupled_wbc.control.real_safe.lowcmd_guard.dds_exclusivity import (
        DdsPublicationMonitor,
        LOWCMD_TOPIC,
    )

    discovery_participant = DomainParticipant(0)
    publication_monitor = DdsPublicationMonitor(discovery_participant)
    baseline = publication_monitor.observe_stable(
        LOWCMD_TOPIC,
        discovery_s=5.0,
        stable_s=1.0,
    )
    exclusivity_policy = LowcmdExclusivityPolicy()
    exclusivity_policy.capture_ai_baseline(baseline)

    def writer_factory():
        return create_exclusivity_checked_writer(
            UnitreeLowCmdWriter,
            publication_monitor,
            exclusivity_policy,
            discovery_s=2.0,
            stable_s=0.5,
        )

    runtime = LowCmdGuardRuntime(
        core,
        source,
        mode,
        writer_factory=writer_factory,
        authorization_commit=(
            (lambda: token_file.consume(token)) if token_file is not None else None
        ),
    )

    def request_signal_recovery(signum, _frame) -> None:
        runtime.signal_local_fault(f"local signal requested recovery: {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, request_signal_recovery)
    signal.signal(signal.SIGTERM, request_signal_recovery)
    try:
        if args.mode == "wbc-commissioning":
            from decoupled_wbc.control.real_safe.gear_wbc_producer import (
                GearWbcReadOnlyProducer,
                GearWbcStandingModel,
                verify_artifact_manifest,
            )
            from decoupled_wbc.control.real_safe.lowcmd_guard import (
                LowerBodyMailbox,
                WbcGuardCommandComposer,
            )

            repository_root = Path(__file__).resolve().parents[4]
            if args.guard_config.resolve() != COMMISSIONING_GUARD_CONFIG.resolve():
                raise PermissionError(
                    "wbc-commissioning requires the hash-locked commissioning guard config"
                )
            verify_artifact_manifest(
                args.artifact_manifest,
                repository_root=repository_root,
            )
            gear_config = (
                repository_root
                / "decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml"
            )
            balance_model = (
                repository_root
                / "decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx"
            )
            mailbox = LowerBodyMailbox()
            producer = GearWbcReadOnlyProducer(
                source,
                core.safety,
                GearWbcStandingModel.from_onnx(gear_config, balance_model),
                mailbox,
            )
            composer = WbcGuardCommandComposer(
                mailbox,
                core.safety,
                mailbox_stale_s=guard_config.lower_body_mailbox_stale_s,
                engage_duration_s=core.safety.limits.engage_duration_s,
                lower_rate_limit=core.safety.limits.lower_target_rate_abs_limit,
                lower_step_limit=core.safety.limits.lower_target_step_abs_limit,
            )
            summary = runtime.execute_wbc_commissioning(
                token=token,
                hardware_transport_enabled=args.hardware_transport_enabled,
                command_publication_enabled=args.command_publication_enabled,
                lifecycle_armed=args.lifecycle_armed,
                producer=producer,
                composer=composer,
            )
        else:
            summary = runtime.execute_current_q_lifecycle(
                token=token,
                hardware_transport_enabled=args.hardware_transport_enabled,
                command_publication_enabled=args.command_publication_enabled,
                lifecycle_armed=args.lifecycle_armed,
            )
        emit_summary(summary, args.summary)
        process_lock.close()
        return 0
    except BaseException as exc:
        # Before ReleaseMode, fail closed and return. After ReleaseMode, never
        # turn a Python exception into process exit while local HOLD is active.
        if not runtime.release_succeeded:
            process_lock.close()
            raise
        if core.state.value == "STOPPED":
            emit_summary(
                {
                    "state": core.state.value,
                    "result": "recovered_after_fault",
                    "fault": f"{type(exc).__name__}: {exc}",
                    "release_calls": runtime.release_calls,
                    "select_calls": runtime.select_calls,
                    "write_calls": runtime.write_calls,
                    "writer_transport_alive": runtime._writer_transport_alive(),
                    **runtime.diagnostic_summary(),
                },
                args.summary,
            )
            process_lock.close()
            return 1
        print(f"FAULT_BLOCKED after ReleaseMode: {type(exc).__name__}: {exc}")
        if runtime._writer_transport_alive():
            print("Local HOLD writer remains active; use hardware emergency stop if required.")
        else:
            print("LowCmd transport is NOT alive; physical support/emergency stop is required.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
