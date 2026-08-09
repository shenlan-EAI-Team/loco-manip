#!/usr/bin/env python3
"""One-shot supported G1 WBC + left-wrist-yaw actuation proof."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import yaml

from decoupled_wbc.control.real_safe import StandaloneSafetyLimits
from decoupled_wbc.control.real_safe.standalone import StandaloneSafetyGate
from decoupled_wbc.control.real_safe.lowcmd_guard import (
    ExclusiveGuardLock,
    GuardConfig,
    LowCmdGuardCore,
    LowcmdExclusivityPolicy,
    OneTimeTokenFile,
    create_exclusivity_checked_writer,
)
from decoupled_wbc.control.real_safe.lowcmd_guard.single_wrist_proof import (
    SingleWristProofComposer,
    SingleWristProofConfig,
    SingleWristProofRuntime,
)


CONFIG_DIR = Path(__file__).with_name("configs")
GUARD_CONFIG = CONFIG_DIR / "g1_wbc_commissioning_guard.yaml"
SAFETY_CONFIG = CONFIG_DIR / "g1_standalone_real_safe.yaml"
BASE_MANIFEST = CONFIG_DIR / "g1_wbc_commissioning_manifest.json"
PROOF_CONFIG = CONFIG_DIR / "g1_single_wrist_proof.json"
PROOF_MANIFEST = CONFIG_DIR / "g1_single_wrist_proof_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--hardware-transport-enabled", action="store_true")
    parser.add_argument("--command-publication-enabled", action="store_true")
    parser.add_argument("--lifecycle-armed", action="store_true")
    return parser.parse_args()


def emit_summary(summary: dict[str, object], path: Path) -> None:
    payload = json.dumps(summary, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")
    print(payload, flush=True)


def main() -> int:
    args = parse_args()
    guard_config = GuardConfig.from_mapping(yaml.safe_load(GUARD_CONFIG.read_text()))
    safety = StandaloneSafetyGate(
        StandaloneSafetyLimits.from_mapping(yaml.safe_load(SAFETY_CONFIG.read_text()))
    )
    proof = SingleWristProofConfig.from_json(PROOF_CONFIG)
    token_file = OneTimeTokenFile(args.token_file)
    token = token_file.load()
    core = LowCmdGuardCore(guard_config, safety, one_time_token=token)
    process_lock = ExclusiveGuardLock(Path("/tmp/g1_lowcmd_guard.lock"))
    process_lock.acquire()

    repository_root = Path(__file__).resolve().parents[4]
    from decoupled_wbc.control.real_safe.gear_wbc_producer import (
        GearWbcReadOnlyProducer,
        GearWbcStandingModel,
        verify_artifact_manifest,
    )

    # The original successful WBC chain stays hash-locked; this proof has a
    # second manifest containing only its isolated additions.
    verify_artifact_manifest(BASE_MANIFEST, repository_root=repository_root)
    verify_artifact_manifest(PROOF_MANIFEST, repository_root=repository_root)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from decoupled_wbc.control.real_safe.lowcmd_guard.unitree_backend import (
        UnitreeGuardStateSource,
        UnitreeLowCmdWriter,
        UnitreeMotionModeClient,
    )
    from cyclonedds.domain import DomainParticipant
    from decoupled_wbc.control.real_safe.lowcmd_guard.dds_exclusivity import (
        DdsPublicationMonitor,
        LOWCMD_TOPIC,
    )
    from decoupled_wbc.control.real_safe.lowcmd_guard import LowerBodyMailbox

    ChannelFactoryInitialize(0, args.interface)
    source = UnitreeGuardStateSource()
    mode = UnitreeMotionModeClient()
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

    runtime = SingleWristProofRuntime(
        core,
        source,
        mode,
        writer_factory=writer_factory,
        authorization_commit=lambda: token_file.consume(token),
        proof=proof,
    )

    def signal_fault(signum, _frame) -> None:
        runtime.signal_local_fault(
            f"local signal froze proof reference: {signal.Signals(signum).name}"
        )

    signal.signal(signal.SIGINT, signal_fault)
    signal.signal(signal.SIGTERM, signal_fault)

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
    composer = SingleWristProofComposer(
        mailbox,
        core.safety,
        mailbox_stale_s=guard_config.lower_body_mailbox_stale_s,
        engage_duration_s=core.safety.limits.engage_duration_s,
        lower_rate_limit=core.safety.limits.lower_target_rate_abs_limit,
        lower_step_limit=core.safety.limits.lower_target_step_abs_limit,
        proof=proof,
        arm_rate_limit=guard_config.target_rate_abs_limit[proof.motor_cmd_index],
        arm_step_limit=guard_config.target_step_abs_limit[proof.motor_cmd_index],
    )

    def completion_hold(summary: dict[str, object]) -> None:
        assert runtime.scheduler is not None
        summary.update(
            {
                "result": "PASS_ACTUATION_PROOF_AWAITING_PHYSICAL_ESTOP",
                "producer_inferences": producer.inference_count,
                "transport_scheduler": runtime.scheduler.metrics.summary(),
                "select_mode_attempted": False,
            }
        )
        emit_summary(summary, args.summary)
        print(
            "PROOF_COMPLETE_HOLD: last valid command remains at 500 Hz; "
            "press the physical emergency stop before terminating this exact process.",
            flush=True,
        )

    try:
        runtime.execute(
            token=token,
            hardware_transport_enabled=args.hardware_transport_enabled,
            command_publication_enabled=args.command_publication_enabled,
            lifecycle_armed=args.lifecycle_armed,
            producer=producer,
            composer=composer,
            completion_hold=completion_hold,
        )
    except BaseException as exc:
        if not runtime.release_succeeded:
            emit_summary(
                {
                    "result": "PRE_RELEASE_BLOCKED",
                    "fault": f"{type(exc).__name__}: {exc}",
                    "release_calls": runtime.release_calls,
                    "select_calls": runtime.select_calls,
                    "write_calls": runtime.write_calls,
                    "token_consumed": not args.token_file.exists(),
                },
                args.summary,
            )
            process_lock.close()
            # Unitree DDS reader threads can keep Python alive after a pre-release
            # exception. At this point no writer or ownership change exists, so a
            # direct process exit is fail-closed and cannot interrupt active control.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        summary = runtime.proof_summary(composer)
        summary.update(
            {
                "result": "FAULT_AWAITING_PHYSICAL_ESTOP",
                "fault": f"{type(exc).__name__}: {exc}",
                "writer_transport_alive": runtime._writer_transport_alive(),
                "select_mode_attempted": False,
            }
        )
        emit_summary(summary, args.summary)
        print(
            "FAULT_HOLD: reference changes are frozen; press the physical "
            "emergency stop before terminating this exact process.",
            flush=True,
        )
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
