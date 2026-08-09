from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .controller import RealBridgeSession
from .gates import GateSettings, OneTimeToken
from .logging import JsonlBridgeLogger


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if config_path.suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    import yaml

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def reject_obsolete_plan(plan_sha256: str) -> None:
    registry = Path(__file__).resolve().parents[1] / "config/obsolete_real_bridge_artifacts.json"
    if not registry.exists():
        return
    value = json.loads(registry.read_text(encoding="utf-8"))
    if any(item.get("plan_sha256") == plan_sha256 for item in value.get("obsolete", [])):
        raise PermissionError("real bridge plan is explicitly obsolete")


def runtime_preflight(config: dict, *, hold_only: bool = False) -> dict:
    network = config["network"]
    target = f"{network['ssh_user']}@{network['g1_host']}"

    def remote(*command: str) -> str:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, *command],
            check=True,
            text=True,
            capture_output=True,
            timeout=10.0,
        )
        return completed.stdout

    can = remote("ip", "-details", "-statistics", "link", "show", "can1")
    can += remote("ip", "-details", "-statistics", "link", "show", "can2")
    if can.count("can state ERROR-ACTIVE") != 2 or can.count("bitrate 1000000") != 2:
        raise RuntimeError("runtime preflight requires can1/can2 ERROR-ACTIVE at 1 Mbps")
    if "BUS-OFF" in can or "ERROR-PASSIVE" in can:
        raise RuntimeError("runtime preflight detected unsafe CAN state")
    listeners = remote("ss", "-ltnp")
    unused_ports = (5555, 5557, 5561) if hold_only else (5555, 5557, 5558, 5561)
    for port in unused_ports:
        if f":{port} " in listeners:
            raise RuntimeError(f"runtime preflight requires port {port} to be unused")
    processes = remote("ps", "-eo", "pid,args")
    forbidden = (
        "g1_deploy_onnx_ref",
        "g1_o6_zmq_driver",
        "glove_teleop_dual_o6.py",
        "remote_o6_agent.py",
        "MotionSwitcherClient",
    )
    found = [name for name in forbidden if name in processes]
    if found:
        raise RuntimeError("runtime preflight found conflicting command process: " + ", ".join(found))
    result = {
        "can1_can2_error_active_1mbps": True,
        "command_ports_unused": list(unused_ports),
        "conflicting_processes": [],
    }
    if hold_only:
        if ":5558 " not in listeners or "o6_feedback_only_relay.py" not in processes:
            raise RuntimeError("hold-only requires audited O6 feedback-only relay on port 5558")
        result.update({
            "o6_feedback_only_port": 5558,
            "o6_position_command_count_required": 0,
        })
    else:
        remote_hash = remote("sha256sum", network["o6_remote_agent"]).split()[0]
        if remote_hash != network["o6_remote_agent_sha256"]:
            raise RuntimeError("remote O6 agent SHA256 does not match audited config")
        result["remote_o6_agent_sha256"] = remote_hash
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safety-gated G1+O6 micro-motion bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="build a transport-free 0.5 second plan")
    preview.add_argument("--feedback", required=True)
    preview.add_argument("--live-jsonl", required=True)
    preview.add_argument("--adapter-config", default="deployment/config/adapter.yaml")
    preview.add_argument("--micro-config", default="deployment/config/micro_motion.yaml")
    preview.add_argument(
        "--checkpoint",
        default="outputs/formal_train_26_corrected_v1/checkpoint-3000",
    )
    preview.add_argument("--output", required=True)
    preview.add_argument("--hold-output", required=True)
    preview.add_argument("--arms-only-output")

    issue = subparsers.add_parser("issue-token", help="issue a plan-bound one-time token")
    issue.add_argument("--token-file", required=True)
    issue.add_argument("--plan", required=True)
    issue.add_argument("--ttl-s", type=float, default=86400.0)

    execute = subparsers.add_parser("execute", help="perform one hold and one micro window")
    execute.add_argument("--config", default="deployment/config/micro_motion.yaml")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--log", required=True)
    execute.add_argument("--token-file", required=True)
    execute.add_argument("--confirmation-token", required=True)
    execute.add_argument(
        "--phase",
        choices=("hold-only", "arms-only-micro", "model-micro"),
        required=True,
    )
    execute.add_argument("--hardware-transport-enabled", action="store_true")
    execute.add_argument("--command-publication-enabled", action="store_true")
    execute.add_argument("--micro-motion-armed", action="store_true")
    return parser


def _create_real_session(
    config: dict,
    logger: JsonlBridgeLogger,
    *,
    hold_only: bool,
) -> RealBridgeSession:
    # Imports and constructors are intentionally reached only after gates and token consumption.
    from .real_g1 import G1ArmSdkTransport
    if hold_only:
        from .feedback_only_o6 import O6FeedbackOnlySubprocessTransport
    else:
        from .real_o6 import RemoteO6Transport

    network = config["network"]
    g1 = G1ArmSdkTransport(network["g1_dds_interface"])
    try:
        if hold_only:
            o6 = O6FeedbackOnlySubprocessTransport(
                network["o6_feedback_reader_python"],
                Path(__file__).resolve().parent / "o6_feedback_stdout.py",
                f"tcp://{network['g1_host']}:5558",
            )
        else:
            o6 = RemoteO6Transport(
                host=network["g1_host"],
                user=network["ssh_user"],
                remote_python=network["o6_remote_python"],
                remote_agent=network["o6_remote_agent"],
            )
    except Exception:
        g1.close()
        raise
    return RealBridgeSession(
        g1,
        o6,
        logger,
        arm_publish_hz=float(config["g1"]["sdk_publish_hz"]),
        o6_publish_hz=float(config["o6"]["publish_hz"]),
        activation_ramp_s=float(config["g1"]["activation_ramp_s"]),
        release_ramp_s=float(config["g1"]["release_ramp_s"]),
        arm_excursion_rad=float(config["micro_motion"]["arm_max_excursion_rad"]),
        arm_velocity_rad_s=float(config["micro_motion"]["arm_max_velocity_rad_s"]),
        arm_acceleration_rad_s2=float(config["micro_motion"]["arm_max_acceleration_rad_s2"]),
        o6_excursion_points=float(config["micro_motion"]["o6_max_excursion_points"]),
        o6_velocity_points_s=float(config["micro_motion"]["o6_max_velocity_points_s"]),
        release_max_arm_rebound_rad=float(config["micro_motion"]["release_max_arm_rebound_rad"]),
        post_release_monitor_s=float(config["micro_motion"]["post_release_monitor_s"]),
        o6_feedback_stale_timeout_s=(
            float(config["micro_motion"]["feedback_stale_timeout_ms"]) / 1000.0
        ),
        scheduler_max_lateness_s=(
            float(config["micro_motion"]["scheduler_max_lateness_ms"]) / 1000.0
        ),
        o6_position_commands_enabled=not hold_only,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preview":
        from .preview import (
            build_arms_only_micro_plan,
            build_hold_only_plan,
            build_plan_from_live_log,
            write_json,
        )

        plan = build_plan_from_live_log(
            args.feedback,
            args.live_jsonl,
            args.adapter_config,
            args.micro_config,
            args.checkpoint,
        )
        write_json(args.output, plan)
        hold_plan = build_hold_only_plan(plan)
        write_json(args.hold_output, hold_plan)
        result = {
            "micro_output": args.output,
            "micro_sha256": sha256_file(args.output),
            "hold_output": args.hold_output,
            "hold_sha256": sha256_file(args.hold_output),
            "confirmation_token": None,
        }
        if args.arms_only_output:
            arms_only_plan = build_arms_only_micro_plan(plan)
            write_json(args.arms_only_output, arms_only_plan)
            result.update({
                "arms_only_output": args.arms_only_output,
                "arms_only_sha256": sha256_file(args.arms_only_output),
            })
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "issue-token":
        digest = sha256_file(args.plan)
        reject_obsolete_plan(digest)
        token = OneTimeToken.issue(args.token_file, ttl_s=args.ttl_s, bound_sha256=digest)
        print(json.dumps({"confirmation_token": token, "plan_sha256": digest}, indent=2))
        return 0

    config = load_config(args.config)
    gates = GateSettings(
        hardware_transport_enabled=bool(args.hardware_transport_enabled),
        command_publication_enabled=bool(args.command_publication_enabled),
        micro_motion_armed=bool(args.micro_motion_armed),
    )
    gates.require_all()
    hold_only = args.phase == "hold-only"
    feedback_only_o6 = args.phase in ("hold-only", "arms-only-micro")
    preflight = runtime_preflight(config, hold_only=feedback_only_o6)
    plan_digest = sha256_file(args.plan)
    reject_obsolete_plan(plan_digest)
    OneTimeToken.consume(
        args.token_file,
        args.confirmation_token,
        bound_sha256=plan_digest,
    )
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("right_o6_feedback_only") is not True or plan.get("right_o6_command_count") != 0:
        raise PermissionError("plan does not prove right O6 feedback-only with zero commands")
    expected_mode = (
        "hold_only_waiting_for_user_confirmation"
        if args.phase == "hold-only"
        else (
            "arms_only_micro_motion_waiting_for_user_confirmation"
            if args.phase == "arms-only-micro"
            else "one_shot_micro_motion"
        )
    )
    if plan.get("mode") != expected_mode:
        raise PermissionError(f"{args.phase} refuses plan mode {plan.get('mode')!r}")
    if args.phase == "arms-only-micro" and (
        plan.get("left_o6_feedback_only") is not True
        or plan.get("left_o6_command_count") != 0
        or plan.get("waist_leg_command_count") != 0
    ):
        raise PermissionError("arms-only plan must prove zero left O6 and waist/leg commands")
    with JsonlBridgeLogger(args.log) as logger:
        logger.write(
            "gates",
            hardware_transport_enabled=True,
            command_publication_enabled=True,
            micro_motion_armed=True,
            one_time_token_consumed=True,
            plan_sha256=plan_digest,
            runtime_preflight=preflight,
        )
        session = _create_real_session(config, logger, hold_only=feedback_only_o6)
        try:
            session.arm_hold()
            session.execute_hold(float(config["micro_motion"]["full_weight_hold_s"]))
            if args.phase == "arms-only-micro":
                session.execute_micro(plan, publish_left_o6=False)
            elif args.phase == "model-micro":
                session.execute_micro(plan)
        except BaseException as exc:
            session.stop(f"{type(exc).__name__}: {exc}", fault=True)
            raise
        else:
            reason = (
                "completed current-position hold only"
                if args.phase == "hold-only"
                else (
                    "completed one hold and one 0.5 second arms-only micro window"
                    if args.phase == "arms-only-micro"
                    else "completed one hold and one 0.5 second micro window"
                )
            )
            release_error = session.stop(reason)
            if release_error is not None:
                raise RuntimeError(release_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
