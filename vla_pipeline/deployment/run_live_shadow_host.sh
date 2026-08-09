#!/usr/bin/env bash
# Fail-closed host entry point for read-only G1 Live Shadow.
# This script never starts a remote process and never creates a command transport.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOYMENT_ROOT="$PROJECT_ROOT/deployment"
LIVE_CONFIG="$DEPLOYMENT_ROOT/config/live_shadow.yaml"
ADAPTER_CONFIG="$DEPLOYMENT_ROOT/config/adapter.yaml"
RUNNER="$DEPLOYMENT_ROOT/run_live_shadow.py"
POLICY_PY="/home/slxy/下载/Isaac-GR00T/.venv/bin/python"
SCENARIO="A"
DURATION="120"
CHECKPOINT="$PROJECT_ROOT/outputs/formal_train_26_corrected_v1/checkpoint-3000"
SHORTENED_CORRECTED="false"

usage() {
  cat <<'EOF'
Usage: deployment/run_live_shadow_host.sh [options]

Options:
  --scenario A|B         A is the stationary 120-second gate; B is visual-response only.
  --duration SECONDS     Run duration (scenario A normally requires at least 120 seconds).
  --checkpoint PATH      Explicit policy checkpoint (default: corrected checkpoint-3000).
  --shortened-corrected  Permit one 30-60 second Scenario A run with corrected checkpoint-3000.
  --interface NAME       Override the configured host-side G1 wired interface.
  --g1-host ADDRESS      Override the configured G1 address.
  --ssh-user USER        G1 SSH user (default: unitree).
  -h, --help             Show this help.

This entry point performs read-only preflight checks and then runs the existing
Null Sink Live Shadow. It does not start camera/O6 services over SSH and does not
start SONIC, g1_deploy_onnx_ref, a LowCmd publisher, or any command bridge.
EOF
}

CONFIG_VALUE() {
  "$POLICY_PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])' "$LIVE_CONFIG" "$1"
}

G1_INTERFACE=""
G1_HOST=""
SSH_USER="unitree"

while (($#)); do
  case "$1" in
    --scenario)
      [[ $# -ge 2 ]] || { echo "ERROR: --scenario requires a value" >&2; exit 2; }
      SCENARIO="$2"
      shift 2
      ;;
    --duration)
      [[ $# -ge 2 ]] || { echo "ERROR: --duration requires a value" >&2; exit 2; }
      DURATION="$2"
      shift 2
      ;;
    --interface)
      [[ $# -ge 2 ]] || { echo "ERROR: --interface requires a value" >&2; exit 2; }
      G1_INTERFACE="$2"
      shift 2
      ;;
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint requires a value" >&2; exit 2; }
      CHECKPOINT="$2"
      shift 2
      ;;
    --shortened-corrected)
      SHORTENED_CORRECTED="true"
      shift
      ;;
    --g1-host)
      [[ $# -ge 2 ]] || { echo "ERROR: --g1-host requires a value" >&2; exit 2; }
      G1_HOST="$2"
      shift 2
      ;;
    --ssh-user)
      [[ $# -ge 2 ]] || { echo "ERROR: --ssh-user requires a value" >&2; exit 2; }
      SSH_USER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$SCENARIO" == "A" || "$SCENARIO" == "B" ]] || {
  echo "ERROR: scenario must be A or B" >&2
  exit 2
}
[[ -x "$POLICY_PY" ]] || { echo "ERROR: policy Python is not executable: $POLICY_PY" >&2; exit 1; }
"$POLICY_PY" -c 'import math, sys; value=float(sys.argv[1]); assert math.isfinite(value) and value > 0' "$DURATION"
if [[ "$SCENARIO" == "A" ]]; then
  if [[ "$SHORTENED_CORRECTED" == "true" ]]; then
    EXPECTED_CORRECTED="$PROJECT_ROOT/outputs/formal_train_26_corrected_v1/checkpoint-3000"
    [[ "$(realpath -e "$CHECKPOINT")" == "$(realpath -e "$EXPECTED_CORRECTED")" ]] || {
      echo "ERROR: --shortened-corrected is pinned to corrected checkpoint-3000" >&2
      exit 2
    }
    "$POLICY_PY" -c 'import sys; value=float(sys.argv[1]); assert 30.0 <= value <= 60.0' "$DURATION" || {
      echo "ERROR: shortened corrected Scenario A requires 30 <= duration <= 60 seconds" >&2
      exit 2
    }
  else
    "$POLICY_PY" -c 'import sys; assert float(sys.argv[1]) >= 120.0' "$DURATION" || {
      echo "ERROR: scenario A requires --duration >= 120 without --shortened-corrected" >&2
      exit 2
    }
  fi
fi

[[ -f "$LIVE_CONFIG" && -f "$ADAPTER_CONFIG" && -f "$RUNNER" ]] || {
  echo "ERROR: required deployment files are missing" >&2
  exit 1
}
[[ -d "$CHECKPOINT" ]] || {
  echo "ERROR: policy checkpoint is missing: $CHECKPOINT" >&2
  exit 1
}

G1_INTERFACE="${G1_INTERFACE:-$(CONFIG_VALUE g1_wired_interface)}"
G1_HOST="${G1_HOST:-$(CONFIG_VALUE g1_host)}"
UNITREE_READER_PY="$(CONFIG_VALUE unitree_reader_python)"
LOWSTATE_TOPIC="$(CONFIG_VALUE unitree_lowstate_topic)"
SSH_TARGET="$SSH_USER@$G1_HOST"
[[ "$G1_INTERFACE" =~ ^[A-Za-z0-9_.:-]+$ ]] || { echo "ERROR: invalid interface name" >&2; exit 2; }
[[ "$G1_HOST" =~ ^[A-Za-z0-9.:-]+$ ]] || { echo "ERROR: invalid G1 host" >&2; exit 2; }
[[ "$SSH_USER" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid SSH user" >&2; exit 2; }

for command in nvidia-smi ip ping ssh ss timeout awk grep find; do
  command -v "$command" >/dev/null || { echo "ERROR: missing command: $command" >&2; exit 1; }
done
[[ -x "$UNITREE_READER_PY" ]] || {
  echo "ERROR: Unitree reader Python is not executable: $UNITREE_READER_PY" >&2
  exit 1
}

PROBE_DIR="$(mktemp -d /tmp/g1_live_shadow_probe.XXXXXX)"
cleanup() {
  rm -rf -- "$PROBE_DIR"
}
trap cleanup EXIT
RUNTIME_CONFIG="$PROBE_DIR/live_shadow.yaml"
"$POLICY_PY" - "$LIVE_CONFIG" "$RUNTIME_CONFIG" "$G1_INTERFACE" "$G1_HOST" <<'PY'
import sys
import yaml

source, target, interface, host = sys.argv[1:]
config = yaml.safe_load(open(source))
config["g1_wired_interface"] = interface
config["g1_host"] = host
config["camera_endpoint"] = f"tcp://{host}:5555"
with open(target, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

echo "READ-ONLY LIVE SHADOW PREFLIGHT"
echo "No commands will be sent and no control ownership will be requested."
echo "Remote sources must already have been started manually."

export PYTHONPATH="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/g1_o6_matplotlib

"$POLICY_PY" - "$RUNTIME_CONFIG" "$ADAPTER_CONFIG" <<'PY'
import sys
import yaml

live = yaml.safe_load(open(sys.argv[1]))
adapter = yaml.safe_load(open(sys.argv[2]))
required_live = {
    "real_hardware_enabled": False,
    "publish_commands": False,
    "shadow_only": True,
    "dry_run": True,
    "command_publish_attempt_limit": 0,
    "control_ownership_request_limit": 0,
}
required_adapter = {"real_hardware_enabled": False, "dry_run": True}
for key, expected in required_live.items():
    if live.get(key) != expected:
        raise SystemExit(f"unsafe live config: {key}={live.get(key)!r}, expected {expected!r}")
for key, expected in required_adapter.items():
    if adapter.get(key) != expected:
        raise SystemExit(f"unsafe adapter config: {key}={adapter.get(key)!r}, expected {expected!r}")
if int(live.get("warmup_inferences", -1)) != 10:
    raise SystemExit("Live Shadow requires exactly 10 warm-up inferences")
if float(live.get("replanning_hz", -1)) != 10.0 or int(live.get("execution_horizon", -1)) != 3:
    raise SystemExit("initial Live Shadow gate requires 10 Hz / horizon 3")
print("Safety config: PASS")
PY

"$POLICY_PY" "$DEPLOYMENT_ROOT/audit_live_safety.py" >/dev/null
"$POLICY_PY" "$DEPLOYMENT_ROOT/audit_o6_feedback_reader.py" >/dev/null
"$POLICY_PY" - "$DEPLOYMENT_ROOT/live_safety_audit.json" "$DEPLOYMENT_ROOT/o6_feedback_reader_safety.json" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    report = json.load(open(path))
    if report.get("gate_passed") is not True:
        raise SystemExit(f"static safety gate failed: {path}")
print("Static safety audits: PASS")
PY

nvidia-smi -L
"$POLICY_PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is unavailable")
print("CUDA:", torch.cuda.get_device_name(0))
PY

ip link show dev "$G1_INTERFACE" >/dev/null
[[ -r "/sys/class/net/$G1_INTERFACE/operstate" ]] || {
  echo "ERROR: cannot read operstate for $G1_INTERFACE" >&2
  exit 1
}
[[ "$(<"/sys/class/net/$G1_INTERFACE/operstate")" == "up" ]] || {
  echo "ERROR: interface $G1_INTERFACE is not UP" >&2
  exit 1
}
ROUTE_LINE="$(ip route get "$G1_HOST")"
awk -v expected="$G1_INTERFACE" '
  { for (i=1; i<=NF; i++) if ($i == "dev" && $(i+1) == expected) found=1 }
  END { exit(found ? 0 : 1) }
' <<<"$ROUTE_LINE" || {
  echo "ERROR: route to $G1_HOST does not use $G1_INTERFACE: $ROUTE_LINE" >&2
  exit 1
}
ping -c 2 -W 1 "$G1_HOST" >/dev/null || {
  echo "ERROR: G1 is not reachable by ping: $G1_HOST" >&2
  exit 1
}
echo "G1 network via $G1_INTERFACE: PASS"

SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" hostname >/dev/null || {
  echo "ERROR: read-only SSH preflight failed for $SSH_TARGET" >&2
  exit 1
}
REMOTE_SS="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" 'ss -ltnpH')"
for port in 5555 5558; do
  awk -v port=":$port" '$4 ~ (port "$") { found=1 } END { exit(found ? 0 : 1) }' <<<"$REMOTE_SS" || {
    echo "ERROR: remote read-only source port $port is not listening" >&2
    exit 1
  }
done
REMOTE_PROCESSES="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" 'ps -eo pid=,args=')"
# The audited camera module lives under the ``gear_sonic`` package. Match the
# command-capable runtime by executable/module token so the required camera
# source is not mistaken for SONIC control.
FORBIDDEN_PROCESS_PATTERN='g1_deploy_onnx_ref|g1_deploy_onnx|(^|[ /])sonic([ /]|$)'
if grep -Eiq "$FORBIDDEN_PROCESS_PATTERN" <<<"$REMOTE_PROCESSES"; then
  echo "ERROR: a forbidden command-capable G1/SONIC process appears to be running:" >&2
  grep -Ei "$FORBIDDEN_PROCESS_PATTERN" <<<"$REMOTE_PROCESSES" >&2
  exit 1
fi
grep -Fq 'o6_feedback_only_relay.py' <<<"$REMOTE_PROCESSES" || {
  echo "ERROR: cannot verify that port 5558 is backed by o6_feedback_only_relay.py" >&2
  exit 1
}
echo "Remote 5555/5558 listeners and feedback-only O6 process: PASS"

set +e
timeout --signal=TERM 10s "$UNITREE_READER_PY" -u \
  "$DEPLOYMENT_ROOT/observation_sources/g1_lowstate_stdout.py" \
  --interface "$G1_INTERFACE" --topic "$LOWSTATE_TOPIC" \
  >"$PROBE_DIR/lowstate.stdout" 2>"$PROBE_DIR/lowstate.stderr"
PROBE_STATUS=$?
set -e
if ! grep -Fq '"schema":"g1_lowstate_readonly_v1"' "$PROBE_DIR/lowstate.stdout"; then
  echo "ERROR: no read-only $LOWSTATE_TOPIC sample received (probe status $PROBE_STATUS)" >&2
  sed -n '1,80p' "$PROBE_DIR/lowstate.stderr" >&2
  exit 1
fi
echo "DDS SUB $LOWSTATE_TOPIC on $G1_INTERFACE: PASS"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$DEPLOYMENT_ROOT/logs/live_shadow_host/$RUN_STAMP"
mkdir -p "$RUN_ROOT"

echo "Starting Null Sink Live Shadow: scenario=$SCENARIO duration=${DURATION}s"
set +e
"$POLICY_PY" "$RUNNER" \
  --config "$RUNTIME_CONFIG" \
  --adapter-config "$ADAPTER_CONFIG" \
  --scenario "$SCENARIO" \
  --duration "$DURATION" \
  --checkpoint "$CHECKPOINT" \
  --log-root "$RUN_ROOT"
RUN_STATUS=$?
set -e

SUMMARY_PATH="$(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -type f -name summary.json -print -quit)"
[[ -n "$SUMMARY_PATH" ]] || {
  echo "ERROR: runner produced no summary.json (status $RUN_STATUS)" >&2
  exit 1
}
"$POLICY_PY" - "$SUMMARY_PATH" "$RUN_STATUS" <<'PY'
import json
import sys

path = sys.argv[1]
runner_status = int(sys.argv[2])
summary = json.load(open(path))
errors = []
if runner_status != 0:
    errors.append(f"runner exit status is {runner_status}")
if summary.get("failure") is not None:
    errors.append(f"runner failure: {summary['failure']}")
if int(summary.get("counts", {}).get("inferences", 0)) <= 0:
    errors.append("no inference completed")
if int(summary.get("warmup_ms", {}).get("count", 0)) != 10:
    errors.append("warm-up count is not 10")
sink = summary.get("null_sink", {})
for key in ("command_publish_attempts", "control_ownership_requests", "real_sdk_objects_created"):
    if int(sink.get(key, -1)) != 0:
        errors.append(f"{key} is not zero")
for key, expected in (
    ("real_hardware_enabled", False),
    ("publish_commands", False),
    ("shadow_only", True),
    ("dry_run", True),
):
    if summary.get(key) is not expected:
        errors.append(f"summary {key} is not {expected}")
if errors:
    raise SystemExit("Live Shadow gate failed: " + "; ".join(errors))
print("Live Shadow gate: PASS")
print("Summary:", path)
print("Inferences:", summary["counts"]["inferences"])
print("Null Sink records:", sink.get("records", 0))
PY

echo "READ-ONLY LIVE SHADOW COMPLETE"
