# First real micro-motion preflight

Date: 2026-08-05 18:45 +08:00

Status: **B - READY, WAITING FOR ONE EXPLICIT USER CONFIRMATION**

## Read-only live snapshot

After the G1 restart, a new read-only run produced 10 synchronized 10 Hz policy
inferences and 100 Null Sink records. Its command, ownership, and real SDK
counters remained zero. A separate synchronized snapshot recorded:

- G1 mode: `mode_machine=5`, `mode_pr=0`
- left arm rad: `[-0.027743, 0.142205, -0.612646, 1.463250, -0.749266, 0.042532, 0.035162]`
- right arm rad: `[0.060197, -0.051233, -0.119411, 1.383962, -0.084465, 0.047493, -0.104383]`
- left O6 percent: `[99.6078, 100, 99.6078, 99.6078, 99.6078, 99.6078]`
- right O6 percent: `[99.6078, 100, 99.6078, 99.6078, 99.6078, 99.6078]`
- waist rad, monitored only: `[-0.241335, -0.000720, -0.000179]`

Snapshot: `deployment/logs/real_bridge_preflight/20260805_183612/current_feedback.json`

## Frozen command preview

Plan SHA256: `c909fc15b7ce3c7699b18b9e066c271cf7d403696c8a7022412628279f05e814`.

- Current-position hold is at the snapshot itself for at most 2 seconds.
- Hold O6 bytes are left `28#01 FE FF FE FE FE FE` and right
  `27#01 FE FF FE FE FE FE`.
- The model preview contains five policy replans, 15 control frames, 25 G1
  arm-SDK ticks, and 15 O6 ticks over 0.5 seconds.
- Left and right arm maximum preview offset is 0.010000 rad.
- Left and right O6 maximum preview offset is 5.0000 points.
- Final preview O6 bytes are left `28#01 F1 F2 F1 F1 F1 F1` and right
  `27#01 F1 F2 F1 F1 F1 F1`; observed command values stay in raw 241-255.
- Right O6 raw policy output remains zero, but the bridge output never leaves
  the arming envelope and therefore never approaches full travel.
- Legs, waist, and slots 30-34 serialize as explicit all-zero, zero-gain motor
  fields. Their feedback/mode is monitored for anomaly detection.

Full per-tick messages:
`deployment/logs/real_bridge_preflight/20260805_183612/micro_motion_plan.json`.

## Current remote state

- `can1` and `can2`: UP, 1 Mbps, restart-ms 100, qlen 1000, ERROR-ACTIVE,
  tx/rx error counters zero.
- Ports 5555, 5557, 5558, and 5561 are unused.
- No camera, feedback relay, O6 command agent, SONIC, policy deploy, or glove
  command process remains from this preflight.
- The audited remote agent file exists, its hash matches, and it is not running.

## Actual preflight commands

Read-only source start used the previously audited commands:

```sh
tmux new-session -d -s rbpf_camera_20260805_183612 "bash -lc 'echo $$ > /home/unitree/real_bridge_preflight_20260805_183612/camera.pid; cd /home/unitree/GR00T-WholeBodyControl; exec /home/unitree/GR00T-WholeBodyControl/.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --ego-view-device-id 342522073568 --fps 30 --port 5555 >> /home/unitree/real_bridge_preflight_20260805_183612/camera.log 2>&1'"
tmux new-session -d -s rbpf_o6_20260805_183612 "bash -lc 'echo $$ > /home/unitree/real_bridge_preflight_20260805_183612/o6.pid; export PYTHONPATH=/home/unitree/linker_hand_python_sdk; exec /usr/bin/python3 /home/unitree/real_bridge_preflight_20260805_183612/o6_feedback_only_relay.py --left-can can2 --right-can can1 --left-sn LHO6-03-1097-L-Z-1-E --right-sn LHO6-03-1093-R-Z-1-E --state-port 5558 --rate 20 >> /home/unitree/real_bridge_preflight_20260805_183612/o6.log 2>&1'"
deployment/run_live_shadow_host.sh --scenario B --duration 1 --interface enp4s0
```

They were stopped only by exact tmux names/PIDs. PID 4185 required exact
`kill -TERM 4185`; no broad kill was used. The remote agent was copied but not
executed.

## Execution remains blocked

The previously displayed command and token must not be used. That token is
bound to the superseded plan hash and runtime verification will reject it. No
replacement token or executable command has been issued. A fresh feedback
check, plan-bound one-time token, and exact command will be produced only after
explicit user confirmation and the required onsite safety confirmation.

## Required onsite confirmation

Before the command may run, the user must explicitly confirm all of the
following: they are beside the robot; physical E-stop is in hand; the G1 is on
a gantry or reliable support; both arms/hands are clear of people, objects, and
cables; cylinder and box are removed; both O6 hands are empty; and this is a
single non-repeating hold plus micro test.
