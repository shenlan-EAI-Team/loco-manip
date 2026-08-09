# G1 arm_sdk joint mapping and takeover proof

Date: 2026-08-05

Status: **proof complete; no real command executed; explicit user confirmation still required**

## 1. Policy/Adapter to DDS mapping

The Policy API and Adapter use the same seven-element order on both sides. The
command and feedback indices are deliberately identical.

|Side|Policy/Adapter index|Joint|`rt/arm_sdk motor_cmd`|`rt/lowstate motor_state`|
|---|---:|---|---:|---:|
|Left|0|shoulder_pitch|15|15|
|Left|1|shoulder_roll|16|16|
|Left|2|shoulder_yaw|17|17|
|Left|3|elbow|18|18|
|Left|4|wrist_roll|19|19|
|Left|5|wrist_pitch|20|20|
|Left|6|wrist_yaw|21|21|
|Right|0|shoulder_pitch|22|22|
|Right|1|shoulder_roll|23|23|
|Right|2|shoulder_yaw|24|24|
|Right|3|elbow|25|25|
|Right|4|wrist_roll|26|26|
|Right|5|wrist_pitch|27|27|
|Right|6|wrist_yaw|28|28|

Evidence: `deployment/config/adapter.yaml` defines the Policy/Adapter order;
`deployment/real_bridge/message_preview.py` and `real_g1.py` use indices 15-28;
`deployment/observation_sources/g1_lowstate_stdout.py` reads the same indices.

## 2. Comparison with Unitree G1 29DoF

The checked official Unitree arm7 example defines the following physical
layout: left leg 0-5, right leg 6-11, waist 12-14, left arm 15-21, right arm
22-28. It defines `kNotUsedJoint = 29` specifically as the arm-SDK blend
weight. Thus all 14 bridge mappings match the official definition one for one.
Slots 30-34 exist in the 35-slot HG `LowCmd_` wire message but are not physical
29DoF joints.

Official source:
`/home/slxy/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py`.

## 3. Exact serialized HG LowCmd values

The message has `mode_pr=0`, `mode_machine=0`, `reserve=[0,0,0,0]`, and 35
motor slots. Incoming feedback at the frozen preview was `mode_pr=0`,
`mode_machine=5`; writing outbound `mode_machine=0` is the factory default used
by the official high-level arm example and does not call a mode-switch client.

Every arm slot 15-28 serializes:

`mode=0, q=<absolute radians>, dq=0.0, kp=60.0, kd=1.5, tau=0.0, reserve=0`.

The current-position hold preview uses these exact `q` values:

|Slot|Joint|Hold q rad|
|---:|---|---:|
|15|left shoulder_pitch|-0.027743481|
|16|left shoulder_roll|0.142204821|
|17|left shoulder_yaw|-0.612645566|
|18|left elbow|1.463249922|
|19|left wrist_roll|-0.749265730|
|20|left wrist_pitch|0.042532016|
|21|left wrist_yaw|0.035161715|
|22|right shoulder_pitch|0.060196765|
|23|right shoulder_roll|-0.051232561|
|24|right shoulder_yaw|-0.119410820|
|25|right elbow|1.383962274|
|26|right wrist_roll|-0.084464818|
|27|right wrist_pitch|0.047493484|
|28|right wrist_yaw|-0.104382604|

Slot 29 serializes
`mode=0, q=<weight>, dq=0.0, kp=0.0, kd=0.0, tau=0.0, reserve=0`.

Each of slots 0-14 and 30-34 serializes the following numeric values, rather
than an unspecified/uninitialized value:

`mode=0, q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0, reserve=0`.

This applies individually to 0 left_hip_pitch, 1 left_hip_roll, 2
left_hip_yaw, 3 left_knee, 4 left_ankle_pitch, 5 left_ankle_roll, 6
right_hip_pitch, 7 right_hip_roll, 8 right_hip_yaw, 9 right_knee, 10
right_ankle_pitch, 11 right_ankle_roll, 12 waist_yaw, 13 waist_roll, 14
waist_pitch, and reserved slots 30, 31, 32, 33, 34.

The Unitree wire packing order is `mode,q,dq,tau,kp,kd,reserve` even though the
logical values above are listed in the requested order. CRC is the Unitree
32-bit polynomial `0x04C11DB7` over the packed HG LowCmd except its final CRC
word. Frozen-preview CRC values are:

- hold at weight 0: `3235253713` (`0xC0D60DD1`)
- hold at weight 1: `4163130563` (`0xF82454C3`)
- first 0.5-second model tick: `2524601241` (`0x967A5B99`)
- last model tick: `577147650` (`0x22669302`)

All 35 slots and each model-tick CRC are retained in
`deployment/logs/real_bridge_preflight/20260805_183612/micro_motion_plan.json`
(SHA256 `c909fc15b7ce3c7699b18b9e066c271cf7d403696c8a7022412628279f05e814`).
At execution, the bridge re-reads the arming feedback; therefore the actual
floating-point `q` bits and CRC must be logged again and may differ from this
snapshot if the robot posture changes. No field is omitted from that log.

## 4. Exact arm_sdk weight curve

- Activation: 50 Hz for 1.00 s, 50 intervals. Tick 0 is `0.00`; ticks 1-50
  add exactly `+0.02` per tick and tick 50 is `1.00`.
- Current-position hold: 100 total arm messages over 2.00 s. Weight is 1.00
  for 50 messages including the message that first reaches 1.00.
- Model window: 25 arm messages over at most 0.50 s, all at weight 1.00.
- Release: 100 messages over a 2.00 s scheduler window, values `0.99, 0.98,
  ..., 0.01, 0.00`, an exact `-0.01` per 20 ms tick from a full-weight run.
  An early fault releases from the actual current weight in 100 equal steps.
- Post-release: after the single weight=0 message, 25 feedback reads over
  0.50 s at 50 Hz with **zero additional command publications**.

The exact arrays are stored under `hold_preview.arm_sdk_weight_curve` in the
machine-readable plan and are asserted against MockG1Transport records.

## 5. Handback and rebound threshold

As weight approaches zero, the original motion controller may progressively
reclaim both arms. The bridge keeps the last safe absolute arm target fixed
during the 2 s ramp and samples `rt/lowstate` after every release message. It
then samples for another 0.5 s after weight reaches zero without publishing.

For every left/right arm joint, the maximum allowed feedback rebound relative
to release-start feedback is `0.01 rad`. A larger rebound records a watchdog
failure. Release continues to weight zero rather than sending zero posture or
trying to seize authority again. The bridge does not request whole-body
ownership; every waist/leg slot remains at the explicitly listed all-zero,
zero-gain serialized value.

## 6. Required first-run per-joint evidence

Every hold and model feedback sample logs one `joint_response` record. Each arm
joint record contains Policy-side index/name, `motor_cmd_index`,
`motor_state_index`, `command_delta`, `feedback_delta`, `sign_consistent`, and
`sign_evaluable`. Each side also logs the maximum-command joint, the
maximum-response joint, and `max_response_matches_max_command`. O6 dimensions
use the same delta/sign fields without G1 motor indices. Release and post-zero
monitor records apply the same logging to all 14 arm joints.

`sign_consistent` is `command_delta * feedback_delta >= 0`; the separate
`sign_evaluable` flag prevents sensor noise or a near-zero command from being
reported as meaningful direction evidence.

## 7. First model-window displacement proof

The regenerated 25-tick preview is limited immediately before transport. Its
maximum absolute offset from arming feedback is 0.010000 rad on both arms
(about 0.57 degrees) and 5 percentage points on each O6 dimension. Arm velocity
is at most 0.12 rad/s and acceleration at most 0.4 rad/s^2. The window is at
most 0.5 s and cannot repeat automatically.

These are risk-reduction limits, not a guarantee that real hardware has zero
risk. A fresh feedback check, physical support, clear workspace, and an E-stop
in hand remain mandatory before execution. No publisher, SDK command object,
O6 setter, or real command was created or sent while producing this proof.
