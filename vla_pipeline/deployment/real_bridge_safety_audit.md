# Real bridge incremental safety audit

Date: 2026-08-05 (joint/takeover proof refresh)

Status: **PASS FOR STATUS C - SCHEDULER-FIXED HOLD COMPLETED**

This audit is incremental to the passed 120-second Live Shadow gate. It covers
only the newly introduced real command path.

## Gating and construction order

- Default YAML and runtime JSON both set `hardware_transport_enabled=false`,
  `command_publication_enabled=false`, and `micro_motion_armed=false`.
- The JSON runtime config is structurally identical to the audited YAML.
- CLI order is: require all three command-line gates, run read-only runtime
  preflight, verify and consume the plan-bound one-time token, then import and
  construct real transports.
- A missing gate leaves the token unconsumed and never calls the transport
  factory. This is covered by `test_real_bridge_no_transport.py`.

## G1 command boundary

- Interface: dedicated DDS `rt/arm_sdk`, message `unitree_hg.msg.dds_.LowCmd_`.
- Written joints: left arm 15-21, right arm 22-28, blend weight 29.
- Legs 0-11 and waist 12-14 serialize exactly as `mode=0, q=0, dq=0, tau=0,
  kp=0, kd=0, reserve=0`; slots 30-34 serialize identically. They therefore
  contain no active gain, feed-forward torque, or nonzero target.
- Each arm joint uses absolute radians, `kp=60`, `kd=1.5`, `dq=0`, `tau=0`,
  matching the official Unitree arm7 SDK DDS example. CRC is applied.
- No `rt/lowcmd`, `MotionSwitcherClient`, controller mode switch, leg command,
  or waist command exists in the new path.
- Arm authority is the documented `rt/arm_sdk` blend weight, not whole-robot
  ownership. Current-position targets are populated before weight rises from
  zero. Release retains the last safe arm target while weight linearly reaches
  zero over two seconds.

## O6 command boundary

- Left is `can2`, arbitration ID `0x28`; right is `can1`, ID `0x27`.
- The only position setter is the deployed SDK's nonblocking
  `LinkerHandO6Can.try_set_joint_positions` (`0x01` plus six bytes).
- Feedback uses `try_request_current_status` (`0x01` query with no target bytes).
- No speed, torque, enable, reset, home, homing, calibration, or fault-clear call
  exists. There is no TCP command listener; the host uses a persistent SSH stdio
  child after all gates pass.
- Stop closes the receive threads and CAN bus handles. It deliberately does not
  call `LinkerHandApi.close_can()`, because that helper sets the Linux CAN link
  down. It sends no zero, hold, or substitute O6 target on timeout/stop.
- Percentage mapping exactly reuses the deployed driver rule:
  `int(round(percent * 255.0 / 100.0))`, including Python round-to-even ties.

## Final bridge limiter

- Arm anchor: feedback at arming; excursion +/-0.01 rad, velocity 0.12 rad/s,
  acceleration 0.4 rad/s^2.
- O6 anchor: independent six-dimensional feedback per side; excursion +/-5
  percentage points and velocity 15 points/s. Integer range is checked again
  after mapping.
- Hold abort: any arm changes by more than 0.01 rad or any O6 dimension by more
  than 2 points.
- Micro aborts on stale/nonfinite/shape/mode/envelope/speed/direction/transport
  failures. A sixth consecutive right-O6 zero-seeking policy replan faults; the
  one-shot plan contains exactly five replans and then stops.
- G1 runs at 50 Hz, O6 at 30 Hz, and the frozen policy plan contains five 10 Hz
  replans over exactly 0.5 seconds.

## Verification

- Requested focused tests: **25 passed** after the scheduler fix.
- AST/config audit: **24/24 checks passed**.
- Mock one-shot exercised hold, micro, logging, and weight-zero release.
- Release feedback is sampled at 50 Hz during the 2 s weight ramp and for a
  further 0.5 s after weight reaches zero without additional publications;
  per-joint rebound is limited to 0.01 rad.
- Remote O6 agent local/remote SHA256:
  `1ad2ae97d38ee0b46c5d8b8b26fb1d6f1a621a1f4978ee9a37226a0ef1a23a81`.
- Combined bridge/config source manifest SHA256:
  `126ca2277fb152bd015e25e4768979d4b9618929222c4b450e539d93b97d6d59`.

## Scheduler-fixed Hold addendum (2026-08-07)

- O6 feedback now runs in a dedicated feedback-only monitor thread; it cannot
  block the 50 Hz arm scheduler.
- A true wall-clock test with a deliberately 50 ms O6 getter passed before the
  real Hold. Focused tests are now **25 passed** and the static audit still
  passes every check.
- The one authorized Hold created one `rt/arm_sdk` publisher and sent current-q
  arm targets only. Weight reached 1 in 1.004138 s, stayed at 1 for 2.008127 s,
  and reached 0 in 2.010316 s.
- O6 setters were never constructed in the Hold path. O6 position commands,
  waist/leg commands, watchdogs, and FAULTs were all zero.
- Maximum arm offset was `9.59e-5 rad`; release rebound was `8.39e-5 rad`.
  Mode stayed `mode_machine=5`, `mode_pr=0`; can1/can2 stayed ERROR-ACTIVE.

The Hold gate is passed. This audit does not authorize or claim execution of a
GR00T model micro-motion or any O6 position command.

## Model-micro decoupling addendum

- Focused tests: **28 passed**; static audit includes explicit checks that the
  arm loop has no O6 transport I/O and release does not require O6 feedback.
- A single O6 worker owns getter/setter calls and consumes a latest-only target;
  right O6 has no command path.
- 20/50/100 ms setter blocking and timeout passed 24.161 seconds of wall-clock
  mocks. The 100 ms case kept arm p99/max at 20.108/20.368 ms.
- The timeout case produced a watchdog, then completed the exact 100-message
  release to weight zero with no release error.
- The new corrected-checkpoint plan is bounded to 0.010 rad per arm joint and
  5.0 points per left O6 dimension. It has no token and sent no real command.

Software preflight is ready for a separately confirmed 0.5-second model micro
run; this audit itself does not authorize or execute it.
