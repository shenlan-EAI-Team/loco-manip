# Scheduler-fixed real Hold-only report

Status: **PASS - CURRENT-POSITION HOLD COMPLETE - MODEL MICRO NOT EXECUTED**

The prior Hold remained `FAIL_TIMING` because synchronous O6 feedback limited
the arm loop to about 22 Hz. O6 feedback is now owned by a feedback-only monitor
thread; the arm scheduler uses absolute wall-clock deadlines and never catches
up with burst publications. A new Hold-only was run once using plan SHA256
`05c20259e968a5d26d3bd4e598fd16a1581c3360349cc82568eb22b68d7bc032`.

## Runtime result

- G1 mode stayed `mode_machine=5`, `mode_pr=0`; outbound messages matched it.
- q target was fresh arming `rt/lowstate` feedback, not the plan snapshot.
- Weight reached exactly 1 after 1.004138 s, remained at 1 for 2.008127 s,
  and reached exactly 0 after a 2.010316 s release.
- 251 arm messages: 51 activation, 100 additional full-weight, 100 release.
- Publish interval: mean 20.082 ms, p99 20.347 ms, min 20.006 ms,
  max 20.420 ms. No deadline crossed the 40 ms FAULT limit.
- Post-release: 25 read-only feedback samples over 0.5 s, no publications.
- O6 position commands: 0 for both hands. Waist/leg commands: 0.
- Runtime FAULT/watchdog events: 0. Release error: none.
- All non-arm slots 0-14 and 30-34 serialized explicit zeros in every message.
- can1/can2 remained 1 Mbps `ERROR-ACTIVE`, tx/rx errors zero. The dedicated
  feedback-only relay was stopped by exact tmux session/PID; port 5558 is clear.

## Per-joint evidence

All values are radians. The command delta is zero for every joint, so a motion
direction sign is not evaluable; no joint exhibited an erroneous response.

|side|motor|joint|initial q|command q|max feedback offset|takeover max|release rebound|error|
|---|---:|---|---:|---:|---:|---:|---:|---|
|L|15|shoulder_pitch|-0.017149426|-0.017149426|0.000047937|0.000047937|0.000023969|no|
|L|16|shoulder_roll|0.101973772|0.101973772|0.000035949|0.000023969|0.000035949|no|
|L|17|shoulder_yaw|-0.063504413|-0.063504413|0.000035957|0.000035957|0.000035957|no|
|L|18|elbow|1.383183360|1.383183360|0.000023961|0.000023961|0.000023961|no|
|L|19|wrist_roll|0.008628642|0.008628642|0.000047937|0.000047937|0.000083890|no|
|L|20|wrist_pitch|0.309456676|0.309456676|0.000035971|0.000023991|0.000035971|no|
|L|21|wrist_yaw|0.121004723|0.121004723|0.000047937|0.000035949|0.000047937|no|
|R|22|shoulder_pitch|-0.095897771|-0.095897771|0.000011988|0.000011988|0.000023969|no|
|R|23|shoulder_roll|-0.050932959|-0.050932959|0.000095874|0.000059921|0.000071906|no|
|R|24|shoulder_yaw|0.085603319|0.085603319|0.000035949|0.000035949|0.000035949|no|
|R|25|elbow|1.511306643|1.511306643|0.000036001|0.000023961|0.000035882|no|
|R|26|wrist_roll|-0.056984991|-0.056984991|0.000023969|0.000023969|0.000023969|no|
|R|27|wrist_pitch|0.019258650|0.019258650|0.000035953|0.000035953|0.000035953|no|
|R|28|wrist_yaw|-0.075752288|-0.075752288|0.000035957|0.000035949|0.000035957|no|

Maximum waist feedback change was `4.56e-5 rad`; the waist was never commanded.
The release/post-release maxima were `8.39e-5` and `7.19e-5 rad`. A subsequent
two-second pure DDS probe stayed in mode `5/0`, with arm span below `8.39e-5`
and waist span below `4.02e-5 rad`.

Evidence:

- `deployment/logs/real_hold/20260807_161300_scheduler_fixed/hold.jsonl`
- `deployment/logs/real_hold/20260807_161300_scheduler_fixed/summary.json`
- `deployment/logs/real_hold/20260807_161300_scheduler_fixed/remote_after_cleanup.txt`

This establishes result **C**: Hold-only passed; model micro-motion was not run.
