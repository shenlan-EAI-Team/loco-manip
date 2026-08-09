# First real arms-only model micro-motion

Status: **SAFE COMPLETION - NO MEASURABLE ARM RESPONSE**

One authorized arms-only run was executed from corrected checkpoint plan SHA256
`6010d7293ab6805ab8f2e7ded813d69859cd4a2190f2ac1f37ffdb6c935190ee`.
The current-position Hold passed, exactly one 0.5-second/25-message model window
ran, and weight was released to exactly zero. There was no automatic repeat.

The command path produced nonzero arm targets, but the largest observed arm
feedback offset was only `4.793704e-05 rad`, below the existing `1e-4 rad`
direction-evaluation threshold. The run therefore proves safe publication and
release, but does not yet prove a controlled physical arm response. Do not
advance to left-O6 or combined motion before this is understood.

## Safety and timing

- Final state: `STOPPED`; release error: null.
- G1 mode remained `(mode_machine=5, mode_pr=0)`.
- Arm command envelope: at most `0.010000 rad` from the live arming feedback.
- Maximum release rebound: `8.392334e-05 rad`.
- Maximum post-release rebound: `7.188320e-05 rad`.
- Maximum waist feedback change: `4.017353e-05 rad`; waist/leg command count: 0.
- Left O6 position commands: 0; right O6 position commands: 0.
- FAULT: 0; watchdog: 0; DDS errors: 0; CAN errors: 0.
- Model-window arm interval mean/p99/max:
  `20.080 / 20.257 / 20.282 ms`.
- Release interval mean/p99/max: `20.085 / 20.335 / 20.342 ms`.
- Weight messages: activation 51 (`0 -> 1`), full hold 100 (`1`), model 25
  (`1`), release 100 (`0.99 -> 0`), then 25 read-only samples.

The feedback-only helper logged one timeout after release was complete while
its local subprocess was being closed. It had `arm_release_required=false`,
did not cause FAULT/watchdog, and the final release error remained null. This is
classified as a shutdown logging artifact, not an in-motion transport fault.

## Per-joint result

`raw` is the last absolute policy target; `final` is the last command after the
startup envelope. Direction is `N/E` because every feedback response remained
below `1e-4 rad` and is therefore not safely distinguishable from quantization
or stationary noise.

|Side|Joint|cmd/state index|initial q|raw|final q|command delta|max feedback delta|direction|release rebound|
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
|L|shoulder_pitch|15/15|0.049231|-0.155443|0.039231|-0.010000|-2.396852e-05|N/E|2.396852e-05|
|L|shoulder_roll|16/16|0.128651|0.244359|0.131852|0.003201|-2.397597e-05|N/E|2.397597e-05|
|L|shoulder_yaw|17/17|-0.042880|-0.039092|-0.032880|0.010000|1.198426e-05|N/E|2.396852e-05|
|L|elbow|18/18|1.381829|1.320006|1.371829|-0.010000|-1.204014e-05|N/E|1.204014e-05|
|L|wrist_roll|19/19|0.008437|-0.082315|-0.001563|-0.010000|-2.396852e-05|N/E|3.595185e-05|
|L|wrist_pitch|20/20|0.260933|0.289461|0.270933|0.010000|2.396107e-05|N/E|2.396107e-05|
|L|wrist_yaw|21/21|0.121017|0.134684|0.131017|0.010000|2.396852e-05|N/E|2.396852e-05|
|R|shoulder_pitch|22/22|0.041873|-0.207962|0.031873|-0.010000|2.396852e-05|N/E|2.396852e-05|
|R|shoulder_roll|23/23|-0.080282|-0.173097|-0.083002|-0.002720|-3.595650e-05|N/E|3.595650e-05|
|R|shoulder_yaw|24/24|0.085651|0.062072|0.075651|-0.010000|4.793704e-05|N/E|3.595650e-05|
|R|elbow|25/25|1.513248|1.444002|1.503248|-0.010000|4.792213e-05|N/E|8.392334e-05|
|R|wrist_roll|26/26|-0.057261|-0.021697|-0.047261|0.010000|-3.595278e-05|N/E|3.595278e-05|
|R|wrist_pitch|27/27|0.024652|0.033120|0.032172|0.007520|-4.793704e-05|N/E|4.793704e-05|
|R|wrist_yaw|28/28|-0.076340|-0.093648|-0.082340|-0.006000|2.396852e-05|N/E|3.594905e-05|

For the left arm, the largest response was shoulder roll, while shoulder roll
was not one of the joints reaching the maximum `0.01 rad` command envelope.
For the right arm, the largest response was shoulder yaw, which was one of five
joints tied at the maximum command magnitude. Since both responses are below
the direction threshold, neither observation proves command/feedback direction.

## Evidence

- Full JSONL: `deployment/logs/real_micro/20260807_165900_arms_only/arms_only.jsonl`
- Machine-readable summary:
  `deployment/logs/real_micro/20260807_165900_arms_only/summary.json`
- Remote O6 feedback-only log:
  `deployment/logs/real_micro/20260807_165900_arms_only/o6_feedback_only_remote.log`
- The one-time token was consumed and cannot be reused.
- The exact feedback-only PID/session were stopped; port 5558 is clear.

