# Real bridge scheduler validation

Status: **PASS**

The arm and O6 feedback paths are decoupled. One feedback-only thread owns the
blocking O6 getter and maintains a timestamped cache. The 50 Hz arm thread never
calls the getter. It uses absolute monotonic deadlines, refuses cumulative
lateness at 40 ms, and schedules each publication no earlier than 20 ms after
the preceding publication, preventing catch-up bursts.

## Offline wall-clock validation

A mock O6 getter was deliberately delayed by 50 ms per call while the complete
1.0 s activation, 2.0 s full-weight hold, 2.0 s release, and 0.5 s read-only
monitor ran against mock transports.

- Arm messages: 251; O6 position commands: 0; waist/leg commands: 0.
- Arm interval mean/p99/max: 20.055 / 20.111 / 20.258 ms.
- Activation/full-weight/release: 1.002647 / 2.005498 / 2.005653 s.
- Deadline misses over 20 ms: 0; watchdogs: 0; release error: none.
- 25 post-release samples and no post-release publication.

Evidence:
`deployment/logs/offline_scheduler/20260807_161218_no_burst/summary.json`.

## Real Hold validation

- Arm interval mean/p99/max: 20.082 / 20.347 / 20.420 ms.
- Activation/full-weight/release: 1.004138 / 2.008127 / 2.010316 s.
- Weight reached exactly 1 and exactly 0; no interval was shorter than
  20.006 ms.
- O6 cached feedback age never exceeded 50.795 ms against a 200 ms stale gate.
- No FAULT, mode change, SDK exception, CAN error, O6 command, or waist/leg
  command occurred.

The scheduler defect from the first Hold is resolved for Hold-only operation.
The O6 position-command path has not been exercised and model micro-motion
remains a separate, not-yet-executed step.

## Model-micro scheduler addendum

The remaining O6 command-path coupling is resolved. `execute_micro` now queues
only the latest enveloped target; a single independent O6 I/O worker owns the
left setter and both feedback getters. Right O6 remains feedback-only. G1 arm
release no longer requires a healthy or fresh O6 cache.

A 24.161-second wall-clock mock covered 20/50/100 ms setter blocking and a
timeout. Under 100 ms blocking, arm p99/max was 20.108/20.368 ms with no interval
below 20.014 ms. The timeout case still completed all 100 release messages,
reached weight zero, and completed the post-release monitor with no release
error. See `deployment/logs/micro_scheduler/20260807_162902/summary.json`.
