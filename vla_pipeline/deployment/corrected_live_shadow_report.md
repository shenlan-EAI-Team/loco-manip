# Corrected checkpoint shortened Live Shadow

Status: **PASS - READ-ONLY / NULL SINK**

- Checkpoint: `outputs/formal_train_26_corrected_v1/checkpoint-3000`
- Scenario A duration: 30 seconds
- Warm-up / inference / Null Sink records: 10 / 300 / 3000
- Effective replanning: 10 Hz; one deadline miss (0.333%), no skipped scheduler periods
- Synchronized observations accepted: 657; cross-modal skew rejections: 84
- Failure, NaN, stale decoder errors: none
- Command publication, ownership requests, real SDK objects: 0 / 0 / 0
- Right O6 command count: 0

## Arm filter comparison

| arm | old velocity | corrected velocity | old acceleration | corrected acceleration |
|---|---:|---:|---:|---:|
| left | 87.81% | 48.97% | 94.15% | 88.87% |
| right | 91.42% | 19.54% | 94.10% | 78.54% |

The corrected raw first-target minus feedback mean absolute errors were
`0.0663 rad` left and `0.0582 rad` right. Successive first-target drift means
were `0.0214 rad` left and `0.0107 rad` right.

## Left O6 guard

Raw adjacent scalar jumps above 30 points occurred at 0.2595%; raw chunk
boundary jumps above 30 points occurred at 0.6689%, with a 58.20 point maximum.
The final guarded output had zero jumps above 30 points and a maximum adjacent
change of 0.5 point. Eighteen scalar spikes were rejected.

## Plans

- Hold-only: `deployment/logs/real_bridge_preflight_corrected/20260807_150000/hold_only_plan.json`
- 0.5-second preview: `deployment/logs/real_bridge_preflight_corrected/20260807_150000/model_micro_motion_plan.json`
- Confirmation token: not issued
- Right O6: feedback-only in both plans, command count zero

The Hold-only plan was subsequently tightened so that **both** O6 hands are
feedback-only. Its SHA256 is
`0925c754b928c28123b9cc95b5b3f0b2a725d17a31771bf01f404c2d278fa88e`.
Its G1 messages carry the stable live mode fields `mode_machine=5` and
`mode_pr=0`; Hold execution re-reads and locks those fields before publishing.

No real command transport was constructed or invoked.
