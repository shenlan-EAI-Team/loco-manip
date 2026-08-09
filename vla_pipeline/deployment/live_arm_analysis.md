# Live arm and O6 analysis

Status: **MEASURED - ANALYSIS ONLY**

The robot feedback was effectively stationary. Mean absolute feedback motion
per selected frame was `8.56e-6 rad` left and `8.90e-6 rad` right.

| group | first target - feedback abs mean | P95 | max | successive first-target drift mean | drift P95 |
|---|---:|---:|---:|---:|---:|
| left arm | 0.338 rad | 1.201 | 1.275 | 0.222 rad | 0.832 |
| right arm | 0.216 rad | 0.781 | 0.997 | 0.246 rad | 0.840 |

Largest left-arm offset was wrist pitch (1.138 rad mean); largest left target
drift was shoulder yaw (0.560 rad mean). Largest right offsets were shoulder
roll (0.427 rad) and wrist pitch (0.379 rad); their target drift means were
0.531 and 0.414 rad.

Across 3,600 filtered control steps, right-arm per-joint velocity/acceleration
trigger rates were:

| joint | velocity | acceleration |
|---|---:|---:|
| shoulder pitch | 93.14% | 95.03% |
| shoulder roll | 97.89% | 93.47% |
| shoulder yaw | 90.22% | 94.61% |
| elbow | 83.67% | 95.64% |
| wrist roll | 89.50% | 95.47% |
| wrist pitch | 95.28% | 90.89% |
| wrist yaw | 90.28% | 93.61% |

Right-arm aggregate velocity and acceleration trigger rates were 91.42% and
94.10%; position and nonfinite triggers were zero. The cause is model chunk
jump, not real feedback change: model first targets changed by 0.246 rad mean
while feedback changed by only 8.90e-6 rad mean. The stateful dry-run filter is
anchored to feedback only on reset and then follows its own filtered output.

Left O6 feedback stayed near `[99.61,100,99.61,99.61,99.61,99.61]`, while raw
first targets spanned 0..100 and changed by 25.45 points mean between replans.
This is an unstable hypothetical open/close trend, not a deployable hand plan.
Right O6 raw output remained exactly all-zero for every inference, confirming
training-data degeneration. Its approximately 100-point mismatch to the real
right-hand feedback is unsafe for any real bridge.
