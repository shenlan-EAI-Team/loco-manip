# G1 WBC single-wrist actuation proof outcome

Result: **FAULT before wrist reference; actuation proof not established.**

The one-shot run released `ai`, established the 500 Hz current-q HOLD, completed
the two-second HOLD and three-second Gear WBC engage, and reached STAND. About
1.12 seconds into STAND, feedback for motor 16 (`left_shoulder_roll_joint`)
crossed the frozen-arm `0.01 rad` envelope. The guard froze the last valid
command and stopped all further reference changes.

The selected motor 21 (`left_wrist_yaw_joint`) proof never started. Its arming
position was `0.031878039 rad`; the planned target was `0.041878039 rad`, but
the command delta remained zero and `proof_trace` is empty. Its largest
pre-proof feedback fluctuation was only `8.39e-05 rad`.

The last persisted valid motor-16 delta was `0.009994842 rad`; the next sample
raised before being appended to the trace. This is a narrow threshold crossing,
not evidence of a large commanded arm motion. No NaN, stale-state, DDS, ONNX
limit, or lower-body hard-limit fault was recorded.

The runtime made one `ReleaseMode` call, 3,083 LowCmd writes, zero `SelectMode`
calls, sent no zero command, and made no automatic retry. The local writer
remained alive with the last valid command until the operator confirmed the
physical emergency stop. Only then was exact PID `277006` terminated.

Post-stop checks found zero `rt/lowcmd` publications, an empty owner, and
150/150 valid read-only snapshots at 50.017 Hz. GR00T, O6 command,
`rt/arm_sdk`, and SONIC were not started.
