# G1 standalone Decoupled WBC commissioning result

Overall result: **FAIL at recovery**. Do not repeat control yet.

The one-shot commissioning passed its preflight, released the `ai` owner, wrote
the current-q HOLD, completed the two-second HOLD, three-second smooth Gear WBC
engage, and five-second standalone STAND window. No earlier safety fault was
raised for stale state, IMU, ONNX output, mailbox, hard limits, frozen arms, or
the 500 Hz writer.

The planned recovery then called `SelectMode("ai")` exactly once. Unitree
returned status `7002`, whose SDK definition is `switcher is busy`. The guard
did not retry, exit, clear gains, or send zero. It continued its last valid
LowCmd at 500 Hz until the operator confirmed the physical emergency stop.
Only then was the exact guard process terminated.

Post-stop read-only checks show no `rt/lowcmd` publication, an empty motion
owner, 100/100 valid lowstate/secondary-IMU snapshots at 50.02 Hz, and a stable
robot state. Do not interpret the empty owner as recovery to `ai`.

The fault-HOLD probe measured 500.15 Hz, 2.426 ms p99 and 2.675 ms maximum
interval, with zero command/state CRC errors. The largest lower-body change
from preflight to post-estop was right knee motor 9 at +0.17294 rad. Right ankle
motor 10 changed -0.11416 rad. The largest arm change was motor 22 at -0.00882
rad, below the 0.01 rad frozen-arm threshold.

The standalone WBC actuation path is therefore demonstrated, but the complete
commissioning lifecycle is not passed. Before another control run, correct and
validate the `SelectMode("ai")` busy/recovery timing. GR00T, O6 command, SONIC,
and `rt/arm_sdk` were not started.
