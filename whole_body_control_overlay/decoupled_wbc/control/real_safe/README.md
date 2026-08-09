# G1 Standalone Real-Safe WBC

This package is the narrow pre-GR00T, pre-O6 path for bringing up Gear WBC on
a supported G1. It does not make the existing general control loop safe for
real deployment.

## State sequence

```text
READ_ONLY
  -> ARM_CONTROL (one-time token, verified fault exit required)
  -> HOLD (29DoF current-q target)
  -> ENGAGE_WBC (smooth lower-body blend, arms remain at arming q)
  -> STAND
  -> FAULT / STOPPED
```

`READ_ONLY` uses only `rt/lowstate` and `rt/secondary_imu`. It does not create a
MotionSwitcher client, LowCmd publisher, Dex3 sender, GR00T policy, or O6 command
transport. The read-only executable is:

```bash
python -m decoupled_wbc.control.main.teleop.run_g1_standalone_read_only \
  --interface <g1-wired-interface> \
  --duration 10
```

This command was not executed against a robot as part of this change.

## Implemented gates

- Real `G1Env` rejects `with_hands=True` before constructing `G1Body` or Dex3;
  `HandCommandSender` independently requires an explicit simulation environment
  before it can construct a Dex3 command publisher.
- `BodyStateProcessor` performs no MotionSwitcher construction or release in
  its constructor. Mode takeover is an explicit, confirmation-gated method.
- Real `BodyCommandSender` starts disarmed. A validated current-q message may be
  prepared before takeover, but `Write()` is impossible until the write gate is
  explicitly armed.
- Snapshot checks cover finite values, 29DoF hard position limits, measured
  velocity limits, both IMUs, DDS freshness, and quaternion norms.
- Command checks cover finite values, hard limits, lower-body per-cycle steps,
  lower-body rates, current-q HOLD drift, and the 50 Hz deadline.
- `ENGAGE_WBC` additionally requires an unbroken valid HOLD heartbeat through
  the full minimum hold interval; elapsed wall time alone is not sufficient.
- The engage transition uses a three-second smoothstep blend followed by the
  lower-body rate limiter. Motors 15-28 remain at the arming snapshot.
- FAULT first disarms the normal LowCmd write gate. Any subsequent supported or
  damped exit command belongs exclusively to the verified platform exit strategy.
- The old general control loop refuses real-hardware startup.

## Deliberate blocker

No generic standing-robot FAULT exit is implemented. Stopping the process,
zeroing torque, clearing gains, or blindly selecting a motion mode are not
accepted substitutes. `ARM_CONTROL` remains blocked unless a concrete
`FaultExitStrategy` declares that it was verified for the supported robot and
test setup.

The exit strategy must specify behavior for at least:

- control-loop deadline loss;
- stale lowstate or secondary IMU;
- DDS/network loss;
- ONNX inference failure or non-finite output;
- measured joint velocity/position violation;
- failure after `ReleaseMode()` but before the first current-q `LowCmd` write;
- operator emergency stop;
- controlled transition from low-level WBC back to the intended robot mode.

Until that strategy is implemented and tested under physical support, only the
read-only phase is eligible for real execution.
