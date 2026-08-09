# G1 READ_ONLY Commissioning - 2026-08-08

## Scope and verdict

The real G1 READ_ONLY layer passed. The process stayed in `READ_ONLY` and did
not construct a LowCmd publisher, Dex3 sender, GR00T/O6 transport, or
MotionSwitcher client. No command write was attempted.

`ARM_CONTROL` remains blocked. This commissioning did not call `ReleaseMode`,
`SelectMode`, `rt/lowcmd`, or `rt/arm_sdk`.

## Environment

- Host interface: `enp4s0`, `192.168.123.222/24`
- G1 address: `192.168.123.164`
- Route: direct through `enp4s0`
- Ping: 3/3 replies, 0% loss, 0.376 ms mean RTT
- SSH host: `unitree-g1-nx`

No SONIC, user WBC deployment, LowCmd process, or command bridge was running in
the inspected G1 NX process, tmux, container, and service lists.

## Real DDS results

The main run lasted 15 seconds and produced 750 accepted synchronized snapshots
with zero missing snapshots.

| Metric | Result |
| --- | ---: |
| Loop mean / p99 / max | 20.000 / 20.063 / 20.104 ms |
| Observed lowstate rate | 50.005 Hz |
| Observed secondary IMU rate | 50.005 Hz |
| lowstate age p99 / max | 0.131 / 0.201 ms |
| secondary IMU age p99 / max | 0.131 / 0.201 ms |
| Maximum 29DoF q peak-to-peak | 0.00010786 rad (motor 21) |
| Maximum measured abs(dq) | 0.05657 rad/s (motor 8) |
| Pelvis quaternion norm range | 0.99999993 to 1.00000010 |
| Secondary quaternion norm range | 0.99999995 to 1.00000013 |
| READ_ONLY watchdog result | PASS |

All 29 q/dq values were finite and inside the configured real-safe limits.
The complete 29-element q/dq telemetry is in
`g1_read_only_commissioning_20260808.json`.

After the mode query, a second five-second run produced 250/250 snapshots, zero
missing samples, a 20.096 ms maximum loop interval, and the same PASS result.
Its raw report is `g1_read_only_post_mode_query_20260808.json`.

These observed source rates are capped by the 50 Hz polling loop. They prove a
fresh sample is available for every 50 Hz WBC tick, not the firmware's maximum
DDS publication rate.

## Mode and ownership

Two explicit mode queries called only MotionSwitcher API 1001 (`CheckMode`).
Both returned:

```json
{"form": "0", "name": "ai"}
```

Both calls returned status 0. API 1002 (`SelectMode`) and API 1003
(`ReleaseMode`) call counts were zero. Therefore the current owner before and
after commissioning was the Unitree `ai` motion mode. The observed
`mode_machine` value was 5; the official G1 low-level example labels this field
as the G1 type and copies it into LowCmd. It is not ownership evidence.

The READ_ONLY runner's zero command counts are supported by both the runtime
summary and its import graph: it constructs only `rt/lowstate` and
`rt/secondary_imu` subscribers. The separate API-1001 queries create RPC
request clients, but no LowCmd publisher and no ownership-changing request.

## Remaining ARM_CONTROL blockers

1. The official G1 low-level examples show `ReleaseMode` followed by continuous
   LowCmd, but do not define or verify a no-jump handoff back to `ai`.
2. The official example writes LowCmd every 2 ms. The project currently defines
   a 50 Hz WBC target loop. The robot-side LowCmd keepalive rate and timeout must
   be measured and the transport scheduler decoupled from the 50 Hz WBC target
   scheduler before takeover.
3. A PC-side watchdog cannot recover ownership after the PC-to-G1 Ethernet link
   fails. The command gate and recovery supervisor must run on the G1 side (or
   an equivalently independent controller) for that fault to be recoverable.
4. The interval after successful `ReleaseMode` but before the first validated
   current-q LowCmd remains unverified.

## Minimal verifiable recovery design

Use one G1-resident `lowcmd_guard` as the only LowCmd publisher and the only
MotionSwitcher client. It starts read-only, records the original mode tuple,
and cannot release `ai` until a current-q command is prepared, a local watchdog
is active, and the physical-support confirmation is consumed.

The guard accepts validated 50 Hz WBC targets but republishes/interpolates the
last accepted target at the empirically verified LowCmd keepalive rate. A PC
heartbeat timeout disables new targets locally and invokes the recovery state
machine without depending on the failed Ethernet link.

The exact handoff sequence between the final safe LowCmd and
`SelectMode("ai")` is not yet accepted. Determine it under a harness by logging
q/dq/IMU and mode ownership while injecting: pre-first-write failure, WBC NaN,
50 Hz deadline loss, PC heartbeat loss, stale lowstate/IMU, SelectMode failure,
and cable removal. Recovery passes only if `CheckMode` returns the original
`ai` owner, no q/dq/IMU threshold is violated, and a post-handoff read-only
window remains stable.

If local lowstate, internal DDS, and MotionSwitcher are all unavailable, there
is no software-only recovery proof. Physical support and the hardware emergency
stop remain mandatory for that case. No generic zero torque, `kp=0`, process
exit, or unverified Damp command is an acceptable replacement.
