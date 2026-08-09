# G1 LowCmd Guard Static Safety Audit - 2026-08-08

## Verdict

The guard is fail-closed and suitable for continued READ_ONLY commissioning,
but it is **not yet authorized for the first real current-q HOLD/recovery
experiment**. Both real execution gates remain false in
`g1_lowcmd_guard.yaml`; no control was executed during this audit.

## Implemented boundary

- The G1-resident process defaults to `read-only`. LowCmd writer construction
  occurs only after config enable, recovery evidence, three CLI gates, a fresh
  current-q command, expected owner `ai`, and persistent one-shot token commit.
- The token must be a non-symlink private regular file. Consumption creates a
  SHA256-bound marker before deleting the issued token, preventing reuse of the
  same token after process restart.
- `ReleaseMode()` is not called by any constructor. The first HOLD write is
  synchronous immediately after a successful `ReleaseMode()` return.
- A monotonic no-catch-up scheduler republishes the immutable current-q command
  at 500 Hz, independently from the future 50 Hz PC target/heartbeat mailbox.
- PC target shape, finiteness, hard limits, monotonic sequence, session,
  heartbeat age, step, and rate are checked before acceptance. The first HOLD
  experiment does not use a PC target.
- The only real-safe `rt/lowcmd` constructor is `UnitreeLowCmdWriter`. Motors
  0-28 serialize explicit mode/q/dq/kp/kd/tau; IDL slots 29-34 serialize all
  zeros; mode_pr, mode_machine, and CRC are assigned for every write.
- `SIGINT` and `SIGTERM` request G1-local recovery. Recovery does not depend on
  the PC path. Automatic SelectMode retry is forbidden.
- Feedback logging is outside the 500 Hz writer callback and includes q, dq,
  tau_est, both IMUs, mode_machine, ownership observations, prepared command,
  scheduler timing, and per-motor maxima.
- A writer failure is explicitly reported as transport unavailable. The CLI no
  longer falsely claims that HOLD is alive after the scheduler has stopped.

## Source-backed frequency statement

Unitree's Python G1 low-level example sets `control_dt_=0.002` and starts a
2 ms recurrent LowCmd writer. The C++ G1 example also creates its command
writer with a 2000 us interval. This project's existing deployer independently
uses `publish_dt_=0.002` and a separate `control_dt_=0.02`.

Therefore 500 Hz is the conservative official/example starting frequency.
There is no source or experiment proving a lower firmware-safe minimum. The G1
NX no-DDS benchmark achieved 499.933 Hz with p99 2.010 ms and max 2.246 ms,
with zero missed deadlines. That benchmark does not include DDS Write latency.

## Verification

- Real-safe tests: 44 passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- G1 NX scheduler baseline and injected blocking benchmarks: completed without
  DDS objects or command writes.
- Post-audit real READ_ONLY smoke test: 250/250 accepted snapshots, owner only
  `ai`, mode_machine only 5, and zero writer/Write/ReleaseMode/SelectMode calls.

## Hard blockers

1. MotionSwitcher client code proves only RPC IDs and return values. Neither the
   SDK nor official G1 examples define a no-jump transition from active LowCmd
   publication to `SelectMode("ai")`, or whether continuing LowCmd while
   selection is pending conflicts with `ai`. The reverse order creates an
   uncontrolled interval. This semantic must not be guessed.
2. The time at which `ReleaseMode()` actually releases `ai` relative to the RPC
   return is undocumented. The guard minimizes return-to-first-write time, but
   cannot bound request-to-release-to-first-write time from client code alone.
3. Current q eliminates position target discontinuity, but it does not prove
   torque/stiffness continuity. The proposed message changes the active
   controller to configured PD gains with dq=0 and tau=0. The official motion
   example is not evidence that this is no-jump or statically stable on this
   supported G1 posture.
4. The single-process lock excludes another local guard, not an arbitrary
   external DDS publisher. An operational publisher/process exclusion check is
   required immediately before any trial.
5. Empty-scheduler timing does not establish real 500 Hz DDS Write latency or a
   safe firmware timeout margin on the G1 NX.

Until blockers 1-3 have authoritative vendor evidence or a separately approved
support-rig commissioning method, `real_execution_enabled=false` and
`recovery_handoff_verified=false` must remain unchanged.

## Audited source hashes

- `core.py`: `841aae5cabb7a97a769f27e79bb4ae8a7c2bc931adad91d2c77350123861492b`
- `runtime.py`: `024e17d573ed2e1c6838b788ef10201713ed0168f97d742f2765dd8cbdbad361`
- `scheduler.py`: `92646122347b414caff78f0bafee1002a00f699f65ea6830f1826700cfa43b32`
- `token_file.py`: `9ecc290040e8b292fc54aa62a8481be3d8012c3ea77ed7208ea37765aa62c55d`
- `unitree_backend.py`: `ae3726e68b36c7004b80963dcebb9313e319914f695e8aef8d15fdeeec2748b6`
- `run_g1_lowcmd_guard.py`: `39aedb78c8b679cb5915636f1afdb5e879ebda2b35bbf00b69ee50cf9d2e1c14`

Hashes must be regenerated after any code change and verified on the G1 before
an experiment.
