# G1 + O6 Live Shadow report

Date: 2026-08-05 17:57:21 +08:00

Status: **SCENARIO A COMPLETE / LIVE SHADOW GATE PASSED**

The final 120-second run used checkpoint-4000, denoising steps 4, exactly 10
warm-up inferences, 10 Hz replanning, execution horizon 3, and a 30 Hz virtual
control timeline. All hypothetical actions terminated in `NullActionSink`.

## Read-only sources

- D435i: PID 6615, tmux `lsro_camera_20260805_174318`, ZMQ 5555, serial
  `342522073568`, `ego_view` RGB `480x640x3`, direct rate 29.98 Hz.
- O6: PID 6762, tmux `lsro_o6_20260805_174318`, ZMQ 5558, audited relay SHA256
  `d76cb5ef41da41abf305d3b09bd8e2e8dda23c2bc14a11998fe625b7a28a14c8`,
  direct rate 19.96 Hz.
- DDS: subscriber-only `rt/lowstate` on `enp4s0`; 7+7 arm, 3 waist and valid
  wxyz quaternion fields.

The O6 serial query established can2 as left (`LHO6-03-1097-L-Z-1-E`) and can1
as right (`LHO6-03-1093-R-Z-1-E`). A four-second CAN capture contained only
DLC=1 status queries and their responses: can2 ID 0x28 `01`, can1 ID 0x27 `01`.
There were no DLC=7 position requests, unknown IDs, or CAN errors.

## Final run

- synchronized observations accepted: 2,501 / 2,904 camera candidates
- cross-modal-skew rejections: 403; no stale/decode/invalid-feedback rejection
- policy inferences: 1,200 over 120 seconds (10.00 Hz)
- Null Sink records: 12,000
- deadline misses: 2 / 1,200 (0.167%)
- end-to-end latency: mean 64.57 ms, P99 77.55 ms, max 161.59 ms
- command publish attempts: 0
- control ownership requests: 0
- real SDK command objects: 0

Both O6 feedback vectors remained constant for all 1,200 selected observations,
and no O6 command frame or setter was present. No physical finger motion was
observed in feedback. No prohibited G1/O6 control process was found.

Two fail-closed pre-runs exposed and fixed host-only defects: the process audit
mistook `gear_sonic.camera` for SONIC control, and the live observation lacked
the trained policy's T=1 axis. Neither pre-run reached an action sink or sent a
command. Five read-only tests and both static audits pass after the fixes.

Full run summary:
`deployment/logs/live_shadow_host/20260805_175132/20260805_175134_A/summary.json`.
Remote logs: `deployment/logs/remote_readonly_sources/live_shadow_readonly_20260805_174318/`.
