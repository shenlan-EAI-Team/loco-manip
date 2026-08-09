# Live Shadow gate report

- Host network, route, SSH, NVIDIA and CUDA: `PASS`
- Static Live Shadow safety audit: `PASS`
- Static O6 feedback-reader audit: `PASS`
- Five read-only regression tests: `PASS`
- can2 left / can1 right at 1 Mbps, restart-ms 100, qlen 1000: `PASS`
- CAN state and errors throughout: `ERROR-ACTIVE`, tx/rx errors 0
- D435i 5555 identity, serial, RGB shape and rate: `PASS`
- O6 5558 identity, SHA256, atomic schema v2 and rate: `PASS`
- Real synchronized observations: 2,501 (`PASS`)
- Warm-up count: 10 (`PASS`)
- Inferences / Null Sink records: 1,200 / 12,000 (`PASS`)
- Command publish attempts: 0 (`PASS`)
- Control ownership requests: 0 (`PASS`)
- Real SDK command objects: 0 (`PASS`)
- Prohibited command publisher/socket observed: 0 (`PASS`)
- Source cleanup limited to this attempt: `PASS`

Gate passed: **True**

The result permits the next stage to be an offline real-SDK bridge
implementation with publication hard-disabled, mock transports, static audits,
and no hardware-side command object instantiation. It does not authorize bridge
execution against G1/O6, command publication, ownership requests, or motion.

Before any hardware-connected bridge test, the very high arm filter trigger
rates, filter re-anchoring semantics, left O6 instability, right O6 state/domain
mismatch, controller-approved limits, and watchdog behavior require separate
review and gates.
