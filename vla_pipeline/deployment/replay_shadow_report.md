# Offline Replay Shadow Report

Mode: **OFFLINE REPLAY SHADOW: NO COMMANDS SENT**.

- Dataset frames: 1199
- Policy calls: 401
- denoising steps: 4
- execution horizon: 3
- NaN/Inf: 0
- out-of-bounds elements: 0
- buffer underruns: 0
- calls exceeding a single 30 Hz period: 401

Inference latency: mean `55.573` ms, p50 `54.001` ms, p90
`58.563` ms, p99 `69.694` ms, max `313.473` ms.

All calls exceed 33.3 ms, so inference cannot be scheduled every camera frame. The recommended
10 Hz planner budget is 100 ms and is satisfied after warmup. The maximum includes first-call
cold start; a future live process must warm up before entering its timed loop.

Full per-frame log: `deployment/logs/replay_shadow/replay_shadow.jsonl`.
