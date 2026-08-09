# Live timing report

Status: **10 HZ STABLE WITH TWO ISOLATED DEADLINE MISSES**

- requested timed interval: 120 seconds
- formal inferences: 1,200 (exactly 10.00 Hz average)
- warm-up: exactly 10; mean 96.78 ms, max 419.63 ms
- policy inference: mean 62.52 ms, P90 66.28 ms, P99 74.98 ms
- observation construction: mean 1.03 ms, P99 1.88 ms
- adapter: mean 0.96 ms, P99 1.14 ms
- Null Sink: mean 0.062 ms, P99 0.103 ms
- end-to-end: mean 64.57 ms, P90 68.24 ms, P99 77.55 ms
- deadline misses over 100 ms: 2 / 1,200 (0.167%); max 161.59 ms
- scheduler periods skipped: 0

Sampled GPU load remained available throughout; peak observed process-era GPU
memory use was about 9.2 GiB. No CUDA, network, or inference loss occurred.
