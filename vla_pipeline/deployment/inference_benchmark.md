# Offline Inference Benchmark

test_2 cached replay on the local GPU; 10 warmups + 100 measured calls per setting.

|setting|mean E2E ms|p50|p90|p99|mean policy ms|theoretical Hz|peak allocated MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|denoising_4|53.617|52.915|55.446|63.512|53.216|18.65|6077.2|
|denoising_8|81.833|81.192|84.097|88.637|81.513|12.22|6077.2|

## Stage breakdown (mean ms)

|setting|data|dict|preprocess|model|postprocess|
|---|---:|---:|---:|---:|---:|
|denoising_4|0.246|0.154|2.677|49.956|0.306|
|denoising_8|0.246|0.075|2.719|78.169|0.324|

## Scheduling recommendation

Initial recommendation: `denoising_steps=4`, 10 Hz replanning, `execution_horizon=3`, with a continuous 30 Hz action timeline and 100 Hz SDK interpolation. The 4-step p99 fits a 100 ms replan budget locally; 8 denoising steps adds latency without useful test error improvement. If future Live Shadow p99 approaches 80–90 ms after camera/network costs, fall back to 7.5 Hz / horizon 4. 5 Hz / horizon 6 has more timing margin but reacts more slowly and executes longer stale chunks.

This benchmark excludes real camera acquisition, image transport, timestamp alignment, G1/O6 network latency, SDK queueing, controller latency, and motor response. Those must be measured separately in Live Shadow.
