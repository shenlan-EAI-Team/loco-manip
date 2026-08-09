# GR00T N1.7 G1+O6 Deployment Preflight

Everything in this directory defaults to offline shadow or mock mode. No script connects to G1,
claims robot control, or publishes G1/O6 commands.

## Required environment

```bash
cd /home/slxy/下载/g1_o6_gr00t
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/g1_o6_matplotlib
PY=/home/slxy/下载/Isaac-GR00T/.venv/bin/python
```

## Run order

1. test_2 evaluation

   ```bash
   $PY deployment/evaluate_open_loop.py
   ```

2. Policy output contract

   ```bash
   $PY deployment/tests/verify_policy_output_contract.py
   ```

3. Inference benchmark

   ```bash
   $PY deployment/benchmark_inference.py --warmup 10 --iterations 100
   ```

4. Replay Shadow

   ```bash
   $PY deployment/g1_o6_shadow_client.py --denoising-steps 4 --execution-horizon 3
   ```

5. Adapter dry-run

   ```bash
   $PY deployment/run_adapter_dry_run.py
   ```

6. Mock failure tests

   These are included in `run_adapter_dry_run.py`: NaN, timeout, network disconnect,
   watchdog, emergency stop, and empty buffer.

7. Real-observation Live Shadow (read-only, Null Sink)

   ```bash
   $PY deployment/audit_live_safety.py
   $PY deployment/tests/test_live_shadow_readonly.py

   # Scene A: unchanged table, at least 120 seconds
   $PY deployment/run_live_shadow.py --scenario A --duration 120

   # Scene B: manually move the cylinder/bin every 5-10 seconds
   $PY deployment/run_live_shadow.py --scenario B --duration 120

   # Only if 10 Hz is unstable
   $PY deployment/run_live_shadow.py \
     --scenario timing_7p5hz --duration 120 --replanning-hz 7.5

   $PY deployment/generate_live_reports.py
   ```

   Required G1-side read-only services are D435i PUB `:5555` and an independently reviewed
   feedback-only dual-O6 PUB `:5558`. Do not start `g1_deploy_onnx_ref` to obtain state: that
   binary releases the current motion mode and creates a real LowCmd publisher. G1 proprioception
   is read directly by a subscriber-only `rt/lowstate` process on the control computer.

Do not change `real_hardware_enabled`, `publish_commands`, `shadow_only`, or `dry_run`. The Live
Shadow runner drains the Action Adapter into an in-memory Null Sink and never calls
`ActionAdapter.drain_to_mock()`. The Action Adapter receives standard Policy API physical-unit
outputs; it never denormalizes and never adds `q_current`.
