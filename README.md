# G1 + O6 GR00T corrected training and deployment code

This repository snapshot contains the code and compact evidence for the corrected
G1 dual-arm + O6 GR00T N1.7 pipeline. Large training data and model tensor shards
are intentionally not included.

## What is included

- `vla_pipeline/`: corrected dataset conversion, modality mapping, training and
  open-loop evaluation scripts, Live Shadow, adapters, safety filters, real-bridge
  source, and tests.
- `isaac_gr00t_overlay/`: the small source overlay used on NVIDIA Isaac-GR00T at
  base commit `b9955401d50c92a29258732e3ad6ccd579f1bdc0`.
- `whole_body_control_overlay/`: the real-safe Decoupled WBC/LowCmd guard overlay
  used on NVlabs/GR00T-WholeBodyControl at base commit
  `bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8`.
- `checkpoint_metadata/`: processor, statistics, model index, trainer state, and
  experiment configuration for corrected `checkpoint-3000`, without tensors.
- `dataset_metadata/`: schemas, statistics, split metadata, and conversion
  manifests. Images, videos, and parquet episodes are excluded.
- `results/`: training curves and corrected official open-loop plots/metrics.
- `docs/`: selected real-hardware commissioning outcomes.

The `whole_body_control_overlay/LICENSE` file preserves the upstream dual-license
notice. The source overlays must be reviewed together with their upstream licenses.

## Corrected model contract

The observation uses one ego-view RGB frame plus state groups for left arm (7),
right arm (7), left O6 (6), right O6 (6), waist (3), and projected gravity (3).
The policy predicts a 16-step action chunk with left arm (7), right arm (7), left
O6 (6), and right O6 (6), for 26 values per step.

The GR00T configuration represents arm actions as relative during model processing,
but the public `Gr00tPolicy.get_action()` API decodes them to **absolute joint
targets in radians**. O6 outputs are **absolute targets in 0-100 percentage points**.
Deployment must not denormalize again and must not add current arm position again.
See `vla_pipeline/deployment/policy_output_contract.md`.

The corrected dataset rebuild uses the already hardware-ordered absolute arm data
and removes the erroneous second reorder/offset/scale. The exact mapping and checks
are implemented in `vla_pipeline/scripts/convert_dataset.py` and
`vla_pipeline/scripts/audit_corrected_pretrain.py`.

## Reproduce the source environment

1. Clone `https://github.com/NVIDIA/Isaac-GR00T.git` and check out
   `b9955401d50c92a29258732e3ad6ccd579f1bdc0`.
2. Copy the contents of `isaac_gr00t_overlay/` over that checkout.
3. Clone `https://github.com/NVlabs/GR00T-WholeBodyControl.git` and check out
   `bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8`.
4. Copy the contents of `whole_body_control_overlay/` over that checkout.
5. Set the environment variables documented by `vla_pipeline/commands/*.sh`
   (`GR00T_REPO`, `SOURCE_DATASET`, and `BACKBONE_MODEL` as applicable) before
   running dataset conversion, training, or evaluation.

## Training and open-loop result

- Corrected training completed 3000/3000 steps.
- Mean loss: 1.093680 over the first 100 steps and 0.043680 over the last 100.
- Corrected test arm MAE: 0.076470 rad left and 0.033057 rad right.
- At deployment-style execution horizon 3: 0.0485 rad left and 0.0253 rad right.
- Left O6 test MAE: 4.948925 percentage points, with rare large spikes that require
  the deployment output guard.
- Right O6 labels are all zero/invalid; it remains feedback-only and its zero error
  must not be interpreted as learned command performance.

See `vla_pipeline/deployment/corrected_training_open_loop_report.md` and
`results/open_loop/`.

## Deployment status and safety boundary

Corrected 30-second real-input Live Shadow passed at 10 Hz with NullActionSink and
zero command publication, ownership requests, and real SDK command objects.

Standalone Gear WBC completed current-q HOLD, 3-second engage, and 5-second STAND
under physical support. The subsequent `SelectMode("ai")` handback returned Unitree
error 7002. A later single-wrist proof faulted on the frozen-arm envelope before the
wrist reference began, so deterministic wrist actuation was **not** proven.

This snapshot is research/commissioning code, not an autonomous production robot
release. Files under `real_bridge/` and `real_safe/` can construct real command
transports when explicit gates are satisfied. Do not run them on hardware without
the documented support, emergency-stop, exclusivity, and one-shot procedures.

## Checkpoint and data

The 12 GB checkpoint is not needed for GitHub code review and cannot be uploaded as
normal GitHub files (individual shards exceed 100 MB). If exact inference
reproduction is required, publish it separately using Hugging Face, institutional
object storage, or another artifact registry and add the URL to `CHECKPOINT.md`.
Do not commit the tensor shards to this repository.
