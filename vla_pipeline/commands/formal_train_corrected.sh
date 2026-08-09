#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GR00T_REPO:?Set GR00T_REPO to the Isaac-GR00T checkout}"
PYTHON="${PYTHON:-$GR00T_REPO/.venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-$GR00T_REPO/checkpoints/GR00T-N1.7-3B}"
: "${BACKBONE_MODEL:?Set BACKBONE_MODEL to the local Cosmos-Reason2-2B directory}"
DATASET="${DATASET:-$PROJECT_ROOT/datasets_corrected_v1/train_26}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$PROJECT_ROOT/configs/g1_o6_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/formal_train_26_corrected_v1}"

[[ -f "$DATASET/meta/stats.json" ]]
[[ -f "$DATASET/meta/relative_stats.json" ]]
jq -e '.status == "PASS"' \
  "$PROJECT_ROOT/deployment/corrected_pretrain_audit.json" >/dev/null
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Refusing to merge with existing output: $OUTPUT_DIR" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=/tmp/g1_o6_matplotlib_corrected
export NO_ALBUMENTATIONS_UPDATE=1

cd "$GR00T_REPO"
exec "$PYTHON" gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL" \
  --backbone-model-path "$BACKBONE_MODEL" \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG" \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 3000 \
  --save-steps 3000 \
  --save-total-limit 1 \
  --save-only-model \
  --global-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --state-dropout-prob 0.0 \
  --learning-rate 1e-4 \
  --dataloader-num-workers 4 \
  --no-use-wandb
