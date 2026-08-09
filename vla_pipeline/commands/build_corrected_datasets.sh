#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GR00T_REPO:?Set GR00T_REPO to the Isaac-GR00T checkout}"
: "${SOURCE_DATASET:?Set SOURCE_DATASET to the original LeRobot dataset}"
PYTHON="${PYTHON:-$GR00T_REPO/.venv/bin/python}"
SOURCE="$SOURCE_DATASET"
OUTPUT_ROOT="$PROJECT_ROOT/datasets_corrected_v1"

[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "Refusing to overwrite existing corrected datasets: $OUTPUT_ROOT" >&2
  exit 2
}

export PYTHONNOUSERSITE=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "$PROJECT_ROOT"
"$PYTHON" scripts/convert_dataset.py \
  --source "$SOURCE" \
  --output-root "$OUTPUT_ROOT" \
  --split split.json

cd "$GR00T_REPO"
for split in smoke_2 train_26 val_2 test_2; do
  "$PYTHON" gr00t/data/stats.py \
    --dataset-path "$OUTPUT_ROOT/$split" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$PROJECT_ROOT/configs/g1_o6_config.py"
done
