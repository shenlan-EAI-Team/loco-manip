#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/checkpoint-N" >&2
  exit 2
fi

CHECKPOINT="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GR00T_REPO:?Set GR00T_REPO to the Isaac-GR00T checkout}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
case "$EVAL_SPLIT" in
  val)
    DATASET="$PROJECT_ROOT/datasets_corrected_v1/val_2"
    ;;
  test)
    DATASET="$PROJECT_ROOT/datasets_corrected_v1/test_2"
    ;;
  *)
    echo "EVAL_SPLIT must be val or test" >&2
    exit 2
    ;;
esac

PYTHON="${PYTHON:-$GR00T_REPO/.venv/bin/python}"
PLOT_DIR="${PLOT_DIR:-$PROJECT_ROOT/outputs/open_loop_${EVAL_SPLIT}}"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/g1_o6_matplotlib
export MPLBACKEND=Agg
export NO_ALBUMENTATIONS_UPDATE=1

cd "$GR00T_REPO"
mkdir -p "$PLOT_DIR"
"$PYTHON" gr00t/eval/open_loop_eval.py \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "$CHECKPOINT" \
  --traj-ids 0 1 \
  --execution-horizon 16 \
  --steps 1000 \
  --denoising-steps 4 \
  --modality-keys left_arm right_arm left_o6 right_o6

cp -a /tmp/open_loop_eval/traj_0.jpeg "$PLOT_DIR/episode_0_actions.jpeg"
cp -a /tmp/open_loop_eval/traj_1.jpeg "$PLOT_DIR/episode_1_actions.jpeg"
