# Corrected Dataset Pre-training Audit

Status: **PASS**

## Model Contract

- Image input: `ego_view`, RGB HWC `480 x 640 x 3`, 30 fps.
- State input: 32 values = left arm 7 + right arm 7 + left O6 6 + right O6 6 + waist 3 + projected gravity 3.
- Action target: 26 values per step; 16-step chunk (`16 x 26`).
- Arm parquet targets are absolute hardware-order radians. The official processor alone converts them to relative actions.
- O6 targets are absolute 0..100 percentages. Waist and projected gravity are inputs only.

## Hard Checks

- Train/validation/test episodes are disjoint and cover source episodes 0..29.
- Every corrected state/action value exactly matches the independently reconstructed source value.
- No second joint reorder, default-angle offset, or action scaling remains.
- All state/action values are finite; all arm targets, arm feedback, and waist feedback are within the official G1 XML limits.
- All timestamps are strictly increasing and frame/global/episode indices are consistent.
- Every copied video has the same SHA256 as its source and is H264 640x480 at 30 fps with matching frame count.
- Projected gravity is body-frame inverse-rotated world gravity and has unit norm.
- `stats.json` and both 16x7 relative-arm statistics were independently recomputed and match.
- Official `LeRobotEpisodeLoader` accepts the dataset and exposes every configured group at the expected dimension.
- The official normalization processor produces finite semantic tensors (`state 1x32`, `action 16x26`) in [-1, 1].
- The N1.7 processor pads them to `state 1x132`, `action 40x132`; exactly 416 action-mask elements are valid.

## Training Split

- Episodes: 26
- Frames: 13112
- Left O6 valid rate: 1.000000
- Right O6 valid rate: 0.000000
- Gravity norm range: 0.999999940 .. 1.000000000

## Known Limitations

- Right O6 feedback/action are all zero and right_hand_valid is always false. Training is valid for both arms and left O6, but cannot teach right O6 behavior.
- Some source episodes include post-success manual reset tails; they remain untrimmed.

## Decision

The corrected dataset passes the hard pre-training gate. It is suitable for retraining both arms and the left O6. Right O6 behavior must not be claimed from this dataset.
