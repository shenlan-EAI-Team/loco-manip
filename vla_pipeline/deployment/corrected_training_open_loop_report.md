# Corrected Training and Open-loop Evaluation

## Contract

- Checkpoint: corrected_v1 checkpoint-3000.
- Official GR00T open-loop script: denoising steps 4, execution horizon 16.
- Deterministic quantitative pass: seed 42, corrected smoke/val/test splits.
- Arm errors are radians; O6 errors are percentage points. They are never averaged together in conclusions.

## Training

- Completed 3000 / 3000 optimizer steps.
- Mean loss over first 100 steps: 1.093680.
- Mean loss over last 100 steps: 0.043680.
- Minimum logged loss: 0.035900 at step 2750.
- Maximum gradient norm: 1.366142; all logged loss and gradients are finite.

## Aggregate MAE

|split|left arm (rad)|right arm (rad)|left O6 (points)|right O6 (points)|
|---|---:|---:|---:|---:|
|smoke|0.042674|0.019139|1.202720|0.000000|
|val|0.054535|0.028201|5.604259|0.000000|
|test|0.076470|0.033057|4.948925|0.000000|

## Old Versus Corrected Test

|group|old MAE|corrected MAE|reduction|
|---|---:|---:|---:|
|left_arm|0.233185|0.076470|67.2%|
|right_arm|0.262069|0.033057|87.4%|
|left_o6|19.792858|4.948925|75.0%|
|right_o6|0.000000|0.000000|n/a|

## Findings

- Both arms improved materially after removing the second reorder/offset/scale. Test MAE is 0.0765 rad left and 0.0331 rad right.
- The test-versus-validation gap is 40.2% for the left arm and 17.2% for the right arm.
- Left O6 test MAE is 4.95 points, but 4.09% of scalar predictions exceed 30 points error; rare outliers dominate RMSE.
- Right O6 is exactly zero because every training state/action is zero and invalid. Zero error is label degeneracy, not learned right-hand control.
- Open-loop curves concatenate independent 16-step chunks. Chunk-boundary discontinuity must be handled by the deployment replanning/interpolation safety layer; this evaluation is not a closed-loop stability proof.
- At deployment-style execution horizon 3, test arm MAE improves to 0.0485 rad left and 0.0253 rad right.
- Horizon 3 does not fix the absolute O6 output: left O6 MAE changes from 4.95 to 5.61 points, so final O6 smoothing/enveloping remains mandatory.
