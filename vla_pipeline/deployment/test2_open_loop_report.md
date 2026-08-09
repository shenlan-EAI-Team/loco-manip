# test_2 Open-loop Evaluation

纯离线评估；test_2 只用于最终未见数据报告，没有用于选择模型或继续调参。

## denoise_4_execute_16

### left_arm (rad)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|shoulder_pitch|0.032099|0.037120|0.081756|
|shoulder_roll|0.444584|0.562598|1.647535|
|shoulder_yaw|0.526116|0.628083|1.946641|
|elbow|0.226982|0.265371|0.665109|
|wrist_roll|0.111919|0.131118|0.343333|
|wrist_pitch|0.111512|0.135796|0.280894|
|wrist_yaw|0.179083|0.210616|0.451834|

Adjacent prediction max jump: `2.313180`; chunk-boundary max jump: `1.991368`.
Velocity abs mean/p95/max: `8.3129` / `32.9973` / `69.3954`.
Acceleration abs mean/p95/max: `437.5978` / `1703.3390` / `4086.7029`.
### right_arm (rad)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|shoulder_pitch|0.224444|0.262018|0.587076|
|shoulder_roll|0.477607|0.564191|1.440139|
|shoulder_yaw|0.125142|0.154362|0.388365|
|elbow|0.097243|0.114751|0.293891|
|wrist_roll|0.164948|0.190526|0.485440|
|wrist_pitch|0.591381|0.693049|1.306234|
|wrist_yaw|0.153720|0.189350|0.473652|

Adjacent prediction max jump: `1.903687`; chunk-boundary max jump: `1.568729`.
Velocity abs mean/p95/max: `8.6345` / `28.6450` / `57.1106`.
Acceleration abs mean/p95/max: `460.2992` / `1462.6542` / `2983.7722`.
### left_o6 (percentage points)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|thumb_cmc_pitch|13.614735|19.249472|56.427456|
|thumb_cmc_yaw|16.590760|21.163966|58.431372|
|index_mcp_pitch|22.580579|34.629778|98.823529|
|middle_mcp_pitch|21.354558|35.840674|100.000000|
|ring_mcp_pitch|22.027768|36.277598|100.000000|
|pinky_mcp_pitch|22.588748|36.287387|100.000000|

Adjacent prediction max jump: `100.000000`; chunk-boundary max jump: `100.000000`.
Velocity abs mean/p95/max: `741.5547` / `2308.3008` / `3000.0000`.
Acceleration abs mean/p95/max: `40467.4062` / `114876.5645` / `180000.0000`.
### right_o6 (percentage points)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|thumb_cmc_pitch|0.000000|0.000000|0.000000|
|thumb_cmc_yaw|0.000000|0.000000|0.000000|
|index_mcp_pitch|0.000000|0.000000|0.000000|
|middle_mcp_pitch|0.000000|0.000000|0.000000|
|ring_mcp_pitch|0.000000|0.000000|0.000000|
|pinky_mcp_pitch|0.000000|0.000000|0.000000|

Adjacent prediction max jump: `0.000000`; chunk-boundary max jump: `0.000000`.
Velocity abs mean/p95/max: `0.0000` / `0.0000` / `0.0000`.
Acceleration abs mean/p95/max: `0.0000` / `0.0000` / `0.0000`.
Right O6 global prediction min/max/mean/std: `0.0` / `0.0` / `0.0` / `0.0`; nonzero frame ratio: `0.0`.

## denoise_4_execute_1

### left_arm (rad)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|shoulder_pitch|0.031162|0.035798|0.083121|
|shoulder_roll|0.282725|0.338055|0.946267|
|shoulder_yaw|0.384650|0.469793|1.374960|
|elbow|0.207033|0.242938|0.595055|
|wrist_roll|0.090608|0.106717|0.230713|
|wrist_pitch|0.113262|0.138169|0.274911|
|wrist_yaw|0.176084|0.204058|0.432081|

Adjacent prediction max jump: `1.557510`; chunk-boundary max jump: `1.557510`.
Velocity abs mean/p95/max: `6.2656` / `22.8500` / `46.7253`.
Acceleration abs mean/p95/max: `335.2864` / `1173.7716` / `2767.6660`.
### right_arm (rad)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|shoulder_pitch|0.168137|0.198622|0.466977|
|shoulder_roll|0.389842|0.452278|1.043084|
|shoulder_yaw|0.106939|0.131912|0.305150|
|elbow|0.074400|0.088147|0.216192|
|wrist_roll|0.135009|0.158523|0.381546|
|wrist_pitch|0.602299|0.701570|1.301964|
|wrist_yaw|0.153799|0.190091|0.465644|

Adjacent prediction max jump: `1.319523`; chunk-boundary max jump: `1.319523`.
Velocity abs mean/p95/max: `7.4485` / `24.7988` / `39.5857`.
Acceleration abs mean/p95/max: `397.4200` / `1249.5769` / `2344.5430`.
### left_o6 (percentage points)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|thumb_cmc_pitch|14.497555|20.596254|57.603924|
|thumb_cmc_yaw|15.996880|20.449778|58.431372|
|index_mcp_pitch|22.096276|34.161392|98.823529|
|middle_mcp_pitch|22.047571|36.804506|100.000000|
|ring_mcp_pitch|21.763215|36.453143|100.000000|
|pinky_mcp_pitch|22.462047|36.054950|100.000000|

Adjacent prediction max jump: `100.000000`; chunk-boundary max jump: `100.000000`.
Velocity abs mean/p95/max: `760.0124` / `2327.7573` / `3000.0000`.
Acceleration abs mean/p95/max: `41639.9851` / `114257.8125` / `180000.0000`.
### right_o6 (percentage points)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|thumb_cmc_pitch|0.000000|0.000000|0.000000|
|thumb_cmc_yaw|0.000000|0.000000|0.000000|
|index_mcp_pitch|0.000000|0.000000|0.000000|
|middle_mcp_pitch|0.000000|0.000000|0.000000|
|ring_mcp_pitch|0.000000|0.000000|0.000000|
|pinky_mcp_pitch|0.000000|0.000000|0.000000|

Adjacent prediction max jump: `0.000000`; chunk-boundary max jump: `0.000000`.
Velocity abs mean/p95/max: `0.0000` / `0.0000` / `0.0000`.
Acceleration abs mean/p95/max: `0.0000` / `0.0000` / `0.0000`.
Right O6 global prediction min/max/mean/std: `0.0` / `0.0` / `0.0` / `0.0`; nonzero frame ratio: `0.0`.

## denoise_8_execute_1

### left_arm (rad)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|shoulder_pitch|0.031153|0.035802|0.083121|
|shoulder_roll|0.283172|0.338145|0.946267|
|shoulder_yaw|0.385659|0.469975|1.383970|
|elbow|0.207098|0.242952|0.595055|
|wrist_roll|0.090687|0.106789|0.230713|
|wrist_pitch|0.113232|0.138162|0.274911|
|wrist_yaw|0.176541|0.204521|0.434299|

Adjacent prediction max jump: `1.570913`; chunk-boundary max jump: `1.570913`.
Velocity abs mean/p95/max: `6.2549` / `22.7799` / `47.1274`.
Acceleration abs mean/p95/max: `334.8667` / `1174.5208` / `2767.6660`.
### right_arm (rad)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|shoulder_pitch|0.167915|0.198312|0.466977|
|shoulder_roll|0.390032|0.452486|1.043084|
|shoulder_yaw|0.106745|0.131647|0.305150|
|elbow|0.074433|0.088080|0.216192|
|wrist_roll|0.134690|0.158082|0.381546|
|wrist_pitch|0.600640|0.700140|1.301964|
|wrist_yaw|0.153989|0.190340|0.465644|

Adjacent prediction max jump: `1.321706`; chunk-boundary max jump: `1.321706`.
Velocity abs mean/p95/max: `7.4425` / `24.8441` / `39.6512`.
Acceleration abs mean/p95/max: `397.0117` / `1250.0499` / `2347.5750`.
### left_o6 (percentage points)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|thumb_cmc_pitch|14.444071|20.544345|57.603924|
|thumb_cmc_yaw|15.965225|20.404495|58.431372|
|index_mcp_pitch|22.071580|34.135391|98.823529|
|middle_mcp_pitch|22.023162|36.776285|100.000000|
|ring_mcp_pitch|21.675532|36.358077|100.000000|
|pinky_mcp_pitch|22.501567|36.096371|100.000000|

Adjacent prediction max jump: `100.000000`; chunk-boundary max jump: `100.000000`.
Velocity abs mean/p95/max: `759.6731` / `2325.8789` / `3000.0000`.
Acceleration abs mean/p95/max: `41629.3253` / `114646.4930` / `180000.0000`.
### right_o6 (percentage points)

|dimension|MAE|RMSE|max abs error|
|---|---:|---:|---:|
|thumb_cmc_pitch|0.000000|0.000000|0.000000|
|thumb_cmc_yaw|0.000000|0.000000|0.000000|
|index_mcp_pitch|0.000000|0.000000|0.000000|
|middle_mcp_pitch|0.000000|0.000000|0.000000|
|ring_mcp_pitch|0.000000|0.000000|0.000000|
|pinky_mcp_pitch|0.000000|0.000000|0.000000|

Adjacent prediction max jump: `0.000000`; chunk-boundary max jump: `0.000000`.
Velocity abs mean/p95/max: `0.0000` / `0.0000` / `0.0000`.
Acceleration abs mean/p95/max: `0.0000` / `0.0000` / `0.0000`.
Right O6 global prediction min/max/mean/std: `0.0` / `0.0` / `0.0` / `0.0`; nonzero frame ratio: `0.0`.

## test_2 versus val_2 (denoise=4, execute=16)

|group|val mean per-dim MAE|test mean per-dim MAE|
|---|---:|---:|
|left_arm|0.222957|0.233185|
|right_arm|0.239752|0.262069|
|left_o6|21.618764|19.792858|
|right_o6|0.000000|0.000000|

结论：execute=1 的 arm MAE 较 execute=16 低，但逐帧随机重规划仍会产生显著跳变；denoise=8 相对 4 没有实质收益。左 O6 出现最大 100 点跳变，必须经过安全过滤。右 O6 始终为零是零方差训练标签造成的退化输出，不代表右手能力。

Plots: `deployment/plots/test2/<configuration>/`.
