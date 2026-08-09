# Action Adapter Dry-run Report

Pipeline: `Policy API -> Action Adapter -> Safety Filter -> Action Buffer -> Mock SDK`.
No real SDK, topic, network endpoint, or motor was used.

|group|max policy velocity|max adapter velocity|max policy accel|max adapter accel|boundary before|boundary after|
|---|---:|---:|---:|---:|---:|---:|
|left_arm|47.7473|0.8000|2802.5811|3.0001|1.5916|0.0267|
|right_arm|41.6739|0.8000|2490.1667|3.0001|1.3777|0.0267|
|left_o6|3000.0000|240.0001|180000.0000|1200.0092|100.0000|8.0000|
|right_o6|0.0000|0.0000|0.0000|0.0000|0.0000|0.0000|

## Filter triggers

|group|position|velocity|acceleration|O6 8-point delta|nonfinite|
|---|---:|---:|---:|---:|---:|
|left_arm|0|7393|7697|0|0|
|right_arm|0|7758|7746|0|0|
|left_o6|0|2641|6081|4929|0|
|right_o6|0|0|0|0|0|

## Mock failure tests

- NaN filtered to finite: `True`
- timeout recorded/hold: `True`
- network disconnect recorded/hold: `True`
- watchdog holds last safe target: `True`
- emergency stop holds last safe target: `True`
- empty buffer underrun recorded: `True`

Main run mock records: G1 `4010`, O6 `4010`; main-run underruns `0`.

Plots: `deployment/plots/adapter_dry_run/`.
