# Policy Output Contract

## Proven contract

- left_arm/right_arm config: `RELATIVE + NON_EEF`.
- left_o6/right_o6 config: `ABSOLUTE + NON_EEF`.
- `Gr00tPolicy.get_action()` returns decoded physical-unit actions.
- Arms returned by the standard API are absolute joint targets in rad.
- O6 returned by the standard API is an absolute 0–100 command.
- Action Adapter must not denormalize and must not add `q_current` again.
- `PolicyClient` returns the server payload unchanged.

## Numerical test on test_2 local episode 0 frame 0

|group|normalized range|pre-relative physical range|public API range|manual decode == API|
|---|---|---|---|---|
|left_arm|[-2.796875, 2.156250]|[-1.280818, 1.322413]|[-1.014048, 1.720117]|True|
|right_arm|[-3.000000, 3.406250]|[-1.281483, 0.479911]|[-1.299179, 0.728952]|True|
|left_o6|[-1.554688, 3.343750]|[0.000000, 100.000000]|[0.000000, 100.000000]|True|
|right_o6|[-2.734375, 2.453125]|[0.000000, 0.000000]|[0.000000, 0.000000]|True|

Arm identity tests:
- `left_arm`: `public_action - current_state == denormalized_relative` → `True`.
- `right_arm`: `public_action - current_state == denormalized_relative` → `True`.

Project duplicate-transform scan hits: `0`.

## Local code path

- `Gr00tPolicy._get_action`: `/home/slxy/下载/Isaac-GR00T/gr00t/policy/gr00t_policy.py:380`–`432`
- `processor.decode_action`: `/home/slxy/下载/Isaac-GR00T/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:382`–`404`
- `StateActionProcessor.unapply_action`: `/home/slxy/下载/Isaac-GR00T/gr00t/data/state_action/state_action_processor.py:421`–`526`
- `PolicyClient._get_action`: `/home/slxy/下载/Isaac-GR00T/gr00t/policy/server_client.py:387`–`393`

`Gr00tPolicy._get_action` obtains normalized `action_pred`, then calls `Gr00tN1d7Processor.decode_action`; this calls `StateActionProcessor.unapply_action`, which denormalizes first and then converts relative arm chunks to absolute using the last raw state.
