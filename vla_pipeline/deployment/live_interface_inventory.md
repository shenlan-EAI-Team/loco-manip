# Live interface inventory

## G1 status observed through read-only SSH

- Host: `unitree-g1-nx`, user `unitree`, IP `192.168.123.164`.
- Wired interface: `enP8p1s0=192.168.123.164/24`; Wi-Fi: `192.168.31.171/24`.
- At audit time there was no tmux runtime and no listener on 5555/5556/5557/5558/5560/5561/60061.
- `g1_deploy_onnx_ref` was not running.

## Selected read-only inputs

|Input|Verified interface|Shape/order|Unit|
|---|---|---|---|
|G1 state|DDS SUB `rt/lowstate`, `unitree_hg::msg::dds_::LowState_`|left arm motor 15:22; right 22:29; waist 12:15|rad|
|IMU|`LowState.imu_state.quaternion`|w,x,y,z|unit quaternion|
|Projected gravity|inverse(wxyz) applied to world `[0,0,-1]`|x,y,z|unit vector|
|Camera|ZMQ SUB `tcp://192.168.123.164:5555`, `ego_view`|480x640x3 RGB|uint8|
|O6|ZMQ SUB `tcp://192.168.123.164:5558`, atomic schema v2|thumb pitch/yaw,index,middle,ring,pinky|training scale 0..100|

SONIC 5557 was rejected as a Live Shadow source: its executable constructs `MotionSwitcherClient`, releases an active mode, creates a `LowCmd` publisher and starts a command writer. It is not read-only.
