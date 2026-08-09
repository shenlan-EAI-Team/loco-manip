# O6 interface contract

- Upstream LinkerHand O6 CAN SDK command and feedback registers are native integer `0..255`.
- `get_current_status()` sends a CAN read request (`0x01` with no target bytes); `set_joint_positions()` sends `0x01` plus six target bytes. These are distinct operations.
- The data-collection ZMQ client rejects anything outside `0..100`; converted train/test state and action are also `0..100`.
- Joint order is `thumb_cmc_pitch, thumb_cmc_yaw, index_mcp_pitch, middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch`.
- Therefore the deployed O6 driver contains a raw-255 to percentage conversion before publishing `actual_q`/`action` and the future command bridge must apply the inverse conversion exactly once.
- The exact rounding/clamping expression in the G1-side `glove_teleop_dual_o6.py` must still be read from the G1 host before any command bridge is developed; this report deliberately does not guess whether it uses round, floor, or truncation.
- Live Shadow creates only a ZMQ SUB socket. It creates no O6 command socket and never imports `LinkerHandApi`.
