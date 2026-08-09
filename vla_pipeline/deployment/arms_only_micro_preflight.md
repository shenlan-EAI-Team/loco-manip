# First arms-only model micro-motion preflight

Status: **EXECUTED ONCE - SAFE COMPLETION - RESPONSE NOT PROVEN**

- Corrected checkpoint manifest:
  `e040f2802954b94eaedd071d73685dbaf416efece2b4fb06e0695aa2f609e012`.
- Window: one 0.5-second model window; automatic repeat is false.
- Both G1 arms participate. Maximum preview offset is 0.010 rad per joint,
  inside the requested +/-0.03 rad outer envelope and the stricter first-run
  +/-0.01 rad effective envelope.
- Velocity and acceleration limits remain 0.12 rad/s and 0.4 rad/s^2.
- Left and right O6 use the feedback-only transport. Both position command
  counts are required to remain zero.
- Waist/leg command count is required to remain zero.
- Normal completion or FAULT uses the existing 100-message, two-second weight
  release to exactly zero, followed by 0.5 seconds of read-only monitoring.
- Runtime logging includes initial q, raw policy target, final command q,
  command/feedback delta and sign, command/state motor index, maximum response
  joint, maximum feedback offset, and release rebound.

Plan SHA256:
`6010d7293ab6805ab8f2e7ded813d69859cd4a2190f2ac1f37ffdb6c935190ee`.

Plan and token are stored under
`deployment/logs/real_bridge_preflight_corrected/20260807_164943_arms_only/`.
The plan-bound token was consumed for one authorized execution. The command and
release path completed safely, but the maximum arm feedback response was only
`4.793704e-05 rad`, so a controlled physical response was not proven. See
`deployment/real_micro_motion_report.md`. No repeat or O6 command test is
authorized from this artifact.
