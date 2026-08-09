# Live synchronization report

Status: **PASS WITH MEASURED SKEW REJECTIONS**

Direct source validation measured D435i at 29.98 Hz, O6 atomic feedback at
19.96 Hz, and DDS at approximately 30.14 Hz. Every accepted camera image was
fresh RGB uint8 `(480,640,3)`; every O6 message was schema v2, feedback-only,
dual 6-D, finite, valid, fresh, and within training scale 0..100.

During the final run the synchronizer examined 2,904 new camera candidates,
accepted 2,501, and rejected 403 (13.88%) only for cross-modal skew above the
strict 20 ms limit. It reported no stale source, decoder rejection, lowstate
error, camera duplicate, or O6 duplicate.

For the 1,200 observations selected for policy inference:

- cross-modal skew: mean 6.45 ms, P95 18.03 ms, P99 19.49 ms, max 19.96 ms
- camera receive age: mean 0.86 ms, max 1.70 ms
- G1 receive age: mean 0.55 ms, max 1.30 ms
- left/right O6 feedback age: mean 11.79/11.66 ms, max 25.71/25.60 ms
- projected gravity remained finite and was computed from normalized wxyz IMU
  quaternion data

The source readers did not zero-fill, forge validity, replay, or reuse a stale
observation.
