# Package verification

Checks run while preparing this source package:

- Public command syntax: `bash -n` passed for corrected dataset build, corrected
  training, open-loop evaluation, and Live Shadow host scripts.
- VLA conversion/adapter/bridge tests: 34 passed.
- Decoupled WBC real-safe/LowCmd guard tests: 67 passed.
- Isaac-GR00T local-backbone resolution checks: 2 passed using the training
  virtual environment.
- Sensitive-material scan: no private key, API key, password, real one-time token,
  consumed-token artifact, or credential file was found.
- Large-file scan: no file above 50 MB is included.

The full Live Shadow read-only pytest module was not rerun during packaging because
the retained Isaac-GR00T training environment does not include pytest and the system
pytest environment does not include the `gr00t` package. Its prior corrected
real-input result is preserved in
`vla_pipeline/deployment/corrected_live_shadow_report.md`.

Private RFC1918 robot addresses remain in hardware configuration and selected
commissioning documentation because they are part of the reproducible interface
configuration. No routable address or authentication material is included.

