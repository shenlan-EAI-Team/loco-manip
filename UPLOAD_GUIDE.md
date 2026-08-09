# GitHub upload guide

Upload this unpacked directory as the source repository. Before making it public:

1. Choose the repository visibility and verify the upstream license requirements.
2. Replace `TODO` in `CHECKPOINT.md` only when a separate checkpoint URL exists.
3. Review author/institution naming and any real-hardware commissioning reports.
4. Initialize Git, commit, and push. Do not force-add files ignored by `.gitignore`.

Keep these outside the GitHub source repository:

- corrected checkpoint tensor shards;
- original and corrected image/video/parquet datasets;
- one-time command tokens and consumed-token markers;
- raw Live Shadow/robot logs, SSH material, credentials, and local environments;
- full copies of the two NVIDIA upstream repositories.

For a teacher reviewing architecture and experimental evidence, the current package
is sufficient. Provide the checkpoint separately only if they need to reproduce
numerical inference rather than inspect code, configuration, curves, and metrics.

