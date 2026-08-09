# Corrected checkpoint publication

Local source artifact:

`outputs/formal_train_26_corrected_v1/checkpoint-3000`

- Training step: 3000
- Approximate total size: 12 GB
- Tensor layout: three safetensors shards
- GitHub source repository: intentionally excluded
- External download URL: `TODO: add only if exact inference reproduction is required`

The small processor/statistics/configuration files are preserved under
`checkpoint_metadata/checkpoint-3000/`. They document the trained artifact but are
not sufficient to run inference without the three tensor shards.

Tensor shard checksums from the retained local checkpoint:

| file | bytes | SHA256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 4,986,649,584 | `cbcaea5ee88f1e0f1465043920a2647c67e7de17d24adfd1c477742a6168edec` |
| `model-00002-of-00003.safetensors` | 4,970,792,616 | `c295637953dea7bbbaade93d87fe7887a5378d91cbea0d3629a7370344183cc8` |
| `model-00003-of-00003.safetensors` | 2,618,758,696 | `f139d77b28ccb1e9eceedde9c75aa92ef6b8df6cb1554fdad64ee1bc58a8bb05` |

Recommended publication options are Hugging Face Hub, institution-managed storage,
or a cloud-drive link with a checksum manifest. Git LFS is technically possible but
is not recommended for this 12 GB artifact because of storage and bandwidth quotas.
