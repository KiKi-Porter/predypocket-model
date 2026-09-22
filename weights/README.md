# Model weights

The TensorFlow checkpoint files are included in this repository. No separate
weight download is required.

Repository layout:

```text
weights/
├── predypocket_model.data-00000-of-00001
└── predypocket_model.index
```

Use the checkpoint prefix, not an individual file, when running commands:

```text
weights/predypocket_model
```

SHA256 checksums are listed in `docs/weight_manifest.md`.
