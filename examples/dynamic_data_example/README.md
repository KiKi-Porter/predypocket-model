# Toy MD Example

This folder contains a tiny synthetic dynamic-data example used for smoke
testing both the data-preparation and inference code paths.

```text
examples/dynamic_data_example/
└── TOY/
    └── TOY_TOY_1.LIG_B_101/
        ├── aa_traj.pdb
        └── complex_reference.pdb
```

- `aa_traj.pdb` is a 12-frame multi-model PDB trajectory.
- `complex_reference.pdb` is the final/reference complex structure used for
  ligand-contact labels.
- The protein contains 8 alanine residues on chain A.
- The ligand is `LIG` on chain B, residue 101, near the middle residues.

The data are deliberately small and synthetic. They are not a benchmark and
should not be used to evaluate model quality. The complete model checkpoint is
included under `weights/`; see the top-level `README.md` and
`docs/weight_manifest.md`.

Run representative-frame sampling:

```bash
PYTHONPATH=src python -u src/sample_predypocket_frames.py \
  --system-dir examples/dynamic_data_example/TOY/TOY_TOY_1.LIG_B_101 \
  --out-prefix outputs/toy_sample
```

Prepare one local-contact training sample:

```bash
PYTHONPATH=src python -u src/prepare_predypocket_data.py \
  --dynamic-root examples/dynamic_data_example \
  --out-dir outputs/toy_prepared \
  --label-methods local_contact \
  --limit 1 \
  --force \
  --workers 1
```

Run inference with the bundled multi-model PDB trajectory:

```bash
MODEL_CHECKPOINT=weights/predypocket_model \
bash scripts/predict_md.sh \
  examples/dynamic_data_example/TOY/TOY_TOY_1.LIG_B_101 \
  outputs/toy_inference/predypocket_toy
```

The wrapper automatically uses `aa_traj.pdb` as the trajectory and
`complex_reference.pdb` as the matching topology. The output prefix produces
the following files:

```text
outputs/toy_inference/predypocket_toy_scores.csv
outputs/toy_inference/predypocket_toy_scores.npy
outputs/toy_inference/predypocket_toy_pockets.csv
outputs/toy_inference/predypocket_toy_selected_frames.npy
outputs/toy_inference/predypocket_toy_residue_keys.npy
```

To call the Python inference entry point directly:

```bash
PYTHONPATH=src python -u src/predypocket_predict.py \
  --trajectory examples/dynamic_data_example/TOY/TOY_TOY_1.LIG_B_101/aa_traj.pdb \
  --checkpoint weights/predypocket_model \
  --out-prefix outputs/toy_inference/predypocket_toy_direct
```

The original files can be regenerated with:

```bash
python examples/dynamic_data_example/make_toy_md.py
```
