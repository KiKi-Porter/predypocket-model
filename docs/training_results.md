# Training results

This repository was validated on an internal `dynamic_data` collection using
expanded residue labels.

Configuration:

- MD representative frames: chronological 3:3:4, ten frames per trajectory.
- Geometric encoder: PocketMiner/GVP representation.
- Training rows after filtering: 1,625.
- Validation rows after filtering: 181.
- Residue cutoff used to avoid GVP OOM during training: `--max-residues 1000`.
- Positive class weight: 14.160815.
- GPU mode: manual replicas over four visible GPUs.

Validation summary:

| Epoch | Train loss | Val loss | Val ROC-AUC | Val PR-AUC |
|---:|---:|---:|---:|---:|
| 0 | 0.631631 | 0.617652 | 0.700145 | 0.148146 |
| 1 | 0.624605 | 0.603696 | 0.700722 | 0.148740 |
| 2 | 0.624863 | 0.595843 | 0.700991 | 0.149346 |
| 3 | 0.622691 | 0.591521 | 0.701070 | 0.149641 |
| 4 | 0.621323 | 0.596785 | 0.701436 | 0.149961 |
| 5 | 0.622108 | 0.604518 | 0.701724 | 0.150165 |
| 6 | 0.623239 | 0.598963 | 0.701866 | 0.150366 |
| 7 | 0.622404 | 0.595461 | 0.702006 | 0.150621 |
| 8 | 0.620574 | 0.592096 | 0.702146 | 0.150801 |
| 9 | 0.621359 | 0.606402 | 0.702444 | 0.150856 |

Recommended checkpoint choice:

- Lowest validation loss: epoch 3.
- Highest validation ROC-AUC and PR-AUC: epoch 9.

The validation positive ratio after filtering was approximately 6.34%, so the
reported PR-AUC values should be interpreted against a low-prevalence baseline.
