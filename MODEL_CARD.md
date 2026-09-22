# Model card

## Model description

PreDyPocket is a PocketMiner-based model for residue-level pocket prediction
from MD trajectories. It uses ten chronological representative conformations
selected with a 3:3:4 early/middle/late split and fuses the resulting temporal
summary with the final reference frame.

## Intended use

- Research prediction of ligandable or cryptic pocket residues from MD trajectories.
- Method development around MD trajectory-based pocket prediction.
- Batch scoring of MD systems after local validation.

## Out-of-scope use

- Clinical diagnosis, clinical decision support, or therapeutic decision-making.
- Direct replacement for experimental binding-site validation.
- Claims about ligand binding without additional structural or biochemical evidence.

## Inputs

- MD trajectory readable by MDTraj.
- Matching topology when required by the trajectory format.
- Optional complex coordinates for training label generation.

## Outputs

- Per-residue pocket probability scores.
- Clustered pocket candidates from thresholded residue scores.

## Training data

The included training code expects user-supplied MD trajectories. The internal
validation run used expanded labels from local ligand heavy-atom contacts and
homolog transferred ligand-contact positives.

## Limitations

- Performance depends on MD trajectory quality, topology consistency, and label quality.
- The current release uses a lightweight temporal model with PocketMiner/GVP
  geometric representations.
- Large proteins may need memory filtering or precomputed backbone embeddings.
