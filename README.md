# PreDyPocket

PreDyPocket is a trajectory-informed deep learning model for residue-level prediction of dynamic precursor pocket signals from molecular-dynamics (MD) simulations. Given a short history of protein backbone coordinates, the model assigns one pocket-association score to each residue.

## Overview

The model combines:

- a shared geometric vector perceptron (GVP) encoder for protein backbone structure;
- temporal transition features computed from consecutive MD frames;
- learned time embeddings;
- a single-layer unidirectional GRU;
- residue-wise temporal attention; and
- a reference-anchored gated fusion module.

The network returns raw logits. Apply a sigmoid function to obtain residue-level probabilities. Pocket regions can then be formed by grouping neighboring residues according to the desired application-specific spatial criteria.

Two model interfaces are provided:

- `DynamicPreDyPocket`: uses the complete coordinate history and temporal modules;
- `StaticAnchorPreDyPocket`: uses only the reference frame and provides a static control model.

The released initializer is a TensorFlow checkpoint at `models/predypocket_initializer`. It initializes the shared structural encoder and classifier before task-specific training.

## Requirements

The reference environment is Linux with Python 3.9.9. The validated dependency versions are:

- TensorFlow 2.6.2
- NumPy 1.19.5
- SciPy 1.7.3
- pandas 1.4.0
- scikit-learn 1.0.2
- MDTraj 1.9.7
- PyYAML 6.0
- tqdm 4.62.3

A CUDA-capable GPU is optional for inference and recommended for training.

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate predypocket
```

Alternatively, install the pinned Python dependencies in an existing Python 3.9 environment:

```bash
python -m pip install -r requirements.txt
```

## Input and output specification

The dynamic model expects four tensors:

| Input | Shape | Description |
| --- | --- | --- |
| `coordinates` | `[batch, frames, residues, 4, 3]` | Backbone coordinates in atom order `N, CA, C, O` |
| `sequence` | `[batch, residues]` | Integer residue sequence indices |
| `residue_mask` | `[batch, residues]` | Boolean mask for valid residues |
| `time_offsets_ps` | `[batch, frames]` | Time offset of each frame in picoseconds |

The output has shape `[batch, residues]` and contains one raw logit per residue. Padding and invalid residues must be excluded with `residue_mask` before scores are interpreted.

The standard ten-frame input contract uses frames at `-900, -800, ..., 0` ps, with the final frame as the reference. The protocol-v2 configuration uses its own eleven-frame contract; use the configuration and data schema together and do not mix the two protocols.

## Quick start

The following command builds the model and runs a deterministic synthetic forward pass without a dataset:

```bash
python - <<'PY'
import numpy as np
from predypocket.model import DynamicPreDyPocket, synthetic_model_inputs

coordinates, sequence, residue_mask, time_offsets_ps = synthetic_model_inputs(
    batch_size=1, residue_count=8, frame_count=10
)
model = DynamicPreDyPocket(input_frame_count=10)
logits = model(
    coordinates,
    sequence,
    residue_mask,
    time_offsets_ps=time_offsets_ps,
    training=False,
)
assert tuple(logits.shape) == (1, 8)
print("logits:", logits.shape)
PY
```

## Inference

The following example loads a preprocessed ten-frame system, initializes the model, and converts logits to residue probabilities:

```python
from pathlib import Path

import numpy as np
import tensorflow as tf

from predypocket.checkpoint import load_predypocket_pretrained
from predypocket.model import DynamicPreDyPocket

system_dir = Path("data/examples/misato/pocket_cache_100ps_v3/10gs")
coordinates = np.load(system_dir / "backbone_coordinates_input.npy")[None]
sequence = np.load(system_dir / "sequence.npy")[None]
residue_mask = np.load(system_dir / "valid_residue_mask.npy")[None].astype(bool)
time_offsets_ps = np.arange(-900, 1, 100, dtype=np.float32)[None]

model = DynamicPreDyPocket(input_frame_count=10)
model(
    coordinates,
    sequence,
    residue_mask,
    time_offsets_ps=time_offsets_ps,
    training=False,
)
load_predypocket_pretrained(model, "models/predypocket_initializer")

logits = model(
    coordinates,
    sequence,
    residue_mask,
    time_offsets_ps=time_offsets_ps,
    training=False,
)
probabilities = tf.math.sigmoid(logits).numpy()[0]
probabilities[~residue_mask[0]] = np.nan
print(probabilities)
```

The initializer produces structural scores before dynamic task-specific training. For a trained model, restore the corresponding TensorFlow checkpoint with the checkpoint utilities in `predypocket.checkpoint`.

## Training

Training uses preprocessed samples containing backbone coordinates, sequence indices, residue masks, time offsets, and residue labels. The labels are consumed by the loss function and are not model inputs. Prepare a manifest and cache directory that follow the schema in the selected configuration.

### MISATO workflow

Use the MISATO entry point for the ten-frame, 100 ps history contract:

```bash
python scripts/misato/train_dynamic_pocket_v1.py \
  --manifest /path/to/misato_manifest.csv \
  --data-dir /path/to/misato_label_data \
  --output-dir outputs/predypocket_misato_dynamic \
  --pretrained-checkpoint models/predypocket_initializer \
  --model-type dynamic \
  --temporal-mode on \
  --encoder-frozen \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-epochs 50 \
  --seed 42
```

For the static control model, set `--model-type static_anchor` and omit `--temporal-mode`.

### Protocol-v2 workflow

The protocol-v2 training and evaluation entry points use the eleven-frame configuration:

```bash
python scripts/predypocket/train_protocol_v2.py \
  --config configs/predypocket_protocol_v2.json \
  --fold 0 \
  --model-variant dynamic \
  --device cpu
```

Run evaluation on a completed fold with:

```bash
python scripts/predypocket/evaluate_protocol_v2.py \
  --config configs/predypocket_protocol_v2.json \
  --fold 0 \
  --model-variant dynamic \
  --split validation \
  --device cpu
```

The five-fold launcher is available at `scripts/predypocket/run_protocol_v2_5fold.py`. Background and status helpers are provided in the same directory.

## Evaluation and post-processing

Evaluation selects decision thresholds on validation data and applies the frozen threshold to the test split. The package reports average precision, PR-AUC, ROC-AUC, F1, precision, recall, and per-protein summaries. Test evaluation must use checkpoints selected without test-label inspection.

The model output is residue-level; it is not a pocket mesh or an atom-level binding pose. A downstream application should define its own residue threshold and spatial clustering rule, then report those choices together with the model checkpoint and configuration.

## Repository layout

```text
PreDyPocket/
├── predypocket/       # Model, data interfaces, losses, metrics, and checkpoint tools
├── src/               # GVP geometric layers used by the model
├── scripts/misato/    # MISATO training, evaluation, and cache preparation
├── scripts/predypocket/ # Protocol-v2 training, evaluation, and utility scripts
├── configs/           # Versioned task and protocol configurations
├── models/            # TensorFlow initializer checkpoint
├── data/examples/     # Small preprocessed systems for inference checks
├── environment.yml    # Conda environment specification
├── requirements.txt   # Pinned pip dependencies
├── CITATION.cff       # Software citation metadata
└── LICENSE
```

## Reproducibility

For a reproducible experiment, record the repository commit, operating system, Python and dependency versions, dataset and preprocessing version, manifest checksum, fold definition, random seed, training command, checkpoint checksum, validation threshold, and configuration file. Keep generated checkpoints and evaluation reports under a versioned output directory.

## License

PreDyPocket is distributed under the MIT License. The GVP components retain their applicable upstream attribution and license notices.

## Citation

Please cite this software release using the metadata in `CITATION.cff`.

