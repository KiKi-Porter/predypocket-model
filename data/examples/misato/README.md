# Included examples

This directory contains exactly two preprocessed MISATO cache systems copied from the source repository:

- `10gs`: ten input frames, 100 ps spacing, source training split.
- `1alw`: ten input frames, 100 ps spacing, source test split.

Each system contains backbone coordinates (`backbone_coordinates_input.npy`), residue sequence indices (`sequence.npy`), a valid-residue mask (`valid_residue_mask.npy`), frame/time metadata, residue mapping, and the original processed labels and pocket-volume artifacts. The raw topology, trajectory, and full manifest are intentionally omitted. `example_systems.csv` is a release-local index and is not a replacement for the official MISATO manifest.

Coordinates are stored in the same units and atom order (`N, CA, C, O`) expected by the copied model code. The model input tensor has shape `[batch, 10, residues, 4, 3]`; padded or invalid residues must be masked before interpretation.

