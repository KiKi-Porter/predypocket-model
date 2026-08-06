"""Variable-residue-length collation for Dynamic PreDyPocket."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class CollateError(ValueError):
    pass


def dynamic_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise CollateError("Cannot collate an empty sample list")
    batch_size = len(samples)
    residue_counts = [int(np.asarray(sample["sequence"]).shape[0]) for sample in samples]
    max_residues = max(residue_counts)
    coordinates = np.zeros((batch_size, 11, max_residues, 4, 3), dtype=np.float32)
    sequence = np.zeros((batch_size, max_residues), dtype=np.int32)
    residue_mask = np.zeros((batch_size, max_residues), dtype=bool)
    label = np.zeros((batch_size, max_residues), dtype=np.float32)
    delta_volume = np.zeros((batch_size, max_residues), dtype=np.float32)
    closed_mask = np.zeros((batch_size, max_residues), dtype=bool)
    training_mask = np.zeros((batch_size, max_residues), dtype=bool)
    time_offsets = np.zeros((batch_size, 11), dtype=np.float32)

    metadata_fields = (
        "protein_id",
        "replica",
        "segment_name",
        "anchor_time_ps",
        "sample_id",
    )
    metadata = {name: [] for name in metadata_fields}
    for batch_index, (sample, residue_count) in enumerate(zip(samples, residue_counts)):
        sample_coordinates = np.asarray(sample["coordinates"], dtype=np.float32)
        if sample_coordinates.shape != (11, residue_count, 4, 3):
            raise CollateError(
                f"Sample coordinates have shape {sample_coordinates.shape}, "
                f"expected {(11, residue_count, 4, 3)}"
            )
        coordinates[batch_index, :, :residue_count] = sample_coordinates
        sequence[batch_index, :residue_count] = np.asarray(sample["sequence"], np.int32)
        residue_mask[batch_index, :residue_count] = np.asarray(
            sample["residue_mask"], bool
        )
        label[batch_index, :residue_count] = np.asarray(sample["label"], np.float32)
        delta_volume[batch_index, :residue_count] = np.asarray(
            sample["delta_volume"], np.float32
        )
        closed_mask[batch_index, :residue_count] = np.asarray(
            sample["closed_mask"], bool
        )
        training_mask[batch_index, :residue_count] = np.asarray(
            sample["training_mask"], bool
        )
        time_offsets[batch_index] = np.asarray(sample["time_offsets_ps"], np.float32)
        for name in metadata_fields:
            metadata[name].append(sample[name])

    training_mask &= residue_mask
    batch = {
        "coordinates": coordinates,
        "sequence": sequence,
        "residue_mask": residue_mask,
        "time_offsets_ps": time_offsets,
        "label": label,
        "delta_volume": delta_volume,
        "closed_mask": closed_mask,
        "training_mask": training_mask,
    }
    batch.update(metadata)
    return batch


def iter_batches(
    dataset: Any,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterable[dict[str, Any]]:
    if batch_size < 1:
        raise CollateError("batch_size must be positive")
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield dynamic_collate([dataset[int(index)] for index in indices[start : start + batch_size]])

