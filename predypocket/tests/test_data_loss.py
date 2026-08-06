from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from predypocket.backbone import BACKBONE_ATOM_ORDER
from predypocket.collate import dynamic_collate
from predypocket.dataset import DynamicDatasetError, assemble_dynamic_sample
from predypocket.losses import (
    PosWeightError,
    compute_train_pos_weight,
    masked_weighted_bce_with_logits,
)


def _in_memory_sample(residue_count: int = 5, sample_valid: bool = True):
    row = {
        "protein_id": "protein_a",
        "replica": "R1",
        "segment_name": "0-50",
        "anchor_time_ps": "1000",
        "sample_id": "sample_a",
    }
    labels = {
        "sample_ids": np.asarray(["sample_a"]),
        "sample_valid_mask": np.asarray([sample_valid]),
        "valid_residue_mask": np.ones(residue_count, dtype=bool),
        "label_20ns_ge20": np.asarray(
            [[index % 2 for index in range(residue_count)]], dtype=bool
        ),
        "delta_volume_20ns": np.arange(residue_count, dtype=np.float32)[None, :],
        "closed_mask_1ns": np.ones((1, residue_count), dtype=bool),
        "training_residue_mask": np.ones((1, residue_count), dtype=bool),
        "anchor_volume": np.full((1, residue_count), 123.0),
        "future_max_volume_20ns": np.full((1, residue_count), 999.0),
    }
    coordinates = np.zeros((11, residue_count, 4, 3), dtype=np.float32)
    sequence = np.arange(residue_count, dtype=np.int32)
    mask = np.ones(residue_count, dtype=bool)
    offsets = np.arange(-1000, 1, 100, dtype=np.float32)
    return assemble_dynamic_sample(
        row, labels, 0, coordinates, sequence, mask, offsets
    )


def test_26_dataset_sample_shape_is_fixed_history():
    sample = _in_memory_sample()
    assert sample["coordinates"].shape == (11, 5, 4, 3)
    assert sample["sequence"].shape == (5,)


def test_27_dataset_returns_only_input_coordinate_history():
    sample = _in_memory_sample()
    assert "future_coordinates" not in sample
    assert sample["coordinates"].shape[0] == 11


def test_28_backbone_order_is_n_ca_c_o():
    assert BACKBONE_ATOM_ORDER == ("N", "CA", "C", "O")


def test_29_label_volume_statistics_do_not_become_inputs():
    sample = _in_memory_sample()
    assert "anchor_volume" not in sample
    assert "future_max_volume" not in sample
    assert "pocket_volume" not in sample


def test_30_dataset_rejects_sample_valid_false():
    with pytest.raises(DynamicDatasetError, match="Excluded"):
        _in_memory_sample(sample_valid=False)


def test_31_collate_pads_different_residue_lengths():
    batch = dynamic_collate([_in_memory_sample(3), _in_memory_sample(5)])
    assert batch["coordinates"].shape == (2, 11, 5, 4, 3)
    assert batch["sequence"].shape == (2, 5)


def test_32_collate_padding_masks_are_false():
    batch = dynamic_collate([_in_memory_sample(3), _in_memory_sample(5)])
    assert not np.any(batch["residue_mask"][0, 3:])
    assert not np.any(batch["training_mask"][0, 3:])


def test_33_masked_bce_matches_manual_value():
    logits = tf.constant([[0.0, 1.0, 100.0]], dtype=tf.float32)
    labels = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
    residue_mask = tf.constant([[True, True, False]])
    training_mask = tf.constant([[True, True, True]])
    result = masked_weighted_bce_with_logits(
        logits, labels, residue_mask, training_mask, pos_weight=2.0
    )
    expected = (
        tf.nn.weighted_cross_entropy_with_logits(
            labels=labels[:, :2], logits=logits[:, :2], pos_weight=2.0
        )
        .numpy()
        .mean()
    )
    assert result.valid
    assert np.isclose(float(result.loss.numpy()), expected, atol=1e-7)


def test_34_padding_does_not_enter_loss():
    base = masked_weighted_bce_with_logits(
        [[0.0, 1.0, 100.0]],
        [[0.0, 1.0, 0.0]],
        [[True, True, False]],
        [[True, True, True]],
        2.0,
    )
    changed = masked_weighted_bce_with_logits(
        [[0.0, 1.0, -100.0]],
        [[0.0, 1.0, 1.0]],
        [[True, True, False]],
        [[True, True, True]],
        2.0,
    )
    assert float(base.loss.numpy()) == float(changed.loss.numpy())


def test_35_empty_effective_mask_is_invalid_not_fake_loss():
    result = masked_weighted_bce_with_logits(
        [[0.0]], [[0.0]], [[True]], [[False]], 1.0
    )
    assert not result.valid
    assert result.loss is None
    assert result.reason == "no_effective_supervision"


def test_36_pos_weight_uses_train_counts_and_caps_at_20():
    samples = [
        {
            "label": np.asarray([1, 0, 0, 0, 0], np.float32),
            "residue_mask": np.ones(5, bool),
            "training_mask": np.ones(5, bool),
        }
    ]
    stats = compute_train_pos_weight(samples, split="train", max_pos_weight=3.0)
    assert stats.raw_pos_weight == 4.0
    assert stats.used_pos_weight == 3.0


def test_37_validation_cannot_supply_pos_weight():
    with pytest.raises(PosWeightError, match="train"):
        compute_train_pos_weight([], split="validation")


def test_38_test_cannot_supply_pos_weight():
    with pytest.raises(PosWeightError, match="train"):
        compute_train_pos_weight([], split="test")

