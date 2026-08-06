"""Masked weighted binary cross entropy operating on raw logits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class MaskedLossResult:
    loss: tf.Tensor | None
    valid: bool
    effective_count: int
    positive_count: int
    negative_count: int
    reason: str | None = None


@dataclass(frozen=True)
class PosWeightStats:
    positive_train_count: int
    negative_train_count: int
    raw_pos_weight: float
    used_pos_weight: float
    max_pos_weight: float
    source_split: str = "train"
    source_system_count: int = 0


class PosWeightError(ValueError):
    pass


def masked_weighted_bce_with_logits(
    logits: tf.Tensor,
    labels: tf.Tensor,
    residue_mask: tf.Tensor,
    training_mask: tf.Tensor,
    pos_weight: float | tf.Tensor,
) -> MaskedLossResult:
    logits = tf.cast(tf.convert_to_tensor(logits), tf.float32)
    labels = tf.cast(labels, logits.dtype)
    residue_mask = tf.cast(residue_mask, tf.bool)
    training_mask = tf.cast(training_mask, tf.bool)
    tf.debugging.assert_equal(tf.shape(logits), tf.shape(labels))
    tf.debugging.assert_equal(tf.shape(logits), tf.shape(residue_mask))
    tf.debugging.assert_equal(tf.shape(logits), tf.shape(training_mask))
    effective_mask = residue_mask & training_mask
    effective_count = int(tf.reduce_sum(tf.cast(effective_mask, tf.int32)).numpy())
    if effective_count == 0:
        return MaskedLossResult(
            loss=None,
            valid=False,
            effective_count=0,
            positive_count=0,
            negative_count=0,
            reason="no_effective_supervision",
        )
    element_loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        pos_weight=tf.cast(pos_weight, logits.dtype),
    )
    mask_float = tf.cast(effective_mask, element_loss.dtype)
    loss = tf.reduce_sum(element_loss * mask_float) / tf.maximum(
        tf.reduce_sum(mask_float), 1.0
    )
    selected_labels = tf.boolean_mask(labels, effective_mask)
    positive_count = int(tf.reduce_sum(tf.cast(selected_labels >= 0.5, tf.int32)).numpy())
    return MaskedLossResult(
        loss=loss,
        valid=True,
        effective_count=effective_count,
        positive_count=positive_count,
        negative_count=effective_count - positive_count,
    )


def compute_train_pos_weight(
    samples: Iterable[Mapping[str, Any]],
    split: str,
    max_pos_weight: float = 20.0,
) -> PosWeightStats:
    """Count only the explicitly identified training split."""

    if split != "train":
        raise PosWeightError(
            f"pos_weight may only be computed from split='train', received {split!r}"
        )
    if max_pos_weight <= 0:
        raise PosWeightError("max_pos_weight must be positive")
    positive = 0
    negative = 0
    system_ids: set[str] = set()
    for sample in samples:
        sample_split = sample.get("split")
        if sample_split is not None and sample_split != "train":
            raise PosWeightError(
                f"pos_weight input contains non-train sample from {sample_split!r}"
            )
        label_key = "labels" if "labels" in sample else "label"
        if label_key not in sample:
            raise PosWeightError("Training supervision is missing labels")
        labels = np.asarray(sample[label_key], dtype=np.float32)
        mask = np.asarray(sample["residue_mask"], dtype=bool) & np.asarray(
            sample["training_mask"], dtype=bool
        )
        if labels.shape != mask.shape:
            raise PosWeightError("Training labels and effective mask differ in shape")
        selected = labels[mask]
        if not np.all(np.isin(selected, (0.0, 1.0))):
            raise PosWeightError("Training supervision contains non-binary labels")
        positive += int(np.sum(selected >= 0.5))
        negative += int(np.sum(selected < 0.5))
        if sample.get("system_id") is not None:
            system_ids.add(str(sample["system_id"]))
    if positive == 0:
        raise PosWeightError("Training split contains no positive supervised residue")
    raw = negative / positive
    used = min(raw, float(max_pos_weight))
    return PosWeightStats(
        positive_train_count=positive,
        negative_train_count=negative,
        raw_pos_weight=float(raw),
        used_pos_weight=float(used),
        max_pos_weight=float(max_pos_weight),
        source_system_count=len(system_ids),
    )


def pos_weight_from_dataset(dataset: Any, max_pos_weight: float = 20.0) -> PosWeightStats:
    if getattr(dataset, "split", None) != "train":
        raise PosWeightError("Dataset must be the current fold's train split")
    return compute_train_pos_weight(
        dataset.iter_supervision(), split="train", max_pos_weight=max_pos_weight
    )
