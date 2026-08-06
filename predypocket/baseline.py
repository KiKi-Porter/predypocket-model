"""Static controls retained for legacy and supplementary evaluations."""

from __future__ import annotations

import tensorflow as tf

from .model import DynamicPreDyPocket, StaticAnchorPreDyPocket


class PublishedPreDyPocketZeroShot(tf.keras.Model):
    """Released PreDyPocket transfer baseline; never a trained static control."""

    def __init__(self, dynamic_model: DynamicPreDyPocket):
        super().__init__(name="published_predypocket_zero_shot", dtype=tf.float32)
        self.dynamic_model = dynamic_model

    def call(
        self,
        coords: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
        time_offsets_ps: tf.Tensor | None = None,
        spatial_frames: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        del time_offsets_ps
        return self.dynamic_model.anchor_only_logits(
            coords,
            sequence,
            residue_mask,
            spatial_frames=spatial_frames,
            training=training,
        )


class HeadMatchedAnchorBaseline(tf.keras.Model):
    """Trainable anchor-only control sharing a PreDyPocket core and classifier head."""

    def __init__(self, dynamic_model: DynamicPreDyPocket):
        super().__init__(name="head_matched_anchor_baseline", dtype=tf.float32)
        self.dynamic_model = dynamic_model

    @property
    def static_model(self):
        return self.dynamic_model.static_model

    @property
    def spatial_encoder(self):
        return self.dynamic_model.spatial_encoder

    def set_spatial_encoder_trainable(self, trainable: bool) -> None:
        self.dynamic_model.set_spatial_encoder_trainable(trainable)

    def call(
        self,
        coords: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
        time_offsets_ps: tf.Tensor | None = None,
        spatial_frames: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        del time_offsets_ps
        return self.dynamic_model.anchor_only_logits(
            coords,
            sequence,
            residue_mask,
            spatial_frames=spatial_frames,
            training=training,
        )
