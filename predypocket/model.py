"""Dynamic PreDyPocket model composed around the released TensorFlow GVP."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gvp import vs_concat  # noqa: E402
from models import MQAModel  # noqa: E402


DEFAULT_NODE_FEATURES = (8, 50)
DEFAULT_EDGE_FEATURES = (1, 32)
DEFAULT_HIDDEN_DIM = (16, 100)
DEFAULT_GVP_LAYERS = 4
DEFAULT_K_NEIGHBORS = 30


def make_static_predypocket(dropout: float = 0.1) -> MQAModel:
    """Construct the exact architecture documented for `models/predypocket_initializer`."""

    previous_policy = tf.keras.mixed_precision.global_policy().name
    if previous_policy != "float32":
        tf.keras.mixed_precision.set_global_policy("float32")
    try:
        return MQAModel(
            node_features=DEFAULT_NODE_FEATURES,
            edge_features=DEFAULT_EDGE_FEATURES,
            hidden_dim=DEFAULT_HIDDEN_DIM,
            num_layers=DEFAULT_GVP_LAYERS,
            k_neighbors=DEFAULT_K_NEIGHBORS,
            dropout=dropout,
            ablate_sidechain_vectors=True,
        )
    finally:
        if previous_policy != "float32":
            tf.keras.mixed_precision.set_global_policy(previous_policy)


class PreDyPocketSpatialEncoder(tf.keras.layers.Layer):
    """Expose the legacy classifier-preceding scalar residue representation."""

    def __init__(self, static_model: MQAModel, name: str = "shared_gvp_encoder"):
        super().__init__(name=name)
        self.static_model = static_model
        self.d_static = int(static_model.hs)

    def call(
        self,
        coordinates: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        model = self.static_model
        features_v, features_e, edge_index = model.features(coordinates, residue_mask)
        sequence_embedding = model.W_s(sequence)
        features_v = vs_concat(features_v, sequence_embedding, model.nv, 0)
        hidden_v = model.W_v(features_v)
        hidden_e = model.W_e(features_e)
        hidden_v = model.encoder(
            hidden_v,
            hidden_e,
            edge_index,
            residue_mask,
            train=training,
        )
        scalar = model.W_V_out(hidden_v)
        return scalar * tf.cast(residue_mask[..., None], scalar.dtype)


def classify_predypocket_logits(
    static_model: MQAModel,
    representation: tf.Tensor,
    training: bool = False,
) -> tf.Tensor:
    """Apply the released residue classifier while bypassing its final sigmoid."""

    if not static_model.dense.built:
        static_model.dense(representation, training=False)
    hidden = representation
    classifier_layers = static_model.dense.layers
    if not classifier_layers:
        raise RuntimeError("Legacy classifier has not been built")
    for layer in classifier_layers[:-1]:
        if isinstance(layer, tf.keras.layers.Dropout):
            hidden = layer(hidden, training=training)
        else:
            hidden = layer(hidden)
    final_layer = classifier_layers[-1]
    if not isinstance(final_layer, tf.keras.layers.Dense):
        raise TypeError("Legacy classifier final layer is not Dense")
    if not final_layer.built:
        final_layer.build(hidden.shape)
    logits = tf.linalg.matmul(hidden, final_layer.kernel) + final_layer.bias
    return tf.squeeze(logits, axis=-1)


def set_predypocket_encoder_trainable(
    static_model: MQAModel, trainable: bool
) -> None:
    """Freeze or unfreeze the shared GVP path while leaving its classifier trainable."""

    for layer in (
        static_model.features,
        static_model.W_s,
        static_model.W_v,
        static_model.W_e,
        static_model.encoder,
        static_model.W_V_out,
    ):
        layer.trainable = trainable
    static_model.dense.trainable = True


def encode_anchor_batch(
    spatial_encoder: PreDyPocketSpatialEncoder,
    coordinates: tf.Tensor,
    sequence: tf.Tensor,
    residue_mask: tf.Tensor,
    training: bool = False,
) -> tf.Tensor:
    """Encode one frame per system after removing padded residues from each graph."""

    coordinates = tf.cast(tf.convert_to_tensor(coordinates), tf.float32)
    sequence = tf.cast(tf.convert_to_tensor(sequence), tf.int32)
    residue_mask = tf.cast(tf.convert_to_tensor(residue_mask), tf.bool)
    tf.debugging.assert_rank(coordinates, 4)
    tf.debugging.assert_rank(sequence, 2)
    tf.debugging.assert_rank(residue_mask, 2)
    shape = tf.shape(coordinates)
    batch_size, residue_count, atom_count, coordinate_dim = tf.unstack(shape)
    tf.debugging.assert_equal(batch_size, tf.shape(sequence)[0])
    tf.debugging.assert_equal(batch_size, tf.shape(residue_mask)[0])
    tf.debugging.assert_equal(residue_count, tf.shape(sequence)[1])
    tf.debugging.assert_equal(residue_count, tf.shape(residue_mask)[1])
    tf.debugging.assert_equal(atom_count, 4)
    tf.debugging.assert_equal(coordinate_dim, 3)
    tf.debugging.assert_all_finite(
        coordinates, "Static anchor coordinates contain NaN or Inf"
    )

    encoded_systems: list[tf.Tensor] = []
    system_values = zip(
        tf.unstack(coordinates, axis=0),
        tf.unstack(sequence, axis=0),
        tf.unstack(residue_mask, axis=0),
    )
    for system_coords, system_sequence, system_mask in system_values:
        valid_indices = tf.cast(tf.where(system_mask)[:, 0], tf.int32)
        tf.debugging.assert_positive(
            tf.size(valid_indices), "A system has no valid residues"
        )
        valid_coords = tf.gather(system_coords, valid_indices)[None, ...]
        valid_sequence = tf.gather(system_sequence, valid_indices)[None, ...]
        valid_count = tf.shape(valid_sequence)[1]
        encoded_valid = spatial_encoder(
            valid_coords,
            valid_sequence,
            tf.ones([1, valid_count], dtype=tf.float32),
            training=training,
        )[0]
        encoded_systems.append(
            tf.scatter_nd(
                valid_indices[:, None],
                encoded_valid,
                [residue_count, spatial_encoder.d_static],
            )
        )
    return tf.stack(encoded_systems, axis=0)


class StaticAnchorPreDyPocket(tf.keras.Model):
    """Independent frame-9 PreDyPocket trained on MISATO Dynamic Pocket v1."""

    model_type = "static_anchor"
    anchor_frame_index = 9

    def __init__(
        self,
        static_model: MQAModel | None = None,
        dropout: float = 0.1,
        name: str = "static_anchor_predypocket",
    ):
        super().__init__(name=name, dtype=tf.float32)
        self.static_model = static_model or make_static_predypocket(dropout=dropout)
        self.d_static = int(self.static_model.hs)
        self.spatial_encoder = PreDyPocketSpatialEncoder(self.static_model)
        self._spatial_frozen = False

    def set_spatial_encoder_trainable(self, trainable: bool) -> None:
        set_predypocket_encoder_trainable(self.static_model, trainable)
        self._spatial_frozen = not trainable

    def classify_logits(
        self, representation: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        return classify_predypocket_logits(
            self.static_model, representation, training=training
        )

    def call(
        self,
        coords: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Return one raw binary logit per residue from frame 9 coordinates only."""

        spatial_training = bool(training) and not self._spatial_frozen
        representation = encode_anchor_batch(
            self.spatial_encoder,
            coords,
            sequence,
            residue_mask,
            training=spatial_training,
        )
        logits = self.classify_logits(representation, training=bool(training))
        mask = tf.cast(tf.convert_to_tensor(residue_mask), tf.bool)
        return tf.where(mask, logits, tf.zeros_like(logits))


class DynamicPreDyPocket(tf.keras.Model):
    """Predict future residue pocket opening from a configurable past history."""

    def __init__(
        self,
        static_model: MQAModel | None = None,
        input_frame_count: int | None = 10,
        time_embedding_dim: int = 16,
        temporal_input_dim: int = 128,
        gru_hidden_dim: int = 128,
        attention_hidden_dim: int = 64,
        dropout: float = 0.1,
        temporal_mode: str = "on",
        name: str = "predypocket",
        spatial_frame_chunk_size: int = 1,
    ):
        super().__init__(name=name)
        if input_frame_count is not None and input_frame_count < 2:
            raise ValueError("Dynamic PreDyPocket requires at least two input frames")
        if gru_hidden_dim != 128:
            raise ValueError("Dynamic PreDyPocket GRU hidden size must be 128")
        if temporal_mode not in {"on", "off"}:
            raise ValueError("temporal_mode must be 'on' or 'off'")
        if (
            not isinstance(spatial_frame_chunk_size, int)
            or isinstance(spatial_frame_chunk_size, bool)
            or spatial_frame_chunk_size < 1
        ):
            raise ValueError("spatial_frame_chunk_size must be a positive integer")
        self.input_frame_count = input_frame_count
        self.temporal_mode = temporal_mode
        self.spatial_frame_chunk_size = spatial_frame_chunk_size
        self.time_embedding_dim = time_embedding_dim
        self.temporal_input_dim = temporal_input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.static_model = static_model or make_static_predypocket(dropout=dropout)
        self.d_static = int(self.static_model.hs)
        self.spatial_encoder = PreDyPocketSpatialEncoder(self.static_model)

        self.time_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, name="linear_1"),
                tf.keras.layers.Activation(tf.nn.silu, name="silu"),
                tf.keras.layers.Dense(time_embedding_dim, name="linear_2"),
            ],
            name="time_embedding",
        )
        self.temporal_projection = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(temporal_input_dim, name="linear"),
                tf.keras.layers.LayerNormalization(name="layer_norm"),
                tf.keras.layers.Activation(tf.nn.silu, name="silu"),
                tf.keras.layers.Dropout(dropout, name="dropout"),
            ],
            name="temporal_input_projection",
        )
        self.temporal_gru = tf.keras.layers.GRU(
            gru_hidden_dim,
            return_sequences=True,
            go_backwards=False,
            name="unidirectional_gru",
        )
        self.attention_hidden = tf.keras.layers.Dense(
            attention_hidden_dim, activation="tanh", name="attention_hidden"
        )
        self.attention_score = tf.keras.layers.Dense(1, name="attention_score")
        self.dynamic_projection = tf.keras.layers.Dense(
            self.d_static,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="zero_initialized_dynamic_projection",
        )
        self.gate_linear = tf.keras.layers.Dense(
            self.d_static, name="vector_residual_gate"
        )
        self._spatial_frozen = False

    model_type = "dynamic"

    @property
    def gru_layers(self) -> int:
        return 1

    @property
    def bidirectional(self) -> bool:
        return False

    def set_spatial_encoder_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze only the legacy spatial path, not its classifier."""

        set_predypocket_encoder_trainable(self.static_model, trainable)
        self._spatial_frozen = not trainable

    def classify_logits(self, representation: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply the pretrained classifier while bypassing only its final sigmoid."""

        return classify_predypocket_logits(
            self.static_model, representation, training=training
        )

    def _encode_spatial_frame_chunks(
        self,
        frame_coords: tf.Tensor,
        sequence: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        """Encode independent frames in bounded batches to cap GVP peak memory."""

        residue_count = tf.shape(sequence)[0]

        def encode_chunk(start: tf.Tensor | int, stop: tf.Tensor | int) -> tf.Tensor:
            chunk_coords = frame_coords[start:stop]
            chunk_count = tf.shape(chunk_coords)[0]
            chunk_sequence = tf.broadcast_to(
                sequence[None, :], [chunk_count, residue_count]
            )
            chunk_mask = tf.ones([chunk_count, residue_count], dtype=tf.float32)
            return self.spatial_encoder(
                chunk_coords,
                chunk_sequence,
                chunk_mask,
                training=training,
            )

        static_frame_count = frame_coords.shape[0]
        if static_frame_count is None:
            raise ValueError(
                "The released GVP requires a statically known frame dimension"
            )
        encoded_chunks = [
            encode_chunk(
                start,
                min(start + self.spatial_frame_chunk_size, static_frame_count),
            )
            for start in range(0, static_frame_count, self.spatial_frame_chunk_size)
        ]
        if not encoded_chunks:
            raise ValueError("At least one spatial frame is required")
        return tf.concat(encoded_chunks, axis=0)

    def encode_frames(
        self,
        coords: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Encode valid residues per system and frame with one shared GVP.

        Each system is cropped to its valid residues before graph construction.
        Consequently padded coordinates never enter distance calculations and
        neither systems nor time frames can share graph edges.
        """

        coords = tf.cast(tf.convert_to_tensor(coords), tf.float32)
        sequence = tf.cast(tf.convert_to_tensor(sequence), tf.int32)
        residue_mask = tf.cast(tf.convert_to_tensor(residue_mask), tf.bool)
        tf.debugging.assert_rank(coords, 5)
        tf.debugging.assert_rank(sequence, 2)
        tf.debugging.assert_rank(residue_mask, 2)
        shape = tf.shape(coords)
        batch_size, frame_count, residue_count, atom_count, coordinate_dim = tf.unstack(
            shape
        )
        tf.debugging.assert_greater_equal(frame_count, 2)
        if self.input_frame_count is not None:
            tf.debugging.assert_equal(frame_count, self.input_frame_count)
        tf.debugging.assert_equal(batch_size, tf.shape(sequence)[0])
        tf.debugging.assert_equal(batch_size, tf.shape(residue_mask)[0])
        tf.debugging.assert_equal(residue_count, tf.shape(sequence)[1])
        tf.debugging.assert_equal(residue_count, tf.shape(residue_mask)[1])
        tf.debugging.assert_equal(atom_count, 4)
        tf.debugging.assert_equal(coordinate_dim, 3)
        tf.debugging.assert_all_finite(coords, "Model coordinates contain NaN or Inf")

        spatial_training = bool(training) and not self._spatial_frozen
        encoded_systems: list[tf.Tensor] = []
        system_values = zip(
            tf.unstack(coords, axis=0),
            tf.unstack(sequence, axis=0),
            tf.unstack(residue_mask, axis=0),
        )
        for system_coords, system_sequence, system_mask in system_values:
            valid_indices = tf.cast(tf.where(system_mask)[:, 0], tf.int32)
            tf.debugging.assert_positive(
                tf.size(valid_indices), "A system has no valid residues"
            )
            valid_coords = tf.gather(system_coords, valid_indices, axis=1)
            valid_sequence = tf.gather(system_sequence, valid_indices)
            valid_count = tf.shape(valid_sequence)[0]
            history_scalar = self._encode_spatial_frame_chunks(
                valid_coords[:-1],
                valid_sequence,
                spatial_training,
            )
            anchor_scalar = self.spatial_encoder(
                valid_coords[-1:],
                valid_sequence[None, :],
                tf.ones([1, valid_count], dtype=tf.float32),
                training=spatial_training,
            )
            encoded_valid = tf.concat([history_scalar, anchor_scalar], axis=0)
            residue_major = tf.transpose(encoded_valid, [1, 0, 2])
            padded_residue_major = tf.scatter_nd(
                valid_indices[:, None],
                residue_major,
                [residue_count, frame_count, self.d_static],
            )
            encoded_systems.append(tf.transpose(padded_residue_major, [1, 0, 2]))
        return tf.stack(encoded_systems, axis=0)

    def _normalized_time_offsets(
        self,
        time_offsets_ps: tf.Tensor | None,
        batch_size: tf.Tensor,
        frame_count: tf.Tensor,
    ) -> tf.Tensor:
        if time_offsets_ps is None:
            raise ValueError("time_offsets_ps is required; the model does not infer timing")
        offsets = tf.convert_to_tensor(time_offsets_ps, dtype=tf.float32)
        if offsets.shape.rank == 1:
            tf.debugging.assert_equal(tf.shape(offsets)[0], frame_count)
            offsets = tf.broadcast_to(offsets[None, :], [batch_size, frame_count])
        else:
            tf.debugging.assert_rank(offsets, 2)
            tf.debugging.assert_equal(tf.shape(offsets), [batch_size, frame_count])
        tf.debugging.assert_all_finite(offsets, "Time offsets contain NaN or Inf")
        relative = offsets - offsets[:, -1:]
        scale = tf.maximum(tf.reduce_max(tf.abs(relative), axis=1, keepdims=True), 1.0)
        normalized = relative / scale
        tf.debugging.assert_less_equal(normalized, 0.0)
        tf.debugging.assert_greater_equal(normalized, -1.0)
        return normalized

    def _validated_spatial_frames(
        self,
        spatial_frames: tf.Tensor,
        coords: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
    ) -> tf.Tensor:
        """Validate a representation produced by the frozen spatial encoder."""

        cached = tf.cast(tf.convert_to_tensor(spatial_frames), tf.float32)
        tf.debugging.assert_rank(cached, 4)
        tf.debugging.assert_equal(tf.shape(cached)[0], tf.shape(coords)[0])
        tf.debugging.assert_equal(tf.shape(cached)[1], tf.shape(coords)[1])
        tf.debugging.assert_equal(tf.shape(cached)[2], tf.shape(sequence)[1])
        tf.debugging.assert_equal(tf.shape(cached)[2], tf.shape(residue_mask)[1])
        tf.debugging.assert_equal(tf.shape(cached)[3], self.d_static)
        if self.input_frame_count is not None:
            tf.debugging.assert_equal(tf.shape(cached)[1], self.input_frame_count)
        tf.debugging.assert_all_finite(
            cached, "Cached spatial representations contain NaN or Inf"
        )
        return cached

    def call(
        self,
        coords: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
        time_offsets_ps: tf.Tensor | None = None,
        spatial_frames: tf.Tensor | None = None,
        training: bool = False,
        return_auxiliary: bool = False,
        temporal_mode: str | None = None,
    ) -> tf.Tensor | tuple[tf.Tensor, dict[str, tf.Tensor]]:
        temporal_mode = self.temporal_mode if temporal_mode is None else temporal_mode
        if temporal_mode not in {"on", "off"}:
            raise ValueError("temporal_mode must be 'on' or 'off'")
        if spatial_frames is None:
            scalar_frames = self.encode_frames(
                coords, sequence, residue_mask, training=training
            )
        else:
            scalar_frames = self._validated_spatial_frames(
                spatial_frames, coords, sequence, residue_mask
            )
        residue_mask = tf.convert_to_tensor(residue_mask, dtype=tf.bool)
        batch_size = tf.shape(scalar_frames)[0]
        frame_count = tf.shape(scalar_frames)[1]
        residue_count = tf.shape(scalar_frames)[2]

        delta = tf.concat(
            [
                tf.zeros_like(scalar_frames[:, :1]),
                scalar_frames[:, 1:] - scalar_frames[:, :-1],
            ],
            axis=1,
        )
        abs_delta = tf.abs(delta)
        normalized_time = self._normalized_time_offsets(
            time_offsets_ps, batch_size, frame_count
        )
        time_embedding = self.time_embedding(normalized_time[..., None], training=training)
        time_embedding = tf.cast(time_embedding, scalar_frames.dtype)
        time_embedding = tf.broadcast_to(
            time_embedding[:, :, None, :],
            [batch_size, frame_count, residue_count, self.time_embedding_dim],
        )
        temporal_features = tf.concat(
            [scalar_frames, delta, abs_delta, time_embedding], axis=-1
        )
        projected = self.temporal_projection(temporal_features, training=training)
        mask_4d = tf.cast(residue_mask[:, None, :, None], projected.dtype)
        projected = projected * mask_4d

        gru_input = tf.transpose(projected, [0, 2, 1, 3])
        gru_input = tf.reshape(
            gru_input,
            [batch_size * residue_count, frame_count, self.temporal_input_dim],
        )
        gru_output = self.temporal_gru(gru_input, training=training)
        gru_output = tf.reshape(
            gru_output,
            [batch_size, residue_count, frame_count, self.gru_hidden_dim],
        )
        gru_output = gru_output * tf.cast(
            residue_mask[:, :, None, None], gru_output.dtype
        )

        attention_scores = self.attention_score(
            self.attention_hidden(gru_output), training=training
        )
        attention_weights = tf.nn.softmax(attention_scores, axis=2)
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        attention_weights = attention_weights * tf.cast(
            residue_mask[:, :, None], attention_weights.dtype
        )
        dynamic_summary = tf.reduce_sum(
            attention_weights[..., None] * gru_output, axis=2
        )
        dynamic_projection = self.dynamic_projection(dynamic_summary)
        z_anchor = scalar_frames[:, -1]
        dynamic_for_fusion = tf.cast(dynamic_projection, z_anchor.dtype)
        gate = tf.math.sigmoid(
            self.gate_linear(tf.concat([z_anchor, dynamic_for_fusion], axis=-1))
        )
        if temporal_mode == "off":
            dynamic_contribution = tf.zeros_like(z_anchor)
            z_fused = z_anchor
        else:
            dynamic_contribution = (
                tf.cast(gate, z_anchor.dtype) * dynamic_for_fusion
            )
            z_fused = z_anchor + dynamic_contribution
        logits = self.classify_logits(z_fused, training=bool(training))
        logits = tf.where(residue_mask, logits, tf.zeros_like(logits))

        if not return_auxiliary:
            return logits
        auxiliary = {
            "scalar_frames": scalar_frames,
            "delta_z": delta,
            "abs_delta_z": abs_delta,
            "normalized_time_offsets": normalized_time,
            "gru_output": tf.transpose(gru_output, [0, 2, 1, 3]),
            "attention_weights": attention_weights,
            "dynamic_summary": dynamic_summary,
            "dynamic_projection": dynamic_projection,
            "gate": gate,
            "dynamic_contribution": dynamic_contribution,
            "temporal_enabled": tf.constant(temporal_mode == "on"),
            "z_anchor": z_anchor,
            "z_fused": z_fused,
        }
        return logits, auxiliary

    def static_anchor_logits(
        self, coords: tf.Tensor, sequence: tf.Tensor, residue_mask: tf.Tensor
    ) -> tf.Tensor:
        """Static baseline using only the final frame and the same raw-logit head."""

        return self.anchor_only_logits(
            coords, sequence, residue_mask, training=False
        )

    def anchor_only_logits(
        self,
        coords: tf.Tensor,
        sequence: tf.Tensor,
        residue_mask: tf.Tensor,
        spatial_frames: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Use only the anchor frame with this instance's current classifier head."""

        coords = tf.cast(tf.convert_to_tensor(coords), tf.float32)
        if coords.shape.rank == 4:
            coords = coords[:, None, ...]
        tf.debugging.assert_rank(coords, 5)
        sequence = tf.cast(tf.convert_to_tensor(sequence), tf.int32)
        residue_mask = tf.cast(tf.convert_to_tensor(residue_mask), tf.bool)
        if spatial_frames is not None:
            representation = self._validated_spatial_frames(
                spatial_frames, coords, sequence, residue_mask
            )[:, -1]
            logits = self.classify_logits(representation, training=bool(training))
            return tf.where(residue_mask, logits, tf.zeros_like(logits))

        anchor = coords[:, -1]
        residue_count = tf.shape(anchor)[1]
        spatial_training = bool(training) and not self._spatial_frozen
        encoded_systems: list[tf.Tensor] = []
        system_values = zip(
            tf.unstack(anchor, axis=0),
            tf.unstack(sequence, axis=0),
            tf.unstack(residue_mask, axis=0),
        )
        for system_coords, system_sequence, system_mask in system_values:
            valid_indices = tf.cast(tf.where(system_mask)[:, 0], tf.int32)
            tf.debugging.assert_positive(
                tf.size(valid_indices), "A system has no valid residues"
            )
            valid_coords = tf.gather(system_coords, valid_indices)[None, ...]
            valid_sequence = tf.gather(system_sequence, valid_indices)[None, ...]
            valid_count = tf.shape(valid_sequence)[1]
            encoded_valid = self.spatial_encoder(
                valid_coords,
                valid_sequence,
                tf.ones([1, valid_count], dtype=tf.float32),
                training=spatial_training,
            )[0]
            encoded_systems.append(
                tf.scatter_nd(
                    valid_indices[:, None],
                    encoded_valid,
                    [residue_count, self.d_static],
                )
            )
        representation = tf.stack(encoded_systems, axis=0)
        logits = self.classify_logits(representation, training=bool(training))
        return tf.where(tf.cast(residue_mask, tf.bool), logits, tf.zeros_like(logits))

    def legacy_anchor_probabilities(
        self, coords: tf.Tensor, sequence: tf.Tensor, residue_mask: tf.Tensor
    ) -> tf.Tensor:
        """Legacy PreDyPocket probabilities with padding-excluded graph building."""

        return tf.math.sigmoid(
            self.anchor_only_logits(coords, sequence, residue_mask, training=False)
        )


def synthetic_model_inputs(
    batch_size: int = 1, residue_count: int = 6, frame_count: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic, past-only backbone-like coordinates for build/smoke tests."""

    if frame_count < 2:
        raise ValueError("Synthetic Dynamic PreDyPocket input requires at least two frames")
    coords = np.zeros((batch_size, frame_count, residue_count, 4, 3), np.float32)
    atom_offsets = np.asarray(
        [[-0.12, 0.02, 0.00], [0.00, 0.00, 0.00],
         [0.13, -0.01, 0.01], [0.20, 0.03, 0.00]],
        dtype=np.float32,
    )
    for residue in range(residue_count):
        base = np.asarray([residue * 0.38, 0.04 * (residue % 2), 0.0], np.float32)
        coords[:, :, residue] = base + atom_offsets
    for frame in range(frame_count):
        coords[:, frame, :, :, 1] += np.float32(frame * 0.001)
    sequence = np.broadcast_to(
        np.arange(residue_count, dtype=np.int32)[None, :] % 20,
        (batch_size, residue_count),
    ).copy()
    residue_mask = np.ones((batch_size, residue_count), dtype=bool)
    time_offsets = np.broadcast_to(
        np.arange(-(frame_count - 1) * 100, 1, 100, dtype=np.float32)[None, :],
        (batch_size, frame_count),
    ).copy()
    return coords, sequence, residue_mask, time_offsets


def build_dynamic_model(model: DynamicPreDyPocket) -> None:
    """Create all variables without loading data or changing learned parameters."""

    frame_count = model.input_frame_count if model.input_frame_count is not None else 10
    coords, sequence, mask, offsets = synthetic_model_inputs(
        frame_count=frame_count
    )
    model(
        coords,
        sequence,
        mask,
        time_offsets_ps=offsets,
        training=False,
        return_auxiliary=False,
    )


def build_static_anchor_model(model: StaticAnchorPreDyPocket) -> None:
    """Create all independent static-anchor variables without reading a dataset."""

    coords, sequence, mask, _ = synthetic_model_inputs(
        batch_size=1, residue_count=6, frame_count=10
    )
    model(coords[:, 9], sequence, mask, training=False)
