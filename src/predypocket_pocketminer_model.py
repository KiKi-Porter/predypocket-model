"""PreDyPocket-style temporal wrapper around the released PocketMiner model."""

from __future__ import print_function

import numpy as np
import tensorflow as tf

from gvp import vs_concat
from models import MQAModel


def make_pocketminer_backbone(dropout=0.1, hidden_dim=100, num_layers=4):
    return MQAModel(
        node_features=(8, 50),
        edge_features=(1, 32),
        hidden_dim=(16, hidden_dim),
        num_layers=num_layers,
        dropout=dropout,
    )


def build_backbone_variables(backbone):
    """Create PocketMiner variables before checkpoint restore."""
    X = np.zeros((1, 4, 4, 3), dtype=np.float32)
    X[0, :, 1, 0] = np.arange(4, dtype=np.float32)
    S = np.zeros((1, 4), dtype=np.int32)
    mask = np.ones((1, 4), dtype=np.float32)
    backbone(X, S, mask, train=False, res_level=True)


def load_pocketminer_backbone(checkpoint_path, optimizer=None):
    backbone = make_pocketminer_backbone()
    build_backbone_variables(backbone)
    restore_model_only(backbone, checkpoint_path)
    return backbone


def restore_model_only(model, checkpoint_path):
    """Restore model weights while deliberately ignoring optimizer state."""
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()
    print("MODEL WEIGHTS RESTORED FROM " + str(checkpoint_path))


def encode_pocketminer_residue_embeddings(backbone, X, S, mask, training=False):
    """Run PocketMiner up to the residue embedding before its dense head."""
    V, E, E_idx = backbone.features(X, mask)
    if backbone.ablate_aa_type:
        h_V = backbone.W_v(V)
    else:
        if backbone.use_lm and not backbone.squeeze_lm:
            h_S = S
        elif backbone.use_lm and backbone.squeeze_lm:
            h_S = backbone.W_s(S)
        else:
            h_S = backbone.W_s(S)
        V = vs_concat(V, h_S, backbone.nv, 0)
        h_V = backbone.W_v(V)
    h_E = backbone.W_e(E)
    h_V = backbone.encoder(h_V, h_E, E_idx, mask, train=training)
    return backbone.W_V_out(h_V)


class PreDyPocketPocketMiner(tf.keras.Model):
    """Temporal MD model that reuses PocketMiner's pretrained GVP encoder.

    Inputs are supplied as a dictionary with keys:
      - X_seq: [B, 10, N, 4, 3] selected 3:3:4 conformations
      - X_ref: [B, N, 4, 3] final/reference conformation
      - S: [B, N] PocketMiner amino-acid ids
      - mask: [B, N] valid residue mask
    """

    def __init__(
        self,
        backbone,
        temporal_dim=128,
        dropout=0.1,
        freeze_backbone=True,
        unfreeze_last_k=0,
        train_classifier=True,
        use_pretrained_classifier=True,
    ):
        super(PreDyPocketPocketMiner, self).__init__()
        self.backbone = backbone
        self.temporal_dim = temporal_dim
        self.dropout_rate = dropout
        self.use_pretrained_classifier = use_pretrained_classifier
        self.stop_backbone_gradient = bool(freeze_backbone and int(unfreeze_last_k) <= 0)

        self.temporal_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(temporal_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Lambda(lambda x: x * tf.math.sigmoid(x)),
            tf.keras.layers.Dropout(dropout),
        ])
        self.temporal_gru = tf.keras.layers.GRU(temporal_dim, return_sequences=True)
        self.attention_logits = tf.keras.layers.Dense(1)
        self.dynamic_to_reference = tf.keras.layers.Dense(self.backbone.hs)
        self.fusion_gate = tf.keras.layers.Dense(self.backbone.hs, activation="sigmoid")
        self.new_classifier = None
        if not use_pretrained_classifier:
            self.new_classifier = tf.keras.Sequential([
                tf.keras.layers.Dense(2 * self.backbone.hs, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(2 * self.backbone.hs, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
        self.configure_trainability(
            freeze_backbone=freeze_backbone,
            unfreeze_last_k=unfreeze_last_k,
            train_classifier=train_classifier,
        )

    def configure_trainability(self, freeze_backbone=True, unfreeze_last_k=0, train_classifier=True):
        if not freeze_backbone:
            self.backbone.trainable = True
            self.backbone.dense.trainable = bool(train_classifier and self.use_pretrained_classifier)
            return

        # Keep the container trainable so Keras still traverses child-layer
        # trainable flags; freeze or unfreeze the actual PocketMiner sublayers.
        self.backbone.trainable = True
        self.backbone.features.trainable = False
        self.backbone.W_v.trainable = False
        self.backbone.W_e.trainable = False
        self.backbone.W_V_out.trainable = False
        if hasattr(self.backbone, "W_s"):
            self.backbone.W_s.trainable = False
        self.backbone.encoder.trainable = bool(unfreeze_last_k > 0)
        for layer in self.backbone.encoder.vglayers:
            layer.trainable = False
        if unfreeze_last_k > 0:
            for layer in self.backbone.encoder.vglayers[-int(unfreeze_last_k):]:
                layer.trainable = True
        self.backbone.dense.trainable = bool(train_classifier and self.use_pretrained_classifier)

    def encode_inputs(self, inputs, training=False):
        X_seq = inputs["X_seq"]
        S = inputs["S"]
        mask = inputs["mask"]
        X_ref = inputs.get("X_ref")
        if X_ref is None:
            X_ref = X_seq[:, -1]
        backbone_training = False if self.stop_backbone_gradient else training

        shape = tf.shape(X_seq)
        batch_size = shape[0]
        n_residues = shape[2]
        time_steps = X_seq.shape[1]
        if time_steps is None:
            time_steps = int(tf.shape(X_seq)[1].numpy())

        # Encode frames one at a time. This is equivalent to flattening the time
        # dimension for inference/training, but avoids 10x peak GVP memory use on
        # large proteins.
        h_seq_frames = []
        for frame_index in range(int(time_steps)):
            h_frame = encode_pocketminer_residue_embeddings(
                self.backbone, X_seq[:, frame_index, :, :, :], S, mask, training=backbone_training
            )
            if self.stop_backbone_gradient:
                h_frame = tf.stop_gradient(h_frame)
            h_seq_frames.append(h_frame)
        h_seq = tf.stack(h_seq_frames, axis=1)
        h_ref = encode_pocketminer_residue_embeddings(
            self.backbone, X_ref, S, mask, training=backbone_training
        )
        if self.stop_backbone_gradient:
            h_ref = tf.stop_gradient(h_ref)
        return h_seq, h_ref, mask

    def classify_embeddings(self, h_seq, h_ref, mask, training=False):
        shape = tf.shape(h_seq)
        batch_size = shape[0]
        n_residues = shape[2]
        time_steps = h_seq.shape[1]
        if time_steps is None:
            time_steps = int(tf.shape(h_seq)[1].numpy())
        delta = h_seq[:, 1:, :, :] - h_seq[:, :-1, :, :]
        delta = tf.concat([tf.zeros_like(h_seq[:, :1, :, :]), delta], axis=1)
        magnitude = tf.norm(delta, axis=-1, keepdims=True)
        time_values = tf.linspace(0.0, 1.0, time_steps)
        time_values = tf.reshape(time_values, [1, time_steps, 1, 1])
        time_values = tf.tile(time_values, [batch_size, 1, n_residues, 1])

        temporal_features = tf.concat([h_seq, delta, magnitude, time_values], axis=-1)
        projected = self.temporal_projection(temporal_features, training=training)
        projected = tf.transpose(projected, [0, 2, 1, 3])
        projected = tf.reshape(projected, [batch_size * n_residues, time_steps, self.temporal_dim])

        recurrent = self.temporal_gru(projected, training=training)
        logits = self.attention_logits(recurrent)
        attention = tf.nn.softmax(logits, axis=1)
        dynamic_summary = tf.reduce_sum(attention * recurrent, axis=1)
        dynamic_summary = tf.reshape(dynamic_summary, [batch_size, n_residues, self.temporal_dim])
        dynamic_summary = self.dynamic_to_reference(dynamic_summary)

        gate = self.fusion_gate(tf.concat([h_ref, dynamic_summary], axis=-1))
        fused = h_ref + gate * dynamic_summary
        fused = fused * tf.expand_dims(tf.cast(mask, tf.float32), -1)

        if self.use_pretrained_classifier:
            out = self.backbone.dense(fused, training=training)
        else:
            if self.new_classifier is None:
                raise ValueError("new_classifier is not initialized")
            out = self.new_classifier(fused, training=training)
        return tf.squeeze(out, axis=-1)

    def call(self, inputs, training=False):
        h_seq, h_ref, mask = self.encode_inputs(inputs, training=training)
        return self.classify_embeddings(h_seq, h_ref, mask, training=training)


def make_predypocket_pocketminer(
    checkpoint_path=None,
    freeze_backbone=True,
    unfreeze_last_k=0,
    train_classifier=True,
    use_pretrained_classifier=True,
    temporal_dim=128,
    dropout=0.1,
):
    if checkpoint_path is None:
        backbone = make_pocketminer_backbone(dropout=dropout)
    else:
        backbone = load_pocketminer_backbone(checkpoint_path)
    return PreDyPocketPocketMiner(
        backbone,
        temporal_dim=temporal_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        unfreeze_last_k=unfreeze_last_k,
        train_classifier=train_classifier,
        use_pretrained_classifier=use_pretrained_classifier,
    )
