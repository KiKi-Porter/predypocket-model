from __future__ import annotations

import inspect

import numpy as np
import pytest
import tensorflow as tf

from predypocket.baseline import PublishedPreDyPocketZeroShot
from predypocket.checkpoint import compare_shape_maps
from predypocket.model import (
    DynamicPreDyPocket,
    PreDyPocketSpatialEncoder,
    synthetic_model_inputs,
)
from predypocket.trainer import model_inputs


def test_01_model_defaults_to_misato_10_frames(model_artifacts):
    assert model_artifacts["model"].input_frame_count == 10
    assert model_artifacts["model"].spatial_frame_chunk_size == 1


def test_118_model_accepts_misato_ten_frame_history():
    model = DynamicPreDyPocket(input_frame_count=10)
    coordinates, sequence, mask, offsets = synthetic_model_inputs(
        batch_size=2, residue_count=6, frame_count=10
    )
    logits, auxiliary = model(
        coordinates,
        sequence,
        mask,
        time_offsets_ps=offsets,
        training=False,
        return_auxiliary=True,
    )
    assert tuple(logits.shape) == (2, 6)
    assert tuple(auxiliary["attention_weights"].shape) == (2, 6, 10)
    assert offsets[0].tolist() == [
        -900.0,
        -800.0,
        -700.0,
        -600.0,
        -500.0,
        -400.0,
        -300.0,
        -200.0,
        -100.0,
        0.0,
    ]


def test_02_shared_single_gvp_encoder(model_artifacts):
    model = model_artifacts["model"]
    assert isinstance(model.spatial_encoder, PreDyPocketSpatialEncoder)
    assert model.spatial_encoder.static_model.encoder is model.static_model.encoder


def test_03_no_eleven_encoder_copies(model_artifacts):
    encoders = [
        layer
        for layer in model_artifacts["model"].submodules
        if isinstance(layer, PreDyPocketSpatialEncoder)
    ]
    assert len(encoders) == 1


def test_04_output_shape_is_batch_by_residue(model_artifacts):
    assert tuple(model_artifacts["logits"].shape) == (2, 6)


def test_05_gru_is_unidirectional(model_artifacts):
    model = model_artifacts["model"]
    assert model.bidirectional is False
    assert model.temporal_gru.go_backwards is False


def test_06_gru_has_one_layer(model_artifacts):
    assert model_artifacts["model"].gru_layers == 1


def test_07_gru_hidden_size_is_128(model_artifacts):
    assert model_artifacts["model"].temporal_gru.units == 128


def test_08_attention_shape(model_artifacts):
    assert tuple(model_artifacts["auxiliary"]["attention_weights"].shape) == (2, 6, 10)


def test_09_gate_shape(model_artifacts):
    assert tuple(model_artifacts["auxiliary"]["gate"].shape) == (2, 6, 100)


def test_10_dynamic_projection_zero_initialized(model_artifacts):
    layer = model_artifacts["model"].dynamic_projection
    assert np.count_nonzero(layer.kernel.numpy()) == 0
    assert np.count_nonzero(layer.bias.numpy()) == 0


def test_11_forward_signature_excludes_future_and_labels():
    parameters = set(inspect.signature(DynamicPreDyPocket.call).parameters)
    prohibited = {
        "label",
        "delta_volume",
        "future_max_volume",
        "pocket_volume",
        "future_coordinates",
    }
    assert not parameters & prohibited


def test_12_delta_first_frame_is_zero(model_artifacts):
    delta = model_artifacts["auxiliary"]["delta_z"].numpy()
    assert np.array_equal(delta[:, 0], np.zeros_like(delta[:, 0]))


def test_13_delta_is_adjacent_difference(model_artifacts):
    scalar = model_artifacts["auxiliary"]["scalar_frames"].numpy()
    delta = model_artifacts["auxiliary"]["delta_z"].numpy()
    assert np.allclose(delta[:, 1:], scalar[:, 1:] - scalar[:, :-1], atol=0, rtol=0)


def test_14_absolute_delta_is_correct(model_artifacts):
    delta = model_artifacts["auxiliary"]["delta_z"].numpy()
    absolute = model_artifacts["auxiliary"]["abs_delta_z"].numpy()
    assert np.array_equal(absolute, np.abs(delta))


def test_15_initial_dynamic_branch_is_zero(model_artifacts):
    dynamic = model_artifacts["auxiliary"]["dynamic_projection"].numpy()
    assert np.array_equal(dynamic, np.zeros_like(dynamic))


def test_16_fused_equals_anchor_at_initialization(model_artifacts):
    auxiliary = model_artifacts["auxiliary"]
    assert np.array_equal(auxiliary["z_fused"].numpy(), auxiliary["z_anchor"].numpy())


def test_17_initial_output_matches_legacy_static_model(model_artifacts):
    probabilities = tf.math.sigmoid(model_artifacts["logits"]).numpy()
    legacy = model_artifacts["legacy_probabilities"].numpy()
    mask = model_artifacts["mask"]
    error = np.abs(probabilities[mask] - legacy[mask])
    assert float(np.max(error)) <= 1e-6


def test_18_attention_normalizes_over_time_for_valid_residues(model_artifacts):
    attention = model_artifacts["auxiliary"]["attention_weights"].numpy()
    mask = model_artifacts["mask"]
    assert np.allclose(np.sum(attention, axis=-1)[mask], 1.0, atol=1e-6)


def test_19_padding_attention_is_zero(model_artifacts):
    attention = model_artifacts["auxiliary"]["attention_weights"].numpy()
    mask = model_artifacts["mask"]
    assert np.array_equal(attention[~mask], np.zeros_like(attention[~mask]))


def test_20_checkpoint_loaded_strictly(loaded_model):
    _, report = loaded_model
    assert report.encoder_loaded
    assert report.classifier_loaded
    assert len(report.loaded_keys) == 177
    assert report.unexpected_keys == []
    assert report.shape_mismatch_keys == []


def test_21_checkpoint_shape_mismatch_is_reported():
    mismatches = compare_shape_maps(
        {"model/example": (10, 20)}, {"model/example": (10, 21)}
    )
    assert mismatches == [
        {
            "key": "model/example",
            "checkpoint_shape": [10, 21],
            "model_shape": [10, 20],
        }
    ]


def test_22_checkpoint_format_is_tensorflow_object_checkpoint(loaded_model):
    assert loaded_model[1].checkpoint_format == "tensorflow_v2_object_checkpoint"


def test_23_dynamic_keys_are_new_not_checkpoint_loaded(loaded_model):
    _, report = loaded_model
    assert report.new_dynamic_keys
    assert not set(report.new_dynamic_keys) & set(report.loaded_keys)


def test_24_model_input_whitelist_excludes_volume_fields(model_artifacts):
    batch = {
        "coordinates": model_artifacts["coordinates"],
        "sequence": model_artifacts["sequence"],
        "residue_mask": model_artifacts["mask"],
        "time_offsets_ps": model_artifacts["offsets"],
        "delta_volume": np.ones((2, 6)),
        "future_max_volume": np.ones((2, 6)),
        "label": np.ones((2, 6)),
    }
    assert set(model_inputs(batch)) == {
        "coords",
        "sequence",
        "residue_mask",
        "time_offsets_ps",
    }


def test_25_logits_are_not_internal_probabilities(model_artifacts):
    logits = model_artifacts["logits"].numpy()
    probabilities = tf.math.sigmoid(logits).numpy()
    assert not np.array_equal(logits, probabilities)


def test_70_static_baseline_stays_float32_under_mixed_policy(model_artifacts):
    previous_policy = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = model_artifacts["model"]
        baseline = PublishedPreDyPocketZeroShot(model)
        expected = model.static_anchor_logits(
            model_artifacts["coordinates"],
            model_artifacts["sequence"],
            model_artifacts["mask"],
        )
        actual = baseline(
            model_artifacts["coordinates"],
            model_artifacts["sequence"],
            model_artifacts["mask"],
            time_offsets_ps=model_artifacts["offsets"],
            training=False,
        )
        direct_half = model.static_anchor_logits(
            tf.cast(model_artifacts["coordinates"], tf.float16),
            model_artifacts["sequence"],
            model_artifacts["mask"],
        )
        assert baseline.compute_dtype == "float32"
        assert actual.dtype == tf.float32
        assert direct_half.dtype == tf.float32
        assert np.allclose(actual.numpy(), expected.numpy(), atol=0, rtol=0)
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)
