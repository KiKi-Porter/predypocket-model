from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from predypocket.checkpoint import load_predypocket_pretrained
from predypocket.baseline import HeadMatchedAnchorBaseline
from predypocket.config import load_config
from predypocket.folds import load_folds
from predypocket.model import DynamicPreDyPocket, synthetic_model_inputs
from predypocket.protocol_v2 import count_protein_supervision
from predypocket.protocol_v2_runtime import initialize_model
from predypocket.trainer import configure_stage


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def loaded_model():
    tf.random.set_seed(42)
    np.random.seed(42)
    model = DynamicPreDyPocket()
    report = load_predypocket_pretrained(
        model,
        REPO_ROOT / "models/predypocket_initializer",
        REPO_ROOT
        / "outputs/misato_model_adaptation_tests/regression/checkpoint_loading_report.json",
    )
    configure_stage(model, stage=1)
    return model, report


@pytest.fixture(scope="session")
def model_artifacts(loaded_model):
    model, _ = loaded_model
    coordinates, sequence, mask, offsets = synthetic_model_inputs(
        batch_size=2, residue_count=6
    )
    mask[1, -2:] = False
    coordinates[1, :, -2:] = 0.0
    logits, auxiliary = model(
        coordinates,
        sequence,
        mask,
        time_offsets_ps=offsets,
        training=False,
        return_auxiliary=True,
    )
    legacy_probabilities = model.legacy_anchor_probabilities(
        coordinates, sequence, mask
    )
    return {
        "model": model,
        "coordinates": coordinates,
        "sequence": sequence,
        "mask": mask,
        "offsets": offsets,
        "logits": logits,
        "auxiliary": auxiliary,
        "legacy_probabilities": legacy_probabilities,
    }


@pytest.fixture(scope="session")
def backward_artifacts(loaded_model):
    from predypocket.trainer import (
        backward_smoke_without_update,
        make_optimizer,
        snapshot_parameters,
    )

    model, _ = loaded_model
    coordinates, sequence, mask, offsets = synthetic_model_inputs(
        batch_size=1, residue_count=6
    )
    batch = {
        "coordinates": coordinates,
        "sequence": sequence,
        "residue_mask": mask,
        "time_offsets_ps": offsets,
        "label": np.asarray([[0, 1, 0, 1, 0, 1]], dtype=np.float32),
        "training_mask": np.ones((1, 6), dtype=bool),
    }
    optimizer = make_optimizer()
    before = snapshot_parameters(model)
    result = backward_smoke_without_update(
        model,
        batch,
        pos_weight=2.0,
        gradient_clip_norm=1.0,
        optimizer=optimizer,
    )
    after = snapshot_parameters(model)
    return {
        "result": result,
        "before": before,
        "after": after,
        "optimizer_iterations": int(optimizer.iterations.numpy()),
    }


@pytest.fixture(scope="session")
def protocol_v2_models(loaded_model):
    previous_policy = tf.keras.mixed_precision.global_policy().name
    config = load_config(REPO_ROOT / "configs/predypocket_protocol_v2.json")
    np.random.seed(42)
    tf.random.set_seed(42)
    anchor, anchor_core, _ = initialize_model(config, "anchor-matched")
    configure_stage(anchor, stage=1)
    tf.keras.mixed_precision.set_global_policy(previous_policy)
    return {
        "dynamic": loaded_model[0],
        "anchor": anchor,
        "anchor_core": anchor_core,
    }


@pytest.fixture(scope="session")
def protocol_v2_split_artifacts():
    config = load_config(REPO_ROOT / "configs/predypocket_protocol_v2.json")
    v1 = load_folds(
        REPO_ROOT / "data/atlas/atlas_10protein_5fold_splits_seed42.json"
    )
    v2 = load_folds(config.repo_path(config.data["folds"]))
    stats = count_protein_supervision(
        config.repo_path(config.data["manifest"]),
        config.repo_path(config.data["backbone_cache_root"]),
        REPO_ROOT,
    )
    return {"config": config, "v1": v1, "v2": v2, "stats": stats}
