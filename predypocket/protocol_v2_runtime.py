"""Shared construction and safety helpers for protocol-v2 CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .baseline import HeadMatchedAnchorBaseline
from .checkpoint import CheckpointLoadReport, load_predypocket_pretrained
from .config import DynamicConfig, REPO_ROOT
from .dataset import DynamicPreDyPocketDataset
from .model import DynamicPreDyPocket, make_static_predypocket


MODEL_VARIANTS = ("dynamic", "anchor-matched")
PROTOCOL_OUTPUT_ROOT = (REPO_ROOT / "outputs/predypocket_protocol_v2").resolve()
TEST_WARNING = (
    "WARNING: protocol v1 test proteins have already been inspected.\n"
    "This evaluation is post-hoc and is not an untouched final test."
)


def tf_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"cpu", "/cpu:0"}:
        return "/CPU:0"
    if normalized in {"cuda", "gpu", "cuda:0", "/gpu:0"}:
        return "/GPU:0"
    return value


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_dynamic_core(config: DynamicConfig) -> DynamicPreDyPocket:
    mixed = bool(config.training.get("mixed_precision", False))
    if mixed:
        tf.keras.mixed_precision.set_global_policy("float32")
        static_model = make_static_predypocket(dropout=float(config.model["dropout"]))
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        static_model = make_static_predypocket(dropout=float(config.model["dropout"]))
    return DynamicPreDyPocket(
        static_model=static_model,
        input_frame_count=int(config.task["input_frame_count"]),
        time_embedding_dim=int(config.model["time_embedding_dim"]),
        temporal_input_dim=int(config.model["temporal_input_dim"]),
        gru_hidden_dim=int(config.model["gru_hidden_dim"]),
        attention_hidden_dim=int(config.model["attention_hidden_dim"]),
        dropout=float(config.model["dropout"]),
    )


def initialize_model(
    config: DynamicConfig,
    model_variant: str,
    checkpoint: str | Path | None = None,
) -> tuple[tf.keras.Model, DynamicPreDyPocket, CheckpointLoadReport]:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant {model_variant!r}")
    core = make_dynamic_core(config)
    report = load_predypocket_pretrained(
        core,
        checkpoint or config.repo_path(config.model["pretrained_checkpoint"]),
    )
    model: tf.keras.Model = (
        core if model_variant == "dynamic" else HeadMatchedAnchorBaseline(core)
    )
    return model, core, report


def make_dataset(
    config: DynamicConfig, fold: int, split: str
) -> DynamicPreDyPocketDataset:
    return DynamicPreDyPocketDataset(
        manifest_path=config.data["manifest"],
        folds_path=config.data["folds"],
        fold=fold,
        split=split,
        backbone_cache_root=config.data["backbone_cache_root"],
        use_backbone_cache=bool(config.data["use_backbone_cache"]),
    )


def variant_directory(model_variant: str) -> str:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant {model_variant!r}")
    return "dynamic" if model_variant == "dynamic" else "anchor_matched"


def training_directory(
    config: DynamicConfig, fold: int, model_variant: str
) -> Path:
    return (
        config.repo_path(config.values["output"]["root"])
        / f"fold{fold}"
        / variant_directory(model_variant)
    )


def evaluation_directory(config: DynamicConfig, fold: int) -> Path:
    return config.repo_path(config.values["output"]["evaluation_root"]) / f"fold{fold}"


def assert_protocol_v2_config(config: DynamicConfig) -> None:
    if config.protocol["name"] != "protocol_v2_posthoc":
        raise ValueError("Protocol-v2 CLI requires protocol_v2_posthoc configuration")
    actual_root = config.repo_path(config.values["output"]["root"])
    try:
        actual_root.relative_to(PROTOCOL_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError("Protocol-v2 training output must remain under its isolated root") from exc


def assert_protocol_output_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(PROTOCOL_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Protocol-v2 output cannot target protocol-v1 or another external directory"
        ) from exc
    return resolved


def validate_evaluation_request(
    split: str, allow_test_evaluation: bool
) -> str | None:
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluation split {split!r}")
    if split == "test" and not allow_test_evaluation:
        raise PermissionError(
            "Test evaluation requires explicit --allow-test-evaluation authorization"
        )
    return TEST_WARNING if split == "test" else None


def evaluation_call_kwargs(
    model_variant: str, temporal_mode: str
) -> dict[str, Any]:
    if temporal_mode not in {"on", "off"}:
        raise ValueError("temporal_mode must be 'on' or 'off'")
    return {"temporal_mode": temporal_mode} if model_variant == "dynamic" else {}
