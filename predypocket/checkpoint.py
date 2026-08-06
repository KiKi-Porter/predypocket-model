"""Audited initialization from the released TensorFlow PreDyPocket checkpoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .model import (
    DynamicPreDyPocket,
    StaticAnchorPreDyPocket,
    build_dynamic_model,
    build_static_anchor_model,
    make_static_predypocket,
    synthetic_model_inputs,
)


INIT_MODES = ("full_predypocket", "encoder_only")
TRAINED_MODEL_TYPES = ("static_anchor", "dynamic")
_ENCODER_COMPONENTS = ("features", "W_s", "W_v", "W_e", "encoder", "W_V_out")
_DYNAMIC_CHECKPOINT_PREFIXES = (
    "model/time_embedding/",
    "model/temporal_projection/",
    "model/temporal_gru/",
    "model/attention_hidden/",
    "model/attention_score/",
    "model/dynamic_projection/",
    "model/gate_linear/",
)


@dataclass
class CheckpointLoadReport:
    checkpoint_path: str
    checkpoint_hash: str = ""
    checkpoint_format: str = "tensorflow_v2_object_checkpoint"
    init_mode: str = "full_predypocket"
    loaded_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatch_keys: list[dict[str, Any]] = field(default_factory=list)
    new_dynamic_keys: list[str] = field(default_factory=list)
    ignored_optimizer_keys: list[str] = field(default_factory=list)
    loaded_tensor_count: int = 0
    loaded_parameter_count: int = 0
    frozen_parameter_count: int = 0
    trainable_parameter_count: int = 0
    model_tensor_count: int = 0
    model_parameter_count: int = 0
    dynamic_parameter_count: int = 0
    encoder_loaded: bool = False
    classifier_loaded: bool = False
    classifier_reinitialized: bool = False
    dynamic_projection_zero_initialized: bool | None = False

    @property
    def shape_mismatches(self) -> list[dict[str, Any]]:
        return self.shape_mismatch_keys

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape_mismatches"] = list(self.shape_mismatch_keys)
        return value


class CheckpointLoadError(RuntimeError):
    def __init__(self, message: str, report: CheckpointLoadReport):
        super().__init__(message)
        self.report = report


class CheckpointModelTypeError(RuntimeError):
    """Raised before restore when a trained checkpoint has the wrong architecture."""


def _checkpoint_files(prefix: Path) -> list[Path]:
    return [
        prefix.with_suffix(".index"),
        *sorted(prefix.parent.glob(prefix.name + ".data-*")),
    ]


def checkpoint_exists(prefix: str | Path) -> bool:
    files = _checkpoint_files(Path(prefix))
    return bool(files[1:]) and all(path.is_file() and path.stat().st_size > 0 for path in files)


def checkpoint_sha256(prefix: str | Path) -> str:
    path = Path(prefix).resolve()
    if not checkpoint_exists(path):
        raise FileNotFoundError(f"Checkpoint prefix is incomplete: {path}")
    digest = hashlib.sha256()
    for artifact in _checkpoint_files(path):
        digest.update(artifact.name.encode("utf-8"))
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_shape_map(prefix: str | Path) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(int(value) for value in shape)
        for name, shape in tf.train.list_variables(str(prefix))
    }


def checkpoint_contract_path(prefix: str | Path) -> Path:
    return Path(str(Path(prefix)) + ".contract.json")


def infer_trained_checkpoint_model_type(prefix: str | Path) -> str:
    """Infer independent-static versus dynamic from the TensorFlow object graph."""

    path = Path(prefix).resolve()
    if not checkpoint_exists(path):
        raise FileNotFoundError(f"Checkpoint prefix is incomplete: {path}")
    names = {
        name
        for name, _ in tf.train.list_variables(str(path))
        if _is_model_tensor(name)
    }
    has_static_path = any(name.startswith("model/static_model/") for name in names)
    has_temporal_path = any(
        name.startswith(prefix_value)
        for name in names
        for prefix_value in _DYNAMIC_CHECKPOINT_PREFIXES
    )
    if has_static_path and has_temporal_path:
        inferred = "dynamic"
    elif has_static_path:
        inferred = "static_anchor"
    else:
        raise CheckpointModelTypeError(
            f"Checkpoint is not a trained DynamicPreDyPocket or independent "
            f"StaticAnchorPreDyPocket: {path}"
        )

    contract_path = checkpoint_contract_path(path)
    if contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointModelTypeError(
                f"Checkpoint contract is unreadable: {contract_path}: {exc}"
            ) from exc
        recorded = contract.get("model_type")
        if recorded != inferred:
            raise CheckpointModelTypeError(
                f"Checkpoint contract records {recorded!r}, but its object graph "
                f"is {inferred!r}: {path}"
            )
    return inferred


def validate_trained_checkpoint_model_type(
    prefix: str | Path, expected_model_type: str
) -> str:
    if expected_model_type not in TRAINED_MODEL_TYPES:
        raise ValueError(f"Unsupported expected model type {expected_model_type!r}")
    actual = infer_trained_checkpoint_model_type(prefix)
    if actual != expected_model_type:
        raise CheckpointModelTypeError(
            f"Checkpoint model type is {actual!r}, expected "
            f"{expected_model_type!r}: {Path(prefix).resolve()}"
        )
    return actual


def write_checkpoint_contract(prefix: str | Path, model: tf.keras.Model) -> Path:
    model_type = getattr(model, "model_type", None)
    if model_type not in TRAINED_MODEL_TYPES:
        raise ValueError(f"Model has no supported checkpoint model_type: {model_type!r}")
    destination = checkpoint_contract_path(prefix)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": model_type,
                "checkpoint_prefix": str(Path(prefix).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def restore_trained_checkpoint(
    model: tf.keras.Model,
    checkpoint_path: str | Path,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
) -> Any:
    """Type-check and strictly restore a trained checkpoint into its own model class."""

    expected_model_type = getattr(model, "model_type", None)
    validate_trained_checkpoint_model_type(checkpoint_path, expected_model_type)
    checkpoint = (
        tf.train.Checkpoint(model=model, optimizer=optimizer)
        if optimizer is not None
        else tf.train.Checkpoint(model=model)
    )
    restore = checkpoint.read(str(Path(checkpoint_path).resolve()))
    restore.assert_existing_objects_matched()
    restore.assert_nontrivial_match()
    restore.expect_partial()
    return restore


def compare_shape_maps(
    expected: Mapping[str, Sequence[int]], actual: Mapping[str, Sequence[int]]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(expected) & set(actual)):
        expected_shape = tuple(int(value) for value in expected[key])
        actual_shape = tuple(int(value) for value in actual[key])
        if expected_shape != actual_shape:
            mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(actual_shape),
                    "model_shape": list(expected_shape),
                }
            )
    return mismatches


def _is_model_tensor(name: str) -> bool:
    return (
        name.startswith("model/")
        and name.endswith("/.ATTRIBUTES/VARIABLE_VALUE")
        and "/.OPTIMIZER_SLOT/" not in name
    )


def _is_optimizer_tensor(name: str) -> bool:
    return name.startswith("optimizer/") or "/.OPTIMIZER_SLOT/" in name


def _write_report(report: CheckpointLoadReport, path: str | Path | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_static_model(model: tf.keras.Model) -> None:
    coords, sequence, mask, _ = synthetic_model_inputs(
        batch_size=1, residue_count=6, frame_count=10
    )
    model(coords[:, -1], sequence, mask.astype(np.float32), train=False, res_level=True)


def _component_variables(model: tf.keras.Model, names: Sequence[str]) -> list[tf.Variable]:
    variables: list[tf.Variable] = []
    for name in names:
        variables.extend(getattr(model, name).variables)
    return variables


def _copy_components(
    source: tf.keras.Model,
    target: tf.keras.Model,
    component_names: Sequence[str],
    report: CheckpointLoadReport,
) -> None:
    for component_name in component_names:
        source_component = getattr(source, component_name)
        target_component = getattr(target, component_name)
        source_weights = source_component.get_weights()
        target_weights = target_component.get_weights()
        if len(source_weights) != len(target_weights):
            report.shape_mismatch_keys.append(
                {
                    "key": component_name,
                    "checkpoint_shape": [len(source_weights)],
                    "model_shape": [len(target_weights)],
                    "error": "component tensor inventory differs",
                }
            )
            continue
        for index, (source_value, target_value) in enumerate(
            zip(source_weights, target_weights)
        ):
            if source_value.shape != target_value.shape:
                report.shape_mismatch_keys.append(
                    {
                        "key": f"{component_name}[{index}]",
                        "checkpoint_shape": list(source_value.shape),
                        "model_shape": list(target_value.shape),
                    }
                )
        if not report.shape_mismatch_keys:
            target_component.set_weights(source_weights)


def load_predypocket_pretrained(
    model: DynamicPreDyPocket | StaticAnchorPreDyPocket,
    checkpoint_path: str | Path,
    report_path: str | Path | None = None,
    init_mode: str = "full_predypocket",
    encoder_frozen: bool = True,
) -> CheckpointLoadReport:
    """Initialize requested components after a strict full checkpoint audit."""

    prefix = Path(checkpoint_path).resolve()
    report = CheckpointLoadReport(
        checkpoint_path=str(prefix),
        init_mode=init_mode,
    )
    if init_mode not in INIT_MODES:
        _write_report(report, report_path)
        raise CheckpointLoadError(f"Unsupported init_mode {init_mode!r}", report)
    if not checkpoint_exists(prefix):
        _write_report(report, report_path)
        raise CheckpointLoadError(f"Checkpoint prefix is incomplete: {prefix}", report)
    report.checkpoint_hash = checkpoint_sha256(prefix)

    if isinstance(model, StaticAnchorPreDyPocket):
        build_static_anchor_model(model)
    elif isinstance(model, DynamicPreDyPocket):
        build_dynamic_model(model)
    else:
        raise TypeError(
            "PreDyPocket initialization requires DynamicPreDyPocket or "
            "StaticAnchorPreDyPocket"
        )
    source = make_static_predypocket(dropout=0.1)
    _build_static_model(source)
    shapes = checkpoint_shape_map(prefix)
    model_keys = sorted(name for name in shapes if _is_model_tensor(name))
    report.ignored_optimizer_keys = sorted(
        name for name in shapes if _is_optimizer_tensor(name)
    )
    source_variables = list(source.variables)
    target_static_variables = list(model.static_model.variables)
    report.model_tensor_count = len(target_static_variables)
    report.model_parameter_count = sum(
        int(tf.size(variable).numpy()) for variable in target_static_variables
    )
    static_ids = {id(variable) for variable in target_static_variables}
    dynamic_variables = [
        variable for variable in model.variables if id(variable) not in static_ids
    ]
    report.dynamic_parameter_count = sum(
        int(tf.size(variable).numpy()) for variable in dynamic_variables
    )
    report.new_dynamic_keys = sorted(variable.name for variable in dynamic_variables)

    if len(model_keys) != len(source_variables):
        report.shape_mismatch_keys.append(
            {
                "key": "<model_tensor_inventory>",
                "checkpoint_shape": [len(model_keys)],
                "model_shape": [len(source_variables)],
            }
        )
        _write_report(report, report_path)
        raise CheckpointLoadError(
            "Checkpoint model-tensor inventory differs from released MQAModel", report
        )
    try:
        restore = tf.train.Checkpoint(model=source).read(str(prefix))
        restore.assert_existing_objects_matched()
        restore.assert_nontrivial_match()
        restore.expect_partial()
    except (AssertionError, ValueError, tf.errors.OpError) as exc:
        report.shape_mismatch_keys.append(
            {
                "key": "<tensorflow_restore>",
                "checkpoint_shape": None,
                "model_shape": None,
                "error": str(exc),
            }
        )
        _write_report(report, report_path)
        raise CheckpointLoadError(f"Strict checkpoint audit failed: {exc}", report) from exc

    components = list(_ENCODER_COMPONENTS)
    if init_mode == "full_predypocket":
        components.append("dense")
    classifier_before = [value.copy() for value in model.static_model.dense.get_weights()]
    _copy_components(source, model.static_model, components, report)
    if report.shape_mismatch_keys:
        _write_report(report, report_path)
        raise CheckpointLoadError("Checkpoint component shapes differ", report)

    loaded_variables = _component_variables(model.static_model, components)
    source_loaded_variables = _component_variables(source, components)
    for target_variable, source_variable in zip(
        loaded_variables, source_loaded_variables
    ):
        if not np.array_equal(target_variable.numpy(), source_variable.numpy()):
            report.shape_mismatch_keys.append(
                {
                    "key": target_variable.name,
                    "checkpoint_shape": list(source_variable.shape),
                    "model_shape": list(target_variable.shape),
                    "error": "post-copy value verification failed",
                }
            )
    if report.shape_mismatch_keys:
        _write_report(report, report_path)
        raise CheckpointLoadError("Checkpoint value verification failed", report)

    encoder_key_prefixes = tuple(f"model/{name}/" for name in _ENCODER_COMPONENTS)
    encoder_keys = [
        key for key in model_keys if key.startswith(encoder_key_prefixes)
    ]
    classifier_keys = [key for key in model_keys if key.startswith("model/dense/")]
    report.loaded_keys = (
        encoder_keys + classifier_keys if init_mode == "full_predypocket" else encoder_keys
    )
    report.missing_keys = list(report.new_dynamic_keys)
    if init_mode == "encoder_only":
        report.missing_keys.extend(classifier_keys)
    report.loaded_tensor_count = len(loaded_variables)
    report.loaded_parameter_count = sum(
        int(tf.size(variable).numpy()) for variable in loaded_variables
    )
    report.encoder_loaded = len(encoder_keys) > 0 and all(
        any(key.startswith(f"model/{component}/") for key in encoder_keys)
        for component in _ENCODER_COMPONENTS
    )
    report.classifier_loaded = (
        init_mode == "full_predypocket" and len(classifier_keys) == 8
    )
    classifier_after = model.static_model.dense.get_weights()
    report.classifier_reinitialized = (
        init_mode == "encoder_only"
        and all(
            np.array_equal(before, after)
            for before, after in zip(classifier_before, classifier_after)
        )
    )
    if isinstance(model, DynamicPreDyPocket):
        report.dynamic_projection_zero_initialized = bool(
            np.count_nonzero(model.dynamic_projection.kernel.numpy()) == 0
            and np.count_nonzero(model.dynamic_projection.bias.numpy()) == 0
        )
    else:
        report.dynamic_projection_zero_initialized = None
    model.set_spatial_encoder_trainable(not encoder_frozen)
    trainable_ids = {id(variable) for variable in model.trainable_variables}
    report.trainable_parameter_count = sum(
        int(tf.size(variable).numpy())
        for variable in model.variables
        if id(variable) in trainable_ids
    )
    report.frozen_parameter_count = sum(
        int(tf.size(variable).numpy())
        for variable in model.variables
        if id(variable) not in trainable_ids
    )
    if (
        not report.encoder_loaded
        or (init_mode == "full_predypocket" and not report.classifier_loaded)
        or (init_mode == "encoder_only" and not report.classifier_reinitialized)
        or (
            isinstance(model, DynamicPreDyPocket)
            and not report.dynamic_projection_zero_initialized
        )
    ):
        _write_report(report, report_path)
        raise CheckpointLoadError("Requested initialization was not fully verified", report)
    _write_report(report, report_path)
    return report
