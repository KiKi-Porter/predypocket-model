"""Configuration loading and fixed-task validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "configs/predypocket_1ns_gap1ns_future20ns.json"
)


class DynamicConfigError(ValueError):
    """Raised when the fixed Dynamic PreDyPocket contract is violated."""


@dataclass(frozen=True)
class DynamicConfig:
    source_path: Path
    values: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, Mapping):
            raise DynamicConfigError(f"Configuration section {name!r} is missing")
        return value

    def repo_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

    @property
    def task(self) -> Mapping[str, Any]:
        return self.section("task")

    @property
    def data(self) -> Mapping[str, Any]:
        return self.section("data")

    @property
    def model(self) -> Mapping[str, Any]:
        return self.section("model")

    @property
    def training(self) -> Mapping[str, Any]:
        return self.section("training")

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self.section("evaluation")

    @property
    def protocol(self) -> Mapping[str, Any]:
        return self.section("protocol")

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self.section("metrics")

    @property
    def default_time_offsets_ps(self) -> tuple[float, ...]:
        count = int(self.task["input_frame_count"])
        interval = float(self.task["model_input_interval_ps"])
        return tuple((index - count + 1) * interval for index in range(count))


def _require_equal(
    section: Mapping[str, Any], name: str, expected: Any, errors: list[str]
) -> None:
    actual = section.get(name)
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            return
    elif actual == expected:
        return
    errors.append(f"{name}: expected {expected!r}, found {actual!r}")


def validate_config(values: Mapping[str, Any]) -> None:
    sections: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for name in ("task", "data", "model", "training", "evaluation"):
        value = values.get(name)
        if not isinstance(value, Mapping):
            errors.append(f"missing object section {name!r}")
        else:
            sections[name] = value
    if errors:
        raise DynamicConfigError("; ".join(errors))

    task = sections["task"]
    for name, expected in {
        "input_history_ns": 1,
        "gap_ns": 1,
        "future_horizon_ns": 20,
        "input_frame_count": 11,
        "model_input_interval_ps": 100,
        "label_threshold_angstrom3": 20.0,
    }.items():
        _require_equal(task, name, expected, errors)

    model = sections["model"]
    for name, expected in {
        "temporal_input_dim": 128,
        "gru_hidden_dim": 128,
        "gru_layers": 1,
        "bidirectional": False,
        "time_embedding_dim": 16,
        "attention_hidden_dim": 64,
        "dropout": 0.1,
        "zero_initialize_dynamic_projection": True,
        "use_vector_gate": True,
    }.items():
        _require_equal(model, name, expected, errors)

    training = sections["training"]
    for name, expected in {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "max_epochs": 50,
        "early_stopping_patience": 8,
        "max_pos_weight": 20.0,
    }.items():
        _require_equal(training, name, expected, errors)
    if bool(training.get("stage2_enabled", False)):
        errors.append("stage2_enabled must remain false in the default configuration")

    protocol = values.get("protocol")
    if protocol is not None:
        if not isinstance(protocol, Mapping):
            errors.append("protocol must be an object")
        else:
            for name, expected in {
                "name": "protocol_v2_posthoc",
                "outer_test_already_inspected": True,
                "train_proteins_per_fold": 6,
                "validation_proteins_per_fold": 2,
                "test_proteins_per_fold": 2,
                "require_both_classes_in_validation": True,
            }.items():
                _require_equal(protocol, name, expected, errors)
            if training.get("stage") != 1:
                errors.append("protocol v2 training.stage must remain 1")
            if training.get("freeze_gvp") is not True:
                errors.append("protocol v2 freeze_gvp must remain true")
            if training.get("allow_single_class_validation") is not False:
                errors.append(
                    "protocol v2 allow_single_class_validation must remain false"
                )
            evaluation = sections["evaluation"]
            if evaluation.get("default_split") != "validation":
                errors.append("protocol v2 default evaluation split must be validation")
            if evaluation.get("test_requires_explicit_flag") is not True:
                errors.append("protocol v2 test evaluation must require an explicit flag")

    if errors:
        raise DynamicConfigError("Invalid fixed task configuration: " + "; ".join(errors))


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DynamicConfig:
    source = Path(path)
    if not source.is_absolute():
        source = (REPO_ROOT / source).resolve()
    if not source.is_file():
        raise DynamicConfigError(f"Configuration file is missing: {source}")
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicConfigError(f"Cannot read {source}: {exc}") from exc
    if not isinstance(values, Mapping):
        raise DynamicConfigError("Dynamic PreDyPocket configuration must be an object")
    validate_config(values)
    return DynamicConfig(source_path=source, values=values)
