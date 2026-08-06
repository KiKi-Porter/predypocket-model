"""Strict JSON serialization helpers for protocol reports and histories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def to_json_compatible(value: Any) -> Any:
    """Recursively replace non-finite numbers with JSON null-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize strict JSON; bare NaN and Infinity are always rejected."""

    options = {"allow_nan": False}
    options.update(kwargs)
    return json.dumps(to_json_compatible(value), **options)


def write_json(path: str | Path, value: Any, **kwargs: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json_dumps(value, **kwargs) + "\n", encoding="utf-8")
