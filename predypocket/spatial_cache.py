"""Persistent frozen-GVP representations for MISATO training.

The formal MISATO training run freezes the released spatial encoder.  This
module stores its deterministic per-system/frame outputs so subsequent epochs
only execute the temporal head.  Cache metadata is deliberately strict: a
cache made for another manifest or checkpoint must never be used silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .checkpoint import checkpoint_sha256


SPATIAL_CACHE_SCHEMA = "misato_dynamic_pocket_spatial_cache_v1"
SPATIAL_CACHE_METADATA = "metadata.json"
SPATIAL_CACHE_SYSTEM_ROOT = "systems"
_SYSTEM_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class SpatialCacheError(RuntimeError):
    """Raised when a spatial representation cache violates its contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(path: str | Path) -> str:
    return sha256_file(path)


def cache_array_path(root: str | Path, system_id: str) -> Path:
    """Return the only permitted cache path for a system ID."""

    normalized = str(system_id).strip().lower()
    if not _SYSTEM_ID_PATTERN.fullmatch(normalized):
        raise SpatialCacheError(f"Invalid system ID for spatial cache: {system_id!r}")
    return Path(root).resolve() / SPATIAL_CACHE_SYSTEM_ROOT / f"{normalized}.npy"


def _validate_array(
    value: np.ndarray,
    *,
    frame_count: int,
    feature_dim: int,
    residue_count: int | None = None,
    system_id: str = "",
) -> np.ndarray:
    array = np.asarray(value)
    expected_prefix = (int(frame_count),)
    if array.ndim != 3 or tuple(array.shape[:1]) != expected_prefix:
        raise SpatialCacheError(
            f"Spatial cache shape for {system_id or '<unknown>'} is {array.shape}; "
            f"expected [T,N,{feature_dim}] with T={frame_count}"
        )
    if int(array.shape[2]) != int(feature_dim):
        raise SpatialCacheError(
            f"Spatial cache feature dimension for {system_id or '<unknown>'} is "
            f"{array.shape[2]}, expected {feature_dim}"
        )
    if residue_count is not None and int(array.shape[1]) != int(residue_count):
        raise SpatialCacheError(
            f"Spatial cache residue count for {system_id or '<unknown>'} is "
            f"{array.shape[1]}, expected {residue_count}"
        )
    if array.dtype != np.float32:
        raise SpatialCacheError(
            f"Spatial cache dtype for {system_id or '<unknown>'} is {array.dtype}, "
            "expected float32"
        )
    return array


def write_spatial_array(
    root: str | Path,
    system_id: str,
    value: np.ndarray,
    *,
    frame_count: int,
    feature_dim: int,
    residue_count: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated float32 representation array."""

    # Validate before writing so callers cannot accidentally hide a precision
    # or shape error through an implicit cast.
    array = _validate_array(
        np.asarray(value),
        frame_count=frame_count,
        feature_dim=feature_dim,
        residue_count=residue_count,
        system_id=system_id,
    )
    destination = cache_array_path(root, system_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}-{time.time_ns()}.npy"
    )
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_spatial_cache(
    root: str | Path,
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    split_system_ids: Mapping[str, Sequence[str]],
    residue_counts: Mapping[str, int],
    frame_count: int,
    feature_dim: int,
    spatial_frame_chunk_size: int,
    include_file_hashes: bool = True,
) -> dict[str, Any]:
    """Validate all shard outputs and write the cache metadata."""

    root_path = Path(root).resolve()
    expected_ids = sorted(
        {str(system_id).strip().lower() for values in split_system_ids.values() for system_id in values}
    )
    if not expected_ids:
        raise SpatialCacheError("Cannot finalize an empty spatial cache")
    split_payload = {
        str(split): sorted(str(system_id).strip().lower() for system_id in values)
        for split, values in split_system_ids.items()
    }
    entries: dict[str, dict[str, Any]] = {}
    for system_id in expected_ids:
        if system_id not in residue_counts:
            raise SpatialCacheError(f"Residue count is missing for {system_id}")
        path = cache_array_path(root_path, system_id)
        if not path.is_file():
            raise SpatialCacheError(f"Spatial cache array is missing: {path}")
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except Exception as exc:  # pragma: no cover - exact NumPy exception varies
            raise SpatialCacheError(f"Cannot load spatial cache array {path}: {exc}") from exc
        _validate_array(
            array,
            frame_count=frame_count,
            feature_dim=feature_dim,
            residue_count=int(residue_counts[system_id]),
            system_id=system_id,
        )
        entry: dict[str, Any] = {
            "path": str(path.relative_to(root_path)),
            "shape": [int(value) for value in array.shape],
            "bytes": int(path.stat().st_size),
        }
        if include_file_hashes:
            entry["sha256"] = sha256_file(path)
        entries[system_id] = entry

    metadata: dict[str, Any] = {
        "schema": SPATIAL_CACHE_SCHEMA,
        "manifest_sha256": _json_sha256(manifest_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "input_frame_count": int(frame_count),
        "feature_dim": int(feature_dim),
        "dtype": "float32",
        "spatial_frame_chunk_size": int(spatial_frame_chunk_size),
        "split_system_ids": split_payload,
        "system_ids": expected_ids,
        "residue_counts": {key: int(value) for key, value in sorted(residue_counts.items())},
        "entries": entries,
    }
    _write_json_atomic(root_path / SPATIAL_CACHE_METADATA, metadata)
    return metadata


class SpatialFeatureCache:
    """Validated loader that attaches cached frames to a collated batch."""

    def __init__(
        self,
        root: str | Path,
        *,
        manifest_path: str | Path,
        checkpoint_path: str | Path,
        expected_system_ids: Sequence[str],
        input_frame_count: int = 10,
        feature_dim: int = 100,
        spatial_frame_chunk_size: int = 9,
        verify_file_hashes: bool = False,
        allow_extra_system_ids: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        metadata_path = self.root / SPATIAL_CACHE_METADATA
        if not metadata_path.is_file():
            raise SpatialCacheError(f"Spatial cache metadata is missing: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SpatialCacheError(f"Spatial cache metadata is invalid: {exc}") from exc
        if metadata.get("schema") != SPATIAL_CACHE_SCHEMA:
            raise SpatialCacheError(
                f"Unsupported spatial cache schema: {metadata.get('schema')!r}"
            )
        if metadata.get("manifest_sha256") != _json_sha256(manifest_path):
            raise SpatialCacheError("Spatial cache was built from a different manifest")
        if metadata.get("checkpoint_sha256") != checkpoint_sha256(checkpoint_path):
            raise SpatialCacheError("Spatial cache was built from a different checkpoint")
        for name, expected in (
            ("input_frame_count", int(input_frame_count)),
            ("feature_dim", int(feature_dim)),
            ("spatial_frame_chunk_size", int(spatial_frame_chunk_size)),
        ):
            if int(metadata.get(name, -1)) != expected:
                raise SpatialCacheError(
                    f"Spatial cache {name}={metadata.get(name)!r} does not match {expected}"
                )
        if metadata.get("dtype") != "float32":
            raise SpatialCacheError("Spatial cache dtype must be float32")

        self.metadata = metadata
        self.frame_count = int(input_frame_count)
        self.feature_dim = int(feature_dim)
        self.entries = dict(metadata.get("entries", {}))
        expected = {str(value).strip().lower() for value in expected_system_ids}
        actual = {str(value).strip().lower() for value in metadata.get("system_ids", [])}
        if allow_extra_system_ids:
            inventory_mismatch = not expected.issubset(actual)
        else:
            inventory_mismatch = expected != actual
        if inventory_mismatch:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise SpatialCacheError(
                f"Spatial cache system inventory differs; missing={missing[:5]}, extra={extra[:5]}"
            )
        if set(self.entries) != actual:
            raise SpatialCacheError("Spatial cache entries do not match its system inventory")
        self.residue_counts = {
            str(key): int(value)
            for key, value in dict(metadata.get("residue_counts", {})).items()
        }
        for system_id in sorted(actual):
            entry = self.entries[system_id]
            path = self.root / str(entry.get("path", ""))
            if path.resolve().parent != (self.root / SPATIAL_CACHE_SYSTEM_ROOT).resolve():
                raise SpatialCacheError(f"Spatial cache path escapes cache root: {path}")
            if not path.is_file():
                raise SpatialCacheError(f"Spatial cache array is missing: {path}")
            expected_shape = tuple(int(value) for value in entry.get("shape", []))
            if expected_shape != (
                self.frame_count,
                self.residue_counts.get(system_id, -1),
                self.feature_dim,
            ):
                raise SpatialCacheError(f"Spatial cache metadata shape is invalid for {system_id}")
            if verify_file_hashes:
                recorded = entry.get("sha256")
                if not recorded or recorded != sha256_file(path):
                    raise SpatialCacheError(f"Spatial cache hash mismatch for {system_id}")
        self._verified_files: set[str] = set()

    def _array_for(self, system_id: str) -> np.ndarray:
        normalized = str(system_id).strip().lower()
        if normalized not in self.entries:
            raise SpatialCacheError(f"System is absent from spatial cache: {normalized}")
        path = self.root / str(self.entries[normalized]["path"])
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        return _validate_array(
            array,
            frame_count=self.frame_count,
            feature_dim=self.feature_dim,
            residue_count=self.residue_counts[normalized],
            system_id=normalized,
        )

    def attach_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Return a batch with padded ``spatial_frames`` added."""

        identities = [str(value).strip().lower() for value in batch.get("system_id", [])]
        if not identities:
            raise SpatialCacheError("A cached batch must contain system_id values")
        sequence = np.asarray(batch["sequence"])
        if sequence.ndim != 2 or sequence.shape[0] != len(identities):
            raise SpatialCacheError("Cached batch sequence/system_id dimensions differ")
        max_residues = int(sequence.shape[1])
        output = np.zeros(
            (len(identities), self.frame_count, max_residues, self.feature_dim),
            dtype=np.float32,
        )
        residue_mask = np.asarray(batch["residue_mask"], dtype=bool)
        if residue_mask.shape != sequence.shape:
            raise SpatialCacheError("Cached batch residue mask shape differs")
        for index, system_id in enumerate(identities):
            array = self._array_for(system_id)
            residue_count = int(array.shape[1])
            if residue_count > max_residues:
                raise SpatialCacheError(
                    f"Cached residue count for {system_id} exceeds collated batch width"
                )
            if np.any(residue_mask[index, residue_count:]):
                raise SpatialCacheError(
                    f"Batch padding/mask disagrees with cached residue count for {system_id}"
                )
            output[index, :, :residue_count] = np.asarray(array, dtype=np.float32)
        result = dict(batch)
        result["spatial_frames"] = tf.convert_to_tensor(output, dtype=tf.float32)
        return result
