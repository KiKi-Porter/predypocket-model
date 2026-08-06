"""Strict, protein-only MISATO Dynamic Pocket v1 dataset and collation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tensorflow as tf


MISATO_SCHEMA = "misato_dynamic_pocket_v1"
MISATO_SPLITS = ("train", "validation", "test")
INPUT_FRAME_INDICES = tuple(range(10))
ANCHOR_FRAME_INDEX = 9
BACKBONE_ATOM_ORDER = ("N", "CA", "C", "O")
DYNAMIC_SAMPLE_FIELDS = frozenset(
    {
        "coords",
        "sequence",
        "residue_mask",
        "training_mask",
        "time_offsets_ps",
        "labels",
        "system_id",
        "split",
    }
)
STATIC_ANCHOR_SAMPLE_FIELDS = frozenset(
    {
        "coords",
        "sequence",
        "residue_mask",
        "training_mask",
        "labels",
        "system_id",
        "split",
    }
)
SAMPLE_FIELDS = DYNAMIC_SAMPLE_FIELDS
DYNAMIC_MODEL_INPUT_FIELDS = frozenset(
    {"coords", "sequence", "residue_mask", "time_offsets_ps"}
)
STATIC_ANCHOR_MODEL_INPUT_FIELDS = frozenset(
    {"coords", "sequence", "residue_mask"}
)
MODEL_INPUT_FIELDS = DYNAMIC_MODEL_INPUT_FIELDS
PROHIBITED_MODEL_FIELDS = frozenset(
    {
        "future_coords",
        "future_coordinates",
        "ligand_coords",
        "ligand_features",
        "pocket_volume",
        "input_end_volume",
        "future_max_volume",
        "delta_volume",
        "future_peak_frame",
        "future_frame_indices",
    }
)
_CACHE_INPUT_FILES = (
    "backbone_coordinates_input.npy",
    "sequence.npy",
    "valid_residue_mask.npy",
    "times_ps.npy",
    "residue_map.csv",
)
_STATIC_ANCHOR_CACHE_INPUT_FILES = (
    "backbone_coordinates_input.npy",
    "sequence.npy",
    "valid_residue_mask.npy",
    "residue_map.csv",
)


class MisatoDatasetError(RuntimeError):
    """Raised when a formal MISATO artifact violates the model-input contract."""


@dataclass(frozen=True)
class SkipEvent:
    system_id: str
    split: str
    reason: str
    detail: str


@dataclass(frozen=True)
class SplitStatistics:
    split: str
    manifest_target_systems: int
    success_systems: int
    excluded_systems: int
    exclusion_reasons: dict[str, int]
    total_residues: int
    positive_labels: int
    negative_labels: int
    positive_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MisatoDatasetError(
            f"Cannot read JSON artifact {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise MisatoDatasetError(f"JSON artifact is not an object: {path}")
    return value


def _load_manifest(path: Path) -> list[dict[str, str]]:
    required = {
        "system_id",
        "split",
        "status",
        "num_frames",
        "num_residues",
        "time_per_frame_ps",
        "first_frame_time_ps",
        "input_frame_indices",
        "input_end_frame_index",
        "future_frame_indices",
    }
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise MisatoDatasetError(f"Cannot read manifest {path}: {exc}") from exc
    if not rows:
        raise MisatoDatasetError(f"Manifest is empty: {path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise MisatoDatasetError(f"Manifest is missing fields: {missing}")

    identities: set[str] = set()
    split_sets = {name: set() for name in MISATO_SPLITS}
    normalized: list[dict[str, str]] = []
    for original in rows:
        row = dict(original)
        system_id = row["system_id"].strip().lower()
        if not system_id or system_id in identities:
            raise MisatoDatasetError(
                f"Manifest contains an empty or duplicate system ID: {system_id!r}"
            )
        split = row.get("split", "")
        if split not in split_sets:
            raise MisatoDatasetError(
                f"Manifest has unsupported split for {system_id}: {split!r}"
            )
        if row.get("status") != "ready":
            raise MisatoDatasetError(
                f"Formal manifest row is not ready for {system_id}: {row.get('status')!r}"
            )
        try:
            input_indices = tuple(int(value) for value in json.loads(row["input_frame_indices"]))
            future_indices = tuple(int(value) for value in json.loads(row["future_frame_indices"]))
            num_frames = int(row["num_frames"])
            num_residues = int(row["num_residues"])
            interval_ps = float(row["time_per_frame_ps"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MisatoDatasetError(
                f"Manifest dimensions/window are invalid for {system_id}: {exc}"
            ) from exc
        if input_indices != INPUT_FRAME_INDICES:
            raise MisatoDatasetError(
                f"MISATO v1 input frames for {system_id} must be exactly 0..9"
            )
        if int(row["input_end_frame_index"]) != 9:
            raise MisatoDatasetError(f"MISATO v1 anchor frame for {system_id} must be 9")
        if num_frames < 11 or future_indices != tuple(range(10, num_frames)):
            raise MisatoDatasetError(
                f"MISATO v1 future window for {system_id} must be frames 10..T-1"
            )
        if num_residues <= 0 or not np.isfinite(interval_ps) or interval_ps <= 0:
            raise MisatoDatasetError(f"Manifest dimensions are invalid for {system_id}")
        row["system_id"] = system_id
        identities.add(system_id)
        split_sets[split].add(system_id)
        normalized.append(row)
    for index, left in enumerate(MISATO_SPLITS):
        for right in MISATO_SPLITS[index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise MisatoDatasetError(
                    f"Manifest split overlap between {left} and {right}: {sorted(overlap)[:10]}"
                )
    return normalized


def manifest_split_summary(manifest_path: str | Path) -> dict[str, Any]:
    rows = _load_manifest(Path(manifest_path).resolve())
    memberships = {
        split: {row["system_id"] for row in rows if row["split"] == split}
        for split in MISATO_SPLITS
    }
    return {
        "target_counts": {split: len(memberships[split]) for split in MISATO_SPLITS},
        "intersections": {
            f"{left}_{right}": sorted(memberships[left] & memberships[right])
            for index, left in enumerate(MISATO_SPLITS)
            for right in MISATO_SPLITS[index + 1 :]
        },
    }


def _residue_ids_from_map(path: Path, expected_count: int) -> np.ndarray:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise MisatoDatasetError(f"Cannot read residue map {path}: {exc}") from exc
    if len(rows) != expected_count:
        raise MisatoDatasetError(
            f"Residue map length {len(rows)} != {expected_count} for {path.parent.name}"
        )
    identifiers: list[str] = []
    for expected_index, row in enumerate(rows):
        try:
            model_index = int(row["model_residue_index"])
            chain_index = int(row["inferred_chain_index"])
            sequence_number = int(row["source_residue_sequence_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MisatoDatasetError(
                f"Residue map has invalid identifiers at row {expected_index}: {path}"
            ) from exc
        if model_index != expected_index:
            raise MisatoDatasetError(
                f"Residue map order differs from model order at row {expected_index}: {path}"
            )
        chain = row.get("source_chain_id", "").strip() or f"chain{chain_index}"
        insertion = row.get("insertion_code", "").strip()
        residue_name = row.get("source_residue_name", "").strip().upper()
        identifiers.append(
            f"{model_index}:{chain}:{sequence_number}{insertion}:{residue_name}"
        )
    if len(identifiers) != len(set(identifiers)):
        raise MisatoDatasetError(f"Residue map identifiers are not unique: {path}")
    return np.asarray(identifiers, dtype=np.str_)


class MisatoDynamicPocketDataset:
    """Success-only view of one official MISATO manifest split.

    Label metadata and schema-3 cache metadata are audited during construction.
    Future/volume arrays are never read into a sample and cannot reach collation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_dir: str | Path,
        split: str,
        system_ids: Sequence[str] | None = None,
        require_complete: bool = False,
        verify_hashes: bool = True,
        model_type: str = "dynamic",
    ) -> None:
        if split not in MISATO_SPLITS:
            raise MisatoDatasetError(f"Unsupported split {split!r}")
        self.manifest_path = Path(manifest_path).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.labels_root = self.data_dir / "labels"
        self.split = split
        self.verify_hashes = bool(verify_hashes)
        if model_type not in {"dynamic", "static_anchor"}:
            raise MisatoDatasetError(f"Unsupported model_type {model_type!r}")
        self.model_type = model_type
        rows = _load_manifest(self.manifest_path)
        self.manifest_target_count = sum(row["split"] == split for row in rows)

        requested = None
        if system_ids is not None:
            requested = {str(value).strip().lower() for value in system_ids}
            manifest_ids = {row["system_id"] for row in rows}
            missing = sorted(requested - manifest_ids)
            if missing:
                raise MisatoDatasetError(
                    f"Requested systems are absent from the manifest: {missing}"
                )
            wrong_split = sorted(
                row["system_id"]
                for row in rows
                if row["system_id"] in requested and row["split"] != split
            )
            if wrong_split:
                raise MisatoDatasetError(
                    f"Requested systems do not belong to split {split!r}: {wrong_split}"
                )

        self.rows: list[dict[str, Any]] = []
        self.skip_events: list[SkipEvent] = []
        for row in rows:
            if row["split"] != split:
                continue
            if requested is not None and row["system_id"] not in requested:
                continue
            try:
                audited = self._audit_success_artifact(row)
            except MisatoDatasetError as exc:
                reason = str(exc).split(":", 1)[0]
                self.skip_events.append(
                    SkipEvent(row["system_id"], split, reason, str(exc))
                )
                continue
            self.rows.append(audited)

        selected_target_count = (
            self.manifest_target_count if requested is None else len(requested)
        )
        excluded = selected_target_count - len(self.rows)
        if require_complete and excluded:
            preview = "; ".join(event.detail for event in self.skip_events[:20])
            raise MisatoDatasetError(
                f"Selected split has {excluded} non-success artifacts: {preview}"
            )
        if requested is not None:
            loaded = {row["system_id"] for row in self.rows}
            not_loaded = sorted(requested - loaded)
            if not_loaded:
                details = [
                    event.detail
                    for event in self.skip_events
                    if event.system_id in not_loaded
                ]
                raise MisatoDatasetError(
                    f"Requested systems are not valid success artifacts: {details}"
                )

        self.system_ids = tuple(row["system_id"] for row in self.rows)
        positive = sum(int(row["positive_count"]) for row in self.rows)
        negative = sum(int(row["negative_count"]) for row in self.rows)
        reason_counts = Counter(event.reason for event in self.skip_events)
        total = positive + negative
        self.statistics = SplitStatistics(
            split=split,
            manifest_target_systems=selected_target_count,
            success_systems=len(self.rows),
            excluded_systems=excluded,
            exclusion_reasons=dict(sorted(reason_counts.items())),
            total_residues=total,
            positive_labels=positive,
            negative_labels=negative,
            positive_rate=(positive / total) if total else None,
        )

    def _audit_success_artifact(self, row: Mapping[str, str]) -> dict[str, Any]:
        system_id = row["system_id"]
        label_dir = self.labels_root / system_id
        metadata_path = label_dir / "metadata.json"
        archive_path = label_dir / "labels.npz"
        if not metadata_path.is_file():
            raise MisatoDatasetError(f"label_metadata_missing: {system_id}")
        try:
            metadata = _read_json(metadata_path)
        except MisatoDatasetError as exc:
            raise MisatoDatasetError(f"label_metadata_invalid: {system_id}: {exc}") from exc
        if metadata.get("status") != "complete":
            raise MisatoDatasetError(
                f"label_status_not_success: {system_id}: {metadata.get('status')!r}"
            )
        if metadata.get("schema") != MISATO_SCHEMA:
            raise MisatoDatasetError(
                f"label_schema_mismatch: {system_id}: {metadata.get('schema')!r}"
            )
        if metadata.get("system_id") != system_id or metadata.get("split") != row["split"]:
            raise MisatoDatasetError(f"label_identity_mismatch: {system_id}")
        if metadata.get("ligand_used_in_model_input") is not False:
            raise MisatoDatasetError(f"ligand_exclusion_unproven: {system_id}")
        if metadata.get("future_used_in_model_input") is not False:
            raise MisatoDatasetError(f"future_exclusion_unproven: {system_id}")
        if metadata.get("closed_mask_applied") is not False:
            raise MisatoDatasetError(f"closed_mask_was_applied: {system_id}")
        try:
            num_frames = int(row["num_frames"])
            num_residues = int(row["num_residues"])
            interval_ps = float(row["time_per_frame_ps"])
            first_frame_time_ps = float(row["first_frame_time_ps"])
            positive = int(metadata["positive_count"])
            negative = int(metadata["negative_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MisatoDatasetError(f"label_dimensions_invalid: {system_id}: {exc}") from exc
        if (
            int(metadata.get("num_frames", -1)) != num_frames
            or int(metadata.get("num_residues", -1)) != num_residues
            or not np.isclose(
                float(metadata.get("time_per_frame_ps", np.nan)),
                interval_ps,
                rtol=0.0,
                atol=1e-12,
            )
            or metadata.get("input_frame_indices") != list(INPUT_FRAME_INDICES)
            or int(metadata.get("input_end_frame", -1)) != 9
            or positive + negative != num_residues
        ):
            raise MisatoDatasetError(f"label_dimensions_mismatch: {system_id}")
        if not archive_path.is_file():
            raise MisatoDatasetError(f"label_archive_missing: {system_id}")
        archive_digest = metadata.get("archive_sha256")
        if not isinstance(archive_digest, str) or not archive_digest:
            raise MisatoDatasetError(f"label_archive_hash_missing: {system_id}")
        if (
            self.verify_hashes
            and self.model_type == "dynamic"
            and archive_digest != _sha256_file(archive_path)
        ):
            raise MisatoDatasetError(f"label_archive_hash_mismatch: {system_id}")

        cache_value = metadata.get("schema3_cache_path")
        if not isinstance(cache_value, str) or not cache_value:
            raise MisatoDatasetError(f"input_cache_path_missing: {system_id}")
        cache_path = Path(cache_value).resolve()
        cache_metadata_path = cache_path / "metadata.json"
        try:
            cache_metadata = _read_json(cache_metadata_path)
        except MisatoDatasetError as exc:
            raise MisatoDatasetError(f"input_cache_metadata_invalid: {system_id}: {exc}") from exc
        if (
            cache_metadata.get("status") != "complete"
            or int(cache_metadata.get("schema_version", -1)) != 3
            or cache_metadata.get("system_id") != system_id
            or cache_metadata.get("split") != row["split"]
            or int(cache_metadata.get("trajectory_frame_count", -1)) != num_frames
            or int(cache_metadata.get("n_residues", -1)) != num_residues
            or cache_metadata.get("input_frame_indices") != list(INPUT_FRAME_INDICES)
            or int(cache_metadata.get("input_end_frame_index", -1)) != 9
            or cache_metadata.get("ligand_used_in_model_input") is not False
            or cache_metadata.get("pdb_atom_order_matches_amber") is not True
            or cache_metadata.get("pdb_residue_order_matches_amber") is not True
        ):
            raise MisatoDatasetError(f"input_cache_contract_mismatch: {system_id}")
        digests = cache_metadata.get("artifact_file_sha256")
        if not isinstance(digests, Mapping):
            raise MisatoDatasetError(f"input_cache_hash_inventory_missing: {system_id}")
        cache_input_files = (
            _STATIC_ANCHOR_CACHE_INPUT_FILES
            if self.model_type == "static_anchor"
            else _CACHE_INPUT_FILES
        )
        for name in cache_input_files:
            artifact = cache_path / name
            if not artifact.is_file():
                raise MisatoDatasetError(f"input_cache_artifact_missing: {system_id}: {name}")
            recorded_digest = digests.get(name)
            if not isinstance(recorded_digest, str) or not recorded_digest:
                raise MisatoDatasetError(
                    f"input_cache_artifact_hash_missing: {system_id}: {name}"
                )
            should_verify_content = not (
                self.model_type == "static_anchor"
                and name == "backbone_coordinates_input.npy"
            )
            if (
                self.verify_hashes
                and should_verify_content
                and recorded_digest != _sha256_file(artifact)
            ):
                raise MisatoDatasetError(
                    f"input_cache_artifact_hash_mismatch: {system_id}: {name}"
                )
        return {
            "system_id": system_id,
            "split": row["split"],
            "num_frames": num_frames,
            "num_residues": num_residues,
            "time_per_frame_ps": interval_ps,
            "first_frame_time_ps": first_frame_time_ps,
            "label_archive": archive_path,
            "cache_path": cache_path,
            "positive_count": positive,
            "negative_count": negative,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        system_id = row["system_id"]
        cache_path = Path(row["cache_path"])
        coordinate_path = cache_path / "backbone_coordinates_input.npy"
        if self.model_type == "static_anchor":
            coordinate_frames = np.load(
                coordinate_path, allow_pickle=False, mmap_mode="r"
            )
            expected_coordinate_shape = (10, int(row["num_residues"]), 4, 3)
            if coordinate_frames.shape != expected_coordinate_shape:
                raise MisatoDatasetError(
                    f"Coordinate shape for {system_id} is {coordinate_frames.shape}, "
                    f"expected {expected_coordinate_shape} in atom order "
                    f"{BACKBONE_ATOM_ORDER}"
                )
            coordinates = np.array(
                coordinate_frames[ANCHOR_FRAME_INDEX], dtype=np.float32, copy=True
            )
            del coordinate_frames
        else:
            coordinates = np.load(coordinate_path, allow_pickle=False)
        sequence = np.load(cache_path / "sequence.npy", allow_pickle=False)
        residue_mask = np.load(
            cache_path / "valid_residue_mask.npy", allow_pickle=False
        )
        times_ps = (
            np.load(cache_path / "times_ps.npy", allow_pickle=False)
            if self.model_type == "dynamic"
            else None
        )
        with np.load(row["label_archive"], allow_pickle=False) as archive:
            schema = str(np.asarray(archive["schema"]).item())
            label_system = str(np.asarray(archive["system_id"]).item())
            label_split = str(np.asarray(archive["split"]).item())
            input_indices = np.asarray(archive["input_frame_indices"], dtype=np.int64)
            input_end_frame = int(np.asarray(archive["input_end_frame"]).item())
            label_num_frames = int(np.asarray(archive["num_frames"]).item())
            label_num_residues = int(np.asarray(archive["num_residues"]).item())
            label_interval_ps = float(np.asarray(archive["time_per_frame_ps"]).item())
            residue_ids = np.asarray(archive["residue_ids"]).copy()
            labels = np.asarray(archive["labels"]).copy()

        if schema != MISATO_SCHEMA:
            raise MisatoDatasetError(
                f"Label schema for {system_id} is {schema!r}, expected {MISATO_SCHEMA!r}"
            )
        if label_system != system_id or label_split != row["split"]:
            raise MisatoDatasetError(f"Label identity differs for {system_id}")
        num_residues = int(row["num_residues"])
        if (
            label_num_frames != int(row["num_frames"])
            or label_num_residues != num_residues
            or not np.isclose(
                label_interval_ps,
                float(row["time_per_frame_ps"]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise MisatoDatasetError(f"Label dimensions differ for {system_id}")
        if not np.array_equal(input_indices, np.asarray(INPUT_FRAME_INDICES)):
            raise MisatoDatasetError(f"Label input frames for {system_id} are not 0..9")
        if input_end_frame != 9:
            raise MisatoDatasetError(f"Label anchor frame for {system_id} is not 9")
        expected_loaded_coordinate_shape = (
            (num_residues, 4, 3)
            if self.model_type == "static_anchor"
            else (10, num_residues, 4, 3)
        )
        if coordinates.shape != expected_loaded_coordinate_shape:
            raise MisatoDatasetError(
                f"Coordinate shape for {system_id} is {coordinates.shape}, expected "
                f"{expected_loaded_coordinate_shape} in atom order {BACKBONE_ATOM_ORDER}"
            )
        if sequence.shape != (num_residues,):
            raise MisatoDatasetError(f"Sequence length differs for {system_id}")
        if residue_mask.shape != (num_residues,):
            raise MisatoDatasetError(f"Residue-mask length differs for {system_id}")
        if labels.shape != (num_residues,) or labels.dtype != np.uint8:
            raise MisatoDatasetError(f"Label length/dtype differs for {system_id}")
        if residue_ids.shape != (num_residues,):
            raise MisatoDatasetError(f"Label residue-ID length differs for {system_id}")
        expected_residue_ids = _residue_ids_from_map(
            cache_path / "residue_map.csv", num_residues
        )
        if not np.array_equal(residue_ids, expected_residue_ids):
            raise MisatoDatasetError(
                f"Label residue order differs from coordinate/sequence order for {system_id}"
            )
        if times_ps is not None:
            if times_ps.shape != (int(row["num_frames"]),):
                raise MisatoDatasetError(f"Frame-time shape differs for {system_id}")
            expected_times = float(row["first_frame_time_ps"]) + np.arange(
                int(row["num_frames"]), dtype=np.float64
            ) * float(row["time_per_frame_ps"])
            if not np.allclose(times_ps, expected_times, rtol=0.0, atol=1e-12):
                raise MisatoDatasetError(f"Frame times differ from manifest for {system_id}")
        if not np.all(np.isfinite(coordinates)):
            raise MisatoDatasetError(f"Coordinates contain NaN or Inf for {system_id}")
        residue_mask = np.asarray(residue_mask, dtype=bool)
        sequence = np.asarray(sequence)
        if not np.all((sequence[residue_mask] >= 0) & (sequence[residue_mask] < 20)):
            raise MisatoDatasetError(f"Valid residues have invalid sequence codes for {system_id}")
        if not np.all(np.isin(labels, (0, 1))):
            raise MisatoDatasetError(f"Labels are not binary for {system_id}")
        if (
            int(np.sum(labels == 1)) != int(row["positive_count"])
            or int(np.sum(labels == 0)) != int(row["negative_count"])
        ):
            raise MisatoDatasetError(
                f"Label class counts differ from metadata for {system_id}"
            )
        sample = {
            "coords": tf.convert_to_tensor(coordinates, dtype=tf.float32),
            "sequence": tf.convert_to_tensor(sequence, dtype=tf.int32),
            "residue_mask": tf.convert_to_tensor(residue_mask, dtype=tf.bool),
            "training_mask": tf.convert_to_tensor(residue_mask.copy(), dtype=tf.bool),
            "labels": tf.convert_to_tensor(labels, dtype=tf.float32),
            "system_id": system_id,
            "split": row["split"],
        }
        expected_sample_fields = (
            STATIC_ANCHOR_SAMPLE_FIELDS
            if self.model_type == "static_anchor"
            else DYNAMIC_SAMPLE_FIELDS
        )
        if times_ps is not None:
            offsets = np.asarray(
                times_ps[np.asarray(INPUT_FRAME_INDICES)] - times_ps[ANCHOR_FRAME_INDEX],
                dtype=np.float32,
            )
            expected_offsets = (
                np.arange(-9, 1, dtype=np.float32)
                * np.float32(row["time_per_frame_ps"])
            )
            if not np.array_equal(offsets, expected_offsets):
                raise MisatoDatasetError(f"Input time offsets differ for {system_id}")
            sample["time_offsets_ps"] = tf.convert_to_tensor(
                offsets, dtype=tf.float32
            )
        if set(sample) != expected_sample_fields or set(sample) & PROHIBITED_MODEL_FIELDS:
            raise AssertionError("MISATO sample field whitelist was violated")
        return sample

    def iter_supervision(self) -> Iterable[dict[str, Any]]:
        """Read only labels and masks; coordinates and all future fields stay unopened."""

        for row in self.rows:
            with np.load(row["label_archive"], allow_pickle=False) as archive:
                labels = np.asarray(archive["labels"], dtype=np.float32).copy()
            residue_mask = np.load(
                Path(row["cache_path"]) / "valid_residue_mask.npy", allow_pickle=False
            ).astype(bool)
            if labels.shape != residue_mask.shape:
                raise MisatoDatasetError(
                    f"Supervision shape differs for {row['system_id']}"
                )
            yield {
                "labels": labels,
                "residue_mask": residue_mask,
                "training_mask": residue_mask.copy(),
                "system_id": row["system_id"],
                "split": row["split"],
            }


class MisatoStaticAnchorDataset(MisatoDynamicPocketDataset):
    """Frame-9-only view with no trajectory or time fields in returned samples."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "model_type" in kwargs:
            raise TypeError("MisatoStaticAnchorDataset fixes model_type='static_anchor'")
        super().__init__(*args, model_type="static_anchor", **kwargs)


# Keep the established import name while enforcing the v1 contract.
MisatoPocketDataset = MisatoDynamicPocketDataset


def misato_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad different protein lengths without exposing extra model-side fields."""

    if not samples:
        raise MisatoDatasetError("Cannot collate an empty sample list")
    for sample in samples:
        if set(sample) != SAMPLE_FIELDS:
            extra = sorted(set(sample) - SAMPLE_FIELDS)
            missing = sorted(SAMPLE_FIELDS - set(sample))
            raise MisatoDatasetError(
                f"MISATO sample violates field whitelist; extra={extra}, missing={missing}"
            )
    splits = {str(sample["split"]) for sample in samples}
    if len(splits) != 1:
        raise MisatoDatasetError(f"A batch cannot mix manifest splits: {sorted(splits)}")
    frame_counts = {int(np.shape(sample["coords"])[0]) for sample in samples}
    if frame_counts != {10}:
        raise MisatoDatasetError(
            f"MISATO v1 batch samples must each have 10 frames, got {sorted(frame_counts)}"
        )
    residue_counts = [int(np.shape(sample["sequence"])[0]) for sample in samples]
    if any(count <= 0 for count in residue_counts):
        raise MisatoDatasetError("MISATO samples must contain at least one residue")
    batch_size = len(samples)
    max_residues = max(residue_counts)
    coords = np.zeros((batch_size, 10, max_residues, 4, 3), dtype=np.float32)
    sequence = np.zeros((batch_size, max_residues), dtype=np.int32)
    labels = np.zeros((batch_size, max_residues), dtype=np.float32)
    residue_mask = np.zeros((batch_size, max_residues), dtype=bool)
    training_mask = np.zeros((batch_size, max_residues), dtype=bool)
    offsets = np.zeros((batch_size, 10), dtype=np.float32)
    for batch_index, (sample, residue_count) in enumerate(zip(samples, residue_counts)):
        sample_coords = np.asarray(sample["coords"], dtype=np.float32)
        sample_sequence = np.asarray(sample["sequence"], dtype=np.int32)
        sample_labels = np.asarray(sample["labels"], dtype=np.float32)
        sample_residue_mask = np.asarray(sample["residue_mask"], dtype=bool)
        sample_training_mask = np.asarray(sample["training_mask"], dtype=bool)
        sample_offsets = np.asarray(sample["time_offsets_ps"], dtype=np.float32)
        expected_residue_shape = (residue_count,)
        if sample_coords.shape != (10, residue_count, 4, 3):
            raise MisatoDatasetError(
                f"Sample coordinate shape is inconsistent for {sample['system_id']}"
            )
        if any(
            value.shape != expected_residue_shape
            for value in (
                sample_sequence,
                sample_labels,
                sample_residue_mask,
                sample_training_mask,
            )
        ):
            raise MisatoDatasetError(
                f"Sample residue fields are misaligned for {sample['system_id']}"
            )
        if sample_offsets.shape != (10,):
            raise MisatoDatasetError(
                f"Sample time offsets have the wrong shape for {sample['system_id']}"
            )
        coords[batch_index, :, :residue_count] = sample_coords
        sequence[batch_index, :residue_count] = sample_sequence
        labels[batch_index, :residue_count] = sample_labels
        residue_mask[batch_index, :residue_count] = sample_residue_mask
        training_mask[batch_index, :residue_count] = (
            sample_training_mask & sample_residue_mask
        )
        offsets[batch_index] = sample_offsets
    batch = {
        "coords": tf.convert_to_tensor(coords, dtype=tf.float32),
        "sequence": tf.convert_to_tensor(sequence, dtype=tf.int32),
        "labels": tf.convert_to_tensor(labels, dtype=tf.float32),
        "residue_mask": tf.convert_to_tensor(residue_mask, dtype=tf.bool),
        "training_mask": tf.convert_to_tensor(training_mask, dtype=tf.bool),
        "time_offsets_ps": tf.convert_to_tensor(offsets, dtype=tf.float32),
        "system_id": [str(sample["system_id"]) for sample in samples],
        "split": [str(sample["split"]) for sample in samples],
    }
    if set(batch) != SAMPLE_FIELDS or set(batch) & PROHIBITED_MODEL_FIELDS:
        raise AssertionError("MISATO batch field whitelist was violated")
    return batch


def misato_static_anchor_collate(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pad frame-9-only samples without creating a time dimension or offsets."""

    if not samples:
        raise MisatoDatasetError("Cannot collate an empty sample list")
    for sample in samples:
        if set(sample) != STATIC_ANCHOR_SAMPLE_FIELDS:
            extra = sorted(set(sample) - STATIC_ANCHOR_SAMPLE_FIELDS)
            missing = sorted(STATIC_ANCHOR_SAMPLE_FIELDS - set(sample))
            raise MisatoDatasetError(
                "Static-anchor sample violates field whitelist; "
                f"extra={extra}, missing={missing}"
            )
    splits = {str(sample["split"]) for sample in samples}
    if len(splits) != 1:
        raise MisatoDatasetError(f"A batch cannot mix manifest splits: {sorted(splits)}")
    residue_counts = [int(np.shape(sample["sequence"])[0]) for sample in samples]
    if any(count <= 0 for count in residue_counts):
        raise MisatoDatasetError("MISATO samples must contain at least one residue")
    batch_size = len(samples)
    max_residues = max(residue_counts)
    coords = np.zeros((batch_size, max_residues, 4, 3), dtype=np.float32)
    sequence = np.zeros((batch_size, max_residues), dtype=np.int32)
    labels = np.zeros((batch_size, max_residues), dtype=np.float32)
    residue_mask = np.zeros((batch_size, max_residues), dtype=bool)
    training_mask = np.zeros((batch_size, max_residues), dtype=bool)
    for batch_index, (sample, residue_count) in enumerate(zip(samples, residue_counts)):
        sample_coords = np.asarray(sample["coords"], dtype=np.float32)
        sample_sequence = np.asarray(sample["sequence"], dtype=np.int32)
        sample_labels = np.asarray(sample["labels"], dtype=np.float32)
        sample_residue_mask = np.asarray(sample["residue_mask"], dtype=bool)
        sample_training_mask = np.asarray(sample["training_mask"], dtype=bool)
        expected_residue_shape = (residue_count,)
        if sample_coords.shape != (residue_count, 4, 3):
            raise MisatoDatasetError(
                f"Static anchor coordinate shape is inconsistent for "
                f"{sample['system_id']}"
            )
        if any(
            value.shape != expected_residue_shape
            for value in (
                sample_sequence,
                sample_labels,
                sample_residue_mask,
                sample_training_mask,
            )
        ):
            raise MisatoDatasetError(
                f"Sample residue fields are misaligned for {sample['system_id']}"
            )
        coords[batch_index, :residue_count] = sample_coords
        sequence[batch_index, :residue_count] = sample_sequence
        labels[batch_index, :residue_count] = sample_labels
        residue_mask[batch_index, :residue_count] = sample_residue_mask
        training_mask[batch_index, :residue_count] = (
            sample_training_mask & sample_residue_mask
        )
    batch = {
        "coords": tf.convert_to_tensor(coords, dtype=tf.float32),
        "sequence": tf.convert_to_tensor(sequence, dtype=tf.int32),
        "labels": tf.convert_to_tensor(labels, dtype=tf.float32),
        "residue_mask": tf.convert_to_tensor(residue_mask, dtype=tf.bool),
        "training_mask": tf.convert_to_tensor(training_mask, dtype=tf.bool),
        "system_id": [str(sample["system_id"]) for sample in samples],
        "split": [str(sample["split"]) for sample in samples],
    }
    if set(batch) != STATIC_ANCHOR_SAMPLE_FIELDS:
        raise AssertionError("Static-anchor batch field whitelist was violated")
    return batch


def iter_misato_batches(
    dataset: MisatoDynamicPocketDataset,
    batch_size: int,
    shuffle: bool | None = None,
    seed: int = 42,
) -> Iterable[dict[str, Any]]:
    if batch_size < 1:
        raise MisatoDatasetError("batch_size must be positive")
    if shuffle is None:
        shuffle = dataset.split == "train"
    if dataset.split != "train" and shuffle:
        raise MisatoDatasetError(
            f"shuffle=True is forbidden for the {dataset.split} split"
        )
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        samples = [dataset[int(index)] for index in selected]
        if dataset.model_type == "static_anchor":
            yield misato_static_anchor_collate(samples)
        else:
            yield misato_collate(samples)


def model_input_whitelist(batch: Mapping[str, Any]) -> dict[str, Any]:
    """The only fields that may cross from a MISATO batch into model.forward."""

    model_fields = (
        DYNAMIC_MODEL_INPUT_FIELDS
        if "time_offsets_ps" in batch
        else STATIC_ANCHOR_MODEL_INPUT_FIELDS
    )
    missing = sorted(model_fields - set(batch))
    if missing:
        raise MisatoDatasetError(f"Model batch is missing required fields: {missing}")
    inputs = {name: batch[name] for name in model_fields}
    # Frozen spatial representations are an optional evaluation/training
    # optimization.  They are attached after collation and never appear in
    # the raw MISATO sample whitelist.
    if "spatial_frames" in batch:
        if model_fields == STATIC_ANCHOR_MODEL_INPUT_FIELDS:
            raise MisatoDatasetError(
                "Independent StaticAnchorPreDyPocket cannot receive a multi-frame "
                "spatial cache"
            )
        inputs["spatial_frames"] = batch["spatial_frames"]
    return inputs
