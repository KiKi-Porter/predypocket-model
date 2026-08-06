"""Formal ATLAS Dynamic PreDyPocket dataset with past-only coordinate loading."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backbone import (
    BACKBONE_ATOM_ORDER,
    build_backbone_layout,
    place_valid_backbone_coordinates,
)
from .config import REPO_ROOT
from .folds import FoldDefinition, load_folds


REQUIRED_LABEL_KEYS = frozenset(
    {
        "sample_ids",
        "delta_volume_20ns",
        "valid_residue_mask",
        "closed_mask_1ns",
        "training_residue_mask",
        "label_20ns_ge20",
        "sample_valid_mask",
    }
)
VALID_MANIFEST_STATUSES = frozenset({"valid", "ready"})


class DynamicDatasetError(RuntimeError):
    """Raised when a formal sample violates the fixed past/future boundary."""


def _resolve_repo_path(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _parse_input_frame_indices(row: Mapping[str, Any]) -> np.ndarray:
    try:
        values = json.loads(str(row["input_frame_indices"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DynamicDatasetError("Manifest input_frame_indices is invalid") from exc
    indices = np.asarray(values, dtype=np.int64)
    if indices.shape != (11,):
        raise DynamicDatasetError(
            f"Exactly 11 input frames are required, found shape {indices.shape}"
        )
    if len(np.unique(indices)) != 11 or not np.all(np.diff(indices) > 0):
        raise DynamicDatasetError("Input frame indices must be unique and increasing")
    return indices


def load_label_arrays(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise DynamicDatasetError(f"Formal label file is missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        missing = REQUIRED_LABEL_KEYS - set(archive.files)
        if missing:
            raise DynamicDatasetError(
                f"Formal label file {source} is missing {sorted(missing)}"
            )
        return {name: archive[name].copy() for name in archive.files}


def find_sample_index(label_arrays: Mapping[str, np.ndarray], sample_id: str) -> int:
    sample_ids = np.asarray(label_arrays["sample_ids"]).astype(str)
    matches = np.flatnonzero(sample_ids == sample_id)
    if matches.shape != (1,):
        raise DynamicDatasetError(
            f"Sample {sample_id!r} occurs {len(matches)} times in its label artifact"
        )
    return int(matches[0])


def assemble_dynamic_sample(
    row: Mapping[str, Any],
    label_arrays: Mapping[str, np.ndarray],
    sample_index: int,
    coordinates: np.ndarray,
    sequence: np.ndarray,
    coordinate_residue_mask: np.ndarray,
    time_offsets_ps: np.ndarray,
) -> dict[str, Any]:
    """Assemble one sample from already loaded arrays; useful for safe unit tests."""

    coords = np.asarray(coordinates, dtype=np.float32)
    seq = np.asarray(sequence, dtype=np.int32)
    cache_mask = np.asarray(coordinate_residue_mask, dtype=bool)
    offsets = np.asarray(time_offsets_ps, dtype=np.float32)
    if coords.ndim != 4 or coords.shape[0] != 11 or coords.shape[2:] != (4, 3):
        raise DynamicDatasetError(
            f"Coordinates must be [11,N,4,3], found {coords.shape}"
        )
    residue_count = coords.shape[1]
    if seq.shape != (residue_count,) or cache_mask.shape != (residue_count,):
        raise DynamicDatasetError("Coordinate, sequence and residue-mask lengths differ")
    if offsets.shape != (11,) or not np.allclose(
        offsets, np.arange(-1000, 1, 100, dtype=np.float32), atol=1e-4, rtol=0.0
    ):
        raise DynamicDatasetError("Input times must be exactly -1000..0 ps at 100 ps")
    if not np.all(np.isfinite(coords)):
        raise DynamicDatasetError("Backbone input coordinates contain NaN or Inf")
    if not bool(np.asarray(label_arrays["sample_valid_mask"])[sample_index]):
        raise DynamicDatasetError("Excluded label sample cannot be loaded")

    label_valid = np.asarray(label_arrays["valid_residue_mask"], dtype=bool)
    if label_valid.shape != (residue_count,):
        raise DynamicDatasetError("Label and coordinate residue counts differ")
    residue_mask = cache_mask & label_valid
    label = np.asarray(label_arrays["label_20ns_ge20"][sample_index], dtype=np.float32)
    delta = np.asarray(label_arrays["delta_volume_20ns"][sample_index], dtype=np.float32)
    closed = np.asarray(label_arrays["closed_mask_1ns"][sample_index], dtype=bool)
    training = np.asarray(
        label_arrays["training_residue_mask"][sample_index], dtype=bool
    )
    for name, value in {
        "label": label,
        "delta_volume": delta,
        "closed_mask": closed,
        "training_mask": training,
    }.items():
        if value.shape != (residue_count,):
            raise DynamicDatasetError(f"{name} has shape {value.shape}, expected {(residue_count,)}")
    training = training & residue_mask

    return {
        "coordinates": coords,
        "sequence": seq,
        "residue_mask": residue_mask,
        "time_offsets_ps": offsets,
        "label": label,
        "delta_volume": delta,
        "closed_mask": closed,
        "training_mask": training,
        "protein_id": str(row["protein_id"]),
        "replica": str(row["replica"]),
        "segment_name": str(row["segment_name"]),
        "anchor_time_ps": float(row["anchor_time_ps"]),
        "sample_id": str(row["sample_id"]),
    }


class DynamicPreDyPocketDataset:
    """Read valid formal samples for one protein-level fold split."""

    def __init__(
        self,
        manifest_path: str | Path,
        folds_path: str | Path | None = None,
        fold: int | None = None,
        split: str | None = None,
        backbone_cache_root: str | Path = "data/atlas/backbone_input_cache_100ps",
        use_backbone_cache: bool = True,
        repo_root: str | Path = REPO_ROOT,
        coordinate_loader: Callable[[Mapping[str, Any]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.manifest_path = _resolve_repo_path(manifest_path, self.repo_root)
        self.backbone_cache_root = _resolve_repo_path(
            backbone_cache_root, self.repo_root
        )
        self.use_backbone_cache = bool(use_backbone_cache)
        self.coordinate_loader = coordinate_loader
        self.fold = fold
        self.split = split
        if not self.manifest_path.is_file():
            raise DynamicDatasetError(f"Formal manifest is missing: {self.manifest_path}")
        if (fold is None) != (split is None):
            raise DynamicDatasetError("fold and split must be provided together")

        definition: FoldDefinition | None = None
        if fold is not None:
            if folds_path is None:
                raise DynamicDatasetError("folds_path is required for split selection")
            definitions = load_folds(_resolve_repo_path(folds_path, self.repo_root))
            try:
                definition = definitions[fold]
            except KeyError as exc:
                raise DynamicDatasetError(f"Fold {fold} is not defined") from exc
            if split not in {"train", "validation", "test"}:
                raise DynamicDatasetError(f"Unsupported split {split!r}")
            allowed_proteins = set(getattr(definition, split))
        else:
            allowed_proteins = None

        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self._label_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._coordinate_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._records: list[tuple[dict[str, str], int, Path]] = []
        for row in rows:
            status = str(row.get("status", "")).strip().lower()
            if status not in VALID_MANIFEST_STATUSES:
                continue
            if allowed_proteins is not None and row.get("protein_id") not in allowed_proteins:
                continue
            _parse_input_frame_indices(row)
            if int(row.get("input_frame_count", 11)) != 11:
                raise DynamicDatasetError("Manifest row does not declare 11 input frames")
            label_path = _resolve_repo_path(row["label_path"], self.repo_root)
            if label_path not in self._label_cache:
                self._label_cache[label_path] = load_label_arrays(label_path)
            arrays = self._label_cache[label_path]
            sample_index = find_sample_index(arrays, str(row["sample_id"]))
            if not bool(np.asarray(arrays["sample_valid_mask"])[sample_index]):
                continue
            self._records.append((dict(row), sample_index, label_path))
        self.protein_ids = tuple(sorted({row[0]["protein_id"] for row in self._records}))

    def __len__(self) -> int:
        return len(self._records)

    def _cache_directory(self, row: Mapping[str, Any]) -> Path:
        return (
            self.backbone_cache_root
            / str(row["protein_id"])
            / str(row["replica"])
            / str(row["segment_name"])
        )

    def _load_backbone_cache(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        root = self._cache_directory(row)
        required = {
            "frame_indices": root / "frame_indices.npy",
            "times_ps": root / "times_ps.npy",
            "coordinates": root / "backbone_coordinates.npy",
            "sequence": root / "sequence.npy",
            "mask": root / "valid_residue_mask.npy",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise DynamicDatasetError(f"Backbone cache is incomplete: {missing}")
        if root not in self._coordinate_cache:
            self._coordinate_cache[root] = {
                name: np.load(path, allow_pickle=False)
                for name, path in required.items()
            }
        cache = self._coordinate_cache[root]
        frame_indices = np.asarray(cache["frame_indices"], dtype=np.int64)
        requested = _parse_input_frame_indices(row)
        positions = np.searchsorted(frame_indices, requested)
        if np.any(positions >= len(frame_indices)) or not np.array_equal(
            frame_indices[positions], requested
        ):
            raise DynamicDatasetError(
                f"Backbone cache lacks requested input frames for {row['sample_id']}"
            )
        coordinates = np.asarray(cache["coordinates"][positions], dtype=np.float32)
        times = np.asarray(cache["times_ps"][positions], dtype=np.float64)
        offsets = (times - float(row["anchor_time_ps"])).astype(np.float32)
        return (
            coordinates,
            np.asarray(cache["sequence"], dtype=np.int32),
            np.asarray(cache["mask"], dtype=bool),
            offsets,
        )

    def _load_direct_frames(self, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import mdtraj as md

        topology_path = _resolve_repo_path(row["topology_path"], self.repo_root)
        xtc_path = _resolve_repo_path(row["xtc_path"], self.repo_root)
        topology = md.load_pdb(str(topology_path)).topology
        layout = build_backbone_layout(topology)
        selected_atoms = layout.valid_atom_indices
        selected_frames = []
        for frame_index in _parse_input_frame_indices(row):
            frame = md.load_frame(
                str(xtc_path),
                int(frame_index),
                top=str(topology_path),
                atom_indices=selected_atoms,
            )
            selected_frames.append(frame.xyz[0])
        coordinates = place_valid_backbone_coordinates(
            np.stack(selected_frames, axis=0), layout
        )
        return (
            coordinates,
            layout.sequence,
            layout.valid_residue_mask,
            np.arange(-1000, 1, 100, dtype=np.float32),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, sample_index, label_path = self._records[index]
        arrays = self._label_cache[label_path]
        if self.coordinate_loader is not None:
            coordinate_values = self.coordinate_loader(row)
        elif self.use_backbone_cache:
            coordinate_values = self._load_backbone_cache(row)
        else:
            coordinate_values = self._load_direct_frames(row)
        return assemble_dynamic_sample(
            row,
            arrays,
            sample_index,
            *coordinate_values,
        )

    @property
    def records(self) -> Sequence[Mapping[str, str]]:
        return tuple(record[0] for record in self._records)

    def iter_supervision(self) -> Sequence[dict[str, np.ndarray]]:
        """Expose labels/masks without reading any coordinate frame."""

        supervision: list[dict[str, np.ndarray]] = []
        for _, sample_index, label_path in self._records:
            arrays = self._label_cache[label_path]
            residue_mask = np.asarray(arrays["valid_residue_mask"], dtype=bool)
            training_mask = np.asarray(
                arrays["training_residue_mask"][sample_index], dtype=bool
            )
            supervision.append(
                {
                    "label": np.asarray(
                        arrays["label_20ns_ge20"][sample_index], dtype=np.float32
                    ),
                    "residue_mask": residue_mask,
                    "training_mask": training_mask & residue_mask,
                }
            )
        return tuple(supervision)


def backbone_atom_order() -> tuple[str, str, str, str]:
    return BACKBONE_ATOM_ORDER
