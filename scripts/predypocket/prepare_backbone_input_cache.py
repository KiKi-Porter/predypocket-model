#!/usr/bin/env python
"""Prepare a 100 ps N/CA/C/O coordinate cache without computing pocket features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.backbone import (
    BACKBONE_ATOM_ORDER,
    build_backbone_layout,
    place_valid_backbone_coordinates,
)
from predypocket.config import DEFAULT_CONFIG_PATH, load_config


class CachePreparationError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--protein-id")
    parser.add_argument("--replica", choices=("R1", "R2", "R3"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_assignments(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise CachePreparationError(f"Assignment CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "protein_id",
        "replica",
        "segment_name",
        "segment_start_ns",
        "segment_end_ns",
        "xtc_path",
        "topology_path",
    }
    if not rows or not required.issubset(rows[0]):
        raise CachePreparationError("Assignment CSV schema is incomplete")
    return rows


def _output_root(task: Mapping[str, Any]) -> Path:
    return (
        Path(task["backbone_cache_root"])
        / str(task["protein_id"])
        / str(task["replica"])
        / str(task["segment_name"])
    )


def _is_complete(root: Path) -> bool:
    required = (
        "frame_indices.npy",
        "times_ps.npy",
        "backbone_coordinates.npy",
        "sequence.npy",
        "valid_residue_mask.npy",
        "metadata.json",
    )
    if not all((root / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "complete" and metadata.get("frame_count") == 501


def _source_schedule(task: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = (
        Path(task["pocket_cache_root"])
        / str(task["protein_id"])
        / str(task["replica"])
        / str(task["segment_name"])
    )
    frame_path = source / "frame_indices.npy"
    time_path = source / "times_ps.npy"
    if not frame_path.is_file() or not time_path.is_file():
        raise CachePreparationError(f"Completed 20 ps source schedule is missing: {source}")
    source_frames = np.load(frame_path, allow_pickle=False)
    source_times = np.load(time_path, allow_pickle=False)
    if source_frames.shape != source_times.shape or source_frames.ndim != 1:
        raise CachePreparationError(f"Invalid source schedule in {source}")
    selected_frames = np.asarray(source_frames[::5], dtype=np.int64)
    selected_times = np.asarray(source_times[::5], dtype=np.float64)
    if selected_frames.shape != (501,) or not np.allclose(
        np.diff(selected_times), 100.0, rtol=0.0, atol=1e-6
    ):
        raise CachePreparationError(f"Source schedule is not a 501-frame 100 ps series: {source}")
    return selected_frames, selected_times


def _load_selected_coordinates(
    xtc_path: Path,
    topology_path: Path,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mdtraj as md

    topology = md.load_pdb(str(topology_path)).topology
    layout = build_backbone_layout(topology)
    selected_atoms = layout.valid_atom_indices
    selected_xyz = np.empty(
        (len(frame_indices), len(selected_atoms), 3), dtype=np.float32
    )
    filled = np.zeros(len(frame_indices), dtype=bool)
    trajectory_offset = 0
    for chunk in md.iterload(
        str(xtc_path),
        chunk=500,
        top=str(topology_path),
        atom_indices=selected_atoms,
    ):
        chunk_end = trajectory_offset + chunk.n_frames
        left = int(np.searchsorted(frame_indices, trajectory_offset, side="left"))
        right = int(np.searchsorted(frame_indices, chunk_end, side="left"))
        if right > left:
            local_indices = frame_indices[left:right] - trajectory_offset
            selected_xyz[left:right] = chunk.xyz[local_indices]
            filled[left:right] = True
        trajectory_offset = chunk_end
        if trajectory_offset > int(frame_indices[-1]):
            break
    if not np.all(filled):
        missing = frame_indices[~filled]
        raise CachePreparationError(
            f"Trajectory does not contain {len(missing)} requested source frames"
        )
    coordinates = place_valid_backbone_coordinates(selected_xyz, layout)
    return coordinates, layout.sequence, layout.valid_residue_mask


def prepare_one(task: Mapping[str, Any]) -> dict[str, Any]:
    root = _output_root(task)
    if _is_complete(root) and task["resume"] and not task["force"]:
        return {"status": "skipped_complete", "output": str(root)}
    if root.exists() and not (task["resume"] or task["force"]):
        raise CachePreparationError(
            f"Output exists; use --resume or --force explicitly: {root}"
        )
    frame_indices, times_ps = _source_schedule(task)
    coordinates, sequence, valid_mask = _load_selected_coordinates(
        Path(task["xtc_path"]), Path(task["topology_path"]), frame_indices
    )
    temporary = root.with_name(root.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.save(temporary / "frame_indices.npy", frame_indices, allow_pickle=False)
    np.save(temporary / "times_ps.npy", times_ps, allow_pickle=False)
    np.save(
        temporary / "backbone_coordinates.npy", coordinates, allow_pickle=False
    )
    np.save(temporary / "sequence.npy", sequence, allow_pickle=False)
    np.save(temporary / "valid_residue_mask.npy", valid_mask, allow_pickle=False)
    metadata = {
        "status": "complete",
        "protein_id": task["protein_id"],
        "replica": task["replica"],
        "segment_name": task["segment_name"],
        "frame_count": 501,
        "sampling_interval_ps": 100,
        "coordinate_unit": "nanometre",
        "backbone_atom_order": list(BACKBONE_ATOM_ORDER),
        "n_residues": int(len(sequence)),
        "n_valid_residues": int(np.sum(valid_mask)),
        "source_xtc": str(task["xtc_path"]),
        "source_topology": str(task["topology_path"]),
        "detector_call_count": 0,
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        shutil.rmtree(root)
    os.replace(temporary, root)
    return {"status": "complete", "output": str(root)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise CachePreparationError("--workers must be positive")
    config = load_config(args.config)
    assignments_path = config.repo_path(config.data["assignments"])
    rows = load_assignments(assignments_path)
    selected = [
        row
        for row in rows
        if (args.protein_id is None or row["protein_id"] == args.protein_id)
        and (args.replica is None or row["replica"] == args.replica)
    ]
    if not selected:
        raise CachePreparationError("No assignment matches the requested filters")
    tasks = []
    for row in selected:
        tasks.append(
            {
                **row,
                "xtc_path": str(config.repo_path(row["xtc_path"])),
                "topology_path": str(config.repo_path(row["topology_path"])),
                "pocket_cache_root": str(config.repo_path(config.data["pocket_cache_root"])),
                "backbone_cache_root": str(
                    config.repo_path(config.data["backbone_cache_root"])
                ),
                "resume": bool(args.resume),
                "force": bool(args.force),
            }
        )
    if args.dry_run:
        payload = {
            "dry_run": True,
            "planned_segments": len(tasks),
            "workers": args.workers,
            "outputs": [str(_output_root(task)) for task in tasks],
            "files_written": 0,
            "coordinate_frames_read": 0,
            "detector_call_count": 0,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.workers == 1:
        results = [prepare_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(prepare_one, tasks))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

