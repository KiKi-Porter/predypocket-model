"""Build the reusable frozen-GVP representation cache for MISATO v1.

The cache contains one ``float32`` array per successful train/validation
system.  Each array has shape ``[10, N, 100]`` and is produced by the exact
released PreDyPocket spatial encoder after strict checkpoint initialization.
The work is shardable by system ID and safe to resume: an intact array is
validated and skipped, while interrupted temporary files are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.checkpoint import load_predypocket_pretrained
from predypocket.misato_dataset import (  # noqa: E402
    MisatoDynamicPocketDataset,
)
from predypocket.model import DynamicPreDyPocket  # noqa: E402
from predypocket.spatial_cache import (  # noqa: E402
    SpatialCacheError,
    cache_array_path,
    finalize_spatial_cache,
    write_spatial_array,
)


FRAME_COUNT = 10
FEATURE_DIM = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--pretrained-checkpoint", default="models/predypocket_initializer")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=("train", "validation"),
        default=None,
        help="Manifest split(s) to cache (default: train and validation).",
    )
    parser.add_argument(
        "--spatial-frame-chunk-size",
        type=int,
        default=9,
        help="Maximum number of frames sent to GVP in one encoder call.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--no-verify-hashes",
        dest="verify_hashes",
        action="store_false",
        default=True,
        help="Skip source artifact SHA-256 checks after the manifest audit.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Validate all arrays and write cache metadata.json, without encoding.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.spatial_frame_chunk_size < 1:
        raise ValueError("--spatial-frame-chunk-size must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    if args.splits is None:
        args.splits = ["train", "validation"]


def _load_datasets(
    manifest: Path,
    data_dir: Path,
    splits: Sequence[str],
    verify_hashes: bool,
) -> dict[str, MisatoDynamicPocketDataset]:
    datasets: dict[str, MisatoDynamicPocketDataset] = {}
    for split in splits:
        datasets[split] = MisatoDynamicPocketDataset(
            manifest,
            data_dir,
            split=split,
            require_complete=False,
            verify_hashes=verify_hashes,
        )
    ids: list[str] = []
    for dataset in datasets.values():
        ids.extend(dataset.system_ids)
    if len(ids) != len(set(ids)):
        raise RuntimeError("Selected manifest splits contain duplicate system IDs")
    if not ids:
        raise RuntimeError("Selected splits contain no successful systems")
    return datasets


def _records(
    datasets: dict[str, MisatoDynamicPocketDataset],
) -> tuple[list[tuple[str, MisatoDynamicPocketDataset, int]], dict[str, list[str]], dict[str, int]]:
    records: list[tuple[str, MisatoDynamicPocketDataset, int]] = []
    split_ids: dict[str, list[str]] = {}
    residue_counts: dict[str, int] = {}
    for split, dataset in datasets.items():
        split_ids[split] = list(dataset.system_ids)
        for index, system_id in enumerate(dataset.system_ids):
            row = dataset.rows[index]
            count = int(row["num_residues"])
            records.append((system_id, dataset, index))
            residue_counts[system_id] = count
    records.sort(key=lambda item: item[0])
    return records, split_ids, residue_counts


def _valid_existing_array(
    path: Path,
    *,
    system_id: str,
    residue_count: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        expected = (FRAME_COUNT, int(residue_count), FEATURE_DIM)
        return value.dtype == np.float32 and tuple(value.shape) == expected and bool(
            np.all(np.isfinite(value))
        )
    except (OSError, ValueError, EOFError):
        return False


def _encode_one(
    model: DynamicPreDyPocket,
    dataset: MisatoDynamicPocketDataset,
    index: int,
) -> np.ndarray:
    sample = dataset[index]
    coords = np.asarray(sample["coords"], dtype=np.float32)
    sequence = np.asarray(sample["sequence"], dtype=np.int32)
    residue_mask = np.asarray(sample["residue_mask"], dtype=bool)
    encoded = model.encode_frames(
        coords[None, ...],
        sequence[None, ...],
        residue_mask[None, ...],
        training=False,
    )[0]
    value = np.asarray(encoded.numpy(), dtype=np.float32)
    if value.shape != (FRAME_COUNT, coords.shape[1], FEATURE_DIM):
        raise RuntimeError(
            f"Encoder output for {sample['system_id']} has shape {value.shape}; "
            f"expected {(FRAME_COUNT, coords.shape[1], FEATURE_DIM)}"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"Encoder output contains NaN/Inf for {sample['system_id']}")
    return value


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _run_shard(
    args: argparse.Namespace,
    manifest: Path,
    data_dir: Path,
    checkpoint: Path,
    cache_dir: Path,
) -> int:
    datasets = _load_datasets(manifest, data_dir, args.splits, args.verify_hashes)
    records, _, residue_counts = _records(datasets)
    assigned = [
        record
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / f"progress.shard-{args.shard_index}-of-{args.num_shards}.jsonl"
    print(
        json.dumps(
            {
                "event": "spatial_cache_shard_start",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "assigned_systems": len(assigned),
                "total_systems": len(records),
                "cache_dir": str(cache_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    tf.random.set_seed(args.seed + args.shard_index)
    model = DynamicPreDyPocket(
        input_frame_count=FRAME_COUNT,
        spatial_frame_chunk_size=args.spatial_frame_chunk_size,
        temporal_mode="on",
    )
    report_path = cache_dir / (
        f"initialization_report.shard-{args.shard_index}-of-{args.num_shards}.json"
    )
    load_predypocket_pretrained(
        model,
        checkpoint,
        report_path,
        init_mode="full_predypocket",
        encoder_frozen=True,
    )

    completed = skipped = 0
    for system_id, dataset, dataset_index in assigned:
        destination = cache_array_path(cache_dir, system_id)
        residue_count = residue_counts[system_id]
        if _valid_existing_array(
            destination, system_id=system_id, residue_count=residue_count
        ):
            skipped += 1
            payload = {"event": "cache_skip", "system_id": system_id}
            _write_progress(progress_path, payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
            continue
        value = _encode_one(model, dataset, dataset_index)
        write_spatial_array(
            cache_dir,
            system_id,
            value,
            frame_count=FRAME_COUNT,
            feature_dim=FEATURE_DIM,
            residue_count=residue_count,
            overwrite=True,
        )
        completed += 1
        payload = {
            "event": "cache_system_complete",
            "system_id": system_id,
            "completed": completed,
            "skipped": skipped,
        }
        _write_progress(progress_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "event": "spatial_cache_shard_complete",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "completed": completed,
                "skipped": skipped,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _finalize(
    args: argparse.Namespace,
    manifest: Path,
    data_dir: Path,
    checkpoint: Path,
    cache_dir: Path,
) -> int:
    datasets = _load_datasets(manifest, data_dir, args.splits, args.verify_hashes)
    _, split_ids, residue_counts = _records(datasets)
    metadata = finalize_spatial_cache(
        cache_dir,
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        split_system_ids=split_ids,
        residue_counts=residue_counts,
        frame_count=FRAME_COUNT,
        feature_dim=FEATURE_DIM,
        spatial_frame_chunk_size=args.spatial_frame_chunk_size,
        include_file_hashes=True,
    )
    print(
        json.dumps(
            {
                "event": "spatial_cache_finalized",
                "cache_dir": str(cache_dir),
                "system_count": len(metadata["system_ids"]),
                "metadata": str(cache_dir / "metadata.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    manifest = Path(args.manifest).resolve()
    data_dir = Path(args.data_dir).resolve()
    checkpoint = Path(args.pretrained_checkpoint).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    if args.finalize:
        return _finalize(args, manifest, data_dir, checkpoint, cache_dir)
    return _run_shard(args, manifest, data_dir, checkpoint, cache_dir)


if __name__ == "__main__":
    raise SystemExit(main())
