#!/usr/bin/env python
"""Build deterministic protocol-v2 splits without using outer-test labels for selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.config import load_config
from predypocket.folds import load_folds
from predypocket.protocol_v2 import (
    build_protocol_v2_definitions,
    build_split_payload,
    count_protein_supervision,
    manifest_replicas,
    membership_rows,
    validate_protocol_v2,
    write_split_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/predypocket_1ns_gap1ns_future20ns.json",
    )
    parser.add_argument(
        "--output-json",
        default="data/atlas/atlas_10protein_protocol_v2_5fold_splits_seed42.json",
    )
    parser.add_argument(
        "--output-membership",
        default="data/atlas/atlas_10protein_protocol_v2_5fold_membership_seed42.csv",
    )
    parser.add_argument(
        "--report",
        default="outputs/predypocket_protocol_v2/SPLIT_REPORT.md",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected_paths = {
        "output_json": REPO_ROOT / "data/atlas/atlas_10protein_protocol_v2_5fold_splits_seed42.json",
        "output_membership": REPO_ROOT / "data/atlas/atlas_10protein_protocol_v2_5fold_membership_seed42.csv",
        "report": REPO_ROOT / "outputs/predypocket_protocol_v2/SPLIT_REPORT.md",
    }
    for name, expected in expected_paths.items():
        supplied = Path(getattr(args, name))
        supplied = supplied.resolve() if supplied.is_absolute() else (REPO_ROOT / supplied).resolve()
        if supplied != expected.resolve():
            raise ValueError(f"{name} must remain isolated at {expected}")
    config = load_config(args.base_config)
    v1 = load_folds(config.repo_path(config.data["folds"]))
    stats = count_protein_supervision(
        config.repo_path(config.data["manifest"]),
        config.repo_path(config.data["backbone_cache_root"]),
        REPO_ROOT,
    )
    definitions, selection = build_protocol_v2_definitions(v1, stats, seed=42)
    validate_protocol_v2(definitions, v1, stats)
    payload = build_split_payload(definitions, selection, stats)
    replicas = manifest_replicas(config.repo_path(config.data["manifest"]))
    rows = membership_rows(definitions, replicas)
    write_split_artifacts(
        config.repo_path(args.output_json),
        config.repo_path(args.output_membership),
        config.repo_path(args.report),
        payload,
        rows,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_membership}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
