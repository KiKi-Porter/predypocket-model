#!/usr/bin/env python
"""Evaluate a trained dynamic checkpoint with validation-only threshold selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.baseline import PublishedPreDyPocketZeroShot
from predypocket.checkpoint import load_predypocket_pretrained
from predypocket.config import DEFAULT_CONFIG_PATH, load_config
from predypocket.dataset import DynamicPreDyPocketDataset
from predypocket.losses import pos_weight_from_dataset
from predypocket.metrics import (
    evaluate_grouped_metrics,
    select_validation_threshold,
    sigmoid,
)
from predypocket.model import DynamicPreDyPocket, make_static_predypocket
from predypocket.readiness import formal_dataset_readiness
from predypocket.serialization import json_dumps, write_json
from predypocket.trainer import collect_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dataset(config, fold: int, split: str) -> DynamicPreDyPocketDataset:
    return DynamicPreDyPocketDataset(
        config.data["manifest"],
        config.data["folds"],
        fold,
        split,
        config.data["backbone_cache_root"],
        bool(config.data["use_backbone_cache"]),
    )


def _metrics_from_predictions(predictions, threshold: float | None):
    return evaluate_grouped_metrics(
        predictions["labels"],
        predictions["logits"],
        predictions["protein_ids"],
        threshold,
        from_logits=True,
    )


def _make_model(config) -> DynamicPreDyPocket:
    mixed = bool(config.training.get("mixed_precision", False))
    if mixed:
        tf.keras.mixed_precision.set_global_policy("float32")
        static_model = make_static_predypocket(dropout=float(config.model["dropout"]))
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    readiness = formal_dataset_readiness(config)
    training_root = config.repo_path(config.values["output"]["root"]) / f"fold{args.fold}"
    dynamic_checkpoint = config.repo_path(
        args.checkpoint or str(training_root / "best_checkpoint")
    )
    output = config.repo_path(
        args.output or str(training_root / "evaluation.json")
    )
    plan = {
        "dry_run": args.dry_run,
        "would_evaluate": not args.dry_run,
        "fold": args.fold,
        "dynamic_checkpoint": str(dynamic_checkpoint),
        "output": str(output),
        "threshold_source": "validation",
        "test_used_for_threshold": False,
        "formal_dataset": readiness,
    }
    if args.dry_run:
        print(json_dumps(plan, indent=2, sort_keys=True))
        return 0
    if not readiness["ready"]:
        raise RuntimeError("Formal ATLAS dataset is not complete; evaluation is refused")

    batch_size = args.batch_size or int(config.training["batch_size"])
    train_dataset = _dataset(config, args.fold, "train")
    validation_dataset = _dataset(config, args.fold, "validation")
    test_dataset = _dataset(config, args.fold, "test")
    pos_weight = pos_weight_from_dataset(
        train_dataset, float(config.training["max_pos_weight"])
    )
    dynamic_model = _make_model(config)
    load_predypocket_pretrained(
        dynamic_model, config.repo_path(config.model["pretrained_checkpoint"])
    )
    restore = tf.train.Checkpoint(model=dynamic_model).read(str(dynamic_checkpoint))
    restore.assert_existing_objects_matched()
    restore.assert_nontrivial_match()
    restore.expect_partial()
    validation = collect_predictions(
        dynamic_model, validation_dataset, batch_size, pos_weight.used_pos_weight
    )
    selected = select_validation_threshold(
        validation["labels"], sigmoid(validation["logits"]), "validation"
    )
    test = collect_predictions(
        dynamic_model, test_dataset, batch_size, pos_weight.used_pos_weight
    )

    baseline_model = _make_model(config)
    load_predypocket_pretrained(
        baseline_model, config.repo_path(config.model["pretrained_checkpoint"])
    )
    static = PublishedPreDyPocketZeroShot(baseline_model)
    baseline_validation = collect_predictions(
        static, validation_dataset, batch_size, pos_weight.used_pos_weight
    )
    baseline_selected = select_validation_threshold(
        baseline_validation["labels"],
        sigmoid(baseline_validation["logits"]),
        "validation",
    )
    baseline_test = collect_predictions(
        static, test_dataset, batch_size, pos_weight.used_pos_weight
    )
    report = {
        "fold": args.fold,
        "dynamic_threshold": selected,
        "dynamic_validation": _metrics_from_predictions(
            validation, selected["threshold"]
        ),
        "dynamic_test": _metrics_from_predictions(test, selected["threshold"]),
        "static_threshold": baseline_selected,
        "static_validation": _metrics_from_predictions(
            baseline_validation, baseline_selected["threshold"]
        ),
        "static_test": _metrics_from_predictions(
            baseline_test, baseline_selected["threshold"]
        ),
        "test_used_for_threshold": False,
    }
    write_json(output, report, indent=2)
    print(json_dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
