#!/usr/bin/env python
"""Evaluate protocol v2 on validation by default; test requires explicit consent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.checkpoint import checkpoint_exists
from predypocket.config import load_config
from predypocket.folds import load_folds, validate_dataset_split
from predypocket.losses import pos_weight_from_dataset
from predypocket.metrics import (
    evaluate_grouped_metrics,
    select_validation_threshold,
    sigmoid,
)
from predypocket.protocol_v2_runtime import (
    MODEL_VARIANTS,
    assert_protocol_output_path,
    assert_protocol_v2_config,
    evaluation_call_kwargs,
    evaluation_directory,
    initialize_model,
    make_dataset,
    set_random_seed,
    tf_device,
    training_directory,
    validate_evaluation_request,
    variant_directory,
)
from predypocket.serialization import json_dumps, write_json
from predypocket.trainer import collect_predictions, configure_stage


DEFAULT_CONFIG = REPO_ROOT / "configs/predypocket_protocol_v2.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--model-variant", required=True, choices=MODEL_VARIANTS)
    parser.add_argument("--temporal-mode", choices=("on", "off"), default="on")
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _metrics(predictions: dict[str, Any], threshold: float | None) -> dict[str, Any]:
    result = evaluate_grouped_metrics(
        predictions["labels"],
        predictions["logits"],
        predictions["protein_ids"],
        threshold,
        from_logits=True,
    )
    result["mean_loss"] = predictions["mean_loss"]
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    assert_protocol_v2_config(config)
    split = args.split or str(config.evaluation["default_split"])
    warning = validate_evaluation_request(split, args.allow_test_evaluation)
    if warning:
        print(warning, file=sys.stderr)
    variant_path = training_directory(config, args.fold, args.model_variant)
    checkpoint = config.repo_path(args.checkpoint) if args.checkpoint else variant_path / "best_checkpoint"
    mode_name = (
        f"temporal_{args.temporal_mode}"
        if args.model_variant == "dynamic"
        else "anchor_only"
    )
    output = (
        config.repo_path(args.output)
        if args.output
        else evaluation_directory(config, args.fold)
        / f"{variant_directory(args.model_variant)}_{mode_name}_{split}.json"
    )
    output = assert_protocol_output_path(output)
    plan = {
        "protocol": config.protocol["name"],
        "posthoc": True,
        "outer_test_already_inspected": True,
        "dry_run": bool(args.dry_run),
        "would_evaluate": not args.dry_run,
        "fold": args.fold,
        "model_variant": args.model_variant,
        "temporal_mode": args.temporal_mode,
        "split": split,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "test_evaluation_explicitly_allowed": bool(args.allow_test_evaluation),
        "test_used_for_threshold": False,
    }
    if args.dry_run:
        print(json_dumps(plan, indent=2, sort_keys=True))
        return 0
    if not checkpoint_exists(checkpoint):
        raise FileNotFoundError(f"Protocol-v2 checkpoint is incomplete: {checkpoint}")

    definitions = load_folds(config.repo_path(config.data["folds"]))
    train_dataset = make_dataset(config, args.fold, "train")
    validation_dataset = make_dataset(config, args.fold, "validation")
    validate_dataset_split(train_dataset, definitions[args.fold], "train")
    validate_dataset_split(
        validation_dataset, definitions[args.fold], "validation"
    )
    pos_weight = pos_weight_from_dataset(
        train_dataset, float(config.training["max_pos_weight"])
    )
    batch_size = args.batch_size or int(config.training["batch_size"])
    set_random_seed(int(config.training["seed"]))
    with tf.device(tf_device(args.device)):
        model, _, _ = initialize_model(config, args.model_variant)
        configure_stage(model, stage=1, stage2_enabled=False)
        restore = tf.train.Checkpoint(model=model).read(str(checkpoint))
        restore.assert_existing_objects_matched()
        restore.assert_nontrivial_match()
        restore.expect_partial()
        call_kwargs = evaluation_call_kwargs(
            args.model_variant, args.temporal_mode
        )
        validation_predictions = collect_predictions(
            model,
            validation_dataset,
            batch_size,
            pos_weight.used_pos_weight,
            model_call_kwargs=call_kwargs,
        )
        threshold = select_validation_threshold(
            validation_predictions["labels"],
            sigmoid(validation_predictions["logits"]),
            split="validation",
        )
        if threshold["threshold"] is None:
            raise RuntimeError(
                "Protocol-v2 validation threshold is undefined; evaluation is refused"
            )
        if split == "validation":
            evaluated_predictions = validation_predictions
        else:
            test_dataset = make_dataset(config, args.fold, "test")
            validate_dataset_split(test_dataset, definitions[args.fold], "test")
            evaluated_predictions = collect_predictions(
                model,
                test_dataset,
                batch_size,
                pos_weight.used_pos_weight,
                model_call_kwargs=call_kwargs,
            )
    report = {
        **plan,
        "would_evaluate": False,
        "evaluation_completed": True,
        "threshold_source": "validation",
        "threshold_selection": threshold,
        "metrics": _metrics(evaluated_predictions, threshold["threshold"]),
    }
    write_json(output, report, indent=2)
    print(json_dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
