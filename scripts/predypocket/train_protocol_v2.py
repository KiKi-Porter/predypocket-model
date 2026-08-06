#!/usr/bin/env python
"""Train one isolated protocol-v2 fold variant, or inspect it without updates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path
from typing import Sequence

import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.collate import dynamic_collate
from predypocket.config import load_config
from predypocket.folds import load_folds, validate_dataset_split
from predypocket.losses import pos_weight_from_dataset
from predypocket.protocol_v2_runtime import (
    MODEL_VARIANTS,
    assert_protocol_output_path,
    assert_protocol_v2_config,
    initialize_model,
    make_dataset,
    set_random_seed,
    tf_device,
    training_directory,
)
from predypocket.readiness import formal_dataset_readiness
from predypocket.serialization import json_dumps
from predypocket.trainer import (
    backward_smoke_without_update,
    configure_stage,
    fit_model,
    make_optimizer,
    validate_validation_for_model_selection,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/predypocket_protocol_v2.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--model-variant", required=True, choices=MODEL_VARIANTS)
    parser.add_argument("--stage", type=int, default=1, choices=(1,))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    assert_protocol_v2_config(config)
    readiness = formal_dataset_readiness(config)
    if not readiness["ready"]:
        raise RuntimeError("Formal ATLAS dataset is not complete; training is refused")
    definitions = load_folds(config.repo_path(config.data["folds"]))
    train_dataset = make_dataset(config, args.fold, "train")
    validation_dataset = make_dataset(config, args.fold, "validation")
    validate_dataset_split(train_dataset, definitions[args.fold], "train")
    validate_dataset_split(
        validation_dataset, definitions[args.fold], "validation"
    )
    validation_counts = validate_validation_for_model_selection(
        validation_dataset,
        allow_single_class_validation=bool(
            config.training.get("allow_single_class_validation", False)
        ),
    )
    pos_weight = pos_weight_from_dataset(
        train_dataset, max_pos_weight=float(config.training["max_pos_weight"])
    )
    seed = args.seed if args.seed is not None else int(config.training["seed"])
    batch_size = args.batch_size or int(config.training["batch_size"])
    accumulation = args.gradient_accumulation or int(
        config.training["gradient_accumulation_steps"]
    )
    max_epochs = args.max_epochs or int(config.training["max_epochs"])
    output = training_directory(config, args.fold, args.model_variant)
    assert_protocol_output_path(output)
    pretrained = config.repo_path(args.checkpoint or config.model["pretrained_checkpoint"])
    plan = {
        "protocol": config.protocol["name"],
        "posthoc": True,
        "dry_run": bool(args.dry_run),
        "would_train": not args.dry_run,
        "fold": args.fold,
        "model_variant": args.model_variant,
        "stage": args.stage,
        "device": tf_device(args.device),
        "seed": seed,
        "pretrained_checkpoint": str(pretrained),
        "resume": args.resume,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "max_epochs": max_epochs,
        "output_directory": str(output),
        "train_proteins": list(definitions[args.fold].train),
        "validation_proteins": list(definitions[args.fold].validation),
        "test_proteins_loaded": False,
        "validation_class_counts": asdict(validation_counts),
        "pos_weight": asdict(pos_weight),
        "epochs_started": 0,
        "optimizer_step_count": 0,
        "parameter_update_count": 0,
    }

    set_random_seed(seed)
    with tf.device(tf_device(args.device)):
        model, _, checkpoint_report = initialize_model(
            config, args.model_variant, pretrained
        )
        configure_stage(model, stage=1, stage2_enabled=False)
        optimizer = make_optimizer(
            float(config.training["learning_rate"]),
            float(config.training["weight_decay"]),
            mixed_precision=bool(config.training.get("mixed_precision", False)),
        )
        plan["pretrained_checkpoint_report"] = checkpoint_report.to_dict()
        if args.dry_run:
            if len(train_dataset) == 0:
                raise RuntimeError("Protocol-v2 train split contains no sample")
            batch = dynamic_collate([train_dataset[0]])
            smoke = backward_smoke_without_update(
                model,
                batch,
                pos_weight=pos_weight.used_pos_weight,
                gradient_clip_norm=float(config.training["gradient_clip_norm"]),
                optimizer=optimizer,
            )
            plan["smoke"] = asdict(smoke)
            plan["optimizer_step_count"] = smoke.optimizer_step_count
            plan["parameter_update_count"] = smoke.parameter_update_count
            print(json_dumps(plan, indent=2, sort_keys=True))
            return 0

        existing_training_artifacts = any(
            output.joinpath(name).exists()
            for name in (
                "history.json",
                "best_checkpoint.index",
                "last_checkpoint.index",
            )
        )
        if existing_training_artifacts and not args.resume:
            raise RuntimeError(
                f"Protocol-v2 output already contains training history: {output}"
            )
        if args.resume:
            resume_prefix = (
                output / "last_checkpoint"
                if args.resume == "auto"
                else config.repo_path(args.resume)
            )
            restore = tf.train.Checkpoint(model=model, optimizer=optimizer).read(
                str(resume_prefix)
            )
            restore.assert_existing_objects_matched()
            restore.assert_nontrivial_match()
        result = fit_model(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            optimizer=optimizer,
            pos_weight=pos_weight.used_pos_weight,
            output_directory=output,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            gradient_clip_norm=float(config.training["gradient_clip_norm"]),
            max_epochs=max_epochs,
            early_stopping_patience=int(config.training["early_stopping_patience"]),
            seed=seed,
            allow_single_class_validation=False,
        )
    print(json_dumps({"fit": asdict(result), "pos_weight": asdict(pos_weight)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
