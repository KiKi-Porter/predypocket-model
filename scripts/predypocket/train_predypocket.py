#!/usr/bin/env python
"""Train one Dynamic PreDyPocket protein fold, or inspect it with --dry-run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.checkpoint import load_predypocket_pretrained
from predypocket.collate import dynamic_collate
from predypocket.config import DEFAULT_CONFIG_PATH, load_config
from predypocket.dataset import DynamicPreDyPocketDataset
from predypocket.folds import load_folds, validate_dataset_split
from predypocket.losses import pos_weight_from_dataset
from predypocket.model import DynamicPreDyPocket, make_static_predypocket
from predypocket.readiness import formal_dataset_readiness
from predypocket.serialization import json_dumps
from predypocket.trainer import (
    backward_smoke_without_update,
    configure_stage,
    fit_model,
    make_optimizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2))
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _tf_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"cpu", "/cpu:0"}:
        return "/CPU:0"
    if normalized in {"cuda", "gpu", "cuda:0", "/gpu:0"}:
        return "/GPU:0"
    return value


def _dataset(config, fold: int, split: str) -> DynamicPreDyPocketDataset:
    return DynamicPreDyPocketDataset(
        manifest_path=config.data["manifest"],
        folds_path=config.data["folds"],
        fold=fold,
        split=split,
        backbone_cache_root=config.data["backbone_cache_root"],
        use_backbone_cache=bool(config.data["use_backbone_cache"]),
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


def _formal_sample_smoke(config, args, checkpoint: Path) -> dict:
    definitions = load_folds(config.repo_path(config.data["folds"]))
    dataset = _dataset(config, args.fold, "train")
    validate_dataset_split(dataset, definitions[args.fold], "train")
    if len(dataset) == 0:
        raise RuntimeError(f"Fold {args.fold} train split contains no formal sample")
    sample = dataset[0]
    batch = dynamic_collate([sample])
    pos_weight = pos_weight_from_dataset(
        dataset, max_pos_weight=float(config.training["max_pos_weight"])
    )
    with tf.device(_tf_device(args.device)):
        model = _make_model(config)
        load_predypocket_pretrained(model, checkpoint)
        configure_stage(
            model,
            args.stage,
            stage2_enabled=bool(config.training.get("stage2_enabled", False)),
        )
        optimizer = make_optimizer(
            float(config.training["learning_rate"]),
            float(config.training["weight_decay"]),
            mixed_precision=bool(config.training.get("mixed_precision", False)),
        )
        smoke = backward_smoke_without_update(
            model,
            batch,
            pos_weight=pos_weight.used_pos_weight,
            gradient_clip_norm=float(config.training["gradient_clip_norm"]),
            optimizer=optimizer,
        )
    if smoke.coordinates_shape[:2] != (1, 11):
        raise AssertionError(f"Formal coordinates have shape {smoke.coordinates_shape}")
    expected_logits_shape = (1, smoke.coordinates_shape[2])
    if smoke.logits_shape != expected_logits_shape:
        raise AssertionError(
            f"Formal logits have shape {smoke.logits_shape}; expected {expected_logits_shape}"
        )
    if not math.isfinite(smoke.loss_value):
        raise AssertionError("Formal masked loss is not finite")
    return {
        "executed": True,
        "sample_id": sample["sample_id"],
        "protein_id": sample["protein_id"],
        "coordinates_shape": list(smoke.coordinates_shape),
        "logits_shape": list(smoke.logits_shape),
        "masked_loss": smoke.loss_value,
        "masked_loss_finite": True,
        "effective_supervision_count": smoke.effective_supervision_count,
        "forward_success": smoke.forward_success,
        "backward_success": smoke.backward_success,
        "gradients_finite": smoke.gradients_finite,
        "gradient_clip_callable": smoke.gradient_clip_callable,
        "parameter_update_count": smoke.parameter_update_count,
        "parameters_unchanged": smoke.parameter_update_count == 0,
        "optimizer_step_count": smoke.optimizer_step_count,
        "scheduler_step_count": smoke.scheduler_step_count,
        "used_pos_weight": pos_weight.used_pos_weight,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    readiness = formal_dataset_readiness(config)
    checkpoint = config.repo_path(
        args.checkpoint or config.model["pretrained_checkpoint"]
    )
    output_root = config.repo_path(config.values["output"]["root"]) / f"fold{args.fold}"
    resolved = {
        "dry_run": bool(args.dry_run),
        "would_train": not args.dry_run,
        "fold": args.fold,
        "stage": args.stage,
        "device": _tf_device(args.device),
        "checkpoint": str(checkpoint),
        "resume": args.resume,
        "batch_size": args.batch_size or int(config.training["batch_size"]),
        "gradient_accumulation_steps": args.gradient_accumulation
        or int(config.training["gradient_accumulation_steps"]),
        "num_workers": args.num_workers,
        "max_epochs": args.max_epochs or int(config.training["max_epochs"]),
        "output_directory": str(output_root),
        "formal_dataset": readiness,
        "optimizer_step_count": 0,
        "scheduler_step_count": 0,
        "epochs_started": 0,
        "formal_sample_smoke": {
            "executed": False,
            "reason": "formal_dataset_not_ready",
        },
    }
    if args.dry_run:
        if readiness["ready"]:
            resolved["formal_sample_smoke"] = _formal_sample_smoke(
                config, args, checkpoint
            )
        print(json_dumps(resolved, indent=2, sort_keys=True))
        return 0
    if not readiness["ready"]:
        raise RuntimeError("Formal ATLAS dataset is not complete; training is refused")
    if args.stage == 2 and not bool(config.training.get("stage2_enabled", False)):
        raise RuntimeError("Stage 2 is disabled in this configuration")

    definitions = load_folds(config.repo_path(config.data["folds"]))
    train_dataset = _dataset(config, args.fold, "train")
    validation_dataset = _dataset(config, args.fold, "validation")
    validate_dataset_split(train_dataset, definitions[args.fold], "train")
    validate_dataset_split(validation_dataset, definitions[args.fold], "validation")
    pos_weight = pos_weight_from_dataset(
        train_dataset, max_pos_weight=float(config.training["max_pos_weight"])
    )

    with tf.device(_tf_device(args.device)):
        model = _make_model(config)
        load_predypocket_pretrained(model, checkpoint)
        configure_stage(
            model,
            args.stage,
            stage2_enabled=bool(config.training.get("stage2_enabled", False)),
        )
        optimizer = make_optimizer(
            float(config.training["learning_rate"]),
            float(config.training["weight_decay"]),
            mixed_precision=bool(config.training.get("mixed_precision", False)),
        )
        if args.resume:
            resume_prefix = (
                output_root / "last_checkpoint"
                if args.resume == "auto"
                else config.repo_path(args.resume)
            )
            status = tf.train.Checkpoint(model=model, optimizer=optimizer).read(
                str(resume_prefix)
            )
            status.assert_existing_objects_matched()
            status.assert_nontrivial_match()
        result = fit_model(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            optimizer=optimizer,
            pos_weight=pos_weight.used_pos_weight,
            output_directory=output_root,
            batch_size=resolved["batch_size"],
            gradient_accumulation_steps=resolved["gradient_accumulation_steps"],
            gradient_clip_norm=float(config.training["gradient_clip_norm"]),
            max_epochs=resolved["max_epochs"],
            early_stopping_patience=int(config.training["early_stopping_patience"]),
            seed=int(config.training["seed"]),
            allow_single_class_validation=bool(
                config.training.get("allow_single_class_validation", False)
            ),
        )
    print(json_dumps({"fit": result.history, "pos_weight": pos_weight.__dict__}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
