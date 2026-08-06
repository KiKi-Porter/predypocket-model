"""Formal DynamicPreDyPocket training on success-only MISATO Dynamic Pocket v1 data."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.checkpoint import (
    INIT_MODES,
    infer_trained_checkpoint_model_type,
    load_predypocket_pretrained,
    restore_trained_checkpoint,
)
from predypocket.losses import compute_train_pos_weight
from predypocket.misato_dataset import (
    MISATO_SPLITS,
    MisatoDynamicPocketDataset,
    MisatoStaticAnchorDataset,
    manifest_split_summary,
)
from predypocket.model import DynamicPreDyPocket, StaticAnchorPreDyPocket
from predypocket.spatial_cache import SpatialFeatureCache, sha256_file
from predypocket.trainer import (
    create_distribution_strategy,
    fit_model,
    make_optimizer,
    validate_global_batch_size,
)


OFFICIAL_SPLIT_COUNTS = {"train": 4133, "validation": 432, "test": 435}
STATIC_ANCHOR_PROTOCOL = {
    "batch_size": 1,
    "gradient_accumulation": 8,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "max_epochs": 50,
    "early_stopping_patience": 8,
    "gradient_clip_norm": 1.0,
    "pos_weight_cap": 20.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-type",
        choices=("dynamic", "static_anchor"),
        default="dynamic",
    )
    parser.add_argument("--pretrained-checkpoint", default="models/predypocket_initializer")
    parser.add_argument("--init-mode", choices=INIT_MODES, default="full_predypocket")
    parser.add_argument("--temporal-mode", choices=("on", "off"), default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--spatial-frame-chunk-size",
        type=int,
        default=1,
        help="Maximum number of trajectory frames encoded by GVP at once.",
    )
    parser.add_argument(
        "--spatial-cache-dir",
        default=None,
        help=(
            "Validated frozen-GVP representation cache. When supplied, the "
            "spatial encoder is skipped during training and validation."
        ),
    )
    parser.add_argument(
        "--distribution-strategy",
        "--strategy",
        choices=("single", "mirrored"),
        default="single",
        help=(
            "Use one device or mirror over every GPU exposed through "
            "CUDA_VISIBLE_DEVICES. --batch-size is the global batch size."
        ),
    )
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--primary-metric", choices=("ap",), default="ap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from CHECKPOINT, or from output-dir/last_checkpoint when omitted.",
    )
    freeze = parser.add_mutually_exclusive_group()
    freeze.add_argument(
        "--encoder-frozen", dest="encoder_frozen", action="store_true", default=True
    )
    freeze.add_argument(
        "--encoder-trainable", dest="encoder_frozen", action="store_false"
    )
    parser.add_argument(
        "--no-verify-hashes",
        dest="verify_hashes",
        action="store_false",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Validate arguments and exit before reading data or creating an optimizer.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "batch_size",
        "spatial_frame_chunk_size",
        "gradient_accumulation",
        "max_epochs",
        "early_stopping_patience",
    )
    for name in positive_integer_fields:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "pos_weight_cap",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.model_type == "dynamic":
        args.temporal_mode = args.temporal_mode or "on"
        if args.temporal_mode != "on":
            raise ValueError(
                "Formal MISATO DynamicPreDyPocket training requires temporal_mode=on"
            )
    else:
        if args.temporal_mode not in {None, "off"}:
            raise ValueError(
                "Independent StaticAnchorPreDyPocket has no temporal mode; omit "
                "--temporal-mode"
            )
        args.temporal_mode = "not_applicable"
        if args.init_mode != "full_predypocket":
            raise ValueError(
                "Formal StaticAnchorPreDyPocket training requires "
                "--init-mode full_predypocket"
            )
        if not args.encoder_frozen:
            raise ValueError(
                "Formal StaticAnchorPreDyPocket Stage 1 requires --encoder-frozen"
            )
        if args.spatial_cache_dir is not None:
            raise ValueError(
                "StaticAnchorPreDyPocket cannot use the multi-frame spatial cache; "
                "it reads raw frame 9 only"
            )
        for name, expected in STATIC_ANCHOR_PROTOCOL.items():
            actual = getattr(args, name)
            if actual != expected:
                raise ValueError(
                    f"Formal StaticAnchorPreDyPocket protocol requires "
                    f"--{name.replace('_', '-')}={expected}, received {actual}"
                )
        if args.distribution_strategy != "single":
            raise ValueError(
                "Formal StaticAnchorPreDyPocket batch_size=1 requires "
                "--distribution-strategy single"
            )
    if args.spatial_cache_dir is not None and not args.encoder_frozen:
        raise ValueError(
            "--spatial-cache-dir requires --encoder-frozen because cached spatial "
            "representations cannot receive encoder gradients"
        )
    if args.distribution_strategy == "mirrored" and args.batch_size < 2:
        raise ValueError("Mirrored training requires --batch-size >= 2")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _assert_label_job_complete(data_dir: Path, manifest: Path) -> dict[str, Any]:
    latest_path = data_dir / "runs" / "latest.json"
    if not latest_path.is_file():
        raise RuntimeError(f"Formal label run status is missing: {latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("run_status") != "complete":
        raise RuntimeError(
            "The MISATO 5k label-generation job is not complete; formal training is refused"
        )
    recorded_manifest = Path(str(latest.get("manifest", ""))).resolve()
    if recorded_manifest != manifest.resolve():
        raise RuntimeError(
            f"Completed label run used a different manifest: {recorded_manifest}"
        )
    return latest


def _parse_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "parse_only": True,
        "would_train": False,
        "optimizer_created": False,
        "manifest": str(Path(args.manifest)),
        "data_dir": str(Path(args.data_dir)),
        "output_dir": str(Path(args.output_dir)),
        "model_type": args.model_type,
        "pretrained_checkpoint": str(Path(args.pretrained_checkpoint)),
        "init_mode": args.init_mode,
        "temporal_mode": args.temporal_mode,
        "input_frame_indices": [9] if args.model_type == "static_anchor" else list(range(10)),
        "temporal_modules_constructed": args.model_type == "dynamic",
        "batch_size": args.batch_size,
        "spatial_frame_chunk_size": args.spatial_frame_chunk_size,
        "spatial_cache_dir": args.spatial_cache_dir,
        "distribution_strategy": args.distribution_strategy,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "gradient_clip_norm": args.gradient_clip_norm,
        "pos_weight_cap": args.pos_weight_cap,
        "primary_metric": args.primary_metric,
        "encoder_frozen": args.encoder_frozen,
        "resume": args.resume,
    }


def _resume_state(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "training_state.json"
    history_path = output_dir / "history.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else []
    )
    return {
        "initial_epoch": int(state.get("next_epoch", len(history))),
        "initial_history": history,
        "initial_best_score": state.get("best_validation_ap"),
        "initial_best_epoch": int(state.get("best_epoch", -1)),
        "initial_stale_epochs": int(state.get("stale_epochs", 0)),
    }


def _assert_output_model_type(output_dir: Path, expected_model_type: str) -> None:
    """Prevent one formal model family from overwriting the other's artifacts."""

    for name in ("best_checkpoint", "last_checkpoint"):
        prefix = output_dir / name
        if prefix.with_suffix(".index").is_file():
            actual = infer_trained_checkpoint_model_type(prefix)
            if actual != expected_model_type:
                raise RuntimeError(
                    f"Training output contains a {actual!r} {name}, expected "
                    f"{expected_model_type!r}: {prefix}"
                )
    state_path = output_dir / "training_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        recorded = state.get("model_type")
        if recorded is not None and recorded != expected_model_type:
            raise RuntimeError(
                f"Training state belongs to model_type={recorded!r}, expected "
                f"{expected_model_type!r}: {state_path}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.parse_only:
        print(json.dumps(_parse_plan(args), indent=2, sort_keys=True))
        return 0

    manifest = Path(args.manifest).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    pretrained = Path(args.pretrained_checkpoint).resolve()
    spatial_cache_dir = (
        Path(args.spatial_cache_dir).resolve()
        if args.spatial_cache_dir is not None
        else None
    )
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise RuntimeError("Training output-dir must be outside the formal label data-dir")
    _assert_output_model_type(output_dir, args.model_type)
    completed_run = _assert_label_job_complete(data_dir, manifest)
    split_summary = manifest_split_summary(manifest)
    if any(split_summary["intersections"].values()):
        raise RuntimeError("Formal train/validation/test manifest splits overlap")
    if split_summary["target_counts"] != OFFICIAL_SPLIT_COUNTS:
        raise RuntimeError(
            "Formal training requires the official 5k split counts "
            f"{OFFICIAL_SPLIT_COUNTS}, received {split_summary['target_counts']}"
        )

    dataset_class = (
        MisatoStaticAnchorDataset
        if args.model_type == "static_anchor"
        else MisatoDynamicPocketDataset
    )
    train_dataset = dataset_class(
        manifest,
        data_dir,
        split="train",
        require_complete=True,
        verify_hashes=args.verify_hashes,
    )
    validation_dataset = dataset_class(
        manifest,
        data_dir,
        split="validation",
        require_complete=True,
        verify_hashes=args.verify_hashes,
    )
    if not len(train_dataset):
        raise RuntimeError("No successful train systems are available")
    if not len(validation_dataset):
        raise RuntimeError("No successful validation systems are available")
    if len(train_dataset) != OFFICIAL_SPLIT_COUNTS["train"]:
        raise RuntimeError(
            "Formal training requires all 4133 train label artifacts to be complete"
        )
    if len(validation_dataset) != OFFICIAL_SPLIT_COUNTS["validation"]:
        raise RuntimeError(
            "Formal model selection requires all 432 validation label artifacts "
            "to be complete"
        )
    if set(train_dataset.system_ids) & set(validation_dataset.system_ids):
        raise RuntimeError("A system appears in both train and validation datasets")
    pos_weight = compute_train_pos_weight(
        train_dataset.iter_supervision(),
        split="train",
        max_pos_weight=args.pos_weight_cap,
    )
    strategy, distribution = create_distribution_strategy(
        args.distribution_strategy
    )
    validate_global_batch_size(args.batch_size, distribution.replica_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest_split_summary.json", split_summary)
    _write_json(output_dir / "train_split_statistics.json", train_dataset.statistics.to_dict())
    _write_json(
        output_dir / "validation_split_statistics.json",
        validation_dataset.statistics.to_dict(),
    )
    _write_json(
        output_dir / "excluded_systems.json",
        {
            "train": [event.__dict__ for event in train_dataset.skip_events],
            "validation": [event.__dict__ for event in validation_dataset.skip_events],
        },
    )
    _write_json(
        output_dir / "used_system_ids.json",
        {
            "train": list(train_dataset.system_ids),
            "validation": list(validation_dataset.system_ids),
            "test": [],
            "test_loaded": False,
        },
    )
    _write_json(output_dir / "pos_weight.json", pos_weight.__dict__)
    _write_json(output_dir / "distribution.json", distribution.to_dict())
    configuration = {
        **_parse_plan(args),
        "parse_only": False,
        "would_train": True,
        "manifest": str(manifest),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "pretrained_checkpoint": str(pretrained),
        "spatial_cache_dir": (
            str(spatial_cache_dir) if spatial_cache_dir is not None else None
        ),
        "label_run_status": completed_run.get("run_status"),
        "pos_weight": pos_weight.__dict__,
        "train_statistics": train_dataset.statistics.to_dict(),
        "validation_statistics": validation_dataset.statistics.to_dict(),
        "test_loaded": False,
        "distribution": distribution.to_dict(),
    }
    _write_json(output_dir / "training_config.json", configuration)
    print(json.dumps({"event": "formal_training_start", **configuration}, sort_keys=True), flush=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    scope = strategy.scope() if strategy is not None else nullcontext()
    with scope:
        if args.model_type == "static_anchor":
            model = StaticAnchorPreDyPocket()
        else:
            model = DynamicPreDyPocket(
                input_frame_count=10,
                spatial_frame_chunk_size=args.spatial_frame_chunk_size,
                temporal_mode="on",
            )
        load_predypocket_pretrained(
            model,
            pretrained,
            output_dir / "initialization_report.json",
            init_mode=args.init_mode,
            encoder_frozen=args.encoder_frozen,
        )
        optimizer = make_optimizer(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            mixed_precision=False,
        )

        resume_values: dict[str, Any] = {}
        if args.resume is not None:
            resume_prefix = (
                output_dir / "last_checkpoint"
                if args.resume == "auto"
                else Path(args.resume).resolve()
            )
            if resume_prefix.with_suffix(".index").is_file():
                restore_trained_checkpoint(
                    model, resume_prefix, optimizer=optimizer
                )
                resume_values = _resume_state(output_dir)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_resumed",
                            "checkpoint": str(resume_prefix),
                            "next_epoch": resume_values["initial_epoch"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            elif args.resume != "auto":
                raise FileNotFoundError(
                    f"Resume checkpoint does not exist: {resume_prefix}"
                )
            else:
                print(
                    json.dumps(
                        {
                            "event": "resume_auto_no_checkpoint",
                            "checkpoint": str(resume_prefix),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        spatial_cache = None
        if spatial_cache_dir is not None:
            expected_system_ids = tuple(
                list(train_dataset.system_ids) + list(validation_dataset.system_ids)
            )
            spatial_cache = SpatialFeatureCache(
                spatial_cache_dir,
                manifest_path=manifest,
                checkpoint_path=pretrained,
                expected_system_ids=expected_system_ids,
                input_frame_count=10,
                feature_dim=int(model.d_static),
                spatial_frame_chunk_size=args.spatial_frame_chunk_size,
                verify_file_hashes=args.verify_hashes,
            )
            configuration["spatial_cache_metadata"] = str(
                spatial_cache_dir / "metadata.json"
            )
            configuration["spatial_cache_metadata_sha256"] = sha256_file(
                spatial_cache_dir / "metadata.json"
            )
            _write_json(output_dir / "training_config.json", configuration)
            print(
                json.dumps(
                    {
                        "event": "spatial_cache_attached",
                        "cache_dir": str(spatial_cache_dir),
                        "system_count": len(expected_system_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            spatial_cache = None

    result = fit_model(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        optimizer=optimizer,
        pos_weight=pos_weight.used_pos_weight,
        output_directory=output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_clip_norm=args.gradient_clip_norm,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        primary_validation_metric=args.primary_metric,
        strategy=strategy,
        spatial_cache=spatial_cache,
        **resume_values,
    )
    print(
        json.dumps(
            {
                "event": "formal_training_complete",
                "best_epoch": result.best_epoch,
                "best_validation_ap": result.best_validation_pr_auc,
                "epochs_completed": result.epochs_completed,
                "optimizer_steps": result.counters.optimizer_step_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
