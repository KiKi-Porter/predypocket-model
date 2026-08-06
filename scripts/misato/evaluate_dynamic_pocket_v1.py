"""Validation/test evaluation for a trained MISATO DynamicPreDyPocket checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predypocket.checkpoint import (
    checkpoint_sha256,
    restore_trained_checkpoint,
)
from predypocket.metrics import (
    evaluate_test_with_frozen_validation_threshold,
    evaluate_grouped_metrics,
    select_validation_threshold,
    sigmoid,
)
from predypocket.misato_dataset import (
    MisatoDynamicPocketDataset,
    MisatoStaticAnchorDataset,
    iter_misato_batches,
    manifest_split_summary,
    model_input_whitelist,
)
from predypocket.model import (
    DynamicPreDyPocket,
    StaticAnchorPreDyPocket,
    build_dynamic_model,
    build_static_anchor_model,
)
from predypocket.spatial_cache import SpatialFeatureCache, sha256_file
from predypocket.trainer import (
    create_distribution_strategy,
    distributed_predict_batch,
    validate_global_batch_size,
)


MISATO_TRAINED_STATIC_ANCHOR_MODE = "misato_trained_static_anchor"
DYNAMIC_TEMPORAL_OFF_MODE = "dynamic_temporal_off"
DYNAMIC_TEMPORAL_ON_MODE = "dynamic_temporal_on"
THREE_MODE_ORDER = (
    MISATO_TRAINED_STATIC_ANCHOR_MODE,
    DYNAMIC_TEMPORAL_OFF_MODE,
    DYNAMIC_TEMPORAL_ON_MODE,
)
SUMMARY_METRICS = (
    "ap",
    "pr_auc",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "mcc",
    "balanced_accuracy",
    "threshold",
    "positive_rate",
    "predicted_positive_rate",
    "protein_macro_ap",
)
EXPECTED_VALIDATION_SYSTEM_COUNT = 432
EXPECTED_VALIDATION_RESIDUE_COUNT = 215_021
EXPECTED_TEST_SYSTEM_COUNT = 435
EXPECTED_TEST_RESIDUE_COUNT = 192_136
PUBLISHED_PREDYPOCKET_ZERO_SHOT = {
    "role": "supplementary_zero_shot_transfer_baseline",
    "included_in_main_three_model_comparison": False,
    "ap": 0.108774,
    "pr_auc": 0.108645,
    "roc_auc": 0.819227,
    "f1": 0.184332,
    "threshold": 0.864180,
}
OFFICIAL_SPLIT_COUNTS = {"train": 4133, "validation": 432, "test": 435}
EXPECTED_SPLIT_SYSTEM_COUNTS = {
    "validation": EXPECTED_VALIDATION_SYSTEM_COUNT,
    "test": EXPECTED_TEST_SYSTEM_COUNT,
}
EXPECTED_SPLIT_RESIDUE_COUNTS = {
    "validation": EXPECTED_VALIDATION_RESIDUE_COUNT,
    "test": EXPECTED_TEST_RESIDUE_COUNT,
}

# Formal test protocol frozen by the completed three-mode validation run. Updating
# any value here requires a new validation-only model/threshold selection review.
FORMAL_TEST_PROTOCOL = "misato_dynamic_pocket_v1_three_mode_test_v1"
FORMAL_MANIFEST_SHA256 = (
    "fd6532a86a99ac34100f7c346002eadef95c3bd45582357fd0647f7cbc315bc4"
)
FORMAL_METRIC_CODE_SHA256 = (
    "0eab8e34340554da8dcc649a543f6537163c25e87b2c6f48a2cc878b2bc0c300"
)
FORMAL_CHECKPOINT_SHA256 = {
    "dynamic": "562b4f304ea7e078ed3bd14bbe04f08bd0fa0164823ef03a460958158e7a178b",
    "static_anchor": "5425ecfbb04ebfb98bd0aee7e9c866a10663b8bc1285fbaade93bb8e84ae4f6c",
}
FORMAL_VALIDATION_THRESHOLDS = {
    MISATO_TRAINED_STATIC_ANCHOR_MODE: 0.6949979246904352,
    DYNAMIC_TEMPORAL_OFF_MODE: 0.5656201080626241,
    DYNAMIC_TEMPORAL_ON_MODE: 0.6965455016639148,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--dynamic-checkpoint",
        "--checkpoint",
        dest="dynamic_checkpoint",
        required=True,
        help="Best checkpoint from independent DynamicPreDyPocket training.",
    )
    parser.add_argument(
        "--static-anchor-checkpoint",
        default=None,
        help=(
            "Best checkpoint from independent MISATO StaticAnchorPreDyPocket "
            "training; required by --compare-three-modes."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pretrained-checkpoint",
        default="models/predypocket_initializer",
        help="Released PreDyPocket checkpoint used to build the spatial cache.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluate validation with threshold selection or test with frozen validation artifacts.",
    )
    parser.add_argument("--temporal-mode", choices=("on", "off"), default="on")
    parser.add_argument(
        "--compare-three-modes",
        action="store_true",
        help=(
            "Evaluate the independently MISATO-trained static anchor and one "
            "dynamic checkpoint with temporal fusion disabled/enabled."
        ),
    )
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
            "spatial encoder is skipped during evaluation."
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
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument(
        "--frozen-validation-report",
        default=None,
        help=(
            "Validation three-mode comparison report used to freeze test "
            "thresholds, checkpoint hashes and manifest identity. Required for --split test."
        ),
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
        help="Validate arguments and exit without loading data or a checkpoint.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.spatial_frame_chunk_size < 1:
        raise ValueError("--spatial-frame-chunk-size must be positive")
    if args.temporal_mode != "on":
        raise ValueError("Formal MISATO evaluation requires temporal_mode=on")
    if args.split == "validation":
        if not args.select_threshold:
            raise ValueError("Validation evaluation requires --select-threshold")
        if args.frozen_validation_report is not None:
            raise ValueError(
                "--frozen-validation-report is only valid for --split test"
            )
    else:
        if not args.compare_three_modes:
            raise ValueError(
                "Formal test evaluation requires --compare-three-modes so all "
                "three frozen thresholds and both checkpoints are evaluated"
            )
        if args.select_threshold:
            raise ValueError(
                "Test evaluation must use frozen validation thresholds; "
                "do not pass --select-threshold"
            )
        if not args.frozen_validation_report:
            raise ValueError(
                "Test evaluation requires --frozen-validation-report from validation"
            )
        if not args.verify_hashes:
            raise ValueError("Formal test evaluation does not allow --no-verify-hashes")
    if args.distribution_strategy == "mirrored" and args.batch_size < 2:
        raise ValueError("Mirrored evaluation requires --batch-size >= 2")
    if args.compare_three_modes and not args.static_anchor_checkpoint:
        raise ValueError(
            "--compare-three-modes requires --static-anchor-checkpoint; the "
            "released PreDyPocket checkpoint is supplementary only"
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _assert_label_job_complete(data_dir: Path, manifest: Path) -> None:
    latest_path = data_dir / "runs" / "latest.json"
    if not latest_path.is_file():
        raise RuntimeError(f"Formal label run status is missing: {latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("run_status") != "complete":
        raise RuntimeError(
            "The MISATO 5k label-generation job is not complete; formal evaluation is refused"
        )
    if Path(str(latest.get("manifest", ""))).resolve() != manifest.resolve():
        raise RuntimeError("Completed label run used a different manifest")


def _load_frozen_validation_artifacts(
    report_path: str | Path,
    *,
    manifest: Path,
    dynamic_checkpoint: Path,
    static_anchor_checkpoint: Path | None,
    compare_three_modes: bool,
) -> dict[str, Any]:
    """Load and verify validation-selected test artifacts before any test data is read."""

    if not compare_three_modes or static_anchor_checkpoint is None:
        raise RuntimeError(
            "Formal test evaluation requires all three modes and both checkpoints"
        )
    path = Path(report_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen validation report does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Frozen validation report is not valid JSON: {path}") from exc
    if not isinstance(report, Mapping):
        raise RuntimeError("Frozen validation report must contain a JSON object")
    if report.get("schema_version") != 2:
        raise RuntimeError("Frozen validation report has an unsupported schema version")
    if report.get("comparison") != "misato_validation_three_model_modes":
        raise RuntimeError("Frozen report is not the formal three-mode validation report")
    if report.get("split") != "validation":
        raise RuntimeError("Frozen report must be generated from the validation split")
    if report.get("test_loaded") is not False:
        raise RuntimeError("Frozen validation report has an invalid test_loaded flag")
    if report.get("threshold_source") != "validation":
        raise RuntimeError("Frozen validation report must declare validation thresholds")
    if report.get("threshold_policy") != "independent_validation_max_f1_per_mode":
        raise RuntimeError("Frozen validation report has an invalid threshold policy")
    if report.get("mode_order") != list(THREE_MODE_ORDER):
        raise RuntimeError("Frozen validation report has an invalid three-mode order")
    if report.get("sample_count") != EXPECTED_VALIDATION_RESIDUE_COUNT:
        raise RuntimeError("Frozen validation report has an invalid residue count")
    if report.get("validation_system_count") != EXPECTED_VALIDATION_SYSTEM_COUNT:
        raise RuntimeError("Frozen validation report has an invalid system count")
    if report.get("checkpoints_are_independent") is not True:
        raise RuntimeError("Frozen validation checkpoints are not independent")
    frozen = report.get("frozen_validation_artifacts")
    if not isinstance(frozen, Mapping):
        raise RuntimeError(
            "Frozen validation report does not contain frozen_validation_artifacts"
        )
    if frozen.get("test_threshold_reselection_forbidden") is not True:
        raise RuntimeError("Frozen report does not forbid test threshold reselection")
    if frozen.get("test_checkpoint_reselection_forbidden") is not True:
        raise RuntimeError("Frozen report does not forbid test checkpoint reselection")

    validation_system_ids = frozen.get("validation_system_ids")
    if (
        not isinstance(validation_system_ids, list)
        or not all(isinstance(system_id, str) for system_id in validation_system_ids)
        or len(validation_system_ids) != EXPECTED_VALIDATION_SYSTEM_COUNT
        or len(set(validation_system_ids)) != EXPECTED_VALIDATION_SYSTEM_COUNT
        or report.get("validation_system_ids") != validation_system_ids
    ):
        raise RuntimeError("Frozen validation system identity/order is invalid")

    manifest_hash = frozen.get("manifest_sha256")
    actual_manifest_hash = sha256_file(manifest)
    if (
        manifest_hash != FORMAL_MANIFEST_SHA256
        or actual_manifest_hash != FORMAL_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Manifest hash differs from the frozen formal validation manifest; "
            "refusing test evaluation"
        )

    metric_code_path = REPO_ROOT / "predypocket" / "metrics.py"
    actual_metric_code_hash = sha256_file(metric_code_path)
    if (
        frozen.get("metric_code_sha256") != FORMAL_METRIC_CODE_SHA256
        or actual_metric_code_hash != FORMAL_METRIC_CODE_SHA256
    ):
        raise RuntimeError(
            "Metric implementation differs from frozen validation; refusing test evaluation"
        )

    checkpoint_reports = {
        "dynamic": frozen.get("dynamic_checkpoint"),
        "static_anchor": frozen.get("static_anchor_checkpoint"),
    }
    expected_checkpoints = {
        "dynamic": dynamic_checkpoint,
        "static_anchor": static_anchor_checkpoint,
    }
    for model_type, checkpoint in expected_checkpoints.items():
        checkpoint_report = checkpoint_reports[model_type]
        if not isinstance(checkpoint_report, Mapping):
            raise RuntimeError(
                f"Frozen validation report lacks the {model_type} checkpoint report"
            )
        if checkpoint_report.get("model_type") != model_type:
            raise RuntimeError(
                f"Frozen {model_type} checkpoint has an invalid model type"
            )
        expected_hash = FORMAL_CHECKPOINT_SHA256[model_type]
        if checkpoint_report.get("checkpoint_hash") != expected_hash:
            raise RuntimeError(
                f"Frozen {model_type} checkpoint is not the selected formal checkpoint"
            )
        top_level_checkpoint = report.get(f"{model_type}_checkpoint")
        if (
            not isinstance(top_level_checkpoint, Mapping)
            or top_level_checkpoint.get("checkpoint_hash") != expected_hash
        ):
            raise RuntimeError(
                f"Frozen {model_type} checkpoint reports are inconsistent"
            )
        actual_hash = checkpoint_sha256(checkpoint)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"{model_type} checkpoint hash differs from validation; "
                "refusing test evaluation"
            )

    threshold_keys = {
        MISATO_TRAINED_STATIC_ANCHOR_MODE: "static_anchor_validation_threshold",
        DYNAMIC_TEMPORAL_OFF_MODE: "dynamic_off_validation_threshold",
        DYNAMIC_TEMPORAL_ON_MODE: "dynamic_on_validation_threshold",
    }
    thresholds: dict[str, float] = {}
    mode_reports = report.get("modes")
    if not isinstance(mode_reports, Mapping):
        raise RuntimeError("Frozen validation report does not contain mode reports")
    for mode, key in threshold_keys.items():
        value = frozen.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Frozen validation threshold is invalid for {mode}")
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError(f"Frozen validation threshold is invalid for {mode}")
        if threshold != FORMAL_VALIDATION_THRESHOLDS[mode]:
            raise RuntimeError(
                f"Frozen validation threshold differs from the formal value for {mode}"
            )
        mode_report = mode_reports.get(mode)
        mode_threshold = (
            mode_report.get("threshold_selection")
            if isinstance(mode_report, Mapping)
            else None
        )
        if (
            not isinstance(mode_threshold, Mapping)
            or mode_threshold.get("threshold_selection_status")
            != "selected_validation_max_f1"
            or mode_threshold.get("threshold") != threshold
        ):
            raise RuntimeError(
                f"Frozen validation mode report is inconsistent for {mode}"
            )
        thresholds[mode] = threshold
    return {
        "protocol": FORMAL_TEST_PROTOCOL,
        "source_report": str(path),
        "source_report_sha256": sha256_file(path),
        "source_split": "validation",
        "manifest_sha256": actual_manifest_hash,
        "metric_code_sha256": actual_metric_code_hash,
        "checkpoint_hashes": {
            model_type: FORMAL_CHECKPOINT_SHA256[model_type]
            for model_type in expected_checkpoints
        },
        "thresholds": thresholds,
        "test_reselection_forbidden": True,
        "test_checkpoint_reselection_forbidden": True,
    }


def _collect_validation_predictions(
    predictors: Mapping[
        str, tuple[tf.keras.Model, Mapping[str, Any]]
    ],
    dataset: MisatoDynamicPocketDataset,
    batch_size: int,
    strategy: tf.distribute.Strategy | None = None,
    spatial_cache: SpatialFeatureCache | None = None,
) -> dict[str, Any]:
    if not predictors:
        raise ValueError("At least one evaluation predictor is required")
    labels: list[np.ndarray] = []
    logits: dict[str, list[np.ndarray]] = {name: [] for name in predictors}
    system_ids: list[str] = []
    residue_indices: list[int] = []
    for batch in iter_misato_batches(
        dataset, batch_size=batch_size, shuffle=False
    ):
        if spatial_cache is not None:
            batch = spatial_cache.attach_batch(batch)
        batch_values: dict[str, np.ndarray] = {}
        for name, (model, call_kwargs) in predictors.items():
            if strategy is not None and int(strategy.num_replicas_in_sync) > 1:
                values = np.asarray(
                    distributed_predict_batch(
                        model,
                        batch,
                        strategy,
                        model_call_kwargs=call_kwargs,
                    ),
                    dtype=np.float64,
                )
            else:
                batch_logits = model(
                    **model_input_whitelist(batch),
                    training=False,
                    **dict(call_kwargs),
                )
                values = np.asarray(batch_logits.numpy(), dtype=np.float64)
            if tuple(values.shape) != tuple(batch["labels"].shape):
                raise RuntimeError(
                    f"{name} logit/label shape mismatch: "
                    f"{values.shape} vs {batch['labels'].shape}"
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{name} evaluation logits contain NaN or Inf")
            batch_values[name] = values
        for index, system_id in enumerate(batch["system_id"]):
            effective_mask = np.asarray(batch["residue_mask"][index], dtype=bool) & np.asarray(
                batch["training_mask"][index], dtype=bool
            )
            effective_indices = np.flatnonzero(effective_mask)
            labels.append(
                np.asarray(batch["labels"][index], dtype=np.float32)[effective_mask]
            )
            for name, values in batch_values.items():
                logits[name].append(values[index][effective_mask])
            system_ids.extend([str(system_id)] * int(np.sum(effective_mask)))
            residue_indices.extend(int(value) for value in effective_indices)
    if not labels:
        raise RuntimeError("Evaluation split contains no effective supervision")
    return {
        "labels": np.concatenate(labels),
        "logits": {
            name: np.concatenate(mode_logits)
            for name, mode_logits in logits.items()
        },
        "system_ids": np.asarray(system_ids, dtype=str),
        "residue_indices": np.asarray(residue_indices, dtype=np.int32),
    }


def _evaluate_predictions(
    labels: np.ndarray,
    logits: np.ndarray,
    system_ids: np.ndarray,
    *,
    split: str,
    frozen_threshold: float | None = None,
) -> dict[str, Any]:
    if split == "validation":
        threshold = select_validation_threshold(
            labels,
            sigmoid(logits),
            split="validation",
        )
        metrics = evaluate_grouped_metrics(
            labels,
            logits,
            system_ids,
            threshold=threshold["threshold"],
            from_logits=True,
        )
    elif split == "test":
        if frozen_threshold is None:
            raise ValueError("Test evaluation requires a frozen validation threshold")
        metrics = evaluate_test_with_frozen_validation_threshold(
            labels,
            logits,
            system_ids,
            threshold=frozen_threshold,
            threshold_source_split="validation",
            from_logits=True,
        )
        threshold = {
            "threshold": float(frozen_threshold),
            "threshold_selection_status": "frozen_validation_threshold",
            "threshold_source_split": "validation",
            "evaluation_split": "test",
            "test_reselection_forbidden": True,
        }
    else:
        raise ValueError(f"Unsupported evaluation split: {split!r}")
    return {
        "threshold_selection": threshold,
        "metrics": metrics,
    }


def _paired_logit_comparison(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError(
            f"Paired prediction shapes differ: {left_values.shape} vs {right_values.shape}"
        )
    logit_difference = np.abs(left_values - right_values)
    probability_difference = np.abs(sigmoid(left_values) - sigmoid(right_values))
    exact_count = int(np.count_nonzero(left_values == right_values))
    return {
        "sample_count": int(left_values.size),
        "exact_logit_match_count": exact_count,
        "all_logits_exactly_equal": bool(exact_count == left_values.size),
        "mean_absolute_logit_difference": float(np.mean(logit_difference)),
        "max_absolute_logit_difference": float(np.max(logit_difference)),
        "mean_absolute_probability_difference": float(
            np.mean(probability_difference)
        ),
        "max_absolute_probability_difference": float(
            np.max(probability_difference)
        ),
    }


def _comparison_summary(
    mode_reports: Mapping[str, Mapping[str, Any]],
    logits: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    pooled: dict[str, dict[str, Any]] = {}
    for mode, report in mode_reports.items():
        pooled_values = report["metrics"]["pooled"]
        values = {
            metric: pooled_values.get(metric)
            for metric in SUMMARY_METRICS
            if metric != "protein_macro_ap"
        }
        values["protein_macro_ap"] = report["metrics"]["macro_protein"].get("ap")
        pooled[mode] = values
    reference = pooled[DYNAMIC_TEMPORAL_ON_MODE]
    deltas: dict[str, dict[str, float | None]] = {}
    for mode in THREE_MODE_ORDER:
        if mode == DYNAMIC_TEMPORAL_ON_MODE:
            continue
        deltas[mode] = {}
        for metric in SUMMARY_METRICS:
            if metric == "threshold":
                continue
            value = pooled[mode][metric]
            reference_value = reference[metric]
            deltas[mode][metric] = (
                None
                if value is None or reference_value is None
                else float(value - reference_value)
            )

    paired: dict[str, Any] = {}
    for left_index, left_mode in enumerate(THREE_MODE_ORDER):
        for right_mode in THREE_MODE_ORDER[left_index + 1 :]:
            paired[f"{left_mode}_vs_{right_mode}"] = _paired_logit_comparison(
                logits[left_mode], logits[right_mode]
            )
    return {
        "pooled_metrics": pooled,
        "metric_delta_comparator_minus_dynamic_temporal_on": deltas,
        "metric_delta_dynamic_temporal_on_minus_comparator": {
            mode: {
                metric: None if value is None else -float(value)
                for metric, value in mode_deltas.items()
            }
            for mode, mode_deltas in deltas.items()
        },
        "paired_logit_comparisons": paired,
    }


def _assert_prediction_alignment(
    static_predictions: Mapping[str, Any],
    dynamic_predictions: Mapping[str, Any],
    *,
    split: str,
    expected_residue_count: int,
) -> None:
    """Require identical labels and residue identity/order across both data paths."""

    for name in ("labels", "system_ids", "residue_indices"):
        static_values = np.asarray(static_predictions[name])
        dynamic_values = np.asarray(dynamic_predictions[name])
        if static_values.shape != dynamic_values.shape or not np.array_equal(
            static_values, dynamic_values
        ):
            raise RuntimeError(
                f"Static/dynamic {split} alignment differs for {name}: "
                f"{static_values.shape} vs {dynamic_values.shape}"
            )
    sample_count = int(np.asarray(dynamic_predictions["labels"]).size)
    if sample_count != expected_residue_count:
        raise RuntimeError(
            f"Formal {split} evaluation must align to exactly "
            f"{expected_residue_count} effective residues, got "
            f"{sample_count}"
        )


def _main_validation_table(
    mode_reports: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    display_names = {
        MISATO_TRAINED_STATIC_ANCHOR_MODE: "MISATO-trained Static Anchor",
        DYNAMIC_TEMPORAL_OFF_MODE: "Dynamic checkpoint temporal-off",
        DYNAMIC_TEMPORAL_ON_MODE: "Dynamic checkpoint temporal-on",
    }
    table: list[dict[str, Any]] = []
    for mode in THREE_MODE_ORDER:
        pooled = mode_reports[mode]["metrics"]["pooled"]
        macro = mode_reports[mode]["metrics"]["macro_protein"]
        table.append(
            {
                "mode": mode,
                "display_name": display_names[mode],
                "ap": pooled.get("ap"),
                "pr_auc": pooled.get("pr_auc"),
                "roc_auc": pooled.get("roc_auc"),
                "precision": pooled.get("precision"),
                "recall": pooled.get("recall"),
                "f1": pooled.get("f1"),
                "specificity": pooled.get("specificity"),
                "mcc": pooled.get("mcc"),
                "threshold": pooled.get("threshold"),
                "positive_rate": pooled.get("positive_rate"),
                "predicted_positive_rate": pooled.get("predicted_positive_rate"),
                "protein_macro_ap": macro.get("ap"),
            }
        )
    return table


def _scientific_comparisons(
    comparison_summary: Mapping[str, Any],
) -> dict[str, Any]:
    on_minus = comparison_summary[
        "metric_delta_dynamic_temporal_on_minus_comparator"
    ]
    return {
        "primary_dynamic_temporal_on_vs_misato_trained_static_anchor": {
            "meaning": (
                "Same official split, labels, pretrained initialization and training "
                "protocol: frames 0-9 dynamics versus frame 9 only."
            ),
            "left": DYNAMIC_TEMPORAL_ON_MODE,
            "right": MISATO_TRAINED_STATIC_ANCHOR_MODE,
            "metric_delta_left_minus_right": on_minus[
                MISATO_TRAINED_STATIC_ANCHOR_MODE
            ],
        },
        "ablation_dynamic_temporal_on_vs_dynamic_temporal_off": {
            "meaning": (
                "Post-training inference ablation of temporal contribution within "
                "the same DynamicPreDyPocket checkpoint and classifier."
            ),
            "left": DYNAMIC_TEMPORAL_ON_MODE,
            "right": DYNAMIC_TEMPORAL_OFF_MODE,
            "metric_delta_left_minus_right": on_minus[DYNAMIC_TEMPORAL_OFF_MODE],
        },
    }


def _trained_checkpoint_report(
    model: tf.keras.Model, checkpoint: Path, model_type: str
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_hash": checkpoint_sha256(checkpoint),
        "model_type": model_type,
        "model_tensor_count": len(model.variables),
        "model_parameter_count": int(
            sum(int(tf.size(variable).numpy()) for variable in model.variables)
        ),
        "missing_model_objects": [],
        "unexpected_non_model_objects": [
            name
            for name, _ in tf.train.list_variables(str(checkpoint))
            if name.startswith("optimizer/") or "/.OPTIMIZER_SLOT/" in name
        ],
        "shape_mismatches": [],
        "restore_assert_existing_objects_matched": True,
        "restore_assert_nontrivial_match": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    parse_plan = {
        "parse_only": bool(args.parse_only),
        "would_evaluate": not args.parse_only,
        "training_enabled": False,
        "optimizer_created": False,
        "manifest": args.manifest,
        "data_dir": args.data_dir,
        "dynamic_checkpoint": args.dynamic_checkpoint,
        "static_anchor_checkpoint": args.static_anchor_checkpoint,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "pretrained_checkpoint_role": "dynamic_spatial_cache_provenance_only",
        "output_dir": args.output_dir,
        "split": args.split,
        "temporal_mode": args.temporal_mode,
        "compare_three_modes": bool(args.compare_three_modes),
        "evaluation_modes": (
            list(THREE_MODE_ORDER)
            if args.compare_three_modes
            else [DYNAMIC_TEMPORAL_ON_MODE]
        ),
        "batch_size": args.batch_size,
        "spatial_frame_chunk_size": args.spatial_frame_chunk_size,
        "spatial_cache_dir": args.spatial_cache_dir,
        "distribution_strategy": args.distribution_strategy,
        "select_threshold": args.select_threshold,
        "frozen_validation_report": args.frozen_validation_report,
        "threshold_source": "validation",
        "threshold_policy": (
            "validation_max_f1_per_mode"
            if args.split == "validation"
            else "frozen_validation_thresholds"
        ),
        "formal_test_protocol": (
            FORMAL_TEST_PROTOCOL if args.split == "test" else None
        ),
        "formal_validation_thresholds": (
            dict(FORMAL_VALIDATION_THRESHOLDS) if args.split == "test" else None
        ),
        "formal_checkpoint_sha256": (
            dict(FORMAL_CHECKPOINT_SHA256) if args.split == "test" else None
        ),
        "would_load_test": args.split == "test" and not args.parse_only,
        "test_loaded": False,
    }
    if args.parse_only:
        print(json.dumps(parse_plan, indent=2, sort_keys=True))
        return 0

    manifest = Path(args.manifest).resolve()
    data_dir = Path(args.data_dir).resolve()
    dynamic_checkpoint = Path(args.dynamic_checkpoint).resolve()
    static_anchor_checkpoint = (
        Path(args.static_anchor_checkpoint).resolve()
        if args.static_anchor_checkpoint is not None
        else None
    )
    pretrained = Path(args.pretrained_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise RuntimeError("Evaluation output-dir must be outside the formal label data-dir")
    if not dynamic_checkpoint.with_suffix(".index").is_file():
        raise FileNotFoundError(
            f"Dynamic trained checkpoint does not exist: {dynamic_checkpoint}"
        )
    if args.compare_three_modes and (
        static_anchor_checkpoint is None
        or not static_anchor_checkpoint.with_suffix(".index").is_file()
    ):
        raise FileNotFoundError(
            f"Static-anchor trained checkpoint does not exist: "
            f"{static_anchor_checkpoint}"
        )
    if (
        args.compare_three_modes
        and static_anchor_checkpoint == dynamic_checkpoint
    ):
        raise RuntimeError(
            "Static Anchor and DynamicPreDyPocket checkpoints must be independent"
        )
    frozen_validation = None
    if args.split == "test":
        frozen_validation = _load_frozen_validation_artifacts(
            args.frozen_validation_report,
            manifest=manifest,
            dynamic_checkpoint=dynamic_checkpoint,
            static_anchor_checkpoint=static_anchor_checkpoint,
            compare_three_modes=args.compare_three_modes,
        )
    _assert_label_job_complete(data_dir, manifest)
    split_summary = manifest_split_summary(manifest)
    if any(split_summary["intersections"].values()):
        raise RuntimeError("Formal manifest splits overlap")
    if split_summary["target_counts"] != OFFICIAL_SPLIT_COUNTS:
        raise RuntimeError(
            "Formal evaluation requires the official 5k split counts "
            f"{OFFICIAL_SPLIT_COUNTS}, received {split_summary['target_counts']}"
        )
    evaluation_dataset = MisatoDynamicPocketDataset(
        manifest,
        data_dir,
        split=args.split,
        require_complete=True,
        verify_hashes=args.verify_hashes,
    )
    expected_system_count = EXPECTED_SPLIT_SYSTEM_COUNTS[args.split]
    expected_residue_count = EXPECTED_SPLIT_RESIDUE_COUNTS[args.split]
    if len(evaluation_dataset) != expected_system_count:
        raise RuntimeError(
            f"Formal {args.split} evaluation requires all {expected_system_count} "
            "systems to be complete"
        )
    static_evaluation_dataset = None
    if args.compare_three_modes:
        static_evaluation_dataset = MisatoStaticAnchorDataset(
            manifest,
            data_dir,
            split=args.split,
            require_complete=True,
            verify_hashes=args.verify_hashes,
        )
        if static_evaluation_dataset.system_ids != evaluation_dataset.system_ids:
            raise RuntimeError(
                f"Static and dynamic {args.split} datasets have different system ordering"
            )

    strategy, distribution = create_distribution_strategy(
        args.distribution_strategy
    )
    validate_global_batch_size(args.batch_size, distribution.replica_count)
    scope = strategy.scope() if strategy is not None else nullcontext()
    static_model: StaticAnchorPreDyPocket | None = None
    with scope:
        dynamic_model = DynamicPreDyPocket(
            input_frame_count=10,
            spatial_frame_chunk_size=args.spatial_frame_chunk_size,
            temporal_mode="on",
        )
        build_dynamic_model(dynamic_model)
        restore_trained_checkpoint(dynamic_model, dynamic_checkpoint)
        if args.compare_three_modes:
            assert static_anchor_checkpoint is not None
            static_model = StaticAnchorPreDyPocket()
            build_static_anchor_model(static_model)
            restore_trained_checkpoint(static_model, static_anchor_checkpoint)
        spatial_cache = None
        if args.spatial_cache_dir is not None:
            spatial_cache = SpatialFeatureCache(
                Path(args.spatial_cache_dir).resolve(),
                manifest_path=manifest,
                checkpoint_path=pretrained,
                expected_system_ids=evaluation_dataset.system_ids,
                input_frame_count=10,
                feature_dim=int(dynamic_model.d_static),
                spatial_frame_chunk_size=args.spatial_frame_chunk_size,
                verify_file_hashes=args.verify_hashes,
                allow_extra_system_ids=True,
            )
    dynamic_checkpoint_report = _trained_checkpoint_report(
        dynamic_model, dynamic_checkpoint, "dynamic"
    )
    static_checkpoint_report = (
        _trained_checkpoint_report(
            static_model, static_anchor_checkpoint, "static_anchor"
        )
        if static_model is not None and static_anchor_checkpoint is not None
        else None
    )

    if args.compare_three_modes:
        if static_model is None or static_evaluation_dataset is None:
            raise RuntimeError("Independent static comparison model was not initialized")
        dynamic_predictors: dict[
            str, tuple[tf.keras.Model, Mapping[str, Any]]
        ] = {
            DYNAMIC_TEMPORAL_OFF_MODE: (
                dynamic_model,
                {"temporal_mode": "off"},
            ),
            DYNAMIC_TEMPORAL_ON_MODE: (
                dynamic_model,
                {"temporal_mode": "on"},
            ),
        }
    else:
        dynamic_predictors = {
            DYNAMIC_TEMPORAL_ON_MODE: (
                dynamic_model,
                {"temporal_mode": "on"},
            )
        }

    dynamic_predictions = _collect_validation_predictions(
        dynamic_predictors,
        evaluation_dataset,
        args.batch_size,
        strategy=strategy,
        spatial_cache=spatial_cache,
    )
    predictions = dynamic_predictions
    if args.compare_three_modes:
        assert static_model is not None
        assert static_evaluation_dataset is not None
        static_predictions = _collect_validation_predictions(
            {MISATO_TRAINED_STATIC_ANCHOR_MODE: (static_model, {})},
            static_evaluation_dataset,
            args.batch_size,
            strategy=strategy,
            spatial_cache=None,
        )
        _assert_prediction_alignment(
            static_predictions,
            dynamic_predictions,
            split=args.split,
            expected_residue_count=expected_residue_count,
        )
        predictions = {
            **dynamic_predictions,
            "logits": {
                MISATO_TRAINED_STATIC_ANCHOR_MODE: static_predictions["logits"][
                    MISATO_TRAINED_STATIC_ANCHOR_MODE
                ],
                **dynamic_predictions["logits"],
            },
        }
    elif int(np.asarray(predictions["labels"]).size) != expected_residue_count:
        raise RuntimeError(
            f"Formal {args.split} evaluation must contain exactly "
            f"{expected_residue_count} effective residues"
        )
    mode_reports = {
        mode: _evaluate_predictions(
            predictions["labels"],
            predictions["logits"][mode],
            predictions["system_ids"],
            split=args.split,
            frozen_threshold=(
                frozen_validation["thresholds"][mode]
                if frozen_validation is not None
                else None
            ),
        )
        for mode in predictions["logits"]
    }
    primary_report = mode_reports[DYNAMIC_TEMPORAL_ON_MODE]
    threshold = primary_report["threshold_selection"]
    metrics = primary_report["metrics"]
    report = {
        "split": args.split,
        "temporal_mode": "on",
        "threshold_selection": threshold,
        "metrics": metrics,
        f"{args.split}_statistics": evaluation_dataset.statistics.to_dict(),
        f"{args.split}_system_count": len(evaluation_dataset.system_ids),
        f"{args.split}_system_ids": list(evaluation_dataset.system_ids),
        "threshold_source": "validation",
        "threshold_policy": (
            "validation_max_f1_per_mode"
            if args.split == "validation"
            else "frozen_validation_thresholds"
        ),
        "formal_test_protocol": (
            FORMAL_TEST_PROTOCOL if args.split == "test" else None
        ),
        "pos_weight_computed": False,
        "training_enabled": False,
        "optimizer_created": False,
        "test_loaded": args.split == "test",
        "test_threshold_reselection_forbidden": args.split == "test",
        "test_checkpoint_reselection_forbidden": args.split == "test",
        "distribution": distribution.to_dict(),
        "checkpoint": dynamic_checkpoint_report,
        "dynamic_checkpoint": dynamic_checkpoint_report,
        "static_anchor_checkpoint": static_checkpoint_report,
        "spatial_cache_dir": (
            str(Path(args.spatial_cache_dir).resolve())
            if args.spatial_cache_dir is not None
            else None
        ),
    }
    if frozen_validation is not None:
        report["frozen_validation_source"] = frozen_validation
    if spatial_cache is not None:
        report["spatial_cache_metadata"] = str(
            spatial_cache.root / "metadata.json"
        )
        report["spatial_cache_metadata_sha256"] = sha256_file(
            spatial_cache.root / "metadata.json"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_table_key = (
        "main_validation_table" if args.split == "validation" else "main_test_table"
    )
    if args.compare_three_modes:
        if static_checkpoint_report is None or static_anchor_checkpoint is None:
            raise RuntimeError("Static-anchor checkpoint report is unavailable")
        mode_definitions = {
            MISATO_TRAINED_STATIC_ANCHOR_MODE: {
                "model": "StaticAnchorPreDyPocket",
                "weight_source": "independent_misato_training_checkpoint",
                "checkpoint_path": str(static_anchor_checkpoint),
                "input_frames_used": [9],
                "input_tensor_contains_other_frames": False,
                "classifier_head": "independently_misato_trained_static_checkpoint",
                "temporal_modules_constructed": False,
                "temporal_fusion": False,
            },
            DYNAMIC_TEMPORAL_OFF_MODE: {
                "model": "DynamicPreDyPocket",
                "weight_source": "dynamic_training_checkpoint",
                "checkpoint_path": str(dynamic_checkpoint),
                "input_frames_encoded": list(range(10)),
                "classifier_head": "dynamic_training_checkpoint",
                "temporal_mode": "off",
                "dynamic_contribution_forced_to_zero": True,
                "temporal_fusion": False,
            },
            DYNAMIC_TEMPORAL_ON_MODE: {
                "model": "DynamicPreDyPocket",
                "weight_source": "dynamic_training_checkpoint",
                "checkpoint_path": str(dynamic_checkpoint),
                "input_frames_encoded": list(range(10)),
                "classifier_head": "dynamic_training_checkpoint",
                "temporal_mode": "on",
                "temporal_fusion": True,
            },
        }
        comparison_summary = _comparison_summary(
            mode_reports, predictions["logits"]
        )
        prediction_path = output_dir / f"{args.split}_three_mode_predictions.npz"
        prediction_values: dict[str, np.ndarray] = {
            "labels": np.asarray(predictions["labels"], dtype=np.float32),
            "system_ids": np.asarray(predictions["system_ids"], dtype=str),
            "residue_indices": np.asarray(
                predictions["residue_indices"], dtype=np.int32
            ),
        }
        for mode in THREE_MODE_ORDER:
            prediction_values[f"logits__{mode}"] = np.asarray(
                predictions["logits"][mode], dtype=np.float64
            )
        np.savez_compressed(prediction_path, **prediction_values)

        comparison_path = output_dir / f"{args.split}_three_mode_comparison.json"
        comparison_report = {
            "schema_version": 2,
            "comparison": f"misato_{args.split}_three_model_modes",
            "split": args.split,
            "mode_order": list(THREE_MODE_ORDER),
            "threshold_policy": (
                "independent_validation_max_f1_per_mode"
                if args.split == "validation"
                else "frozen_validation_thresholds"
            ),
            "threshold_source": "validation",
            "test_loaded": args.split == "test",
            "sample_count": int(predictions["labels"].size),
            f"{args.split}_system_count": len(evaluation_dataset.system_ids),
            f"{args.split}_system_ids": list(evaluation_dataset.system_ids),
            f"{args.split}_statistics": evaluation_dataset.statistics.to_dict(),
            f"static_{args.split}_statistics": (
                static_evaluation_dataset.statistics.to_dict()
                if static_evaluation_dataset is not None
                else None
            ),
            "residue_alignment": {
                "sample_count": int(predictions["labels"].size),
                "labels_exactly_equal": True,
                "system_ids_exactly_equal": True,
                "residue_indices_exactly_equal": True,
                "residue_mask_and_training_mask_policy": "intersection",
                "all_logits_finite": True,
            },
            "distribution": distribution.to_dict(),
            "dynamic_checkpoint": dynamic_checkpoint_report,
            "static_anchor_checkpoint": static_checkpoint_report,
            "checkpoints_are_independent": bool(
                dynamic_checkpoint != static_anchor_checkpoint
                and dynamic_checkpoint_report["checkpoint_hash"]
                != static_checkpoint_report["checkpoint_hash"]
            ),
            "spatial_cache_dir": report["spatial_cache_dir"],
            "prediction_archive": str(prediction_path),
            "prediction_archive_sha256": sha256_file(prediction_path),
            "prediction_archive_fields": list(prediction_values),
            "summary": comparison_summary,
            comparison_table_key: _main_validation_table(mode_reports),
            "scientific_comparisons": _scientific_comparisons(comparison_summary),
            "modes": {
                mode: {
                    "definition": mode_definitions[mode],
                    **mode_reports[mode],
                }
                for mode in THREE_MODE_ORDER
            },
        }
        if args.split == "validation":
            comparison_report["supplementary_baselines"] = {
                "published_predypocket_zero_shot": PUBLISHED_PREDYPOCKET_ZERO_SHOT
            }
            comparison_report["frozen_validation_artifacts"] = {
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "validation_system_ids": list(evaluation_dataset.system_ids),
                "static_anchor_checkpoint": static_checkpoint_report,
                "dynamic_checkpoint": dynamic_checkpoint_report,
                "static_anchor_validation_threshold": mode_reports[
                    MISATO_TRAINED_STATIC_ANCHOR_MODE
                ]["threshold_selection"]["threshold"],
                "dynamic_off_validation_threshold": mode_reports[
                    DYNAMIC_TEMPORAL_OFF_MODE
                ]["threshold_selection"]["threshold"],
                "dynamic_on_validation_threshold": mode_reports[
                    DYNAMIC_TEMPORAL_ON_MODE
                ]["threshold_selection"]["threshold"],
                "model_definitions": mode_definitions,
                "metric_code": str(REPO_ROOT / "predypocket" / "metrics.py"),
                "metric_code_sha256": sha256_file(
                    REPO_ROOT / "predypocket" / "metrics.py"
                ),
                "test_threshold_reselection_forbidden": True,
                "test_checkpoint_reselection_forbidden": True,
            }
        else:
            comparison_report["frozen_validation_source"] = frozen_validation
            comparison_report["test_reselection_forbidden"] = True
            comparison_report["test_checkpoint_reselection_forbidden"] = True
        if spatial_cache is not None:
            comparison_report["spatial_cache_metadata"] = report[
                "spatial_cache_metadata"
            ]
            comparison_report["spatial_cache_metadata_sha256"] = report[
                "spatial_cache_metadata_sha256"
            ]
        _write_json(comparison_path, comparison_report)
        if args.split == "validation":
            _write_json(
                output_dir / "selected_thresholds.json",
                {
                    "threshold_source": "validation",
                    "threshold_policy": "independent_validation_max_f1_per_mode",
                    "static_anchor_validation_threshold": mode_reports[
                        MISATO_TRAINED_STATIC_ANCHOR_MODE
                    ]["threshold_selection"],
                    "dynamic_off_validation_threshold": mode_reports[
                        DYNAMIC_TEMPORAL_OFF_MODE
                    ]["threshold_selection"],
                    "dynamic_on_validation_threshold": mode_reports[
                        DYNAMIC_TEMPORAL_ON_MODE
                    ]["threshold_selection"],
                    "test_reselection_forbidden": True,
                },
            )
        else:
            _write_json(
                output_dir / "frozen_thresholds_used.json",
                frozen_validation,
            )
        report["three_mode_comparison"] = {
            "mode_order": list(THREE_MODE_ORDER),
            "report": str(comparison_path),
            "report_sha256": sha256_file(comparison_path),
            "prediction_archive": str(prediction_path),
            "prediction_archive_sha256": comparison_report[
                "prediction_archive_sha256"
            ],
            "summary": comparison_summary,
            comparison_table_key: comparison_report[comparison_table_key],
            "scientific_comparisons": comparison_report[
                "scientific_comparisons"
            ],
        }
        if args.split == "validation":
            report["three_mode_comparison"]["supplementary_baselines"] = comparison_report[
                "supplementary_baselines"
            ]
    _write_json(output_dir / f"{args.split}_metrics.json", report)
    _write_json(output_dir / "distribution.json", distribution.to_dict())
    _write_json(
        output_dir
        / ("selected_threshold.json" if args.split == "validation" else "frozen_threshold.json"),
        threshold,
    )
    _write_json(
        output_dir / f"{args.split}_system_ids.json",
        list(evaluation_dataset.system_ids),
    )
    _write_json(
        output_dir / f"excluded_{args.split}_systems.json",
        [event.__dict__ for event in evaluation_dataset.skip_events],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
