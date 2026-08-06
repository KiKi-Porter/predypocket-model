"""Pooled, per-protein and macro protein binary classification metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


class MetricError(ValueError):
    pass


SINGLE_CLASS_REASON = "metric requires both positive and negative labels"
NO_POSITIVE_REASON = "metric requires at least one positive label"
NO_NEGATIVE_REASON = "metric requires at least one negative label"
NO_THRESHOLD_REASON = "threshold-dependent metric requires a defined threshold"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    output[~nonnegative] = exponential / (1.0 + exponential)
    return output


def _validated_arrays(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.shape != scores.shape or y_true.size == 0:
        raise MetricError("Metric labels/scores must be non-empty and equal length")
    if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isfinite(scores)):
        raise MetricError("Metrics require finite probabilities and binary labels")
    return y_true, scores


def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def _set_undefined(
    result: dict[str, Any], reasons: dict[str, str], name: str, reason: str
) -> None:
    result[name] = None
    reasons[name] = reason


def compute_binary_metrics(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float | None,
) -> dict[str, Any]:
    """Compute metrics without manufacturing scores for single-class data."""

    y_true, scores = _validated_arrays(labels, probabilities)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    sample_count = int(y_true.size)
    has_positive = positive_count > 0
    has_negative = negative_count > 0
    has_both_classes = has_positive and has_negative
    class_status = (
        "both_classes"
        if has_both_classes
        else "all_positive"
        if has_positive
        else "all_negative"
    )
    result: dict[str, Any] = {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / sample_count,
        "random_ap_baseline": positive_count / sample_count,
        "has_positive": has_positive,
        "has_negative": has_negative,
        "has_both_classes": has_both_classes,
        "class_status": class_status,
        "mean_probability": float(np.mean(scores)),
        "max_probability": float(np.max(scores)),
    }
    undefined: dict[str, str] = {}

    if has_both_classes:
        result["pr_auc"] = _pr_auc(y_true, scores)
        result["average_precision"] = float(average_precision_score(y_true, scores))
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        _set_undefined(result, undefined, "pr_auc", SINGLE_CLASS_REASON)
        _set_undefined(result, undefined, "roc_auc", SINGLE_CLASS_REASON)
        _set_undefined(
            result, undefined, "average_precision", SINGLE_CLASS_REASON
        )

    threshold_value: float | None
    if threshold is None:
        threshold_value = None
    else:
        threshold_value = float(threshold)
        if not np.isfinite(threshold_value):
            raise MetricError("Metric threshold must be finite or None")
    result["threshold"] = threshold_value

    threshold_fields = (
        "true_positive_count",
        "true_negative_count",
        "false_positive_count",
        "false_negative_count",
        "predicted_positive_count",
        "predicted_negative_count",
        "predicted_positive_rate",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "specificity",
        "balanced_accuracy",
        "matthews_correlation_coefficient",
    )
    if threshold_value is None:
        for name in threshold_fields:
            _set_undefined(result, undefined, name, NO_THRESHOLD_REASON)
    else:
        predictions = (scores >= threshold_value).astype(np.int32)
        true_positive = int(np.sum((y_true == 1) & (predictions == 1)))
        true_negative = int(np.sum((y_true == 0) & (predictions == 0)))
        false_positive = int(np.sum((y_true == 0) & (predictions == 1)))
        false_negative = int(np.sum((y_true == 1) & (predictions == 0)))
        predicted_positive = true_positive + false_positive
        predicted_negative = true_negative + false_negative
        result.update(
            {
                "true_positive_count": true_positive,
                "true_negative_count": true_negative,
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
                "predicted_positive_count": predicted_positive,
                "predicted_negative_count": predicted_negative,
                "predicted_positive_rate": predicted_positive / sample_count,
            }
        )
        if predicted_positive > 0:
            result["precision"] = true_positive / predicted_positive
        else:
            _set_undefined(
                result,
                undefined,
                "precision",
                "precision requires at least one predicted positive",
            )
        if has_positive:
            recall = true_positive / positive_count
            result["recall"] = recall
            result["f1"] = (
                0.0
                if predicted_positive == 0
                else 2.0 * result["precision"] * recall
                / (result["precision"] + recall)
                if result["precision"] + recall > 0
                else 0.0
            )
        else:
            _set_undefined(result, undefined, "recall", NO_POSITIVE_REASON)
            _set_undefined(result, undefined, "f1", NO_POSITIVE_REASON)
        if has_negative:
            result["false_positive_rate"] = false_positive / negative_count
            result["specificity"] = true_negative / negative_count
        else:
            _set_undefined(
                result, undefined, "false_positive_rate", NO_NEGATIVE_REASON
            )
            _set_undefined(result, undefined, "specificity", NO_NEGATIVE_REASON)
        if has_both_classes:
            result["balanced_accuracy"] = float(
                balanced_accuracy_score(y_true, predictions)
            )
            result["matthews_correlation_coefficient"] = float(
                matthews_corrcoef(y_true, predictions)
            )
        else:
            _set_undefined(
                result, undefined, "balanced_accuracy", SINGLE_CLASS_REASON
            )
            _set_undefined(
                result,
                undefined,
                "matthews_correlation_coefficient",
                SINGLE_CLASS_REASON,
            )

    result["ap"] = result["average_precision"]
    result["mcc"] = result.get("matthews_correlation_coefficient")
    result["undefined_metrics"] = sorted(undefined)
    result["undefined_reason"] = undefined
    return result


def select_validation_threshold(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    split: str,
) -> dict[str, Any]:
    if split != "validation":
        raise MetricError(
            f"Threshold selection requires validation data, received split={split!r}"
        )
    y_true, scores = _validated_arrays(labels, probabilities)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    has_both_classes = positive_count > 0 and negative_count > 0
    base = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "has_both_classes": has_both_classes,
    }
    if not has_both_classes:
        return {
            **base,
            "threshold": None,
            "validation_f1": None,
            "threshold_selection_status": "undefined_single_class_validation",
            "undefined_reason": (
                "Validation split does not contain both positive and negative labels; "
                "max-F1 threshold selection is undefined."
            ),
        }
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return {
            **base,
            "threshold": None,
            "validation_f1": None,
            "threshold_selection_status": "undefined_no_threshold_candidates",
            "undefined_reason": "Validation scores produced no threshold candidates.",
        }
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return {
        **base,
        "threshold": float(thresholds[best_index]),
        "validation_f1": float(f1_values[best_index]),
        "threshold_selection_status": "selected_validation_max_f1",
        "undefined_reason": None,
    }


def _defined_metric_values(
    per_protein: dict[str, dict[str, Any]], name: str
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    undefined_ids: list[str] = []
    for protein_id, metrics in per_protein.items():
        value = metrics.get(name)
        if value is None or not np.isfinite(float(value)):
            undefined_ids.append(protein_id)
        else:
            values.append(float(value))
    return values, undefined_ids


def evaluate_grouped_metrics(
    labels: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    protein_ids: Sequence[str] | np.ndarray,
    threshold: float | None,
    from_logits: bool = True,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    raw_values = np.asarray(values, dtype=np.float64).reshape(-1)
    proteins = np.asarray(protein_ids, dtype=str).reshape(-1)
    if not (y_true.shape == raw_values.shape == proteins.shape):
        raise MetricError("Grouped metric arrays must have equal shape")
    probabilities = sigmoid(raw_values) if from_logits else raw_values
    pooled = compute_binary_metrics(y_true, probabilities, threshold)
    per_protein: dict[str, dict[str, Any]] = {}
    for protein in sorted(np.unique(proteins)):
        selected = proteins == protein
        per_protein[protein] = compute_binary_metrics(
            y_true[selected], probabilities[selected], threshold
        )

    metric_names = (
        "pr_auc",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "matthews_correlation_coefficient",
        "false_positive_rate",
        "specificity",
        "predicted_positive_rate",
        "mean_probability",
        "max_probability",
        "random_ap_baseline",
    )
    total = len(per_protein)
    macro: dict[str, Any] = {
        "protein_count": total,
        "total_protein_count": total,
    }
    for name in metric_names:
        defined, undefined_ids = _defined_metric_values(per_protein, name)
        macro[name] = float(np.mean(defined)) if defined else None
        macro[f"defined_{name}_protein_count"] = len(defined)
        macro[f"undefined_{name}_protein_count"] = len(undefined_ids)
        macro[f"undefined_{name}_protein_ids"] = undefined_ids

    all_negative_ids = [
        protein_id
        for protein_id, metrics in per_protein.items()
        if metrics["positive_count"] == 0
    ]
    all_negative_fpr = [
        float(per_protein[protein_id]["false_positive_rate"])
        for protein_id in all_negative_ids
        if per_protein[protein_id]["false_positive_rate"] is not None
    ]
    all_negative_predicted_rate = [
        float(per_protein[protein_id]["predicted_positive_rate"])
        for protein_id in all_negative_ids
        if per_protein[protein_id]["predicted_positive_rate"] is not None
    ]
    macro.update(
        {
            "all_negative_protein_count": len(all_negative_ids),
            "all_negative_protein_ids": all_negative_ids,
            "macro_all_negative_false_positive_rate": (
                float(np.mean(all_negative_fpr)) if all_negative_fpr else None
            ),
            "macro_all_negative_predicted_positive_rate": (
                float(np.mean(all_negative_predicted_rate))
                if all_negative_predicted_rate
                else None
            ),
        }
    )
    macro["undefined_metrics"] = [
        name for name in metric_names if macro[name] is None
    ]
    macro["undefined_reason"] = {
        name: "no protein has a defined value for this metric"
        for name in macro["undefined_metrics"]
    }
    macro["ap"] = macro["average_precision"]
    macro["mcc"] = macro["matthews_correlation_coefficient"]
    return {
        "pooled": pooled,
        "per_protein": per_protein,
        "macro_protein": macro,
    }


def evaluate_test_with_frozen_validation_threshold(
    labels: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    protein_ids: Sequence[str] | np.ndarray,
    threshold: float,
    threshold_source_split: str,
    from_logits: bool = True,
) -> dict[str, Any]:
    """Evaluate test values only with a threshold frozen on validation."""

    if threshold_source_split != "validation":
        raise MetricError(
            "Test evaluation threshold must have been selected on validation"
        )
    if threshold is None or not np.isfinite(float(threshold)):
        raise MetricError("A finite frozen validation threshold is required")
    report = evaluate_grouped_metrics(
        labels,
        values,
        protein_ids,
        float(threshold),
        from_logits=from_logits,
    )
    report["threshold_source_split"] = "validation"
    report["threshold_frozen_for_test"] = True
    return report
