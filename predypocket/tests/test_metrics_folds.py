from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from predypocket.folds import load_folds
from predypocket.metrics import (
    MetricError,
    compute_binary_metrics,
    evaluate_grouped_metrics,
    select_validation_threshold,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_39_all_required_metrics_are_computed():
    metrics = compute_binary_metrics(
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.6, 0.9]), 0.5
    )
    required = {
        "pr_auc",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "matthews_correlation_coefficient",
    }
    assert required <= set(metrics)


def test_40_grouped_metrics_include_pooled_per_protein_and_macro():
    report = evaluate_grouped_metrics(
        [0, 1, 0, 1], [-2.0, 2.0, -1.0, 1.0], ["a", "a", "b", "b"], 0.5
    )
    assert set(report) == {"pooled", "per_protein", "macro_protein"}
    assert set(report["per_protein"]) == {"a", "b"}


def test_41_threshold_is_selected_on_validation():
    selected = select_validation_threshold(
        [0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6], split="validation"
    )
    assert 0.0 <= selected["threshold"] <= 1.0


def test_42_test_threshold_selection_is_refused():
    with pytest.raises(MetricError, match="validation"):
        select_validation_threshold([0, 1], [0.1, 0.9], split="test")


def test_43_five_fold_file_has_no_protein_leakage():
    folds = load_folds(
        REPO_ROOT / "data/atlas/atlas_10protein_5fold_splits_seed42.json"
    )
    assert len(folds) == 5
    for definition in folds.values():
        assert not set(definition.train) & set(definition.validation)
        assert not set(definition.train) & set(definition.test)
        assert not set(definition.validation) & set(definition.test)


def test_44_each_fold_has_7_1_2_proteins():
    folds = load_folds(
        REPO_ROOT / "data/atlas/atlas_10protein_5fold_splits_seed42.json"
    )
    assert all(
        (len(fold.train), len(fold.validation), len(fold.test)) == (7, 1, 2)
        for fold in folds.values()
    )

