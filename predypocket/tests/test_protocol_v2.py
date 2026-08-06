from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np
import pytest

from predypocket.config import load_config
from predypocket.folds import load_folds
from predypocket.metrics import (
    compute_binary_metrics,
    evaluate_grouped_metrics,
    select_validation_threshold,
)
from predypocket.model import synthetic_model_inputs
from predypocket.protocol_v2 import (
    build_protocol_v2_definitions,
    select_inner_validation,
    validate_protocol_v2,
)
from predypocket.protocol_v2_runtime import (
    TEST_WARNING,
    assert_protocol_output_path,
    training_directory,
    validate_evaluation_request,
)
from predypocket.serialization import json_dumps
from predypocket.trainer import (
    update_model_selection,
    validate_validation_for_model_selection,
)
from scripts.predypocket.run_protocol_v2_5fold import (
    assign_folds,
    commands as protocol_v2_commands,
    fold_commands,
    parse_gpu_ids,
)
import scripts.predypocket.run_protocol_v2_5fold as protocol_v2_runner


REPO_ROOT = Path(__file__).resolve().parents[2]
V2_SPLITS = REPO_ROOT / "data/atlas/atlas_10protein_protocol_v2_5fold_splits_seed42.json"


class _SupervisionDataset:
    def __init__(self, labels):
        values = np.asarray(labels, dtype=np.float32)
        self._samples = (
            {
                "label": values,
                "residue_mask": np.ones_like(values, dtype=bool),
                "training_mask": np.ones_like(values, dtype=bool),
            },
        )

    def iter_supervision(self):
        return self._samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative_display = item.relative_to(REPO_ROOT)
        digest.update(f"{_sha256(item)}  {relative_display}\n".encode("utf-8"))
    return digest.hexdigest()


def test_71_all_negative_pr_auc_is_undefined():
    result = compute_binary_metrics([0, 0], [0.1, 0.9], 0.5)
    assert result["pr_auc"] is None


def test_72_all_negative_average_precision_is_undefined():
    result = compute_binary_metrics([0, 0], [0.1, 0.9], 0.5)
    assert result["average_precision"] is None


def test_73_all_negative_roc_auc_is_undefined():
    result = compute_binary_metrics([0, 0], [0.1, 0.9], 0.5)
    assert result["roc_auc"] is None


def test_74_all_positive_roc_auc_is_undefined():
    result = compute_binary_metrics([1, 1], [0.1, 0.9], 0.5)
    assert result["roc_auc"] is None
    assert result["average_precision"] is None
    assert not result["has_both_classes"]


def test_75_two_class_metrics_remain_defined():
    result = compute_binary_metrics([0, 1], [0.1, 0.9], 0.5)
    assert result["has_both_classes"]
    assert all(result[name] is not None for name in ("pr_auc", "average_precision", "roc_auc"))


def test_76_undefined_json_is_null_and_strict():
    result = compute_binary_metrics([0, 0], [0.1, 0.9], 0.5)
    encoded = json_dumps(result)
    assert "NaN" not in encoded
    assert json.loads(encoded)["pr_auc"] is None


def test_77_all_negative_false_positive_rate_is_correct():
    result = compute_binary_metrics([0, 0], [0.1, 0.9], 0.5)
    assert result["false_positive_count"] == 1
    assert result["false_positive_rate"] == 0.5
    assert result["specificity"] == 0.5


def test_78_all_negative_predicted_positive_rate_is_correct():
    result = compute_binary_metrics([0, 0], [0.1, 0.9], 0.5)
    assert result["predicted_positive_count"] == 1
    assert result["predicted_positive_rate"] == 0.5


def _single_class_grouped_report():
    return evaluate_grouped_metrics(
        [0, 0, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        ["all_negative", "all_negative", "mixed", "mixed"],
        0.5,
        from_logits=False,
    )


def test_79_macro_pr_auc_ignores_undefined():
    report = _single_class_grouped_report()
    assert report["macro_protein"]["pr_auc"] == report["per_protein"]["mixed"]["pr_auc"]


def test_80_macro_reports_defined_protein_count():
    macro = _single_class_grouped_report()["macro_protein"]
    assert macro["total_protein_count"] == 2
    assert macro["defined_pr_auc_protein_count"] == 1
    assert macro["undefined_pr_auc_protein_count"] == 1


def test_81_macro_reports_undefined_protein_ids():
    macro = _single_class_grouped_report()["macro_protein"]
    assert macro["undefined_pr_auc_protein_ids"] == ["all_negative"]


def test_82_macro_does_not_insert_half_for_all_negative():
    report = _single_class_grouped_report()
    assert report["per_protein"]["all_negative"]["pr_auc"] is None
    assert report["macro_protein"]["pr_auc"] != 0.5


def test_83_single_class_validation_is_rejected_before_training():
    with pytest.raises(ValueError, match="does not contain both"):
        validate_validation_for_model_selection(_SupervisionDataset([0, 0]))


def test_84_undefined_primary_metric_does_not_update_best():
    decision = update_model_selection(None, best_score=0.2, stale_epochs=3)
    assert not decision.primary_metric_defined
    assert not decision.best_checkpoint_updated
    assert decision.best_score == 0.2
    assert decision.stale_epochs == 3


def test_85_single_class_validation_has_no_max_f1_threshold():
    selected = select_validation_threshold([0, 0], [0.1, 0.9], "validation")
    assert selected["threshold"] is None
    assert selected["validation_f1"] is None
    assert selected["threshold_selection_status"] == "undefined_single_class_validation"


def test_86_two_class_model_selection_updates_normally():
    first = update_model_selection(0.2, best_score=None, stale_epochs=0)
    stale = update_model_selection(0.1, best_score=first.best_score, stale_epochs=0)
    assert first.best_checkpoint_updated and first.best_score == 0.2
    assert not stale.best_checkpoint_updated and stale.stale_epochs == 1


def test_87_every_protocol_v2_fold_is_6_2_2(protocol_v2_split_artifacts):
    assert all(
        (len(fold.train), len(fold.validation), len(fold.test)) == (6, 2, 2)
        for fold in protocol_v2_split_artifacts["v2"].values()
    )


def test_88_outer_test_matches_protocol_v1(protocol_v2_split_artifacts):
    data = protocol_v2_split_artifacts
    assert all(data["v2"][fold].test == data["v1"][fold].test for fold in range(5))


def test_89_every_validation_has_both_classes(protocol_v2_split_artifacts):
    data = protocol_v2_split_artifacts
    for fold in data["v2"].values():
        selected = [data["stats"][protein] for protein in fold.validation]
        assert sum(item.positive_count for item in selected) > 0
        assert sum(item.negative_count for item in selected) > 0


def test_90_every_train_split_has_both_classes(protocol_v2_split_artifacts):
    data = protocol_v2_split_artifacts
    for fold in data["v2"].values():
        selected = [data["stats"][protein] for protein in fold.train]
        assert sum(item.positive_count for item in selected) > 0
        assert sum(item.negative_count for item in selected) > 0


def test_91_protein_does_not_cross_split(protocol_v2_split_artifacts):
    for fold in protocol_v2_split_artifacts["v2"].values():
        assigned = (*fold.train, *fold.validation, *fold.test)
        assert len(assigned) == len(set(assigned)) == 10


def test_92_replicas_do_not_cross_split():
    path = REPO_ROOT / "data/atlas/atlas_10protein_protocol_v2_5fold_membership_seed42.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 50
    assert len({(row["fold"], row["protein_id"]) for row in rows}) == 50
    assert all(row["replicas"] == "R1,R2,R3" for row in rows)
    membership = {
        (int(row["fold"]), row["protein_id"]): row["split"] for row in rows
    }
    manifest = REPO_ROOT / "data/atlas/atlas_dynamic_manifest_10proteins_1ns_gap1ns_future20ns.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    for fold in range(5):
        entity_splits = {}
        for row in manifest_rows:
            split = membership[(fold, row["protein_id"])]
            entity = (row["protein_id"], row["replica"], row["segment_name"])
            entity_splits.setdefault(entity, set()).add(split)
        assert all(len(splits) == 1 for splits in entity_splits.values())


def test_93_protocol_v2_selection_is_deterministic(protocol_v2_split_artifacts):
    data = protocol_v2_split_artifacts
    first, first_details = build_protocol_v2_definitions(data["v1"], data["stats"], 42)
    second, second_details = build_protocol_v2_definitions(data["v1"], data["stats"], 42)
    assert first == second
    assert first_details == second_details


def test_94_selection_does_not_read_test_labels(protocol_v2_split_artifacts):
    data = protocol_v2_split_artifacts
    outer = data["v1"][0]
    non_test = sorted(set(data["stats"]) - set(outer.test))

    class GuardedStats(dict):
        def __getitem__(self, key):
            if key in outer.test:
                raise AssertionError("outer-test statistics were accessed")
            return super().__getitem__(key)

    guarded = GuardedStats(data["stats"])
    train, validation, details = select_inner_validation(non_test, guarded)
    assert len(train) == 6 and len(validation) == 2
    assert details["outer_test_labels_used_for_selection"] is False


def test_95_all_five_protocol_v2_folds_validate(protocol_v2_split_artifacts):
    data = protocol_v2_split_artifacts
    validate_protocol_v2(data["v2"], data["v1"], data["stats"])


def test_96_head_matched_uses_only_anchor_frame(protocol_v2_models):
    coordinates, sequence, mask, offsets = synthetic_model_inputs()
    changed = coordinates.copy()
    changed[:, :-1] += 10.0
    first = protocol_v2_models["anchor"](coordinates, sequence, mask, offsets, training=False)
    second = protocol_v2_models["anchor"](changed, sequence, mask, offsets, training=False)
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_97_head_matched_gvp_initialization_matches_dynamic(protocol_v2_models):
    dynamic = protocol_v2_models["dynamic"].static_model
    anchor = protocol_v2_models["anchor_core"].static_model
    for left, right in zip(dynamic.variables, anchor.variables):
        np.testing.assert_array_equal(left.numpy(), right.numpy())


def test_98_head_matched_classifier_initialization_matches_dynamic(protocol_v2_models):
    dynamic = protocol_v2_models["dynamic"].static_model.dense.variables
    anchor = protocol_v2_models["anchor_core"].static_model.dense.variables
    assert len(dynamic) == len(anchor) > 0
    for left, right in zip(dynamic, anchor):
        np.testing.assert_array_equal(left.numpy(), right.numpy())


def test_99_head_matched_and_dynamic_freeze_gvp_consistently(protocol_v2_models):
    dynamic = protocol_v2_models["dynamic"]
    anchor_core = protocol_v2_models["anchor_core"]
    assert dynamic._spatial_frozen and anchor_core._spatial_frozen
    assert dynamic.static_model.dense.trainable
    assert anchor_core.static_model.dense.trainable


def test_100_non_anchor_frame_changes_do_not_affect_anchor_output(protocol_v2_models):
    coordinates, sequence, mask, offsets = synthetic_model_inputs()
    baseline = protocol_v2_models["anchor"]
    expected = baseline.dynamic_model.anchor_only_logits(coordinates, sequence, mask)
    actual = baseline(coordinates, sequence, mask, offsets, training=False)
    np.testing.assert_array_equal(expected.numpy(), actual.numpy())


def test_101_temporal_off_contribution_is_exactly_zero(protocol_v2_models):
    coordinates, sequence, mask, offsets = synthetic_model_inputs()
    _, auxiliary = protocol_v2_models["dynamic"](
        coordinates, sequence, mask, offsets, training=False,
        return_auxiliary=True, temporal_mode="off"
    )
    np.testing.assert_array_equal(
        auxiliary["dynamic_contribution"].numpy(),
        np.zeros_like(auxiliary["dynamic_contribution"].numpy()),
    )
    np.testing.assert_array_equal(auxiliary["z_fused"].numpy(), auxiliary["z_anchor"].numpy())


def test_102_temporal_off_uses_current_classifier_head(protocol_v2_models):
    model = protocol_v2_models["dynamic"]
    coordinates, sequence, mask, offsets = synthetic_model_inputs()
    final_bias = model.static_model.dense.layers[-1].bias
    before_bias = final_bias.numpy().copy()
    before = model(coordinates, sequence, mask, offsets, training=False, temporal_mode="off")
    try:
        final_bias.assign_add(np.ones_like(before_bias))
        after = model(coordinates, sequence, mask, offsets, training=False, temporal_mode="off")
        assert not np.array_equal(before.numpy(), after.numpy())
    finally:
        final_bias.assign(before_bias)
    np.testing.assert_array_equal(final_bias.numpy(), before_bias)


def test_103_temporal_off_equals_current_head_on_anchor(protocol_v2_models):
    model = protocol_v2_models["dynamic"]
    coordinates, sequence, mask, offsets = synthetic_model_inputs()
    off = model(coordinates, sequence, mask, offsets, training=False, temporal_mode="off")
    anchor = model.static_anchor_logits(coordinates, sequence, mask)
    np.testing.assert_array_equal(off.numpy(), anchor.numpy())


def test_104_temporal_on_executes_temporal_path(protocol_v2_models):
    coordinates, sequence, mask, offsets = synthetic_model_inputs()
    _, auxiliary = protocol_v2_models["dynamic"](
        coordinates, sequence, mask, offsets, training=False,
        return_auxiliary=True, temporal_mode="on"
    )
    assert bool(auxiliary["temporal_enabled"].numpy())
    expected = auxiliary["z_anchor"] + auxiliary["dynamic_contribution"]
    np.testing.assert_array_equal(auxiliary["z_fused"].numpy(), expected.numpy())


def test_105_temporal_modes_do_not_modify_checkpoint_files(protocol_v2_models):
    index = REPO_ROOT / "models/predypocket_initializer.index"
    data = REPO_ROOT / "models/predypocket_initializer.data-00000-of-00001"
    before = (_sha256(index), _sha256(data))
    coordinates, sequence, mask, offsets = synthetic_model_inputs()
    model = protocol_v2_models["dynamic"]
    model(coordinates, sequence, mask, offsets, training=False, temporal_mode="on")
    model(coordinates, sequence, mask, offsets, training=False, temporal_mode="off")
    assert before == (_sha256(index), _sha256(data))


def test_106_protocol_v1_results_are_bitwise_unchanged():
    expected = {
        0: "68e3759f7617b0e611ec5735896ad5c3f9d77448143bc316900203df0b970901",
        1: "df080f848cbb8fb074817198c88a0cdc4336b1dd8733f5aad55035caad8bc351",
        2: "73257571d5884c40958f43f6edd879d9caa2cf8ba909e1b908afd5c8dfeb70a9",
        3: "c73a4574d88a4a9440c2d345231305d99841cf915d0875668ce629777fc3f715",
        4: "34e54589f95ef91b611f83057ac713ab4b14ba05a17ace20165a8a86dfa3a88b",
    }
    root = REPO_ROOT / "outputs/predypocket_model/training"
    assert {fold: _tree_sha256(root / f"fold{fold}") for fold in range(5)} == expected


def test_107_protocol_v2_output_cannot_target_v1():
    config = load_config(REPO_ROOT / "configs/predypocket_protocol_v2.json")
    output = training_directory(config, 0, "dynamic")
    assert output.is_relative_to(REPO_ROOT / "outputs/predypocket_protocol_v2")
    assert "predypocket_model" not in str(output)
    with pytest.raises(ValueError, match="cannot target"):
        assert_protocol_output_path(
            REPO_ROOT / "outputs/predypocket_model/training/fold0"
        )


def test_108_no_epoch_was_started_by_protocol_tests():
    root = REPO_ROOT / "outputs/predypocket_protocol_v2/training"
    assert not root.exists() or not list(root.rglob("history.json"))


def test_109_optimizer_step_count_remains_zero(backward_artifacts):
    assert backward_artifacts["result"].optimizer_step_count == 0
    assert backward_artifacts["optimizer_iterations"] == 0


def test_110_test_evaluation_is_not_automatic():
    assert validate_evaluation_request("validation", False) is None
    with pytest.raises(PermissionError, match="explicit"):
        validate_evaluation_request("test", False)
    assert validate_evaluation_request("test", True) == TEST_WARNING


def test_111_protocol_v2_cannot_enter_stage2():
    config = load_config(REPO_ROOT / "configs/predypocket_protocol_v2.json")
    assert config.training["stage"] == 1
    assert config.training["stage2_enabled"] is False
    source = (REPO_ROOT / "scripts/predypocket/train_protocol_v2.py").read_text()
    assert "choices=(1,)" in source


def test_112_protocol_v2_fold_pipeline_preserves_step_order():
    planned = fold_commands("config.json", "/GPU:0", 3)
    assert len(planned) == 5
    assert [Path(command[1]).name for command in planned] == [
        "train_protocol_v2.py",
        "evaluate_protocol_v2.py",
        "train_protocol_v2.py",
        "evaluate_protocol_v2.py",
        "evaluate_protocol_v2.py",
    ]
    assert [
        command[command.index("--model-variant") + 1] for command in planned
    ] == ["anchor-matched", "anchor-matched", "dynamic", "dynamic", "dynamic"]
    evaluation_modes = [
        command[command.index("--temporal-mode") + 1]
        for command in planned
        if "--temporal-mode" in command
    ]
    assert evaluation_modes == ["off", "on", "off"]


def test_113_protocol_v2_complete_schedule_contains_all_five_folds():
    planned = protocol_v2_commands("config.json", "/GPU:0")
    assert len(planned) == 25
    folds = [int(command[command.index("--fold") + 1]) for command in planned]
    assert {fold: folds.count(fold) for fold in range(5)} == {
        fold: 5 for fold in range(5)
    }


def test_114_four_gpu_assignment_never_overlaps_folds():
    assignments = assign_folds(("2", "3", "4", "5"))
    assert assignments == (
        ("2", (0, 4)),
        ("3", (1,)),
        ("4", (2,)),
        ("5", (3,)),
    )
    assigned = [fold for _, folds in assignments for fold in folds]
    assert sorted(assigned) == list(range(5))


def test_115_gpu_ids_are_unique_physical_indices():
    assert parse_gpu_ids("2, 3,4,5") == ("2", "3", "4", "5")
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        parse_gpu_ids("2,2,3,4")
    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        parse_gpu_ids("2,GPU-3,4,5")


def test_116_all_folds_wrapper_defaults_to_four_gpus():
    source = (
        REPO_ROOT / "scripts/predypocket/run_protocol_v2_all_folds.sh"
    ).read_text()
    assert 'GPU_IDS="${GPU_IDS:-2,3,4,5}"' in source
    assert '--gpu-ids "$GPU_IDS"' in source


def test_117_parallel_workers_isolate_visible_gpu():
    original_fold_commands = protocol_v2_runner.fold_commands

    def fake_commands(config, device, fold):
        del config, device, fold
        return [
            [
                sys.executable,
                "-c",
                "import os; print('visible=' + os.environ['CUDA_VISIBLE_DEVICES'])",
                "--model-variant",
                "dynamic",
            ]
        ]

    try:
        protocol_v2_runner.fold_commands = fake_commands
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop_event = threading.Event()
            console_lock = threading.Lock()
            assignments = assign_folds(("2", "3", "4", "5"))
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        protocol_v2_runner._run_worker,
                        gpu_id,
                        folds[:1],
                        "config.json",
                        "/GPU:0",
                        root,
                        stop_event,
                        console_lock,
                    )
                    for gpu_id, folds in assignments
                ]
                for future in futures:
                    future.result()
            for gpu_id, folds in assignments:
                log = (root / f"fold{folds[0]}" / "pipeline.log").read_text()
                assert f"visible={gpu_id}" in log
            assert not stop_event.is_set()
    finally:
        protocol_v2_runner.fold_commands = original_fold_commands
