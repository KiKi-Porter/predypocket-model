from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_45_synthetic_forward_succeeded(backward_artifacts):
    assert backward_artifacts["result"].forward_success


def test_46_synthetic_loss_succeeded(backward_artifacts):
    assert backward_artifacts["result"].loss_success


def test_47_single_backward_succeeded(backward_artifacts):
    assert backward_artifacts["result"].backward_success


def test_48_gradients_exist_and_are_finite(backward_artifacts):
    result = backward_artifacts["result"]
    assert result.gradient_count > 0
    assert result.gradients_finite
    assert result.finite_gradient_count == result.gradient_count


def test_49_gradient_clipping_is_callable(backward_artifacts):
    result = backward_artifacts["result"]
    assert result.gradient_clip_callable
    assert result.gradient_global_norm_after_clip <= 1.000001


def test_50_optimizer_step_count_is_zero(backward_artifacts):
    assert backward_artifacts["result"].optimizer_step_count == 0
    assert backward_artifacts["optimizer_iterations"] == 0


def test_51_scheduler_step_count_is_zero(backward_artifacts):
    assert backward_artifacts["result"].scheduler_step_count == 0


def test_52_all_parameters_are_bitwise_unchanged(backward_artifacts):
    before = backward_artifacts["before"]
    after = backward_artifacts["after"]
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[name], after[name]) for name in before)
    assert backward_artifacts["result"].parameter_update_count == 0


def test_53_no_formal_training_artifacts_exist():
    root = REPO_ROOT / "outputs/predypocket_protocol_v2/training"
    prohibited = (
        "best_checkpoint.index",
        "last_checkpoint.index",
        "training.pid",
        "training.log",
    )
    assert not any(path.name in prohibited for path in root.rglob("*"))


def test_54_no_epoch_or_background_process_was_started(backward_artifacts):
    result = backward_artifacts["result"]
    assert result.optimizer_step_count == 0
    root = REPO_ROOT / "outputs/predypocket_protocol_v2/training"
    assert not root.exists() or not list(root.rglob("history.json"))


def test_55_protected_atlas_python_files_keep_initial_hashes():
    expected = {
        "__init__.py": "af217aa809cb9c68d373f713b754d325b341e21127b822fa9a14861095afb559",
        "atlas_config.py": "706d7873ae89bad883e5b094f49facbf320608436adfab52874a69338de6c516",
        "atlas_detector.py": "8bf9a8f8c59664227c87dd02b7dd415389d71777b962949b4cbc03b95ed6757d",
        "atlas_topology.py": "c830b1f1f0c6957a6bb2640b82fb1dd362e8ff36679aa0b9b6ff6b0c7253a94b",
        "assign_replica_segments.py": "89b919c311a9a60c1b9ca05ae5bec24e9a82297223991b45daf33e31d9ba80c8",
        "build_dynamic_labels.py": "1cb5576e43207c138c7a0089a0c95bda03cc9bf432b9c87ee6bd176be204f17a",
        "build_dynamic_manifest.py": "8c25f4fb17fad62bc0282d7966046dafd503ef748598f085e394861d0e4b6abb",
        "build_protein_folds.py": "d30c7546c624575cd7865c346c008e0cf12c2522fbe997839b2cd03b7a62de45",
        "discover_atlas_data.py": "1d35449abcc203de243ee1c728419f8c84e5ca86a10da6fc2b58e44f41bbd1b1",
        "generate_segment_pocket_cache.py": "67062107025a5624ec54b8bb2d07be610db9e7f250cdba5fc62fc9d9e3a9fb34",
        "run_atlas_label_pipeline.py": "100ff714faf40ac7ea50af4f68aa156dbf50a1c75f04bf70f162d173b4befb4e",
        "test_atlas_preparation.py": "c2112109d39092afdd751144bdefe38c19aee09b092c04187fe21728c9dc6b37",
        "validate_atlas_dataset.py": "6b05faaba0a56261ce4dcaf575f45f18c1b601a137f55f2ed9d72e68dfd8992c",
    }
    root = REPO_ROOT / "scripts/atlas"
    assert {name: _sha256(root / name) for name in expected} == expected


def test_56_protected_label_config_keeps_initial_hash():
    path = REPO_ROOT / "configs/atlas_10proteins_1ns_gap1ns_future20ns.json"
    assert _sha256(path) == "b0c71dedd4d4cb686595de7bb73b091687865329d1dae2b30d03bd5da8f43ab9"


def test_57_original_checkpoint_is_unchanged():
    assert _sha256(REPO_ROOT / "models/predypocket_initializer.index") == (
        "09e36a62a987b14bc49cf7a1fa53fa734f85d85d088777f98c6aca253b47bf46"
    )
    assert _sha256(REPO_ROOT / "models/predypocket_initializer.data-00000-of-00001") == (
        "6f5ab62b9fb38b54040053ac321f3362d616bae9b31569249a032dbb50890b70"
    )


def test_58_cache_script_has_no_forbidden_detector_api():
    source = (
        REPO_ROOT / "scripts/predypocket/prepare_backbone_input_cache.py"
    ).read_text(encoding="utf-8")
    assert "enspara.geometry.pockets" not in source
    assert "get_pocket_cells" not in source
    assert "LIGSITE" not in source


def test_59_cache_writer_targets_only_new_backbone_root():
    source = (
        REPO_ROOT / "scripts/predypocket/prepare_backbone_input_cache.py"
    ).read_text(encoding="utf-8")
    assert 'Path(task["backbone_cache_root"])' in source
    assert "np.save(source" not in source


def test_60_no_pseudo_manifest_or_label_was_created_by_tests():
    test_root = REPO_ROOT / "predypocket/tests"
    assert not list(test_root.rglob("*.csv"))
    assert not list(test_root.rglob("*.npz"))
