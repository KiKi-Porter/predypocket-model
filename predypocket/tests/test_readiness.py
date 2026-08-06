from __future__ import annotations

import tempfile
from pathlib import Path

from predypocket.readiness import (
    parse_report_statuses,
    readiness_from_paths,
)


def _readiness_fixture(
    root: Path,
    status: str = "Status: **passed**",
    manifest: str = "sample_id,status\nsample_1,ready\n",
    write_manifest: bool = True,
    use_membership_only: bool = False,
):
    paths = {
        "manifest": root / "manifest.csv",
        "folds": root / "folds.json",
        "fold_membership": root / "membership.csv",
        "formal_dataset_report": root / "FINAL_DATASET_REPORT.md",
    }
    if write_manifest:
        paths["manifest"].write_text(manifest, encoding="utf-8")
    if use_membership_only:
        paths["fold_membership"].write_text("fold,protein_id\n0,p1\n", encoding="utf-8")
    else:
        paths["folds"].write_text("{}\n", encoding="utf-8")
    paths["formal_dataset_report"].write_text(status + "\n", encoding="utf-8")
    return readiness_from_paths(paths)


def test_61_status_bold_passed_is_recognized():
    assert parse_report_statuses("Status: **passed**") == ("passed",)


def test_62_status_plain_passed_is_recognized():
    assert parse_report_statuses("Status: passed") == ("passed",)


def test_63_status_bold_complete_is_recognized():
    assert parse_report_statuses("Status: **complete**") == ("complete",)


def test_64_status_failed_is_explicit_failure():
    with tempfile.TemporaryDirectory() as directory:
        result = _readiness_fixture(Path(directory), status="Status: failed")
    assert result["explicit_failed"] is True
    assert result["ready"] is False


def test_65_missing_manifest_is_not_ready():
    with tempfile.TemporaryDirectory() as directory:
        result = _readiness_fixture(Path(directory), write_manifest=False)
    assert result["manifest_row_count"] == 0
    assert result["ready"] is False


def test_66_empty_manifest_is_not_ready():
    with tempfile.TemporaryDirectory() as directory:
        result = _readiness_fixture(
            Path(directory), manifest="sample_id,status\n"
        )
    assert result["manifest_row_count"] == 0
    assert result["ready"] is False


def test_67_fold_membership_alone_satisfies_fold_presence():
    with tempfile.TemporaryDirectory() as directory:
        result = _readiness_fixture(Path(directory), use_membership_only=True)
    assert result["fold_definition_exists"] is True
    assert result["ready"] is True


def test_68_status_matching_ignores_case_backticks_and_whitespace():
    assert parse_report_statuses("  Status : `  COMPLETE  `  ") == ("complete",)


def test_69_status_plain_complete_is_recognized():
    assert parse_report_statuses("Status: complete") == ("complete",)
