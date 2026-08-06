"""Read-only readiness checks shared by dry-run CLIs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

from .config import DynamicConfig


_STATUS_LINE = re.compile(
    r"^\s*(?:[-+]\s+)?status\s*[:=]\s*(.*?)\s*$", re.IGNORECASE
)
_MARKDOWN_DECORATION = re.compile(r"[`*_~]+")
_STATUS_TOKEN = re.compile(r"^[a-z]+", re.IGNORECASE)
_COMPLETE_STATUSES = frozenset({"passed", "complete"})


def parse_report_statuses(text: str) -> tuple[str, ...]:
    """Extract normalized explicit Status fields from a Markdown report."""

    statuses: list[str] = []
    for line in text.splitlines():
        match = _STATUS_LINE.match(line)
        if match is None:
            continue
        undecorated = _MARKDOWN_DECORATION.sub("", match.group(1)).strip()
        token = _STATUS_TOKEN.match(undecorated)
        if token is not None:
            statuses.append(token.group(0).casefold())
    return tuple(statuses)


def manifest_data_row_count(path: Path) -> int:
    """Count non-empty CSV data rows, excluding the header."""

    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            if next(reader, None) is None:
                return 0
            return sum(1 for row in reader if any(cell.strip() for cell in row))
    except (OSError, csv.Error):
        return 0


def readiness_from_paths(paths: Mapping[str, Path]) -> dict[str, Any]:
    required = {"manifest", "folds", "fold_membership", "formal_dataset_report"}
    missing_keys = required - set(paths)
    if missing_keys:
        raise ValueError(f"Readiness paths are missing keys: {sorted(missing_keys)}")
    resolved = {name: Path(path) for name, path in paths.items()}
    exists = {name: path.is_file() for name, path in resolved.items()}
    statuses: tuple[str, ...] = ()
    if exists["formal_dataset_report"]:
        text = resolved["formal_dataset_report"].read_text(
            encoding="utf-8", errors="replace"
        )
        statuses = parse_report_statuses(text)
    completion_declared = any(status in _COMPLETE_STATUSES for status in statuses)
    explicit_failed = "failed" in statuses
    manifest_rows = manifest_data_row_count(resolved["manifest"])
    fold_definition_exists = exists["folds"] or exists["fold_membership"]
    ready = (
        exists["manifest"]
        and fold_definition_exists
        and exists["formal_dataset_report"]
        and completion_declared
        and manifest_rows > 0
        and not explicit_failed
    )
    return {
        "ready": ready,
        "exists": exists,
        "completion_declared": completion_declared,
        "explicit_failed": explicit_failed,
        "report_statuses": list(statuses),
        "manifest_row_count": manifest_rows,
        "fold_definition_exists": fold_definition_exists,
        "paths": {name: str(path) for name, path in resolved.items()},
    }


def formal_dataset_readiness(config: DynamicConfig) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "manifest": config.repo_path(config.data["manifest"]),
        "folds": config.repo_path(config.data["folds"]),
        "fold_membership": config.repo_path(config.data["fold_membership"]),
        "formal_dataset_report": config.repo_path(config.data["formal_dataset_report"]),
    }
    return readiness_from_paths(paths)
