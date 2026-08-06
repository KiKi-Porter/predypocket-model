"""Protein-level ATLAS five-fold split validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class FoldError(ValueError):
    """Raised for missing proteins, overlap, or malformed fold definitions."""


@dataclass(frozen=True)
class FoldDefinition:
    fold: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def all_proteins(self) -> frozenset[str]:
        return frozenset((*self.train, *self.validation, *self.test))

    def split_proteins(self, split: str) -> tuple[str, ...]:
        if split not in {"train", "validation", "test"}:
            raise FoldError(f"Unknown split {split!r}")
        return getattr(self, split)


def validate_fold(
    definition: FoldDefinition,
    expected_sizes: tuple[int, int, int] = (7, 1, 2),
) -> None:
    sets = {
        "train": set(definition.train),
        "validation": set(definition.validation),
        "test": set(definition.test),
    }
    actual_sizes = (
        len(definition.train),
        len(definition.validation),
        len(definition.test),
    )
    if actual_sizes != expected_sizes:
        raise FoldError(
            f"Fold {definition.fold} must contain {expected_sizes[0]} train, "
            f"{expected_sizes[1]} validation, {expected_sizes[2]} test proteins"
        )
    if sets["train"] & sets["validation"]:
        raise FoldError(f"Fold {definition.fold} leaks train into validation")
    if sets["train"] & sets["test"]:
        raise FoldError(f"Fold {definition.fold} leaks train into test")
    if sets["validation"] & sets["test"]:
        raise FoldError(f"Fold {definition.fold} leaks validation into test")
    if len(definition.all_proteins) != 10:
        raise FoldError(f"Fold {definition.fold} does not contain 10 unique proteins")


def load_folds(path: str | Path) -> dict[int, FoldDefinition]:
    source = Path(path)
    if not source.is_file():
        raise FoldError(f"Fold JSON is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FoldError(f"Cannot read fold JSON {source}: {exc}") from exc
    if payload.get("split_unit") != "protein_id":
        raise FoldError("Fold split unit must be protein_id")
    if payload.get("replicas_kept_together") is not True:
        raise FoldError("Fold file does not guarantee replica grouping")
    if payload.get("segments_and_anchors_kept_together") is not True:
        raise FoldError("Fold file does not guarantee segment/anchor grouping")
    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != 5:
        raise FoldError("Exactly five folds are required")
    raw_sizes = payload.get("fold_sizes", {"train": 7, "validation": 1, "test": 2})
    if not isinstance(raw_sizes, Mapping):
        raise FoldError("fold_sizes must be an object")
    try:
        expected_sizes = tuple(
            int(raw_sizes[name]) for name in ("train", "validation", "test")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FoldError("fold_sizes must define integer train/validation/test counts") from exc
    if expected_sizes not in {(7, 1, 2), (6, 2, 2)}:
        raise FoldError(f"Unsupported fold sizes {expected_sizes}")

    definitions: dict[int, FoldDefinition] = {}
    for raw in raw_folds:
        definition = FoldDefinition(
            fold=int(raw["fold"]),
            train=tuple(str(value) for value in raw["train"]),
            validation=tuple(str(value) for value in raw["validation"]),
            test=tuple(str(value) for value in raw["test"]),
        )
        validate_fold(definition, expected_sizes=expected_sizes)
        if definition.fold in definitions:
            raise FoldError(f"Duplicate fold number {definition.fold}")
        definitions[definition.fold] = definition
    if set(definitions) != set(range(5)):
        raise FoldError("Fold numbers must be 0 through 4")
    return definitions


def validate_dataset_split(dataset: Any, definition: FoldDefinition, split: str) -> None:
    expected = set(definition.split_proteins(split))
    observed = set(dataset.protein_ids)
    if not observed <= expected:
        raise FoldError(
            f"Dataset split {split} contains proteins outside its fold: {sorted(observed - expected)}"
        )


def filter_records_for_split(
    records: Sequence[Mapping[str, Any]], definition: FoldDefinition, split: str
) -> list[Mapping[str, Any]]:
    proteins = set(definition.split_proteins(split))
    return [record for record in records if str(record["protein_id"]) in proteins]
