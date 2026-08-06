"""Deterministic protocol-v2 split construction and validation."""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import DynamicPreDyPocketDataset
from .folds import FoldDefinition, FoldError, validate_fold
from .serialization import write_json


PROTOCOL_V2_NAME = "protocol_v2_posthoc"
PROTOCOL_V2_SEED = 42
PROTOCOL_V2_SIZES = (6, 2, 2)


@dataclass(frozen=True)
class ProteinSupervisionStats:
    protein_id: str
    positive_count: int
    negative_count: int

    @property
    def sample_count(self) -> int:
        return self.positive_count + self.negative_count

    @property
    def positive_rate(self) -> float:
        return self.positive_count / self.sample_count if self.sample_count else 0.0

    @property
    def has_both_classes(self) -> bool:
        return self.positive_count > 0 and self.negative_count > 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sample_count"] = self.sample_count
        result["positive_rate"] = self.positive_rate
        result["has_both_classes"] = self.has_both_classes
        return result


def count_protein_supervision(
    manifest_path: str | Path,
    backbone_cache_root: str | Path,
    repo_root: str | Path,
) -> dict[str, ProteinSupervisionStats]:
    """Count effective labels without loading coordinates or making predictions."""

    root = Path(repo_root).resolve()
    dataset = DynamicPreDyPocketDataset(
        manifest_path=manifest_path,
        backbone_cache_root=backbone_cache_root,
        use_backbone_cache=True,
        repo_root=root,
    )
    counts: dict[str, list[int]] = {}
    cache_masks: dict[tuple[str, str, str], np.ndarray] = {}
    for record, supervision in zip(dataset.records, dataset.iter_supervision()):
        protein_id = str(record["protein_id"])
        cache_key = (
            protein_id,
            str(record["replica"]),
            str(record["segment_name"]),
        )
        if cache_key not in cache_masks:
            mask_path = (
                dataset.backbone_cache_root
                / cache_key[0]
                / cache_key[1]
                / cache_key[2]
                / "valid_residue_mask.npy"
            )
            if not mask_path.is_file():
                raise FoldError(f"Backbone validity mask is missing: {mask_path}")
            cache_masks[cache_key] = np.asarray(
                np.load(mask_path, allow_pickle=False), dtype=bool
            )
        labels = np.asarray(supervision["label"], dtype=np.float32)
        effective = (
            np.asarray(supervision["residue_mask"], dtype=bool)
            & np.asarray(supervision["training_mask"], dtype=bool)
            & cache_masks[cache_key]
        )
        if labels.shape != effective.shape:
            raise FoldError(f"Supervision shape mismatch for {protein_id}")
        selected = labels[effective]
        current = counts.setdefault(protein_id, [0, 0])
        current[0] += int(np.sum(selected >= 0.5))
        current[1] += int(np.sum(selected < 0.5))
    return {
        protein_id: ProteinSupervisionStats(protein_id, values[0], values[1])
        for protein_id, values in sorted(counts.items())
    }


def combine_stats(
    protein_ids: Sequence[str],
    stats: Mapping[str, ProteinSupervisionStats],
) -> ProteinSupervisionStats:
    selected = [stats[protein_id] for protein_id in protein_ids]
    return ProteinSupervisionStats(
        protein_id="+".join(sorted(protein_ids)),
        positive_count=sum(item.positive_count for item in selected),
        negative_count=sum(item.negative_count for item in selected),
    )


def select_inner_validation(
    non_test_proteins: Sequence[str],
    stats: Mapping[str, ProteinSupervisionStats],
    epsilon: float = 1e-12,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Choose validation proteins without accepting or consulting outer-test IDs."""

    proteins = tuple(sorted(str(value) for value in non_test_proteins))
    if len(proteins) != 8 or len(set(proteins)) != 8:
        raise FoldError("Protocol v2 selection requires eight unique non-test proteins")
    candidates: list[tuple[tuple[Any, ...], tuple[str, ...], tuple[str, ...], Any, Any]] = []
    for validation in itertools.combinations(proteins, 2):
        validation_set = set(validation)
        train = tuple(protein for protein in proteins if protein not in validation_set)
        validation_stats = combine_stats(validation, stats)
        train_stats = combine_stats(train, stats)
        if not validation_stats.has_both_classes or not train_stats.has_both_classes:
            continue
        rate_gap = abs(
            math.log(
                max(validation_stats.positive_rate, epsilon)
                / max(train_stats.positive_rate, epsilon)
            )
        )
        score = (
            rate_gap,
            -validation_stats.positive_count,
            tuple(validation),
        )
        candidates.append(
            (score, tuple(validation), train, validation_stats, train_stats)
        )
    if not candidates:
        raise FoldError("No valid two-protein validation combination has both classes")
    candidates.sort(key=lambda item: item[0])
    score, validation, train, validation_stats, train_stats = candidates[0]
    details = {
        "candidate_count": math.comb(8, 2),
        "eligible_candidate_count": len(candidates),
        "objective_abs_log_positive_rate_ratio": float(score[0]),
        "tie_break_validation_positive_count": validation_stats.positive_count,
        "tie_break_validation_protein_ids": list(validation),
        "train_positive_rate": train_stats.positive_rate,
        "validation_positive_rate": validation_stats.positive_rate,
        "outer_test_labels_used_for_selection": False,
        "epsilon": epsilon,
    }
    return train, validation, details


def build_protocol_v2_definitions(
    v1_definitions: Mapping[int, FoldDefinition],
    stats: Mapping[str, ProteinSupervisionStats],
    seed: int = PROTOCOL_V2_SEED,
) -> tuple[dict[int, FoldDefinition], dict[int, dict[str, Any]]]:
    if seed != PROTOCOL_V2_SEED:
        raise FoldError(f"Protocol v2 seed must remain {PROTOCOL_V2_SEED}")
    definitions: dict[int, FoldDefinition] = {}
    selection: dict[int, dict[str, Any]] = {}
    all_proteins = set(stats)
    for fold in range(5):
        outer = v1_definitions[fold]
        test = tuple(outer.test)
        non_test = tuple(sorted(all_proteins - set(test)))
        # Only non-test statistics are exposed to the selection function.
        eligible_stats = {protein_id: stats[protein_id] for protein_id in non_test}
        train, validation, details = select_inner_validation(
            non_test, eligible_stats
        )
        definition = FoldDefinition(
            fold=fold,
            train=train,
            validation=validation,
            test=test,
        )
        validate_fold(definition, expected_sizes=PROTOCOL_V2_SIZES)
        definitions[fold] = definition
        selection[fold] = details
    return definitions, selection


def fold_split_stats(
    definition: FoldDefinition,
    stats: Mapping[str, ProteinSupervisionStats],
) -> dict[str, dict[str, Any]]:
    return {
        split: combine_stats(definition.split_proteins(split), stats).to_dict()
        for split in ("train", "validation", "test")
    }


def build_split_payload(
    definitions: Mapping[int, FoldDefinition],
    selection: Mapping[int, Mapping[str, Any]],
    stats: Mapping[str, ProteinSupervisionStats],
) -> dict[str, Any]:
    folds = []
    for fold in range(5):
        definition = definitions[fold]
        folds.append(
            {
                "fold": fold,
                "train": list(definition.train),
                "validation": list(definition.validation),
                "test": list(definition.test),
                "selection": dict(selection[fold]),
                "supervision": fold_split_stats(definition, stats),
            }
        )
    return {
        "schema_version": 2,
        "protocol": PROTOCOL_V2_NAME,
        "algorithm": (
            "enumerate-nontest-validation-pairs-min-abs-log-rate-"
            "then-max-positive-then-lexical-v1"
        ),
        "seed": PROTOCOL_V2_SEED,
        "split_unit": "protein_id",
        "replicas_kept_together": True,
        "segments_and_anchors_kept_together": True,
        "outer_test_source": "atlas_10protein_5fold_splits_seed42.json",
        "outer_test_already_inspected": True,
        "outer_test_labels_used_for_validation_selection": False,
        "fold_sizes": {"train": 6, "validation": 2, "test": 2},
        "folds": folds,
    }


def manifest_replicas(manifest_path: str | Path) -> dict[str, tuple[str, ...]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        found: dict[str, set[str]] = {}
        for row in rows:
            found.setdefault(str(row["protein_id"]), set()).add(str(row["replica"]))
    return {
        protein_id: tuple(sorted(replicas))
        for protein_id, replicas in sorted(found.items())
    }


def membership_rows(
    definitions: Mapping[int, FoldDefinition],
    replicas: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        definition = definitions[fold]
        for split in ("train", "validation", "test"):
            for protein_id in definition.split_proteins(split):
                rows.append(
                    {
                        "fold": fold,
                        "protein_id": protein_id,
                        "split": split,
                        "replicas": ",".join(replicas[protein_id]),
                        "split_unit": "protein_id",
                        "seed": PROTOCOL_V2_SEED,
                        "protocol": PROTOCOL_V2_NAME,
                    }
                )
    return rows


def write_membership(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "fold",
        "protein_id",
        "split",
        "replicas",
        "split_unit",
        "seed",
        "protocol",
    )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_protocol_v2(
    definitions: Mapping[int, FoldDefinition],
    v1_definitions: Mapping[int, FoldDefinition],
    stats: Mapping[str, ProteinSupervisionStats],
) -> None:
    if set(definitions) != set(range(5)):
        raise FoldError("Protocol v2 must define folds 0 through 4")
    for fold, definition in definitions.items():
        validate_fold(definition, expected_sizes=PROTOCOL_V2_SIZES)
        if tuple(definition.test) != tuple(v1_definitions[fold].test):
            raise FoldError(f"Fold {fold} outer test differs from protocol v1")
        train_stats = combine_stats(definition.train, stats)
        validation_stats = combine_stats(definition.validation, stats)
        if not train_stats.has_both_classes:
            raise FoldError(f"Fold {fold} train split does not contain both classes")
        if not validation_stats.has_both_classes:
            raise FoldError(f"Fold {fold} validation does not contain both classes")


def render_split_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Dynamic PreDyPocket Protocol v2 Split Report",
        "",
        "Protocol: `protocol_v2_posthoc`  ",
        "Seed: `42`  ",
        "Split size: `6 train / 2 validation / 2 outer test`  ",
        "Outer test labels were not used to select inner validation proteins.",
        "Outer test counts below are descriptive only and were added after selection.",
        "",
    ]
    for fold in payload["folds"]:
        lines.extend(
            [
                f"## Fold {fold['fold']}",
                "",
                f"- Train proteins: `{', '.join(fold['train'])}`",
                f"- Validation proteins: `{', '.join(fold['validation'])}`",
                f"- Test proteins: `{', '.join(fold['test'])}`",
                f"- Eligible validation pairs: `{fold['selection']['eligible_candidate_count']}/28`",
                f"- Selection score: `{fold['selection']['objective_abs_log_positive_rate_ratio']:.12g}`",
                "",
                "| Split | Positive | Negative | Positive rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for split in ("train", "validation", "test"):
            item = fold["supervision"][split]
            lines.append(
                f"| {split} | {item['positive_count']} | {item['negative_count']} | "
                f"{item['positive_rate']:.10f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            "- Every fold contains exactly 6/2/2 proteins.",
            "- Every validation and train split contains positive and negative labels.",
            "- Every outer-test pair is identical to protocol v1.",
            "- Assignment is protein-level; replicas, segments, and anchors cannot cross splits.",
            "- Each protein belongs to exactly one split within each fold.",
            "",
        ]
    )
    return "\n".join(lines)


def write_split_artifacts(
    json_path: str | Path,
    membership_path: str | Path,
    report_path: str | Path,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    write_json(json_path, payload, indent=2, sort_keys=False)
    write_membership(membership_path, rows)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_split_report(payload), encoding="utf-8")
