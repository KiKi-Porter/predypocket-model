"""PreDyPocket-compatible backbone layout and sequence encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BACKBONE_ATOM_ORDER = ("N", "CA", "C", "O")
THREE_TO_INDEX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "CYM": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}


class BackboneLayoutError(ValueError):
    """Raised when residues cannot be mapped unambiguously to N/CA/C/O."""


@dataclass(frozen=True)
class BackboneLayout:
    atom_indices: np.ndarray
    sequence: np.ndarray
    valid_residue_mask: np.ndarray
    residue_names: tuple[str, ...]

    @property
    def residue_count(self) -> int:
        return int(len(self.sequence))

    @property
    def valid_atom_indices(self) -> np.ndarray:
        return self.atom_indices[self.valid_residue_mask].reshape(-1)


def build_backbone_layout(topology: Any) -> BackboneLayout:
    protein_atoms = [int(index) for index in topology.select("protein")]
    if not protein_atoms:
        raise BackboneLayoutError("Topology contains no protein atoms")
    residue_indices: list[int] = []
    for atom_index in protein_atoms:
        residue_index = int(topology.atom(atom_index).residue.index)
        if not residue_indices or residue_indices[-1] != residue_index:
            residue_indices.append(residue_index)
    if len(residue_indices) != len(set(residue_indices)):
        raise BackboneLayoutError("Protein residue order is not contiguous")

    atoms = np.full((len(residue_indices), 4), -1, dtype=np.int64)
    sequence = np.zeros(len(residue_indices), dtype=np.int32)
    valid = np.zeros(len(residue_indices), dtype=bool)
    names: list[str] = []
    for model_index, topology_index in enumerate(residue_indices):
        residue = topology.residue(topology_index)
        residue_name = str(residue.name).upper()
        names.append(residue_name)
        by_name = {
            name: [int(atom.index) for atom in residue.atoms if atom.name == name]
            for name in BACKBONE_ATOM_ORDER
        }
        has_unique_backbone = all(len(by_name[name]) == 1 for name in BACKBONE_ATOM_ORDER)
        known_sequence = residue_name in THREE_TO_INDEX
        if known_sequence:
            sequence[model_index] = THREE_TO_INDEX[residue_name]
        if has_unique_backbone and known_sequence:
            atoms[model_index] = [by_name[name][0] for name in BACKBONE_ATOM_ORDER]
            valid[model_index] = True
    if not np.any(valid):
        raise BackboneLayoutError("No residue has a complete PreDyPocket backbone")
    return BackboneLayout(
        atom_indices=atoms,
        sequence=sequence,
        valid_residue_mask=valid,
        residue_names=tuple(names),
    )


def place_valid_backbone_coordinates(
    valid_coordinates: np.ndarray, layout: BackboneLayout
) -> np.ndarray:
    """Place atom-sliced MDTraj coordinates into an all-residue tensor."""

    coordinates = np.asarray(valid_coordinates, dtype=np.float32)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise BackboneLayoutError(
            f"Atom coordinates must be [F,A,3], found {coordinates.shape}"
        )
    valid_count = int(np.sum(layout.valid_residue_mask))
    if coordinates.shape[1] != valid_count * 4:
        raise BackboneLayoutError(
            f"Expected {valid_count * 4} selected atoms, found {coordinates.shape[1]}"
        )
    output = np.zeros(
        (coordinates.shape[0], layout.residue_count, 4, 3), dtype=np.float32
    )
    output[:, layout.valid_residue_mask] = coordinates.reshape(
        coordinates.shape[0], valid_count, 4, 3
    )
    return output

