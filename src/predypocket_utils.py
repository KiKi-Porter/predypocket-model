"""PreDyPocket data preparation utilities for MD trajectory training.

The functions in this module are intentionally dependency-light.  MD trajectory
loading is isolated behind mdtraj imports so label and alignment tests can run in
plain Python environments.
"""

from __future__ import print_function

import json
import hashlib
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "CYM": "C", "CYX": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
    "HIS": "H", "HID": "H", "HIE": "H", "HIP": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}

AA1_TO_ID = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6,
    "G": 7, "H": 8, "I": 9, "L": 10, "K": 11, "M": 12,
    "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18,
    "V": 19,
}

SOLVENT_RESNAMES = {"HOH", "WAT", "SOL", "H2O", "DOD"}
COMMON_NON_LIGAND_RESNAMES = SOLVENT_RESNAMES | {
    "NA", "K", "CL", "CA", "MG", "MN", "ZN", "FE", "CU", "NI", "CO",
    "CD", "HG", "IOD", "BR", "F", "SO4", "PO4", "NO3", "ACT", "ACE",
    "EDO", "GOL", "DMS", "DMSO", "PEG", "PE4", "MPD", "TRS", "MES",
    "HEP", "BME", "FMT", "ACY", "IPA", "EOH", "MOH",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")
SUPPORTED_TRAJECTORY_EXTENSIONS = (
    ".nc", ".xtc", ".dcd", ".trr", ".h5", ".pdb", ".pdb.gz",
)


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    resseq: str
    icode: str
    resname: str

    def compact(self):
        icode = self.icode if self.icode else "-"
        return "%s:%s:%s:%s" % (self.chain or "-", self.resseq, icode, self.resname)

    def pdb_identity(self):
        return (self.chain or "", str(self.resseq), self.icode or "")


@dataclass(frozen=True)
class LigandSpec:
    resname: str
    chain: str
    resseq: str
    icode: str = ""


@dataclass(frozen=True)
class AtomRecord:
    record: str
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: str
    icode: str
    x: float
    y: float
    z: float
    element: str

    @property
    def residue_key(self):
        return ResidueKey(self.chain, self.resseq, self.icode, self.resname)

    @property
    def is_hydrogen(self):
        element = (self.element or "").upper()
        return element == "H" or self.name.upper().startswith("H")

    @property
    def is_protein(self):
        return self.resname.upper() in AA3_TO_1 and self.record == "ATOM"


def parse_residue_key(text):
    parts = str(text).split(":")
    if len(parts) != 4:
        raise ValueError("Invalid residue key string: %r" % text)
    chain, resseq, icode, resname = parts
    return ResidueKey("" if chain == "-" else chain, resseq, "" if icode == "-" else icode, resname)


def residue_keys_to_strings(keys):
    return np.asarray([k.compact() for k in keys], dtype=object)


def residue_keys_from_strings(values):
    return [parse_residue_key(v) for v in values]


def infer_element(atom_name):
    cleaned = re.sub(r"[^A-Za-z]", "", atom_name or "")
    if not cleaned:
        return ""
    if cleaned[0].upper() == "H":
        return "H"
    return cleaned[0].upper()


def parse_pdb_atoms(pdb_path):
    atoms = []
    with open(pdb_path, "r") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A", "1"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            element = line[76:78].strip() if len(line) >= 78 else ""
            atom = AtomRecord(
                record=record,
                name=line[12:16].strip(),
                altloc=altloc,
                resname=line[17:20].strip().upper(),
                chain=line[21].strip(),
                resseq=line[22:26].strip(),
                icode=line[26].strip(),
                x=x,
                y=y,
                z=z,
                element=(element or infer_element(line[12:16])).upper(),
            )
            atoms.append(atom)
    return atoms


def ordered_protein_residue_keys(atoms, chain_id=None):
    keys = []
    seen = set()
    for atom in atoms:
        if not atom.is_protein:
            continue
        if chain_id is not None and atom.chain != chain_id:
            continue
        identity = atom.residue_key.pdb_identity()
        if identity not in seen:
            keys.append(atom.residue_key)
            seen.add(identity)
    return keys


def sequence_from_residue_keys(keys):
    chars = []
    for key in keys:
        aa = AA3_TO_1.get(key.resname.upper())
        if aa:
            chars.append(aa)
    return "".join(chars)


def sequence_ids_from_residue_keys(keys):
    ids = []
    for key in keys:
        aa = AA3_TO_1.get(key.resname.upper(), "A")
        ids.append(AA1_TO_ID.get(aa, 0))
    return np.asarray(ids, dtype=np.int32)


def _residue_type_code(resname):
    return AA3_TO_1.get(str(resname).upper(), str(resname).upper())


def _same_residue_type(left, right):
    return _residue_type_code(left) == _residue_type_code(right)


def parse_ligand_spec_from_system_dir(system_dir):
    """Parse dynamic_data directory names such as 4KVK_4KVK_1.PG4_A_703."""
    base = Path(system_dir).name
    parts = base.split("_")
    if len(parts) < 4:
        return None
    ligand_token = parts[-3]
    ligand = ligand_token.split(".")[-1].upper()
    chain = parts[-2]
    resseq = parts[-1]
    match = re.match(r"(-?\d+)([A-Za-z]?)$", resseq)
    if match:
        resseq, icode = match.group(1), match.group(2)
    else:
        icode = ""
    if not ligand or not chain or not resseq:
        return None
    return LigandSpec(ligand, chain, resseq, icode)


def _select_ligand_atoms(atoms, ligand_spec=None, include_common_additives=False):
    selected = []
    resname_only = []
    for atom in atoms:
        if atom.is_hydrogen:
            continue
        if ligand_spec is not None:
            if atom.resname != ligand_spec.resname:
                continue
            resname_only.append(atom)
            if ligand_spec.chain and atom.chain != ligand_spec.chain:
                continue
            if ligand_spec.resseq and atom.resseq != str(ligand_spec.resseq):
                continue
            if ligand_spec.icode and atom.icode != ligand_spec.icode:
                continue
            selected.append(atom)
            continue
        if atom.record != "HETATM":
            continue
        if atom.resname in AA3_TO_1:
            continue
        if atom.resname in SOLVENT_RESNAMES:
            continue
        if (not include_common_additives) and atom.resname in COMMON_NON_LIGAND_RESNAMES:
            continue
        selected.append(atom)
    if ligand_spec is not None and not selected:
        return resname_only
    return selected


def _ordered_protein_atom_groups(atoms):
    keys = []
    groups = []
    by_identity = {}
    for atom in atoms:
        if atom.is_hydrogen or not atom.is_protein:
            continue
        identity = atom.residue_key.pdb_identity()
        group_index = by_identity.get(identity)
        if group_index is None:
            group_index = len(groups)
            by_identity[identity] = group_index
            keys.append(atom.residue_key)
            groups.append([])
        groups[group_index].append(atom)
    return keys, groups


def _map_target_residues_to_atom_groups(target_residue_keys, source_keys, source_groups):
    exact = {}
    loose = {}
    for index, key in enumerate(source_keys):
        exact.setdefault(key.pdb_identity(), index)
        loose.setdefault((str(key.resseq), key.icode or "", _residue_type_code(key.resname)), index)

    mapped = []
    stats = {"exact": 0, "ignore_chain": 0, "order": 0, "unmapped": 0}
    for index, key in enumerate(target_residue_keys):
        source_index = exact.get(key.pdb_identity())
        if source_index is not None and _same_residue_type(source_keys[source_index].resname, key.resname):
            mapped.append(source_groups[source_index])
            stats["exact"] += 1
            continue

        source_index = loose.get((str(key.resseq), key.icode or "", _residue_type_code(key.resname)))
        if source_index is not None:
            mapped.append(source_groups[source_index])
            stats["ignore_chain"] += 1
            continue

        if index < len(source_keys) and _same_residue_type(source_keys[index].resname, key.resname):
            mapped.append(source_groups[index])
            stats["order"] += 1
            continue

        mapped.append([])
        stats["unmapped"] += 1
    return mapped, stats


def _protein_atom_map(atoms):
    by_identity = {}
    for key, group in zip(*_ordered_protein_atom_groups(atoms)):
        by_identity[key.pdb_identity()] = group
    return by_identity


def _minimum_distance(atom_group, ligand_xyz):
    if not atom_group or ligand_xyz.size == 0:
        return None
    coords = np.asarray([[a.x, a.y, a.z] for a in atom_group], dtype=np.float64)
    delta = coords[:, None, :] - ligand_xyz[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    return float(np.min(distances))


def _minimum_distances_for_groups(atom_groups, ligand_xyz):
    if ligand_xyz.size == 0:
        return [None for _ in atom_groups]
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(ligand_xyz)
        distances = []
        for atom_group in atom_groups:
            if not atom_group:
                distances.append(None)
                continue
            coords = np.asarray([[a.x, a.y, a.z] for a in atom_group], dtype=np.float64)
            nearest, _ = tree.query(coords, k=1)
            distances.append(float(np.min(nearest)))
        return distances
    except Exception:
        return [_minimum_distance(atom_group, ligand_xyz) for atom_group in atom_groups]


def local_contact_labels(
    pdb_path,
    target_residue_keys,
    ligand_spec=None,
    contact_cutoff=4.5,
    buffer_cutoff=6.0,
    negative_policy="all_non_positive",
    include_common_additives=False,
):
    """Create residue labels from contacts to ligand atoms in a complex PDB.

    Labels are 1 for residues within `contact_cutoff` Angstrom of a ligand, 0
    for confident non-contacts, and -1 for ignored/uncertain residues.
    """
    atoms = parse_pdb_atoms(pdb_path)
    ligand_atoms = _select_ligand_atoms(
        atoms, ligand_spec=ligand_spec, include_common_additives=include_common_additives
    )
    default = 0 if negative_policy == "all_non_positive" else -1
    labels = np.zeros(len(target_residue_keys), dtype=np.int32) + default
    distances = np.zeros(len(target_residue_keys), dtype=np.float32) + np.nan
    if not ligand_atoms:
        return labels * 0 - 1, {
            "positive_count": 0,
            "ignored_count": len(labels),
            "ligand_atom_count": 0,
            "warning": "no ligand atoms selected",
        }

    source_keys, source_groups = _ordered_protein_atom_groups(atoms)
    mapped_groups, mapping_stats = _map_target_residues_to_atom_groups(
        target_residue_keys, source_keys, source_groups
    )
    ligand_xyz = np.asarray([[a.x, a.y, a.z] for a in ligand_atoms], dtype=np.float64)
    for i, dist in enumerate(_minimum_distances_for_groups(mapped_groups, ligand_xyz)):
        if dist is None:
            labels[i] = -1
            continue
        distances[i] = dist
        if dist <= contact_cutoff:
            labels[i] = 1
        elif buffer_cutoff and dist <= buffer_cutoff:
            labels[i] = -1
        elif negative_policy == "all_non_positive":
            labels[i] = 0

    return labels, {
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "ignored_count": int(np.sum(labels < 0)),
        "ligand_atom_count": len(ligand_atoms),
        "mapping": mapping_stats,
        "min_distance": float(np.nanmin(distances)) if np.isfinite(distances).any() else None,
    }


def _read_prmtop_flags(prmtop_path):
    flags = {}
    current_name = None
    current_format = None
    current_lines = []
    with open(prmtop_path, "r") as handle:
        for line in handle:
            if line.startswith("%FLAG"):
                if current_name is not None:
                    flags[current_name] = (current_format, current_lines)
                current_name = line.split(None, 1)[1].strip()
                current_format = None
                current_lines = []
            elif line.startswith("%FORMAT"):
                current_format = line.strip()
            elif current_name is not None:
                current_lines.append(line.rstrip("\n"))
    if current_name is not None:
        flags[current_name] = (current_format, current_lines)
    return flags


def _parse_prmtop_char_flag(flags, name):
    fmt, lines = flags[name]
    match = re.search(r"(\d+)a(\d+)", fmt or "", re.IGNORECASE)
    width = int(match.group(2)) if match else 4
    values = []
    for line in lines:
        for start in range(0, len(line), width):
            item = line[start:start + width].strip()
            if item:
                values.append(item)
    return values


def _parse_prmtop_int_flag(flags, name):
    _, lines = flags[name]
    values = []
    for line in lines:
        values.extend(int(item) for item in line.split())
    return values


def parse_amber_prmtop(prmtop_path):
    flags = _read_prmtop_flags(prmtop_path)
    pointers = _parse_prmtop_int_flag(flags, "POINTERS")
    if len(pointers) < 12:
        raise ValueError("Amber prmtop POINTERS block is too short: %s" % prmtop_path)
    natom = int(pointers[0])
    nres = int(pointers[11])
    atom_names = _parse_prmtop_char_flag(flags, "ATOM_NAME")[:natom]
    residue_labels = _parse_prmtop_char_flag(flags, "RESIDUE_LABEL")[:nres]
    residue_pointers = _parse_prmtop_int_flag(flags, "RESIDUE_POINTER")[:nres]
    if len(atom_names) != natom or len(residue_labels) != nres or len(residue_pointers) != nres:
        raise ValueError("Incomplete Amber prmtop blocks in %s" % prmtop_path)
    return {
        "natom": natom,
        "nres": nres,
        "atom_names": atom_names,
        "residue_labels": residue_labels,
        "residue_pointers": residue_pointers,
    }


def parse_amber_restart_coordinates(inpcrd_path, natom=None):
    with open(inpcrd_path, "r") as handle:
        lines = handle.readlines()
    if len(lines) < 2:
        raise ValueError("Amber restart file is too short: %s" % inpcrd_path)
    header_tokens = lines[1].split()
    if not header_tokens:
        raise ValueError("Amber restart atom-count line is empty: %s" % inpcrd_path)
    file_natom = int(float(header_tokens[0]))
    if natom is None:
        natom = file_natom
    elif int(natom) != file_natom:
        raise ValueError("Amber topology/restart atom-count mismatch: %s vs %s" % (natom, file_natom))
    values = []
    for line in lines[2:]:
        values.extend(_parse_amber_float_line(line))
        if len(values) >= 3 * int(natom):
            break
    if len(values) < 3 * int(natom):
        raise ValueError("Amber restart has fewer than 3*N coordinates: %s" % inpcrd_path)
    return np.asarray(values[:3 * int(natom)], dtype=np.float64).reshape((int(natom), 3))


def _parse_amber_float_line(line):
    tokens = line.split()
    if tokens:
        try:
            return [float(item) for item in tokens]
        except ValueError:
            pass
    values = []
    stripped = line.rstrip("\n")
    for start in range(0, len(stripped), 12):
        item = stripped[start:start + 12].strip()
        if item:
            values.append(float(item))
    return values


def amber_atoms_from_prmtop_inpcrd(prmtop_path, inpcrd_path):
    topology = parse_amber_prmtop(prmtop_path)
    coords = parse_amber_restart_coordinates(inpcrd_path, topology["natom"])
    atoms = []
    residue_pointers = topology["residue_pointers"]
    atom_names = topology["atom_names"]
    residue_labels = topology["residue_labels"]
    for residue_index, resname in enumerate(residue_labels):
        start = residue_pointers[residue_index] - 1
        if residue_index + 1 < len(residue_pointers):
            end = residue_pointers[residue_index + 1] - 1
        else:
            end = topology["natom"]
        record = "ATOM" if resname.upper() in AA3_TO_1 else "HETATM"
        for atom_index in range(start, end):
            x, y, z = coords[atom_index]
            atoms.append(AtomRecord(
                record=record,
                name=atom_names[atom_index].strip(),
                altloc="",
                resname=resname.strip().upper(),
                chain="",
                resseq=str(residue_index + 1),
                icode="",
                x=float(x),
                y=float(y),
                z=float(z),
                element=infer_element(atom_names[atom_index]),
            ))
    return atoms


def local_contact_labels_from_amber(
    prmtop_path,
    inpcrd_path,
    target_residue_keys,
    ligand_spec=None,
    contact_cutoff=4.5,
    buffer_cutoff=6.0,
    negative_policy="all_non_positive",
    include_common_additives=False,
):
    atoms = amber_atoms_from_prmtop_inpcrd(prmtop_path, inpcrd_path)
    labels, stats = local_contact_labels_from_atoms(
        atoms,
        target_residue_keys,
        ligand_spec=ligand_spec,
        contact_cutoff=contact_cutoff,
        buffer_cutoff=buffer_cutoff,
        negative_policy=negative_policy,
        include_common_additives=include_common_additives,
    )
    stats["source"] = "amber_prmtop_inpcrd"
    return labels, stats


def local_contact_labels_from_atoms(
    atoms,
    target_residue_keys,
    ligand_spec=None,
    contact_cutoff=4.5,
    buffer_cutoff=6.0,
    negative_policy="all_non_positive",
    include_common_additives=False,
):
    ligand_atoms = _select_ligand_atoms(
        atoms, ligand_spec=ligand_spec, include_common_additives=include_common_additives
    )
    default = 0 if negative_policy == "all_non_positive" else -1
    labels = np.zeros(len(target_residue_keys), dtype=np.int32) + default
    distances = np.zeros(len(target_residue_keys), dtype=np.float32) + np.nan
    if not ligand_atoms:
        return labels * 0 - 1, {
            "positive_count": 0,
            "ignored_count": len(labels),
            "ligand_atom_count": 0,
            "warning": "no ligand atoms selected",
        }

    source_keys, source_groups = _ordered_protein_atom_groups(atoms)
    mapped_groups, mapping_stats = _map_target_residues_to_atom_groups(
        target_residue_keys, source_keys, source_groups
    )
    ligand_xyz = np.asarray([[a.x, a.y, a.z] for a in ligand_atoms], dtype=np.float64)
    for i, dist in enumerate(_minimum_distances_for_groups(mapped_groups, ligand_xyz)):
        if dist is None:
            labels[i] = -1
            continue
        distances[i] = dist
        if dist <= contact_cutoff:
            labels[i] = 1
        elif buffer_cutoff and dist <= buffer_cutoff:
            labels[i] = -1
        elif negative_policy == "all_non_positive":
            labels[i] = 0

    return labels, {
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "ignored_count": int(np.sum(labels < 0)),
        "ligand_atom_count": len(ligand_atoms),
        "mapping": mapping_stats,
        "min_distance": float(np.nanmin(distances)) if np.isfinite(distances).any() else None,
    }


def combine_label_vectors(label_vectors):
    if not label_vectors:
        raise ValueError("No label vectors supplied")
    arr = np.vstack([np.asarray(v, dtype=np.int32) for v in label_vectors])
    combined = np.zeros(arr.shape[1], dtype=np.int32) - 1
    combined[np.any(arr == 0, axis=0)] = 0
    combined[np.any(arr == 1, axis=0)] = 1
    return combined


def chain_sequences_from_atoms(atoms):
    by_chain = {}
    for key in ordered_protein_residue_keys(atoms):
        by_chain.setdefault(key.chain, []).append(key)
    return {chain: (keys, sequence_from_residue_keys(keys)) for chain, keys in by_chain.items()}


def needleman_wunsch(query, target, match=2, mismatch=-1, gap=-2):
    n, m = len(query), len(target)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap
        trace[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap
        trace[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match if query[i - 1] == target[j - 1] else mismatch)
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            trace[i, j] = 0 if best == diag else (1 if best == up else 2)

    aq, at = [], []
    i, j = n, m
    while i > 0 or j > 0:
        direction = trace[i, j]
        if i > 0 and j > 0 and direction == 0:
            aq.append(query[i - 1])
            at.append(target[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or direction == 1):
            aq.append(query[i - 1])
            at.append("-")
            i -= 1
        else:
            aq.append("-")
            at.append(target[j - 1])
            j -= 1
    return "".join(reversed(aq)), "".join(reversed(at))


def alignment_mapping(query, target):
    aligned_query, aligned_target = needleman_wunsch(query, target)
    q_i = -1
    t_i = -1
    mapping = {}
    aligned_pairs = 0
    matches = 0
    for q_char, t_char in zip(aligned_query, aligned_target):
        if q_char != "-":
            q_i += 1
        if t_char != "-":
            t_i += 1
        if q_char != "-" and t_char != "-":
            mapping[q_i] = t_i
            aligned_pairs += 1
            if q_char == t_char:
                matches += 1
    coverage = float(aligned_pairs) / float(len(query)) if query else 0.0
    identity = float(matches) / float(aligned_pairs) if aligned_pairs else 0.0
    return mapping, {"identity": identity, "coverage": coverage}


def rcsb_sequence_search(sequence, identity_cutoff=0.7, max_hits=25, evalue_cutoff=1.0, timeout=60, cache_dir=None, retries=3):
    cache_path = None
    if cache_dir:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
        cache_name = "seq_%s_id%.3f_e%.1g_n%d.json" % (digest, identity_cutoff, evalue_cutoff, max_hits)
        cache_path = cache / cache_name
        if cache_path.exists() and cache_path.stat().st_size > 0:
            with open(cache_path, "r") as handle:
                return json.load(handle).get("identifiers", [])

    payload = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": evalue_cutoff,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": max_hits},
            "results_content_type": ["experimental"],
        },
    }
    request = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= max(1, int(retries)):
                raise
            time.sleep(2.0 * float(attempt + 1))
    else:  # pragma: no cover - defensive, loop should break or raise
        raise last_error
    identifiers = [item["identifier"] for item in data.get("result_set", [])]
    if cache_path:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp.%d" % os.getpid())
        with open(tmp_path, "w") as handle:
            json.dump({"identifiers": identifiers, "query": payload}, handle, indent=2, sort_keys=True)
        os.replace(str(tmp_path), str(cache_path))
    return identifiers


def rcsb_entry_id(identifier):
    return re.split(r"[_:.]", identifier)[0].upper()


def download_rcsb_pdb(entry_id, cache_dir, timeout=20, max_bytes=20000000):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (entry_id.upper() + ".pdb")
    if path.exists() and path.stat().st_size > 0:
        return str(path)
    url = "https://files.rcsb.org/download/%s.pdb" % entry_id.upper()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > int(max_bytes):
            raise IOError("PDB download for %s exceeds max_bytes=%d" % (entry_id, max_bytes))
        chunks = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > int(max_bytes):
                raise IOError("PDB download for %s exceeds max_bytes=%d" % (entry_id, max_bytes))
        content = b"".join(chunks)
    if not content.strip():
        raise IOError("Empty PDB download for %s" % entry_id)
    tmp_path = cache / (entry_id.upper() + ".pdb.tmp.%d" % os.getpid())
    with open(tmp_path, "wb") as handle:
        handle.write(content)
    os.replace(str(tmp_path), str(path))
    return str(path)


def homolog_contact_labels(
    target_sequence,
    target_residue_keys,
    current_entry_id=None,
    cache_dir="rcsb_cache",
    identity_cutoff=0.7,
    coverage_cutoff=0.8,
    max_hits=25,
    contact_cutoff=4.5,
    buffer_cutoff=6.0,
    include_common_additives=False,
    download_timeout=20,
    max_download_bytes=20000000,
):
    """Transfer ligand-contact residue positives from RCSB homolog complexes.

    Non-positive residues remain -1 because missing homolog contact evidence is
    not a confident negative label.
    """
    labels = np.zeros(len(target_residue_keys), dtype=np.int32) - 1
    details = []
    try:
        hits = rcsb_sequence_search(
            target_sequence,
            identity_cutoff=identity_cutoff,
            max_hits=max_hits,
            cache_dir=Path(cache_dir) / "sequence_searches",
        )
    except Exception as exc:  # pragma: no cover - network defensive path
        return labels, {
            "positive_count": 0,
            "ignored_count": int(np.sum(labels < 0)),
            "hit_count": 0,
            "hits_used": 0,
            "details": [],
            "warning": "RCSB sequence search failed; homolog labels skipped: %s" % exc,
        }
    for identifier in hits:
        entry_id = rcsb_entry_id(identifier)
        if current_entry_id and entry_id == current_entry_id.upper():
            continue
        try:
            pdb_path = download_rcsb_pdb(
                entry_id,
                cache_dir,
                timeout=download_timeout,
                max_bytes=max_download_bytes,
            )
            atoms = parse_pdb_atoms(pdb_path)
        except Exception as exc:  # pragma: no cover - network/file defensive path
            details.append({"entry_id": entry_id, "error": str(exc)})
            continue
        for chain_id, (hit_keys, hit_sequence) in chain_sequences_from_atoms(atoms).items():
            mapping, stats = alignment_mapping(target_sequence, hit_sequence)
            if stats["identity"] < identity_cutoff or stats["coverage"] < coverage_cutoff:
                continue
            hit_labels, label_stats = local_contact_labels(
                pdb_path,
                hit_keys,
                ligand_spec=None,
                contact_cutoff=contact_cutoff,
                buffer_cutoff=buffer_cutoff,
                negative_policy="unknown_unlabeled",
                include_common_additives=include_common_additives,
            )
            positive_hit_indices = set(np.where(hit_labels == 1)[0].tolist())
            transferred = 0
            for q_index, h_index in mapping.items():
                if h_index in positive_hit_indices:
                    labels[q_index] = 1
                    transferred += 1
            if transferred:
                detail = {
                    "entry_id": entry_id,
                    "chain_id": chain_id,
                    "identity": stats["identity"],
                    "coverage": stats["coverage"],
                    "transferred_positive_count": transferred,
                    "ligand_contact_count": label_stats.get("positive_count", 0),
                }
                details.append(detail)
    return labels, {
        "positive_count": int(np.sum(labels == 1)),
        "ignored_count": int(np.sum(labels < 0)),
        "hit_count": len(hits),
        "hits_used": len([d for d in details if d.get("transferred_positive_count", 0) > 0]),
        "details": details,
    }


def segment_bounds_334(n_frames):
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    first = int(round(n_frames * 0.3))
    second = int(round(n_frames * 0.6))
    first = min(max(first, 1), n_frames)
    second = min(max(second, first + 1 if n_frames > 1 else first), n_frames)
    bounds = [(0, first), (first, second), (second, n_frames)]
    cleaned = []
    for start, end in bounds:
        if start >= n_frames:
            cleaned.append((n_frames - 1, n_frames))
        elif end <= start:
            cleaned.append((start, min(start + 1, n_frames)))
        else:
            cleaned.append((start, end))
    return cleaned


def _kabsch(P, Q):
    P_center = P - P.mean(axis=0, keepdims=True)
    Q_center = Q - Q.mean(axis=0, keepdims=True)
    covariance = P_center.T.dot(Q_center)
    V, _, Wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(V.dot(Wt)) < 0:
        correction[-1, -1] = -1
    rotation = V.dot(correction).dot(Wt)
    return P_center.dot(rotation)


def _aligned_flattened_coords(coords):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape [frames, atoms, 3]")
    ref = coords[0]
    aligned = []
    for frame in coords:
        aligned.append(_kabsch(frame, ref).reshape(-1))
    return np.asarray(aligned, dtype=np.float64)


def _pairwise_rmsd(coords):
    flat = _aligned_flattened_coords(coords)
    norms = np.sum(flat * flat, axis=1)
    distances2 = norms[:, None] + norms[None, :] - 2.0 * flat.dot(flat.T)
    distances2 = np.maximum(distances2, 0.0)
    n_atoms = max(int(coords.shape[1]), 1)
    return np.sqrt(distances2 / float(n_atoms))


def _k_medoids(distance_matrix, k, max_iterations=8):
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    k = min(int(k), n)
    medoids = [0]
    while len(medoids) < k:
        min_dist = np.min(distance_matrix[:, medoids], axis=1)
        min_dist[medoids] = -1.0
        medoids.append(int(np.argmax(min_dist)))

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iterations):
        labels = np.argmin(distance_matrix[:, medoids], axis=1)
        new_medoids = []
        for cluster_index in range(k):
            members = np.where(labels == cluster_index)[0]
            if len(members) == 0:
                new_medoids.append(medoids[cluster_index])
                continue
            intra = distance_matrix[np.ix_(members, members)]
            new_medoids.append(int(members[np.argmin(np.sum(intra, axis=1))]))
        if new_medoids == medoids:
            break
        medoids = new_medoids
    return sorted(medoids)


def select_representative_indices_334(ca_coords, counts=(3, 3, 4), max_iterations=8):
    """Select 3/3/4 representative frames by per-segment k-medoids clustering.

    `ca_coords` must be an array with shape `[n_frames, n_ca_atoms, 3]`.
    Returned indices are original trajectory frame indices in chronological order.
    """
    ca_coords = np.asarray(ca_coords, dtype=np.float64)
    n_frames = ca_coords.shape[0]
    if n_frames == 0:
        raise ValueError("Cannot select representatives from an empty trajectory")
    selected = []
    for (start, end), k in zip(segment_bounds_334(n_frames), counts):
        local = ca_coords[start:end]
        if local.shape[0] == 0:
            chosen = [max(start - 1, 0)] * int(k)
        elif local.shape[0] == 1:
            chosen = [start] * int(k)
        else:
            distances = _pairwise_rmsd(local)
            local_medoids = _k_medoids(distances, k, max_iterations=max_iterations)
            chosen = [start + idx for idx in local_medoids]
            if len(chosen) < int(k):
                chosen.extend([chosen[-1]] * (int(k) - len(chosen)))
        selected.extend(chosen[: int(k)])
    return sorted([int(i) for i in selected])


def find_dynamic_system_dirs(dynamic_root):
    root = Path(dynamic_root)
    return sorted([p for p in root.glob("*/*") if p.is_dir()])


def infer_current_entry_id(system_dir):
    base = Path(system_dir).name
    parts = base.split("_")
    for part in parts:
        if re.match(r"^[0-9][A-Za-z0-9]{3}$", part):
            return part.upper()
    parent = Path(system_dir).parent.name
    if re.match(r"^[0-9][A-Za-z0-9]{3}$", parent):
        return parent.upper()
    return None


def find_trajectory_and_topology(system_dir):
    path = Path(system_dir)
    trajectory_candidates = [path / "md_dry.nc", path / "traj.nc"]
    for ext in SUPPORTED_TRAJECTORY_EXTENSIONS:
        trajectory_candidates.extend(sorted(path.glob("*" + ext)))
    trajectory = next((p for p in trajectory_candidates if p.exists() and p.is_file()), None)
    topology_candidates = [
        path / "complex.prmtop",
        path / "protein.prmtop",
        path / "md_200.pdb",
        path / "complex_reference.pdb",
    ]
    topology = next((p for p in topology_candidates if p.exists() and p.is_file()), None)
    reference_candidates = [path / "complex_reference.pdb", path / "md_200.pdb"]
    reference = next((p for p in reference_candidates if p.exists() and p.is_file()), None)
    return trajectory, topology, reference


def _import_mdtraj():
    try:
        import mdtraj as md  # pylint: disable=import-error
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "mdtraj is required for trajectory preparation. Install the "
            "PocketMiner environment before running this command."
        ) from exc
    return md


def _residue_key_from_mdtraj_residue(residue):
    chain = getattr(residue.chain, "chain_id", None) or str(getattr(residue.chain, "index", ""))
    resseq = getattr(residue, "resSeq", None)
    try:
        resseq_int = int(resseq) if resseq is not None else None
    except (TypeError, ValueError):
        resseq_int = None
    if resseq is None or (resseq_int is not None and (resseq_int <= 0 or resseq_int == residue.index)):
        resseq = residue.index + 1
    return ResidueKey(str(chain), str(resseq), "", residue.name.upper())


def extract_pocketminer_arrays(traj, frame_indices):
    residues = [r for r in traj.topology.residues if r.name.upper() in AA3_TO_1]
    if not residues:
        raise ValueError("No standard protein residues found in trajectory topology")
    frame_indices = [int(i) for i in frame_indices]
    T = len(frame_indices)
    L = len(residues)
    X = np.zeros((T, L, 4, 3), dtype=np.float32)
    mask = np.ones((L,), dtype=np.float32)
    keys = []
    for r_index, residue in enumerate(residues):
        keys.append(_residue_key_from_mdtraj_residue(residue))
        atoms_by_name = {atom.name.upper(): atom.index for atom in residue.atoms}
        if not all(name in atoms_by_name for name in BACKBONE_ATOMS):
            mask[r_index] = 0.0
            continue
        for atom_pos, atom_name in enumerate(BACKBONE_ATOMS):
            X[:, r_index, atom_pos, :] = traj.xyz[frame_indices, atoms_by_name[atom_name], :]
    S = sequence_ids_from_residue_keys(keys)
    return X, S, mask, keys


def build_md_features(trajectory_path, topology_path=None, cluster_iterations=8):
    md = _import_mdtraj()
    if topology_path:
        traj = md.load(str(trajectory_path), top=str(topology_path))
    else:
        traj = md.load(str(trajectory_path))
    ca_indices = traj.topology.select("protein and name CA")
    if len(ca_indices) == 0:
        raise ValueError("No protein C-alpha atoms found in trajectory")
    selected_frames = select_representative_indices_334(
        traj.xyz[:, ca_indices, :], max_iterations=cluster_iterations
    )
    reference_frame = traj.n_frames - 1
    X_seq, S, mask, residue_keys = extract_pocketminer_arrays(traj, selected_frames)
    X_ref, _, ref_mask, _ = extract_pocketminer_arrays(traj, [reference_frame])
    mask = np.minimum(mask, ref_mask)
    return {
        "X_seq": X_seq,
        "X_ref": X_ref[0],
        "S": S,
        "mask": mask,
        "residue_keys": residue_keys,
        "selected_frames": np.asarray(selected_frames, dtype=np.int32),
        "reference_frame": int(reference_frame),
        "n_frames": int(traj.n_frames),
    }


def sample_id_from_system_dir(system_dir):
    parent = Path(system_dir).parent.name
    name = Path(system_dir).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", parent + "__" + name)


def write_feature_npz(out_path, feature_dict):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out),
        X_seq=feature_dict["X_seq"],
        X_ref=feature_dict["X_ref"],
        S=feature_dict["S"],
        mask=feature_dict["mask"],
        residue_keys=residue_keys_to_strings(feature_dict["residue_keys"]),
        selected_frames=feature_dict["selected_frames"],
        reference_frame=np.asarray([feature_dict["reference_frame"]], dtype=np.int32),
        n_frames=np.asarray([feature_dict["n_frames"]], dtype=np.int32),
    )


def load_feature_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "X_seq": data["X_seq"].astype(np.float32),
        "X_ref": data["X_ref"].astype(np.float32),
        "S": data["S"].astype(np.int32),
        "mask": data["mask"].astype(np.float32),
        "residue_keys": residue_keys_from_strings(data["residue_keys"]),
        "selected_frames": data["selected_frames"].astype(np.int32),
        "reference_frame": int(data["reference_frame"][0]),
        "n_frames": int(data["n_frames"][0]),
    }


def utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
