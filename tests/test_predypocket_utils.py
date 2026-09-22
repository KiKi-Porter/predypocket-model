import os
import sys
import tempfile
import unittest

import numpy as np

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from predypocket_utils import (  # noqa: E402
    LigandSpec,
    ResidueKey,
    alignment_mapping,
    amber_atoms_from_prmtop_inpcrd,
    combine_label_vectors,
    local_contact_labels,
    local_contact_labels_from_amber,
    parse_amber_restart_coordinates,
    parse_ligand_spec_from_system_dir,
    segment_bounds_334,
    select_representative_indices_334,
)
from prepare_predypocket_data import parse_methods  # noqa: E402


def pdb_line(record, serial, name, resname, chain, resseq, x, y, z, element):
    return (
        "{record:<6}{serial:>5} {name:<4} {resname:>3} {chain:1}{resseq:>4}    "
        "{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 20.00          {element:>2}\n"
    ).format(
        record=record,
        serial=serial,
        name=name,
        resname=resname,
        chain=chain,
        resseq=resseq,
        x=x,
        y=y,
        z=z,
        element=element,
    )


class PreDyPocketUtilsTest(unittest.TestCase):
    def test_segment_bounds_334(self):
        self.assertEqual(segment_bounds_334(20), [(0, 6), (6, 12), (12, 20)])

    def test_select_representative_indices_334(self):
        coords = np.zeros((20, 3, 3), dtype=np.float64)
        coords[:, :, 0] = np.arange(20)[:, None]
        coords[:, 1, 1] = 1.0
        coords[:, 2, 2] = 1.0
        frames = select_representative_indices_334(coords)
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames, sorted(frames))
        self.assertEqual(sum(0 <= f < 6 for f in frames), 3)
        self.assertEqual(sum(6 <= f < 12 for f in frames), 3)
        self.assertEqual(sum(12 <= f < 20 for f in frames), 4)

    def test_local_contact_labels(self):
        content = "".join([
            pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
            pdb_line("ATOM", 2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
            pdb_line("ATOM", 3, "C", "ALA", "A", 1, 1.0, 1.0, 0.0, "C"),
            pdb_line("ATOM", 4, "O", "ALA", "A", 1, 1.0, 1.0, 1.0, "O"),
            pdb_line("ATOM", 5, "N", "GLY", "A", 2, 20.0, 0.0, 0.0, "N"),
            pdb_line("ATOM", 6, "CA", "GLY", "A", 2, 21.0, 0.0, 0.0, "C"),
            pdb_line("ATOM", 7, "C", "GLY", "A", 2, 21.0, 1.0, 0.0, "C"),
            pdb_line("ATOM", 8, "O", "GLY", "A", 2, 21.0, 1.0, 1.0, "O"),
            pdb_line("HETATM", 9, "C1", "LIG", "A", 101, 2.0, 0.0, 0.0, "C"),
        ])
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            labels, stats = local_contact_labels(
                path,
                [ResidueKey("A", "1", "", "ALA"), ResidueKey("A", "2", "", "GLY")],
                ligand_spec=LigandSpec("LIG", "A", "101"),
                contact_cutoff=4.5,
                buffer_cutoff=6.0,
            )
            self.assertEqual(labels.tolist(), [1, 0])
            self.assertEqual(stats["positive_count"], 1)
            self.assertEqual(stats["negative_count"], 1)
        finally:
            os.unlink(path)

    def test_alignment_mapping(self):
        mapping, stats = alignment_mapping("ACDE", "ACXDE")
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 3, 3: 4})
        self.assertAlmostEqual(stats["coverage"], 1.0)
        self.assertAlmostEqual(stats["identity"], 1.0)

    def test_label_combine(self):
        combined = combine_label_vectors([
            np.asarray([0, -1, 0, -1]),
            np.asarray([-1, 1, 1, -1]),
        ])
        self.assertEqual(combined.tolist(), [0, 1, 1, -1])

    def test_ligand_spec_parse(self):
        spec = parse_ligand_spec_from_system_dir("dynamic_data/4KVK/4KVK_4KVK_1.PG4_A_703")
        self.assertEqual(spec, LigandSpec("PG4", "A", "703", ""))

    def test_expanded_contact_method_parse(self):
        self.assertEqual(parse_methods("local_contact,expanded_contact"), ["local_contact", "expanded_contact"])

    def test_hetatm_standard_residue_is_not_ligand(self):
        content = "".join([
            pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
            pdb_line("ATOM", 2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
            pdb_line("HETATM", 3, "CA", "ASN", "A", 2, 1.5, 0.0, 0.0, "C"),
            pdb_line("HETATM", 4, "C1", "LIG", "A", 101, 20.0, 0.0, 0.0, "C"),
        ])
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            labels, stats = local_contact_labels(
                path,
                [ResidueKey("A", "1", "", "ALA")],
                ligand_spec=None,
                contact_cutoff=4.5,
                buffer_cutoff=6.0,
            )
            self.assertEqual(labels.tolist(), [0])
            self.assertEqual(stats["positive_count"], 0)
        finally:
            os.unlink(path)

    def test_amber_contact_labels(self):
        prmtop = """%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
test
%FLAG POINTERS
%FORMAT(10I8)
       5       1       0       0       0       0       0       0       0       0
       0       3       0       0       0       0       0       0       0       0
       0       0       0       0       0       0       0       0       0       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
N   CA  N   CA  C1
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
ALA GLY LIG
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1       3       5
"""
        inpcrd = """test
     5
   0.0000000   0.0000000   0.0000000   1.0000000   0.0000000   0.0000000
  20.0000000   0.0000000   0.0000000  21.0000000   0.0000000   0.0000000
   2.0000000   0.0000000   0.0000000
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = os.path.join(tmpdir, "complex.prmtop")
            crd_path = os.path.join(tmpdir, "complex.inpcrd")
            with open(top_path, "w") as handle:
                handle.write(prmtop)
            with open(crd_path, "w") as handle:
                handle.write(inpcrd)

            atoms = amber_atoms_from_prmtop_inpcrd(top_path, crd_path)
            self.assertEqual(len(atoms), 5)
            labels, stats = local_contact_labels_from_amber(
                top_path,
                crd_path,
                [ResidueKey("0", "1", "", "ALA"), ResidueKey("0", "2", "", "GLY")],
                ligand_spec=LigandSpec("LIG", "A", "999"),
                contact_cutoff=4.5,
                buffer_cutoff=6.0,
            )
            self.assertEqual(labels.tolist(), [1, 0])
            self.assertEqual(stats["ligand_atom_count"], 1)
            self.assertEqual(stats["mapping"]["ignore_chain"], 2)

    def test_amber_restart_fixed_width_without_spaces(self):
        content = """test
     2  0.0000000
   5.9820000-100.3280000   1.5000000  -2.0000000   3.0000000  -4.0000000
"""
        with tempfile.NamedTemporaryFile("w", suffix=".inpcrd", delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            coords = parse_amber_restart_coordinates(path, natom=2)
            self.assertEqual(coords.shape, (2, 3))
            self.assertAlmostEqual(float(coords[0, 0]), 5.982)
            self.assertAlmostEqual(float(coords[0, 1]), -100.328)
            self.assertAlmostEqual(float(coords[1, 2]), -4.0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
