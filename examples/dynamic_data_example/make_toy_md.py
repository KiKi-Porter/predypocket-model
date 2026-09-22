"""Generate the tiny MD PDB example shipped with this repository."""

from __future__ import print_function

import math
from pathlib import Path


SYSTEM_DIR = Path(__file__).resolve().parent / "TOY" / "TOY_TOY_1.LIG_B_101"


def pdb_atom_line(serial, record, name, resname, chain, resseq, x, y, z, element):
    return (
        "{record:<6}{serial:5d} {name:^4s} {resname:>3s} {chain:1s}"
        "{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
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


def frame_lines(frame_index=0, include_model=False):
    lines = []
    if include_model:
        lines.append("MODEL     %4d\n" % (frame_index + 1))
    serial = 1
    phase = 0.45 * math.sin(frame_index / 3.0)
    for resseq in range(1, 9):
        x0 = 3.8 * (resseq - 1)
        bend = 0.22 * math.sin((frame_index + resseq) / 2.0)
        coords = [
            ("N", x0 - 1.25, phase + bend, 0.05 * frame_index, "N"),
            ("CA", x0, phase + bend + 0.35, 0.02 * resseq, "C"),
            ("C", x0 + 1.30, phase + bend, -0.02 * frame_index, "C"),
            ("O", x0 + 2.05, phase + bend - 0.85, -0.04 * frame_index, "O"),
        ]
        for name, x, y, z, element in coords:
            lines.append(pdb_atom_line(serial, "ATOM", name, "ALA", "A", resseq, x, y, z, element))
            serial += 1
    ligand_shift = 0.18 * math.sin(frame_index / 2.0)
    ligand = [
        ("C1", 11.5 + ligand_shift, 1.05, 0.20, "C"),
        ("O1", 12.7 + ligand_shift, 1.05, 0.20, "O"),
    ]
    for name, x, y, z, element in ligand:
        lines.append(pdb_atom_line(serial, "HETATM", name, "LIG", "B", 101, x, y, z, element))
        serial += 1
    if include_model:
        lines.append("ENDMDL\n")
    return lines


def main():
    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    traj_lines = []
    for frame_index in range(12):
        traj_lines.extend(frame_lines(frame_index, include_model=True))
    traj_lines.append("END\n")
    (SYSTEM_DIR / "aa_traj.pdb").write_text("".join(traj_lines))

    ref_lines = frame_lines(11, include_model=False)
    ref_lines.append("END\n")
    (SYSTEM_DIR / "complex_reference.pdb").write_text("".join(ref_lines))
    print("Wrote toy MD example to %s" % SYSTEM_DIR)


if __name__ == "__main__":
    main()
