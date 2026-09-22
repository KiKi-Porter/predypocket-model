# Data format

PreDyPocket expects one directory per MD system. The default helper recognizes
the `dynamic_data` layout used during development:

```text
dynamic_data/
└── <entry_id>/
    └── <system_id>/
        ├── md_dry.nc
        ├── complex.prmtop
        ├── complex.inpcrd
        ├── protein.prmtop
        └── complex_reference.pdb
```

Required files for feature extraction:

- MD trajectory: `md_dry.nc`, `traj.nc`, or another MDTraj-supported trajectory.
- Topology: preferably `complex.prmtop`; `protein.prmtop`, `md_200.pdb`, or `complex_reference.pdb` are fallback options.
- Frame sampling: trajectories are split chronologically into early/middle/late
  segments with a 3:3:4 ratio, then 3/3/4 representative frames are selected
  by per-segment structural clustering.

Required files for local-contact training labels:

- `complex.prmtop` and `complex.inpcrd`, or a complex PDB containing protein and ligand heavy atoms.

Labels use residue-level values:

- `1`: pocket/contact positive residue.
- `0`: confident negative residue.
- `-1`: ignored residue, excluded from loss and metrics.

The default expanded label is:

```text
expanded_contact = local heavy-atom ligand contact positives union homolog transferred positives
```

Local positives are protein residues with any protein heavy atom within 4.5 A of
any ligand heavy atom. Homolog transfer searches RCSB sequence-similar complex
structures and transfers only positive ligand-contact evidence to the query
sequence.

## Included toy example

The release includes a synthetic smoke-test dataset:

```text
examples/dynamic_data_example/TOY/TOY_TOY_1.LIG_B_101/
├── aa_traj.pdb
└── complex_reference.pdb
```

This example follows the same directory discovery path as real `dynamic_data`
systems. It is intentionally tiny and should be used only to verify sampling,
label preparation, one-epoch training, and inference commands.
