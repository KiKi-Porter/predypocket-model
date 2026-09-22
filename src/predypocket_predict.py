"""Run PreDyPocket-style PocketMiner inference on an MD trajectory."""

from __future__ import print_function

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from predypocket_pocketminer_model import make_predypocket_pocketminer, restore_model_only
from predypocket_utils import (
    AA3_TO_1,
    build_md_features,
    find_trajectory_and_topology,
    residue_keys_to_strings,
)


def connected_components(indices, coords_angstrom, max_ca_distance=8.0):
    indices = [int(i) for i in indices]
    if not indices:
        return []
    selected = coords_angstrom[indices]
    delta = selected[:, None, :] - selected[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    neighbors = distances <= max_ca_distance
    visited = set()
    components = []
    for local_index, residue_index in enumerate(indices):
        if local_index in visited:
            continue
        stack = [local_index]
        visited.add(local_index)
        component = []
        while stack:
            node = stack.pop()
            component.append(indices[node])
            for nbr in np.where(neighbors[node])[0].tolist():
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        components.append(sorted(component))
    return components


def write_scores_csv(path, residue_keys, scores, selected_frames, reference_frame):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "residue_index", "residue_key", "aa", "score", "selected_frames", "reference_frame",
        ])
        writer.writeheader()
        frame_text = ";".join(str(int(i)) for i in selected_frames)
        for i, (key, score) in enumerate(zip(residue_keys, scores), 1):
            writer.writerow({
                "residue_index": i,
                "residue_key": key.compact(),
                "aa": AA3_TO_1.get(key.resname.upper(), "X"),
                "score": "%.6f" % float(score),
                "selected_frames": frame_text,
                "reference_frame": int(reference_frame),
            })


def write_pockets_csv(path, residue_keys, scores, X_ref, threshold=0.6, max_ca_distance=8.0, min_residues=4):
    ca_angstrom = X_ref[:, 1, :] * 10.0
    positive = np.where(scores >= threshold)[0].tolist()
    components = connected_components(positive, ca_angstrom, max_ca_distance=max_ca_distance)
    components = [component for component in components if len(component) >= min_residues]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "pocket_id", "n_residues", "mean_score", "max_score", "residue_indices", "residue_keys",
        ])
        writer.writeheader()
        for pocket_id, component in enumerate(components, 1):
            pocket_scores = scores[component]
            writer.writerow({
                "pocket_id": pocket_id,
                "n_residues": len(component),
                "mean_score": "%.6f" % float(np.mean(pocket_scores)),
                "max_score": "%.6f" % float(np.max(pocket_scores)),
                "residue_indices": ";".join(str(i + 1) for i in component),
                "residue_keys": ";".join(residue_keys[i].compact() for i in component),
            })
    return components


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Predict pocket residues from an MD trajectory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--system-dir", help="dynamic_data system directory containing md_dry.nc/protein.prmtop.")
    source.add_argument("--trajectory", help="Trajectory path supported by mdtraj, e.g. .nc, .xtc, .dcd, .trr, .pdb.")
    parser.add_argument("--topology", help="Topology path required for most trajectory formats, e.g. .prmtop or .pdb.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Complete PreDyPocket checkpoint prefix to restore.",
    )
    parser.add_argument("--out-prefix", required=True, help="Output prefix for .npy and .csv files.")
    parser.add_argument("--temporal-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-new-classifier", action="store_true", help="Match training if a new classifier was used.")
    parser.add_argument("--cluster-iterations", type=int, default=8)
    parser.add_argument("--pocket-threshold", type=float, default=0.6)
    parser.add_argument("--pocket-ca-distance", type=float, default=8.0)
    parser.add_argument("--min-pocket-residues", type=int, default=4)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.system_dir:
        trajectory, topology, _ = find_trajectory_and_topology(args.system_dir)
        if trajectory is None or topology is None:
            raise SystemExit("Could not infer trajectory/topology from %s" % args.system_dir)
    else:
        trajectory = Path(args.trajectory)
        topology = Path(args.topology) if args.topology else None

    features = build_md_features(trajectory, topology_path=topology, cluster_iterations=args.cluster_iterations)
    inputs = {
        "X_seq": tf.convert_to_tensor(features["X_seq"][None, :, :, :, :]),
        "X_ref": tf.convert_to_tensor(features["X_ref"][None, :, :, :]),
        "S": tf.convert_to_tensor(features["S"][None, :]),
        "mask": tf.convert_to_tensor(features["mask"][None, :]),
    }
    model = make_predypocket_pocketminer(
        None,
        freeze_backbone=True,
        train_classifier=False,
        use_pretrained_classifier=not args.use_new_classifier,
        temporal_dim=args.temporal_dim,
        dropout=args.dropout,
    )
    model(inputs, training=False)  # Build variables before restoring the complete checkpoint.
    restore_model_only(model, args.checkpoint)
    scores = model(inputs, training=False).numpy()[0]
    scores = scores * features["mask"]

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_prefix) + "_scores.npy", scores)
    np.save(str(out_prefix) + "_selected_frames.npy", features["selected_frames"])
    np.save(str(out_prefix) + "_residue_keys.npy", residue_keys_to_strings(features["residue_keys"]))
    write_scores_csv(
        str(out_prefix) + "_scores.csv",
        features["residue_keys"],
        scores,
        features["selected_frames"],
        features["reference_frame"],
    )
    pockets = write_pockets_csv(
        str(out_prefix) + "_pockets.csv",
        features["residue_keys"],
        scores,
        features["X_ref"],
        threshold=args.pocket_threshold,
        max_ca_distance=args.pocket_ca_distance,
        min_residues=args.min_pocket_residues,
    )
    print("Wrote %d residue scores and %d pocket candidates to %s_*" % (len(scores), len(pockets), out_prefix))


if __name__ == "__main__":
    main()
