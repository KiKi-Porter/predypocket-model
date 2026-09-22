"""Sample 3:3:4 representative MD frames for PreDyPocket."""

from __future__ import print_function

import argparse
import csv
from pathlib import Path

import numpy as np

from predypocket_utils import (
    build_md_features,
    find_trajectory_and_topology,
    residue_keys_to_strings,
    segment_bounds_334,
)


def segment_name(frame_index, n_frames):
    names = ("early", "middle", "late")
    for name, (start, end) in zip(names, segment_bounds_334(n_frames)):
        if int(start) <= int(frame_index) < int(end):
            return name
    return "late"


def write_selected_frames_csv(path, selected_frames, n_frames, reference_frame):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "selected_order",
            "frame_index",
            "segment",
            "normalized_time",
            "n_frames",
            "reference_frame",
        ])
        writer.writeheader()
        denom = max(int(n_frames) - 1, 1)
        for order, frame_index in enumerate(selected_frames, 1):
            writer.writerow({
                "selected_order": order,
                "frame_index": int(frame_index),
                "segment": segment_name(frame_index, n_frames),
                "normalized_time": "%.6f" % (float(frame_index) / float(denom)),
                "n_frames": int(n_frames),
                "reference_frame": int(reference_frame),
            })


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Select the 3 early, 3 middle, and 4 late representative MD frames used by PreDyPocket."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--system-dir", help="dynamic_data system directory containing trajectory/topology files.")
    source.add_argument("--trajectory", help="Trajectory path supported by MDTraj, e.g. .nc, .xtc, .dcd, .trr, .pdb.")
    parser.add_argument("--topology", help="Topology path required for most trajectory formats.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for selected-frame files.")
    parser.add_argument("--cluster-iterations", type=int, default=8)
    parser.add_argument(
        "--write-features",
        action="store_true",
        help="Also write a compact .npz with sampled PocketMiner tensors for debugging.",
    )
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
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_prefix) + "_selected_frames.npy", features["selected_frames"])
    np.save(str(out_prefix) + "_residue_keys.npy", residue_keys_to_strings(features["residue_keys"]))
    write_selected_frames_csv(
        str(out_prefix) + "_selected_frames.csv",
        features["selected_frames"],
        features["n_frames"],
        features["reference_frame"],
    )
    if args.write_features:
        np.savez_compressed(
            str(out_prefix) + "_features.npz",
            X_seq=features["X_seq"],
            X_ref=features["X_ref"],
            S=features["S"],
            mask=features["mask"],
            selected_frames=features["selected_frames"],
        )
    frame_text = ",".join(str(int(i)) for i in features["selected_frames"])
    print(
        "Selected 3:3:4 frames [%s] from %d-frame trajectory; reference frame is %d."
        % (frame_text, int(features["n_frames"]), int(features["reference_frame"]))
    )
    print("Wrote sampled-frame metadata to %s_selected_frames.csv" % out_prefix)


if __name__ == "__main__":
    main()
