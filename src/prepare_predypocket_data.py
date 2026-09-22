"""Prepare PreDyPocket-style dynamic-data features and labels.

This script deliberately ignores existing `cluster_info.txt` and
`cluster_summary.txt` files.  It always recomputes representative frames by the
paper's chronological 3:3:4 split: early/middle/late segments with 3, 3, and 4
cluster representatives respectively.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from predypocket_utils import (
    build_md_features,
    combine_label_vectors,
    find_dynamic_system_dirs,
    find_trajectory_and_topology,
    homolog_contact_labels,
    infer_current_entry_id,
    load_feature_npz,
    local_contact_labels,
    local_contact_labels_from_amber,
    parse_ligand_spec_from_system_dir,
    sample_id_from_system_dir,
    sequence_from_residue_keys,
    utc_timestamp,
    write_feature_npz,
)


POCKETMINER_LABEL_SCHEMES = {
    "ligsite_gp_to_nearest_residue": {
        "filestem": "gp-to-nearest-resi-procedure-min-rank-{6,7}-window-40-stride-1",
        "meaning": "LIGSITE grid-point pocket-volume increase assigned to nearest residue",
        "released_thresholds": [20, 30],
    },
    "ligsite_nearby_pocket_volume": {
        "filestem": "nearby-pv-procedure-min-rank-{6,7}-window-40-stride-1",
        "meaning": "LIGSITE pocket-volume signal assigned to nearby residues",
        "released_thresholds": [87, 116, 145],
    },
    "fpocket_drug_score": {
        "filestem": "fpocket-drug-scores-{max,difference}-cutoff-0.3-window-40ns-stride-{5,25}",
        "meaning": "fpocket druggability-score labels used as soft labels",
        "released_thresholds": [0.3],
    },
    "pm_cryptic_validation": {
        "filestem": "pm-dataset label dictionaries",
        "meaning": "0 negative, 1 cryptic-pocket residue, 2 uncertain/excluded",
        "released_thresholds": None,
    },
}


def parse_methods(value):
    if not value:
        return ["local_contact", "combined_contact"]
    methods = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"local_contact", "homolog_contact", "combined_contact", "expanded_contact"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError("Unknown label method(s): %s" % ", ".join(unknown))
    return methods


def needs_local_labels(methods):
    return any(method in methods for method in ("local_contact", "combined_contact", "expanded_contact"))


def needs_homolog_labels(methods):
    return any(method in methods for method in ("homolog_contact", "combined_contact", "expanded_contact"))


def write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_dataset_csv(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "system_dir", "feature_path", "label_path", "label_method",
        "positive_count", "negative_count", "ignored_count", "n_residues",
        "n_frames", "selected_frames", "reference_frame", "trajectory_path",
        "topology_path", "reference_pdb", "status",
    ]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def append_progress_log(path, row):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "done_count", "total", "source_index", "status", "system_dir",
        "row_count", "positive_count", "error",
    ]
    write_header = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        handle.flush()


def label_stats(labels):
    labels = np.asarray(labels)
    return {
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "ignored_count": int(np.sum(labels < 0)),
    }


def save_label(out_dir, method, sample_id, labels):
    path = Path(out_dir) / "labels" / method / (sample_id + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.asarray(labels, dtype=np.int32))
    return str(path)


def rows_from_metadata(metadata, methods):
    rows = []
    for method in methods:
        method_meta = metadata.get("label_methods", {}).get(method)
        if not method_meta:
            continue
        stats = method_meta.get("stats", {})
        n_residues = metadata.get("n_residues", "")
        if not n_residues and method_meta.get("path") and Path(method_meta["path"]).exists():
            n_residues = int(np.load(method_meta["path"], mmap_mode="r").shape[0])
        rows.append({
            "sample_id": metadata.get("sample_id", ""),
            "system_dir": metadata.get("system_dir", ""),
            "feature_path": metadata.get("feature_path", ""),
            "label_path": method_meta.get("path", ""),
            "label_method": method,
            "positive_count": stats.get("positive_count", 0),
            "negative_count": stats.get("negative_count", 0),
            "ignored_count": stats.get("ignored_count", 0),
            "n_residues": n_residues,
            "n_frames": metadata.get("n_frames", ""),
            "selected_frames": ";".join(str(i) for i in metadata.get("selected_frames", [])),
            "reference_frame": metadata.get("reference_frame", ""),
            "trajectory_path": metadata.get("trajectory_path", ""),
            "topology_path": metadata.get("topology_path", ""),
            "reference_pdb": metadata.get("reference_pdb", "") or "",
            "status": "ok",
        })
    return rows


def collect_rows_from_metadata(out_dir, methods, sample_ids):
    rows = []
    metadata_dir = Path(out_dir) / "metadata"
    for sample_id in sorted(sample_ids):
        path = metadata_dir / (sample_id + ".json")
        if not path.exists():
            continue
        with open(path, "r") as handle:
            rows.extend(rows_from_metadata(json.load(handle), methods))
    return rows


def prepare_one_system(system_dir, args, methods):
    sample_id = sample_id_from_system_dir(system_dir)
    metadata_path = Path(args.out_dir) / "metadata" / (sample_id + ".json")
    if metadata_path.exists() and not args.force:
        try:
            with open(metadata_path, "r") as handle:
                metadata = json.load(handle)
            feature_ok = metadata.get("feature_path") and Path(metadata["feature_path"]).exists()
            labels_ok = all(
                metadata.get("label_methods", {}).get(method, {}).get("path")
                and Path(metadata["label_methods"][method]["path"]).exists()
                for method in methods
            )
            if feature_ok and labels_ok:
                return rows_from_metadata(metadata, methods)
        except Exception:
            pass
    trajectory, topology, reference = find_trajectory_and_topology(system_dir)
    if trajectory is None or topology is None:
        raise ValueError("missing trajectory or topology")
    system_path = Path(system_dir)
    complex_prmtop = system_path / "complex.prmtop"
    complex_inpcrd = system_path / "complex.inpcrd"

    feature_path = Path(args.out_dir) / "features" / (sample_id + ".npz")
    if feature_path.exists() and not args.force:
        try:
            features = load_feature_npz(str(feature_path))
        except Exception:
            features = build_md_features(
                trajectory,
                topology_path=topology,
                cluster_iterations=args.cluster_iterations,
            )
            write_feature_npz(str(feature_path), features)
    else:
        features = build_md_features(
            trajectory,
            topology_path=topology,
            cluster_iterations=args.cluster_iterations,
        )
        write_feature_npz(str(feature_path), features)

    residue_keys = features["residue_keys"]
    sequence = sequence_from_residue_keys(residue_keys)
    local_labels = None
    local_meta = {"warning": "not requested"}
    if (
        complex_prmtop.exists()
        and complex_inpcrd.exists()
        and needs_local_labels(methods)
    ):
        local_labels, local_meta = local_contact_labels_from_amber(
            str(complex_prmtop),
            str(complex_inpcrd),
            residue_keys,
            ligand_spec=parse_ligand_spec_from_system_dir(system_dir),
            contact_cutoff=args.contact_cutoff,
            buffer_cutoff=args.buffer_cutoff,
            negative_policy="all_non_positive",
            include_common_additives=args.include_common_additives,
        )
        local_meta["reference"] = "%s;%s" % (complex_prmtop, complex_inpcrd)
    elif reference is not None and needs_local_labels(methods):
        local_labels, local_meta = local_contact_labels(
            str(reference),
            residue_keys,
            ligand_spec=parse_ligand_spec_from_system_dir(system_dir),
            contact_cutoff=args.contact_cutoff,
            buffer_cutoff=args.buffer_cutoff,
            negative_policy="all_non_positive",
            include_common_additives=args.include_common_additives,
        )
        local_meta["reference"] = str(reference)
    elif needs_local_labels(methods):
        local_labels = np.zeros(len(residue_keys), dtype=np.int32) - 1
        local_meta = {"warning": "missing reference complex", "ignored_count": len(residue_keys)}

    homolog_labels = None
    homolog_meta = {"warning": "not requested"}
    if args.enable_rcsb and needs_homolog_labels(methods):
        homolog_labels, homolog_meta = homolog_contact_labels(
            sequence,
            residue_keys,
            current_entry_id=infer_current_entry_id(system_dir),
            cache_dir=args.rcsb_cache,
            identity_cutoff=args.homolog_identity,
            coverage_cutoff=args.homolog_coverage,
            max_hits=args.max_homologs,
            contact_cutoff=args.contact_cutoff,
            buffer_cutoff=args.buffer_cutoff,
            include_common_additives=args.include_common_additives,
            download_timeout=args.rcsb_download_timeout,
            max_download_bytes=args.max_rcsb_pdb_bytes,
        )
    elif "homolog_contact" in methods or "expanded_contact" in methods:
        raise ValueError("homolog_contact and expanded_contact require --enable-rcsb")

    rows = []
    label_meta = {}
    for method in methods:
        if method == "local_contact":
            labels = local_labels
            meta = local_meta
        elif method == "homolog_contact":
            labels = homolog_labels
            meta = homolog_meta
        elif method in ("combined_contact", "expanded_contact"):
            vectors = []
            if local_labels is not None:
                vectors.append(local_labels)
            if homolog_labels is not None:
                vectors.append(homolog_labels)
            labels = combine_label_vectors(vectors) if vectors else np.zeros(len(residue_keys), dtype=np.int32) - 1
            meta = {"sources": {"local_contact": local_meta, "homolog_contact": homolog_meta}}
        else:  # pragma: no cover - guarded by parse_methods
            raise ValueError("Unsupported label method: %s" % method)

        stats = label_stats(labels)
        label_path = save_label(args.out_dir, method, sample_id, labels)
        label_meta[method] = {"path": label_path, "stats": stats, "details": meta}
        rows.append({
            "sample_id": sample_id,
            "system_dir": str(system_dir),
            "feature_path": str(feature_path),
            "label_path": label_path,
            "label_method": method,
            "positive_count": stats["positive_count"],
            "negative_count": stats["negative_count"],
            "ignored_count": stats["ignored_count"],
            "n_residues": len(residue_keys),
            "n_frames": features["n_frames"],
            "selected_frames": ";".join(str(i) for i in features["selected_frames"].tolist()),
            "reference_frame": features["reference_frame"],
            "trajectory_path": str(trajectory),
            "topology_path": str(topology),
            "reference_pdb": str(reference) if reference is not None else "",
            "status": "ok",
        })

    write_json(metadata_path, {
        "sample_id": sample_id,
        "system_dir": str(system_dir),
        "created_at": utc_timestamp(),
        "trajectory_path": str(trajectory),
        "topology_path": str(topology),
        "reference_pdb": str(reference) if reference is not None else None,
        "feature_path": str(feature_path),
        "n_frames": features["n_frames"],
        "n_residues": len(residue_keys),
        "selected_frames": features["selected_frames"].tolist(),
        "reference_frame": features["reference_frame"],
        "label_methods": label_meta,
    })
    return rows


def child_command_for_system(script_path, system_dir, args):
    cmd = [
        sys.executable,
        script_path,
        "--system-dir", str(system_dir),
        "--out-dir", str(args.out_dir),
        "--label-methods", str(args.label_methods),
        "--contact-cutoff", str(args.contact_cutoff),
        "--buffer-cutoff", str(args.buffer_cutoff),
        "--cluster-iterations", str(args.cluster_iterations),
        "--workers", "1",
        "--rcsb-cache", str(args.rcsb_cache),
        "--homolog-identity", str(args.homolog_identity),
        "--homolog-coverage", str(args.homolog_coverage),
        "--max-homologs", str(args.max_homologs),
        "--rcsb-download-timeout", str(args.rcsb_download_timeout),
        "--max-rcsb-pdb-bytes", str(args.max_rcsb_pdb_bytes),
        "--no-finalize",
    ]
    if args.force:
        cmd.append("--force")
    if args.enable_rcsb:
        cmd.append("--enable-rcsb")
    if args.include_common_additives:
        cmd.append("--include-common-additives")
    return cmd


def tail_text(path, n_lines=6):
    if not Path(path).exists():
        return ""
    with open(path, "r", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-n_lines:]).strip()


def run_parallel_subprocesses(system_dirs, args, methods):
    workers = max(1, int(args.workers or 1))
    script_path = os.path.abspath(__file__)
    log_dir = Path(args.worker_log_dir or (Path(args.out_dir) / "worker_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(system_dirs, 1))
    active = []
    errors = []
    successes = set()
    done_count = 0
    total = len(system_dirs)

    while pending or active:
        while pending and len(active) < workers:
            index, system_dir = pending.pop(0)
            sample_id = sample_id_from_system_dir(system_dir)
            log_path = log_dir / (sample_id + ".log")
            handle = open(log_path, "w")
            process = subprocess.Popen(
                child_command_for_system(script_path, system_dir, args),
                cwd=os.getcwd(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            active.append({
                "index": index,
                "system_dir": str(system_dir),
                "sample_id": sample_id,
                "log_path": str(log_path),
                "handle": handle,
                "process": process,
                "started_at": time.time(),
            })

        time.sleep(1.0)
        still_active = []
        for item in active:
            process = item["process"]
            return_code = process.poll()
            timed_out = False
            if return_code is None and args.worker_timeout and time.time() - item["started_at"] > args.worker_timeout:
                process.kill()
                return_code = process.wait()
                timed_out = True
            if return_code is None:
                still_active.append(item)
                continue

            item["handle"].close()
            done_count += 1
            metadata_path = Path(args.out_dir) / "metadata" / (item["sample_id"] + ".json")
            if return_code == 0 and metadata_path.exists():
                with open(metadata_path, "r") as handle:
                    rows = rows_from_metadata(json.load(handle), methods)
                successes.add(item["sample_id"])
                positives = sum(int(row.get("positive_count", 0)) for row in rows)
                append_progress_log(args.progress_log, {
                    "done_count": done_count,
                    "total": total,
                    "source_index": item["index"],
                    "status": "ok",
                    "system_dir": item["system_dir"],
                    "row_count": len(rows),
                    "positive_count": positives,
                })
                print(
                    "[%d/%d done; source %d/%d] OK %s rows=%d positives=%d" % (
                        done_count, total, item["index"], total, item["system_dir"], len(rows), positives,
                    ),
                    flush=True,
                )
            else:
                if timed_out:
                    error = "worker timeout after %.0f seconds" % float(args.worker_timeout)
                else:
                    error = tail_text(item["log_path"]) or "return code %s" % return_code
                errors.append({"system_dir": item["system_dir"], "error": error})
                append_progress_log(args.progress_log, {
                    "done_count": done_count,
                    "total": total,
                    "source_index": item["index"],
                    "status": "error",
                    "system_dir": item["system_dir"],
                    "error": error,
                })
                print(
                    "[%d/%d done; source %d/%d] ERROR %s :: %s" % (
                        done_count, total, item["index"], total, item["system_dir"], error,
                    ),
                    flush=True,
                )
        active = still_active
    return successes, errors


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Prepare strict 3:3:4 PreDyPocket-style data from dynamic_data."
    )
    parser.add_argument("--dynamic-root", default="dynamic_data", help="Root containing dynamic_data systems.")
    parser.add_argument(
        "--out-dir",
        default="PockerMiner/gvp-pocket_pred/data/predypocket_dynamic",
        help="Output directory for features, labels, and manifests.",
    )
    parser.add_argument(
        "--label-methods",
        default="local_contact,combined_contact",
        help="Comma-separated: local_contact, homolog_contact, combined_contact, expanded_contact.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument(
        "--system-dir",
        action="append",
        default=None,
        help="Prepare only this exact system directory. May be supplied multiple times.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute existing feature files.")
    parser.add_argument("--contact-cutoff", type=float, default=4.5, help="Ligand contact cutoff in Angstrom.")
    parser.add_argument("--buffer-cutoff", type=float, default=6.0, help="Residues within this distance but outside contact cutoff are ignored.")
    parser.add_argument("--cluster-iterations", type=int, default=8, help="K-medoids refinement iterations per segment.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel systems to prepare.")
    parser.add_argument(
        "--progress-log",
        default=None,
        help="TSV progress log. Defaults to <out-dir>/progress.tsv.",
    )
    parser.add_argument(
        "--worker-log-dir",
        default=None,
        help="Directory for per-system worker logs in subprocess parallel mode.",
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=1800.0,
        help="Maximum seconds allowed per single-system worker before marking it as failed.",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Prepare systems but do not write dataset CSVs or manifest. Used by subprocess workers.",
    )
    parser.add_argument("--enable-rcsb", action="store_true", help="Enable RCSB homolog-complex contact transfer.")
    parser.add_argument("--homolog-identity", type=float, default=0.70, help="Minimum sequence identity for homolog transfer.")
    parser.add_argument("--homolog-coverage", type=float, default=0.80, help="Minimum query coverage for homolog transfer.")
    parser.add_argument("--max-homologs", type=int, default=25, help="Max RCSB polymer_entity hits per target sequence.")
    parser.add_argument("--rcsb-download-timeout", type=int, default=20, help="Per-RCSB-PDB download timeout in seconds.")
    parser.add_argument("--max-rcsb-pdb-bytes", type=int, default=20000000, help="Skip RCSB PDB downloads larger than this many bytes.")
    parser.add_argument(
        "--rcsb-cache",
        default=None,
        help="Cache directory for downloaded RCSB PDB files. Defaults to <out-dir>/rcsb_cache.",
    )
    parser.add_argument(
        "--include-common-additives",
        action="store_true",
        help="Do not filter common ions/additives when selecting ligand atoms.",
    )
    parser.add_argument(
        "--write-pocketminer-label-summary",
        action="store_true",
        help="Write a JSON summary of released PocketMiner label families.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    methods = parse_methods(args.label_methods)
    out_dir = Path(args.out_dir)
    if args.rcsb_cache is None:
        args.rcsb_cache = str(out_dir / "rcsb_cache")
    if args.progress_log is None:
        args.progress_log = str(out_dir / "progress.tsv")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "prep_config.json", vars(args))
    if args.write_pocketminer_label_summary and not args.no_finalize:
        write_json(out_dir / "pocketminer_label_schemes.json", POCKETMINER_LABEL_SCHEMES)

    if args.system_dir:
        system_dirs = [Path(item) for item in args.system_dir]
    else:
        system_dirs = find_dynamic_system_dirs(args.dynamic_root)
    if args.limit is not None:
        system_dirs = system_dirs[: args.limit]

    all_rows = []
    errors = []
    workers = max(1, int(args.workers or 1))
    success_sample_ids = set()
    if workers == 1:
        for index, system_dir in enumerate(system_dirs, 1):
            print("[%d/%d] %s" % (index, len(system_dirs), system_dir), flush=True)
            try:
                rows = prepare_one_system(system_dir, args, methods)
                all_rows.extend(rows)
                success_sample_ids.add(sample_id_from_system_dir(system_dir))
                print("  OK rows=%d" % len(rows), flush=True)
                append_progress_log(args.progress_log, {
                    "done_count": index,
                    "total": len(system_dirs),
                    "source_index": index,
                    "status": "ok",
                    "system_dir": str(system_dir),
                    "row_count": len(rows),
                    "positive_count": sum(int(row.get("positive_count", 0)) for row in rows),
                })
            except Exception as exc:  # pragma: no cover - operational path
                print("  ERROR: %s" % exc, flush=True)
                errors.append({"system_dir": str(system_dir), "error": str(exc)})
                append_progress_log(args.progress_log, {
                    "done_count": index,
                    "total": len(system_dirs),
                    "source_index": index,
                    "status": "error",
                    "system_dir": str(system_dir),
                    "error": str(exc),
                })
    else:
        success_sample_ids, errors = run_parallel_subprocesses(system_dirs, args, methods)
        all_rows = collect_rows_from_metadata(args.out_dir, methods, success_sample_ids)

    if args.no_finalize:
        if errors:
            raise SystemExit(1)
        return

    for method in methods:
        rows = [row for row in all_rows if row["label_method"] == method]
        write_dataset_csv(out_dir / ("dataset_%s.csv" % method), rows)
    write_json(out_dir / "manifest.json", {
        "created_at": utc_timestamp(),
        "dynamic_root": args.dynamic_root,
        "out_dir": str(out_dir),
        "label_methods": methods,
        "sample_count": len(set(row["sample_id"] for row in all_rows)),
        "row_count": len(all_rows),
        "error_count": len(errors),
        "errors": errors,
    })
    print("Wrote %d dataset rows with %d errors to %s" % (len(all_rows), len(errors), out_dir))


if __name__ == "__main__":
    main()
