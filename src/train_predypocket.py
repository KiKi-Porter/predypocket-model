"""Train the PreDyPocket temporal MD model."""

from __future__ import print_function

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf

from predypocket_pocketminer_model import make_predypocket_pocketminer, restore_model_only
from predypocket_utils import load_feature_npz, utc_timestamp
from util import save_checkpoint


tf.get_logger().setLevel("ERROR")


def load_dataset_rows(dataset_csv):
    with open(dataset_csv, "r") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("status", "ok") == "ok"]
    usable = []
    for row in rows:
        if not os.path.exists(row["feature_path"]):
            continue
        if not os.path.exists(row["label_path"]):
            continue
        usable.append(row)
    return usable


def residue_count_for_row(row):
    with np.load(row["feature_path"]) as feature:
        return int(feature["X_ref"].shape[0])


def filter_rows_by_residue_count(rows, max_residues):
    kept = []
    skipped = 0
    max_seen = 0
    for row in rows:
        n_residues = residue_count_for_row(row)
        max_seen = max(max_seen, n_residues)
        if n_residues > max_residues:
            skipped += 1
            continue
        row = dict(row)
        row["n_residues"] = str(n_residues)
        kept.append(row)
    return kept, skipped, max_seen


def split_rows(rows, val_fraction=0.1, seed=42):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_fraction)) if len(rows) > 1 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    return train_rows, val_rows


def iter_batches(rows, batch_size, shuffle=True, seed=42, drop_remainder=False):
    rows = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        if drop_remainder and len(batch) < batch_size:
            continue
        yield batch


def load_batch(rows, as_tensors=True):
    features = [load_feature_npz(row["feature_path"]) for row in rows]
    labels = [np.load(row["label_path"]).astype(np.float32) for row in rows]
    max_len = max(item["X_seq"].shape[1] for item in features)
    batch_size = len(rows)
    time_steps = features[0]["X_seq"].shape[0]

    X_seq = np.zeros((batch_size, time_steps, max_len, 4, 3), dtype=np.float32)
    X_ref = np.zeros((batch_size, max_len, 4, 3), dtype=np.float32)
    S = np.zeros((batch_size, max_len), dtype=np.int32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)
    y = np.zeros((batch_size, max_len), dtype=np.float32) - 1.0

    for i, (feature, label) in enumerate(zip(features, labels)):
        length = feature["X_seq"].shape[1]
        if label.shape[0] != length:
            raise ValueError("Label length mismatch for %s" % rows[i]["sample_id"])
        X_seq[i, :, :length, :, :] = feature["X_seq"]
        X_ref[i, :length, :, :] = feature["X_ref"]
        S[i, :length] = feature["S"]
        mask[i, :length] = feature["mask"]
        y[i, :length] = label
    inputs = {
        "X_seq": X_seq,
        "X_ref": X_ref,
        "S": S,
        "mask": mask,
    }
    if not as_tensors:
        return inputs, y, rows
    return {
        "X_seq": tf.convert_to_tensor(X_seq),
        "X_ref": tf.convert_to_tensor(X_ref),
        "S": tf.convert_to_tensor(S),
        "mask": tf.convert_to_tensor(mask),
    }, tf.convert_to_tensor(y), rows


def count_labels(rows):
    positives = 0
    negatives = 0
    ignored = 0
    for row in rows:
        labels = np.load(row["label_path"])
        positives += int(np.sum(labels == 1))
        negatives += int(np.sum(labels == 0))
        ignored += int(np.sum(labels < 0))
    return positives, negatives, ignored


def masked_weighted_bce(y_true, y_pred, mask, positive_weight=1.0):
    numerator, denominator, valid_count = masked_weighted_bce_sums(y_true, y_pred, mask, positive_weight)
    if int(valid_count.numpy()) == 0:
        return None, 0
    loss = numerator / tf.maximum(denominator, tf.keras.backend.epsilon())
    return loss, int(valid_count.numpy())


def masked_weighted_bce_sums(y_true, y_pred, mask, positive_weight=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    valid = tf.cast(tf.logical_and(mask > 0.0, y_true >= 0.0), tf.float32)
    labels = tf.where(y_true > 0.5, 1.0, 0.0)
    eps = tf.keras.backend.epsilon()
    preds = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    per_residue = -(labels * tf.math.log(preds) + (1.0 - labels) * tf.math.log(1.0 - preds))
    positive_weight = tf.cast(positive_weight, tf.float32)
    weights = valid * tf.where(labels > 0.5, positive_weight, 1.0)
    numerator = tf.reduce_sum(per_residue * weights)
    denominator = tf.reduce_sum(weights)
    valid_count = tf.reduce_sum(valid)
    return numerator, denominator, valid_count


def make_distributed_batch(inputs, labels, strategy):
    num_replicas = strategy.num_replicas_in_sync
    input_splits = {key: tf.split(value, num_replicas, axis=0) for key, value in inputs.items()}
    label_splits = tf.split(labels, num_replicas, axis=0)

    def value_fn(ctx):
        replica_id = ctx.replica_id_in_sync_group
        replica_inputs = {key: splits[replica_id] for key, splits in input_splits.items()}
        return replica_inputs, label_splits[replica_id]

    return strategy.experimental_distribute_values_from_function(value_fn)


def make_distributed_train_step(strategy, model, optimizer, positive_weight):
    positive_weight = tf.constant(float(positive_weight), dtype=tf.float32)

    @tf.function(experimental_relax_shapes=True)
    def distributed_train_step(dist_batch):
        def replica_step(batch):
            inputs, labels = batch
            with tf.GradientTape() as tape:
                preds = model(inputs, training=True)
                numerator, denominator, valid_count = masked_weighted_bce_sums(
                    labels, preds, inputs["mask"], positive_weight
                )
                replica_context = tf.distribute.get_replica_context()
                global_denominator = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, denominator)
                loss = numerator / tf.maximum(global_denominator, tf.keras.backend.epsilon())
            variables = model.trainable_variables
            grads = tape.gradient(loss, variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
            if grads_and_vars:
                optimizer.apply_gradients(grads_and_vars)
            return numerator, denominator, valid_count

        per_replica_numerator, per_replica_denominator, per_replica_valid_count = strategy.run(
            replica_step, args=(dist_batch,)
        )
        total_numerator = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_numerator, axis=None)
        total_denominator = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_denominator, axis=None)
        total_valid_count = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_valid_count, axis=None)
        loss = total_numerator / tf.maximum(total_denominator, tf.keras.backend.epsilon())
        return loss, total_valid_count

    return distributed_train_step


def make_single_model(args, train_rows, device_name=None):
    def build():
        model = make_predypocket_pocketminer(
            args.checkpoint,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_k=args.unfreeze_last_k,
            train_classifier=not args.freeze_classifier,
            use_pretrained_classifier=not args.use_new_classifier,
            temporal_dim=args.temporal_dim,
            dropout=args.dropout,
        )
        dummy_inputs, _, _ = load_batch([train_rows[0]])
        model(dummy_inputs, training=False)
        if args.resume_checkpoint:
            restore_model_only(model, args.resume_checkpoint)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        return model, optimizer

    if device_name is None:
        return build()
    with tf.device(device_name):
        return build()


def make_manual_gpu_replicas(args, train_rows, devices):
    models = []
    optimizers = []
    for device_name in devices:
        model, optimizer = make_single_model(args, train_rows, device_name=device_name)
        models.append(model)
        optimizers.append(optimizer)
    initial_weights = models[0].get_weights()
    for model in models[1:]:
        model.set_weights(initial_weights)
    return models, optimizers


def weighted_label_denominator(labels, mask, positive_weight):
    valid = np.logical_and(mask > 0.0, labels >= 0.0)
    positive_weights = np.where(labels > 0.5, float(positive_weight), 1.0)
    weights = valid.astype(np.float32) * positive_weights.astype(np.float32)
    return float(np.sum(weights)), int(np.sum(valid))


def train_manual_gpu_batch(models, optimizers, devices, inputs_np, labels_np, positive_weight):
    num_replicas = len(models)
    batch_size = labels_np.shape[0]
    if batch_size % num_replicas != 0:
        raise ValueError("manual GPU batch size must be divisible by replica count")
    per_replica = batch_size // num_replicas
    global_denominator, global_valid_count = weighted_label_denominator(
        labels_np, inputs_np["mask"], positive_weight
    )
    if global_valid_count == 0:
        return None, 0

    def replica_gradients(replica_id):
        start = replica_id * per_replica
        end = start + per_replica
        with tf.device(devices[replica_id]):
            replica_inputs = {
                key: tf.convert_to_tensor(value[start:end])
                for key, value in inputs_np.items()
            }
            replica_labels = tf.convert_to_tensor(labels_np[start:end])
            if getattr(models[replica_id], "stop_backbone_gradient", False):
                h_seq, h_ref, replica_mask = models[replica_id].encode_inputs(
                    replica_inputs, training=False
                )
                h_seq = tf.stop_gradient(h_seq)
                h_ref = tf.stop_gradient(h_ref)
                with tf.GradientTape() as tape:
                    preds = models[replica_id].classify_embeddings(
                        h_seq, h_ref, replica_mask, training=True
                    )
                    numerator, _, _ = masked_weighted_bce_sums(
                        replica_labels, preds, replica_mask, positive_weight
                    )
                    loss = numerator / tf.maximum(
                        tf.constant(global_denominator, dtype=tf.float32),
                        tf.keras.backend.epsilon(),
                    )
                variables = models[replica_id].trainable_variables
                grads = tape.gradient(loss, variables)
                return float(numerator.numpy()), grads

            with tf.GradientTape() as tape:
                preds = models[replica_id](replica_inputs, training=True)
                numerator, _, _ = masked_weighted_bce_sums(
                    replica_labels, preds, replica_inputs["mask"], positive_weight
                )
                loss = numerator / tf.maximum(
                    tf.constant(global_denominator, dtype=tf.float32),
                    tf.keras.backend.epsilon(),
                )
            variables = models[replica_id].trainable_variables
            grads = tape.gradient(loss, variables)
            return float(numerator.numpy()), grads

    with ThreadPoolExecutor(max_workers=num_replicas) as pool:
        results = list(pool.map(replica_gradients, range(num_replicas)))

    total_numerator = sum(item[0] for item in results)
    aggregated_grads = []
    for grads_for_var in zip(*[item[1] for item in results]):
        valid_grads = [grad for grad in grads_for_var if grad is not None]
        if not valid_grads:
            aggregated_grads.append(None)
            continue
        with tf.device("/CPU:0"):
            aggregated_grads.append(tf.add_n([tf.identity(grad) for grad in valid_grads]))

    for model, optimizer in zip(models, optimizers):
        grads_and_vars = [
            (grad, variable)
            for grad, variable in zip(aggregated_grads, model.trainable_variables)
            if grad is not None
        ]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)
    return total_numerator / max(global_denominator, float(tf.keras.backend.epsilon())), global_valid_count


def evaluate(model, rows, batch_size, positive_weight=1.0):
    losses = []
    valid_counts = []
    auc = tf.keras.metrics.AUC(name="auc")
    pr_auc = tf.keras.metrics.AUC(curve="PR", name="pr_auc")
    for batch_rows in iter_batches(rows, batch_size, shuffle=False):
        inputs, labels, _ = load_batch(batch_rows)
        preds = model(inputs, training=False)
        loss, valid_count = masked_weighted_bce(labels, preds, inputs["mask"], positive_weight)
        if loss is None:
            continue
        valid = tf.logical_and(inputs["mask"] > 0.0, labels >= 0.0)
        auc.update_state(tf.boolean_mask(labels, valid), tf.boolean_mask(preds, valid))
        pr_auc.update_state(tf.boolean_mask(labels, valid), tf.boolean_mask(preds, valid))
        losses.append(float(loss.numpy()))
        valid_counts.append(valid_count)
    if not losses:
        return {"loss": None, "auc": None, "pr_auc": None, "valid_count": 0}
    return {
        "loss": float(np.average(losses, weights=valid_counts)),
        "auc": float(auc.result().numpy()),
        "pr_auc": float(pr_auc.result().numpy()),
        "valid_count": int(sum(valid_counts)),
    }


def write_history(path, rows):
    fields = ["epoch", "train_loss", "val_loss", "val_auc", "val_pr_auc", "train_valid_count", "val_valid_count"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train PreDyPocket on prepared MD trajectory data.")
    parser.add_argument("--dataset-csv", required=True, help="dataset_<label_method>.csv from prepare_predypocket_data.py")
    parser.add_argument(
        "--checkpoint",
        default="PockerMiner/gvp-pocket_pred/models/pocketminer",
        help="Released PocketMiner checkpoint prefix used to initialize the backbone.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional model checkpoint prefix to resume training from.",
    )
    parser.add_argument(
        "--out-dir",
        default="PockerMiner/gvp-pocket_pred/models/predypocket",
        help="Output directory for checkpoints and history.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Validation batch size. Defaults to --batch-size; use 1 to reduce validation GPU memory.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-rows", type=int, default=None, help="Debug only: use the first N usable dataset rows.")
    parser.add_argument(
        "--max-residues",
        type=int,
        default=None,
        help="Skip samples with more than this many protein residues to avoid GVP OOM on very large complexes.",
    )
    parser.add_argument("--temporal-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--unfreeze-last-k", type=int, default=0, help="Unfreeze the last K PocketMiner encoder layers.")
    parser.add_argument("--freeze-classifier", action="store_true", help="Also freeze PocketMiner's pretrained dense classifier.")
    parser.add_argument("--use-new-classifier", action="store_true", help="Train a new classifier instead of reusing PocketMiner's classifier.")
    parser.add_argument("--positive-weight", type=float, default=None, help="Override class weight for positive labels.")
    parser.add_argument(
        "--mirrored-strategy",
        action="store_true",
        help="Use tf.distribute.MirroredStrategy across all visible GPUs. Set CUDA_VISIBLE_DEVICES first to choose cards.",
    )
    parser.add_argument(
        "--manual-gpu-replicas",
        action="store_true",
        help="Use one model replica per visible GPU and manually aggregate gradients. Useful when MirroredStrategy is incompatible.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    rows = load_dataset_rows(args.dataset_csv)
    if not rows:
        raise SystemExit("No usable rows found in %s" % args.dataset_csv)
    skipped_large = 0
    max_n_residues = None
    if args.max_residues is not None:
        rows, skipped_large, max_n_residues = filter_rows_by_residue_count(rows, args.max_residues)
        if not rows:
            raise SystemExit("No rows remain after --max-residues %d" % args.max_residues)
        print(
            "Filtered %d rows above --max-residues %d; kept %d rows (max seen %d)" % (
                skipped_large, args.max_residues, len(rows), max_n_residues
            )
        )
    if args.limit_rows is not None:
        rows = rows[:args.limit_rows]
    train_rows, val_rows = split_rows(rows, val_fraction=args.val_fraction, seed=args.seed)
    positives, negatives, ignored = count_labels(train_rows)
    positive_weight = args.positive_weight
    if positive_weight is None:
        positive_weight = float(negatives) / float(max(positives, 1))
        positive_weight = max(positive_weight, 1.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mirrored_strategy and args.manual_gpu_replicas:
        raise SystemExit("Choose only one of --mirrored-strategy or --manual-gpu-replicas")

    strategy = None
    distributed_train_step = None
    manual_devices = None
    replica_models = None
    replica_optimizers = None
    if args.mirrored_strategy:
        visible_gpus = tf.config.list_physical_devices("GPU")
        if not visible_gpus:
            raise SystemExit("--mirrored-strategy requested, but TensorFlow sees no visible GPUs")
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        if args.batch_size % strategy.num_replicas_in_sync != 0:
            raise SystemExit(
                "--batch-size must be divisible by the number of MirroredStrategy replicas (%d)" %
                strategy.num_replicas_in_sync
            )
        print(
            "Using MirroredStrategy with %d replicas: %s" % (
                strategy.num_replicas_in_sync,
                list(strategy.extended.worker_devices),
            )
        )

    if args.manual_gpu_replicas:
        visible_gpus = tf.config.list_physical_devices("GPU")
        if not visible_gpus:
            raise SystemExit("--manual-gpu-replicas requested, but TensorFlow sees no visible GPUs")
        manual_devices = ["/GPU:%d" % idx for idx in range(len(visible_gpus))]
        if args.batch_size % len(manual_devices) != 0:
            raise SystemExit(
                "--batch-size must be divisible by visible GPU count (%d)" % len(manual_devices)
            )
        print("Using manual GPU replicas on devices: %s" % manual_devices)

    if manual_devices is not None:
        replica_models, replica_optimizers = make_manual_gpu_replicas(args, train_rows, manual_devices)
        model = replica_models[0]
        optimizer = replica_optimizers[0]
    else:
        build_scope = strategy.scope() if strategy is not None else None
        if build_scope is not None:
            build_scope.__enter__()
        try:
            model, optimizer = make_single_model(args, train_rows)
            if strategy is not None:
                distributed_train_step = make_distributed_train_step(strategy, model, optimizer, positive_weight)
        finally:
            if build_scope is not None:
                build_scope.__exit__(None, None, None)

    with open(out_dir / "training_config.json", "w") as handle:
        payload = vars(args).copy()
        payload.update({
            "created_at": utc_timestamp(),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_positives": positives,
            "train_negatives": negatives,
            "train_ignored": ignored,
            "positive_weight": positive_weight,
            "distributed_replicas": 1 if strategy is None else strategy.num_replicas_in_sync,
            "manual_gpu_replicas": 0 if manual_devices is None else len(manual_devices),
            "manual_gpu_devices": manual_devices,
            "skipped_large_rows": skipped_large,
            "max_n_residues_seen": max_n_residues,
        })
        json.dump(payload, handle, indent=2, sort_keys=True)
    model_id = int(time.time())
    model_path = str(out_dir / "{}_{}")

    history = []
    best_val = None
    best_epoch = None
    for epoch in range(args.epochs):
        losses = []
        valid_counts = []
        for batch_index, batch_rows in enumerate(
            iter_batches(
                train_rows,
                args.batch_size,
                seed=args.seed + epoch,
                drop_remainder=(strategy is not None or manual_devices is not None),
            ),
            1,
        ):
            if manual_devices is not None:
                inputs_np, labels_np, _ = load_batch(batch_rows, as_tensors=False)
                loss, valid_count = train_manual_gpu_batch(
                    replica_models, replica_optimizers, manual_devices, inputs_np, labels_np, positive_weight
                )
                if loss is None:
                    continue
                losses.append(float(loss))
                valid_counts.append(valid_count)
                if batch_index % 25 == 0:
                    print("epoch %d batch %d loss %.5f" % (epoch, batch_index, losses[-1]))
                continue

            inputs, labels, _ = load_batch(batch_rows)
            if strategy is not None:
                dist_batch = make_distributed_batch(inputs, labels, strategy)
                loss, valid_count = distributed_train_step(dist_batch)
                valid_count = int(valid_count.numpy())
                if valid_count == 0:
                    continue
            else:
                with tf.GradientTape() as tape:
                    preds = model(inputs, training=True)
                    loss, valid_count = masked_weighted_bce(labels, preds, inputs["mask"], positive_weight)
                if loss is None:
                    continue
                variables = model.trainable_variables
                grads = tape.gradient(loss, variables)
                grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
                if grads_and_vars:
                    optimizer.apply_gradients(grads_and_vars)
            losses.append(float(loss.numpy()))
            valid_counts.append(valid_count)
            if batch_index % 25 == 0:
                print("epoch %d batch %d loss %.5f" % (epoch, batch_index, losses[-1]))

        train_loss = float(np.average(losses, weights=valid_counts)) if losses else None
        val_metrics = evaluate(model, val_rows, args.eval_batch_size, positive_weight) if val_rows else {
            "loss": None, "auc": None, "pr_auc": None, "valid_count": 0,
        }
        print(
            "epoch %d train_loss=%s val_loss=%s val_auc=%s val_pr_auc=%s" % (
                epoch, train_loss, val_metrics["loss"], val_metrics["auc"], val_metrics["pr_auc"]
            )
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "train_valid_count": int(sum(valid_counts)),
            "val_valid_count": val_metrics["valid_count"],
        })
        save_checkpoint(model_path, model, optimizer, model_id, epoch)
        if val_metrics["loss"] is not None and (best_val is None or val_metrics["loss"] < best_val):
            best_val = val_metrics["loss"]
            best_epoch = epoch

    write_history(out_dir / "history.csv", history)
    with open(out_dir / "best_epoch.txt", "w") as handle:
        handle.write("%s\n" % ("" if best_epoch is None else best_epoch))


if __name__ == "__main__":
    main()
