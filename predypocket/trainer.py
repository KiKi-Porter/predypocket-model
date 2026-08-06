"""Training/evaluation interfaces; dry-run and smoke helpers never update weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .collate import iter_batches
from .checkpoint import write_checkpoint_contract
from .losses import masked_weighted_bce_with_logits
from .metrics import evaluate_grouped_metrics, select_validation_threshold, sigmoid
from .misato_dataset import MisatoDynamicPocketDataset, iter_misato_batches
from .serialization import write_json


VALIDATION_CLASS_ERROR = (
    "Validation split does not contain both positive and negative labels; "
    "PR-AUC-based model selection is undefined."
)


@tf.keras.utils.register_keras_serializable(package="predypocket")
class AdamW(tf.keras.optimizers.Adam):
    """TensorFlow 2.6-compatible Adam with decoupled weight decay."""

    def __init__(self, weight_decay: float = 1e-4, name: str = "AdamW", **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self._set_hyper("weight_decay", weight_decay)

    def _decay_variable(self, variable: tf.Variable) -> tf.Operation:
        variable_dtype = variable.dtype.base_dtype
        learning_rate = self._decayed_lr(variable_dtype)
        weight_decay = self._get_hyper("weight_decay", variable_dtype)
        return variable.assign_sub(
            learning_rate * weight_decay * variable,
            use_locking=self._use_locking,
        )

    def _resource_apply_dense(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        apply_state: Mapping[str, Any] | None = None,
    ) -> tf.Operation:
        decay = self._decay_variable(var)
        with tf.control_dependencies([decay]):
            return super()._resource_apply_dense(grad, var, apply_state=apply_state)

    def _resource_apply_sparse(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        indices: tf.Tensor,
        apply_state: Mapping[str, Any] | None = None,
    ) -> tf.Operation:
        decay = self._decay_variable(var)
        with tf.control_dependencies([decay]):
            return super()._resource_apply_sparse(
                grad, var, indices, apply_state=apply_state
            )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["weight_decay"] = self._serialize_hyperparameter("weight_decay")
        return config


@dataclass
class TrainingCounters:
    optimizer_step_count: int = 0
    scheduler_step_count: int = 0
    epochs_started: int = 0
    skipped_invalid_batches: int = 0


@dataclass
class BackwardSmokeResult:
    forward_success: bool
    loss_success: bool
    backward_success: bool
    gradient_count: int
    finite_gradient_count: int
    gradients_finite: bool
    gradient_global_norm_before_clip: float
    gradient_global_norm_after_clip: float
    gradient_clip_callable: bool
    parameter_update_count: int
    optimizer_step_count: int
    scheduler_step_count: int
    effective_supervision_count: int
    coordinates_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    loss_value: float


@dataclass
class EpochResult:
    mean_loss: float
    valid_batch_count: int
    skipped_batch_count: int
    optimizer_steps: int


@dataclass
class FitResult:
    best_epoch: int
    best_validation_pr_auc: float | None
    epochs_completed: int
    history: list[dict[str, Any]] = field(default_factory=list)
    counters: TrainingCounters = field(default_factory=TrainingCounters)


@dataclass(frozen=True)
class ClassCounts:
    sample_count: int
    positive_count: int
    negative_count: int
    has_both_classes: bool


@dataclass(frozen=True)
class ModelSelectionDecision:
    primary_metric_defined: bool
    primary_metric: float | None
    best_score: float | None
    stale_epochs: int
    best_checkpoint_updated: bool


@dataclass(frozen=True)
class DistributionInfo:
    """Resolved device layout used by a formal training process."""

    requested: str
    strategy: str
    replica_count: int
    devices: tuple[str, ...]
    cuda_visible_devices: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "strategy": self.strategy,
            "replica_count": self.replica_count,
            "devices": list(self.devices),
            "cuda_visible_devices": self.cuda_visible_devices,
        }


def create_distribution_strategy(
    requested: str = "single",
) -> tuple[tf.distribute.Strategy | None, DistributionInfo]:
    """Resolve the requested local strategy after CUDA visibility is configured.

    Physical GPU selection belongs to the process environment (normally
    ``CUDA_VISIBLE_DEVICES``).  TensorFlow exposes those cards as logical
    ``/GPU:0``, ``/GPU:1``, ... and MirroredStrategy mirrors over all visible
    logical GPUs.  Keeping this explicit prevents a normal single-GPU launch
    from unexpectedly consuming every card on a host.
    """

    if requested not in {"single", "mirrored"}:
        raise ValueError(f"Unsupported distribution strategy: {requested!r}")
    visible_gpus = tuple(device.name for device in tf.config.list_logical_devices("GPU"))
    if requested == "single":
        device = visible_gpus[:1] or ("/CPU:0",)
        return None, DistributionInfo(
            requested=requested,
            strategy="default",
            replica_count=1,
            devices=device,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    if len(visible_gpus) < 2:
        raise RuntimeError(
            "Mirrored training requires at least two visible logical GPUs; "
            f"found {len(visible_gpus)}. Set CUDA_VISIBLE_DEVICES before Python starts."
        )
    strategy = tf.distribute.MirroredStrategy(devices=list(visible_gpus))
    return strategy, DistributionInfo(
        requested=requested,
        strategy="MirroredStrategy",
        replica_count=int(strategy.num_replicas_in_sync),
        devices=tuple(visible_gpus),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def validate_global_batch_size(batch_size: int, replica_count: int) -> None:
    """Require enough global examples for every replica to do useful work."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if replica_count > 1 and batch_size < replica_count:
        raise ValueError(
            f"Global batch_size={batch_size} is smaller than the "
            f"{replica_count} replicas; use at least {replica_count}"
        )


def supervision_class_counts(dataset: Any) -> ClassCounts:
    """Count effective labels without loading coordinates or running a model."""

    positive_count = 0
    negative_count = 0
    for sample in dataset.iter_supervision():
        label_key = "labels" if "labels" in sample else "label"
        labels = np.asarray(sample[label_key], dtype=np.float32)
        mask = np.asarray(sample["residue_mask"], dtype=bool) & np.asarray(
            sample["training_mask"], dtype=bool
        )
        if labels.shape != mask.shape:
            raise ValueError("Validation labels and effective mask differ in shape")
        selected = labels[mask]
        positive_count += int(np.sum(selected >= 0.5))
        negative_count += int(np.sum(selected < 0.5))
    sample_count = positive_count + negative_count
    return ClassCounts(
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=negative_count,
        has_both_classes=positive_count > 0 and negative_count > 0,
    )


def validate_validation_for_model_selection(
    validation_dataset: Any,
    allow_single_class_validation: bool = False,
) -> ClassCounts:
    counts = supervision_class_counts(validation_dataset)
    if counts.sample_count == 0:
        raise ValueError("Validation split contains no effective supervision")
    if not counts.has_both_classes and not allow_single_class_validation:
        raise ValueError(VALIDATION_CLASS_ERROR)
    return counts


def update_model_selection(
    primary_metric: float | None,
    best_score: float | None,
    stale_epochs: int,
) -> ModelSelectionDecision:
    """Update early-stopping state without assigning meaning to undefined metrics."""

    metric_defined = primary_metric is not None and np.isfinite(primary_metric)
    if not metric_defined:
        return ModelSelectionDecision(
            primary_metric_defined=False,
            primary_metric=None,
            best_score=best_score,
            stale_epochs=stale_epochs,
            best_checkpoint_updated=False,
        )
    score = float(primary_metric)
    updated = best_score is None or score > best_score
    return ModelSelectionDecision(
        primary_metric_defined=True,
        primary_metric=score,
        best_score=score if updated else best_score,
        stale_epochs=0 if updated else stale_epochs + 1,
        best_checkpoint_updated=updated,
    )


def make_optimizer(
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    mixed_precision: bool = False,
) -> tf.keras.optimizers.Optimizer:
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    if mixed_precision:
        return tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer


def configure_stage(
    model: tf.keras.Model,
    stage: int,
    stage2_enabled: bool = False,
) -> None:
    if stage == 1:
        model.set_spatial_encoder_trainable(False)
        return
    if stage == 2 and stage2_enabled:
        model.set_spatial_encoder_trainable(True)
        return
    if stage == 2:
        raise ValueError("Stage 2 is implemented but disabled by configuration")
    raise ValueError(f"Unsupported training stage {stage}")


def model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist past-only model fields; labels and volume statistics cannot pass."""

    inputs = {
        "coords": batch["coords"] if "coords" in batch else batch["coordinates"],
        "sequence": batch["sequence"],
        "residue_mask": batch["residue_mask"],
    }
    if "time_offsets_ps" in batch:
        inputs["time_offsets_ps"] = batch["time_offsets_ps"]
    if "spatial_frames" in batch:
        if "time_offsets_ps" not in inputs:
            raise ValueError(
                "Independent StaticAnchorPreDyPocket batches cannot use temporal "
                "spatial-frame caches"
            )
        inputs["spatial_frames"] = batch["spatial_frames"]
    return inputs


def _batch_labels(batch: Mapping[str, Any]) -> Any:
    return batch["labels"] if "labels" in batch else batch["label"]


def _iter_batches(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    spatial_cache: Any | None = None,
) -> Iterable[dict[str, Any]]:
    if isinstance(dataset, MisatoDynamicPocketDataset):
        batches = iter_misato_batches(dataset, batch_size, shuffle=shuffle, seed=seed)
    else:
        batches = iter_batches(dataset, batch_size, shuffle=shuffle, seed=seed)
    for batch in batches:
        yield spatial_cache.attach_batch(batch) if spatial_cache is not None else batch


def _distributed_enabled(strategy: tf.distribute.Strategy | None) -> bool:
    return strategy is not None and int(strategy.num_replicas_in_sync) > 1


def _canonical_batch_tensors(
    batch: Mapping[str, Any], include_labels: bool = True
) -> dict[str, tf.Tensor]:
    """Convert the whitelisted batch fields to tensors for replica slicing."""

    inputs = model_inputs(batch)
    tensors = {
        "coords": tf.convert_to_tensor(inputs["coords"], dtype=tf.float32),
        "sequence": tf.convert_to_tensor(inputs["sequence"], dtype=tf.int32),
        "residue_mask": tf.convert_to_tensor(inputs["residue_mask"], dtype=tf.bool),
    }
    if inputs.get("time_offsets_ps") is not None:
        tensors["time_offsets_ps"] = tf.convert_to_tensor(
            inputs["time_offsets_ps"], dtype=tf.float32
        )
    if inputs.get("spatial_frames") is not None:
        tensors["spatial_frames"] = tf.convert_to_tensor(
            inputs["spatial_frames"], dtype=tf.float32
        )
    if include_labels:
        tensors["labels"] = tf.convert_to_tensor(_batch_labels(batch), dtype=tf.float32)
        tensors["training_mask"] = tf.convert_to_tensor(
            batch["training_mask"], dtype=tf.bool
        )
    return tensors


def _tensor_batch_size(value: tf.Tensor) -> int:
    first_dimension = value.shape[0]
    if first_dimension is None:
        first_dimension = int(tf.shape(value)[0].numpy())
    return int(first_dimension)


def _replica_bounds(
    batch_size: int, replica_id: int, replica_count: int
) -> tuple[int, int]:
    """Split a global batch contiguously, retaining a possible short tail."""

    base, remainder = divmod(batch_size, replica_count)
    start = replica_id * base + min(replica_id, remainder)
    length = base + int(replica_id < remainder)
    return start, start + length


def _distribute_tensor_batch(
    strategy: tf.distribute.Strategy,
    tensors: Mapping[str, tf.Tensor],
) -> Mapping[str, Any]:
    """Create per-replica values from a (possibly uneven) global batch."""

    if not tensors:
        raise ValueError("Cannot distribute an empty tensor batch")
    batch_size = _tensor_batch_size(next(iter(tensors.values())))
    if any(_tensor_batch_size(value) != batch_size for value in tensors.values()):
        raise ValueError("All distributed batch tensors must share their first dimension")
    replica_count = int(strategy.num_replicas_in_sync)

    def value_fn(context: tf.distribute.InputContext) -> dict[str, tf.Tensor]:
        replica_id = int(context.replica_id_in_sync_group)
        start, stop = _replica_bounds(batch_size, replica_id, replica_count)
        return {name: value[start:stop] for name, value in tensors.items()}

    return strategy.experimental_distribute_values_from_function(value_fn)


def _masked_loss_sum_and_count(
    logits: tf.Tensor,
    labels: tf.Tensor,
    residue_mask: tf.Tensor,
    training_mask: tf.Tensor,
    pos_weight: float | tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return an unreduced masked BCE numerator and effective-residue count."""

    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, logits.dtype)
    effective_mask = tf.cast(
        tf.cast(residue_mask, tf.bool) & tf.cast(training_mask, tf.bool),
        logits.dtype,
    )
    element_loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        pos_weight=tf.cast(pos_weight, logits.dtype),
    )
    return (
        tf.reduce_sum(element_loss * effective_mask),
        tf.reduce_sum(effective_mask),
    )


def _reduce_gradients(
    strategy: tf.distribute.Strategy,
    per_replica_gradients: Sequence[Any],
    per_replica_presence: Sequence[Any],
    variables: Sequence[tf.Variable],
) -> list[tf.Tensor | None]:
    """Sum replica gradients and preserve ``None`` for disconnected variables."""

    reduced: list[tf.Tensor | None] = []
    for gradient, presence, variable in zip(
        per_replica_gradients, per_replica_presence, variables
    ):
        present = int(
            strategy.reduce(tf.distribute.ReduceOp.SUM, presence, axis=None).numpy()
        )
        if present == 0:
            reduced.append(None)
        else:
            reduced.append(
                strategy.reduce(tf.distribute.ReduceOp.SUM, gradient, axis=None)
            )
    return reduced


def _distributed_gradient_step(
    model: tf.keras.Model,
    batch: Mapping[str, Any],
    pos_weight: float,
    strategy: tf.distribute.Strategy,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
) -> tuple[float | None, int, list[tf.Tensor | None]]:
    """Run one replica-synchronised forward/backward pass without applying grads."""

    tensors = _canonical_batch_tensors(batch, include_labels=True)
    effective_mask = tensors["residue_mask"] & tensors["training_mask"]
    global_count = int(tf.reduce_sum(tf.cast(effective_mask, tf.int32)).numpy())
    variables = list(model.trainable_variables)
    if global_count == 0:
        return None, 0, [None] * len(variables)
    distributed = _distribute_tensor_batch(strategy, tensors)
    normalizer = tf.constant(float(global_count), dtype=tf.float32)

    def replica_step(local: Mapping[str, tf.Tensor]):
        local_batch_size = _tensor_batch_size(local["coords"])
        if local_batch_size == 0:
            zero_gradients = tuple(tf.zeros_like(variable) for variable in variables)
            absent = tuple(tf.constant(0, dtype=tf.int32) for _ in variables)
            return tf.constant(0.0), tf.constant(0.0), zero_gradients, absent
        # Keras layer freezing excludes variables from this list but does not
        # change each Variable's auto-watch flag. Explicit watching prevents a
        # frozen GVP from retaining every frame's internal backward graph.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            local_inputs = {
                name: value
                for name, value in local.items()
                if name not in {"labels", "training_mask"}
            }
            logits = model(**local_inputs, training=True)
            numerator, count = _masked_loss_sum_and_count(
                logits,
                local["labels"],
                local["residue_mask"],
                local["training_mask"],
                pos_weight,
            )
            loss = numerator / normalizer
            gradient_loss = loss
            if optimizer is not None and hasattr(optimizer, "get_scaled_loss"):
                gradient_loss = optimizer.get_scaled_loss(gradient_loss)
        gradients = tape.gradient(gradient_loss, variables)
        if optimizer is not None and hasattr(optimizer, "get_unscaled_gradients"):
            gradients = optimizer.get_unscaled_gradients(gradients)
        dense_gradients: list[tf.Tensor] = []
        presence: list[tf.Tensor] = []
        for gradient, variable in zip(gradients, variables):
            if gradient is None:
                dense_gradients.append(tf.zeros_like(variable))
                presence.append(tf.constant(0, dtype=tf.int32))
            else:
                dense_gradients.append(tf.convert_to_tensor(gradient))
                presence.append(tf.constant(1, dtype=tf.int32))
        return numerator, count, tuple(dense_gradients), tuple(presence)

    per_replica_numerator, per_replica_count, per_replica_gradients, per_replica_presence = (
        strategy.run(replica_step, args=(distributed,))
    )
    numerator = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_numerator, axis=None
    )
    count = int(
        strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_count, axis=None)
        .numpy()
    )
    if count != global_count:
        raise RuntimeError(
            f"Distributed effective-residue count mismatch: {count} != {global_count}"
        )
    gradients = _reduce_gradients(
        strategy,
        per_replica_gradients,
        per_replica_presence,
        variables,
    )
    return float((numerator / normalizer).numpy()), count, gradients


def distributed_predict_batch(
    model: tf.keras.Model,
    batch: Mapping[str, Any],
    strategy: tf.distribute.Strategy,
    model_call_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Run inference on all replicas and restore the original batch order."""

    tensors = _canonical_batch_tensors(batch, include_labels=False)
    distributed = _distribute_tensor_batch(strategy, tensors)
    call_kwargs = dict(model_call_kwargs or {})

    def replica_predict(local: Mapping[str, tf.Tensor]) -> tf.Tensor:
        local_batch_size = _tensor_batch_size(local["coords"])
        if local_batch_size == 0:
            residue_count = tf.shape(local["sequence"])[1]
            return tf.zeros([0, residue_count], dtype=tf.float32)
        return model(**local, training=False, **call_kwargs)

    per_replica_logits = strategy.run(replica_predict, args=(distributed,))
    local_logits = strategy.experimental_local_results(per_replica_logits)
    arrays = [np.asarray(value.numpy(), dtype=np.float32) for value in local_logits]
    arrays = [value for value in arrays if value.shape[0] > 0]
    residue_count = int(tf.shape(tensors["coords"])[2].numpy())
    if not arrays:
        return np.zeros((0, residue_count), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


def snapshot_parameters(model: tf.keras.Model) -> dict[str, np.ndarray]:
    return {
        f"{index}:{variable.name}": variable.numpy().copy()
        for index, variable in enumerate(model.variables)
    }


def count_parameter_updates(
    model: tf.keras.Model, before: Mapping[str, np.ndarray]
) -> int:
    updates = 0
    current_keys = []
    for index, variable in enumerate(model.variables):
        key = f"{index}:{variable.name}"
        current_keys.append(key)
        if key not in before or not np.array_equal(variable.numpy(), before[key]):
            updates += 1
    updates += len(set(before) - set(current_keys))
    return updates


def clip_gradients(
    gradients: Sequence[tf.Tensor | tf.IndexedSlices | None], max_norm: float
) -> tuple[list[tf.Tensor | tf.IndexedSlices | None], tf.Tensor, tf.Tensor]:
    non_null = [gradient for gradient in gradients if gradient is not None]
    if not non_null:
        zero = tf.constant(0.0, dtype=tf.float32)
        return list(gradients), zero, zero
    before_norm = tf.linalg.global_norm(non_null)
    clipped_non_null, _ = tf.clip_by_global_norm(non_null, max_norm)
    iterator = iter(clipped_non_null)
    clipped = [next(iterator) if gradient is not None else None for gradient in gradients]
    after_norm = tf.linalg.global_norm(clipped_non_null)
    return clipped, before_norm, after_norm


def backward_smoke_without_update(
    model: tf.keras.Model,
    batch: Mapping[str, Any],
    pos_weight: float,
    gradient_clip_norm: float = 1.0,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    strategy: tf.distribute.Strategy | None = None,
) -> BackwardSmokeResult:
    """Run forward/loss/backward/clip and prove that no variable was changed."""

    before = snapshot_parameters(model)
    optimizer_iterations_before = (
        int(optimizer.iterations.numpy()) if optimizer is not None else 0
    )
    if _distributed_enabled(strategy):
        assert strategy is not None
        loss_value, effective_count, gradients = _distributed_gradient_step(
            model,
            batch,
            pos_weight,
            strategy,
            optimizer=optimizer,
        )
        if loss_value is None:
            raise ValueError("Smoke batch contains no effective supervision")
        logits_shape = tuple(int(value) for value in np.shape(_batch_labels(batch)))
    else:
        variables = list(model.trainable_variables)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            logits = model(**model_inputs(batch), training=True)
            loss_result = masked_weighted_bce_with_logits(
                logits,
                _batch_labels(batch),
                batch["residue_mask"],
                batch["training_mask"],
                pos_weight,
            )
            if not loss_result.valid or loss_result.loss is None:
                raise ValueError("Smoke batch contains no effective supervision")
            gradient_loss = loss_result.loss
            if optimizer is not None and hasattr(optimizer, "get_scaled_loss"):
                gradient_loss = optimizer.get_scaled_loss(gradient_loss)
        gradients = tape.gradient(gradient_loss, variables)
        if optimizer is not None and hasattr(optimizer, "get_unscaled_gradients"):
            gradients = optimizer.get_unscaled_gradients(gradients)
        loss_value = float(loss_result.loss.numpy())
        effective_count = loss_result.effective_count
        logits_shape = tuple(int(value) for value in tf.shape(logits).numpy())
    clipped, before_norm, after_norm = clip_gradients(gradients, gradient_clip_norm)
    finite = []
    for gradient in clipped:
        if gradient is None:
            continue
        values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
        finite.append(bool(tf.reduce_all(tf.math.is_finite(values)).numpy()))
    optimizer_iterations_after = (
        int(optimizer.iterations.numpy()) if optimizer is not None else 0
    )
    optimizer_steps = optimizer_iterations_after - optimizer_iterations_before
    parameter_updates = count_parameter_updates(model, before)
    if parameter_updates:
        raise AssertionError(f"Smoke backward changed {parameter_updates} model variables")
    if optimizer_steps:
        raise AssertionError("Smoke backward changed optimizer iterations")
    return BackwardSmokeResult(
        forward_success=True,
        loss_success=True,
        backward_success=any(gradient is not None for gradient in gradients),
        gradient_count=sum(gradient is not None for gradient in gradients),
        finite_gradient_count=sum(finite),
        gradients_finite=bool(finite) and all(finite),
        gradient_global_norm_before_clip=float(before_norm.numpy()),
        gradient_global_norm_after_clip=float(after_norm.numpy()),
        gradient_clip_callable=True,
        parameter_update_count=parameter_updates,
        optimizer_step_count=optimizer_steps,
        scheduler_step_count=0,
        effective_supervision_count=effective_count,
        coordinates_shape=tuple(
            int(value)
            for value in np.shape(
                batch["coords"] if "coords" in batch else batch["coordinates"]
            )
        ),
        logits_shape=logits_shape,
        loss_value=loss_value,
    )


def _accumulate_gradients(
    accumulated: list[tf.Tensor | tf.IndexedSlices | None],
    gradients: Sequence[tf.Tensor | tf.IndexedSlices | None],
) -> list[tf.Tensor | tf.IndexedSlices | None]:
    output: list[tf.Tensor | tf.IndexedSlices | None] = []
    for current, gradient in zip(accumulated, gradients):
        if gradient is None:
            output.append(current)
        elif current is None:
            output.append(tf.convert_to_tensor(gradient))
        else:
            output.append(tf.convert_to_tensor(current) + tf.convert_to_tensor(gradient))
    return output


def _apply_gradients(
    optimizer: tf.keras.optimizers.Optimizer,
    gradients: Sequence[tf.Tensor | tf.IndexedSlices | None],
    variables: Sequence[tf.Variable],
    strategy: tf.distribute.Strategy | None = None,
) -> None:
    pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, variables)
        if gradient is not None
    ]
    if not pairs:
        return
    if not _distributed_enabled(strategy):
        optimizer.apply_gradients(pairs)
        return
    assert strategy is not None
    dense_gradients = tuple(tf.convert_to_tensor(gradient) for gradient, _ in pairs)
    active_variables = tuple(variable for _, variable in pairs)

    def replica_apply(local_gradients: Sequence[tf.Tensor]) -> tf.Tensor:
        optimizer.apply_gradients(zip(local_gradients, active_variables))
        return tf.identity(optimizer.iterations)

    strategy.run(replica_apply, args=(dense_gradients,))


def train_one_epoch(
    model: tf.keras.Model,
    dataset: Any,
    optimizer: tf.keras.optimizers.Optimizer,
    pos_weight: float,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float = 1.0,
    seed: int = 42,
    counters: TrainingCounters | None = None,
    strategy: tf.distribute.Strategy | None = None,
    spatial_cache: Any | None = None,
) -> EpochResult:
    """One explicit epoch. This function is never called by dry-run or tests."""

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    counters = counters or TrainingCounters()
    counters.epochs_started += 1
    variables = list(model.trainable_variables)
    accumulated: list[tf.Tensor | tf.IndexedSlices | None] = [None] * len(variables)
    accumulation_count = 0
    losses: list[float] = []
    skipped = 0

    def apply_accumulated() -> None:
        nonlocal accumulated, accumulation_count
        if accumulation_count == 0:
            return
        averaged = [
            gradient / float(accumulation_count) if gradient is not None else None
            for gradient in accumulated
        ]
        clipped, _, _ = clip_gradients(averaged, gradient_clip_norm)
        _apply_gradients(optimizer, clipped, variables, strategy=strategy)
        counters.optimizer_step_count += 1
        accumulated = [None] * len(variables)
        accumulation_count = 0

    for batch in _iter_batches(
        dataset, batch_size, shuffle=True, seed=seed, spatial_cache=spatial_cache
    ):
        if _distributed_enabled(strategy):
            assert strategy is not None
            loss_value, _, gradients = _distributed_gradient_step(
                model,
                batch,
                pos_weight,
                strategy,
                optimizer=optimizer,
            )
            if loss_value is None:
                counters.skipped_invalid_batches += 1
                skipped += 1
                continue
        else:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables)
                logits = model(**model_inputs(batch), training=True)
                loss_result = masked_weighted_bce_with_logits(
                    logits,
                    _batch_labels(batch),
                    batch["residue_mask"],
                    batch["training_mask"],
                    pos_weight,
                )
                gradient_loss = loss_result.loss
                if gradient_loss is not None and hasattr(
                    optimizer, "get_scaled_loss"
                ):
                    gradient_loss = optimizer.get_scaled_loss(gradient_loss)
            if not loss_result.valid or loss_result.loss is None:
                counters.skipped_invalid_batches += 1
                skipped += 1
                continue
            gradients = tape.gradient(gradient_loss, variables)
            if hasattr(optimizer, "get_unscaled_gradients"):
                gradients = optimizer.get_unscaled_gradients(gradients)
            loss_value = float(loss_result.loss.numpy())
        accumulated = _accumulate_gradients(accumulated, gradients)
        accumulation_count += 1
        losses.append(loss_value)
        if accumulation_count == gradient_accumulation_steps:
            apply_accumulated()
    apply_accumulated()
    return EpochResult(
        mean_loss=float(np.mean(losses)) if losses else float("nan"),
        valid_batch_count=len(losses),
        skipped_batch_count=skipped,
        optimizer_steps=counters.optimizer_step_count,
    )


def collect_predictions(
    model: tf.keras.Model,
    dataset: Any,
    batch_size: int,
    pos_weight: float,
    model_call_kwargs: Mapping[str, Any] | None = None,
    strategy: tf.distribute.Strategy | None = None,
    spatial_cache: Any | None = None,
) -> dict[str, Any]:
    labels: list[np.ndarray] = []
    logits_values: list[np.ndarray] = []
    proteins: list[str] = []
    losses: list[float] = []
    call_kwargs = dict(model_call_kwargs or {})
    if "training" in call_kwargs:
        raise ValueError("model_call_kwargs cannot override training=False")
    for batch in _iter_batches(
        dataset, batch_size, shuffle=False, spatial_cache=spatial_cache
    ):
        if _distributed_enabled(strategy):
            assert strategy is not None
            logits_array = distributed_predict_batch(
                model,
                batch,
                strategy,
                model_call_kwargs=call_kwargs,
            )
            numerator, count = _masked_loss_sum_and_count(
                tf.convert_to_tensor(logits_array, dtype=tf.float32),
                _batch_labels(batch),
                batch["residue_mask"],
                batch["training_mask"],
                pos_weight,
            )
            count_value = int(count.numpy())
            if count_value == 0:
                continue
            losses.append(float((numerator / count).numpy()))
        else:
            logits = model(**model_inputs(batch), training=False, **call_kwargs)
            loss_result = masked_weighted_bce_with_logits(
                logits,
                _batch_labels(batch),
                batch["residue_mask"],
                batch["training_mask"],
                pos_weight,
            )
            if not loss_result.valid or loss_result.loss is None:
                continue
            losses.append(float(loss_result.loss.numpy()))
            logits_array = np.asarray(logits.numpy())
        identities = batch["system_id"] if "system_id" in batch else batch["protein_id"]
        if logits_array.shape[0] != len(identities):
            raise RuntimeError(
                "Distributed prediction order/size differs from the global batch"
            )
        for index, protein in enumerate(identities):
            mask = np.asarray(batch["residue_mask"][index], bool) & np.asarray(
                batch["training_mask"][index], bool
            )
            labels.append(np.asarray(_batch_labels(batch)[index], np.float32)[mask])
            logits_values.append(logits_array[index][mask])
            proteins.extend([str(protein)] * int(np.sum(mask)))
    if not labels:
        raise ValueError("Evaluation split contains no effective supervision")
    return {
        "labels": np.concatenate(labels),
        "logits": np.concatenate(logits_values),
        "protein_ids": np.asarray(proteins, dtype=str),
        "mean_loss": float(np.mean(losses)),
    }


def evaluate_dataset(
    model: tf.keras.Model,
    dataset: Any,
    batch_size: int,
    pos_weight: float,
    threshold: float | None,
    model_call_kwargs: Mapping[str, Any] | None = None,
    strategy: tf.distribute.Strategy | None = None,
    spatial_cache: Any | None = None,
) -> dict[str, Any]:
    predictions = collect_predictions(
        model,
        dataset,
        batch_size,
        pos_weight,
        model_call_kwargs=model_call_kwargs,
        strategy=strategy,
        spatial_cache=spatial_cache,
    )
    metrics = evaluate_grouped_metrics(
        predictions["labels"],
        predictions["logits"],
        predictions["protein_ids"],
        threshold,
        from_logits=True,
    )
    metrics["mean_loss"] = predictions["mean_loss"]
    return metrics


def fit_model(
    model: tf.keras.Model,
    train_dataset: Any,
    validation_dataset: Any,
    optimizer: tf.keras.optimizers.Optimizer,
    pos_weight: float,
    output_directory: str | Path,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    gradient_clip_norm: float = 1.0,
    max_epochs: int = 50,
    early_stopping_patience: int = 8,
    seed: int = 42,
    allow_single_class_validation: bool = False,
    primary_validation_metric: str = "ap",
    initial_epoch: int = 0,
    initial_history: Sequence[Mapping[str, Any]] | None = None,
    initial_best_score: float | None = None,
    initial_best_epoch: int = -1,
    initial_stale_epochs: int = 0,
    strategy: tf.distribute.Strategy | None = None,
    spatial_cache: Any | None = None,
) -> FitResult:
    """Full future training loop. Calling this function performs optimization."""

    if primary_validation_metric.lower() not in {"ap", "average_precision"}:
        raise ValueError("MISATO primary_validation_metric must be AP")
    validate_validation_for_model_selection(
        validation_dataset,
        allow_single_class_validation=allow_single_class_validation,
    )
    replica_count = (
        int(strategy.num_replicas_in_sync)
        if _distributed_enabled(strategy)
        else 1
    )
    validate_global_batch_size(batch_size, replica_count)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    counters = TrainingCounters()
    if initial_epoch < 0 or initial_epoch > max_epochs:
        raise ValueError("initial_epoch must be between zero and max_epochs")
    history: list[dict[str, Any]] = [dict(entry) for entry in (initial_history or [])]
    best_score = initial_best_score
    best_epoch = initial_best_epoch
    stale_epochs = initial_stale_epochs
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    for epoch in range(initial_epoch, max_epochs):
        epoch_result = train_one_epoch(
            model,
            train_dataset,
            optimizer,
            pos_weight,
            batch_size,
            gradient_accumulation_steps,
            gradient_clip_norm,
            seed + epoch,
            counters,
            strategy,
            spatial_cache=spatial_cache,
        )
        validation = collect_predictions(
            model,
            validation_dataset,
            batch_size,
            pos_weight,
            strategy=strategy,
            spatial_cache=spatial_cache,
        )
        threshold = select_validation_threshold(
            validation["labels"], sigmoid(validation["logits"]), split="validation"
        )
        validation_metrics = evaluate_grouped_metrics(
            validation["labels"],
            validation["logits"],
            validation["protein_ids"],
            threshold["threshold"],
            from_logits=True,
        )
        score = validation_metrics["pooled"]["average_precision"]
        decision = update_model_selection(score, best_score, stale_epochs)
        entry = {
            "epoch": epoch,
            "train": asdict(epoch_result),
            "validation": validation_metrics,
            "threshold_selection": threshold,
            "validation_primary_metric_defined": decision.primary_metric_defined,
            "validation_primary_metric": decision.primary_metric,
            "validation_primary_metric_name": "average_precision",
            "best_checkpoint_updated": decision.best_checkpoint_updated,
        }
        history.append(entry)
        last_prefix = checkpoint.write(str(output / "last_checkpoint"))
        write_checkpoint_contract(last_prefix, model)
        if decision.best_checkpoint_updated:
            best_score = decision.best_score
            best_epoch = epoch
            best_prefix = checkpoint.write(str(output / "best_checkpoint"))
            write_checkpoint_contract(best_prefix, model)
        stale_epochs = decision.stale_epochs
        write_json(output / "history.json", history, indent=2)
        write_json(
            output / "training_state.json",
            {
                "next_epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_validation_ap": best_score,
                "stale_epochs": stale_epochs,
                "optimizer_step_count_this_process": counters.optimizer_step_count,
                "model_type": getattr(model, "model_type", None),
            },
            indent=2,
        )
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "train_loss": epoch_result.mean_loss,
                    "validation_ap": score,
                    "best_validation_ap": best_score,
                    "best_checkpoint_updated": decision.best_checkpoint_updated,
                    "stale_epochs": stale_epochs,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if decision.primary_metric_defined and stale_epochs >= early_stopping_patience:
            break
    return FitResult(
        best_epoch=best_epoch,
        best_validation_pr_auc=best_score,
        epochs_completed=len(history),
        history=history,
        counters=counters,
    )
