#!/usr/bin/env python
"""Plan or execute protocol-v2 folds sequentially or on isolated GPU workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN = REPO_ROOT / "scripts/predypocket/train_protocol_v2.py"
EVALUATE = REPO_ROOT / "scripts/predypocket/evaluate_protocol_v2.py"
CONFIG = REPO_ROOT / "configs/predypocket_protocol_v2.json"
PROTOCOL_OUTPUT_ROOT = (REPO_ROOT / "outputs/predypocket_protocol_v2").resolve()
FOLD_COUNT = 5
TRAINING_ARTIFACTS = (
    "training.pid",
    "history.json",
    "best_checkpoint.index",
    "last_checkpoint.index",
)


def fold_commands(config: str, device: str, fold: int) -> list[list[str]]:
    if fold not in range(FOLD_COUNT):
        raise ValueError(f"fold must be between 0 and {FOLD_COUNT - 1}")
    planned: list[list[str]] = []
    for variant in ("anchor-matched", "dynamic"):
        planned.append(
            [
                sys.executable,
                str(TRAIN),
                "--config",
                config,
                "--fold",
                str(fold),
                "--model-variant",
                variant,
                "--stage",
                "1",
                "--device",
                device,
                "--seed",
                "42",
            ]
        )
        modes = ("on", "off") if variant == "dynamic" else ("off",)
        for mode in modes:
            planned.append(
                [
                    sys.executable,
                    str(EVALUATE),
                    "--config",
                    config,
                    "--fold",
                    str(fold),
                    "--model-variant",
                    variant,
                    "--temporal-mode",
                    mode,
                    "--split",
                    "validation",
                    "--device",
                    device,
                ]
            )
    return planned


def commands(config: str, device: str) -> list[list[str]]:
    return [
        command
        for fold in range(FOLD_COUNT)
        for command in fold_commands(config, device, fold)
    ]


def parse_gpu_ids(value: str) -> tuple[str, ...]:
    gpu_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpu_ids:
        raise argparse.ArgumentTypeError("--gpu-ids must contain at least one GPU ID")
    if len(gpu_ids) > FOLD_COUNT:
        raise argparse.ArgumentTypeError(
            f"--gpu-ids accepts at most {FOLD_COUNT} GPU IDs"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids must not contain duplicates")
    if any(not gpu_id.isdigit() for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids must be comma-separated integers")
    return gpu_ids


def assign_folds(gpu_ids: Sequence[str]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
    return tuple(
        (gpu_id, tuple(range(index, FOLD_COUNT, len(gpu_ids))))
        for index, gpu_id in enumerate(gpu_ids)
        if index < FOLD_COUNT
    )


def _training_root(config_path: str) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    output = Path(config["output"]["root"])
    training_root = (output if output.is_absolute() else REPO_ROOT / output).resolve()
    try:
        training_root.relative_to(PROTOCOL_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Protocol-v2 parallel output must remain under its isolated root"
        ) from exc
    return training_root


def _validate_available_gpus(gpu_ids: Sequence[str]) -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    available = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    missing = sorted(set(gpu_ids) - available, key=int)
    if missing:
        raise RuntimeError(
            "Requested physical GPU IDs are unavailable: " + ",".join(missing)
        )


def _assert_clean_training_outputs(training_root: Path) -> None:
    collisions = []
    for fold in range(FOLD_COUNT):
        for variant in ("anchor_matched", "dynamic"):
            directory = training_root / f"fold{fold}" / variant
            for name in TRAINING_ARTIFACTS:
                artifact = directory / name
                if artifact.exists():
                    collisions.append(artifact)
    if collisions:
        displayed = "\n".join(f"- {path}" for path in collisions)
        raise RuntimeError(
            "Protocol-v2 training artifacts already exist; refusing a partial launch:\n"
            f"{displayed}"
        )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _variant_from_command(command: Sequence[str]) -> str:
    value = command[command.index("--model-variant") + 1]
    return "anchor_matched" if value == "anchor-matched" else value


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_command(
    command: Sequence[str],
    env: dict[str, str],
    log_handle,
    pid_file: Path | None,
    stop_event: threading.Event,
) -> None:
    print(f"[{_timestamp()}] START {shlex.join(command)}", file=log_handle, flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{process.pid}\n", encoding="ascii")
    try:
        while True:
            try:
                return_code = process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if stop_event.is_set():
                    _terminate(process)
                    raise RuntimeError("Parallel protocol-v2 launch was cancelled")
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if pid_file is not None and pid_file.exists():
            recorded_pid = pid_file.read_text(encoding="ascii").strip()
            if recorded_pid == str(process.pid):
                pid_file.unlink()
    print(f"[{_timestamp()}] DONE  {shlex.join(command)}", file=log_handle, flush=True)


def _run_worker(
    gpu_id: str,
    folds: Sequence[int],
    config: str,
    device: str,
    training_root: Path,
    stop_event: threading.Event,
    console_lock: threading.Lock,
) -> None:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    for fold in folds:
        if stop_event.is_set():
            return
        fold_root = training_root / f"fold{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        log_path = fold_root / "pipeline.log"
        with console_lock:
            print(f"physical_gpu={gpu_id} fold={fold} state=started log={log_path}")
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                print(
                    f"[{_timestamp()}] physical_gpu={gpu_id} fold={fold} pipeline=started",
                    file=log_handle,
                    flush=True,
                )
                try:
                    for command in fold_commands(config, device, fold):
                        is_training = Path(command[1]).resolve() == TRAIN.resolve()
                        pid_file = None
                        if is_training:
                            variant = _variant_from_command(command)
                            pid_file = fold_root / variant / "training.pid"
                        _run_command(command, env, log_handle, pid_file, stop_event)
                except Exception as exc:
                    print(
                        f"[{_timestamp()}] physical_gpu={gpu_id} fold={fold} "
                        f"pipeline=failed error={exc!r}",
                        file=log_handle,
                        flush=True,
                    )
                    raise
                else:
                    print(
                        f"[{_timestamp()}] physical_gpu={gpu_id} fold={fold} "
                        "pipeline=completed",
                        file=log_handle,
                        flush=True,
                    )
        except Exception:
            stop_event.set()
            raise
        with console_lock:
            print(f"physical_gpu={gpu_id} fold={fold} state=completed log={log_path}")


def _print_parallel_plan(
    config: str, device: str, assignments: Sequence[tuple[str, Sequence[int]]]
) -> None:
    print(f"mode=parallel workers={len(assignments)}")
    for gpu_id, folds in assignments:
        print(f"physical_gpu={gpu_id} folds={','.join(map(str, folds))}")
        for fold in folds:
            for command in fold_commands(config, device, fold):
                print(
                    f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu_id} "
                    f"{shlex.join(command)}"
                )


def _execute_parallel(config: str, device: str, gpu_ids: Sequence[str]) -> None:
    assignments = assign_folds(gpu_ids)
    training_root = _training_root(config)
    training_root.mkdir(parents=True, exist_ok=True)
    lock_path = training_root / ".parallel_launcher.lock"
    with lock_path.open("w", encoding="ascii") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another protocol-v2 parallel launcher is running") from exc
        lock_handle.write(f"{os.getpid()}\n")
        lock_handle.flush()
        _assert_clean_training_outputs(training_root)
        _validate_available_gpus(gpu_ids)
        stop_event = threading.Event()
        console_lock = threading.Lock()
        executor = ThreadPoolExecutor(max_workers=len(assignments))
        futures = [
            executor.submit(
                _run_worker,
                gpu_id,
                folds,
                config,
                device,
                training_root,
                stop_event,
                console_lock,
            )
            for gpu_id, folds in assignments
        ]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop_event.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        help=(
            "Comma-separated physical GPU IDs. Each worker exposes one GPU and runs its "
            "assigned folds sequentially."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_ids:
        assignments = assign_folds(args.gpu_ids)
        _print_parallel_plan(args.config, args.device, assignments)
        if not args.execute:
            print("execute=False; no training or evaluation was started")
            return 0
        _execute_parallel(args.config, args.device, args.gpu_ids)
        return 0

    planned = commands(args.config, args.device)
    for command in planned:
        print(shlex.join(command))
    if not args.execute:
        print("execute=False; no training or evaluation was started")
        return 0
    for command in planned:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
