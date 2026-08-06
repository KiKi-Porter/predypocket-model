#!/usr/bin/env python
"""Run five protein-level folds sequentially; --dry-run only prints commands."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/predypocket_1ns_gap1ns_future20ns.json"
TRAIN_SCRIPT = REPO_ROOT / "scripts/predypocket/train_predypocket.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def commands(args: argparse.Namespace) -> list[list[str]]:
    output = []
    for fold in range(5):
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            args.config,
            "--fold",
            str(fold),
            "--stage",
            str(args.stage),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
        ]
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.gradient_accumulation is not None:
            command.extend(
                ["--gradient-accumulation", str(args.gradient_accumulation)]
            )
        if args.max_epochs is not None:
            command.extend(["--max-epochs", str(args.max_epochs)])
        output.append(command)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    planned = commands(args)
    for command in planned:
        print(shlex.join(command))
    if args.dry_run:
        print("dry_run=True folds_started=0")
        return 0
    for command in planned:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

