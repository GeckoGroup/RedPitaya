#!/usr/bin/env python3
"""Dump LibCST trees for a Python file or directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import libcst as cst
from libcst.display import dump


def collect_python_files(target: Path) -> list[Path]:
    if target.is_file():
        if target.suffix != ".py":
            raise ValueError(f"Target file is not Python: {target}")
        return [target]

    if target.is_dir():
        return sorted(path for path in target.rglob("*.py") if path.is_file())

    raise ValueError(f"Target does not exist: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display LibCST concrete syntax trees for Python code."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="fitting",
        help="Python file or directory to parse (default: fitting).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit how many files are dumped.",
    )
    parser.add_argument(
        "--show-defaults",
        action="store_true",
        help="Include default-valued CST fields in the output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.target)

    try:
        files = collect_python_files(target)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        print(f"No Python files found under: {target}", file=sys.stderr)
        return 1

    for path in files:
        source = path.read_text(encoding="utf-8")
        print(f"\n=== {path} ===")
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError as exc:
            print(f"Failed to parse {path}: {exc}", file=sys.stderr)
            continue
        print(dump(module, show_defaults=args.show_defaults))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
