#!/usr/bin/env python3
"""Report function/class definitions whose names appear only once in a codebase.

The script:
1) walks Python files under a root path,
2) extracts class/function/async-function definitions via AST,
3) counts all NAME tokens across the same files,
4) prints definitions where the bare name token count is exactly one.

This is intentionally a quick heuristic: token counts include comments-free code
tokens only, but they are not semantic references.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Definition:
    kind: str
    name: str
    qualname: str
    path: Path
    line: int
    col: int


def iter_python_files(root: Path, exclude: Iterable[str]) -> Iterator[Path]:
    patterns = tuple(exclude)
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            continue
        yield path


def iter_name_tokens(source: str) -> Iterator[str]:
    reader = io.StringIO(source).readline
    try:
        for tok in tokenize.generate_tokens(reader):
            if tok.type == tokenize.NAME:
                yield tok.string
    except tokenize.TokenError:
        return


def extract_definitions(source: str, path: Path) -> list[Definition]:
    tree = ast.parse(source, filename=str(path))
    definitions: list[Definition] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def _qualname(self, name: str) -> str:
            return ".".join((*self.scope, name)) if self.scope else name

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            definitions.append(
                Definition(
                    kind="class",
                    name=node.name,
                    qualname=self._qualname(node.name),
                    path=path,
                    line=int(node.lineno),
                    col=int(node.col_offset) + 1,
                )
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            definitions.append(
                Definition(
                    kind="function",
                    name=node.name,
                    qualname=self._qualname(node.name),
                    path=path,
                    line=int(node.lineno),
                    col=int(node.col_offset) + 1,
                )
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            definitions.append(
                Definition(
                    kind="async-func",
                    name=node.name,
                    qualname=self._qualname(node.name),
                    path=path,
                    line=int(node.lineno),
                    col=int(node.col_offset) + 1,
                )
            )
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

    Visitor().visit(tree)
    return definitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List class/function definitions whose bare name token count is exactly one."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Glob pattern (relative to root) to skip; repeatable. "
            "Example: --exclude 'fitting/jaxfit/*'"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Target token count to report (default: 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    exclude = [
        ".git/*",
        "**/.git/*",
        "**/__pycache__/*",
        "**/.ruff_cache/*",
        *list(args.exclude or []),
    ]

    token_counts: Counter[str] = Counter()
    definitions: list[Definition] = []
    parse_failures: list[Path] = []
    scanned = 0

    for py_file in iter_python_files(root, exclude):
        scanned += 1
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        token_counts.update(iter_name_tokens(source))
        try:
            definitions.extend(extract_definitions(source, py_file))
        except SyntaxError:
            parse_failures.append(py_file)

    target = int(args.count)
    matches = [d for d in definitions if token_counts.get(d.name, 0) == target]
    matches.sort(key=lambda d: (str(d.path), d.line, d.col, d.qualname))

    print(f"Scanned files: {scanned}")
    print(f"Definitions found: {len(definitions)}")
    print(f"Definitions with token-count == {target}: {len(matches)}")
    if parse_failures:
        print(f"Files skipped due to parse errors: {len(parse_failures)}")
    print()
    print("kind       count  name                           location")
    print("---------  -----  -----------------------------  ---------------------------")
    for d in matches:
        rel = d.path.relative_to(root).as_posix()
        count = token_counts.get(d.name, 0)
        print(
            f"{d.kind:<9}  {count:>5}  {d.name:<29}  "
            f"{rel}:{d.line}:{d.col} ({d.qualname})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
