"""Hierarchical coloured debug logging for fit procedures.

Output hierarchy:
  Level 0  ──  Procedure  (header / footer)
  Level 1  ──  Step       (fit, set_param, randomize, set_boundaries, …)
  Level 2  ──  Attempt    (retry within a fit step)
  Level 3  ──  Solver     (piecewise / multi-channel stages, restarts)
  Level 4  ──  Detail     (JAX init, cache, segment-spec builder)

Colour scheme (ANSI, respects $NO_COLOR and non-tty):
  Procedure  → bold cyan
  Step       → bold yellow
  Pass       → bold green
  Fail       → bold red
  Timing     → magenta
  Solver     → dim
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

# ── Config ────────────────────────────────────────────────────────


def enabled() -> bool:
    """Return whether fit-debug output is active."""
    raw = str(os.environ.get("REDPITAYA_FIT_DEBUG", "1")).strip().lower()
    return raw not in {"", "0", "false", "off", "no"}


# ── ANSI helpers ──────────────────────────────────────────────────


def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_COLOR = _use_color()


def _sgr(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _dim(t: str) -> str:
    return _sgr("2", t)


def _green(t: str) -> str:
    return _sgr("32", t)


def _red(t: str) -> str:
    return _sgr("31", t)


def _yellow(t: str) -> str:
    return _sgr("33", t)


def _magenta(t: str) -> str:
    return _sgr("35", t)


def _bold_cyan(t: str) -> str:
    return _sgr("1;36", t)


def _bold_yellow(t: str) -> str:
    return _sgr("1;33", t)


def _bold_green(t: str) -> str:
    return _sgr("1;32", t)


def _bold_red(t: str) -> str:
    return _sgr("1;31", t)


# ── Core emit ─────────────────────────────────────────────────────

_PREFIX = "[fit] "
_INDENT = "  "  # 2 spaces per level


def _emit(level: int, line: str) -> None:
    if not enabled():
        return
    indent = _INDENT * level
    print(f"{_PREFIX}{indent}{line}", flush=True)


# ── Formatting helpers ────────────────────────────────────────────


def _names(names: Sequence[str], limit: int = 8) -> str:
    """Compact comma-separated name list, truncated with '…+N'."""
    items = [str(n).strip() for n in names if str(n).strip()]
    if not items:
        return ""
    if len(items) <= limit:
        return ",".join(items)
    return f"{','.join(items[:limit])}…+{len(items) - limit}"


def _r2(r2) -> str:
    if r2 is None:
        return "R²=?"
    return f"R²={float(r2):.6f}"


def _t(seconds: float) -> str:
    return _magenta(f"{seconds:.3f}s")


def _status_icon(status: str) -> str:
    s = str(status).strip().lower()
    if s in ("pass", "ok"):
        return _bold_green("✓ PASS")
    if s == "fail":
        return _bold_red("✗ FAIL")
    if s == "skipped":
        return _yellow("⊘ SKIP")
    return status


# ══════════════════════════════════════════════════════════════════
# Level 0 — Procedure
# ══════════════════════════════════════════════════════════════════


def procedure_start(name: str, n_steps: int, backend: str = "") -> None:
    plural = "s" if n_steps != 1 else ""
    tag = f" - {backend}" if backend else ""
    _emit(0, _bold_cyan(f"══ {name} ({n_steps} step{plural}{tag}) " + "═" * 30))


def procedure_done(elapsed: float, r2=None) -> None:
    parts = [f"Done {elapsed:.3f}s"]
    if r2 is not None:
        parts.append(_r2(r2))
    _emit(0, _bold_cyan(f"══ {' │ '.join(parts)} " + "═" * 30))


# ══════════════════════════════════════════════════════════════════
# Level 1 — Step
# ══════════════════════════════════════════════════════════════════


def step_start(
    index: int,
    total: int,
    step_type: str,
    label: str,
    *,
    channels: Sequence[str] = (),
    n_free: int = 0,
    n_fixed: int = 0,
    n_locked_boundaries: int = 0,
    max_attempts: int = 1,
    retry_mode: str = "",
    seed_from_siblings: bool = False,
) -> None:
    parts = [f"{_bold_yellow(f'Step {index}/{total}')} [{step_type}] {label!r}"]
    if channels:
        parts.append(_dim(f"ch={_names(channels)}"))
    if n_free or n_fixed:
        parts.append(_dim(f"free={n_free} fixed={n_fixed}"))
    if n_locked_boundaries:
        parts.append(_dim(f"locked_bnd={n_locked_boundaries}"))
    if max_attempts > 1:
        parts.append(_dim(f"attempts={max_attempts} mode={retry_mode}"))
    if seed_from_siblings:
        parts.append(_green("sibling-seed=ON"))
    _emit(1, " ".join(parts))


def step_done(
    status: str,
    *,
    r2=None,
    retries_used: int = 0,
    elapsed: float = 0,
    message: str = "",
) -> None:
    parts = [_status_icon(status)]
    if elapsed > 0:
        parts.append(_t(elapsed))
    if message:
        parts.append(_dim(message))
    _emit(1, " ".join(parts))


# ══════════════════════════════════════════════════════════════════
# Level 2 — Attempt (retry within a fit step)
# ══════════════════════════════════════════════════════════════════


def attempt_start(index: int, total: int, strategy: str) -> None:
    _emit(2, f"Attempt {index}/{total} {_dim(strategy)}")


def attempt_done(elapsed: float, r2=None, *, is_best: bool = False) -> None:
    parts = [_t(elapsed)]
    if r2 is not None:
        parts.append(_r2(r2))
    if is_best:
        parts.append(_green("★"))
    _emit(2, " ".join(parts))


def attempt_fail(elapsed: float, error: str) -> None:
    _emit(2, _red(f"✗ {elapsed:.3f}s {error}"))


# ══════════════════════════════════════════════════════════════════
# Level 3 — Solver  (suppressed — keep signatures for callers)
# ══════════════════════════════════════════════════════════════════


def solver_start(kind: str, **_kw) -> None:  # noqa: ARG001
    pass


def solver_stage(name: str, elapsed: float, **_kw) -> None:  # noqa: ARG001
    pass


def solver_restart(index: int, total: int, elapsed: float, **_kw) -> None:  # noqa: ARG001
    pass


def solver_done(kind: str, elapsed: float, **_kw) -> None:  # noqa: ARG001
    pass


# ══════════════════════════════════════════════════════════════════
# Level 4 — Detail
# ══════════════════════════════════════════════════════════════════


def detail(message: str) -> None:
    _emit(2, _dim(str(message)))


# ══════════════════════════════════════════════════════════════════
# Legacy / raw — unstructured single-line debug
# ══════════════════════════════════════════════════════════════════


def raw(message: str) -> None:
    """Backward-compatible unstructured debug line."""
    _emit(0, message)
