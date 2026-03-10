#!/usr/bin/env python3
"""
Efficient generic ordered piecewise solver.

Core pipeline:
1) coarse global search over ordered breakpoint indices + quick local segment fits
2) robust full-parameter refinement (segment parameters + ordered breakpoint ratios)

The implementation is generic with respect to segment functional forms.
Each segment is represented by a callable `model_func(x, *params)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from numba import njit as _numba_njit

    NUMBA_AVAILABLE = True
except Exception:
    _numba_njit = None
    NUMBA_AVAILABLE = False

_NUMBA_WARNING_PRINTED = False


def njit_or_noop(*jit_args: Any, **jit_kwargs: Any) -> Callable:
    """
    Decorator shim:
    - uses numba.njit when available
    - otherwise returns the function unchanged and prints one warning
    """

    def decorate(fn: Callable) -> Callable:
        global _NUMBA_WARNING_PRINTED
        if NUMBA_AVAILABLE:
            return _numba_njit(*jit_args, **jit_kwargs)(fn)
        if not _NUMBA_WARNING_PRINTED:
            print("Warning: numba is not available; running without JIT acceleration.")
            _NUMBA_WARNING_PRINTED = True
        return fn

    # Support both @njit_or_noop and @njit_or_noop(...)
    if len(jit_args) == 1 and callable(jit_args[0]) and not jit_kwargs:
        fn = jit_args[0]
        return decorate(fn)
    return decorate


ArrayLike = Union[Sequence[float], np.ndarray]
SegmentCallable = Callable[..., np.ndarray]
_JAX_FIT_MANAGER: Any = None


def _get_jax_fit_manager_cached():
    global _JAX_FIT_MANAGER
    if _JAX_FIT_MANAGER is None:
        from jax_backend import get_jax_fit_manager

        _JAX_FIT_MANAGER = get_jax_fit_manager()
    return _JAX_FIT_MANAGER


@dataclass(frozen=True)
class SegmentSpec:
    model_func: SegmentCallable
    p0: Sequence[float]
    bounds: Tuple[Sequence[float], Sequence[float]]
    periodic_mask: Sequence[bool] = ()
    periodic_periods: Sequence[float] = ()
    periodic_offsets: Sequence[float] = ()
    n_starts: int = 4
    maxfev: int = 3000


@dataclass(frozen=True)
class OrderedPiecewiseConfig:
    min_segment_points: int = 16
    grid_stride: int = 12
    max_grid_evals: int = 160
    local_n_starts: int = 1
    local_maxfev: int = 700
    robust_max_nfev: int = 45000
    prefer_jit: bool = True


@dataclass
class OrderedPiecewiseResult:
    params: np.ndarray
    y_hat: np.ndarray
    boundaries: np.ndarray
    sse: float
    diagnostics: Dict[str, Any]


def _sse_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yh = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.size != yh.size:
        raise ValueError("y_true and y_pred must have equal length.")
    if NUMBA_AVAILABLE:
        return float(_sse_kernel(y, yh))
    valid = np.isfinite(y) & np.isfinite(yh)
    if not np.any(valid):
        return float("inf")
    r = y[valid] - yh[valid]
    return float(np.dot(r, r))


def _boundary_ratio_diff_step_from_x(x_sorted: np.ndarray) -> float:
    x_arr = np.asarray(x_sorted, dtype=float).reshape(-1)
    finite = x_arr[np.isfinite(x_arr)]
    if finite.size < 2:
        return 0.02
    span = max(float(finite[-1] - finite[0]), 1e-12)
    deltas = np.diff(finite)
    deltas = deltas[deltas > 0.0]
    if deltas.size > 0:
        base = float(np.median(deltas)) / span
    else:
        base = 1.0 / max(2, int(finite.size) - 1)
    return float(np.clip(2.0 * base, 1e-4, 0.25))


@njit_or_noop(cache=True, fastmath=True)
def _sse_kernel(y: np.ndarray, yh: np.ndarray) -> float:
    total = 0.0
    found = False
    for i in range(y.size):
        yi = y[i]
        yhi = yh[i]
        if np.isfinite(yi) and np.isfinite(yhi):
            d = yi - yhi
            total += d * d
            found = True
    if not found:
        return np.inf
    return total


@njit_or_noop(cache=True, fastmath=True)
def _ratios_to_pcts_kernel(ratios: np.ndarray) -> np.ndarray:
    out = np.empty(ratios.size, dtype=np.float64)
    acc = 0.0
    for i in range(ratios.size):
        r = ratios[i]
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        acc = acc + r * (1.0 - acc)
        if acc < 0.0:
            acc = 0.0
        elif acc > 1.0:
            acc = 1.0
        out[i] = acc
    return out


@njit_or_noop(cache=True, fastmath=True)
def _pcts_to_ratios_kernel(pcts: np.ndarray) -> np.ndarray:
    out = np.empty(pcts.size, dtype=np.float64)
    prev = 0.0
    for i in range(pcts.size):
        pct = pcts[i]
        if pct < prev:
            pct = prev
        elif pct > 1.0:
            pct = 1.0
        denom = 1.0 - prev
        if denom < 1e-12:
            denom = 1e-12
        out[i] = (pct - prev) / denom
        prev = pct
    return out


@njit_or_noop(cache=True, fastmath=True)
def _blend_sequence_kernel(
    x: np.ndarray,
    boundaries: np.ndarray,
    bw: float,
    seg_outputs: np.ndarray,
    n_seg: int,
    n_points: int,
) -> np.ndarray:
    """Compute blended piecewise prediction in a single compiled pass.

    *seg_outputs* is a contiguous (n_seg * n_points,) flat buffer where
    segment *s* occupies indices ``s * n_points`` through ``(s+1) * n_points``.
    """
    inv_bw = 1.0 / bw if bw > 0.0 else 0.0
    y_hat = np.empty(n_points, dtype=np.float64)
    n_boundaries = n_seg - 1
    for i in range(n_points):
        xi = x[i]
        acc = 0.0
        for s in range(n_seg):
            if n_seg == 1:
                w = 1.0
            elif s == 0:
                t = (xi - boundaries[0]) * inv_bw + 0.5
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                w = 1.0 - t
            elif s == n_boundaries:
                t = (xi - boundaries[s - 1]) * inv_bw + 0.5
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                w = t
            else:
                t_prev = (xi - boundaries[s - 1]) * inv_bw + 0.5
                if t_prev < 0.0:
                    t_prev = 0.0
                elif t_prev > 1.0:
                    t_prev = 1.0
                t_curr = (xi - boundaries[s]) * inv_bw + 0.5
                if t_curr < 0.0:
                    t_curr = 0.0
                elif t_curr > 1.0:
                    t_curr = 1.0
                w = t_prev * (1.0 - t_curr)
            acc += w * seg_outputs[s * n_points + i]
        y_hat[i] = acc
    return y_hat


@njit_or_noop(cache=True, fastmath=True)
def _ratios_to_boundary_values_kernel(
    ratios: np.ndarray, x_min: float, x_span: float
) -> np.ndarray:
    out = np.empty(ratios.size, dtype=np.float64)
    acc = 0.0
    for i in range(ratios.size):
        r = ratios[i]
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        acc = acc + r * (1.0 - acc)
        if acc < 0.0:
            acc = 0.0
        elif acc > 1.0:
            acc = 1.0
        out[i] = x_min + acc * x_span
    return out


@njit_or_noop(cache=True, fastmath=True)
def _boundary_values_to_ratios_kernel(
    boundaries: np.ndarray, x_min: float, inv_span: float
) -> np.ndarray:
    out = np.empty(boundaries.size, dtype=np.float64)
    prev = 0.0
    for i in range(boundaries.size):
        pct = (boundaries[i] - x_min) * inv_span
        if pct < prev:
            pct = prev
        elif pct > 1.0:
            pct = 1.0
        denom = 1.0 - prev
        if denom < 1e-12:
            denom = 1e-12
        out[i] = (pct - prev) / denom
        prev = pct
    return out


def _n_boundaries(segments: Sequence[SegmentSpec]) -> int:
    return max(0, len(segments) - 1)


def _segment_dims(segments: Sequence[SegmentSpec]) -> List[int]:
    return [len(np.asarray(seg.p0, dtype=float).reshape(-1)) for seg in segments]


def _validate_segments(segments: Sequence[SegmentSpec]) -> None:
    if not segments:
        raise ValueError("At least one segment is required.")
    for idx, seg in enumerate(segments):
        lo = np.asarray(seg.bounds[0], dtype=float).reshape(-1)
        hi = np.asarray(seg.bounds[1], dtype=float).reshape(-1)
        p0 = np.asarray(seg.p0, dtype=float).reshape(-1)
        if not (lo.size == hi.size == p0.size):
            raise ValueError(
                f"Segment {idx} has inconsistent param dimensions (bounds/p0 mismatch)."
            )
        p_mask = np.asarray(seg.periodic_mask, dtype=bool).reshape(-1)
        if p_mask.size > 0 and p_mask.size != p0.size:
            raise ValueError(
                f"Segment {idx} has periodic_mask length mismatch: "
                f"{p_mask.size} != {p0.size}."
            )
        p_periods = np.asarray(seg.periodic_periods, dtype=float).reshape(-1)
        if p_periods.size > 0 and p_periods.size != p0.size:
            raise ValueError(
                f"Segment {idx} has periodic_periods length mismatch: "
                f"{p_periods.size} != {p0.size}."
            )
        p_offsets = np.asarray(seg.periodic_offsets, dtype=float).reshape(-1)
        if p_offsets.size > 0 and p_offsets.size != p0.size:
            raise ValueError(
                f"Segment {idx} has periodic_offsets length mismatch: "
                f"{p_offsets.size} != {p0.size}."
            )


def _segment_periodic_arrays(
    seg: SegmentSpec,
    n_params: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return periodic arrays aligned to a segment parameter vector."""
    if n_params is None:
        n_params = int(np.asarray(seg.p0, dtype=float).reshape(-1).size)
    n = int(n_params)

    raw_mask = np.asarray(seg.periodic_mask, dtype=bool).reshape(-1)
    mask = raw_mask if raw_mask.size > 0 else np.zeros(n, dtype=bool)
    if mask.size != n:
        raise ValueError("Segment periodic_mask length must match parameter count.")

    raw_periods = np.asarray(seg.periodic_periods, dtype=float).reshape(-1)
    periods = raw_periods if raw_periods.size > 0 else np.ones(n, dtype=float)
    if periods.size != n:
        raise ValueError("Segment periodic_periods length must match parameter count.")

    raw_offsets = np.asarray(seg.periodic_offsets, dtype=float).reshape(-1)
    offsets = raw_offsets if raw_offsets.size > 0 else np.zeros(n, dtype=float)
    if offsets.size != n:
        raise ValueError("Segment periodic_offsets length must match parameter count.")

    invalid_periods = mask & (~np.isfinite(periods) | (periods <= 0.0))
    if np.any(invalid_periods):
        raise ValueError("Periodic periods must be finite and > 0.")
    invalid_offsets = mask & ~np.isfinite(offsets)
    if np.any(invalid_offsets):
        raise ValueError("Periodic offsets must be finite.")

    return mask, np.where(mask, periods, 1.0), np.where(mask, offsets, 0.0)


def boundary_ratios_to_pcts(ratios: ArrayLike) -> np.ndarray:
    ratios_arr = np.asarray(ratios, dtype=np.float64).reshape(-1)
    if ratios_arr.size == 0:
        return np.asarray([], dtype=float)
    if NUMBA_AVAILABLE:
        return np.asarray(_ratios_to_pcts_kernel(ratios_arr), dtype=float)
    clipped = np.clip(ratios_arr, 0.0, 1.0)
    return np.asarray(1.0 - np.cumprod(1.0 - clipped), dtype=float)


def pcts_to_boundary_ratios(pcts: ArrayLike) -> np.ndarray:
    pcts_arr = np.asarray(pcts, dtype=np.float64).reshape(-1)
    if pcts_arr.size == 0:
        return np.asarray([], dtype=float)
    if NUMBA_AVAILABLE:
        return np.asarray(_pcts_to_ratios_kernel(pcts_arr), dtype=float)
    monotone = np.maximum.accumulate(np.clip(pcts_arr, 0.0, 1.0))
    prev = np.concatenate([np.asarray([0.0], dtype=float), monotone[:-1]])
    denom = np.maximum(1.0 - prev, 1e-12)
    return np.asarray((monotone - prev) / denom, dtype=float)


def _ratios_to_boundary_values_from_stats(
    ratios: np.ndarray,
    x_min: float,
    x_span: float,
    use_jit: bool,
) -> np.ndarray:
    ratios_arr = np.asarray(ratios, dtype=np.float64).reshape(-1)
    if ratios_arr.size == 0:
        return np.asarray([], dtype=float)
    if use_jit and NUMBA_AVAILABLE:
        return np.asarray(
            _ratios_to_boundary_values_kernel(ratios_arr, float(x_min), float(x_span)),
            dtype=float,
        )
    pcts = boundary_ratios_to_pcts(ratios_arr)
    return np.asarray(float(x_min) + pcts * float(x_span), dtype=float)


def _boundary_values_to_ratios_from_stats(
    boundaries: np.ndarray,
    x_min: float,
    inv_span: float,
    use_jit: bool,
) -> np.ndarray:
    b_arr = np.asarray(boundaries, dtype=np.float64).reshape(-1)
    if b_arr.size == 0:
        return np.asarray([], dtype=float)
    if use_jit and NUMBA_AVAILABLE:
        return np.asarray(
            _boundary_values_to_ratios_kernel(b_arr, float(x_min), float(inv_span)),
            dtype=float,
        )
    pcts = np.clip((b_arr - float(x_min)) * float(inv_span), 0.0, 1.0)
    monotone = np.maximum.accumulate(pcts)
    prev = np.concatenate([np.asarray([0.0], dtype=float), monotone[:-1]])
    denom = np.maximum(1.0 - prev, 1e-12)
    return np.asarray((monotone - prev) / denom, dtype=float)


def _default_boundary_ratios(segments: Sequence[SegmentSpec]) -> np.ndarray:
    m = _n_boundaries(segments)
    if m <= 0:
        return np.asarray([], dtype=float)
    pcts = np.linspace(1.0 / (m + 1), m / (m + 1), m)
    return pcts_to_boundary_ratios(pcts)


def _pack_flat_params(
    segments: Sequence[SegmentSpec],
    seg_params: Sequence[ArrayLike],
    boundary_ratios: ArrayLike,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for p in seg_params:
        parts.append(np.asarray(p, dtype=float).reshape(-1))
    if _n_boundaries(segments) > 0:
        parts.append(
            np.clip(np.asarray(boundary_ratios, dtype=float).reshape(-1), 0.0, 1.0)
        )
    if not parts:
        return np.asarray([], dtype=float)
    return np.concatenate(parts)


def _unpack_flat_params(
    segments: Sequence[SegmentSpec],
    flat_params: ArrayLike,
) -> Tuple[List[np.ndarray], np.ndarray]:
    flat = np.asarray(flat_params, dtype=float).reshape(-1)
    dims = _segment_dims(segments)
    n_seg_param = int(np.sum(dims))
    if flat.size < n_seg_param:
        raise ValueError("Flat parameter vector is too short.")

    seg_params: List[np.ndarray] = []
    cursor = 0
    for d in dims:
        seg_params.append(np.asarray(flat[cursor : cursor + d], dtype=float))
        cursor += d

    m = _n_boundaries(segments)
    if m > 0:
        ratios = flat[cursor : cursor + m]
        if ratios.size < m:
            ratios = _default_boundary_ratios(segments)
        ratios = np.clip(np.asarray(ratios, dtype=float), 0.0, 1.0)
    else:
        ratios = np.asarray([], dtype=float)

    return seg_params, ratios


def _piecewise_bounds_and_p0(
    segments: Sequence[SegmentSpec],
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    lower: List[float] = []
    upper: List[float] = []
    p0: List[float] = []
    for seg in segments:
        lo = np.asarray(seg.bounds[0], dtype=float).reshape(-1)
        hi = np.asarray(seg.bounds[1], dtype=float).reshape(-1)
        p0_seg = np.asarray(seg.p0, dtype=float).reshape(-1)
        lower.extend(lo.tolist())
        upper.extend(hi.tolist())
        p0.extend(np.clip(p0_seg, lo, hi).tolist())

    m = _n_boundaries(segments)
    if m > 0:
        lower.extend([0.0] * m)
        upper.extend([1.0] * m)
        p0.extend(_default_boundary_ratios(segments).tolist())

    return (
        np.asarray(p0, dtype=float),
        (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
    )


def _piecewise_periodic_arrays(
    segments: Sequence[SegmentSpec],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    periodic_mask: List[bool] = []
    periodic_periods: List[float] = []
    periodic_offsets: List[float] = []
    for seg in segments:
        mask, periods, offsets = _segment_periodic_arrays(seg)
        periodic_mask.extend(mask.tolist())
        periodic_periods.extend(periods.tolist())
        periodic_offsets.extend(offsets.tolist())

    m = _n_boundaries(segments)
    if m > 0:
        periodic_mask.extend([False] * m)
        periodic_periods.extend([1.0] * m)
        periodic_offsets.extend([0.0] * m)

    return (
        np.asarray(periodic_mask, dtype=bool),
        np.asarray(periodic_periods, dtype=float),
        np.asarray(periodic_offsets, dtype=float),
    )


def _auto_blend_width(x_sorted: np.ndarray) -> float:
    """Compute a blend half-width for lerp transitions.

    The width is large enough to span several data-point spacings so that
    the optimizer sees a smooth gradient when moving a boundary, but small
    enough not to distort the overall fit shape.
    """
    if x_sorted.size < 2:
        return 0.0
    span = float(x_sorted[-1] - x_sorted[0])
    if span <= 0:
        return 0.0
    deltas = np.diff(x_sorted)
    deltas = deltas[deltas > 0.0]
    if deltas.size > 0:
        median_dx = float(np.median(deltas))
    else:
        median_dx = span / max(1, x_sorted.size - 1)
    # Use ~2% of range, but at least 4 data-point spacings
    return float(max(0.02 * span, 4.0 * median_dx))


def _predict_piecewise(
    segments: Sequence[SegmentSpec],
    x: ArrayLike,
    flat_params: ArrayLike,
    use_jit: bool = True,
    blend_width: float = 0.0,
    _x_min: Optional[float] = None,
    _x_span: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    seg_params, ratios = _unpack_flat_params(segments, flat_params)
    if _x_min is not None and _x_span is not None:
        x_min = float(_x_min)
        x_span = float(_x_span)
    elif x_arr.size > 0:
        x_min = float(np.min(x_arr))
        x_span = float(np.max(x_arr) - x_min)
    else:
        x_min = 0.0
        x_span = 0.0
    boundaries = _ratios_to_boundary_values_from_stats(
        ratios, x_min, x_span, use_jit=bool(use_jit)
    )

    n_seg = len(segments)
    bw = float(blend_width)

    if bw > 0.0 and boundaries.size > 0 and n_seg > 1:
        # --- Lerp-blended prediction (smooth boundary transitions) ---
        # Evaluate every segment over the full x range.  If any segment
        # function cannot handle the full range we fall back to the hard
        # boundary path.
        seg_outputs: List[np.ndarray] = []
        blend_ok = True
        for seg_idx, (seg, p) in enumerate(zip(segments, seg_params)):
            try:
                y_seg = np.asarray(seg.model_func(x_arr, *p), dtype=float).reshape(-1)
                if y_seg.size != x_arr.size:
                    blend_ok = False
                    break
            except Exception:
                blend_ok = False
                break
            seg_outputs.append(y_seg)

        if blend_ok:
            if use_jit and NUMBA_AVAILABLE and n_seg > 1:
                # JIT-compiled blend: pack outputs into a flat buffer and
                # compute weights + weighted sum in a single compiled pass.
                seg_flat = np.empty(n_seg * x_arr.size, dtype=np.float64)
                for si in range(n_seg):
                    seg_flat[si * x_arr.size : (si + 1) * x_arr.size] = seg_outputs[si]
                y_hat = _blend_sequence_kernel(
                    x_arr,
                    boundaries,
                    bw,
                    seg_flat,
                    n_seg,
                    x_arr.size,
                )
            else:
                # NumPy fallback: build per-boundary smooth step functions.
                steps: List[np.ndarray] = []
                for b in boundaries:
                    t = np.clip((x_arr - b) / bw + 0.5, 0.0, 1.0)
                    steps.append(t)
                y_hat = np.zeros_like(x_arr, dtype=float)
                for seg_idx in range(n_seg):
                    if seg_idx == 0:
                        w = 1.0 - steps[0]
                    elif seg_idx == n_seg - 1:
                        w = steps[seg_idx - 1]
                    else:
                        w = steps[seg_idx - 1] * (1.0 - steps[seg_idx])
                    y_hat += w * seg_outputs[seg_idx]

            return {
                "y_hat": np.asarray(y_hat, dtype=float),
                "boundaries": np.asarray(boundaries, dtype=float),
            }

    # --- Hard boundary prediction (original behaviour) ---
    idx = np.searchsorted(boundaries, x_arr, side="right")
    y_hat = np.empty_like(x_arr, dtype=float)
    for seg_idx, (seg, p) in enumerate(zip(segments, seg_params)):
        mask = idx == seg_idx
        if not np.any(mask):
            continue
        y_seg = np.asarray(seg.model_func(x_arr[mask], *p), dtype=float).reshape(-1)
        if y_seg.size != int(np.count_nonzero(mask)):
            raise ValueError(f"Segment {seg_idx} output size mismatch.")
        y_hat[mask] = y_seg

    return {
        "y_hat": np.asarray(y_hat, dtype=float),
        "boundaries": np.asarray(boundaries, dtype=float),
    }


def _fit_segment_local(
    seg: SegmentSpec,
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    rng_seed: int,
    seg_index: int = -1,
    n_starts: Optional[int] = None,
    maxfev: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    lo = np.asarray(seg.bounds[0], dtype=float).reshape(-1)
    hi = np.asarray(seg.bounds[1], dtype=float).reshape(-1)
    p0 = np.clip(np.asarray(seg.p0, dtype=float).reshape(-1), lo, hi)
    n_params = int(p0.size)
    periodic_mask, periodic_periods, periodic_offsets = _segment_periodic_arrays(
        seg,
        n_params=n_params,
    )

    if p0.size == 0:
        y_hat = np.asarray(seg.model_func(x_seg), dtype=float).reshape(-1)
        return np.asarray([], dtype=float), _sse_score(y_seg, y_hat)

    n_try = max(1, int(seg.n_starts if n_starts is None else n_starts))
    use_maxfev = max(50, int(seg.maxfev if maxfev is None else maxfev))
    starts: List[np.ndarray] = [p0]
    if n_try > 1:
        rng = np.random.default_rng(int(rng_seed))
        for _ in range(n_try - 1):
            starts.append(rng.uniform(lo, hi))

    # JAXFit is mandatory for local segment curve fitting.
    try:
        _jax_mgr = _get_jax_fit_manager_cached()
    except Exception as _jax_err:
        if seg_index >= 0:
            raise RuntimeError(
                f"Segment {seg_index} JAX backend unavailable: {_jax_err}"
            ) from _jax_err
        raise RuntimeError(f"JAX backend unavailable: {_jax_err}") from _jax_err

    best_params: Optional[np.ndarray] = None
    best_sse = float("inf")
    for start in starts:
        try:
            popt, _pcov = _jax_mgr.curve_fit(
                seg.model_func,
                x_seg,
                y_seg,
                p0=np.asarray(start, dtype=float),
                bounds=(lo, hi),
                periodic_mask=periodic_mask,
                periodic_periods=periodic_periods,
                periodic_offsets=periodic_offsets,
                max_nfev=use_maxfev,
            )
            y_hat = np.asarray(seg.model_func(x_seg, *popt), dtype=float).reshape(-1)
            sse = _sse_score(y_seg, y_hat)
            if sse < best_sse:
                best_sse = float(sse)
                best_params = np.asarray(popt, dtype=float)
        except Exception:
            continue

    if best_params is None:
        if seg_index >= 0:
            raise RuntimeError(f"Segment {seg_index} local fit failed.")
        raise RuntimeError("Segment local fit failed.")
    return best_params, float(best_sse)


def _combo_to_slices(n_points: int, combo: Sequence[int]) -> List[slice]:
    if not combo:
        return [slice(0, n_points)]
    out: List[slice] = []
    start = 0
    for idx in combo:
        end = int(idx) + 1
        out.append(slice(start, end))
        start = end
    out.append(slice(start, n_points))
    return out


def _iter_boundary_index_combos(
    n_points: int,
    n_boundaries: int,
    min_segment_points: int,
    stride: int,
) -> Iterable[Tuple[int, ...]]:
    if n_boundaries <= 0:
        yield ()
        return

    min_seg = max(1, int(min_segment_points))
    step = max(1, int(stride))
    candidates = range(min_seg - 1, n_points - min_seg, step)
    for combo in combinations(candidates, int(n_boundaries)):
        prev = -1
        valid = True
        for idx in combo:
            if (idx - prev) < min_seg:
                valid = False
                break
            prev = idx
        if not valid:
            continue
        if (n_points - 1 - combo[-1]) < min_seg:
            continue
        yield tuple(int(v) for v in combo)


def _sort_xy(x: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have equal length.")
    order = np.argsort(x_arr, kind="mergesort")
    return x_arr[order], y_arr[order]


class _OrderedPiecewiseSolver:
    def __init__(
        self,
        segments: Sequence[SegmentSpec],
        config: Optional[OrderedPiecewiseConfig] = None,
    ) -> None:
        self.segments: Tuple[SegmentSpec, ...] = tuple(segments)
        _validate_segments(self.segments)
        self.config = (
            config
            if isinstance(config, OrderedPiecewiseConfig)
            else OrderedPiecewiseConfig()
        )
        self._use_jit = bool(self.config.prefer_jit)
        self._base_p0, (self._lower, self._upper) = _piecewise_bounds_and_p0(
            self.segments
        )
        (
            self._periodic_mask,
            self._periodic_periods,
            self._periodic_offsets,
        ) = _piecewise_periodic_arrays(self.segments)

    def _predict(
        self,
        x: ArrayLike,
        params: ArrayLike,
        blend_width: float = 0.0,
        _x_min: Optional[float] = None,
        _x_span: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        return _predict_piecewise(
            self.segments,
            x,
            params,
            use_jit=self._use_jit,
            blend_width=blend_width,
            _x_min=_x_min,
            _x_span=_x_span,
        )

    def _fit_global_seed(
        self, x_sorted: np.ndarray, y_sorted: np.ndarray, seed: int = 0
    ) -> Dict[str, Any]:
        if x_sorted.size < (
            len(self.segments) * max(2, self.config.min_segment_points)
        ):
            raise ValueError("Insufficient points for requested piecewise model.")

        n = int(x_sorted.size)
        nb = _n_boundaries(self.segments)
        x_min = float(x_sorted[0]) if n > 0 else 0.0
        x_span = max(float(x_sorted[-1] - x_min), 1e-12) if n > 0 else 1e-12
        inv_span = 1.0 / x_span
        max_grid_evals = int(self.config.max_grid_evals)
        min_seg_pts = max(3, int(self.config.min_segment_points))
        local_n_starts = int(self.config.local_n_starts)
        local_maxfev = int(self.config.local_maxfev)

        best: Optional[Dict[str, Any]] = None
        best_sse = float("inf")
        eval_count = 0
        for combo in _iter_boundary_index_combos(
            n_points=n,
            n_boundaries=nb,
            min_segment_points=self.config.min_segment_points,
            stride=self.config.grid_stride,
        ):
            eval_count += 1
            if eval_count > max_grid_evals:
                break
            slices = _combo_to_slices(n, combo)
            if len(slices) != len(self.segments):
                continue

            total_sse = 0.0
            seg_params: List[np.ndarray] = []
            failed = False
            for seg_idx, (seg, seg_slice) in enumerate(zip(self.segments, slices)):
                xs = x_sorted[seg_slice]
                ys = y_sorted[seg_slice]
                if xs.size < min_seg_pts:
                    failed = True
                    break
                try:
                    p, sse = _fit_segment_local(
                        seg,
                        xs,
                        ys,
                        rng_seed=int(seed) + 997 * (seg_idx + 1) + 65537 * eval_count,
                        seg_index=seg_idx,
                        n_starts=local_n_starts,
                        maxfev=local_maxfev,
                    )
                except Exception:
                    failed = True
                    break
                seg_params.append(np.asarray(p, dtype=float))
                total_sse += float(sse)

            if failed or total_sse >= best_sse:
                continue

            boundaries = (
                x_sorted[np.asarray(combo, dtype=int)]
                if nb > 0
                else np.asarray([], dtype=float)
            )
            ratios = _boundary_values_to_ratios_from_stats(
                boundaries,
                x_min=x_min,
                inv_span=inv_span,
                use_jit=self._use_jit,
            )
            flat = _pack_flat_params(self.segments, seg_params, ratios)
            pred = self._predict(x_sorted, flat, _x_min=x_min, _x_span=x_span)
            best = {
                "params": np.asarray(flat, dtype=float),
                "y_hat": np.asarray(pred["y_hat"], dtype=float),
                "boundaries": np.asarray(pred["boundaries"], dtype=float),
                "sse": float(total_sse),
                "evaluated": int(eval_count),
            }
            best_sse = float(total_sse)

        if best is None:
            raise RuntimeError("Global breakpoint seeding failed.")
        return best

    def _fit_robust_refine(
        self,
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        seed_flat: ArrayLike,
    ) -> Dict[str, Any]:
        p0 = np.asarray(seed_flat, dtype=float).reshape(-1)
        if p0.size != self._base_p0.size:
            p0 = np.asarray(self._base_p0, dtype=float)
        p0 = np.clip(p0, self._lower, self._upper)

        bw = _auto_blend_width(x_sorted)
        _xmin = float(x_sorted[0]) if x_sorted.size > 0 else 0.0
        _xspan = float(x_sorted[-1] - x_sorted[0]) if x_sorted.size > 1 else 0.0

        def residuals(flat: np.ndarray) -> np.ndarray:
            y_hat = self._predict(
                x_sorted,
                flat,
                blend_width=bw,
                _x_min=_xmin,
                _x_span=_xspan,
            )["y_hat"]
            return np.asarray(y_hat - y_sorted, dtype=float)

        diff_step = None
        nb = _n_boundaries(self.segments)
        if nb > 0 and p0.size >= nb:
            diff_step = np.full(p0.size, 1e-6, dtype=float)
            diff_step[-nb:] = _boundary_ratio_diff_step_from_x(x_sorted)

        jax_mgr = _get_jax_fit_manager_cached()
        ls = jax_mgr.least_squares(
            residuals,
            p0,
            bounds=(self._lower, self._upper),
            periodic_mask=self._periodic_mask,
            periodic_periods=self._periodic_periods,
            periodic_offsets=self._periodic_offsets,
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=int(self.config.robust_max_nfev),
            diff_step=diff_step,
            x_scale=1.0,
        )
        popt = np.asarray(ls.x, dtype=float)
        pred = self._predict(x_sorted, popt, _x_min=_xmin, _x_span=_xspan)
        return {
            "params": popt,
            "y_hat": np.asarray(pred["y_hat"], dtype=float),
            "boundaries": np.asarray(pred["boundaries"], dtype=float),
            "sse": float(_sse_score(y_sorted, pred["y_hat"])),
        }

    def fit(self, x: ArrayLike, y: ArrayLike, seed: int = 0) -> OrderedPiecewiseResult:
        x_sorted, y_sorted = _sort_xy(x, y)
        global_seed = self._fit_global_seed(x_sorted, y_sorted, seed=seed)
        robust = self._fit_robust_refine(x_sorted, y_sorted, global_seed["params"])
        best = robust

        diagnostics: Dict[str, Any] = {
            "global_seed_sse": float(global_seed["sse"]),
            "robust_sse": float(robust["sse"]),
            "final_sse": float(best["sse"]),
            "selected_stage": "robust_only",
            "evaluated_grid_candidates": int(global_seed.get("evaluated", 0)),
            "numba_available": bool(NUMBA_AVAILABLE),
            "jit_enabled": bool(self._use_jit),
            "jax_enabled": True,
        }

        return OrderedPiecewiseResult(
            params=np.asarray(best["params"], dtype=float),
            y_hat=np.asarray(best["y_hat"], dtype=float),
            boundaries=np.asarray(best["boundaries"], dtype=float),
            sse=float(best["sse"]),
            diagnostics=diagnostics,
        )


def refit_ordered_piecewise_from_seed(
    x: ArrayLike,
    y: ArrayLike,
    segments: Sequence[SegmentSpec],
    seed_params: ArrayLike,
    config: Optional[OrderedPiecewiseConfig] = None,
) -> OrderedPiecewiseResult:
    solver = _OrderedPiecewiseSolver(segments=segments, config=config)
    x_sorted, y_sorted = _sort_xy(x, y)

    seed_flat = np.asarray(seed_params, dtype=float).reshape(-1)
    if seed_flat.size != solver._base_p0.size:
        raise ValueError(
            f"seed_params size mismatch: got {seed_flat.size}, expected {solver._base_p0.size}."
        )
    seed_flat = np.clip(seed_flat, solver._lower, solver._upper)

    seed_pred = solver._predict(x_sorted, seed_flat)
    seed_sse = float(_sse_score(y_sorted, seed_pred["y_hat"]))
    robust = solver._fit_robust_refine(x_sorted, y_sorted, seed_flat)

    # Guard: keep the seed if robust refinement produced worse SSE.
    if float(robust["sse"]) <= seed_sse:
        best = robust
        selected_stage = "robust_from_seed"
    else:
        best = {
            "params": seed_flat,
            "y_hat": seed_pred["y_hat"],
            "boundaries": seed_pred["boundaries"],
            "sse": seed_sse,
        }
        selected_stage = "seed_kept (robust worse)"

    diagnostics: Dict[str, Any] = {
        "seed_sse": float(seed_sse),
        "robust_sse": float(robust["sse"]),
        "final_sse": float(best["sse"]),
        "selected_stage": selected_stage,
        "numba_available": bool(NUMBA_AVAILABLE),
        "jit_enabled": bool(solver._use_jit),
        "jax_enabled": True,
    }

    return OrderedPiecewiseResult(
        params=np.asarray(best["params"], dtype=float),
        y_hat=np.asarray(best["y_hat"], dtype=float),
        boundaries=np.asarray(best["boundaries"], dtype=float),
        sse=float(best["sse"]),
        diagnostics=diagnostics,
    )


def predict_ordered_piecewise(
    x: ArrayLike,
    segments: Sequence[SegmentSpec],
    params: ArrayLike,
    prefer_jit: bool = True,
    blend_width: float = 0.0,
    _x_min: Optional[float] = None,
    _x_span: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    return _predict_piecewise(
        segments=segments,
        x=x,
        flat_params=params,
        use_jit=bool(prefer_jit),
        blend_width=float(blend_width),
        _x_min=_x_min,
        _x_span=_x_span,
    )


__all__ = [
    "SegmentSpec",
    "OrderedPiecewiseConfig",
    "OrderedPiecewiseResult",
    "refit_ordered_piecewise_from_seed",
    "predict_ordered_piecewise",
    "pcts_to_boundary_ratios",
    "boundary_ratios_to_pcts",
]
