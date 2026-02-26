"""JAX/jaxfit GPU-accelerated fitting backend.

Best practices (https://jaxfit.readthedocs.io/en/latest/):
- Double precision required: ``jax_enable_x64 = True`` before importing JAX
- Reuse ``CurveFit`` instances with ``flength`` to avoid JIT retracing
- Model functions must be pure and use ``jax.numpy`` (jnp)
- Let jaxfit compute Jacobians via auto-differentiation (faster than numeric)
- Round ``flength`` up to a power of 2 to minimise the cache size

Usage
-----
>>> from jax_backend import jax_available, get_jax_fit_manager, build_jax_model_func
"""

from __future__ import annotations

import ast
import math
import pathlib
import sys
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

# Re-use existing expression helpers
from expression import (
    _EXPRESSION_ALLOWED_CONSTANTS,
)
from solver import SegmentSpec


# ---------------------------------------------------------------------------
# Local fit-debug helper
# ---------------------------------------------------------------------------
import fit_log as _fit_log


def _fit_debug(message: str) -> None:
    _fit_log.detail(f"[jax] {message}")


# ---------------------------------------------------------------------------
# Lazy JAX / jaxfit initialisation  (thread-safe, runs at most once)
# ---------------------------------------------------------------------------
_jax_init_lock = threading.Lock()
_jax_initialized = False
_jax_ok = False
_jax_gpu = False
_jnp: Any = None  # jax.numpy module ref
_CurveFit_cls: Any = None  # jaxfit.CurveFit class ref
_JAX_INIT_ERROR: Optional[str] = None


def _ensure_jax_init() -> bool:
    """Lazily import JAX + jaxfit.  Returns *True* on success."""
    global _jax_initialized, _jax_ok, _jax_gpu
    global _jnp, _CurveFit_cls, _JAX_INIT_ERROR

    if _jax_initialized:
        return _jax_ok

    with _jax_init_lock:
        if _jax_initialized:
            return _jax_ok
        try:
            # x64 MUST be enabled before any JAX computation (jaxfit requirement)
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp_mod

            _jnp = jnp_mod

            # Add the vendored jaxfit package to sys.path
            _jaxfit_dir = str(pathlib.Path(__file__).resolve().parent / "jaxfit")
            if _jaxfit_dir not in sys.path:
                sys.path.insert(0, _jaxfit_dir)

            from jaxfit import CurveFit

            _CurveFit_cls = CurveFit

            # Detect GPU / TPU
            try:
                devices = jax.devices()
                _jax_gpu = any(d.platform == "gpu" for d in devices)
                dev_strs = []
                for d in devices:
                    kind = getattr(d, "device_kind", "")
                    dev_strs.append(
                        f"{d.platform.upper()}" + (f" ({kind})" if kind else "")
                    )
                _fit_debug(
                    f"init ok: devices=[{', '.join(dev_strs)}] "
                    f"gpu={'yes' if _jax_gpu else 'no'}"
                )
            except Exception:
                _jax_gpu = False
                _fit_debug("init ok: device detection failed")

            _jax_ok = True
        except Exception as exc:
            _JAX_INIT_ERROR = str(exc)
            _jax_ok = False
            _fit_debug(f"init FAILED: {exc}")

        _jax_initialized = True
        return _jax_ok


# ---------------------------------------------------------------------------
# Public availability helpers
# ---------------------------------------------------------------------------
def jax_available() -> bool:
    """Return *True* if JAX and jaxfit are importable."""
    return _ensure_jax_init()


def backend_tag(use_jax: bool = False) -> str:
    """Return a short backend label for log headers.

    Returns one of:
    - ``"NON-JIT"`` — JAX not requested or not available
    - GPU device name (e.g. ``"NVIDIA RTX 4090"``) when a GPU is detected
    - ``"JIT"`` — JAX available on CPU only
    """
    if not use_jax or not _ensure_jax_init():
        return "NON-JIT"
    if _jax_gpu:
        try:
            import jax

            for d in jax.devices():
                if d.platform == "gpu":
                    kind = getattr(d, "device_kind", "")
                    return kind if kind else "GPU"
        except Exception:
            return "GPU"
    return "JIT"


# ---------------------------------------------------------------------------
# JAX-compatible math function mapping
# ---------------------------------------------------------------------------
def _get_jax_allowed_functions() -> Dict[str, Any]:
    """Build mapping of allowed math functions using jax.numpy."""
    if _jnp is None:
        raise RuntimeError("JAX not initialised — call jax_available() first")
    return {
        "abs": _jnp.abs,
        "sin": _jnp.sin,
        "cos": _jnp.cos,
        "tan": _jnp.tan,
        "arcsin": _jnp.arcsin,
        "arccos": _jnp.arccos,
        "arctan": _jnp.arctan,
        "sinh": _jnp.sinh,
        "cosh": _jnp.cosh,
        "tanh": _jnp.tanh,
        "exp": _jnp.exp,
        "log": _jnp.log,
        "log10": _jnp.log10,
        "sqrt": _jnp.sqrt,
        "power": _jnp.power,
        "minimum": _jnp.minimum,
        "maximum": _jnp.maximum,
        "clip": _jnp.clip,
    }


# ---------------------------------------------------------------------------
# Expression compiler — JAX edition
# ---------------------------------------------------------------------------
def compile_segment_expression_jax(
    expression_text: str,
    parameter_names: Sequence[str],
) -> Callable:
    """Compile a segment expression into a JAX-traceable evaluator.

    Returns ``callable(x_data, param_values_dict) -> jnp.ndarray``.
    The result is a pure function suitable for auto-differentiation.
    """
    if not _ensure_jax_init():
        raise RuntimeError("JAX not available")

    text = str(expression_text).strip()
    if not text:
        raise ValueError("Segment expression is empty.")
    ordered_names = list(parameter_names)

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid segment expression: {exc.msg}") from exc

    code = compile(tree, "<jax_segment>", "eval")
    jax_funcs = _get_jax_allowed_functions()

    eval_globals: Dict[str, Any] = {
        "__builtins__": {},
        "np": _jnp,  # redirect np -> jnp inside expressions
        "math": math,
    }
    eval_globals.update(jax_funcs)
    eval_globals.update(_EXPRESSION_ALLOWED_CONSTANTS)

    def _evaluate(
        x_data,
        param_values: Mapping[str, float],
    ):
        eval_locals: Dict[str, Any] = {"x": x_data}
        for name in ordered_names:
            if name not in param_values:
                raise ValueError(f"Missing parameter '{name}'.")
            eval_locals[name] = param_values[name]
        result = eval(code, eval_globals, eval_locals)
        out = _jnp.asarray(result)
        if out.shape == ():
            return _jnp.broadcast_to(out, x_data.shape)
        return out.reshape(-1)

    return _evaluate


# ---------------------------------------------------------------------------
# Build a JAX-compatible model function from an expression
# ---------------------------------------------------------------------------
def build_jax_model_func(
    expression_text: str,
    parameter_names: Sequence[str],
    fixed_params: Optional[Mapping[str, float]] = None,
) -> Tuple[Callable, List[str]]:
    """Return ``(model_func, free_param_names)`` ready for jaxfit CurveFit.

    ``model_func(x, *free_values)`` uses jax.numpy and is auto-differentiable.
    """
    evaluator = compile_segment_expression_jax(expression_text, parameter_names)
    fixed_map = dict(fixed_params or {})
    free_names = [n for n in parameter_names if n not in fixed_map]

    def model_func(x, *free_values):
        param_dict: Dict[str, Any] = dict(fixed_map)
        for name, val in zip(free_names, free_values):
            param_dict[name] = val
        return evaluator(x, param_dict)

    # Attach a content-based identity so JaxFitManager can cache one
    # CurveFit per *distinct mathematical function*, not per Python object.
    # jaxfit detects function changes via bytecode comparison, but every
    # closure produced by this factory has identical bytecode.  Without a
    # per-function CurveFit, switching between segments silently reuses a
    # stale JIT trace → wrong results.
    model_func._jax_func_id = (
        str(expression_text),
        tuple(free_names),
        tuple(sorted(fixed_map.items())),
    )

    return model_func, free_names


# ---------------------------------------------------------------------------
# CurveFit Manager — singleton with flength-based caching
# ---------------------------------------------------------------------------
def _next_power_of_2(n: int) -> int:
    """Round *n* up to the next power of 2 (minimum 64)."""
    n = max(64, int(n))
    return 1 << (n - 1).bit_length()


class JaxFitManager:
    """Singleton that caches ``CurveFit`` instances keyed by *(flength, func_id)*.

    **Why per-function caching is required:**
    jaxfit's ``LeastSquares`` detects function changes by comparing bytecode
    (``__code__.co_code``).  All closures produced by ``build_jax_model_func``
    share the *same* bytecode because they come from the same factory, so
    jaxfit never detects the switch and silently reuses a stale JIT trace.
    Giving each distinct mathematical function its own ``CurveFit`` instance
    eliminates the problem entirely — the function never "changes" within
    a single instance.

    **Why flength caching matters** (from the jaxfit docs):
    Creating a new ``CurveFit`` or changing ``flength`` triggers expensive
    JIT retracing of all internal XLA kernels.  Re-using an instance with
    the same ``flength`` skips retracing entirely and keeps the fit running
    on the GPU hot-path.

    The ``flength`` is rounded to the next power of 2, so a handful of
    cached instances cover a wide range of data sizes.
    """

    _instance: Optional[JaxFitManager] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    # keyed by (flength, func_id_hash)
                    inst._cache: Dict[Tuple, Any] = {}
                    inst._cache_lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _func_id(model_func: Callable) -> int:
        """Return a deterministic hash for *model_func*.

        Functions produced by ``build_jax_model_func`` carry a
        ``_jax_func_id`` tuple ``(expr, free_names, fixed_items)``.
        For arbitrary callables we fall back to ``id()``.
        """
        raw = getattr(model_func, "_jax_func_id", None)
        if raw is not None:
            return hash(raw)
        return id(model_func)

    def _get_cf(
        self,
        data_length: int,
        model_func: Callable,
    ) -> Tuple[Any, int, int, bool]:
        """Return ``(CurveFit, flength, func_id, was_cached)``."""
        flength = _next_power_of_2(data_length)
        fid = self._func_id(model_func)
        cache_key = (flength, fid)
        with self._cache_lock:
            if cache_key not in self._cache:
                t0 = time.perf_counter()
                self._cache[cache_key] = _CurveFit_cls(flength=flength)
                _fit_debug(
                    f"CurveFit COLD  flength={flength} "
                    f"func_id={fid}  "
                    f"(data_len={data_length})  "
                    f"create={time.perf_counter() - t0:.4f}s  "
                    f"cached_total={len(self._cache)}"
                )
                return self._cache[cache_key], flength, fid, False
            return self._cache[cache_key], flength, fid, True

    # -- public API --------------------------------------------------------

    def curve_fit(
        self,
        model_func: Callable,
        x_data: np.ndarray,
        y_data: np.ndarray,
        p0: np.ndarray,
        bounds: Tuple = (-np.inf, np.inf),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ``CurveFit.curve_fit`` with automatic flength caching.

        Parameters
        ----------
        model_func : callable
            Pure function ``f(x, *params)`` using jax.numpy operations.
        x_data, y_data : ndarray
            Fit data (will be cast to float64).
        p0 : ndarray
            Initial parameter guesses.
        bounds : tuple
            ``(lower, upper)`` parameter bounds.

        Returns
        -------
        popt, pcov : ndarray, ndarray
        """
        n = int(x_data.shape[-1] if x_data.ndim > 1 else x_data.size)
        cf, flength, fid, was_cached = self._get_cf(n, model_func)
        t0 = time.perf_counter()
        popt, pcov = cf.curve_fit(
            model_func,
            np.asarray(x_data, dtype=np.float64),
            np.asarray(y_data, dtype=np.float64),
            p0=np.asarray(p0, dtype=np.float64),
            bounds=bounds,
        )
        elapsed = time.perf_counter() - t0
        _fit_debug(
            f"curve_fit  n={n}  flength={flength}  "
            f"func_id={fid}  "
            f"{'WARM' if was_cached else 'COLD'}  "
            f"elapsed={elapsed:.4f}s  "
            f"device={'GPU' if _jax_gpu else 'CPU'}"
        )
        return np.asarray(popt), np.asarray(pcov)


def get_jax_fit_manager() -> JaxFitManager:
    """Return the singleton ``JaxFitManager`` (creates JAX on first call)."""
    if not _ensure_jax_init():
        raise RuntimeError("JAX/jaxfit not available")
    return JaxFitManager()


# ---------------------------------------------------------------------------
# Make JAX-compatible SegmentSpec objects
# ---------------------------------------------------------------------------
def make_jax_segment_specs(
    segment_exprs: Sequence[str],
    segment_param_names: Sequence[Sequence[str]],
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    fixed_param_values: Optional[Mapping[str, float]] = None,
) -> Tuple[List[SegmentSpec], List[Callable], List[List[str]]]:
    """Build ``SegmentSpec`` objects whose *model_func*s are JAX-traceable.

    Returns
    -------
    specs : list of SegmentSpec
    jax_funcs : list of model callables ``f(x, *free_vals)``
    free_names_per_seg : list of list of free parameter names
    """
    fixed_map = {
        str(k): float(v)
        for k, v in dict(fixed_param_values or {}).items()
        if str(k).strip()
    }
    specs: List[SegmentSpec] = []
    jax_funcs: List[Callable] = []
    free_names_list: List[List[str]] = []

    t0 = time.perf_counter()
    for seg_names, expr in zip(segment_param_names, segment_exprs):
        func, free_names = build_jax_model_func(expr, seg_names, fixed_map)
        jax_funcs.append(func)
        free_names_list.append(free_names)

        lo, hi, p0_vals = [], [], []
        for name in free_names:
            low, high = bounds_map[name]
            if low > high:
                low, high = high, low
            lo.append(float(low))
            hi.append(float(high))
            p0_vals.append(float(np.clip(seed_map[name], low, high)))

        specs.append(
            SegmentSpec(
                model_func=func,
                p0=p0_vals,
                bounds=(lo, hi),
            )
        )

    _fit_debug(
        "make_jax_segment_specs: "
        f"segments={len(specs)} "
        f"free_params={[len(fn) for fn in free_names_list]} "
        f"device={'GPU' if _jax_gpu else 'CPU'} "
        f"elapsed={time.perf_counter() - t0:.4f}s"
    )
    return specs, jax_funcs, free_names_list
