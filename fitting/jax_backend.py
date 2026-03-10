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
import os
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
    Union,
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


def _fit_notice_use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _fit_notice_sgr(code: str, text: str) -> str:
    if not _fit_notice_use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _fit_notice(
    message: str,
    *,
    command: Optional[str] = None,
    is_error: bool = False,
) -> None:
    """Emit important JAX backend notices even when fit-debug is off."""
    text = f"[jax] {message}"
    if command:
        text = f"{text}\n      Try: {command}"
    if _fit_log.enabled():
        _fit_log.detail(text)
        return
    prefix = "[fit] [jax]"
    if is_error:
        headline = _fit_notice_sgr("1;31", message)
    else:
        headline = message
    if command:
        label = _fit_notice_sgr("1;33", "Try:")
        cmd = _fit_notice_sgr("36", command)
        print(f"{prefix} {headline}\n{prefix}   {label} {cmd}", flush=True)
        return
    print(f"{prefix} {headline}", flush=True)


# ---------------------------------------------------------------------------
# Lazy JAX / jaxfit initialisation  (thread-safe, runs at most once)
# ---------------------------------------------------------------------------
_jax_init_lock = threading.Lock()
_jax_initialized = False
_jax_ok = False
_jax_gpu = False
_jnp: Any = None  # jax.numpy module ref
_CurveFit_cls: Any = None  # jaxfit.CurveFit class ref
_LeastSquares_cls: Any = None  # jaxfit.LeastSquares class ref
_JAX_INIT_ERROR: Optional[str] = None


def _probe_nvidia_uvm() -> Tuple[bool, str]:
    """Check whether NVIDIA UVM device node is usable in this process."""
    if os.name != "posix":
        return True, ""
    uvm_path = "/dev/nvidia-uvm"
    if not os.path.exists(uvm_path):
        return True, f"{uvm_path} not present"
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(uvm_path, flags)
        os.close(fd)
        return True, f"{uvm_path} open ok"
    except OSError as exc:
        return False, f"{uvm_path} open failed: errno={exc.errno} ({exc.strerror})"


def _ensure_jax_init() -> bool:
    """Lazily import JAX + jaxfit.  Returns *True* on success."""
    global _jax_initialized, _jax_ok, _jax_gpu
    global _jnp, _CurveFit_cls, _LeastSquares_cls, _JAX_INIT_ERROR

    if _jax_initialized:
        return _jax_ok

    with _jax_init_lock:
        if _jax_initialized:
            return _jax_ok
        try:
            # Default is "auto" (let JAX choose, typically GPU when available).
            # REDPITAYA_JAX_PLATFORM can force cpu/gpu/tpu when needed.
            platform_pref = str(
                os.environ.get("REDPITAYA_JAX_PLATFORM", "auto")
            ).strip().lower()
            raw_jax_platforms = str(os.environ.get("JAX_PLATFORMS", "")).strip().lower()
            if platform_pref in {"cpu", "gpu", "tpu"}:
                # Explicit override from app-level setting.
                os.environ["JAX_PLATFORMS"] = platform_pref
            else:
                platform_pref = "auto"
                # Avoid inherited shell state forcing CUDA/GPU only.
                # In auto mode we want JAX to choose any available backend.
                if raw_jax_platforms in {"cuda", "gpu"}:
                    os.environ["JAX_PLATFORMS"] = ""
                    _fit_debug(
                        "cleared stale JAX_PLATFORMS override "
                        f"(was '{raw_jax_platforms}') for auto backend selection"
                    )

            _fit_debug(
                "init env: "
                f"platform_pref={platform_pref} "
                f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '')!r} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
            )
            if platform_pref in {"auto", "gpu"}:
                uvm_ok, uvm_status = _probe_nvidia_uvm()
                if uvm_ok:
                    _fit_debug(f"nvidia uvm probe: {uvm_status}")
                else:
                    repair_cmd = "sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
                    hint = (
                        "CUDA device node is unhealthy; try "
                        f"'{repair_cmd}' "
                        "or reboot."
                    )
                    _fit_notice(
                        "gpu init error (auto mode): "
                        f"{uvm_status}. Falling back to CPU.",
                        command=repair_cmd,
                        is_error=True,
                    )
                    if platform_pref == "gpu":
                        raise RuntimeError(
                            "REDPITAYA_JAX_PLATFORM='gpu' requested but "
                            f"{uvm_status}. {hint}"
                        )
                    os.environ["JAX_PLATFORMS"] = "cpu"
                    _fit_debug(
                        "forcing JAX_PLATFORMS='cpu' in auto mode because "
                        "GPU init would fail"
                    )

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
            from jaxfit.least_squares import LeastSquares

            _CurveFit_cls = CurveFit
            _LeastSquares_cls = LeastSquares

            # Detect active backend/devices. This must succeed; otherwise fitting
            # will fail later with harder-to-debug runtime errors.
            devices = jax.devices()
            _jax_gpu = any(d.platform == "gpu" for d in devices)
            if platform_pref == "gpu" and not _jax_gpu:
                raise RuntimeError(
                    "REDPITAYA_JAX_PLATFORM='gpu' requested but no visible GPU devices "
                    f"were found. devices={devices!r}"
                )
            dev_strs = []
            for d in devices:
                kind = getattr(d, "device_kind", "")
                dev_strs.append(
                    f"{d.platform.upper()}" + (f" ({kind})" if kind else "")
                )
            _fit_debug(
                f"init ok: devices=[{', '.join(dev_strs)}] "
                f"gpu={'yes' if _jax_gpu else 'no'} "
                f"platform_pref={platform_pref}"
            )

            _jax_ok = True
        except Exception as exc:
            _JAX_INIT_ERROR = str(exc)
            _jax_ok = False
            if platform_pref == "auto":
                _fit_notice(
                    f"gpu init error (auto mode): {exc}",
                    is_error=True,
                )
            _fit_debug(f"init FAILED: {exc}")

        _jax_initialized = True
        return _jax_ok


# ---------------------------------------------------------------------------
# Public availability helpers
# ---------------------------------------------------------------------------
def jax_available() -> bool:
    """Return *True* if JAX and jaxfit are importable."""
    return _ensure_jax_init()


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
                    inst._least_squares = None
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

    def _get_ls(self):
        if self._least_squares is None:
            with self._cache_lock:
                if self._least_squares is None:
                    self._least_squares = _LeastSquares_cls()
        return self._least_squares

    @staticmethod
    def _as_param_array(values, n: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=float)
        out = np.asarray(arr, dtype=float).reshape(-1)
        if out.size != int(n):
            raise ValueError(f"{name} must be scalar or length {n}.")
        return out

    @staticmethod
    def _parse_periodic_mask(periodic, n: int) -> np.ndarray:
        if periodic is None:
            return np.zeros(n, dtype=bool)
        arr = np.asarray(periodic)
        if arr.ndim == 0:
            return np.full(n, bool(arr), dtype=bool)
        mask = np.asarray(arr, dtype=bool).reshape(-1)
        if mask.size != int(n):
            raise ValueError(f"`periodic_mask` must be scalar or length {n}.")
        return mask

    @classmethod
    def _parse_periodic_values(
        cls,
        values,
        n: int,
        name: str,
        default: float,
    ) -> np.ndarray:
        if values is None:
            return np.full(n, float(default), dtype=float)
        return cls._as_param_array(values, n, name)

    @staticmethod
    def _wrap_periodic_numpy(
        params: np.ndarray,
        periodic_mask: np.ndarray,
        periodic_periods: np.ndarray,
        periodic_offsets: np.ndarray,
    ) -> np.ndarray:
        out = np.asarray(params, dtype=float).reshape(-1).copy()
        if out.size == 0 or not np.any(periodic_mask):
            return out
        mask = np.asarray(periodic_mask, dtype=bool).reshape(-1)
        out[mask] = periodic_offsets[mask] + np.mod(
            out[mask] - periodic_offsets[mask],
            periodic_periods[mask],
        )
        return out

    @classmethod
    def _prepare_periodic_least_squares_config(
        cls,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        periodic_mask,
        periodic_periods,
        periodic_offsets,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(x0.size)
        mask = cls._parse_periodic_mask(periodic_mask, n)
        if not np.any(mask):
            return (
                mask,
                np.ones(n, dtype=float),
                np.zeros(n, dtype=float),
                np.asarray(lb, dtype=float).copy(),
                np.asarray(ub, dtype=float).copy(),
                np.asarray(x0, dtype=float).copy(),
            )

        if periodic_periods is None:
            periods = np.ones(n, dtype=float)
            lb_arr = np.asarray(lb, dtype=float)
            ub_arr = np.asarray(ub, dtype=float)
            spans = ub_arr - lb_arr
            valid = mask & np.isfinite(lb_arr) & np.isfinite(ub_arr) & (spans > 0.0)
            if not np.all(valid[mask]):
                raise ValueError(
                    "Periodic least-squares parameters require finite bounds "
                    "with non-zero span when `periodic_periods` is omitted."
                )
            periods[mask] = spans[mask]
        else:
            periods = cls._parse_periodic_values(
                periodic_periods,
                n,
                "`periodic_periods`",
                1.0,
            )
            invalid = mask & (~np.isfinite(periods) | (periods <= 0.0))
            if np.any(invalid):
                raise ValueError(
                    "`periodic_periods` must be finite and > 0 for periodic parameters."
                )
            periods = np.where(mask, periods, 1.0)

        if periodic_offsets is None:
            offsets = np.zeros(n, dtype=float)
            lb_arr = np.asarray(lb, dtype=float)
            finite_lb = mask & np.isfinite(lb_arr)
            offsets[finite_lb] = lb_arr[finite_lb]
        else:
            offsets = cls._parse_periodic_values(
                periodic_offsets,
                n,
                "`periodic_offsets`",
                0.0,
            )
            invalid_offsets = mask & ~np.isfinite(offsets)
            if np.any(invalid_offsets):
                raise ValueError(
                    "`periodic_offsets` must be finite for periodic parameters."
                )
            offsets = np.where(mask, offsets, 0.0)

        wrapped_x0 = cls._wrap_periodic_numpy(
            np.asarray(x0, dtype=float),
            mask,
            periods,
            offsets,
        )
        fit_lb = np.asarray(lb, dtype=float).copy()
        fit_ub = np.asarray(ub, dtype=float).copy()
        fit_lb[mask] = -np.inf
        fit_ub[mask] = np.inf
        return mask, periods, offsets, fit_lb, fit_ub, wrapped_x0

    @classmethod
    def _build_fd_jacobian(
        cls,
        residual_func: Callable[[np.ndarray], np.ndarray],
        lb: np.ndarray,
        ub: np.ndarray,
        diff_step,
    ) -> Callable[[np.ndarray], np.ndarray]:
        lb_arr = np.asarray(lb, dtype=float).reshape(-1)
        ub_arr = np.asarray(ub, dtype=float).reshape(-1)

        def jac(xvals: np.ndarray) -> np.ndarray:
            x = np.asarray(xvals, dtype=float).reshape(-1)
            n = int(x.size)
            steps = cls._as_param_array(
                np.sqrt(np.finfo(float).eps) if diff_step is None else diff_step,
                n,
                "`diff_step`",
            )
            f0 = np.asarray(residual_func(x), dtype=float).reshape(-1)
            m = int(f0.size)
            jacobian = np.zeros((m, n), dtype=float)
            for col in range(n):
                base = float(abs(steps[col]))
                h = float(base * max(1.0, abs(x[col])))
                if not np.isfinite(h) or h <= 0.0:
                    h = float(np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[col])))
                x_f = np.asarray(x, dtype=float).copy()
                x_b = np.asarray(x, dtype=float).copy()
                forward = x[col] + h
                backward = x[col] - h
                if np.isfinite(ub_arr[col]):
                    forward = min(forward, ub_arr[col])
                if np.isfinite(lb_arr[col]):
                    backward = max(backward, lb_arr[col])
                x_f[col] = forward
                x_b[col] = backward
                # Use exact step-size checks here. Relative-tolerance based
                # comparisons (np.isclose defaults) can incorrectly treat valid
                # finite-difference perturbations as "equal" and zero-out a
                # Jacobian column, freezing optimization variables.
                if float(x_f[col] - x_b[col]) == 0.0:
                    continue
                if float(x_b[col] - x[col]) == 0.0:
                    f_f = np.asarray(residual_func(x_f), dtype=float).reshape(-1)
                    denom = float(x_f[col] - x[col])
                    if denom != 0.0:
                        jacobian[:, col] = (f_f - f0) / denom
                    continue
                if float(x_f[col] - x[col]) == 0.0:
                    f_b = np.asarray(residual_func(x_b), dtype=float).reshape(-1)
                    denom = float(x[col] - x_b[col])
                    if denom != 0.0:
                        jacobian[:, col] = (f0 - f_b) / denom
                    continue
                f_f = np.asarray(residual_func(x_f), dtype=float).reshape(-1)
                f_b = np.asarray(residual_func(x_b), dtype=float).reshape(-1)
                jacobian[:, col] = (f_f - f_b) / float(x_f[col] - x_b[col])
            return jacobian

        return jac

    # -- public API --------------------------------------------------------

    def curve_fit(
        self,
        model_func: Callable,
        x_data: np.ndarray,
        y_data: np.ndarray,
        p0: np.ndarray,
        bounds: Tuple = (-np.inf, np.inf),
        periodic_mask: Optional[np.ndarray] = None,
        periodic_periods: Optional[np.ndarray] = None,
        periodic_offsets: Optional[np.ndarray] = None,
        max_nfev: Optional[int] = None,
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
        periodic_mask : ndarray, optional
            Boolean mask of periodic parameters.
        periodic_periods : ndarray, optional
            Per-parameter wrap periods.
        periodic_offsets : ndarray, optional
            Per-parameter wrap offsets.
        max_nfev : int, optional
            Maximum function evaluations for local TRF solve.

        Returns
        -------
        popt, pcov : ndarray, ndarray
        """
        n = int(x_data.shape[-1] if x_data.ndim > 1 else x_data.size)
        cf, flength, fid, was_cached = self._get_cf(n, model_func)
        t0 = time.perf_counter()
        fit_kwargs: Dict[str, Any] = {}
        if max_nfev is not None:
            fit_kwargs["max_nfev"] = int(max_nfev)
        popt, pcov = cf.curve_fit(
            model_func,
            np.asarray(x_data, dtype=np.float64),
            np.asarray(y_data, dtype=np.float64),
            p0=np.asarray(p0, dtype=np.float64),
            bounds=bounds,
            periodic=periodic_mask,
            periodic_periods=periodic_periods,
            periodic_offsets=periodic_offsets,
            **fit_kwargs,
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

    def least_squares(
        self,
        residual_func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: Tuple = (-np.inf, np.inf),
        periodic_mask: Optional[np.ndarray] = None,
        periodic_periods: Optional[np.ndarray] = None,
        periodic_offsets: Optional[np.ndarray] = None,
        loss: str = "linear",
        f_scale: float = 1.0,
        max_nfev: Optional[int] = None,
        diff_step=None,
        x_scale: Union[str, np.ndarray, float] = 1.0,
    ):
        """Run jaxfit LeastSquares with finite-difference Jacobian.

        Periodic parameters are optimized in an unbounded variable and wrapped
        into one period before each residual/Jacobian evaluation.
        """
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        n = int(x0_arr.size)
        lb_arr = self._as_param_array(bounds[0], n, "lower bounds")
        ub_arr = self._as_param_array(bounds[1], n, "upper bounds")
        (
            periodic_mask_arr,
            periodic_period_arr,
            periodic_offset_arr,
            fit_lb_arr,
            fit_ub_arr,
            fit_x0_arr,
        ) = self._prepare_periodic_least_squares_config(
            x0_arr,
            lb_arr,
            ub_arr,
            periodic_mask,
            periodic_periods,
            periodic_offsets,
        )

        wrapped_residual = residual_func
        if np.any(periodic_mask_arr):

            def _wrapped_residual(xvals: np.ndarray) -> np.ndarray:
                wrapped = self._wrap_periodic_numpy(
                    np.asarray(xvals, dtype=float),
                    periodic_mask_arr,
                    periodic_period_arr,
                    periodic_offset_arr,
                )
                return residual_func(wrapped)

            wrapped_residual = _wrapped_residual

        jac_func = self._build_fd_jacobian(
            wrapped_residual,
            fit_lb_arr,
            fit_ub_arr,
            diff_step,
        )
        ls = self._get_ls()
        t0 = time.perf_counter()
        result = ls.least_squares(
            wrapped_residual,
            fit_x0_arr,
            jac=jac_func,
            bounds=(fit_lb_arr, fit_ub_arr),
            method="trf",
            loss=str(loss),
            f_scale=float(f_scale),
            max_nfev=(int(max_nfev) if max_nfev is not None else None),
            x_scale=x_scale,
        )
        if np.any(periodic_mask_arr):
            result.x = self._wrap_periodic_numpy(
                np.asarray(result.x, dtype=float),
                periodic_mask_arr,
                periodic_period_arr,
                periodic_offset_arr,
            )
        elapsed = time.perf_counter() - t0
        _fit_debug(
            f"least_squares  n={n}  m={np.asarray(result.fun).size}  "
            f"nfev={getattr(result, 'nfev', '?')}  "
            f"periodic={'yes' if np.any(periodic_mask_arr) else 'no'}  "
            f"elapsed={elapsed:.4f}s"
        )
        return result


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
