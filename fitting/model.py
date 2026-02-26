"""Parameter specs, model definitions, fitting pipeline, and helper metrics."""

import ast
import math
import time
from dataclasses import dataclass
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
from scipy.optimize import least_squares

from fit_results import fit_get

from expression import (
    _ExpressionParameterCollector,
    _PARAMETER_NAME_RE,
    _EXPRESSION_HELPER_NAMES,
    _EXPRESSION_ALLOWED_FUNCTIONS,
    _EXPRESSION_ALLOWED_CONSTANTS,
)
from solver import (
    OrderedPiecewiseConfig,
    OrderedPiecewiseResult,
    SegmentSpec,
    predict_ordered_piecewise,
    refit_ordered_piecewise_from_seed,
    pcts_to_boundary_ratios,
    _boundary_ratio_diff_step_from_x,
    _auto_blend_width,
)


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    symbol: str
    description: str
    default: float
    min_value: float  # Lower bound for parameter (see TODO: Tidy up parameter min/max in model)
    max_value: float  # Upper bound for parameter (see TODO: Tidy up parameter min/max in model)
    decimals: int = 4

    @property
    def inferred_step(self) -> float:
        """
        Infer a sensible spinbox step from parameter span.
        Ensures that the step is always positive and respects the parameter's precision.
        """
        precision_step = 10 ** (-self.decimals)
        span = abs(self.max_value - self.min_value)
        if span <= 0.0:
            return precision_step
        span_order = int(np.floor(np.log10(span)))
        span_step = 10 ** (span_order - 4)
        return max(precision_step, span_step)

    def bounds(self) -> Tuple[float, float]:
        """
        Return the (min, max) bounds for this parameter.
        Reference: TODO - Tidy up parameter min/max in model
        """
        return (self.min_value, self.max_value)

    @property
    def column_name(self) -> str:
        return self.key


DEFAULT_WINDOW_TITLE = "Curve Fitting"
FIT_DETAILS_FILENAME = "fit_details.json"
DEFAULT_TARGET_CHANNEL = "CH2"
DEFAULT_EXPRESSION = (
    "m*x+c ; abs(A_MI*sin(A_mod*sin(2*pi*f_mod*x+pi*phi_mod)+pi*phi_MI))**2+V_0 ; m*x+c"
)
DEFAULT_PARAM_SPECS = (
    # Parameter bounds are now consistently defined (see TODO: Tidy up parameter min/max in model)
    ParameterSpec(
        key="m",
        symbol="m",
        description="Linear slope",
        default=0.0,
        min_value=-2.0,
        max_value=2.0,
    ),
    ParameterSpec(
        key="c",
        symbol="c",
        description="Linear offset",
        default=4.0,
        min_value=0.0,
        max_value=10.0,
    ),
    ParameterSpec(
        key="A_MI",
        symbol="A_{MI}",
        description="Outer sinusoid amplitude",
        default=2.0,
        min_value=0.0,
        max_value=10.0,
    ),
    ParameterSpec(
        key="A_mod",
        symbol="A_{mod}",
        description="Inner modulation amplitude",
        default=2.0,
        min_value=0.0,
        max_value=10.0,
    ),
    ParameterSpec(
        key="f_mod",
        symbol="f_{mod}",
        description="Modulation frequency",
        default=100.0,
        min_value=0.0,
        max_value=1e6,
    ),
    ParameterSpec(
        key="phi_mod",
        symbol="\\phi_{mod}",
        description="Modulation phase",
        default=0.0,
        min_value=-4,
        max_value=4,
    ),
    ParameterSpec(
        key="phi_MI",
        symbol="\\phi_{MI}",
        description="Outer phase offset",
        default=0.0,
        min_value=-4,
        max_value=4,
    ),
    ParameterSpec(
        key="V_0",
        symbol="V_{0}",
        description="Vertical offset",
        default=0.0,
        min_value=0.0,
        max_value=10.0,
    ),
)
FIT_CURVE_COLOR = "#16a34a"
CHANNEL_PLOT_COLORS = (
    "#1e293b",  # dark grey / near-black
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#f59e0b",  # amber
    "#7c3aed",  # violet
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#be123c",  # rose
    "#64748b",  # slate
)
# Companion colours for fit curves — similar hue family, visually distinct.
FIT_COMPANION_COLORS = (
    "#94a3b8",  # light grey  (companion to dark grey)
    "#7c3aed",  # violet      (companion to blue)
    "#f472b6",  # pink        (companion to red)
    "#0d9488",  # teal        (companion to green)
    "#d97706",  # dark amber  (companion to amber)
    "#c084fc",  # light purple(companion to violet)
    "#fb923c",  # light orange(companion to orange)
    "#2563eb",  # blue        (companion to cyan)
    "#fb7185",  # light rose  (companion to rose)
    "#1e293b",  # dark slate  (companion to slate)
)


def fit_companion_color(index: int) -> str:
    """Return the fit-curve companion colour for the given palette index."""
    idx = int(index)
    if FIT_COMPANION_COLORS:
        return FIT_COMPANION_COLORS[idx % len(FIT_COMPANION_COLORS)]
    return f"C{(idx + 1) % 10}"


def palette_color(index: int) -> str:
    idx = int(index)
    if CHANNEL_PLOT_COLORS:
        return CHANNEL_PLOT_COLORS[idx % len(CHANNEL_PLOT_COLORS)]
    return f"C{idx % 10}"


import fit_log as _fit_log  # noqa: E402


def fit_debug(message: str) -> None:
    _fit_log.raw(message)


def compute_r2(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if np.count_nonzero(valid) < 2:
        return None
    yt = y_true_arr[valid]
    yp = y_pred_arr[valid]
    diff = yt - yp
    ss_res = np.dot(diff, diff)
    mean_diff = yt - np.mean(yt)
    ss_tot = np.dot(mean_diff, mean_diff)
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else 0.0
    return float(1.0 - ss_res / ss_tot)


def smooth_channel_array(values, window_size):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    try:
        window = int(window_size)
    except Exception:
        window = 1
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    if window > arr.size:
        window = arr.size if arr.size % 2 == 1 else max(1, arr.size - 1)
    if window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.asarray(smoothed, dtype=float)


@dataclass(frozen=True)
class PiecewiseModelDefinition:
    target_col: str
    segment_exprs: Tuple[str, ...]
    segment_param_names: Tuple[Tuple[str, ...], ...]
    segment_evaluators: Tuple[Callable[..., np.ndarray], ...]
    global_param_names: Tuple[str, ...]


@dataclass(frozen=True)
class MultiChannelModelDefinition:
    """Multi-channel model with shared parameters across channel equations."""

    channel_models: Tuple[PiecewiseModelDefinition, ...]
    global_param_names: Tuple[str, ...]
    boundary_links: Tuple[Tuple[Tuple[str, int], ...], ...] = ()
    # boundary_links: tuple of link groups.  Each group is a tuple of
    # (target_col, boundary_index) pairs that share the same ratio value
    # during optimisation.  Only groups with >=2 members are meaningful.

    @property
    def primary(self) -> PiecewiseModelDefinition:
        return self.channel_models[0]

    @property
    def target_channels(self) -> Tuple[str, ...]:
        return tuple(m.target_col for m in self.channel_models)

    @property
    def is_multi_channel(self) -> bool:
        return len(self.channel_models) > 1

    @property
    def all_boundary_ids(self) -> Tuple[Tuple[str, int], ...]:
        """All (target_col, boundary_index) pairs across channels."""
        ids: list = []
        for m in self.channel_models:
            for i in range(max(0, len(m.segment_exprs) - 1)):
                ids.append((m.target_col, i))
        return tuple(ids)


class FitCancelledError(RuntimeError):
    """Internal exception used to abort fitting when cancellation is requested."""


def has_nonempty_values(values) -> bool:
    if values is None:
        return False
    if isinstance(values, np.ndarray):
        return bool(values.size > 0)
    try:
        return bool(len(values) > 0)
    except Exception:
        return bool(values)


def finite_float_or_none(value) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _row_has_error(row: Mapping[str, Any]) -> bool:
    pattern_error = str(row.get("pattern_error") or "").strip()
    fit_error = str(fit_get(row, "error") or "").strip()
    normalized_fit_error = fit_error.lower().replace(".", "").strip()
    if normalized_fit_error in {"cancelled", "canceled"}:
        fit_error = ""
    return bool(pattern_error or fit_error)


def default_boundary_ratios(n_boundaries: int) -> np.ndarray:
    n = int(max(0, n_boundaries))
    if n <= 0:
        return np.asarray([], dtype=float)
    pcts = np.linspace(1.0 / (n + 1), n / (n + 1), n)
    return pcts_to_boundary_ratios(pcts)


def boundary_ratios_to_positions(
    ratios: Sequence[float], n_boundaries: int
) -> np.ndarray:
    n = int(max(0, n_boundaries))
    if n <= 0:
        return np.asarray([], dtype=float)
    ratio_arr = np.clip(np.asarray(ratios, dtype=float).reshape(-1), 0.0, 1.0)
    if ratio_arr.size != n:
        ratio_arr = default_boundary_ratios(n)
    positions = np.empty(n, dtype=float)
    prev = 0.0
    for idx, ratio in enumerate(ratio_arr):
        current = prev + (1.0 - prev) * float(ratio)
        if idx > 0:
            current = max(current, positions[idx - 1])
        current = float(np.clip(current, 0.0, 1.0))
        positions[idx] = current
        prev = current
    return positions


def boundary_ratios_to_x_values(
    ratios: Sequence[float], x_values: Sequence[float], n_boundaries: int
) -> np.ndarray:
    n = int(max(0, n_boundaries))
    if n <= 0:
        return np.asarray([], dtype=float)
    x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    finite = x_arr[np.isfinite(x_arr)]
    if finite.size == 0:
        return np.asarray([], dtype=float)
    x_min = float(np.min(finite))
    x_max = float(np.max(finite))
    if np.isclose(x_min, x_max):
        x_max = x_min + 1.0
    positions = boundary_ratios_to_positions(ratios, n)
    span = float(x_max - x_min)
    return np.asarray(x_min + span * positions, dtype=float)


def extract_segment_parameter_names(
    expression_text: str,
    reserved_names: Optional[Sequence[str]] = None,
) -> List[str]:
    text = str(expression_text).strip()
    if not text:
        raise ValueError("Segment expression is empty.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid segment expression: {exc.msg}") from exc

    reserved = set(reserved_names or ())
    reserved |= {"np", "x"} | _EXPRESSION_HELPER_NAMES
    collector = _ExpressionParameterCollector(reserved_names=reserved)
    collector.visit(tree)

    for name in collector.names:
        if not _PARAMETER_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid parameter name '{name}' in segment expression.")
    return collector.names


def compile_segment_expression(
    expression_text: str,
    parameter_names: Sequence[str],
) -> Callable[[np.ndarray, Mapping[str, float]], np.ndarray]:
    text = str(expression_text).strip()
    if not text:
        raise ValueError("Segment expression is empty.")
    ordered_names = list(parameter_names)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid segment expression: {exc.msg}") from exc

    code = compile(tree, "<piecewise_segment>", "eval")
    eval_globals: Dict[str, Any] = {
        "__builtins__": {},
        "np": np,
        "math": math,
    }
    eval_globals.update(_EXPRESSION_ALLOWED_FUNCTIONS)
    eval_globals.update(_EXPRESSION_ALLOWED_CONSTANTS)

    def _evaluate(x_data: np.ndarray, param_values: Mapping[str, float]) -> np.ndarray:
        x_arr = np.asarray(x_data, dtype=float).reshape(-1)
        eval_locals: Dict[str, Any] = {"x": x_arr}
        for name in ordered_names:
            if name not in param_values:
                raise ValueError(f"Missing parameter '{name}'.")
            eval_locals[name] = float(param_values[name])
        try:
            result = eval(code, eval_globals, eval_locals)
        except Exception as exc:
            raise ValueError(f"Segment evaluation failed: {exc}") from exc
        out = np.asarray(result, dtype=float)
        if out.shape == ():
            return np.full_like(x_arr, float(out), dtype=float)
        out = out.reshape(-1)
        if out.size != x_arr.size:
            raise ValueError("Segment output length does not match input length.")
        return out

    return _evaluate


def build_piecewise_model_definition(
    target_col: str,
    segment_exprs: Sequence[str],
    channel_names: Sequence[str],
) -> PiecewiseModelDefinition:
    seg_exprs = [str(expr).strip() for expr in segment_exprs]
    if len(seg_exprs) < 1:
        raise ValueError("Piecewise model must contain at least 1 segment expression.")
    for idx, expr in enumerate(seg_exprs):
        if not expr:
            raise ValueError(f"Segment {idx + 1} expression is empty.")

    channel_set = {
        str(name).strip() for name in (channel_names or []) if str(name).strip()
    }
    segment_param_names: List[Tuple[str, ...]] = []
    segment_evaluators: List[Callable[..., np.ndarray]] = []
    global_names: List[str] = []
    seen_global = set()
    for seg_idx, expr in enumerate(seg_exprs):
        seg_names = extract_segment_parameter_names(expr)
        for name in seg_names:
            if name in channel_set:
                raise ValueError(
                    f"Segment {seg_idx + 1} uses channel token '{name}'. Use 'x' for the selected x-axis channel."
                )
            if name not in seen_global:
                seen_global.add(name)
                global_names.append(name)
        segment_param_names.append(tuple(seg_names))
        segment_evaluators.append(compile_segment_expression(expr, seg_names))

    return PiecewiseModelDefinition(
        target_col=str(target_col),
        segment_exprs=tuple(seg_exprs),
        segment_param_names=tuple(segment_param_names),
        segment_evaluators=tuple(segment_evaluators),
        global_param_names=tuple(global_names),
    )


def build_multi_channel_model_definition(
    channel_equations: Sequence[Tuple[str, Sequence[str]]],
    channel_names: Sequence[str],
    boundary_links: Sequence[Sequence[Tuple[str, int]]] = (),
) -> MultiChannelModelDefinition:
    """Build a multi-channel model from a list of (target_col, segment_exprs) tuples.

    Parameters with the same name across channels are shared.
    *boundary_links* is a sequence of link groups.  Each group is a sequence of
    ``(target_col, boundary_index)`` pairs that should share the same boundary
    ratio value during fitting.
    """
    if not channel_equations:
        raise ValueError("At least one channel equation is required.")
    models: List[PiecewiseModelDefinition] = []
    global_names: List[str] = []
    seen: set = set()
    for target_col, seg_exprs in channel_equations:
        model = build_piecewise_model_definition(target_col, seg_exprs, channel_names)
        models.append(model)
        for name in model.global_param_names:
            if name not in seen:
                seen.add(name)
                global_names.append(name)
    normalised_links: List[Tuple[Tuple[str, int], ...]] = []
    for group in boundary_links:
        members = tuple(sorted({(str(t), int(i)) for t, i in group}))
        if len(members) >= 2:
            normalised_links.append(members)
    return MultiChannelModelDefinition(
        channel_models=tuple(models),
        global_param_names=tuple(global_names),
        boundary_links=tuple(normalised_links),
    )


def make_segment_specs(
    model_def: PiecewiseModelDefinition,
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    fixed_param_values: Optional[Mapping[str, float]] = None,
    use_jax: bool = False,
) -> Sequence[SegmentSpec]:
    # When JAX backend is enabled, build JAX-traceable model functions
    # so that jaxfit can auto-differentiate through them on the GPU.
    if use_jax:
        try:
            from jax_backend import make_jax_segment_specs

            t0 = time.perf_counter()
            specs, _, _ = make_jax_segment_specs(
                model_def.segment_exprs,
                model_def.segment_param_names,
                seed_map,
                bounds_map,
                fixed_param_values,
            )
            _fit_log.detail(
                f"jax segment specs: {len(specs)}seg {time.perf_counter() - t0:.4f}s"
            )
            return tuple(specs)
        except Exception as _jax_exc:
            _fit_log.detail(f"jax segment specs fallback: {_jax_exc}")
            pass  # fall back to numpy path

    fixed_map = {
        str(key): float(value)
        for key, value in dict(fixed_param_values or {}).items()
        if str(key).strip()
    }
    segments: List[SegmentSpec] = []
    for seg_names, seg_eval in zip(
        model_def.segment_param_names, model_def.segment_evaluators
    ):
        lo: List[float] = []
        hi: List[float] = []
        p0: List[float] = []
        free_names: List[str] = []
        fixed_items: List[Tuple[str, float]] = []
        for name in seg_names:
            if name in fixed_map:
                fixed_items.append((name, float(fixed_map[name])))
                continue
            low, high = bounds_map[name]
            if low > high:
                low, high = high, low
            start = float(seed_map[name])
            lo.append(float(low))
            hi.append(float(high))
            p0.append(float(np.clip(start, low, high)))
            free_names.append(name)

        # Pre-build fixed param template (values already float, skip per-call conversion).
        _fixed_template = {key: float(value) for key, value in fixed_items}

        def model_func(
            x_vals,
            *vals,
            _free_names=tuple(free_names),
            _template=_fixed_template,
            _eval=seg_eval,
        ):
            local_map = dict(_template)
            for idx, key in enumerate(_free_names):
                local_map[key] = float(vals[idx])
            return _eval(x_vals, local_map)

        segments.append(
            SegmentSpec(
                model_func=model_func,
                p0=p0,
                bounds=(lo, hi),
                n_starts=2,
                maxfev=3000,
            )
        )
    return tuple(segments)


def shared_to_local_flat(
    model_def: PiecewiseModelDefinition,
    shared_values: np.ndarray,
    boundary_ratios: np.ndarray,
    fixed_param_values: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    fixed_names = {
        str(key).strip()
        for key in dict(fixed_param_values or {}).keys()
        if str(key).strip()
    }
    shared_map = {
        key: float(shared_values[idx])
        for idx, key in enumerate(model_def.global_param_names)
    }
    parts: List[float] = []
    for seg_names in model_def.segment_param_names:
        for name in seg_names:
            if name in fixed_names:
                continue
            parts.append(float(shared_map[name]))
    if boundary_ratios.size > 0:
        parts.extend(np.asarray(boundary_ratios, dtype=float).reshape(-1).tolist())
    return np.asarray(parts, dtype=float)


def _random_seed_within_bounds(
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    fixed_params: Optional[Mapping[str, float]],
    rng: "np.random.Generator",
) -> Dict[str, float]:
    """Return a seed map where each free parameter is drawn uniformly from its bounds."""
    fixed_keys = set(dict(fixed_params or {}).keys())
    result: Dict[str, float] = dict(seed_map)
    for name, bounds in bounds_map.items():
        if name in fixed_keys:
            continue
        try:
            low = float(min(bounds[0], bounds[1]))
            high = float(max(bounds[0], bounds[1]))
        except Exception:
            continue
        if not (np.isfinite(low) and np.isfinite(high)):
            continue
        if np.isclose(low, high):
            result[name] = float(low)
        else:
            result[name] = float(rng.uniform(low, high))
    return result


def _normalize_fixed_boundary_ratios(
    fixed_boundary_ratios: Optional[Mapping[int, float]],
    n_boundaries: int,
) -> Dict[int, float]:
    """Normalise a sparse {boundary_index: ratio} map."""
    out: Dict[int, float] = {}
    raw = dict(fixed_boundary_ratios or {})
    for raw_idx, raw_val in raw.items():
        try:
            idx = int(raw_idx)
            val = float(raw_val)
        except Exception:
            continue
        if idx < 0 or idx >= int(max(0, n_boundaries)):
            continue
        if not np.isfinite(val):
            continue
        out[idx] = float(np.clip(val, 0.0, 1.0))
    return out


def _run_initial_fit_stage(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    segments: Sequence[SegmentSpec],
    local_seed_flat: np.ndarray,
    n_boundaries: int,
    max_nfev: int,
    use_jax: bool = False,
) -> "OrderedPiecewiseResult":
    """Run seed-based refinement for stage-A initialisation."""
    _ = n_boundaries
    return refit_ordered_piecewise_from_seed(
        x_arr,
        y_arr,
        segments,
        seed_params=local_seed_flat,
        config=OrderedPiecewiseConfig(
            robust_max_nfev=max_nfev,
            prefer_jit=True,
            use_jax=use_jax,
        ),
    )


def run_piecewise_fit_pipeline(
    x_data: np.ndarray,
    y_data: np.ndarray,
    model_def: PiecewiseModelDefinition,
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    boundary_seed: Optional[np.ndarray] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    fixed_params: Optional[Mapping[str, float]] = None,
    fixed_boundary_ratios: Optional[Mapping[int, float]] = None,
    n_random_restarts: int = 0,
    rng_seed: Optional[int] = None,
    use_jax: bool = False,
) -> Dict[str, Any]:
    total_t0 = time.perf_counter()
    x_arr = np.asarray(x_data, dtype=float).reshape(-1)
    y_arr = np.asarray(y_data, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have equal length.")
    if x_arr.size < 8:
        raise ValueError("Not enough points to run piecewise fitting.")

    global_names = list(model_def.global_param_names)
    allowed_names = set(global_names)
    global_index = {name: idx for idx, name in enumerate(global_names)}

    seed_by_key: Dict[str, float] = {}
    for name in global_names:
        if name not in seed_map:
            raise ValueError(f"Missing seed for parameter '{name}'.")
        try:
            value = float(seed_map[name])
        except Exception as exc:
            raise ValueError(f"Seed for parameter '{name}' is non-numeric.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Seed for parameter '{name}' is not finite.")
        seed_by_key[name] = float(value)

    fixed_by_key: Dict[str, float] = {}
    for raw_key, raw_value in dict(fixed_params or {}).items():
        key = str(raw_key).strip()
        if not key:
            continue
        if key not in allowed_names:
            raise ValueError(f"Unknown fixed parameter '{key}'.")
        try:
            numeric = float(raw_value)
        except Exception as exc:
            raise ValueError(
                f"Fixed value for parameter '{key}' is non-numeric."
            ) from exc
        if not np.isfinite(numeric):
            raise ValueError(f"Fixed value for parameter '{key}' is not finite.")
        if key in bounds_map:
            low_raw, high_raw = bounds_map[key]
            low = float(min(low_raw, high_raw))
            high = float(max(low_raw, high_raw))
            tolerance = 1e-12 * max(1.0, abs(low), abs(high))
            if numeric < (low - tolerance) or numeric > (high + tolerance):
                raise ValueError(
                    f"Fixed value for parameter '{key}'={numeric:.6g} is outside [{low:.6g}, {high:.6g}]."
                )
        fixed_by_key[key] = float(numeric)
        seed_by_key[key] = float(numeric)

    def is_cancelled() -> bool:
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except Exception:
            return False

    if is_cancelled():
        raise FitCancelledError("cancelled")

    free_param_names = [name for name in global_names if name not in fixed_by_key]
    free_param_indices = np.asarray(
        [global_index[name] for name in free_param_names], dtype=int
    )
    fixed_index_values = [
        (int(global_index[name]), float(value))
        for name, value in fixed_by_key.items()
        if name in global_index
    ]

    segments = make_segment_specs(
        model_def,
        seed_by_key,
        bounds_map,
        fixed_param_values=fixed_by_key,
        use_jax=use_jax,
    )
    n_boundaries = max(0, len(segments) - 1)
    if boundary_seed is None:
        boundary_seed_arr = default_boundary_ratios(n_boundaries)
    else:
        boundary_seed_arr = np.clip(
            np.asarray(boundary_seed, dtype=float).reshape(-1), 0.0, 1.0
        )
        if boundary_seed_arr.size != n_boundaries:
            boundary_seed_arr = default_boundary_ratios(n_boundaries)
    fixed_boundary_map = _normalize_fixed_boundary_ratios(
        fixed_boundary_ratios, n_boundaries
    )
    if fixed_boundary_map:
        for idx, value in fixed_boundary_map.items():
            boundary_seed_arr[idx] = float(value)

    shared_seed = np.asarray(
        [float(seed_by_key[name]) for name in global_names], dtype=float
    )

    local_to_global_idx = []
    for seg_names in model_def.segment_param_names:
        for name in seg_names:
            if name in fixed_by_key:
                continue
            local_to_global_idx.append(global_index[name])
    local_to_global_idx = np.asarray(local_to_global_idx, dtype=int)
    n_local_params = int(local_to_global_idx.size)

    local_seed_flat = np.empty(n_local_params + n_boundaries, dtype=float)
    if n_local_params > 0:
        local_seed_flat[:n_local_params] = shared_seed[local_to_global_idx]
    if n_boundaries > 0:
        local_seed_flat[n_local_params:] = boundary_seed_arr
    free_boundary_indices = np.asarray(
        [idx for idx in range(n_boundaries) if idx not in fixed_boundary_map],
        dtype=int,
    )
    _fit_log.solver_start(
        "piecewise",
        points=x_arr.size,
        segments=len(segments),
        n_free=len(free_param_names),
        n_fixed=len(fixed_by_key),
        boundaries=n_boundaries,
        free_boundaries=int(free_boundary_indices.size),
        restarts=int(max(0, n_random_restarts)),
        use_jax=use_jax,
    )

    if n_local_params == 0 and free_boundary_indices.size == 0:
        fixed_pred = predict_ordered_piecewise(
            x_arr,
            segments,
            local_seed_flat,
            prefer_jit=True,
        )
        fixed_result = {
            "params_by_key": {name: float(seed_by_key[name]) for name in global_names},
            "params_vector": np.asarray(
                [float(seed_by_key[name]) for name in global_names], dtype=float
            ),
            "local_flat": np.asarray(local_seed_flat, dtype=float),
            "boundary_ratios": np.asarray(boundary_seed_arr, dtype=float),
            "boundaries": np.asarray(fixed_pred["boundaries"], dtype=float),
            "y_hat": np.asarray(fixed_pred["y_hat"], dtype=float),
            "r2": compute_r2(y_arr, fixed_pred["y_hat"]),
        }
        _fit_log.solver_done(
            "piecewise", time.perf_counter() - total_t0, r2=fixed_result.get("r2")
        )
        return fixed_result

    stage_a_nfev = int(np.clip(700 * max(1, local_seed_flat.size), 3000, 14000))
    stage_a_t0 = time.perf_counter()
    stage_a = _run_initial_fit_stage(
        x_arr,
        y_arr,
        segments,
        local_seed_flat,
        n_boundaries,
        stage_a_nfev,
        use_jax=use_jax,
    )
    stage_a_elapsed = time.perf_counter() - stage_a_t0
    _fit_log.solver_stage(
        "stage-a",
        stage_a_elapsed,
        sse=getattr(stage_a, "sse", None),
        nfev=stage_a_nfev,
        use_jax=use_jax,
    )
    if is_cancelled():
        raise FitCancelledError("cancelled")

    n_local = local_seed_flat.size - n_boundaries
    stage_a_local = np.asarray(stage_a.params[:n_local], dtype=float)
    stage_a_boundaries = np.asarray(stage_a.params[n_local:], dtype=float)

    shared_init = np.asarray(shared_seed, dtype=float)
    if n_local > 0:
        sums = np.zeros(shared_init.size, dtype=float)
        counts = np.zeros(shared_init.size, dtype=float)
        np.add.at(sums, local_to_global_idx, stage_a_local[:n_local])
        np.add.at(counts, local_to_global_idx, 1.0)
        use_mask = counts > 0.0
        shared_init[use_mask] = sums[use_mask] / counts[use_mask]
    for idx, value in fixed_index_values:
        shared_init[idx] = float(value)

    shared_lo = np.asarray(
        [float(bounds_map[name][0]) for name in free_param_names], dtype=float
    )
    shared_hi = np.asarray(
        [float(bounds_map[name][1]) for name in free_param_names], dtype=float
    )
    free_lower = np.minimum(shared_lo, shared_hi)
    free_upper = np.maximum(shared_lo, shared_hi)
    shared_init_free = (
        np.asarray(shared_init[free_param_indices], dtype=float)
        if free_param_indices.size > 0
        else np.asarray([], dtype=float)
    )
    free_boundary_seed = (
        np.asarray(stage_a_boundaries[free_boundary_indices], dtype=float)
        if free_boundary_indices.size > 0
        else np.asarray([], dtype=float)
    )
    lower = np.concatenate(
        [free_lower, np.zeros(int(free_boundary_indices.size), dtype=float)]
    )
    upper = np.concatenate(
        [free_upper, np.ones(int(free_boundary_indices.size), dtype=float)]
    )
    x0 = np.concatenate(
        [
            np.clip(shared_init_free, free_lower, free_upper),
            np.clip(free_boundary_seed, 0.0, 1.0),
        ]
    )

    shared_template = np.asarray(shared_seed, dtype=float)
    for idx, value in fixed_index_values:
        shared_template[idx] = float(value)
    shared_work = np.asarray(shared_template, dtype=float).copy()
    local_work = np.empty(n_local_params + n_boundaries, dtype=float)
    boundary_work = np.asarray(boundary_seed_arr, dtype=float).copy()
    n_free = int(free_param_indices.size)

    def _compose_shared_vector(free_values: np.ndarray) -> np.ndarray:
        np.copyto(shared_work, shared_template)
        if n_free > 0:
            shared_work[free_param_indices] = np.asarray(free_values, dtype=float)
        return shared_work

    def _compose_local_flat(
        shared_values: np.ndarray, free_boundary_values: np.ndarray
    ) -> np.ndarray:
        if n_local_params > 0:
            local_work[:n_local_params] = shared_values[local_to_global_idx]
        if n_boundaries > 0:
            np.copyto(boundary_work, boundary_seed_arr)
            if free_boundary_indices.size > 0:
                boundary_work[free_boundary_indices] = np.asarray(
                    free_boundary_values, dtype=float
                )
            for idx, value in fixed_boundary_map.items():
                boundary_work[idx] = float(value)
        if n_boundaries > 0:
            local_work[n_local_params:] = np.asarray(boundary_work, dtype=float)
        return local_work

    _blend_w = _auto_blend_width(np.sort(x_arr)) if n_boundaries > 0 else 0.0
    _xmin = float(np.min(x_arr))
    _xspan = float(np.max(x_arr) - _xmin)

    def residuals(vals: np.ndarray) -> np.ndarray:
        if is_cancelled():
            raise FitCancelledError("cancelled")
        vals = np.asarray(vals, dtype=float).reshape(-1)
        shared_vals = _compose_shared_vector(vals[:n_free])
        local_flat = _compose_local_flat(shared_vals, vals[n_free:])
        pred = predict_ordered_piecewise(
            x_arr,
            segments,
            local_flat,
            prefer_jit=True,
            blend_width=_blend_w,
            _x_min=_xmin,
            _x_span=_xspan,
        )
        return np.asarray(pred["y_hat"] - y_arr, dtype=float)

    best_shared = np.asarray(shared_init, dtype=float)
    best_boundary = np.asarray(np.clip(stage_a_boundaries, 0.0, 1.0), dtype=float)
    for idx, value in fixed_boundary_map.items():
        best_boundary[idx] = float(value)
    best_local_flat = np.asarray(
        _compose_local_flat(
            best_shared,
            best_boundary[free_boundary_indices]
            if free_boundary_indices.size > 0
            else np.asarray([], dtype=float),
        ),
        dtype=float,
    ).copy()
    best_pred = predict_ordered_piecewise(
        x_arr,
        segments,
        best_local_flat,
        prefer_jit=True,
        _x_min=_xmin,
        _x_span=_xspan,
    )
    stage_b_elapsed = 0.0
    stage_b_nfev = None
    stage_b_status = "skipped"
    if x0.size > 0:
        stage_b_t0 = time.perf_counter()
        try:
            diff_step = None
            if free_boundary_indices.size > 0:
                diff_step = np.full(x0.size, 1e-6, dtype=float)
                diff_step[n_free:] = _boundary_ratio_diff_step_from_x(x_arr)
            shared_refine_nfev = int(np.clip(500 * max(1, x0.size), 2500, 14000))
            ls = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                method="trf",
                loss="soft_l1",
                f_scale=0.5,
                max_nfev=shared_refine_nfev,
                diff_step=diff_step,
                x_scale="jac",
            )
            refined_vals = np.asarray(ls.x, dtype=float)
            stage_b_status = "ok"
            stage_b_nfev = int(getattr(ls, "nfev", 0))
            refined_shared = np.asarray(
                _compose_shared_vector(refined_vals[:n_free]), dtype=float
            ).copy()
            refined_free_boundary = np.asarray(refined_vals[n_free:], dtype=float)
            refined_boundary = np.asarray(boundary_seed_arr, dtype=float).copy()
            if free_boundary_indices.size > 0:
                refined_boundary[free_boundary_indices] = refined_free_boundary
            for idx, value in fixed_boundary_map.items():
                refined_boundary[idx] = float(value)
            refined_local = np.asarray(
                _compose_local_flat(refined_shared, refined_free_boundary),
                dtype=float,
            ).copy()
            refined_pred = predict_ordered_piecewise(
                x_arr,
                segments,
                refined_local,
                prefer_jit=True,
                _x_min=_xmin,
                _x_span=_xspan,
            )
            # Guard: only accept stage-B result if it actually improved R².
            refined_r2 = compute_r2(y_arr, refined_pred["y_hat"])
            pre_r2 = compute_r2(y_arr, best_pred["y_hat"])
            if refined_r2 is None or (
                pre_r2 is not None and refined_r2 < pre_r2 - 1e-12
            ):
                stage_b_status = "rejected (R² worse)"
            else:
                best_shared = refined_shared
                best_boundary = refined_boundary
                best_local_flat = refined_local
                best_pred = refined_pred

        except FitCancelledError:
            raise
        except Exception as exc:
            stage_b_status = f"error:{type(exc).__name__}"
        finally:
            stage_b_elapsed = time.perf_counter() - stage_b_t0
    _fit_log.solver_stage(
        "stage-b", stage_b_elapsed, status=stage_b_status, nfev=stage_b_nfev
    )

    best_result = {
        "params_by_key": {
            name: float(best_shared[idx]) for idx, name in enumerate(global_names)
        },
        "params_vector": np.asarray(best_shared, dtype=float),
        "local_flat": np.asarray(best_local_flat, dtype=float),
        "boundary_ratios": np.asarray(best_boundary, dtype=float),
        "boundaries": np.asarray(best_pred["boundaries"], dtype=float),
        "y_hat": np.asarray(best_pred["y_hat"], dtype=float),
        "r2": compute_r2(y_arr, best_pred["y_hat"]),
    }

    # ── Random restarts: try N additional random starting points ──────
    n_restarts = int(n_random_restarts)
    restart_improved_count = 0
    restarts_elapsed = 0.0
    if n_restarts > 0:
        restarts_t0 = time.perf_counter()
        rng = np.random.default_rng(rng_seed)
        best_r2 = best_result.get("r2")
        for restart_idx in range(n_restarts):
            if is_cancelled():
                break
            restart_t0 = time.perf_counter()
            restart_seed = _random_seed_within_bounds(
                seed_map, bounds_map, fixed_params, rng
            )
            restart_boundary: Optional[np.ndarray] = (
                np.asarray(rng.uniform(0.0, 1.0, size=int(n_boundaries)), dtype=float)
                if n_boundaries > 0
                else np.asarray([], dtype=float)
            )
            for idx, value in fixed_boundary_map.items():
                if idx < restart_boundary.size:
                    restart_boundary[idx] = float(value)
            try:
                candidate = run_piecewise_fit_pipeline(
                    x_arr,
                    y_arr,
                    model_def,
                    restart_seed,
                    bounds_map,
                    boundary_seed=restart_boundary,
                    cancel_check=cancel_check,
                    fixed_params=fixed_params,
                    fixed_boundary_ratios=fixed_boundary_map,
                    n_random_restarts=0,
                )
            except FitCancelledError:
                raise
            except Exception as exc:
                _fit_log.solver_restart(
                    restart_idx + 1,
                    n_restarts,
                    time.perf_counter() - restart_t0,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            candidate_r2 = candidate.get("r2")
            _fit_log.solver_restart(
                restart_idx + 1,
                n_restarts,
                time.perf_counter() - restart_t0,
                r2=candidate_r2,
            )
            if candidate_r2 is not None and (
                best_r2 is None or float(candidate_r2) > float(best_r2)
            ):
                best_result = candidate
                best_r2 = float(candidate_r2)
                restart_improved_count += 1
        restarts_elapsed = time.perf_counter() - restarts_t0

    _fit_log.solver_done(
        "piecewise",
        time.perf_counter() - total_t0,
        r2=best_result.get("r2"),
        stage_a=stage_a_elapsed,
        stage_b=stage_b_elapsed,
        restarts=n_restarts,
        restart_elapsed=restarts_elapsed,
        restart_improved=restart_improved_count,
        use_jax=use_jax,
    )
    return best_result


def run_multi_channel_fit_pipeline(
    x_data: np.ndarray,
    y_data_by_channel: Mapping[str, np.ndarray],
    multi_model: MultiChannelModelDefinition,
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    boundary_seeds: Optional[Mapping[str, np.ndarray]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    fixed_params: Optional[Mapping[str, float]] = None,
    fixed_boundary_ratios_by_channel: Optional[
        Mapping[str, Mapping[int, float]]
    ] = None,
    n_random_restarts: int = 0,
    rng_seed: Optional[int] = None,
    use_jax: bool = False,
) -> Dict[str, Any]:
    """Fit shared parameters across multiple channel equations simultaneously.

    Each channel equation is a piecewise model with its own boundary ratios.
    Parameters with the same name are shared across channels.
    Returns per-channel results plus combined metrics.
    """
    total_t0 = time.perf_counter()
    x_arr = np.asarray(x_data, dtype=float).reshape(-1)
    if x_arr.size < 8:
        raise ValueError("Not enough points to run multi-channel fitting.")

    global_names = list(multi_model.global_param_names)
    global_index = {name: idx for idx, name in enumerate(global_names)}
    boundary_seeds_map = dict(boundary_seeds or {})
    fixed_boundary_raw = {
        str(k): dict(v or {})
        for k, v in dict(fixed_boundary_ratios_by_channel or {}).items()
    }

    fixed_by_key: Dict[str, float] = {}
    for raw_key, raw_value in dict(fixed_params or {}).items():
        key = str(raw_key).strip()
        if not key or key not in global_index:
            continue
        try:
            numeric = float(raw_value)
        except Exception:
            continue
        if not np.isfinite(numeric):
            continue
        fixed_by_key[key] = float(numeric)

    seed_by_key: Dict[str, float] = {}
    for name in global_names:
        if name not in seed_map:
            raise ValueError(f"Missing seed for parameter '{name}'.")
        value = float(seed_map[name])
        if not np.isfinite(value):
            raise ValueError(f"Seed for parameter '{name}' is not finite.")
        seed_by_key[name] = float(value)
    for key, value in fixed_by_key.items():
        seed_by_key[key] = float(value)

    def is_cancelled() -> bool:
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except Exception:
            return False

    if is_cancelled():
        raise FitCancelledError("cancelled")

    free_param_names = [name for name in global_names if name not in fixed_by_key]
    free_param_indices = np.asarray(
        [global_index[name] for name in free_param_names], dtype=int
    )
    fixed_index_values = [
        (int(global_index[name]), float(value))
        for name, value in fixed_by_key.items()
        if name in global_index
    ]

    # Build per-channel segment specs and boundary info.
    channel_info: List[Dict[str, Any]] = []
    fixed_boundary_by_bid: Dict[Tuple[str, int], float] = {}
    for ch_model in multi_model.channel_models:
        target = ch_model.target_col
        if target not in y_data_by_channel:
            raise ValueError(f"Missing y-data for channel '{target}'.")
        y_arr = np.asarray(y_data_by_channel[target], dtype=float).reshape(-1)
        if y_arr.size != x_arr.size:
            raise ValueError(
                f"Channel '{target}' data length ({y_arr.size}) != x length ({x_arr.size})."
            )
        segments = make_segment_specs(
            ch_model,
            seed_by_key,
            bounds_map,
            fixed_param_values=fixed_by_key,
            use_jax=use_jax,
        )
        n_boundaries = max(0, len(segments) - 1)
        b_seed = boundary_seeds_map.get(target)
        if b_seed is None:
            b_seed_arr = default_boundary_ratios(n_boundaries)
        else:
            b_seed_arr = np.clip(np.asarray(b_seed, dtype=float).reshape(-1), 0.0, 1.0)
            if b_seed_arr.size != n_boundaries:
                b_seed_arr = default_boundary_ratios(n_boundaries)
        fixed_local = _normalize_fixed_boundary_ratios(
            fixed_boundary_raw.get(target), n_boundaries
        )
        for idx, value in fixed_local.items():
            b_seed_arr[idx] = float(value)
            fixed_boundary_by_bid[(target, int(idx))] = float(value)

        # Per-channel local-to-global index mapping.
        local_to_global: List[int] = []
        for seg_names in ch_model.segment_param_names:
            for name in seg_names:
                if name in fixed_by_key:
                    continue
                local_to_global.append(global_index[name])
        local_to_global_arr = np.asarray(local_to_global, dtype=int)

        channel_info.append(
            {
                "model": ch_model,
                "target": target,
                "y_arr": y_arr,
                "segments": segments,
                "n_boundaries": n_boundaries,
                "boundary_seed": b_seed_arr,
                "fixed_boundary_map": fixed_local,
                "local_to_global": local_to_global_arr,
                "n_local_params": int(local_to_global_arr.size),
            }
        )
    channel_targets = [str(ch["target"]) for ch in channel_info]
    _fit_log.solver_start(
        "multi",
        points=x_arr.size,
        channels=channel_targets,
        n_free=len(free_param_names),
        n_fixed=len(fixed_by_key),
        restarts=int(max(0, n_random_restarts)),
        use_jax=use_jax,
    )

    # Build shared optimization vector: [free_params, boundary_ch1, boundary_ch2, ...]
    shared_seed = np.asarray(
        [float(seed_by_key[name]) for name in global_names], dtype=float
    )
    n_free = int(free_param_indices.size)

    # Stage A: per-channel initial fit, then average to get shared seed.
    stage_a_total_elapsed = 0.0
    for ch in channel_info:
        if is_cancelled():
            raise FitCancelledError("cancelled")
        n_local = ch["n_local_params"]
        n_b = ch["n_boundaries"]
        local_seed = np.empty(n_local + n_b, dtype=float)
        if n_local > 0:
            local_seed[:n_local] = shared_seed[ch["local_to_global"]]
        if n_b > 0:
            local_seed[n_local:] = ch["boundary_seed"]
        nfev = int(np.clip(700 * max(1, local_seed.size), 3000, 14000))
        ch_stage_t0 = time.perf_counter()
        stage = _run_initial_fit_stage(
            x_arr,
            ch["y_arr"],
            ch["segments"],
            local_seed,
            n_b,
            nfev,
            use_jax=use_jax,
        )
        ch_stage_elapsed = time.perf_counter() - ch_stage_t0
        stage_a_total_elapsed += ch_stage_elapsed
        _fit_log.solver_stage(
            "stage-a",
            ch_stage_elapsed,
            channel=ch["target"],
            sse=getattr(stage, "sse", None),
            nfev=nfev,
            use_jax=use_jax,
        )
        ch["stage_a_local"] = np.asarray(stage.params[:n_local], dtype=float)
        ch["stage_a_boundary"] = np.asarray(stage.params[n_local:], dtype=float)

    # Average per-channel stage A results for shared parameter seed.
    shared_init = np.asarray(shared_seed, dtype=float)
    sums = np.zeros(shared_init.size, dtype=float)
    counts = np.zeros(shared_init.size, dtype=float)
    for ch in channel_info:
        n_local = ch["n_local_params"]
        if n_local > 0:
            np.add.at(sums, ch["local_to_global"], ch["stage_a_local"][:n_local])
            np.add.at(counts, ch["local_to_global"], 1.0)
    use_mask = counts > 0.0
    shared_init[use_mask] = sums[use_mask] / counts[use_mask]
    for idx, value in fixed_index_values:
        shared_init[idx] = float(value)

    # ---- Boundary variable mapping from link groups ----
    all_bids: List[Tuple[str, int]] = []
    for ch in channel_info:
        for i in range(ch["n_boundaries"]):
            all_bids.append((ch["target"], i))
    all_bids_set = set(all_bids)
    ch_info_by_target = {ch["target"]: ch for ch in channel_info}

    # If a linked boundary group has any fixed member, force the full group fixed.
    for group in multi_model.boundary_links:
        valid = [bid for bid in group if bid in all_bids_set]
        if not valid:
            continue
        fixed_vals = [
            fixed_boundary_by_bid.get(bid)
            for bid in valid
            if bid in fixed_boundary_by_bid
        ]
        if not fixed_vals:
            continue
        forced = float(np.clip(float(fixed_vals[0]), 0.0, 1.0))
        for bid in valid:
            fixed_boundary_by_bid[bid] = forced

    bid_to_var: Dict[Tuple[str, int], int] = {}
    linked_bids: set = set()
    boundary_var_count = 0
    for group in multi_model.boundary_links:
        valid = [bid for bid in group if bid in all_bids_set]
        if len(valid) < 2:
            continue
        free_valid = [bid for bid in valid if bid not in fixed_boundary_by_bid]
        if not free_valid:
            continue
        for bid in free_valid:
            bid_to_var[bid] = boundary_var_count
            linked_bids.add(bid)
        boundary_var_count += 1
    for bid in all_bids:
        if bid in fixed_boundary_by_bid:
            continue
        if bid not in bid_to_var:
            bid_to_var[bid] = boundary_var_count
            boundary_var_count += 1
    total_boundary = boundary_var_count
    _fit_log.solver_stage(
        "setup",
        0,
        status=f"free_bounds={total_boundary} fixed_bids={len(fixed_boundary_by_bid)}",
    )

    # Build combined optimization vector.
    shared_lo = np.asarray(
        [float(bounds_map[name][0]) for name in free_param_names], dtype=float
    )
    shared_hi = np.asarray(
        [float(bounds_map[name][1]) for name in free_param_names], dtype=float
    )
    free_lower = np.minimum(shared_lo, shared_hi)
    free_upper = np.maximum(shared_lo, shared_hi)

    shared_init_free = (
        np.asarray(shared_init[free_param_indices], dtype=float)
        if free_param_indices.size > 0
        else np.asarray([], dtype=float)
    )

    boundary_lo = np.zeros(total_boundary, dtype=float)
    boundary_hi = np.ones(total_boundary, dtype=float)
    boundary_init = np.zeros(total_boundary, dtype=float)

    # Seed linked groups by averaging stage-A seeds of their members.
    for group in multi_model.boundary_links:
        valid = [bid for bid in group if bid in linked_bids]
        if not valid:
            continue
        var_idx = bid_to_var[valid[0]]
        seeds_for_group: List[float] = []
        for target, bidx in valid:
            ch = ch_info_by_target.get(target)
            if ch is not None and bidx < ch["stage_a_boundary"].size:
                seeds_for_group.append(float(ch["stage_a_boundary"][bidx]))
        if seeds_for_group:
            boundary_init[var_idx] = float(np.clip(np.mean(seeds_for_group), 0.0, 1.0))
    # Seed unlinked boundaries from their own stage-A value.
    for bid in all_bids:
        if bid in linked_bids:
            continue
        if bid in fixed_boundary_by_bid:
            continue
        var_idx = bid_to_var[bid]
        target, bidx = bid
        ch = ch_info_by_target.get(target)
        if ch is not None and bidx < ch["stage_a_boundary"].size:
            boundary_init[var_idx] = float(
                np.clip(float(ch["stage_a_boundary"][bidx]), 0.0, 1.0)
            )

    lower = np.concatenate([free_lower, boundary_lo])
    upper = np.concatenate([free_upper, boundary_hi])
    x0 = np.concatenate(
        [
            np.clip(shared_init_free, free_lower, free_upper),
            boundary_init,
        ]
    )

    def _boundary_value_for_bid(
        bid: Tuple[str, int], boundary_values: np.ndarray
    ) -> float:
        if bid in fixed_boundary_by_bid:
            return float(fixed_boundary_by_bid[bid])
        var_idx = bid_to_var.get(bid)
        if var_idx is None:
            return 0.5
        return float(boundary_values[var_idx])

    if x0.size == 0:
        # All parameters are fixed. Just evaluate.
        results_by_channel: Dict[str, Any] = {}
        for ch in channel_info:
            n_local = ch["n_local_params"]
            n_b = ch["n_boundaries"]
            local_flat = np.empty(n_local + n_b, dtype=float)
            if n_local > 0:
                local_flat[:n_local] = shared_init[ch["local_to_global"]]
            if n_b > 0:
                ch_ratios = np.asarray(ch["boundary_seed"], dtype=float).copy()
                for i in range(n_b):
                    ch_ratios[i] = _boundary_value_for_bid(
                        (ch["target"], i), np.asarray([], dtype=float)
                    )
                local_flat[n_local:] = ch_ratios
            pred = predict_ordered_piecewise(
                x_arr, ch["segments"], local_flat, prefer_jit=True
            )
            results_by_channel[ch["target"]] = {
                "y_hat": np.asarray(pred["y_hat"], dtype=float),
                "boundary_ratios": np.asarray(local_flat[n_local:], dtype=float),
                "boundaries": np.asarray(pred["boundaries"], dtype=float),
                "r2": compute_r2(ch["y_arr"], pred["y_hat"]),
            }
        fixed_result = {
            "params_by_key": {
                name: float(shared_init[idx]) for idx, name in enumerate(global_names)
            },
            "params_vector": np.asarray(shared_init, dtype=float),
            "channel_results": results_by_channel,
            "r2": _combine_channel_r2(results_by_channel),
        }
        _fit_log.solver_done(
            "multi", time.perf_counter() - total_t0, r2=fixed_result.get("r2")
        )
        return fixed_result

    # Build template and work arrays for composing vectors.
    shared_template = np.asarray(
        [float(seed_by_key[name]) for name in global_names], dtype=float
    ).copy()
    for idx, value in fixed_index_values:
        shared_template[idx] = float(value)
    shared_work = shared_template.copy()

    def _compose_shared(free_values: np.ndarray) -> np.ndarray:
        np.copyto(shared_work, shared_template)
        if n_free > 0:
            shared_work[free_param_indices] = np.asarray(free_values, dtype=float)
        return shared_work

    _mc_blend_w = _auto_blend_width(np.sort(x_arr)) if total_boundary > 0 else 0.0
    _mc_xmin = float(np.min(x_arr))
    _mc_xspan = float(np.max(x_arr) - _mc_xmin)

    def residuals(vals: np.ndarray) -> np.ndarray:
        if is_cancelled():
            raise FitCancelledError("cancelled")
        vals = np.asarray(vals, dtype=float).reshape(-1)
        shared_vals = _compose_shared(vals[:n_free])
        boundary_vals = np.asarray(vals[n_free:], dtype=float)
        all_residuals: List[np.ndarray] = []
        for ch in channel_info:
            n_local = ch["n_local_params"]
            n_b = ch["n_boundaries"]
            local_flat = np.empty(n_local + n_b, dtype=float)
            if n_local > 0:
                local_flat[:n_local] = shared_vals[ch["local_to_global"]]
            for i in range(n_b):
                local_flat[n_local + i] = _boundary_value_for_bid(
                    (ch["target"], i),
                    boundary_vals,
                )
            pred = predict_ordered_piecewise(
                x_arr,
                ch["segments"],
                local_flat,
                prefer_jit=True,
                blend_width=_mc_blend_w,
                _x_min=_mc_xmin,
                _x_span=_mc_xspan,
            )
            all_residuals.append(np.asarray(pred["y_hat"] - ch["y_arr"], dtype=float))
        return np.concatenate(all_residuals)

    # Run combined least-squares optimization.
    stage_b_t0 = time.perf_counter()
    stage_b_nfev = None
    stage_b_status = "skipped"
    try:
        diff_step = np.full(x0.size, 1e-6, dtype=float)
        if total_boundary > 0:
            diff_step[n_free:] = _boundary_ratio_diff_step_from_x(x_arr)
        refine_nfev = int(np.clip(500 * max(1, x0.size), 2500, 20000))
        ls = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=0.5,
            max_nfev=refine_nfev,
            diff_step=diff_step,
            x_scale="jac",
        )
        refined_vals = np.asarray(ls.x, dtype=float)
        stage_b_nfev = int(getattr(ls, "nfev", 0))
        stage_b_status = "ok"
    except FitCancelledError:
        raise
    except Exception as exc:
        refined_vals = x0
        stage_b_status = f"error:{type(exc).__name__}"
    stage_b_elapsed = time.perf_counter() - stage_b_t0
    _fit_log.solver_stage(
        "stage-b", stage_b_elapsed, status=stage_b_status, nfev=stage_b_nfev
    )

    np.asarray(_compose_shared(refined_vals[:n_free]), dtype=float).copy()

    # Helper to evaluate per-channel results from an optimization vector.
    def _eval_multi_channel(vals: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        s = np.asarray(_compose_shared(vals[:n_free]), dtype=float).copy()
        bvals = np.asarray(vals[n_free:], dtype=float)
        ch_results: Dict[str, Any] = {}
        for ch in channel_info:
            n_local = ch["n_local_params"]
            n_b = ch["n_boundaries"]
            local_flat = np.empty(n_local + n_b, dtype=float)
            if n_local > 0:
                local_flat[:n_local] = s[ch["local_to_global"]]
            ch_ratios = np.zeros(n_b, dtype=float)
            for i in range(n_b):
                val = _boundary_value_for_bid((ch["target"], i), bvals)
                local_flat[n_local + i] = val
                ch_ratios[i] = val
            pred = predict_ordered_piecewise(
                x_arr, ch["segments"], local_flat, prefer_jit=True
            )
            ch_results[ch["target"]] = {
                "y_hat": np.asarray(pred["y_hat"], dtype=float),
                "boundary_ratios": np.asarray(ch_ratios, dtype=float),
                "boundaries": np.asarray(pred["boundaries"], dtype=float),
                "r2": compute_r2(ch["y_arr"], pred["y_hat"]),
            }
        result = {
            "params_by_key": {
                name: float(s[idx]) for idx, name in enumerate(global_names)
            },
            "params_vector": np.asarray(s, dtype=float),
            "channel_results": ch_results,
            "r2": _combine_channel_r2(ch_results),
        }
        return s, result

    # Evaluate both pre- and post-optimization states.
    _, refined_result = _eval_multi_channel(refined_vals)
    _, pre_result = _eval_multi_channel(x0)

    # Guard: keep pre-optimization state if stage B made R² worse.
    pre_r2 = pre_result.get("r2")
    refined_r2 = refined_result.get("r2")
    if refined_r2 is None or (pre_r2 is not None and refined_r2 < pre_r2 - 1e-12):
        best_result = pre_result
        np.asarray(_compose_shared(x0[:n_free]), dtype=float).copy()
        stage_b_status = "rejected (R² worse)"
    else:
        best_result = refined_result

    # ── Random restarts: try N additional random starting points ──────
    n_restarts = int(n_random_restarts)
    restart_improved_count = 0
    restarts_elapsed = 0.0
    if n_restarts > 0:
        restarts_t0 = time.perf_counter()
        rng = np.random.default_rng(rng_seed)
        best_r2 = best_result.get("r2")
        for restart_idx in range(n_restarts):
            if is_cancelled():
                break
            restart_t0 = time.perf_counter()
            restart_seed = _random_seed_within_bounds(
                seed_map, bounds_map, fixed_params, rng
            )
            restart_boundary_seeds: Dict[str, np.ndarray] = {}
            for ch in channel_info:
                n_b = ch["n_boundaries"]
                candidate = (
                    np.asarray(rng.uniform(0.0, 1.0, size=n_b), dtype=float)
                    if n_b > 0
                    else np.asarray([], dtype=float)
                )
                for i in range(n_b):
                    bid = (ch["target"], i)
                    if bid in fixed_boundary_by_bid:
                        candidate[i] = float(fixed_boundary_by_bid[bid])
                restart_boundary_seeds[ch["target"]] = candidate
            try:
                candidate = run_multi_channel_fit_pipeline(
                    x_arr,
                    y_data_by_channel,
                    multi_model,
                    restart_seed,
                    bounds_map,
                    boundary_seeds=restart_boundary_seeds,
                    cancel_check=cancel_check,
                    fixed_params=fixed_params,
                    fixed_boundary_ratios_by_channel=fixed_boundary_ratios_by_channel,
                    n_random_restarts=0,
                )
            except FitCancelledError:
                raise
            except Exception as exc:
                _fit_log.solver_restart(
                    restart_idx + 1,
                    n_restarts,
                    time.perf_counter() - restart_t0,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            candidate_r2 = candidate.get("r2")
            _fit_log.solver_restart(
                restart_idx + 1,
                n_restarts,
                time.perf_counter() - restart_t0,
                r2=candidate_r2,
            )
            if candidate_r2 is not None and (
                best_r2 is None or float(candidate_r2) > float(best_r2)
            ):
                best_result = candidate
                best_r2 = float(candidate_r2)
                restart_improved_count += 1
        restarts_elapsed = time.perf_counter() - restarts_t0

    _fit_log.solver_done(
        "multi",
        time.perf_counter() - total_t0,
        r2=best_result.get("r2"),
        stage_a=stage_a_total_elapsed,
        stage_b=stage_b_elapsed,
        restarts=n_restarts,
        restart_elapsed=restarts_elapsed,
        restart_improved=restart_improved_count,
        use_jax=use_jax,
    )
    return best_result


def _combine_channel_r2(results_by_channel: Mapping[str, Any]) -> Optional[float]:
    """Average R² across channels, ignoring those without a value."""
    r2_values = []
    for ch_result in results_by_channel.values():
        r2 = ch_result.get("r2")
        if r2 is not None:
            r2_values.append(float(r2))
    if not r2_values:
        return None
    return float(np.mean(r2_values))
