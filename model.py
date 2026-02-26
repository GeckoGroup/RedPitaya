"""Parameter specs, model definitions, fitting pipeline, and helper metrics."""

import ast
import math
import re
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
from sklearn.metrics import r2_score
from scipy.optimize import least_squares

from expression import (
    _ExpressionParameterCollector,
    _PARAMETER_NAME_RE,
    _EXPRESSION_HELPER_NAMES,
    _EXPRESSION_ALLOWED_FUNCTIONS,
    _EXPRESSION_ALLOWED_CONSTANTS,
)
from solver import (
    OrderedPiecewiseConfig,
    SegmentSpec,
    predict_ordered_piecewise,
    refit_ordered_piecewise_from_seed,
)


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    symbol: str
    description: str
    default: float
    min_value: float
    max_value: float
    decimals: int = 4

    @property
    def inferred_step(self) -> float:
        # Infer a sensible spinbox step from parameter span.
        precision_step = 10 ** (-self.decimals)
        span = abs(self.max_value - self.min_value)
        if span <= 0.0:
            return precision_step
        span_order = int(np.floor(np.log10(span)))
        span_step = 10 ** (span_order - 4)
        return max(precision_step, span_step)

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
    "#2563eb",  # blue
    "#f59e0b",  # amber
    "#7c3aed",  # violet
    "#dc2626",  # red
    "#0891b2",  # cyan
    "#ea580c",  # orange
    "#0f766e",  # teal
    "#a855f7",  # purple
    "#64748b",  # slate
    "#be123c",  # rose
)


def palette_color(index: int) -> str:
    idx = int(index)
    if CHANNEL_PLOT_COLORS:
        return CHANNEL_PLOT_COLORS[idx % len(CHANNEL_PLOT_COLORS)]
    return f"C{idx % 10}"


def compute_r2(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if np.count_nonzero(valid) < 2:
        return None
    try:
        return float(r2_score(y_true_arr[valid], y_pred_arr[valid]))
    except Exception:
        return None


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


def _finite_float_or_none(value) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _row_has_error(row: Mapping[str, Any]) -> bool:
    pattern_error = str(row.get("pattern_error") or "").strip()
    fit_error = str(row.get("error") or "").strip()
    normalized_fit_error = fit_error.lower().replace(".", "").strip()
    if normalized_fit_error in {"cancelled", "canceled"}:
        fit_error = ""
    return bool(pattern_error or fit_error)


def is_fit_row_improved(
    candidate_row: Mapping[str, Any], baseline_row: Mapping[str, Any]
) -> bool:
    candidate_has_fit = has_nonempty_values(candidate_row.get("params"))
    baseline_has_fit = has_nonempty_values(baseline_row.get("params"))
    if candidate_has_fit and not baseline_has_fit:
        return True
    if not candidate_has_fit:
        return False
    if not baseline_has_fit:
        return True

    candidate_has_error = _row_has_error(candidate_row)
    baseline_has_error = _row_has_error(baseline_row)
    if baseline_has_error and not candidate_has_error:
        return True
    if candidate_has_error and not baseline_has_error:
        return False

    candidate_r2 = _finite_float_or_none(candidate_row.get("r2"))
    baseline_r2 = _finite_float_or_none(baseline_row.get("r2"))
    if candidate_r2 is not None and baseline_r2 is not None:
        tolerance = 1e-12
        return bool(candidate_r2 > (baseline_r2 + tolerance))
    if candidate_r2 is not None and baseline_r2 is None:
        return True
    return False


def default_boundary_ratios(n_boundaries: int) -> np.ndarray:
    n = int(max(0, n_boundaries))
    if n <= 0:
        return np.asarray([], dtype=float)
    pcts = np.linspace(1.0 / (n + 1), n / (n + 1), n)
    ratios = np.empty(n, dtype=float)
    prev = 0.0
    for i, pct in enumerate(pcts):
        denom = max(1.0 - prev, 1e-12)
        ratios[i] = (float(pct) - prev) / denom
        prev = float(pct)
    return np.clip(ratios, 0.0, 1.0)


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


def _boundary_ratio_diff_step_from_x(x_values: np.ndarray) -> float:
    x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    finite = np.sort(x_arr[np.isfinite(x_arr)])
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
    eval_globals = {
        "__builtins__": __builtins__,
        "np": np,
        "math": math,
        **_EXPRESSION_ALLOWED_FUNCTIONS,
        **_EXPRESSION_ALLOWED_CONSTANTS,
    }

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


def _make_segment_specs(
    model_def: PiecewiseModelDefinition,
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    fixed_param_values: Optional[Mapping[str, float]] = None,
) -> Sequence[SegmentSpec]:
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

        def model_func(
            x_vals,
            *vals,
            _free_names=tuple(free_names),
            _fixed_items=tuple(fixed_items),
            _eval=seg_eval,
        ):
            local_map = {key: float(value) for key, value in _fixed_items}
            local_map.update(
                {key: float(vals[idx]) for idx, key in enumerate(_free_names)}
            )
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


def _shared_to_local_flat(
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


def run_piecewise_fit_pipeline(
    x_data: np.ndarray,
    y_data: np.ndarray,
    model_def: PiecewiseModelDefinition,
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    boundary_seed: Optional[np.ndarray] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    fixed_params: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
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

    segments = _make_segment_specs(
        model_def,
        seed_by_key,
        bounds_map,
        fixed_param_values=fixed_by_key,
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

    if local_seed_flat.size == 0:
        fixed_pred = predict_ordered_piecewise(
            x_arr,
            segments,
            local_seed_flat,
            prefer_jit=True,
        )
        return {
            "params_by_key": {name: float(seed_by_key[name]) for name in global_names},
            "params_vector": np.asarray(
                [float(seed_by_key[name]) for name in global_names], dtype=float
            ),
            "local_flat": np.asarray(local_seed_flat, dtype=float),
            "boundary_ratios": np.asarray([], dtype=float),
            "boundaries": np.asarray(fixed_pred["boundaries"], dtype=float),
            "y_hat": np.asarray(fixed_pred["y_hat"], dtype=float),
            "r2": compute_r2(y_arr, fixed_pred["y_hat"]),
        }

    stage_a_nfev = int(np.clip(700 * max(1, local_seed_flat.size), 3000, 14000))
    stage_a = refit_ordered_piecewise_from_seed(
        x_arr,
        y_arr,
        segments,
        seed_params=local_seed_flat,
        config=OrderedPiecewiseConfig(
            robust_max_nfev=stage_a_nfev,
            prefer_jit=True,
        ),
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
    lower = np.concatenate([free_lower, np.zeros(n_boundaries, dtype=float)])
    upper = np.concatenate([free_upper, np.ones(n_boundaries, dtype=float)])
    x0 = np.concatenate(
        [
            np.clip(shared_init_free, free_lower, free_upper),
            np.clip(stage_a_boundaries, 0.0, 1.0),
        ]
    )

    shared_template = np.asarray(shared_seed, dtype=float)
    for idx, value in fixed_index_values:
        shared_template[idx] = float(value)
    shared_work = np.asarray(shared_template, dtype=float).copy()
    local_work = np.empty(n_local_params + n_boundaries, dtype=float)
    n_free = int(free_param_indices.size)

    def _compose_shared_vector(free_values: np.ndarray) -> np.ndarray:
        np.copyto(shared_work, shared_template)
        if n_free > 0:
            shared_work[free_param_indices] = np.asarray(free_values, dtype=float)
        return shared_work

    def _compose_local_flat(
        shared_values: np.ndarray, boundary_values: np.ndarray
    ) -> np.ndarray:
        if n_local_params > 0:
            local_work[:n_local_params] = shared_values[local_to_global_idx]
        if n_boundaries > 0:
            local_work[n_local_params:] = np.asarray(boundary_values, dtype=float)
        return local_work

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
        )
        return np.asarray(pred["y_hat"] - y_arr, dtype=float)

    best_shared = np.asarray(shared_init, dtype=float)
    best_boundary = np.asarray(np.clip(stage_a_boundaries, 0.0, 1.0), dtype=float)
    best_local_flat = np.asarray(
        _compose_local_flat(best_shared, best_boundary),
        dtype=float,
    ).copy()
    best_pred = predict_ordered_piecewise(
        x_arr,
        segments,
        best_local_flat,
        prefer_jit=True,
    )
    if x0.size > 0:
        try:
            diff_step = None
            if n_boundaries > 0:
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
            refined_shared = np.asarray(
                _compose_shared_vector(refined_vals[:n_free]), dtype=float
            ).copy()
            refined_boundary = np.asarray(refined_vals[n_free:], dtype=float)
            refined_local = np.asarray(
                _compose_local_flat(refined_shared, refined_boundary),
                dtype=float,
            ).copy()
            refined_pred = predict_ordered_piecewise(
                x_arr,
                segments,
                refined_local,
                prefer_jit=True,
            )
            best_shared = refined_shared
            best_boundary = refined_boundary
            best_local_flat = refined_local
            best_pred = refined_pred

        except FitCancelledError:
            raise

    return {
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
