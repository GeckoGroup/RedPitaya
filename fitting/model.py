"""Parameter specs, model definitions, fitting pipeline, and helper metrics."""

import ast
import math
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
    fit_ordered_piecewise,
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

    @property
    def has_boundary_links(self) -> bool:
        return len(self.boundary_links) > 0


# ── Fitting procedure data classes ──────────────────────────────────


@dataclass(frozen=True)
class FitProcedureStep:
    """One step in a multi-step fitting procedure.

    *channels*  – target channel names to fit in this step (empty → all).
    *free_params* – parameter keys to optimise (empty → all non-fixed).
    *fixed_params* – parameter keys to hold at their current values.
    *bound_params* – mapping of param_key → capture_field_name.
                     Bound params are fixed at the value extracted from the
                     filename pattern, like the main GUI's field-binding.
    *min_r2*   – minimum R² required to continue to the next step.
                 ``None`` means always continue.
    *label*    – optional human-readable label for this step.
    """

    channels: Tuple[str, ...] = ()
    free_params: Tuple[str, ...] = ()
    fixed_params: Tuple[str, ...] = ()
    bound_params: Tuple[Tuple[str, str], ...] = ()  # ((param_key, field_name), ...)
    min_r2: Optional[float] = None
    label: str = ""

    def serialize(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.channels:
            d["channels"] = list(self.channels)
        if self.free_params:
            d["free_params"] = list(self.free_params)
        if self.fixed_params:
            d["fixed_params"] = list(self.fixed_params)
        if self.bound_params:
            d["bound_params"] = {str(k): str(v) for k, v in self.bound_params}
        if self.min_r2 is not None:
            d["min_r2"] = float(self.min_r2)
        if self.label:
            d["label"] = str(self.label)
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> "FitProcedureStep":
        raw_bound = data.get("bound_params")
        bound: Tuple[Tuple[str, str], ...] = ()
        if isinstance(raw_bound, Mapping):
            bound = tuple(
                (str(k), str(v)) for k, v in raw_bound.items() if v not in (None, "")
            )
        return cls(
            channels=tuple(str(c) for c in (data.get("channels") or ())),
            free_params=tuple(str(p) for p in (data.get("free_params") or ())),
            fixed_params=tuple(str(p) for p in (data.get("fixed_params") or ())),
            bound_params=bound,
            min_r2=finite_float_or_none(data.get("min_r2")),
            label=str(data.get("label") or ""),
        )


@dataclass(frozen=True)
class FitProcedure:
    """An ordered sequence of :class:`FitProcedureStep` instances."""

    name: str = "Procedure"
    steps: Tuple[FitProcedureStep, ...] = ()

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "steps": [s.serialize() for s in self.steps],
        }

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> "FitProcedure":
        return cls(
            name=str(data.get("name") or "Procedure"),
            steps=tuple(
                FitProcedureStep.deserialize(s)
                for s in (data.get("steps") or ())
                if isinstance(s, Mapping)
            ),
        )


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

    candidate_r2 = finite_float_or_none(candidate_row.get("r2"))
    baseline_r2 = finite_float_or_none(baseline_row.get("r2"))
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
        members = tuple(sorted(set((str(t), int(i)) for t, i in group)))
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


def _run_initial_fit_stage(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    segments: Sequence[SegmentSpec],
    local_seed_flat: np.ndarray,
    n_boundaries: int,
    max_nfev: int,
) -> "OrderedPiecewiseResult":
    """Run coarse grid search (when boundaries exist) or seed-based fit.

    Falls back to seed-based refinement if grid search fails.
    """
    if n_boundaries > 0:
        stride = max(12, x_arr.size // 150)
        try:
            return fit_ordered_piecewise(
                x_arr,
                y_arr,
                segments,
                config=OrderedPiecewiseConfig(
                    grid_stride=stride,
                    robust_max_nfev=max_nfev,
                    prefer_jit=True,
                ),
            )
        except Exception:
            pass  # fall through to seed-based refinement
    return refit_ordered_piecewise_from_seed(
        x_arr,
        y_arr,
        segments,
        seed_params=local_seed_flat,
        config=OrderedPiecewiseConfig(
            robust_max_nfev=max_nfev,
            prefer_jit=True,
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
    n_random_restarts: int = 0,
    rng_seed: Optional[int] = None,
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

    segments = make_segment_specs(
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
    stage_a = _run_initial_fit_stage(
        x_arr,
        y_arr,
        segments,
        local_seed_flat,
        n_boundaries,
        stage_a_nfev,
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

    _blend_w = _auto_blend_width(np.sort(x_arr)) if n_boundaries > 0 else 0.0

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
        except Exception:
            pass

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
    if n_restarts > 0:
        rng = np.random.default_rng(rng_seed)
        best_r2 = best_result.get("r2")
        for _ in range(n_restarts):
            if is_cancelled():
                break
            restart_seed = _random_seed_within_bounds(
                seed_map, bounds_map, fixed_params, rng
            )
            restart_boundary: Optional[np.ndarray] = (
                np.asarray(
                    rng.uniform(0.0, 1.0, size=int(n_boundaries)), dtype=float
                )
                if n_boundaries > 0
                else np.asarray([], dtype=float)
            )
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
                    n_random_restarts=0,
                )
            except FitCancelledError:
                raise
            except Exception:
                continue
            candidate_r2 = candidate.get("r2")
            if candidate_r2 is not None and (
                best_r2 is None or float(candidate_r2) > float(best_r2)
            ):
                best_result = candidate
                best_r2 = float(candidate_r2)

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
    n_random_restarts: int = 0,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Fit shared parameters across multiple channel equations simultaneously.

    Each channel equation is a piecewise model with its own boundary ratios.
    Parameters with the same name are shared across channels.
    Returns per-channel results plus combined metrics.
    """
    x_arr = np.asarray(x_data, dtype=float).reshape(-1)
    if x_arr.size < 8:
        raise ValueError("Not enough points to run multi-channel fitting.")

    global_names = list(multi_model.global_param_names)
    global_index = {name: idx for idx, name in enumerate(global_names)}
    boundary_seeds_map = dict(boundary_seeds or {})

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
            ch_model, seed_by_key, bounds_map, fixed_param_values=fixed_by_key
        )
        n_boundaries = max(0, len(segments) - 1)
        b_seed = boundary_seeds_map.get(target)
        if b_seed is None:
            b_seed_arr = default_boundary_ratios(n_boundaries)
        else:
            b_seed_arr = np.clip(np.asarray(b_seed, dtype=float).reshape(-1), 0.0, 1.0)
            if b_seed_arr.size != n_boundaries:
                b_seed_arr = default_boundary_ratios(n_boundaries)

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
                "local_to_global": local_to_global_arr,
                "n_local_params": int(local_to_global_arr.size),
            }
        )

    # Build shared optimization vector: [free_params, boundary_ch1, boundary_ch2, ...]
    shared_seed = np.asarray(
        [float(seed_by_key[name]) for name in global_names], dtype=float
    )
    n_free = int(free_param_indices.size)

    # Stage A: per-channel initial fit, then average to get shared seed.
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
        stage = _run_initial_fit_stage(
            x_arr,
            ch["y_arr"],
            ch["segments"],
            local_seed,
            n_b,
            nfev,
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

    bid_to_var: Dict[Tuple[str, int], int] = {}
    linked_bids: set = set()
    boundary_var_count = 0
    for group in multi_model.boundary_links:
        valid = [bid for bid in group if bid in all_bids_set]
        if len(valid) < 2:
            continue
        for bid in valid:
            bid_to_var[bid] = boundary_var_count
            linked_bids.add(bid)
        boundary_var_count += 1
    for bid in all_bids:
        if bid not in bid_to_var:
            bid_to_var[bid] = boundary_var_count
            boundary_var_count += 1
    total_boundary = boundary_var_count

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
                local_flat[n_local:] = ch["boundary_seed"]
            pred = predict_ordered_piecewise(
                x_arr, ch["segments"], local_flat, prefer_jit=True
            )
            results_by_channel[ch["target"]] = {
                "y_hat": np.asarray(pred["y_hat"], dtype=float),
                "boundary_ratios": np.asarray([], dtype=float),
                "boundaries": np.asarray(pred["boundaries"], dtype=float),
                "r2": compute_r2(ch["y_arr"], pred["y_hat"]),
            }
        return {
            "params_by_key": {
                name: float(shared_init[idx]) for idx, name in enumerate(global_names)
            },
            "params_vector": np.asarray(shared_init, dtype=float),
            "channel_results": results_by_channel,
            "r2": _combine_channel_r2(results_by_channel),
        }

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

    def residuals(vals: np.ndarray) -> np.ndarray:
        if is_cancelled():
            raise FitCancelledError("cancelled")
        vals = np.asarray(vals, dtype=float).reshape(-1)
        shared_vals = _compose_shared(vals[:n_free])
        all_residuals: List[np.ndarray] = []
        for ch in channel_info:
            n_local = ch["n_local_params"]
            n_b = ch["n_boundaries"]
            local_flat = np.empty(n_local + n_b, dtype=float)
            if n_local > 0:
                local_flat[:n_local] = shared_vals[ch["local_to_global"]]
            for i in range(n_b):
                var_idx = bid_to_var[(ch["target"], i)]
                local_flat[n_local + i] = vals[n_free + var_idx]
            pred = predict_ordered_piecewise(
                x_arr,
                ch["segments"],
                local_flat,
                prefer_jit=True,
                blend_width=_mc_blend_w,
            )
            all_residuals.append(np.asarray(pred["y_hat"] - ch["y_arr"], dtype=float))
        return np.concatenate(all_residuals)

    # Run combined least-squares optimization.
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
    except FitCancelledError:
        raise
    except Exception:
        refined_vals = x0

    best_shared = np.asarray(_compose_shared(refined_vals[:n_free]), dtype=float).copy()

    # Extract per-channel results.
    results_by_channel = {}
    for ch in channel_info:
        n_local = ch["n_local_params"]
        n_b = ch["n_boundaries"]
        local_flat = np.empty(n_local + n_b, dtype=float)
        if n_local > 0:
            local_flat[:n_local] = best_shared[ch["local_to_global"]]
        ch_ratios = np.zeros(n_b, dtype=float)
        for i in range(n_b):
            var_idx = bid_to_var[(ch["target"], i)]
            val = float(refined_vals[n_free + var_idx])
            local_flat[n_local + i] = val
            ch_ratios[i] = val
        pred = predict_ordered_piecewise(
            x_arr, ch["segments"], local_flat, prefer_jit=True
        )
        results_by_channel[ch["target"]] = {
            "y_hat": np.asarray(pred["y_hat"], dtype=float),
            "boundary_ratios": np.asarray(ch_ratios, dtype=float),
            "boundaries": np.asarray(pred["boundaries"], dtype=float),
            "r2": compute_r2(ch["y_arr"], pred["y_hat"]),
        }

    best_result = {
        "params_by_key": {
            name: float(best_shared[idx]) for idx, name in enumerate(global_names)
        },
        "params_vector": np.asarray(best_shared, dtype=float),
        "channel_results": results_by_channel,
        "r2": _combine_channel_r2(results_by_channel),
    }

    # ── Random restarts: try N additional random starting points ──────
    n_restarts = int(n_random_restarts)
    if n_restarts > 0:
        rng = np.random.default_rng(rng_seed)
        best_r2 = best_result.get("r2")
        for _ in range(n_restarts):
            if is_cancelled():
                break
            restart_seed = _random_seed_within_bounds(
                seed_map, bounds_map, fixed_params, rng
            )
            restart_boundary_seeds: Dict[str, np.ndarray] = {}
            for ch in channel_info:
                n_b = ch["n_boundaries"]
                restart_boundary_seeds[ch["target"]] = (
                    np.asarray(
                        rng.uniform(0.0, 1.0, size=n_b), dtype=float
                    )
                    if n_b > 0
                    else np.asarray([], dtype=float)
                )
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
                    n_random_restarts=0,
                )
            except FitCancelledError:
                raise
            except Exception:
                continue
            candidate_r2 = candidate.get("r2")
            if candidate_r2 is not None and (
                best_r2 is None or float(candidate_r2) > float(best_r2)
            ):
                best_result = candidate
                best_r2 = float(candidate_r2)

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


# ── Procedure pipeline ──────────────────────────────────────────────


def run_procedure_pipeline(
    x_data: np.ndarray,
    y_data_by_channel: Mapping[str, np.ndarray],
    multi_model: MultiChannelModelDefinition,
    procedure: FitProcedure,
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    boundary_seeds: Optional[Mapping[str, np.ndarray]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    step_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    bound_values: Optional[Mapping[str, float]] = None,
    n_random_restarts: int = 0,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a multi-step fitting procedure, feeding results forward.

    Parameters
    ----------
    step_callback : callable(step_index, step_result) or None
        Called after each completed step so the GUI can report progress.

    Returns
    -------
    dict with keys:
        - ``step_results``: list of per-step result dicts
        - ``params_by_key``: final merged parameter values
        - ``r2``: final combined R²
        - ``channel_results``: final per-channel breakdown
        - ``stopped_at_step``: index of the step that caused early stop, or None
    """
    x_arr = np.asarray(x_data, dtype=float).reshape(-1)
    running_seed = dict(seed_map)
    running_boundary_seeds: Dict[str, np.ndarray] = dict(boundary_seeds or {})
    step_results: List[Dict[str, Any]] = []
    last_result: Optional[Dict[str, Any]] = None
    stopped_at_step: Optional[int] = None

    global_names = set(multi_model.global_param_names)

    def _is_cancelled() -> bool:
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except Exception:
            return False

    for step_idx, step in enumerate(procedure.steps):
        if _is_cancelled():
            raise FitCancelledError("cancelled")

        # --- Determine fixed params for this step ---
        # If free_params is specified, everything else is fixed.
        # If fixed_params is specified, everything else is free.
        # If both are specified, free_params takes priority.
        fixed_for_step: Dict[str, float] = {}

        # Apply bound params first (param → capture field → numeric value).
        _bound_vals = dict(bound_values or {})
        for param_key, field_name in step.bound_params:
            if param_key in global_names and field_name in _bound_vals:
                fixed_for_step[param_key] = float(_bound_vals[field_name])
                running_seed[param_key] = float(_bound_vals[field_name])

        if step.free_params:
            free_set = set(step.free_params) & global_names
            for name in global_names:
                if name not in free_set:
                    fixed_for_step[name] = float(running_seed.get(name, 0.0))
        elif step.fixed_params:
            for name in step.fixed_params:
                if name in global_names:
                    fixed_for_step[name] = float(running_seed.get(name, 0.0))

        # --- Filter to step channels ---
        step_channels = set(step.channels) if step.channels else None
        if step_channels:
            enabled_models = tuple(
                m for m in multi_model.channel_models if m.target_col in step_channels
            )
        else:
            enabled_models = multi_model.channel_models

        if not enabled_models:
            # No matching channels; skip step but record it.
            step_results.append({"skipped": True, "reason": "no matching channels"})
            if step_callback is not None:
                step_callback(step_idx, step_results[-1])
            continue

        # Build a filtered multi-model for this step's channels.
        enabled_targets = {m.target_col for m in enabled_models}
        filtered_links: List[Tuple[Tuple[str, int], ...]] = []
        for group in multi_model.boundary_links:
            filtered = tuple(bid for bid in group if bid[0] in enabled_targets)
            if len(filtered) >= 2:
                filtered_links.append(filtered)
        filtered_global: List[str] = []
        seen_global: set = set()
        for m in enabled_models:
            for seg_names in m.segment_param_names:
                for name in seg_names:
                    if name not in seen_global:
                        seen_global.add(name)
                        filtered_global.append(name)
        step_multi = MultiChannelModelDefinition(
            channel_models=enabled_models,
            global_param_names=tuple(filtered_global),
            boundary_links=tuple(filtered_links),
        )

        # Gather y-data for step channels.
        step_y = {
            ch: y_data_by_channel[ch]
            for ch in enabled_targets
            if ch in y_data_by_channel
        }
        if not step_y:
            step_results.append({"skipped": True, "reason": "no y-data for channels"})
            if step_callback is not None:
                step_callback(step_idx, step_results[-1])
            continue

        # Filter boundary seeds to step channels.
        step_boundary_seeds = {
            ch: running_boundary_seeds[ch]
            for ch in enabled_targets
            if ch in running_boundary_seeds
        }

        # Filter fixed params to only those in the step model's global names.
        step_fixed_filtered = {
            k: v for k, v in fixed_for_step.items() if k in seen_global
        }

        # Run the fit.
        if step_multi.is_multi_channel:
            result = run_multi_channel_fit_pipeline(
                x_arr,
                step_y,
                step_multi,
                running_seed,
                bounds_map,
                boundary_seeds=step_boundary_seeds,
                cancel_check=cancel_check,
                fixed_params=step_fixed_filtered,
                n_random_restarts=int(n_random_restarts),
                rng_seed=rng_seed,
            )
        else:
            # Single-channel path.
            ch_model = enabled_models[0]
            ch_target = ch_model.target_col
            b_seed = step_boundary_seeds.get(ch_target)
            result = run_piecewise_fit_pipeline(
                x_arr,
                step_y[ch_target],
                ch_model,
                running_seed,
                bounds_map,
                boundary_seed=b_seed,
                cancel_check=cancel_check,
                fixed_params=step_fixed_filtered,
                n_random_restarts=int(n_random_restarts),
                rng_seed=rng_seed,
            )

        last_result = result

        # Feed results forward into the running seed.
        params_by_key = dict(result.get("params_by_key") or {})
        for key, value in params_by_key.items():
            running_seed[key] = float(value)

        # Update boundary seeds from channel results.
        channel_results = result.get("channel_results")
        if isinstance(channel_results, dict):
            for ch_target, ch_result in channel_results.items():
                ch_ratios = ch_result.get("boundary_ratios")
                if ch_ratios is not None:
                    running_boundary_seeds[ch_target] = np.asarray(
                        ch_ratios, dtype=float
                    )
        elif result.get("boundary_ratios") is not None:
            ch_target = enabled_models[0].target_col
            running_boundary_seeds[ch_target] = np.asarray(
                result["boundary_ratios"], dtype=float
            )

        step_r2 = result.get("r2")
        step_result_entry = {
            "step_index": step_idx,
            "label": step.label or f"Step {step_idx + 1}",
            "params_by_key": dict(params_by_key),
            "r2": float(step_r2) if step_r2 is not None else None,
            "channels": list(enabled_targets),
            "free_params": list(step.free_params)
            if step.free_params
            else list(seen_global - set(fixed_for_step)),
            "fixed_params": list(fixed_for_step.keys()),
        }
        step_results.append(step_result_entry)

        if step_callback is not None:
            step_callback(step_idx, step_result_entry)

        # Check R² threshold.
        if step.min_r2 is not None and step_r2 is not None:
            if float(step_r2) < float(step.min_r2):
                stopped_at_step = step_idx
                break

    # Build final output.
    final_channel_results = {}
    if last_result is not None:
        final_channel_results = last_result.get("channel_results") or {}

    return {
        "step_results": step_results,
        "params_by_key": dict(running_seed),
        "r2": last_result.get("r2") if last_result is not None else None,
        "channel_results": final_channel_results,
        "stopped_at_step": stopped_at_step,
    }
