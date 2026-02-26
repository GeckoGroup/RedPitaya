"""Polymorphic procedure step types with a plugin-style registry."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import math
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import numpy as np


_SCALAR_EXPRESSION_ALLOWED_FUNCTIONS: Dict[str, Any] = {
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "power": np.power,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}
_SCALAR_EXPRESSION_ALLOWED_CONSTANTS: Dict[str, float] = {
    "pi": float(np.pi),
    "e": float(np.e),
}
_SCALAR_EXPRESSION_ALLOWED_NODE_TYPES: Tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)
_ASCII_TO_SUBSCRIPT_TRANS = str.maketrans(
    {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "-": "₋",
        "+": "₊",
    }
)
_SUBSCRIPT_TO_ASCII_TRANS = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₋": "-",
        "₊": "+",
    }
)


def _finite_float_or_none(value) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _default_boundary_ratios_local(n_boundaries: int) -> np.ndarray:
    n = int(max(0, n_boundaries))
    if n <= 0:
        return np.asarray([], dtype=float)
    pcts = np.linspace(1.0 / (n + 1), n / (n + 1), n)
    prev = 0.0
    out = []
    for pct in pcts:
        denom = max(1e-12, 1.0 - prev)
        ratio = (float(pct) - prev) / denom
        ratio = float(np.clip(ratio, 0.0, 1.0))
        out.append(ratio)
        prev = prev + ratio * (1.0 - prev)
    return np.asarray(out, dtype=float)


def _normalise_boundary_name_map(
    mapping: Optional[Mapping[str, Sequence[Sequence[Any]]]],
) -> Dict[str, Tuple[Tuple[str, int], ...]]:
    out: Dict[str, Tuple[Tuple[str, int], ...]] = {}
    raw = dict(mapping or {})
    for raw_name, raw_members in raw.items():
        name = str(raw_name).strip()
        if not name:
            continue
        members: List[Tuple[str, int]] = []
        for member in raw_members or ():
            if not isinstance(member, (tuple, list)) or len(member) != 2:
                continue
            try:
                target = str(member[0]).strip()
                index = int(member[1])
            except Exception:
                continue
            if not target or index < 0:
                continue
            bid = (target, index)
            if bid not in members:
                members.append(bid)
        if members:
            out[name] = tuple(members)
    return out


def _compile_scalar_expression(
    expression_text: str,
) -> Tuple[Callable[[Mapping[str, float]], float], Tuple[str, ...]]:
    text = str(expression_text).strip()
    if not text:
        raise ValueError("Expression is empty.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc.msg}") from exc

    variable_names: List[str] = []
    seen_names: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, _SCALAR_EXPRESSION_ALLOWED_NODE_TYPES):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed.")
            if node.func.id not in _SCALAR_EXPRESSION_ALLOWED_FUNCTIONS:
                raise ValueError(f"Unsupported function '{node.func.id}'.")
        if isinstance(node, ast.Name):
            name = str(node.id)
            if (
                name in _SCALAR_EXPRESSION_ALLOWED_FUNCTIONS
                or name in _SCALAR_EXPRESSION_ALLOWED_CONSTANTS
            ):
                continue
            if name not in seen_names:
                seen_names.add(name)
                variable_names.append(name)

    code = compile(tree, "<boundary_expression>", "eval")
    eval_globals: Dict[str, Any] = {
        "__builtins__": {},
        "math": math,
    }
    eval_globals.update(_SCALAR_EXPRESSION_ALLOWED_FUNCTIONS)
    eval_globals.update(_SCALAR_EXPRESSION_ALLOWED_CONSTANTS)

    def _evaluate(values_by_name: Mapping[str, float]) -> float:
        eval_locals: Dict[str, float] = {}
        for name in variable_names:
            if name not in values_by_name:
                raise ValueError(f"Missing '{name}' in expression context.")
            numeric = _finite_float_or_none(values_by_name.get(name))
            if numeric is None:
                raise ValueError(f"Non-numeric value for '{name}'.")
            eval_locals[name] = float(numeric)
        try:
            result = eval(code, eval_globals, eval_locals)
        except Exception as exc:
            raise ValueError(f"Expression evaluation failed: {exc}") from exc
        numeric_result = _finite_float_or_none(result)
        if numeric_result is None:
            raise ValueError("Expression result is not finite.")
        return float(numeric_result)

    return _evaluate, tuple(variable_names)


def _boundary_counts_by_target(multi_model: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ch_model in tuple(getattr(multi_model, "channel_models", ()) or ()):
        target = str(getattr(ch_model, "target_col", "") or "").strip()
        if not target:
            continue
        n_boundaries = max(0, len(getattr(ch_model, "segment_exprs", ()) or ()) - 1)
        out[target] = int(n_boundaries)
    return out


def _boundary_name_aliases(name: str) -> Tuple[str, ...]:
    base = str(name or "").strip()
    if not base:
        return ()
    aliases: List[str] = []
    for candidate in (
        base,
        str(base).translate(_SUBSCRIPT_TO_ASCII_TRANS),
        str(base).translate(_ASCII_TO_SUBSCRIPT_TRANS),
    ):
        key = str(candidate).strip()
        if key and key not in aliases:
            aliases.append(key)
    return tuple(aliases)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    """Outcome of a single procedure step execution."""

    status: str = "pass"  # "pass" | "fail" | "skipped"
    message: str = ""
    params_by_key: Dict[str, float] = field(default_factory=dict)
    boundary_ratios: Optional[Dict[str, Any]] = None  # {target_col: np.ndarray}
    r2: Optional[float] = None
    per_channel_r2: Optional[Dict[str, Optional[float]]] = None
    channels: Tuple[str, ...] = ()
    free_params: Tuple[str, ...] = ()
    fixed_params: Tuple[str, ...] = ()
    retries_used: int = 0
    retry_r2_history: Tuple[float, ...] = ()


# ---------------------------------------------------------------------------
# Execution context  (mutable bag carried across steps)
# ---------------------------------------------------------------------------


class ProcedureContext:
    """Mutable state passed through procedure steps."""

    def __init__(
        self,
        *,
        seed_map: Dict[str, float],
        bounds_map: Dict[str, Tuple[float, float]],
        boundary_seeds: Dict[str, Any],
        x_data: Any,
        y_data_by_channel: Dict[str, Any],
        multi_model: Any,
        global_names: set,
        bound_values: Optional[Dict[str, float]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        rng_seed: Optional[int] = None,
        boundary_name_to_ids: Optional[Mapping[str, Sequence[Sequence[Any]]]] = None,
        use_jax: bool = False,
        # -- Cross-file sibling seeding --
        captures: Optional[Dict[str, Any]] = None,
        sibling_results: Optional[Mapping[str, Mapping[str, Any]]] = None,
        capture_seed_keys: Optional[Sequence[str]] = None,
        seed_from_siblings: bool = False,
    ):
        self.seed_map = dict(seed_map)
        self.bounds_map = dict(bounds_map)
        self.boundary_seeds = {
            str(k): np.asarray(v, dtype=float).reshape(-1)
            for k, v in dict(boundary_seeds).items()
        }
        self.x_data = np.asarray(x_data, dtype=float).reshape(-1)
        self.y_data_by_channel = {
            str(k): np.asarray(v, dtype=float) for k, v in y_data_by_channel.items()
        }
        self.multi_model = multi_model
        self.global_names = set(global_names)
        self.bound_values = dict(bound_values or {})
        self.cancel_check = cancel_check
        self.rng_seed = rng_seed
        self.boundary_name_to_ids = _normalise_boundary_name_map(boundary_name_to_ids)
        self.use_jax = bool(use_jax)
        # Cross-file sibling seeding state.
        self.captures: Dict[str, Any] = dict(captures or {})
        self.sibling_results: Dict[str, Dict[str, Any]] = dict(sibling_results or {})
        self.capture_seed_keys: Tuple[str, ...] = tuple(capture_seed_keys or ())
        self.seed_from_siblings: bool = bool(seed_from_siblings)
        self.step_index: int = 0  # set by pipeline before each step
        self.step_total: int = 0  # set by pipeline before first step
        # Per-attempt callback set by run_procedure_pipeline.
        self.attempt_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = (
            None
        )

    def is_cancelled(self) -> bool:
        if self.cancel_check is None:
            return False
        try:
            return bool(self.cancel_check())
        except Exception:
            return False

    def ensure_boundary_seed_size(self, target: str, n_boundaries: int) -> np.ndarray:
        arr = np.asarray(
            self.boundary_seeds.get(
                target, _default_boundary_ratios_local(n_boundaries)
            ),
            dtype=float,
        ).reshape(-1)
        if arr.size != int(max(0, n_boundaries)):
            arr = _default_boundary_ratios_local(n_boundaries)
        arr = np.clip(arr, 0.0, 1.0)
        self.boundary_seeds[str(target)] = np.asarray(arr, dtype=float)
        return self.boundary_seeds[str(target)]


# ---------------------------------------------------------------------------
# Base class + registry
# ---------------------------------------------------------------------------

# Global registry: step_type string → class
_STEP_TYPE_REGISTRY: Dict[str, Type[ProcedureStepBase]] = {}


def register_step_type(cls: Type[ProcedureStepBase]) -> Type[ProcedureStepBase]:
    """Class decorator that registers a step type."""
    _STEP_TYPE_REGISTRY[cls.step_type] = cls
    return cls


def deserialize_step(data: Mapping[str, Any]) -> ProcedureStepBase:
    """Deserialise a step from a dict, dispatching to the correct subclass."""
    step_type = str(data.get("step_type") or "").strip()
    if not step_type:
        raise ValueError("Missing step_type.")
    cls = _STEP_TYPE_REGISTRY.get(step_type)
    if cls is None:
        raise ValueError(f"Unknown step type: {step_type!r}")
    return cls.deserialize(data)


def available_step_types() -> List[Tuple[str, str]]:
    """Return (step_type, label) pairs for all registered step types."""
    return [(cls.step_type, cls.step_label) for cls in _STEP_TYPE_REGISTRY.values()]


@dataclass(frozen=True)
class ProcedureStepBase:
    """Abstract base for all procedure step types."""

    step_type: ClassVar[str] = "base"
    step_label: ClassVar[str] = "Base Step"

    label: str = ""

    def serialize(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"step_type": self.step_type}
        if self.label:
            d["label"] = str(self.label)
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> ProcedureStepBase:
        return cls(label=str(data.get("label") or ""))

    def execute(self, context: ProcedureContext) -> StepResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fit step
# ---------------------------------------------------------------------------


@register_step_type
@dataclass(frozen=True)
class FitStep(ProcedureStepBase):
    """Run curve-fitting on selected channels with selected free parameters.

    Channel selection semantics
    --------------------------
    - ``channels = None``  → use **all** channels (default, backward compat).
    - ``channels = ()``    → **no** channels selected — boundary-only fit.
      All parameters are forced fixed from the current seed map and only
      boundary ratios are optimised, using every channel's y-data.
    - ``channels = ("ch1", ...)`` → use only the listed channels.
    """

    step_type: ClassVar[str] = "fit"
    step_label: ClassVar[str] = "Fit"

    channels: Optional[Tuple[str, ...]] = None
    free_params: Tuple[str, ...] = ()
    fixed_params: Tuple[str, ...] = ()
    # Capture/field values that seed parameters before this fit step.
    # They do not force-fix by themselves; fixed/free controls decide that.
    bound_params: Tuple[Tuple[str, str], ...] = ()  # ((param_key, field_name), ...)
    min_r2: Optional[float] = None
    max_retries: int = 0
    retry_scale: float = 0.3
    retry_mode: str = "jitter_then_random"  # jitter | random | jitter_then_random
    locked_boundary_names: Tuple[str, ...] = ()
    on_fail: str = "stop"  # stop | continue

    def serialize(self) -> Dict[str, Any]:
        d = super().serialize()
        if self.channels is not None:
            d["channels"] = list(self.channels)
        if self.free_params:
            d["free_params"] = list(self.free_params)
        if self.fixed_params:
            d["fixed_params"] = list(self.fixed_params)
        if self.bound_params:
            d["bound_params"] = {str(k): str(v) for k, v in self.bound_params}
        if self.min_r2 is not None:
            d["min_r2"] = float(self.min_r2)
        if self.max_retries > 0:
            d["max_retries"] = int(self.max_retries)
        if self.retry_scale != 0.3:
            d["retry_scale"] = float(self.retry_scale)
        if self.retry_mode != "jitter_then_random":
            d["retry_mode"] = str(self.retry_mode)
        if self.locked_boundary_names:
            d["locked_boundary_names"] = [
                str(name) for name in self.locked_boundary_names
            ]
        if self.on_fail != "stop":
            d["on_fail"] = str(self.on_fail)
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> FitStep:
        raw_bound = data.get("bound_params")
        bound: Tuple[Tuple[str, str], ...] = ()
        if isinstance(raw_bound, Mapping):
            bound = tuple(
                (str(k), str(v)) for k, v in raw_bound.items() if v not in (None, "")
            )
        retry_mode = str(data.get("retry_mode") or "jitter_then_random").strip()
        if retry_mode not in {"jitter", "random", "jitter_then_random"}:
            retry_mode = "jitter_then_random"
        raw_locked = data.get("locked_boundary_names")
        if raw_locked is None:
            raw_locked = data.get("boundary_names")
        locked_boundary_names: Tuple[str, ...] = ()
        if isinstance(raw_locked, Sequence) and not isinstance(
            raw_locked, (str, bytes)
        ):
            seen_locked: List[str] = []
            for raw_name in raw_locked:
                name = str(raw_name).strip()
                if name and name not in seen_locked:
                    seen_locked.append(name)
            locked_boundary_names = tuple(seen_locked)
        on_fail = str(data.get("on_fail") or "stop").strip()
        if on_fail not in {"stop", "continue"}:
            on_fail = "stop"
        raw_channels = data.get("channels", None)
        if raw_channels is None:
            channels: Optional[Tuple[str, ...]] = None  # key absent → all channels
        else:
            channels = tuple(str(c) for c in raw_channels)  # [] → () = no channels
        return cls(
            channels=channels,
            free_params=tuple(str(p) for p in (data.get("free_params") or ())),
            fixed_params=tuple(str(p) for p in (data.get("fixed_params") or ())),
            bound_params=bound,
            min_r2=_finite_float_or_none(data.get("min_r2")),
            max_retries=max(0, int(data.get("max_retries") or 0)),
            retry_scale=float(data.get("retry_scale", 0.3)),
            retry_mode=retry_mode,
            locked_boundary_names=locked_boundary_names,
            on_fail=on_fail,
            label=str(data.get("label") or ""),
        )

    def execute(self, context: ProcedureContext) -> StepResult:
        # Lazy import to avoid circular dependency.
        from procedure import _execute_fit_step

        return _execute_fit_step(self, context)


# ---------------------------------------------------------------------------
# Set Parameter step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterAssignment:
    """Single parameter assignment entry for ``SetParameterStep``."""

    target_key: str = ""
    source_kind: str = "literal"  # literal | param | capture
    source_key: str = ""
    literal_value: Optional[float] = None
    scale: float = 1.0
    offset: float = 0.0
    clamp_to_bounds: bool = True
    on_missing: str = "skip"  # skip | fail

    def serialize(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "target_key": str(self.target_key),
            "source_kind": str(self.source_kind),
            "source_key": str(self.source_key),
            "scale": float(self.scale),
            "offset": float(self.offset),
            "clamp_to_bounds": bool(self.clamp_to_bounds),
            "on_missing": str(self.on_missing),
        }
        if self.literal_value is not None:
            data["literal_value"] = float(self.literal_value)
        return data

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> ParameterAssignment:
        target_key = str(data.get("target_key") or "").strip()
        source_kind = str(data.get("source_kind") or "literal").strip()
        if source_kind not in {"literal", "param", "capture"}:
            source_kind = "literal"
        source_key = str(data.get("source_key") or "").strip()
        literal_value = _finite_float_or_none(data.get("literal_value"))
        scale = _finite_float_or_none(data.get("scale"))
        offset = _finite_float_or_none(data.get("offset"))
        on_missing = str(data.get("on_missing") or "skip").strip()
        if on_missing not in {"skip", "fail"}:
            on_missing = "skip"
        return cls(
            target_key=target_key,
            source_kind=source_kind,
            source_key=source_key,
            literal_value=literal_value,
            scale=1.0 if scale is None else float(scale),
            offset=0.0 if offset is None else float(offset),
            clamp_to_bounds=bool(data.get("clamp_to_bounds", True)),
            on_missing=on_missing,
        )


@register_step_type
@dataclass(frozen=True)
class SetParameterStep(ProcedureStepBase):
    """Set parameter seeds from literals, existing params, or capture fields."""

    step_type: ClassVar[str] = "set_parameter"
    step_label: ClassVar[str] = "Set Parameter"

    assignments: Tuple[ParameterAssignment, ...] = ()

    def serialize(self) -> Dict[str, Any]:
        d = super().serialize()
        if self.assignments:
            d["assignments"] = [a.serialize() for a in self.assignments]
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> SetParameterStep:
        assignments: List[ParameterAssignment] = []
        raw_assign = data.get("assignments")
        if isinstance(raw_assign, Sequence):
            for entry in raw_assign:
                if not isinstance(entry, Mapping):
                    continue
                assignment = ParameterAssignment.deserialize(entry)
                if assignment.target_key:
                    assignments.append(assignment)
        return cls(
            assignments=tuple(assignments),
            label=str(data.get("label") or ""),
        )

    def execute(self, context: ProcedureContext) -> StepResult:
        changed: Dict[str, float] = {}
        warnings: List[str] = []
        for assignment in self.assignments:
            key = str(assignment.target_key).strip()
            if not key or key not in context.global_names:
                continue

            source_value: Optional[float] = None
            source_label = ""
            if assignment.source_kind == "literal":
                source_value = (
                    float(assignment.literal_value)
                    if assignment.literal_value is not None
                    else None
                )
                source_label = "literal"
            elif assignment.source_kind == "param":
                source_label = f"param:{assignment.source_key}"
                source_value = _finite_float_or_none(
                    context.seed_map.get(str(assignment.source_key))
                )
            elif assignment.source_kind == "capture":
                source_label = f"capture:{assignment.source_key}"
                source_value = _finite_float_or_none(
                    context.bound_values.get(str(assignment.source_key))
                )
            else:
                source_label = assignment.source_kind

            if source_value is None:
                msg = f"{key}: missing source ({source_label})."
                if assignment.on_missing == "fail":
                    return StepResult(
                        status="fail",
                        message=msg,
                        params_by_key=dict(context.seed_map),
                    )
                warnings.append(msg)
                continue

            out_value = float(
                source_value * float(assignment.scale) + assignment.offset
            )
            if assignment.clamp_to_bounds and key in context.bounds_map:
                low, high = context.bounds_map[key]
                if low > high:
                    low, high = high, low
                out_value = float(np.clip(out_value, low, high))

            if np.isfinite(out_value):
                context.seed_map[key] = out_value
                changed[key] = out_value

        message = f"Set {len(changed)} parameter(s)."
        if warnings:
            message += f" Skipped {len(warnings)} missing source(s)."
        return StepResult(
            status="pass",
            message=message,
            params_by_key=dict(context.seed_map),
        )


# ---------------------------------------------------------------------------
# Set Boundaries step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryAssignment:
    """Single boundary assignment entry for ``SetBoundariesStep``."""

    target_name: str = ""
    source_kind: str = "literal"  # literal | boundary | expression
    source_name: str = ""
    literal_value: Optional[float] = None
    expression: str = ""
    on_missing: str = "skip"  # skip | fail

    def serialize(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "target_name": str(self.target_name),
            "source_kind": str(self.source_kind),
            "source_name": str(self.source_name),
            "expression": str(self.expression),
            "on_missing": str(self.on_missing),
        }
        if self.literal_value is not None:
            data["literal_value"] = float(self.literal_value)
        return data

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> BoundaryAssignment:
        target_name = str(
            data.get("target_name") or data.get("target_key") or ""
        ).strip()
        source_kind = str(data.get("source_kind") or "literal").strip().lower()
        if source_kind == "expr":
            source_kind = "expression"
        if source_kind not in {"literal", "boundary", "expression"}:
            source_kind = "literal"
        source_name = str(
            data.get("source_name") or data.get("source_key") or ""
        ).strip()
        literal_value = _finite_float_or_none(data.get("literal_value"))
        expression = str(data.get("expression") or data.get("expr") or "").strip()
        on_missing = str(data.get("on_missing") or "skip").strip().lower()
        if on_missing not in {"skip", "fail"}:
            on_missing = "skip"
        return cls(
            target_name=target_name,
            source_kind=source_kind,
            source_name=source_name,
            literal_value=literal_value,
            expression=expression,
            on_missing=on_missing,
        )


@register_step_type
@dataclass(frozen=True)
class SetBoundariesStep(ProcedureStepBase):
    """Set boundary-group ratios from literals, other boundaries, or expressions."""

    step_type: ClassVar[str] = "set_boundaries"
    step_label: ClassVar[str] = "Set Boundaries"

    assignments: Tuple[BoundaryAssignment, ...] = ()

    def serialize(self) -> Dict[str, Any]:
        d = super().serialize()
        if self.assignments:
            d["assignments"] = [a.serialize() for a in self.assignments]
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> SetBoundariesStep:
        assignments: List[BoundaryAssignment] = []
        raw_assign = data.get("assignments")
        if isinstance(raw_assign, Sequence):
            for entry in raw_assign:
                if not isinstance(entry, Mapping):
                    continue
                assignment = BoundaryAssignment.deserialize(entry)
                if assignment.target_name:
                    assignments.append(assignment)
        return cls(
            assignments=tuple(assignments),
            label=str(data.get("label") or ""),
        )

    def execute(self, context: ProcedureContext) -> StepResult:
        counts_by_target = _boundary_counts_by_target(context.multi_model)
        if not counts_by_target:
            return StepResult(
                status="skipped",
                message="No boundary channels available.",
                params_by_key=dict(context.seed_map),
            )

        available_groups = dict(context.boundary_name_to_ids or {})
        resolved_name_cache: Dict[str, str] = {}

        def _resolve_group_name(raw_name: str) -> str:
            key = str(raw_name or "").strip()
            if not key:
                return ""
            cached = resolved_name_cache.get(key)
            if cached is not None:
                return cached
            resolved = ""
            for alias in _boundary_name_aliases(key):
                if alias in available_groups:
                    resolved = str(alias)
                    break
            resolved_name_cache[key] = resolved
            return resolved

        def _group_value(group_name: str) -> Optional[float]:
            resolved = _resolve_group_name(group_name)
            if not resolved:
                return None
            members = tuple(available_groups.get(resolved, ()))
            for target, bidx in members:
                n_boundaries = counts_by_target.get(str(target))
                if n_boundaries is None:
                    continue
                arr = context.ensure_boundary_seed_size(str(target), int(n_boundaries))
                if 0 <= int(bidx) < arr.size:
                    return _finite_float_or_none(arr[int(bidx)])
            return None

        def _apply_group_value(group_name: str, value: float) -> Tuple[int, set]:
            resolved = _resolve_group_name(group_name)
            if not resolved:
                return 0, set()
            applied = 0
            touched_targets: set = set()
            members = tuple(available_groups.get(resolved, ()))
            clipped = float(np.clip(float(value), 0.0, 1.0))
            for target, bidx in members:
                target_key = str(target)
                n_boundaries = counts_by_target.get(target_key)
                if n_boundaries is None:
                    continue
                arr = context.ensure_boundary_seed_size(target_key, int(n_boundaries))
                idx = int(bidx)
                if idx < 0 or idx >= arr.size:
                    continue
                arr[idx] = clipped
                context.boundary_seeds[target_key] = np.clip(
                    np.asarray(arr, dtype=float).reshape(-1), 0.0, 1.0
                )
                applied += 1
                touched_targets.add(target_key)
            return applied, touched_targets

        changed_groups: List[str] = []
        changed_targets: set = set()
        warnings: List[str] = []
        compiled_expr_cache: Dict[
            str, Tuple[Callable[[Mapping[str, float]], float], Tuple[str, ...]]
        ] = {}

        for assignment in self.assignments:
            target_name_raw = str(assignment.target_name).strip()
            if not target_name_raw:
                continue
            target_name = _resolve_group_name(target_name_raw)
            if not target_name:
                msg = f"{target_name_raw}: unknown boundary group."
                if assignment.on_missing == "fail":
                    return StepResult(
                        status="fail",
                        message=msg,
                        params_by_key=dict(context.seed_map),
                    )
                warnings.append(msg)
                continue

            source_kind = str(assignment.source_kind or "literal").strip().lower()
            source_value: Optional[float] = None

            if source_kind == "literal":
                source_value = _finite_float_or_none(assignment.literal_value)
            elif source_kind == "boundary":
                source_value = _group_value(str(assignment.source_name))
            elif source_kind == "expression":
                expr_text = str(assignment.expression or "").strip()
                if expr_text:
                    cached = compiled_expr_cache.get(expr_text)
                    if cached is None:
                        try:
                            cached = _compile_scalar_expression(expr_text)
                        except ValueError as exc:
                            msg = f"{target_name}: invalid expression ({exc})."
                            if assignment.on_missing == "fail":
                                return StepResult(
                                    status="fail",
                                    message=msg,
                                    params_by_key=dict(context.seed_map),
                                )
                            warnings.append(msg)
                            continue
                        compiled_expr_cache[expr_text] = cached
                    evaluator, expr_names = cached
                    env: Dict[str, float] = {}
                    missing_names: List[str] = []
                    for name in expr_names:
                        value = _group_value(name)
                        if value is None:
                            missing_names.append(str(name))
                        else:
                            env[str(name)] = float(value)
                    if missing_names:
                        msg = (
                            f"{target_name}: missing expression input(s): "
                            f"{', '.join(str(n) for n in missing_names)}."
                        )
                        if assignment.on_missing == "fail":
                            return StepResult(
                                status="fail",
                                message=msg,
                                params_by_key=dict(context.seed_map),
                            )
                        warnings.append(msg)
                        continue
                    try:
                        source_value = _finite_float_or_none(evaluator(env))
                    except ValueError as exc:
                        msg = f"{target_name}: expression error ({exc})."
                        if assignment.on_missing == "fail":
                            return StepResult(
                                status="fail",
                                message=msg,
                                params_by_key=dict(context.seed_map),
                            )
                        warnings.append(msg)
                        continue

            if source_value is None:
                msg = f"{target_name}: missing source ({source_kind})."
                if assignment.on_missing == "fail":
                    return StepResult(
                        status="fail",
                        message=msg,
                        params_by_key=dict(context.seed_map),
                    )
                warnings.append(msg)
                continue

            applied, touched_targets = _apply_group_value(
                target_name, float(source_value)
            )
            if applied <= 0:
                msg = f"{target_name}: no usable boundary members."
                if assignment.on_missing == "fail":
                    return StepResult(
                        status="fail",
                        message=msg,
                        params_by_key=dict(context.seed_map),
                    )
                warnings.append(msg)
                continue

            if target_name not in changed_groups:
                changed_groups.append(target_name)
            changed_targets.update(touched_targets)

        message = f"Set {len(changed_groups)} boundary group(s)."
        if warnings:
            message += f" Skipped {len(warnings)} assignment(s)."
        return StepResult(
            status="pass",
            message=message,
            params_by_key=dict(context.seed_map),
            boundary_ratios={
                str(target): np.asarray(
                    context.boundary_seeds.get(
                        str(target), np.asarray([], dtype=float)
                    ),
                    dtype=float,
                )
                .reshape(-1)
                .copy()
                for target in sorted(changed_targets)
            }
            if changed_targets
            else None,
        )


# ---------------------------------------------------------------------------
# Randomize Seeds step
# ---------------------------------------------------------------------------


@register_step_type
@dataclass(frozen=True)
class RandomizeSeedsStep(ProcedureStepBase):
    """Randomly perturb parameter seeds within their bounds."""

    step_type: ClassVar[str] = "randomize_seeds"
    step_label: ClassVar[str] = "Randomize Seeds"

    params: Tuple[str, ...] = ()  # empty = all free params
    scale: float = 0.2  # fraction of (high - low) range

    def serialize(self) -> Dict[str, Any]:
        d = super().serialize()
        if self.params:
            d["params"] = list(self.params)
        if self.scale != 0.2:
            d["scale"] = float(self.scale)
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> RandomizeSeedsStep:
        return cls(
            params=tuple(str(p) for p in (data.get("params") or ())),
            scale=float(data.get("scale", 0.2)),
            label=str(data.get("label") or ""),
        )

    def execute(self, context: ProcedureContext) -> StepResult:
        rng = np.random.default_rng(context.rng_seed)
        target_params = (
            [p for p in self.params if p in context.global_names]
            if self.params
            else list(context.global_names)
        )
        perturbed = {}
        for key in target_params:
            if key not in context.bounds_map:
                continue
            low, high = context.bounds_map[key]
            if low > high:
                low, high = high, low
            span = high - low
            current = context.seed_map.get(key, (low + high) / 2.0)
            delta = rng.uniform(-self.scale, self.scale) * span
            new_val = float(np.clip(current + delta, low, high))
            context.seed_map[key] = new_val
            perturbed[key] = new_val
        return StepResult(
            status="pass",
            message=f"Randomised {len(perturbed)} parameter(s) (scale={self.scale:.2f}).",
            params_by_key=dict(context.seed_map),
        )
