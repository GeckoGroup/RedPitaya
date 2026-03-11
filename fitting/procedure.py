"""Procedure pipeline — executes a sequence of polymorphic procedure steps.

This module replaces the monolithic ``run_procedure_pipeline`` from *model.py*
with a new version that dispatches to each step's ``execute(context)`` method.
New code should import from here.

Key public API
--------------
- ``run_procedure_pipeline()`` — the new polymorphic pipeline
- ``_execute_fit_step()``        — called by ``FitStep.execute()``
- ``FitProcedure``             — a procedure that holds ``ProcedureStepBase`` steps
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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

from procedure_steps import (
    ProcedureStepBase,
    ProcedureContext,
    StepResult,
    FitStep,
    deserialize_step,
)
import fit_log as _fit_log


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return int(max(int(minimum), value))


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(max(float(minimum), value))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


_FIT_SCREEN_MAX_POINTS = _env_int("REDPITAYA_FIT_SCREEN_MAX_POINTS", 4096, minimum=8)
_FIT_SCREEN_STAGEA_SCALE = _env_float(
    "REDPITAYA_FIT_SCREEN_STAGEA_SCALE", 0.30, minimum=1e-6
)
_FIT_SCREEN_STAGEB_SCALE = _env_float(
    "REDPITAYA_FIT_SCREEN_STAGEB_SCALE", 0.25, minimum=1e-6
)
_FIT_FULL_PATIENCE = _env_int("REDPITAYA_FIT_FULL_PATIENCE", 2, minimum=1)
_FIT_FULL_MIN_DELTA_R2 = _env_float(
    "REDPITAYA_FIT_FULL_MIN_DELTA_R2", 2e-4, minimum=0.0
)
_FIT_FULL_SCREEN_MAX_R2_GAP = _env_float(
    "REDPITAYA_FIT_FULL_SCREEN_MAX_R2_GAP", 0.12, minimum=0.0
)
_FIT_FULL_SCREEN_MIN_REL_R2 = _env_float(
    "REDPITAYA_FIT_FULL_SCREEN_MIN_REL_R2", 0.85, minimum=0.0
)
_FIT_FULL_SCREEN_ABS_MIN_R2 = _env_float(
    "REDPITAYA_FIT_FULL_SCREEN_ABS_MIN_R2", 0.0, minimum=-1.0
)
_FIT_RETRY_SCREENING_ENABLED = _env_bool(
    "REDPITAYA_FIT_RETRY_SCREENING_ENABLED", True
)
_FIT_SIBLING_MIN_R2 = _env_float("REDPITAYA_FIT_SIBLING_MIN_R2", 0.0, minimum=-1.0)
_FIT_SIBLING_MIN_R2_MARGIN = _env_float(
    "REDPITAYA_FIT_SIBLING_MIN_R2_MARGIN", 0.25, minimum=0.0
)
_FIT_SIBLING_EXACT_BONUS = _env_float(
    "REDPITAYA_FIT_SIBLING_EXACT_BONUS", 0.25, minimum=0.0
)


# ---------------------------------------------------------------------------
# FitProcedure — holds polymorphic steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitProcedure:
    """An ordered sequence of polymorphic procedure steps."""

    name: str = "Procedure"
    steps: Tuple[ProcedureStepBase, ...] = ()
    seed_from_siblings: bool = False  # global: use sibling results during retries

    def serialize(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": str(self.name),
            "steps": [s.serialize() for s in self.steps],
        }
        if self.seed_from_siblings:
            d["seed_from_siblings"] = True
        return d

    @classmethod
    def deserialize(cls, data: Mapping[str, Any]) -> FitProcedure:
        steps: list = []
        for idx, raw in enumerate(data.get("steps") or ()):
            if not isinstance(raw, Mapping):
                raise ValueError(f"Invalid step payload at index {idx}.")
            try:
                steps.append(deserialize_step(raw))
            except Exception as exc:
                raise ValueError(f"Invalid step payload at index {idx}: {exc}") from exc
        return cls(
            name=str(data.get("name") or "Procedure"),
            steps=tuple(steps),
            seed_from_siblings=bool(data.get("seed_from_siblings", False)),
        )


# ---------------------------------------------------------------------------
# Fit step execution  (called by FitStep.execute)
# ---------------------------------------------------------------------------

# Lazy import cache to avoid circular imports with model.py
_model = None


def _get_model():
    global _model
    if _model is None:
        import model as _m

        _model = _m
    return _model


# ---------------------------------------------------------------------------
# Cross-file sibling seeding helpers
# ---------------------------------------------------------------------------


def _capture_seed_signature(
    captures: Mapping[str, Any],
    seed_keys: Sequence[str],
) -> Optional[Tuple[Tuple[str, str], ...]]:
    """Create a hashable signature from capture values for matching."""
    if not seed_keys:
        return None
    sig: List[Tuple[str, str]] = []
    for key in seed_keys:
        value = captures.get(key)
        if value in (None, ""):
            return None
        sig.append((str(key), str(value)))
    return tuple(sig)


def _capture_value_distance(left: Any, right: Any) -> float:
    """Distance between two capture values (numeric-aware)."""
    try:
        ln = float(left)
        rn = float(right)
        if np.isfinite(ln) and np.isfinite(rn):
            denom = abs(ln) + abs(rn) + 1.0
            return float(abs(ln - rn) / denom)
    except (TypeError, ValueError):
        pass
    return 0.0 if str(left) == str(right) else 1.0


def _capture_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    seed_keys: Sequence[str],
) -> Optional[float]:
    """Average distance between two capture sets over *seed_keys*."""
    if not seed_keys:
        return None
    total = 0.0
    count = 0
    for key in seed_keys:
        lv = left.get(key)
        rv = right.get(key)
        if lv in (None, "") or rv in (None, ""):
            total += 1.0
        else:
            total += _capture_value_distance(lv, rv)
        count += 1
    if count <= 0:
        return None
    return float(total / float(count))


def _best_sibling_seed_from_context(
    context: ProcedureContext,
    relevant_params: set,
    fixed_params: Dict[str, float],
    min_seed_r2: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    ranked = _ranked_sibling_seed_candidates(
        context,
        relevant_params=relevant_params,
        fixed_params=fixed_params,
        enabled_targets=None,
        min_seed_r2=min_seed_r2,
        top_k=1,
    )
    if not ranked:
        return None
    return ranked[0]


def _ranked_sibling_seed_candidates(
    context: ProcedureContext,
    *,
    relevant_params: set,
    fixed_params: Mapping[str, float],
    enabled_targets: Optional[Sequence[str]],
    min_seed_r2: Optional[float] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Return top sibling candidates ranked by match quality + R²."""
    if not context.sibling_results or not context.captures:
        return []
    def _finite_r2(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except Exception:
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    quality_floor = float(_FIT_SIBLING_MIN_R2)
    min_seed_r2_val = _finite_r2(min_seed_r2)
    if min_seed_r2_val is not None:
        quality_floor = max(
            quality_floor, float(min_seed_r2_val - float(_FIT_SIBLING_MIN_R2_MARGIN))
        )

    seed_keys = context.capture_seed_keys
    if not seed_keys:
        return []
    targets_filter = set(str(t) for t in (enabled_targets or ()))
    current_sig = _capture_seed_signature(context.captures, seed_keys)
    ranked: List[Dict[str, Any]] = []
    for _file_key, sibling in context.sibling_results.items():
        sibling_captures = sibling.get("captures")
        if not isinstance(sibling_captures, Mapping):
            continue
        sibling_params = dict(sibling.get("params_by_key") or {})
        if not sibling_params:
            continue
        candidate_sig = _capture_seed_signature(sibling_captures, seed_keys)
        exact_match = (
            current_sig is not None
            and candidate_sig is not None
            and current_sig == candidate_sig
        )
        distance = _capture_distance(context.captures, sibling_captures, seed_keys)
        if distance is None:
            distance = 1.0
        sibling_r2 = _finite_r2(sibling.get("r2"))
        if sibling_r2 is not None and float(sibling_r2) < float(quality_floor):
            continue
        # Scoring: low distance + high R², exact match as a light tie-breaker.
        score = (
            (float(_FIT_SIBLING_EXACT_BONUS) if exact_match else 0.0)
            - float(distance)
            + float(sibling_r2 or 0.0)
        )

        seed_map: Dict[str, float] = {}
        for key in relevant_params:
            if key in fixed_params:
                continue
            if key not in sibling_params:
                continue
            try:
                val = float(sibling_params[key])
            except Exception:
                continue
            if np.isfinite(val):
                seed_map[str(key)] = float(val)
        if not seed_map:
            continue

        boundary_seeds: Dict[str, np.ndarray] = {}
        for ch, ratios in (sibling.get("boundary_ratios_by_channel") or {}).items():
            ch_key = str(ch)
            if targets_filter and ch_key not in targets_filter:
                continue
            try:
                arr = np.asarray(ratios, dtype=float).reshape(-1)
            except Exception:
                continue
            if arr.size > 0:
                boundary_seeds[ch_key] = arr

        ranked.append(
            {
                "seed_map": seed_map,
                "boundary_seeds": boundary_seeds,
                "source_kind": "matching-captures" if exact_match else "closest-captures",
                "distance": float(distance),
                "exact_match": bool(exact_match),
                "r2": sibling_r2,
                "score": float(score),
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("r2") or -1e9),
            int(bool(item.get("exact_match"))),
            -float(item.get("distance") or 1e9),
        ),
        reverse=True,
    )
    limit = max(0, int(top_k))
    if limit > 0:
        return ranked[:limit]
    return ranked


def _attempt_seed_signature(
    seed: Mapping[str, float],
    boundary_seeds: Mapping[str, Any],
) -> Tuple[float, ...]:
    """Build a hashable signature for attempt deduplication."""
    parts: List[float] = []
    for key in sorted(seed.keys()):
        try:
            parts.append(round(float(seed[key]), 10))
        except Exception:
            parts.append(0.0)
    for ch in sorted(boundary_seeds.keys()):
        try:
            arr = np.asarray(boundary_seeds[ch], dtype=float).reshape(-1)
            parts.extend(round(float(v), 10) for v in arr)
        except Exception:
            pass
    return tuple(parts)


def _default_boundary_name_groups(
    multi_model: Any,
) -> Dict[str, Tuple[Tuple[str, int], ...]]:
    """Build default boundary-name groups from model link topology."""
    all_ids = list(getattr(multi_model, "all_boundary_ids", ()) or ())
    all_ids = [(str(t), int(i)) for t, i in all_ids]
    if not all_ids:
        return {}

    parent: Dict[Tuple[str, int], Tuple[str, int]] = {bid: bid for bid in all_ids}

    def _find(bid: Tuple[str, int]) -> Tuple[str, int]:
        cur = bid
        while parent[cur] != cur:
            parent[cur] = parent[parent[cur]]
            cur = parent[cur]
        return cur

    def _union(left: Tuple[str, int], right: Tuple[str, int]) -> None:
        rl = _find(left)
        rr = _find(right)
        if rl != rr:
            parent[rr] = rl

    for group in getattr(multi_model, "boundary_links", ()) or ():
        valid = []
        for raw in group:
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                continue
            bid = (str(raw[0]), int(raw[1]))
            if bid in parent:
                valid.append(bid)
        if len(valid) < 2:
            continue
        anchor = valid[0]
        for bid in valid[1:]:
            _union(anchor, bid)

    grouped: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for bid in all_ids:
        root = _find(bid)
        grouped.setdefault(root, []).append(bid)

    out: Dict[str, Tuple[Tuple[str, int], ...]] = {}
    used: set = set()
    next_idx = 0
    for bid in all_ids:
        root = _find(bid)
        if root in used:
            continue
        used.add(root)
        name = f"X{next_idx}"
        next_idx += 1
        out[name] = tuple(grouped.get(root, ()))
    return out


def _boundary_link_groups_for_context(
    context: ProcedureContext,
) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """Return normalised boundary link groups for the current procedure context."""
    from_model = tuple(getattr(context.multi_model, "boundary_links", ()) or ())
    raw_groups: Sequence[Sequence[Any]]
    if from_model:
        raw_groups = from_model
    else:
        raw_groups = tuple(
            (members or ()) for members in context.boundary_name_to_ids.values()
        )

    out: List[Tuple[Tuple[str, int], ...]] = []
    seen: set = set()
    for raw_group in raw_groups:
        members: List[Tuple[str, int]] = []
        for raw_member in tuple(raw_group or ()):
            if not isinstance(raw_member, (tuple, list)) or len(raw_member) != 2:
                continue
            target = str(raw_member[0]).strip()
            if not target:
                continue
            try:
                idx = int(raw_member[1])
            except Exception:
                continue
            if idx < 0:
                continue
            bid = (target, idx)
            if bid not in members:
                members.append(bid)
        if len(members) < 2:
            continue
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)
        out.append(tuple(members))
    return tuple(out)


def _propagate_linked_boundary_seeds(
    context: ProcedureContext,
    *,
    source_targets: Sequence[str],
) -> None:
    """Mirror linked boundary ratios across channels, like the manual fit path."""
    link_groups = _boundary_link_groups_for_context(context)
    if not link_groups:
        return

    # Ensure each channel has a correctly sized boundary array before propagation.
    for ch_model in tuple(getattr(context.multi_model, "channel_models", ()) or ()):
        target = str(getattr(ch_model, "target_col", "") or "").strip()
        if not target:
            continue
        n_boundaries = max(0, len(getattr(ch_model, "segment_exprs", ()) or ()) - 1)
        context.ensure_boundary_seed_size(target, n_boundaries)

    preferred: List[str] = []
    for raw_target in source_targets:
        target = str(raw_target).strip()
        if target and target not in preferred:
            preferred.append(target)

    for members in link_groups:
        reference = None
        for target in preferred:
            for member_target, member_idx in members:
                if member_target != target:
                    continue
                arr_raw = context.boundary_seeds.get(member_target)
                if arr_raw is None:
                    continue
                arr = np.asarray(arr_raw, dtype=float).reshape(-1)
                if 0 <= int(member_idx) < arr.size:
                    reference = float(arr[int(member_idx)])
                    break
            if reference is not None:
                break

        if reference is None:
            for member_target, member_idx in members:
                arr_raw = context.boundary_seeds.get(member_target)
                if arr_raw is None:
                    continue
                arr = np.asarray(arr_raw, dtype=float).reshape(-1)
                if 0 <= int(member_idx) < arr.size:
                    reference = float(arr[int(member_idx)])
                    break

        if reference is None or not np.isfinite(reference):
            continue
        linked_value = float(np.clip(reference, 0.0, 1.0))

        for member_target, member_idx in members:
            arr_raw = context.boundary_seeds.get(member_target)
            if arr_raw is None:
                continue
            arr = np.asarray(arr_raw, dtype=float).reshape(-1).copy()
            if 0 <= int(member_idx) < arr.size:
                arr[int(member_idx)] = linked_value
                context.boundary_seeds[str(member_target)] = arr


def _execute_fit_step(step: FitStep, context: ProcedureContext) -> StepResult:
    """Core fit logic for a single FitStep, including retry support."""
    model = _get_model()
    global_names = context.global_names

    # --- Boundary-only mode ---
    # channels=() (empty tuple) means "no channels — fit boundaries only."
    # In this mode every parameter is forced fixed from the current seed map
    # and the fit optimises only boundary ratios, using all channels' y-data
    # (same behaviour as clicking Fit with everything fixed in the normal UI).
    boundary_only = step.channels is not None and len(step.channels) == 0

    # --- Resolve fixed params ---
    fixed_for_step: Dict[str, float] = {}

    if boundary_only:
        # Fix every parameter from the current context seeds.
        for name in global_names:
            fixed_for_step[name] = float(context.seed_map.get(name, 0.0))
    else:
        # Apply field-driven parameters as seed overrides (normal-fit behavior).
        # They are not forced fixed unless the step's free/fixed config says so.
        for param_key, field_name in step.bound_params:
            if param_key in global_names and field_name in context.bound_values:
                context.seed_map[param_key] = float(context.bound_values[field_name])

        if step.free_params:
            free_set = set(step.free_params) & global_names
            for name in global_names:
                if name not in free_set:
                    fixed_for_step[name] = float(context.seed_map.get(name, 0.0))
        elif step.fixed_params:
            for name in step.fixed_params:
                if name in global_names:
                    fixed_for_step[name] = float(context.seed_map.get(name, 0.0))

    # --- Filter to step channels ---
    if boundary_only:
        # Boundary-only: use every channel's data.
        enabled_models = context.multi_model.channel_models
    elif step.channels:
        step_channels = set(step.channels)
        enabled_models = tuple(
            m
            for m in context.multi_model.channel_models
            if m.target_col in step_channels
        )
    else:
        # channels is None → all channels.
        enabled_models = context.multi_model.channel_models

    if not enabled_models:
        return StepResult(status="skipped", message="No matching channels.")

    # Build filtered multi-model.
    enabled_targets = {m.target_col for m in enabled_models}
    filtered_links: List[Tuple[Tuple[str, int], ...]] = []
    for group in context.multi_model.boundary_links:
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

    step_multi = model.MultiChannelModelDefinition(
        channel_models=enabled_models,
        global_param_names=tuple(filtered_global),
        boundary_links=tuple(filtered_links),
    )

    # Gather y-data.
    step_y = {
        ch: context.y_data_by_channel[ch]
        for ch in enabled_targets
        if ch in context.y_data_by_channel
    }
    if not step_y:
        return StepResult(status="skipped", message="No y-data for channels.")

    step_boundary_seeds = {
        ch: context.boundary_seeds[ch]
        for ch in enabled_targets
        if ch in context.boundary_seeds
    }

    step_fixed_filtered = {k: v for k, v in fixed_for_step.items() if k in seen_global}

    # Resolve locked boundary names to (target, index) pairs.
    selected_locked_ids: set = set()
    for boundary_name in step.locked_boundary_names:
        for bid in context.boundary_name_to_ids.get(str(boundary_name), ()):
            if isinstance(bid, (tuple, list)) and len(bid) == 2:
                selected_locked_ids.add((str(bid[0]), int(bid[1])))

    n_boundaries_by_target = {
        m.target_col: max(0, len(m.segment_exprs) - 1) for m in enabled_models
    }
    fixed_boundary_by_channel: Dict[str, Dict[int, float]] = {}
    for target, bidx in sorted(selected_locked_ids):
        if target not in n_boundaries_by_target:
            continue
        n_b = n_boundaries_by_target[target]
        if bidx < 0 or bidx >= n_b:
            continue
        arr = context.ensure_boundary_seed_size(target, n_b)
        if bidx >= arr.size:
            continue
        fixed_boundary_by_channel.setdefault(target, {})[int(bidx)] = float(arr[bidx])

    # --- Run the fit (with retries) ---
    max_attempts = 1 + max(0, step.max_retries)
    retry_mode = str(step.retry_mode or "jitter_then_random").strip()
    if retry_mode not in {"jitter", "random", "jitter_then_random"}:
        retry_mode = "jitter_then_random"
    step_free_for_log = (
        sorted(str(p) for p in step.free_params if str(p) in seen_global)
        if step.free_params
        else sorted(str(p) for p in seen_global if str(p) not in step_fixed_filtered)
    )
    step_label = str(getattr(step, "label", "") or "fit")

    # --- Pre-compute sibling seed (global procedure setting) ---
    _sibling_seed_info: Optional[Dict[str, Any]] = None
    if context.seed_from_siblings:
        if not context.sibling_results:
            _fit_log.detail("sibling-seed: enabled but no sibling results available")
        elif not context.captures:
            _fit_log.detail("sibling-seed: enabled but no captures for current file")
        elif not context.capture_seed_keys:
            _fit_log.detail(
                "sibling-seed: enabled but no capture_seed_keys configured "
                "(need bound_params in procedure steps)"
            )
        else:
            _sibling_seed_info = _best_sibling_seed_from_context(
                context,
                seen_global,
                step_fixed_filtered,
                min_seed_r2=step.min_r2,
            )
            if _sibling_seed_info:
                n_seed_params = len(_sibling_seed_info.get("seed_map") or {})
                n_seed_boundaries = len(_sibling_seed_info.get("boundary_seeds") or {})
                _fit_log.detail(
                    "sibling seed found: "
                    f"kind={_sibling_seed_info.get('source_kind')} "
                    f"r2={_sibling_seed_info.get('r2')} "
                    f"params={n_seed_params} boundaries={n_seed_boundaries}"
                )
            else:
                _fit_log.detail(
                    f"sibling-seed: enabled, {len(context.sibling_results)} "
                    "sibling(s) available but no match found for captures"
                )

    _fit_log.step_start(
        context.step_index,
        context.step_total,
        "fit",
        step_label,
        channels=sorted(enabled_targets),
        n_free=len(step_free_for_log),
        n_fixed=len(step_fixed_filtered),
        n_locked_boundaries=len(step.locked_boundary_names),
        max_attempts=max_attempts,
        retry_mode=retry_mode,
        seed_from_siblings=bool(_sibling_seed_info is not None),
    )

    # --- Dedup tracking ---
    _seen_attempt_sigs: set = set()
    best_result = None
    best_r2 = None
    retry_r2_history: List[float] = []
    attempt_counter = 0
    high_retry_heuristics = (
        int(step.max_retries) >= 2 and bool(_FIT_RETRY_SCREENING_ENABLED)
    )
    sibling_pool = (
        _ranked_sibling_seed_candidates(
            context,
            relevant_params=set(seen_global),
            fixed_params=step_fixed_filtered,
            enabled_targets=tuple(enabled_targets),
            min_seed_r2=step.min_r2,
            top_k=3,
        )
        if context.seed_from_siblings
        else []
    )

    def _clone_boundary_map(boundary_map: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for ch, arr in dict(boundary_map).items():
            out[str(ch)] = np.asarray(arr, dtype=float).reshape(-1).copy()
        return out

    def _boundary_template() -> Dict[str, np.ndarray]:
        template: Dict[str, np.ndarray] = {}
        for target, n_b in n_boundaries_by_target.items():
            base = step_boundary_seeds.get(target)
            if base is None:
                base_arr = context.ensure_boundary_seed_size(target, n_b)
            else:
                base_arr = np.asarray(base, dtype=float).reshape(-1)
                if base_arr.size != n_b:
                    base_arr = context.ensure_boundary_seed_size(target, n_b)
            template[target] = np.clip(np.asarray(base_arr, dtype=float), 0.0, 1.0)
        for target, locked in fixed_boundary_by_channel.items():
            arr = template.get(target)
            if arr is None:
                continue
            arr = arr.copy()
            for idx, value in locked.items():
                if 0 <= int(idx) < arr.size:
                    arr[int(idx)] = float(np.clip(value, 0.0, 1.0))
            template[target] = arr
        return template

    base_boundary_template = _boundary_template()

    def _sanitize_seed(seed_like: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        raw = dict(seed_like or {})
        for key in seen_global:
            if key in step_fixed_filtered:
                out[key] = float(step_fixed_filtered[key])
                continue
            low_raw, high_raw = context.bounds_map.get(key, (-np.inf, np.inf))
            low = float(min(low_raw, high_raw))
            high = float(max(low_raw, high_raw))
            default = context.seed_map.get(key, (low + high) / 2.0)
            try:
                value = float(raw.get(key, default))
            except Exception:
                value = float(default)
            if np.isfinite(low) and np.isfinite(high):
                if bool(context.periodic_params.get(str(key), False)) and (high > low):
                    value = float(low + np.mod(value - low, high - low))
                else:
                    value = float(np.clip(value, low, high))
            out[key] = float(value)
        return out

    def _sanitize_boundaries(boundary_like: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        out = _clone_boundary_map(base_boundary_template)
        for ch, raw_arr in dict(boundary_like or {}).items():
            ch_key = str(ch)
            if ch_key not in out:
                continue
            target_arr = out[ch_key]
            try:
                candidate = np.asarray(raw_arr, dtype=float).reshape(-1)
            except Exception:
                continue
            if candidate.size != target_arr.size:
                continue
            out[ch_key] = np.clip(candidate, 0.0, 1.0)
        for target, locked in fixed_boundary_by_channel.items():
            arr = out.get(target)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float).reshape(-1).copy()
            for idx, value in locked.items():
                if 0 <= int(idx) < arr.size:
                    arr[int(idx)] = float(np.clip(value, 0.0, 1.0))
            out[target] = arr
        return out

    def _extract_boundary_seed_map(result: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        ch_res = result.get("channel_results")
        if isinstance(ch_res, Mapping):
            for ch_t, ch_r in ch_res.items():
                if not isinstance(ch_r, Mapping):
                    continue
                br = ch_r.get("boundary_ratios")
                if br is None:
                    continue
                try:
                    out[str(ch_t)] = np.asarray(br, dtype=float).reshape(-1)
                except Exception:
                    pass
            return out
        if (
            len(enabled_models) == 1
            and result.get("boundary_ratios") is not None
            and enabled_models[0].target_col in n_boundaries_by_target
        ):
            ch_t = str(enabled_models[0].target_col)
            try:
                out[ch_t] = np.asarray(result["boundary_ratios"], dtype=float).reshape(-1)
            except Exception:
                pass
        return out

    def _seed_from_fit_result(
        base_seed: Mapping[str, float],
        base_boundaries: Mapping[str, Any],
        fit_result: Mapping[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        seed_out = dict(base_seed)
        for key, value in dict(fit_result.get("params_by_key") or {}).items():
            k = str(key)
            if k in seen_global and k not in step_fixed_filtered:
                try:
                    v = float(value)
                except Exception:
                    continue
                if np.isfinite(v):
                    seed_out[k] = v
        boundary_out = _clone_boundary_map(_sanitize_boundaries(base_boundaries))
        boundary_from_result = _extract_boundary_seed_map(fit_result)
        for ch, arr in boundary_from_result.items():
            if ch not in boundary_out:
                continue
            candidate = np.asarray(arr, dtype=float).reshape(-1)
            if candidate.size == boundary_out[ch].size:
                boundary_out[ch] = np.clip(candidate, 0.0, 1.0)
        return _sanitize_seed(seed_out), _sanitize_boundaries(boundary_out)

    def _randomise_seed(
        rng: np.random.Generator,
        base_seed: Mapping[str, float],
        scale: float,
        *,
        force_random: bool = False,
    ) -> Dict[str, float]:
        out = dict(base_seed)
        for key in seen_global:
            if key in step_fixed_filtered:
                out[key] = float(step_fixed_filtered[key])
                continue
            if key not in context.bounds_map:
                continue
            low_raw, high_raw = context.bounds_map[key]
            low = float(min(low_raw, high_raw))
            high = float(max(low_raw, high_raw))
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            if np.isclose(low, high):
                out[key] = float(low)
                continue
            if force_random or retry_mode == "random":
                out[key] = float(rng.uniform(low, high))
                continue
            span = high - low
            center = float(out.get(key, (low + high) / 2.0))
            delta = float(rng.uniform(-scale, scale) * span)
            out[key] = float(np.clip(center + delta, low, high))
        return _sanitize_seed(out)

    def _jitter_seed(
        rng: np.random.Generator,
        center_seed: Mapping[str, float],
        scale: float,
    ) -> Dict[str, float]:
        out = dict(center_seed)
        for key in seen_global:
            if key in step_fixed_filtered or key not in context.bounds_map:
                continue
            low_raw, high_raw = context.bounds_map[key]
            low = float(min(low_raw, high_raw))
            high = float(max(low_raw, high_raw))
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            if np.isclose(low, high):
                out[key] = float(low)
                continue
            span = high - low
            center = float(out.get(key, (low + high) / 2.0))
            delta = float(rng.normal(0.0, max(1e-6, scale) * span))
            out[key] = float(np.clip(center + delta, low, high))
        return _sanitize_seed(out)

    def _jitter_boundaries(
        rng: np.random.Generator,
        boundary_seed_map: Mapping[str, Any],
        scale: float,
    ) -> Dict[str, np.ndarray]:
        out = _clone_boundary_map(_sanitize_boundaries(boundary_seed_map))
        for ch, arr in list(out.items()):
            if arr.size <= 0:
                continue
            jitter = rng.normal(0.0, max(1e-6, scale) * 0.35, size=arr.size)
            out[ch] = np.clip(np.asarray(arr, dtype=float) + jitter, 0.0, 1.0)
        return _sanitize_boundaries(out)

    def _blend_from_siblings(
        siblings: Sequence[Mapping[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        base_seed = _sanitize_seed(context.seed_map)
        seed_out = dict(base_seed)
        if not siblings:
            return seed_out, _sanitize_boundaries(base_boundary_template)
        base_w = 0.45
        sib_weights = [0.30, 0.18, 0.07][: len(siblings)]
        for key in seen_global:
            if key in step_fixed_filtered:
                seed_out[key] = float(step_fixed_filtered[key])
                continue
            weighted = base_w * float(base_seed.get(key, context.seed_map.get(key, 0.0)))
            wsum = base_w
            for idx, sib in enumerate(siblings):
                s_map = dict(sib.get("seed_map") or {})
                if key not in s_map:
                    continue
                w = sib_weights[idx]
                weighted += w * float(s_map[key])
                wsum += w
            if wsum > 0:
                seed_out[key] = float(weighted / wsum)
        boundary_out = _clone_boundary_map(_sanitize_boundaries(base_boundary_template))
        for ch, arr in list(boundary_out.items()):
            if arr.size <= 0:
                continue
            weighted = base_w * np.asarray(arr, dtype=float)
            wsum = np.full(arr.size, base_w, dtype=float)
            for idx, sib in enumerate(siblings):
                s_b = (sib.get("boundary_seeds") or {}).get(ch)
                if s_b is None:
                    continue
                candidate = np.asarray(s_b, dtype=float).reshape(-1)
                if candidate.size != arr.size:
                    continue
                w = sib_weights[idx]
                weighted += w * candidate
                wsum += w
            safe = np.where(wsum > 0.0, wsum, 1.0)
            boundary_out[ch] = np.clip(weighted / safe, 0.0, 1.0)
        return _sanitize_seed(seed_out), _sanitize_boundaries(boundary_out)

    def _make_candidate(
        seed: Mapping[str, float],
        boundaries: Mapping[str, Any],
        seed_source: str,
    ) -> Dict[str, Any]:
        return {
            "seed": _sanitize_seed(seed),
            "boundaries": _sanitize_boundaries(boundaries),
            "seed_source": str(seed_source),
        }

    def _emit_attempt_callback(
        *,
        attempt_idx: int,
        attempt_r2: Optional[float],
        is_new_best: bool,
        attempt_strategy: str,
        elapsed: float,
        phase: str,
        seed_source: str,
        result: Mapping[str, Any],
    ) -> None:
        if context.attempt_callback is None:
            return
        attempt_params = dict(result.get("params_by_key") or {})
        attempt_boundary_ratios_out: Dict[str, Any] = {}
        attempt_per_channel_r2: Dict[str, Optional[float]] = {}
        ch_res = result.get("channel_results")
        if isinstance(ch_res, Mapping):
            for ch_t, ch_r in ch_res.items():
                if not isinstance(ch_r, Mapping):
                    continue
                br = ch_r.get("boundary_ratios")
                if br is not None:
                    attempt_boundary_ratios_out[str(ch_t)] = (
                        np.asarray(br, dtype=float).reshape(-1).tolist()
                    )
                ch_r2 = ch_r.get("r2")
                if ch_r2 is not None:
                    try:
                        val = float(ch_r2)
                    except Exception:
                        val = None
                    if val is not None and np.isfinite(val):
                        attempt_per_channel_r2[str(ch_t)] = float(val)
        elif result.get("boundary_ratios") is not None and len(enabled_models) == 1:
            ch_t = str(enabled_models[0].target_col)
            attempt_boundary_ratios_out[ch_t] = (
                np.asarray(result["boundary_ratios"], dtype=float).reshape(-1).tolist()
            )
        try:
            context.attempt_callback(
                context.step_index - 1,  # 0-based step index
                int(attempt_idx),
                {
                    "r2": float(attempt_r2) if attempt_r2 is not None else None,
                    "is_new_best": bool(is_new_best),
                    "strategy": attempt_strategy,
                    "phase": str(phase),
                    "seed_source": str(seed_source),
                    "attempt": int(attempt_idx),
                    "max_attempts": int(max(int(max_attempts), int(attempt_counter))),
                    "best_r2": best_r2,
                    "elapsed": float(elapsed),
                    "retry_r2_history": list(retry_r2_history),
                    "step_label": step_label,
                    "params_by_key": attempt_params,
                    "boundary_ratios": attempt_boundary_ratios_out,
                    "per_channel_r2": dict(attempt_per_channel_r2),
                    "channels": list(sorted(enabled_targets)),
                    "free_params": list(step_free_for_log),
                    "fixed_params": list(step_fixed_filtered.keys()),
                },
            )
        except Exception:
            pass

    def _run_attempt(
        candidate: Mapping[str, Any],
        *,
        phase: str,
        screening_mode: bool,
    ) -> Optional[Dict[str, Any]]:
        nonlocal best_result, best_r2, attempt_counter
        if context.is_cancelled():
            raise model.FitCancelledError("cancelled")
        attempt_seed = _sanitize_seed(candidate.get("seed") or {})
        attempt_boundary_seeds = _sanitize_boundaries(candidate.get("boundaries") or {})
        seed_source = str(candidate.get("seed_source") or "base")
        sig = _attempt_seed_signature(attempt_seed, attempt_boundary_seeds)
        attempt_strategy = f"{phase}:{seed_source}"
        if sig in _seen_attempt_sigs:
            _fit_log.attempt_start(
                int(attempt_counter + 1),
                int(max(int(max_attempts), int(attempt_counter + 1))),
                f"{attempt_strategy}(dedup-skip)",
            )
            return None
        _seen_attempt_sigs.add(sig)
        attempt_counter += 1
        _fit_log.attempt_start(
            int(attempt_counter),
            int(max(int(max_attempts), int(attempt_counter))),
            attempt_strategy,
        )
        attempt_t0 = time.perf_counter()
        try:
            if step_multi.is_multi_channel:
                result = model.run_multi_channel_fit_pipeline(
                    context.x_data,
                    step_y,
                    step_multi,
                    attempt_seed,
                    context.bounds_map,
                    periodic_params=context.periodic_params,
                    boundary_seeds=attempt_boundary_seeds,
                    cancel_check=context.cancel_check,
                    fixed_params=step_fixed_filtered,
                    fixed_boundary_ratios_by_channel=fixed_boundary_by_channel,
                    n_random_restarts=0,
                    rng_seed=context.rng_seed,
                    screening=bool(screening_mode),
                    max_points=(
                        int(_FIT_SCREEN_MAX_POINTS)
                        if bool(screening_mode)
                        else None
                    ),
                    stage_a_nfev_scale=(
                        float(_FIT_SCREEN_STAGEA_SCALE)
                        if bool(screening_mode)
                        else 1.0
                    ),
                    stage_b_nfev_scale=(
                        float(_FIT_SCREEN_STAGEB_SCALE)
                        if bool(screening_mode)
                        else 1.0
                    ),
                )
            else:
                ch_model = enabled_models[0]
                ch_target = ch_model.target_col
                b_seed = attempt_boundary_seeds.get(ch_target)
                result = model.run_piecewise_fit_pipeline(
                    context.x_data,
                    step_y[ch_target],
                    ch_model,
                    attempt_seed,
                    context.bounds_map,
                    periodic_params=context.periodic_params,
                    boundary_seed=b_seed,
                    cancel_check=context.cancel_check,
                    fixed_params=step_fixed_filtered,
                    fixed_boundary_ratios=fixed_boundary_by_channel.get(ch_target, {}),
                    n_random_restarts=0,
                    rng_seed=context.rng_seed,
                    screening=bool(screening_mode),
                    max_points=(
                        int(_FIT_SCREEN_MAX_POINTS)
                        if bool(screening_mode)
                        else None
                    ),
                    stage_a_nfev_scale=(
                        float(_FIT_SCREEN_STAGEA_SCALE)
                        if bool(screening_mode)
                        else 1.0
                    ),
                    stage_b_nfev_scale=(
                        float(_FIT_SCREEN_STAGEB_SCALE)
                        if bool(screening_mode)
                        else 1.0
                    ),
                )
        except model.FitCancelledError:
            raise
        except Exception as exc:
            _fit_log.attempt_fail(
                time.perf_counter() - attempt_t0, f"{type(exc).__name__}: {exc}"
            )
            return None

        elapsed = time.perf_counter() - attempt_t0
        raw_r2 = result.get("r2")
        attempt_r2 = float(raw_r2) if raw_r2 is not None else None
        is_new_best = best_result is None or (
            attempt_r2 is not None and (best_r2 is None or attempt_r2 > float(best_r2))
        )
        _fit_log.attempt_done(elapsed, r2=attempt_r2, is_best=is_new_best)
        if attempt_r2 is not None:
            retry_r2_history.append(float(attempt_r2))
        if is_new_best:
            best_result = result
            best_r2 = float(attempt_r2) if attempt_r2 is not None else best_r2
        _emit_attempt_callback(
            attempt_idx=int(attempt_counter - 1),
            attempt_r2=attempt_r2,
            is_new_best=is_new_best,
            attempt_strategy=attempt_strategy,
            elapsed=elapsed,
            phase=phase,
            seed_source=seed_source,
            result=result,
        )
        return {
            "result": result,
            "r2": attempt_r2,
            "elapsed": float(elapsed),
            "seed": attempt_seed,
            "boundaries": attempt_boundary_seeds,
            "seed_source": seed_source,
            "phase": phase,
        }

    # Two-stage attempt pipeline for high-retry fit steps.
    if high_retry_heuristics:
        rng = np.random.default_rng((context.rng_seed or 42) + int(context.step_index))
        candidate_queue: List[Dict[str, Any]] = []
        candidate_queue.append(
            _make_candidate(context.seed_map, base_boundary_template, "base")
        )
        if _sibling_seed_info is not None:
            sibling_seed = dict(context.seed_map)
            sibling_seed.update(dict(_sibling_seed_info.get("seed_map") or {}))
            sibling_boundaries = _clone_boundary_map(base_boundary_template)
            for ch, arr in dict(_sibling_seed_info.get("boundary_seeds") or {}).items():
                if str(ch) in sibling_boundaries:
                    sibling_boundaries[str(ch)] = np.asarray(arr, dtype=float).reshape(-1)
            candidate_queue.append(
                _make_candidate(sibling_seed, sibling_boundaries, "sibling_blend")
            )
        if sibling_pool:
            blend_seed, blend_boundaries = _blend_from_siblings(sibling_pool)
            candidate_queue.append(
                _make_candidate(blend_seed, blend_boundaries, "sibling_blend")
            )
            for sibling in sibling_pool:
                s_seed = _sanitize_seed(sibling.get("seed_map") or {})
                s_boundaries = _sanitize_boundaries(sibling.get("boundary_seeds") or {})
                candidate_queue.append(
                    _make_candidate(
                        _jitter_seed(rng, s_seed, max(0.05, float(step.retry_scale) * 0.5)),
                        _jitter_boundaries(
                            rng, s_boundaries, max(0.05, float(step.retry_scale) * 0.4)
                        ),
                        "sibling_jitter",
                    )
                )
        if len(candidate_queue) > max_attempts:
            candidate_queue = candidate_queue[:max_attempts]

        reserve_for_best = 1 if max_attempts >= 4 else 0
        while len(candidate_queue) < max(1, max_attempts - reserve_for_best):
            candidate_queue.append(
                _make_candidate(
                    _randomise_seed(
                        rng,
                        context.seed_map,
                        max(0.1, float(step.retry_scale) if retry_mode != "random" else 1.0),
                        force_random=True,
                    ),
                    _jitter_boundaries(
                        rng,
                        base_boundary_template,
                        0.75 if retry_mode == "random" else max(0.1, float(step.retry_scale)),
                    ),
                    "random",
                )
            )

        screening_results: List[Dict[str, Any]] = []
        best_screen: Optional[Dict[str, Any]] = None
        screen_target_hit = False
        queue_idx = 0
        while queue_idx < len(candidate_queue):
            outcome = _run_attempt(candidate_queue[queue_idx], phase="screen", screening_mode=True)
            queue_idx += 1
            if outcome is None:
                continue
            screening_results.append(outcome)
            if (
                best_screen is None
                or (
                    (outcome.get("r2") is not None and best_screen.get("r2") is None)
                    or (
                        outcome.get("r2") is not None
                        and best_screen.get("r2") is not None
                        and float(outcome["r2"]) > float(best_screen["r2"])
                    )
                    or (
                        outcome.get("r2") is not None
                        and best_screen.get("r2") is not None
                        and np.isclose(float(outcome["r2"]), float(best_screen["r2"]))
                        and float(outcome.get("elapsed") or np.inf)
                        < float(best_screen.get("elapsed") or np.inf)
                    )
                )
            ):
                best_screen = outcome
                if len(candidate_queue) < max_attempts:
                    tuned_seed, tuned_boundaries = _seed_from_fit_result(
                        best_screen["seed"],
                        best_screen["boundaries"],
                        best_screen["result"],
                    )
                    candidate_queue.append(
                        _make_candidate(
                            _jitter_seed(
                                rng,
                                tuned_seed,
                                max(0.05, float(step.retry_scale) * 0.4),
                            ),
                            _jitter_boundaries(
                                rng,
                                tuned_boundaries,
                                max(0.05, float(step.retry_scale) * 0.35),
                            ),
                            "best_jitter",
                        )
                    )
            if (
                step.min_r2 is not None
                and outcome.get("r2") is not None
                and float(outcome["r2"]) >= float(step.min_r2)
            ):
                screen_target_hit = True
                _fit_log.detail(
                    "screen target reached: "
                    f"r2={float(outcome['r2']):.6f} >= min_r2={float(step.min_r2):.6f}; "
                    "skipping remaining screen/full attempts"
                )
                break

        if screen_target_hit and best_screen is not None:
            best_result = best_screen["result"]
            best_r2 = (
                float(best_screen["r2"]) if best_screen.get("r2") is not None else best_r2
            )
        else:
            ranked_screen = sorted(
                screening_results,
                key=lambda item: (
                    float(item.get("r2") if item.get("r2") is not None else -1e12),
                    -float(item.get("elapsed") or np.inf),
                ),
                reverse=True,
            )
            best_screen_r2: Optional[float] = None
            if ranked_screen:
                top_screen_r2 = ranked_screen[0].get("r2")
                if top_screen_r2 is not None:
                    try:
                        top_numeric = float(top_screen_r2)
                    except Exception:
                        top_numeric = None
                    if top_numeric is not None and np.isfinite(top_numeric):
                        best_screen_r2 = float(top_numeric)
            full_screen_floor_r2: Optional[float] = None
            if best_screen_r2 is not None:
                full_screen_floor_r2 = float(
                    max(
                        float(_FIT_FULL_SCREEN_ABS_MIN_R2),
                        float(best_screen_r2 - float(_FIT_FULL_SCREEN_MAX_R2_GAP)),
                        float(best_screen_r2 * float(_FIT_FULL_SCREEN_MIN_REL_R2)),
                    )
                )
                _fit_log.detail(
                    "full-candidate screen floor: "
                    f"best={best_screen_r2:.6f} floor={full_screen_floor_r2:.6f}"
                )

            def _screen_candidate_allowed(entry: Mapping[str, Any], *, is_best: bool) -> bool:
                if is_best:
                    return True
                if full_screen_floor_r2 is None:
                    return True
                r2_raw = entry.get("r2")
                if r2_raw is None:
                    _fit_log.detail(
                        "full-candidate dropped (screen): "
                        f"seed_source={entry.get('seed_source')} reason=no-r2"
                    )
                    return False
                try:
                    r2_value = float(r2_raw)
                except Exception:
                    _fit_log.detail(
                        "full-candidate dropped (screen): "
                        f"seed_source={entry.get('seed_source')} reason=invalid-r2"
                    )
                    return False
                if not np.isfinite(r2_value):
                    _fit_log.detail(
                        "full-candidate dropped (screen): "
                        f"seed_source={entry.get('seed_source')} reason=non-finite-r2"
                    )
                    return False
                if float(r2_value) < float(full_screen_floor_r2):
                    _fit_log.detail(
                        "full-candidate dropped (screen): "
                        f"seed_source={entry.get('seed_source')} "
                        f"r2={r2_value:.6f} floor={float(full_screen_floor_r2):.6f}"
                    )
                    return False
                return True

            n_full = min(3, max(1, int(np.ceil(np.sqrt(max_attempts)))))
            full_candidates: List[Dict[str, Any]] = []
            anchor_sources = {"base", "sibling_blend"}
            if ranked_screen:
                for idx, entry in enumerate(ranked_screen):
                    if str(entry.get("seed_source") or "") not in anchor_sources:
                        continue
                    if not _screen_candidate_allowed(entry, is_best=(idx == 0)):
                        continue
                    tuned_seed, tuned_boundaries = _seed_from_fit_result(
                        entry["seed"],
                        entry["boundaries"],
                        entry["result"],
                    )
                    full_candidates.append(
                        _make_candidate(
                            tuned_seed,
                            tuned_boundaries,
                            str(entry.get("seed_source") or "base"),
                        )
                    )
                    break
            for idx, entry in enumerate(ranked_screen):
                if not _screen_candidate_allowed(entry, is_best=(idx == 0)):
                    continue
                tuned_seed, tuned_boundaries = _seed_from_fit_result(
                    entry["seed"],
                    entry["boundaries"],
                    entry["result"],
                )
                candidate = _make_candidate(
                    tuned_seed, tuned_boundaries, str(entry.get("seed_source") or "base")
                )
                sig = _attempt_seed_signature(candidate["seed"], candidate["boundaries"])
                if any(
                    _attempt_seed_signature(c["seed"], c["boundaries"]) == sig
                    for c in full_candidates
                ):
                    continue
                full_candidates.append(candidate)
                if len(full_candidates) >= n_full:
                    break

            best_full_result = None
            best_full_r2 = None
            no_progress = 0
            for candidate in full_candidates:
                outcome = _run_attempt(candidate, phase="full", screening_mode=False)
                if outcome is None:
                    continue
                current_r2 = outcome.get("r2")
                improved = False
                if current_r2 is not None and (
                    best_full_r2 is None or float(current_r2) > float(best_full_r2)
                ):
                    delta = (
                        float(current_r2) - float(best_full_r2)
                        if best_full_r2 is not None
                        else np.inf
                    )
                    improved = bool(delta >= float(_FIT_FULL_MIN_DELTA_R2))
                    best_full_r2 = float(current_r2)
                    best_full_result = outcome["result"]
                elif best_full_result is None:
                    best_full_result = outcome["result"]

                if improved:
                    no_progress = 0
                else:
                    no_progress += 1

                if (
                    step.min_r2 is not None
                    and current_r2 is not None
                    and float(current_r2) >= float(step.min_r2)
                ):
                    break
                if no_progress >= int(_FIT_FULL_PATIENCE):
                    break

            if best_full_result is not None:
                best_result = best_full_result
                full_r2 = best_full_result.get("r2")
                best_r2 = float(full_r2) if full_r2 is not None else best_r2
    else:
        for attempt in range(max_attempts):
            if context.is_cancelled():
                raise model.FitCancelledError("cancelled")
            attempt_seed = dict(context.seed_map)
            attempt_boundary_seeds = _clone_boundary_map(base_boundary_template)
            seed_source = "base"
            if attempt == 0 and _sibling_seed_info is not None:
                for key, val in dict(_sibling_seed_info.get("seed_map") or {}).items():
                    if key in seen_global and key not in step_fixed_filtered:
                        attempt_seed[str(key)] = float(val)
                for ch, arr in dict(_sibling_seed_info.get("boundary_seeds") or {}).items():
                    if str(ch) in attempt_boundary_seeds:
                        attempt_boundary_seeds[str(ch)] = np.asarray(
                            arr, dtype=float
                        ).reshape(-1)
                seed_source = "sibling_blend"
            if attempt > 0:
                rng = np.random.default_rng((context.rng_seed or 42) + int(attempt))
                if retry_mode == "jitter":
                    seed_source = "best_jitter"
                    attempt_seed = _jitter_seed(
                        rng, attempt_seed, max(0.05, float(step.retry_scale))
                    )
                elif retry_mode == "random":
                    seed_source = "random"
                    attempt_seed = _randomise_seed(rng, attempt_seed, 1.0, force_random=True)
                else:
                    if attempt == 1:
                        seed_source = "best_jitter"
                        attempt_seed = _jitter_seed(
                            rng, attempt_seed, max(0.05, float(step.retry_scale))
                        )
                    else:
                        seed_source = "random"
                        attempt_seed = _randomise_seed(
                            rng, attempt_seed, 1.0, force_random=True
                        )

            outcome = _run_attempt(
                _make_candidate(attempt_seed, attempt_boundary_seeds, seed_source),
                phase="full",
                screening_mode=False,
            )
            if outcome is None:
                continue
            attempt_r2 = outcome.get("r2")
            if (
                step.min_r2 is not None
                and attempt_r2 is not None
                and float(attempt_r2) >= float(step.min_r2)
            ):
                break

    if best_result is None:
        _fit_log.step_done("fail", message="All fit attempts failed.")
        return StepResult(
            status="fail",
            message="All fit attempts failed.",
            params_by_key=dict(context.seed_map),
            retries_used=max(0, len(retry_r2_history)),
            retry_r2_history=tuple(retry_r2_history),
        )

    # --- Feed results back into context ---
    params_by_key = dict(best_result.get("params_by_key") or {})
    for key, value in params_by_key.items():
        context.seed_map[key] = float(value)

    source_targets: List[str] = []
    channel_results = best_result.get("channel_results")
    if isinstance(channel_results, dict):
        for ch_target, ch_result in channel_results.items():
            ch_ratios = ch_result.get("boundary_ratios")
            if ch_ratios is not None:
                context.boundary_seeds[ch_target] = np.asarray(ch_ratios, dtype=float)
                source_targets.append(str(ch_target))
    elif best_result.get("boundary_ratios") is not None:
        ch_target = enabled_models[0].target_col
        context.boundary_seeds[ch_target] = np.asarray(
            best_result["boundary_ratios"], dtype=float
        )
        source_targets.append(str(ch_target))

    if source_targets:
        _propagate_linked_boundary_seeds(
            context,
            source_targets=tuple(source_targets),
        )

    step_r2 = best_result.get("r2")
    effective_free = (
        list(step.free_params)
        if step.free_params
        else list(seen_global - set(step_fixed_filtered))
    )

    # Compute per-channel R².
    per_channel_r2 = None
    if isinstance(channel_results, dict):
        per_channel_r2 = {}
        for ch, cr in channel_results.items():
            cr2 = cr.get("r2")
            per_channel_r2[ch] = float(cr2) if cr2 is not None else None

    # Check R² threshold.
    status = "pass"
    if step.min_r2 is not None and step_r2 is not None:
        if float(step_r2) < float(step.min_r2):
            status = "fail"
    _fit_log.step_done(
        status, r2=step_r2, retries_used=max(0, len(retry_r2_history) - 1)
    )

    return StepResult(
        status=status,
        message=f"R²={step_r2:.6f}" if step_r2 is not None else "Fit complete.",
        params_by_key=dict(params_by_key),
        boundary_ratios=(
            {
                ch: context.boundary_seeds[ch]
                for ch in enabled_targets
                if ch in context.boundary_seeds
            }
        ),
        r2=float(step_r2) if step_r2 is not None else None,
        per_channel_r2=per_channel_r2,
        channels=tuple(sorted(enabled_targets)),
        free_params=tuple(effective_free),
        fixed_params=tuple(step_fixed_filtered.keys()),
        retries_used=max(0, len(retry_r2_history) - 1),
        retry_r2_history=tuple(retry_r2_history),
    )


# ---------------------------------------------------------------------------
# New polymorphic pipeline
# ---------------------------------------------------------------------------


def run_procedure_pipeline(
    x_data: np.ndarray,
    y_data_by_channel: Mapping[str, np.ndarray],
    multi_model: Any,
    procedure: FitProcedure,
    seed_map: Mapping[str, float],
    bounds_map: Mapping[str, Tuple[float, float]],
    periodic_params: Optional[Mapping[str, bool]] = None,
    boundary_seeds: Optional[Mapping[str, np.ndarray]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    step_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    attempt_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    bound_values: Optional[Mapping[str, float]] = None,
    boundary_name_groups: Optional[Mapping[str, Sequence[Sequence[Any]]]] = None,
    rng_seed: Optional[int] = None,
    # -- Cross-file sibling seeding --
    captures: Optional[Mapping[str, Any]] = None,
    sibling_results: Optional[Mapping[str, Mapping[str, Any]]] = None,
    capture_seed_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute a multi-step procedure with polymorphic step types.

    Each step's ``execute(context)`` is called in order.  Results feed forward
    through the shared ``ProcedureContext``.

    Parameters (cross-file seeding)
    ------
    captures : dict, optional
        Capture values for the current file (used for sibling matching).
    sibling_results : dict, optional
        Normalised results from other files in the batch:
        ``{file_key: {"captures": ..., "params_by_key": ..., "boundary_ratios_by_channel": ..., "r2": ...}}``.
    capture_seed_keys : sequence of str, optional
        Which capture keys to use for sibling matching (e.g. ``["V", "f"]``).

    Returns
    -------
    dict with keys:
        - ``step_results``: list of StepResult objects
        - ``params_by_key``: final merged parameter values
        - ``r2``: final combined R² (from last fit step, or None)
        - ``channel_results``: final per-channel breakdown
        - ``stopped_at_step``: index of the step that caused early stop, or None
    """
    model = _get_model()

    context = ProcedureContext(
        seed_map=dict(seed_map),
        bounds_map=dict(bounds_map),
        periodic_params=periodic_params,
        boundary_seeds=dict(boundary_seeds or {}),
        x_data=x_data,
        y_data_by_channel=dict(y_data_by_channel),
        multi_model=multi_model,
        global_names=set(multi_model.global_param_names),
        bound_values=dict(bound_values or {}),
        cancel_check=cancel_check,
        rng_seed=rng_seed,
        boundary_name_to_ids=(
            dict(boundary_name_groups)
            if boundary_name_groups is not None
            else _default_boundary_name_groups(multi_model)
        ),
        captures=dict(captures or {}),
        sibling_results=dict(sibling_results or {}),
        capture_seed_keys=tuple(capture_seed_keys or ()),
        seed_from_siblings=bool(procedure.seed_from_siblings),
    )
    context.attempt_callback = attempt_callback

    step_results: List[StepResult] = []
    last_fit_result: Optional[StepResult] = None
    stopped_at_step: Optional[int] = None

    n_steps = len(procedure.steps)
    context.step_total = n_steps
    _fit_log.procedure_start(procedure.name, n_steps)
    procedure_t0 = time.perf_counter()

    for step_idx, step in enumerate(procedure.steps):
        if context.is_cancelled():
            raise model.FitCancelledError("cancelled")

        context.step_index = step_idx + 1
        # Log non-fit steps here; FitStep logs itself via _execute_fit_step.
        if not isinstance(step, FitStep):
            _fit_log.step_start(
                step_idx + 1,
                n_steps,
                step.step_type,
                step.label or step.step_label,
            )
        step_t0 = time.perf_counter()
        result = step.execute(context)
        if not isinstance(step, FitStep):
            _fit_log.step_done(
                result.status,
                r2=result.r2,
                elapsed=time.perf_counter() - step_t0,
                message=result.message,
            )
        step_results.append(result)

        # Track last fit-type result for final R².
        if isinstance(step, FitStep) and result.r2 is not None:
            last_fit_result = result

        # Report progress.
        step_elapsed = time.perf_counter() - step_t0
        result_dict = {
            "step_index": step_idx,
            "step_total": n_steps,
            "step_type": step.step_type,
            "label": step.label or f"Step {step_idx + 1}",
            "status": result.status,
            "message": result.message,
            "params_by_key": dict(result.params_by_key),
            "boundary_ratios": (
                {
                    str(target): np.asarray(values, dtype=float).reshape(-1).tolist()
                    for target, values in dict(result.boundary_ratios or {}).items()
                }
                if result.boundary_ratios is not None
                else None
            ),
            "r2": result.r2,
            "per_channel_r2": result.per_channel_r2,
            "channels": list(result.channels),
            "free_params": list(result.free_params),
            "fixed_params": list(result.fixed_params),
            "retries_used": result.retries_used,
            "retry_r2_history": list(result.retry_r2_history),
            "elapsed": step_elapsed,
        }
        if step_callback is not None:
            step_callback(step_idx, result_dict)

        # R² gate: FitStep checks its own threshold and sets status="fail".
        if isinstance(step, FitStep) and result.status == "fail":
            if str(step.on_fail or "stop") != "continue":
                stopped_at_step = step_idx
                break

    # Build final per-channel boundary state from context so channels fitted
    # earlier in the procedure are not dropped when the last fit step targets
    # a different channel.
    final_r2 = last_fit_result.r2 if last_fit_result is not None else None
    _fit_log.procedure_done(time.perf_counter() - procedure_t0, r2=final_r2)
    final_channel_results = {}
    for ch_model in tuple(getattr(multi_model, "channel_models", ()) or ()):
        ch_target = str(getattr(ch_model, "target_col", "") or "").strip()
        if not ch_target:
            continue
        n_boundaries = max(0, len(getattr(ch_model, "segment_exprs", ()) or ()) - 1)
        ratios = context.ensure_boundary_seed_size(ch_target, n_boundaries)
        final_channel_results[ch_target] = {
            "boundary_ratios": np.asarray(ratios, dtype=float).reshape(-1).copy()
        }

    # Preserve per-channel R² from the last fit step where available.
    if last_fit_result is not None and last_fit_result.per_channel_r2:
        for ch, r2 in last_fit_result.per_channel_r2.items():
            ch_key = str(ch)
            if ch_key not in final_channel_results:
                final_channel_results[ch_key] = {}
            final_channel_results[ch_key]["r2"] = r2

    return {
        "step_results": [
            {
                "step_index": i,
                "step_type": procedure.steps[i].step_type
                if i < len(procedure.steps)
                else "?",
                "label": sr.message
                if not (procedure.steps[i].label if i < len(procedure.steps) else "")
                else (
                    procedure.steps[i].label
                    if i < len(procedure.steps)
                    else f"Step {i + 1}"
                ),
                "status": sr.status,
                "message": sr.message,
                "params_by_key": dict(sr.params_by_key),
                "boundary_ratios": (
                    {
                        str(target): np.asarray(values, dtype=float)
                        .reshape(-1)
                        .tolist()
                        for target, values in dict(sr.boundary_ratios or {}).items()
                    }
                    if sr.boundary_ratios is not None
                    else None
                ),
                "r2": sr.r2,
                "per_channel_r2": sr.per_channel_r2,
                "channels": list(sr.channels),
                "free_params": list(sr.free_params),
                "fixed_params": list(sr.fixed_params),
                "retries_used": sr.retries_used,
                "retry_r2_history": list(sr.retry_r2_history),
            }
            for i, sr in enumerate(step_results)
        ],
        "params_by_key": dict(context.seed_map),
        "r2": last_fit_result.r2 if last_fit_result is not None else None,
        "channel_results": final_channel_results,
        "stopped_at_step": stopped_at_step,
    }
