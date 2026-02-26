"""Helpers for storing batch row fit data in a single ``fit_results`` mapping."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Dict


FIT_RESULT_KEYS = (
    "params",
    "r2",
    "error",
    "boundary_ratios",
    "boundary_values",
    "channel_results",
)
_FIT_RESULT_KEY_SET = set(FIT_RESULT_KEYS)


def _normalized_fit_results(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in FIT_RESULT_KEYS:
        if key in value:
            out[key] = value.get(key)
    return out


def ensure_fit_results(container: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Ensure ``container['fit_results']`` exists."""
    fit_results = _normalized_fit_results(container.get("fit_results"))
    container["fit_results"] = fit_results
    return fit_results


def fit_get(container: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    """Read a fit value from ``fit_results``."""
    if key not in _FIT_RESULT_KEY_SET:
        if isinstance(container, Mapping):
            return container.get(key, default)
        return default
    if not isinstance(container, Mapping):
        return default

    fit_results = container.get("fit_results")
    if not isinstance(fit_results, Mapping):
        return default
    if key not in fit_results:
        return default
    return fit_results.get(key)


def fit_set(container: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Write a fit value into ``fit_results``."""
    if key not in _FIT_RESULT_KEY_SET:
        container[key] = value
        return
    fit_results = ensure_fit_results(container)
    fit_results[key] = value
    container["fit_results"] = fit_results


def canonicalize_fit_row(row: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return a plain dict row with canonical ``fit_results`` storage."""
    normalized: Dict[str, Any] = dict(row or {})
    ensure_fit_results(normalized)
    return normalized
