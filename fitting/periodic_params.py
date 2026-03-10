"""Shared helpers for periodic-parameter configuration."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def normalize_periodic_params(
    periodic_params: Optional[Mapping[str, Any]],
) -> Dict[str, bool]:
    """Normalize periodic parameter mapping to ``{param_key: bool}``."""
    out: Dict[str, bool] = {}
    for raw_key, raw_value in dict(periodic_params or {}).items():
        key = str(raw_key).strip()
        if not key:
            continue
        out[key] = bool(raw_value)
    return out

