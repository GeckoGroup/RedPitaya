from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from model import (
    boundary_ratios_to_positions,
    default_boundary_ratios,
    pcts_to_boundary_ratios,
)


BoundaryId = Tuple[str, int]
BoundaryLinkGroup = Sequence[BoundaryId]


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, atol=1e-12, rtol=0.0))


def _normalize_positions(positions: Sequence[float]) -> np.ndarray:
    arr = np.asarray(positions, dtype=float).reshape(-1)
    if arr.size <= 0:
        return np.asarray([], dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.maximum.accumulate(arr)
    return np.asarray(arr, dtype=float)


def _normalize_channel_ratios(ratios: Sequence[float], count: int) -> np.ndarray:
    n = max(0, int(count))
    if n <= 0:
        return np.asarray([], dtype=float)
    arr = np.asarray(ratios, dtype=float).reshape(-1)
    if arr.size != n:
        arr = default_boundary_ratios(n)
    arr = np.clip(arr, 0.0, 1.0)
    positions = boundary_ratios_to_positions(arr, n)
    positions = _normalize_positions(positions)
    return np.asarray(pcts_to_boundary_ratios(positions), dtype=float).reshape(-1)


@dataclass
class BoundaryState:
    """Canonical boundary-ratio store keyed by channel target."""

    _topology: Dict[str, int] = field(default_factory=dict)
    _ratios_by_target: Dict[str, np.ndarray] = field(default_factory=dict)
    _primary_target: Optional[str] = None

    def targets(self) -> Tuple[str, ...]:
        return tuple(self._topology.keys())

    @property
    def primary_target(self) -> Optional[str]:
        if self._primary_target in self._topology:
            return self._primary_target
        if self._topology:
            return next(iter(self._topology))
        return None

    def topology(self) -> Dict[str, int]:
        return {str(target): int(count) for target, count in self._topology.items()}

    def set_topology(
        self,
        topology: Mapping[str, int],
        *,
        primary_target: Optional[str] = None,
        preserve_existing: bool = True,
    ) -> None:
        clean_topology: Dict[str, int] = {}
        for raw_target, raw_count in dict(topology or {}).items():
            target = str(raw_target).strip()
            if not target:
                continue
            try:
                count = int(raw_count)
            except Exception:
                continue
            clean_topology[target] = max(0, count)

        if primary_target is None:
            if self._primary_target in clean_topology:
                primary = self._primary_target
            elif clean_topology:
                primary = next(iter(clean_topology))
            else:
                primary = None
        else:
            primary = str(primary_target)
            if primary not in clean_topology:
                primary = next(iter(clean_topology), None)

        old = dict(self._ratios_by_target)
        new_map: Dict[str, np.ndarray] = {}
        for target, count in clean_topology.items():
            seed = None
            if preserve_existing:
                old_arr = old.get(target)
                if old_arr is not None and np.asarray(old_arr).reshape(-1).size == int(
                    count
                ):
                    seed = old_arr
            if seed is None:
                seed = default_boundary_ratios(count)
            new_map[target] = _normalize_channel_ratios(seed, count)

        self._topology = clean_topology
        self._ratios_by_target = new_map
        self._primary_target = primary

    def channel_count(self, target: str) -> int:
        return int(self._topology.get(str(target), 0))

    def channel_ratios(self, target: str) -> np.ndarray:
        key = str(target)
        count = self.channel_count(key)
        arr = self._ratios_by_target.get(key)
        if arr is None or np.asarray(arr).reshape(-1).size != count:
            arr = _normalize_channel_ratios(arr if arr is not None else [], count)
            self._ratios_by_target[key] = arr
        return np.asarray(arr, dtype=float).reshape(-1).copy()

    def primary_ratios(self) -> np.ndarray:
        target = self.primary_target
        if target is None:
            return np.asarray([], dtype=float)
        return self.channel_ratios(target)

    def set_channel_ratios(self, target: str, ratios: Sequence[float]) -> bool:
        key = str(target)
        if key not in self._topology:
            return False
        count = self.channel_count(key)
        old = self.channel_ratios(key)
        normalized = _normalize_channel_ratios(ratios, count)
        if _allclose(old, normalized):
            self._ratios_by_target[key] = normalized
            return False
        self._ratios_by_target[key] = normalized
        return True

    def update_channels(
        self, ratios_by_target: Mapping[str, Sequence[float]]
    ) -> Set[str]:
        changed: Set[str] = set()
        for raw_target, raw_ratios in dict(ratios_by_target or {}).items():
            target = str(raw_target)
            if target not in self._topology:
                continue
            if self.set_channel_ratios(target, raw_ratios):
                changed.add(target)
        return changed

    def set_primary_ratios(self, ratios: Sequence[float]) -> bool:
        target = self.primary_target
        if target is None:
            return False
        return self.set_channel_ratios(target, ratios)

    def boundary_ratio(self, target: str, idx: int) -> Optional[float]:
        key = str(target)
        try:
            i = int(idx)
        except Exception:
            return None
        if i < 0:
            return None
        ratios = self.channel_ratios(key)
        if i >= ratios.size:
            return None
        value = float(ratios[i])
        if not np.isfinite(value):
            return None
        return float(np.clip(value, 0.0, 1.0))

    def set_boundary_ratio(self, target: str, idx: int, value: float) -> bool:
        key = str(target)
        try:
            i = int(idx)
        except Exception:
            return False
        if i < 0:
            return False
        ratios = self.channel_ratios(key)
        if i >= ratios.size:
            return False
        ratios[i] = float(np.clip(float(value), 0.0, 1.0))
        return self.set_channel_ratios(key, ratios)

    def apply_link_groups(
        self,
        link_groups: Iterable[BoundaryLinkGroup],
        *,
        source_boundary: Optional[BoundaryId] = None,
        source_target: Optional[str] = None,
        prefer_targets: Optional[Iterable[str]] = None,
    ) -> Set[str]:
        source_bid = None
        if source_boundary is not None:
            try:
                source_bid = (str(source_boundary[0]), int(source_boundary[1]))
            except Exception:
                source_bid = None
        source_target_key = (
            str(source_target).strip() if source_target not in (None, "") else None
        )
        preferred = []
        for item in list(prefer_targets or []):
            key = str(item).strip()
            if key and key not in preferred:
                preferred.append(key)

        pending: Dict[str, np.ndarray] = {}

        for group in tuple(link_groups or ()):  # make deterministic during edits
            members = []
            for raw_target, raw_idx in tuple(group or ()):  # tolerate malformed groups
                target = str(raw_target)
                if target not in self._topology:
                    continue
                try:
                    idx = int(raw_idx)
                except Exception:
                    continue
                if idx < 0 or idx >= self.channel_count(target):
                    continue
                members.append((target, idx))
            if len(members) < 2:
                continue

            for target, _idx in members:
                if target not in pending:
                    pending[target] = self.channel_ratios(target)

            reference_value = None
            if source_bid is not None and source_bid in members:
                source_arr = pending.get(source_bid[0])
                if source_arr is not None and source_bid[1] < source_arr.size:
                    reference_value = float(source_arr[source_bid[1]])

            if reference_value is None and source_target_key:
                for target, idx in members:
                    if target != source_target_key:
                        continue
                    source_arr = pending.get(target)
                    if source_arr is not None and idx < source_arr.size:
                        reference_value = float(source_arr[idx])
                        break

            if reference_value is None and preferred:
                for pref_target in preferred:
                    for target, idx in members:
                        if target != pref_target:
                            continue
                        pref_arr = pending.get(target)
                        if pref_arr is not None and idx < pref_arr.size:
                            reference_value = float(pref_arr[idx])
                            break
                    if reference_value is not None:
                        break

            if reference_value is None:
                first_target, first_idx = members[0]
                first_arr = pending.get(first_target)
                if first_arr is not None and first_idx < first_arr.size:
                    reference_value = float(first_arr[first_idx])

            if reference_value is None:
                continue
            reference_value = float(np.clip(reference_value, 0.0, 1.0))

            for target, idx in members:
                arr = pending.get(target)
                if arr is None or idx >= arr.size:
                    continue
                arr[idx] = reference_value

        changed_targets: Set[str] = set()
        for target, pending_ratios in pending.items():
            count = self.channel_count(target)
            old = self.channel_ratios(target)
            normalized = _normalize_channel_ratios(pending_ratios, count)
            self._ratios_by_target[target] = normalized
            if not _allclose(old, normalized):
                changed_targets.add(target)

        return changed_targets

    def as_per_channel_map(self) -> Dict[str, np.ndarray]:
        return {target: self.channel_ratios(target) for target in self.targets()}
