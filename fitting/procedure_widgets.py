"""Procedure panel widget — extracted from fit_gui.py.

Provides ``ProcedurePanel``, a self-contained QWidget that implements the
Procedures tab: step card editor, run/cancel controls, results table, and
status display.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QComboBox,
    QLineEdit,
    QScrollArea,
    QCheckBox,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from model import (
    MultiChannelModelDefinition,
)
from procedure_steps import (
    ProcedureStepBase,
    FitStep,
    SetParameterStep,
    ParameterAssignment,
    SetBoundariesStep,
    BoundaryAssignment,
    RandomizeSeedsStep,
    available_step_types,
)
from procedure import FitProcedure
from widgets import RichTextComboBox
from expression import parameter_symbol_to_html


def _finite_float_or_none(value) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


_ASCII_TO_SUBSCRIPT_BOUNDARY_TRANS = str.maketrans(
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
_SUBSCRIPT_TO_ASCII_BOUNDARY_TRANS = str.maketrans(
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
_BOUNDARY_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _boundary_name_aliases(name: str) -> List[str]:
    base = str(name or "").strip()
    if not base:
        return []
    out: List[str] = []
    for candidate in (
        base,
        str(base).translate(_SUBSCRIPT_TO_ASCII_BOUNDARY_TRANS),
        str(base).translate(_ASCII_TO_SUBSCRIPT_BOUNDARY_TRANS),
    ):
        key = str(candidate).strip()
        if key and key not in out:
            out.append(key)
    return out


def _resolve_boundary_alias(name: str, valid_names: set[str]) -> str:
    for alias in _boundary_name_aliases(str(name or "")):
        if alias in valid_names:
            return str(alias)
    return ""


def _normalise_boundary_reference_text(text: str) -> str:
    return str(text or "").translate(_ASCII_TO_SUBSCRIPT_BOUNDARY_TRANS)


def _normalise_boundary_expression_text(text: str) -> str:
    raw = str(text or "")

    def _replace_ident(match) -> str:
        token = str(match.group(0))
        if any(ch.isdigit() for ch in token):
            return token.translate(_ASCII_TO_SUBSCRIPT_BOUNDARY_TRANS)
        return token

    return _BOUNDARY_IDENT_RE.sub(_replace_ident, raw)


# ---------------------------------------------------------------------------
# Protocol for the host (ManualFitGUI adapter)
# ---------------------------------------------------------------------------


class ProcedureHost:
    """Minimal interface that ProcedurePanel requires from its host GUI.

    Subclass or duck-type this in ManualFitGUI to provide data/model access.
    """

    def proc_available_params(self) -> list:
        return []

    def proc_available_channels(self) -> list:
        return []

    def proc_available_capture_keys(self) -> list:
        return []

    def proc_capture_preview_values(self) -> dict:
        return {}

    def proc_build_fit_context(self, fixed_params=None) -> dict:
        _ = fixed_params
        return {}

    def proc_get_multi_channel_model(self) -> Optional[MultiChannelModelDefinition]:
        return None

    def proc_get_piecewise_model(self):
        return None

    def proc_get_current_data(self):
        return None

    def proc_get_x_channel(self) -> str:
        return "TIME"

    def proc_get_boundary_ratios_per_channel(self) -> dict:
        return {}

    def proc_available_boundary_groups(self) -> list:
        """Return [(display_name, ((target, idx), ...)), ...] for boundary groups."""
        return []

    def proc_get_smoothing(self) -> Tuple[bool, int]:
        return (False, 1)

    def proc_get_random_restarts(self) -> int:
        return 0

    def proc_channel_display_name(self, channel_key: str) -> str:
        """Return the user-facing display name for *channel_key*."""
        return str(channel_key)

    def proc_display_symbol_html(self, key: str) -> str:
        return parameter_symbol_to_html(key)

    def proc_on_fit_finished(self, result: dict) -> None:
        pass

    def proc_autosave(self) -> None:
        pass

    def proc_log(self, message: str) -> None:
        pass

    def proc_current_dir(self) -> str:
        return "."


# ---------------------------------------------------------------------------
# Step results table widget
# ---------------------------------------------------------------------------


class StepResultsTable(QWidget):
    """Table showing per-step results after a procedure run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            [
                "#",
                "Label",
                "Type",
                "Status",
                "R²",
                "Retries",
                "Key Changes",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(180)
        layout.addWidget(self.table)

    _STATUS_COLORS = {
        "pass": QColor("#dcfce7"),  # Green
        "fail": QColor("#fee2e2"),  # Red
        "skipped": QColor("#f1f5f9"),  # Gray
    }

    def populate(self, step_results: list):
        """Fill the table from a list of step result dicts."""
        self.table.setRowCount(0)
        if not step_results:
            return

        self.table.setRowCount(len(step_results))
        for row_idx, sr in enumerate(step_results):
            step_idx = sr.get("step_index", row_idx)
            step_type = sr.get("step_type", "fit")
            label = sr.get("label", f"Step {step_idx + 1}")
            status = sr.get("status", "pass")
            r2 = sr.get("r2")
            retries = sr.get("retries_used", 0)
            params = sr.get("params_by_key") or {}

            # Format key changes: show up to 4 params.
            free_keys = sr.get("free_params") or []
            change_parts = []
            for k in free_keys[:4]:
                v = params.get(k)
                if v is not None:
                    change_parts.append(f"{k}={v:.4g}")
            key_changes = ", ".join(change_parts)
            if len(free_keys) > 4:
                key_changes += f" (+{len(free_keys) - 4})"

            items = [
                QTableWidgetItem(str(step_idx + 1)),
                QTableWidgetItem(str(label)),
                QTableWidgetItem(str(step_type)),
                QTableWidgetItem(str(status).upper()),
                QTableWidgetItem(f"{r2:.6f}" if r2 is not None else "N/A"),
                QTableWidgetItem(str(retries) if retries > 0 else ""),
                QTableWidgetItem(key_changes),
            ]

            bg = self._STATUS_COLORS.get(status, QColor("#ffffff"))
            for col, item in enumerate(items):
                item.setBackground(bg)
                self.table.setItem(row_idx, col, item)


# ---------------------------------------------------------------------------
# ProcedurePanel — the main widget
# ---------------------------------------------------------------------------


class ProcedurePanel(QWidget):
    """Self-contained Procedure tab widget.

    Communicates with the host GUI exclusively through a ``ProcedureHost``
    adapter, avoiding tight coupling.
    """

    # Signals
    procedure_changed = pyqtSignal()  # emitted when steps/name change

    def __init__(self, host: ProcedureHost, parent=None):
        super().__init__(parent)
        self.host = host
        self._procedure_steps: List[ProcedureStepBase] = []
        self._procedure_name = "Procedure"
        self._seed_from_siblings = False
        self._procedure_running = False
        self._procedure_task_id = None
        self._run_btn = None
        self._cancel_btn = None
        self._procedure_log_lines: List[str] = []
        self._procedure_log_max_lines = 200
        self._live_prev_params_by_key: Dict[str, float] = {}
        self._last_step_results: List[dict] = []
        self._last_status_text = ""
        self._build_ui()
        self._rebuild_step_cards()

    # ── Public API ────────────────────────────────────────────────

    @property
    def procedure_steps(self) -> list:
        return self._procedure_steps

    @procedure_steps.setter
    def procedure_steps(self, steps: list):
        self._procedure_steps = list(steps)
        self._rebuild_step_cards()

    @property
    def procedure_name(self) -> str:
        return self._procedure_name

    @procedure_name.setter
    def procedure_name(self, name: str):
        self._procedure_name = str(name).strip() or "Procedure"

    def build_procedure(self) -> FitProcedure:
        return FitProcedure(
            name=self._procedure_name,
            steps=tuple(self._procedure_steps),
            seed_from_siblings=bool(self._seed_from_siblings),
        )

    def prune_invalid_params(self):
        """Remove references to params/channels that no longer exist."""
        valid_params = set(self.host.proc_available_params())
        valid_channels = set(self.host.proc_available_channels())
        valid_boundaries = {
            str(name)
            for name, _members in (self.host.proc_available_boundary_groups() or [])
        }
        changed = False
        for idx, step in enumerate(self._procedure_steps):
            if isinstance(step, FitStep):
                new_free = tuple(p for p in step.free_params if p in valid_params)
                new_fixed = tuple(p for p in step.fixed_params if p in valid_params)
                if step.channels is None:
                    new_channels = None  # preserve "all channels"
                else:
                    new_channels = tuple(
                        c for c in step.channels if c in valid_channels
                    )
                new_locked = tuple(
                    name
                    for name in step.locked_boundary_names
                    if name in valid_boundaries
                )
                new_bound = tuple(
                    (k, v) for k, v in step.bound_params if k in valid_params
                )
                if (
                    new_free != step.free_params
                    or new_fixed != step.fixed_params
                    or new_channels != step.channels
                    or new_locked != step.locked_boundary_names
                    or new_bound != step.bound_params
                ):
                    self._procedure_steps[idx] = FitStep(
                        channels=new_channels,
                        free_params=new_free,
                        fixed_params=new_fixed,
                        bound_params=new_bound,
                        min_r2=step.min_r2,
                        max_retries=step.max_retries,
                        retry_scale=step.retry_scale,
                        retry_mode=step.retry_mode,
                        locked_boundary_names=new_locked,
                        on_fail=step.on_fail,
                        label=step.label,
                    )
                    changed = True
            elif isinstance(step, SetParameterStep):
                new_assignments = []
                for assignment in step.assignments:
                    if assignment.target_key not in valid_params:
                        continue
                    if (
                        assignment.source_kind == "param"
                        and assignment.source_key not in valid_params
                    ):
                        continue
                    new_assignments.append(assignment)
                if tuple(new_assignments) != tuple(step.assignments):
                    self._procedure_steps[idx] = SetParameterStep(
                        assignments=tuple(new_assignments),
                        label=step.label,
                    )
                    changed = True
            elif isinstance(step, SetBoundariesStep):
                new_assignments = []
                for assignment in step.assignments:
                    target_name = _resolve_boundary_alias(
                        str(assignment.target_name), valid_boundaries
                    )
                    if target_name not in valid_boundaries:
                        continue
                    source_name = str(assignment.source_name or "")
                    if str(
                        assignment.source_kind
                    ) == "boundary" and not _resolve_boundary_alias(
                        source_name, valid_boundaries
                    ):
                        continue
                    if str(assignment.source_kind) == "boundary":
                        source_name = _resolve_boundary_alias(
                            source_name, valid_boundaries
                        )
                    new_assignments.append(
                        BoundaryAssignment(
                            target_name=target_name,
                            source_kind=str(assignment.source_kind),
                            source_name=source_name,
                            literal_value=assignment.literal_value,
                            expression=str(assignment.expression or ""),
                            on_missing=str(assignment.on_missing or "skip"),
                        )
                    )
                if tuple(new_assignments) != tuple(step.assignments):
                    self._procedure_steps[idx] = SetBoundariesStep(
                        assignments=tuple(new_assignments),
                        label=step.label,
                    )
                    changed = True
        if changed:
            self._rebuild_step_cards()

    def restore_from_serialized(self, data: Mapping):
        """Restore procedure from serialised dict (fit_details or standalone)."""
        steps_raw = data.get("steps") or ()
        for raw in steps_raw:
            if not isinstance(raw, Mapping):
                continue
            if not str(raw.get("step_type") or "").strip():
                raise ValueError(
                    "Unsupported procedure format: each step must include 'step_type'."
                )
        proc = FitProcedure.deserialize(data)
        self._procedure_steps = list(proc.steps)
        self._procedure_name = str(data.get("name") or "Procedure")
        self._seed_from_siblings = bool(proc.seed_from_siblings)
        cb = getattr(self, "_seed_from_siblings_cb", None)
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(self._seed_from_siblings)
            cb.blockSignals(False)
        raw_step_results = data.get("last_step_results")
        if isinstance(raw_step_results, (list, tuple)):
            self._last_step_results = [
                dict(item) for item in raw_step_results if isinstance(item, Mapping)
            ]
        else:
            self._last_step_results = []
        self._clear_run_log()
        raw_key_changes = data.get("latest_key_changes")
        if isinstance(raw_key_changes, (list, tuple)):
            self._procedure_log_lines = [
                str(line).strip() for line in raw_key_changes if str(line).strip()
            ][-int(self._procedure_log_max_lines) :]
        self._last_status_text = str(data.get("last_status") or "")
        self.prune_invalid_params()
        self._rebuild_step_cards()
        self._restore_run_state_ui()

    def serialize_procedure(self) -> dict:
        """Return serialised procedure dict for persistence."""
        payload = self.build_procedure().serialize()
        if self._last_step_results:
            payload["last_step_results"] = [
                self._json_safe_value(item) for item in list(self._last_step_results)
            ]
        if self._procedure_log_lines:
            payload["latest_key_changes"] = list(self._procedure_log_lines)
            payload["run_log"] = list(self._procedure_log_lines)
        if self._last_status_text:
            payload["last_status"] = str(self._last_status_text)
        return payload

    def record_external_procedure_start(
        self,
        *,
        procedure_name: str,
        file_label: str = "",
        step_count: Optional[int] = None,
    ) -> None:
        """Update live UI state for procedure runs started outside this panel."""
        proc_name = str(procedure_name).strip() or "Procedure"
        file_name = str(file_label).strip()
        file_text = f" [{file_name}]" if file_name else ""
        steps_text = (
            f" ({int(step_count)} steps)"
            if step_count is not None and int(step_count) > 0
            else ""
        )
        self._clear_run_log()
        self._set_status_text(f"Running {proc_name}{steps_text}{file_text}...")
        self.host.proc_autosave()

    def record_external_procedure_result(
        self, result: Mapping, *, file_label: str = ""
    ) -> None:
        """Show and persist procedure output produced by external run paths."""
        result_map = dict(result or {})
        file_name = str(file_label).strip()
        step_results = result_map.get("step_results")
        if isinstance(step_results, (list, tuple)):
            self._last_step_results = [
                dict(item) for item in step_results if isinstance(item, Mapping)
            ]
            self._results_table.populate(self._last_step_results)
            self._rebuild_live_key_changes(self._last_step_results)
        else:
            self._last_step_results = []
            self._results_table.populate([])
            self._clear_run_log()

        stopped = result_map.get("stopped_at_step")
        r2 = _finite_float_or_none(result_map.get("r2"))
        file_text = f" [{file_name}]" if file_name else ""
        if stopped is not None:
            try:
                step_no = int(stopped) + 1
                msg = f"Procedure stopped early at step {step_no}{file_text}."
            except Exception:
                msg = f"Procedure stopped early{file_text}."
        else:
            r2_text = f" R²={r2:.6f}" if r2 is not None else ""
            msg = f"Procedure complete{file_text}.{r2_text}".strip()
        self._set_status_text(msg)
        self.host.proc_autosave()

    def record_external_procedure_failure(
        self, message: str, *, file_label: str = ""
    ) -> None:
        file_name = str(file_label).strip()
        msg = str(message).strip() or "Procedure failed."
        display_msg = f"Procedure failed [{file_name}]: {msg}" if file_name else msg
        self._set_status_text(display_msg)
        self.host.proc_autosave()

    def record_external_procedure_cancelled(self, *, file_label: str = "") -> None:
        file_name = str(file_label).strip()
        file_text = f" [{file_name}]" if file_name else ""
        msg = f"Procedure cancelled{file_text}."
        self._set_status_text(msg)
        self.host.proc_autosave()

    def clear_run_history(self) -> None:
        self._clear_run_log()
        self._last_step_results = []
        self._last_status_text = ""
        self._restore_run_state_ui()

    def refresh_display_context(self) -> None:
        """Refresh card labels/tooltips after host display-name changes."""
        self._rebuild_step_cards()
        # Keep status/results widgets in sync with current step ordering.
        self._restore_run_state_ui()

    def request_close_shutdown(self, *, force_terminate: bool = False) -> None:
        """Request cancellation of an active procedure run during window close."""
        _ = bool(force_terminate)
        task_id = getattr(self, "_procedure_task_id", None)
        if task_id is not None:
            worker_thread = getattr(self.host, "_fit_worker_thread", None)
            if worker_thread is not None:
                worker_thread.cancel_tasks({task_id})
        self._disconnect_worker_signals()

    def finalize_close_shutdown(self) -> None:
        """Reset run-state pointers/buttons after host forces shutdown on close."""
        self._procedure_running = False
        self._procedure_task_id = None
        if self._run_btn is not None:
            self._run_btn.setEnabled(True)
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(False)

    def _set_status_text(self, text: str) -> None:
        self._last_status_text = str(text or "").strip()
        self._status_label.setText(self._last_status_text)

    def _coerce_params_by_key(self, params_by_key) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not isinstance(params_by_key, Mapping):
            return out
        for key, raw_value in params_by_key.items():
            numeric = _finite_float_or_none(raw_value)
            if numeric is None:
                continue
            out[str(key)] = float(numeric)
        return out

    @classmethod
    def _json_safe_value(cls, value):
        if isinstance(value, np.ndarray):
            return np.asarray(value).reshape(-1).tolist()
        if isinstance(value, np.generic):
            try:
                return value.item()
            except Exception:
                return str(value)
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe_value(raw_val)
                for key, raw_val in dict(value).items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_safe_value(item) for item in value]
        return value

    @staticmethod
    def _dedupe_str_values(values) -> List[str]:
        seen: set = set()
        ordered: List[str] = []
        for raw in values or ():
            key = str(raw).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    def _step_candidate_keys(self, step_idx: int, step_result: Mapping) -> List[str]:
        free_params = self._dedupe_str_values(step_result.get("free_params") or ())
        if free_params:
            return free_params
        step_obj: ProcedureStepBase | None = None
        if 0 <= int(step_idx) < len(self._procedure_steps):
            step_obj = self._procedure_steps[int(step_idx)]
        if isinstance(step_obj, SetParameterStep):
            return self._dedupe_str_values(
                assignment.target_key for assignment in step_obj.assignments
            )
        if isinstance(step_obj, RandomizeSeedsStep) and step_obj.params:
            return self._dedupe_str_values(step_obj.params)
        return []

    def _format_step_key_change_line(
        self,
        step_idx: int,
        step_result: Mapping,
        *,
        previous_params: Optional[Mapping[str, float]] = None,
    ) -> None | str:
        params_by_key = self._coerce_params_by_key(step_result.get("params_by_key"))
        if not params_by_key:
            return None
        keys = self._step_candidate_keys(step_idx, step_result)
        if not keys and previous_params:
            changed = []
            for key, value in params_by_key.items():
                prev = _finite_float_or_none(previous_params.get(key))
                if prev is None:
                    continue
                if not np.isclose(
                    float(value), float(prev), rtol=1e-9, atol=1e-12, equal_nan=False
                ):
                    changed.append(str(key))
            keys = self._dedupe_str_values(changed)
        if not keys:
            return None

        value_parts: List[str] = []
        for key in keys:
            if key not in params_by_key:
                continue
            value_parts.append(f"{key}={params_by_key[key]:.6g}")
        if not value_parts:
            return None
        max_shown = 6
        suffix = ""
        if len(value_parts) > max_shown:
            suffix = f" (+{len(value_parts) - max_shown})"
            value_parts = value_parts[:max_shown]
        label = str(step_result.get("label") or f"Step {int(step_idx) + 1}")
        return f"{label}: {', '.join(value_parts)}{suffix}"

    def _rebuild_live_key_changes(self, step_results: List[Mapping]) -> None:
        lines: List[str] = []
        prev_params: Dict[str, float] = {}
        for fallback_idx, raw_result in enumerate(step_results):
            if not isinstance(raw_result, Mapping):
                continue
            raw_step_idx = raw_result.get("step_index", fallback_idx)
            try:
                step_idx = int(raw_step_idx)
            except Exception:
                step_idx = int(fallback_idx)
            line = self._format_step_key_change_line(
                step_idx,
                raw_result,
                previous_params=prev_params,
            )
            if line:
                lines.append(line)
            params_by_key = self._coerce_params_by_key(raw_result.get("params_by_key"))
            if params_by_key:
                prev_params = params_by_key
        self._live_prev_params_by_key = dict(prev_params)
        self._procedure_log_lines = lines[-int(self._procedure_log_max_lines) :]
        self._render_run_log()

    def _clear_run_log(self) -> None:
        self._procedure_log_lines = []
        self._live_prev_params_by_key = {}
        self._render_run_log()

    def _render_run_log(self) -> None:
        log_widget = getattr(self, "_results_log", None)
        if log_widget is None:
            return
        log_widget.setPlainText("\n".join(self._procedure_log_lines))
        scrollbar = log_widget.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _append_run_log_line(
        self,
        line: str,
        *,
        section_key: Optional[str] = None,
        section_title: str = "",
    ) -> None:
        text = str(line or "").strip()
        if not text:
            return
        _ = section_key
        _ = section_title
        self._procedure_log_lines.append(text)
        if len(self._procedure_log_lines) > int(self._procedure_log_max_lines):
            self._procedure_log_lines = self._procedure_log_lines[
                -int(self._procedure_log_max_lines) :
            ]
        self._render_run_log()

    def _restore_run_state_ui(self) -> None:
        self._status_label.setText(str(self._last_status_text or ""))
        self._results_table.populate(self._last_step_results)
        if self._last_step_results:
            self._rebuild_live_key_changes(self._last_step_results)
        else:
            self._render_run_log()

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header row: just the Add Step dropdown.
        header = QHBoxLayout()
        header.setSpacing(6)

        # Add Step dropdown.
        self._add_step_btn = QPushButton("+ Add Step")
        self._add_step_btn.setProperty("primary", True)
        self._add_step_menu = QMenu(self._add_step_btn)
        for step_type, step_label in available_step_types():
            action = self._add_step_menu.addAction(step_label)
            action.setData(step_type)
            action.triggered.connect(
                lambda _checked, st=step_type: self._add_step(st)
            )
        self._add_step_btn.setMenu(self._add_step_menu)
        header.addWidget(self._add_step_btn)
        self._template_btn = QPushButton("Templates")
        self._template_menu = QMenu(self._template_btn)
        tpl_action = self._template_menu.addAction("MI/TTL/SigGen Extract")
        tpl_action.triggered.connect(self._add_template_mi_extract)
        self._template_btn.setMenu(self._template_menu)
        header.addWidget(self._template_btn)
        header.addStretch()

        self._seed_from_siblings_cb = QCheckBox("Seed from siblings")
        self._seed_from_siblings_cb.setToolTip(
            "During batch fits, use results from other files with similar\n"
            "captures as an additional seed during retries."
        )
        self._seed_from_siblings_cb.setChecked(self._seed_from_siblings)
        self._seed_from_siblings_cb.toggled.connect(self._on_seed_from_siblings_toggled)
        header.addWidget(self._seed_from_siblings_cb)
        layout.addLayout(header)

        # Step list scroll area.
        self._step_scroll = QScrollArea()
        self._step_scroll.setWidgetResizable(True)
        self._step_scroll.setMinimumHeight(80)
        self._step_scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #e3e8ef; border-radius: 6px; background: #f8fafc; }"
        )
        self._step_container = QWidget()
        self._step_container.setStyleSheet("background: transparent;")
        self._step_layout = QVBoxLayout(self._step_container)
        self._step_layout.setContentsMargins(4, 4, 4, 4)
        self._step_layout.setSpacing(4)
        self._step_layout.addStretch()
        self._step_scroll.setWidget(self._step_container)
        layout.addWidget(self._step_scroll, 1)

        # Status label.
        self._status_label = self._make_label("", object_name="statusLabel")
        layout.addWidget(self._status_label)

        # Step results table.
        self._results_table = StepResultsTable()
        layout.addWidget(self._results_table)

    # ── Widget helpers ────────────────────────────────────────────

    @staticmethod
    def _make_label(
        text="",
        *,
        object_name=None,
        tooltip=None,
        width=None,
        alignment=None,
        style_sheet=None,
    ):
        label = QLabel(str(text))
        if object_name:
            label.setObjectName(str(object_name))
        if tooltip:
            label.setToolTip(str(tooltip))
        if alignment is not None:
            label.setAlignment(alignment)
        if width is not None:
            label.setFixedWidth(int(width))
        if style_sheet is not None:
            label.setStyleSheet(str(style_sheet))
        return label

    @staticmethod
    def _make_button(
        text,
        *,
        handler=None,
        tooltip=None,
        enabled=None,
        fixed_width=None,
        primary=False,
        style_sheet=None,
    ):
        button = QPushButton(str(text))
        if tooltip:
            button.setToolTip(str(tooltip))
        if enabled is not None:
            button.setEnabled(bool(enabled))
        if fixed_width is not None:
            button.setFixedWidth(int(fixed_width))
        if primary:
            button.setProperty("primary", True)
        if style_sheet:
            button.setStyleSheet(str(style_sheet))
        if callable(handler):
            button.clicked.connect(handler)
        return button

    # ── Step cards ────────────────────────────────────────────────

    _CARD_STYLE = """
        QGroupBox#procStepCard {
            background: #ffffff;
            border: 1px solid #d3dae3;
            border-radius: 8px;
            padding: 6px 8px 5px 8px;
            margin: 0px;
        }
        QGroupBox#procStepCard QCheckBox { spacing: 3px; }
        QGroupBox#procStepCard QCheckBox::indicator {
            width: 13px; height: 13px;
            border: 2px solid #9ca3af; border-radius: 3px;
            background: #ffffff;
        }
        QGroupBox#procStepCard QCheckBox::indicator:checked {
            background: #2563eb; border-color: #2563eb;
        }
        QGroupBox#procStepCard QCheckBox::indicator:hover { border-color: #6b7280; }
    """

    def _rebuild_step_cards(self):
        layout = self._step_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for idx, step in enumerate(self._procedure_steps):
            card = self._make_step_card(idx, step)
            layout.addWidget(card)
        layout.addStretch()

    def _make_step_card(self, idx: int, step: ProcedureStepBase) -> QWidget:
        """Dispatch to the appropriate card builder based on step type."""
        if isinstance(step, FitStep):
            return self._build_fit_step_card(idx, step)
        if isinstance(step, SetParameterStep):
            return self._build_set_param_card(idx, step)
        if isinstance(step, SetBoundariesStep):
            return self._build_set_boundaries_card(idx, step)
        if isinstance(step, RandomizeSeedsStep):
            return self._build_randomize_card(idx, step)
        # Fallback generic card.
        return self._build_generic_card(idx, step)

    def _card_shell(self, idx: int, type_label: str) -> Tuple[QGroupBox, QVBoxLayout]:
        """Create the common card frame with title row and move/remove buttons."""
        card = QGroupBox()
        card.setObjectName("procStepCard")
        card.setStyleSheet(self._CARD_STYLE)
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(6, 4, 6, 4)
        vlayout.setSpacing(4)

        # Title row.
        title_row = QHBoxLayout()
        title_row.setSpacing(4)
        number_label = QLabel(
            f"<b style='color:#2563eb;'>Step {idx + 1}</b>"
            f" <i style='color:#6b7280;'>[{type_label}]</i>"
        )
        number_label.setTextFormat(Qt.TextFormat.RichText)
        title_row.addWidget(number_label)
        title_row.addStretch()

        for icon_text, delta in [("▲", -1), ("▼", 1)]:
            btn = self._make_button(
                icon_text,
                handler=lambda _checked, i=idx, d=delta: self._move_step(i, d),
                fixed_width=24,
                tooltip="Move step",
            )
            title_row.addWidget(btn)
        remove_btn = self._make_button(
            "✕",
            handler=lambda _checked, i=idx: self._remove_step(i),
            fixed_width=24,
            tooltip="Remove step",
            style_sheet="QPushButton { color: #dc2626; } QPushButton:hover { background: #fee2e2; }",
        )
        title_row.addWidget(remove_btn)
        vlayout.addLayout(title_row)

        # Separator.
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #e5e7eb;")
        vlayout.addWidget(sep)

        card.setLayout(vlayout)
        return card, vlayout

    # ── Parameter equation grouping ───────────────────────────────

    def _get_param_equation_groups(
        self, all_params: list, selected_channels: Optional[set[str]] = None
    ) -> List[Tuple[Optional[str], List[str]]]:
        """Group parameters by equation/channel.

        Returns a list of ``(label, [param_keys])`` tuples.  *label* is the
        channel target name for multi-channel models, or ``None`` when there
        is only a single equation. Parameters are ordered to match
        *all_params* and each displayed parameter appears exactly once.
        """
        multi = self.host.proc_get_multi_channel_model()
        if multi is not None and multi.is_multi_channel:
            groups: List[Tuple[Optional[str], List[str]]] = []
            seen: set = set()
            all_channel_models = list(multi.channel_models)
            if selected_channels:
                channel_models = [
                    ch_model
                    for ch_model in all_channel_models
                    if str(ch_model.target_col) in selected_channels
                ]
                if not channel_models:
                    channel_models = all_channel_models
            else:
                channel_models = all_channel_models
            available_params = {str(param) for param in all_params}
            for ch_model in channel_models:
                ch_keys: List[str] = []
                for seg_names in ch_model.segment_param_names:
                    for name in seg_names:
                        key = str(name)
                        if key not in seen and key in available_params:
                            ch_keys.append(key)
                            seen.add(key)
                if ch_keys:
                    groups.append((str(ch_model.target_col), ch_keys))
            # Catch any remaining params not in any channel only when the step
            # is not channel-filtered. When specific channels are selected,
            # showing trailing params can expose controls from other channels.
            if not selected_channels:
                trailing = [
                    str(param) for param in all_params if str(param) not in seen
                ]
                if trailing:
                    groups.append((None, trailing))
            return groups

        # Single equation (single channel or piecewise) — one group.
        return [(None, list(all_params))]

    # ── Fit step card ─────────────────────────────────────────────

    def _build_fit_step_card(self, idx: int, step: FitStep) -> QWidget:
        card, vlayout = self._card_shell(idx, "Fit")
        all_params = self.host.proc_available_params()
        all_channels = self.host.proc_available_channels()
        capture_keys = self.host.proc_available_capture_keys()
        bound_map = dict(step.bound_params)

        # Channels (multi-channel only).
        if len(all_channels) > 1:
            ch_row = QHBoxLayout()
            ch_row.setSpacing(4)
            ch_row.addWidget(self._make_label("Channels:", object_name="paramHeader"))
            for ch in all_channels:
                cb = QCheckBox(self.host.proc_channel_display_name(ch))
                cb.setChecked(step.channels is None or ch in step.channels)
                cb.setProperty("_step_idx", idx)
                cb.setProperty("_channel", ch)
                cb.stateChanged.connect(self._on_fit_channel_toggled)
                ch_row.addWidget(cb)
            ch_row.addStretch()
            vlayout.addLayout(ch_row)

        # Boundary groups to hold fixed during this fit step.
        boundary_groups = self.host.proc_available_boundary_groups() or []
        # Resolve the effective set of channels for visibility filtering.
        # None (all channels) and () (boundary-only / no channels) both
        # show boundaries from every channel.
        step_channels: set[str]
        if step.channels is None or len(step.channels) == 0:
            step_channels = {str(ch) for ch in all_channels}
        else:
            step_channels = {str(ch) for ch in step.channels}
        visible_boundary_groups = []
        for name, members in boundary_groups:
            if not step_channels:
                visible_boundary_groups.append((str(name), tuple(members or ())))
                continue
            include = False
            for member in tuple(members or ()):
                if (
                    isinstance(member, (tuple, list))
                    and len(member) == 2
                    and str(member[0]) in step_channels
                ):
                    include = True
                    break
            if include:
                visible_boundary_groups.append((str(name), tuple(members or ())))

        locked_boundary_names = set(step.locked_boundary_names)
        boundary_row = QHBoxLayout()
        boundary_row.setSpacing(4)
        boundary_row.addWidget(
            self._make_label("Fit boundaries:", object_name="paramHeader")
        )
        if not visible_boundary_groups:
            boundary_row.addWidget(
                self._make_label(
                    "<i>None available for selected channels.</i>",
                    style_sheet="color:#64748b;",
                )
            )
        else:
            for name, members in visible_boundary_groups:
                member_parts: List[str] = []
                for member in members:
                    if not isinstance(member, (tuple, list)) or len(member) != 2:
                        continue
                    try:
                        member_parts.append(
                            f"{self.host.proc_channel_display_name(str(member[0]))}[{int(member[1]) + 1}]"
                        )
                    except Exception:
                        continue
                member_label = ", ".join(member_parts)
                cb = QCheckBox(str(name))
                cb.setChecked(str(name) not in locked_boundary_names)
                cb.setToolTip(
                    "Fit this boundary group in this step."
                    "\nUnchecked groups are held fixed."
                    + (f"\n{member_label}" if member_label else "")
                )
                cb.setProperty("_step_idx", idx)
                cb.setProperty("_boundary_name", str(name))
                cb.stateChanged.connect(self._on_fit_boundary_name_toggled)
                boundary_row.addWidget(cb)
        boundary_row.addStretch()
        vlayout.addLayout(boundary_row)

        # Boundary-only mode: when channels is explicitly empty, all params
        # are forced fixed at execution time and only boundaries are fit.
        boundary_only = step.channels is not None and len(step.channels) == 0

        # Parameter grid: Free | Symbol | Seed from field.
        # Group parameters by equation with dividers between different equations.
        if not boundary_only:
            eq_groups = self._get_param_equation_groups(
                all_params,
                selected_channels=step_channels if len(all_channels) > 1 else None,
            )

            has_multiple_equations = len(eq_groups) > 1

            param_grid = QGridLayout()
            param_grid.setSpacing(2)
            param_grid.setContentsMargins(0, 1, 0, 1)
            param_grid.setColumnStretch(0, 0)
            param_grid.setColumnStretch(1, 0)
            param_grid.setColumnStretch(2, 0)
            param_grid.setColumnStretch(3, 1)

            for col, (text, width) in enumerate(
                [
                    ("Fit", 36),
                    ("Parameter", 120),
                    ("Seed from field", 140),
                ]
            ):
                hdr = self._make_label(text, object_name="paramHeader")
                if width:
                    hdr.setMinimumWidth(width)
                param_grid.addWidget(hdr, 0, col)

            free_set = set(step.free_params)
            row = 1
            for grp_idx, (eq_label, eq_params) in enumerate(eq_groups):
                # Equation divider (between equations, not before the first).
                if has_multiple_equations and grp_idx > 0:
                    sep = QWidget()
                    sep.setFixedHeight(2)
                    sep.setStyleSheet(
                        "background: #94a3b8; margin-top: 3px; margin-bottom: 1px;"
                    )
                    param_grid.addWidget(sep, row, 0, 1, 4)
                    row += 1
                # Equation header label.
                if has_multiple_equations and eq_label:
                    eq_hdr = self._make_label(
                        self.host.proc_channel_display_name(eq_label),
                        style_sheet="font-weight: 600; color: #475569; font-size: 11px; padding: 1px 0;",
                    )
                    param_grid.addWidget(eq_hdr, row, 1, 1, 3)
                    row += 1

                for p in eq_params:
                    free_cb = QCheckBox()
                    free_cb.setChecked(p in free_set or len(free_set) == 0)
                    free_cb.setToolTip(f"Fit {p} in this step")
                    free_cb.setProperty("_step_idx", idx)
                    free_cb.setProperty("_param", p)
                    free_cb.stateChanged.connect(self._on_fit_param_toggled)
                    param_grid.addWidget(free_cb, row, 0, Qt.AlignmentFlag.AlignCenter)

                    symbol_html = self.host.proc_display_symbol_html(p)
                    plabel = QLabel(symbol_html)
                    plabel.setTextFormat(Qt.TextFormat.RichText)
                    plabel.setObjectName("paramInline")
                    plabel.setToolTip(p)
                    param_grid.addWidget(plabel, row, 1)

                    combo = RichTextComboBox()
                    combo.add_rich_item("(none)", "", "(none)")
                    for ck in capture_keys:
                        combo.add_rich_item(ck, ck, ck)
                    current_field = bound_map.get(p, "")
                    target_idx = combo.findData(current_field)
                    if target_idx < 0:
                        target_idx = 0
                    combo.setCurrentIndex(target_idx)
                    combo.setProperty("_step_idx", idx)
                    combo.setProperty("_param", p)
                    combo.setToolTip(
                        "Optional capture field used to seed this parameter before fitting."
                    )
                    combo.currentIndexChanged.connect(self._on_fit_bound_changed)
                    param_grid.addWidget(combo, row, 2)
                    row += 1

            param_wrapper = QHBoxLayout()
            param_wrapper.setContentsMargins(0, 0, 0, 0)
            param_wrapper.addLayout(param_grid)
            param_wrapper.addStretch()
            vlayout.addLayout(param_wrapper)

        # R² threshold row.
        r2_row = QHBoxLayout()
        r2_row.setSpacing(4)
        r2_row.addWidget(self._make_label("Min R²:", object_name="paramHeader"))
        r2_spin = QDoubleSpinBox()
        r2_spin.setRange(-1.0, 1.0)
        r2_spin.setDecimals(4)
        r2_spin.setSingleStep(0.01)
        r2_spin.setSpecialValueText("none")
        r2_spin.setValue(step.min_r2 if step.min_r2 is not None else -1.0)
        r2_spin.setMaximumWidth(130)
        r2_spin.setProperty("_step_idx", idx)
        r2_spin.valueChanged.connect(self._on_fit_r2_changed)
        r2_row.addWidget(r2_spin)

        # Retry controls.
        r2_row.addWidget(self._make_label("Retries:", object_name="paramHeader"))
        retry_spin = QSpinBox()
        retry_spin.setRange(0, 20)
        retry_spin.setValue(step.max_retries)
        retry_spin.setMaximumWidth(60)
        retry_spin.setToolTip("Number of retry attempts if R² threshold not met")
        retry_spin.setProperty("_step_idx", idx)
        retry_spin.valueChanged.connect(self._on_fit_retry_changed)
        r2_row.addWidget(retry_spin)

        r2_row.addWidget(self._make_label("Scale:", object_name="paramHeader"))
        scale_spin = QDoubleSpinBox()
        scale_spin.setRange(0.01, 1.0)
        scale_spin.setDecimals(2)
        scale_spin.setSingleStep(0.05)
        scale_spin.setValue(step.retry_scale)
        scale_spin.setMaximumWidth(70)
        scale_spin.setToolTip("Randomisation scale for retry seed perturbation")
        scale_spin.setProperty("_step_idx", idx)
        scale_spin.valueChanged.connect(self._on_fit_retry_scale_changed)
        r2_row.addWidget(scale_spin)

        r2_row.addStretch()
        vlayout.addLayout(r2_row)

        flow_row = QHBoxLayout()
        flow_row.setSpacing(4)
        flow_row.addWidget(self._make_label("Retry Mode:", object_name="paramHeader"))
        retry_mode_combo = QComboBox()
        retry_mode_combo.addItem("Jitter then random", "jitter_then_random")
        retry_mode_combo.addItem("Jitter only", "jitter")
        retry_mode_combo.addItem("Random only", "random")
        mode_idx = retry_mode_combo.findData(step.retry_mode)
        retry_mode_combo.setCurrentIndex(0 if mode_idx < 0 else mode_idx)
        retry_mode_combo.setProperty("_step_idx", idx)
        retry_mode_combo.currentIndexChanged.connect(self._on_fit_retry_mode_changed)
        flow_row.addWidget(retry_mode_combo)

        flow_row.addWidget(self._make_label("On Fail:", object_name="paramHeader"))
        on_fail_combo = QComboBox()
        on_fail_combo.addItem("Stop procedure", "stop")
        on_fail_combo.addItem("Continue", "continue")
        fail_idx = on_fail_combo.findData(step.on_fail)
        on_fail_combo.setCurrentIndex(0 if fail_idx < 0 else fail_idx)
        on_fail_combo.setProperty("_step_idx", idx)
        on_fail_combo.currentIndexChanged.connect(self._on_fit_on_fail_changed)
        flow_row.addWidget(on_fail_combo)
        flow_row.addStretch()
        vlayout.addLayout(flow_row)

        return card

    # ── Set Parameter step card ───────────────────────────────────

    def _build_set_param_card(self, idx: int, step: SetParameterStep) -> QWidget:
        card, vlayout = self._card_shell(idx, "Set Parameter")
        all_params = self.host.proc_available_params()
        selected_keys = [
            str(a.target_key) for a in step.assignments if str(a.target_key)
        ]

        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        add_row.addWidget(self._make_label("Add parameter:", object_name="paramHeader"))
        add_combo = QComboBox()
        add_combo.addItem("(select)", "")
        for p in all_params:
            if str(p) not in set(selected_keys):
                add_combo.addItem(str(p), str(p))
        add_combo.setMaximumWidth(180)
        add_combo.setProperty("_step_idx", idx)
        add_combo.currentIndexChanged.connect(self._on_set_param_add_selected)
        add_row.addWidget(add_combo)
        add_row.addStretch()
        vlayout.addLayout(add_row)

        if not step.assignments:
            vlayout.addWidget(
                self._make_label(
                    "<i>No parameters selected.</i>",
                    style_sheet="color:#64748b;",
                )
            )
            return card

        grid = QGridLayout()
        grid.setSpacing(3)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.addWidget(self._make_label("Parameter", object_name="paramHeader"), 0, 0)
        grid.addWidget(self._make_label("Source", object_name="paramHeader"), 0, 1)
        grid.addWidget(self._make_label("Key/Value", object_name="paramHeader"), 0, 2)
        grid.addWidget(self._make_label("Scale", object_name="paramHeader"), 0, 3)
        grid.addWidget(self._make_label("Offset", object_name="paramHeader"), 0, 4)
        grid.addWidget(self._make_label("Clamp", object_name="paramHeader"), 0, 5)
        grid.addWidget(self._make_label("Missing", object_name="paramHeader"), 0, 6)
        grid.addWidget(self._make_label("", object_name="paramHeader"), 0, 7)
        grid.setColumnStretch(8, 1)

        row = 1
        for assignment in step.assignments:
            p = str(assignment.target_key)
            plabel = QLabel(self.host.proc_display_symbol_html(p))
            plabel.setTextFormat(Qt.TextFormat.RichText)
            plabel.setObjectName("paramInline")
            plabel.setToolTip(p)
            grid.addWidget(plabel, row, 0)

            source_combo = QComboBox()
            source_combo.addItem("Literal", "literal")
            source_combo.addItem("From param", "param")
            source_combo.addItem("From capture", "capture")
            source_idx = source_combo.findData(assignment.source_kind)
            source_combo.setCurrentIndex(0 if source_idx < 0 else source_idx)
            source_combo.setMaximumWidth(110)
            source_combo.setProperty("_step_idx", idx)
            source_combo.setProperty("_param", p)
            source_combo.currentIndexChanged.connect(self._on_set_param_source_changed)
            grid.addWidget(source_combo, row, 1)

            key_or_val = ""
            if assignment.source_kind == "literal":
                key_or_val = (
                    ""
                    if assignment.literal_value is None
                    else f"{float(assignment.literal_value):.9g}"
                )
            else:
                key_or_val = str(assignment.source_key or "")
            ref_edit = QLineEdit(key_or_val)
            ref_edit.setPlaceholderText("value or key")
            ref_edit.setMaximumWidth(130)
            ref_edit.setProperty("_step_idx", idx)
            ref_edit.setProperty("_param", p)
            ref_edit.textChanged.connect(self._on_set_param_ref_changed)
            grid.addWidget(ref_edit, row, 2)

            scale_spin = QDoubleSpinBox()
            scale_spin.setRange(-1e9, 1e9)
            scale_spin.setDecimals(6)
            scale_spin.setSingleStep(0.1)
            scale_spin.setValue(float(assignment.scale))
            scale_spin.setMaximumWidth(88)
            scale_spin.setProperty("_step_idx", idx)
            scale_spin.setProperty("_param", p)
            scale_spin.valueChanged.connect(self._on_set_param_scale_changed)
            grid.addWidget(scale_spin, row, 3)

            offset_spin = QDoubleSpinBox()
            offset_spin.setRange(-1e9, 1e9)
            offset_spin.setDecimals(6)
            offset_spin.setSingleStep(0.1)
            offset_spin.setValue(float(assignment.offset))
            offset_spin.setMaximumWidth(88)
            offset_spin.setProperty("_step_idx", idx)
            offset_spin.setProperty("_param", p)
            offset_spin.valueChanged.connect(self._on_set_param_offset_changed)
            grid.addWidget(offset_spin, row, 4)

            clamp_cb = QCheckBox()
            clamp_cb.setChecked(bool(assignment.clamp_to_bounds))
            clamp_cb.setProperty("_step_idx", idx)
            clamp_cb.setProperty("_param", p)
            clamp_cb.stateChanged.connect(self._on_set_param_clamp_toggled)
            grid.addWidget(clamp_cb, row, 5, Qt.AlignmentFlag.AlignCenter)

            missing_combo = QComboBox()
            missing_combo.addItem("Skip", "skip")
            missing_combo.addItem("Fail", "fail")
            miss_idx = missing_combo.findData(str(assignment.on_missing))
            missing_combo.setCurrentIndex(0 if miss_idx < 0 else miss_idx)
            missing_combo.setMaximumWidth(78)
            missing_combo.setProperty("_step_idx", idx)
            missing_combo.setProperty("_param", p)
            missing_combo.currentIndexChanged.connect(
                self._on_set_param_missing_changed
            )
            grid.addWidget(missing_combo, row, 6)

            remove_btn = self._make_button(
                "✕",
                handler=lambda _checked=False,
                i=idx,
                key=p: self._on_set_param_remove_clicked(i, key),
                fixed_width=24,
                tooltip="Remove parameter from this step",
                style_sheet="QPushButton { color: #dc2626; } QPushButton:hover { background: #fee2e2; }",
            )
            grid.addWidget(remove_btn, row, 7)
            row += 1

        wrapper = QHBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addLayout(grid)
        wrapper.addStretch()
        vlayout.addLayout(wrapper)
        return card

    # ── Set Boundaries step card ──────────────────────────────────

    def _build_set_boundaries_card(self, idx: int, step: SetBoundariesStep) -> QWidget:
        card, vlayout = self._card_shell(idx, "Set Boundaries")
        groups = self.host.proc_available_boundary_groups() or []
        available_names = [str(name) for name, _members in groups if str(name).strip()]
        selected_names = [
            str(a.target_name) for a in step.assignments if str(a.target_name).strip()
        ]

        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        add_row.addWidget(self._make_label("Add boundary:", object_name="paramHeader"))
        add_combo = QComboBox()
        add_combo.addItem("(select)", "")
        for name in available_names:
            if name not in set(selected_names):
                add_combo.addItem(name, name)
        add_combo.setMaximumWidth(180)
        add_combo.setProperty("_step_idx", idx)
        add_combo.currentIndexChanged.connect(self._on_set_boundary_add_selected)
        add_row.addWidget(add_combo)
        add_row.addStretch()
        vlayout.addLayout(add_row)

        if not step.assignments:
            vlayout.addWidget(
                self._make_label(
                    "<i>No boundaries selected.</i>",
                    style_sheet="color:#64748b;",
                )
            )
            return card

        members_by_name = {
            str(name): tuple(members or ()) for name, members in tuple(groups or ())
        }

        grid = QGridLayout()
        grid.setSpacing(3)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.addWidget(self._make_label("Boundary", object_name="paramHeader"), 0, 0)
        grid.addWidget(self._make_label("Source", object_name="paramHeader"), 0, 1)
        grid.addWidget(self._make_label("Value / Ref", object_name="paramHeader"), 0, 2)
        grid.addWidget(self._make_label("Missing", object_name="paramHeader"), 0, 3)
        grid.addWidget(self._make_label("", object_name="paramHeader"), 0, 4)
        grid.setColumnStretch(5, 1)

        row = 1
        for assignment in step.assignments:
            target_name = str(assignment.target_name)
            member_label = ", ".join(
                f"{str(target)}[{int(i) + 1}]"
                for target, i in tuple(members_by_name.get(target_name, ()))
            )
            target_label = self._make_label(target_name or "(unknown)")
            if member_label:
                target_label.setToolTip(member_label)
            grid.addWidget(target_label, row, 0)

            source_combo = QComboBox()
            source_combo.addItem("Literal", "literal")
            source_combo.addItem("From boundary", "boundary")
            source_combo.addItem("Expression", "expression")
            source_idx = source_combo.findData(str(assignment.source_kind))
            source_combo.setCurrentIndex(0 if source_idx < 0 else source_idx)
            source_combo.setMaximumWidth(130)
            source_combo.setProperty("_step_idx", idx)
            source_combo.setProperty("_boundary", target_name)
            source_combo.currentIndexChanged.connect(
                self._on_set_boundary_source_changed
            )
            grid.addWidget(source_combo, row, 1)

            ref_text = ""
            source_kind = str(assignment.source_kind)
            if source_kind == "literal":
                ref_text = (
                    ""
                    if assignment.literal_value is None
                    else f"{float(assignment.literal_value):.9g}"
                )
            elif source_kind == "boundary":
                ref_text = _normalise_boundary_reference_text(
                    str(assignment.source_name or "")
                )
            else:
                ref_text = _normalise_boundary_expression_text(
                    str(assignment.expression or "")
                )
            ref_edit = QLineEdit(ref_text)
            if source_kind == "literal":
                ref_edit.setPlaceholderText("ratio (0-1)")
            elif source_kind == "boundary":
                ref_edit.setPlaceholderText("boundary name")
            else:
                ref_edit.setPlaceholderText("e.g. X0 + 0.05")
            ref_edit.setMaximumWidth(180)
            if source_kind == "boundary" and available_names:
                ref_edit.setToolTip("Available: " + ", ".join(available_names))
            ref_edit.setProperty("_step_idx", idx)
            ref_edit.setProperty("_boundary", target_name)
            ref_edit.setProperty("_boundary_source_kind", source_kind)
            ref_edit.textEdited.connect(self._on_set_boundary_ref_edited)
            ref_edit.textChanged.connect(self._on_set_boundary_ref_changed)
            grid.addWidget(ref_edit, row, 2)

            missing_combo = QComboBox()
            missing_combo.addItem("Skip", "skip")
            missing_combo.addItem("Fail", "fail")
            miss_idx = missing_combo.findData(str(assignment.on_missing))
            missing_combo.setCurrentIndex(0 if miss_idx < 0 else miss_idx)
            missing_combo.setMaximumWidth(78)
            missing_combo.setProperty("_step_idx", idx)
            missing_combo.setProperty("_boundary", target_name)
            missing_combo.currentIndexChanged.connect(
                self._on_set_boundary_missing_changed
            )
            grid.addWidget(missing_combo, row, 3)

            remove_btn = self._make_button(
                "✕",
                handler=lambda _checked=False,
                i=idx,
                key=target_name: self._on_set_boundary_remove_clicked(i, key),
                fixed_width=24,
                tooltip="Remove boundary from this step",
                style_sheet="QPushButton { color: #dc2626; } QPushButton:hover { background: #fee2e2; }",
            )
            grid.addWidget(remove_btn, row, 4)
            row += 1

        wrapper = QHBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addLayout(grid)
        wrapper.addStretch()
        vlayout.addLayout(wrapper)
        return card

    # ── Randomize Seeds step card ─────────────────────────────────

    def _build_randomize_card(self, idx: int, step: RandomizeSeedsStep) -> QWidget:
        card, vlayout = self._card_shell(idx, "Randomize Seeds")
        all_params = self.host.proc_available_params()

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self._make_label("Scale:", object_name="paramHeader"))
        scale_spin = QDoubleSpinBox()
        scale_spin.setRange(0.01, 1.0)
        scale_spin.setDecimals(2)
        scale_spin.setSingleStep(0.05)
        scale_spin.setValue(step.scale)
        scale_spin.setMaximumWidth(80)
        scale_spin.setToolTip("Perturbation as fraction of (high - low)")
        scale_spin.setProperty("_step_idx", idx)
        scale_spin.valueChanged.connect(self._on_randomize_scale_changed)
        row.addWidget(scale_spin)
        row.addStretch()
        vlayout.addLayout(row)

        # Parameter checkboxes (unchecked = all).
        params_row = QHBoxLayout()
        params_row.setSpacing(4)
        params_row.addWidget(self._make_label("Params:", object_name="paramHeader"))
        target_set = set(step.params)
        for p in all_params:
            cb = QCheckBox(p)
            cb.setChecked(p in target_set or len(target_set) == 0)
            cb.setProperty("_step_idx", idx)
            cb.setProperty("_param", p)
            cb.stateChanged.connect(self._on_randomize_param_toggled)
            params_row.addWidget(cb)
        params_row.addStretch()
        vlayout.addLayout(params_row)
        return card

    # ── Generic fallback card ─────────────────────────────────────

    def _build_generic_card(self, idx: int, step: ProcedureStepBase) -> QWidget:
        card, vlayout = self._card_shell(idx, step.step_type)
        vlayout.addWidget(
            self._make_label(
                f"<i>Step type '{step.step_type}' has no custom editor.</i>",
                style_sheet="color: #9ca3af;",
            )
        )
        return card

    # ── Step editing slots ────────────────────────────────────────

    def _notify_change(self):
        self.procedure_changed.emit()
        self.host.proc_autosave()

    # Fit step slots
    def _on_fit_channel_toggled(self, _state):
        sender = self.sender()
        idx = sender.property("_step_idx")
        ch = sender.property("_channel")
        if idx is None or ch is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        if step.channels is None:
            current = set(self.host.proc_available_channels())
        elif step.channels:
            current = set(step.channels)
        else:
            current = set()  # boundary-only: start with none checked
        if sender.isChecked():
            current.add(ch)
        else:
            current.discard(ch)
        all_ch = set(self.host.proc_available_channels())
        if current == all_ch:
            channels = None  # all channels selected → None
        else:
            channels = tuple(sorted(current))  # may be () → boundary-only
        # Determine effective channels for boundary-group filtering.
        selected_channels = set(channels) if channels else set(all_ch)
        valid_lock_names: set = set()
        for name, members in self.host.proc_available_boundary_groups() or []:
            if not selected_channels:
                valid_lock_names.add(str(name))
                continue
            for member in tuple(members or ()):
                if (
                    isinstance(member, (tuple, list))
                    and len(member) == 2
                    and str(member[0]) in selected_channels
                ):
                    valid_lock_names.add(str(name))
                    break
        new_locked = tuple(
            n for n in step.locked_boundary_names if str(n) in valid_lock_names
        )
        self._procedure_steps[idx] = FitStep(
            channels=channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=step.bound_params,
            min_r2=step.min_r2,
            max_retries=step.max_retries,
            retry_scale=step.retry_scale,
            retry_mode=step.retry_mode,
            locked_boundary_names=new_locked,
            on_fail=step.on_fail,
            label=step.label,
        )
        self._rebuild_step_cards()
        self._notify_change()

    def _on_fit_param_toggled(self, _state):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        all_params = set(self.host.proc_available_params())
        current_free = set(step.free_params) if step.free_params else set(all_params)
        if sender.isChecked():
            current_free.add(param)
        else:
            current_free.discard(param)
        free = tuple(sorted(current_free)) if current_free != all_params else ()
        fixed = tuple(sorted(all_params - current_free))
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=free,
            fixed_params=fixed,
            bound_params=step.bound_params,
            min_r2=step.min_r2,
            max_retries=step.max_retries,
            retry_scale=step.retry_scale,
            retry_mode=step.retry_mode,
            locked_boundary_names=step.locked_boundary_names,
            on_fail=step.on_fail,
            label=step.label,
        )
        self._notify_change()

    def _on_fit_bound_changed(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        existing = dict(step.bound_params)
        field = sender.currentData()
        if field not in (None, ""):
            existing[str(param)] = str(field)
        else:
            existing.pop(str(param), None)
        new_bound = tuple((k, v) for k, v in existing.items() if v)
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=new_bound,
            min_r2=step.min_r2,
            max_retries=step.max_retries,
            retry_scale=step.retry_scale,
            retry_mode=step.retry_mode,
            locked_boundary_names=step.locked_boundary_names,
            on_fail=step.on_fail,
            label=step.label,
        )
        self._notify_change()

    def _on_seed_from_siblings_toggled(self, checked: bool):
        self._seed_from_siblings = bool(checked)
        self._notify_change()

    def _on_fit_r2_changed(self, value):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        r2 = float(value) if value > -1.0 else None
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=step.bound_params,
            min_r2=r2,
            max_retries=step.max_retries,
            retry_scale=step.retry_scale,
            retry_mode=step.retry_mode,
            locked_boundary_names=step.locked_boundary_names,
            on_fail=step.on_fail,
            label=step.label,
        )
        self._notify_change()

    def _on_fit_retry_changed(self, value):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=step.bound_params,
            min_r2=step.min_r2,
            max_retries=max(0, int(value)),
            retry_scale=step.retry_scale,
            retry_mode=step.retry_mode,
            locked_boundary_names=step.locked_boundary_names,
            on_fail=step.on_fail,
            label=step.label,
        )
        self._notify_change()

    def _on_fit_retry_scale_changed(self, value):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=step.bound_params,
            min_r2=step.min_r2,
            max_retries=step.max_retries,
            retry_scale=float(value),
            retry_mode=step.retry_mode,
            locked_boundary_names=step.locked_boundary_names,
            on_fail=step.on_fail,
            label=step.label,
        )
        self._notify_change()

    def _on_fit_retry_mode_changed(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        retry_mode = str(sender.currentData() or "jitter_then_random")
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=step.bound_params,
            min_r2=step.min_r2,
            max_retries=step.max_retries,
            retry_scale=step.retry_scale,
            retry_mode=retry_mode,
            locked_boundary_names=step.locked_boundary_names,
            on_fail=step.on_fail,
            label=step.label,
        )
        self._notify_change()

    def _on_fit_on_fail_changed(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        on_fail = str(sender.currentData() or "stop")
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=step.bound_params,
            min_r2=step.min_r2,
            max_retries=step.max_retries,
            retry_scale=step.retry_scale,
            retry_mode=step.retry_mode,
            locked_boundary_names=step.locked_boundary_names,
            on_fail=on_fail,
            label=step.label,
        )
        self._notify_change()

    def _on_fit_boundary_name_toggled(self, _state):
        sender = self.sender()
        idx = sender.property("_step_idx")
        name = sender.property("_boundary_name")
        if idx is None or name is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, FitStep):
            return
        names = set(step.locked_boundary_names)
        if sender.isChecked():
            names.discard(str(name))
        else:
            names.add(str(name))
        self._procedure_steps[idx] = FitStep(
            channels=step.channels,
            free_params=step.free_params,
            fixed_params=step.fixed_params,
            bound_params=step.bound_params,
            min_r2=step.min_r2,
            max_retries=step.max_retries,
            retry_scale=step.retry_scale,
            retry_mode=step.retry_mode,
            locked_boundary_names=tuple(sorted(names)),
            on_fail=step.on_fail,
            label=step.label,
        )
        self._notify_change()

    # Set Parameter slots
    def _set_param_assignment_map(
        self, step: SetParameterStep
    ) -> Dict[str, ParameterAssignment]:
        return {a.target_key: a for a in step.assignments}

    def _set_param_replace_assignment(
        self,
        step: SetParameterStep,
        assignment: Optional[ParameterAssignment],
        remove_key: Optional[str] = None,
    ) -> SetParameterStep:
        current = self._set_param_assignment_map(step)
        key = assignment.target_key if assignment is not None else str(remove_key or "")
        if assignment is None or not key:
            if key:
                current.pop(str(key), None)
        else:
            current[key] = assignment
        out = tuple(current[k] for k in sorted(current.keys()))
        return SetParameterStep(assignments=out, label=step.label)

    def _on_set_param_add_selected(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetParameterStep):
            return
        param = str(sender.currentData() or "").strip()
        if not param:
            return
        if param in self._set_param_assignment_map(step):
            return
        assignment = ParameterAssignment(
            target_key=param,
            source_kind="literal",
            literal_value=None,
            scale=1.0,
            offset=0.0,
            clamp_to_bounds=True,
            on_missing="skip",
        )
        self._procedure_steps[idx] = self._set_param_replace_assignment(
            step, assignment
        )
        self._rebuild_step_cards()
        self._notify_change()

    def _on_set_param_remove_clicked(self, idx: int, param: str):
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetParameterStep):
            return
        self._procedure_steps[idx] = self._set_param_replace_assignment(
            step, None, remove_key=str(param)
        )
        self._rebuild_step_cards()
        self._notify_change()

    def _on_set_param_source_changed(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetParameterStep):
            return
        source_kind = str(sender.currentData() or "literal")
        if source_kind not in {"literal", "param", "capture"}:
            source_kind = "literal"
        existing = self._set_param_assignment_map(step).get(
            str(param),
            ParameterAssignment(target_key=str(param)),
        )
        literal = existing.literal_value if source_kind == "literal" else None
        source_key = existing.source_key if source_kind in {"param", "capture"} else ""
        assignment = ParameterAssignment(
            target_key=str(param),
            source_kind=source_kind,
            source_key=source_key,
            literal_value=literal,
            scale=existing.scale,
            offset=existing.offset,
            clamp_to_bounds=existing.clamp_to_bounds,
            on_missing=existing.on_missing,
        )
        self._procedure_steps[idx] = self._set_param_replace_assignment(
            step, assignment
        )
        self._rebuild_step_cards()
        self._notify_change()

    def _on_set_param_ref_changed(self, text):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetParameterStep):
            return
        assignment = self._set_param_assignment_map(step).get(str(param))
        if assignment is None:
            return
        text = str(text).strip()
        if assignment.source_kind == "literal":
            literal = _finite_float_or_none(text)
            assignment = ParameterAssignment(
                target_key=assignment.target_key,
                source_kind=assignment.source_kind,
                source_key="",
                literal_value=literal,
                scale=assignment.scale,
                offset=assignment.offset,
                clamp_to_bounds=assignment.clamp_to_bounds,
                on_missing=assignment.on_missing,
            )
        else:
            assignment = ParameterAssignment(
                target_key=assignment.target_key,
                source_kind=assignment.source_kind,
                source_key=text,
                literal_value=None,
                scale=assignment.scale,
                offset=assignment.offset,
                clamp_to_bounds=assignment.clamp_to_bounds,
                on_missing=assignment.on_missing,
            )
        self._procedure_steps[idx] = self._set_param_replace_assignment(
            step, assignment
        )
        self._notify_change()

    def _on_set_param_scale_changed(self, value):
        self._on_set_param_numeric_field_changed("scale", value)

    def _on_set_param_offset_changed(self, value):
        self._on_set_param_numeric_field_changed("offset", value)

    def _on_set_param_numeric_field_changed(self, field: str, value):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetParameterStep):
            return
        assignment = self._set_param_assignment_map(step).get(str(param))
        if assignment is None:
            return
        scale = float(value) if field == "scale" else float(assignment.scale)
        offset = float(value) if field == "offset" else float(assignment.offset)
        assignment = ParameterAssignment(
            target_key=assignment.target_key,
            source_kind=assignment.source_kind,
            source_key=assignment.source_key,
            literal_value=assignment.literal_value,
            scale=scale,
            offset=offset,
            clamp_to_bounds=assignment.clamp_to_bounds,
            on_missing=assignment.on_missing,
        )
        self._procedure_steps[idx] = self._set_param_replace_assignment(
            step, assignment
        )
        self._notify_change()

    def _on_set_param_clamp_toggled(self, _state):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetParameterStep):
            return
        assignment = self._set_param_assignment_map(step).get(str(param))
        if assignment is None:
            return
        assignment = ParameterAssignment(
            target_key=assignment.target_key,
            source_kind=assignment.source_kind,
            source_key=assignment.source_key,
            literal_value=assignment.literal_value,
            scale=assignment.scale,
            offset=assignment.offset,
            clamp_to_bounds=sender.isChecked(),
            on_missing=assignment.on_missing,
        )
        self._procedure_steps[idx] = self._set_param_replace_assignment(
            step, assignment
        )
        self._notify_change()

    def _on_set_param_missing_changed(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetParameterStep):
            return
        assignment = self._set_param_assignment_map(step).get(str(param))
        if assignment is None:
            return
        assignment = ParameterAssignment(
            target_key=assignment.target_key,
            source_kind=assignment.source_kind,
            source_key=assignment.source_key,
            literal_value=assignment.literal_value,
            scale=assignment.scale,
            offset=assignment.offset,
            clamp_to_bounds=assignment.clamp_to_bounds,
            on_missing=str(sender.currentData() or "skip"),
        )
        self._procedure_steps[idx] = self._set_param_replace_assignment(
            step, assignment
        )
        self._notify_change()

    # Set Boundaries slots
    def _set_boundary_assignment_map(
        self, step: SetBoundariesStep
    ) -> Dict[str, BoundaryAssignment]:
        return {a.target_name: a for a in step.assignments}

    def _set_boundary_replace_assignment(
        self,
        step: SetBoundariesStep,
        assignment: Optional[BoundaryAssignment],
        remove_key: Optional[str] = None,
    ) -> SetBoundariesStep:
        current = self._set_boundary_assignment_map(step)
        key = (
            assignment.target_name if assignment is not None else str(remove_key or "")
        )
        if assignment is None or not key:
            if key:
                current.pop(str(key), None)
        else:
            current[key] = assignment
        out = tuple(current[k] for k in sorted(current.keys()))
        return SetBoundariesStep(assignments=out, label=step.label)

    def _on_set_boundary_add_selected(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetBoundariesStep):
            return
        name = str(sender.currentData() or "").strip()
        if not name:
            return
        if name in self._set_boundary_assignment_map(step):
            return
        assignment = BoundaryAssignment(
            target_name=name,
            source_kind="literal",
            literal_value=None,
            on_missing="skip",
        )
        self._procedure_steps[idx] = self._set_boundary_replace_assignment(
            step, assignment
        )
        self._rebuild_step_cards()
        self._notify_change()

    def _on_set_boundary_remove_clicked(self, idx: int, boundary_name: str):
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetBoundariesStep):
            return
        self._procedure_steps[idx] = self._set_boundary_replace_assignment(
            step, None, remove_key=str(boundary_name)
        )
        self._rebuild_step_cards()
        self._notify_change()

    def _on_set_boundary_source_changed(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        boundary_name = sender.property("_boundary")
        if idx is None or boundary_name is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetBoundariesStep):
            return
        source_kind = str(sender.currentData() or "literal")
        if source_kind not in {"literal", "boundary", "expression"}:
            source_kind = "literal"
        existing = self._set_boundary_assignment_map(step).get(
            str(boundary_name),
            BoundaryAssignment(target_name=str(boundary_name)),
        )
        literal_value = existing.literal_value if source_kind == "literal" else None
        source_name = existing.source_name if source_kind == "boundary" else ""
        expression = existing.expression if source_kind == "expression" else ""
        assignment = BoundaryAssignment(
            target_name=str(boundary_name),
            source_kind=source_kind,
            source_name=source_name,
            literal_value=literal_value,
            expression=expression,
            on_missing=existing.on_missing,
        )
        self._procedure_steps[idx] = self._set_boundary_replace_assignment(
            step, assignment
        )
        self._rebuild_step_cards()
        self._notify_change()

    def _on_set_boundary_ref_edited(self, _text):
        sender = self.sender()
        if sender is None:
            return
        source_kind = str(sender.property("_boundary_source_kind") or "").strip()
        current_text = str(sender.text() or "")
        converted = current_text
        if source_kind == "boundary":
            converted = _normalise_boundary_reference_text(current_text)
        elif source_kind == "expression":
            converted = _normalise_boundary_expression_text(current_text)
        if converted == current_text:
            return
        cursor_pos = sender.cursorPosition()
        sender.setText(converted)
        sender.setCursorPosition(min(max(0, int(cursor_pos)), len(converted)))

    def _on_set_boundary_ref_changed(self, text):
        sender = self.sender()
        idx = sender.property("_step_idx")
        boundary_name = sender.property("_boundary")
        if idx is None or boundary_name is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetBoundariesStep):
            return
        assignment = self._set_boundary_assignment_map(step).get(str(boundary_name))
        if assignment is None:
            return
        value_text = str(text).strip()
        if assignment.source_kind == "literal":
            assignment = BoundaryAssignment(
                target_name=assignment.target_name,
                source_kind=assignment.source_kind,
                source_name="",
                literal_value=_finite_float_or_none(value_text),
                expression="",
                on_missing=assignment.on_missing,
            )
        elif assignment.source_kind == "boundary":
            value_text = _normalise_boundary_reference_text(value_text)
            assignment = BoundaryAssignment(
                target_name=assignment.target_name,
                source_kind=assignment.source_kind,
                source_name=value_text,
                literal_value=None,
                expression="",
                on_missing=assignment.on_missing,
            )
        else:
            value_text = _normalise_boundary_expression_text(value_text)
            assignment = BoundaryAssignment(
                target_name=assignment.target_name,
                source_kind=assignment.source_kind,
                source_name="",
                literal_value=None,
                expression=value_text,
                on_missing=assignment.on_missing,
            )
        self._procedure_steps[idx] = self._set_boundary_replace_assignment(
            step, assignment
        )
        self._notify_change()

    def _on_set_boundary_missing_changed(self, _index):
        sender = self.sender()
        idx = sender.property("_step_idx")
        boundary_name = sender.property("_boundary")
        if idx is None or boundary_name is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, SetBoundariesStep):
            return
        assignment = self._set_boundary_assignment_map(step).get(str(boundary_name))
        if assignment is None:
            return
        assignment = BoundaryAssignment(
            target_name=assignment.target_name,
            source_kind=assignment.source_kind,
            source_name=assignment.source_name,
            literal_value=assignment.literal_value,
            expression=assignment.expression,
            on_missing=str(sender.currentData() or "skip"),
        )
        self._procedure_steps[idx] = self._set_boundary_replace_assignment(
            step, assignment
        )
        self._notify_change()

    # Randomize Seeds slots
    def _on_randomize_scale_changed(self, value):
        sender = self.sender()
        idx = sender.property("_step_idx")
        if idx is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, RandomizeSeedsStep):
            return
        self._procedure_steps[idx] = RandomizeSeedsStep(
            params=step.params,
            scale=float(value),
            label=step.label,
        )
        self._notify_change()

    def _on_randomize_param_toggled(self, _state):
        sender = self.sender()
        idx = sender.property("_step_idx")
        param = sender.property("_param")
        if idx is None or param is None or idx >= len(self._procedure_steps):
            return
        step = self._procedure_steps[idx]
        if not isinstance(step, RandomizeSeedsStep):
            return
        all_params = set(self.host.proc_available_params())
        current = set(step.params) if step.params else set(all_params)
        if sender.isChecked():
            current.add(param)
        else:
            current.discard(param)
        params = tuple(sorted(current)) if current != all_params else ()
        self._procedure_steps[idx] = RandomizeSeedsStep(
            params=params,
            scale=step.scale,
            label=step.label,
        )
        self._notify_change()

    def _channel_param_map(self) -> Dict[str, set]:
        out: Dict[str, set] = {}
        multi = self.host.proc_get_multi_channel_model()
        if multi is not None:
            for ch_model in multi.channel_models:
                names = out.setdefault(str(ch_model.target_col), set())
                for seg_names in ch_model.segment_param_names:
                    for key in seg_names:
                        names.add(str(key))
            return out
        piecewise = self.host.proc_get_piecewise_model()
        if piecewise is not None:
            names = out.setdefault(str(piecewise.target_col), set())
            for seg_names in piecewise.segment_param_names:
                for key in seg_names:
                    names.add(str(key))
        return out

    def _add_template_mi_extract(self):
        channels = list(self.host.proc_available_channels())
        if len(channels) < 2:
            self.host.proc_log("Template requires at least 2 fit channels.")
            return
        capture_keys = list(self.host.proc_available_capture_keys())

        dlg = QDialog(self)
        dlg.setWindowTitle("MI/TTL/SigGen Template")
        dlg_layout = QVBoxLayout(dlg)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)

        def _combo(items: List[str], default: Optional[str] = None) -> QComboBox:
            cb = QComboBox()
            for item in items:
                cb.addItem(self.host.proc_channel_display_name(str(item)), str(item))
            if default is not None:
                idx = cb.findData(str(default))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            return cb

        default_mi = "CH2" if "CH2" in channels else channels[0]
        default_ttl = (
            "CH4" if "CH4" in channels else channels[min(1, len(channels) - 1)]
        )
        default_sig = "CH3" if "CH3" in channels else channels[0]

        mi_combo = _combo(channels, default_mi)
        ttl_combo = _combo(channels, default_ttl)
        sig_combo = _combo(channels, default_sig)

        capture_items = ["(none)"] + capture_keys
        fmod_combo = _combo(
            capture_items, "f_mod" if "f_mod" in capture_keys else "(none)"
        )

        grid.addWidget(self._make_label("MI channel:"), 0, 0)
        grid.addWidget(mi_combo, 0, 1)
        grid.addWidget(self._make_label("TTL channel:"), 1, 0)
        grid.addWidget(ttl_combo, 1, 1)
        grid.addWidget(self._make_label("SigGen channel:"), 2, 0)
        grid.addWidget(sig_combo, 2, 1)
        grid.addWidget(self._make_label("f_mod capture field:"), 3, 0)
        grid.addWidget(fmod_combo, 3, 1)
        dlg_layout.addLayout(grid)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != int(QDialog.DialogCode.Accepted):
            return

        mi_channel = str(mi_combo.currentData() or "")
        ttl_channel = str(ttl_combo.currentData() or "")
        sig_channel = str(sig_combo.currentData() or "")
        if not mi_channel or not ttl_channel or not sig_channel:
            self.host.proc_log("Template cancelled: missing channel selection.")
            return
        if len({mi_channel, ttl_channel, sig_channel}) < 3:
            QMessageBox.warning(
                self,
                "Template Setup",
                "MI, TTL, and SigGen channels must be different.",
            )
            return

        param_map = self._channel_param_map()
        all_params = set(self.host.proc_available_params())
        ttl_params = set(param_map.get(ttl_channel, set()))
        sig_params = set(param_map.get(sig_channel, set()))
        mi_params = set(param_map.get(mi_channel, set()))
        if not ttl_params:
            ttl_params = set(all_params)
        if not sig_params:
            sig_params = set(all_params)
        if not mi_params:
            mi_params = set(all_params)

        if "f_mod" in all_params:
            sig_params.add("f_mod")

        ttl_free = tuple(sorted(ttl_params))
        ttl_fixed = tuple(sorted(all_params - set(ttl_free)))

        sig_free = tuple(sorted(sig_params))
        sig_fixed = tuple(sorted(all_params - set(sig_free)))

        solved_shared = (ttl_params | sig_params) & mi_params
        mi_free = tuple(sorted(mi_params - solved_shared))
        if not mi_free:
            mi_free = tuple(sorted(mi_params))
        mi_fixed = tuple(sorted(all_params - set(mi_free)))

        locked_boundary_names = tuple(
            str(name)
            for name, _members in (self.host.proc_available_boundary_groups() or [])
        )

        steps: List[ProcedureStepBase] = [
            FitStep(
                channels=(ttl_channel,),
                free_params=ttl_free,
                fixed_params=ttl_fixed,
                min_r2=0.9,
                max_retries=4,
                retry_scale=0.25,
                retry_mode="jitter_then_random",
                on_fail="stop",
                label=f"Fit {self.host.proc_channel_display_name(ttl_channel)} levels + boundaries",
            ),
        ]

        f_mod_capture = str(fmod_combo.currentData() or "").strip()
        if f_mod_capture and f_mod_capture != "(none)" and "f_mod" in all_params:
            steps.append(
                SetParameterStep(
                    assignments=(
                        ParameterAssignment(
                            target_key="f_mod",
                            source_kind="capture",
                            source_key=f_mod_capture,
                            scale=1.0,
                            offset=0.0,
                            clamp_to_bounds=True,
                            on_missing="skip",
                        ),
                    ),
                    label="Seed f_mod from filename",
                )
            )

        steps.extend(
            [
                FitStep(
                    channels=(sig_channel,),
                    free_params=sig_free,
                    fixed_params=sig_fixed,
                    min_r2=0.9,
                    max_retries=4,
                    retry_scale=0.25,
                    retry_mode="jitter_then_random",
                    locked_boundary_names=locked_boundary_names,
                    on_fail="stop",
                    label=f"Fit {self.host.proc_channel_display_name(sig_channel)} modulation",
                ),
                FitStep(
                    channels=(mi_channel,),
                    free_params=mi_free,
                    fixed_params=mi_fixed,
                    min_r2=0.9,
                    max_retries=6,
                    retry_scale=0.35,
                    retry_mode="jitter_then_random",
                    locked_boundary_names=locked_boundary_names,
                    on_fail="stop",
                    label=f"Fit {self.host.proc_channel_display_name(mi_channel)} MI output",
                ),
            ]
        )

        self._procedure_steps.extend(steps)
        self._rebuild_step_cards()
        self._notify_change()
        self.host.proc_log(
            "Inserted template: MI/TTL/SigGen Extract "
            f"({mi_channel}, {ttl_channel}, {sig_channel})."
        )

    # ── Step management ───────────────────────────────────────────

    def _add_step(self, step_type: str):
        """Add a new step of the given type."""
        from procedure_steps import _STEP_TYPE_REGISTRY

        cls = _STEP_TYPE_REGISTRY.get(step_type)
        if cls is None:
            return
        new_step = cls()
        self._procedure_steps.append(new_step)
        self._rebuild_step_cards()
        self._notify_change()

    def _remove_step(self, idx: int):
        if idx < 0 or idx >= len(self._procedure_steps):
            return
        self._procedure_steps.pop(idx)
        self._rebuild_step_cards()
        self._notify_change()

    def _move_step(self, idx: int, direction: int):
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self._procedure_steps):
            return
        steps = self._procedure_steps
        steps[idx], steps[new_idx] = steps[new_idx], steps[idx]
        self._rebuild_step_cards()
        self._notify_change()

    # -- Signal adapters: FitWorkerThread -> existing panel handlers ----------
    def _disconnect_worker_signals(self):
        """Disconnect from the host worker thread signals."""
        worker_thread = getattr(self.host, "_fit_worker_thread", None)
        if worker_thread is None:
            return
        for sig, slot in (
            (worker_thread.task_progress, self._on_worker_progress),
            (worker_thread.task_finished, self._on_worker_finished),
            (worker_thread.task_failed, self._on_worker_failed),
            (worker_thread.task_cancelled, self._on_worker_cancelled),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def _on_worker_progress(self, task_id, step_idx, _total, step_result):
        if task_id != getattr(self, "_procedure_task_id", None):
            return
        if isinstance(step_result, dict):
            self._on_step_completed(step_idx, step_result)

    def _on_worker_finished(self, task_id, results):
        if task_id != getattr(self, "_procedure_task_id", None):
            return
        self._disconnect_worker_signals()
        result = results[0] if results else {}
        self._on_finished(result)

    def _on_worker_failed(self, task_id, error_text):
        if task_id != getattr(self, "_procedure_task_id", None):
            return
        self._disconnect_worker_signals()
        self._on_failed(error_text)

    def _on_worker_cancelled(self, task_id):
        if task_id != getattr(self, "_procedure_task_id", None):
            return
        self._disconnect_worker_signals()
        self._on_cancelled()

    def _on_step_completed(self, step_idx, step_result):
        label = step_result.get("label", f"Step {step_idx + 1}")
        r2 = step_result.get("r2")
        status = step_result.get("status", "pass")
        step_type = step_result.get("step_type", "fit")
        retries = step_result.get("retries_used", 0)

        r2_text = f"R²={r2:.6f}" if r2 is not None else ""
        retry_text = f" ({retries} retries)" if retries > 0 else ""
        self._set_status_text(
            f"Completed {label}: {status.upper()} {r2_text}{retry_text}"
        )
        self.host.proc_log(
            f"  {label} [{step_type}]: {status.upper()} {r2_text}{retry_text}"
        )

        step_idx_int = int(step_idx)
        while len(self._last_step_results) <= step_idx_int:
            self._last_step_results.append({})
        self._last_step_results[step_idx_int] = dict(step_result)
        self._results_table.populate(self._last_step_results)

        line = self._format_step_key_change_line(
            step_idx_int,
            step_result,
            previous_params=self._live_prev_params_by_key,
        )
        if line:
            self._append_run_log_line(line)
        params_by_key = self._coerce_params_by_key(step_result.get("params_by_key"))
        if params_by_key:
            self._live_prev_params_by_key = params_by_key

    def _on_finished(self, result):
        self._procedure_running = False
        self._procedure_task_id = None
        if self._run_btn is not None:
            self._run_btn.setEnabled(True)
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(False)

        # Populate results table.
        step_results = result.get("step_results") or []
        if isinstance(step_results, (list, tuple)):
            self._last_step_results = [
                dict(item) for item in step_results if isinstance(item, Mapping)
            ]
        else:
            self._last_step_results = []
        self._results_table.populate(self._last_step_results)
        self._rebuild_live_key_changes(self._last_step_results)

        stopped = result.get("stopped_at_step")
        if stopped is not None:
            msg = f"Procedure stopped early at step {stopped + 1} (R² threshold)."
            self._set_status_text(msg)
            self.host.proc_log(msg)
        else:
            r2 = result.get("r2")
            r2_text = f"R²={r2:.6f}" if r2 is not None else ""
            msg = f"Procedure complete. {r2_text}".strip()
            self._set_status_text(msg)
            self.host.proc_log(msg)

        self.host.proc_on_fit_finished(result)
        self.host.proc_autosave()

    def _on_failed(self, error_msg):
        self._procedure_running = False
        self._procedure_task_id = None
        if self._run_btn is not None:
            self._run_btn.setEnabled(True)
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(False)
        msg = f"Procedure failed: {error_msg}"
        self._set_status_text(msg)
        self.host.proc_log(msg)
        self.host.proc_autosave()

    def _on_cancelled(self):
        self._procedure_running = False
        self._procedure_task_id = None
        if self._run_btn is not None:
            self._run_btn.setEnabled(True)
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(False)
        self._set_status_text("Procedure cancelled.")
        self.host.proc_log("Procedure cancelled.")
        self.host.proc_autosave()
