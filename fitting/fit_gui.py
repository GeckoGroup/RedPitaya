#!/usr/bin/env python3
"""
Manual Curve Fitting GUI for MI Model
Allows manual adjustment of parameters for failed automatic fits.
"""

import argparse
import os
import re
import sys
import html
import json
import time
import warnings
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QGraphicsDropShadowEffect,
    QMainWindow,
    QWidget,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QSlider,
    QPushButton,
    QComboBox,
    QTextEdit,
    QFileDialog,
    QLineEdit,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QAbstractSpinBox,
    QSizePolicy,
    QCheckBox,
    QMenu,
    QMessageBox,
    QFrame,
    QToolButton,
    QProgressBar,
)
from PyQt6.QtCore import (
    QCoreApplication,
    QPoint,
    QRect,
    QAbstractItemModel,
    Qt,
    QTimer,
    QSize,
    QEvent,
    QObject,
    pyqtSignal,
)
from PyQt6.QtCore import QThread
from PyQt6.QtGui import (
    QFontMetrics,
    QAction,
    QIcon,
    QPalette,
    QColor,
    QBrush,
    QTextDocument,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

# use Qt5Agg backend for better performance
from matplotlib.pyplot import switch_backend

from data_io import (
    is_supported_archive_path,
    open_archive_csv_member_stream,
    read_measurement_csv,
    stem_for_file_ref,
)
from fit_results import canonicalize_fit_row, fit_get, fit_set
from expression import (
    COLUMN_COLOR,
    parameter_symbol_to_html,
    parameter_symbol_to_mathtext,
    resolve_parameter_symbol,
    format_expression_pretty,
    format_equation_pretty,
    ExpressionSyntaxHighlighter,
    colorize_expression_html,
    is_valid_parameter_name,
)
from model import PiecewiseModelDefinition
from batch import CapturePatternConfig
from solver import SegmentSpec
from model import (
    ParameterSpec,
    MultiChannelModelDefinition,
    DEFAULT_WINDOW_TITLE,
    FIT_DETAILS_FILENAME,
    DEFAULT_TARGET_CHANNEL,
    DEFAULT_EXPRESSION,
    DEFAULT_PARAM_SPECS,
    palette_color,
    fit_companion_color,
    predict_ordered_piecewise,
    has_nonempty_values,
    finite_float_or_none,
    default_boundary_ratios,
    boundary_ratios_to_positions,
    boundary_ratios_to_x_values,
    pcts_to_boundary_ratios,
    extract_segment_parameter_names,
    build_piecewise_model_definition,
    build_multi_channel_model_definition,
    shared_to_local_flat,
    compute_r2,
    smooth_channel_array,
    fit_debug,
    _row_has_error,
)
from widgets import (
    clear_layout,
    NumericSortTableWidgetItem,
    CompactDoubleSpinBox,
    ClickableLabel,
    SingleLineStatusLabel,
    VerticallyCenteredTextEdit,
    format_boundary_display_name,
    RichTextComboBox,
    RichTextHeaderView,
    MultiHandleSlider,
    _UNICODE_SUBSCRIPT_TRANS,
    TABLE_SORT_ROLE,
)
from batch import (
    parse_capture_pattern,
    extract_captures,
    resolve_fixed_params_from_captures,
    make_batch_result_row,
    FitWorkerThread,
    ThumbnailRenderWorker,
    _BATCH_PATTERN_MISMATCH_ERROR,
    _FIT_PARAM_RANGE_ERROR_PREFIX,
)
from procedure_widgets import ProcedurePanel, ProcedureHost
from fit_state import BoundaryState


switch_backend("Qt5Agg")

APP_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "redpitaya_icon.png"


class _ScrollEatFilter(QObject):
    """Event filter that eats wheel events on unfocused spinboxes."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------------
# Live Procedure Panel — replaces param sliders during multi-step runs
# ---------------------------------------------------------------------------


class ProcedureLivePanel(QWidget):
    """Live-updating view of procedure execution.

    Shown in place of the parameter sliders while a multi-step procedure is
    running.  Mirrors the console fit-log layout: step metadata on one line,
    attempt details indented underneath the active step.
    """

    _STATUS_ICONS = {
        "pending": "\u2022",  # bullet
        "running": "\u25b6",  # play triangle
        "pass": "\u2714",  # check mark
        "fail": "\u2718",  # cross mark
        "skipped": "\u2013",  # en dash
    }
    _STATUS_COLORS = {
        "pending": "#94a3b8",
        "running": "#2563eb",
        "pass": "#16a34a",
        "fail": "#dc2626",
        "skipped": "#9ca3af",
    }

    # ── Styling tokens ──
    _FONT_HDR = "font-size: 14px; font-weight: 700;"
    _FONT_STEP = "font-size: 13px;"
    _FONT_STEP_BOLD = "font-size: 13px; font-weight: 600;"
    _FONT_DETAIL = "font-size: 12px;"
    _FONT_DETAIL_BOLD = "font-size: 12px; font-weight: 600;"
    _COLOR_PRIMARY = "color: #334155;"
    _COLOR_SECONDARY = "color: #64748b;"
    _COLOR_DIM = "color: #94a3b8;"
    _COLOR_HEADER = "color: #475569;"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._step_rows: List[dict] = []
        self._step_statuses: List[str] = []
        self._n_steps: int = 0
        self._procedure_t0: float = 0.0
        self._active_step_idx: int = -1
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────

    dismissed = pyqtSignal()

    def _build_ui(self):
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            "ProcedureLivePanel {"
            "  background: #ffffff;"
            "  border-top: 1px solid #94a3b8;"
            "}"
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, -4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar — opaque background
        header_widget = QWidget()
        header_widget.setAutoFillBackground(True)
        header_widget.setStyleSheet(
            "background: #f8fafc; border-bottom: 1px solid #e2e8f0;"
        )
        header_inner = QHBoxLayout(header_widget)
        header_inner.setContentsMargins(10, 6, 6, 6)
        header_inner.setSpacing(8)
        self._procedure_name_label = QLabel("Procedure")
        self._procedure_name_label.setStyleSheet(
            f"{self._FONT_HDR} {self._COLOR_HEADER}"
        )
        header_inner.addWidget(self._procedure_name_label)
        self._file_label = QLabel("")
        self._file_label.setStyleSheet(f"{self._FONT_STEP} {self._COLOR_SECONDARY}")
        header_inner.addWidget(self._file_label, 1)
        self._elapsed_label = QLabel("")
        self._elapsed_label.setStyleSheet(f"{self._FONT_STEP} {self._COLOR_DIM}")
        self._elapsed_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        header_inner.addWidget(self._elapsed_label)

        self._dismiss_btn = QPushButton("\u2715")
        self._dismiss_btn.setFixedSize(22, 22)
        self._dismiss_btn.setStyleSheet(
            "QPushButton { border: none; font-size: 14px; color: #94a3b8;"
            " border-radius: 3px; padding: 0; background: transparent; }"
            "QPushButton:hover { background: #e2e8f0; color: #334155; }"
        )
        self._dismiss_btn.setToolTip("Hide procedure log")
        self._dismiss_btn.clicked.connect(self._on_dismiss)
        header_inner.addWidget(self._dismiss_btn)
        layout.addWidget(header_widget)

        # Steps scroll area
        self._steps_scroll = QScrollArea()
        self._steps_scroll.setWidgetResizable(True)
        self._steps_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._steps_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._steps_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea QWidget#qt_scrollarea_viewport {"
            " background: transparent; }"
        )
        self._steps_widget = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_widget)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._steps_layout.setSpacing(2)
        self._steps_scroll.setWidget(self._steps_widget)
        layout.addWidget(self._steps_scroll, 1)

    # ── Public API ─────────────────────────────────────────────────

    def start_procedure(
        self,
        procedure_name: str,
        step_infos: List[dict],
        file_label: str = "",
    ):
        """Initialise the panel for a new procedure run.

        Parameters
        ----------
        step_infos : list of dict
            Each dict has at minimum ``"label"`` and ``"step_type"``, and
            optionally ``"channels"``, ``"n_free"``, ``"n_fixed"``,
            ``"max_retries"``, ``"retry_mode"``, ``"locked_boundary_names"``.
        """
        self._procedure_name_label.setText(procedure_name or "Procedure")
        self._file_label.setText(file_label)
        self._file_label.setVisible(bool(file_label))
        self._procedure_t0 = time.perf_counter()
        self._elapsed_label.setText("")
        self._active_step_idx = -1

        # Clear existing step rows
        for i in reversed(range(self._steps_layout.count())):
            item = self._steps_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
            elif item and item.spacerItem():
                self._steps_layout.removeItem(item)
        self._step_rows.clear()
        self._step_statuses.clear()
        self._n_steps = len(step_infos)

        for idx, info in enumerate(step_infos):
            step_type = str(info.get("step_type", ""))

            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 3, 4, 3)
            row_layout.setSpacing(1)

            # ── Single compact line:  ● Step 1/5 [fit] ch2 — 5 free   R²=…  1.2s ──
            top_row = QHBoxLayout()
            top_row.setSpacing(4)

            icon_lbl = QLabel(self._STATUS_ICONS["pending"])
            icon_lbl.setFixedWidth(14)
            icon_lbl.setStyleSheet(
                f"{self._FONT_STEP} color: {self._STATUS_COLORS['pending']};"
            )
            top_row.addWidget(icon_lbl)

            # Build compact heading: "Step 1/5 [fit] ch2 — 5 free"
            heading_parts: List[str] = [
                f"Step {idx + 1}/{len(step_infos)}",
                f"[{step_type}]",
            ]
            channels = info.get("channels")
            if channels:
                heading_parts.append(", ".join(str(c) for c in channels))
            desc_bits: List[str] = []
            n_free = info.get("n_free")
            if n_free:
                desc_bits.append(f"{n_free} free")
            n_fixed = info.get("n_fixed")
            if n_fixed:
                desc_bits.append(f"{n_fixed} fixed")
            locked = info.get("locked_boundary_names")
            if locked:
                desc_bits.append(f"{len(locked)} locked bnd")
            if desc_bits:
                heading_parts.append("\u2014 " + ", ".join(desc_bits))

            text_lbl = QLabel(" ".join(heading_parts))
            text_lbl.setStyleSheet(f"{self._FONT_STEP_BOLD} {self._COLOR_PRIMARY}")
            top_row.addWidget(text_lbl, 1)

            r2_lbl = QLabel("")
            r2_lbl.setMinimumWidth(180)
            r2_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            r2_lbl.setStyleSheet(f"{self._FONT_STEP} {self._COLOR_SECONDARY}")
            top_row.addWidget(r2_lbl)

            elapsed_lbl = QLabel("")
            elapsed_lbl.setFixedWidth(52)
            elapsed_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            elapsed_lbl.setStyleSheet(f"{self._FONT_STEP} {self._COLOR_DIM}")
            top_row.addWidget(elapsed_lbl)

            row_layout.addLayout(top_row)

            # ── Attempt sub-area (hidden; shown only when retries occur) ──
            attempt_widget = QWidget()
            attempt_layout = QVBoxLayout(attempt_widget)
            attempt_layout.setContentsMargins(19, 1, 0, 1)
            attempt_layout.setSpacing(0)

            att_bar = QProgressBar()
            att_bar.setMaximumHeight(4)
            att_bar.setTextVisible(False)
            att_bar.setStyleSheet(
                "QProgressBar { border: 1px solid #cbd5e1; border-radius: 2px; "
                "background: #f1f5f9; }"
                "QProgressBar::chunk { background: #2563eb; border-radius: 1px; }"
            )
            att_bar.hide()
            attempt_layout.addWidget(att_bar)

            attempt_widget.hide()
            row_layout.addWidget(attempt_widget)

            self._steps_layout.addWidget(row_widget)
            self._step_rows.append(
                {
                    "icon": icon_lbl,
                    "text": text_lbl,
                    "r2": r2_lbl,
                    "elapsed": elapsed_lbl,
                    "attempt_widget": attempt_widget,
                    "attempt_layout": attempt_layout,
                    "att_bar": att_bar,
                    "att_rows": [],
                    "widget": row_widget,
                }
            )
            self._step_statuses.append("pending")

        self._steps_layout.addStretch()

    @staticmethod
    def _format_per_channel_r2(per_channel_r2: Optional[Mapping[str, Any]]) -> str:
        """Render per-channel R² mapping as compact text."""
        if not isinstance(per_channel_r2, Mapping) or not per_channel_r2:
            return ""
        parts: List[str] = []
        for raw_channel in sorted(per_channel_r2.keys(), key=lambda key: str(key)):
            channel_key: str = str(raw_channel).strip()
            if not channel_key:
                continue
            r2_val = finite_float_or_none(per_channel_r2.get(raw_channel))
            if r2_val is None:
                continue
            parts.append(f"{channel_key}={float(r2_val):.6f}")
        return ", ".join(parts)

    def update_step(
        self,
        step_idx: int,
        status: str,
        r2: Optional[float] = None,
        step_result: Optional[dict] = None,
    ):
        """Update a step's status, R², and timing on the single heading line."""
        if step_idx < 0 or step_idx >= len(self._step_rows):
            return
        row = self._step_rows[step_idx]
        icon = self._STATUS_ICONS.get(status, self._STATUS_ICONS["pending"])
        color = self._STATUS_COLORS.get(status, self._STATUS_COLORS["pending"])
        row["icon"].setText(icon)
        row["icon"].setStyleSheet(f"{self._FONT_STEP} color: {color};")
        self._step_statuses[step_idx] = status

        # Hide attempt sub-area once step is done
        if status in ("pass", "fail", "skipped"):
            row["attempt_widget"].hide()

        per_channel_text = ""
        if isinstance(step_result, dict):
            per_channel_text = self._format_per_channel_r2(
                step_result.get("per_channel_r2")
            )

        # R² display on the heading line
        if r2 is not None:
            r2_color = (
                "#16a34a" if r2 > 0.99 else "#d97706" if r2 > 0.95 else "#dc2626"
            )
            summary = f"R\u00b2={r2:.6f}"
            if per_channel_text:
                summary = f"{summary} | {per_channel_text}"
            row["r2"].setText(summary)
            row["r2"].setToolTip(summary if per_channel_text else "")
            row["r2"].setStyleSheet(f"{self._FONT_STEP_BOLD} color: {r2_color};")
        elif status in ("pass", "fail", "skipped"):
            msg = ""
            if step_result and step_result.get("message"):
                msg = str(step_result["message"])
            summary = msg or status
            if per_channel_text:
                summary = (
                    f"{summary} | {per_channel_text}" if summary else per_channel_text
                )
            row["r2"].setText(summary)
            row["r2"].setToolTip(summary if per_channel_text else "")
            row["r2"].setStyleSheet(f"{self._FONT_STEP} color: {color};")

        # Elapsed time
        if step_result and step_result.get("elapsed") is not None:
            elapsed = float(step_result["elapsed"])
            row["elapsed"].setText(f"{elapsed:.2f}s")

        # Update overall elapsed
        self._elapsed_label.setText(f"{time.perf_counter() - self._procedure_t0:.1f}s")

        # Auto-scroll to keep active area visible
        self._scroll_to_step(step_idx)

    def mark_step_running(self, step_idx: int):
        """Mark a step as currently running (before completion)."""
        self.update_step(step_idx, "running")

    def update_attempt(
        self,
        step_idx: int,
        attempt: int,
        max_attempts: int,
        r2: Optional[float],
        best_r2: Optional[float],
        is_new_best: bool,
        strategy: str,
        elapsed: float = 0.0,
        per_channel_r2: Optional[Mapping[str, Any]] = None,
    ):
        """Update the step heading R² live; show attempt detail rows only when retries occur."""
        if step_idx < 0 or step_idx >= len(self._step_rows):
            return
        row = self._step_rows[step_idx]

        # Mark this step as running
        if self._step_statuses[step_idx] != "running":
            self.mark_step_running(step_idx)
        if self._active_step_idx != step_idx:
            if 0 <= self._active_step_idx < len(self._step_rows):
                self._step_rows[self._active_step_idx]["attempt_widget"].hide()
            self._active_step_idx = step_idx

        # Always update the heading R² with the current best
        display_r2 = best_r2 if best_r2 is not None else r2
        per_channel_text = self._format_per_channel_r2(per_channel_r2)
        if display_r2 is not None:
            r2_color = (
                "#16a34a"
                if display_r2 > 0.99
                else "#d97706"
                if display_r2 > 0.95
                else "#dc2626"
            )
            summary = f"R\u00b2={display_r2:.6f}"
            if per_channel_text:
                summary = f"{summary} | {per_channel_text}"
            row["r2"].setText(summary)
            row["r2"].setToolTip(summary if per_channel_text else "")
            row["r2"].setStyleSheet(f"{self._FONT_STEP_BOLD} color: {r2_color};")
        elif per_channel_text:
            row["r2"].setText(per_channel_text)
            row["r2"].setToolTip(per_channel_text)
            row["r2"].setStyleSheet(f"{self._FONT_STEP} {self._COLOR_SECONDARY}")

        # Show elapsed on heading
        if elapsed > 0:
            row["elapsed"].setText(f"{elapsed:.2f}s")

        # Always show attempt rows with R² and time
        row["attempt_widget"].show()

        # Build compact attempt line:  #1/3 best_init  R²=0.999456 ★  0.12s
        parts: List[str] = [f"#{attempt + 1}/{max_attempts}"]
        if strategy:
            parts.append(strategy)
        if r2 is not None:
            parts.append(f"R\u00b2={r2:.6f}")
        if per_channel_text:
            parts.append(per_channel_text)
        if is_new_best and attempt > 0:
            parts.append("\u2605")
        if elapsed > 0:
            parts.append(f"{elapsed:.3f}s")

        att_lbl = QLabel("  ".join(parts))
        att_color = "#16a34a" if is_new_best else "#94a3b8"
        att_lbl.setStyleSheet(f"{self._FONT_DETAIL} color: {att_color};")

        att_layout = row["attempt_layout"]
        bar_index = att_layout.indexOf(row["att_bar"])
        att_layout.insertWidget(bar_index, att_lbl)
        row["att_rows"].append(att_lbl)

        # Progress bar (only when there are multiple attempts)
        if max_attempts > 1:
            row["att_bar"].show()
            row["att_bar"].setMaximum(max_attempts)
            row["att_bar"].setValue(attempt + 1)
            bar_color = "#16a34a" if is_new_best else "#d97706"
            row["att_bar"].setStyleSheet(
                "QProgressBar { border: 1px solid #cbd5e1;"
                " border-radius: 2px; background: #f1f5f9; }"
                f"QProgressBar::chunk {{ background: {bar_color};"
                " border-radius: 1px; }"
            )

        self._steps_scroll.ensureWidgetVisible(att_lbl, 0, 40)

        # Update overall elapsed
        self._elapsed_label.setText(f"{time.perf_counter() - self._procedure_t0:.1f}s")

        # Auto-scroll to step
        self._scroll_to_step(step_idx)

    def finish_procedure(self):
        """Called when the procedure completes — update header."""
        total = time.perf_counter() - self._procedure_t0
        self._elapsed_label.setText(f"Done \u2014 {total:.1f}s")
        # Hide any lingering attempt widget
        if 0 <= self._active_step_idx < len(self._step_rows):
            self._step_rows[self._active_step_idx]["attempt_widget"].hide()
        self._active_step_idx = -1

    def _scroll_to_step(self, step_idx: int):
        """Ensure the given step row is visible in the scroll area."""
        if step_idx < 0 or step_idx >= len(self._step_rows):
            return
        widget = self._step_rows[step_idx]["widget"]
        self._steps_scroll.ensureWidgetVisible(widget, 0, 40)

    # ── Overlay positioning ─────────────────────────────────────

    def reposition(self):
        """Full-width, anchored to the bottom of the parent widget."""
        p = self.parentWidget()
        if p is None:
            return
        pw, ph = p.width(), p.height()
        h = min(int(ph * 0.55), max(220, ph - 40))
        self.setGeometry(0, ph - h, pw, h)

    def _on_dismiss(self):
        self.hide()
        self.dismissed.emit()


class DataPreloadWorker(QObject):
    """Background CSV reader that preloads file frames for faster navigation."""

    file_loaded = pyqtSignal(int, str, object, object)  # (session, file_ref, frame, error)
    finished = pyqtSignal(int)  # (session)

    def __init__(self, session_id: int, file_refs) -> None:
        super().__init__()
        self.session_id = int(session_id)
        self.file_refs: List[str] = [
            str(ref).strip() for ref in list(file_refs or []) if str(ref).strip()
        ]
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        for file_ref in self.file_refs:
            if self.cancel_requested:
                break
            frame = None
            error = None
            try:
                frame = read_measurement_csv(file_ref)
            except Exception as exc:
                error = str(exc)
            if self.cancel_requested:
                break
            self.file_loaded.emit(self.session_id, str(file_ref), frame, error)
        self.finished.emit(self.session_id)


class ManualFitGUI(QMainWindow):
    def __init__(self, source_path: Optional[str] = None):
        super().__init__()
        self._scroll_eat_filter = _ScrollEatFilter(self)
        self.param_specs = list(DEFAULT_PARAM_SPECS)
        self.param_spinboxes = {}
        self.param_sliders = {}
        self.param_min_spinboxes = {}
        self.param_max_spinboxes = {}
        self._model_param_min_spinboxes = {}
        self._model_param_max_spinboxes = {}
        self._model_param_periodic_checkboxes = {}
        self.param_lock_status_labels = {}
        self.param_tail_spacers_by_key = {}
        self.breakpoint_controls = {}
        self.param_row_tail_spacers = []
        self._param_slider_steps = 2000
        self._param_name_width = 88
        self._param_bound_width = 88
        self._param_value_width = 94
        self._show_plot_param_bounds = False
        self._fit_option_label_width = 64
        self._param_tail_placeholder_width = 0
        self._last_fit_active_keys = []
        self._apply_expression_in_progress = False
        self._channel_sync_in_progress = False
        self._highlight_refresh_in_progress = False
        self._file_load_in_progress = False
        self._expression_edit_mode = False
        self.current_expression = f"{DEFAULT_TARGET_CHANNEL} = {DEFAULT_EXPRESSION}"
        try:
            self._piecewise_model = build_piecewise_model_definition(
                target_col=DEFAULT_TARGET_CHANNEL,
                segment_exprs=[part.strip() for part in DEFAULT_EXPRESSION.split(";")],
                channel_names=["TIME", "CH2", "CH3", "CH4"],
            )
        except Exception:
            self._piecewise_model = None
        # Multi-channel model: wraps one or more PiecewiseModelDefinition objects.
        # Parameters with the same name are shared across channels.
        self._multi_channel_model = None
        self._fit_state = BoundaryState()
        self._boundary_slider_mapping = None  # Used for multi-channel slider dispatch
        self._boundary_handle_map = (
            None  # list of [(target, idx), ...] per slider handle
        )
        self._boundary_name_map = {}  # {(target, idx): "X₀"/"X₁"/...} boundary name assignment
        self._boundary_name_edits = {}  # {(target, idx): QLineEdit} name editors
        self._manually_fixed_boundary_ids = set()  # {(target_col, boundary_idx), ...}
        self._boundary_fix_checkboxes = {}
        self._manually_fixed_params = set()  # Param keys manually fixed by user
        self._periodic_param_keys = set()  # Param keys treated as periodic
        self.param_fix_checkboxes = {}  # key -> QCheckBox
        self._param_channel_header_labels = {}  # target_col -> QLabel
        self._fit_channel_enabled = {}  # {target_col: bool} per-equation fit toggles
        self._fit_channel_checkboxes = {}  # {target_col: QCheckBox}
        self._last_per_channel_r2 = {}  # {target_col: float|None}
        self._mapped_param_seed_file_key = None
        try:
            self._multi_channel_model: MultiChannelModelDefinition = (
                build_multi_channel_model_definition(
                    [
                        (
                            DEFAULT_TARGET_CHANNEL,
                            [part.strip() for part in DEFAULT_EXPRESSION.split(";")],
                        )
                    ],
                    channel_names=["TIME", "CH2", "CH3", "CH4"],
                )
            )
        except Exception:
            pass

        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        if APP_ICON_PATH.exists():
            icon = QIcon(str(APP_ICON_PATH))
            if not icon.isNull():
                self.setWindowIcon(icon)
        self.setGeometry(100, 100, 1000, 800)

        self.data_files = []
        self.current_file_idx = 0
        self.current_data = None
        self.channels = {
            "CH2": "MI output voltage",
            "CH3": "Sig Gen",
            "CH4": "Trigger",
            "TIME": "Time",
        }
        self.channel_units = {
            "CH2": "s",
            "CH3": "V",
            "CH4": "V",
            "TIME": "V",
        }
        self.x_channel = "TIME"
        self.last_popt = None
        self.auto_fit_btn_default_text = "Auto Fit"
        self._auto_fit_run_mode = "fit"
        self._batch_fit_run_mode = "fit"
        self._fit_compute_mode = self._normalize_fit_compute_mode(
            os.environ.get("REDPITAYA_JAX_PLATFORM", "gpu")
        )
        self._apply_fit_compute_mode_env(self._fit_compute_mode)
        self._auto_fit_mode_actions = {}
        self._batch_fit_mode_actions = {}
        initial_boundary_count = max(
            0,
            len(self._piecewise_model.segment_exprs) - 1
            if self._piecewise_model is not None
            else 2,
        )
        initial_target = (
            str(self._piecewise_model.target_col)
            if self._piecewise_model is not None
            else None
        )
        initial_topology = (
            {initial_target: initial_boundary_count}
            if initial_target is not None
            else {}
        )
        self._fit_state.set_topology(
            initial_topology,
            primary_target=initial_target,
            preserve_existing=False,
        )
        self._last_r2 = None
        self._last_per_channel_r2 = {}
        self.param_capture_map = {}
        self.param_capture_combos = {}
        self.fit_tasks: Dict[int, dict] = {}
        self._fit_task_counter = 0
        self._fit_worker_thread = FitWorkerThread(parent=self)
        self._fit_worker_thread.task_progress.connect(self._on_fit_task_progress)
        self._fit_worker_thread.task_step_completed.connect(self._on_fit_task_step)
        self._fit_worker_thread.task_attempt_completed.connect(
            self._on_fit_task_attempt
        )
        self._fit_worker_thread.task_finished.connect(self._on_fit_task_finished)
        self._fit_worker_thread.task_failed.connect(self._on_fit_task_failed)
        self._fit_worker_thread.task_cancelled.connect(self._on_fit_task_cancelled)
        self._fit_worker_thread.start()
        self._data_preload_cache: Dict[str, Any] = {}
        self._data_preload_failed: set[str] = set()
        self._data_preload_thread: QThread | None = None
        self._data_preload_worker: DataPreloadWorker | None = None
        self._data_preload_session: int = 0
        self._batch_refit_random_restarts = 2
        self._batch_total_tasks = 0
        self._batch_progress_started_at = 0.0
        self._batch_cancel_requested = False
        self.batch_fit_in_progress = False
        self._batch_cancel_pending = False
        self.batch_results = []
        self.batch_files = []
        self.batch_capture_keys = []
        self.batch_match_count = 0
        self.batch_unmatched_files = []
        self.analysis_records = []
        self.analysis_columns = []
        self.analysis_numeric_data = {}
        self.analysis_param_columns = []
        self._analysis_math_controls_refreshing = False
        self._analysis_row_x_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._analysis_point_pick_cid = None
        self._analysis_hover_cid = None
        self._analysis_hover_leave_cid = None
        self._analysis_hover_artists = []
        self._analysis_hover_annotations = {}
        self._analysis_scatter_files = {}
        self._analysis_pending_pick = None
        self._analysis_pick_load_timer = QTimer()
        self._analysis_pick_load_timer.setSingleShot(True)
        self._analysis_pick_load_timer.timeout.connect(
            self._flush_pending_analysis_point_pick
        )
        self.batch_row_height = 64
        self.batch_row_height_min = 40
        self.batch_row_height_max = 320
        self.batch_thumbnail_aspect = 1.5
        self.batch_thumbnail_supersample = 2.0
        self._batch_row_height_sync = False
        self._batch_progress_done = 0
        self._batch_submission_context = None
        self._close_shutdown_in_progress = False
        self._close_force_accept = False
        self._close_shutdown_deadline = 0.0
        self.regex_timer = QTimer()
        self.regex_timer.setSingleShot(True)
        self.regex_timer.timeout.connect(self._do_prepare_batch_preview)
        self._batch_analysis_refresh_pending = False
        self._batch_analysis_refresh_timer = QTimer()
        self._batch_analysis_refresh_timer.setSingleShot(True)
        self._batch_analysis_refresh_timer.timeout.connect(
            self._flush_batch_analysis_refresh
        )
        self._idle_archive_scan_queue: List[str] = []
        self._idle_archive_scan_total: int = 0
        self._idle_archive_scan_done: int = 0
        self._idle_archive_scan_added: int = 0
        self._idle_archive_scan_session: int = 0
        self._idle_archive_scan_stream = None
        self._idle_archive_scan_current_archive: str | None = None
        self._idle_archive_scan_current_found: int = 0
        self._last_source_load_cancelled: bool = False
        self._idle_archive_scan_timer = QTimer()
        self._idle_archive_scan_timer.setSingleShot(True)
        self._idle_archive_scan_timer.timeout.connect(self._process_idle_archive_scan)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        self._pending_thumbnail_rows = set()
        self._batch_preview_ready = False
        app: QCoreApplication | None = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # Parameter initial values default to the midpoint of each parameter range.
        self.defaults = self._default_param_values(self.param_specs)

        # Optimization: timer for debouncing updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.do_full_update)
        self.slider_active = False

        # Cache for plot data
        self.cached_time_data = None
        self.raw_channel_cache = {}
        self.channel_cache = {}
        self._expression_channel_data_cache = None
        self._display_target_points = 3000
        self._plot_has_residual_axis = False
        self._channel_visibility = {}  # channel_name -> bool
        self._channel_toggle_buttons = {}  # channel_name -> QPushButton
        self._last_file_load_error: str = ""
        self.smoothing_enabled = True
        self.smoothing_window = 101

        # Current source path (directory/archive/csv)
        default_source = "../AFG_measurements/"
        if source_path not in (None, ""):
            self.current_dir = str(Path(str(source_path)).expanduser())
        else:
            self.current_dir = str(Path(default_source).expanduser())
        self._source_display_override = None
        self._source_selected_paths = []
        self._fit_details_restore_in_progress = False
        self._fit_details_autoload_attempted = False

        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("root")
        self.setCentralWidget(central_widget)
        self._enforce_light_mode()
        self._apply_compact_ui_defaults()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        self.stats_text = SingleLineStatusLabel("")
        self.stats_text.setObjectName("statsLine")
        self.stats_text.setStyleSheet("padding: 0px 2px; margin: 0px;")

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self.tabs, 1)

        self.manual_tab = QWidget()
        self.model_tab = QWidget()
        self.batch_tab = QWidget()
        self.analysis_tab = QWidget()
        self.procedure_tab = QWidget()
        self.tabs.addTab(self.manual_tab, "Plot")
        self.tabs.addTab(self.batch_tab, "Batch")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.model_tab, "Model")
        self.tabs.addTab(self.procedure_tab, "Procedure")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._attach_tab_corner_controls()

        manual_layout = QVBoxLayout(self.manual_tab)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(4)

        # Add draggable sizer (QSplitter) to GUI panes.
        # This splitter allows the user to resize the parameter controls and plot area interactively.
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.setChildrenCollapsible(False)

        self.param_controls_container = QWidget()
        self.param_controls_layout = QVBoxLayout(self.param_controls_container)
        self.param_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.param_controls_layout.setSpacing(0)
        self.create_parameters_frame(self.param_controls_layout)
        self._refresh_fit_action_buttons()
        self._refresh_batch_controls()

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)
        self.create_plot_frame(self.plot_layout)

        self.main_splitter.addWidget(self.param_controls_container)
        self.main_splitter.addWidget(self.plot_container)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        manual_layout.addWidget(self.main_splitter)

        model_layout = QVBoxLayout(self.model_tab)
        model_layout.setContentsMargins(6, 6, 6, 6)
        model_layout.setSpacing(6)
        self.create_model_tab(model_layout)

        batch_layout = QVBoxLayout(self.batch_tab)
        batch_layout.setContentsMargins(6, 6, 6, 6)
        batch_layout.setSpacing(6)
        self.create_batch_results_frame(batch_layout)

        analysis_layout = QVBoxLayout(self.analysis_tab)
        analysis_layout.setContentsMargins(6, 6, 6, 6)
        analysis_layout.setSpacing(6)
        self.create_batch_analysis_frame(analysis_layout)

        procedure_layout = QVBoxLayout(self.procedure_tab)
        procedure_layout.setContentsMargins(6, 6, 6, 6)
        procedure_layout.setSpacing(6)
        self._procedure_panel = ProcedurePanel(
            host=self._make_procedure_host(), parent=self.procedure_tab
        )
        procedure_layout.addWidget(self._procedure_panel, 1)

        QTimer.singleShot(0, self._apply_default_splitter_sizes)
        # Defer initial file discovery/load until the UI has painted once.
        self.stats_text.setText("Loading data sources...")
        QTimer.singleShot(0, self.load_files)

    def _on_tab_changed(self, _index):
        current_widget: QWidget | None = (
            self.tabs.currentWidget() if hasattr(self, "tabs") else None
        )
        if hasattr(self, "tab_r2_label"):
            self.tab_r2_label.setVisible(current_widget is self.manual_tab)
        if current_widget not in (self.batch_tab, self.analysis_tab):
            return
        if not self._batch_preview_ready:
            self._batch_preview_ready = True
            self.prepare_batch_preview()
            self._expand_file_column_for_selected_files()
        if current_widget == self.analysis_tab:
            self._refresh_batch_analysis_if_run()

    def _attach_tab_corner_controls(self):
        if not hasattr(self, "tabs"):
            return
        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(4)
        if hasattr(self, "stats_text"):
            self.stats_text.setSizePolicy(
                QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
            )
            self.stats_text.setMinimumWidth(260)
            self.stats_text.setMaximumWidth(560)
            corner_layout.addWidget(self.stats_text)
        self.tabs.setCornerWidget(corner_widget, Qt.Corner.TopRightCorner)

    def sizeHint(self):
        return QSize(1000, 800)

    def _apply_default_splitter_sizes(self):
        main_splitter: Any | None = getattr(self, "main_splitter", None)
        if main_splitter is not None:
            total: int = max(700, int(self.height()) or 700)
            top = int(total * 0.40)
            bottom: int = max(260, total - top)
            main_splitter.setSizes([top, bottom])

        param_splitter: Any | None = getattr(self, "param_fit_splitter", None)
        if param_splitter is not None:
            total_w: int = max(900, int(self.width()) or 900)
            left = int(total_w * 0.62)
            right: int = max(320, total_w - left)
            param_splitter.setSizes([left, right])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_source_path_label()
        QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
        # Reposition the procedure-log overlay if visible
        panel = getattr(self, "_procedure_live_panel", None)
        if panel is not None and panel.isVisible():
            panel.reposition()

    def eventFilter(self, watched, event):
        if (
            watched is getattr(self, "param_header_widget", None)
            and event is not None
            and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)
        ):
            QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
        if event is not None and event.type() == QEvent.Type.MouseButtonPress:
            self._defocus_editors_on_outside_click(watched, event)
        if (
            self._expression_edit_mode
            and event is not None
            and event.type() == QEvent.Type.MouseButtonPress
            and QApplication.activePopupWidget() is None
        ):
            active_modal: QWidget | None = QApplication.activeModalWidget()
            if active_modal is not None and active_modal is not self:
                return super().eventFilter(watched, event)
            clicked_widget: QWidget | None = (
                watched if isinstance(watched, QWidget) else None
            )
            global_pos = None
            if hasattr(event, "globalPosition"):
                try:
                    global_pos = event.globalPosition().toPoint()
                except Exception:
                    global_pos = None
            elif hasattr(event, "globalPos"):
                try:
                    global_pos = event.globalPos()
                except Exception:
                    global_pos = None
            if global_pos is not None:
                widget_at_pos: QWidget | None = QApplication.widgetAt(global_pos)
                if widget_at_pos is not None:
                    clicked_widget: QWidget = widget_at_pos
            if not self._is_expression_editor_child(clicked_widget):
                QTimer.singleShot(
                    0, lambda: self._apply_expression_on_focus_leave(force=True)
                )
        return super().eventFilter(watched, event)

    @staticmethod
    def _editable_parent_widget(widget: QWidget | None) -> QWidget | None:
        candidate: QWidget | None = None
        probe: QWidget | None = widget
        while isinstance(probe, QWidget):
            if isinstance(probe, QAbstractSpinBox):
                return probe
            if candidate is None and isinstance(probe, QLineEdit):
                if probe.isEnabled() and not probe.isReadOnly():
                    candidate = probe
            elif candidate is None and isinstance(probe, QTextEdit):
                if probe.isEnabled() and not probe.isReadOnly():
                    candidate = probe
            probe = probe.parentWidget()
        return candidate

    def _defocus_editors_on_outside_click(self, watched, event):
        active_popup: QWidget | None = QApplication.activePopupWidget()
        if active_popup is not None:
            return

        focused_widget: QWidget | None = QApplication.focusWidget()
        focused_editor: QWidget | None = self._editable_parent_widget(focused_widget)
        if focused_editor is None:
            return

        clicked_widget: QWidget | None = (
            watched if isinstance(watched, QWidget) else None
        )
        global_pos = None
        if hasattr(event, "globalPosition"):
            try:
                global_pos = event.globalPosition().toPoint()
            except Exception:
                global_pos = None
        elif hasattr(event, "globalPos"):
            try:
                global_pos = event.globalPos()
            except Exception:
                global_pos = None
        if global_pos is not None:
            widget_at_pos: QWidget | None = QApplication.widgetAt(global_pos)
            if widget_at_pos is not None:
                clicked_widget: QWidget = widget_at_pos

        if clicked_widget is not None and (
            clicked_widget is focused_editor
            or focused_editor.isAncestorOf(clicked_widget)
        ):
            return
        clicked_editor: QWidget | None = self._editable_parent_widget(clicked_widget)
        if clicked_editor is not None:
            return

        if focused_widget is not None:
            focused_widget.clearFocus()
        focused_editor.clearFocus()

    def _enforce_light_mode(self):
        """Force a light Qt palette regardless of system appearance."""
        app: QCoreApplication | None = QApplication.instance()
        if app is None:
            return

        app.setStyle("Fusion")

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f5f7fa"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f8fafc"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#dbeafe"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#111827"))

        palette.setColor(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#9ca3af")
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            QColor("#9ca3af"),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.WindowText,
            QColor("#9ca3af"),
        )

        app.setPalette(palette)

    def _apply_compact_ui_defaults(self):
        """Apply a clean, compact light UI stylesheet."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root, QTabWidget > QWidget {
                background: #f5f7fa;
                color: #111827;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #e3e8ef;
                border-radius: 8px;
                margin-top: 0px;
                padding-top: 2px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0px;
                margin: 0px;
                color: transparent;
                max-height: 0px;
            }
            QPushButton {
                min-height: 22px;
                padding: 2px 8px;
                background: #ffffff;
                color: #111827;
                border-radius: 8px;
                border: 1px solid #d3dae3;
            }
            QPushButton:hover {
                background: #f3f6f9;
                border-color: #c7d0dc;
            }
            QPushButton:pressed {
                background: #eaf0f5;
            }
            QPushButton:checked {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            QPushButton:disabled {
                color: #9ca3af;
                background: #f5f7fa;
                border-color: #e4e9ef;
            }
            QToolButton#actionButton {
                min-height: 22px;
                padding: 2px 10px 2px 8px;
                background: #ffffff;
                color: #111827;
                border-radius: 8px;
                border: 1px solid #d3dae3;
            }
            QToolButton#actionButton::menu-button {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #d3dae3;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QToolButton#actionButton:hover {
                background: #f3f6f9;
                border-color: #c7d0dc;
            }
            QToolButton#actionButton:hover::menu-button {
                border-left-color: #c7d0dc;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QToolButton#actionButton:pressed {
                background: #eaf0f5;
            }
            QToolButton#actionButton:pressed::menu-button {
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QToolButton#actionButton:disabled {
                color: #9ca3af;
                background: #f5f7fa;
                border-color: #e4e9ef;
            }
            QToolButton#actionButton:disabled::menu-button {
                border-left-color: #e4e9ef;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QPushButton[primary="true"] {
                background: #2563eb;
                color: white;
                border-color: #1d4ed8;
            }
            QPushButton[primary="true"]:hover {
                background: #1d4ed8;
            }
            QPushButton[primary="true"]:pressed {
                background: #1e40af;
            }
            QToolButton#actionButton[primary="true"] {
                background: #2563eb;
                color: white;
                border-color: #1d4ed8;
            }
            QToolButton#actionButton[primary="true"]::menu-button {
                border-left: 1px solid #1d4ed8;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QToolButton#actionButton[primary="true"]:hover {
                background: #1d4ed8;
            }
            QToolButton#actionButton[primary="true"]:hover::menu-button {
                border-left-color: #1e40af;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QToolButton#actionButton[primary="true"]:pressed {
                background: #1e40af;
            }
            QToolButton#actionButton[primary="true"]:pressed::menu-button {
                border-left-color: #1e3a8a;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QLabel#sourcePathLabel {
                color: #1d4ed8;
                font-weight: 600;
                text-decoration: underline;
                padding: 1px 2px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                min-height: 22px;
                background: #ffffff;
                color: #111827;
                border: 1px solid #d3dae3;
                border-radius: 6px;
                padding: 1px 6px;
            }
            QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
                color: #9ca3af;
                background: #f5f7fa;
                border-color: #e4e9ef;
            }
            QLineEdit:read-only {
                background: #f3f6f9;
                color: #4b5563;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QTextEdit {
                padding: 3px;
                background: #ffffff;
                color: #111827;
                border: 1px solid #d3dae3;
                border-radius: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #e3e8ef;
                border-radius: 8px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #eef2f6;
                border: 1px solid #d7dde6;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 4px 10px;
                margin-right: 2px;
                color: #4b5563;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #111827;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #d3dae3;
                border-radius: 6px;
                gridline-color: #e7ecf2;
                alternate-background-color: #f8fafc;
            }
            QHeaderView::section {
                background: #f3f6f9;
                color: #374151;
                border: 1px solid #e2e8f0;
                padding: 4px;
            }
            QLabel#statusLabel {
                color: #2563eb;
                font-weight: 600;
                padding: 1px 2px;
            }
            QLabel#columnTokenLabel {
                color: #6b7280;
                font-weight: 600;
                padding: 0px 2px;
            }
            QLabel#paramInline {
                color: #374151;
                padding: 0px 2px 0px 0px;
            }
            QLabel#paramHeader {
                color: #6b7280;
                font-size: 11px;
                font-weight: 600;
                padding: 0px 2px;
            }
            QDoubleSpinBox#paramBoundBox {
                background: #ffffff;
                color: #0f172a;
                border-color: #94a3b8;
            }
            QDoubleSpinBox#paramValueBox {
                background: #ffffff;
                color: #0f172a;
                border-color: #94a3b8;
                font-weight: 600;
            }
            QSlider::groove:horizontal {
                border: none;
                background: #dbe3ec;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6b7280;
                border: 1px solid #4b5563;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
            QToolBar {
                background: #ffffff;
                border: 1px solid #e3e8ef;
                border-radius: 6px;
                spacing: 2px;
            }
            QToolButton {
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 2px;
            }
            QToolButton:hover {
                background: #f3f6f9;
                border-color: #d3dae3;
            }
            QCheckBox { spacing: 4px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 2px solid #9ca3af;
                border-radius: 3px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #2563eb;
                border-color: #2563eb;
            }
            QCheckBox::indicator:hover {
                border-color: #6b7280;
            }
            QCheckBox::indicator:disabled {
                background: #f5f7fa;
                border-color: #d3dae3;
            }
            QCheckBox::indicator:checked:disabled {
                background: #93c5fd;
                border-color: #93c5fd;
            }
            QScrollArea {
                border: 1px solid #e3e8ef;
                border-radius: 8px;
                background: #f8fafc;
            }
            QSplitter::handle {
                background: #e2e8f0;
            }
            QSplitter::handle:horizontal {
                width: 6px;
                margin: 0 2px;
                border-radius: 2px;
            }
            QSplitter::handle:vertical {
                height: 6px;
                margin: 2px 0;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background: #cbd5e1;
            }
            """
        )

    def _new_label(
        self,
        text="",
        *,
        object_name=None,
        tooltip=None,
        width=None,
        alignment=None,
        style_sheet=None,
        word_wrap=None,
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
        if word_wrap is not None:
            label.setWordWrap(bool(word_wrap))
        return label

    def _new_button(
        self,
        text,
        *,
        handler=None,
        toggled_handler=None,
        checkable=False,
        checked=None,
        tooltip=None,
        enabled=None,
        fixed_width=None,
        min_height=None,
        object_name=None,
        primary=False,
        style_sheet=None,
    ):
        button = QPushButton(str(text))
        if object_name:
            button.setObjectName(str(object_name))
        if checkable:
            button.setCheckable(True)
        if checked is not None:
            button.setChecked(bool(checked))
        if tooltip:
            button.setToolTip(str(tooltip))
        if enabled is not None:
            button.setEnabled(bool(enabled))
        if fixed_width is not None:
            button.setFixedWidth(int(fixed_width))
        if min_height is not None:
            button.setMinimumHeight(int(min_height))
        if primary:
            button.setProperty("primary", True)
        if style_sheet is not None:
            button.setStyleSheet(str(style_sheet))
        if callable(handler):
            button.clicked.connect(handler)
        if callable(toggled_handler):
            button.toggled.connect(toggled_handler)
        return button

    def _new_checkbox(
        self,
        text,
        *,
        checked=False,
        tooltip=None,
        enabled=None,
        toggled_handler=None,
    ):
        checkbox = QCheckBox(str(text))
        checkbox.setChecked(bool(checked))
        if tooltip:
            checkbox.setToolTip(str(tooltip))
        if enabled is not None:
            checkbox.setEnabled(bool(enabled))
        if callable(toggled_handler):
            checkbox.toggled.connect(toggled_handler)
        return checkbox

    def _new_combobox(
        self,
        *,
        items=None,
        current_text=None,
        current_data=None,
        tooltip=None,
        enabled=None,
        minimum_width=None,
        fixed_width=None,
        current_index_changed=None,
        rich_text=False,
    ):
        combo: RichTextComboBox | QComboBox = (
            RichTextComboBox() if rich_text else QComboBox()
        )
        if items:
            for item in items:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    combo.addItem(str(item[0]), item[1])
                else:
                    combo.addItem(str(item))
        if tooltip:
            combo.setToolTip(str(tooltip))
        if enabled is not None:
            combo.setEnabled(bool(enabled))
        if minimum_width is not None:
            combo.setMinimumWidth(int(minimum_width))
        if fixed_width is not None:
            combo.setFixedWidth(int(fixed_width))
        if current_text is not None:
            combo.setCurrentText(str(current_text))
        elif current_data is not None:
            index: int = combo.findData(current_data)
            if index >= 0:
                combo.setCurrentIndex(index)
        if callable(current_index_changed):
            combo.currentIndexChanged.connect(current_index_changed)
        return combo

    def _new_line_edit(
        self,
        text="",
        *,
        placeholder=None,
        tooltip=None,
        enabled=None,
        read_only=None,
        object_name=None,
        fixed_width=None,
        text_changed=None,
    ):
        line_edit = QLineEdit(str(text))
        if placeholder:
            line_edit.setPlaceholderText(str(placeholder))
        if tooltip:
            line_edit.setToolTip(str(tooltip))
        if enabled is not None:
            line_edit.setEnabled(bool(enabled))
        if read_only is not None:
            line_edit.setReadOnly(bool(read_only))
        if object_name:
            line_edit.setObjectName(str(object_name))
        if fixed_width is not None:
            line_edit.setFixedWidth(int(fixed_width))
        if callable(text_changed):
            line_edit.textChanged.connect(text_changed)
        return line_edit

    def _make_compact_tool_button(self, text, tooltip, handler):
        """Build a fixed-width toolbar-like button for file navigation."""
        return self._new_button(
            text,
            tooltip=tooltip,
            handler=handler,
            fixed_width=30,
        )

    @staticmethod
    def _set_split_action_min_width(button, labels, *, extra_px=38) -> None:
        """Set a minimum width so split-button text is not cropped by the arrow area."""
        if button is None:
            return
        metrics: QFontMetrics = button.fontMetrics()
        widest: int = 0
        for label in list(labels or ()):
            text: str = str(label or "").strip()
            if not text:
                continue
            widest = max(widest, int(metrics.horizontalAdvance(text)))
        if widest <= 0:
            return
        button.setMinimumWidth(int(widest + max(24, int(extra_px))))

    def _refresh_source_path_label(self):
        if not hasattr(self, "source_path_label"):
            return
        source_text: str = (
            str(self._source_display_override).strip()
            if self._source_display_override
            else str(self.current_dir).strip()
        ) or "."
        prefix = "📁 "
        max_width: int = max(220, int(self.width() * 0.4))
        self.source_path_label.setMaximumWidth(max_width)
        metrics: QFontMetrics = self.source_path_label.fontMetrics()
        available_for_path: int = max(24, max_width - metrics.horizontalAdvance(prefix))
        elided_path: str = metrics.elidedText(
            source_text,
            Qt.TextElideMode.ElideLeft,
            available_for_path,
        )
        self.source_path_label.setText(f"{prefix}{elided_path}")
        tooltip: List[str] = [f"Current data source:\n{source_text}"]
        selected_paths: List[Any] = list(
            getattr(self, "_source_selected_paths", []) or []
        )
        if selected_paths:
            preview: str = "\n".join(
                stem_for_file_ref(path) for path in selected_paths[:12]
            )
            remaining: int = len(selected_paths) - 12
            if remaining > 0:
                preview += f"\n... +{remaining} more"
            tooltip.append(f"Selected files ({len(selected_paths)}):\n{preview}")
        tooltip.append(
            "Click to choose a folder, an archive, or one/more CSV files."
        )
        self.source_path_label.setToolTip("\n\n".join(tooltip))

    def _sync_file_navigation_buttons(self):
        total: int = len(getattr(self, "data_files", []) or [])
        current_idx = int(getattr(self, "current_file_idx", 0))
        at_start: bool = current_idx <= 0
        at_end: bool = current_idx >= max(0, total - 1)
        busy = bool(getattr(self, "_file_load_in_progress", False))

        if hasattr(self, "prev_file_btn"):
            self.prev_file_btn.setEnabled((total > 0) and (not at_start) and (not busy))
        if hasattr(self, "next_file_btn"):
            self.next_file_btn.setEnabled((total > 0) and (not at_end) and (not busy))

    def _current_loaded_file_path(self):
        files: List[Any] = list(getattr(self, "data_files", []) or [])
        idx = int(getattr(self, "current_file_idx", -1))
        if 0 <= idx < len(files):
            return files[idx]
        return None

    def _fit_task_file_key(self, file_path):
        text: str = str(file_path or "").strip()
        if not text:
            return ""
        if "::" in text:
            archive_text, member = text.split("::", 1)
            archive: Path = Path(archive_text).expanduser()
            try:
                archive_key = str(archive.resolve(strict=False))
            except Exception:
                archive_key = str(archive)
            member_key: str = str(member).strip().replace("\\", "/")
            return f"{archive_key}::{member_key}"
        path_obj: Path = Path(text).expanduser()
        try:
            return str(path_obj.resolve(strict=False))
        except Exception:
            return str(path_obj)

    def _next_fit_task_id(self):
        self._fit_task_counter: int = int(getattr(self, "_fit_task_counter", 0)) + 1
        return int(self._fit_task_counter)

    def _active_fit_tasks_for_file(self, file_path):
        target: str = self._fit_task_file_key(file_path)
        if not target:
            return []
        return [
            meta
            for meta in self.fit_tasks.values()
            if str(
                meta.get("file_key") or self._fit_task_file_key(meta.get("file_path"))
            )
            == target
        ]

    def _is_file_fit_active(self, file_path):
        return bool(self._active_fit_tasks_for_file(file_path))

    def _refresh_fit_action_buttons(self):
        current_file: Any | None = self._current_loaded_file_path()
        active_tasks = self._active_fit_tasks_for_file(current_file)
        manual_tasks = [
            meta for meta in active_tasks if str(meta.get("kind")) == "manual"
        ]

        piecewise_running = bool(manual_tasks)
        any_running = bool(active_tasks)

        if hasattr(self, "auto_fit_btn"):
            self.auto_fit_btn.setEnabled((not any_running) or piecewise_running)
            self.auto_fit_btn.setText(
                "Cancel" if piecewise_running else self.auto_fit_btn_default_text
            )
        self._set_auto_fit_mode_selector_enabled(not any_running)
        if hasattr(self, "reset_from_batch_btn"):
            self.reset_from_batch_btn.setEnabled(not any_running)
        if hasattr(self, "clear_previous_result_btn"):
            self.clear_previous_result_btn.setEnabled(not any_running)
        if hasattr(self, "fit_compute_mode_btn"):
            self.fit_compute_mode_btn.setEnabled(not bool(self.fit_tasks))

    def _refresh_batch_controls(self):
        if not hasattr(self, "run_batch_btn") or not hasattr(self, "cancel_batch_btn"):
            return
        if not bool(self.batch_fit_in_progress):
            self.run_batch_btn.setEnabled(True)
            self.run_batch_btn.setText(self.run_batch_btn_default_text)
            self.cancel_batch_btn.setEnabled(False)
            self.cancel_batch_btn.setText("Cancel")
            self._set_batch_fit_mode_selector_enabled(True)
            return
        total: int = max(0, int(getattr(self, "_batch_total_tasks", 0)))
        done: int = max(0, int(getattr(self, "_batch_progress_done", 0)))
        running_batch = any(
            str(task.get("kind")) == "batch" and str(task.get("status")) == "running"
            for task in self.fit_tasks.values()
        )
        display_done: int = min(
            total,
            done + (1 if running_batch and done < total else 0),
        )
        eta_text: str = self._progress_eta_text(
            done=done,
            total=total,
            started_at=float(getattr(self, "_batch_progress_started_at", 0.0) or 0.0),
        )
        self.run_batch_btn.setEnabled(False)
        if self._current_batch_fit_run_mode() == "procedure":
            self.run_batch_btn.setText(f"{display_done}/{total} ({eta_text})")
        else:
            self.run_batch_btn.setText(
                f"{self.run_batch_btn_default_text} ({done}/{total})"
            )
        self.cancel_batch_btn.setEnabled(True)
        self.cancel_batch_btn.setText(
            "Force Stop"
            if bool(getattr(self, "_batch_cancel_pending", False))
            else "Cancel"
        )
        self._set_batch_fit_mode_selector_enabled(False)

    @staticmethod
    def _format_time_left(seconds: float) -> str:
        if not np.isfinite(seconds) or float(seconds) <= 0.0:
            return "0s left"
        total_seconds: int = max(1, int(round(float(seconds))))
        if total_seconds < 60:
            return f"{total_seconds}s left"
        total_minutes = int(round(total_seconds / 60.0))
        if total_minutes < 60:
            return f"{total_minutes}m left"
        hours, minutes = divmod(total_minutes, 60)
        if hours < 24:
            if minutes == 0:
                return f"{hours}h left"
            return f"{hours}h {minutes}m left"
        days, hours = divmod(hours, 24)
        if hours == 0:
            return f"{days}d left"
        return f"{days}d {hours}h left"

    def _progress_eta_text(self, *, done: int, total: int, started_at: float) -> str:
        total_i: int = max(0, int(total))
        done_i: int = max(0, int(done))
        if total_i <= 0:
            return "estimating..."
        remaining: int = max(0, total_i - done_i)
        if remaining <= 0:
            return "0s left"
        if done_i <= 0 or float(started_at) <= 0.0:
            return "estimating..."
        elapsed = max(0.0, float(time.perf_counter()) - float(started_at))
        if elapsed <= 0.0:
            return "estimating..."
        eta_seconds = elapsed * (float(remaining) / float(done_i))
        return self._format_time_left(float(eta_seconds))

    def _update_batch_procedure_status(
        self,
        *,
        current_task: Optional[Mapping[str, Any]] = None,
        step_done: Optional[int] = None,
        step_total: Optional[int] = None,
    ) -> None:
        if not bool(getattr(self, "batch_fit_in_progress", False)):
            return
        if self._current_batch_fit_run_mode() != "procedure":
            return
        total: int = max(0, int(getattr(self, "_batch_total_tasks", 0)))
        done: int = max(0, int(getattr(self, "_batch_progress_done", 0)))
        running_batch = any(
            str(task.get("kind")) == "batch" and str(task.get("status")) == "running"
            for task in self.fit_tasks.values()
        )
        display_done: int = min(
            total,
            done + (1 if running_batch and done < total else 0),
        )
        eta_text: str = self._progress_eta_text(
            done=done,
            total=total,
            started_at=float(getattr(self, "_batch_progress_started_at", 0.0) or 0.0),
        )
        file_label = ""
        task_ref = current_task or {}
        if task_ref:
            file_label = stem_for_file_ref(task_ref.get("file_path"))
        if not file_label:
            running_tasks = [
                task
                for task in self.fit_tasks.values()
                if str(task.get("kind")) == "batch" and str(task.get("status")) == "running"
            ]
            if running_tasks:
                file_label = stem_for_file_ref(running_tasks[0].get("file_path"))
        detail = ""
        if step_done is not None and step_total is not None and int(step_total) > 0:
            detail = f" step {max(0, int(step_done))}/{max(1, int(step_total))}"
        message = f"Procedures {display_done}/{total} ({eta_text})"
        if file_label:
            message += f" - {file_label}"
        if detail:
            message += detail
        self.stats_text.setText(message)

    def _update_manual_procedure_status(
        self,
        task: Mapping[str, Any],
        *,
        step_done: int,
        step_total: int,
    ) -> None:
        if str(task.get("kind")) != "manual":
            return
        if self._normalize_fit_run_mode(task.get("execution_mode")) != "procedure":
            return
        total_i: int = max(1, int(step_total))
        done_i: int = max(0, min(int(step_done), total_i))
        started_at_raw = task.get("_progress_started_at")
        started_at_val = (
            finite_float_or_none(started_at_raw) if started_at_raw is not None else None
        )
        if started_at_val is None or started_at_val <= 0.0:
            started_at_val = float(time.perf_counter())
            try:
                task["_progress_started_at"] = float(started_at_val)
            except Exception:
                pass
        eta_text: str = self._progress_eta_text(
            done=done_i,
            total=total_i,
            started_at=float(started_at_val),
        )
        file_label: str = stem_for_file_ref(task.get("file_path"))
        self.stats_text.setText(
            f"Procedure {done_i}/{total_i} ({eta_text}) - {file_label}"
        )

    @staticmethod
    def _normalize_fit_run_mode(mode_value) -> str:
        mode: str = str(mode_value or "").strip().lower()
        if mode in {"procedure", "proc"}:
            return "procedure"
        return "fit"

    @staticmethod
    def _normalize_fit_compute_mode(mode_value) -> str:
        mode: str = str(mode_value or "").strip().lower()
        if mode == "cpu":
            return "cpu"
        return "gpu"

    @staticmethod
    def _apply_fit_compute_mode_env(mode_value) -> None:
        mode: str = ManualFitGUI._normalize_fit_compute_mode(mode_value)
        os.environ["REDPITAYA_JAX_PLATFORM"] = mode

    @staticmethod
    def _is_jax_backend_initialized() -> bool:
        module = sys.modules.get("jax_backend")
        return bool(module is not None and getattr(module, "_jax_initialized", False))

    @staticmethod
    def _auto_fit_button_text_for_mode(mode_value) -> str:
        mode: str = ManualFitGUI._normalize_fit_run_mode(mode_value)
        return "Run Procedure" if mode == "procedure" else "Auto Fit"

    @staticmethod
    def _batch_fit_button_text_for_mode(mode_value) -> str:
        mode: str = ManualFitGUI._normalize_fit_run_mode(mode_value)
        return "Run Procedures" if mode == "procedure" else "Auto Fits"

    def _current_auto_fit_run_mode(self) -> str:
        return self._normalize_fit_run_mode(getattr(self, "_auto_fit_run_mode", "fit"))

    def _current_batch_fit_run_mode(self) -> str:
        return self._normalize_fit_run_mode(getattr(self, "_batch_fit_run_mode", "fit"))

    def _current_fit_compute_mode(self) -> str:
        return self._normalize_fit_compute_mode(
            getattr(self, "_fit_compute_mode", "gpu")
        )

    def _set_fit_compute_mode(
        self,
        mode_value,
        *,
        autosave=True,
        show_status=False,
    ) -> None:
        mode: str = self._normalize_fit_compute_mode(mode_value)
        previous_mode: str = self._current_fit_compute_mode()
        changed: bool = mode != previous_mode
        self._fit_compute_mode = mode
        self._apply_fit_compute_mode_env(mode)
        if hasattr(self, "fit_compute_mode_btn"):
            checked = mode == "gpu"
            self.fit_compute_mode_btn.blockSignals(True)
            self.fit_compute_mode_btn.setChecked(checked)
            self.fit_compute_mode_btn.setText("GPU" if checked else "CPU")
            self.fit_compute_mode_btn.blockSignals(False)
        if show_status and changed and hasattr(self, "stats_text"):
            mode_label: str = "GPU" if mode == "gpu" else "CPU"
            if self._is_jax_backend_initialized():
                self.stats_text.append(
                    f"Fit compute mode set to {mode_label}. "
                    "Restart the app to switch backend."
                )
            else:
                self.stats_text.append(f"Fit compute mode set to {mode_label}.")
        if autosave:
            self._autosave_fit_details()

    def _on_fit_compute_mode_toggled(self, checked: bool) -> None:
        self._set_fit_compute_mode(
            "gpu" if bool(checked) else "cpu",
            autosave=True,
            show_status=True,
        )

    @staticmethod
    def _set_mode_actions_enabled(
        actions_by_mode: Mapping[str, QAction],
        enabled: bool,
    ) -> None:
        for action in dict(actions_by_mode or {}).values():
            if action is not None:
                action.setEnabled(bool(enabled))

    def _set_auto_fit_mode_selector_enabled(self, enabled: bool) -> None:
        self._set_mode_actions_enabled(self._auto_fit_mode_actions, bool(enabled))

    def _set_batch_fit_mode_selector_enabled(self, enabled: bool) -> None:
        self._set_mode_actions_enabled(self._batch_fit_mode_actions, bool(enabled))

    def _set_auto_fit_mode(self, mode_value, *, autosave=True) -> None:
        mode: str = self._normalize_fit_run_mode(mode_value)
        self._auto_fit_run_mode = mode
        self.auto_fit_btn_default_text = self._auto_fit_button_text_for_mode(mode)
        if hasattr(self, "auto_fit_btn"):
            current_file: Any | None = self._current_loaded_file_path()
            active_tasks = self._active_fit_tasks_for_file(current_file)
            manual_tasks = [
                meta for meta in active_tasks if str(meta.get("kind")) == "manual"
            ]
            self.auto_fit_btn.setText(
                "Cancel" if bool(manual_tasks) else self.auto_fit_btn_default_text
            )
        if autosave:
            self._autosave_fit_details()

    def _set_batch_fit_mode(self, mode_value, *, autosave=True) -> None:
        mode: str = self._normalize_fit_run_mode(mode_value)
        self._batch_fit_run_mode = mode
        self.run_batch_btn_default_text = self._batch_fit_button_text_for_mode(mode)
        if hasattr(self, "run_batch_btn") and not bool(
            getattr(self, "batch_fit_in_progress", False)
        ):
            self.run_batch_btn.setText(self.run_batch_btn_default_text)
        if autosave:
            self._autosave_fit_details()

    def _current_procedure_for_run(self):
        panel: Any | None = getattr(self, "_procedure_panel", None)
        if panel is None:
            return (None, "Procedure panel is not available.")
        try:
            procedure = panel.build_procedure()
        except Exception as exc:
            return (None, f"Procedure setup error: {exc}")
        if not list(getattr(procedure, "steps", ()) or ()):
            return (None, "No procedure steps defined.")
        return (procedure, None)

    @staticmethod
    def _procedure_capture_field_map(procedure) -> Dict[str, str]:
        """Return field-name identity mapping for capture-driven procedure steps."""
        out: Dict[str, str] = {}
        steps = list(getattr(procedure, "steps", ()) or ())
        for step in steps:
            for pair in tuple(getattr(step, "bound_params", ()) or ()):
                if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                    continue
                field_name: str = str(pair[1] or "").strip()
                if field_name:
                    out[field_name] = field_name
            for assignment in tuple(getattr(step, "assignments", ()) or ()):
                source_kind: str = str(
                    getattr(assignment, "source_kind", "") or ""
                ).strip()
                if source_kind != "capture":
                    continue
                field_name: str = str(
                    getattr(assignment, "source_key", "") or ""
                ).strip()
                if field_name:
                    out[field_name] = field_name
        return out

    def _default_param_values(self, specs):
        defaults = []
        for spec in specs:
            low = float(spec.min_value)
            high = float(spec.max_value)
            if low > high:
                low, high = high, low
            defaults.append(self._param_default_from_limits(low, high))
        return defaults

    @staticmethod
    def _param_decimals_from_limits(
        low, high, *, sig_figs: int = 6, max_decimals: int = 12
    ) -> int:
        low_f = float(low)
        high_f = float(high)
        span = abs(high_f - low_f)
        scale_candidates = [abs(low_f), abs(high_f), span]
        finite_scales = [value for value in scale_candidates if np.isfinite(value)]
        scale = max(finite_scales) if finite_scales else 0.0
        if scale <= 0.0:
            sig_fig_decimals: int = int(sig_figs - 1)
        else:
            order = int(np.floor(np.log10(scale)))
            sig_fig_decimals = int(sig_figs - 1 - order)
        decimals = max(2, sig_fig_decimals)
        return int(min(max_decimals, decimals))

    @staticmethod
    def _param_step_from_limits(low, high, decimals: int) -> float:
        precision_step = 10.0 ** (-int(decimals))
        span = abs(float(high) - float(low))
        if (not np.isfinite(span)) or span <= 0.0:
            return float(precision_step)
        span_order = int(np.floor(np.log10(span)))
        span_step = 10.0 ** (span_order - 4)
        return float(max(precision_step, span_step))

    @staticmethod
    def _param_default_from_limits(low, high) -> float:
        low_f = float(low)
        high_f = float(high)
        if low_f > high_f:
            low_f, high_f = high_f, low_f
        return float(0.5 * (low_f + high_f))

    def _apply_param_spec_defaults_to_controls(self) -> bool:
        defaults = self._default_param_values(self.param_specs)
        self.defaults = list(defaults)
        changed = False
        for idx, spec in enumerate(self.param_specs):
            spinbox = self.param_spinboxes.get(spec.key)
            if spinbox is None:
                continue
            low = float(spinbox.minimum())
            high = float(spinbox.maximum())
            if low > high:
                low, high = high, low
            target = float(np.clip(defaults[idx], low, high))
            if not np.isclose(float(spinbox.value()), target):
                changed = True
                spinbox.blockSignals(True)
                spinbox.setValue(target)
                spinbox.blockSignals(False)
            self._sync_slider_from_spinbox(spec.key)
        return changed

    def _new_compact_int_spinbox(
        self,
        minimum,
        maximum,
        value,
        *,
        single_step=None,
        tooltip=None,
    ) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(int(minimum), int(maximum))
        if single_step is not None:
            spinbox.setSingleStep(int(single_step))
        spinbox.setValue(int(value))
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(False)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        spinbox.setFixedWidth(56)
        spinbox.setProperty("defocus_on_outside_click", True)
        if tooltip:
            spinbox.setToolTip(str(tooltip))
        return spinbox

    def _new_compact_param_spinbox(
        self,
        spec,
        value,
        *,
        minimum=None,
        maximum=None,
        precision_min=None,
        precision_max=None,
        width=72,
        object_name=None,
        tooltip=None,
    ) -> CompactDoubleSpinBox:
        spinbox = CompactDoubleSpinBox()
        if object_name:
            spinbox.setObjectName(str(object_name))
        if minimum is None:
            minimum = float(spec.min_value)
        if maximum is None:
            maximum = float(spec.max_value)
        low = float(min(minimum, maximum))
        high = float(max(minimum, maximum))
        precision_low = float(low if precision_min is None else precision_min)
        precision_high = float(high if precision_max is None else precision_max)
        if precision_low > precision_high:
            precision_low, precision_high = precision_high, precision_low
        decimals = self._param_decimals_from_limits(precision_low, precision_high)
        spinbox.setDecimals(int(decimals))
        spinbox.setRange(low, high)
        spinbox.setSingleStep(
            self._param_step_from_limits(precision_low, precision_high, decimals)
        )
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(False)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        spinbox.setFixedWidth(int(width))
        spinbox.setProperty("defocus_on_outside_click", True)
        if tooltip:
            spinbox.setToolTip(str(tooltip))
        spinbox.setValue(float(np.clip(float(value), low, high)))
        return spinbox

    def _effective_smoothing_window(self) -> int:
        window = int(getattr(self, "smoothing_window", 1))
        if window <= 1:
            return 1
        if window % 2 == 0:
            window += 1
        return window

    def _smooth_channel_values(
        self, values
    ) -> (
        np.ndarray[Tuple[int, ...], np.dtype[Any]]
        | np.ndarray[Tuple[int], np.dtype[Any]]
    ):
        if not bool(getattr(self, "smoothing_enabled", False)):
            return np.asarray(values, dtype=float)
        return smooth_channel_array(values, self._effective_smoothing_window())

    def _rebuild_channel_cache_from_raw(self) -> None:
        rebuilt = {}
        for key, values in (self.raw_channel_cache or {}).items():
            try:
                rebuilt[str(key)] = self._smooth_channel_values(values)
            except Exception:
                continue
        self.channel_cache = rebuilt
        self._expression_channel_data_cache = dict(rebuilt)

    def _sync_smoothing_window_enabled(self) -> None:
        spin: Any | None = getattr(self, "smoothing_window_spin", None)
        if spin is None:
            return
        toggle: Any | None = getattr(self, "smoothing_toggle_btn", None)
        if toggle is None:
            toggle: Any | None = getattr(self, "smoothing_enable_cb", None)
        enabled = bool(toggle and toggle.isChecked())
        spin.setEnabled(enabled)

    def _on_smoothing_controls_changed(self) -> None:
        toggle: Any | None = getattr(self, "smoothing_toggle_btn", None)
        if toggle is None:
            toggle: Any | None = getattr(self, "smoothing_enable_cb", None)
        enabled = bool(toggle and toggle.isChecked())
        self._sync_smoothing_window_enabled()
        window: int = (
            int(self.smoothing_window_spin.value())
            if hasattr(self, "smoothing_window_spin")
            else int(getattr(self, "smoothing_window", 1))
        )
        if window % 2 == 0:
            window += 1
            if hasattr(self, "smoothing_window_spin"):
                self.smoothing_window_spin.blockSignals(True)
                self.smoothing_window_spin.setValue(window)
                self.smoothing_window_spin.blockSignals(False)

        changed: bool = (enabled != self.smoothing_enabled) or (
            window != self.smoothing_window
        )
        self.smoothing_enabled: bool = enabled
        self.smoothing_window: int = window

        if not changed:
            return

        main_xlim = None
        main_ylim = None
        residual_ylim = None
        if hasattr(self, "ax") and self.ax is not None:
            try:
                main_xlim: Tuple[float, ...] = tuple(self.ax.get_xlim())
            except Exception:
                main_xlim = None
            try:
                main_ylim: Tuple[float, ...] = tuple(self.ax.get_ylim())
            except Exception:
                main_ylim = None
        if hasattr(self, "ax_residual") and self.ax_residual is not None:
            try:
                residual_ylim: Tuple[float, ...] = tuple(self.ax_residual.get_ylim())
            except Exception:
                residual_ylim = None

        self._rebuild_channel_cache_from_raw()
        self._sync_breakpoint_sliders_from_state()
        for row in self.batch_results:
            row["plot_full"] = None
            row["plot"] = None
            row["plot_render_size"] = None
        if self.batch_results:
            self.update_batch_table()
            self.queue_visible_thumbnail_render()
        self.update_plot(fast=False, preserve_view=False)
        if hasattr(self, "ax") and self.ax is not None:
            if main_xlim is not None:
                try:
                    self.ax.set_xlim(*main_xlim)
                except Exception:
                    pass
            if main_ylim is not None:
                try:
                    self.ax.set_ylim(*main_ylim)
                except Exception:
                    pass
        if (
            residual_ylim is not None
            and hasattr(self, "ax_residual")
            and self.ax_residual is not None
        ):
            try:
                self.ax_residual.set_ylim(*residual_ylim)
            except Exception:
                pass
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.draw_idle()
        self._autosave_fit_details()

    def _rebuild_equation_toggles(self) -> None:
        """Rebuild per-equation checkboxes in the fit group."""
        layout: Any | None = getattr(self, "equation_toggles_layout", None)
        if layout is None:
            return
        clear_layout(layout)
        self._fit_channel_checkboxes.clear()

        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is None:
            return

        targets: List[Any] = list(multi_model.target_channels)
        if not targets:
            return

        for target in targets:
            cb = QCheckBox(self._channel_display_name(target))
            cb.setToolTip(
                f"Include {self._channel_display_name(target)} equation in fitting"
            )
            enabled = self._fit_channel_enabled.get(target, True)
            cb.setChecked(enabled)
            cb.toggled.connect(
                lambda checked, t=target: self._on_equation_toggle_changed(t, checked)
            )
            layout.addWidget(cb)
            self._fit_channel_checkboxes[target] = cb

        layout.addStretch(1)

    def _on_equation_toggle_changed(self, target, checked) -> None:
        """Handle toggling of a per-equation fit checkbox."""
        self._fit_channel_enabled[target] = checked
        self.update_plot(fast=False)

    def _boundary_topology(self):
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is not None and getattr(multi_model, "channel_models", None):
            topology: Dict[str, int] = {
                str(ch_model.target_col): max(0, len(ch_model.segment_exprs) - 1)
                for ch_model in multi_model.channel_models
            }
            primary_target = str(multi_model.primary.target_col)
            return (topology, primary_target)
        model_def: Any | None = getattr(self, "_piecewise_model", None)
        if model_def is None:
            return ({}, None)
        target = str(model_def.target_col)
        n_boundaries: int = max(0, len(model_def.segment_exprs) - 1)
        return ({target: n_boundaries}, target)

    def _refresh_boundary_state_topology(self, *, preserve_existing=True) -> None:
        state: Any | None = getattr(self, "_fit_state", None)
        if state is None:
            state = BoundaryState()
            self._fit_state: BoundaryState = state
        topology, primary_target = self._boundary_topology()
        state.set_topology(
            topology,
            primary_target=primary_target,
            preserve_existing=bool(preserve_existing),
        )

    def _apply_boundary_links_to_state(
        self,
        *,
        source_boundary=None,
        source_target=None,
        prefer_targets=None,
    ) -> Set[str]:
        self._refresh_boundary_state_topology(preserve_existing=True)
        changed_targets: Set[str] = self._fit_state.apply_link_groups(
            self._boundary_links_from_map(),
            source_boundary=source_boundary,
            source_target=source_target,
            prefer_targets=prefer_targets,
        )
        return changed_targets

    def _valid_boundary_ids(self):
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is not None and multi_model.is_multi_channel:
            return {
                (str(target), int(idx))
                for target, idx in list(multi_model.all_boundary_ids or ())
            }
        model_def: Any | None = getattr(self, "_piecewise_model", None)
        if model_def is None:
            return set()
        n_boundaries: int = max(0, len(model_def.segment_exprs) - 1)
        target = str(model_def.target_col)
        return {(target, int(idx)) for idx in range(n_boundaries)}

    def _prune_fixed_boundary_ids(self) -> None:
        valid = self._valid_boundary_ids()
        current = set(getattr(self, "_manually_fixed_boundary_ids", set()))
        self._manually_fixed_boundary_ids = {bid for bid in current if bid in valid}

    def _boundary_ratio_for_id(self, boundary_id) -> None | float:
        try:
            target, idx = boundary_id
            target = str(target)
            idx = int(idx)
        except Exception:
            return None
        self._refresh_boundary_state_topology(preserve_existing=True)
        return self._fit_state.boundary_ratio(target, idx)

    def _fixed_boundary_maps_for_fit(self):
        self._prune_fixed_boundary_ids()
        fixed_ids = set(getattr(self, "_manually_fixed_boundary_ids", set()))
        if not fixed_ids:
            return ({}, {})

        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is not None and multi_model.is_multi_channel:
            per_channel: Dict[str, Dict[int, float]] = {}
            for boundary_id in sorted(fixed_ids):
                ratio: None | float = self._boundary_ratio_for_id(boundary_id)
                if ratio is None:
                    continue
                target, idx = boundary_id
                per_channel.setdefault(str(target), {})[int(idx)] = float(ratio)
            return ({}, per_channel)

        single_map: Dict[int, float] = {}
        model_def: Any | None = getattr(self, "_piecewise_model", None)
        target: str = str(model_def.target_col) if model_def is not None else ""
        for boundary_id in sorted(fixed_ids):
            ch_target, idx = boundary_id
            if str(ch_target) != target:
                continue
            ratio: None | float = self._boundary_ratio_for_id(boundary_id)
            if ratio is None:
                continue
            single_map[int(idx)] = float(ratio)
        return (single_map, {})

    def _rebuild_boundary_fix_controls(self) -> None:
        layout: Any | None = getattr(self, "boundary_fix_checks_layout", None)
        widget: Any | None = getattr(self, "boundary_fix_widget", None)
        if layout is None or widget is None:
            return

        clear_layout(layout)
        self._boundary_fix_checkboxes = {}
        self._prune_fixed_boundary_ids()

        options = []
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is not None and multi_model.is_multi_channel:
            self._ensure_boundary_names()
            name_map: Dict[Any, Any] = dict(
                getattr(self, "_boundary_name_map", {}) or {}
            )
            groups: Dict[str, List[Tuple[str, int]]] = {}
            for target, idx in list(multi_model.all_boundary_ids or ()):
                bid: Tuple[str, int] = (str(target), int(idx))
                name = str(name_map.get(bid, format_boundary_display_name(int(idx))))
                groups.setdefault(name, []).append(bid)
            for name in sorted(groups.keys()):
                members: Tuple[Tuple[str, int], ...] = tuple(groups[name])
                if members:
                    tooltip: str = ", ".join(
                        f"{self._channel_display_name(t)}[{int(i) + 1}]"
                        for t, i in members
                    )
                    options.append((str(name), members, tooltip))
        else:
            model_def: Any | None = getattr(self, "_piecewise_model", None)
            if model_def is not None:
                target = str(model_def.target_col)
                n_boundaries: int = max(0, len(model_def.segment_exprs) - 1)
                for idx in range(n_boundaries):
                    bid: Tuple[str, int] = (target, int(idx))
                    name: str = format_boundary_display_name(int(idx))
                    options.append(
                        (
                            name,
                            (bid,),
                            f"{self._channel_display_name(target)} boundary {int(idx) + 1}",
                        )
                    )

        if not options:
            widget.setVisible(False)
            return

        fixed_ids = set(getattr(self, "_manually_fixed_boundary_ids", set()))
        for name, members, tooltip in options:
            cb = QCheckBox(str(name))
            cb.setToolTip(
                f"Include {name} in fitting.\n{tooltip}"
                if tooltip
                else f"Include {name} in fitting."
            )
            cb.setChecked(all(member not in fixed_ids for member in members))
            cb.setProperty("_boundary_members", tuple(members))
            cb.toggled.connect(self._on_boundary_fit_toggled)
            layout.addWidget(cb)
            self._boundary_fix_checkboxes[str(name)] = cb
        layout.addStretch(1)
        widget.setVisible(True)

    def _set_boundary_members_fixed(self, members, fixed_enabled: bool) -> None:
        fixed_ids = set(getattr(self, "_manually_fixed_boundary_ids", set()))
        for item in members or ():
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            bid: Tuple[str, int] = (str(item[0]), int(item[1]))
            if fixed_enabled:
                fixed_ids.add(bid)
            else:
                fixed_ids.discard(bid)
        self._manually_fixed_boundary_ids = fixed_ids
        self._autosave_fit_details()

    def _on_boundary_fit_toggled(self, checked) -> None:
        """Handle boundary fit checkbox toggle (checked = fit)."""
        sender: QObject | None = self.sender()
        members: Any | Tuple[np.Never] = (
            sender.property("_boundary_members") if sender is not None else ()
        )
        self._set_boundary_members_fixed(members, not bool(checked))

    def _get_enabled_fit_channels(self):
        """Return list of target channel names that are enabled for fitting."""
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is None:
            primary_target: str = self._primary_target_channel()
            return [primary_target] if primary_target else []
        targets: List[Any] = list(multi_model.target_channels)
        enabled: List[Any] = [
            t for t in targets if self._fit_channel_enabled.get(t, True)
        ]
        return enabled if enabled else targets  # fallback: keep all if none checked

    def _primary_target_channel(self) -> str:
        """Return the current primary fit target channel."""
        state: Any | None = getattr(self, "_fit_state", None)
        if state is not None:
            state_target: str = str(getattr(state, "primary_target", "") or "").strip()
            if state_target:
                return state_target

        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is not None:
            primary_model: Any | None = getattr(multi_model, "primary", None)
            primary_target: str = str(
                getattr(primary_model, "target_col", "") or ""
            ).strip()
            if primary_target:
                return primary_target
            target_channels: List[str] = [
                str(channel).strip()
                for channel in list(getattr(multi_model, "target_channels", []) or [])
            ]
            target_channels = [channel for channel in target_channels if channel]
            if target_channels:
                return target_channels[0]

        model_def: Any | None = getattr(self, "_piecewise_model", None)
        model_target: str = str(getattr(model_def, "target_col", "") or "").strip()
        if model_target:
            return model_target

        available_channels: List[str] = list(self._available_channel_names())
        if not available_channels:
            return ""
        if self.x_channel in available_channels and len(available_channels) > 1:
            for channel in available_channels:
                if channel != self.x_channel:
                    return channel
        return str(available_channels[0]).strip()

    def _sync_fit_panel_top_spacing(self) -> None:
        spacer: Any | None = getattr(self, "fit_panel_top_spacer", None)
        header_widget: Any | None = getattr(self, "param_header_widget", None)
        if spacer is None or header_widget is None:
            return
        header_height = 0
        try:
            header_height: int = max(
                int(header_widget.minimumSizeHint().height()),
                int(header_widget.sizeHint().height()),
            )
        except Exception:
            header_height = 0
        header_height: int = max(0, int(header_height))
        if (
            header_widget.minimumHeight() != header_height
            or header_widget.maximumHeight() != header_height
        ):
            header_widget.setFixedHeight(header_height)
        # Keep right-side controls top-aligned with no extra offset.
        spacer.setFixedHeight(0)
        self._sync_param_pane_height()

    def _sync_param_pane_height(self) -> None:
        scroll: Any | None = getattr(self, "param_controls_scroll", None)
        if scroll is None:
            return
        scroll.setMinimumHeight(0)
        scroll.setMaximumHeight(16777215)

    def _make_param_header_label(self, text, width=None) -> QLabel:
        return self._new_label(
            text,
            object_name="paramHeader",
            width=width,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    def _display_symbol_for_param(self, key, symbol_hint=None) -> str:
        return resolve_parameter_symbol(key, symbol_hint)

    def _display_symbol_for_param_html(self, key, symbol_hint=None) -> str:
        return parameter_symbol_to_html(
            self._display_symbol_for_param(key, symbol_hint)
        )

    def _parameter_symbol_map(self):
        mapping = {}
        for spec in self.param_specs:
            mapping[spec.key] = self._display_symbol_for_param(spec.key, spec.symbol)
        return mapping

    def _display_name_for_param_key(self, key) -> str:
        for spec in self.param_specs:
            if spec.key == key:
                return self._display_symbol_for_param(spec.key, spec.symbol)
        return self._display_symbol_for_param(key, key)

    def _display_name_for_param_key_mathtext(self, key) -> str:
        return parameter_symbol_to_mathtext(self._display_name_for_param_key(key))

    def _ordered_parameter_sections(self):
        ordered_keys: List[str] = self._ordered_param_keys()
        model_def: PiecewiseModelDefinition | None = self._piecewise_model
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)

        # Multi-channel: show sections for each channel, then truly shared at the end.
        if multi_model is not None and multi_model.is_multi_channel:
            sections = []
            seen = set()
            truly_shared = set()
            # Identify parameters that appear in more than one channel.
            param_channel_count = {}
            for ch_model in multi_model.channel_models:
                for seg_names in ch_model.segment_param_names:
                    for name in seg_names:
                        param_channel_count[name] = param_channel_count.get(name, 0) + 1
            for name, count in param_channel_count.items():
                if count > 1:
                    truly_shared.add(name)

            for ch_idx, ch_model in enumerate(multi_model.channel_models):
                target = ch_model.target_col
                seg_param_names: List[Any] = list(ch_model.segment_param_names)
                total_segs: int = len(seg_param_names)
                sections.append(
                    {
                        "kind": "channel_header",
                        "index": ch_idx,
                        "target": target,
                        "keys": [],
                    }
                )
                for seg_idx, seg_names in enumerate(seg_param_names, start=1):
                    unique_keys = []
                    for raw_name in seg_names:
                        key = str(raw_name)
                        if key in seen or key not in ordered_keys:
                            continue
                        # Show shared params in their FIRST occurrence.
                        unique_keys.append(key)
                        seen.add(key)
                    label_idx: int = seg_idx
                    sections.append(
                        {
                            "kind": "segment",
                            "index": label_idx,
                            "keys": unique_keys,
                            "target": target,
                        }
                    )
                    if seg_idx < total_segs:
                        sections.append(
                            {
                                "kind": "boundary",
                                "index": seg_idx,
                                "keys": [],
                                "target": target,
                            }
                        )

            # Catch any remaining keys not yet listed.
            trailing: List[str] = [key for key in ordered_keys if key not in seen]
            if trailing:
                sections.append({"kind": "shared", "index": 0, "keys": trailing})
                for key in trailing:
                    seen.add(key)
            return sections

        if (
            model_def is None
            or not getattr(model_def, "segment_param_names", None)
            or len(model_def.segment_param_names) == 0
        ):
            return [{"kind": "segment", "index": 1, "keys": list(ordered_keys)}]

        sections = []
        seen = set()
        segment_param_names: List[Tuple[str, ...]] = list(model_def.segment_param_names)
        total_segments: int = len(segment_param_names)
        for seg_idx, seg_names in enumerate(segment_param_names, start=1):
            unique_keys = []
            for raw_name in seg_names:
                key = str(raw_name)
                if key in seen or key not in ordered_keys:
                    continue
                unique_keys.append(key)
                seen.add(key)
            sections.append({"kind": "segment", "index": seg_idx, "keys": unique_keys})
            if seg_idx < total_segments:
                sections.append({"kind": "boundary", "index": seg_idx, "keys": []})

        trailing: List[str] = [key for key in ordered_keys if key not in seen]
        if trailing:
            sections.append({"kind": "shared", "index": 0, "keys": trailing})
        return sections

    def _build_param_section_header(self, title, tooltip="") -> QLabel:
        label: QLabel = self._new_label(
            str(title),
            object_name="statusLabel",
            tooltip=tooltip,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            style_sheet="font-weight: 700; color: #0f172a; padding: 2px 0 0 0;",
        )
        return label

    def _build_segment_header_widget(self, seg_idx, expr_text, target=None) -> QLabel:
        """Build a segment header QLabel with syntax-highlighted expression.

        Returns a QLabel showing "Segment N:  <colorized expression>" using
        ``colorize_expression_html`` for rich rendering.
        """
        # -- resolve pretty display text -----------------------------------
        symbol_map = self._parameter_symbol_map()
        display_expr: str = (
            format_expression_pretty(expr_text, name_map=symbol_map)
            if expr_text
            else ""
        )

        # -- gather tokens for colorizer -----------------------------------
        param_tokens: set[str] = set()
        if expr_text:
            try:
                for name in extract_segment_parameter_names(expr_text):
                    param_tokens.add(str(symbol_map.get(name, name)))
            except Exception:
                pass

        column_names = list(self._available_channel_names())
        if target:
            column_names.append(str(target))

        html_symbol_map: Dict[str, str] = {
            tok: parameter_symbol_to_html(tok) or tok for tok in param_tokens
        }

        colorized: str = (
            colorize_expression_html(
                display_expr, column_names, param_tokens, symbol_map=html_symbol_map
            )
            if display_expr
            else ""
        )

        # -- build label ---------------------------------------------------
        prefix: str = (
            f"<span style='font-weight:700; color:#0f172a;'>Segment {seg_idx}</span>"
        )
        if colorized:
            html_text: str = (
                f"{prefix}<span style='color:#64748b;'> :  </span>{colorized}"
            )
        else:
            html_text: str = prefix

        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setText(html_text)
        label.setWordWrap(True)
        label.setStyleSheet("padding: 2px 0 0 0;")
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _build_equation_divider(self) -> QFrame:
        """Build a prominent visual divider between different equations/channels."""
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(
            "QFrame { color: #94a3b8; background-color: #94a3b8; "
            "max-height: 2px; margin-top: 6px; margin-bottom: 2px; }"
        )
        return divider

    def _build_param_boundary_marker(self, boundary_index, target=None) -> QFrame:
        """Build a subtle visual divider between parameter segments."""
        name_map: Any | Dict[Any, Any] = getattr(self, "_boundary_name_map", {})
        bid = (target, boundary_index - 1) if target else (None, boundary_index - 1)
        name = name_map.get(bid, format_boundary_display_name(boundary_index - 1))

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(
            "QFrame { color: #e2e8f0; background-color: #e2e8f0; max-height: 1px; "
            "margin-top: 1px; margin-bottom: 1px; }"
        )
        divider.setToolTip(f"Boundary {name}")
        return divider

    def _build_top_breakpoint_controls_widget(self) -> QWidget:
        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )

        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(4)

        # Header label — uses text-label token.
        boundaries_label: QLabel = self._new_label(
            "Segment Boundaries",
            style_sheet="font-weight: 600; color: #334155; font-size: 11px; padding: 0;",
        )
        outer.addWidget(boundaries_label)

        # Single-channel slider row (used when only one channel).
        min_label: QLabel = self._new_label(
            "Start",
            object_name="paramHeader",
            width=40,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        slider = MultiHandleSlider()
        slider.valuesChanged.connect(self._on_breakpoint_values_changed)
        slider.sliderPressed.connect(self._on_breakpoint_slider_pressed)
        slider.sliderReleased.connect(self._on_breakpoint_slider_released)

        max_label: QLabel = self._new_label(
            "End",
            object_name="paramHeader",
            width=40,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self._single_channel_slider_row = QWidget()
        sr_layout = QHBoxLayout(self._single_channel_slider_row)
        sr_layout.setContentsMargins(0, 0, 0, 0)
        sr_layout.setSpacing(4)
        sr_layout.addWidget(min_label)
        sr_layout.addWidget(slider, 1)
        sr_layout.addWidget(max_label)
        outer.addWidget(self._single_channel_slider_row)

        # Multi-channel slider area: dynamically populated per channel.
        self._multi_channel_slider_panel = QWidget()
        self._multi_channel_slider_panel.setVisible(False)
        self._multi_channel_slider_layout = QVBoxLayout(
            self._multi_channel_slider_panel
        )
        self._multi_channel_slider_layout.setContentsMargins(0, 0, 0, 0)
        self._multi_channel_slider_layout.setSpacing(4)
        outer.addWidget(self._multi_channel_slider_panel)

        # Per-channel slider registry: {target_col: MultiHandleSlider}
        self._per_channel_sliders = {}

        self.breakpoint_controls = {
            "slider": slider,
            "container": container,
        }
        return container

    def _boundary_links_from_map(self):
        """Convert the boundary name map to link groups for the model.

        Boundaries sharing the same assigned name are linked together.
        """
        name_map: Any | Dict[Any, Any] = getattr(self, "_boundary_name_map", {})
        if not name_map:
            return ()
        groups_by_name: dict[str, list[tuple[str, int]]] = {}
        for bid, name in name_map.items():
            if name not in (None, ""):
                groups_by_name.setdefault(str(name), []).append(bid)
        result = []
        for _name, members in sorted(groups_by_name.items()):
            if len(members) >= 2:
                result.append(tuple(sorted(members)))
        return tuple(result)

    def _ensure_boundary_names(self) -> None:
        """Ensure every boundary has an assigned name in _boundary_name_map.

        New boundaries receive globally unique names (X₀, X₁, X₂, …)
        so that nothing is linked by default.  Existing assignments are
        preserved — the user creates links explicitly by assigning the
        same name to two or more boundaries.
        """
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is None or not multi_model.is_multi_channel:
            return
        name_map: Any | Dict[Any, Any] = getattr(self, "_boundary_name_map", {})
        all_bids: List[Any] = list(multi_model.all_boundary_ids)

        # Names already claimed by existing assignments.
        used_names: set[Any] = {name_map[bid] for bid in all_bids if bid in name_map}

        # Assign a unique name to every new boundary.
        next_idx = 0
        for bid in all_bids:
            if bid not in name_map:
                while format_boundary_display_name(next_idx) in used_names:
                    next_idx += 1
                name: str = format_boundary_display_name(next_idx)
                name_map[bid] = name
                used_names.add(name)
                next_idx += 1

        # Prune entries that no longer exist.
        valid: set[Any] = set(all_bids)
        self._boundary_name_map: Dict[Any, Any] = {
            bid: name for bid, name in name_map.items() if bid in valid
        }
        # Links are derived on-the-fly from _boundary_name_map via
        # _boundary_links_from_map(), so no extra sync is needed.

    def _rebuild_boundary_name_panel(self) -> None:
        """Rebuild the interactive boundary-name assignment panel."""
        panel: Any | None = getattr(self, "_boundary_name_panel", None)
        layout: Any | None = getattr(self, "_boundary_name_panel_layout", None)
        if panel is None or layout is None:
            return
        # Clear old widgets.
        clear_layout(layout)
        self._boundary_name_edits.clear()

        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is None or not multi_model.is_multi_channel:
            panel.setVisible(False)
            self._sync_model_settings_sep()
            return

        # Collect channels with boundaries.
        channels_with_boundaries = []
        for ch_model in multi_model.channel_models:
            n_b: int = max(0, len(ch_model.segment_exprs) - 1)
            if n_b > 0:
                channels_with_boundaries.append(ch_model)
        if not channels_with_boundaries:
            panel.setVisible(False)
            self._sync_model_settings_sep()
            return

        self._ensure_boundary_names()
        name_map: Dict[Any, Any] | Dict[Tuple[str, int], str] = self._boundary_name_map

        # Section header — uses text-label token.
        link_header: QLabel = self._new_label(
            "Link Across Channels",
            style_sheet="font-weight: 600; color: #334155; font-size: 11px; padding: 0;",
            tooltip="Assign the same name to boundaries on different channels to link them.\n"
            "Linked boundaries (shown in purple) move together when dragged.",
        )
        layout.addWidget(link_header)

        # Hint line — text-muted token.
        hint: QLabel = self._new_label(
            "Type the same name on two boundaries to link them.",
            style_sheet="color: #64748b; font-size: 10px; padding: 0;",
        )
        layout.addWidget(hint)

        # Build one row per channel: "CH2: [X₀] [X₃]"
        for ch_model in channels_with_boundaries:
            target = ch_model.target_col
            n_b: int = max(0, len(ch_model.segment_exprs) - 1)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)

            ch_label: QLabel = self._new_label(
                self._channel_display_name(target),
                object_name="paramHeader",
                width=56,
                alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            row.addWidget(ch_label)

            for bidx in range(n_b):
                bid = (target, bidx)
                current_name: str | Any = name_map.get(
                    bid, format_boundary_display_name(bidx)
                )
                edit: QLineEdit = self._new_line_edit(
                    current_name,
                    fixed_width=50,
                    tooltip=(
                        f"Boundary name for {self._channel_display_name(target)} boundary {bidx + 1}.\n"
                        "Give two boundaries the same name to link them.\n"
                        "Press Enter or click away to apply."
                    ),
                )
                edit.textEdited.connect(
                    lambda _text, editor=edit: self._auto_subscript_digits(editor)
                )
                edit.editingFinished.connect(
                    lambda t=target, bi=bidx: self._on_boundary_name_changed(t, bi)
                )
                edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
                row.addWidget(edit)
                self._boundary_name_edits[bid] = edit

            row.addStretch(1)
            layout.addLayout(row)

        self._refresh_boundary_link_highlights()

        # Show panel.
        panel.setVisible(True)
        self._sync_model_settings_sep()

    def _refresh_boundary_link_highlights(self) -> None:
        """Colour boundary name edits: purple when linked, default otherwise."""
        links = self._boundary_links_from_map()
        linked_bids = set()
        for group in links:
            linked_bids.update(group)
        for bid, edit in self._boundary_name_edits.items():
            if bid in linked_bids:
                edit.setStyleSheet("QLineEdit { color: #7c3aed; font-weight: 600; }")
            else:
                edit.setStyleSheet("")

    @staticmethod
    def _auto_subscript_digits(edit: QLineEdit) -> None:
        """Replace regular digits with Unicode subscript characters in-place."""
        text: str = edit.text()
        converted: str = text.translate(_UNICODE_SUBSCRIPT_TRANS)
        if converted != text:
            pos: int = edit.cursorPosition()
            edit.blockSignals(True)
            edit.setText(converted)
            edit.setCursorPosition(pos)
            edit.blockSignals(False)

    def _on_boundary_name_changed(self, target: str, boundary_idx: int) -> None:
        """Handle boundary name edit change."""
        bid: Tuple[str, int] = (target, boundary_idx)
        edit = self._boundary_name_edits.get(bid)
        if edit is None:
            return
        # Auto-convert regular digits to Unicode subscript.
        self._auto_subscript_digits(edit)
        new_name = edit.text().strip()
        if not new_name:
            return

        self._boundary_name_map[bid] = str(new_name)
        self._refresh_boundary_link_highlights()

        # Sync ratios: move the edited boundary to match the position of
        # the other boundaries already in its link group.
        self._apply_boundary_links_to_model()

        self._refresh_boundary_state_topology(preserve_existing=True)
        links = self._boundary_links_from_map()
        for group in links:
            if bid in group:
                other_ratios = []
                for gt, gi in group:
                    if (gt, gi) == bid:
                        continue
                    ratio: float | None = self._fit_state.boundary_ratio(gt, gi)
                    if ratio is not None:
                        other_ratios.append(float(ratio))
                if other_ratios:
                    target_ratio = float(np.clip(np.mean(other_ratios), 0.0, 1.0))
                    self._fit_state.set_boundary_ratio(
                        target, boundary_idx, target_ratio
                    )
                break
        self._apply_boundary_links_to_state(source_boundary=bid)
        self._sync_breakpoint_sliders_from_state()
        self._rebuild_boundary_fix_controls()
        self.update_plot()
        self._autosave_fit_details()

    def _apply_boundary_links_to_model(self) -> None:
        """Rebuild the multi-channel model with current boundary link groups."""
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is None or not multi_model.is_multi_channel:
            return
        try:
            channel_equations: List[Tuple[Any, List[Any]]] = [
                (m.target_col, list(m.segment_exprs))
                for m in multi_model.channel_models
            ]
            links = self._boundary_links_from_map()
            self._multi_channel_model: MultiChannelModelDefinition = (
                build_multi_channel_model_definition(
                    channel_equations,
                    channel_names=list(self.channels.keys()),
                    boundary_links=links,
                )
            )
        except Exception:
            pass

    def _current_segment_boundary_count(self) -> int:
        model_def: Any | None = getattr(self, "_piecewise_model", None)
        if model_def is None:
            return 0
        return max(0, len(model_def.segment_exprs) - 1)

    def _boundary_positions_to_ratios(self, positions, n_boundaries):
        n = int(max(0, n_boundaries))
        if n <= 0:
            return np.asarray([], dtype=float)
        pos_arr: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
            positions, dtype=float
        ).reshape(-1)
        if pos_arr.size != n:
            return default_boundary_ratios(n)
        return pcts_to_boundary_ratios(pos_arr)

    def _x_axis_range_for_boundary_controls(self) -> Tuple[float, float]:
        if self.current_data is None:
            return (0.0, 1.0)
        try:
            x_values: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                self._get_channel_data(self.x_channel), dtype=float
            ).reshape(-1)
        except Exception:
            return (0.0, 1.0)
        finite: np.ndarray[Tuple[int, ...], np.dtype[Any]] = x_values[
            np.isfinite(x_values)
        ]
        if finite.size == 0:
            return (0.0, 1.0)
        x_min = float(np.min(finite))
        x_max = float(np.max(finite))
        if np.isclose(x_min, x_max):
            x_max: float = x_min + 1.0
        return (x_min, x_max)

    def _format_compact_number(self, value) -> str:
        numeric = float(value)
        if not np.isfinite(numeric):
            return "n/a"
        magnitude: float = abs(numeric)
        if magnitude >= 1e4 or (magnitude > 0.0 and magnitude < 1e-3):
            return f"{numeric:.3e}"
        return f"{numeric:.6g}"

    def _sync_breakpoint_sliders_from_state(self) -> None:
        self._refresh_boundary_state_topology(preserve_existing=True)
        self._prune_fixed_boundary_ids()
        self._rebuild_boundary_fix_controls()
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        control = (
            self.breakpoint_controls
            if isinstance(self.breakpoint_controls, dict)
            else {}
        )
        slider = control.get("slider")

        # Multi-channel: build per-channel sliders.
        if multi_model is not None and multi_model.is_multi_channel:
            # Hide the single-channel slider row.
            single_row: Any | None = getattr(self, "_single_channel_slider_row", None)
            if single_row is not None:
                single_row.setVisible(False)

            self._rebuild_boundary_name_panel()
            stored = self._fit_state.as_per_channel_map()

            # Build linked boundary lookup for painting linked handles.
            linked_groups = {}  # bid -> set of linked bids
            for group in multi_model.boundary_links:
                group_set: set[Any] = set(group)
                for bid in group:
                    linked_groups[bid] = group_set

            # Build / update per-channel slider panel.
            mc_panel: Any | None = getattr(self, "_multi_channel_slider_panel", None)
            mc_layout: Any | None = getattr(self, "_multi_channel_slider_layout", None)
            per_ch_sliders: Any | Dict[Any, Any] = getattr(
                self, "_per_channel_sliders", {}
            )

            # Determine which channels need sliders (those with boundaries).
            channels_with_boundaries = []
            for ch_model in multi_model.channel_models:
                n_b: int = max(0, len(ch_model.segment_exprs) - 1)
                if n_b > 0:
                    channels_with_boundaries.append(ch_model)

            # Rebuild slider widgets if channel set changed.
            existing_targets: set[Any] = set(per_ch_sliders.keys())
            needed_targets = {ch.target_col for ch in channels_with_boundaries}
            if existing_targets != needed_targets:
                # Clear old.
                if mc_layout is not None:
                    clear_layout(mc_layout)
                per_ch_sliders.clear()

                for ch_model in channels_with_boundaries:
                    target = ch_model.target_col
                    row = QHBoxLayout()
                    row.setContentsMargins(0, 0, 0, 0)
                    row.setSpacing(4)

                    ch_label: QLabel = self._new_label(
                        target,
                        object_name="paramHeader",
                        width=40,
                        alignment=Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter,
                    )
                    row.addWidget(ch_label)

                    ch_slider = MultiHandleSlider()
                    ch_slider.setMinimumHeight(28)
                    ch_slider.setMaximumHeight(28)
                    ch_slider.valuesChanged.connect(
                        lambda pos, t=target: self._on_channel_breakpoint_changed(
                            t, pos
                        )
                    )
                    ch_slider.sliderPressed.connect(self._on_breakpoint_slider_pressed)
                    ch_slider.sliderReleased.connect(
                        self._on_breakpoint_slider_released
                    )
                    row.addWidget(ch_slider, 1)
                    per_ch_sliders[target] = ch_slider

                    if mc_layout is not None:
                        mc_layout.addLayout(row)

                self._per_channel_sliders: Any | Dict[Any, Any] = per_ch_sliders

            # Update slider values and labels.
            for ch_model in channels_with_boundaries:
                target = ch_model.target_col
                ch_slider: Any | None = per_ch_sliders.get(target)
                if ch_slider is None:
                    continue
                n_b: int = max(0, len(ch_model.segment_exprs) - 1)
                ch_ratios = stored.get(target, default_boundary_ratios(n_b))
                positions = boundary_ratios_to_positions(ch_ratios, n_b)

                labels = []
                linked_idx_set = set()
                name_map: Any | Dict[Any, Any] = getattr(self, "_boundary_name_map", {})
                for bidx in range(n_b):
                    bid = (target, bidx)
                    bnd_name = name_map.get(bid, format_boundary_display_name(bidx))
                    labels.append(bnd_name)
                    if bid in linked_groups and len(linked_groups[bid]) >= 2:
                        linked_idx_set.add(bidx)

                ch_slider.blockSignals(True)
                ch_slider.set_values(positions.tolist())
                ch_slider.set_labels(labels)
                ch_slider.set_linked_indices(linked_idx_set)
                ch_slider.blockSignals(False)
                ch_slider.setEnabled(True)

            if mc_panel is not None:
                mc_panel.setVisible(bool(channels_with_boundaries))

            # Not used by per-channel sliders.
            self._boundary_handle_map = None

            if hasattr(self, "formula_label") and not getattr(
                self, "_expression_edit_mode", False
            ):
                self._set_formula_label()
            return
        single_row: Any | None = getattr(self, "_single_channel_slider_row", None)
        if single_row is not None:
            single_row.setVisible(True)
        mc_panel: Any | None = getattr(self, "_multi_channel_slider_panel", None)
        if mc_panel is not None:
            mc_panel.setVisible(False)
        panel: Any | None = getattr(self, "_boundary_name_panel", None)
        if panel is not None:
            panel.setVisible(False)
        sep: Any | None = getattr(self, "_boundary_link_sep", None)
        if sep is not None:
            sep.setVisible(False)
        self._boundary_handle_map = None

        n_boundaries: int = self._current_segment_boundary_count()

        if n_boundaries <= 0:
            if slider is not None:
                slider.blockSignals(True)
                slider.set_values([])
                slider.blockSignals(False)
                slider.setEnabled(False)
            return

        ratios: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
            self._fit_state.primary_ratios(),
            dtype=float,
        ).reshape(-1)
        if ratios.size != n_boundaries:
            ratios = default_boundary_ratios(n_boundaries)
        ratios: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.clip(ratios, 0.0, 1.0)
        positions = boundary_ratios_to_positions(ratios, n_boundaries)
        x_min, x_max = self._x_axis_range_for_boundary_controls()
        axis_label: str = self._channel_axis_label(self.x_channel)
        self._boundary_slider_mapping = None
        if slider is not None:
            slider.blockSignals(True)
            slider.set_values(positions.tolist())
            slider.set_labels(
                [format_boundary_display_name(idx) for idx in range(n_boundaries)]
            )
            slider.blockSignals(False)
            slider.setEnabled(True)
            slider.setToolTip(f"Boundary positions X₀, X₁, ... on {axis_label}.")
        if hasattr(self, "formula_label") and not getattr(
            self, "_expression_edit_mode", False
        ):
            self._set_formula_label()

    def _on_channel_breakpoint_changed(self, target: str, positions) -> None:
        """Handle boundary slider change for a specific channel."""
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is None or not multi_model.is_multi_channel:
            return
        self._refresh_boundary_state_topology(preserve_existing=True)
        target = str(target)
        n_b: int = self._fit_state.channel_count(target)
        if n_b <= 0:
            return
        pos_arr: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
            positions, dtype=float
        ).reshape(-1)
        if pos_arr.size != n_b:
            return
        pos_arr: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.clip(pos_arr, 0.0, 1.0)
        pos_arr: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.maximum.accumulate(
            pos_arr
        )
        new_ratios = self._boundary_positions_to_ratios(pos_arr, n_b)
        self._fit_state.set_channel_ratios(target, new_ratios)
        self._fit_state.apply_link_groups(
            self._boundary_links_from_map(),
            source_target=target,
        )
        stored = self._fit_state.as_per_channel_map()

        # Update other channel sliders that were affected by link propagation.
        per_ch_sliders: Any | Dict[Any, Any] = getattr(self, "_per_channel_sliders", {})
        for ch_model in multi_model.channel_models:
            t = ch_model.target_col
            if t == target:
                continue  # skip the slider being dragged
            ch_slider: Any | None = per_ch_sliders.get(t)
            if ch_slider is None:
                continue
            ch_r = stored.get(t)
            if ch_r is None:
                continue
            ch_n: int = len(ch_r)
            ch_pos = boundary_ratios_to_positions(ch_r, ch_n)
            ch_slider.blockSignals(True)
            ch_slider.set_values(ch_pos.tolist())
            ch_slider.blockSignals(False)

        self.update_plot(fast=True)

    def _on_breakpoint_values_changed(self, positions) -> None:
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)

        # Multi-channel uses per-channel sliders.
        if multi_model is not None and multi_model.is_multi_channel:
            # Per-channel sliders handle this via _on_channel_breakpoint_changed.
            return

        n_boundaries: int = self._current_segment_boundary_count()
        if n_boundaries <= 0:
            return

        self._refresh_boundary_state_topology(preserve_existing=True)
        pos_arr: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
            positions, dtype=float
        ).reshape(-1)
        if pos_arr.size != n_boundaries:
            pos_arr = boundary_ratios_to_positions(
                self._fit_state.primary_ratios(),
                n_boundaries,
            )
        pos_arr: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.clip(pos_arr, 0.0, 1.0)
        pos_arr: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.maximum.accumulate(
            pos_arr
        )
        self._fit_state.set_primary_ratios(
            self._boundary_positions_to_ratios(pos_arr, n_boundaries)
        )
        self._sync_breakpoint_sliders_from_state()
        self.update_plot(fast=True)

    def _on_breakpoint_slider_pressed(self) -> None:
        self.slider_active = True

    def _on_breakpoint_slider_released(self) -> None:
        self.slider_active = False
        self.do_full_update()
        self._autosave_fit_details()

    def _set_param_fixed(self, key: str, fixed_enabled: bool) -> None:
        """Set the fixed/unfixed state for a parameter."""
        fixed_keys = set(getattr(self, "_manually_fixed_params", set()))
        if fixed_enabled:
            fixed_keys.add(key)
        else:
            fixed_keys.discard(key)
        self._manually_fixed_params = fixed_keys
        # Visually dim slider/bounds when fixed.
        slider = self.param_sliders.get(key)
        min_box = self.param_min_spinboxes.get(key)
        max_box = self.param_max_spinboxes.get(key)
        if slider is not None:
            slider.setEnabled(not fixed_enabled)
        if min_box is not None:
            min_box.setEnabled(not fixed_enabled)
        if max_box is not None:
            max_box.setEnabled(not fixed_enabled)
        # Keep current values when toggling fixed/fit state for a single param.
        # Restoring batch-fitted values here resets unrelated controls.
        self._autosave_fit_details()

    def _on_param_fit_toggled(self, key: str, checked: bool) -> None:
        """Handle the per-parameter fit checkbox toggle (checked = fit)."""
        self._set_param_fixed(key, not bool(checked))
        self._sync_param_slider_lock_state()

    def _create_param_label(self, spec, width) -> QLabel:
        """Create a one-line parameter label."""
        symbol_text: str = self._display_symbol_for_param(spec.key, spec.symbol)
        symbol_html: str = self._display_symbol_for_param_html(spec.key, spec.symbol)
        tooltip = str(spec.description)
        if symbol_text != spec.key:
            tooltip: str = f"{tooltip} ({spec.key})"
        label: QLabel = self._new_label(
            f"{symbol_html}:",
            object_name="paramInline",
            tooltip=tooltip,
            width=width,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    def _ordered_param_keys(self) -> List[str]:
        return [spec.key for spec in self.param_specs]

    def _channel_display_name(self, channel_name) -> str:
        key = str(channel_name)
        alias: str = str(self.channels.get(key, "")).strip()
        return alias if alias else key

    def _channel_unit(self, channel_name) -> str:
        key = str(channel_name)
        unit: str = str(getattr(self, "channel_units", {}).get(key, "")).strip()
        return unit

    def _channel_axis_label(self, channel_name) -> str:
        base: str = self._channel_display_name(channel_name)
        unit: str = self._channel_unit(channel_name)
        return f"{base} [{unit}]" if unit else base

    def _channel_legend_label(self, channel_name) -> str:
        key = str(channel_name)
        base: str = self._channel_display_name(key)
        if base and base != key:
            text: str = f"{key} ({base})"
        else:
            text: str = key
        unit: str = self._channel_unit(key)
        return f"{text} [{unit}]" if unit else text

    def get_current_param_map(self):
        values = {}
        for idx, spec in enumerate(self.param_specs):
            if spec.key in self.param_spinboxes:
                values[spec.key] = float(self.param_spinboxes[spec.key].value())
            elif idx < len(self.defaults):
                values[spec.key] = float(self.defaults[idx])
            else:
                values[spec.key] = 0.0
        return values

    def _available_capture_keys(self):
        keys = []
        config: CapturePatternConfig | None = self._resolve_batch_capture_config(
            show_errors=False
        )
        if config is not None and config.regex is not None:
            for key in config.regex.groupindex.keys():
                text: str = str(key).strip()
                if text and text not in keys:
                    keys.append(text)
        return keys

    def _capture_preview_values(self):
        config: CapturePatternConfig | None = self._resolve_batch_capture_config(
            show_errors=False
        )
        if config is None or config.regex is None:
            return {}

        file_path: Any | None = self._current_loaded_file_path()
        if not file_path:
            candidates: List[Any] = list(getattr(self, "batch_files", []) or [])
            if not candidates:
                candidates: List[Any] = list(getattr(self, "data_files", []) or [])
            file_path: Any | None = candidates[0] if candidates else None
        if not file_path:
            return {}

        extracted: Dict[str, str] | None = extract_captures(
            stem_for_file_ref(file_path),
            config.regex,
            config.defaults,
        )
        if extracted is None:
            return {}
        return {str(key): str(value) for key, value in dict(extracted).items()}

    def _on_param_capture_mapping_changed(self, capture_key, _index) -> None:
        combo = self.param_capture_combos.get(capture_key)
        if combo is None:
            return
        selected = combo.currentData()
        self.param_capture_map[str(capture_key)] = (
            str(selected) if selected not in (None, "") else None
        )
        self._mapped_param_seed_file_key = None
        self._refresh_param_capture_mapping_controls()
        self._autosave_fit_details()

    def _parsed_numeric_param_values_from_mapping(self) -> Dict[str, float]:
        preview_values = self._capture_preview_values()
        mapping: Dict[str, None] = self._current_param_capture_map()
        out: Dict[str, float] = {}
        for param_key, capture_key in mapping.items():
            if capture_key in (None, ""):
                continue
            raw_value = preview_values.get(str(capture_key))
            text: str = str(raw_value).strip() if raw_value is not None else ""
            if not text:
                continue
            try:
                numeric = float(text)
            except Exception:
                continue
            if not np.isfinite(numeric):
                continue
            out[str(param_key)] = float(numeric)
        return out

    def _effective_param_capture_map_for_fixing(self) -> Dict[str, None]:
        """Return param->capture mapping used for field seeding only.

        Every bound mapping seeds its parameter.
        Manually fixed parameters are excluded so capture edits do not
        overwrite values while Fit is disabled in the parameter pane.
        """
        mapping: Dict[str, None] = dict(self._current_param_capture_map())
        manually_fixed_keys: set[str] = {
            str(key) for key in set(getattr(self, "_manually_fixed_params", set()))
        }
        for key, capture_key in list(mapping.items()):
            if str(key) in manually_fixed_keys:
                mapping[str(key)] = None
                continue
            if capture_key in (None, ""):
                mapping[str(key)] = None
        return mapping

    def _sync_param_slider_lock_state(
        self, *, allow_seed_for_fixed: bool = False
    ) -> None:
        param_capture_map: Dict[str, None] = self._current_param_capture_map()
        mapped_values: Dict[str, float] = (
            self._parsed_numeric_param_values_from_mapping()
        )
        manually_fixed = getattr(self, "_manually_fixed_params", set())
        has_seed_mapped_fields = any(
            value not in (None, "") for value in param_capture_map.values()
        )
        current_file_key: str = self._fit_task_file_key(
            self._current_loaded_file_path()
        )
        should_seed_mapped_values: Any | bool = (
            has_seed_mapped_fields
            and bool(current_file_key)
            and current_file_key != getattr(self, "_mapped_param_seed_file_key", None)
        )
        any_value_changed = False
        seeded_target_param_present = False
        for key, slider in self.param_sliders.items():
            if slider is None:
                continue
            capture_key: None = param_capture_map.get(str(key))
            is_capture_mapped: bool = capture_key not in (None, "")
            is_capture_seed_enabled: bool = is_capture_mapped
            is_manually_fixed: bool = key in manually_fixed
            min_box = self.param_min_spinboxes.get(key)
            max_box = self.param_max_spinboxes.get(key)
            model_min_box = self._model_param_min_spinboxes.get(key)
            model_max_box = self._model_param_max_spinboxes.get(key)
            value_box = self.param_spinboxes.get(key)
            lock_status_label = self.param_lock_status_labels.get(key)
            tail_spacer = self.param_tail_spacers_by_key.get(key)
            fix_cb = self.param_fix_checkboxes.get(key)

            # Capture mappings are seed-only; keep parameter controls visible.
            for widget in (slider, min_box, max_box, value_box, tail_spacer):
                if widget is not None:
                    widget.setVisible(True)
            if fix_cb is not None:
                fix_cb.setVisible(True)

            if is_capture_seed_enabled and (
                allow_seed_for_fixed or not is_manually_fixed
            ):
                seeded_target_param_present = True

            if (
                should_seed_mapped_values
                and is_capture_seed_enabled
                and (allow_seed_for_fixed or not is_manually_fixed)
                and str(key) in mapped_values
                and value_box is not None
            ):
                seeded_value = float(mapped_values[str(key)])
                low = float(value_box.minimum())
                high = float(value_box.maximum())
                if not np.isclose(float(value_box.value()), seeded_value):
                    value_box.blockSignals(True)
                    value_box.setValue(float(np.clip(seeded_value, low, high)))
                    value_box.blockSignals(False)
                    any_value_changed = True

            if lock_status_label is not None and is_capture_mapped:
                if is_capture_seed_enabled:
                    if is_manually_fixed:
                        lock_status_label.setText(
                            f'Seed from field "{capture_key}" (Fit disabled)'
                        )
                        lock_status_label.setToolTip(
                            f'Field "{capture_key}" is selected for seeding.\n'
                            "Value updates are paused while Fit is disabled for this parameter."
                        )
                    else:
                        lock_status_label.setText(f'Seed from field "{capture_key}"')
                        lock_status_label.setToolTip(
                            f'Field "{capture_key}" seeds this parameter before fitting.'
                        )
                    lock_status_label.show()
            elif lock_status_label is not None:
                lock_status_label.hide()

            # When manually fixed, disable slider and bounds but keep value editable.
            if is_manually_fixed:
                if slider is not None:
                    slider.setEnabled(False)
                if min_box is not None:
                    min_box.setEnabled(False)
                if max_box is not None:
                    max_box.setEnabled(False)
                for model_box in (model_min_box, model_max_box):
                    if model_box is not None:
                        model_box.setEnabled(False)
                if value_box is not None:
                    value_box.setEnabled(True)
                    value_box.setToolTip("Fixed value (check Fit to unlock)")
            else:
                if slider is not None:
                    slider.setEnabled(True)
                    slider.setToolTip("Sweep value across active bounds")
                if min_box is not None:
                    min_box.setEnabled(True)
                    min_box.setToolTip("Lower bound")
                if max_box is not None:
                    max_box.setEnabled(True)
                    max_box.setToolTip("Upper bound")
                for model_box in (model_min_box, model_max_box):
                    if model_box is not None:
                        model_box.setEnabled(True)
                if value_box is not None:
                    value_box.setEnabled(True)
                    value_box.setToolTip("Current value")
            if not bool(getattr(self, "_show_plot_param_bounds", True)):
                if min_box is not None:
                    min_box.setVisible(False)
                if max_box is not None:
                    max_box.setVisible(False)
        if has_seed_mapped_fields and seeded_target_param_present:
            self._mapped_param_seed_file_key = current_file_key
        else:
            self._mapped_param_seed_file_key = None
        if any_value_changed:
            self.update_plot(fast=False)

    def _parameter_display_items(self):
        items = []
        seen = {}
        for param_key in self._ordered_param_keys():
            symbol_token: str = str(self._display_name_for_param_key(param_key)).strip()
            if not symbol_token:
                symbol_token = str(param_key)
            count: int = int(seen.get(symbol_token, 0)) + 1
            seen[symbol_token] = count
            plain_label: str = symbol_token if count == 1 else f"{symbol_token} {count}"
            rich_base: str = parameter_symbol_to_html(symbol_token) or html.escape(
                symbol_token
            )
            rich_label: str = (
                rich_base
                if count == 1
                else f"{rich_base} <span style='color:#64748b;'>{count}</span>"
            )
            items.append(
                {
                    "plain": plain_label,
                    "html": rich_label,
                    "key": str(param_key),
                }
            )
        return items

    def _refresh_param_capture_mapping_controls(
        self, *, allow_seed_for_fixed: bool = False
    ) -> None:
        if not hasattr(self, "capture_mapping_layout"):
            return
        clear_layout(self.capture_mapping_layout)
        self.param_capture_combos = {}

        capture_keys = self._available_capture_keys()
        param_keys: List[str] = self._ordered_param_keys()
        parameter_items = self._parameter_display_items()
        next_map = {}
        if not capture_keys:
            self.capture_mapping_layout.addWidget(
                self._new_label(
                    "No named fields in Pattern.",
                    object_name="statusLabel",
                    style_sheet="color: #64748b; font-style: italic;",
                ),
                0,
                0,
                1,
                3,
            )
            self.param_capture_map = {}
            self._sync_param_slider_lock_state(
                allow_seed_for_fixed=allow_seed_for_fixed
            )
            return

        self.capture_mapping_layout.addWidget(
            self._new_label(
                "Field",
                object_name="statusLabel",
                style_sheet="font-weight: 700; color: #334155;",
            ),
            0,
            0,
        )
        self.capture_mapping_layout.addWidget(
            self._new_label(
                "Parameter",
                object_name="statusLabel",
                style_sheet="font-weight: 700; color: #334155;",
            ),
            0,
            1,
        )
        self.capture_mapping_layout.addWidget(
            self._new_label(
                "Value",
                object_name="statusLabel",
                style_sheet="font-weight: 700; color: #334155;",
            ),
            0,
            2,
        )
        preview_values = self._capture_preview_values()

        for row_idx, capture_key in enumerate(capture_keys, start=1):
            mapped: str | None = self.param_capture_map.get(capture_key)
            if mapped not in param_keys:
                mapped = None
            next_map[capture_key] = mapped

            label: QLabel = self._new_label(
                str(capture_key),
                object_name="paramInline",
                tooltip=f"Filename field '{capture_key}'",
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            combo: RichTextComboBox | QComboBox = self._new_combobox(
                minimum_width=150, rich_text=True
            )
            if isinstance(combo, RichTextComboBox):
                combo.add_rich_item("Unbound", None, "Unbound")
                for item in parameter_items:
                    combo.add_rich_item(item["plain"], item["key"], item["html"])
            else:
                combo.addItem("Unbound", None)
                for item in parameter_items:
                    combo.addItem(str(item["plain"]), item["key"])
            target_idx: int = combo.findData(mapped)
            if target_idx < 0:
                target_idx = 0
            combo.setCurrentIndex(target_idx)
            combo.currentIndexChanged.connect(
                lambda index, key=capture_key: self._on_param_capture_mapping_changed(
                    key, index
                )
            )
            value_text: str = str(preview_values.get(str(capture_key), "")).strip()
            value_label: QLabel = self._new_label(
                value_text if value_text else "—",
                object_name="paramInline",
                tooltip=(
                    f"Parsed value for '{capture_key}' from current file."
                    if value_text
                    else f"No parsed value available for '{capture_key}'."
                ),
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                style_sheet="color: #475569;",
            )
            self.capture_mapping_layout.addWidget(label, row_idx, 0)
            self.capture_mapping_layout.addWidget(combo, row_idx, 1)
            self.capture_mapping_layout.addWidget(value_label, row_idx, 2)
            self.param_capture_combos[capture_key] = combo

        self.param_capture_map = next_map
        self._sync_param_slider_lock_state(allow_seed_for_fixed=allow_seed_for_fixed)

    def _current_param_capture_map(self) -> Dict[str, None]:
        mapping: Dict[str, None] = {key: None for key in self._ordered_param_keys()}
        for capture_key, param_key in self.param_capture_map.items():
            target: str | None = str(param_key) if param_key not in (None, "") else None
            if target in mapping:
                mapping[target] = str(capture_key)
        return mapping

    def _current_file_seed_overrides_from_mapping(
        self,
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        parameter_capture_map: Dict[str, None] = (
            self._effective_param_capture_map_for_fixing()
        )
        if not any(value not in (None, "") for value in parameter_capture_map.values()):
            return {}, None

        capture_config: CapturePatternConfig | None = (
            self._resolve_batch_capture_config(show_errors=True)
        )
        if capture_config is None:
            return None, "Capture pattern is invalid."
        if capture_config.regex is None:
            return (
                None,
                "Pattern is required when field-to-parameter seed mappings are used.",
            )

        file_path: Any | None = self._current_loaded_file_path()
        if not file_path:
            return None, "No current file loaded for field-to-parameter seed mapping."

        extracted: Dict[str, str] | None = extract_captures(
            stem_for_file_ref(file_path),
            capture_config.regex,
            capture_config.defaults,
        )
        if extracted is None:
            return None, _BATCH_PATTERN_MISMATCH_ERROR
        return resolve_fixed_params_from_captures(parameter_capture_map, extracted)

    @staticmethod
    def _channel_sort_key(channel_name) -> tuple[int, str, int, str]:
        text: str = str(channel_name).strip()
        upper: str = text.upper()
        if upper == "TIME":
            return (0, "", -1, upper)
        match = re.fullmatch(r"([A-Z_]+)(\d+)", upper)
        if match is not None:
            return (1, str(match.group(1)), int(match.group(2)), upper)
        return (2, upper, 0, upper)

    def _sorted_channel_names(self, names) -> List[str]:
        unique = []
        seen = set()
        for raw_name in names:
            name: str = str(raw_name).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            unique.append(name)
        return sorted(unique, key=self._channel_sort_key)

    def _numeric_channel_columns(self) -> List[str]:
        if self.current_data is None:
            return []
        names = []
        seen_columns = set()
        for col in self.current_data.columns:
            key: str = str(col).strip()
            if not key or key in seen_columns:
                continue
            try:
                self.current_data[col].to_numpy(dtype=float, copy=False)
            except Exception:
                continue
            seen_columns.add(key)
            names.append(key)
        return self._sorted_channel_names(names)

    def _available_channel_names(self):
        if self.current_data is not None:
            names = [str(col).strip() for col in self.current_data.columns]
            return self._sorted_channel_names(names)
        return self._sorted_channel_names(self.channels.keys())

    def _expression_channel_data(self):
        if self.current_data is None:
            self._expression_channel_data_cache = None
            return {}
        if self._expression_channel_data_cache is not None:
            return self._expression_channel_data_cache

        channels = {}
        for col in self.current_data.columns:
            key: str = str(col).strip()
            if not key:
                continue
            try:
                channels[key] = self._get_channel_data(key)
            except Exception:
                continue
        self._expression_channel_data_cache = channels
        return channels

    def _slice_channel_data(self, channel_data, subset):
        if not channel_data:
            return {}
        sliced = {}
        for key, values in channel_data.items():
            try:
                sliced[key] = np.asarray(values, dtype=float)[subset]
            except Exception:
                continue
        return sliced

    def _parameter_bounds(self):
        lower = []
        upper = []
        for spec in self.param_specs:
            low = float(spec.min_value)
            high = float(spec.max_value)
            if low > high:
                low, high = high, low
            lower.append(low)
            upper.append(high)
        return lower, upper

    def _value_to_slider_position(self, key, value) -> int:
        min_box = self.param_min_spinboxes.get(key)
        max_box = self.param_max_spinboxes.get(key)
        if min_box is None or max_box is None:
            return 0
        low = float(min_box.value())
        high = float(max_box.value())
        if np.isclose(low, high):
            return 0
        ratio: float = (float(value) - low) / (high - low)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return int(round(ratio * self._param_slider_steps))

    def _slider_position_to_value(self, key, slider_position) -> float:
        min_box = self.param_min_spinboxes.get(key)
        max_box = self.param_max_spinboxes.get(key)
        if min_box is None or max_box is None:
            return float(slider_position)
        low = float(min_box.value())
        high = float(max_box.value())
        if np.isclose(low, high):
            return low
        ratio: float = float(slider_position) / float(self._param_slider_steps)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return low + (high - low) * ratio

    def _sync_slider_from_spinbox(self, key) -> None:
        spinbox = self.param_spinboxes.get(key)
        slider = self.param_sliders.get(key)
        if spinbox is None or slider is None:
            return
        slider.blockSignals(True)
        slider.setValue(self._value_to_slider_position(key, spinbox.value()))
        slider.blockSignals(False)

    def _on_param_bounds_changed(self, key, source) -> None:
        min_box = self.param_min_spinboxes.get(key)
        max_box = self.param_max_spinboxes.get(key)
        value_box = self.param_spinboxes.get(key)
        slider = self.param_sliders.get(key)
        if min_box is None or max_box is None or value_box is None or slider is None:
            return

        low = float(min_box.value())
        high = float(max_box.value())
        if low > high:
            if source == "min":
                max_box.blockSignals(True)
                max_box.setValue(low)
                max_box.blockSignals(False)
                high: float = low
            else:
                min_box.blockSignals(True)
                min_box.setValue(high)
                min_box.blockSignals(False)
                low: float = high

        updated_specs = []
        updated_decimals = self._param_decimals_from_limits(low, high)
        for spec in self.param_specs:
            if spec.key != key:
                updated_specs.append(spec)
                continue
            midpoint_default = self._param_default_from_limits(low, high)
            updated_specs.append(
                ParameterSpec(
                    key=spec.key,
                    symbol=spec.symbol,
                    description=spec.description,
                    default=midpoint_default,
                    min_value=float(low),
                    max_value=float(high),
                    decimals=int(updated_decimals),
                )
            )
        self.param_specs = updated_specs
        self.defaults = self._default_param_values(self.param_specs)

        updated_step = self._param_step_from_limits(low, high, updated_decimals)
        for box in (min_box, max_box, value_box):
            box.blockSignals(True)
            box.setDecimals(int(updated_decimals))
            box.setSingleStep(float(updated_step))
            box.blockSignals(False)

        value_box.blockSignals(True)
        value_box.setMinimum(low)
        value_box.setMaximum(high)
        value_box.setValue(float(np.clip(value_box.value(), low, high)))
        value_box.blockSignals(False)
        self._sync_slider_from_spinbox(key)
        self._sync_model_param_limits_from_primary(key)
        self.update_plot(fast=False)
        self._autosave_fit_details()

    def _build_fit_context(
        self,
        seed_overrides=None,
        fixed_params=None,
        include_current_target=False,
        respect_enabled_channels=True,
    ):
        base_model_def: PiecewiseModelDefinition | None = self._piecewise_model
        if base_model_def is None:
            raise ValueError("No compiled piecewise model is available.")
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        # Keep row param order aligned with the full model/UI controls.
        if multi_model is not None and getattr(multi_model, "global_param_names", None):
            ordered_keys: List[Any] = list(multi_model.global_param_names)
        else:
            ordered_keys: List[str] = list(base_model_def.global_param_names)
        if not ordered_keys:
            raise ValueError("No parameters are available for fitting.")

        # Filter multi-channel model to enabled channels.
        fit_multi_model: Any | None = multi_model
        if (
            multi_model is not None
            and multi_model.is_multi_channel
            and bool(respect_enabled_channels)
        ):
            enabled_channels = self._get_enabled_fit_channels()
            current_target: str = self._primary_target_channel()
            if (
                bool(include_current_target)
                and current_target
                and current_target in set(multi_model.target_channels)
                and current_target not in enabled_channels
            ):
                enabled_channels = [current_target, *enabled_channels]
                fit_debug(
                    "fit-context include-current-target: "
                    f"added={current_target} "
                    f"enabled={','.join(enabled_channels)}"
                )
            enabled_ch_models: List[Any] = [
                m
                for m in multi_model.channel_models
                if m.target_col in enabled_channels
            ]
            if enabled_ch_models and len(enabled_ch_models) < len(
                multi_model.channel_models
            ):
                enabled_targets: set[Any] = {m.target_col for m in enabled_ch_models}
                filtered_links = []
                for group in multi_model.boundary_links:
                    filtered: Tuple[Any, ...] = tuple(
                        bid for bid in group if bid[0] in enabled_targets
                    )
                    if len(filtered) >= 2:
                        filtered_links.append(filtered)
                filtered_global = []
                filtered_seen = set()
                for m in enabled_ch_models:
                    for name in m.global_param_names:
                        if name not in filtered_seen:
                            filtered_seen.add(name)
                            filtered_global.append(name)
                fit_multi_model = MultiChannelModelDefinition(
                    channel_models=tuple(enabled_ch_models),
                    global_param_names=tuple(filtered_global),
                    boundary_links=tuple(filtered_links),
                )
            elif not enabled_ch_models:
                fit_multi_model = multi_model

        # Decide which model is used for the actual numerical fit.
        active_model_def: PiecewiseModelDefinition = base_model_def
        active_multi_model: MultiChannelModelDefinition | Any | None = fit_multi_model
        single_channel_target = None
        if fit_multi_model is not None and len(fit_multi_model.channel_models) == 1:
            active_model_def: PiecewiseModelDefinition | Any = (
                fit_multi_model.channel_models[0]
            )
            active_multi_model = None
            single_channel_target = str(active_model_def.target_col)

        if active_multi_model is not None and active_multi_model.is_multi_channel:
            fit_param_keys: List[str] = list(active_multi_model.global_param_names)
        else:
            fit_param_keys: List[str] = list(active_model_def.global_param_names)
        fit_param_key_set: set[str] = set(fit_param_keys)

        fixed_map: Dict[str, float] = {
            str(key): float(value)
            for key, value in dict(fixed_params or {}).items()
            if str(key).strip()
        }

        # Merge manually fixed parameters from UI checkboxes.
        manually_fixed = getattr(self, "_manually_fixed_params", set())
        current_values_raw = self.get_current_param_map()
        current_values: Dict[str, float] = {}
        for raw_key, raw_value in dict(current_values_raw or {}).items():
            key_text: str = str(raw_key).strip()
            if not key_text:
                continue
            numeric: float | None = finite_float_or_none(raw_value)
            if numeric is None:
                continue
            current_values[key_text] = float(numeric)
        for key in manually_fixed:
            key_text: str = str(key).strip()
            if key_text and key_text not in fixed_map and key_text in current_values:
                try:
                    fixed_map[key_text] = float(current_values[key_text])
                except (TypeError, ValueError):
                    pass
        spec_by_key: Dict[str, ParameterSpec] = {
            str(spec.key): spec for spec in self.param_specs
        }
        periodic_selected: set[str] = {
            str(key).strip()
            for key in set(getattr(self, "_periodic_param_keys", set()) or set())
            if str(key).strip()
        }
        periodic_by_key: Dict[str, bool] = {
            key: True
            for key in periodic_selected
            if key in fit_param_key_set and key in spec_by_key
        }
        bounds_by_key = {}
        seed_map = {}
        missing_keys = []

        def _coerce_to_param_bounds(key: str, value: float) -> float:
            if key not in bounds_by_key:
                return float(value)
            low_raw, high_raw = bounds_by_key[key]
            low = float(min(low_raw, high_raw))
            high = float(max(low_raw, high_raw))
            numeric = float(value)
            if (
                bool(periodic_by_key.get(str(key)))
                and np.isfinite(low)
                and np.isfinite(high)
                and (high > low)
            ):
                return float(low + np.mod(numeric - low, high - low))
            return float(np.clip(numeric, low, high))

        for key in ordered_keys:
            key_text: str = str(key)
            spec: ParameterSpec | None = spec_by_key.get(key_text)
            if spec is None:
                missing_keys.append(key_text)
                continue
            low = float(spec.min_value)
            high = float(spec.max_value)
            if low > high:
                low, high = high, low
            bounds_by_key[key_text] = (low, high)

            if key_text in current_values:
                seed = float(current_values[key_text])
            else:
                spec: ParameterSpec | None = spec_by_key.get(key_text)
                if spec is None:
                    missing_keys.append(key_text)
                    continue
                seed = self._param_default_from_limits(spec.min_value, spec.max_value)
            seed_map[key_text] = _coerce_to_param_bounds(key_text, float(seed))
        if missing_keys:
            missing_text: str = ", ".join(dict.fromkeys(missing_keys))
            raise ValueError(
                f"Model/UI parameter mismatch. Missing controls for: {missing_text}"
            )
        for key, value in list(fixed_map.items()):
            if key in bounds_by_key:
                fixed_map[key] = _coerce_to_param_bounds(key, float(value))
            if key in seed_map:
                seed_map[key] = float(fixed_map[key])
        if seed_overrides:
            for key, value in seed_overrides.items():
                if key not in seed_map or key in fixed_map:
                    continue
                seed_map[key] = _coerce_to_param_bounds(str(key), float(value))

        for key in fit_param_keys:
            if key in fixed_map:
                continue
            if key not in bounds_by_key:
                raise ValueError(f"Missing bounds for fit parameter '{key}'.")
            low, high = bounds_by_key[key]
            if np.isclose(low, high):
                raise ValueError(
                    f"Bounds for '{key}' are equal; expand them before fitting."
                )
            if bool(periodic_by_key.get(key)):
                if not (np.isfinite(low) and np.isfinite(high)):
                    raise ValueError(
                        f"Periodic parameter '{key}' requires finite bounds."
                    )

        self._refresh_boundary_state_topology(preserve_existing=True)
        # Procedure workers consume per-channel boundary seeds for all fit modes.
        boundary_seeds_per_channel: Dict[str, np.ndarray] = {}
        if active_multi_model is not None and active_multi_model.is_multi_channel:
            for ch_model in active_multi_model.channel_models:
                ch_target: str | Any = ch_model.target_col
                ch_n_b: int = max(0, len(ch_model.segment_exprs) - 1)
                ch_seed: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                    self._fit_state.channel_ratios(ch_target), dtype=float
                ).reshape(-1)
                if ch_seed.size != ch_n_b:
                    ch_seed = default_boundary_ratios(ch_n_b)
                boundary_seeds_per_channel[ch_target] = ch_seed
        else:
            seed_target: str = str(
                single_channel_target
                if single_channel_target is not None
                else getattr(active_model_def, "target_col", "")
            ).strip()
            if seed_target:
                seed_count: int = max(0, len(active_model_def.segment_exprs) - 1)
                seed_values: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                    self._fit_state.channel_ratios(seed_target),
                    dtype=float,
                ).reshape(-1)
                if seed_values.size != seed_count:
                    seed_values = default_boundary_ratios(seed_count)
                boundary_seeds_per_channel[seed_target] = seed_values

        fixed_boundary_ratios, fixed_boundary_ratios_by_channel = (
            self._fixed_boundary_maps_for_fit()
        )
        fixed_boundary_by_channel_all: Dict[str, Dict[int, float]] = dict(
            fixed_boundary_ratios_by_channel or {}
        )
        if active_multi_model is not None and active_multi_model.is_multi_channel:
            enabled_targets: set[str] = {
                str(ch_model.target_col)
                for ch_model in active_multi_model.channel_models
            }
            fixed_boundary_ratios_by_channel: Dict[str, Dict[int, float]] = {
                str(target): {
                    int(idx): float(value) for idx, value in dict(ch_map or {}).items()
                }
                for target, ch_map in fixed_boundary_by_channel_all.items()
                if str(target) in enabled_targets
            }
            fixed_boundary_ratios = {}
        else:
            fixed_boundary_ratios_by_channel = {}
            if single_channel_target is not None:
                single_map: Dict[int, float] = fixed_boundary_by_channel_all.get(
                    single_channel_target, {}
                )
                fixed_boundary_ratios: Dict[int, float] = {
                    int(idx): float(value)
                    for idx, value in dict(single_map or {}).items()
                }

        fit_fixed_map: Dict[str, float] = {
            key: float(value)
            for key, value in fixed_map.items()
            if key in fit_param_key_set
        }
        fit_channel_targets = []
        if active_multi_model is not None and active_multi_model.is_multi_channel:
            fit_channel_targets: List[str] = [
                str(ch.target_col) for ch in active_multi_model.channel_models
            ]
        fit_debug(
            "fit-context: "
            f"ordered_keys={len(ordered_keys)} "
            f"fit_model_target={active_model_def.target_col} "
            "fit_multi_channels="
            f"{len(active_multi_model.channel_models) if active_multi_model is not None else 0} "
            f"fit_targets=[{','.join(fit_channel_targets) if fit_channel_targets else '-'}] "
            f"fit_param_keys={len(fit_param_keys)} "
            f"fixed_params={len(fit_fixed_map)}"
        )

        return {
            "ordered_keys": ordered_keys,
            "seed_map": seed_map,
            "bounds_map": bounds_by_key,
            "periodic_params": periodic_by_key,
            "model_def": active_model_def,
            "multi_channel_model": active_multi_model,
            "boundary_seeds_per_channel": boundary_seeds_per_channel,
            "fixed_params": fit_fixed_map,
            "fixed_boundary_ratios": fixed_boundary_ratios,
            "fixed_boundary_ratios_by_channel": fixed_boundary_ratios_by_channel,
        }

    @staticmethod
    def _segment_model_from_evaluator(
        evaluator: Callable[[np.ndarray, Mapping[str, float]], np.ndarray],
        ordered_names: Sequence[str],
    ) -> Callable[..., np.ndarray]:
        names: Tuple[str, ...] = tuple(str(name) for name in ordered_names)

        def _model(x_data, *params) -> np.ndarray[Tuple[int, ...], np.dtype[Any]]:
            if len(params) != len(names):
                raise ValueError(
                    f"Expected {len(names)} parameters, got {len(params)}."
                )
            x_arr: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                x_data, dtype=float
            ).reshape(-1)
            values: Dict[str, float] = {
                name: float(params[idx]) for idx, name in enumerate(names)
            }
            return np.asarray(evaluator(x_arr, values), dtype=float).reshape(-1)

        return _model

    def _make_plot_segment_specs(
        self,
        model_def: PiecewiseModelDefinition,
        seed_map: Mapping[str, float],
        bounds_map: Mapping[str, Tuple[float, float]],
    ) -> Tuple[SegmentSpec, ...]:
        """Build non-JAX segment specs for plotting/evaluation only."""
        specs: List[SegmentSpec] = []
        for seg_names, evaluator in zip(
            model_def.segment_param_names, model_def.segment_evaluators
        ):
            names: Tuple[str, ...] = tuple(str(name) for name in seg_names)
            lower: List[float] = []
            upper: List[float] = []
            p0: List[float] = []
            for name in names:
                if name not in bounds_map:
                    raise ValueError(f"Missing bounds for fit parameter '{name}'.")
                low_raw, high_raw = bounds_map[name]
                low = float(min(low_raw, high_raw))
                high = float(max(low_raw, high_raw))
                if name not in seed_map:
                    raise ValueError(f"Missing parameter '{name}'.")
                value = float(seed_map[name])
                if np.isfinite(low) and np.isfinite(high):
                    value = float(np.clip(value, low, high))
                lower.append(low)
                upper.append(high)
                p0.append(value)
            specs.append(
                SegmentSpec(
                    model_func=self._segment_model_from_evaluator(evaluator, names),
                    p0=p0,
                    bounds=(lower, upper),
                )
            )
        return tuple(specs)

    def evaluate_model_map(
        self,
        x_data,
        param_values,
        channel_data=None,
        boundary_ratios=None,
    ) -> np.ndarray[Tuple[int, ...], np.dtype[Any]]:
        _ = channel_data
        model_def: PiecewiseModelDefinition | None = self._piecewise_model
        boundary_target: str | None = None
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if (
            model_def is not None
            and multi_model is not None
            and multi_model.is_multi_channel
        ):
            selected_target: str = self._primary_target_channel()
            for ch_model in multi_model.channel_models:
                if str(ch_model.target_col) == selected_target:
                    model_def = ch_model
                    boundary_target = selected_target
                    break
        if model_def is None:
            raise ValueError("No compiled piecewise model is available.")
        spec_by_key: Dict[str, ParameterSpec] = {
            spec.key: spec for spec in self.param_specs
        }
        bounds_map: Dict[str, Tuple[float, float]] = {
            key: (
                float(min(spec.min_value, spec.max_value)),
                float(max(spec.min_value, spec.max_value)),
            )
            for key, spec in spec_by_key.items()
        }
        missing_bounds: List[str] = [
            key for key in model_def.global_param_names if key not in bounds_map
        ]
        if missing_bounds:
            missing_text: str = ", ".join(missing_bounds)
            raise ValueError(
                f"Model/UI parameter mismatch. Missing bounds for: {missing_text}"
            )
        seed_map = {}
        missing_keys = []
        current_values = self.get_current_param_map()
        for key in model_def.global_param_names:
            if key in param_values:
                seed_map[key] = float(param_values[key])
                continue
            if key in current_values:
                seed_map[key] = float(current_values[key])
                continue
            missing_keys.append(key)
        if missing_keys:
            missing_text: str = ", ".join(missing_keys)
            raise ValueError(
                f"Model/UI parameter mismatch. Missing parameter values: {missing_text}"
            )
        segments: Tuple[SegmentSpec, ...] = self._make_plot_segment_specs(
            model_def, seed_map, bounds_map
        )
        shared: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
            [seed_map[key] for key in model_def.global_param_names], dtype=float
        )
        n_boundaries: int = max(0, len(segments) - 1)
        if boundary_ratios is None:
            if boundary_target:
                b: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                    self._fit_state.channel_ratios(boundary_target), dtype=float
                ).reshape(-1)
            else:
                b: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                    self._fit_state.primary_ratios(), dtype=float
                ).reshape(-1)
        else:
            b: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                boundary_ratios, dtype=float
            )
        if b.size != n_boundaries:
            b = default_boundary_ratios(n_boundaries)
        local_flat = shared_to_local_flat(model_def, shared, np.clip(b, 0.0, 1.0))
        pred = predict_ordered_piecewise(
            np.asarray(x_data, dtype=float).reshape(-1),
            segments,
            local_flat,
            prefer_jit=True,
        )
        return np.asarray(pred["y_hat"], dtype=float)

    def evaluate_channel_model(
        self, channel_model, x_data, param_values, boundary_ratios=None
    ) -> np.ndarray[Tuple[int, ...], np.dtype[Any]]:
        """Evaluate a specific channel's piecewise model with shared parameters."""
        spec_by_key: Dict[str, ParameterSpec] = {
            spec.key: spec for spec in self.param_specs
        }
        bounds_map: Dict[str, Tuple[float, float]] = {
            key: (
                float(min(spec.min_value, spec.max_value)),
                float(max(spec.min_value, spec.max_value)),
            )
            for key, spec in spec_by_key.items()
        }
        seed_map = {}
        current_values = self.get_current_param_map()
        for key in channel_model.global_param_names:
            if key in param_values:
                seed_map[key] = float(param_values[key])
            elif key in current_values:
                seed_map[key] = float(current_values[key])
            else:
                raise ValueError(f"Missing parameter '{key}'.")
        segments: Tuple[SegmentSpec, ...] = self._make_plot_segment_specs(
            channel_model, seed_map, bounds_map
        )
        shared: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
            [seed_map[key] for key in channel_model.global_param_names], dtype=float
        )
        n_boundaries: int = max(0, len(segments) - 1)
        if boundary_ratios is None:
            b: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                self._fit_state.channel_ratios(channel_model.target_col), dtype=float
            ).reshape(-1)
        else:
            b: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                boundary_ratios, dtype=float
            )
        if b.size != n_boundaries:
            b = default_boundary_ratios(n_boundaries)
        local_flat = shared_to_local_flat(channel_model, shared, np.clip(b, 0.0, 1.0))
        pred = predict_ordered_piecewise(
            np.asarray(x_data, dtype=float).reshape(-1),
            segments,
            local_flat,
            prefer_jit=True,
        )
        return np.asarray(pred["y_hat"], dtype=float)

    def evaluate_model(
        self, x_data, params, channel_data=None, boundary_ratios=None
    ) -> np.ndarray[Tuple[int, ...], np.dtype[Any]]:
        """Evaluate active piecewise model from ordered list or key-value map."""
        ordered_keys: List[str] = self._ordered_param_keys()
        if isinstance(params, dict):
            values: Dict[str, float] = {
                key: float(params[key]) for key in ordered_keys if key in params
            }
            if len(values) != len(ordered_keys):
                missing: List[str] = [key for key in ordered_keys if key not in values]
                raise ValueError(f"Missing model parameters: {', '.join(missing)}")
            return self.evaluate_model_map(
                x_data,
                values,
                channel_data=channel_data,
                boundary_ratios=boundary_ratios,
            )
        if len(params) != len(ordered_keys):
            raise ValueError(
                f"Expected {len(ordered_keys)} parameters, got {len(params)}."
            )
        values: Dict[str, float] = {
            key: float(params[idx]) for idx, key in enumerate(ordered_keys)
        }
        return self.evaluate_model_map(
            x_data,
            values,
            channel_data=channel_data,
            boundary_ratios=boundary_ratios,
        )

    def _snapshot_full_model_function(self):
        ordered_keys: List[str] = list(self._ordered_param_keys())

        def model_func(
            x_data, *params, column_data=None, boundary_ratios=None
        ) -> np.ndarray[Tuple[int, ...], np.dtype[Any]]:
            if len(params) != len(ordered_keys):
                raise ValueError(
                    f"Expected {len(ordered_keys)} parameters, got {len(params)}."
                )
            values: Dict[str, float] = {
                key: float(params[idx]) for idx, key in enumerate(ordered_keys)
            }
            return self.evaluate_model_map(
                x_data,
                values,
                channel_data=column_data,
                boundary_ratios=boundary_ratios,
            )

        return model_func

    def _breakpoint_value_map(self, n_boundaries):
        n = int(max(0, n_boundaries))
        if n <= 0:
            return {}
        self._refresh_boundary_state_topology(preserve_existing=True)
        ratios: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
            self._fit_state.primary_ratios(),
            dtype=float,
        ).reshape(-1)
        if ratios.size != n:
            ratios = default_boundary_ratios(n)
        ratios: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.clip(ratios, 0.0, 1.0)
        positions = boundary_ratios_to_positions(ratios, n)
        x_min, x_max = self._x_axis_range_for_boundary_controls()
        span = float(x_max - x_min)
        values = x_min + span * positions
        return {f"break{idx + 1}": float(val) for idx, val in enumerate(values)}

    def _piecewise_boundary_conditions(
        self, segment_count, include_break_values=False, target=None
    ):
        n_segments = int(max(0, segment_count))
        if n_segments <= 0:
            return []
        if n_segments == 1:
            return ["all x"]
        n_boundaries: int = max(0, n_segments - 1)
        value_map = (
            self._breakpoint_value_map(n_boundaries) if include_break_values else {}
        )
        name_map: Any | Dict[Any, Any] = getattr(self, "_boundary_name_map", {})

        def break_display_name(name) -> str:
            text = str(name)
            match: re.Match[str] | None = re.fullmatch(r"break(\d+)", text)
            if match is None:
                return text
            try:
                index: int = int(match.group(1)) - 1
            except Exception:
                index = 0
            # Use assigned name from _boundary_name_map if target is given.
            if target is not None:
                bid = (target, max(0, index))
                assigned: Any | None = name_map.get(bid)
                if assigned:
                    return str(assigned)
            return format_boundary_display_name(max(0, index))

        def break_token(name) -> str:
            display_name: str = break_display_name(name)
            if name not in value_map:
                return display_name
            return f"{display_name} ({self._format_compact_number(value_map[name])})"

        conditions = []
        for seg_idx in range(1, n_segments + 1):
            if seg_idx == 1:
                conditions.append(f"x < {break_token('break1')}")
            elif seg_idx == n_segments:
                last_break: str = f"break{n_boundaries}"
                conditions.append(f"x >= {break_token(last_break)}")
            else:
                left_break: str = f"break{seg_idx - 1}"
                right_break: str = f"break{seg_idx}"
                conditions.append(
                    f"{break_token(left_break)} <= x < {break_token(right_break)}"
                )
        return conditions

    def _piecewise_left_brace_rows(self, row_count):
        rows = int(max(1, row_count))
        if rows == 1:
            return ["{"]
        if rows == 2:
            return ["⎧", "⎩"]
        mid_idx: int = rows // 2
        out = []
        for idx in range(rows):
            if idx == 0:
                out.append("⎧")
            elif idx == rows - 1:
                out.append("⎩")
            elif idx == mid_idx:
                out.append("⎨")
            else:
                out.append("⎪")
        return out

    def _build_multi_channel_formula_html(self, channel_equations) -> str:
        """Build a single unified table for all channels so domain columns align."""
        symbol_map = self._parameter_symbol_map()
        brace_cell_style = (
            "padding:0 4px 1px 2px; font-family:serif; font-size:18px; "
            "font-weight:700; line-height:1.0; color:#111827;"
        )
        pipe_cell_style = (
            "padding:0 4px 1px 8px; font-family:serif; font-size:18px; "
            "font-weight:700; line-height:1.0; color:#111827;"
        )
        all_rows = []
        for ch_idx, (ch_target, ch_seg_exprs) in enumerate(channel_equations):
            conditions = self._piecewise_boundary_conditions(
                len(ch_seg_exprs), target=ch_target
            )
            brace_rows = self._piecewise_left_brace_rows(len(ch_seg_exprs))
            n_segs: int = len(ch_seg_exprs)
            # Add a small gap row between channels.
            if ch_idx > 0:
                all_rows.append(
                    "<tr><td colspan='5' style='padding:4px 0 0 0;'></td></tr>"
                )
            for seg_i, (expr_text, cond_text) in enumerate(
                zip(ch_seg_exprs, conditions)
            ):
                pretty_expr: str = format_expression_pretty(
                    expr_text, name_map=symbol_map
                )
                colored_expr: str = self._colorize_formula_text_html(
                    pretty_expr,
                    target_col=ch_target,
                    rhs_expression=expr_text,
                )
                # First row of each channel shows the "target(x) =" label.
                if seg_i == 0:
                    label_cell: str = (
                        f"<td rowspan='{n_segs}' style='font-family:serif; font-size:15px; "
                        "color:#111827; white-space:nowrap; vertical-align:middle; "
                        f"padding:0 2px 0 0;'>{html.escape(str(ch_target))}(x) =</td>"
                    )
                    brace_inner: str = "".join(
                        f"<tr><td style='{brace_cell_style}'>{html.escape(str(b))}</td></tr>"
                        for b in brace_rows
                    )
                    brace_cell: str = (
                        f"<td rowspan='{n_segs}' style='vertical-align:middle;'>"
                        f"<table style='border-collapse:collapse;'>{brace_inner}</table></td>"
                    )
                else:
                    label_cell: str = ""
                    brace_cell: str = ""

                all_rows.append(
                    "<tr>"
                    f"{label_cell}"
                    f"{brace_cell}"
                    f"<td style='padding:0 0 1px 0;'>{colored_expr}</td>"
                    f"<td style='{pipe_cell_style}'>|</td>"
                    "<td style='padding:0 0 1px 0; color:#334155; white-space:nowrap;'>"
                    f"{html.escape(str(cond_text))}</td>"
                    "</tr>"
                )
        rows_html: str = "".join(all_rows)
        return (
            "<table style='margin:0 auto; border-collapse:collapse;'>"
            f"{rows_html}"
            "</table>"
        )

    def _build_piecewise_formula_html(self, target_col, segment_exprs) -> str:
        symbol_map = self._parameter_symbol_map()
        conditions = self._piecewise_boundary_conditions(len(segment_exprs))
        brace_rows = self._piecewise_left_brace_rows(len(segment_exprs))
        brace_cell_style = (
            "padding:0 4px 1px 2px; font-family:serif; font-size:18px; "
            "font-weight:700; line-height:1.0; color:#111827;"
        )
        pipe_cell_style = (
            "padding:0 4px 1px 8px; font-family:serif; font-size:18px; "
            "font-weight:700; line-height:1.0; color:#111827;"
        )
        rows = []
        brace_cells = []
        for expr_text, cond_text in zip(segment_exprs, conditions):
            pretty_expr: str = format_expression_pretty(expr_text, name_map=symbol_map)
            colored_expr: str = self._colorize_formula_text_html(
                pretty_expr,
                target_col=target_col,
                rhs_expression=expr_text,
            )
            rows.append(
                "<tr>"
                f"<td style='padding:0 0 1px 0;'>{colored_expr}</td>"
                f"<td style='{pipe_cell_style}'>|</td>"
                f"<td style='padding:0 0 1px 0; color:#334155;'>{html.escape(str(cond_text))}</td>"
                "</tr>"
            )
        for brace_char in brace_rows:
            brace_cells.append(
                "<tr>"
                f"<td style='{brace_cell_style}'>{html.escape(str(brace_char))}</td>"
                "</tr>"
            )
        rows_html: str = "".join(rows)
        brace_html: str = "".join(brace_cells)
        return (
            "<table style='margin:0 auto; border-collapse:collapse;'>"
            "<tr>"
            f"<td style='font-family:serif; font-size:15px; color:#111827; white-space:nowrap; vertical-align:middle; padding:0 2px 0 0;'>{html.escape(str(target_col))}(x) =</td>"
            "<td style='vertical-align:middle;'>"
            "<table style='border-collapse:collapse;'>"
            f"{brace_html}"
            "</table>"
            "</td>"
            "<td style='vertical-align:middle;'>"
            "<table style='border-collapse:collapse;'>"
            f"{rows_html}"
            "</table>"
            "</td>"
            "</tr>"
            "</table>"
        )

    def _set_formula_label(self) -> None:
        """Populate the formula label from the active expression."""
        # Multi-channel: show stacked formula tables for each channel.
        multi: Any | None = getattr(self, "_multi_channel_model", None)
        if multi is not None and multi.is_multi_channel:
            try:
                channel_equations = self._parse_multi_equation_text(
                    self.current_expression,
                    strict=False,
                )
            except Exception:
                channel_equations = []
            if channel_equations:
                combined_html: str = self._build_multi_channel_formula_html(
                    channel_equations
                )
                tooltip_parts = []
                for ch_target, ch_seg_exprs in channel_equations:
                    boundary_help: str = "\n".join(
                        f"  Segment {idx}: {cond}"
                        for idx, cond in enumerate(
                            self._piecewise_boundary_conditions(
                                len(ch_seg_exprs), target=ch_target
                            ),
                            start=1,
                        )
                    )
                    tooltip_parts.append(
                        f"{ch_target} = {' ; '.join(ch_seg_exprs)}\n{boundary_help}"
                    )
                self.formula_label.setTextFormat(Qt.TextFormat.RichText)
                self.formula_label.setText(combined_html)
                self.formula_label.setToolTip(
                    "\n\n".join(tooltip_parts) + "\n\nClick equation to edit."
                )
                return

        target_col = None
        rhs_expression = None
        segment_exprs = None
        try:
            target_col, seg_exprs = self._parse_equation_text(
                self.current_expression,
                strict=False,
            )
            segment_exprs: List[str] = list(seg_exprs)
            rhs_expression: str = " ; ".join(seg_exprs)
        except Exception:
            target_col = None
            rhs_expression = None

        self.formula_label.setTextFormat(Qt.TextFormat.RichText)
        if target_col is not None and segment_exprs is not None:
            self.formula_label.setText(
                self._build_piecewise_formula_html(target_col, segment_exprs)
            )
            boundary_help: str = "\n".join(
                f"Segment {idx}: {cond}"
                for idx, cond in enumerate(
                    self._piecewise_boundary_conditions(len(segment_exprs)),
                    start=1,
                )
            )
            display_text: str = f"{target_col} = {' ; '.join(segment_exprs)}"
            self.formula_label.setToolTip(
                f"Python: {self.current_expression}\nDisplay: {display_text}\n\n"
                f"{boundary_help}\n\n"
                "Click equation to edit."
            )
            return

        pretty_equation: str = format_equation_pretty(
            self.current_expression,
            name_map=self._parameter_symbol_map(),
        )
        display_text: str = (
            pretty_equation if pretty_equation else self.current_expression
        )
        colored_text: str = self._colorize_formula_text_html(
            display_text,
            target_col=target_col,
            rhs_expression=rhs_expression,
        )
        self.formula_label.setText(
            '<span style="font-family:serif; font-size:15px; '
            'white-space: normal;">'
            f"{colored_text}</span>"
        )
        self.formula_label.setToolTip(
            f"Python: {self.current_expression}\nDisplay: {display_text}\n\n"
            "Click equation to edit."
        )

    def _colorize_formula_text_html(
        self, display_text, target_col=None, rhs_expression=None
    ) -> str:
        text: str = str(display_text).strip()
        if not text:
            return ""

        symbol_map = self._parameter_symbol_map()
        param_tokens = set()
        if rhs_expression:
            try:
                _target, seg_exprs = self._parse_equation_text(
                    f"{target_col or self._primary_target_channel()} = {rhs_expression}",
                    strict=True,
                )
                for seg_expr in seg_exprs:
                    for name in extract_segment_parameter_names(seg_expr):
                        param_tokens.add(str(symbol_map.get(name, name)))
            except Exception:
                param_tokens = set()

        column_names = list(self._available_channel_names())
        if target_col:
            column_names.append(str(target_col))

        html_symbol_map = {
            token: parameter_symbol_to_html(token) or token for token in param_tokens
        }
        return colorize_expression_html(
            text, column_names, param_tokens, symbol_map=html_symbol_map
        )

    def _is_expression_editor_child(self, widget) -> bool:
        if widget is None or not hasattr(self, "expression_editor_widget"):
            return False
        current = widget
        while current is not None:
            if current is self.expression_editor_widget:
                return True
            current = current.parentWidget()
        return False

    def _set_expression_edit_mode(self, enabled) -> None:
        self._expression_edit_mode = bool(enabled)
        if hasattr(self, "formula_label"):
            self.formula_label.setVisible(not self._expression_edit_mode)
        if hasattr(self, "expression_editor_widget"):
            self.expression_editor_widget.setVisible(self._expression_edit_mode)
        if hasattr(self, "formula_label"):
            self.formula_label.setCursor(
                Qt.CursorShape.IBeamCursor
                if self._expression_edit_mode
                else Qt.CursorShape.PointingHandCursor
            )

    def _enter_expression_edit_mode(self) -> None:
        if self._expression_edit_mode:
            return
        self._set_expression_editor_text(self.current_expression)
        self._refresh_expression_highlighting()
        self._set_function_status("", is_error=False)
        self._set_expression_edit_mode(True)
        if hasattr(self, "function_input"):
            self.function_input.setFocus()
            self.function_input.selectAll()

    def _on_expression_input_focus_left(self) -> None:
        if not self._expression_edit_mode:
            return
        QTimer.singleShot(0, self._apply_expression_on_focus_leave)

    def _on_expression_input_apply_requested(self) -> None:
        if not self._expression_edit_mode:
            return
        self._apply_expression_on_focus_leave(force=True)

    def _apply_expression_on_focus_leave(self, force=False) -> None:
        if not self._expression_edit_mode:
            return
        if QApplication.activePopupWidget() is not None:
            return
        active_modal: QWidget | None = QApplication.activeModalWidget()
        if active_modal is not None and active_modal is not self:
            return
        if not force:
            focus_widget: QWidget | None = QApplication.focusWidget()
            if self._is_expression_editor_child(focus_widget):
                return
        if self.apply_expression_from_input():
            self._set_expression_edit_mode(False)

    def create_plot_frame(self, parent_layout) -> None:
        """Create plot section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.ax: Axes = self.fig.add_subplot(111)
        self.ax_residual = None
        self.canvas: FigureCanvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.16)

        # Add matplotlib navigation toolbar
        self.toolbar: NavigationToolbar[FigureCanvas] = NavigationToolbar(
            self.canvas, self
        )
        self.toolbar.setIconSize(QSize(14, 14))
        self.toolbar.setMaximumHeight(28)
        toolbar_spacer = QWidget()
        toolbar_spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.toolbar.addWidget(toolbar_spacer)

        # Container widget for channel-visibility toggles, embedded in the toolbar.
        self.plot_channel_toggle_container = QWidget()
        self.plot_channel_toggle_container.setMaximumHeight(24)
        self.plot_channel_toggles_layout = QHBoxLayout(
            self.plot_channel_toggle_container
        )
        self.plot_channel_toggles_layout.setContentsMargins(4, 0, 0, 0)
        self.plot_channel_toggles_layout.setSpacing(6)
        self.plot_channel_toggles_buttons_widget = QWidget()
        self.plot_channel_toggles_buttons_layout = QHBoxLayout(
            self.plot_channel_toggles_buttons_widget
        )
        self.plot_channel_toggles_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_channel_toggles_buttons_layout.setSpacing(3)
        self.plot_channel_toggles_layout.addWidget(
            self.plot_channel_toggles_buttons_widget
        )
        self.plot_channel_toggles_layout.addStretch(1)
        self.plot_toolbar_status_widget = QWidget()
        self.plot_toolbar_status_layout = QHBoxLayout(self.plot_toolbar_status_widget)
        self.plot_toolbar_status_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_toolbar_status_layout.setSpacing(6)
        self.tab_r2_label: QLabel = self._new_label(
            "R²: N/A",
            object_name="statusLabel",
            style_sheet="font-weight: 600; color: #334155; padding: 0px 2px;",
        )
        self.plot_toolbar_status_layout.addWidget(self.tab_r2_label)
        self.plot_channel_toggles_layout.addWidget(self.plot_toolbar_status_widget)
        self.plot_channel_toggle_container.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        self.plot_channel_toggle_container.setVisible(True)
        self._toolbar_toggle_action = self.toolbar.addWidget(
            self.plot_channel_toggle_container
        )
        self.toolbar.setStyleSheet(
            self.toolbar.styleSheet() + " QToolBar { padding-right: 0px; }"
            if self.toolbar.styleSheet()
            else "QToolBar { padding-right: 0px; }"
        )

        layout.addWidget(self.toolbar)

        layout.addWidget(self.canvas)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_model_tab(self, parent_layout) -> None:
        """Create the Model tab with equation editor and segment overview."""
        group = QGroupBox("")
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(6)

        # Section label.
        group_layout.addWidget(
            self._new_label(
                "Equation",
                style_sheet="font-weight: 700; color: #0f172a; padding: 1px 2px;",
            )
        )

        # Equation editor (formula display + text editor).
        equation_host: Any | None = getattr(self, "_equation_host_widget", None)
        if equation_host is not None:
            equation_host.setMaximumWidth(16777215)  # remove old 760px cap
            equation_host.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
            )
            group_layout.addWidget(equation_host)

        # Separator before settings panels.
        self._boundary_link_sep = QWidget()
        self._boundary_link_sep.setFixedHeight(1)
        self._boundary_link_sep.setStyleSheet("background: #e5e7eb;")
        self._boundary_link_sep.setVisible(False)
        group_layout.addWidget(self._boundary_link_sep)

        # Horizontal container for boundary linking + channel names panels.
        self._model_settings_row = QWidget()
        self._model_settings_row_layout = QHBoxLayout(self._model_settings_row)
        self._model_settings_row_layout.setContentsMargins(0, 6, 0, 0)
        self._model_settings_row_layout.setSpacing(12)

        # Boundary linking panel (for multi-channel: assign same name to link).
        self._boundary_name_panel = QWidget()
        self._boundary_name_panel.setVisible(False)
        self._boundary_name_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        self._boundary_name_panel_layout = QVBoxLayout(self._boundary_name_panel)
        self._boundary_name_panel_layout.setContentsMargins(0, 0, 0, 0)
        self._boundary_name_panel_layout.setSpacing(6)
        self._model_settings_row_layout.addWidget(
            self._boundary_name_panel, 0, Qt.AlignmentFlag.AlignTop
        )

        # Channel names / units inline editing panel.
        self._channel_names_panel = QWidget()
        self._channel_names_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        self._channel_names_panel_layout = QVBoxLayout(self._channel_names_panel)
        self._channel_names_panel_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_names_panel_layout.setSpacing(4)
        self._channel_name_edits = {}  # channel_key -> (name_edit, unit_edit)
        self._model_settings_row_layout.addWidget(
            self._channel_names_panel, 0, Qt.AlignmentFlag.AlignTop
        )

        # Parameter limits panel (keep it compact in the same settings row).
        self._model_param_limits_panel = QWidget()
        self._model_param_limits_panel_layout = QVBoxLayout(
            self._model_param_limits_panel
        )
        self._model_param_limits_panel_layout.setContentsMargins(0, 0, 0, 0)
        self._model_param_limits_panel_layout.setSpacing(4)
        self._model_param_limits_panel.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        self._model_settings_row_layout.addWidget(
            self._model_param_limits_panel, 0, Qt.AlignmentFlag.AlignTop
        )

        self._model_settings_row_layout.addStretch(1)
        group_layout.addWidget(self._model_settings_row)

        self._rebuild_model_param_limits_panel()

        group_layout.addStretch()
        group.setLayout(group_layout)
        parent_layout.addWidget(group, 1)

    def _rebuild_model_segment_info(self) -> None:
        """Refresh the boundary-linking panel in the Model tab."""
        self._rebuild_boundary_name_panel()
        self._rebuild_channel_names_panel()
        self._rebuild_model_param_limits_panel()

    def _sync_model_settings_sep(self) -> None:
        """Show the separator above the settings row when any settings panel is visible."""
        sep: Any | None = getattr(self, "_boundary_link_sep", None)
        if sep is None:
            return
        boundary_visible: bool = (
            getattr(self, "_boundary_name_panel", None) is not None
            and self._boundary_name_panel.isVisible()
        )
        names_visible: bool = (
            getattr(self, "_channel_names_panel", None) is not None
            and self._channel_names_panel.isVisible()
        )
        limits_visible: bool = (
            getattr(self, "_model_param_limits_panel", None) is not None
            and self._model_param_limits_panel.isVisible()
        )
        sep.setVisible(boundary_visible or names_visible or limits_visible)

    def _rebuild_channel_names_panel(self) -> None:
        """Rebuild inline channel name / unit editors in the Model tab."""
        panel: Any | None = getattr(self, "_channel_names_panel", None)
        layout: Any | None = getattr(self, "_channel_names_panel_layout", None)
        if panel is None or layout is None:
            return
        clear_layout(layout)
        self._channel_name_edits.clear()

        channel_names = self._available_channel_names()
        if not channel_names:
            panel.setVisible(False)
            self._sync_model_settings_sep()
            return

        header: QLabel = self._new_label(
            "Channel Names & Units",
            style_sheet="font-weight: 600; color: #334155; font-size: 11px; padding: 0;",
            tooltip="Set display names and units for legends and axis labels.",
        )
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        grid.addWidget(
            self._new_label(
                "Channel",
                style_sheet="font-weight: 700; color: #64748b; font-size: 10px;",
            ),
            0,
            0,
        )
        grid.addWidget(
            self._new_label(
                "Display Name",
                style_sheet="font-weight: 700; color: #64748b; font-size: 10px;",
            ),
            0,
            1,
        )
        grid.addWidget(
            self._new_label(
                "Unit",
                style_sheet="font-weight: 700; color: #64748b; font-size: 10px;",
            ),
            0,
            2,
        )

        for row_idx, ch_key in enumerate(channel_names, start=1):
            color: str = self._channel_plot_color(ch_key)
            ch_label: QLabel = self._new_label(
                str(ch_key),
                style_sheet=(f"font-weight: 600; color: {color}; font-size: 11px;"),
            )
            name_edit: QLineEdit = self._new_line_edit(
                str(self.channels.get(ch_key, ch_key)),
                fixed_width=140,
                tooltip=f"Display label for {ch_key}. Leave blank to use {ch_key}.",
            )
            name_edit.setPlaceholderText(str(ch_key))
            name_edit.editingFinished.connect(
                lambda k=ch_key: self._on_channel_name_edited(k)
            )
            unit_edit: QLineEdit = self._new_line_edit(
                str(getattr(self, "channel_units", {}).get(ch_key, "")),
                fixed_width=50,
                tooltip=f"Unit for {ch_key} (e.g. V, mV, s, ms).",
            )
            unit_edit.setPlaceholderText("unit")
            unit_edit.editingFinished.connect(
                lambda k=ch_key: self._on_channel_name_edited(k)
            )
            grid.addWidget(ch_label, row_idx, 0)
            grid.addWidget(name_edit, row_idx, 1)
            grid.addWidget(unit_edit, row_idx, 2)
            self._channel_name_edits[ch_key] = (name_edit, unit_edit)

        layout.addLayout(grid)
        panel.setVisible(True)
        self._sync_model_settings_sep()

    def _on_channel_name_edited(self, channel_key) -> None:
        """Apply an inline channel name / unit edit."""
        editors = self._channel_name_edits.get(channel_key)
        if editors is None:
            return
        name_edit, unit_edit = editors
        value = name_edit.text().strip()
        unit = unit_edit.text().strip()
        self.channels[channel_key] = value or channel_key
        self.channel_units[channel_key] = unit
        self._refresh_channel_name_references(autosave=True)

    def _sync_param_channel_header_labels(self) -> None:
        labels: Dict[str, QLabel] = dict(
            getattr(self, "_param_channel_header_labels", {}) or {}
        )
        for target, label in labels.items():
            if label is None:
                continue
            display_name: str = self._channel_display_name(target)
            label.setText(f"Channel: {display_name}")
            label.setToolTip(f"Parameters for {display_name} equation")

    def _refresh_channel_name_references(
        self, *, refresh_plot=True, autosave=False
    ) -> None:
        """Refresh widgets that display channel aliases/units."""
        self._sync_param_channel_header_labels()
        if self.current_data is not None:
            self._refresh_channel_combos()
        else:
            self._rebuild_channel_visibility_toggles()
            self._rebuild_channel_names_panel()
        self._rebuild_model_segment_info()
        self._rebuild_equation_toggles()
        self._rebuild_boundary_fix_controls()
        panel: Any | None = getattr(self, "_procedure_panel", None)
        if panel is not None:
            refresh_ctx = getattr(panel, "refresh_display_context", None)
            if callable(refresh_ctx):
                refresh_ctx()
        if hasattr(self, "formula_label") and not getattr(
            self, "_expression_edit_mode", False
        ):
            self._set_formula_label()
        if bool(getattr(self, "batch_results", [])):
            self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        if refresh_plot:
            self.update_plot(fast=False, preserve_view=False)
        if autosave:
            self._autosave_fit_details()

    def _rebuild_model_param_limits_panel(self) -> None:
        panel: Any | None = getattr(self, "_model_param_limits_panel", None)
        layout: Any | None = getattr(self, "_model_param_limits_panel_layout", None)
        if panel is None or layout is None:
            return
        clear_layout(layout)
        self._model_param_min_spinboxes = {}
        self._model_param_max_spinboxes = {}
        self._model_param_periodic_checkboxes = {}

        if not self.param_specs:
            panel.setVisible(False)
            self._sync_model_settings_sep()
            return

        layout.addWidget(
            self._new_label(
                "Parameter Limits",
                style_sheet="font-weight: 600; color: #334155; font-size: 11px; padding: 0;",
                tooltip="Edit fit bounds here. Plot tab keeps only value sliders.",
            )
        )

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)
        grid.addWidget(
            self._new_label(
                "Parameter",
                style_sheet="font-weight: 700; color: #64748b; font-size: 10px;",
            ),
            0,
            0,
        )
        grid.addWidget(
            self._new_label(
                "Lower",
                style_sheet="font-weight: 700; color: #64748b; font-size: 10px;",
            ),
            0,
            1,
        )
        grid.addWidget(
            self._new_label(
                "Upper",
                style_sheet="font-weight: 700; color: #64748b; font-size: 10px;",
            ),
            0,
            2,
        )
        grid.addWidget(
            self._new_label(
                "Periodic",
                style_sheet="font-weight: 700; color: #64748b; font-size: 10px;",
            ),
            0,
            3,
        )

        for row_idx, spec in enumerate(self.param_specs, start=1):
            key = str(spec.key)
            symbol: str = self._display_symbol_for_param_html(spec.key, spec.symbol)
            label: QLabel = self._new_label(
                f"{symbol}:",
                object_name="paramInline",
                tooltip=str(spec.description),
            )
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setSizePolicy(
                QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
            )
            grid.addWidget(label, row_idx, 0)

            src_min = self.param_min_spinboxes.get(key)
            src_max = self.param_max_spinboxes.get(key)
            low: float = (
                float(src_min.value()) if src_min is not None else float(spec.min_value)
            )
            high: float = (
                float(src_max.value()) if src_max is not None else float(spec.max_value)
            )
            if low > high:
                low, high = high, low

            _bound_range = 1e12
            min_box: CompactDoubleSpinBox = self._new_compact_param_spinbox(
                spec,
                low,
                minimum=-_bound_range,
                maximum=_bound_range,
                precision_min=low,
                precision_max=high,
                width=80,
                object_name="paramBoundBox",
                tooltip=f"Lower bound for {key}",
            )
            min_box.valueChanged.connect(
                lambda _value, name=key: self._on_model_param_bounds_changed(
                    name, "min"
                )
            )
            max_box: CompactDoubleSpinBox = self._new_compact_param_spinbox(
                spec,
                high,
                minimum=-_bound_range,
                maximum=_bound_range,
                precision_min=low,
                precision_max=high,
                width=80,
                object_name="paramBoundBox",
                tooltip=f"Upper bound for {key}",
            )
            max_box.valueChanged.connect(
                lambda _value, name=key: self._on_model_param_bounds_changed(
                    name, "max"
                )
            )
            periodic_cb: QCheckBox = self._new_checkbox(
                "",
                checked=bool(key in getattr(self, "_periodic_param_keys", set())),
                tooltip=(
                    "Treat this parameter as periodic during fitting. "
                    "One period is inferred from current [lower, upper] bounds."
                ),
            )
            periodic_cb.stateChanged.connect(
                lambda _state, name=key: self._on_model_param_periodic_toggled(name)
            )
            grid.addWidget(min_box, row_idx, 1)
            grid.addWidget(max_box, row_idx, 2)
            grid.addWidget(periodic_cb, row_idx, 3, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._model_param_min_spinboxes[key] = min_box
            self._model_param_max_spinboxes[key] = max_box
            self._model_param_periodic_checkboxes[key] = periodic_cb

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addLayout(grid)
        row.addStretch(1)
        layout.addLayout(row)
        panel.setVisible(True)
        self._sync_model_param_limits_from_primary()
        self._sync_model_settings_sep()

    def _sync_model_param_limits_from_primary(self, key=None) -> None:
        keys = (
            [str(key)]
            if key is not None
            else list(self._model_param_min_spinboxes.keys())
        )
        for param_key in keys:
            model_min = self._model_param_min_spinboxes.get(param_key)
            model_max = self._model_param_max_spinboxes.get(param_key)
            src_min = self.param_min_spinboxes.get(param_key)
            src_max = self.param_max_spinboxes.get(param_key)
            if (
                model_min is None
                or model_max is None
                or src_min is None
                or src_max is None
            ):
                continue
            model_min.blockSignals(True)
            model_max.blockSignals(True)
            model_min.setValue(float(src_min.value()))
            model_max.setValue(float(src_max.value()))
            model_min.blockSignals(False)
            model_max.blockSignals(False)

    def _on_model_param_bounds_changed(self, key, source) -> None:
        src_min = self.param_min_spinboxes.get(key)
        src_max = self.param_max_spinboxes.get(key)
        model_min = self._model_param_min_spinboxes.get(key)
        model_max = self._model_param_max_spinboxes.get(key)
        if src_min is None or src_max is None or model_min is None or model_max is None:
            return

        if source == "min":
            src_min.setValue(float(model_min.value()))
        else:
            src_max.setValue(float(model_max.value()))

    def _on_model_param_periodic_toggled(self, key) -> None:
        cb: QCheckBox | None = self._model_param_periodic_checkboxes.get(str(key))
        if cb is None:
            return
        periodic_keys: set[str] = set(getattr(self, "_periodic_param_keys", set()))
        if cb.isChecked():
            periodic_keys.add(str(key))
        else:
            periodic_keys.discard(str(key))
        self._periodic_param_keys = periodic_keys
        self._autosave_fit_details()

    def create_parameters_frame(self, parent_layout) -> None:
        """Create full-width controls + parameters section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        equation_host = QWidget()
        equation_host.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        equation_host_layout = QVBoxLayout(equation_host)
        equation_host_layout.setContentsMargins(0, 0, 0, 0)
        equation_host_layout.setSpacing(0)
        equation_slot_layout = QGridLayout()
        equation_slot_layout.setContentsMargins(0, 0, 0, 0)
        equation_slot_layout.setSpacing(0)
        self.formula_label = ClickableLabel(self.current_expression)
        self.formula_label.setMinimumHeight(40)
        self.formula_label.setMaximumHeight(10000)
        self.formula_label.setWordWrap(True)
        self.formula_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        self.formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_label.clicked.connect(self._enter_expression_edit_mode)
        equation_slot_layout.addWidget(self.formula_label, 0, 0)

        self.expression_editor_widget = QWidget()
        self.expression_editor_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        self.expression_editor_widget.setMinimumHeight(
            self.formula_label.minimumHeight()
        )
        self.expression_editor_widget.setMaximumHeight(
            self.formula_label.maximumHeight()
        )
        editor_layout = QVBoxLayout(self.expression_editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)

        expr_layout = QHBoxLayout()
        expr_layout.setContentsMargins(0, 0, 0, 0)
        expr_layout.setSpacing(2)
        self.function_input = VerticallyCenteredTextEdit()
        self.function_input.setAcceptRichText(False)
        self.function_input.setPlaceholderText(
            "Example: CH2 = seg1 ; seg2 ; ... ; segN"
        )
        self.function_input.setPlainText(self.current_expression)
        self.function_input.setMinimumHeight(self.formula_label.minimumHeight())
        self.function_input.setMaximumHeight(self.formula_label.maximumHeight())
        self.function_input.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self.function_input.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.function_input.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.function_input.setStyleSheet("padding: 1px 4px;")
        self.function_input.document().setDocumentMargin(1)
        self.function_input.setToolTip(
            "Equation format: TARGET = seg1 ; seg2 ; ... ; segN (semicolon-separated)"
        )
        self.function_input.textChanged.connect(self._on_expression_text_changed)
        self.function_input.focus_left.connect(self._on_expression_input_focus_left)
        self.function_input.apply_requested.connect(
            self._on_expression_input_apply_requested
        )
        expr_layout.addWidget(self.function_input, 1)

        # Insert button/menu intentionally removed for a cleaner equation editor.
        editor_layout.addLayout(expr_layout)

        self.function_status_label: QLabel = self._new_label(
            "", object_name="statusLabel"
        )
        self.function_status_label.hide()
        editor_layout.addWidget(self.function_status_label)
        equation_slot_layout.addWidget(self.expression_editor_widget, 0, 0)
        equation_host_layout.addLayout(equation_slot_layout)
        self._equation_host_widget: QWidget = equation_host
        self.breakpoint_top_widget: QWidget = (
            self._build_top_breakpoint_controls_widget()
        )

        source_file_widget = QWidget()
        source_file_layout = QVBoxLayout(source_file_widget)
        source_file_layout.setContentsMargins(0, 0, 0, 0)
        source_file_layout.setSpacing(4)
        source_layout = QHBoxLayout()
        source_layout.setSpacing(4)
        source_layout.addWidget(self._make_param_header_label("Source", width=48))
        self.source_path_label = ClickableLabel("")
        self.source_path_label.setObjectName("sourcePathLabel")
        self.source_path_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.source_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.source_path_label.clicked.connect(self.browse_directory)
        self._refresh_source_path_label()
        source_layout.addWidget(self.source_path_label, 1)
        source_file_layout.addLayout(source_layout)

        file_layout = QHBoxLayout()
        file_layout.setSpacing(4)
        file_layout.addWidget(self._make_param_header_label("File", width=48))
        self.file_combo: RichTextComboBox | QComboBox = self._new_combobox(
            current_index_changed=self.on_file_changed
        )
        file_layout.addWidget(self.file_combo, 1)
        self.prev_file_btn: QPushButton = self._make_compact_tool_button(
            "◀", "Previous File", self.prev_file
        )
        file_layout.addWidget(self.prev_file_btn)
        self.next_file_btn: QPushButton = self._make_compact_tool_button(
            "▶", "Next File", self.next_file
        )
        file_layout.addWidget(self.next_file_btn)
        self._sync_file_navigation_buttons()
        source_file_layout.addLayout(file_layout)

        channel_layout = QHBoxLayout()
        channel_layout.setSpacing(4)
        channel_layout.addWidget(self._make_param_header_label("X", width=20))
        self.x_channel_combo: RichTextComboBox | QComboBox = self._new_combobox(
            current_index_changed=self._on_x_channel_changed
        )
        channel_layout.addWidget(self.x_channel_combo, 1)
        source_file_layout.addLayout(channel_layout)

        fit_widget = QWidget()
        fit_widget_layout = QVBoxLayout(fit_widget)
        fit_widget_layout.setContentsMargins(0, 0, 0, 0)
        fit_widget_layout.setSpacing(4)

        file_group = QGroupBox("")
        file_group_layout = QVBoxLayout(file_group)
        file_group_layout.setContentsMargins(6, 6, 6, 6)
        file_group_layout.setSpacing(4)
        file_group_layout.addWidget(
            self._new_label(
                "File Options",
                style_sheet="font-weight: 600; color: #374151; padding: 1px 2px;",
            )
        )
        file_group_layout.addWidget(source_file_widget)
        file_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        fit_widget_layout.addWidget(file_group, 1)

        fit_group = QGroupBox("")
        fit_group_layout = QVBoxLayout(fit_group)
        fit_group_layout.setContentsMargins(6, 6, 6, 6)
        fit_group_layout.setSpacing(4)
        fit_group_layout.addWidget(
            self._new_label(
                "Fit Options",
                style_sheet="font-weight: 600; color: #374151; padding: 1px 2px;",
            )
        )

        self.show_residuals_cb: QPushButton = self._new_button(
            "Residuals",
            checkable=True,
            checked=False,
            toggled_handler=lambda: self.update_plot(fast=False),
        )

        self.smoothing_toggle_btn: QPushButton = self._new_button(
            "Smooth",
            checkable=True,
            checked=self.smoothing_enabled,
            toggled_handler=self._on_smoothing_controls_changed,
            tooltip="Apply moving-average smoothing to channels before fitting/analysis.",
        )
        self.smoothing_enable_cb: QPushButton = self.smoothing_toggle_btn

        fit_actions_row = QHBoxLayout()
        fit_actions_row.setSpacing(4)
        self.auto_fit_btn: QToolButton = QToolButton()
        self.auto_fit_btn.setObjectName("actionButton")
        self.auto_fit_btn.setProperty("primary", True)
        self.auto_fit_btn.setText(self.auto_fit_btn_default_text)
        self.auto_fit_btn.setToolTip(
            "Click to run fit. Use the arrow to choose Straightforward or Procedure mode."
        )
        self.auto_fit_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.auto_fit_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.auto_fit_btn.clicked.connect(self.auto_fit)
        self.auto_fit_mode_menu = QMenu(self.auto_fit_btn)
        self._auto_fit_mode_actions = {}
        for mode_label, mode_value in (
            ("Straightforward", "fit"),
            ("Procedure", "procedure"),
        ):
            action: QAction | None = self.auto_fit_mode_menu.addAction(mode_label)
            if action is None:
                continue
            action.triggered.connect(
                lambda _checked=False, m=mode_value: self._set_auto_fit_mode(m)
            )
            self._auto_fit_mode_actions[str(mode_value)] = action
        self.auto_fit_btn.setMenu(self.auto_fit_mode_menu)
        self._set_auto_fit_mode(self._auto_fit_run_mode, autosave=False)
        self._set_split_action_min_width(
            self.auto_fit_btn,
            [
                self._auto_fit_button_text_for_mode("fit"),
                self._auto_fit_button_text_for_mode("procedure"),
                "Cancel",
            ],
        )

        self.reset_from_batch_btn: QPushButton = self._new_button(
            "Reset",
            handler=self.reset_params_from_batch,
            tooltip="Load parameters for the current file from the batch table row.",
        )
        self.fit_compute_mode_btn: QPushButton = self._new_button(
            "GPU",
            checkable=True,
            checked=(self._current_fit_compute_mode() == "gpu"),
            toggled_handler=self._on_fit_compute_mode_toggled,
            tooltip=(
                "Fit backend target for JAX (GPU or CPU). "
                "If JAX is already initialized, restart the app to apply a switch."
            ),
            fixed_width=64,
        )
        fit_actions_row.addWidget(self.auto_fit_btn)
        fit_actions_row.addWidget(self.reset_from_batch_btn)
        fit_actions_row.addWidget(self.fit_compute_mode_btn)
        self._set_fit_compute_mode(
            self._current_fit_compute_mode(),
            autosave=False,
            show_status=False,
        )

        fit_actions_row.addStretch(1)
        fit_group_layout.addLayout(fit_actions_row)

        fit_policy_row = QHBoxLayout()
        fit_policy_row.setSpacing(4)
        self.clear_previous_result_btn: QPushButton = self._new_button(
            "Clear Previous Result",
            handler=self.clear_previous_result_for_current_file,
            tooltip=(
                "Clear the stored fit result for the currently loaded file "
                "(single-file action)."
            ),
        )
        fit_policy_row.addWidget(self.clear_previous_result_btn)
        fit_policy_row.addStretch(1)
        fit_group_layout.addLayout(fit_policy_row)

        # --- Per-equation fit toggles ---
        self.equation_toggles_widget = QWidget()
        self.equation_toggles_layout = QHBoxLayout(self.equation_toggles_widget)
        self.equation_toggles_layout.setContentsMargins(0, 0, 0, 0)
        self.equation_toggles_layout.setSpacing(6)
        fit_group_layout.addWidget(self.equation_toggles_widget)
        self._rebuild_equation_toggles()

        self.boundary_fix_widget = QWidget()
        self.boundary_fix_layout = QVBoxLayout(self.boundary_fix_widget)
        self.boundary_fix_layout.setContentsMargins(0, 0, 0, 0)
        self.boundary_fix_layout.setSpacing(2)
        self.boundary_fix_layout.addWidget(
            self._new_label(
                "Fit Boundaries",
                object_name="paramHeader",
                style_sheet="font-weight: 600; color: #475569;",
            )
        )
        self.boundary_fix_checks_widget = QWidget()
        self.boundary_fix_checks_layout = QHBoxLayout(self.boundary_fix_checks_widget)
        self.boundary_fix_checks_layout.setContentsMargins(0, 0, 0, 0)
        self.boundary_fix_checks_layout.setSpacing(4)
        self.boundary_fix_layout.addWidget(self.boundary_fix_checks_widget)
        fit_group_layout.addWidget(self.boundary_fix_widget)

        fit_view_row = QHBoxLayout()
        fit_view_row.setSpacing(4)
        fit_view_row.addWidget(self.show_residuals_cb)
        fit_view_row.addWidget(self.smoothing_toggle_btn)
        fit_view_row.addWidget(
            self._new_label(
                "N",
                object_name="paramInline",
                tooltip="Smoothing window size (odd samples).",
                style_sheet="font-weight: 600; color: #475569;",
            )
        )

        self.smoothing_window_spin: QSpinBox = self._new_compact_int_spinbox(
            1,
            101,
            self._effective_smoothing_window(),
            single_step=2,
            tooltip="Smoothing window size (odd samples).",
        )
        self.smoothing_window_spin.valueChanged.connect(
            self._on_smoothing_controls_changed
        )
        fit_view_row.addWidget(self.smoothing_window_spin)
        fit_view_row.addStretch(1)
        fit_group_layout.addLayout(fit_view_row)
        self._rebuild_boundary_fix_controls()

        fit_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        fit_widget_layout.addWidget(fit_group, 1)
        self._sync_smoothing_window_enabled()
        self.create_batch_controls_frame(fit_widget_layout)

        self.expression_highlighter: ExpressionSyntaxHighlighter[
            QTextDocument | None
        ] = ExpressionSyntaxHighlighter(self.function_input.document())

        self.param_header_widget = QWidget()
        self.param_header_widget.installEventFilter(self)
        self.param_header_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        param_header_layout = QHBoxLayout(self.param_header_widget)
        param_header_layout.setContentsMargins(0, 0, 0, 0)
        param_header_layout.setSpacing(6)
        param_header_layout.addWidget(
            self._make_param_header_label("Parameter", width=self._param_name_width)
        )
        self._param_header_lower_label: QLabel = self._make_param_header_label(
            "Lower", width=self._param_bound_width
        )
        if self._show_plot_param_bounds:
            param_header_layout.addWidget(self._param_header_lower_label)
        slider_header: QLabel = self._make_param_header_label("Range")
        slider_header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        param_header_layout.addWidget(slider_header, 1)
        self._param_header_upper_label: QLabel = self._make_param_header_label(
            "Upper", width=self._param_bound_width
        )
        if self._show_plot_param_bounds:
            param_header_layout.addWidget(self._param_header_upper_label)
        param_header_layout.addWidget(
            self._make_param_header_label("Value", width=self._param_value_width)
        )
        param_header_layout.addWidget(self._make_param_header_label("Fit", width=20))
        params_left_widget = QWidget()
        params_left_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        params_left_layout = QVBoxLayout(params_left_widget)
        params_left_layout.setContentsMargins(0, 0, 0, 0)
        params_left_layout.setSpacing(6)
        self._param_header_to_rows_gap: int = params_left_layout.spacing()
        params_left_layout.addWidget(self.param_header_widget)

        self.param_controls_widget = QWidget()
        self.param_controls_layout = QVBoxLayout(self.param_controls_widget)
        self.param_controls_layout.setSpacing(6)
        self.param_controls_layout.setContentsMargins(0, 0, 0, 0)

        self.param_controls_scroll = QScrollArea()
        self.param_controls_scroll.setWidgetResizable(True)
        self.param_controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.param_controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.param_controls_scroll.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.param_controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.param_controls_scroll.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background: #ffffff;
            }
            QScrollArea QWidget#qt_scrollarea_viewport {
                background: #ffffff;
            }
            """
        )
        self.param_controls_scroll.setWidget(self.param_controls_widget)

        # Param scroll takes the full space; procedure panel overlays on top.
        params_left_layout.addWidget(self.param_controls_scroll, 1)
        self._params_left_widget = params_left_widget

        # Floating procedure-log overlay (hidden by default)
        self._procedure_live_panel = ProcedureLivePanel(parent=params_left_widget)
        self._procedure_live_panel.hide()
        self._procedure_live_panel.dismissed.connect(self._on_procedure_panel_dismissed)
        self._panel_active_task_id: int | None = None

        # Toggle button to show/hide procedure log (appears during a run)
        self._proc_log_toggle_btn = QPushButton("\u25b2 Procedure Log")
        self._proc_log_toggle_btn.setStyleSheet(
            "QPushButton { border: 1px solid #cbd5e1; border-radius: 4px;"
            " background: #f8fafc; color: #475569; font-size: 12px;"
            " font-weight: 600; padding: 3px 10px; }"
            "QPushButton:hover { background: #e2e8f0; }"
        )
        self._proc_log_toggle_btn.setFixedHeight(24)
        self._proc_log_toggle_btn.hide()
        self._proc_log_toggle_btn.clicked.connect(self._on_proc_log_toggle)
        params_left_layout.addWidget(self._proc_log_toggle_btn)

        fit_widget.setMinimumWidth(340)
        fit_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self.fit_panel_widget: QWidget = fit_widget
        fit_right_widget = QWidget()
        fit_right_widget.setMinimumWidth(340)
        fit_right_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        fit_right_layout = QVBoxLayout(fit_right_widget)
        fit_right_layout.setContentsMargins(0, 0, 0, 0)
        fit_right_layout.setSpacing(0)
        self.fit_panel_top_spacer = QWidget()
        self.fit_panel_top_spacer.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.fit_panel_top_spacer.setFixedHeight(0)
        fit_right_layout.addWidget(self.fit_panel_top_spacer)
        fit_right_layout.addWidget(
            fit_widget,
            1,
        )
        self.param_fit_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.param_fit_splitter.setChildrenCollapsible(False)
        self.param_fit_splitter.addWidget(params_left_widget)
        self.param_fit_splitter.addWidget(fit_right_widget)
        self.param_fit_splitter.setStretchFactor(0, 1)
        self.param_fit_splitter.setStretchFactor(1, 0)
        self.param_fit_splitter.splitterMoved.connect(
            lambda *_: self._autosave_fit_details()
        )
        params_status_layout = QVBoxLayout()
        params_status_layout.setContentsMargins(0, 0, 0, 0)
        params_status_layout.setSpacing(0)
        params_status_layout.addWidget(self.param_fit_splitter)
        layout.addLayout(params_status_layout)
        self.rebuild_manual_param_controls()
        self._rebuild_model_segment_info()
        self._rebuild_channel_token_buttons()
        self._set_formula_label()
        self._set_expression_edit_mode(False)
        self._sync_fit_panel_top_spacing()
        QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
        self._sync_param_row_tail_spacers()
        QTimer.singleShot(0, self._sync_param_row_tail_spacers)

        group.setLayout(layout)
        self._parameters_group: QGroupBox = group
        parent_layout.addWidget(group)

    def _set_function_status(self, message, is_error=False) -> None:
        if not hasattr(self, "function_status_label"):
            return
        if is_error:
            self.function_status_label.setText(str(message))
            self.function_status_label.setStyleSheet(
                "color: #b91c1c; font-weight: 600; padding: 1px 2px;"
            )
            self.function_status_label.show()
        else:
            self.function_status_label.clear()
            self.function_status_label.setStyleSheet("")
            self.function_status_label.hide()

    def _expression_editor_text(self) -> str:
        if not hasattr(self, "function_input"):
            return ""
        raw: str = self.function_input.toPlainText()
        lines: List[str] = [line.strip() for line in raw.splitlines() if line.strip()]
        # Preserve newlines between lines to support multi-channel equations.
        return "\n".join(lines).strip()

    def _set_expression_editor_text(self, text) -> None:
        if not hasattr(self, "function_input"):
            return
        self.function_input.blockSignals(True)
        self.function_input.setPlainText(str(text))
        self.function_input.blockSignals(False)

    def _resolve_column_name(self, name):
        target: str = str(name).strip()
        if not target:
            return None
        available = self._available_channel_names()
        if target in available:
            return target
        lookup = {col.upper(): col for col in available}
        return lookup.get(target.upper())

    def _parse_equation_text(self, text, strict=False):
        equation: str = str(text).strip()
        if not equation:
            raise ValueError("Expression is empty.")

        if "=" in equation:
            left, right = equation.split("=", 1)
            lhs_text: str = left.strip()
            rhs_text: str = right.strip()
        else:
            if strict:
                raise ValueError("Use equation form: TARGET = seg1 ; seg2 ; ... ; segN")
            lhs_text = self._primary_target_channel()
            rhs_text: str = equation

        if not is_valid_parameter_name(lhs_text):
            raise ValueError("Invalid left-hand column. Use a CSV column name.")
        if not rhs_text:
            raise ValueError("Right-hand expression is empty.")

        target_col = self._resolve_column_name(lhs_text)
        if target_col is None:
            available: str = ", ".join(self._available_channel_names()) or "none"
            raise ValueError(
                f"Target column '{lhs_text}' is not in CSV columns ({available})."
            )
        segments: List[str] = [part.strip() for part in rhs_text.split(";")]
        if len(segments) < 1:
            raise ValueError(
                "Use one or more segment expressions: TARGET = seg1 ; seg2 ; ... ; segN"
            )
        if any(not seg for seg in segments):
            raise ValueError("Each segment expression must be non-empty.")
        return target_col, segments

    def _parse_multi_equation_text(self, text, strict=False):
        """Parse multi-line expression text into a list of (target_col, segments) tuples.

        Each line is a separate channel equation: ``TARGET = seg1 ; seg2``.
        If only a single line is present, falls back to single-channel behaviour.
        Returns a list of ``(target_col, [seg1, seg2, ...])`` tuples.
        """
        raw: str = str(text).strip()
        if not raw:
            raise ValueError("Expression is empty.")

        # Split on newlines (each line is a separate channel equation).
        lines: List[str] = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) == 0:
            raise ValueError("Expression is empty.")

        equations = []
        seen_targets = set()
        for line_num, line in enumerate(lines, start=1):
            try:
                target_col, segments = self._parse_equation_text(line, strict=strict)
            except ValueError as exc:
                if len(lines) > 1:
                    raise ValueError(f"Line {line_num}: {exc}") from exc
                raise
            if target_col in seen_targets:
                raise ValueError(
                    f"Line {line_num}: duplicate target channel '{target_col}'."
                )
            seen_targets.add(target_col)
            equations.append((target_col, segments))
        return equations

    def _on_expression_text_changed(self) -> None:
        self._refresh_expression_highlighting()

    def _insert_expression_token(self, token_text) -> None:
        if not hasattr(self, "function_input"):
            return
        self.function_input.insertPlainText(str(token_text))
        self.function_input.setFocus()

    def _add_channel_token_button(self, label, token) -> None:
        button: QPushButton = self._new_button(
            label,
            handler=lambda _checked=False, t=token: self._insert_expression_token(t),
            min_height=20,
            tooltip="Insert column token into expression.",
            style_sheet=(
                f"""
            QPushButton {{
                color: {COLUMN_COLOR};
                background: #ffffff;
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                font-weight: 600;
                padding: 1px 8px;
            }}
            QPushButton:hover {{
                background: #ffffff;
                border-color: #60a5fa;
            }}
            QPushButton:pressed {{
                background: #ffffff;
                border-color: #3b82f6;
            }}
            """
            ),
        )
        self.channel_tokens_layout.addWidget(button)

    def _refresh_expression_highlighting(self) -> None:
        if self._highlight_refresh_in_progress:
            return
        if not hasattr(self, "expression_highlighter"):
            return
        self._highlight_refresh_in_progress = True
        try:
            expression_text: str = self._expression_editor_text()
            columns = self._available_channel_names()
            params = []
            if expression_text:
                try:
                    channel_equations = self._parse_multi_equation_text(
                        expression_text,
                        strict=False,
                    )
                    seen = set()
                    for _target, seg_exprs in channel_equations:
                        for seg_expr in seg_exprs:
                            seg_params: List[str] = extract_segment_parameter_names(
                                seg_expr
                            )
                            for name in seg_params:
                                if name not in seen:
                                    seen.add(name)
                                    params.append(name)
                except Exception:
                    params = []
            self.expression_highlighter.set_context(columns, params)
        finally:
            self._highlight_refresh_in_progress = False

    def _rebuild_channel_token_buttons(self) -> None:
        tokens: List[str] = ["x"]

        if hasattr(self, "channel_token_menu"):
            self.channel_token_menu.clear()
            for token_name in tokens:
                action = self.channel_token_menu.addAction(token_name)
                action.triggered.connect(
                    lambda _checked=False, t=token_name: self._insert_expression_token(
                        t
                    )
                )
            if hasattr(self, "insert_token_btn"):
                self.insert_token_btn.setEnabled(bool(tokens))

        if hasattr(self, "channel_tokens_layout"):
            clear_layout(self.channel_tokens_layout)
            for token_name in tokens:
                self._add_channel_token_button(token_name, token_name)
            self.channel_tokens_layout.addStretch(1)
        self._refresh_expression_highlighting()

    def _remap_param_values_by_key(
        self,
        raw_params,
        *,
        source_keys: Sequence[str],
        target_keys: Sequence[str],
        fallback_by_key: Mapping[str, Any] | None = None,
    ) -> None | List[float]:
        """Remap a parameter vector from source key order into target key order."""
        values: np.ndarray[Tuple[int], np.dtype[Any]] = self._as_float_array(raw_params)
        if values.size <= 0:
            return None

        source_map: Dict[str, float] = {}
        for idx, key in enumerate(list(source_keys)):
            if idx >= values.size:
                break
            numeric: float | None = finite_float_or_none(values[idx])
            if numeric is None:
                continue
            source_map[str(key)] = float(numeric)

        fallback_map: Dict[str, Any] = dict(fallback_by_key or {})
        remapped: List[float] = []
        for key in list(target_keys):
            target_key: str = str(key)
            value: float | None = source_map.get(target_key)
            if value is None:
                fallback_val: float | None = finite_float_or_none(
                    fallback_map.get(target_key)
                )
                value = float(fallback_val) if fallback_val is not None else 0.0
            remapped.append(float(value))
        return remapped

    def _remap_batch_result_params_on_model_change(
        self,
        *,
        old_param_keys: Sequence[str],
        new_param_specs: Sequence[ParameterSpec],
    ) -> int:
        """Remap stored batch fit vectors to the current parameter key order."""
        rows: List[Any] = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return 0

        source_keys: List[str] = [str(key) for key in list(old_param_keys or [])]
        target_keys: List[str] = [str(spec.key) for spec in list(new_param_specs or [])]
        if not source_keys or not target_keys:
            return 0

        spec_by_key: Dict[str, ParameterSpec] = {
            str(spec.key): spec for spec in list(new_param_specs or [])
        }
        fallback_by_key: Dict[str, float] = {
            str(spec.key): self._param_default_from_limits(
                spec.min_value, spec.max_value
            )
            for spec in list(new_param_specs or [])
        }
        remapped_rows: int = 0
        for idx, raw_row in enumerate(rows):
            row = canonicalize_fit_row(raw_row)
            prior_values: np.ndarray[Tuple[int], np.dtype[Any]] = self._as_float_array(
                fit_get(row, "params")
            )
            remapped: None | List[float] = self._remap_param_values_by_key(
                fit_get(row, "params"),
                source_keys=source_keys,
                target_keys=target_keys,
                fallback_by_key=fallback_by_key,
            )
            if remapped is None:
                self.batch_results[idx] = row
                continue

            clipped_remapped: List[float] = []
            for key, raw_value in zip(target_keys, remapped):
                numeric: float = float(raw_value)
                spec: ParameterSpec | None = spec_by_key.get(str(key))
                if spec is not None:
                    low = float(min(spec.min_value, spec.max_value))
                    high = float(max(spec.min_value, spec.max_value))
                    if (
                        str(key) in set(getattr(self, "_periodic_param_keys", set()))
                        and np.isfinite(low)
                        and np.isfinite(high)
                        and (high > low)
                    ):
                        numeric = float(low + np.mod(float(numeric - low), high - low))
                    else:
                        numeric = float(np.clip(numeric, low, high))
                clipped_remapped.append(float(numeric))

            remapped_arr: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                clipped_remapped, dtype=float
            ).reshape(-1)
            changed: bool = prior_values.size != remapped_arr.size or not np.allclose(
                prior_values,
                remapped_arr,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            if changed:
                remapped_rows += 1
            fit_set(row, "params", [float(v) for v in remapped_arr.tolist()])
            row["plot_has_fit"] = bool(has_nonempty_values(fit_get(row, "params")))
            row = self._apply_param_range_validation_to_row(row)
            self.batch_results[idx] = row
        return remapped_rows

    def apply_expression_from_input(self) -> bool:
        if self._apply_expression_in_progress:
            return False
        self._apply_expression_in_progress = True
        try:
            expression_text = self._expression_editor_text()
            old_param_keys: List[str] = [
                str(spec.key) for spec in list(getattr(self, "param_specs", []) or [])
            ]
            if (
                expression_text
                and expression_text != self.function_input.toPlainText().strip()
            ):
                self._set_expression_editor_text(expression_text)
            lower, upper = self._parameter_bounds()
            bounds_map = {
                spec.key: (lower[idx], upper[idx])
                for idx, spec in enumerate(self.param_specs)
            }
            existing_specs: Dict[str, ParameterSpec] = {
                spec.key: spec for spec in self.param_specs
            }

            try:
                channel_equations = self._parse_multi_equation_text(
                    expression_text, strict=True
                )
                # Primary target is the first channel equation.
                target_col = channel_equations[0][0]
                segment_exprs = channel_equations[0][1]
                if self.x_channel == target_col:
                    available = [
                        col
                        for col in self._available_channel_names()
                        if col != target_col
                    ]
                    if available:
                        self.x_channel = available[0]

                model_def: PiecewiseModelDefinition = build_piecewise_model_definition(
                    target_col=target_col,
                    segment_exprs=segment_exprs,
                    channel_names=self._available_channel_names(),
                )
                # Build multi-channel model wrapping all channel equations.
                links = self._boundary_links_from_map()
                multi_model: MultiChannelModelDefinition = (
                    build_multi_channel_model_definition(
                        channel_equations,
                        channel_names=self._available_channel_names(),
                        boundary_links=links,
                    )
                )
                param_names: List[str] = list(multi_model.global_param_names)
                new_specs = []
                new_defaults = []
                for key in param_names:
                    existing: ParameterSpec | None = existing_specs.get(key)
                    if existing is not None:
                        symbol_hint: str = existing.symbol
                        description: str = existing.description
                        min_val, max_val = bounds_map.get(
                            key, (existing.min_value, existing.max_value)
                        )
                    else:
                        symbol_hint: str = key
                        description: str = f"Parameter {key}"
                        min_val, max_val = -10.0, 10.0

                    min_val = float(min_val)
                    max_val = float(max_val)
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    decimals = self._param_decimals_from_limits(min_val, max_val)
                    default_val = self._param_default_from_limits(min_val, max_val)
                    new_specs.append(
                        ParameterSpec(
                            key=key,
                            symbol=resolve_parameter_symbol(key, symbol_hint),
                            description=description,
                            default=default_val,
                            min_value=min_val,
                            max_value=max_val,
                            decimals=decimals,
                        )
                    )
                    new_defaults.append(default_val)
            except Exception as exc:
                self._set_function_status(f"Function error: {exc}", is_error=True)
                self._refresh_expression_highlighting()
                return False

            new_expression_text: str = "\n".join(
                f"{tc} = {' ; '.join(segs)}" for tc, segs in channel_equations
            )
            old_expression_text: str = str(
                getattr(self, "current_expression", "")
            ).strip()
            try:
                old_channel_equations = self._parse_multi_equation_text(
                    old_expression_text, strict=False
                )
                old_expression_text = "\n".join(
                    f"{tc} = {' ; '.join(segs)}" for tc, segs in old_channel_equations
                )
            except Exception:
                old_expression_text = "\n".join(
                    line.strip()
                    for line in old_expression_text.splitlines()
                    if line.strip()
                )
            expression_changed: bool = new_expression_text != old_expression_text

            self.param_specs = new_specs
            self.defaults = new_defaults
            # Store expression as multi-line text for multi-channel.
            self.current_expression: str = new_expression_text
            self._set_expression_editor_text(self.current_expression)
            self._piecewise_model: PiecewiseModelDefinition = model_def
            self._multi_channel_model: MultiChannelModelDefinition = multi_model
            # Prune manually fixed params that no longer exist.
            valid_keys: set[str] = set(multi_model.global_param_names)
            self._manually_fixed_params = (
                getattr(self, "_manually_fixed_params", set()) & valid_keys
            )
            self._periodic_param_keys = (
                getattr(self, "_periodic_param_keys", set()) & valid_keys
            )
            # Prune procedure steps for removed params/channels.
            if hasattr(self, "_procedure_panel"):
                self._procedure_panel.prune_invalid_params()
            # Prune boundary name map for IDs that no longer exist.
            valid_bids: set[Tuple[str, int]] = set(multi_model.all_boundary_ids)
            self._boundary_name_map: Dict[Any, Any] = {
                bid: name
                for bid, name in getattr(self, "_boundary_name_map", {}).items()
                if bid in valid_bids
            }
            self._manually_fixed_boundary_ids: set[tuple[str, int]] = {
                (str(t), int(i))
                for t, i in getattr(self, "_manually_fixed_boundary_ids", set())
                if (str(t), int(i)) in valid_bids
            }
            self._ensure_boundary_names()

            # Rebuild multi-channel model with links derived from boundary names.
            # _ensure_boundary_names assigns default names (X₀, X₁, ...) so
            # boundaries at the same index across channels share a name and
            # become linked.  The model built earlier may lack these links.
            updated_links = self._boundary_links_from_map()
            if updated_links != tuple(multi_model.boundary_links):
                try:
                    multi_model: MultiChannelModelDefinition = (
                        build_multi_channel_model_definition(
                            channel_equations,
                            channel_names=self._available_channel_names(),
                            boundary_links=updated_links,
                        )
                    )
                    self._multi_channel_model: MultiChannelModelDefinition = multi_model
                except Exception:
                    pass

            old_targets: Tuple[str, ...] = tuple(self._fit_state.targets())
            self._refresh_boundary_state_topology(preserve_existing=True)
            self._fit_state.apply_link_groups(
                updated_links,
                source_target=channel_equations[0][0],
                prefer_targets=old_targets,
            )
            if expression_changed:
                self.last_popt = None
                self._last_r2 = None
                self._last_per_channel_r2 = {}
                self._last_fit_active_keys = []
            # Sync per-equation enable state: default new channels to True.
            multi_targets: List[str] = list(multi_model.target_channels)
            for t in multi_targets:
                if t not in self._fit_channel_enabled:
                    self._fit_channel_enabled[t] = True
            # Remove stale entries.
            self._fit_channel_enabled = {
                t: v for t, v in self._fit_channel_enabled.items() if t in multi_targets
            }
            self.rebuild_manual_param_controls()
            self._rebuild_model_segment_info()
            self._rebuild_equation_toggles()
            self._rebuild_boundary_fix_controls()
            self._refresh_channel_combos()
            self._set_formula_label()
            self._set_function_status("", is_error=False)
            self._refresh_expression_highlighting()
            self.update_plot(fast=False, preserve_view=False)
            self._reset_plot_home_view()
            if expression_changed and self.batch_results:
                batch_action: str = self._prompt_batch_results_on_equation_change()
                if batch_action == "cancel":
                    return False
                if batch_action == "wipe":
                    for row in self.batch_results:
                        fit_set(row, "params", None)
                        fit_set(row, "r2", None)
                        fit_set(row, "error", None)
                        fit_set(row, "channel_results", None)
                        row["plot_full"] = None
                        row["plot"] = None
                        row["plot_render_size"] = None
                        row["plot_has_fit"] = None
                        row["_equation_stale"] = False
                else:
                    remapped_rows: int = self._remap_batch_result_params_on_model_change(
                        old_param_keys=old_param_keys,
                        new_param_specs=self.param_specs,
                    )
                    for row in self.batch_results:
                        row["_equation_stale"] = True
                    if remapped_rows > 0:
                        plural: str = "s" if remapped_rows != 1 else ""
                        self.stats_text.append(
                            "Equation updated; "
                            f"remapped fit parameters for {remapped_rows} kept result{plural} "
                            "and marked all kept results stale."
                        )
                    else:
                        self.stats_text.append(
                            "Equation updated; existing batch results were kept and marked stale."
                        )
                self.update_batch_table()
                self._refresh_batch_analysis_if_run()
                self.queue_visible_thumbnail_render()
                current_file: Any | None = self._current_loaded_file_path()
                if current_file:
                    self._apply_batch_params_for_file(current_file)
                self.update_plot(fast=False, preserve_view=True)
            if expression_changed:
                self._autosave_fit_details()
            return True
        finally:
            self._apply_expression_in_progress = False

    def create_param_control(
        self, spec, default_val
    ) -> Tuple[
        QVBoxLayout,
        CompactDoubleSpinBox,
        QSlider,
        CompactDoubleSpinBox,
        CompactDoubleSpinBox,
        QLabel,
        QWidget,
        QCheckBox,
    ]:
        """Create a compact row for lower/upper bounds and current value."""
        key = spec.key
        show_bounds = bool(getattr(self, "_show_plot_param_bounds", True))
        layout = QHBoxLayout()
        layout.setSpacing(6)

        name_label: QLabel = self._create_param_label(
            spec, width=self._param_name_width
        )
        layout.addWidget(name_label)

        lock_status_label: QLabel = self._new_label(
            "",
            object_name="statusLabel",
            alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            style_sheet="color: #64748b; font-style: italic;",
        )
        lock_status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        lock_status_label.hide()
        layout.addWidget(lock_status_label, 1)

        _bound_range = 1e12
        min_box: CompactDoubleSpinBox = self._new_compact_param_spinbox(
            spec,
            spec.min_value,
            minimum=-_bound_range,
            maximum=_bound_range,
            precision_min=spec.min_value,
            precision_max=spec.max_value,
            width=self._param_bound_width,
            object_name="paramBoundBox",
            tooltip="Lower bound",
        )
        min_box.valueChanged.connect(
            lambda _value, name=key: self._on_param_bounds_changed(name, "min")
        )
        min_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        min_box.installEventFilter(self._scroll_eat_filter)
        layout.addWidget(min_box)
        if not show_bounds:
            min_box.setVisible(False)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(self._param_slider_steps)
        slider.setFixedHeight(18)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.setToolTip("Sweep value across active bounds")
        layout.addWidget(slider, 1)

        max_box: CompactDoubleSpinBox = self._new_compact_param_spinbox(
            spec,
            spec.max_value,
            minimum=-_bound_range,
            maximum=_bound_range,
            precision_min=spec.min_value,
            precision_max=spec.max_value,
            width=self._param_bound_width,
            object_name="paramBoundBox",
            tooltip="Upper bound",
        )
        max_box.valueChanged.connect(
            lambda _value, name=key: self._on_param_bounds_changed(name, "max")
        )
        max_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        max_box.installEventFilter(self._scroll_eat_filter)
        layout.addWidget(max_box)
        if not show_bounds:
            max_box.setVisible(False)

        low = float(min_box.value())
        high = float(max_box.value())
        value_box: CompactDoubleSpinBox = self._new_compact_param_spinbox(
            spec,
            np.clip(default_val, low, high),
            minimum=low,
            maximum=high,
            precision_min=low,
            precision_max=high,
            width=self._param_value_width,
            object_name="paramValueBox",
            tooltip="Current value",
        )
        value_box.valueChanged.connect(lambda: self.update_plot(fast=False))
        value_box.valueChanged.connect(
            lambda _value, name=key: self._sync_slider_from_spinbox(name)
        )
        value_box.valueChanged.connect(self._autosave_fit_details)
        layout.addWidget(value_box)

        fix_checkbox = QCheckBox()
        fix_checkbox.setToolTip("Include this parameter in fitting")
        fix_checkbox.setFixedWidth(20)
        fix_checkbox.setChecked(
            key not in getattr(self, "_manually_fixed_params", set())
        )
        fix_checkbox.toggled.connect(
            lambda checked, name=key: self._on_param_fit_toggled(name, checked)
        )
        layout.addWidget(fix_checkbox)

        tail_spacer = QWidget()
        tail_spacer.setFixedWidth(max(0, int(self._param_tail_placeholder_width)))
        tail_spacer.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(tail_spacer)

        container_layout = QVBoxLayout()
        container_layout.setSpacing(1)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addLayout(layout)

        def slider_to_spinbox(position) -> None:
            value: float = self._slider_position_to_value(key, position)
            value_box.blockSignals(True)
            value_box.setValue(value)
            value_box.blockSignals(False)
            self.update_plot(fast=True)

        def slider_pressed() -> None:
            self.slider_active = True

        def slider_released() -> None:
            self.slider_active = False
            self.do_full_update()
            self._autosave_fit_details()

        slider.valueChanged.connect(slider_to_spinbox)
        slider.sliderPressed.connect(slider_pressed)
        slider.sliderReleased.connect(slider_released)

        return (
            container_layout,
            value_box,
            slider,
            min_box,
            max_box,
            lock_status_label,
            tail_spacer,
            fix_checkbox,
        )

    def _sync_param_row_tail_spacers(self) -> None:
        if not hasattr(self, "param_row_tail_spacers"):
            return
        actions_widget: Any | None = getattr(self, "param_header_actions_widget", None)
        width = int(self._param_tail_placeholder_width)
        if actions_widget is not None:
            try:
                width: int = max(
                    width,
                    int(actions_widget.minimumSizeHint().width()),
                    int(actions_widget.sizeHint().width()),
                )
            except Exception:
                pass
        width: int = max(0, int(width))
        self._param_tail_placeholder_width: int = width
        for spacer in self.param_row_tail_spacers:
            if spacer is None:
                continue
            spacer.setFixedWidth(width)

    def rebuild_manual_param_controls(self) -> None:
        if not hasattr(self, "param_controls_layout"):
            return
        # Remove the breakpoint widget from the layout before clearing so it
        # is not destroyed by clear_layout()'s deleteLater() calls.
        bp: Any | None = getattr(self, "breakpoint_top_widget", None)
        if bp is not None:
            self.param_controls_layout.removeWidget(bp)
            bp.setParent(None)
        clear_layout(self.param_controls_layout)
        # Re-parent the boundary sliders as the first item inside the scroll.
        if bp is not None:
            self.param_controls_layout.addWidget(bp)
        self.param_spinboxes.clear()
        self.param_sliders.clear()
        self.param_min_spinboxes.clear()
        self.param_max_spinboxes.clear()
        self.param_lock_status_labels.clear()
        self.param_tail_spacers_by_key.clear()
        self.param_row_tail_spacers.clear()
        self.param_fix_checkboxes.clear()
        self._param_channel_header_labels = {}

        spec_by_key: Dict[str, ParameterSpec] = {
            spec.key: spec for spec in self.param_specs
        }
        default_by_key = {
            spec.key: (self.defaults[idx] if idx < len(self.defaults) else 0.0)
            for idx, spec in enumerate(self.param_specs)
        }

        _first_channel = True
        for section in self._ordered_parameter_sections():
            kind = str(section.get("kind"))
            keys: List[str] = [str(key) for key in (section.get("keys") or [])]

            if kind == "channel_header":
                target = str(section.get("target", ""))
                if not _first_channel:
                    self.param_controls_layout.addWidget(self._build_equation_divider())
                _first_channel = False
                header_label: QLabel = self._build_param_section_header(
                    f"Channel: {self._channel_display_name(target)}",
                    tooltip=f"Parameters for {self._channel_display_name(target)} equation",
                )
                self.param_controls_layout.addWidget(header_label)
                self._param_channel_header_labels[target] = header_label
            elif kind == "segment":
                seg_idx = int(section.get("index", 0))
                target = section.get("target")
                expr_text: str = ""
                # Resolve expression from the correct channel model.
                if target is not None:
                    multi_model: Any | None = getattr(
                        self, "_multi_channel_model", None
                    )
                    if multi_model is not None:
                        for ch_m in multi_model.channel_models:
                            if ch_m.target_col == target and 1 <= seg_idx <= len(
                                ch_m.segment_exprs
                            ):
                                expr_text = str(ch_m.segment_exprs[seg_idx - 1])
                                break
                else:
                    model_def: PiecewiseModelDefinition | None = self._piecewise_model
                    if model_def is not None and 1 <= seg_idx <= len(
                        model_def.segment_exprs
                    ):
                        expr_text = str(model_def.segment_exprs[seg_idx - 1])
                self.param_controls_layout.addWidget(
                    self._build_segment_header_widget(seg_idx, expr_text, target),
                )

            elif kind == "boundary":
                self.param_controls_layout.addWidget(
                    self._build_param_boundary_marker(
                        section.get("index", 0),
                        target=section.get("target"),
                    )
                )
            elif kind == "shared":
                self.param_controls_layout.addWidget(
                    self._build_param_section_header("Shared Parameters")
                )

            for key in keys:
                spec: ParameterSpec | None = spec_by_key.get(key)
                if spec is None:
                    continue
                default_val = float(default_by_key.get(key, 0.0))
                (
                    control_layout,
                    spinbox,
                    slider,
                    min_box,
                    max_box,
                    lock_status_label,
                    tail_spacer,
                    fix_checkbox,
                ) = self.create_param_control(spec, default_val)
                self.param_spinboxes[spec.key] = spinbox
                self.param_sliders[spec.key] = slider
                self.param_min_spinboxes[spec.key] = min_box
                self.param_max_spinboxes[spec.key] = max_box
                self.param_lock_status_labels[spec.key] = lock_status_label
                self.param_tail_spacers_by_key[spec.key] = tail_spacer
                self.param_row_tail_spacers.append(tail_spacer)
                self.param_fix_checkboxes[spec.key] = fix_checkbox
                self.param_controls_layout.addLayout(control_layout)
                self._sync_slider_from_spinbox(spec.key)

        self.param_controls_layout.addStretch(1)
        self._sync_breakpoint_sliders_from_state()
        self._refresh_param_capture_mapping_controls()
        self._sync_param_row_tail_spacers()
        QTimer.singleShot(0, self._sync_param_row_tail_spacers)
        QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
        self._sync_model_param_limits_from_primary()

    def _refresh_channel_combos(self) -> None:
        if self._channel_sync_in_progress:
            return
        if self.current_data is None:
            return
        self._channel_sync_in_progress = True
        try:
            channel_columns = self._numeric_channel_columns()
            if not channel_columns:
                return
            for key in channel_columns:
                existing_label: str = str(self.channels.get(key, "")).strip()
                if not existing_label:
                    self.channels[key] = key
                if key not in self.channel_units:
                    self.channel_units[key] = ""

            primary_target: str = self._primary_target_channel()
            x_fallback: None | str = "TIME" if "TIME" in channel_columns else None
            if x_fallback is None:
                for col in channel_columns:
                    if col != primary_target:
                        x_fallback = col
                        break
                if x_fallback is None:
                    x_fallback = channel_columns[0]
            x_choice = (
                self.x_channel if self.x_channel in channel_columns else x_fallback
            )
            if x_choice == primary_target and len(channel_columns) > 1:
                for col in channel_columns:
                    if col != primary_target:
                        x_choice = col
                        break
            self.x_channel = x_choice
            if hasattr(self, "function_input"):
                expr_text: str = self._expression_editor_text()
                if expr_text:
                    try:
                        resolved_target, seg_exprs = self._parse_equation_text(
                            expr_text, strict=False
                        )
                        normalized: str = (
                            f"{resolved_target} = {' ; '.join(seg_exprs)}"
                        )
                        if normalized != expr_text:
                            self.current_expression: str = normalized
                            self._set_expression_editor_text(normalized)
                    except Exception:
                        pass
            if hasattr(self, "x_channel_combo"):
                self.x_channel_combo.blockSignals(True)
                self.x_channel_combo.clear()
                for col in channel_columns:
                    self.x_channel_combo.addItem(self._channel_display_name(col), col)
                x_idx: int = self.x_channel_combo.findData(self.x_channel)
                if x_idx >= 0:
                    self.x_channel_combo.setCurrentIndex(x_idx)
                self.x_channel_combo.blockSignals(False)
            self._sync_breakpoint_sliders_from_state()
            self._rebuild_channel_token_buttons()
        finally:
            self._channel_sync_in_progress = False
        self._rebuild_channel_visibility_toggles()
        self._rebuild_channel_names_panel()

    def _rebuild_channel_visibility_toggles(self) -> None:
        """Rebuild the channel-visibility toggle buttons from current data columns."""
        layout: Any | None = getattr(self, "plot_channel_toggles_layout", None)
        buttons_layout: Any | None = getattr(
            self, "plot_channel_toggles_buttons_layout", None
        )
        container: Any | None = getattr(self, "plot_channel_toggle_container", None)
        toggle_action: Any | None = getattr(self, "_toolbar_toggle_action", None)
        if layout is None or container is None:
            return
        if buttons_layout is None:
            buttons_layout = layout

        def _set_toggles_visible(vis: bool) -> None:
            container.setVisible(vis)
            if toggle_action is not None:
                toggle_action.setVisible(vis)

        # Remove existing toggle buttons.
        clear_layout(buttons_layout)
        for btn in list(self._channel_toggle_buttons.values()):
            btn.deleteLater()
        self._channel_toggle_buttons.clear()

        has_status_area: bool = bool(
            getattr(self, "plot_toolbar_status_widget", None) is not None
        )
        if self.current_data is None:
            _set_toggles_visible(has_status_area)
            return

        channel_names = [
            key for key in self._numeric_channel_columns() if key != self.x_channel
        ]

        if len(channel_names) <= 1:
            _set_toggles_visible(has_status_area)
            return

        # Preserve existing visibility state; default new channels to visible.
        for name in channel_names:
            if name not in self._channel_visibility:
                self._channel_visibility[name] = True

        buttons_layout.addWidget(
            self._new_label(
                "Show:",
                style_sheet="font-weight: 600; color: #475569; font-size: 11px;",
            )
        )
        for name in channel_names:
            color: str = self._channel_plot_color(name)
            visible = self._channel_visibility.get(name, True)
            btn = QPushButton(self._channel_display_name(name))
            btn.setCheckable(True)
            btn.setChecked(visible)
            btn.setFixedHeight(18)
            btn.setStyleSheet(self._channel_toggle_stylesheet(color, visible))
            btn.setProperty("_ch_name", name)
            btn.setProperty("_ch_color", color)
            btn.toggled.connect(self._on_channel_visibility_toggled)
            buttons_layout.addWidget(btn)
            self._channel_toggle_buttons[name] = btn

        _set_toggles_visible(True)

    @staticmethod
    def _channel_toggle_stylesheet(color, checked) -> str:
        if checked:
            return (
                f"QPushButton {{ background: {color}; color: #fff; border: 1px solid {color};"
                " border-radius: 3px; padding: 0px 6px; font-size: 11px; font-weight: 600; }"
                " QPushButton:hover { opacity: 0.85; }"
            )
        return (
            "QPushButton { background: #f1f5f9; color: #94a3b8; border: 1px solid #cbd5e1;"
            " border-radius: 3px; padding: 0px 6px; font-size: 11px; }"
            " QPushButton:hover { background: #e2e8f0; }"
        )

    def _on_channel_visibility_toggled(self) -> None:
        btn: QObject | None = self.sender()
        if btn is None:
            return
        ch_name = btn.property("_ch_name")
        ch_color = btn.property("_ch_color")
        checked = btn.isChecked()
        if ch_name:
            self._channel_visibility[str(ch_name)] = checked
        if ch_color:
            btn.setStyleSheet(self._channel_toggle_stylesheet(str(ch_color), checked))
        self.update_plot(fast=False, preserve_view=False)

    def _is_channel_visible(self, channel_name):
        """Return True if a channel should be plotted."""
        return self._channel_visibility.get(str(channel_name).strip(), True)

    def _channel_plot_color(self, channel_name) -> str:
        """Return a stable palette color for *channel_name* based on data column order."""
        if self.current_data is not None:
            idx = 0
            for key in self._numeric_channel_columns():
                if not key or key == self.x_channel:
                    continue
                if key == str(channel_name).strip():
                    return palette_color(idx)
                idx += 1
        return palette_color(0)

    def _fit_companion_color(self, channel_name) -> str:
        """Return a companion colour for fit curves that is similar but distinct."""
        if self.current_data is not None:
            idx = 0
            for key in self._numeric_channel_columns():
                if not key or key == self.x_channel:
                    continue
                if key == str(channel_name).strip():
                    return fit_companion_color(idx)
                idx += 1
        return fit_companion_color(0)

    def _on_x_channel_changed(self, _index) -> None:
        if self._channel_sync_in_progress:
            return
        if not hasattr(self, "x_channel_combo"):
            return
        data = self.x_channel_combo.currentData()
        if not data:
            return
        self.x_channel = str(data)
        primary_target: str = self._primary_target_channel()
        if self.x_channel == primary_target and self.current_data is not None:
            for col in self._available_channel_names():
                if col != primary_target:
                    self.x_channel = col
                    idx: int = self.x_channel_combo.findData(self.x_channel)
                    if idx >= 0:
                        self.x_channel_combo.blockSignals(True)
                        self.x_channel_combo.setCurrentIndex(idx)
                        self.x_channel_combo.blockSignals(False)
                    break
        self._sync_breakpoint_sliders_from_state()
        self._rebuild_channel_visibility_toggles()
        self.update_plot(fast=False, preserve_view=False)

    def _set_toolbar_home_limits(
        self,
        home_main_xlim,
        home_main_ylim,
        home_residual_ylim=None,
        *,
        keep_current_view=False,
    ):
        toolbar: Any | None = getattr(self, "toolbar", None)
        if toolbar is None or not hasattr(self, "ax") or self.ax is None:
            return
        current_main_xlim = None
        current_main_ylim = None
        current_residual_ylim = None
        if keep_current_view:
            try:
                current_main_xlim: Tuple[float, ...] = tuple(self.ax.get_xlim())
            except Exception:
                current_main_xlim = None
            try:
                current_main_ylim: Tuple[float, ...] = tuple(self.ax.get_ylim())
            except Exception:
                current_main_ylim = None
            if hasattr(self, "ax_residual") and self.ax_residual is not None:
                try:
                    current_residual_ylim: Tuple[float, ...] = tuple(
                        self.ax_residual.get_ylim()
                    )
                except Exception:
                    current_residual_ylim = None
        try:
            if home_main_xlim is not None:
                self.ax.set_xlim(*home_main_xlim)
            if home_main_ylim is not None:
                self.ax.set_ylim(*home_main_ylim)
            if (
                home_residual_ylim is not None
                and hasattr(self, "ax_residual")
                and self.ax_residual is not None
            ):
                self.ax_residual.set_ylim(*home_residual_ylim)

            nav_stack: Any | None = getattr(toolbar, "_nav_stack", None)
            if nav_stack is not None:
                nav_stack.clear()
            push_current: Any | None = getattr(toolbar, "push_current", None)
            if callable(push_current):
                push_current()
            set_history_buttons: Any | None = getattr(
                toolbar, "set_history_buttons", None
            )
            if callable(set_history_buttons):
                set_history_buttons()
        except Exception:
            return
        if not keep_current_view:
            return
        try:
            if current_main_xlim is not None:
                self.ax.set_xlim(*current_main_xlim)
            if current_main_ylim is not None:
                self.ax.set_ylim(*current_main_ylim)
            if (
                current_residual_ylim is not None
                and hasattr(self, "ax_residual")
                and self.ax_residual is not None
            ):
                self.ax_residual.set_ylim(*current_residual_ylim)
        except Exception:
            pass

    def _reset_plot_home_view(self) -> None:
        """Reset toolbar Home target to the current plotted extents."""
        if not hasattr(self, "ax") or self.ax is None:
            return
        main_xlim = None
        main_ylim = None
        residual_ylim = None
        try:
            main_xlim: Tuple[float, ...] = tuple(self.ax.get_xlim())
        except Exception:
            pass
        try:
            main_ylim: Tuple[float, ...] = tuple(self.ax.get_ylim())
        except Exception:
            pass
        if hasattr(self, "ax_residual") and self.ax_residual is not None:
            try:
                residual_ylim: Tuple[float, ...] = tuple(self.ax_residual.get_ylim())
            except Exception:
                residual_ylim = None
        self._set_toolbar_home_limits(main_xlim, main_ylim, residual_ylim)

    def create_batch_controls_frame(self, parent_layout) -> None:
        """Create batch-only controls (shared params/settings are above tabs)."""
        group = QGroupBox("")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        batch_label: QLabel = self._new_label(
            "Batch Actions",
            style_sheet="font-weight: 600; color: #374151; padding: 1px 2px;",
        )
        layout.addWidget(batch_label)

        self.run_batch_btn_default_text = "Run Batch"
        self.run_batch_btn: QToolButton = QToolButton()
        self.run_batch_btn.setObjectName("actionButton")
        self.run_batch_btn.setText(self.run_batch_btn_default_text)
        self.run_batch_btn.setToolTip(
            "Click to run batch. Use the arrow to choose Straightforward or Procedure mode."
        )
        self.run_batch_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.run_batch_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.run_batch_btn.clicked.connect(self.run_batch_fit)
        self.run_batch_mode_menu = QMenu(self.run_batch_btn)
        self._batch_fit_mode_actions = {}
        for mode_label, mode_value in (
            ("Straightforward", "fit"),
            ("Procedure", "procedure"),
        ):
            action: QAction | None = self.run_batch_mode_menu.addAction(mode_label)
            if action is None:
                continue
            action.triggered.connect(
                lambda _checked=False, m=mode_value: self._set_batch_fit_mode(m)
            )
            self._batch_fit_mode_actions[str(mode_value)] = action
        self.run_batch_btn.setMenu(self.run_batch_mode_menu)
        self._set_batch_fit_mode(self._batch_fit_run_mode, autosave=False)
        self._set_split_action_min_width(
            self.run_batch_btn,
            [
                self._batch_fit_button_text_for_mode("fit"),
                self._batch_fit_button_text_for_mode("procedure"),
                "Force Stop",
            ],
            extra_px=42,
        )

        regex_layout = QHBoxLayout()
        regex_layout.setSpacing(4)
        self.regex_input: QLineEdit = self._new_line_edit(
            "",
            placeholder="Example: data_{freq}_{idx}_ALL",
            tooltip=(
                "Template mode: use {field} placeholders (and * wildcard).\n"
                "Optional with default: {field=default} (e.g. filename_{idx=0}).\n"
                "Optional affix with default: {field=default|prefix|suffix}.\n"
                "Optional affix without default: {field|prefix|suffix} "
                "(e.g. filename{ver|_v} or filename{ver||)}).\n"
                "If both prefix and suffix are provided, either side may be absent."
            ),
            text_changed=self._on_regex_changed,
        )
        regex_layout.addWidget(self.regex_input)
        layout.addLayout(regex_layout)

        self.batch_parse_feedback_label: QLabel = self._new_label(
            "",
            object_name="statusLabel",
        )
        self.batch_parse_feedback_label.hide()
        layout.addWidget(self.batch_parse_feedback_label)

        self.capture_mapping_widget = QWidget()
        self.capture_mapping_layout = QGridLayout(self.capture_mapping_widget)
        self.capture_mapping_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_mapping_layout.setHorizontalSpacing(6)
        self.capture_mapping_layout.setVerticalSpacing(2)
        layout.addWidget(self.capture_mapping_widget)
        self._refresh_param_capture_mapping_controls()

        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)
        actions_row.addWidget(self.run_batch_btn)
        self.cancel_batch_btn: QPushButton = self._new_button(
            "Cancel",
            handler=self.cancel_batch_fit,
            enabled=False,
        )
        actions_row.addWidget(self.cancel_batch_btn)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.batch_status_label: QLabel = self._new_label("", object_name="statusLabel")
        self.batch_status_label.hide()
        layout.addWidget(self.batch_status_label)
        layout.addStretch(1)

        group.setLayout(layout)
        parent_layout.addWidget(group, 1)

    def create_batch_results_frame(self, parent_layout) -> None:
        """Create batch results table."""
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(0)
        self.batch_table.setRowCount(0)
        self.batch_table.cellClicked.connect(self._on_batch_table_cell_clicked)
        self.batch_table_header: RichTextHeaderView = RichTextHeaderView(
            Qt.Orientation.Horizontal, self.batch_table
        )
        self.batch_table.setHorizontalHeader(self.batch_table_header)
        batch_header: QHeaderView | None = self.batch_table.horizontalHeader()
        batch_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        batch_header.setStretchLastSection(True)
        batch_header.setMinimumSectionSize(60)
        v_header: QHeaderView | None = self.batch_table.verticalHeader()
        v_header.setVisible(True)
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        v_header.setMinimumSectionSize(self.batch_row_height_min)
        v_header.setToolTip("Drag row borders to resize all rows")
        v_header.sectionResized.connect(self._on_batch_row_resized_by_user)
        # Apply a uniform row height for plot previews.
        v_header.setDefaultSectionSize(self._current_batch_row_height())
        self.batch_table.setSortingEnabled(True)
        batch_header.setSortIndicatorShown(True)
        batch_header.setSectionsClickable(True)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.verticalScrollBar().valueChanged.connect(
            self.queue_visible_thumbnail_render
        )
        parent_layout.addWidget(self.batch_table)

    def create_batch_analysis_frame(self, parent_layout) -> None:
        """Create interactive batch analysis plot controls."""
        group = QGroupBox("")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        source_row = QHBoxLayout()
        source_row.setSpacing(4)
        self.analysis_status_label: QLabel = self._new_label(
            "Total: 0 | With Results: 0",
            object_name="statusLabel",
        )
        source_row.addWidget(self.analysis_status_label, 1)
        layout.addLayout(source_row)

        axis_row = QHBoxLayout()
        axis_row.setSpacing(4)
        axis_row.addWidget(self._new_label("Field (X):"))
        self.analysis_x_combo: RichTextComboBox | QComboBox = self._new_combobox(
            current_index_changed=self.update_batch_analysis_plot,
            rich_text=True,
            fixed_width=220,
        )
        axis_row.addWidget(self.analysis_x_combo)
        axis_row.addWidget(self._new_label("Plot:"))
        self.analysis_mode_combo: RichTextComboBox | QComboBox = self._new_combobox(
            items=[
                ("Combined", "combined"),
                ("One per parameter", "separate"),
            ],
            current_index_changed=self.update_batch_analysis_plot,
            fixed_width=170,
        )
        axis_row.addWidget(self.analysis_mode_combo)
        axis_row.addWidget(self._new_label("Layers:"))
        self.analysis_show_points_btn: QPushButton = self._new_button(
            "Points",
            checkable=True,
            checked=True,
            toggled_handler=self.update_batch_analysis_plot,
        )
        axis_row.addWidget(self.analysis_show_points_btn)

        self.analysis_show_series_line_btn: QPushButton = self._new_button(
            "Series Line",
            checkable=True,
            checked=False,
            toggled_handler=self.update_batch_analysis_plot,
        )
        axis_row.addWidget(self.analysis_show_series_line_btn)

        self.analysis_fit_line_btn: QPushButton = self._new_button(
            "Best-Fit Lines",
            checkable=True,
            checked=True,
            toggled_handler=self.update_batch_analysis_plot,
        )
        axis_row.addWidget(self.analysis_fit_line_btn)

        self.analysis_legend_btn: QPushButton = self._new_button(
            "Legend",
            checkable=True,
            checked=True,
            toggled_handler=self.update_batch_analysis_plot,
        )
        axis_row.addWidget(self.analysis_legend_btn)

        self.analysis_log_x_btn: QPushButton = self._new_button(
            "Log X",
            checkable=True,
            checked=False,
            toggled_handler=self.update_batch_analysis_plot,
            tooltip="Use logarithmic scaling on the X axis.",
        )
        axis_row.addWidget(self.analysis_log_x_btn)
        axis_row.addStretch(1)
        layout.addLayout(axis_row)

        math_row = QHBoxLayout()
        math_row.setSpacing(4)
        math_row.addWidget(self._new_label("Derived (Y):"))
        self.analysis_math_enable_btn: QCheckBox = self._new_checkbox(
            "Use",
            checked=False,
            toggled_handler=self._on_analysis_math_controls_changed,
            tooltip="Plot a derived series from two fit parameters.",
        )
        math_row.addWidget(self.analysis_math_enable_btn)
        self.analysis_math_left_combo: RichTextComboBox | QComboBox = (
            self._new_combobox(
                current_index_changed=self._on_analysis_math_controls_changed,
                rich_text=True,
                fixed_width=190,
            )
        )
        math_row.addWidget(self.analysis_math_left_combo)
        self.analysis_math_op_combo: RichTextComboBox | QComboBox = self._new_combobox(
            items=[
                ("+", "+"),
                ("-", "-"),
                ("*", "*"),
                ("/", "/"),
            ],
            current_data="-",
            current_index_changed=self._on_analysis_math_controls_changed,
            fixed_width=64,
        )
        math_row.addWidget(self.analysis_math_op_combo)
        self.analysis_math_right_combo: RichTextComboBox | QComboBox = (
            self._new_combobox(
                current_index_changed=self._on_analysis_math_controls_changed,
                rich_text=True,
                fixed_width=190,
            )
        )
        math_row.addWidget(self.analysis_math_right_combo)
        self.analysis_math_enable_btn.setEnabled(False)
        self.analysis_math_left_combo.setEnabled(False)
        self.analysis_math_op_combo.setEnabled(False)
        self.analysis_math_right_combo.setEnabled(False)
        math_row.addStretch(1)
        layout.addLayout(math_row)

        self.analysis_param_buttons = {}
        self.analysis_params_widget = QWidget()
        self.analysis_params_widget.setObjectName("analysisParamsPanel")
        self.analysis_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.analysis_params_widget.setStyleSheet(
            "QWidget#analysisParamsPanel { "
            "border: 1px solid #e5e7eb; border-radius: 6px; background: #f8fafc; }"
        )
        self.analysis_params_button_layout = QVBoxLayout(self.analysis_params_widget)
        self.analysis_params_button_layout.setContentsMargins(6, 4, 6, 4)
        self.analysis_params_button_layout.setSpacing(6)
        layout.addWidget(self.analysis_params_widget)

        self.analysis_fig = Figure(figsize=(10, 3.2), dpi=100)
        self.analysis_fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.2)
        self.analysis_canvas: FigureCanvas = FigureCanvas(self.analysis_fig)
        self._analysis_axis_height_px = 240
        self._analysis_canvas_margin_px = 80
        self.analysis_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.analysis_scroll_area = QScrollArea()
        self.analysis_scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.analysis_scroll_area.setWidgetResizable(True)
        self.analysis_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.analysis_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.analysis_scroll_area.setWidget(self.analysis_canvas)
        self.analysis_scroll_area.setMinimumHeight(280)
        self._set_analysis_canvas_height(1)
        if self._analysis_point_pick_cid is None:
            self._analysis_point_pick_cid: int = self.analysis_canvas.mpl_connect(
                "pick_event",
                self._on_analysis_point_picked,
            )
        if self._analysis_hover_cid is None:
            self._analysis_hover_cid: int = self.analysis_canvas.mpl_connect(
                "motion_notify_event",
                self._on_analysis_plot_hover,
            )
        if self._analysis_hover_leave_cid is None:
            self._analysis_hover_leave_cid: int = self.analysis_canvas.mpl_connect(
                "figure_leave_event",
                self._on_analysis_plot_leave,
            )
        layout.addWidget(self.analysis_scroll_area, 1)

        group.setLayout(layout)
        parent_layout.addWidget(group, 1)

    # ── Procedure tab ───────────────────────────────────────────────

    def _make_procedure_host(self) -> ProcedureHost:
        """Create a ProcedureHost adapter that delegates to this GUI."""
        gui = self

        class _Host(ProcedureHost):
            def proc_available_params(self) -> List[Any] | List[str]:
                return gui._proc_available_params()

            def proc_available_channels(self):
                return gui._proc_available_channels()

            def proc_available_capture_keys(self):
                return gui._available_capture_keys()

            def proc_capture_preview_values(self):
                return gui._capture_preview_values()

            def proc_build_fit_context(self, fixed_params=None):
                return gui._build_fit_context(
                    fixed_params=fixed_params,
                    respect_enabled_channels=False,
                )

            def proc_get_multi_channel_model(self) -> Any | None:
                return getattr(gui, "_multi_channel_model", None)

            def proc_get_piecewise_model(self) -> Any | None:
                return getattr(gui, "_piecewise_model", None)

            def proc_get_current_data(self) -> Any | None:
                return getattr(gui, "data", None)

            def proc_get_x_channel(self) -> str:
                return str(getattr(gui, "x_channel", "TIME"))

            def proc_get_boundary_ratios_per_channel(self):
                gui._refresh_boundary_state_topology(preserve_existing=True)
                return gui._fit_state.as_per_channel_map()

            def proc_available_boundary_groups(self):
                return gui._proc_available_boundary_groups()

            def proc_get_smoothing(self) -> Tuple[bool, int]:
                enabled = bool(getattr(gui, "smoothing_enabled", False))
                window = int(getattr(gui, "smoothing_window", 1) or 1)
                return (enabled, window)

            def proc_get_random_restarts(self) -> int:
                spin: Any | None = getattr(gui, "random_restart_spinbox", None)
                return int(spin.value()) if spin is not None else 0

            def proc_channel_display_name(self, channel_key: str) -> str:
                return gui._channel_display_name(channel_key)

            def proc_display_symbol_html(self, key) -> str:
                return gui._display_symbol_for_param_html(key)

            def proc_on_fit_finished(self, result) -> None:
                gui.on_fit_finished(result)

            def proc_autosave(self) -> None:
                gui._autosave_fit_details()

            def proc_log(self, message) -> None:
                if hasattr(gui, "stats_text"):
                    gui.stats_text.append(str(message))

            def proc_current_dir(self) -> str:
                d: Any | None = getattr(gui, "current_dir", None)
                return str(Path(d).expanduser()) if d else str(Path.cwd())

        return _Host()

    def _proc_available_params(self) -> List[Any] | List[str]:
        """Return list of currently available parameter keys."""
        multi: Any | None = getattr(self, "_multi_channel_model", None)
        if multi is not None:
            return list(multi.global_param_names)
        model: Any | None = getattr(self, "_piecewise_model", None)
        if model is not None:
            return list(model.global_param_names)
        return [spec.key for spec in self.param_specs]

    def _proc_available_channels(self):
        """Return list of currently available target channels."""
        multi: Any | None = getattr(self, "_multi_channel_model", None)
        if multi is not None:
            return list(multi.target_channels)
        model: Any | None = getattr(self, "_piecewise_model", None)
        if model is not None:
            return [model.target_col]
        primary_target: str = self._primary_target_channel()
        return [primary_target] if primary_target else []

    def _proc_available_boundary_groups(self):
        """Return boundary groups as [(name, ((target, idx), ...)), ...]."""
        multi: Any | None = getattr(self, "_multi_channel_model", None)
        if multi is None:
            return []
        name_map: Any | Dict[Any, Any] = getattr(self, "_boundary_name_map", {}) or {}
        groups: Dict[str, List[Tuple[str, int]]] = {}
        for target, bidx in list(getattr(multi, "all_boundary_ids", ()) or ()):
            bid = (target, int(bidx))
            name = str(name_map.get(bid, format_boundary_display_name(int(bidx))))
            groups.setdefault(name, []).append((str(target), int(bidx)))
        out = []
        for name in sorted(groups.keys()):
            members: Tuple[Tuple[str, int], ...] = tuple(groups[name])
            if members:
                out.append((name, members))
        return out

    def _batch_row_error_text(self, row) -> str:
        pattern_error: str = str(row.get("pattern_error") or "").strip()
        fit_error: str = str(fit_get(row, "error") or "").strip()
        is_stale = bool(row.get("_equation_stale"))

        normalized_fit_error: str = fit_error.lower().replace(".", "").strip()
        if normalized_fit_error in {"cancelled", "canceled"}:
            fit_error: str = ""

        parts = []
        if is_stale:
            parts.append("Stale fit (equation changed)")
        if pattern_error:
            parts.append(pattern_error)
        if fit_error and fit_error not in parts:
            parts.append(fit_error)
        return " | ".join(parts)

    def _batch_row_has_stored_result(self, row, *, include_errors: bool = True) -> bool:
        if not isinstance(row, Mapping):
            return False
        if include_errors and fit_get(row, "error") not in (None, ""):
            return True
        return bool(
            has_nonempty_values(fit_get(row, "params"))
            or (finite_float_or_none(fit_get(row, "r2")) is not None)
            or bool(fit_get(row, "channel_results"))
            or bool(row.get("_procedure_result"))
        )

    def _fit_param_range_violations(self, params):
        values: (
            np.ndarray[Tuple[int, ...], np.dtype[Any]]
            | np.ndarray[Tuple[int], np.dtype[Any]]
        ) = self._as_float_array(params)
        if values.size <= 0:
            return []
        violations = []
        for idx, spec in enumerate(self.param_specs):
            if idx >= values.size:
                break
            value = float(values[idx])
            if not np.isfinite(value):
                continue
            low = float(min(spec.min_value, spec.max_value))
            high = float(max(spec.min_value, spec.max_value))
            tolerance: float = 1e-12 * max(1.0, abs(low), abs(high))
            if value < (low - tolerance) or value > (high + tolerance):
                violations.append(
                    {
                        "index": int(idx),
                        "key": str(spec.key),
                        "value": float(value),
                        "low": float(low),
                        "high": float(high),
                    }
                )
        return violations

    def _fit_param_range_error_text(self, violations) -> None | str:
        rows = list(violations or [])
        if not rows:
            return None
        samples = []
        for item in rows[:3]:
            key = str(item.get("key") or "")
            label: str = self._display_name_for_param_key(key) if key else key
            value = float(item.get("value"))
            low = float(item.get("low"))
            high = float(item.get("high"))
            samples.append(f"{label}={value:.6g} not in [{low:.6g}, {high:.6g}]")
        remainder: int = len(rows) - len(samples)
        suffix: str = f" (+{remainder} more)" if remainder > 0 else ""
        return f"{_FIT_PARAM_RANGE_ERROR_PREFIX} {'; '.join(samples)}{suffix}"

    def _apply_param_range_validation_to_row(self, row):
        normalized = canonicalize_fit_row(row)
        violations = self._fit_param_range_violations(fit_get(normalized, "params"))
        violation_keys: List[str] = [
            str(item.get("key")) for item in violations if item.get("key")
        ]
        normalized["_param_range_violation_keys"] = violation_keys

        range_error: None | str = self._fit_param_range_error_text(violations)
        existing_parts: List[str] = [
            part.strip()
            for part in str(fit_get(normalized, "error") or "").split("|")
            if str(part).strip()
        ]
        existing_parts: List[str] = [
            part
            for part in existing_parts
            if not str(part).startswith(_FIT_PARAM_RANGE_ERROR_PREFIX)
        ]
        if range_error:
            existing_parts.append(range_error)
        fit_set(
            normalized,
            "error",
            " | ".join(dict.fromkeys(existing_parts)) if existing_parts else None,
        )
        return normalized

    def _current_file_param_violation_keys(self):
        file_path: Any | None = self._current_loaded_file_path()
        if not file_path:
            return set()
        row_index: None | int = self._find_batch_result_index_by_file(file_path)
        if row_index is None or row_index < 0 or row_index >= len(self.batch_results):
            return set()
        row = self.batch_results[row_index]
        violations = self._fit_param_range_violations(fit_get(row, "params"))
        return {str(item.get("key")) for item in violations if item.get("key")}

    def _refresh_param_value_error_highlighting(self) -> None:
        invalid_by_row = self._current_file_param_violation_keys()
        for spec in self.param_specs:
            key = str(spec.key)
            box = self.param_spinboxes.get(key)
            if box is None:
                continue
            numeric: float | None = finite_float_or_none(box.value())
            low = float(min(spec.min_value, spec.max_value))
            high = float(max(spec.min_value, spec.max_value))
            tolerance: float = 1e-12 * max(1.0, abs(low), abs(high))
            out_of_spec: bool = numeric is not None and (
                numeric < (low - tolerance) or numeric > (high + tolerance)
            )
            invalid = bool(out_of_spec or key in invalid_by_row)
            box.setStyleSheet("color: #b91c1c;" if invalid else "")

    def _x_values_for_analysis_row(self, row) -> np.ndarray[Tuple[int], np.dtype[Any]]:
        file_ref: str = str(row.get("file") or row.get("__file_ref") or "").strip()
        if not file_ref:
            return np.asarray([], dtype=float)
        x_channel: str = str(row.get("x_channel") or self.x_channel or "").strip()
        if not x_channel:
            return np.asarray([], dtype=float)

        file_key: str = self._fit_task_file_key(file_ref)
        cache_key: Tuple[str, str] = (file_key or file_ref, x_channel)
        cached = self._analysis_row_x_cache.get(cache_key)
        if cached is not None:
            return np.asarray(cached, dtype=float).reshape(-1)

        x_values: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray([], dtype=float)
        current_file: Any | None = self._current_loaded_file_path()
        if (
            current_file
            and file_key
            and self._fit_task_file_key(current_file) == file_key
            and self.current_data is not None
        ):
            try:
                x_values = np.asarray(
                    self._get_channel_data(x_channel), dtype=float
                ).reshape(-1)
            except Exception:
                x_values = np.asarray([], dtype=float)
        else:
            frame = self._data_preload_cache.get(file_ref)
            if frame is None and file_ref not in self._data_preload_failed:
                try:
                    frame = read_measurement_csv(file_ref)
                    self._data_preload_cache[file_ref] = frame
                except Exception:
                    self._data_preload_failed.add(file_ref)
            if frame is not None:
                try:
                    source_col = (
                        x_channel
                        if x_channel in frame.columns
                        else str(frame.columns[0])
                    )
                    x_values = np.asarray(
                        frame[source_col].to_numpy(dtype=float, copy=False),
                        dtype=float,
                    ).reshape(-1)
                except Exception:
                    x_values = np.asarray([], dtype=float)

        x_finite = x_values[np.isfinite(x_values)]
        self._analysis_row_x_cache[cache_key] = np.asarray(x_finite, dtype=float)
        return np.asarray(x_finite, dtype=float).reshape(-1)

    def _channel_boundary_values_for_analysis_row(
        self,
        row,
        target: str,
    ) -> np.ndarray[Tuple[int], np.dtype[Any]]:
        target_key: str = str(target or "").strip()
        if not target_key:
            return np.asarray([], dtype=float)

        ch_results_raw = fit_get(row, "channel_results")
        if not isinstance(ch_results_raw, Mapping):
            return np.asarray([], dtype=float)
        ch_data = ch_results_raw.get(target_key)
        if not isinstance(ch_data, Mapping):
            # Normalize target matching for loaded payloads where channel keys
            # may differ by whitespace/case.
            for raw_target, raw_entry in ch_results_raw.items():
                raw_target_text: str = str(raw_target).strip()
                if raw_target_text == target_key:
                    target_key = raw_target_text
                    ch_data = raw_entry
                    break
                if raw_target_text.lower() == target_key.lower():
                    target_key = raw_target_text
                    ch_data = raw_entry
                    break
            if not isinstance(ch_data, Mapping):
                return np.asarray([], dtype=float)

        boundaries = self._as_float_array(ch_data.get("boundaries"))
        if boundaries.size > 0:
            return boundaries

        ratios = self._as_float_array(ch_data.get("boundary_ratios"))
        if ratios.size <= 0:
            return np.asarray([], dtype=float)

        x_values = self._x_values_for_analysis_row(row)
        if x_values.size <= 0:
            return np.asarray([], dtype=float)

        try:
            computed: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                boundary_ratios_to_x_values(
                    ratios,
                    x_values,
                    int(ratios.size),
                ),
                dtype=float,
            ).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)
        if computed.size <= 0:
            return np.asarray([], dtype=float)

        # Cache computed channel boundaries on the row so follow-up analysis
        # refreshes do not need to recompute/read source data.
        if isinstance(ch_results_raw, dict):
            updated_entry: Dict[str, Any] = dict(ch_data)
            updated_entry["boundaries"] = np.asarray(computed, dtype=float).copy()
            ch_results_raw[target_key] = updated_entry
            fit_set(row, "channel_results", ch_results_raw)

        return np.asarray(computed, dtype=float).reshape(-1)

    def _extract_analysis_records_from_batch(self):
        records = []
        param_columns = self._batch_parameter_column_items()
        for row in self.batch_results:
            record = {
                "File": stem_for_file_ref(row["file"]),
                "__file_ref": row["file"],
            }
            captures = row.get("captures") or {}
            for key, value in captures.items():
                record[key] = value
            params: (
                np.ndarray[Tuple[int, ...], np.dtype[Any]]
                | np.ndarray[Tuple[int], np.dtype[Any]]
            ) = self._as_float_array(fit_get(row, "params"))
            boundary_targets: Set[str] = {
                str(item.get("target") or "").strip()
                for item in param_columns
                if str(item.get("kind") or "") == "boundary"
                and str(item.get("target") or "").strip()
            }
            ch_boundary_values: Dict[str, np.ndarray] = {}
            for target in boundary_targets:
                ch_vals = self._channel_boundary_values_for_analysis_row(row, target)
                if ch_vals.size > 0:
                    ch_boundary_values[target] = ch_vals
            for item in param_columns:
                idx = int(item["index"])
                if item["kind"] == "param":
                    value: float | None = (
                        float(params[idx]) if params.size > idx else None
                    )
                else:
                    target_col = str(item.get("target") or "").strip()
                    if target_col:
                        bv_arr = ch_boundary_values.get(target_col)
                        if bv_arr is None:
                            bv_arr = np.asarray([], dtype=float)
                    else:
                        bv_arr = np.asarray([], dtype=float)
                    value: float | None = (
                        float(bv_arr[idx])
                        if bv_arr.size > idx
                        else None
                    )
                record[str(item["key"])] = value
            record["R2"] = fit_get(row, "r2")
            record["Error"] = self._batch_row_error_text(row)
            records.append(record)
        return records

    def _extract_analysis_columns(self, records):
        columns = []
        for row in records:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
        return columns

    def _coerce_numeric_array(
        self, values
    ) -> np.ndarray[Tuple[int, ...], np.dtype[Any]]:
        numeric = []
        for value in values:
            if value is None:
                numeric.append(np.nan)
                continue
            text: str = str(value).strip()
            if text == "":
                numeric.append(np.nan)
                continue
            try:
                numeric.append(float(text))
            except Exception:
                numeric.append(np.nan)
        return np.asarray(numeric, dtype=float)

    def _analysis_field_display_text(
        self, field_key
    ) -> Tuple[Literal[""], Literal[""]] | Tuple[str, str]:
        key_text: str = str(field_key).strip()
        if not key_text:
            return ("", "")
        if key_text == "R2":
            return ("R²", "R²")
        label_text: str = key_text
        for item in self._batch_parameter_column_items():
            item_key: str = str(item.get("key", "")).strip()
            if item_key != key_text:
                continue
            token: str = str(item.get("token", "")).strip()
            if token:
                label_text: str = token
            break
        label_html: str = parameter_symbol_to_html(label_text) or html.escape(
            label_text
        )
        return (label_text, label_html)

    def _analysis_param_target_by_key(self) -> Dict[str, None | str]:
        """Map parameter key -> equation target channel used in parameter sliders."""
        target_by_key: Dict[str, None | str] = {}
        current_target: None | str = None
        for section in self._ordered_parameter_sections():
            kind: str = str(section.get("kind") or "")
            if kind == "channel_header":
                target_text: str = str(section.get("target") or "").strip()
                current_target = target_text or None
                continue
            if kind not in {"segment", "shared"}:
                continue
            section_target: None | str = current_target
            raw_target = section.get("target")
            if raw_target is not None:
                target_text = str(raw_target).strip()
                section_target = target_text or None
            if kind == "shared":
                section_target = None
            for raw_key in list(section.get("keys") or []):
                key_text = str(raw_key).strip()
                if key_text and key_text not in target_by_key:
                    target_by_key[key_text] = section_target
        return target_by_key

    def _analysis_param_sections(self):
        """Return grouped analysis Y-parameter keys by type and equation."""
        item_by_key: Dict[str, Dict[Any, Any]] = {
            str(item.get("key")): dict(item)
            for item in self._batch_parameter_column_items()
            if str(item.get("key") or "").strip()
        }
        param_target_by_key = self._analysis_param_target_by_key()
        target_order: Dict[str, int] = {}
        multi: Any | None = getattr(self, "_multi_channel_model", None)
        if multi is not None and getattr(multi, "target_channels", None):
            target_order = {
                str(target): idx
                for idx, target in enumerate(list(multi.target_channels))
            }
        elif getattr(self, "_piecewise_model", None) is not None:
            target_order = {str(self._piecewise_model.target_col): 0}

        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {
            "param": {},
            "metric": {},
        }
        boundaries_by_target: Dict[None | str, List[str]] = {}
        for raw_key in self.analysis_param_columns:
            key_text: str = str(raw_key or "").strip()
            if not key_text:
                continue
            if key_text == "R2":
                group_key: str = "__metrics__"
                group = grouped["metric"].setdefault(
                    group_key,
                    {
                        "group_key": group_key,
                        "title": "",
                        "target": None,
                        "keys": [],
                    },
                )
                group["keys"].append(key_text)
                continue

            item: Dict[Any, Any] = dict(item_by_key.get(key_text) or {})
            kind: str = str(item.get("kind") or "param")
            if kind == "boundary":
                target: None | str = (
                    str(item.get("target")).strip()
                    if item.get("target") not in (None, "")
                    else None
                )
                boundaries_by_target.setdefault(target, []).append(key_text)
                continue

            target = param_target_by_key.get(key_text)
            if target:
                group_key = f"target:{target}"
                title = self._channel_display_name(target)
            else:
                group_key = "__shared__"
                title = "Shared"
            group = grouped["param"].setdefault(
                group_key,
                {
                    "group_key": group_key,
                    "title": str(title),
                    "target": target,
                    "keys": [],
                },
            )
            group["keys"].append(key_text)

        # Append boundaries at the end of their corresponding parameter/equation groups.
        for target, boundary_keys in boundaries_by_target.items():
            if not boundary_keys:
                continue
            if target:
                group_key = f"target:{target}"
                title = self._channel_display_name(target)
            elif "__shared__" in grouped["param"]:
                group_key = "__shared__"
                title = str(grouped["param"][group_key].get("title") or "Shared")
            else:
                group_key = "__boundaries__"
                title = "Boundaries"
            group = grouped["param"].setdefault(
                group_key,
                {
                    "group_key": group_key,
                    "title": str(title),
                    "target": target if target else None,
                    "keys": [],
                },
            )
            group["keys"].extend(list(boundary_keys))

        def _sorted_groups(kind_name: str):
            values: List[Dict[str, Any]] = list(grouped[kind_name].values())
            if kind_name != "param":
                return values

            def _sort_key(group: Dict[str, Any]):
                target: None | str = group.get("target")
                if target:
                    return (
                        0,
                        int(target_order.get(str(target), 10_000)),
                        str(self._channel_display_name(target)).lower(),
                    )
                if str(group.get("group_key")) == "__shared__":
                    return (1, 0, "")
                return (2, 0, str(group.get("title") or "").lower())

            return sorted(values, key=_sort_key)

        sections = []
        for kind_name, title in (
            ("param", "Parameters"),
            ("metric", "Metrics"),
        ):
            groups = _sorted_groups(kind_name)
            if not groups:
                continue
            sections.append({"kind": kind_name, "title": title, "groups": groups})
        return sections

    def _default_analysis_x_field(self, numeric_columns):
        for key in self.batch_capture_keys:
            if key in numeric_columns:
                return key
        for key in numeric_columns:
            if key not in self.analysis_param_columns and key != "R2":
                return key
        return numeric_columns[0] if numeric_columns else None

    def _refresh_batch_analysis_data(self, preserve_selection) -> None:
        raw_records = self._extract_analysis_records_from_batch()

        records = list(raw_records)
        total_rows: int = len(records)
        rows_with_results: int = sum(
            1
            for row in list(getattr(self, "batch_results", []) or [])
            if self._batch_row_has_stored_result(row, include_errors=False)
        )
        self.analysis_status_label.setText(
            f"Total: {int(total_rows)} | With Results: {int(rows_with_results)}"
        )

        self.analysis_records = records
        self.analysis_columns = self._extract_analysis_columns(records)
        self.analysis_numeric_data = {}
        for column in self.analysis_columns:
            values = [row.get(column, "") for row in records]
            as_numeric: np.ndarray[Tuple[int, ...], np.dtype[Any]] = (
                self._coerce_numeric_array(values)
            )
            if np.isfinite(as_numeric).sum() > 0:
                self.analysis_numeric_data[column] = as_numeric

        numeric_columns = list(self.analysis_numeric_data.keys())
        self.analysis_param_columns: List[str] = [
            str(item["key"])
            for item in self._batch_parameter_column_items()
            if str(item["key"]) in self.analysis_numeric_data
        ]
        if (
            "R2" in self.analysis_numeric_data
            and "R2" not in self.analysis_param_columns
        ):
            self.analysis_param_columns.append("R2")
        if not self.analysis_param_columns:
            self.analysis_param_columns = [
                key for key in numeric_columns if key not in ("R2",)
            ]
        self._refresh_analysis_math_controls(
            preserve_selection=bool(preserve_selection)
        )

        previous_x: Any | None = (
            self.analysis_x_combo.currentData() if preserve_selection else None
        )
        previous_params = (
            set(self._selected_analysis_params()) if preserve_selection else set()
        )

        self.analysis_x_combo.blockSignals(True)
        self.analysis_x_combo.clear()
        if isinstance(self.analysis_x_combo, RichTextComboBox):
            self.analysis_x_combo.add_rich_item(
                "Select X Axis...", None, "Select X Axis..."
            )
        else:
            self.analysis_x_combo.addItem("Select X Axis...", None)
        for key in numeric_columns:
            plain_label, rich_label = self._analysis_field_display_text(key)
            if isinstance(self.analysis_x_combo, RichTextComboBox):
                self.analysis_x_combo.add_rich_item(plain_label, key, rich_label)
            else:
                self.analysis_x_combo.addItem(plain_label, key)
        self.analysis_x_combo.blockSignals(False)

        chosen_x: Any | None = (
            previous_x
            if (preserve_selection and previous_x in numeric_columns)
            else None
        )
        if chosen_x is None:
            chosen_x = self._default_analysis_x_field(numeric_columns)
        x_idx: int = self.analysis_x_combo.findData(chosen_x)
        if x_idx < 0:
            x_idx: int = self.analysis_x_combo.findData(None)
        if x_idx >= 0:
            self.analysis_x_combo.setCurrentIndex(x_idx)

        self._rebuild_analysis_param_buttons(previous_params)

        self.update_batch_analysis_plot()

    def _selected_analysis_params(self):
        return [
            key
            for key, button in self.analysis_param_buttons.items()
            if button.isChecked()
        ]

    def _analysis_math_operand_keys(self) -> List[str]:
        keys: List[str] = [
            str(key).strip()
            for key in list(getattr(self, "analysis_param_columns", []) or [])
            if str(key).strip()
        ]
        non_metric_keys: List[str] = [key for key in keys if key != "R2"]
        return non_metric_keys if non_metric_keys else keys

    def _refresh_analysis_math_controls(self, *, preserve_selection: bool) -> None:
        if not hasattr(self, "analysis_math_left_combo"):
            return

        left_combo = self.analysis_math_left_combo
        right_combo = self.analysis_math_right_combo
        op_combo = self.analysis_math_op_combo
        enable_btn = self.analysis_math_enable_btn

        operand_keys: List[str] = self._analysis_math_operand_keys()
        previous_left: Any | None = (
            left_combo.currentData() if preserve_selection else None
        )
        previous_right: Any | None = (
            right_combo.currentData() if preserve_selection else None
        )

        self._analysis_math_controls_refreshing = True
        try:
            for combo in (left_combo, right_combo):
                blocked = combo.blockSignals(True)
                combo.clear()
                if isinstance(combo, RichTextComboBox):
                    combo.add_rich_item(
                        "Select parameter...",
                        None,
                        "Select parameter...",
                    )
                else:
                    combo.addItem("Select parameter...", None)
                for key in operand_keys:
                    plain_label, rich_label = self._analysis_field_display_text(key)
                    if not plain_label:
                        plain_label = str(key)
                    if not rich_label:
                        rich_label = html.escape(plain_label)
                    if isinstance(combo, RichTextComboBox):
                        combo.add_rich_item(plain_label, key, rich_label)
                    else:
                        combo.addItem(plain_label, key)
                combo.blockSignals(blocked)

            chosen_left: Any | None = (
                previous_left if previous_left in operand_keys else None
            )
            chosen_right: Any | None = (
                previous_right if previous_right in operand_keys else None
            )
            if chosen_left is None and operand_keys:
                chosen_left = operand_keys[0]
            if chosen_right is None and operand_keys:
                chosen_right = (
                    operand_keys[1] if len(operand_keys) > 1 else operand_keys[0]
                )

            for combo, chosen in (
                (left_combo, chosen_left),
                (right_combo, chosen_right),
            ):
                blocked = combo.blockSignals(True)
                combo_idx: int = combo.findData(chosen)
                if combo_idx < 0:
                    combo_idx = combo.findData(None)
                if combo_idx >= 0:
                    combo.setCurrentIndex(combo_idx)
                combo.blockSignals(blocked)

            has_operands: bool = bool(operand_keys)
            left_combo.setEnabled(has_operands)
            right_combo.setEnabled(has_operands)
            op_combo.setEnabled(has_operands)
            enable_btn.setEnabled(has_operands)
            if not preserve_selection:
                blocked = enable_btn.blockSignals(True)
                enable_btn.setChecked(False)
                enable_btn.blockSignals(blocked)
            if not has_operands:
                blocked = enable_btn.blockSignals(True)
                enable_btn.setChecked(False)
                enable_btn.blockSignals(blocked)
        finally:
            self._analysis_math_controls_refreshing = False

    def _on_analysis_math_controls_changed(self, *_args) -> None:
        if getattr(self, "_analysis_math_controls_refreshing", False):
            return
        self.update_batch_analysis_plot()

    def _analysis_math_series_spec(self) -> None | Dict[str, Any]:
        if not hasattr(self, "analysis_math_enable_btn"):
            return None
        if not self.analysis_math_enable_btn.isChecked():
            return None

        left_key: Any | None = self.analysis_math_left_combo.currentData()
        right_key: Any | None = self.analysis_math_right_combo.currentData()
        op: str = str(self.analysis_math_op_combo.currentData() or "").strip()
        if op not in {"+", "-", "*", "/"}:
            return None
        if left_key not in self.analysis_numeric_data:
            return None
        if right_key not in self.analysis_numeric_data:
            return None

        left_values: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
            self.analysis_numeric_data[left_key],
            dtype=float,
        ).reshape(-1)
        right_values: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
            self.analysis_numeric_data[right_key],
            dtype=float,
        ).reshape(-1)
        if left_values.size <= 0 or right_values.size <= 0:
            return None
        if left_values.size != right_values.size:
            return None

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if op == "+":
                y_values = left_values + right_values
            elif op == "-":
                y_values = left_values - right_values
            elif op == "*":
                y_values = left_values * right_values
            else:
                y_values = left_values / right_values

        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        y_values[~np.isfinite(y_values)] = np.nan

        left_label, _ = self._analysis_field_display_text(left_key)
        right_label, _ = self._analysis_field_display_text(right_key)
        if not left_label:
            left_label = str(left_key)
        if not right_label:
            right_label = str(right_key)
        plot_label_text: str = f"{left_label} {op} {right_label}"
        hover_op: str = "\u00d7" if op == "*" else op
        hover_label: str = f"{left_label} {hover_op} {right_label}"
        return {
            "key": "__analysis_math__",
            "y_values": y_values,
            "plot_label": parameter_symbol_to_mathtext(plot_label_text),
            "hover_label": hover_label,
        }

    def _collect_analysis_ui_state(self) -> Dict[str, Any]:
        """Serialize current Analysis tab control state."""
        state: Dict[str, Any] = {
            "x_field": None,
            "selected_params": [],
            "mode": "combined",
            "show_points": True,
            "show_series_line": False,
            "show_fit_lines": True,
            "show_legend": True,
            "log_x": False,
            "math_enabled": False,
            "math_left": None,
            "math_op": "-",
            "math_right": None,
        }
        if hasattr(self, "analysis_x_combo"):
            state["x_field"] = self.analysis_x_combo.currentData()
        if hasattr(self, "analysis_mode_combo"):
            mode = self.analysis_mode_combo.currentData()
            if mode not in (None, ""):
                state["mode"] = str(mode)
        if hasattr(self, "analysis_show_points_btn"):
            state["show_points"] = bool(self.analysis_show_points_btn.isChecked())
        if hasattr(self, "analysis_show_series_line_btn"):
            state["show_series_line"] = bool(
                self.analysis_show_series_line_btn.isChecked()
            )
        if hasattr(self, "analysis_fit_line_btn"):
            state["show_fit_lines"] = bool(self.analysis_fit_line_btn.isChecked())
        if hasattr(self, "analysis_legend_btn"):
            state["show_legend"] = bool(self.analysis_legend_btn.isChecked())
        if hasattr(self, "analysis_log_x_btn"):
            state["log_x"] = bool(self.analysis_log_x_btn.isChecked())
        if hasattr(self, "analysis_math_enable_btn"):
            state["math_enabled"] = bool(self.analysis_math_enable_btn.isChecked())
        if hasattr(self, "analysis_math_left_combo"):
            state["math_left"] = self.analysis_math_left_combo.currentData()
        if hasattr(self, "analysis_math_op_combo"):
            op_text: str = str(self.analysis_math_op_combo.currentData() or "").strip()
            if op_text in {"+", "-", "*", "/"}:
                state["math_op"] = op_text
        if hasattr(self, "analysis_math_right_combo"):
            state["math_right"] = self.analysis_math_right_combo.currentData()
        if hasattr(self, "analysis_param_buttons"):
            state["selected_params"] = [
                str(key)
                for key, button in self.analysis_param_buttons.items()
                if button.isChecked()
            ]
        return state

    def _restore_analysis_ui_state(self, state) -> None:
        """Restore Analysis tab control state from serialized payload."""
        if not isinstance(state, Mapping):
            return
        if not hasattr(self, "analysis_x_combo"):
            return

        mode_combo: Any | None = getattr(self, "analysis_mode_combo", None)
        saved_mode: str = str(state.get("mode") or "").strip()
        if mode_combo is not None and saved_mode:
            idx: int = mode_combo.findData(saved_mode)
            if idx >= 0:
                blocked = mode_combo.blockSignals(True)
                mode_combo.setCurrentIndex(idx)
                mode_combo.blockSignals(blocked)

        toggle_fields = (
            ("analysis_show_points_btn", "show_points"),
            ("analysis_show_series_line_btn", "show_series_line"),
            ("analysis_fit_line_btn", "show_fit_lines"),
            ("analysis_legend_btn", "show_legend"),
            ("analysis_log_x_btn", "log_x"),
        )
        for attr_name, state_key in toggle_fields:
            btn: Any | None = getattr(self, attr_name, None)
            if btn is None:
                continue
            if state_key not in state:
                continue
            blocked = btn.blockSignals(True)
            btn.setChecked(bool(state.get(state_key)))
            btn.blockSignals(blocked)

        math_enable_btn: Any | None = getattr(self, "analysis_math_enable_btn", None)
        if math_enable_btn is not None and "math_enabled" in state:
            blocked = math_enable_btn.blockSignals(True)
            math_enable_btn.setChecked(bool(state.get("math_enabled")))
            math_enable_btn.blockSignals(blocked)

        math_combo_fields = (
            ("analysis_math_left_combo", "math_left"),
            ("analysis_math_op_combo", "math_op"),
            ("analysis_math_right_combo", "math_right"),
        )
        for attr_name, state_key in math_combo_fields:
            combo: Any | None = getattr(self, attr_name, None)
            if combo is None:
                continue
            if state_key not in state:
                continue
            idx = combo.findData(state.get(state_key))
            if idx < 0:
                continue
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(blocked)

        saved_x = state.get("x_field")
        x_idx: int = self.analysis_x_combo.findData(saved_x)
        if x_idx < 0:
            x_idx = self.analysis_x_combo.findData(None)
        if x_idx >= 0:
            blocked = self.analysis_x_combo.blockSignals(True)
            self.analysis_x_combo.setCurrentIndex(x_idx)
            self.analysis_x_combo.blockSignals(blocked)

        selected: set[str] = set()
        raw_selected = state.get("selected_params")
        if isinstance(raw_selected, list):
            selected = {
                str(value).strip() for value in raw_selected if str(value).strip()
            }
        for key, button in self.analysis_param_buttons.items():
            blocked = button.blockSignals(True)
            button.setChecked(str(key) in selected)
            button.blockSignals(blocked)

        self.update_batch_analysis_plot()

    def _toggle_analysis_param(self, key) -> None:
        control = self.analysis_param_buttons.get(str(key))
        if control is None:
            return
        control.setChecked(not control.isChecked())

    def _rebuild_analysis_param_buttons(self, previous_params) -> None:
        clear_layout(self.analysis_params_button_layout)
        self.analysis_param_buttons = {}
        sections = self._analysis_param_sections()
        if not sections:
            self.analysis_params_button_layout.addWidget(
                self._new_label(
                    "No numeric fit parameters available.",
                    style_sheet="color: #64748b; font-size: 11px;",
                )
            )
            self.analysis_params_button_layout.addStretch(1)
            return

        row_specs = []
        for section in sections:
            section_kind: str = str(section.get("kind") or "")
            for group in list(section.get("groups") or []):
                group_title: str = str(group.get("title") or "").strip()
                if not group_title and section_kind == "boundary":
                    group_title = "Boundaries"
                if not group_title and section_kind == "metric":
                    group_title = "Metrics"
                row_specs.append(
                    {
                        "kind": section_kind,
                        "title": group_title,
                        "keys": [str(key) for key in list(group.get("keys") or [])],
                    }
                )

        title_labels: List[str] = [
            f"{str(row.get('title') or '').strip()}:"
            for row in row_specs
            if str(row.get("title") or "").strip()
        ]
        title_col_width: int = 0
        if title_labels:
            metrics: QFontMetrics = self.fontMetrics()
            title_col_width = max(
                metrics.horizontalAdvance(text) for text in title_labels
            )
            title_col_width += 10

        previous_kind: str = ""
        for row in row_specs:
            row_kind: str = str(row.get("kind") or "")
            if previous_kind and row_kind and row_kind != previous_kind:
                spacer = QWidget()
                spacer.setFixedHeight(2)
                self.analysis_params_button_layout.addWidget(spacer)
            previous_kind = row_kind

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(2, 0, 0, 0)
            row_layout.setSpacing(6)

            title_text: str = str(row.get("title") or "").strip()
            if title_col_width > 0:
                title_label: QLabel = self._new_label(
                    f"{title_text}:" if title_text else "",
                    style_sheet="font-weight: 600; color: #475569; font-size: 10px;",
                )
                title_label.setFixedWidth(int(title_col_width))
                row_layout.addWidget(title_label)

            for key_text in list(row.get("keys") or []):
                plain_label, rich_label = self._analysis_field_display_text(key_text)
                if not plain_label:
                    plain_label = key_text
                if not rich_label:
                    rich_label = html.escape(plain_label)
                tooltip_text: str = (
                    "Fit R²" if key_text == "R2" else f"{plain_label} ({key_text})"
                )
                control: QCheckBox = self._new_checkbox(
                    "",
                    checked=(key_text in previous_params),
                    toggled_handler=self.update_batch_analysis_plot,
                    tooltip=tooltip_text,
                )
                label = ClickableLabel(rich_label)
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setToolTip(tooltip_text)
                label.clicked.connect(
                    lambda _checked=False,
                    param_key=key_text: self._toggle_analysis_param(param_key)
                )

                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(0, 0, 0, 0)
                item_layout.setSpacing(2)
                item_layout.addWidget(control)
                item_layout.addWidget(label)
                row_layout.addWidget(item_widget)
                self.analysis_param_buttons[key_text] = control

            row_layout.addStretch(1)
            self.analysis_params_button_layout.addWidget(row_widget)
        self.analysis_params_button_layout.addStretch(1)

    def _show_analysis_message(self, message) -> None:
        self._analysis_scatter_files = {}
        self._analysis_pending_pick = None
        timer = getattr(self, "_analysis_pick_load_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._set_analysis_canvas_height(1)
        self._clear_analysis_figure()
        ax: Axes = self.analysis_fig.add_subplot(111)
        self._set_hover_axes([ax])
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        self.analysis_canvas.draw_idle()

    def _clear_analysis_figure(self) -> None:
        fig: Figure | None = getattr(self, "analysis_fig", None)
        if fig is None:
            return

        # Clearing a figure with log-scaled axes can trigger matplotlib warnings
        # when stale non-positive limits exist. Normalize scales before clear.
        for axis in list(fig.axes):
            try:
                axis.set_xscale("linear")
            except Exception:
                pass
            try:
                axis.set_yscale("linear")
            except Exception:
                pass

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Attempt to set non-positive xlim on a log-scaled axis "
                    "will be ignored."
                ),
                category=UserWarning,
            )
            fig.clear()

    def _set_analysis_canvas_height(self, axis_count: int) -> None:
        if not hasattr(self, "analysis_canvas"):
            return
        panels: int = max(1, int(axis_count))
        axis_height: int = max(120, int(getattr(self, "_analysis_axis_height_px", 240)))
        margin_px: int = max(0, int(getattr(self, "_analysis_canvas_margin_px", 80)))
        target_height: int = int(min(4096, margin_px + (panels * axis_height)))
        self.analysis_canvas.setFixedHeight(target_height)

    def _analysis_x_limits_from_data(
        self, x_min: float, x_max: float, *, use_log_x: bool
    ) -> Tuple[float, float]:
        x_lo: float = float(x_min)
        x_hi: float = float(x_max)
        if not np.isfinite(x_lo) or not np.isfinite(x_hi):
            return (1.0, 10.0) if use_log_x else (0.0, 1.0)
        if x_hi < x_lo:
            x_lo, x_hi = x_hi, x_lo

        if np.isclose(x_lo, x_hi):
            if use_log_x:
                base: float = max(float(np.nextafter(0.0, 1.0)), float(x_lo))
                return (max(float(np.nextafter(0.0, 1.0)), base / 1.25), base * 1.25)
            span: float = max(abs(float(x_lo)) * 0.1, 1.0)
            lower: float = x_lo - span
            upper: float = x_hi + span
        else:
            span = x_hi - x_lo
            pad = max(span * 0.05, 0.0)
            lower = x_lo - pad
            upper = x_hi + pad

        if use_log_x:
            min_pos: float = max(float(np.nextafter(0.0, 1.0)), float(x_lo))
            if lower <= 0.0:
                lower = max(float(np.nextafter(0.0, 1.0)), min_pos * 0.8)
            if upper <= lower:
                upper = lower * 10.0
        else:
            # Keep linear plots non-negative whenever the source x-data are non-negative.
            if x_lo >= 0.0 and lower < 0.0:
                lower = 0.0
            if upper <= lower:
                upper = lower + max(abs(float(x_hi)) * 0.05, 1.0)

        return float(lower), float(upper)

    def _analysis_record_file_ref(self, record):
        file_ref: str = str(record.get("__file_ref") or "").strip()
        if file_ref:
            return file_ref

        display_name: str = str(record.get("File") or "").strip()
        if not display_name:
            return None

        matches = [
            file_path
            for file_path in self.data_files
            if stem_for_file_ref(file_path) == display_name
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _format_hover_number(value) -> str:
        try:
            number: float = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(number):
            return "NaN"
        return f"{number:.12g}"

    def _build_hover_annotation(self, axis: Axes):
        annotation = axis.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(10, 10),
            textcoords="offset pixels",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#f8fafc",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#111827",
                "edgecolor": "#334155",
                "alpha": 0.95,
            },
            zorder=10_000,
        )
        annotation.set_annotation_clip(True)
        annotation.set_clip_on(True)
        annotation.set_clip_box(axis.bbox)
        annotation.set_visible(False)
        return annotation

    def _set_hover_axes(self, axes) -> None:
        valid_axes = [axis for axis in list(np.atleast_1d(axes)) if axis is not None]
        annotations = {axis: self._build_hover_annotation(axis) for axis in valid_axes}
        self._analysis_hover_artists = []
        self._analysis_hover_annotations = annotations

    def _register_hover_artist(
        self,
        *,
        artist,
        title: str,
        x_label: str,
        y_label: str,
        file_refs=None,
    ) -> None:
        if artist is None:
            return
        try:
            artist.set_picker(5)
        except Exception:
            pass
        if isinstance(artist, Line2D):
            try:
                artist.set_pickradius(5)
            except Exception:
                pass

        entry = {
            "artist": artist,
            "axis": getattr(artist, "axes", None),
            "title": str(title),
            "x_label": str(x_label),
            "y_label": str(y_label),
            "file_refs": list(file_refs) if file_refs is not None else None,
        }
        self._analysis_hover_artists.append(entry)

    @staticmethod
    def _extract_hover_points(
        artist,
        contains_info,
        *,
        event=None,
        max_points: int = 4,
    ) -> List[Tuple[int, float, float]]:
        indices = []
        if isinstance(contains_info, Mapping):
            raw = contains_info.get("ind", [])
            indices = list(raw) if raw is not None else []
        if not indices:
            return []

        points: List[Tuple[int, float, float]] = []

        if isinstance(artist, PathCollection):
            offsets: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                artist.get_offsets(),
                dtype=float,
            )
            if offsets.ndim != 2 or offsets.shape[1] < 2:
                return []
            for idx in indices:
                point_idx: int = int(idx)
                if point_idx < 0 or point_idx >= offsets.shape[0]:
                    continue
                points.append(
                    (
                        point_idx,
                        float(offsets[point_idx, 0]),
                        float(offsets[point_idx, 1]),
                    )
                )

        elif isinstance(artist, Line2D):
            x_data: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                artist.get_xdata(orig=False),
                dtype=float,
            ).reshape(-1)
            y_data: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                artist.get_ydata(orig=False),
                dtype=float,
            ).reshape(-1)
            if x_data.size == 0 or y_data.size == 0:
                return []
            max_idx: int = min(x_data.size, y_data.size) - 1
            for idx in indices:
                point_idx = max(0, min(int(idx), max_idx))
                points.append(
                    (point_idx, float(x_data[point_idx]), float(y_data[point_idx]))
                )

        if not points:
            return []

        # Keep nearest points first so tooltip content reflects cursor position.
        if (
            event is not None
            and hasattr(event, "x")
            and hasattr(event, "y")
            and getattr(artist, "axes", None) is not None
        ):
            axis: Axes = artist.axes
            coords = np.asarray([(p[1], p[2]) for p in points], dtype=float)
            disp = axis.transData.transform(coords)
            ex: float = float(event.x)
            ey: float = float(event.y)
            d2: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.square(
                disp[:, 0] - ex
            ) + np.square(disp[:, 1] - ey)
            order = np.argsort(d2)
            points = [points[int(i)] for i in order]

        # Deduplicate indices while preserving order.
        unique_points: List[Tuple[int, float, float]] = []
        seen_indices: Set[int] = set()
        for point_idx, x_val, y_val in points:
            if point_idx in seen_indices:
                continue
            seen_indices.add(point_idx)
            unique_points.append((point_idx, x_val, y_val))
            if len(unique_points) >= max(1, int(max_points)):
                break

        return unique_points

    def _hover_text_for_entry(
        self, entry, points: List[Tuple[int, float, float]]
    ) -> str:
        if not points:
            return ""
        text_lines = []
        title: str = str(entry.get("title") or "").strip()
        if title:
            text_lines.append(title)

        file_refs = entry.get("file_refs")
        x_label: str = str(entry.get("x_label", "x"))
        y_label: str = str(entry.get("y_label", "y"))

        if len(points) == 1:
            point_idx, x_val, y_val = points[0]
            if file_refs and 0 <= point_idx < len(file_refs):
                file_ref: str = str(file_refs[point_idx] or "").strip()
                if file_ref:
                    text_lines.insert(0, stem_for_file_ref(file_ref))
            text_lines.append(f"{x_label}: {self._format_hover_number(x_val)}")
            text_lines.append(f"{y_label}: {self._format_hover_number(y_val)}")
            return "\n".join(text_lines)

        text_lines.append(f"{len(points)} nearby points")
        for point_idx, x_val, y_val in points:
            point_name: str = f"#{point_idx + 1}"
            if file_refs and 0 <= point_idx < len(file_refs):
                file_ref: str = str(file_refs[point_idx] or "").strip()
                if file_ref:
                    point_name = stem_for_file_ref(file_ref)
            text_lines.append(
                f"{point_name}: {x_label}={self._format_hover_number(x_val)}, "
                f"{y_label}={self._format_hover_number(y_val)}"
            )
        return "\n".join(text_lines)

    def _hide_hover_annotations(self) -> None:
        annotations = self._analysis_hover_annotations
        canvas = self.analysis_canvas
        changed: bool = False
        for annotation in annotations.values():
            if annotation.get_visible():
                annotation.set_visible(False)
                changed = True
        if changed:
            canvas.draw_idle()

    @staticmethod
    def _find_hover_hit(
        event, entries
    ) -> None | Tuple[dict, List[Tuple[int, float, float]]]:
        event_axis = getattr(event, "inaxes", None)
        if event_axis is None:
            return None
        for entry in reversed(entries):
            artist = entry.get("artist")
            axis = entry.get("axis")
            if artist is None or axis is None or axis is not event_axis:
                continue
            try:
                contains, info = artist.contains(event)
            except Exception:
                continue
            if not contains:
                continue
            points = ManualFitGUI._extract_hover_points(
                artist, info, event=event, max_points=4
            )
            if not points:
                continue
            return (entry, points)
        return None

    @staticmethod
    def _position_hover_annotation(annotation, axis: Axes, event) -> None:
        bbox = axis.bbox
        try:
            cursor_x: float = float(event.x)
            cursor_y: float = float(event.y)
        except Exception:
            cursor_x = float(bbox.x0 + (bbox.width * 0.5))
            cursor_y = float(bbox.y0 + (bbox.height * 0.5))

        is_right_half: bool = cursor_x >= float(bbox.x0 + (bbox.width * 0.5))
        is_top_half: bool = cursor_y >= float(bbox.y0 + (bbox.height * 0.5))
        x_offset: int = -12 if is_right_half else 12
        y_offset: int = -12 if is_top_half else 12

        annotation.set_position((x_offset, y_offset))
        annotation.set_ha("right" if is_right_half else "left")
        annotation.set_va("top" if is_top_half else "bottom")

        # Nudge annotation fully inside the axis bbox.
        fig = getattr(annotation, "figure", None)
        canvas = getattr(fig, "canvas", None)
        if canvas is None:
            return
        try:
            renderer = canvas.get_renderer()
            ann_bbox = annotation.get_window_extent(renderer=renderer)
        except Exception:
            return

        margin_px: float = 4.0
        left_bound: float = float(bbox.x0 + margin_px)
        right_bound: float = float(bbox.x1 - margin_px)
        bottom_bound: float = float(bbox.y0 + margin_px)
        top_bound: float = float(bbox.y1 - margin_px)

        adjust_x: float = 0.0
        adjust_y: float = 0.0
        if ann_bbox.x0 < left_bound:
            adjust_x += left_bound - float(ann_bbox.x0)
        if ann_bbox.x1 > right_bound:
            adjust_x -= float(ann_bbox.x1) - right_bound
        if ann_bbox.y0 < bottom_bound:
            adjust_y += bottom_bound - float(ann_bbox.y0)
        if ann_bbox.y1 > top_bound:
            adjust_y -= float(ann_bbox.y1) - top_bound
        if not (np.isclose(adjust_x, 0.0) and np.isclose(adjust_y, 0.0)):
            cur_x, cur_y = annotation.get_position()
            annotation.set_position((float(cur_x) + adjust_x, float(cur_y) + adjust_y))

    def _update_hover_annotations(self, event) -> None:
        annotations = self._analysis_hover_annotations
        entries = self._analysis_hover_artists
        canvas = self.analysis_canvas

        if getattr(event, "inaxes", None) is None:
            self._hide_hover_annotations()
            return

        hit = self._find_hover_hit(event, entries)
        if hit is None:
            self._hide_hover_annotations()
            return
        entry, points = hit
        axis = entry.get("axis")
        annotation = annotations.get(axis)
        if annotation is None:
            self._hide_hover_annotations()
            return

        anchor_x: float = float(points[0][1])
        anchor_y: float = float(points[0][2])
        text = self._hover_text_for_entry(entry, points)
        changed: bool = False

        for ann_axis, ann in annotations.items():
            if ann_axis is axis:
                continue
            if ann.get_visible():
                ann.set_visible(False)
                changed = True

        needs_update: bool = (
            not annotation.get_visible() or annotation.get_text() != text
        )
        if not needs_update:
            try:
                old_x, old_y = annotation.xy
                needs_update = not (
                    np.isclose(float(old_x), float(anchor_x))
                    and np.isclose(float(old_y), float(anchor_y))
                )
            except Exception:
                needs_update = True

        old_pos: Tuple[float, ...] = tuple(annotation.get_position())
        old_ha: str = str(annotation.get_ha())
        old_va: str = str(annotation.get_va())
        if needs_update:
            annotation.xy = (float(anchor_x), float(anchor_y))
            annotation.set_text(text)
            changed = True
        if not annotation.get_visible():
            annotation.set_visible(True)
            changed = True
        self._position_hover_annotation(annotation, axis, event)
        new_pos: Tuple[float, ...] = tuple(annotation.get_position())
        new_ha: str = str(annotation.get_ha())
        new_va: str = str(annotation.get_va())
        if old_pos != new_pos or old_ha != new_ha or old_va != new_va:
            changed = True

        if changed:
            canvas.draw_idle()

    def _on_analysis_plot_hover(self, event) -> None:
        self._update_hover_annotations(event)

    def _on_analysis_plot_leave(self, _event) -> None:
        self._hide_hover_annotations()

    @staticmethod
    def _analysis_pick_signature(mouse_event) -> Any:
        if mouse_event is None:
            return None
        return id(mouse_event)

    @staticmethod
    def _analysis_pick_nearest_index(
        artist,
        picked_indices,
        mouse_event,
    ) -> Tuple[None | int, float]:
        if picked_indices is None:
            return (None, float("inf"))

        try:
            indices = np.asarray(list(picked_indices), dtype=int).reshape(-1)
        except Exception:
            return (None, float("inf"))
        if indices.size == 0:
            return (None, float("inf"))

        if (
            mouse_event is None
            or not hasattr(mouse_event, "x")
            or not hasattr(mouse_event, "y")
            or not isinstance(artist, PathCollection)
        ):
            return (int(indices[0]), float("inf"))

        try:
            offsets: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                artist.get_offsets(),
                dtype=float,
            )
        except Exception:
            return (int(indices[0]), float("inf"))
        if offsets.ndim != 2 or offsets.shape[1] < 2:
            return (int(indices[0]), float("inf"))

        valid = indices[(indices >= 0) & (indices < offsets.shape[0])]
        if valid.size == 0:
            return (None, float("inf"))

        axis: Axes | None = getattr(artist, "axes", None)
        if axis is None:
            return (int(valid[0]), float("inf"))

        try:
            display_xy: np.ndarray[Tuple[int, ...], np.dtype[Any]] = axis.transData.transform(
                offsets[valid, :2]
            )
            ex = float(mouse_event.x)
            ey = float(mouse_event.y)
            d2: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.square(
                display_xy[:, 0] - ex
            ) + np.square(display_xy[:, 1] - ey)
            best_local_idx: int = int(np.argmin(d2))
            return (int(valid[best_local_idx]), float(d2[best_local_idx]))
        except Exception:
            return (int(valid[0]), float("inf"))

    def _flush_pending_analysis_point_pick(self) -> None:
        pending = getattr(self, "_analysis_pending_pick", None)
        self._analysis_pending_pick = None
        if not isinstance(pending, dict):
            return
        file_ref: str = str(pending.get("file_ref") or "").strip()
        if not file_ref:
            return
        self._open_file_in_plot_tab(file_ref)

    def _on_analysis_point_picked(self, event) -> None:
        artist: Any | None = getattr(event, "artist", None)
        if artist is None:
            return
        file_refs = self._analysis_scatter_files.get(artist)
        if not file_refs:
            return

        picked: Any | None = getattr(event, "ind", None)
        if picked is None or len(picked) == 0:
            return
        mouse_event = getattr(event, "mouseevent", None)
        point_idx, distance2 = self._analysis_pick_nearest_index(
            artist,
            picked,
            mouse_event,
        )
        if point_idx is None:
            return
        if point_idx < 0 or point_idx >= len(file_refs):
            return

        file_ref = file_refs[point_idx]
        if not file_ref:
            self.stats_text.append(
                "Unable to resolve clicked point to a unique source file."
            )
            return
        signature = self._analysis_pick_signature(mouse_event)
        pending = getattr(self, "_analysis_pending_pick", None)
        should_replace: bool = not isinstance(pending, dict)
        if not should_replace:
            if pending.get("signature") != signature:
                should_replace = True
            else:
                prev_distance2 = float(pending.get("distance2", float("inf")))
                should_replace = float(distance2) < prev_distance2

        if should_replace:
            self._analysis_pending_pick = {
                "signature": signature,
                "distance2": float(distance2),
                "file_ref": str(file_ref),
            }
        timer = getattr(self, "_analysis_pick_load_timer", None)
        if timer is None:
            self._flush_pending_analysis_point_pick()
            return
        if not timer.isActive():
            timer.start(20)

    def _linear_fit(self, x_data, y_data) -> None | Tuple[float, float]:
        if x_data.size < 2:
            return None
        if np.isclose(float(np.ptp(x_data)), 0.0):
            return None
        try:
            slope, intercept = np.polyfit(x_data, y_data, 1)
            return float(slope), float(intercept)
        except Exception:
            return None

    def update_batch_analysis_plot(self) -> None:
        """Plot parameter variation against selected field."""
        if not hasattr(self, "analysis_fig"):
            return
        self._analysis_scatter_files = {}
        self._analysis_pending_pick = None
        timer = getattr(self, "_analysis_pick_load_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        if not self.analysis_numeric_data:
            self._show_analysis_message("No numeric data available for analysis.")
            return

        x_field = self.analysis_x_combo.currentData()
        selected_params = self._selected_analysis_params()
        series_specs: List[Dict[str, Any]] = []
        for param_name in selected_params:
            y_values = self.analysis_numeric_data.get(param_name)
            if y_values is None:
                continue
            series_specs.append(
                {
                    "key": str(param_name),
                    "y_values": np.asarray(y_values, dtype=float).reshape(-1),
                    "plot_label": self._display_name_for_param_key_mathtext(param_name),
                    "hover_label": self._display_name_for_param_key(param_name),
                }
            )
        math_series = self._analysis_math_series_spec()
        if math_series is not None:
            series_specs.append(math_series)
        if x_field not in self.analysis_numeric_data:
            self._show_analysis_message("Select an X field to plot.")
            return
        if not series_specs:
            self._show_analysis_message(
                "Select at least one parameter or enable a math-derived series."
            )
            return

        x_values = self.analysis_numeric_data[x_field]
        x_hover_label, _ = self._analysis_field_display_text(x_field)
        analysis_x_hover_label: str = x_hover_label or str(x_field)
        mode = self.analysis_mode_combo.currentData()
        show_points: bool = self.analysis_show_points_btn.isChecked()
        show_series_line: bool = self.analysis_show_series_line_btn.isChecked()
        show_fit_lines: bool = self.analysis_fit_line_btn.isChecked()
        show_legend: bool = self.analysis_legend_btn.isChecked()
        use_log_x: bool = bool(
            getattr(self, "analysis_log_x_btn", None)
            and self.analysis_log_x_btn.isChecked()
        )
        if use_log_x:
            log_invalid_reason: str = ""
            for spec in series_specs:
                y_values = np.asarray(
                    spec.get("y_values", np.asarray([], dtype=float)),
                    dtype=float,
                ).reshape(-1)
                if y_values.size != np.size(x_values):
                    continue
                valid_pairs = np.isfinite(x_values) & np.isfinite(y_values)
                valid_count: int = int(np.count_nonzero(valid_pairs))
                if valid_count <= 0:
                    continue
                positive_count: int = int(
                    np.count_nonzero(valid_pairs & (x_values > 0.0))
                )
                if positive_count < valid_count:
                    display_name: str = str(
                        spec.get("plot_label") or spec.get("hover_label") or "series"
                    )
                    log_invalid_reason = (
                        f"X values for {display_name} include zero/negative points."
                    )
                    break
            if log_invalid_reason:
                use_log_x = False
                log_btn: Any | None = getattr(self, "analysis_log_x_btn", None)
                if log_btn is not None and log_btn.isChecked():
                    blocked: bool = log_btn.blockSignals(True)
                    log_btn.setChecked(False)
                    log_btn.blockSignals(blocked)
                if hasattr(self, "stats_text"):
                    self.stats_text.append(
                        f"✗ Batch analysis: disabled Log X; reverted to linear ({log_invalid_reason})"
                    )

        if not (show_points or show_series_line or show_fit_lines):
            self._show_analysis_message(
                "Enable at least one plot layer (Points/Line/Fit)."
            )
            return

        axis_count: int = (
            len(series_specs)
            if mode == "separate" and len(series_specs) > 1
            else 1
        )
        self._set_analysis_canvas_height(axis_count)
        self._clear_analysis_figure()
        if mode == "separate" and len(series_specs) > 1:
            axes = self.analysis_fig.subplots(len(series_specs), 1, sharex=True)
            axes: List[Any] = list(np.atleast_1d(axes))
        else:
            axes: List[Axes] = [self.analysis_fig.add_subplot(111)]
        self._set_hover_axes(axes)

        plotted_any = False
        x_min_plot: None | float = None
        x_max_plot: None | float = None
        for idx, spec in enumerate(series_specs):
            y_values = np.asarray(
                spec.get("y_values", np.asarray([], dtype=float)),
                dtype=float,
            ).reshape(-1)
            if y_values.size <= 0 or y_values.size != np.size(x_values):
                continue
            mask = np.isfinite(x_values) & np.isfinite(y_values)
            if use_log_x:
                mask &= x_values > 0.0
            if np.count_nonzero(mask) == 0:
                continue

            plotted_any = True
            x_plot = x_values[mask]
            y_plot = y_values[mask]
            local_x_min: float = float(np.min(x_plot))
            local_x_max: float = float(np.max(x_plot))
            if x_min_plot is None or local_x_min < x_min_plot:
                x_min_plot = local_x_min
            if x_max_plot is None or local_x_max > x_max_plot:
                x_max_plot = local_x_max
            file_refs = [
                self._analysis_record_file_ref(self.analysis_records[row_idx])
                for row_idx in np.flatnonzero(mask)
            ]
            order: np.ndarray[
                Tuple[int, ...], np.dtype[np.signedinteger[np._32Bit | np._64Bit]]
            ] = np.argsort(x_plot)
            x_sorted = x_plot[order]
            y_sorted = y_plot[order]
            file_refs_sorted = [file_refs[int(order_idx)] for order_idx in order]
            color: str = palette_color(idx)
            r2_values = self.analysis_numeric_data.get("R2")
            if r2_values is not None and np.size(r2_values) == np.size(mask):
                r2_plot = r2_values[mask]
                r2_sorted = r2_plot[order]
                point_alphas = np.clip(r2_sorted, 0.5, 1.0)
                point_alphas: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.where(
                    np.isfinite(point_alphas), point_alphas, 0.5
                )
            else:
                point_alphas: np.ndarray[Tuple[int], np.dtype[Any]] = np.full(
                    np.size(x_sorted), 0.5, dtype=float
                )
            target_ax: Any | Axes = axes[idx] if len(axes) > 1 else axes[0]
            param_plot_label: str = str(
                spec.get("plot_label") or spec.get("hover_label") or spec.get("key")
            )
            param_hover_label: str = str(
                spec.get("hover_label") or spec.get("plot_label") or spec.get("key")
            )

            if show_points:
                scatter_label: str = (
                    param_plot_label if not show_series_line else "_nolegend_"
                )
                rgba: Tuple[float, float, float, float] = mcolors.to_rgba(color)
                point_colors: List[Tuple[float, float, float, float]] = [
                    (rgba[0], rgba[1], rgba[2], float(alpha)) for alpha in point_alphas
                ]
                scatter: PathCollection | Any = target_ax.scatter(
                    x_sorted,
                    y_sorted,
                    s=26,
                    color=point_colors,
                    label=scatter_label,
                )
                scatter.set_picker(5)
                self._analysis_scatter_files[scatter] = file_refs_sorted
                self._register_hover_artist(
                    artist=scatter,
                    title=param_hover_label,
                    x_label=analysis_x_hover_label,
                    y_label=param_hover_label,
                    file_refs=file_refs_sorted,
                )
            if show_series_line:
                (series_line,) = target_ax.plot(
                    x_sorted,
                    y_sorted,
                    linewidth=1.4,
                    alpha=0.85,
                    color=color,
                    label=param_plot_label,
                )
                self._register_hover_artist(
                    artist=series_line,
                    title=f"{param_hover_label} series",
                    x_label=analysis_x_hover_label,
                    y_label=param_hover_label,
                )

            if show_fit_lines:
                fit: None | Tuple[float, float] = self._linear_fit(x_sorted, y_sorted)
                if fit is not None:
                    slope, intercept = fit
                    x_line: np.ndarray[Tuple[int, ...], np.dtype[np.floating[Any]]] = (
                        np.linspace(
                            float(np.min(x_sorted)), float(np.max(x_sorted)), 200
                        )
                    )
                    y_line = slope * x_line + intercept
                    # Exclude best fit lines from legend
                    (fit_line,) = target_ax.plot(
                        x_line,
                        y_line,
                        linewidth=1.6,
                        color=color,
                        label="_nolegend_",
                    )
                    self._register_hover_artist(
                        artist=fit_line,
                        title=f"{param_hover_label} linear fit",
                        x_label=analysis_x_hover_label,
                        y_label=param_hover_label,
                    )

            if len(axes) > 1:
                target_ax.set_ylabel(param_plot_label)
                target_ax.grid(True, alpha=0.25)
                if show_legend:
                    target_ax.legend(loc="best", fontsize=8)

        if not plotted_any:
            if use_log_x:
                self._show_analysis_message(
                    "No positive finite X values available for log-scale plotting."
                )
            else:
                self._show_analysis_message(
                    "No finite X/Y pairs available for the selected fields."
                )
            return

        if x_min_plot is not None and x_max_plot is not None:
            x_left, x_right = self._analysis_x_limits_from_data(
                x_min_plot,
                x_max_plot,
                use_log_x=use_log_x,
            )
            for axis in axes:
                if use_log_x:
                    # Ensure valid positive limits before enabling log scale.
                    axis.set_xlim(x_left, x_right)
                    axis.set_xscale("log")
                else:
                    axis.set_xscale("linear")
                    axis.set_xlim(x_left, x_right)
        else:
            for axis in axes:
                axis.set_xscale("linear")

        if len(axes) == 1:
            if len(series_specs) == 1:
                single_label: str = str(
                    series_specs[0].get("plot_label")
                    or series_specs[0].get("hover_label")
                    or "Parameter Value"
                )
                axes[0].set_ylabel(single_label)
            else:
                axes[0].set_ylabel("Parameter Value")
            if show_legend:
                axes[0].legend(loc="best", fontsize=8)
            axes[0].grid(True, alpha=0.3)

        if x_field in self.analysis_param_columns:
            x_axis_label: str = self._display_name_for_param_key_mathtext(x_field)
        else:
            x_axis_label = x_field

        axes[-1].set_xlabel(x_axis_label)
        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw_idle()

    def _current_batch_row_height(self) -> int:
        return max(
            self.batch_row_height_min,
            min(self.batch_row_height_max, int(self.batch_row_height)),
        )

    def _current_batch_thumbnail_size(self) -> Tuple[int, int]:
        row_height: int = self._current_batch_row_height()
        thumb_height: int = max(24, row_height - 8)
        thumb_width: int = max(
            36, int(round(thumb_height * self.batch_thumbnail_aspect))
        )
        return (thumb_width, thumb_height)

    def _full_batch_thumbnail_size(self) -> Tuple[int, int]:
        full_height: int = max(
            48,
            int(
                round(
                    (self._current_batch_row_height() - 8)
                    * self.batch_thumbnail_supersample
                )
            ),
        )
        full_width: int = max(36, int(round(full_height * self.batch_thumbnail_aspect)))
        return (full_width, full_height)

    def _apply_batch_row_heights(self) -> None:
        if not hasattr(self, "batch_table"):
            return
        if self._batch_row_height_sync:
            return

        row_height: int = self._current_batch_row_height()
        self._batch_row_height_sync = True
        try:
            self.batch_table.verticalHeader().setDefaultSectionSize(row_height)
            if self.batch_table.columnCount() > 0:
                thumb_width, _ = self._current_batch_thumbnail_size()
                self.batch_table.setColumnWidth(0, thumb_width + 18)
            for row_idx in range(self.batch_table.rowCount()):
                self.batch_table.setRowHeight(row_idx, row_height)
        finally:
            self._batch_row_height_sync = False

    def _scaled_batch_plot(self, row):
        source = row.get("plot_full") or row.get("plot")
        if source is None:
            return None
        target_width, target_height = self._current_batch_thumbnail_size()
        return source.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _find_table_row_by_file(self, file_path) -> None | int:
        """Find table row index by file path stored in item user data."""
        if self.batch_table.columnCount() == 0:
            return None
        target_key: str = self._fit_task_file_key(file_path)
        if not target_key:
            return None
        for row_idx in range(self.batch_table.rowCount()):
            item: QTableWidgetItem | None = self.batch_table.item(row_idx, 0)
            if not item:
                continue
            item_key: str = self._fit_task_file_key(item.data(Qt.ItemDataRole.UserRole))
            if item_key and item_key == target_key:
                return row_idx
        return None

    def _find_batch_result_index_by_file(self, file_path) -> None | int:
        target_key: str = self._fit_task_file_key(file_path)
        if not target_key:
            return None
        for idx, row in enumerate(self.batch_results):
            row_key: str = self._fit_task_file_key(row.get("file"))
            if row_key and row_key == target_key:
                return idx
        return None

    def _rebuild_batch_capture_keys_from_rows(self) -> None:
        keys = []
        for row in self.batch_results:
            captures = dict(row.get("captures") or {})
            for key in captures.keys():
                text: str = str(key).strip()
                if text and text not in keys:
                    keys.append(text)
        self.batch_capture_keys = keys

    def _batch_parameter_column_items(self):
        items = []
        for idx, spec in enumerate(self.param_specs):
            token: str = (
                str(self._display_name_for_param_key(spec.key)).strip()
                or spec.column_name
            )
            items.append(
                {
                    "kind": "param",
                    "index": int(idx),
                    "key": str(spec.column_name),
                    "token": token,
                }
            )
        # Collect boundaries from all channels (multi-channel aware).
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        name_map: Dict[Any, Any] = getattr(self, "_boundary_name_map", {}) or {}
        if multi_model is not None and multi_model.is_multi_channel:
            for ch_model in multi_model.channel_models:
                target = str(ch_model.target_col)
                n_b = max(0, len(ch_model.segment_exprs) - 1)
                for idx in range(n_b):
                    bid = (target, idx)
                    display_name = name_map.get(bid, format_boundary_display_name(idx))
                    # Prefix with channel name when multiple channels exist.
                    token = f"{self._channel_display_name(target)}:{display_name}"
                    items.append(
                        {
                            "kind": "boundary",
                            "index": int(idx),
                            "key": f"{target}:X_{idx}",
                            "token": token,
                            "target": target,
                        }
                    )
        else:
            n_boundaries: int = max(
                0,
                len(self._piecewise_model.segment_exprs) - 1
                if self._piecewise_model is not None
                else 0,
            )
            target = (
                str(self._piecewise_model.target_col)
                if self._piecewise_model is not None
                else None
            )
            for idx in range(n_boundaries):
                items.append(
                    {
                        "kind": "boundary",
                        "index": int(idx),
                        "key": f"X_{idx}",
                        "token": format_boundary_display_name(idx),
                        "target": target,
                    }
                )
        return items

    @staticmethod
    def _as_float_array(
        values,
    ) -> (
        np.ndarray[Tuple[int, ...], np.dtype[Any]]
        | np.ndarray[Tuple[int], np.dtype[Any]]
    ):
        if values is None:
            return np.asarray([], dtype=float)
        try:
            return np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)

    @staticmethod
    def _float_list_or_none(values) -> None | List[float]:
        if values is None:
            return None
        try:
            arr: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                values, dtype=float
            ).reshape(-1)
        except Exception:
            return None
        finite: np.ndarray[Tuple[int, ...], np.dtype[Any]] = arr[np.isfinite(arr)]
        if finite.size != arr.size:
            return None
        return [float(v) for v in arr]

    def _fit_details_sidecar_path(self) -> None | Path:
        base_dir = None
        source_text: str = str(getattr(self, "current_dir", "")).strip()
        if source_text:
            source_path: Path = Path(source_text).expanduser()
            if source_path.is_dir():
                base_dir: Path = source_path
            elif source_path.is_file():
                base_dir: Path = source_path.parent
        if base_dir is None:
            selected: List[Any] = list(
                getattr(self, "_source_selected_paths", []) or []
            )
            if selected:
                selected_parent: Path = Path(selected[0]).expanduser().parent
                if selected_parent.exists():
                    base_dir: Path = selected_parent
        if base_dir is None:
            return None
        return Path(base_dir) / FIT_DETAILS_FILENAME

    def _serialize_fit_parameter_specs(self):
        serialized = []
        manually_fixed = getattr(self, "_manually_fixed_params", set())
        periodic_keys = set(getattr(self, "_periodic_param_keys", set()) or set())
        for spec in self.param_specs:
            min_box = self.param_min_spinboxes.get(spec.key)
            max_box = self.param_max_spinboxes.get(spec.key)
            value_box = self.param_spinboxes.get(spec.key)
            low: float = (
                float(min_box.value()) if min_box is not None else float(spec.min_value)
            )
            high: float = (
                float(max_box.value()) if max_box is not None else float(spec.max_value)
            )
            if low > high:
                low, high = high, low
            value: float = (
                float(value_box.value())
                if value_box is not None
                else self._param_default_from_limits(low, high)
            )
            serialized.append(
                {
                    "key": str(spec.key),
                    "symbol": str(spec.symbol),
                    "min_value": float(low),
                    "max_value": float(high),
                    "value": float(np.clip(value, low, high)),
                    "fixed": bool(spec.key in manually_fixed),
                    "periodic": bool(spec.key in periodic_keys),
                }
            )
        return serialized

    def _serialize_fit_batch_rows(self):
        rows = []
        for row in list(getattr(self, "batch_results", []) or []):
            file_ref: str = str(row.get("file") or "").strip()
            if not file_ref:
                continue
            params: None | List[float] = self._float_list_or_none(
                fit_get(row, "params")
            )
            r2: float | None = finite_float_or_none(fit_get(row, "r2"))
            error: str | None = (
                str(fit_get(row, "error"))
                if fit_get(row, "error") not in (None, "")
                else None
            )
            captures: Dict[Any, Any] = dict(row.get("captures") or {})
            # Skip rows with no meaningful saved data.
            if (
                params is None
                and r2 is None
                and error is None
                and not fit_get(row, "channel_results")
            ):
                continue
            entry: Dict[str, Any] = {
                "file": file_ref,
                "file_stem": stem_for_file_ref(file_ref),
            }
            if captures:
                entry["captures"] = captures
            fit_results_payload = {}
            if params is not None:
                fit_results_payload["params"] = params
            if r2 is not None:
                fit_results_payload["r2"] = r2
            if error is not None:
                fit_results_payload["error"] = error
            # Serialize per-channel boundary results from channel_results.
            ch_results = fit_get(row, "channel_results")
            if isinstance(ch_results, dict) and ch_results:
                ch_out = {}
                for ch_target, ch_result in ch_results.items():
                    if not isinstance(ch_result, Mapping):
                        continue
                    ch_entry = {}
                    ch_br = ch_result.get("boundary_ratios")
                    if ch_br is not None:
                        ch_list: None | List[float] = self._float_list_or_none(ch_br)
                        if ch_list is not None:
                            ch_entry["boundary_ratios"] = ch_list
                    ch_bv = ch_result.get("boundaries")
                    if ch_bv is not None:
                        ch_bv_list: None | List[float] = self._float_list_or_none(ch_bv)
                        if ch_bv_list is not None:
                            ch_entry["boundaries"] = ch_bv_list
                    ch_r2 = finite_float_or_none(ch_result.get("r2"))
                    if ch_r2 is not None:
                        ch_entry["r2"] = ch_r2
                    if ch_entry:
                        ch_out[str(ch_target)] = ch_entry
                if ch_out:
                    fit_results_payload["channel_results"] = ch_out
            if fit_results_payload:
                entry["fit_results"] = fit_results_payload
            rows.append(entry)
        return rows

    @staticmethod
    def _serialize_splitter_sizes(splitter) -> None | List[int]:
        if splitter is None:
            return None
        try:
            sizes: List[int] = [int(v) for v in splitter.sizes()]
        except Exception:
            return None
        if len(sizes) < 2:
            return None
        if any(v <= 0 for v in sizes):
            return None
        return sizes

    @staticmethod
    def _sanitized_splitter_sizes(raw_sizes):
        if not isinstance(raw_sizes, (list, tuple)):
            return None
        out = []
        for item in raw_sizes:
            try:
                value = int(item)
            except Exception:
                return None
            if value <= 0:
                return None
            out.append(value)
        return out if len(out) >= 2 else None

    def _restore_splitter_sizes(self, splitter, raw_sizes) -> None:
        sizes = self._sanitized_splitter_sizes(raw_sizes)
        if splitter is None or sizes is None:
            return
        count = int(splitter.count())
        if count < 2:
            return
        if len(sizes) != count:
            return
        splitter.setSizes([int(v) for v in sizes])

    def _collect_fit_details_payload(self):
        self._refresh_boundary_state_topology(preserve_existing=True)
        # Multi-channel boundary ratios (per channel).
        boundary_ratios_per_channel = {}
        multi: Any | None = getattr(self, "_multi_channel_model", None)
        if multi is not None and multi.is_multi_channel:
            raw = self._fit_state.as_per_channel_map()
            for ch_target, ch_ratios in raw.items():
                boundary_ratios_per_channel[str(ch_target)] = (
                    self._float_list_or_none(ch_ratios) or []
                )
        channel_keys = self._sorted_channel_names(
            set(str(key).strip() for key in dict(self.channels or {}).keys())
            | set(str(key).strip() for key in dict(self.channel_units or {}).keys())
        )
        channel_keys = [key for key in channel_keys if key]
        batch_rows = self._serialize_fit_batch_rows()
        payload = {
            "format": "manual_fit_gui_details",
            "version": 15,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "gui": {
                "equation": str(getattr(self, "current_expression", "")).strip(),
                "x_channel": str(getattr(self, "x_channel", "")).strip(),
                "channel_display_names": {
                    str(key): str(self.channels.get(key, "")).strip()
                    for key in channel_keys
                },
                "channel_units": {
                    str(key): str(self.channel_units.get(key, "")).strip()
                    for key in channel_keys
                },
                "auto_fit_mode": self._current_auto_fit_run_mode(),
                "batch_fit_mode": self._current_batch_fit_run_mode(),
                "fit_compute_mode": self._current_fit_compute_mode(),
                "smoothing_enabled": bool(getattr(self, "smoothing_enabled", False)),
                "smoothing_window": int(getattr(self, "smoothing_window", 1) or 1),
                "capture_pattern": (
                    self.regex_input.text().strip()
                    if hasattr(self, "regex_input")
                    else ""
                ),
                "capture_to_param": {
                    str(key): (str(value) if value not in (None, "") else None)
                    for key, value in dict(
                        getattr(self, "param_capture_map", {}) or {}
                    ).items()
                },
                "param_to_capture": {
                    str(key): (str(value) if value not in (None, "") else None)
                    for key, value in dict(
                        self._current_param_capture_map() or {}
                    ).items()
                },
                "boundary_ratios_per_channel": boundary_ratios_per_channel,
                "boundary_name_map": {
                    f"{t}:{i}": str(name)
                    for (t, i), name in getattr(self, "_boundary_name_map", {}).items()
                    if name not in (None, "")
                },
                "fixed_boundary_ids": [
                    f"{str(t)}:{int(i)}"
                    for t, i in sorted(
                        set(getattr(self, "_manually_fixed_boundary_ids", set()))
                    )
                ],
                "manually_fixed_params": sorted(
                    str(k) for k in getattr(self, "_manually_fixed_params", set())
                ),
                "main_splitter_sizes": self._serialize_splitter_sizes(
                    getattr(self, "main_splitter", None)
                ),
                "param_fit_splitter_sizes": self._serialize_splitter_sizes(
                    getattr(self, "param_fit_splitter", None)
                ),
                "analysis_state": self._collect_analysis_ui_state(),
                "procedure": self._procedure_panel.serialize_procedure(),
            },
            "parameters": self._serialize_fit_parameter_specs(),
            "batch_results": batch_rows,
        }
        return payload

    def _write_fit_details_file(self, file_path, *, quiet) -> bool:
        path: Path = Path(file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._collect_fit_details_payload()
        temp_path: Path = path.with_name(f".{path.name}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
            os.replace(temp_path, path)
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            raise
        if not quiet:
            self.stats_text.append(f"✓ Saved fit details to {path}")
        return True

    def _autosave_fit_details(self) -> bool:
        if bool(getattr(self, "_fit_details_restore_in_progress", False)):
            return False
        sidecar: None | Path = self._fit_details_sidecar_path()
        if sidecar is None:
            return False
        try:
            return self._write_fit_details_file(sidecar, quiet=True)
        except Exception as exc:
            self.stats_text.append(f"✗ Auto-save fit details failed: {exc}")
            return False

    def _resolve_import_file_ref(self, row_data):
        file_ref: str = str(row_data.get("file") or "").strip()
        if file_ref and file_ref in self.data_files:
            return file_ref
        if file_ref:
            target_key: str = self._fit_task_file_key(file_ref)
            if target_key:
                canonical_matches = [
                    data_file
                    for data_file in self.data_files
                    if self._fit_task_file_key(data_file) == target_key
                ]
                if len(canonical_matches) == 1:
                    return canonical_matches[0]
            if "::" in file_ref:
                raw_archive, raw_member = file_ref.split("::", 1)
                member_key: str = str(raw_member).strip().replace("\\", "/").lower()
                archive_name: str = str(Path(raw_archive).name).strip().lower()
                archive_member_matches = []
                member_only_matches = []
                for data_file in self.data_files:
                    data_text: str = str(data_file).strip()
                    if "::" not in data_text:
                        continue
                    data_archive, data_member = data_text.split("::", 1)
                    data_member_key: str = (
                        str(data_member).strip().replace("\\", "/").lower()
                    )
                    if data_member_key != member_key:
                        continue
                    member_only_matches.append(data_file)
                    if str(Path(data_archive).name).strip().lower() == archive_name:
                        archive_member_matches.append(data_file)
                if len(archive_member_matches) == 1:
                    return archive_member_matches[0]
                if len(member_only_matches) == 1:
                    return member_only_matches[0]
        stem: str = str(row_data.get("file_stem") or "").strip()
        if not stem and file_ref:
            stem: str = stem_for_file_ref(file_ref)
        if not stem:
            return None
        matches = [
            data_file
            for data_file in self.data_files
            if stem_for_file_ref(data_file) == stem
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _apply_imported_batch_rows(self, payload):
        imported_rows = list(payload.get("batch_results") or [])
        if not imported_rows:
            return (0, 0)
        if not self.data_files:
            return (0, len(imported_rows))

        existing_by_file: Dict[str, Dict[Any, Any]] = {
            str(row.get("file")): canonicalize_fit_row(row)
            for row in list(getattr(self, "batch_results", []) or [])
            if str(row.get("file") or "").strip()
        }
        for source_index, file_ref in enumerate(self.data_files):
            if file_ref not in existing_by_file:
                existing_by_file[file_ref] = make_batch_result_row(
                    source_index=source_index,
                    file_path=file_ref,
                    x_channel=self.x_channel,
                    captures={},
                )

        applied = 0
        skipped = 0
        expected_params: int = len(self.param_specs)

        for raw_row in imported_rows:
            if not isinstance(raw_row, Mapping):
                skipped += 1
                continue
            file_ref = self._resolve_import_file_ref(raw_row)
            if not file_ref:
                skipped += 1
                continue
            row: Dict[Any, Any] = canonicalize_fit_row(existing_by_file.get(file_ref))
            row["file"] = file_ref
            row["captures"] = dict(raw_row.get("captures") or row.get("captures") or {})
            row["x_channel"] = self.x_channel
            raw_fit_results = dict(raw_row.get("fit_results") or {})
            fit_set(
                row,
                "error",
                str(raw_fit_results.get("error"))
                if raw_fit_results.get("error") not in (None, "")
                else None,
            )
            row["pattern_error"] = (
                str(raw_row.get("pattern_error"))
                if raw_row.get("pattern_error") not in (None, "")
                else row.get("pattern_error")
            )
            fit_set(row, "r2", finite_float_or_none(raw_fit_results.get("r2")))

            params: (
                np.ndarray[Tuple[int, ...], np.dtype[Any]]
                | np.ndarray[Tuple[int], np.dtype[Any]]
            ) = self._as_float_array(raw_fit_results.get("params"))
            if params.size > 0:
                if params.size >= expected_params:
                    fit_set(
                        row,
                        "params",
                        [float(params[idx]) for idx in range(expected_params)],
                    )
                else:
                    padded = list(self.get_current_params())
                    for idx, value in enumerate(params.tolist()):
                        if idx < len(padded):
                            padded[idx] = float(value)
                    fit_set(row, "params", padded)

            saved_ch_results = raw_fit_results.get("channel_results")
            if isinstance(saved_ch_results, Mapping):
                normalized_ch_results = {}
                for ch_target, ch_result in saved_ch_results.items():
                    if not isinstance(ch_result, Mapping):
                        continue
                    entry = {}
                    ch_br = self._as_float_array(ch_result.get("boundary_ratios"))
                    if ch_br.size > 0:
                        entry["boundary_ratios"] = np.clip(ch_br, 0.0, 1.0)
                    ch_bv = self._as_float_array(ch_result.get("boundaries"))
                    if ch_bv.size > 0:
                        entry["boundaries"] = ch_bv
                    ch_r2 = finite_float_or_none(ch_result.get("r2"))
                    if ch_r2 is not None:
                        entry["r2"] = ch_r2
                    if entry:
                        normalized_ch_results[str(ch_target)] = entry
                if normalized_ch_results:
                    fit_set(row, "channel_results", normalized_ch_results)
                else:
                    fit_set(row, "channel_results", None)
            else:
                fit_set(row, "channel_results", None)

            row["plot_full"] = None
            row["plot"] = None
            row["plot_render_size"] = None
            row["plot_has_fit"] = has_nonempty_values(fit_get(row, "params"))
            has_imported_fit: bool = bool(
                row.get("plot_has_fit")
                or bool(fit_get(row, "channel_results"))
            )
            if has_imported_fit:
                row["_equation_stale"] = False
                row["_fit_conditions_stale"] = False
            row = self._apply_param_range_validation_to_row(row)
            existing_by_file[file_ref] = row
            applied += 1

        ordered_rows = []
        for source_index, file_ref in enumerate(self.data_files):
            row: Dict[Any, Any] = dict(existing_by_file.get(file_ref) or {})
            row["_source_index"] = int(source_index)
            row["file"] = file_ref
            row["x_channel"] = self.x_channel
            ordered_rows.append(row)

        self.batch_results = ordered_rows
        self._rebuild_batch_capture_keys_from_rows()
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        self.queue_visible_thumbnail_render()
        current_file: Any | None = self._current_loaded_file_path()
        if current_file and self._apply_batch_params_for_file(current_file):
            self.update_plot(fast=False)
        return applied, skipped

    def _apply_fit_details_payload(self, payload, *, source_path=None, auto_load=False):
        if not isinstance(payload, Mapping):
            raise ValueError("Fit details file must contain a JSON object.")

        gui = dict(payload.get("gui") or {})
        saved_analysis_state = gui.get("analysis_state")
        expression_text: str = str(gui.get("equation") or "").strip()
        fit_debug(
            "fit-details apply start: "
            f"source={str(source_path) if source_path is not None else '-'} "
            f"auto_load={bool(auto_load)} "
            f"expr_lines={len(expression_text.splitlines()) if expression_text else 0} "
            f"params={len(list(payload.get('parameters') or []))} "
            f"batch_rows={len(list(payload.get('batch_results') or []))}"
        )

        self._fit_details_restore_in_progress = True
        try:
            if expression_text:
                self.current_expression: str = expression_text
                self._set_expression_editor_text(expression_text)
                if not self.apply_expression_from_input():
                    fit_debug(
                        "fit-details apply rejected stored equation: "
                        f"source={str(source_path) if source_path is not None else '-'} "
                        f"equation={expression_text!r}"
                    )
                    raise ValueError("Failed to apply stored equation.")

            imported_params = list(payload.get("parameters") or [])
            if imported_params:
                spec_by_key = {}
                value_by_key = {}
                for entry in imported_params:
                    if not isinstance(entry, Mapping):
                        continue
                    key: str = str(entry.get("key") or "").strip()
                    if not key:
                        continue
                    fallback: ParameterSpec | None = next(
                        (spec for spec in self.param_specs if spec.key == key), None
                    )
                    if fallback is None:
                        continue
                    min_value: float | None = finite_float_or_none(
                        entry.get("min_value")
                    )
                    max_value: float | None = finite_float_or_none(
                        entry.get("max_value")
                    )
                    if min_value is None:
                        min_value = float(fallback.min_value)
                    if max_value is None:
                        max_value = float(fallback.max_value)
                    if min_value > max_value:
                        min_value, max_value = max_value, min_value
                    default_value = self._param_default_from_limits(
                        min_value, max_value
                    )
                    decimals = self._param_decimals_from_limits(min_value, max_value)
                    spec_by_key[key] = ParameterSpec(
                        key=key,
                        symbol=str(entry.get("symbol") or fallback.symbol),
                        description=str(fallback.description),
                        default=default_value,
                        min_value=min_value,
                        max_value=max_value,
                        decimals=decimals,
                    )
                    periodic_enabled: bool = bool(entry.get("periodic", False))
                    value_by_key[key] = {
                        "value": entry.get("value"),
                        "periodic": periodic_enabled,
                    }

                if spec_by_key:
                    merged_specs = []
                    for spec in self.param_specs:
                        merged_specs.append(spec_by_key.get(spec.key, spec))
                    self.param_specs = merged_specs
                    valid_param_keys: set[str] = {str(spec.key) for spec in self.param_specs}
                    self._periodic_param_keys = {
                        str(key)
                        for key, state in value_by_key.items()
                        if bool(state.get("periodic")) and str(key) in valid_param_keys
                    }
                    self.defaults = self._default_param_values(self.param_specs)
                    self.rebuild_manual_param_controls()
                    self._rebuild_model_segment_info()

                    for spec in self.param_specs:
                        state = value_by_key.get(spec.key)
                        if state is None:
                            continue
                        min_box = self.param_min_spinboxes.get(spec.key)
                        max_box = self.param_max_spinboxes.get(spec.key)
                        spinbox = self.param_spinboxes.get(spec.key)
                        if min_box is None or max_box is None or spinbox is None:
                            continue
                        low = float(min(spec.min_value, spec.max_value))
                        high = float(max(spec.min_value, spec.max_value))
                        min_box.setValue(float(low))
                        max_box.setValue(float(high))
                        value: float | None = finite_float_or_none(state.get("value"))
                        if value is None:
                            value = self._param_default_from_limits(low, high)
                        spinbox.setValue(float(np.clip(value, low, high)))

            saved_channel_names = gui.get("channel_display_names")
            if isinstance(saved_channel_names, Mapping):
                for raw_key, raw_value in saved_channel_names.items():
                    key: str = str(raw_key).strip()
                    if not key:
                        continue
                    alias: str = (
                        str(raw_value).strip() if raw_value not in (None, "") else ""
                    )
                    self.channels[key] = alias or key

            saved_channel_units = gui.get("channel_units")
            if isinstance(saved_channel_units, Mapping):
                for raw_key, raw_value in saved_channel_units.items():
                    key: str = str(raw_key).strip()
                    if not key:
                        continue
                    unit_text: str = (
                        str(raw_value).strip() if raw_value not in (None, "") else ""
                    )
                    self.channel_units[key] = unit_text

            x_channel: str = str(gui.get("x_channel") or "").strip()
            channel_names = list(self._available_channel_names())
            if x_channel in channel_names and hasattr(self, "x_channel_combo"):
                idx: int = self.x_channel_combo.findData(x_channel)
                if idx >= 0:
                    self.x_channel_combo.setCurrentIndex(idx)

            self._set_auto_fit_mode(
                gui.get("auto_fit_mode", self._auto_fit_run_mode),
                autosave=False,
            )
            self._set_batch_fit_mode(
                gui.get("batch_fit_mode", self._batch_fit_run_mode),
                autosave=False,
            )
            self._set_fit_compute_mode(
                gui.get("fit_compute_mode", self._fit_compute_mode),
                autosave=False,
                show_status=False,
            )

            if hasattr(self, "smoothing_toggle_btn"):
                self.smoothing_toggle_btn.setChecked(
                    bool(gui.get("smoothing_enabled", self.smoothing_enabled))
                )
            if hasattr(self, "smoothing_window_spin"):
                window = int(gui.get("smoothing_window", self.smoothing_window) or 1)
                self.smoothing_window_spin.setValue(max(1, window))
            self._on_smoothing_controls_changed()

            if hasattr(self, "regex_input"):
                pattern_text: str = str(gui.get("capture_pattern") or "").strip()
                self.regex_input.blockSignals(True)
                self.regex_input.setText(pattern_text)
                self.regex_input.blockSignals(False)

            mapping = dict(gui.get("capture_to_param") or {})
            self.param_capture_map: Dict[str, str | None] = {
                str(key): (str(value) if value not in (None, "") else None)
                for key, value in mapping.items()
            }
            self._mapped_param_seed_file_key = None
            self._refresh_param_capture_mapping_controls()

            self._refresh_boundary_state_topology(preserve_existing=True)

            # Restore per-channel boundary ratios for multi-channel.
            saved_per_channel = gui.get("boundary_ratios_per_channel")
            if isinstance(saved_per_channel, Mapping):
                multi: Any | None = getattr(self, "_multi_channel_model", None)
                if multi is not None and multi.is_multi_channel:
                    restored = {}
                    for ch_model in multi.channel_models:
                        ch_target = ch_model.target_col
                        ch_ratios_raw = saved_per_channel.get(ch_target)
                        if ch_ratios_raw is not None:
                            ch_ratios: (
                                np.ndarray[Tuple[int, ...], np.dtype[Any]]
                                | np.ndarray[Tuple[int], np.dtype[Any]]
                            ) = self._as_float_array(ch_ratios_raw)
                            n_expected: int = max(0, len(ch_model.segment_exprs) - 1)
                            if ch_ratios.size == n_expected:
                                restored[ch_target] = np.clip(ch_ratios, 0.0, 1.0)
                    if restored:
                        self._fit_state.update_channels(restored)

            # Restore boundary name map.
            saved_name_map = gui.get("boundary_name_map")
            if isinstance(saved_name_map, Mapping) and saved_name_map:
                restored_names: Dict[Tuple[str, int], str] = {}
                for key_str, name in saved_name_map.items():
                    parts: List[str] = str(key_str).rsplit(":", 1)
                    if len(parts) == 2:
                        try:
                            restored_names[(parts[0], int(parts[1]))] = str(name)
                        except (TypeError, ValueError):
                            pass
                self._boundary_name_map: Dict[Tuple[str, int], str] = restored_names
            else:
                self._boundary_name_map = {}
            self._apply_boundary_links_to_model()
            self._refresh_boundary_state_topology(preserve_existing=True)
            self._fit_state.apply_link_groups(
                self._boundary_links_from_map(),
                source_target=self._fit_state.primary_target,
            )

            saved_fixed_boundary_ids = gui.get("fixed_boundary_ids")
            restored_fixed_boundaries = set()
            if isinstance(saved_fixed_boundary_ids, list):
                for raw in saved_fixed_boundary_ids:
                    parts: List[str] = str(raw).rsplit(":", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        restored_fixed_boundaries.add((str(parts[0]), int(parts[1])))
                    except Exception:
                        continue
            self._manually_fixed_boundary_ids = restored_fixed_boundaries
            self._prune_fixed_boundary_ids()
            self._rebuild_boundary_fix_controls()

            # Restore manually fixed parameters.
            saved_fixed = gui.get("manually_fixed_params")
            if isinstance(saved_fixed, list):
                self._manually_fixed_params: set[str] = {str(k) for k in saved_fixed}
            else:
                self._manually_fixed_params = set()
            # Sync fix checkbox states.
            for key, cb in self.param_fix_checkboxes.items():
                cb.blockSignals(True)
                cb.setChecked(key not in self._manually_fixed_params)
                cb.blockSignals(False)
            valid_param_keys: set[str] = {str(spec.key) for spec in self.param_specs}
            self._periodic_param_keys = {
                str(k)
                for k in getattr(self, "_periodic_param_keys", set())
                if str(k) in valid_param_keys
            }
            for key, cb in getattr(self, "_model_param_periodic_checkboxes", {}).items():
                cb.blockSignals(True)
                cb.setChecked(str(key) in self._periodic_param_keys)
                cb.blockSignals(False)

            # Restore fitting procedure.
            saved_procedure = gui.get("procedure")
            if isinstance(saved_procedure, Mapping) and saved_procedure:
                try:
                    self._procedure_panel.restore_from_serialized(saved_procedure)
                except Exception as exc:
                    self.stats_text.append(f"Procedure load warning: {exc}")
                    fit_debug(
                        "fit-details procedure restore warning: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                self._procedure_panel.procedure_steps = []
                self._procedure_panel.procedure_name = "Procedure"
                self._procedure_panel.clear_run_history()

            self._restore_splitter_sizes(
                getattr(self, "main_splitter", None),
                gui.get("main_splitter_sizes"),
            )
            self._restore_splitter_sizes(
                getattr(self, "param_fit_splitter", None),
                gui.get("param_fit_splitter_sizes"),
            )
            QTimer.singleShot(
                0,
                lambda sizes=gui.get(
                    "main_splitter_sizes"
                ): self._restore_splitter_sizes(
                    getattr(self, "main_splitter", None), sizes
                ),
            )
            QTimer.singleShot(
                0,
                lambda sizes=gui.get(
                    "param_fit_splitter_sizes"
                ): self._restore_splitter_sizes(
                    getattr(self, "param_fit_splitter", None), sizes
                ),
            )
            self._sync_param_slider_lock_state()
            self._sync_breakpoint_sliders_from_state()

            applied_rows, skipped_rows = self._apply_imported_batch_rows(payload)
            fit_debug(
                "fit-details apply rows: "
                f"applied={int(applied_rows)} skipped={int(skipped_rows)}"
            )

            if source_path is not None:
                load_mode: str = "Auto-loaded" if auto_load else "Imported"
                self.stats_text.append(
                    f"✓ {load_mode} fit details from {Path(source_path).name}"
                )
            if applied_rows > 0:
                self.stats_text.append(
                    f"Applied fit details to {applied_rows} file(s)."
                )
            if skipped_rows > 0:
                self.stats_text.append(
                    f"Skipped {skipped_rows} saved file row(s) that did not match current files."
                )
            self._refresh_channel_name_references(refresh_plot=False, autosave=False)
            self._restore_analysis_ui_state(saved_analysis_state)
            self.update_plot(fast=False)
        finally:
            self._fit_details_restore_in_progress = False

    @staticmethod
    def _rename_corrupt_fit_details(path: Path) -> Path | None:
        """Rename a corrupt fit_details file with a timestamp suffix."""
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = path.with_suffix(f".corrupt_{stamp}.json")
            path.rename(backup)
            return backup
        except OSError:
            return None

    def _reset_to_default_equation(self) -> None:
        """Reset the equation editor and model back to the built-in default."""
        default_expr = f"{DEFAULT_TARGET_CHANNEL} = {DEFAULT_EXPRESSION}"
        self.current_expression = default_expr
        self._set_expression_editor_text(default_expr)
        self.apply_expression_from_input()
        fit_debug("fit-details: equation reset to default after load failure")

    def _warn_fit_details_failed(
        self, backup: Path | None, reason: str, exc: Exception
    ) -> None:
        """Show a warning popup when fit_details loading fails."""
        error_detail = f"{type(exc).__name__}: {exc}"
        if backup is not None:
            rename_line = f"The original file has been renamed to:\n{backup}"
        else:
            rename_line = "(Could not rename the original file.)"

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Fit Details Load Failed")
        dialog.setText(reason)
        dialog.setInformativeText(f"{error_detail}\n\n{rename_line}")
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        # Force the dialog wide enough to show full paths without clipping.
        from PyQt6.QtWidgets import QSpacerItem, QSizePolicy

        spacer = QSpacerItem(
            600, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )
        layout = dialog.layout()
        if layout is not None:
            layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
        dialog.exec()

    def _load_fit_details_file(self, file_path, *, auto_load) -> bool:
        path: Path = Path(file_path).expanduser()
        fit_debug(
            f"fit-details load requested: path={path} auto_load={bool(auto_load)}"
        )
        if not path.exists():
            fit_debug(
                "fit-details load skipped (missing file): "
                f"path={path} auto_load={bool(auto_load)}"
            )
            if not auto_load:
                self.stats_text.append(f"Fit details file not found: {path}")
            return False
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            backup = self._rename_corrupt_fit_details(path)
            self.stats_text.append(f"✗ Failed to read fit details: {exc}")
            fit_debug(
                "fit-details load read failed: "
                f"path={path} error={type(exc).__name__}: {exc}"
            )
            self._reset_to_default_equation()
            self._warn_fit_details_failed(backup, "Failed to read fit details", exc)
            return False
        try:
            self._apply_fit_details_payload(
                payload, source_path=path, auto_load=auto_load
            )
            fit_debug(f"fit-details load success: path={path}")
            return True
        except Exception as exc:
            backup = self._rename_corrupt_fit_details(path)
            self.stats_text.append(f"✗ Failed to apply fit details: {exc}")
            fit_debug(
                "fit-details load apply failed: "
                f"path={path} error={type(exc).__name__}: {exc}"
            )
            self._reset_to_default_equation()
            self._warn_fit_details_failed(backup, "Failed to apply fit details", exc)
            return False

    def _attempt_fit_details_autoload_once(self, *, reason: str = "") -> bool:
        if bool(getattr(self, "_fit_details_autoload_attempted", False)):
            return False
        self._fit_details_autoload_attempted = True
        loaded: bool = bool(self._autoload_fit_details_from_source())
        fit_debug(
            "fit-details autoload attempt: "
            f"reason={reason or '-'} loaded={loaded}"
        )
        return loaded

    def _autoload_fit_details_from_source(self) -> bool:
        if bool(getattr(self, "_fit_details_restore_in_progress", False)):
            return False
        sidecar: None | Path = self._fit_details_sidecar_path()
        if sidecar is None:
            fit_debug("fit-details autoload skipped: no sidecar path")
            return False
        if not sidecar.exists():
            fit_debug(f"fit-details autoload skipped: sidecar missing path={sidecar}")
            return False
        return self._load_fit_details_file(sidecar, auto_load=True)

    def _upsert_batch_row_from_fit(
        self,
        file_path,
        ordered_keys,
        fit_result,
    ) -> None | int:
        if not file_path:
            return None
        params_by_key = dict((fit_result or {}).get("params_by_key") or {})
        row_params: List[float] = [
            float(params_by_key.get(key, 0.0)) for key in ordered_keys
        ]
        row_r2: float | None = (
            float(fit_result["r2"])
            if fit_result is not None and fit_result.get("r2") is not None
            else None
        )

        row_idx: None | int = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            captures = {}
            pattern_error = None
            capture_config: CapturePatternConfig | None = (
                self._resolve_batch_capture_config(show_errors=False)
            )
            if capture_config is not None:
                extracted: Dict[str, str] | None = extract_captures(
                    stem_for_file_ref(file_path),
                    capture_config.regex,
                    capture_config.defaults,
                )
                if extracted is None:
                    pattern_error: str = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures: Dict[str, str] = dict(extracted)
            row = make_batch_result_row(
                source_index=len(self.batch_results),
                file_path=file_path,
                x_channel=self.x_channel,
                captures=captures,
                pattern_error=pattern_error,
            )
            self.batch_results.append(row)
            row_idx: int = len(self.batch_results) - 1
        else:
            row = canonicalize_fit_row(self.batch_results[row_idx])

        fit_set(row, "params", list(row_params))
        fit_set(row, "r2", row_r2)

        # Preserve known per-channel fit quality/boundaries across subset fits.
        def _normalize_channel_results(raw_results) -> Dict[str, Dict[str, Any]]:
            normalized: Dict[str, Dict[str, Any]] = {}
            if not isinstance(raw_results, Mapping):
                return normalized
            for raw_target, raw_entry in raw_results.items():
                target: str = str(raw_target).strip()
                if not target or not isinstance(raw_entry, Mapping):
                    continue
                entry: Dict[str, Any] = {}
                ch_r2 = finite_float_or_none(raw_entry.get("r2"))
                if ch_r2 is not None:
                    entry["r2"] = float(ch_r2)
                ch_boundary = self._as_float_array(raw_entry.get("boundary_ratios"))
                if ch_boundary.size > 0:
                    entry["boundary_ratios"] = np.clip(ch_boundary, 0.0, 1.0)
                ch_boundaries = self._as_float_array(raw_entry.get("boundaries"))
                if ch_boundaries.size > 0:
                    entry["boundaries"] = ch_boundaries
                if entry:
                    normalized[target] = entry
            return normalized

        merged_channel_results: Dict[str, Dict[str, Any]] = _normalize_channel_results(
            fit_get(row, "channel_results")
        )
        candidate_channel_results: Dict[str, Dict[str, Any]] = (
            _normalize_channel_results(
                fit_result.get("channel_results")
                if isinstance(fit_result, dict)
                else None
            )
        )
        for target, entry in candidate_channel_results.items():
            merged_channel_results[str(target)] = dict(entry)
        fit_set(
            row,
            "channel_results",
            merged_channel_results if merged_channel_results else None,
        )
        row["x_channel"] = self.x_channel
        row["plot_full"] = None
        row["plot"] = None
        row["plot_render_size"] = None
        row["plot_has_fit"] = True
        row["_equation_stale"] = False
        row["_fit_conditions"] = self._fit_conditions_fingerprint()
        row = self._apply_param_range_validation_to_row(row)
        self.batch_results[row_idx] = row
        self._rebuild_batch_capture_keys_from_rows()

        if self.batch_table.rowCount() != len(self.batch_results):
            self.update_batch_table()
        else:
            table_row_idx: None | int = self._find_table_row_by_file(file_path)
            if table_row_idx is None:
                self.update_batch_table()
            else:
                self.update_batch_table_row(table_row_idx, row)
        self._start_thumbnail_render(row_indices=[row_idx])
        return row_idx

    def _on_batch_row_resized_by_user(
        self, _logical_index, _old_size, new_size
    ) -> None:
        if self._batch_row_height_sync:
            return
        self.batch_row_height: int = max(
            self.batch_row_height_min,
            min(self.batch_row_height_max, int(new_size)),
        )
        self._apply_batch_row_heights()
        for row in self.batch_results:
            row_idx: None | int = self._find_table_row_by_file(row["file"])
            if row_idx is not None:
                self._update_batch_plot_cell(row_idx, row)
        self.queue_visible_thumbnail_render()

    def _sync_batch_files_from_shared(self, sync_pattern=True) -> None:
        """Mirror batch files from shared file list (default: all files in folder)."""
        self.batch_files = list(self.data_files)

        if sync_pattern and hasattr(self, "regex_input") and self.batch_files:
            first_name: str = stem_for_file_ref(self.batch_files[0])
            if self.regex_input.text() != first_name:
                self.regex_input.blockSignals(True)
                self.regex_input.setText(first_name)
                self.regex_input.blockSignals(False)

        preview_needed_now = bool(self._batch_preview_ready)
        if hasattr(self, "tabs"):
            preview_needed_now: bool = preview_needed_now or (
                self.tabs.currentWidget() in (self.batch_tab, self.analysis_tab)
            )

        if preview_needed_now:
            self._batch_preview_ready = True
            self.prepare_batch_preview()
            self._expand_file_column_for_selected_files()
        else:
            # Keep startup fast by deferring batch preview/table build
            # until a batch-related tab is opened.
            self._stop_thumbnail_render()
            self.batch_results = []
            self.batch_capture_keys = []
            self.batch_match_count = 0
            self.batch_unmatched_files = []
            self.update_batch_table()

    def _build_batch_result_row(
        self,
        source_index,
        file_path,
        captures,
        pattern_error=None,
        existing=None,
        preserve_fit_result=True,
    ):
        existing_row = canonicalize_fit_row(existing)
        existing_plot_full = None
        existing_plot = None
        existing_plot_render_size = None
        existing_plot_has_fit = existing_row.get("plot_has_fit")
        if existing_plot_has_fit is None:
            existing_plot_has_fit: bool = has_nonempty_values(
                fit_get(existing_row, "params")
            )
        if preserve_fit_result:
            existing_plot_full = existing_row.get("plot_full")
            if existing_plot_full is None:
                existing_plot_full = existing_row.get("plot")
            if existing_plot_full is None:
                existing_plot_full = existing_row.get("thumbnail")
            existing_plot = existing_row.get("plot")
            existing_plot_render_size = existing_row.get("plot_render_size")
        elif not existing_plot_has_fit:
            existing_plot_full = existing_row.get("plot_full")
            if existing_plot_full is None:
                existing_plot_full = existing_row.get("plot")
            if existing_plot_full is None:
                existing_plot_full = existing_row.get("thumbnail")
            existing_plot = existing_row.get("plot")
            existing_plot_render_size = existing_row.get("plot_render_size")
        row = make_batch_result_row(
            source_index=source_index,
            file_path=file_path,
            x_channel=self.x_channel,
            captures=captures,
            params=fit_get(existing_row, "params") if preserve_fit_result else None,
            r2=fit_get(existing_row, "r2") if preserve_fit_result else None,
            error=fit_get(existing_row, "error") if preserve_fit_result else None,
            plot_full=existing_plot_full,
            plot=existing_plot,
            plot_has_fit=(
                bool(existing_plot_has_fit) if existing_plot_full is not None else None
            ),
            plot_render_size=(
                existing_plot_render_size if existing_plot_full is not None else None
            ),
            pattern_error=pattern_error,
            equation_stale=bool(existing_row.get("_equation_stale")),
        )
        existing_ch_results = (
            fit_get(existing_row, "channel_results") if preserve_fit_result else None
        )
        fit_set(
            row,
            "channel_results",
            existing_ch_results if isinstance(existing_ch_results, Mapping) else None,
        )
        return self._apply_param_range_validation_to_row(row)

    def _source_dialog_start_dir(self) -> Path:
        current_path: Path = Path(self.current_dir).expanduser()
        start_dir: Path = (
            current_path.parent if current_path.is_file() else current_path
        )
        if not start_dir.exists():
            selected: List[Any] = list(
                getattr(self, "_source_selected_paths", []) or []
            )
            if selected:
                selected_parent: Path = Path(selected[0]).expanduser().parent
                if selected_parent.exists():
                    start_dir: Path = selected_parent
        if not start_dir.exists():
            start_dir: Path = Path.cwd()
        return start_dir

    def _confirm_clear_batch_results(self, action_label) -> bool:
        rows: List[Any] = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return True

        action_text: str = str(action_label or "this action").strip() or "this action"
        row_count: int = len(rows)
        plural: str = "s" if row_count != 1 else ""
        reply: QMessageBox.StandardButton = QMessageBox.question(
            self,
            "Clear Batch Results?",
            (
                f"{action_text} will clear {row_count} existing batch result{plural}.\n"
                "Do you want to continue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _prompt_batch_results_on_rerun(self) -> str:
        rows: List[Any] = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return "keep"

        row_count: int = len(rows)
        plural: str = "s" if row_count != 1 else ""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Run Batch")
        dialog.setText(f"Batch rerun found {row_count} existing result{plural}.")
        dialog.setInformativeText(
            "Choose whether to keep existing results for comparison/seeding, or clear them first."
        )

        keep_btn: QPushButton | None = dialog.addButton(
            "Proceed and Keep", QMessageBox.ButtonRole.AcceptRole
        )
        clear_btn: QPushButton | None = dialog.addButton(
            "Proceed and Clear", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn: QPushButton | None = dialog.addButton(
            QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(keep_btn)

        dialog.exec()
        clicked: QAbstractButton | None = dialog.clickedButton()
        if clicked == cancel_btn:
            return "cancel"
        if clicked == clear_btn:
            return "clear"
        return "keep"

    def _prompt_batch_results_on_equation_change(self) -> str:
        rows: List[Any] = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return "wipe"

        row_count: int = len(rows)
        plural: str = "s" if row_count != 1 else ""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Equation Changed")
        dialog.setText(
            f"Changing the equation affects {row_count} existing batch result{plural}."
        )
        dialog.setInformativeText(
            "Choose whether to keep old batch results or clear them before continuing."
        )

        keep_btn: QPushButton | None = dialog.addButton(
            "Proceed and Keep", QMessageBox.ButtonRole.AcceptRole
        )
        wipe_btn: QPushButton | None = dialog.addButton(
            "Proceed and Clear", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn: QPushButton | None = dialog.addButton(
            QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(keep_btn)

        dialog.exec()
        clicked: QAbstractButton | None = dialog.clickedButton()
        if clicked == cancel_btn:
            return "cancel"
        if clicked == wipe_btn:
            return "wipe"
        return "keep"

    def _stop_background_data_preload(self, *, wait_ms: int = 80) -> None:
        worker: DataPreloadWorker | None = getattr(self, "_data_preload_worker", None)
        thread: QThread | None = getattr(self, "_data_preload_thread", None)
        self._request_worker_cancel(worker)
        stopped = True
        if thread is not None:
            if bool(thread.isRunning()):
                stopped = bool(
                    self._shutdown_thread(
                        thread,
                        wait_ms=wait_ms,
                        force_terminate=True,
                    )
                )
            else:
                try:
                    thread.deleteLater()
                except Exception:
                    pass
        if stopped:
            self._data_preload_worker = None
            self._data_preload_thread = None

    def _start_background_data_preload(self, *, prioritize_file: str | None = None) -> None:
        files: List[str] = [str(ref).strip() for ref in list(self.data_files or [])]
        files = [ref for ref in files if ref]
        if not files:
            return

        self._stop_background_data_preload(wait_ms=80)
        if getattr(self, "_data_preload_thread", None) is not None:
            fit_debug("data-preload start skipped: previous worker still stopping")
            return
        self._data_preload_session = int(getattr(self, "_data_preload_session", 0)) + 1
        session_id: int = int(self._data_preload_session)

        queue: List[str] = []
        preferred: str = str(prioritize_file or "").strip()
        if (
            preferred
            and preferred in files
            and preferred not in self._data_preload_cache
            and preferred not in self._data_preload_failed
        ):
            queue.append(preferred)
        for ref in files:
            if ref in self._data_preload_cache:
                continue
            if ref in self._data_preload_failed:
                continue
            if ref in queue:
                continue
            queue.append(ref)
        if not queue:
            return

        worker = DataPreloadWorker(session_id, queue)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.file_loaded.connect(self._on_background_data_preload_loaded)
        worker.finished.connect(self._on_background_data_preload_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._data_preload_worker = worker
        self._data_preload_thread = thread
        fit_debug(
            "data-preload start: "
            f"session={session_id} files={len(queue)} cached={len(self._data_preload_cache)}"
        )
        thread.start()

    def _on_background_data_preload_loaded(
        self,
        session_id: int,
        file_ref: str,
        frame: Any,
        error: Any,
    ) -> None:
        if int(session_id) != int(getattr(self, "_data_preload_session", 0)):
            return
        file_key: str = str(file_ref).strip()
        if not file_key:
            return
        if frame is not None:
            self._data_preload_cache[file_key] = frame
            self._data_preload_failed.discard(file_key)
            return
        if error not in (None, ""):
            self._data_preload_failed.add(file_key)
            fit_debug(
                "data-preload file failed: "
                f"file={stem_for_file_ref(file_key)} error={error}"
            )

    def _on_background_data_preload_finished(self, session_id: int) -> None:
        if int(session_id) != int(getattr(self, "_data_preload_session", 0)):
            return
        fit_debug(
            "data-preload done: "
            f"session={session_id} cached={len(self._data_preload_cache)}"
        )
        self._data_preload_worker = None
        self._data_preload_thread = None
        pending = [
            str(ref).strip()
            for ref in list(self.data_files or [])
            if str(ref).strip()
            and str(ref).strip() not in self._data_preload_cache
            and str(ref).strip() not in self._data_preload_failed
        ]
        if pending:
            active_file: Any | None = self._current_loaded_file_path()
            self._start_background_data_preload(
                prioritize_file=str(active_file) if active_file else None
            )

    def _cancel_idle_archive_scan(self) -> None:
        timer: QTimer | None = getattr(self, "_idle_archive_scan_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        stream: Any | None = getattr(self, "_idle_archive_scan_stream", None)
        if stream is not None:
            close_method = getattr(stream, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
        self._idle_archive_scan_queue = []
        self._idle_archive_scan_total = 0
        self._idle_archive_scan_done = 0
        self._idle_archive_scan_added = 0
        self._idle_archive_scan_stream = None
        self._idle_archive_scan_current_archive = None
        self._idle_archive_scan_current_found = 0
        self._idle_archive_scan_session = int(
            getattr(self, "_idle_archive_scan_session", 0)
        ) + 1

    def _queue_idle_archive_scan(self, archive_paths) -> None:
        deduped: List[str] = []
        seen: set[str] = set()
        for raw_path in list(archive_paths or []):
            text: str = str(raw_path).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        if not deduped:
            return

        self._idle_archive_scan_session = int(
            getattr(self, "_idle_archive_scan_session", 0)
        ) + 1
        self._idle_archive_scan_queue = deduped
        self._idle_archive_scan_total = len(deduped)
        self._idle_archive_scan_done = 0
        self._idle_archive_scan_added = 0
        self._idle_archive_scan_stream = None
        self._idle_archive_scan_current_archive = None
        self._idle_archive_scan_current_found = 0
        self.stats_text.setText(
            f"Scanning {self._idle_archive_scan_total} archive(s) in background..."
        )
        timer: QTimer | None = getattr(self, "_idle_archive_scan_timer", None)
        if timer is not None:
            if timer.isActive():
                timer.stop()
            timer.start(0)

    def _append_archive_file_refs(self, file_refs) -> int:
        refs: List[str] = [
            str(ref).strip() for ref in list(file_refs or []) if str(ref).strip()
        ]
        if not refs:
            return 0
        existing: set[str] = {
            str(item).strip() for item in list(self.data_files or [])
        }
        new_refs: List[str] = [ref for ref in refs if ref not in existing]
        if not new_refs:
            return 0

        combo: Any | None = getattr(self, "file_combo", None)
        combo_was_blocked: bool = (
            bool(combo.blockSignals(True)) if combo is not None else False
        )
        try:
            for ref in new_refs:
                self.data_files.append(ref)
                if combo is not None:
                    combo.addItem(stem_for_file_ref(ref), ref)
        finally:
            if combo is not None:
                combo.blockSignals(combo_was_blocked)

        self._sync_file_navigation_buttons()
        self._sync_batch_files_from_shared(sync_pattern=False)
        return len(new_refs)

    def _process_idle_archive_scan(self) -> None:
        timer: QTimer | None = getattr(self, "_idle_archive_scan_timer", None)
        queue: List[str] = list(getattr(self, "_idle_archive_scan_queue", []) or [])
        stream: Any | None = getattr(self, "_idle_archive_scan_stream", None)
        total: int = max(1, int(getattr(self, "_idle_archive_scan_total", 0)))
        done: int = int(getattr(self, "_idle_archive_scan_done", 0))

        if stream is None:
            if not queue:
                added: int = int(getattr(self, "_idle_archive_scan_added", 0))
                if int(getattr(self, "_idle_archive_scan_total", 0)) > 0:
                    if added > 0:
                        if self.current_data is None:
                            self.stats_text.setText(
                                f"Archive scan complete: added {added} file(s). Select a file to load."
                            )
                        else:
                            self.stats_text.setText(
                                f"Archive scan complete: added {added} file(s)."
                            )
                    elif not self.data_files:
                        self.stats_text.setText("No CSV files found in selected source.")
                self._idle_archive_scan_total = 0
                self._idle_archive_scan_done = 0
                self._idle_archive_scan_added = 0
                self._idle_archive_scan_current_archive = None
                self._idle_archive_scan_current_found = 0
                # Apply one full batch sync after archive expansion completes.
                if self.data_files:
                    self._sync_batch_files_from_shared(sync_pattern=True)
                    # If the source started as archive-only, we may have had no
                    # file loaded when scanning began. Load the first readable
                    # file now so the main plot is populated.
                    if self.current_data is None:
                        candidate_indices: List[int] = []
                        preferred_idx = int(getattr(self, "current_file_idx", 0))
                        if 0 <= preferred_idx < len(self.data_files):
                            candidate_indices.append(preferred_idx)
                        if 0 not in candidate_indices:
                            candidate_indices.append(0)
                        for idx in candidate_indices:
                            if self.load_file(idx, report_errors=False):
                                break
                    self._attempt_fit_details_autoload_once(
                        reason="archive_scan_complete"
                    )
                    if getattr(self, "_data_preload_thread", None) is None:
                        active_file: Any | None = self._current_loaded_file_path()
                        self._start_background_data_preload(
                            prioritize_file=str(active_file) if active_file else None
                        )
                return

            archive_path: str = str(queue.pop(0))
            self._idle_archive_scan_queue = queue
            self._idle_archive_scan_current_archive = archive_path
            self._idle_archive_scan_current_found = 0
            try:
                stream = open_archive_csv_member_stream(archive_path)
            except Exception as exc:
                archive_name: str = Path(archive_path).name
                self.stats_text.setText(
                    f"Archive scan failed for '{archive_name}': {exc}"
                )
                self._idle_archive_scan_done = done + 1
                if timer is not None:
                    timer.start(10)
                return
            self._idle_archive_scan_stream = stream

        archive_path = str(getattr(self, "_idle_archive_scan_current_archive", "") or "")
        archive_name: str = Path(archive_path).name if archive_path else "archive"

        members_batch: List[str] = []
        try:
            members_batch = list(stream.next_batch(max_items=96))
        except Exception as exc:
            self.stats_text.setText(f"Archive scan failed for '{archive_name}': {exc}")
            close_method = getattr(stream, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
            self._idle_archive_scan_stream = None
            self._idle_archive_scan_current_archive = None
            self._idle_archive_scan_current_found = 0
            self._idle_archive_scan_done = done + 1
            if timer is not None:
                timer.start(10)
            return

        if members_batch:
            added_count: int = self._append_archive_file_refs(
                [f"{archive_path}::{member}" for member in members_batch]
            )
            self._idle_archive_scan_added = int(
                getattr(self, "_idle_archive_scan_added", 0)
            ) + int(max(0, added_count))
            self._idle_archive_scan_current_found = int(
                getattr(self, "_idle_archive_scan_current_found", 0)
            ) + int(len(members_batch))

        done_for_archive: bool = bool(getattr(stream, "done", False))
        if done_for_archive:
            close_method = getattr(stream, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
            found_count: int = int(getattr(self, "_idle_archive_scan_current_found", 0))
            self._idle_archive_scan_stream = None
            self._idle_archive_scan_current_archive = None
            self._idle_archive_scan_current_found = 0
            self._idle_archive_scan_done = done + 1
            self.stats_text.setText(
                f"Scanned archive {self._idle_archive_scan_done}/{total} '{archive_name}' ({found_count} member(s))."
            )
            QCoreApplication.processEvents()
            if timer is not None:
                timer.start(10)
            return

        found_count: int = int(getattr(self, "_idle_archive_scan_current_found", 0))
        self.stats_text.setText(
            f"Scanning archive {done + 1}/{total} '{archive_name}'... ({found_count} member(s) found)"
        )
        QCoreApplication.processEvents()
        if timer is not None:
            timer.start(0)

    def _apply_data_file_list(
        self,
        files,
        *,
        empty_message,
        confirm_clear_batch: bool = True,
    ) -> bool:
        deduped_files = []
        seen = set()
        for file_ref in files:
            text: str = str(file_ref).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped_files.append(text)

        if bool(confirm_clear_batch):
            if not self._confirm_clear_batch_results("Loading a new data source"):
                self._last_source_load_cancelled = True
                self.stats_text.append("Load cancelled; existing batch results kept.")
                return False
        self._last_source_load_cancelled = False
        self._stop_background_data_preload(wait_ms=80)
        self._data_preload_cache = {}
        self._data_preload_failed = set()
        self._fit_details_autoload_attempted = False

        self.data_files = deduped_files
        self.file_combo.clear()
        self.current_file_idx = 0

        if not self.data_files:
            self.current_data = None
            self.cached_time_data = None
            self.raw_channel_cache = {}
            self.channel_cache = {}
            self._expression_channel_data_cache = None
            self._last_file_load_error: str = ""
            self._sync_batch_files_from_shared(sync_pattern=False)
            self.stats_text.setText(str(empty_message))
            self._clear_main_plot("No data loaded.")
            self._sync_file_navigation_buttons()
            self._refresh_fit_action_buttons()
            return False

        for file_ref in self.data_files:
            self.file_combo.addItem(stem_for_file_ref(file_ref), file_ref)
        self._sync_file_navigation_buttons()

        self._sync_batch_files_from_shared(sync_pattern=True)
        loaded_ok = False
        loaded_idx = -1
        candidate_indices: List[int] = []
        preferred_idx = 0
        if 0 <= int(getattr(self, "current_file_idx", 0)) < len(self.data_files):
            preferred_idx = int(getattr(self, "current_file_idx", 0))
        if preferred_idx not in candidate_indices:
            candidate_indices.append(preferred_idx)
        if 0 not in candidate_indices:
            candidate_indices.append(0)
        # Keep startup responsive: probe only a few files instead of scanning
        # the full list before showing the GUI as ready.
        max_probe = min(4, len(self.data_files))
        for idx in range(max_probe):
            if idx not in candidate_indices:
                candidate_indices.append(idx)
        for idx in candidate_indices:
            if self.load_file(idx, report_errors=False):
                loaded_ok = True
                loaded_idx = int(idx)
                break

        if loaded_ok:
            if loaded_idx > 0:
                loaded_name: str = stem_for_file_ref(self.data_files[loaded_idx])
                self.stats_text.setText(
                    f"Loaded '{loaded_name}' after skipping {loaded_idx} unreadable source file(s)."
                )
            elif self.stats_text.text().strip() == "Loading data sources...":
                self.stats_text.clear()
            self._attempt_fit_details_autoload_once(reason="initial_file_load")
            if getattr(self, "_data_preload_thread", None) is None:
                loaded_file_ref: str = str(self.data_files[loaded_idx]).strip()
                self._start_background_data_preload(prioritize_file=loaded_file_ref)
            self._sync_file_navigation_buttons()
            return True

        detail: str = self._last_file_load_error or "No readable data found."
        self.stats_text.setText(f"Failed to load any selected data source. {detail}")
        self._clear_main_plot("No readable data loaded.")
        self._sync_file_navigation_buttons()
        return False

    def _load_selected_csv_files(self, csv_paths) -> bool:
        selected_csv = []
        for path_text in csv_paths:
            path_obj: Path = Path(path_text).expanduser()
            if path_obj.is_file() and path_obj.suffix.lower() == ".csv":
                selected_csv.append(str(path_obj))

        if not selected_csv:
            self.stats_text.append("No valid CSV files were selected.")
            return False

        self.current_dir = str(Path(selected_csv[0]).parent)
        count: int = len(selected_csv)
        plural: str = "s" if count != 1 else ""
        self._source_display_override: str = f"{count} selected CSV file{plural}"
        self._source_selected_paths = list(selected_csv)
        self._refresh_source_path_label()
        loaded_ok: bool = self._apply_data_file_list(
            selected_csv,
            empty_message="No readable CSV files found in selected set.",
        )
        if loaded_ok:
            self.stats_text.append(f"Loaded {count} selected CSV file{plural}.")
        return loaded_ok

    def load_files(self) -> None:
        """Load CSV sources from a directory root, archive, or single CSV file."""
        self._cancel_idle_archive_scan()
        source_path: Path = Path(self.current_dir).expanduser()
        files: List[str] = []
        archive_paths: List[str] = []
        empty_message = "No CSV files found in selected source."

        self._source_display_override = None
        self._source_selected_paths = []

        if source_path.is_dir():
            csv_files: List[str] = []
            archive_candidates: List[str] = []
            for candidate in sorted(source_path.rglob("*")):
                if not candidate.is_file():
                    continue
                suffix: str = candidate.suffix.lower()
                if suffix == ".csv":
                    csv_files.append(str(candidate))
                elif is_supported_archive_path(candidate):
                    archive_candidates.append(str(candidate))
            files = list(csv_files)
            archive_paths = list(archive_candidates)
            if not files and archive_paths:
                empty_message = "Scanning archive members in background..."
        elif source_path.is_file() and is_supported_archive_path(source_path):
            files = []
            archive_paths = [str(source_path)]
            empty_message = "Scanning archive members in background..."
        elif source_path.is_file() and source_path.suffix.lower() == ".csv":
            files: List[str] = [str(source_path)]
        elif not source_path.exists():
            empty_message: str = f"Selected source does not exist: {source_path}"

        self._refresh_source_path_label()
        self._apply_data_file_list(
            files,
            empty_message=empty_message,
            confirm_clear_batch=not bool(source_path.is_dir()),
        )
        if archive_paths and not bool(getattr(self, "_last_source_load_cancelled", False)):
            self._queue_idle_archive_scan(archive_paths)

    def _clear_main_plot(self, message="No data loaded.") -> None:
        if not hasattr(self, "fig") or not hasattr(self, "canvas"):
            return
        self.fig.clear()
        self.ax: Axes = self.fig.add_subplot(111)
        self.ax_residual = None
        self._plot_has_residual_axis = False
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel(
            self._channel_axis_label(self.x_channel)
            if hasattr(self, "x_channel")
            else "X"
        )
        primary_target: str = self._primary_target_channel()
        self.ax.set_ylabel(
            self._channel_axis_label(primary_target) if primary_target else "Signal"
        )
        if message:
            self.ax.text(
                0.5,
                0.5,
                str(message),
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
                fontsize=10,
            )
        self._reset_plot_home_view()
        self.canvas.draw_idle()

    def load_file(self, idx, *, report_errors=True) -> bool:
        """Load a specific file."""
        if self._file_load_in_progress:
            return False
        if idx < 0 or idx >= len(self.data_files):
            return False

        self._file_load_in_progress = True
        loaded_ok = False
        self._last_file_load_error: str = ""
        try:
            self.current_file_idx = idx
            if self.file_combo.currentIndex() != idx:
                self.file_combo.blockSignals(True)
                self.file_combo.setCurrentIndex(idx)
                self.file_combo.blockSignals(False)
            self.last_popt = None
            self._last_r2 = None
            self._last_per_channel_r2 = {}

            try:
                file_path = self.data_files[idx]
                file_name: str = stem_for_file_ref(file_path)
                load_started_at: float = time.monotonic()
                last_progress_ui_time: float = 0.0

                def _on_file_load_progress(
                    message: str, fraction: None | float = None
                ) -> None:
                    nonlocal last_progress_ui_time
                    now: float = time.monotonic()
                    if (
                        fraction not in (None, 0.0, 1.0)
                        and (now - last_progress_ui_time) < 0.06
                    ):
                        return
                    last_progress_ui_time = now
                    text: str = f"Loading '{file_name}'"
                    detail: str = str(message or "").strip()
                    if detail:
                        text = f"{text}: {detail}"
                    if fraction is not None:
                        pct: int = int(
                            round(max(0.0, min(1.0, float(fraction))) * 100.0)
                        )
                        text = f"{text} ({pct}%)"
                    self.stats_text.setText(text)
                    QCoreApplication.processEvents()

                cached_frame = self._data_preload_cache.get(str(file_path))
                if cached_frame is not None:
                    self.current_data = cached_frame
                    fit_debug(f"data-preload cache hit: file={stem_for_file_ref(file_path)}")
                else:
                    _on_file_load_progress("Preparing...", 0.0)
                    self.current_data = read_measurement_csv(
                        file_path,
                        progress_cb=_on_file_load_progress,
                    )
                    self._data_preload_cache[str(file_path)] = self.current_data
                self._data_preload_failed.discard(str(file_path))
                # Cache data for faster updates
                time_src = (
                    "TIME"
                    if "TIME" in self.current_data.columns
                    else self.current_data.columns[0]
                )
                self.cached_time_data = (
                    self.current_data[time_src].to_numpy(dtype=float, copy=True) * 1e3
                )
                self.raw_channel_cache = {}
                self.channel_cache = {}
                for col in self.current_data.columns:
                    try:
                        self.raw_channel_cache[col] = self.current_data[col].to_numpy(
                            dtype=float,
                            copy=True,
                        )
                    except Exception:
                        continue
                self._rebuild_channel_cache_from_raw()
                self._refresh_channel_combos()
                self._apply_param_spec_defaults_to_controls()
                has_valid_fit_to_load: bool = self._has_valid_batch_fit_for_file(
                    file_path
                )
                self._apply_batch_params_for_file(file_path)
                # If this file already has a stored fitted row, preserve those
                # values on load instead of immediately reseeding from bound fields.
                if has_valid_fit_to_load:
                    self._mapped_param_seed_file_key = self._fit_task_file_key(file_path)
                else:
                    self._mapped_param_seed_file_key = None
                # Refresh capture mapping controls after restoring any batch row
                # so per-file seed values are re-applied when no valid fit exists.
                self._refresh_param_capture_mapping_controls(
                    allow_seed_for_fixed=(not has_valid_fit_to_load)
                )
                self.update_plot(fast=False, preserve_view=False)
                self._reset_plot_home_view()
                load_elapsed_s: float = max(0.0, time.monotonic() - load_started_at)
                self.stats_text.setText(
                    f"Loaded '{file_name}' ({load_elapsed_s:.2f}s)."
                )
                loaded_ok = True
                if len(self._data_preload_cache) < len(self.data_files):
                    if getattr(self, "_data_preload_thread", None) is None:
                        self._start_background_data_preload(
                            prioritize_file=str(file_path)
                        )
            except Exception as e:
                self.current_data = None
                self.cached_time_data = None
                self.raw_channel_cache = {}
                self.channel_cache = {}
                self._expression_channel_data_cache = None
                file_path = self.data_files[idx]
                file_name: str = stem_for_file_ref(file_path)
                self._data_preload_failed.add(str(file_path))
                self._last_file_load_error: str = f"Error loading '{file_name}': {e}"
                if report_errors:
                    self.stats_text.setText(self._last_file_load_error)
                    self._clear_main_plot(f"Failed to load: {file_name}")
        finally:
            self._file_load_in_progress = False
            self._sync_file_navigation_buttons()
            self._refresh_fit_action_buttons()
        return loaded_ok

    def on_file_changed(self, idx) -> None:
        """Handle file selection change."""
        if self._file_load_in_progress:
            return
        if idx >= 0:
            self.load_file(idx)

    def prev_file(self) -> None:
        """Load previous file."""
        if self.current_file_idx > 0:
            self.load_file(self.current_file_idx - 1)

    def next_file(self) -> None:
        """Load next file."""
        if self.current_file_idx < len(self.data_files) - 1:
            self.load_file(self.current_file_idx + 1)

    def get_current_params(self):
        """Get current parameter values."""
        result = []
        for idx, spec in enumerate(self.param_specs):
            sb = self.param_spinboxes.get(spec.key)
            if sb is not None:
                result.append(float(sb.value()))
            elif idx < len(self.defaults):
                result.append(float(self.defaults[idx]))
            else:
                result.append(0.0)
        return result

    def _fit_conditions_fingerprint(self):
        """Return a hashable fingerprint of the current fit constraints.

        The fingerprint captures every setting that, if changed, means a
        previous fit result is no longer comparable via R² – i.e. the old
        result was optimised under different constraints.

        Included:
          * equation text
          * set of manually fixed parameter keys
          * set of periodic parameter keys
          * set of manually fixed boundary IDs
          * set of enabled fit channels
        """
        expr = str(getattr(self, "current_expression", "") or "")
        fixed_params: frozenset[str] = frozenset(
            str(k) for k in getattr(self, "_manually_fixed_params", set())
        )
        periodic_params: frozenset[str] = frozenset(
            str(k) for k in getattr(self, "_periodic_param_keys", set())
        )
        fixed_bids = frozenset(getattr(self, "_manually_fixed_boundary_ids", set()))
        enabled_channels: Tuple[Any, ...] = tuple(
            sorted(self._get_enabled_fit_channels())
        )
        return (expr, fixed_params, periodic_params, fixed_bids, enabled_channels)

    def _restore_fitted_state_if_available(self) -> None:
        """Re-apply the stored batch fit result for the current file.

        Called when a fixed boundary or parameter is released so that
        values snap back to their last optimised positions.
        """
        current_file: Any | None = self._current_loaded_file_path()
        if not current_file:
            return
        if self._apply_batch_params_for_file(current_file):
            self.update_plot(fast=False)

    def _has_valid_batch_fit_for_file(self, file_path) -> bool:
        """Return whether this file has a usable stored fit row."""
        if not file_path or not getattr(self, "batch_results", None):
            return False

        row_idx: None | int = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            return False
        row = self.batch_results[row_idx]
        params = fit_get(row, "params")
        if params is None:
            return False
        try:
            params = list(np.asarray(params, dtype=float).reshape(-1))
        except Exception:
            return False
        return len(params) >= len(self.param_specs)

    def _apply_batch_params_for_file(self, file_path):
        """Apply batch-fitted parameters for this file if available."""
        if not file_path or not getattr(self, "batch_results", None):
            return False

        row_idx: None | int = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            return False
        matched_row = self.batch_results[row_idx]
        row_is_stale: bool = bool(
            matched_row.get("_equation_stale")
            or matched_row.get("_fit_conditions_stale")
        )

        params = fit_get(matched_row, "params")
        if params is None:
            return False
        try:
            params: List[Any] = list(np.asarray(params, dtype=float).reshape(-1))
        except Exception:
            return False
        if len(params) < len(self.param_specs):
            return False

        changed = False
        boundary_changed = False

        # Block spinbox signals while applying batch values so that
        # intermediate update_plot calls don't fire with stale boundaries.
        blocked_spinboxes = []
        updated_param_keys: List[str] = []
        for idx, spec in enumerate(self.param_specs):
            spinbox = self.param_spinboxes.get(spec.key)
            if spinbox is None:
                continue
            try:
                value = float(params[idx])
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            if not np.isclose(float(spinbox.value()), value):
                changed = True
            was_blocked = spinbox.blockSignals(True)
            spinbox.setValue(value)
            updated_param_keys.append(str(spec.key))
            if not was_blocked:
                blocked_spinboxes.append(spinbox)
        # Unblock after all values are set.
        for spinbox in blocked_spinboxes:
            spinbox.blockSignals(False)
        for key in updated_param_keys:
            self._sync_slider_from_spinbox(key)
        self._refresh_boundary_state_topology(preserve_existing=True)
        boundary_source_targets = []
        applied_boundary_targets: Set[str] = set()
        has_any_stored_boundaries: bool = False

        # Restore per-channel boundary ratios from batch row (multi-channel).
        channel_results = fit_get(matched_row, "channel_results")
        multi: Any | None = getattr(self, "_multi_channel_model", None)
        if (
            (not row_is_stale)
            and
            isinstance(channel_results, dict)
            and multi is not None
            and multi.is_multi_channel
        ):
            for ch_target, ch_result in channel_results.items():
                ch_ratios = (
                    ch_result.get("boundary_ratios")
                    if isinstance(ch_result, Mapping)
                    else None
                )
                if ch_ratios is not None:
                    has_any_stored_boundaries = True
                    ch_key: str = str(ch_target).strip()
                    if not ch_key:
                        continue
                    try:
                        ch_b: np.ndarray[Tuple[int], np.dtype[Any]] = np.asarray(
                            ch_ratios, dtype=float
                        ).reshape(-1)
                    except Exception:
                        continue
                    expected_n: int = int(self._fit_state.channel_count(ch_key))
                    if expected_n <= 0:
                        continue
                    if ch_b.size != expected_n:
                        fit_debug(
                            "apply-batch-boundary SKIPPED channel (size mismatch): "
                            f"file={file_path} "
                            f"target={ch_key} "
                            f"b.size={int(ch_b.size)} "
                            f"expected={int(expected_n)}"
                        )
                        continue
                    if self._fit_state.set_channel_ratios(ch_key, np.clip(ch_b, 0.0, 1.0)):
                        boundary_changed = True
                    boundary_source_targets.append(ch_key)
                    applied_boundary_targets.add(ch_key)

        # Avoid leaking boundary state from previously loaded files when this row
        # does not carry per-channel ratios for every target.
        if (
            (not row_is_stale)
            and has_any_stored_boundaries
            and multi is not None
            and multi.is_multi_channel
        ):
            for target in self._fit_state.targets():
                target_key: str = str(target).strip()
                if not target_key or target_key in applied_boundary_targets:
                    continue
                n_target: int = int(self._fit_state.channel_count(target_key))
                if n_target <= 0:
                    continue
                if self._fit_state.set_channel_ratios(
                    target_key, default_boundary_ratios(n_target)
                ):
                    boundary_changed = True

        seen_sources = set()
        ordered_sources = []
        for source_target in boundary_source_targets:
            key = str(source_target)
            if not key or key in seen_sources:
                continue
            seen_sources.add(key)
            ordered_sources.append(key)
        for source_target in ordered_sources:
            changed_targets: Set[str] = self._fit_state.apply_link_groups(
                self._boundary_links_from_map(),
                source_target=source_target,
                prefer_targets=[source_target],
            )
            if changed_targets:
                boundary_changed = True
        self._sync_breakpoint_sliders_from_state()
        self._refresh_param_value_error_highlighting()
        return bool(changed or boundary_changed)

    def reset_params_from_batch(self) -> None:
        """Load parameter values for the current file from batch results."""
        current_file: Any | None = self._current_loaded_file_path()
        if not current_file:
            self.stats_text.append("No current file loaded.")
            return
        if self._is_file_fit_active(current_file):
            self.stats_text.append("Cannot reset from batch while a fit is running.")
            return
        row_idx: None | int = self._find_batch_result_index_by_file(current_file)
        if row_idx is None or not has_nonempty_values(
            fit_get(self.batch_results[row_idx], "params")
        ):
            self.stats_text.append("No batch-fit parameters found for this file.")
            return
        changed = self._apply_batch_params_for_file(current_file)
        if changed:
            self.update_plot(fast=False)
            self.stats_text.append(
                "Loaded parameters from batch table for current file."
            )
        # else: do nothing (no dialog, no message)

    def clear_previous_result_for_current_file(self) -> None:
        """Clear stored fit output for the currently loaded file."""
        current_file: Any | None = self._current_loaded_file_path()
        if not current_file:
            self.stats_text.append("No current file loaded.")
            return
        if self._is_file_fit_active(current_file):
            self.stats_text.append(
                "Cannot clear previous result while a fit is running."
            )
            return
        row_idx: None | int = self._find_batch_result_index_by_file(current_file)
        if row_idx is None:
            self.stats_text.append("No stored result found for this file.")
            return

        row = canonicalize_fit_row(self.batch_results[row_idx])
        has_stored_result: bool = self._batch_row_has_stored_result(
            row, include_errors=True
        )
        if not has_stored_result:
            self.stats_text.append("No previous fit result to clear for this file.")
            return

        fit_set(row, "params", None)
        fit_set(row, "r2", None)
        fit_set(row, "error", None)
        fit_set(row, "channel_results", None)
        row["plot_full"] = None
        row["plot"] = None
        row["plot_render_size"] = None
        row["plot_has_fit"] = None
        for stale_key in (
            "_fit_conditions",
            "_fit_conditions_stale",
            "_fit_no_change",
            "_seed_source",
            "_seed_source_file",
            "_procedure_result",
        ):
            row.pop(stale_key, None)
        row["_fit_status"] = None
        row["_queue_position"] = None
        row["_r2_old"] = None
        row["_fit_task_id"] = None

        self.batch_results[row_idx] = canonicalize_fit_row(row)
        table_row_idx: None | int = self._find_table_row_by_file(current_file)
        if table_row_idx is None:
            self.update_batch_table()
        else:
            self.update_batch_table_row(table_row_idx, self.batch_results[row_idx])
        self._start_thumbnail_render(row_indices=[int(row_idx)])
        self._refresh_batch_analysis_if_run()
        self._autosave_fit_details()
        self.stats_text.append(
            f"Cleared previous fit result for {stem_for_file_ref(current_file)}."
        )

    def do_full_update(self) -> None:
        """Perform a complete update including stats."""
        self.update_plot(fast=False)

    def browse_directory(self) -> None:
        """Choose source mode and load from folder, archive, or selected CSV files."""
        start_dir: Path = self._source_dialog_start_dir()

        source_menu = QMenu(self)
        folder_action: QAction | None = source_menu.addAction("Load Folder...")
        csv_action: QAction | None = source_menu.addAction("Load CSV File(s)...")
        archive_action: QAction | None = source_menu.addAction("Load Archive...")

        if hasattr(self, "source_path_label"):
            anchor: QPoint = self.source_path_label.mapToGlobal(
                self.source_path_label.rect().bottomLeft()
            )
        else:
            anchor: QPoint = self.mapToGlobal(self.rect().center())
        selected_action: QAction | None = source_menu.exec(anchor)
        if selected_action is None:
            return
        if selected_action not in {folder_action, csv_action, archive_action}:
            return

        if selected_action == folder_action:
            selected_dir: str = QFileDialog.getExistingDirectory(
                self,
                "Select Data Folder",
                str(start_dir),
                options=QFileDialog.Option.ShowDirsOnly,
            )
            if not selected_dir:
                return
            self.current_dir = str(Path(selected_dir).expanduser())
            self._source_display_override = None
            self._source_selected_paths = []
            self._refresh_source_path_label()
            self.load_files()
            return

        if selected_action == archive_action:
            selected_archive, _ = QFileDialog.getOpenFileName(
                self,
                "Select Archive",
                str(start_dir),
                "Archive Files (*.zip *.tar.xz);;ZIP Archives (*.zip);;"
                "TAR.XZ Archives (*.tar.xz);;All Files (*.*)",
            )
            if not selected_archive:
                return
            chosen_archive: Path = Path(selected_archive).expanduser()
            if not chosen_archive.exists():
                self.stats_text.append(
                    f"Selected path does not exist: {chosen_archive}"
                )
                return
            self.current_dir = str(chosen_archive)
            self._source_display_override = None
            self._source_selected_paths = []
            self._refresh_source_path_label()
            self.load_files()
            return

        selected_csvs, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV File(s)",
            str(start_dir),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not selected_csvs:
            return
        self._load_selected_csv_files(selected_csvs)

    def auto_fit(self) -> None:
        """Start auto-fit in a worker thread to keep GUI responsive.

        Toggle behaviour: if a manual fit is already running for the current
        file, the first press cancels it.  A second press submits a new fit
        with priority, preempting whatever is running.
        """
        if self.current_data is None:
            self.stats_text.append("No data loaded!")
            return

        current_file: Any | None = self._current_loaded_file_path()
        if not current_file:
            self.stats_text.append("No current file loaded!")
            return

        active_tasks = self._active_fit_tasks_for_file(current_file)
        active_manual_tasks = [
            meta for meta in active_tasks if str(meta.get("kind")) == "manual"
        ]
        if active_manual_tasks:
            # Toggle: cancel existing manual fits for this file.
            self.cancel_auto_fit()
            return
        # Allow submitting even when a batch task is active on this file —
        # manual fits use preempt() so they jump the queue.

        run_mode: str = self._current_auto_fit_run_mode()
        procedure = None
        if run_mode == "procedure":
            procedure, procedure_error = self._current_procedure_for_run()
            if procedure_error:
                self.stats_text.append(procedure_error)
                return

        if run_mode == "fit":
            seed_overrides, mapping_error = (
                self._current_file_seed_overrides_from_mapping()
            )
            if mapping_error:
                self.stats_text.append(f"Fit setup error: {mapping_error}")
                return
        else:
            seed_overrides = {}

        try:
            # Seed from current UI controls so refits follow the slider state.
            fit_context = self._build_fit_context(
                seed_overrides=seed_overrides,
                fixed_params={},
                respect_enabled_channels=(run_mode != "procedure"),
            )
        except Exception as exc:
            self.stats_text.append(f"Fit setup error: {exc}")
            return

        capture_config: CapturePatternConfig | None = (
            self._resolve_batch_capture_config(show_errors=False)
        )
        if capture_config is None:
            capture_config: CapturePatternConfig = parse_capture_pattern("")

        source_index = int(getattr(self, "current_file_idx", 0))
        try:
            task_id = self._start_file_fit_task(
                kind="manual",
                file_path=current_file,
                source_index=source_index,
                fit_context=fit_context,
                capture_regex_pattern=capture_config.regex_pattern,
                capture_defaults=capture_config.defaults,
                parameter_capture_map=self._effective_param_capture_map_for_fixing(),
                execution_mode=run_mode,
                procedure=procedure,
                priority=True,
            )
        except Exception as exc:
            self.stats_text.append(f"Fit setup error: {exc}")
            return
        action_label: str = "Procedure fit" if run_mode == "procedure" else "Auto-fit"
        self.stats_text.append(
            f"{action_label} started for {stem_for_file_ref(current_file)} (full trace)."
        )
        if run_mode == "procedure":
            panel: Any | None = getattr(self, "_procedure_panel", None)
            if panel is not None and procedure is not None:
                try:
                    panel.record_external_procedure_start(
                        procedure_name=str(getattr(procedure, "name", "Procedure")),
                        file_label=stem_for_file_ref(current_file),
                        step_count=len(getattr(procedure, "steps", ()) or ()),
                    )
                except Exception:
                    pass
            if procedure is not None:
                total_steps = len(getattr(procedure, "steps", ()) or ())
                if total_steps > 0:
                    task = self.fit_tasks.get(int(task_id))
                    if task is not None:
                        task["_progress_started_at"] = float(time.perf_counter())
                        self._update_manual_procedure_status(
                            task,
                            step_done=0,
                            step_total=int(total_steps),
                        )
        self._refresh_fit_action_buttons()

    def on_fit_finished(self, fit_result) -> None:
        """Handle successful fit completion."""
        model_def: PiecewiseModelDefinition | None = self._piecewise_model
        if model_def is None:
            self.on_fit_failed("No compiled piecewise model.")
            return
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        ordered_keys: List[str] = list(
            multi_model.global_param_names
            if multi_model is not None
            else model_def.global_param_names
        )
        params_by_key = dict((fit_result or {}).get("params_by_key") or {})
        best_params: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
            [float(params_by_key.get(key, 0.0)) for key in ordered_keys], dtype=float
        )
        self.last_popt = best_params
        self._last_fit_active_keys: List[str] = list(ordered_keys)
        self._last_r2: float | None = (
            float(fit_result["r2"])
            if fit_result is not None and fit_result.get("r2") is not None
            else None
        )
        self._refresh_boundary_state_topology(preserve_existing=True)
        # Handle per-channel boundary ratios from multi-channel fit results.
        channel_results = (fit_result or {}).get("channel_results")
        if isinstance(channel_results, dict) and channel_results:
            source_targets = []
            for ch_target, ch_result in channel_results.items():
                ch_ratios = (
                    ch_result.get("boundary_ratios")
                    if isinstance(ch_result, Mapping)
                    else None
                )
                if ch_ratios is not None:
                    self._fit_state.set_channel_ratios(ch_target, ch_ratios)
                    source_targets.append(str(ch_target))
            for source_target in dict.fromkeys(source_targets):
                self._fit_state.apply_link_groups(
                    self._boundary_links_from_map(),
                    source_target=source_target,
                    prefer_targets=[source_target],
                )
            # Report per-channel R² for multi-channel fits.
            if multi_model is not None and multi_model.is_multi_channel:
                ch_r2_parts = []
                per_ch_r2 = {}
                for ch_target, ch_result in channel_results.items():
                    ch_r2 = (
                        ch_result.get("r2") if isinstance(ch_result, Mapping) else None
                    )
                    per_ch_r2[ch_target] = ch_r2
                    if ch_r2 is not None:
                        ch_r2_parts.append(f"{ch_target}: {ch_r2:.6f}")
                self._last_per_channel_r2 = per_ch_r2
                if ch_r2_parts:
                    self.stats_text.append(f"Per-channel R²: {', '.join(ch_r2_parts)}")
        self._sync_breakpoint_sliders_from_state()

        # Block spinbox signals while setting values to prevent intermediate
        # update_plot calls that would fire before all params are applied.
        blocked_spinboxes = []
        updated_param_keys: List[str] = []
        for idx, key in enumerate(ordered_keys):
            if key in self.param_spinboxes and idx < len(self.last_popt):
                spinbox = self.param_spinboxes[key]
                was_blocked = spinbox.blockSignals(True)
                spinbox.setValue(self.last_popt[idx])
                updated_param_keys.append(str(key))
                if not was_blocked:
                    blocked_spinboxes.append(spinbox)
        for spinbox in blocked_spinboxes:
            spinbox.blockSignals(False)
        for key in updated_param_keys:
            self._sync_slider_from_spinbox(key)
        self.defaults: List[Any] = list(self.last_popt)

        r2_text: str = f"{self._last_r2:.6f}" if self._last_r2 is not None else "N/A"
        violations = self._fit_param_range_violations(best_params)
        range_error: None | str = self._fit_param_range_error_text(violations)
        if range_error:
            self.stats_text.append(f"✗ Auto-fit failed: {range_error}")
        else:
            self.stats_text.append(
                f"✓ Auto-fit successful! R² (full trace) = {r2_text}"
            )
        summary: str = ", ".join(
            f"{self._display_name_for_param_key(key)}={self.last_popt[idx]:.4f}"
            for idx, key in enumerate(ordered_keys)
            if idx < len(self.last_popt)
        )
        self.stats_text.append(summary)

        current_file: Any | None = self._current_loaded_file_path()
        updated_batch_row: bool = (
            self._upsert_batch_row_from_fit(
                current_file,
                ordered_keys,
                fit_result,
            )
            is not None
        )

        self.update_plot()
        if updated_batch_row:
            self._refresh_batch_analysis_if_run()
        self._autosave_fit_details()
        self._refresh_fit_action_buttons()

    def on_fit_failed(self, error_text) -> None:
        """Handle fit failures."""
        self.stats_text.append(f"✗ Auto-fit failed: {error_text}")
        self._refresh_fit_action_buttons()

    def cancel_auto_fit(self) -> None:
        """Request cancellation of an in-flight auto-fit."""
        current_file: Any | None = self._current_loaded_file_path()
        if not current_file:
            return
        current_key: str = self._fit_task_file_key(current_file)
        manual_task_ids: set[int] = set()
        for task_id, task in list(self.fit_tasks.items()):
            if str(task.get("kind")) != "manual":
                continue
            task_key = str(
                task.get("file_key") or self._fit_task_file_key(task.get("file_path"))
            )
            if task_key != current_key:
                continue
            manual_task_ids.add(int(task_id))
        if manual_task_ids:
            self._fit_worker_thread.cancel_tasks(manual_task_ids)
            self.stats_text.append("Fit cancellation requested...")

    @staticmethod
    def _request_worker_cancel(worker) -> None:
        if worker is None:
            return
        request_cancel: Any | None = getattr(worker, "request_cancel", None)
        if callable(request_cancel):
            try:
                request_cancel()
            except Exception:
                pass

    @staticmethod
    def _shutdown_thread(thread, wait_ms=None, force_terminate=False) -> bool:
        if thread is None:
            return True
        try:
            thread.requestInterruption()
        except Exception:
            pass
        thread.quit()
        stopped = True
        if wait_ms is None:
            thread.wait()
        else:
            stopped = bool(thread.wait(int(wait_ms)))
            if (not stopped) and force_terminate:
                try:
                    thread.terminate()
                except Exception:
                    pass
                stopped = bool(thread.wait(int(max(100, wait_ms))))
        if stopped:
            thread.deleteLater()
        return stopped

    def _get_channel_data(self, channel_name):
        if channel_name in self.channel_cache:
            return self.channel_cache[channel_name]
        if self.current_data is None:
            raise ValueError("No data loaded.")
        if channel_name not in self.current_data.columns:
            raise KeyError(f"Channel '{channel_name}' not found in data.")
        values = self.current_data[channel_name].to_numpy(dtype=float, copy=True)
        self.raw_channel_cache[channel_name] = values
        smoothed: (
            np.ndarray[Tuple[int, ...], np.dtype[Any]]
            | np.ndarray[Tuple[int], np.dtype[Any]]
        ) = self._smooth_channel_values(values)
        self.channel_cache[channel_name] = smoothed
        return smoothed

    def _display_indices(
        self, n_points
    ) -> (
        np.ndarray[Tuple[int, ...], np.dtype[Any]]
        | np.ndarray[Tuple[int], np.dtype[Any]]
    ):
        if n_points <= 0:
            return np.asarray([], dtype=int)
        target: int = max(1000, int(self._display_target_points))
        stride: int = max(1, int(np.ceil(n_points / float(target))))
        return np.arange(0, n_points, stride, dtype=int)

    def _ensure_plot_axes(self, show_residuals) -> None:
        if show_residuals == self._plot_has_residual_axis and hasattr(self, "ax"):
            return
        self.fig.clear()
        if show_residuals:
            grid: GridSpec = self.fig.add_gridspec(
                2, 1, height_ratios=[3, 1], hspace=0.05
            )
            self.ax: Axes = self.fig.add_subplot(grid[0])
            self.ax_residual: Axes = self.fig.add_subplot(grid[1], sharex=self.ax)
            self.ax.tick_params(labelbottom=False)
        else:
            self.ax: Axes = self.fig.add_subplot(111)
            self.ax_residual = None
        self._plot_has_residual_axis = bool(show_residuals)

    def _finite_min_max(self, *arrays) -> Tuple[float, float]:
        y_min = None
        y_max = None
        for arr in arrays:
            if arr is None:
                continue
            values: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                arr, dtype=float
            )
            finite: np.ndarray[Tuple[int, ...], np.dtype[Any]] = values[
                np.isfinite(values)
            ]
            if finite.size == 0:
                continue
            cur_min = float(np.min(finite))
            cur_max = float(np.max(finite))
            y_min: float = cur_min if y_min is None else min(y_min, cur_min)
            y_max: float = cur_max if y_max is None else max(y_max, cur_max)

        if y_min is None or y_max is None:
            return (-1.0, 1.0)
        if np.isclose(y_min, y_max):
            pad: float = 1.0 if np.isclose(y_min, 0.0) else max(1e-6, abs(y_min) * 0.05)
            return (y_min - pad, y_max + pad)

        pad: float = (y_max - y_min) * 0.05
        if pad <= 0.0:
            pad = 1.0
        return (y_min - pad, y_max + pad)

    def _apply_unique_legend(self, axis, loc="lower right") -> None:
        handles, labels = axis.get_legend_handles_labels()
        unique_handles = []
        unique_labels = []
        seen = set()
        for handle, label in zip(handles, labels):
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        if unique_handles:
            axis.legend(unique_handles, unique_labels, loc=loc)

    def _boundary_marker_entries(self, x_values: Sequence[float]) -> List[Dict[str, Any]]:
        """Return boundary marker line entries for plotting."""
        x_arr = np.asarray(x_values, dtype=float).reshape(-1)
        x_finite = x_arr[np.isfinite(x_arr)]
        if x_finite.size == 0:
            return []

        entries: List[Dict[str, Any]] = []
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)

        if multi_model is not None and multi_model.is_multi_channel:
            for ch_model in multi_model.channel_models:
                target = str(ch_model.target_col).strip()
                if not target or not self._is_channel_visible(target):
                    continue
                n_boundaries = max(0, len(ch_model.segment_exprs) - 1)
                if n_boundaries <= 0:
                    continue
                ratios = np.asarray(
                    self._fit_state.channel_ratios(target), dtype=float
                ).reshape(-1)
                if ratios.size != n_boundaries:
                    ratios = default_boundary_ratios(n_boundaries)
                x_boundaries = boundary_ratios_to_x_values(
                    ratios,
                    x_finite,
                    n_boundaries,
                )
                for bidx, bval in enumerate(np.asarray(x_boundaries, dtype=float).reshape(-1)):
                    if not np.isfinite(bval):
                        continue
                    entries.append(
                        {
                            "x": float(bval),
                            "color": str(self._fit_companion_color(target)),
                        }
                    )
        else:
            target = str(
                getattr(self._fit_state, "primary_target", "")
                or self._primary_target_channel()
            )
            n_boundaries = int(self._fit_state.channel_count(target))
            if n_boundaries > 0:
                ratios = np.asarray(self._fit_state.channel_ratios(target), dtype=float).reshape(
                    -1
                )
                if ratios.size != n_boundaries:
                    ratios = default_boundary_ratios(n_boundaries)
                x_boundaries = boundary_ratios_to_x_values(
                    ratios,
                    x_finite,
                    n_boundaries,
                )
                for bidx, bval in enumerate(np.asarray(x_boundaries, dtype=float).reshape(-1)):
                    if not np.isfinite(bval):
                        continue
                    entries.append(
                        {
                            "x": float(bval),
                            "color": "#475569",
                        }
                    )
        return entries

    def _draw_boundary_markers_on_axis(
        self,
        axis: Axes,
        entries: Sequence[Mapping[str, Any]],
    ) -> List[Any]:
        """Draw boundary markers and return artists for later updates/removal."""
        artists: List[Any] = []
        if not entries:
            return artists
        for entry in entries:
            x_val = finite_float_or_none(entry.get("x"))
            if x_val is None:
                continue
            color = str(entry.get("color") or "#64748b")
            line = axis.axvline(
                float(x_val),
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                zorder=3.5,
            )
            artists.append(line)
        return artists

    def _refresh_boundary_markers(self, axis: Axes, x_values: Sequence[float]) -> None:
        """Recompute and redraw boundary markers from current boundary ratios."""
        old_artists = self._plot_lines.get("boundary_artists")
        if isinstance(old_artists, list):
            for artist in old_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
        entries = self._boundary_marker_entries(x_values)
        self._plot_lines["boundary_artists"] = self._draw_boundary_markers_on_axis(
            axis, entries
        )

    def _prepare_plot_context(self, params):
        x_data = self._get_channel_data(self.x_channel)
        primary_target: str = self._primary_target_channel()
        y_data = self._get_channel_data(primary_target)
        n_points: int = len(x_data)
        if n_points == 0:
            return None

        channel_data_full = self._expression_channel_data()
        display_idx: (
            np.ndarray[Tuple[int, ...], np.dtype[Any]]
            | np.ndarray[Tuple[int], np.dtype[Any]]
        ) = self._display_indices(n_points)
        if display_idx.size == 0:
            return None

        x_display = x_data[display_idx]
        y_display = y_data[display_idx]
        channel_data_display = self._slice_channel_data(channel_data_full, display_idx)

        # Determine which channel targets are being fitted.
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        fit_target_channels = [primary_target]
        if multi_model is not None and multi_model.is_multi_channel:
            fit_target_channels: List[Any] = list(multi_model.target_channels)

        plot_channel_names = []
        # Prioritize fitted target channels.
        for name in fit_target_channels:
            key: str = str(name).strip()
            if not key or key in plot_channel_names:
                continue
            if key in channel_data_display:
                plot_channel_names.append(key)

        if self.current_data is not None:
            for key in self._numeric_channel_columns():
                if not key or key == self.x_channel or key in plot_channel_names:
                    continue
                if key in channel_data_display:
                    plot_channel_names.append(key)
        else:
            for key in channel_data_display.keys():
                key_text: str = str(key).strip()
                if (
                    not key_text
                    or key_text == self.x_channel
                    or key_text in plot_channel_names
                ):
                    continue
                plot_channel_names.append(key_text)

        plot_channel_displays = {
            key: np.asarray(channel_data_display[key], dtype=float)
            for key in plot_channel_names
            if key in channel_data_display
        }

        return {
            "params": params,
            "x_data": x_data,
            "y_data": y_data,
            "n_points": n_points,
            "channel_data_full": channel_data_full,
            "display_idx": display_idx,
            "x_display": x_display,
            "y_display": y_display,
            "channel_data_display": channel_data_display,
            "plot_channel_displays": plot_channel_displays,
            "fit_target_channels": fit_target_channels,
        }

    def _compute_display_series(self, context):
        fitted_display: np.ndarray[Tuple[int, ...], np.dtype[Any]] = (
            self.evaluate_model(
                context["x_display"],
                context["params"],
                channel_data=context["channel_data_display"],
            )
        )
        residuals_display = context["y_display"] - fitted_display

        # Per-channel fitted curves for multi-channel models.
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        channel_fitted = {}
        channel_residuals = {}
        if multi_model is not None and multi_model.is_multi_channel:
            param_map = (
                context["params"]
                if isinstance(context["params"], dict)
                else {
                    key: float(context["params"][idx])
                    for idx, key in enumerate(self._ordered_param_keys())
                    if idx < len(context["params"])
                }
            )
            for ch_model in multi_model.channel_models:
                target = ch_model.target_col
                try:
                    ch_fitted: np.ndarray[Tuple[int, ...], np.dtype[Any]] = (
                        self.evaluate_channel_model(
                            ch_model, context["x_display"], param_map
                        )
                    )
                    channel_fitted[target] = ch_fitted
                    ch_data = context["channel_data_display"].get(target)
                    if ch_data is not None:
                        channel_residuals[target] = (
                            np.asarray(ch_data, dtype=float) - ch_fitted
                        )
                except Exception:
                    pass

        return {
            "fitted_display": fitted_display,
            "residuals_display": residuals_display,
            "channel_fitted": channel_fitted,
            "channel_residuals": channel_residuals,
        }

    def _try_fast_plot_update(self, context, series, show_residuals) -> bool:
        if not hasattr(self, "_plot_lines"):
            return False
        if show_residuals != self._plot_has_residual_axis:
            return False

        fitted_line = self._plot_lines.get("fitted")
        if fitted_line is not None:
            fitted_line.set_ydata(series["fitted_display"])
        residuals_line = self._plot_lines.get("residuals")
        if residuals_line is not None and show_residuals:
            residuals_line.set_ydata(series["residuals_display"])

        # Update per-channel fit curves for multi-channel.
        channel_fitted = series.get("channel_fitted", {})
        for ch_target, ch_fitted_data in channel_fitted.items():
            line_key: str = f"fitted_{ch_target}"
            ch_line = self._plot_lines.get(line_key)
            if ch_line is not None:
                ch_line.set_ydata(ch_fitted_data)

        # Update per-channel residuals.
        if show_residuals:
            channel_residuals = series.get("channel_residuals", {})
            for ch_target, ch_resid in channel_residuals.items():
                res_key: str = f"residuals_{ch_target}"
                ch_res_line = self._plot_lines.get(res_key)
                if ch_res_line is not None:
                    ch_res_line.set_ydata(ch_resid)

        # Compute y-limits including only visible channels/fit curves.
        channel_arrays = [
            v
            for k, v in context.get("plot_channel_displays", {}).items()
            if self._is_channel_visible(k)
        ]
        if channel_fitted:
            fit_arrays = [
                v for k, v in channel_fitted.items() if self._is_channel_visible(k)
            ]
        else:
            fit_arrays = [series["fitted_display"]]
        y_min, y_max = self._finite_min_max(*channel_arrays, *fit_arrays)
        self.ax.set_ylim(y_min, y_max)
        x_vals: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
            context["x_display"], dtype=float
        )
        x_finite: np.ndarray[Tuple[int, ...], np.dtype[Any]] = x_vals[
            np.isfinite(x_vals)
        ]
        if x_finite.size > 0:
            x_min = float(np.min(x_finite))
            x_max = float(np.max(x_finite))
            if np.isclose(x_min, x_max):
                pad: float = (
                    1.0 if np.isclose(x_min, 0.0) else max(1e-6, abs(x_min) * 0.05)
                )
                x_min -= pad
                x_max += pad
            self.ax.set_xlim(x_min, x_max)
        if show_residuals and self.ax_residual is not None:
            resid_arrays = [
                v
                for k, v in series.get("channel_residuals", {}).items()
                if self._is_channel_visible(k)
            ]
            if not resid_arrays:
                resid_arrays = [series["residuals_display"]]
            r_min, r_max = self._finite_min_max(*resid_arrays)
            self.ax_residual.set_ylim(r_min, r_max)
        self._refresh_boundary_markers(self.ax, context["x_data"])
        self.canvas.draw_idle()
        return True

    def _update_stats_panel(self, r2_value, per_channel_r2=None) -> None:
        """Update the R² display in the tab corner and per-equation labels."""
        if per_channel_r2 is None:
            per_channel_r2: Any | Dict[Any, Any] = getattr(
                self, "_last_per_channel_r2", {}
            )

        # Tab corner: show per-channel R² summary for multi-channel.
        multi_model: Any | None = getattr(self, "_multi_channel_model", None)
        if multi_model is not None and multi_model.is_multi_channel and per_channel_r2:
            parts = []
            for target in multi_model.target_channels:
                ch_r2 = per_channel_r2.get(target)
                if ch_r2 is not None:
                    parts.append(f"{self._channel_display_name(target)}: {ch_r2:.4f}")
                else:
                    parts.append(f"{self._channel_display_name(target)}: N/A")
            tab_text: str = "R² " + " | ".join(parts)
        else:
            tab_text: str = f"R²: {r2_value:.6f}" if r2_value is not None else "R²: N/A"
        if hasattr(self, "tab_r2_label"):
            self.tab_r2_label.setText(tab_text)
            # Ensure the R² label is always visible on the plot control bar (manual tab)
            if hasattr(self, "tabs") and hasattr(self, "manual_tab"):
                self.tab_r2_label.setVisible(
                    self.tabs.currentWidget() is self.manual_tab
                )

        # Update per-equation R² labels (if any exist).
        labels: Any | Dict[Any, Any] = getattr(self, "_equation_r2_labels", {})
        for target, label in labels.items():
            ch_r2 = per_channel_r2.get(target)
            if ch_r2 is not None:
                label.setText(f"R²: {ch_r2:.6f}")
                label.setStyleSheet(
                    "color: #334155; font-size: 11px; font-weight: 600; padding: 0px 2px;"
                )
            else:
                label.setText("R²: N/A")
                label.setStyleSheet(
                    "color: #64748b; font-size: 11px; padding: 0px 2px;"
                )
        self._sync_param_row_tail_spacers()

    def update_plot(self, fast=False, preserve_view=True):
        """Update plot with current parameters.

        Args:
            fast: If True, skip expensive operations for smooth slider interaction
            preserve_view: If True, keep current zoom/pan limits when possible
        """
        if self.current_data is None:
            self._refresh_param_value_error_highlighting()
            return

        # Debounce full updates during slider movement
        if fast and not self.slider_active:
            # Use timer to batch rapid updates
            self.update_timer.stop()
            self.update_timer.start(50)  # 50ms debounce
            return

        try:
            self._refresh_param_value_error_highlighting()
            params = self.get_current_params()
            context = self._prepare_plot_context(params)
            if context is None:
                return
            series = self._compute_display_series(context)

            show_residuals: bool = self.show_residuals_cb.isChecked()
            can_preserve_view = bool(
                preserve_view
                and hasattr(self, "ax")
                and self.ax is not None
                and show_residuals == self._plot_has_residual_axis
            )
            main_xlim = None
            main_ylim = None
            residual_ylim = None
            if can_preserve_view:
                try:
                    main_xlim: Tuple[float, ...] = tuple(self.ax.get_xlim())
                except Exception:
                    main_xlim = None
                try:
                    main_ylim: Tuple[float, ...] = tuple(self.ax.get_ylim())
                except Exception:
                    main_ylim = None
                if show_residuals and self.ax_residual is not None:
                    try:
                        residual_ylim: Tuple[float, ...] = tuple(
                            self.ax_residual.get_ylim()
                        )
                    except Exception:
                        residual_ylim = None

            if fast and self._try_fast_plot_update(context, series, show_residuals):
                if can_preserve_view:
                    try:
                        if main_xlim is not None:
                            self.ax.set_xlim(*main_xlim)
                        if main_ylim is not None:
                            self.ax.set_ylim(*main_ylim)
                        if (
                            show_residuals
                            and self.ax_residual is not None
                            and residual_ylim is not None
                        ):
                            self.ax_residual.set_ylim(*residual_ylim)
                        self.canvas.draw_idle()
                    except Exception:
                        pass
                return

            self._ensure_plot_axes(show_residuals)
            self.ax.clear()
            if self.ax_residual is not None:
                self.ax_residual.clear()

            primary_target: str = self._primary_target_channel()
            self._plot_lines = {}
            fit_target_channels = context.get(
                "fit_target_channels",
                [primary_target] if primary_target else [],
            )
            for idx, (channel_name, values) in enumerate(
                context.get("plot_channel_displays", {}).items()
            ):
                if not self._is_channel_visible(channel_name):
                    continue
                color: str = self._channel_plot_color(channel_name)
                channel_label: str = self._channel_legend_label(channel_name)
                is_target: bool = channel_name in fit_target_channels
                self.ax.plot(
                    context["x_display"],
                    values,
                    label=channel_label,
                    color=color,
                    linewidth=1.5 if is_target else 1.2,
                    alpha=1.0 if is_target else 0.9,
                )

            # Plot fit curves using a companion colour that is similar but
            # visually distinct from the data-channel colour.
            channel_fitted = series.get("channel_fitted", {})
            if channel_fitted and len(channel_fitted) > 1:
                for fit_idx, (ch_target, ch_fitted_data) in enumerate(
                    channel_fitted.items()
                ):
                    if not self._is_channel_visible(ch_target):
                        continue
                    fit_color: str = self._fit_companion_color(ch_target)
                    fit_label: str = f"Fit ({ch_target})"
                    (fitted_line,) = self.ax.plot(
                        context["x_display"],
                        ch_fitted_data,
                        label=fit_label,
                        color=fit_color,
                        linewidth=1.4,
                    )
                    self._plot_lines[f"fitted_{ch_target}"] = fitted_line
                # Expose the primary fit line under the standard key.
                if primary_target in channel_fitted:
                    primary_fit_line = self._plot_lines.get(f"fitted_{primary_target}")
                    if primary_fit_line is not None:
                        self._plot_lines["fitted"] = primary_fit_line
            else:
                # Single-channel: use companion colour.
                fit_color: str = self._fit_companion_color(primary_target)
                (fitted_line,) = self.ax.plot(
                    context["x_display"],
                    series["fitted_display"],
                    label="Fitted",
                    color=fit_color,
                    linewidth=1.4,
                )
                self._plot_lines["fitted"] = fitted_line

            if show_residuals and self.ax_residual is not None:
                # Plot residuals for all fitted channels.
                channel_residuals = series.get("channel_residuals", {})
                if channel_residuals and len(channel_residuals) > 1:
                    for res_idx, (ch_target, ch_resid) in enumerate(
                        channel_residuals.items()
                    ):
                        if not self._is_channel_visible(ch_target):
                            continue
                        res_color: str = self._fit_companion_color(ch_target)
                        (residuals_line,) = self.ax_residual.plot(
                            context["x_display"],
                            ch_resid,
                            label=f"Residuals ({ch_target})",
                            color=res_color,
                            linewidth=1.0,
                        )
                        self._plot_lines[f"residuals_{ch_target}"] = residuals_line
                else:
                    (residuals_line,) = self.ax_residual.plot(
                        context["x_display"],
                        series["residuals_display"],
                        label="Residuals",
                        color="black",
                        linewidth=1.0,
                    )
                    self._plot_lines["residuals"] = residuals_line
                self.ax_residual.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.6)

            if not fast:
                per_channel_r2 = {}
                multi_model: Any | None = getattr(self, "_multi_channel_model", None)
                if multi_model is not None and multi_model.is_multi_channel:
                    param_map: dict[str, float] = (
                        params
                        if isinstance(params, dict)
                        else {
                            key: float(params[idx])
                            for idx, key in enumerate(self._ordered_param_keys())
                            if idx < len(params)
                        }
                    )
                    for ch_model in multi_model.channel_models:
                        target = ch_model.target_col
                        try:
                            ch_fitted_full: np.ndarray[
                                Tuple[int, ...], np.dtype[Any]
                            ] = self.evaluate_channel_model(
                                ch_model, context["x_data"], param_map
                            )
                            ch_actual = context["channel_data_full"].get(target)
                            if ch_actual is not None:
                                ch_r2: None | float = compute_r2(
                                    np.asarray(ch_actual, dtype=float), ch_fitted_full
                                )
                                per_channel_r2[target] = ch_r2
                        except Exception:
                            per_channel_r2[target] = None
                    valid_channel_r2 = [
                        float(value)
                        for value in per_channel_r2.values()
                        if value is not None
                    ]
                    r2_value: None | float = (
                        float(np.mean(valid_channel_r2)) if valid_channel_r2 else None
                    )
                else:
                    fitted_full: np.ndarray[Tuple[int, ...], np.dtype[Any]] = (
                        self.evaluate_model(
                            context["x_data"],
                            params,
                            channel_data=context["channel_data_full"],
                        )
                    )
                    r2_value: None | float = compute_r2(context["y_data"], fitted_full)
                    if primary_target:
                        per_channel_r2[primary_target] = r2_value
                self._last_r2: None | float = r2_value
                self._last_per_channel_r2 = per_channel_r2
            else:
                r2_value: None | float = self._last_r2
                per_channel_r2: Any | Dict[Any, Any] = getattr(
                    self, "_last_per_channel_r2", {}
                )

            # Only include visible channels for y-limits.
            channel_arrays = [
                v
                for k, v in context.get("plot_channel_displays", {}).items()
                if self._is_channel_visible(k)
            ]
            # Include only visible fit curve arrays in the y-limits.
            if channel_fitted:
                fit_arrays = [
                    v for k, v in channel_fitted.items() if self._is_channel_visible(k)
                ]
            else:
                fit_arrays = [series["fitted_display"]]
            y_min, y_max = self._finite_min_max(*channel_arrays, *fit_arrays)
            self.ax.set_ylim(y_min, y_max)
            self._apply_unique_legend(self.ax, loc="lower right")
            self.ax.set_xlabel(
                "" if show_residuals else self._channel_axis_label(self.x_channel)
            )
            # Multi-channel: show generic y-label.
            multi_model: Any | None = getattr(self, "_multi_channel_model", None)
            if multi_model is not None and multi_model.is_multi_channel:
                ch_labels: List[str] = [
                    self._channel_axis_label(t) for t in multi_model.target_channels
                ]
                self.ax.set_ylabel(" / ".join(ch_labels))
            else:
                self.ax.set_ylabel(
                    self._channel_axis_label(primary_target) if primary_target else "Signal"
                )
            x_vals: np.ndarray[Tuple[int, ...], np.dtype[Any]] = np.asarray(
                context["x_display"], dtype=float
            )
            x_finite: np.ndarray[Tuple[int, ...], np.dtype[Any]] = x_vals[
                np.isfinite(x_vals)
            ]
            if x_finite.size > 0:
                x_min = float(np.min(x_finite))
                x_max = float(np.max(x_finite))
                if np.isclose(x_min, x_max):
                    pad: float = (
                        1.0 if np.isclose(x_min, 0.0) else max(1e-6, abs(x_min) * 0.05)
                    )
                    x_min -= pad
                    x_max += pad
                self.ax.set_xlim(x_min, x_max)
            self.ax.grid(True, alpha=0.3)
            self._refresh_boundary_markers(self.ax, context["x_data"])
            if show_residuals and self.ax_residual is not None:
                resid_arrays = [
                    v
                    for k, v in series.get("channel_residuals", {}).items()
                    if self._is_channel_visible(k)
                ]
                if not resid_arrays:
                    resid_arrays = [series["residuals_display"]]
                r_min, r_max = self._finite_min_max(*resid_arrays)
                self.ax_residual.set_ylim(r_min, r_max)
                self.ax_residual.set_ylabel("Residual")
                self.ax_residual.set_xlabel(self._channel_axis_label(self.x_channel))
                self.ax_residual.grid(True, alpha=0.25)
                # Residual legend intentionally omitted to reduce clutter.

            home_main_xlim = None
            home_main_ylim = None
            home_residual_ylim = None
            try:
                home_main_xlim: Tuple[float, ...] = tuple(self.ax.get_xlim())
            except Exception:
                home_main_xlim = None
            try:
                home_main_ylim: Tuple[float, ...] = tuple(self.ax.get_ylim())
            except Exception:
                home_main_ylim = None
            if show_residuals and self.ax_residual is not None:
                try:
                    home_residual_ylim: Tuple[float, ...] = tuple(
                        self.ax_residual.get_ylim()
                    )
                except Exception:
                    home_residual_ylim = None

            if can_preserve_view:
                if main_xlim is not None:
                    self.ax.set_xlim(*main_xlim)
                if main_ylim is not None:
                    self.ax.set_ylim(*main_ylim)
                if (
                    show_residuals
                    and self.ax_residual is not None
                    and residual_ylim is not None
                ):
                    self.ax_residual.set_ylim(*residual_ylim)

            self._set_toolbar_home_limits(
                home_main_xlim,
                home_main_ylim,
                home_residual_ylim,
                keep_current_view=can_preserve_view,
            )

            self.canvas.draw_idle()
            self._update_stats_panel(r2_value, per_channel_r2)
        except Exception as e:
            self.stats_text.setText(f"Error updating stats: {e}")
            print(f"Error updating stats: {type(e).__name__}: {e}", file=sys.stderr)

    def _apply_fit_row_update(self, row):
        if has_nonempty_values(fit_get(row, "params")):
            row["_equation_stale"] = False
            row["_fit_conditions"] = self._fit_conditions_fingerprint()
        row = self._apply_param_range_validation_to_row(row)
        row_file: str = str(row.get("file") or "").strip()
        row_file_key: str = self._fit_task_file_key(row_file)
        row_index = row.get("_source_index")
        try:
            row_index: int | None = int(row_index) if row_index is not None else None
        except Exception:
            row_index = None
        if (
            row_index is not None
            and 0 <= row_index < len(self.batch_results)
            and row_file_key
        ):
            existing_file: str = str(
                self.batch_results[row_index].get("file") or ""
            ).strip()
            existing_file_key: str = self._fit_task_file_key(existing_file)
            if existing_file_key != row_file_key:
                by_file: None | int = self._find_batch_result_index_by_file(row_file)
                fit_debug(
                    "fit-row-index-mismatch: "
                    f"row_file={row_file} "
                    f"row_index={row_index} "
                    f"existing_file={existing_file or '-'} "
                    f"fallback_index={by_file if by_file is not None else -1}"
                )
                row_index: int | None = by_file
        if row_index is None:
            row_index = self._find_batch_result_index_by_file(row.get("file"))
        if row_index is None or row_index < 0 or row_index >= len(self.batch_results):
            self.batch_results.append(canonicalize_fit_row(row))
            self._rebuild_batch_capture_keys_from_rows()
            self.update_batch_table()
            row_index: int = len(self.batch_results) - 1
        else:
            existing = self.batch_results[row_index]
            for runtime_key in (
                "_fit_status",
                "_queue_position",
                "_r2_old",
                "_fit_task_id",
            ):
                if runtime_key not in row:
                    row[runtime_key] = existing.get(runtime_key)
            row_has_fit: bool = has_nonempty_values(fit_get(row, "params"))
            if (not row_has_fit) and ("_equation_stale" not in row):
                row["_equation_stale"] = bool(existing.get("_equation_stale"))
            existing_plot_has_fit = existing.get("plot_has_fit")
            if existing_plot_has_fit is None:
                existing_plot_has_fit: bool = has_nonempty_values(
                    fit_get(existing, "params")
                )
            if (
                (not row_has_fit)
                and (not existing_plot_has_fit)
                and existing.get("plot_full") is not None
                and row.get("plot_full") is None
            ):
                row["plot_full"] = existing["plot_full"]
                row["plot_has_fit"] = False
                row["plot_render_size"] = existing.get("plot_render_size")
            elif (
                (not row_has_fit)
                and (not existing_plot_has_fit)
                and existing.get("plot") is not None
                and row.get("plot") is None
            ):
                row["plot"] = existing["plot"]
                row["plot_has_fit"] = False
                row["plot_render_size"] = existing.get("plot_render_size")
            self.batch_results[row_index] = canonicalize_fit_row(row)
            table_row_idx: None | int = self._find_table_row_by_file(row.get("file"))
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)
            else:
                self.update_batch_table()

        row_has_fit: bool = has_nonempty_values(fit_get(row, "params"))
        if row_has_fit and row.get("plot_full") is None:
            self._start_thumbnail_render(row_indices=[int(row_index)])

        current_file: Any | None = self._current_loaded_file_path()
        current_file_key: str = self._fit_task_file_key(current_file)
        stored_row = (
            self.batch_results[row_index]
            if 0 <= int(row_index) < len(self.batch_results)
            else row
        )
        stored_file_key: str = self._fit_task_file_key(stored_row.get("file"))
        stored_has_fit: bool = has_nonempty_values(fit_get(stored_row, "params"))
        if (
            current_file_key
            and stored_file_key
            and current_file_key == stored_file_key
            and stored_has_fit
        ):
            self._apply_batch_params_for_file(current_file)
            self.update_plot(fast=False)
        self._refresh_batch_analysis_if_run()

    def _set_batch_row_runtime_fields(self, file_path, **updates) -> bool:
        row_idx: None | int = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            return False
        row = canonicalize_fit_row(self.batch_results[row_idx])
        changed = False
        for key, value in updates.items():
            if row.get(key) == value:
                continue
            row[key] = value
            changed = True
        if not changed:
            return False
        self.batch_results[row_idx] = canonicalize_fit_row(row)
        table_row_idx: None | int = self._find_table_row_by_file(file_path)
        if table_row_idx is None:
            self.update_batch_table()
        else:
            self.update_batch_table_row(table_row_idx, row)
        return True

    def _upsert_fit_error_row(self, file_path, source_index, error_text) -> None:
        row_idx: None | int = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            captures = {}
            pattern_error = None
            capture_config: CapturePatternConfig | None = (
                self._resolve_batch_capture_config(show_errors=False)
            )
            if capture_config is not None:
                extracted: Dict[str, str] | None = extract_captures(
                    stem_for_file_ref(file_path),
                    capture_config.regex,
                    capture_config.defaults,
                )
                if extracted is None:
                    pattern_error: str = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures: Dict[str, str] = dict(extracted)
            row = make_batch_result_row(
                source_index=source_index,
                file_path=file_path,
                x_channel=self.x_channel,
                captures=captures,
                pattern_error=pattern_error,
            )
            row_idx: int = len(self.batch_results)
            self.batch_results.append(row)
        row = canonicalize_fit_row(self.batch_results[row_idx])
        if not has_nonempty_values(fit_get(row, "params")):
            fit_set(row, "error", str(error_text))
        self.batch_results[row_idx] = canonicalize_fit_row(row)
        table_row_idx: None | int = self._find_table_row_by_file(file_path)
        if table_row_idx is None:
            self._rebuild_batch_capture_keys_from_rows()
            self.update_batch_table()
        else:
            self.update_batch_table_row(table_row_idx, row)
        self._refresh_batch_analysis_if_run()

    def _start_file_fit_task(
        self,
        *,
        kind,
        file_path,
        source_index,
        fit_context,
        capture_regex_pattern,
        capture_defaults,
        parameter_capture_map,
        queue_position=None,
        execution_mode="fit",
        procedure=None,
        batch_existing_rows_by_file=None,
        batch_fit_conditions_fingerprint=None,
        batch_procedure_capture_map=None,
        batch_filtered_capture_map=None,
        priority=False,
    ):
        # Preload reads can contend with fit worker data loads (especially for
        # archive-backed sources). Stop preload before starting any fit task.
        if getattr(self, "_data_preload_thread", None) is not None:
            self._stop_background_data_preload(wait_ms=120)

        run_mode: str = self._normalize_fit_run_mode(execution_mode)
        single_proc_sibling_context_requested = (
            str(kind) != "batch"
            and run_mode == "procedure"
            and bool(getattr(procedure, "seed_from_siblings", False))
        )
        self._apply_fit_compute_mode_env(self._current_fit_compute_mode())
        task_id: int = self._next_fit_task_id()
        existing_idx: None | int = self._find_batch_result_index_by_file(file_path)
        existing_row = (
            canonicalize_fit_row(self.batch_results[existing_idx])
            if existing_idx is not None
            else {}
        )
        if str(kind) == "batch":
            current_file: Any | None = self._current_loaded_file_path()
            if current_file and str(file_path) == str(current_file):
                overridden = canonicalize_fit_row(existing_row)
                ordered_keys: List[np.Never] = list(
                    fit_context.get("ordered_keys") or ()
                )
                seed_map = dict(fit_context.get("seed_map") or {})
                if ordered_keys and seed_map:
                    try:
                        fit_set(
                            overridden,
                            "params",
                            [
                                float(seed_map[key])
                                for key in ordered_keys
                                if key in seed_map
                            ],
                        )
                    except Exception:
                        pass
                seed_boundaries_by_channel = dict(
                    fit_context.get("boundary_seeds_per_channel") or {}
                )
                if seed_boundaries_by_channel:
                    seeded_channel_results: Dict[str, Dict[str, Any]] = {}
                    for raw_target, raw_ratios in seed_boundaries_by_channel.items():
                        target_key: str = str(raw_target).strip()
                        if not target_key:
                            continue
                        ratios_arr = self._as_float_array(raw_ratios)
                        if ratios_arr.size <= 0:
                            continue
                        seeded_channel_results[target_key] = {
                            "boundary_ratios": np.clip(ratios_arr, 0.0, 1.0)
                        }
                    fit_set(
                        overridden,
                        "channel_results",
                        seeded_channel_results if seeded_channel_results else None,
                    )
                existing_row = overridden
        needs_fit_condition_invalidation = True
        if str(kind) == "batch" and isinstance(batch_existing_rows_by_file, Mapping):
            existing_rows_by_file = {
                str(key): canonicalize_fit_row(value)
                for key, value in dict(batch_existing_rows_by_file).items()
                if str(key).strip()
            }
            needs_fit_condition_invalidation = False
            if existing_row:
                existing_rows_by_file[str(file_path)] = canonicalize_fit_row(
                    existing_row
                )
        elif str(kind) == "batch":
            existing_rows_by_file = {
                str(row.get("file")): canonicalize_fit_row(row)
                for row in list(self.batch_results or [])
                if row.get("file")
            }
            if existing_row:
                existing_rows_by_file[str(file_path)] = canonicalize_fit_row(
                    existing_row
                )
        else:
            # Manual/single-file runs normally avoid copying the full batch table
            # for responsiveness.  When a single procedure explicitly enables
            # sibling seeding, include the current batch table so capture-matched
            # siblings are available to the worker.
            if single_proc_sibling_context_requested:
                existing_rows_by_file = {
                    str(row.get("file")): canonicalize_fit_row(row)
                    for row in list(self.batch_results or [])
                    if row.get("file")
                }
                needs_fit_condition_invalidation = False
                fit_debug(
                    "fit-task sibling context: "
                    "enabled for single procedure run "
                    f"rows={len(existing_rows_by_file)} "
                    f"file={file_path}"
                )
            else:
                existing_rows_by_file = {}
                needs_fit_condition_invalidation = False
            if existing_row:
                existing_rows_by_file[str(file_path)] = canonicalize_fit_row(
                    existing_row
                )

        if needs_fit_condition_invalidation:
            current_fp = (
                batch_fit_conditions_fingerprint
                if batch_fit_conditions_fingerprint is not None
                else self._fit_conditions_fingerprint()
            )
            for _file_key, erow in existing_rows_by_file.items():
                stored_fp = erow.get("_fit_conditions")
                if stored_fp is not None and stored_fp != current_fp:
                    fit_set(erow, "r2", None)
                    erow["_fit_conditions_stale"] = True
                    fit_debug(
                        "fit-conditions-stale: "
                        f"file={_file_key} "
                        f"stored_fp={stored_fp!r} "
                        f"current_fp={current_fp!r}"
                    )

        multi_channel_model = fit_context.get("multi_channel_model")
        model_def_for_task = fit_context.get("model_def")
        boundary_seeds_per_channel = fit_context.get("boundary_seeds_per_channel", {})
        fit_param_keys = set()
        if multi_channel_model is not None and multi_channel_model.is_multi_channel:
            fit_param_keys = {
                str(key).strip()
                for key in getattr(multi_channel_model, "global_param_names", ())
                if str(key).strip()
            }
        elif model_def_for_task is not None:
            fit_param_keys = {
                str(key).strip()
                for key in getattr(model_def_for_task, "global_param_names", ())
                if str(key).strip()
            }
        if isinstance(batch_filtered_capture_map, Mapping):
            filtered_capture_map = {
                str(raw_key).strip(): (
                    str(raw_value) if raw_value not in (None, "") else None
                )
                for raw_key, raw_value in dict(batch_filtered_capture_map).items()
                if str(raw_key).strip()
            }
        else:
            filtered_capture_map = {}
            for raw_key, raw_value in dict(parameter_capture_map or {}).items():
                key: str = str(raw_key).strip()
                if not key:
                    continue
                if fit_param_keys and key not in fit_param_keys:
                    continue
                filtered_capture_map[key] = (
                    str(raw_value) if raw_value not in (None, "") else None
                )
        fit_debug(
            "fit-task mapping: "
            f"kind={kind} "
            f"mode={run_mode} "
            f"file={file_path} "
            f"map_keys={len(filtered_capture_map)} "
            f"fit_param_keys={len(fit_param_keys)}"
        )

        # Build job descriptor for FitWorkerThread.
        # All fits go through the procedure pipeline.  When run_mode is not
        # "procedure" we auto-build a trivial single-step FitProcedure so
        # that BatchProcedureFitWorker handles every fit uniformly.
        descriptor: dict
        if run_mode == "procedure":
            proc = procedure
            if proc is None:
                proc, _ = self._current_procedure_for_run()
            if proc is None:
                raise ValueError("No procedure steps defined.")
            # If procedure sibling-seeding is enabled and this is a non-batch
            # run, ensure we have a sibling context even when the caller did not
            # pre-populate one.
            if str(kind) != "batch" and bool(getattr(proc, "seed_from_siblings", False)):
                if len(existing_rows_by_file) <= 1:
                    existing_rows_by_file = {
                        str(row.get("file")): canonicalize_fit_row(row)
                        for row in list(self.batch_results or [])
                        if row.get("file")
                    }
                    if existing_row:
                        existing_rows_by_file[str(file_path)] = canonicalize_fit_row(
                            existing_row
                        )
                fit_debug(
                    "fit-task sibling context: "
                    "single procedure seed_from_siblings enabled "
                    f"rows={len(existing_rows_by_file)} "
                    f"file={file_path}"
                )

            if isinstance(batch_procedure_capture_map, Mapping):
                task_capture_map: Dict[str, str] = {
                    str(key): str(value)
                    for key, value in dict(batch_procedure_capture_map).items()
                    if str(key).strip() and value not in (None, "")
                }
            else:
                task_capture_map = self._procedure_capture_field_map(proc)
        else:
            # ---- auto-build a trivial procedure for plain fits ----
            from procedure import FitProcedure
            from procedure_steps import FitStep

            fixed_params_dict: Dict[str, float] = dict(
                fit_context.get("fixed_params") or {}
            )
            random_restarts: int = (
                int(getattr(self, "_batch_refit_random_restarts", 0))
                if str(kind) == "batch"
                else 0
            )

            # Translate locked boundary IDs → boundary group names.
            boundary_groups_list = self._proc_available_boundary_groups() or []
            bid_to_name: Dict[Tuple[str, int], str] = {}
            for _bg_name, _bg_members in boundary_groups_list:
                for _bg_target, _bg_idx in _bg_members:
                    bid_to_name[(str(_bg_target), int(_bg_idx))] = _bg_name
            locked_names: Set[str] = set()
            for bid in getattr(self, "_manually_fixed_boundary_ids", set()):
                _ln = bid_to_name.get((str(bid[0]), int(bid[1])))
                if _ln:
                    locked_names.add(_ln)

            proc = FitProcedure(
                name="Auto Fit",
                steps=(
                    FitStep(
                        fixed_params=tuple(sorted(fixed_params_dict.keys())),
                        max_retries=random_restarts,
                        retry_mode=(
                            "random" if random_restarts > 0 else "jitter_then_random"
                        ),
                        retry_scale=1.0 if random_restarts > 0 else 0.3,
                        locked_boundary_names=tuple(sorted(locked_names)),
                    ),
                ),
            )

            # Inject fixed-param values into seed_map so the procedure step
            # reads the correct locked values.
            if fixed_params_dict:
                patched_seed_map = dict(fit_context["seed_map"])
                for key, value in fixed_params_dict.items():
                    patched_seed_map[key] = float(value)
                fit_context = {**fit_context, "seed_map": patched_seed_map}

            task_capture_map = filtered_capture_map

        # Ensure a multi-channel model is available (wrap single-channel if
        # needed).
        proc_multi_model = multi_channel_model
        if proc_multi_model is None and model_def_for_task is not None:
            proc_multi_model = MultiChannelModelDefinition(
                channel_models=(model_def_for_task,),
                global_param_names=tuple(model_def_for_task.global_param_names),
            )
        if proc_multi_model is None:
            raise ValueError("No compiled model available for fit.")

        use_existing_fit_seed = str(kind) == "batch" or (
            run_mode == "procedure"
            and str(kind) != "batch"
            and bool(getattr(proc, "seed_from_siblings", False))
        )

        descriptor = {
            "worker_kind": "procedure_batch",
            "worker_args": {
                "file_paths": [file_path],
                "source_indices": [int(source_index)],
                "regex_pattern": capture_regex_pattern,
                "capture_defaults": capture_defaults,
                "parameter_capture_map": task_capture_map,
                "multi_channel_model": proc_multi_model,
                "ordered_param_keys": fit_context["ordered_keys"],
                "seed_map": fit_context["seed_map"],
                "bounds_map": fit_context["bounds_map"],
                "periodic_params": fit_context.get("periodic_params", {}),
                "boundary_seeds_per_channel": boundary_seeds_per_channel,
                "x_channel": self.x_channel,
                "procedure": proc,
                "smoothing_enabled": self.smoothing_enabled,
                "smoothing_window": self._effective_smoothing_window(),
                "boundary_name_groups": dict(
                    self._proc_available_boundary_groups() or []
                ),
                "existing_rows_by_file": existing_rows_by_file,
                "use_existing_fit_seed": bool(use_existing_fit_seed),
            },
        }

        self.fit_tasks[task_id] = {
            "id": int(task_id),
            "kind": str(kind),
            "execution_mode": run_mode,
            "file_path": str(file_path),
            "file_key": self._fit_task_file_key(file_path),
            "source_index": int(source_index),
            "procedure_name": (
                str(getattr(proc, "name", "") or "Procedure")
                if run_mode == "procedure"
                else ""
            ),
            "procedure_step_count": (
                len(getattr(proc, "steps", ()) or ())
                if run_mode == "procedure"
                else None
            ),
            "queue_position": (
                int(queue_position) if queue_position not in (None, "") else None
            ),
            "status": "pending",
            "_pre_fit_r2": finite_float_or_none(getattr(self, "_last_r2", None)),
            "_progress_started_at": None,
        }
        if str(kind) == "batch":
            existing_r2 = None
            row_idx: None | int = self._find_batch_result_index_by_file(file_path)
            if row_idx is not None and 0 <= row_idx < len(self.batch_results):
                existing_r2 = fit_get(self.batch_results[row_idx], "r2")
            self._set_batch_row_runtime_fields(
                file_path,
                _fit_status="Queued",
                _fit_task_id=int(task_id),
                _queue_position=(
                    int(queue_position) if queue_position not in (None, "") else None
                ),
                _r2_old=existing_r2,
            )
            if run_mode == "procedure":
                panel: Any | None = getattr(self, "_procedure_panel", None)
                if panel is not None:
                    file_name: str = stem_for_file_ref(file_path)
                    panel.record_external_procedure_start(
                        procedure_name=str(
                            getattr(proc, "name", "") or "Procedure"
                        ),
                        file_label=file_name,
                        step_count=len(getattr(proc, "steps", ()) or ()),
                    )

        # Submit to the single worker thread.
        if priority:
            self._fit_worker_thread.preempt(task_id, descriptor)
        else:
            self._fit_worker_thread.submit(task_id, descriptor)

        # Show the live procedure panel for multi-step procedures.
        # For batch runs we always activate it so the user sees progress
        # across all files, not just the currently-loaded one.
        n_steps = len(getattr(proc, "steps", ()) or ())
        current_file = self._current_loaded_file_path()
        is_current = current_file and self._fit_task_file_key(
            current_file
        ) == self._fit_task_file_key(file_path)
        if run_mode == "procedure" and n_steps > 1:
            step_infos: List[dict] = []
            for s in proc.steps:
                lbl = (
                    getattr(s, "label", "")
                    or getattr(s, "step_label", "")
                    or s.step_type
                )
                info: dict = {"label": str(lbl), "step_type": s.step_type}
                channels = getattr(s, "channels", None)
                if channels:
                    display_channels: List[str] = []
                    for raw_channel in channels:
                        channel_key: str = str(raw_channel).strip()
                        if not channel_key:
                            continue
                        display_name: str = str(
                            self._channel_display_name(channel_key)
                        ).strip()
                        display_channels.append(display_name or channel_key)
                    if display_channels:
                        info["channels"] = display_channels
                free = getattr(s, "free_params", None)
                fixed = getattr(s, "fixed_params", None)
                info["n_free"] = len(free) if free else 0
                info["n_fixed"] = len(fixed) if fixed else 0
                info["max_retries"] = int(getattr(s, "max_retries", 0) or 0)
                info["retry_mode"] = str(getattr(s, "retry_mode", "") or "")
                locked = getattr(s, "locked_boundary_names", None)
                if locked:
                    info["locked_boundary_names"] = list(locked)
                step_infos.append(info)
            # Store step_infos on the task so the panel can re-init on task switch.
            self.fit_tasks[task_id]["_step_infos"] = step_infos
            self.fit_tasks[task_id]["_proc_name"] = str(
                getattr(proc, "name", "") or "Procedure"
            )
            self.fit_tasks[task_id]["_file_label"] = (
                stem_for_file_ref(file_path) if file_path else ""
            )
            self.fit_tasks[task_id]["_live_panel_active"] = True
            self.fit_tasks[task_id]["_is_current_file"] = bool(is_current)

            # Only init & show the overlay for the first procedure task, or
            # when there is no task currently displayed on the panel.
            if (
                not hasattr(self, "_panel_active_task_id")
                or self._panel_active_task_id is None
            ):
                self._procedure_live_panel.start_procedure(
                    self.fit_tasks[task_id]["_proc_name"],
                    step_infos,
                    file_label=self.fit_tasks[task_id]["_file_label"],
                )
                self._show_procedure_overlay()
                self._panel_active_task_id = task_id

        self._refresh_fit_action_buttons()
        self._refresh_batch_controls()
        return int(task_id)

    def _on_fit_task_progress(self, task_id, _completed, _total, row) -> None:
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return
        task["status"] = "running"
        if not finite_float_or_none(task.get("_progress_started_at")):
            task["_progress_started_at"] = float(time.perf_counter())
        # Batch-procedure workers emit a final per-file row on progress and then
        # emit the same row again on finished. Applying both doubles expensive UI
        # work on the main thread.
        if (
            str(task.get("kind")) == "batch"
            and self._normalize_fit_run_mode(task.get("execution_mode")) == "procedure"
        ):
            self._update_batch_procedure_status(current_task=task)
            return
        self._apply_fit_row_update(row)

    # -- Procedure overlay helpers --

    def _show_procedure_overlay(self):
        """Show the procedure-log overlay at the bottom of the param pane."""
        self._procedure_live_panel.reposition()
        self._procedure_live_panel.raise_()
        self._procedure_live_panel.show()
        self._proc_log_toggle_btn.hide()

    def _on_procedure_panel_dismissed(self):
        """User clicked the X on the overlay."""
        self._proc_log_toggle_btn.setText("\u25b2 Show Procedure Log")
        self._proc_log_toggle_btn.show()

    def _on_proc_log_toggle(self):
        """Toggle the procedure-log overlay visibility."""
        if self._procedure_live_panel.isVisible():
            self._procedure_live_panel.hide()
            self._proc_log_toggle_btn.setText("\u25b2 Show Procedure Log")
        else:
            self._procedure_live_panel.reposition()
            self._procedure_live_panel.raise_()
            self._procedure_live_panel.show()
            self._proc_log_toggle_btn.hide()

    # -- Per-step and per-attempt live feedback --

    def _on_fit_task_step(self, task_id, step_idx, step_result) -> None:
        """Handler for task_step_completed signal — update live panel and plot."""
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return
        if not task.get("_live_panel_active"):
            return

        # If this task is different from what the panel is currently showing,
        # re-initialise the panel for this task (happens during batch runs).
        if getattr(self, "_panel_active_task_id", None) != int(task_id):
            step_infos = task.get("_step_infos")
            if step_infos:
                self._procedure_live_panel.start_procedure(
                    str(task.get("_proc_name", "Procedure")),
                    step_infos,
                    file_label=str(task.get("_file_label", "")),
                )
                if not self._procedure_live_panel.isVisible():
                    self._show_procedure_overlay()
                self._panel_active_task_id = int(task_id)

        status = (
            str(step_result.get("status", "pass"))
            if isinstance(step_result, dict)
            else "pass"
        )
        r2 = step_result.get("r2") if isinstance(step_result, dict) else None
        self._procedure_live_panel.update_step(
            step_idx,
            status,
            r2=r2,
            step_result=step_result if isinstance(step_result, dict) else None,
        )
        task_mode: str = self._normalize_fit_run_mode(task.get("execution_mode"))
        if task_mode == "procedure":
            total_steps = int(task.get("procedure_step_count") or 0)
            done_steps = max(0, int(step_idx) + 1)
            if str(task.get("kind")) == "manual" and total_steps > 0:
                self._update_manual_procedure_status(
                    task,
                    step_done=done_steps,
                    step_total=total_steps,
                )
            elif str(task.get("kind")) == "batch":
                self._update_batch_procedure_status(
                    current_task=task,
                    step_done=done_steps,
                    step_total=(total_steps if total_steps > 0 else None),
                )

        # Apply params to spinboxes and update plot on each step completion,
        # but only if this task is for the currently loaded file.
        if isinstance(step_result, dict) and task.get("_is_current_file"):
            params_by_key = step_result.get("params_by_key")
            if isinstance(params_by_key, dict) and params_by_key:
                self._apply_step_params_to_ui(params_by_key)
            # Apply boundary ratios from the step result.
            boundary_ratios = step_result.get("boundary_ratios")
            if isinstance(boundary_ratios, dict):
                self._apply_step_boundaries_to_ui(boundary_ratios)
            self.update_plot(fast=True, preserve_view=True)

    def _on_fit_task_attempt(self, task_id, step_idx, attempt, info) -> None:
        """Handler for task_attempt_completed signal — update live panel and flash plot."""
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return
        if not task.get("_live_panel_active"):
            return
        if not isinstance(info, dict):
            return

        # If this task is different from what the panel is currently showing,
        # re-initialise the panel for this task (happens during batch runs).
        if getattr(self, "_panel_active_task_id", None) != int(task_id):
            step_infos = task.get("_step_infos")
            if step_infos:
                self._procedure_live_panel.start_procedure(
                    str(task.get("_proc_name", "Procedure")),
                    step_infos,
                    file_label=str(task.get("_file_label", "")),
                )
                if not self._procedure_live_panel.isVisible():
                    self._show_procedure_overlay()
                self._panel_active_task_id = int(task_id)

        r2 = info.get("r2")
        best_r2 = info.get("best_r2")
        is_new_best = bool(info.get("is_new_best", False))
        strategy = str(info.get("strategy", ""))
        max_attempts = int(info.get("max_attempts", 1))
        attempt_num = int(info.get("attempt", attempt))

        self._procedure_live_panel.update_attempt(
            step_idx,
            attempt_num,
            max_attempts,
            r2=r2,
            best_r2=best_r2,
            is_new_best=is_new_best,
            strategy=strategy,
            elapsed=float(info.get("elapsed", 0) or 0),
            per_channel_r2=info.get("per_channel_r2"),
        )

        # Apply this attempt's params to the plot — only for the current file.
        is_current_file = task.get("_is_current_file")
        params_by_key = info.get("params_by_key")
        boundary_ratios = info.get("boundary_ratios")
        if is_current_file:
            if isinstance(params_by_key, dict) and params_by_key:
                self._apply_step_params_to_ui(params_by_key)
            if isinstance(boundary_ratios, dict):
                self._apply_step_boundaries_to_ui(boundary_ratios)
            self.update_plot(fast=True, preserve_view=True)

        if not is_new_best:
            # Schedule a revert to best params after a brief flash.
            if is_current_file:
                best_params = task.get("_best_params_by_key")
                best_boundaries = task.get("_best_boundaries")
                if best_params is not None:
                    _revert_p = dict(best_params)
                    _revert_b = dict(best_boundaries or {})

                    def _revert(p=_revert_p, b=_revert_b):
                        if p:
                            self._apply_step_params_to_ui(p)
                        if b:
                            self._apply_step_boundaries_to_ui(b)
                        self.update_plot(fast=True, preserve_view=True)

                    QTimer.singleShot(150, _revert)
        else:
            # This is the new best — update saved state.
            task["_best_params_by_key"] = dict(params_by_key or {})
            task["_best_boundaries"] = dict(boundary_ratios or {})

    def _apply_step_params_to_ui(self, params_by_key: dict) -> None:
        """Push parameter values from a procedure step into the spinboxes."""
        blocked = []
        for key, value in params_by_key.items():
            spinbox = self.param_spinboxes.get(str(key))
            if spinbox is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            was_blocked = spinbox.blockSignals(True)
            spinbox.setValue(v)
            if not was_blocked:
                blocked.append(spinbox)
        for sb in blocked:
            sb.blockSignals(False)
        for key in params_by_key:
            if str(key) in self.param_spinboxes:
                self._sync_slider_from_spinbox(str(key))

    def _apply_step_boundaries_to_ui(self, boundary_ratios: dict) -> None:
        """Push boundary ratios from a procedure step into the fit state."""
        updated_targets: set[str] = set()
        for ch_target, ratios in boundary_ratios.items():
            try:
                arr = np.asarray(ratios, dtype=float).reshape(-1)
            except Exception:
                continue
            expected = self._fit_state.channel_count(str(ch_target))
            if arr.size == expected and expected > 0:
                self._fit_state.set_channel_ratios(
                    str(ch_target), np.clip(arr, 0.0, 1.0)
                )
                updated_targets.add(str(ch_target))
        if updated_targets:
            self._fit_state.apply_link_groups(
                self._boundary_links_from_map(),
                prefer_targets=sorted(updated_targets),
            )

    def _on_fit_task_finished(self, task_id, results) -> None:
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return

        row = None
        if isinstance(results, (list, tuple)) and results:
            row = results[0]
            if row is not None:
                row = self._apply_param_range_validation_to_row(row)
                # Guard: for manual fits, reject result if R² is worse than
                # the pre-fit state (avoids optimizer regression).
                if task.get("kind") == "manual":
                    pre_fit_r2 = finite_float_or_none(task.get("_pre_fit_r2"))
                    new_r2 = finite_float_or_none(fit_get(row, "r2"))
                    if (
                        pre_fit_r2 is not None
                        and new_r2 is not None
                        and new_r2 < pre_fit_r2 - 1e-12
                    ):
                        file_name = stem_for_file_ref(str(task.get("file_path", "")))
                        self.stats_text.append(
                            f"Auto-fit rejected [{file_name}]: "
                            f"R²={new_r2:.6f} worse than pre-fit "
                            f"R²={pre_fit_r2:.6f}; keeping current state."
                        )
                        row = None
                if row is not None:
                    self._apply_fit_row_update(row)

        task_mode: str = self._normalize_fit_run_mode(task.get("execution_mode"))
        manual_label: str = "Procedure fit" if task_mode == "procedure" else "Auto-fit"
        procedure_panel: Any | None = getattr(self, "_procedure_panel", None)

        if task.get("kind") == "manual":
            file_path = str(task.get("file_path"))
            file_name: str = stem_for_file_ref(file_path)
            if row is None:
                self.stats_text.append(
                    f"{manual_label} finished with no result: {file_name}"
                )
            elif has_nonempty_values(fit_get(row, "params")):
                r2_val: float | None = finite_float_or_none(fit_get(row, "r2"))
                r2_text: str = f"{float(r2_val):.6f}" if r2_val is not None else "N/A"
                row_error: str = self._batch_row_error_text(row)
                if row_error:
                    self.stats_text.append(
                        f"✗ {manual_label} failed [{file_name}]: {row_error}"
                    )
                else:
                    self.stats_text.append(
                        f"✓ {manual_label} successful [{file_name}] R²={r2_text}"
                    )
                current_file: Any | None = self._current_loaded_file_path()
                if self._fit_task_file_key(current_file) == self._fit_task_file_key(
                    file_path
                ):
                    params: (
                        np.ndarray[Tuple[int, ...], np.dtype[Any]]
                        | np.ndarray[Tuple[int], np.dtype[Any]]
                    ) = self._as_float_array(fit_get(row, "params"))
                    self.last_popt: (
                        np.ndarray[Tuple[int, ...], np.dtype[Any]]
                        | np.ndarray[Tuple[int], np.dtype[Any]]
                    ) = params
                    self._last_fit_active_keys: List[str] = self._ordered_param_keys()
                    self._last_r2: float | None = r2_val
                    self._apply_batch_params_for_file(file_path)
                    self.update_plot(fast=False)
            else:
                error_text = str(fit_get(row, "error") or "No fit result.")
                self.stats_text.append(
                    f"✗ {manual_label} failed [{file_name}]: {error_text}"
                )
            if task_mode == "procedure" and procedure_panel is not None:
                if row is None:
                    procedure_panel.record_external_procedure_failure(
                        "Procedure run produced no result.",
                        file_label=file_name,
                    )
                else:
                    proc_result = row.get("_procedure_result")
                    if isinstance(proc_result, Mapping):
                        procedure_panel.record_external_procedure_result(
                            proc_result,
                            file_label=file_name,
                        )
                    else:
                        row_error = self._batch_row_error_text(row)
                        if row_error:
                            procedure_panel.record_external_procedure_failure(
                                row_error,
                                file_label=file_name,
                            )
                        else:
                            procedure_panel.record_external_procedure_result(
                                {
                                    "step_results": [],
                                    "r2": finite_float_or_none(fit_get(row, "r2")),
                                    "stopped_at_step": None,
                                },
                                file_label=file_name,
                            )
        elif task.get("kind") == "batch" and row is not None:
            status_text = "Done"
            if bool(row.get("_fit_no_change")):
                status_text = "No Change"
            elif self._batch_row_error_text(row):
                status_text = "Failed"
            elif not has_nonempty_values(fit_get(row, "params")):
                status_text = "No Result"
            self._set_batch_row_runtime_fields(
                task.get("file_path"),
                _fit_status=status_text,
                _fit_task_id=None,
            )
            if has_nonempty_values(fit_get(row, "params")):
                seed_source: str = str(row.get("_seed_source") or "").strip().lower()
                if seed_source in {"matching-captures", "closest-captures"}:
                    file_path = str(task.get("file_path"))
                    file_name: str = stem_for_file_ref(file_path)
                    source_file: str = str(row.get("_seed_source_file") or "").strip()
                    source_name: str = (
                        stem_for_file_ref(source_file) if source_file else "another row"
                    )
                    if seed_source == "matching-captures":
                        self.stats_text.append(
                            f"ℹ Seed used [{file_name}]: matched extracted fields from {source_name}."
                        )
                    else:
                        self.stats_text.append(
                            f"ℹ Seed used [{file_name}]: closest extracted fields from {source_name}."
                        )
            if task_mode == "procedure" and procedure_panel is not None:
                file_path = str(task.get("file_path"))
                file_name: str = stem_for_file_ref(file_path)
                proc_result = row.get("_procedure_result")
                if isinstance(proc_result, Mapping):
                    procedure_panel.record_external_procedure_result(
                        proc_result,
                        file_label=file_name,
                    )
                else:
                    row_error = self._batch_row_error_text(row)
                    if row_error:
                        procedure_panel.record_external_procedure_failure(
                            row_error,
                            file_label=file_name,
                        )
                    else:
                        procedure_panel.record_external_procedure_result(
                            {
                                "step_results": [],
                                "r2": finite_float_or_none(fit_get(row, "r2")),
                                "stopped_at_step": None,
                            },
                            file_label=file_name,
                        )

        if row is not None and has_nonempty_values(fit_get(row, "params")):
            current_file: Any | None = self._current_loaded_file_path()
            if self._fit_task_file_key(current_file) == self._fit_task_file_key(
                task.get("file_path")
            ):
                self._apply_batch_params_for_file(current_file)
                self.update_plot(fast=False)

        # --- Write-back: update shared existing_rows_by_file so the next file
        # in the batch queue can use this result as a sibling seed. ---
        if (
            task.get("kind") == "batch"
            and row is not None
            and has_nonempty_values(fit_get(row, "params"))
            and not _row_has_error(row)
        ):
            context = getattr(self, "_batch_submission_context", None)
            if isinstance(context, dict):
                shared: Dict[str, Any] | None = context.get(
                    "batch_existing_rows_by_file"
                )
                if isinstance(shared, dict):
                    file_key = str(task.get("file_path", ""))
                    if file_key:
                        shared[file_key] = canonicalize_fit_row(row)

        self._finish_fit_task(int(task_id))

    def _on_fit_task_failed(self, task_id, error_text) -> None:
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return
        file_path = str(task.get("file_path"))
        source_index = int(task.get("source_index", 0))
        self._upsert_fit_error_row(file_path, source_index, error_text)
        if task.get("kind") == "batch":
            self._set_batch_row_runtime_fields(
                file_path,
                _fit_status="Failed",
                _fit_task_id=None,
            )
            task_mode: str = self._normalize_fit_run_mode(task.get("execution_mode"))
            if task_mode == "procedure":
                panel: Any | None = getattr(self, "_procedure_panel", None)
                if panel is not None:
                    file_name: str = stem_for_file_ref(file_path)
                    panel.record_external_procedure_failure(
                        error_text,
                        file_label=file_name,
                    )
        if task.get("kind") == "manual":
            task_mode: str = self._normalize_fit_run_mode(task.get("execution_mode"))
            manual_label: str = (
                "Procedure fit" if task_mode == "procedure" else "Auto-fit"
            )
            file_name: str = stem_for_file_ref(file_path)
            self.stats_text.append(
                f"✗ {manual_label} failed [{file_name}]: {error_text}"
            )
            if task_mode == "procedure":
                panel: Any | None = getattr(self, "_procedure_panel", None)
                if panel is not None:
                    panel.record_external_procedure_failure(
                        error_text,
                        file_label=file_name,
                    )
        self._finish_fit_task(int(task_id))

    def _on_fit_task_cancelled(self, task_id) -> None:
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return
        if task.get("kind") == "batch":
            self._set_batch_row_runtime_fields(
                task.get("file_path"),
                _fit_status="Cancelled",
                _fit_task_id=None,
            )
            task_mode: str = self._normalize_fit_run_mode(task.get("execution_mode"))
            if task_mode == "procedure":
                panel: Any | None = getattr(self, "_procedure_panel", None)
                if panel is not None:
                    file_name: str = stem_for_file_ref(task.get("file_path"))
                    panel.record_external_procedure_cancelled(file_label=file_name)
        if task.get("kind") == "manual":
            task_mode: str = self._normalize_fit_run_mode(task.get("execution_mode"))
            manual_label: str = (
                "Procedure fit" if task_mode == "procedure" else "Auto-fit"
            )
            file_name: str = stem_for_file_ref(task.get("file_path"))
            self.stats_text.append(f"{manual_label} cancelled: {file_name}")
            if task_mode == "procedure":
                panel: Any | None = getattr(self, "_procedure_panel", None)
                if panel is not None:
                    panel.record_external_procedure_cancelled(file_label=file_name)
        self._finish_fit_task(int(task_id))

    def _finish_fit_task(self, task_id) -> None:
        task = self.fit_tasks.pop(int(task_id), None)
        if task is None:
            return

        # Restore param sliders if we had the live procedure panel showing.
        if task.get("_live_panel_active"):
            # For batch tasks, only call finish_procedure on the last task.
            # Clear the active panel task_id so the next task re-inits the panel.
            is_batch = str(task.get("kind")) == "batch"
            remaining_batch = (
                sum(
                    1
                    for t in self.fit_tasks.values()
                    if str(t.get("kind")) == "batch" and t.get("_live_panel_active")
                )
                if is_batch
                else 0
            )
            if getattr(self, "_panel_active_task_id", None) == int(task_id):
                self._panel_active_task_id = None
            if not is_batch or remaining_batch == 0:
                self._procedure_live_panel.finish_procedure()
                # Leave the overlay visible so the user can review results,
                # but update the toggle button text.
                self._proc_log_toggle_btn.setText("\u25b2 Procedure Log (done)")

        kind = str(task.get("kind"))
        if kind == "batch":
            self._batch_progress_done: int = min(
                int(self._batch_total_tasks),
                int(self._batch_progress_done) + 1,
            )
            self._set_batch_row_runtime_fields(
                task.get("file_path"),
                _fit_task_id=None,
            )
            if self._current_batch_fit_run_mode() == "procedure":
                self._update_batch_procedure_status()

        if kind == "batch" and self.batch_fit_in_progress:
            if bool(getattr(self, "_batch_cancel_requested", False)):
                self._cancel_unscheduled_batch_rows()
                remaining = sum(
                    1 for t in self.fit_tasks.values() if str(t.get("kind")) == "batch"
                )
                if remaining == 0:
                    self._complete_batch_fit_run()
            elif int(self._batch_progress_done) >= int(self._batch_total_tasks):
                self._complete_batch_fit_run()

        self._refresh_fit_action_buttons()
        self._refresh_batch_controls()

        # Resume background preload once all fit tasks are idle.
        if (
            not self.fit_tasks
            and self.data_files
            and len(self._data_preload_cache) < len(self.data_files)
            and getattr(self, "_data_preload_thread", None) is None
        ):
            active_file: Any | None = self._current_loaded_file_path()
            self._start_background_data_preload(
                prioritize_file=str(active_file) if active_file else None
            )

    def _cancel_unscheduled_batch_rows(self) -> int:
        """Mark queued rows without active tasks as Cancelled."""
        rows: List[Any] = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return 0
        active_file_keys: set[str] = set()
        for task in self.fit_tasks.values():
            if str(task.get("kind")) != "batch":
                continue
            file_key: str = str(
                task.get("file_key") or self._fit_task_file_key(task.get("file_path"))
            )
            if file_key:
                active_file_keys.add(file_key)

        changed = False
        cancelled_count = 0
        for row_idx, row in enumerate(rows):
            status_text: str = str(row.get("_fit_status") or "").strip().lower()
            if status_text != "queued":
                continue
            file_key: str = self._fit_task_file_key(row.get("file"))
            if file_key and file_key in active_file_keys:
                continue
            updated = canonicalize_fit_row(row)
            updated["_fit_status"] = "Cancelled"
            updated["_fit_task_id"] = None
            self.batch_results[row_idx] = canonicalize_fit_row(updated)
            cancelled_count += 1
            changed = True

        if changed:
            self.update_batch_table()
            self._refresh_batch_analysis_if_run()
        return int(cancelled_count)

    def _complete_batch_fit_run(self) -> None:
        cancelled = bool(getattr(self, "_batch_cancel_requested", False))
        if cancelled:
            self._cancel_unscheduled_batch_rows()
            self.stats_text.append("Batch fit cancelled.")
        else:
            self.stats_text.append("✓ Batch fit completed.")

        self.batch_fit_in_progress = False
        self._batch_submission_context = None
        self._batch_cancel_pending = False
        self._batch_cancel_requested = False
        self._batch_progress_done = 0
        self._batch_total_tasks = 0
        self._batch_progress_started_at = 0.0
        self.update_batch_table()
        if not bool(getattr(self, "_close_shutdown_in_progress", False)):
            self.queue_visible_thumbnail_render()
        self._flush_batch_analysis_refresh()
        self._autosave_fit_details()
        self._refresh_batch_controls()

    def run_batch_fit(self) -> None:
        """Run batch fitting using the shared file list."""
        if self.batch_fit_in_progress:
            self.stats_text.append("Batch fit is already running.")
            return
        run_mode: str = self._current_batch_fit_run_mode()
        procedure = None
        if run_mode == "procedure":
            procedure, procedure_error = self._current_procedure_for_run()
            if procedure_error:
                self.stats_text.append(procedure_error)
                return
        rerun_choice: str = self._prompt_batch_results_on_rerun()
        if rerun_choice == "cancel":
            self.stats_text.append("Batch fit cancelled; existing batch results kept.")
            return
        clear_existing_results: bool = rerun_choice == "clear"
        previous_rows: List[Any] = list(getattr(self, "batch_results", []) or [])
        self._sync_batch_files_from_shared(sync_pattern=False)
        if not self.batch_files:
            self.stats_text.append("No files available from the shared folder list.")
            return

        capture_config: CapturePatternConfig | None = (
            self._resolve_batch_capture_config(show_errors=True)
        )
        if capture_config is None:
            return

        try:
            fit_context = self._build_fit_context(
                respect_enabled_channels=(run_mode != "procedure")
            )
        except Exception as exc:
            self.stats_text.append(f"Batch fit setup error: {exc}")
            return

        self._stop_thumbnail_render()
        self._batch_progress_done = 0
        self.batch_fit_in_progress = True
        self._batch_cancel_pending = False
        self._batch_cancel_requested = False
        self._batch_submission_context = None

        existing_by_file = {}
        if not clear_existing_results:
            for row in previous_rows:
                file_ref: str = str(row.get("file") or "").strip()
                if not file_ref:
                    continue
                existing_by_file[self._fit_task_file_key(file_ref)] = row
            for row in list(getattr(self, "batch_results", []) or []):
                file_ref: str = str(row.get("file") or "").strip()
                if not file_ref:
                    continue
                existing_by_file[self._fit_task_file_key(file_ref)] = row
        self.batch_results = []
        work_items = []

        def _refit_priority(
            existing_row, source_index_value, has_existing_fit
        ) -> Tuple[Literal[0], float, int] | Tuple[Literal[1], float, int]:
            if not has_existing_fit:
                return (0, 0.0, int(source_index_value))
            r2_val: float | None = finite_float_or_none(fit_get(existing_row, "r2"))
            if r2_val is None:
                distance = float("inf")
            else:
                distance = float(abs(1.0 - float(r2_val)))
            return (1, -distance, int(source_index_value))

        existing_r2_by_file_key: Dict[str, Any] = {}
        for source_index, file_path in enumerate(self.batch_files):
            existing = existing_by_file.get(self._fit_task_file_key(file_path), {})
            file_key_for_row: str = self._fit_task_file_key(file_path)
            existing_r2_by_file_key[file_key_for_row] = fit_get(existing, "r2")
            extracted: Dict[str, str] | None = extract_captures(
                stem_for_file_ref(file_path),
                capture_config.regex,
                capture_config.defaults,
            )
            captures: Dict[str, str] = extracted if extracted is not None else {}
            pattern_error: None | str = (
                _BATCH_PATTERN_MISMATCH_ERROR if extracted is None else None
            )
            self.batch_results.append(
                self._build_batch_result_row(
                    source_index=source_index,
                    file_path=file_path,
                    captures=captures,
                    pattern_error=pattern_error,
                    existing=existing,
                    preserve_fit_result=(not clear_existing_results),
                )
            )
            has_existing_fit: bool = (
                has_nonempty_values(fit_get(existing, "params"))
                if not clear_existing_results
                else False
            )
            work_items.append(
                (
                    source_index,
                    file_path,
                    has_existing_fit,
                    _refit_priority(existing, source_index, has_existing_fit),
                )
            )

        prioritized_items = sorted(work_items, key=lambda item: item[3])
        queue_position_by_file_key: Dict[str, int] = {}
        for queue_position, (
            _source_index,
            file_path,
            _already_fitted,
            _priority,
        ) in enumerate(prioritized_items, start=1):
            queue_position_by_file_key[self._fit_task_file_key(file_path)] = int(
                queue_position
            )

        # Show the full batch queue immediately, even when worker creation is deferred.
        for row_idx, row in enumerate(list(self.batch_results or [])):
            file_ref: str = str(row.get("file") or "").strip()
            file_key = self._fit_task_file_key(file_ref)
            queue_pos = queue_position_by_file_key.get(file_key)
            if queue_pos is None:
                continue
            updated = canonicalize_fit_row(row)
            updated["_fit_status"] = "Queued"
            updated["_queue_position"] = int(queue_pos)
            updated["_fit_task_id"] = None
            updated["_r2_old"] = existing_r2_by_file_key.get(file_key)
            self.batch_results[row_idx] = canonicalize_fit_row(updated)

        self.update_batch_table()
        self._batch_total_tasks: int = len(prioritized_items)
        self._batch_progress_started_at = float(time.perf_counter())
        self._refresh_batch_controls()
        mode_text: str = (
            "procedure mode" if run_mode == "procedure" else "straightforward mode"
        )
        self.stats_text.append(f"Batch fit started ({mode_text}, queued).")
        if run_mode == "procedure":
            self._update_batch_procedure_status()

        parameter_capture_map: Dict[str, None] = (
            self._effective_param_capture_map_for_fixing()
        )
        fit_param_keys: set[str] = set()
        fit_multi_model = fit_context.get("multi_channel_model")
        fit_model_def = fit_context.get("model_def")
        if fit_multi_model is not None and fit_multi_model.is_multi_channel:
            fit_param_keys = {
                str(key).strip()
                for key in getattr(fit_multi_model, "global_param_names", ())
                if str(key).strip()
            }
        elif fit_model_def is not None:
            fit_param_keys = {
                str(key).strip()
                for key in getattr(fit_model_def, "global_param_names", ())
                if str(key).strip()
            }
        filtered_capture_map: Dict[str, Optional[str]] = {}
        for raw_key, raw_value in dict(parameter_capture_map or {}).items():
            key_text: str = str(raw_key).strip()
            if not key_text:
                continue
            if fit_param_keys and key_text not in fit_param_keys:
                continue
            filtered_capture_map[key_text] = (
                str(raw_value) if raw_value not in (None, "") else None
            )
        procedure_capture_map: Dict[str, str] = (
            self._procedure_capture_field_map(procedure)
            if run_mode == "procedure" and procedure is not None
            else {}
        )

        shared_existing_rows_by_file: Dict[str, Dict[str, Any]] = {
            str(row.get("file")): canonicalize_fit_row(row)
            for row in list(self.batch_results or [])
            if row.get("file")
        }
        current_fp = self._fit_conditions_fingerprint()
        for file_key, existing_row in shared_existing_rows_by_file.items():
            stored_fp = existing_row.get("_fit_conditions")
            if stored_fp is not None and stored_fp != current_fp:
                fit_set(existing_row, "r2", None)
                existing_row["_fit_conditions_stale"] = True
                fit_debug(
                    "fit-conditions-stale: "
                    f"file={file_key} "
                    f"stored_fp={stored_fp!r} "
                    f"current_fp={current_fp!r}"
                )

        self._batch_submission_context = {
            "fit_context": fit_context,
            "capture_regex_pattern": capture_config.regex_pattern,
            "capture_defaults": dict(capture_config.defaults),
            "parameter_capture_map": parameter_capture_map,
            "execution_mode": run_mode,
            "procedure": procedure,
            "batch_existing_rows_by_file": shared_existing_rows_by_file,
            "batch_fit_conditions_fingerprint": current_fp,
            "batch_procedure_capture_map": procedure_capture_map,
            "batch_filtered_capture_map": filtered_capture_map,
        }

        # Submit all batch tasks directly to the single worker thread.
        # Yield to the event loop periodically so the GUI stays responsive.
        for queue_position, (
            source_index,
            file_path,
            _already_fitted,
            _priority,
        ) in enumerate(prioritized_items, start=1):
            try:
                self._start_file_fit_task(
                    kind="batch",
                    file_path=file_path,
                    source_index=source_index,
                    fit_context=fit_context,
                    capture_regex_pattern=capture_config.regex_pattern,
                    capture_defaults=dict(capture_config.defaults),
                    parameter_capture_map=parameter_capture_map,
                    queue_position=queue_position,
                    execution_mode=run_mode,
                    procedure=procedure,
                    batch_existing_rows_by_file=shared_existing_rows_by_file,
                    batch_fit_conditions_fingerprint=current_fp,
                    batch_procedure_capture_map=procedure_capture_map,
                    batch_filtered_capture_map=filtered_capture_map,
                )
            except Exception as exc:
                self._force_stop_batch_fit(f"Batch fit setup error: {exc}")
                return
            # Yield to UI event loop every 5 submissions to prevent hang.
            if queue_position % 5 == 0:
                QApplication.processEvents()

    def _force_stop_batch_fit(self, reason_text) -> None:
        if not self.batch_fit_in_progress:
            return
        self._batch_cancel_requested = True
        self._batch_submission_context = None
        self.stats_text.append(str(reason_text))

        # Cancel all pending and running tasks on the worker thread.
        batch_task_ids = {
            int(tid)
            for tid, task in self.fit_tasks.items()
            if str(task.get("kind")) == "batch"
        }
        if batch_task_ids:
            self._fit_worker_thread.cancel_tasks(batch_task_ids)
        for task_id in list(batch_task_ids):
            task = self.fit_tasks.get(int(task_id))
            if task is not None:
                self._set_batch_row_runtime_fields(
                    task.get("file_path"),
                    _fit_status="Cancelled",
                    _fit_task_id=None,
                )
            self._finish_fit_task(int(task_id))
        self._cancel_unscheduled_batch_rows()
        remaining = sum(
            1 for t in self.fit_tasks.values() if str(t.get("kind")) == "batch"
        )
        if self.batch_fit_in_progress and remaining == 0:
            self._complete_batch_fit_run()

    def cancel_batch_fit(self) -> None:
        """Request cancellation of an in-flight batch fit."""
        if not self.batch_fit_in_progress:
            return
        if not self._batch_cancel_pending:
            self._batch_cancel_pending = True
            self._batch_cancel_requested = True
            self._batch_submission_context = None
            self._cancel_unscheduled_batch_rows()

            # Cancel all batch tasks on the worker thread.
            batch_task_ids = {
                int(tid)
                for tid, task in self.fit_tasks.items()
                if str(task.get("kind")) == "batch"
            }
            if batch_task_ids:
                self._fit_worker_thread.cancel_tasks(batch_task_ids)

            self.stats_text.append(
                "Batch cancellation requested... click Cancel again to force stop."
            )
            self._refresh_batch_controls()
            remaining = sum(
                1 for t in self.fit_tasks.values() if str(t.get("kind")) == "batch"
            )
            if remaining == 0:
                self._complete_batch_fit_run()
                return
            QTimer.singleShot(
                1500,
                lambda: (
                    self._force_stop_batch_fit(
                        "⚠ Batch fit did not stop promptly; force-stopped."
                    )
                    if self.batch_fit_in_progress and self._batch_cancel_pending
                    else None
                ),
            )
            return
        self._force_stop_batch_fit("⚠ Batch fit force-stopped.")

    def update_batch_table(self) -> None:
        """Refresh batch results table with captures and fit params."""
        if not self.batch_results:
            self.batch_table.setRowCount(0)
            self.batch_table.setColumnCount(0)
            rich_header: Any | None = getattr(self, "batch_table_header", None)
            if isinstance(rich_header, RichTextHeaderView):
                rich_header.set_section_html_map({})
            return

        sorting_enabled: bool = self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            param_columns = self._batch_parameter_column_items()
            param_column_tokens: List[str] = [
                str(item["token"]) for item in param_columns
            ]
            columns: List[str] = (
                ["Plot"]
                + ["File"]
                + self.batch_capture_keys
                + ["Queue", "Status", "R² Old", "R² New"]
                + param_column_tokens
                + ["Error"]
            )
            self.batch_table.setColumnCount(len(columns))
            self.batch_table.setHorizontalHeaderLabels(columns)
            rich_header: Any | None = getattr(self, "batch_table_header", None)
            if isinstance(rich_header, RichTextHeaderView):
                rich_map = {}
                param_start: int = 6 + len(self.batch_capture_keys)
                for offset, token in enumerate(param_column_tokens):
                    rich_html: str = parameter_symbol_to_html(token)
                    if rich_html:
                        rich_map[param_start + offset] = rich_html
                rich_header.set_section_html_map(rich_map)
            self.batch_table.setRowCount(len(self.batch_results))
            self._apply_batch_row_heights()

            for row_idx, row in enumerate(self.batch_results):
                self.update_batch_table_row(row_idx, row, suspend_sorting=False)
        finally:
            if sorting_enabled:
                self.batch_table.setSortingEnabled(True)
        self.queue_visible_thumbnail_render()

    def update_batch_table_row(self, row_idx, row, suspend_sorting=True) -> None:
        """Update a single batch row in the results table."""
        sorting_enabled: bool = suspend_sorting and self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            queue_value = row.get("_queue_position")
            queue_text: str = (
                str(int(queue_value))
                if isinstance(queue_value, (int, np.integer))
                else ""
            )
            status_text: str = str(row.get("_fit_status") or "").strip()
            status_item = NumericSortTableWidgetItem(status_text)
            status_lower: str = status_text.lower()
            status_sort_rank = {"done": 0, "running": 1, "queued": 2}.get(
                status_lower, 3
            )
            status_item.setData(TABLE_SORT_ROLE, int(status_sort_rank))
            if status_lower == "running":
                status_item.setForeground(QBrush(QColor("#1d4ed8")))
            elif status_lower == "queued":
                status_item.setForeground(QBrush(QColor("#64748b")))
            elif status_lower == "done":
                status_item.setForeground(QBrush(QColor("#15803d")))
            elif status_lower == "no change":
                status_item.setForeground(QBrush(QColor("#7c3aed")))
            elif status_lower in {"failed", "cancelled"}:
                status_item.setForeground(QBrush(QColor("#b91c1c")))

            r2_old: float | None = finite_float_or_none(row.get("_r2_old"))
            r2_old_item = NumericSortTableWidgetItem(
                f"{float(r2_old):.6f}" if r2_old is not None else ""
            )

            # Plot column (index 0)
            self._update_batch_plot_cell(row_idx, row)

            # File name column (index 1)
            file_name: str = stem_for_file_ref(row["file"])
            file_item = NumericSortTableWidgetItem(file_name)
            file_item.setData(Qt.ItemDataRole.UserRole, row["file"])
            self.batch_table.setItem(row_idx, 1, file_item)

            # Capture columns (start at index 2)
            for col_idx, key in enumerate(self.batch_capture_keys, start=2):
                value = row.get("captures", {}).get(key, "")
                self.batch_table.setItem(
                    row_idx, col_idx, NumericSortTableWidgetItem(str(value))
                )

            # Runtime columns come right after capture columns.
            queue_col: int = 2 + len(self.batch_capture_keys)
            status_col: int = queue_col + 1
            r2_old_col: int = status_col + 1
            r2_new_col: int = r2_old_col + 1
            # Parameter columns come after runtime columns.
            param_start: int = r2_new_col + 1
            param_columns = self._batch_parameter_column_items()
            error_col: int = param_start + len(param_columns)

            self.batch_table.setItem(
                row_idx,
                queue_col,
                NumericSortTableWidgetItem(queue_text),
            )
            self.batch_table.setItem(row_idx, status_col, status_item)

            self.batch_table.setItem(
                row_idx,
                r2_old_col,
                r2_old_item,
            )
            r2_val: float | None = finite_float_or_none(fit_get(row, "r2"))
            self.batch_table.setItem(
                row_idx,
                r2_new_col,
                NumericSortTableWidgetItem(
                    f"{float(r2_val):.6f}" if r2_val is not None else ""
                ),
            )

            params: (
                np.ndarray[Tuple[int, ...], np.dtype[Any]]
                | np.ndarray[Tuple[int], np.dtype[Any]]
            ) = self._as_float_array(fit_get(row, "params"))
            # Build per-channel boundary-value lookup from channel_results.
            ch_results_raw = fit_get(row, "channel_results")
            ch_boundary_values: Dict[str, np.ndarray] = {}
            if isinstance(ch_results_raw, Mapping):
                for ch_target, ch_data in ch_results_raw.items():
                    if isinstance(ch_data, Mapping):
                        ch_bv = ch_data.get("boundaries")
                        if ch_bv is not None:
                            ch_boundary_values[str(ch_target)] = self._as_float_array(
                                ch_bv
                            )
            violation_indices: set[int] = {
                int(item.get("index"))
                for item in self._fit_param_range_violations(fit_get(row, "params"))
            }
            for offset, item in enumerate(param_columns):
                idx = int(item["index"])
                if item["kind"] == "param":
                    value: float | None = (
                        float(params[idx]) if params.size > idx else None
                    )
                else:
                    # Use per-channel boundary values when available.
                    target_col = item.get("target")
                    if target_col:
                        target_key: str = str(target_col)
                        bv_arr = ch_boundary_values.get(target_key)
                        if bv_arr is None:
                            bv_arr = np.asarray([], dtype=float)
                    else:
                        bv_arr = np.asarray([], dtype=float)
                    value: float | None = (
                        float(bv_arr[idx]) if bv_arr.size > idx else None
                    )
                cell_text: str = f"{value:.6f}" if value is not None else ""
                cell_item = NumericSortTableWidgetItem(cell_text)
                if item["kind"] == "param" and idx in violation_indices:
                    cell_item.setForeground(QBrush(QColor("#b91c1c")))
                self.batch_table.setItem(row_idx, param_start + offset, cell_item)
            error_text: str = self._batch_row_error_text(row)
            self.batch_table.setItem(
                row_idx,
                error_col,
                NumericSortTableWidgetItem(error_text),
            )
            self._apply_batch_row_error_background(row_idx, bool(error_text))
        finally:
            if sorting_enabled:
                self.batch_table.setSortingEnabled(True)

    def _apply_batch_row_error_background(self, row_idx, is_error) -> None:
        """Tint errored rows pale red; force white for non-error rows."""
        if row_idx < 0 or row_idx >= self.batch_table.rowCount():
            return
        color: QColor = QColor("#fee2e2") if is_error else QColor("#ffffff")
        for col_idx in range(self.batch_table.columnCount()):
            item: QTableWidgetItem | None = self.batch_table.item(row_idx, col_idx)
            if item is not None:
                item.setBackground(color)

    def _update_batch_plot_cell(self, row_idx, row) -> None:
        """Update only the plot thumbnail cell for a batch row."""
        thumb_item = QTableWidgetItem()
        pixmap = self._scaled_batch_plot(row)
        if pixmap is not None:
            thumb_item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
        else:
            thumb_item.setData(Qt.ItemDataRole.DecorationRole, None)
        thumb_item.setData(Qt.ItemDataRole.UserRole, row["file"])  # Store file path
        self.batch_table.setItem(row_idx, 0, thumb_item)

    def _on_batch_table_cell_clicked(self, row_idx, col_idx) -> None:
        """Load selected batch row into the shared Plot tab."""
        if row_idx < 0:
            return

        clicked_item: QTableWidgetItem | None = self.batch_table.item(row_idx, col_idx)
        file_path: Any | None = (
            clicked_item.data(Qt.ItemDataRole.UserRole) if clicked_item else None
        )
        if not file_path:
            for fallback_col in (1, 0):
                file_item: QTableWidgetItem | None = self.batch_table.item(
                    row_idx, fallback_col
                )
                file_path: Any | None = (
                    file_item.data(Qt.ItemDataRole.UserRole)
                    if file_item is not None
                    else None
                )
                if file_path:
                    break
        if not file_path:
            return

        self._open_file_in_plot_tab(file_path)

    def _open_file_in_plot_tab(self, file_path) -> bool:
        if not file_path:
            return False

        current_file = self._current_loaded_file_path()
        if current_file is not None and str(current_file) == str(file_path):
            self.tabs.setCurrentWidget(self.manual_tab)
            return True

        try:
            file_idx: int = self.data_files.index(file_path)
        except ValueError:
            self.data_files.append(file_path)
            self.file_combo.addItem(stem_for_file_ref(file_path), file_path)
            self._sync_batch_files_from_shared(sync_pattern=False)
            file_idx = len(self.data_files) - 1

        self.load_file(file_idx)
        self.tabs.setCurrentWidget(self.manual_tab)
        return True

    def _expand_file_column_for_selected_files(self) -> None:
        """Expand file column width to show the longest selected file name."""
        if not self.batch_files or self.batch_table.columnCount() < 2:
            return

        font_metrics: QFontMetrics = self.batch_table.fontMetrics()
        longest_width = 0
        for file_path in self.batch_files:
            file_name: str = stem_for_file_ref(file_path)
            longest_width: int = max(
                longest_width, font_metrics.horizontalAdvance(file_name)
            )

        # Account for text padding and small header/sort margin.
        target_width: int = longest_width + 36
        current_width: int = self.batch_table.columnWidth(1)
        if target_width > current_width:
            self.batch_table.setColumnWidth(1, target_width)

    def _visible_batch_row_indices(self):
        if not hasattr(self, "batch_table") or self.batch_table.rowCount() == 0:
            return []
        viewport: QRect = self.batch_table.viewport().rect()
        model: QAbstractItemModel | None = self.batch_table.model()
        visible = []
        for row_idx in range(self.batch_table.rowCount()):
            rect: QRect = self.batch_table.visualRect(model.index(row_idx, 0))
            if rect.isValid() and rect.intersects(viewport):
                visible.append(row_idx)
        return visible

    def _visible_batch_result_indices(self):
        """Return visible rows mapped to indices in self.batch_results."""
        if not self.batch_results:
            return []
        visible_table_rows = self._visible_batch_row_indices()
        if not visible_table_rows:
            return []

        result_index_by_file = {
            row.get("file"): idx for idx, row in enumerate(self.batch_results)
        }
        visible_result_rows = []
        seen = set()
        for table_row in visible_table_rows:
            file_path = None
            for col_idx in (0, 1):
                item: QTableWidgetItem | None = self.batch_table.item(
                    table_row, col_idx
                )
                if item is not None:
                    file_path = item.data(Qt.ItemDataRole.UserRole)
                if file_path:
                    break
            if not file_path:
                continue
            result_idx: int | None = result_index_by_file.get(file_path)
            if result_idx is None or result_idx in seen:
                continue
            visible_result_rows.append(result_idx)
            seen.add(result_idx)
        return visible_result_rows

    def _prioritize_thumbnail_rows(self, row_indices):
        ordered = []
        seen = set()
        for idx in row_indices:
            try:
                row_idx = int(idx)
            except Exception:
                continue
            if 0 <= row_idx < len(self.batch_results) and row_idx not in seen:
                ordered.append(row_idx)
                seen.add(row_idx)

        if not ordered:
            return []

        visible_set = set(self._visible_batch_result_indices())
        visible_first = [idx for idx in ordered if idx in visible_set]
        not_visible = [idx for idx in ordered if idx not in visible_set]
        return visible_first + not_visible

    def _row_thumbnail_render_size(self, row) -> Tuple[int, int] | None:
        size_meta = row.get("plot_render_size")
        if isinstance(size_meta, (tuple, list)) and len(size_meta) == 2:
            try:
                return (int(size_meta[0]), int(size_meta[1]))
            except Exception:
                pass

        source = row.get("plot_full") or row.get("plot")
        if source is not None:
            try:
                width = int(source.width())
                height = int(source.height())
                if width > 0 and height > 0:
                    return (width, height)
            except Exception:
                pass
        return None

    def _batch_row_thumbnail_needs_render(self, row, expected_size) -> bool:
        source = row.get("plot_full") or row.get("plot")
        if source is None:
            return True

        rendered_size: Tuple[int, int] | None = self._row_thumbnail_render_size(row)
        if rendered_size is None:
            return True
        return tuple(rendered_size) != tuple(expected_size)

    def queue_visible_thumbnail_render(self, *_args) -> None:
        if not self.batch_results:
            return
        row_indices = self._visible_batch_result_indices()
        if not row_indices:
            row_indices: List[int] = list(range(min(len(self.batch_results), 10)))
        row_indices = self._prioritize_thumbnail_rows(row_indices)
        self._start_thumbnail_render(row_indices=row_indices)

    def _start_thumbnail_render(self, row_indices=None) -> None:
        """Start background thread to render missing thumbnails."""
        if not self.batch_results:
            return

        expected_size: Tuple[int, int] = self._full_batch_thumbnail_size()

        if row_indices is None:
            candidate_rows: List[int] = list(range(len(self.batch_results)))
        else:
            candidate_rows = self._prioritize_thumbnail_rows(row_indices)
        candidate_rows = self._prioritize_thumbnail_rows(candidate_rows)
        candidate_rows = [
            idx
            for idx in candidate_rows
            if self._batch_row_thumbnail_needs_render(
                self.batch_results[idx], expected_size
            )
        ]
        if not candidate_rows:
            return

        if self.thumb_render_in_progress:
            self._pending_thumbnail_rows.update(candidate_rows)
            return

        try:
            thumbnail_model_func = self._snapshot_full_model_function()
        except Exception:
            return

        self.thumb_render_in_progress = True
        self.thumb_thread = QThread(self)
        self.thumb_worker: ThumbnailRenderWorker[
            List, Callable[..., np.ndarray[tuple[int, ...], np.dtype[Any]]]
        ] = ThumbnailRenderWorker(
            self.batch_results,
            thumbnail_model_func,
            full_thumbnail_size=expected_size,
            row_indices=candidate_rows,
            smoothing_enabled=self.smoothing_enabled,
            smoothing_window=self._effective_smoothing_window(),
        )
        self.thumb_worker.moveToThread(self.thumb_thread)

        self.thumb_thread.started.connect(self.thumb_worker.run)
        self.thumb_worker.progress.connect(self._on_thumbnail_rendered)
        self.thumb_worker.finished.connect(self._on_thumbnails_finished)
        self.thumb_worker.cancelled.connect(self._on_thumbnails_finished)
        self.thumb_thread.start()

    def _stop_thumbnail_render(self) -> None:
        """Stop thumbnail worker/thread if active."""
        self._request_worker_cancel(self.thumb_worker)
        self._shutdown_thread(self.thumb_thread, wait_ms=2000)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        self._pending_thumbnail_rows.clear()

    def _on_thumbnail_rendered(self, _idx, _total, row_idx) -> None:
        """Update table cell when thumbnail is rendered."""
        if row_idx < len(self.batch_results):
            row = self.batch_results[row_idx]
            table_row_idx: None | int = self._find_table_row_by_file(row["file"])
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)

    def _on_thumbnails_finished(self) -> None:
        """Clean up thumbnail worker when finished."""
        self._shutdown_thread(self.thumb_thread, wait_ms=500, force_terminate=True)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        if self._pending_thumbnail_rows and not bool(
            getattr(self, "_close_shutdown_in_progress", False)
        ):
            queued = self._prioritize_thumbnail_rows(self._pending_thumbnail_rows)
            self._pending_thumbnail_rows.clear()
            self._start_thumbnail_render(row_indices=queued)

    def _set_batch_parse_feedback(self, message, is_error=False, tooltip="") -> None:
        if is_error:
            self.batch_parse_feedback_label.setText(str(message))
            self.batch_parse_feedback_label.setToolTip(str(tooltip or message))
            self.batch_parse_feedback_label.setStyleSheet(
                "color: #b91c1c; font-weight: 600; padding: 1px 2px;"
            )
            self.batch_parse_feedback_label.show()
        else:
            self.batch_parse_feedback_label.clear()
            self.batch_parse_feedback_label.setToolTip("")
            self.batch_parse_feedback_label.setStyleSheet("")
            self.batch_parse_feedback_label.hide()

    def _resolve_batch_capture_config(self, show_errors) -> CapturePatternConfig | None:
        pattern_text: str = (
            self.regex_input.text().strip() if hasattr(self, "regex_input") else ""
        )
        try:
            return parse_capture_pattern(pattern_text)
        except Exception as exc:
            if show_errors:
                self._set_batch_parse_feedback(f"Error: {exc}", is_error=True)
                self.batch_status_label.setText(f"Error: {exc}")
                self.batch_status_label.show()
            return None

    def _update_batch_capture_feedback(self, config) -> None:
        _ = config
        self._set_batch_parse_feedback("", is_error=False)

    def _on_regex_changed(self) -> None:
        """Debounce filename pattern changes to avoid excessive updates."""
        self._refresh_param_capture_mapping_controls()
        self.regex_timer.stop()
        self.regex_timer.start(300)  # 300ms debounce

    def prepare_batch_preview(self) -> None:
        """Populate preview results before running batch fit."""
        self.regex_timer.stop()
        self._do_prepare_batch_preview()

    def _do_prepare_batch_preview(self) -> None:
        """Actually perform the batch preview update."""
        if not self.batch_files:
            self._stop_thumbnail_render()
            self.batch_match_count = 0
            self.batch_unmatched_files = []
            self.batch_capture_keys = []
            self.batch_results = []
            config: CapturePatternConfig | None = self._resolve_batch_capture_config(
                show_errors=True
            )
            if config is None:
                self._refresh_param_capture_mapping_controls()
                self.update_batch_table()
                self._refresh_batch_analysis_if_run()
                return
            self.batch_status_label.hide()
            self._update_batch_capture_feedback(config)
            self._refresh_param_capture_mapping_controls()
            self.update_batch_table()
            self._refresh_batch_analysis_if_run()
            self._autosave_fit_details()
            return

        capture_config: CapturePatternConfig | None = (
            self._resolve_batch_capture_config(show_errors=True)
        )
        if capture_config is None:
            return

        self.batch_status_label.hide()

        existing_file_order = [row["file"] for row in self.batch_results]
        files_unchanged: bool = existing_file_order == self.batch_files and bool(
            self.batch_results
        )

        self.batch_capture_keys = []
        self.batch_match_count = 0
        self.batch_unmatched_files = []

        if files_unchanged:
            for source_index, row in enumerate(self.batch_results):
                row["_source_index"] = source_index
                row["x_channel"] = self.x_channel
                extracted: Dict[str, str] | None = extract_captures(
                    stem_for_file_ref(row["file"]),
                    capture_config.regex,
                    capture_config.defaults,
                )
                captures = {}
                if extracted is None:
                    self.batch_unmatched_files.append(stem_for_file_ref(row["file"]))
                    row["pattern_error"] = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures: Dict[str, str] = extracted
                    row["pattern_error"] = None
                    if capture_config.mode != "off":
                        self.batch_match_count += 1
                    for key in captures.keys():
                        if key not in self.batch_capture_keys:
                            self.batch_capture_keys.append(key)
                row["captures"] = captures
        else:
            self._stop_thumbnail_render()

            # Build a map of existing results by file path to preserve params/r2/error/plot.
            existing_results = {row["file"]: row for row in self.batch_results}
            rebuilt_results = []

            for source_index, file_path in enumerate(self.batch_files):
                captures = {}
                extracted: Dict[str, str] | None = extract_captures(
                    stem_for_file_ref(file_path),
                    capture_config.regex,
                    capture_config.defaults,
                )
                pattern_error = None
                if extracted is None:
                    self.batch_unmatched_files.append(stem_for_file_ref(file_path))
                    pattern_error: str = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures: Dict[str, str] = extracted
                    if capture_config.mode != "off":
                        self.batch_match_count += 1
                    for key in captures.keys():
                        if key not in self.batch_capture_keys:
                            self.batch_capture_keys.append(key)

                existing = existing_results.get(file_path)
                rebuilt_results.append(
                    self._build_batch_result_row(
                        source_index=source_index,
                        file_path=file_path,
                        captures=captures,
                        pattern_error=pattern_error,
                        existing=existing,
                        preserve_fit_result=True,
                    )
                )

            self.batch_results = rebuilt_results

        self._update_batch_capture_feedback(capture_config)
        self._refresh_param_capture_mapping_controls()
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        if any(
            row.get("plot_full") is None and row.get("plot") is None
            for row in self.batch_results
        ):
            self.queue_visible_thumbnail_render()
        self._autosave_fit_details()

    def _flush_batch_analysis_refresh(self) -> None:
        if not hasattr(self, "analysis_status_label"):
            return
        self._batch_analysis_refresh_pending = False
        self._refresh_batch_analysis_data(preserve_selection=True)

    def _refresh_batch_analysis_if_run(self) -> None:
        if not hasattr(self, "analysis_status_label"):
            return
        timer = getattr(self, "_batch_analysis_refresh_timer", None)
        if bool(getattr(self, "batch_fit_in_progress", False)):
            self._batch_analysis_refresh_pending = True
            if timer is not None and not timer.isActive():
                timer.start(250)
            return
        if timer is not None and timer.isActive():
            timer.stop()
        self._flush_batch_analysis_refresh()

    def _close_threads_snapshot(self) -> List[Any]:
        """Return all live QThread objects that need stopping on close."""
        threads: List[Any] = []
        seen: set[int] = set()
        # Worker thread
        wt = getattr(self, "_fit_worker_thread", None)
        if wt is not None:
            ident = int(id(wt))
            seen.add(ident)
            threads.append(wt)
        # Thumbnail thread
        thumb = getattr(self, "thumb_thread", None)
        if thumb is not None:
            ident = int(id(thumb))
            if ident not in seen:
                seen.add(ident)
                threads.append(thumb)
        preload = getattr(self, "_data_preload_thread", None)
        if preload is not None:
            ident = int(id(preload))
            if ident not in seen:
                seen.add(ident)
                threads.append(preload)
        return threads

    def _request_procedure_close_shutdown(
        self, *, force_terminate: bool = False
    ) -> None:
        panel: Any | None = getattr(self, "_procedure_panel", None)
        if panel is None:
            return
        request_shutdown = panel.request_close_shutdown
        if callable(request_shutdown):
            try:
                request_shutdown(force_terminate=bool(force_terminate))
            except Exception:
                pass

    def _request_close_shutdown(self, *, force_terminate: bool = False) -> None:
        self._batch_cancel_requested = True
        self._batch_cancel_pending = True
        self._batch_submission_context = None
        self._cancel_idle_archive_scan()
        self._stop_background_data_preload(wait_ms=120)

        # Cancel everything on the worker thread and tell it to exit its
        # run-loop.  shutdown() must be called *before* any terminate() so
        # the thread can exit cleanly without leaving the QMutex locked.
        self._fit_worker_thread.cancel_all()
        self._fit_worker_thread.shutdown()

        self._request_worker_cancel(getattr(self, "thumb_worker", None))
        self._request_procedure_close_shutdown(force_terminate=force_terminate)

        for thread in self._close_threads_snapshot():
            try:
                thread.requestInterruption()
            except Exception:
                pass
            try:
                thread.quit()
            except Exception:
                pass
            if force_terminate and bool(thread.isRunning()):
                # FitWorkerThread owns an internal QMutex. If terminate() kills it
                # while that mutex is locked, later cancel_all()/shutdown() calls
                # in closeEvent can deadlock the GUI thread.
                if thread is getattr(self, "_fit_worker_thread", None):
                    continue
                try:
                    thread.terminate()
                except Exception:
                    pass

    def _poll_close_shutdown(self) -> None:
        if not bool(getattr(self, "_close_shutdown_in_progress", False)):
            return
        running_threads = [
            t for t in self._close_threads_snapshot() if bool(t.isRunning())
        ]
        if not running_threads:
            self._close_force_accept = True
            self.close()
            return
        if time.monotonic() >= float(getattr(self, "_close_shutdown_deadline", 0.0)):
            self._request_close_shutdown(force_terminate=True)
            self._close_force_accept = True
            self.close()
            return
        QTimer.singleShot(75, self._poll_close_shutdown)

    def closeEvent(self, event) -> None:
        """Ensure worker thread is stopped before closing."""
        if bool(getattr(self, "_close_shutdown_in_progress", False)) and not bool(
            getattr(self, "_close_force_accept", False)
        ):
            event.ignore()
            return

        if not bool(getattr(self, "_close_force_accept", False)):
            active_threads = self._close_threads_snapshot()
            if bool(getattr(self, "fit_tasks", {})) or any(
                bool(thread.isRunning()) for thread in active_threads
            ):
                self._close_shutdown_in_progress = True
                self._close_shutdown_deadline = time.monotonic() + 2.0
                self._request_close_shutdown(force_terminate=False)
                QTimer.singleShot(75, self._poll_close_shutdown)
                event.ignore()
                return

        self._autosave_fit_details()
        app: QCoreApplication | None = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        analysis_timer = getattr(self, "_batch_analysis_refresh_timer", None)
        if analysis_timer is not None:
            try:
                analysis_timer.stop()
            except Exception:
                pass

        # Shut down the single worker thread.
        # cancel_all() + shutdown() tell the run-loop to exit.  If the
        # thread does not stop within 1.5 s we force-terminate it.  We must
        # NEVER call shutdown() (which locks _mutex) after terminate()
        # because terminate() can leave the QMutex permanently locked,
        # causing a deadlock on the GUI thread.
        self._fit_worker_thread.cancel_all()
        self._fit_worker_thread.shutdown()
        if not self._fit_worker_thread.wait(1500):
            try:
                self._fit_worker_thread.terminate()
            except Exception:
                pass
            self._fit_worker_thread.wait(500)

        self.fit_tasks = {}
        self._batch_submission_context = None

        self._request_worker_cancel(self.thumb_worker)
        self._request_procedure_close_shutdown(force_terminate=True)
        panel: Any | None = getattr(self, "_procedure_panel", None)
        if panel is not None:
            finalize_shutdown = panel.finalize_close_shutdown
            if callable(finalize_shutdown):
                try:
                    finalize_shutdown()
                except Exception:
                    pass
        self._shutdown_thread(self.thumb_thread, wait_ms=150, force_terminate=True)
        self._stop_background_data_preload(wait_ms=150)
        self._close_shutdown_in_progress = False
        self._close_force_accept = False
        self._close_shutdown_deadline = 0.0
        super().closeEvent(event)


def _parse_cli_source_path(
    argv: Sequence[str],
) -> Tuple[Optional[str], List[str]]:
    """Parse CLI source-path arguments and return (source_path, qt_argv)."""
    parser = argparse.ArgumentParser(
        prog=Path(str(argv[0] if argv else "fit_gui.py")).name,
        description="Launch the RedPitaya fit GUI.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Optional data source to open at startup (folder, archive, or CSV).",
    )
    parser.add_argument(
        "-s",
        "--source",
        dest="source_option",
        help="Data source to open at startup (folder, archive, or CSV).",
    )
    parsed, remaining = parser.parse_known_args(list(argv[1:]))
    source_path = str(parsed.source_option or parsed.source or "").strip() or None
    qt_argv = [str(argv[0] if argv else "fit_gui.py"), *remaining]
    return source_path, qt_argv


if __name__ == "__main__":
    startup_source, qt_argv = _parse_cli_source_path(sys.argv)
    app = QApplication(qt_argv)
    app_icon = QIcon()
    if APP_ICON_PATH.exists():
        app_icon = QIcon(str(APP_ICON_PATH))
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
    window = ManualFitGUI(source_path=startup_source)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    window.resize(1200, 800)
    sys.exit(app.exec())
