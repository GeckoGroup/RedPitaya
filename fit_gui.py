#!/usr/bin/env python3
"""
Manual Curve Fitting GUI for MI Model
Allows manual adjustment of parameters for failed automatic fits.
"""

import ast
import html
import os
import math
import sys
import re
import json
import zipfile
from collections import deque
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
)
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
from pandas import read_csv
from scipy.optimize import least_squares

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
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
    QDialog,
    QDialogButtonBox,
    QStyle,
    QStyleOptionComboBox,
    QStyleOptionHeader,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QEvent, QRectF
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QIcon,
    QPixmap,
    QPalette,
    QColor,
    QPainter,
    QPen,
    QBrush,
    QTextCharFormat,
    QSyntaxHighlighter,
    QFont,
    QTextDocument,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from solver import (
    OrderedPiecewiseConfig,
    SegmentSpec,
    predict_ordered_piecewise,
    refit_ordered_piecewise_from_seed,
)

# use Qt5Agg backend for better performance
from matplotlib.pyplot import switch_backend


def normalize_column_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("(s)", "").replace("(v)", "")
    if text in {"time", "times"}:
        return "TIME"
    if text.startswith("ch"):
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return f"CH{digits}"
    return str(name).strip().upper()


def read_measurement_csv(file_ref: str):
    """Read CSV data from plain files or zip members and normalize channel names."""

    def detect_header_row(lines, max_lines=256):
        for idx, raw_line in enumerate(list(lines)[:max_lines]):
            line = str(raw_line).strip()
            if not line:
                continue
            cells = [cell.strip().strip('"').strip("'") for cell in line.split(",")]
            if not cells:
                continue
            if normalize_column_name(cells[0]) != "TIME":
                continue
            nonempty = [cell for cell in cells if cell]
            if len(nonempty) >= 2:
                return idx
        return 0

    if "::" in file_ref and file_ref.split("::", 1)[0].lower().endswith(".zip"):
        zip_path, member = file_ref.split("::", 1)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member) as handle:
                raw = handle.read()
        preview_lines = raw.decode("utf-8", errors="ignore").splitlines()
        header_row = detect_header_row(preview_lines)
        read_kwargs = {"header": 0, "low_memory": False}
        if header_row > 0:
            read_kwargs["skiprows"] = header_row
        frame = read_csv(BytesIO(raw), **read_kwargs)
        if frame.shape[1] < 2 and header_row == 0:
            frame = read_csv(BytesIO(raw), skiprows=13, header=0, low_memory=False)
    else:
        preview_lines = []
        try:
            with open(file_ref, "r", encoding="utf-8", errors="ignore") as handle:
                for _ in range(256):
                    line = handle.readline()
                    if line == "":
                        break
                    preview_lines.append(line)
        except Exception:
            preview_lines = []
        header_row = detect_header_row(preview_lines)
        read_kwargs = {"header": 0, "low_memory": False}
        if header_row > 0:
            read_kwargs["skiprows"] = header_row
        frame = read_csv(file_ref, **read_kwargs)
        if frame.shape[1] < 2 and header_row == 0:
            frame = read_csv(file_ref, skiprows=13, header=0, low_memory=False)

    frame = frame.rename(
        columns={col: normalize_column_name(col) for col in frame.columns}
    )
    if "TIME" not in frame.columns and "TIME(S)" in frame.columns:
        frame = frame.rename(columns={"TIME(S)": "TIME"})
    return frame


def stem_for_file_ref(file_ref: str) -> str:
    if "::" in file_ref:
        _zip_path, member = file_ref.split("::", 1)
        return Path(member).stem
    return Path(file_ref).stem


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)


class NumericSortTableWidgetItem(QTableWidgetItem):
    """Sort numerically when both cells contain valid finite numbers."""

    @staticmethod
    def _to_number(text):
        if text is None:
            return None
        stripped = str(text).strip()
        if not stripped:
            return None
        try:
            value = float(stripped)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            left_num = self._to_number(self.text())
            right_num = self._to_number(other.text())
            if left_num is not None and right_num is not None:
                return left_num < right_num
        return super().__lt__(other)


class CompactDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that omits unnecessary trailing decimals in display text."""

    def textFromValue(self, value):
        decimals = max(0, int(self.decimals()))
        text = f"{float(value):.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text in {"", "-0"}:
            return "0"
        return text


class ClickableLabel(QLabel):
    """Simple clickable label used for source-path selection."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class SingleLineStatusLabel(QLabel):
    """Single-line status label with QTextEdit-like append/setText helpers."""

    def __init__(self, text=""):
        super().__init__("")
        self.setObjectName("statusLabel")
        self.setText(text)

    @staticmethod
    def _normalize(text):
        raw = str(text) if text is not None else ""
        raw = raw.replace("\n", " ")
        return re.sub(r"\s+", " ", raw).strip()

    def setText(self, text):
        normalized = self._normalize(text)
        super().setText(normalized)
        super().setToolTip(str(text) if text is not None else "")

    def append(self, text):
        self.setText(text)

    def clear(self):
        super().clear()
        super().setToolTip("")


class VerticallyCenteredTextEdit(QTextEdit):
    """QTextEdit that keeps short content centered vertically."""

    focus_left = pyqtSignal()
    apply_requested = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._top_margin = -1
        self.textChanged.connect(self._queue_vertical_centering)
        self.document().documentLayout().documentSizeChanged.connect(
            self._queue_vertical_centering
        )
        QTimer.singleShot(0, self._apply_vertical_centering)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._queue_vertical_centering()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.focus_left.emit()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            modifiers = event.modifiers()
            if modifiers == Qt.KeyboardModifier.NoModifier:
                self.apply_requested.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def _queue_vertical_centering(self, *_args):
        QTimer.singleShot(0, self._apply_vertical_centering)

    def _apply_vertical_centering(self):
        doc_height = int(math.ceil(self.document().size().height()))
        viewport_height = int(self.viewport().height())
        top_margin = max(0, (viewport_height - doc_height) // 2)
        if top_margin == self._top_margin:
            return
        self._top_margin = top_margin
        self.setViewportMargins(0, top_margin, 0, 0)


_UNICODE_SUBSCRIPT_TRANS = str.maketrans(
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


def _format_boundary_display_name(index: int) -> str:
    idx = max(0, int(index))
    return f"X{str(idx).translate(_UNICODE_SUBSCRIPT_TRANS)}"


_RICH_TEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 11


class RichTextItemDelegate(QStyledItemDelegate):
    """Render HTML snippets stored in _RICH_TEXT_ROLE."""

    def paint(self, painter, option, index):
        html_text = index.data(_RICH_TEXT_ROLE)
        if not html_text:
            super().paint(painter, option, index)
            return

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget
        )

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget
        )
        if not text_rect.isValid():
            return
        doc = QTextDocument()
        doc.setDefaultFont(opt.font)
        doc.setHtml(str(html_text))
        doc.setTextWidth(float(max(0, text_rect.width())))
        painter.save()
        painter.setClipRect(text_rect)
        y_offset = text_rect.top() + (text_rect.height() - doc.size().height()) * 0.5
        painter.translate(text_rect.left(), y_offset)
        doc.drawContents(
            painter,
            QRectF(0.0, 0.0, float(text_rect.width()), float(text_rect.height())),
        )
        painter.restore()

    def sizeHint(self, option, index):
        html_text = index.data(_RICH_TEXT_ROLE)
        if not html_text:
            return super().sizeHint(option, index)
        doc = QTextDocument()
        doc.setDefaultFont(option.font)
        doc.setHtml(str(html_text))
        size = doc.size()
        base = super().sizeHint(option, index)
        return QSize(
            max(base.width(), int(math.ceil(size.width())) + 10),
            max(base.height(), int(math.ceil(size.height())) + 6),
        )


class RichTextComboBox(QComboBox):
    """QComboBox that renders HTML labels for popup items and current selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(RichTextItemDelegate(self))

    def add_rich_item(self, plain_text, user_data=None, html_text=None):
        self.addItem(str(plain_text), user_data)
        idx = self.count() - 1
        if html_text is not None:
            self.setItemData(idx, str(html_text), _RICH_TEXT_ROLE)

    def paintEvent(self, event):
        _ = event
        idx = self.currentIndex()
        rich_html = self.itemData(idx, _RICH_TEXT_ROLE) if idx >= 0 else None
        if not rich_html:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = ""
        style = self.style()
        style.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt, painter, self)
        style.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt, painter, self)

        text_rect = style.subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            opt,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        if not text_rect.isValid():
            return
        doc = QTextDocument()
        doc.setDefaultFont(self.font())
        doc.setHtml(str(rich_html))
        doc.setTextWidth(float(max(0, text_rect.width())))
        painter.save()
        painter.setClipRect(text_rect)
        y_offset = text_rect.top() + (text_rect.height() - doc.size().height()) * 0.5
        painter.translate(text_rect.left(), y_offset)
        doc.drawContents(
            painter,
            QRectF(0.0, 0.0, float(text_rect.width()), float(text_rect.height())),
        )
        painter.restore()


class RichTextHeaderView(QHeaderView):
    """Header view that can render selected section labels as rich text."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._section_html_map = {}

    def set_section_html_map(self, mapping):
        self._section_html_map = {
            int(key): str(value)
            for key, value in dict(mapping or {}).items()
            if value not in (None, "")
        }
        self.viewport().update()

    def paintSection(self, painter, rect, logical_index):
        html_text = self._section_html_map.get(int(logical_index))
        if not html_text:
            super().paintSection(painter, rect, logical_index)
            return

        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        if hasattr(self, "initStyleOptionForIndex"):
            self.initStyleOptionForIndex(opt, int(logical_index))
        opt.rect = rect
        opt.section = int(logical_index)
        opt.text = ""
        style = self.style()
        style.drawControl(QStyle.ControlElement.CE_HeaderSection, opt, painter, self)
        style.drawControl(QStyle.ControlElement.CE_HeaderLabel, opt, painter, self)

        text_rect = style.subElementRect(QStyle.SubElement.SE_HeaderLabel, opt, self)
        if not text_rect.isValid():
            text_rect = rect.adjusted(4, 0, -4, 0)

        doc = QTextDocument()
        doc.setDefaultFont(self.font())
        doc.setHtml(str(html_text))
        doc.setTextWidth(float(max(0, text_rect.width())))
        painter.save()
        painter.setClipRect(text_rect)
        y_offset = text_rect.top() + (text_rect.height() - doc.size().height()) * 0.5
        painter.translate(text_rect.left(), y_offset)
        doc.drawContents(
            painter,
            QRectF(0.0, 0.0, float(text_rect.width()), float(text_rect.height())),
        )
        painter.restore()


class MultiHandleSlider(QWidget):
    """Horizontal slider with multiple draggable handles in [0, 1]."""

    valuesChanged = pyqtSignal(object)
    sliderPressed = pyqtSignal()
    sliderReleased = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = []
        self._labels = []
        self._active_index = -1
        self._handle_radius = 7
        self._track_margin = 10
        self.setMinimumHeight(32)
        self.setMaximumHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def values(self):
        return list(self._values)

    def set_values(self, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            self._values = []
            self.update()
            return
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.maximum.accumulate(arr)
        self._values = [float(v) for v in arr.tolist()]
        self.update()

    def set_labels(self, labels):
        if labels is None:
            self._labels = []
            self.update()
            return
        self._labels = [str(item) for item in list(labels)]
        self.update()

    def _track_geometry(self):
        margin = int(min(max(6, self._handle_radius + 1), max(6, self.width() // 5)))
        x0 = margin
        x1 = max(margin + 1, self.width() - margin)
        y = int(round(self.height() * 0.68))
        return x0, x1, y

    def _value_to_x(self, value):
        x0, x1, _ = self._track_geometry()
        ratio = float(np.clip(value, 0.0, 1.0))
        return x0 + ratio * float(x1 - x0)

    def _x_to_value(self, x_pos):
        x0, x1, _ = self._track_geometry()
        span = max(1.0, float(x1 - x0))
        ratio = (float(x_pos) - float(x0)) / span
        return float(np.clip(ratio, 0.0, 1.0))

    def _nearest_handle_index(self, x_pos):
        if not self._values:
            return -1
        distances = [abs(self._value_to_x(v) - float(x_pos)) for v in self._values]
        return int(np.argmin(distances))

    def _set_handle_from_x(self, index, x_pos, *, emit_signal):
        if index < 0 or index >= len(self._values):
            return
        proposed = self._x_to_value(x_pos)
        x0, x1, _ = self._track_geometry()
        epsilon = 1.0 / max(1.0, float(x1 - x0))
        lower = self._values[index - 1] + epsilon if index > 0 else 0.0
        upper = (
            self._values[index + 1] - epsilon if index < len(self._values) - 1 else 1.0
        )
        if lower > upper:
            lower = self._values[index - 1] if index > 0 else 0.0
            upper = self._values[index + 1] if index < len(self._values) - 1 else 1.0
        value = float(np.clip(proposed, lower, upper))
        changed = not np.isclose(value, self._values[index])
        self._values[index] = value
        self.update()
        if changed and emit_signal:
            self.valuesChanged.emit(self.values())

    def paintEvent(self, event):
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        x0, x1, y = self._track_geometry()

        track_pen = QPen(QColor("#cbd5e1"), 4)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawLine(x0, y, x1, y)

        label_font = QFont(painter.font())
        label_font.setPointSize(max(7, int(label_font.pointSize()) - 1))
        painter.setFont(label_font)
        metrics = painter.fontMetrics()

        for idx, value in enumerate(self._values):
            x_pos = int(round(self._value_to_x(value)))
            is_active = idx == self._active_index
            painter.setPen(
                QPen(QColor("#4b5563") if not is_active else QColor("#1d4ed8"), 1.2)
            )
            painter.setBrush(
                QBrush(QColor("#6b7280") if not is_active else QColor("#2563eb"))
            )
            diameter = int(self._handle_radius * 2)
            painter.drawEllipse(
                int(x_pos - self._handle_radius),
                int(y - self._handle_radius),
                diameter,
                diameter,
            )

            label = (
                str(self._labels[idx])
                if idx < len(self._labels) and str(self._labels[idx]).strip()
                else _format_boundary_display_name(idx)
            )
            text_width = int(metrics.horizontalAdvance(label))
            tx = int(round(x_pos - (text_width * 0.5)))
            tx = max(0, min(tx, max(0, self.width() - text_width)))
            baseline = max(metrics.ascent(), int(y - self._handle_radius - 4))
            painter.setPen(QColor("#475569"))
            painter.drawText(tx, baseline, label)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._values:
            super().mousePressEvent(event)
            return
        self._active_index = self._nearest_handle_index(event.position().x())
        if self._active_index >= 0:
            self.sliderPressed.emit()
            self._set_handle_from_x(
                self._active_index,
                event.position().x(),
                emit_signal=True,
            )
        event.accept()

    def mouseMoveEvent(self, event):
        if self._active_index >= 0 and (event.buttons() & Qt.MouseButton.LeftButton):
            self._set_handle_from_x(
                self._active_index,
                event.position().x(),
                emit_signal=True,
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._active_index >= 0:
            self._set_handle_from_x(
                self._active_index,
                event.position().x(),
                emit_signal=True,
            )
            self._active_index = -1
            self.sliderReleased.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


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


# Initialize backend after fit defaults.
switch_backend("Qt5Agg")

APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "redpitaya_icon.png"


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


_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPRESSION_COLUMN_COLOR = "#1d4ed8"
_EXPRESSION_PARAM_COLOR = "#047857"
_EXPRESSION_CONSTANT_COLOR = "#9333ea"
_EXPRESSION_ALLOWED_FUNCTIONS = {
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
_EXPRESSION_ALLOWED_CONSTANTS = {"pi": float(np.pi), "e": float(np.e)}
_EXPRESSION_HELPER_NAMES = {"col", "columns", "C", "math"}
_LATEX_HTML_COMMANDS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ϵ",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}
_DISPLAY_FUNCTION_NAMES = {
    "abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "log10": "log10",
    "sqrt": "sqrt",
    "power": "pow",
    "minimum": "min",
    "maximum": "max",
    "clip": "clip",
}


def _normalize_latex_symbol_token(token_text: str) -> str:
    token = str(token_text).strip()
    if not token:
        return ""
    if token.startswith("$") and token.endswith("$") and len(token) >= 2:
        token = token[1:-1].strip()
    if token.startswith("{") and token.endswith("}") and len(token) >= 3:
        token = token[1:-1].strip()
    token = token.strip()
    if token.startswith("{") and token.endswith("}") and len(token) >= 3:
        token = token[1:-1].strip()
    return token


def _normalize_latex_script_token(token_text: str) -> str:
    token = str(token_text).strip()
    if not token:
        return ""
    token = re.sub(r"_(?!\{)([A-Za-z0-9+\-]+)", r"_{\1}", token)
    token = re.sub(r"\^(?!\{)([A-Za-z0-9+\-]+)", r"^{\1}", token)
    balance = token.count("{") - token.count("}")
    if balance > 0:
        token = token + ("}" * balance)
    return token


def _strip_outer_braces(text: str) -> str:
    token = str(text).strip()
    if token.startswith("{") and token.endswith("}") and len(token) >= 2:
        return token[1:-1].strip()
    return token


def _latex_fragment_to_html(text: str) -> str:
    token = str(text).strip()
    if not token:
        return ""
    if token.startswith("\\"):
        command = token[1:]
        mapped = _LATEX_HTML_COMMANDS.get(command)
        if mapped is not None:
            return mapped
    return html.escape(token)


def latex_symbol_to_plain(symbol_text: str) -> Optional[str]:
    token = _normalize_latex_symbol_token(symbol_text)
    if not token:
        return None
    return _normalize_latex_script_token(token)


def parameter_symbol_to_html(symbol_text: str) -> str:
    token = _normalize_latex_script_token(
        _normalize_latex_symbol_token(str(symbol_text))
    )
    if not token:
        return ""
    match = re.match(
        r"^(?P<base>.+?)(?:_(?P<sub>\{[^{}]*\}|[^_^]+))?(?:\^(?P<sup>\{[^{}]*\}|[^_^]+))?$",
        token,
    )
    if not match:
        return _latex_fragment_to_html(token)
    base = _latex_fragment_to_html(match.group("base") or "")
    sub = _strip_outer_braces(match.group("sub") or "")
    sup = _strip_outer_braces(match.group("sup") or "")
    out = str(base)
    if sub:
        out += f"<sub>{_latex_fragment_to_html(sub)}</sub>"
    if sup:
        out += f"<sup>{_latex_fragment_to_html(sup)}</sup>"
    return out


def parameter_symbol_to_mathtext(symbol_text: str) -> str:
    token = _normalize_latex_script_token(
        _normalize_latex_symbol_token(str(symbol_text))
    )
    if not token:
        return ""
    if token.startswith("$") and token.endswith("$") and len(token) >= 2:
        return token
    return f"${token}$"


def resolve_parameter_symbol(param_key: str, symbol_hint: Optional[str] = None) -> str:
    key_text = str(param_key).strip()
    if symbol_hint is not None:
        raw_symbol = str(symbol_hint).strip()
        if raw_symbol:
            mapped_symbol = latex_symbol_to_plain(raw_symbol)
            return mapped_symbol if mapped_symbol else raw_symbol
    mapped_from_key = latex_symbol_to_plain(key_text)
    return mapped_from_key if mapped_from_key else key_text


def _format_number_literal(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isfinite(value):
            if np.isclose(value, round(value), atol=1e-12):
                return str(int(round(value)))
            return f"{value:.8g}"
        return str(value)
    return str(value)


def _ast_callable_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _ast_callable_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _parenthesize_if_binop(node, rendered: str) -> str:
    if isinstance(node, ast.BinOp):
        return f"({rendered})"
    return rendered


def _parenthesize_if_add_sub(node, rendered: str) -> str:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return f"({rendered})"
    return rendered


def _render_expression_pretty(
    node,
    name_map: Optional[Mapping[str, str]] = None,
) -> str:
    if isinstance(node, ast.BinOp):
        left = _render_expression_pretty(node.left, name_map=name_map)
        right = _render_expression_pretty(node.right, name_map=name_map)

        if isinstance(node.op, ast.Add):
            return f"{left} + {right}"
        if isinstance(node.op, ast.Sub):
            return f"{left} - {right}"
        if isinstance(node.op, ast.Mult):
            return (
                f"{_parenthesize_if_add_sub(node.left, left)} · "
                f"{_parenthesize_if_add_sub(node.right, right)}"
            )
        if isinstance(node.op, ast.Div):
            return (
                f"{_parenthesize_if_binop(node.left, left)} / "
                f"{_parenthesize_if_binop(node.right, right)}"
            )
        if isinstance(node.op, ast.Pow):
            left_render = _parenthesize_if_binop(node.left, left)
            if (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float))
                and float(node.right.value).is_integer()
            ):
                exponent = str(int(node.right.value))
                return f"{left_render}^{exponent}"
            right_render = _render_expression_pretty(node.right, name_map=name_map)
            return f"{left_render}^{_parenthesize_if_binop(node.right, right_render)}"
        if isinstance(node.op, ast.Mod):
            return (
                f"{_parenthesize_if_binop(node.left, left)} mod "
                f"{_parenthesize_if_binop(node.right, right)}"
            )
        return f"{left} ? {right}"

    if isinstance(node, ast.UnaryOp):
        operand = _render_expression_pretty(node.operand, name_map=name_map)
        operand = _parenthesize_if_binop(node.operand, operand)
        if isinstance(node.op, ast.USub):
            return f"-{operand}"
        if isinstance(node.op, ast.UAdd):
            return f"+{operand}"
        return operand

    if isinstance(node, ast.Call):
        call_name = _ast_callable_name(node.func)
        short_name = call_name.split(".")[-1] if call_name else ""
        display_name = _DISPLAY_FUNCTION_NAMES.get(
            call_name, _DISPLAY_FUNCTION_NAMES.get(short_name, short_name or "f")
        )
        args = [_render_expression_pretty(arg, name_map=name_map) for arg in node.args]
        if display_name == "abs" and len(args) == 1:
            return f"|{args[0]}|"
        return f"{display_name}({', '.join(args)})"

    if isinstance(node, ast.Attribute):
        return node.attr

    if isinstance(node, ast.Name):
        if node.id == "pi":
            return "pi"
        if node.id == "e":
            return "e"
        if name_map:
            mapped = name_map.get(node.id)
            if mapped:
                return str(mapped)
        return node.id

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            return repr(value)
        return _format_number_literal(value)

    try:
        return ast.unparse(node)
    except Exception:
        return str(node)


def format_expression_pretty(
    expression_text: str,
    name_map: Optional[Mapping[str, str]] = None,
) -> str:
    text = str(expression_text).strip()
    if not text:
        return ""
    try:
        tree = ast.parse(text, mode="eval")
        return _render_expression_pretty(tree.body, name_map=name_map)
    except Exception:
        fallback = text
        fallback = re.sub(r"\b(?:np|math)\.pi\b", "pi", fallback)
        fallback = re.sub(r"\b(?:np|math)\.e\b", "e", fallback)
        fallback = re.sub(
            r"\b(?:np|math)\.(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|exp|log10|log|sqrt|abs|power|minimum|maximum|clip)\b",
            lambda m: _DISPLAY_FUNCTION_NAMES.get(m.group(1), m.group(1)),
            fallback,
        )
        fallback = re.sub(r"\s*\*\s*", " · ", fallback)
        if name_map:
            for name, symbol in sorted(
                name_map.items(),
                key=lambda item: len(str(item[0])),
                reverse=True,
            ):
                raw_name = str(name).strip()
                rendered_symbol = str(symbol).strip()
                if not raw_name or not rendered_symbol or raw_name == rendered_symbol:
                    continue
                fallback = re.sub(
                    rf"\b{re.escape(raw_name)}\b", rendered_symbol, fallback
                )
        return re.sub(r"\s+", " ", fallback).strip()


def format_equation_pretty(
    equation_text: str,
    name_map: Optional[Mapping[str, str]] = None,
) -> str:
    text = str(equation_text).strip()
    if not text:
        return ""
    if "=" not in text:
        return format_expression_pretty(text, name_map=name_map)
    lhs, rhs = text.split("=", 1)
    return f"{lhs.strip()} = {format_expression_pretty(rhs.strip(), name_map=name_map)}"


class ExpressionSyntaxHighlighter(QSyntaxHighlighter):
    """Colorize columns, parameters, and constants in the expression editor."""

    _WORD_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    _NUMBER_RE = re.compile(r"(?<![A-Za-z_])(?:\d+\.\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?")

    def __init__(self, document):
        super().__init__(document)
        self.column_names = set()
        self.param_names = set()

        self.column_format = QTextCharFormat()
        self.column_format.setForeground(QColor(_EXPRESSION_COLUMN_COLOR))
        self.column_format.setFontWeight(QFont.Weight.Bold)

        self.param_format = QTextCharFormat()
        self.param_format.setForeground(QColor(_EXPRESSION_PARAM_COLOR))
        self.param_format.setFontWeight(QFont.Weight.Bold)

        self.constant_format = QTextCharFormat()
        self.constant_format.setForeground(QColor(_EXPRESSION_CONSTANT_COLOR))

    def set_context(self, column_names: Sequence[str], param_names: Sequence[str]):
        self.column_names = {str(name) for name in (column_names or [])}
        self.param_names = {str(name) for name in (param_names or [])}
        self.rehighlight()

    def highlightBlock(self, text):
        if not text:
            return

        for match in self._NUMBER_RE.finditer(text):
            start = match.start()
            end = match.end()
            self.setFormat(start, end - start, self.constant_format)

        for match in self._WORD_RE.finditer(text):
            token = match.group(0)
            start = match.start()
            end = match.end()
            if token in self.column_names:
                self.setFormat(start, end - start, self.column_format)
            elif token in self.param_names:
                self.setFormat(start, end - start, self.param_format)
            elif token in _EXPRESSION_ALLOWED_CONSTANTS:
                self.setFormat(start, end - start, self.constant_format)


class _ExpressionParameterCollector(ast.NodeVisitor):
    """Collect user-defined parameter names in first-seen order."""

    def __init__(self, reserved_names=None):
        super().__init__()
        self.names = []
        self._seen = set()
        self.reserved_names = set(reserved_names or ())

    def visit_Call(self, node):
        # Function tokens are validated separately; only parse argument expressions.
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            return
        name = node.id
        if name in self.reserved_names:
            return
        if name in _EXPRESSION_ALLOWED_FUNCTIONS:
            return
        if name in _EXPRESSION_ALLOWED_CONSTANTS:
            return
        if name not in self._seen:
            self._seen.add(name)
            self.names.append(name)


def extract_expression_parameter_names(
    expression_text: str,
    reserved_names: Optional[Sequence[str]] = None,
) -> List[str]:
    text = str(expression_text).strip()
    if not text:
        raise ValueError("Function expression is empty.")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid function expression: {exc.msg}") from exc

    reserved = set(reserved_names or ())
    reserved |= {"np"} | _EXPRESSION_HELPER_NAMES
    collector = _ExpressionParameterCollector(reserved_names=reserved)
    collector.visit(tree)

    if not collector.names:
        raise ValueError("Function must reference at least one fit parameter.")

    for name in collector.names:
        if name == "x":
            raise ValueError(
                "Bare 'x' is not supported. Use explicit CSV columns (for example CH3 or TIME)."
            )
        if not _PARAMETER_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid parameter name '{name}' in expression.")
    return collector.names


def compile_expression_function(
    expression_text: str,
    parameter_names: Sequence[str],
):
    text = str(expression_text).strip()
    if not text:
        raise ValueError("Function expression is empty.")

    ordered_names = list(parameter_names)
    if not ordered_names:
        raise ValueError("No parameters are defined for this function.")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid function expression: {exc.msg}") from exc

    code = compile(tree, "<fit_expression>", "eval")
    eval_globals = {
        "__builtins__": __builtins__,
        "np": np,
        "math": math,
        **_EXPRESSION_ALLOWED_FUNCTIONS,
        **_EXPRESSION_ALLOWED_CONSTANTS,
    }

    def _prepare_channel_array(values, target_length: int):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == target_length:
            return array
        if array.size == 1:
            return np.full(target_length, float(array[0]), dtype=float)
        raise ValueError(
            f"Column length {array.size} does not match input length {target_length}."
        )

    def _evaluate(
        x_data,
        param_values,
        column_data=None,
    ):
        input_array = np.asarray(x_data, dtype=float).reshape(-1)
        n_points = input_array.size
        columns = {}
        if column_data:
            for name, values in column_data.items():
                try:
                    columns[str(name)] = _prepare_channel_array(values, n_points)
                except Exception:
                    continue

        def col(name):
            key = str(name)
            if key in columns:
                return columns[key]
            if key.upper() in columns:
                return columns[key.upper()]
            if key.lower() in columns:
                return columns[key.lower()]
            raise KeyError(f"Column '{key}' not found.")

        eval_locals = {
            "col": col,
            "columns": columns,
            "C": columns,
        }
        if "TIME" in columns:
            eval_locals["TIME"] = columns["TIME"]

        for key, values in columns.items():
            if _PARAMETER_NAME_RE.fullmatch(key):
                eval_locals[key] = values

        for name in ordered_names:
            if name not in param_values:
                raise ValueError(f"Missing parameter '{name}' for expression.")
            eval_locals[name] = float(param_values[name])

        try:
            result = eval(code, eval_globals, eval_locals)
        except Exception as exc:
            raise ValueError(f"Function evaluation failed: {exc}") from exc

        result_array = np.asarray(result, dtype=float)
        if result_array.shape == ():
            return np.full_like(input_array, float(result_array), dtype=float)
        result_array = result_array.reshape(-1)
        if result_array.size != n_points:
            raise ValueError("Function output length does not match input length.")
        return result_array

    return _evaluate


@dataclass(frozen=True)
class CapturePatternConfig:
    mode: str
    regex_pattern: str
    regex: Optional[Pattern[str]]
    defaults: Dict[str, str]


_FIELD_NAME_RE = _PARAMETER_NAME_RE


def _is_optional_delimiter(char: str) -> bool:
    """Return True for punctuation delimiters that may wrap optional fields."""
    return (
        bool(char)
        and (not char.isalnum())
        and (char not in "{}*")
        and (not char.isspace())
    )


def _template_to_regex(template_text: str) -> Tuple[str, Dict[str, str]]:
    """Convert a template into regex and default values for optional fields."""
    if not template_text:
        raise ValueError("Template is empty.")

    pieces = ["^"]
    defaults = {}
    seen_fields = set()
    idx = 0
    length = len(template_text)
    literal_buffer = []

    def flush_literal():
        if literal_buffer:
            pieces.append(re.escape("".join(literal_buffer)))
            literal_buffer.clear()

    while idx < length:
        char = template_text[idx]
        if char == "{":
            end = template_text.find("}", idx + 1)
            if end < 0:
                raise ValueError("Missing closing '}' in template.")
            field_spec = template_text[idx + 1 : end].strip()
            manual_prefix = None
            manual_suffix = None
            if "=" in field_spec:
                field_name, default_spec = field_spec.split("=", 1)
                field_name = field_name.strip()
                affix_parts = default_spec.split("|")
                if len(affix_parts) == 1:
                    default_value = affix_parts[0].strip()
                elif len(affix_parts) == 2:
                    default_value = affix_parts[0].strip()
                    manual_prefix = affix_parts[1]
                elif len(affix_parts) == 3:
                    default_value = affix_parts[0].strip()
                    manual_prefix = affix_parts[1]
                    manual_suffix = affix_parts[2]
                else:
                    raise ValueError(
                        f"Optional field '{field_name}' has too many '|' parts. "
                        "Use {field=default|prefix|suffix} or {field|prefix|suffix}."
                    )
                optional = True
            elif "|" in field_spec:
                affix_parts = field_spec.split("|")
                field_name = affix_parts[0].strip()
                default_value = ""
                if len(affix_parts) == 2:
                    manual_prefix = affix_parts[1]
                elif len(affix_parts) == 3:
                    manual_prefix = affix_parts[1]
                    manual_suffix = affix_parts[2]
                else:
                    raise ValueError(
                        "Optional field has too many '|' parts. "
                        "Use {field|prefix|suffix}."
                    )
                optional = True
            else:
                field_name = field_spec
                default_value = None
                optional = False

            if not _FIELD_NAME_RE.fullmatch(field_name):
                raise ValueError(
                    f"Invalid field name '{field_name}'. Use letters, numbers, underscore."
                )
            if field_name in seen_fields:
                raise ValueError(f"Duplicate field name '{field_name}'.")

            prefix = ""
            suffix = ""
            if optional:
                if manual_prefix is not None:
                    prefix = manual_prefix
                elif literal_buffer and _is_optional_delimiter(literal_buffer[-1]):
                    prefix = literal_buffer.pop()

            flush_literal()

            if optional:
                defaults[field_name] = default_value
                if manual_suffix is not None:
                    suffix = manual_suffix
                    idx = end + 1
                else:
                    next_idx = end + 1
                    if next_idx < length and _is_optional_delimiter(
                        template_text[next_idx]
                    ):
                        suffix = template_text[next_idx]
                        idx = next_idx + 1
                    else:
                        idx = end + 1

                optional_pieces = []
                if prefix and suffix:
                    # If both affixes are configured, allow either side to be
                    # missing in filenames while still supporting both.
                    optional_pieces.append(f"(?:{re.escape(prefix)})?")
                    optional_pieces.append(f"(?P<{field_name}>.+?)")
                    optional_pieces.append(f"(?:{re.escape(suffix)})?")
                else:
                    if prefix:
                        optional_pieces.append(re.escape(prefix))
                    optional_pieces.append(f"(?P<{field_name}>.+?)")
                    if suffix:
                        optional_pieces.append(re.escape(suffix))
                pieces.append(f"(?:{''.join(optional_pieces)})?")
            else:
                pieces.append(f"(?P<{field_name}>.+?)")
                idx = end + 1
            seen_fields.add(field_name)
            continue
        if char == "}":
            raise ValueError("Unexpected '}' in template.")
        if char == "*":
            flush_literal()
            pieces.append(".*")
            idx += 1
            continue

        literal_buffer.append(char)
        idx += 1

    flush_literal()
    pieces.append("$")
    return "".join(pieces), defaults


def parse_capture_pattern(pattern_text: str) -> CapturePatternConfig:
    """Parse capture input as template syntax only."""
    text = pattern_text.strip()
    if not text:
        return CapturePatternConfig(
            mode="off", regex_pattern="", regex=None, defaults={}
        )

    regex_pattern, defaults = _template_to_regex(text)
    regex = re.compile(regex_pattern)
    return CapturePatternConfig(
        mode="template", regex_pattern=regex_pattern, regex=regex, defaults=defaults
    )


def extract_captures(
    stem: str,
    regex: Optional[Pattern[str]],
    defaults: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    """Extract captures from a filename stem, supporting named and positional groups."""
    if regex is None:
        return {}
    match = regex.search(stem)
    if not match:
        return None

    capture_defaults = defaults or {}
    captures = match.groupdict()
    if captures:
        for key, default_value in capture_defaults.items():
            if captures.get(key) in (None, ""):
                captures[key] = default_value
        return captures

    groups = match.groups()
    if groups:
        return {f"group_{idx + 1}": value for idx, value in enumerate(groups)}

    return {"match": match.group(0)}


_BATCH_PATTERN_MISMATCH_ERROR = "Pattern mismatch: filename does not match pattern."
_FIT_PARAM_RANGE_ERROR_PREFIX = "Out-of-range fit parameter(s):"


def resolve_fixed_params_from_captures(
    parameter_capture_map: Mapping[str, Optional[str]],
    captures: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    fixed_values: Dict[str, float] = {}
    capture_map = dict(parameter_capture_map or {})
    capture_values = dict(captures or {})
    for param_key, capture_key in capture_map.items():
        if not capture_key:
            continue
        if capture_key not in capture_values:
            return None, (
                f"Mapped capture '{capture_key}' for parameter '{param_key}' is missing."
            )
        raw_value = capture_values.get(capture_key)
        text = str(raw_value).strip() if raw_value is not None else ""
        if text == "":
            return None, (
                f"Mapped capture '{capture_key}' for parameter '{param_key}' is empty."
            )
        try:
            numeric = float(text)
        except Exception:
            return None, (
                f"Mapped capture '{capture_key}' for parameter '{param_key}' is non-numeric: {raw_value!r}"
            )
        if not np.isfinite(numeric):
            return None, (
                f"Mapped capture '{capture_key}' for parameter '{param_key}' is not finite: {raw_value!r}"
            )
        fixed_values[str(param_key)] = float(numeric)
    return fixed_values, None


def make_batch_result_row(
    source_index,
    file_path,
    x_channel,
    y_channel,
    captures=None,
    params=None,
    r2=None,
    error=None,
    plot_full=None,
    plot=None,
    plot_has_fit=None,
    plot_render_size=None,
    boundary_ratios=None,
    boundary_values=None,
    pattern_error=None,
    equation_stale=False,
    fit_status=None,
    queue_position=None,
    r2_old=None,
    fit_task_id=None,
):
    return {
        "_source_index": int(source_index),
        "file": file_path,
        "captures": dict(captures or {}),
        "params": params,
        "r2": r2,
        "error": error,
        "x_channel": x_channel,
        "y_channel": y_channel,
        "plot_full": plot_full,
        "plot": plot,
        "plot_has_fit": plot_has_fit,
        "plot_render_size": plot_render_size,
        "boundary_ratios": boundary_ratios,
        "boundary_values": boundary_values,
        "pattern_error": pattern_error,
        "_equation_stale": bool(equation_stale),
        "_fit_status": (str(fit_status) if fit_status not in (None, "") else None),
        "_queue_position": (
            int(queue_position) if queue_position not in (None, "") else None
        ),
        "_r2_old": r2_old,
        "_fit_task_id": (int(fit_task_id) if fit_task_id not in (None, "") else None),
    }


def render_batch_thumbnail(
    row,
    model_func,
    full_thumbnail_size=(468, 312),
    smoothing_enabled=False,
    smoothing_window=1,
):
    """Render a row thumbnail pixmap, including all channels and fitted curve."""
    try:
        data = read_measurement_csv(row["file"])
        x_col = row.get("x_channel") or (
            "CH3" if "CH3" in data.columns else data.columns[0]
        )
        y_col = row.get("y_channel") or (
            "CH2" if "CH2" in data.columns else data.columns[0]
        )
        x_data = data[x_col].to_numpy(dtype=float, copy=True)
        y_data = data[y_col].to_numpy(dtype=float, copy=True)
        if smoothing_enabled:
            y_data = smooth_channel_array(y_data, smoothing_window)
            x_data = smooth_channel_array(x_data, smoothing_window)

        # Keep thumbnail rendering lightweight: use a reduced sample count.
        sample_count = len(x_data)
        max_thumbnail_points = 600
        if sample_count > max_thumbnail_points:
            step = int(np.ceil(sample_count / float(max_thumbnail_points)))
            y_data = y_data[::step]
            x_data = x_data[::step]
        column_data = {}
        for column in data.columns:
            key = str(column).strip()
            if not key:
                continue
            try:
                column_values = data[column].to_numpy(dtype=float, copy=True)
                if smoothing_enabled:
                    column_values = smooth_channel_array(
                        column_values, smoothing_window
                    )
                if sample_count > max_thumbnail_points:
                    column_values = column_values[::step]
                column_data[key] = column_values
            except Exception:
                continue

        target_width = max(24, int(full_thumbnail_size[0]))
        target_height = max(24, int(full_thumbnail_size[1]))
        render_dpi = 72
        fig = Figure(
            figsize=(target_width / render_dpi, target_height / render_dpi),
            dpi=render_dpi,
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
        ax = fig.add_subplot(111)
        plot_channel_names = []
        if y_col in column_data:
            plot_channel_names.append(str(y_col))
        for column in data.columns:
            key = str(column).strip()
            if (
                not key
                or key == str(x_col)
                or key in plot_channel_names
                or key not in column_data
            ):
                continue
            plot_channel_names.append(key)

        for idx, channel_name in enumerate(plot_channel_names):
            channel_values = np.asarray(column_data[channel_name], dtype=float)
            if channel_values.size != x_data.size:
                continue
            is_target = str(channel_name) == str(y_col)
            ax.plot(
                x_data,
                channel_values,
                linewidth=1.25 if is_target else 1.0,
                alpha=1.0 if is_target else 0.8,
                color=palette_color(idx),
            )

        params = row.get("params")
        if params is not None:
            try:
                params_arr = np.asarray(params, dtype=float).reshape(-1)
            except Exception:
                params_arr = np.asarray([], dtype=float)
        else:
            params_arr = np.asarray([], dtype=float)
        if params_arr.size > 0:
            fitted_y = model_func(
                x_data,
                *params_arr.tolist(),
                column_data=column_data,
                boundary_ratios=row.get("boundary_ratios"),
            )
            ax.plot(x_data, fitted_y, linewidth=1.25, color=FIT_CURVE_COLOR)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=render_dpi)
        buf.seek(0)
        data_bytes = buf.getvalue()
        buf.close()

        pixmap = QPixmap()
        pixmap.loadFromData(data_bytes, "PNG")
        return pixmap
    except Exception:
        pixmap = QPixmap(full_thumbnail_size[0], full_thumbnail_size[1])
        pixmap.fill(Qt.GlobalColor.white)
        return pixmap


class FitWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, x_data, y_data, fit_context):
        super().__init__()
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.fit_context = dict(fit_context or {})
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            if self.cancel_requested:
                self.cancelled.emit()
                return

            result = run_piecewise_fit_pipeline(
                self.x_data,
                self.y_data,
                self.fit_context["model_def"],
                self.fit_context["seed_map"],
                self.fit_context["bounds_map"],
                boundary_seed=self.fit_context.get("boundary_seed"),
                cancel_check=lambda: self.cancel_requested,
                fixed_params=self.fit_context.get("fixed_params"),
            )
            if self.cancel_requested:
                self.cancelled.emit()
                return
            self.finished.emit(result)
        except FitCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            if self.cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class BatchFitWorker(QObject):
    progress = pyqtSignal(int, int, object)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        file_paths,
        source_indices,
        existing_rows_by_file,
        regex_pattern,
        capture_defaults,
        parameter_capture_map,
        model_def,
        ordered_param_keys,
        seed_map,
        bounds_map,
        boundary_seed,
        x_channel,
        y_channel,
        use_existing_fit_seed=True,
        random_restarts=0,
        smoothing_enabled=False,
        smoothing_window=1,
    ):
        super().__init__()
        self.file_paths = list(file_paths)
        self.source_indices = [int(idx) for idx in list(source_indices or ())]
        if len(self.source_indices) != len(self.file_paths):
            self.source_indices = list(range(len(self.file_paths)))
        self.existing_rows_by_file = {
            str(key): dict(value or {})
            for key, value in dict(existing_rows_by_file or {}).items()
        }
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.capture_defaults = dict(capture_defaults or {})
        self.parameter_capture_map = {
            str(key): (str(value) if value not in (None, "") else None)
            for key, value in dict(parameter_capture_map or {}).items()
        }
        self._capture_seed_keys = tuple(
            sorted(
                {
                    str(field)
                    for field in self.parameter_capture_map.values()
                    if field not in (None, "")
                }
            )
        )
        self.model_def = model_def
        self.ordered_param_keys = list(ordered_param_keys or ())
        self.seed_map = {
            str(key): float(val) for key, val in dict(seed_map or {}).items()
        }
        self.bounds_map = {
            str(key): (float(v[0]), float(v[1]))
            for key, v in dict(bounds_map or {}).items()
        }
        self.boundary_seed = np.asarray(boundary_seed, dtype=float)
        self.x_channel = str(x_channel)
        self.y_channel = str(y_channel)
        self.use_existing_fit_seed = bool(use_existing_fit_seed)
        self.random_restarts = max(0, int(random_restarts))
        self._rng = np.random.default_rng()
        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoothing_window = int(smoothing_window)
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    def _mapping_fixed_params(
        self, captures: Mapping[str, Any]
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        return resolve_fixed_params_from_captures(self.parameter_capture_map, captures)

    @staticmethod
    def _copy_fit_fields_from_existing(
        row: Dict[str, Any], existing_row: Mapping[str, Any]
    ) -> Dict[str, Any]:
        preserved = dict(row)
        existing_params = existing_row.get("params")
        if existing_params is not None:
            try:
                existing_params = (
                    np.asarray(existing_params, dtype=float).reshape(-1).tolist()
                )
            except Exception:
                pass
        preserved["params"] = existing_params
        preserved["r2"] = existing_row.get("r2")
        preserved["error"] = (
            None if has_nonempty_values(existing_params) else existing_row.get("error")
        )
        boundary_ratios = existing_row.get("boundary_ratios")
        if boundary_ratios is not None:
            try:
                boundary_ratios = (
                    np.asarray(boundary_ratios, dtype=float).reshape(-1).copy()
                )
            except Exception:
                pass
        boundary_values = existing_row.get("boundary_values")
        if boundary_values is not None:
            try:
                boundary_values = (
                    np.asarray(boundary_values, dtype=float).reshape(-1).copy()
                )
            except Exception:
                pass
        preserved["boundary_ratios"] = boundary_ratios
        preserved["boundary_values"] = boundary_values
        return preserved

    def _seed_map_with_existing_fit(
        self, seed_map: Dict[str, float], existing_row: Mapping[str, Any]
    ) -> Dict[str, float]:
        updated_seed = dict(seed_map)
        if not has_nonempty_values(existing_row.get("params")):
            return updated_seed
        try:
            existing_params = np.asarray(
                existing_row.get("params"), dtype=float
            ).reshape(-1)
        except Exception:
            return updated_seed
        for idx, key in enumerate(self.ordered_param_keys):
            if idx >= existing_params.size:
                break
            value = float(existing_params[idx])
            if (not np.isfinite(value)) or key not in updated_seed:
                continue
            updated_seed[key] = value
        return updated_seed

    def _capture_seed_signature(
        self, captures: Mapping[str, Any]
    ) -> Optional[Tuple[Tuple[str, str], ...]]:
        if not self._capture_seed_keys:
            return None
        signature: List[Tuple[str, str]] = []
        for key in self._capture_seed_keys:
            value = captures.get(key)
            if value in (None, ""):
                return None
            signature.append((key, str(value)))
        return tuple(signature)

    @staticmethod
    def _capture_value_distance(left: Any, right: Any) -> float:
        left_num = _finite_float_or_none(left)
        right_num = _finite_float_or_none(right)
        if left_num is not None and right_num is not None:
            denom = abs(left_num) + abs(right_num) + 1.0
            return float(abs(left_num - right_num) / denom)
        return 0.0 if str(left) == str(right) else 1.0

    @staticmethod
    def _is_seed_candidate_good(row: Mapping[str, Any]) -> bool:
        if not has_nonempty_values(row.get("params")):
            return False
        return not _row_has_error(row)

    def _capture_distance(
        self,
        left_captures: Mapping[str, Any],
        right_captures: Mapping[str, Any],
    ) -> Optional[float]:
        if not self._capture_seed_keys:
            return None
        total = 0.0
        count = 0
        for key in self._capture_seed_keys:
            left_value = left_captures.get(key)
            right_value = right_captures.get(key)
            if left_value in (None, "") or right_value in (None, ""):
                total += 1.0
                count += 1
                continue
            total += self._capture_value_distance(left_value, right_value)
            count += 1
        if count <= 0:
            return None
        return float(total / float(count))

    def _best_matching_capture_seed_row(
        self, *, captures: Mapping[str, Any], exclude_file_path: str
    ) -> Tuple[Optional[Mapping[str, Any]], Optional[str], Optional[str]]:
        target_signature = self._capture_seed_signature(captures)
        best_row: Optional[Mapping[str, Any]] = None
        best_file: Optional[str] = None
        closest_row: Optional[Mapping[str, Any]] = None
        closest_file: Optional[str] = None
        closest_distance: Optional[float] = None
        exclude_key = str(exclude_file_path)
        for file_key, candidate_row in self.existing_rows_by_file.items():
            if str(file_key) == exclude_key:
                continue
            if not self._is_seed_candidate_good(candidate_row):
                continue
            candidate_captures = candidate_row.get("captures")
            if not isinstance(candidate_captures, Mapping):
                continue
            candidate_signature = self._capture_seed_signature(candidate_captures)
            if (
                target_signature is not None
                and candidate_signature is not None
                and candidate_signature == target_signature
            ):
                if best_row is None or is_fit_row_improved(candidate_row, best_row):
                    best_row = candidate_row
                    best_file = str(file_key)
                continue
            distance = self._capture_distance(captures, candidate_captures)
            if distance is None:
                continue
            if closest_row is None:
                closest_row = candidate_row
                closest_file = str(file_key)
                closest_distance = float(distance)
                continue
            if float(distance) < float(closest_distance):
                closest_row = candidate_row
                closest_file = str(file_key)
                closest_distance = float(distance)
                continue
            if np.isclose(
                float(distance), float(closest_distance)
            ) and is_fit_row_improved(candidate_row, closest_row):
                closest_row = candidate_row
                closest_file = str(file_key)
                closest_distance = float(distance)

        if best_row is not None:
            return best_row, "matching-captures", best_file
        if closest_row is not None:
            return closest_row, "closest-captures", closest_file
        return None, None, None

    def _boundary_seed_for_file(self, existing_row: Mapping[str, Any]) -> np.ndarray:
        candidate = np.asarray(self.boundary_seed, dtype=float).reshape(-1)
        if candidate.size <= 0:
            return candidate
        if not has_nonempty_values(existing_row.get("params")):
            return candidate
        existing_ratios = existing_row.get("boundary_ratios")
        if existing_ratios is None:
            return candidate
        try:
            existing_arr = np.asarray(existing_ratios, dtype=float).reshape(-1)
        except Exception:
            return candidate
        if existing_arr.size != candidate.size:
            return candidate
        if np.any(~np.isfinite(existing_arr)):
            return candidate
        return np.clip(existing_arr, 0.0, 1.0)

    def _randomized_seed_map(self, seed_map: Mapping[str, float]) -> Dict[str, float]:
        randomized = {str(key): float(val) for key, val in dict(seed_map or {}).items()}
        for key, bounds in self.bounds_map.items():
            try:
                low_raw, high_raw = bounds
                low = float(min(low_raw, high_raw))
                high = float(max(low_raw, high_raw))
            except Exception:
                continue
            if not np.isfinite(low) or not np.isfinite(high):
                continue
            if np.isclose(low, high):
                randomized[str(key)] = float(low)
                continue
            randomized[str(key)] = float(self._rng.uniform(low, high))
        return randomized

    def _randomized_boundary_seed(self, boundary_seed: np.ndarray) -> np.ndarray:
        base = np.asarray(boundary_seed, dtype=float).reshape(-1)
        if base.size <= 0:
            return base
        return np.asarray(self._rng.uniform(0.0, 1.0, size=base.size), dtype=float)

    def _seed_signature(
        self, seed_map: Mapping[str, float], boundary_seed: np.ndarray
    ) -> Tuple[float, ...]:
        values: List[float] = []
        for key in self.ordered_param_keys:
            try:
                values.append(float(seed_map.get(key, 0.0)))
            except Exception:
                values.append(0.0)
        boundary = np.asarray(boundary_seed, dtype=float).reshape(-1)
        if boundary.size > 0:
            values.extend(boundary.tolist())
        return tuple(round(float(v), 12) for v in values)

    def _fit_single_file(self, source_index, file_path):
        if self.cancel_requested:
            return None

        existing_row = self.existing_rows_by_file.get(str(file_path), {})
        existing_has_fit = has_nonempty_values(existing_row.get("params"))
        extracted = extract_captures(
            stem_for_file_ref(file_path),
            self.regex,
            self.capture_defaults,
        )
        captures = extracted if extracted is not None else {}
        pattern_error = _BATCH_PATTERN_MISMATCH_ERROR if extracted is None else None
        row = make_batch_result_row(
            source_index=source_index,
            file_path=file_path,
            x_channel=self.x_channel,
            y_channel=self.y_channel,
            captures=captures,
            pattern_error=pattern_error,
        )

        def _copy_existing_no_change_row():
            copied = self._copy_fit_fields_from_existing(row, existing_row)
            copied["_fit_no_change"] = True
            return copied

        try:
            data = read_measurement_csv(file_path)
            if self.x_channel not in data.columns:
                raise KeyError(f"Missing x channel '{self.x_channel}'.")
            if self.y_channel not in data.columns:
                raise KeyError(f"Missing y channel '{self.y_channel}'.")
            x_data = data[self.x_channel].to_numpy(dtype=float, copy=True)
            y_data = data[self.y_channel].to_numpy(dtype=float, copy=True)
            if self.smoothing_enabled:
                x_data = smooth_channel_array(x_data, self.smoothing_window)
                y_data = smooth_channel_array(y_data, self.smoothing_window)

            fixed_params, mapping_error = self._mapping_fixed_params(captures)
            if mapping_error:
                if existing_has_fit:
                    return _copy_existing_no_change_row()
                row["error"] = mapping_error
                return row

            if self.cancel_requested:
                return None

            attempt_inputs: List[
                Tuple[Dict[str, float], np.ndarray, str, Optional[str]]
            ] = []
            if self.use_existing_fit_seed:
                matching_seed_row, matching_seed_kind, matching_seed_file = (
                    self._best_matching_capture_seed_row(
                        captures=captures,
                        exclude_file_path=str(file_path),
                    )
                )

                derived_row = matching_seed_row
                derived_kind = str(matching_seed_kind or "")
                derived_file: Optional[str] = (
                    str(matching_seed_file)
                    if matching_seed_file not in (None, "")
                    else None
                )
                if derived_row is None and self._is_seed_candidate_good(existing_row):
                    derived_row = existing_row
                    derived_kind = "existing-row"
                    derived_file = str(file_path)

                if derived_row is not None:
                    derived_seed = self._seed_map_with_existing_fit(
                        self.seed_map, derived_row
                    )
                    derived_boundary = self._boundary_seed_for_file(derived_row)
                    attempt_inputs.append(
                        (
                            dict(derived_seed),
                            np.asarray(derived_boundary, dtype=float).reshape(-1),
                            (derived_kind or "matching-captures"),
                            derived_file,
                        )
                    )
                else:
                    attempt_inputs.append(
                        (
                            dict(self.seed_map),
                            np.asarray(self.boundary_seed, dtype=float).reshape(-1),
                            "default",
                            None,
                        )
                    )

                attempt_inputs.append(
                    (
                        self._randomized_seed_map(self.seed_map),
                        self._randomized_boundary_seed(self.boundary_seed),
                        "random-restart",
                        None,
                    )
                )
            else:
                attempt_inputs.append(
                    (
                        dict(self.seed_map),
                        np.asarray(self.boundary_seed, dtype=float).reshape(-1),
                        "default",
                        None,
                    )
                )

            deduped_attempts: List[
                Tuple[Dict[str, float], np.ndarray, str, Optional[str]]
            ] = []
            seen_signatures: Set[Tuple[float, ...]] = set()
            for (
                seed_candidate,
                boundary_candidate,
                seed_source,
                seed_source_file,
            ) in attempt_inputs:
                sig = self._seed_signature(seed_candidate, boundary_candidate)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                deduped_attempts.append(
                    (seed_candidate, boundary_candidate, seed_source, seed_source_file)
                )

            best_row = None
            last_error = None
            for (
                seed_candidate,
                boundary_candidate,
                seed_source,
                seed_source_file,
            ) in deduped_attempts:
                if self.cancel_requested:
                    return None
                seed_map = dict(seed_candidate)
                if fixed_params:
                    for key, value in fixed_params.items():
                        if key in seed_map:
                            seed_map[key] = float(value)

                try:
                    result = run_piecewise_fit_pipeline(
                        x_data,
                        y_data,
                        self.model_def,
                        seed_map,
                        self.bounds_map,
                        boundary_seed=boundary_candidate,
                        cancel_check=lambda: self.cancel_requested,
                        fixed_params=fixed_params,
                    )
                except FitCancelledError:
                    return None
                except Exception as exc:
                    last_error = str(exc)
                    continue

                candidate_row = dict(row)
                params_by_key = dict(result.get("params_by_key") or {})
                candidate_row["params"] = [
                    float(params_by_key.get(key, seed_map.get(key, 0.0)))
                    for key in self.ordered_param_keys
                ]
                candidate_row["r2"] = (
                    float(result["r2"]) if result.get("r2") is not None else None
                )
                boundary_values = result.get("boundary_ratios")
                if boundary_values is None:
                    boundary_values = []
                candidate_row["boundary_ratios"] = np.asarray(
                    boundary_values, dtype=float
                )
                n_boundaries = max(0, len(self.model_def.segment_exprs) - 1)
                candidate_row["boundary_values"] = boundary_ratios_to_x_values(
                    candidate_row["boundary_ratios"],
                    x_data,
                    n_boundaries,
                )
                candidate_row["error"] = None
                candidate_row["_seed_source"] = str(seed_source)
                candidate_row["_seed_source_file"] = seed_source_file
                if best_row is None or is_fit_row_improved(candidate_row, best_row):
                    best_row = candidate_row

            if best_row is None:
                if existing_has_fit:
                    return _copy_existing_no_change_row()
                row["error"] = str(last_error or "No fit result.")
                return row
            if existing_has_fit and not is_fit_row_improved(best_row, existing_row):
                return _copy_existing_no_change_row()
            return best_row
        except FitCancelledError:
            return None
        except Exception as exc:
            if existing_has_fit:
                return _copy_existing_no_change_row()
            row["error"] = str(exc)
            return row

    @pyqtSlot()
    def run(self):
        results = []
        try:
            total = len(self.file_paths)
            if total == 0:
                self.finished.emit([])
                return

            completed = 0
            for idx, file_path in enumerate(self.file_paths):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return
                source_index = (
                    self.source_indices[idx] if idx < len(self.source_indices) else idx
                )
                row = self._fit_single_file(source_index, file_path)
                if row is None:
                    if self.cancel_requested:
                        self.cancelled.emit()
                        return
                    continue
                results.append(row)
                completed += 1
                self.progress.emit(completed, total, row)

            if self.cancel_requested:
                self.cancelled.emit()
                return
            self.finished.emit(results)
        except Exception as exc:
            if self.cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))


class ThumbnailRenderWorker(QObject):
    progress = pyqtSignal(int, int, object)
    finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(
        self,
        batch_results,
        model_func,
        full_thumbnail_size=(468, 312),
        row_indices=None,
        smoothing_enabled=False,
        smoothing_window=1,
    ):
        super().__init__()
        self.batch_results = batch_results
        self.model_func = model_func
        self.full_thumbnail_size = full_thumbnail_size
        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoothing_window = int(smoothing_window)
        if row_indices is None:
            self.row_indices = list(range(len(batch_results)))
        else:
            ordered = []
            seen = set()
            for idx in row_indices:
                try:
                    row_idx = int(idx)
                except Exception:
                    continue
                if 0 <= row_idx < len(batch_results) and row_idx not in seen:
                    ordered.append(row_idx)
                    seen.add(row_idx)
            self.row_indices = ordered
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    def _row_render_size(self, row):
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

    def _needs_render(self, row):
        source = row.get("plot_full") or row.get("plot")
        if source is None:
            return True
        rendered_size = self._row_render_size(row)
        if rendered_size is None:
            return True
        target_size = (
            int(self.full_thumbnail_size[0]),
            int(self.full_thumbnail_size[1]),
        )
        return tuple(rendered_size) != target_size

    @pyqtSlot()
    def run(self):
        try:
            total = len(self.row_indices)
            for done_idx, row_idx in enumerate(self.row_indices):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return

                row = self.batch_results[row_idx]
                if not self._needs_render(row):
                    self.progress.emit(done_idx + 1, total, row_idx)
                    continue

                pixmap = self.render_thumbnail(row)
                row["plot_full"] = pixmap
                row["plot_has_fit"] = has_nonempty_values(row.get("params"))
                row["plot_render_size"] = (
                    int(self.full_thumbnail_size[0]),
                    int(self.full_thumbnail_size[1]),
                )
                self.progress.emit(done_idx + 1, total, row_idx)

            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit()
        except Exception:
            self.finished.emit()

    def render_thumbnail(self, row):
        """Render a plot to a QPixmap for embedding in table."""
        return render_batch_thumbnail(
            row,
            self.model_func,
            full_thumbnail_size=self.full_thumbnail_size,
            smoothing_enabled=self.smoothing_enabled,
            smoothing_window=self.smoothing_window,
        )


class ManualFitGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.param_specs = list(DEFAULT_PARAM_SPECS)
        self.param_spinboxes = {}
        self.param_sliders = {}
        self.param_min_spinboxes = {}
        self.param_max_spinboxes = {}
        self.param_lock_status_labels = {}
        self.param_tail_spacers_by_key = {}
        self.breakpoint_controls = {}
        self.param_row_tail_spacers = []
        self._param_slider_steps = 2000
        self._param_name_width = 88
        self._param_bound_width = 72
        self._param_value_width = 78
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

        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.setGeometry(100, 100, 900, 900)

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
        self.y_channel = "CH2"
        self.last_popt = None
        self.last_pcov = None
        self.last_r2 = None
        self.auto_fit_btn_default_text = "Auto Fit"
        self.current_boundary_ratios = default_boundary_ratios(
            max(0, len(self._piecewise_model.segment_exprs) - 1)
            if self._piecewise_model is not None
            else 2
        )
        self._last_r2 = None
        self.param_capture_map = {}
        self.param_capture_combos = {}
        self.fit_thread = None
        self.fit_worker = None
        self.batch_thread = None
        self.batch_worker = None
        self.fit_tasks = {}
        self._fit_task_counter = 0
        self._fit_max_concurrent = max(1, int((os.cpu_count() or 2) - 1))
        self._batch_refit_random_restarts = 2
        self._pending_fit_task_ids = deque()
        self._batch_active_task_ids = set()
        self._batch_total_tasks = 0
        self._batch_cancel_requested = False
        self._manual_active_task_ids = set()
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
        self._analysis_point_pick_cid = None
        self._analysis_scatter_files = {}
        self.max_thumbnails = 8
        self.thumb_cols = 1
        self.batch_row_height = 64
        self.batch_row_height_min = 40
        self.batch_row_height_max = 320
        self.batch_thumbnail_aspect = 1.5
        self.batch_thumbnail_supersample = 2.0
        self._batch_row_height_sync = False
        self._batch_progress_done = 0
        self.regex_timer = QTimer()
        self.regex_timer.setSingleShot(True)
        self.regex_timer.timeout.connect(self._do_prepare_batch_preview)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        self._pending_thumbnail_rows = set()
        self._batch_preview_ready = False
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # Parameter initial values default to ParameterSpec.default (clipped to bounds).
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
        self._last_file_load_error = ""
        self.smoothing_enabled = True
        self.smoothing_window = 101

        # Current directory
        self.current_dir = "./AFG_measurements/"
        self._source_display_override = None
        self._source_selected_paths = []
        self._fit_details_restore_in_progress = False

        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("root")
        self.setCentralWidget(central_widget)
        self._enforce_light_mode()
        self._apply_compact_ui_defaults()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Shared controls/parameters/results for all modes.
        self.create_parameters_frame(main_layout)
        self._refresh_fit_action_buttons()
        self._refresh_batch_controls()

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        if hasattr(self, "stats_text"):
            main_layout.addWidget(self.stats_text)

        self.manual_tab = QWidget()
        self.batch_tab = QWidget()
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.manual_tab, "Plot")
        self.tabs.addTab(self.batch_tab, "Batch Processing")
        self.tabs.addTab(self.analysis_tab, "Batch Analysis")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._attach_tab_corner_controls()

        manual_layout = QVBoxLayout(self.manual_tab)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(4)

        # Manual mode: interactive plot only (controls are shared above tabs).
        self.create_plot_frame(manual_layout)

        batch_layout = QVBoxLayout(self.batch_tab)
        batch_layout.setContentsMargins(6, 6, 6, 6)
        batch_layout.setSpacing(6)
        self.create_batch_results_frame(batch_layout)

        analysis_layout = QVBoxLayout(self.analysis_tab)
        analysis_layout.setContentsMargins(6, 6, 6, 6)
        analysis_layout.setSpacing(6)
        self.create_batch_analysis_frame(analysis_layout)
        analysis_layout.addStretch()

        # Defer initial file discovery/load until the UI has painted once.
        self.stats_text.setText("Loading data sources...")
        QTimer.singleShot(0, self.load_files)

    def _on_tab_changed(self, _index):
        current_widget = self.tabs.currentWidget() if hasattr(self, "tabs") else None
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
        self.tab_r2_label = self._new_label(
            "R²: N/A",
            object_name="statusLabel",
            style_sheet="font-weight: 600; color: #334155; padding: 1px 2px;",
        )
        corner_layout.addWidget(self.tab_r2_label)
        self.tabs.setCornerWidget(corner_widget, Qt.Corner.TopRightCorner)
        self.tab_corner_controls = corner_widget
        self.tab_r2_label.setVisible(self.tabs.currentWidget() is self.manual_tab)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_source_path_label()
        QTimer.singleShot(0, self._sync_fit_panel_top_spacing)

    def eventFilter(self, watched, event):
        if (
            watched is getattr(self, "param_header_widget", None)
            and event is not None
            and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)
        ):
            QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
        if event is not None and event.type() == QEvent.Type.MouseButtonPress:
            self._defocus_numeric_editor_on_outside_click(watched, event)
        if (
            self._expression_edit_mode
            and event is not None
            and event.type() == QEvent.Type.MouseButtonPress
            and QApplication.activePopupWidget() is None
        ):
            active_modal = QApplication.activeModalWidget()
            if active_modal is not None and active_modal is not self:
                return super().eventFilter(watched, event)
            clicked_widget = watched if isinstance(watched, QWidget) else None
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
                widget_at_pos = QApplication.widgetAt(global_pos)
                if widget_at_pos is not None:
                    clicked_widget = widget_at_pos
            if not self._is_expression_editor_child(clicked_widget):
                QTimer.singleShot(
                    0, lambda: self._apply_expression_on_focus_leave(force=True)
                )
        return super().eventFilter(watched, event)

    def _defocus_numeric_editor_on_outside_click(self, watched, event):
        active_popup = QApplication.activePopupWidget()
        if active_popup is not None:
            return

        focused_widget = QApplication.focusWidget()
        spinbox = None
        probe = focused_widget
        while isinstance(probe, QWidget):
            if isinstance(probe, QAbstractSpinBox):
                spinbox = probe
                break
            probe = probe.parentWidget()

        if spinbox is None:
            return
        if not bool(spinbox.property("defocus_on_outside_click")):
            return

        clicked_widget = watched if isinstance(watched, QWidget) else None
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
            widget_at_pos = QApplication.widgetAt(global_pos)
            if widget_at_pos is not None:
                clicked_widget = widget_at_pos

        if clicked_widget is not None and (
            clicked_widget is spinbox or spinbox.isAncestorOf(clicked_widget)
        ):
            return

        if focused_widget is not None:
            focused_widget.clearFocus()
        spinbox.clearFocus()

    def _enforce_light_mode(self):
        """Force a light Qt palette regardless of system appearance."""
        app = QApplication.instance()
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
        combo = RichTextComboBox() if rich_text else QComboBox()
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
            index = combo.findData(current_data)
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

    def _refresh_source_path_label(self):
        if not hasattr(self, "source_path_label"):
            return
        source_text = (
            str(self._source_display_override).strip()
            if self._source_display_override
            else str(self.current_dir).strip()
        ) or "."
        prefix = "📁 "
        max_width = max(220, int(self.width() * 0.4))
        self.source_path_label.setMaximumWidth(max_width)
        metrics = self.source_path_label.fontMetrics()
        available_for_path = max(24, max_width - metrics.horizontalAdvance(prefix))
        elided_path = metrics.elidedText(
            source_text,
            Qt.TextElideMode.ElideLeft,
            available_for_path,
        )
        self.source_path_label.setText(f"{prefix}{elided_path}")
        tooltip = [f"Current data source:\n{source_text}"]
        selected_paths = list(getattr(self, "_source_selected_paths", []) or [])
        if selected_paths:
            preview = "\n".join(stem_for_file_ref(path) for path in selected_paths[:12])
            remaining = len(selected_paths) - 12
            if remaining > 0:
                preview += f"\n... +{remaining} more"
            tooltip.append(f"Selected files ({len(selected_paths)}):\n{preview}")
        tooltip.append(
            "Click to choose a folder, a ZIP archive, or one/more CSV files."
        )
        self.source_path_label.setToolTip("\n\n".join(tooltip))

    def _sync_file_navigation_buttons(self):
        total = len(getattr(self, "data_files", []) or [])
        current_idx = int(getattr(self, "current_file_idx", 0))
        at_start = current_idx <= 0
        at_end = current_idx >= max(0, total - 1)
        busy = bool(getattr(self, "_file_load_in_progress", False))

        if hasattr(self, "prev_file_btn"):
            self.prev_file_btn.setEnabled((total > 0) and (not at_start) and (not busy))
        if hasattr(self, "next_file_btn"):
            self.next_file_btn.setEnabled((total > 0) and (not at_end) and (not busy))

    def _current_loaded_file_path(self):
        files = list(getattr(self, "data_files", []) or [])
        idx = int(getattr(self, "current_file_idx", -1))
        if 0 <= idx < len(files):
            return files[idx]
        return None

    def _fit_task_file_key(self, file_path):
        text = str(file_path or "").strip()
        if not text:
            return ""
        if "::" in text:
            archive_text, member = text.split("::", 1)
            archive = Path(archive_text).expanduser()
            try:
                archive_key = str(archive.resolve(strict=False))
            except Exception:
                archive_key = str(archive)
            member_key = str(member).strip().replace("\\", "/")
            return f"{archive_key}::{member_key}"
        path_obj = Path(text).expanduser()
        try:
            return str(path_obj.resolve(strict=False))
        except Exception:
            return str(path_obj)

    def _next_fit_task_id(self):
        self._fit_task_counter = int(getattr(self, "_fit_task_counter", 0)) + 1
        return int(self._fit_task_counter)

    def _running_fit_task_count(self):
        return sum(
            1
            for meta in self.fit_tasks.values()
            if str(meta.get("status")) == "running"
        )

    def _schedule_fit_tasks(self):
        max_running = max(1, int(getattr(self, "_fit_max_concurrent", 1)))
        while (
            self._running_fit_task_count() < max_running
            and len(self._pending_fit_task_ids) > 0
        ):
            try:
                task_id = int(self._pending_fit_task_ids.popleft())
            except Exception:
                continue
            task = self.fit_tasks.get(task_id)
            if task is None:
                continue
            if str(task.get("status")) != "pending":
                continue
            task["status"] = "running"
            if str(task.get("kind")) == "batch":
                self._set_batch_row_runtime_fields(
                    task.get("file_path"),
                    _fit_status="Running",
                )
            thread = task.get("thread")
            if thread is not None:
                thread.start()

    def _active_fit_tasks_for_file(self, file_path):
        target = self._fit_task_file_key(file_path)
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
        current_file = self._current_loaded_file_path()
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
        if hasattr(self, "reset_from_batch_btn"):
            self.reset_from_batch_btn.setEnabled(not any_running)

    def _refresh_batch_controls(self):
        if not hasattr(self, "run_batch_btn") or not hasattr(self, "cancel_batch_btn"):
            return
        if not bool(self.batch_fit_in_progress):
            self.run_batch_btn.setEnabled(True)
            self.run_batch_btn.setText(self.run_batch_btn_default_text)
            self.cancel_batch_btn.setEnabled(False)
            self.cancel_batch_btn.setText("Cancel")
            return
        total = max(0, int(getattr(self, "_batch_total_tasks", 0)))
        done = max(0, int(getattr(self, "_batch_progress_done", 0)))
        self.run_batch_btn.setEnabled(False)
        self.run_batch_btn.setText(f"Run Batch ({done}/{total})")
        self.cancel_batch_btn.setEnabled(True)
        self.cancel_batch_btn.setText(
            "Force Stop"
            if bool(getattr(self, "_batch_cancel_pending", False))
            else "Cancel"
        )

    def _default_param_midpoints(self, specs):
        midpoints = []
        for spec in specs:
            low = float(spec.min_value)
            high = float(spec.max_value)
            if low > high:
                low, high = high, low
            midpoints.append((low + high) * 0.5)
        return midpoints

    def _default_param_values(self, specs):
        defaults = []
        for spec in specs:
            low = float(spec.min_value)
            high = float(spec.max_value)
            if low > high:
                low, high = high, low
            defaults.append(float(np.clip(float(spec.default), low, high)))
        return defaults

    def _apply_param_spec_defaults_to_controls(self):
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
    ):
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

    def _new_compact_float_spinbox(
        self,
        minimum,
        maximum,
        value,
        *,
        decimals=3,
        single_step=None,
        tooltip=None,
    ):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(float(minimum), float(maximum))
        spinbox.setDecimals(int(decimals))
        if single_step is not None:
            spinbox.setSingleStep(float(single_step))
        spinbox.setValue(float(value))
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(False)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        spinbox.setFixedWidth(56)
        spinbox.setProperty("defocus_on_outside_click", True)
        if tooltip:
            spinbox.setToolTip(str(tooltip))
        return spinbox

    def _build_fit_option_cell(self, label_widget, value_widget):
        cell = QWidget()
        cell_layout = QHBoxLayout(cell)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(4)
        if isinstance(label_widget, QLabel):
            label_widget.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            label_widget.setFixedWidth(int(self._fit_option_label_width))
        cell_layout.addWidget(label_widget)
        cell_layout.addWidget(value_widget)
        cell_layout.addStretch(1)
        return cell

    def _new_compact_param_spinbox(
        self,
        spec,
        value,
        *,
        minimum=None,
        maximum=None,
        width=72,
        object_name=None,
        tooltip=None,
    ):
        spinbox = CompactDoubleSpinBox()
        if object_name:
            spinbox.setObjectName(str(object_name))
        if minimum is None:
            minimum = float(spec.min_value)
        if maximum is None:
            maximum = float(spec.max_value)
        low = float(min(minimum, maximum))
        high = float(max(minimum, maximum))
        spinbox.setDecimals(int(spec.decimals))
        spinbox.setRange(low, high)
        spinbox.setSingleStep(float(spec.inferred_step))
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(False)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        spinbox.setFixedWidth(int(width))
        spinbox.setProperty("defocus_on_outside_click", True)
        if tooltip:
            spinbox.setToolTip(str(tooltip))
        spinbox.setValue(float(np.clip(float(value), low, high)))
        return spinbox

    def _effective_smoothing_window(self):
        window = int(getattr(self, "smoothing_window", 1))
        if window <= 1:
            return 1
        if window % 2 == 0:
            window += 1
        return window

    def _smooth_channel_values(self, values):
        if not bool(getattr(self, "smoothing_enabled", False)):
            return np.asarray(values, dtype=float)
        return smooth_channel_array(values, self._effective_smoothing_window())

    def _rebuild_channel_cache_from_raw(self):
        rebuilt = {}
        for key, values in (self.raw_channel_cache or {}).items():
            try:
                rebuilt[str(key)] = self._smooth_channel_values(values)
            except Exception:
                continue
        self.channel_cache = rebuilt
        self._expression_channel_data_cache = dict(rebuilt)

    def _sync_smoothing_window_enabled(self):
        spin = getattr(self, "smoothing_window_spin", None)
        if spin is None:
            return
        toggle = getattr(self, "smoothing_toggle_btn", None)
        if toggle is None:
            toggle = getattr(self, "smoothing_enable_cb", None)
        enabled = bool(toggle and toggle.isChecked())
        spin.setEnabled(enabled)

    def _on_smoothing_controls_changed(self):
        toggle = getattr(self, "smoothing_toggle_btn", None)
        if toggle is None:
            toggle = getattr(self, "smoothing_enable_cb", None)
        enabled = bool(toggle and toggle.isChecked())
        self._sync_smoothing_window_enabled()
        window = (
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

        changed = (enabled != self.smoothing_enabled) or (
            window != self.smoothing_window
        )
        self.smoothing_enabled = enabled
        self.smoothing_window = window

        if not changed:
            return

        main_xlim = None
        main_ylim = None
        residual_ylim = None
        if hasattr(self, "ax") and self.ax is not None:
            try:
                main_xlim = tuple(self.ax.get_xlim())
            except Exception:
                main_xlim = None
            try:
                main_ylim = tuple(self.ax.get_ylim())
            except Exception:
                main_ylim = None
        if hasattr(self, "ax_residual") and self.ax_residual is not None:
            try:
                residual_ylim = tuple(self.ax_residual.get_ylim())
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

    def _sync_fit_panel_top_spacing(self):
        spacer = getattr(self, "fit_panel_top_spacer", None)
        header_widget = getattr(self, "param_header_widget", None)
        if spacer is None or header_widget is None:
            return
        header_height = 0
        try:
            header_height = max(
                int(header_widget.minimumSizeHint().height()),
                int(header_widget.sizeHint().height()),
            )
        except Exception:
            header_height = 0
        header_height = max(0, int(header_height))
        # Keep the header row from stretching vertically and creating layout feedback.
        if (
            header_widget.minimumHeight() != header_height
            or header_widget.maximumHeight() != header_height
        ):
            header_widget.setFixedHeight(header_height)
        target_height = (
            int(getattr(self, "_param_header_to_rows_gap", 0)) + header_height
        )
        spacer.setFixedHeight(max(0, target_height))
        self._sync_param_pane_height()

    def _sync_param_pane_height(self):
        scroll = getattr(self, "param_controls_scroll", None)
        fit_panel = getattr(self, "fit_panel_widget", None)
        if scroll is None or fit_panel is None:
            return
        fit_height = 0
        try:
            fit_height = max(
                int(fit_panel.height()),
                int(fit_panel.minimumSizeHint().height()),
                int(fit_panel.sizeHint().height()),
            )
        except Exception:
            fit_height = int(getattr(fit_panel, "height", lambda: 0)())
        fit_height = max(0, int(fit_height))
        if fit_height <= 0:
            return
        if scroll.minimumHeight() != fit_height or scroll.maximumHeight() != fit_height:
            scroll.setFixedHeight(fit_height)

    def _make_param_header_label(self, text, width=None):
        return self._new_label(
            text,
            object_name="paramHeader",
            width=width,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    def _display_symbol_for_param(self, key, symbol_hint=None):
        return resolve_parameter_symbol(key, symbol_hint)

    def _display_symbol_for_param_html(self, key, symbol_hint=None):
        return parameter_symbol_to_html(
            self._display_symbol_for_param(key, symbol_hint)
        )

    def _parameter_symbol_map(self):
        mapping = {}
        for spec in self.param_specs:
            mapping[spec.key] = self._display_symbol_for_param(spec.key, spec.symbol)
        return mapping

    def _display_name_for_param_key(self, key):
        for spec in self.param_specs:
            if spec.key == key:
                return self._display_symbol_for_param(spec.key, spec.symbol)
        return self._display_symbol_for_param(key, key)

    def _display_name_for_param_key_mathtext(self, key):
        return parameter_symbol_to_mathtext(self._display_name_for_param_key(key))

    def _ordered_parameter_sections(self):
        ordered_keys = self._ordered_param_keys()
        model_def = self._piecewise_model
        if (
            model_def is None
            or not getattr(model_def, "segment_param_names", None)
            or len(model_def.segment_param_names) == 0
        ):
            return [{"kind": "segment", "index": 1, "keys": list(ordered_keys)}]

        sections = []
        seen = set()
        segment_param_names = list(model_def.segment_param_names)
        total_segments = len(segment_param_names)
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

        trailing = [key for key in ordered_keys if key not in seen]
        if trailing:
            sections.append({"kind": "shared", "index": 0, "keys": trailing})
        return sections

    def _build_param_section_header(self, title, tooltip=""):
        label = self._new_label(
            str(title),
            object_name="statusLabel",
            tooltip=tooltip,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            style_sheet="font-weight: 700; color: #0f172a; padding: 2px 0 0 0;",
        )
        return label

    def _build_param_boundary_marker(self, boundary_index):
        _ = boundary_index
        divider = QWidget()
        divider.setFixedHeight(8)
        divider.setStyleSheet("border-top: 1px solid #cbd5e1;")
        return divider

    def _build_top_breakpoint_controls_widget(self):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        container.setMinimumWidth(320)

        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        min_label = self._new_label(
            "Start",
            object_name="paramInline",
            width=56,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            style_sheet="color: #64748b;",
        )
        row.addWidget(min_label)

        slider = MultiHandleSlider()
        slider.valuesChanged.connect(self._on_breakpoint_values_changed)
        slider.sliderPressed.connect(self._on_breakpoint_slider_pressed)
        slider.sliderReleased.connect(self._on_breakpoint_slider_released)
        row.addWidget(slider, 1)

        max_label = self._new_label(
            "End",
            object_name="paramInline",
            width=56,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            style_sheet="color: #64748b;",
        )
        row.addWidget(max_label)
        outer.addLayout(row)

        self.breakpoint_controls = {
            "slider": slider,
            "container": container,
        }
        return container

    def _current_segment_boundary_count(self):
        model_def = getattr(self, "_piecewise_model", None)
        if model_def is None:
            return 0
        return max(0, len(model_def.segment_exprs) - 1)

    def _boundary_ratios_to_positions(self, ratios, n_boundaries):
        return boundary_ratios_to_positions(ratios, n_boundaries)

    def _boundary_positions_to_ratios(self, positions, n_boundaries):
        n = int(max(0, n_boundaries))
        if n <= 0:
            return np.asarray([], dtype=float)
        pos_arr = np.asarray(positions, dtype=float).reshape(-1)
        if pos_arr.size != n:
            return default_boundary_ratios(n)
        pos_arr = np.clip(pos_arr, 0.0, 1.0)
        pos_arr = np.maximum.accumulate(pos_arr)
        ratios = np.empty(n, dtype=float)
        prev = 0.0
        for idx, position in enumerate(pos_arr):
            denom = max(1.0 - prev, 1e-12)
            ratios[idx] = float(np.clip((float(position) - prev) / denom, 0.0, 1.0))
            prev = float(position)
        return ratios

    def _x_axis_range_for_boundary_controls(self):
        if self.current_data is None:
            return (0.0, 1.0)
        try:
            x_values = np.asarray(
                self._get_channel_data(self.x_channel), dtype=float
            ).reshape(-1)
        except Exception:
            return (0.0, 1.0)
        finite = x_values[np.isfinite(x_values)]
        if finite.size == 0:
            return (0.0, 1.0)
        x_min = float(np.min(finite))
        x_max = float(np.max(finite))
        if np.isclose(x_min, x_max):
            x_max = x_min + 1.0
        return (x_min, x_max)

    def _format_compact_number(self, value):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "n/a"
        magnitude = abs(numeric)
        if magnitude >= 1e4 or (magnitude > 0.0 and magnitude < 1e-3):
            return f"{numeric:.3e}"
        return f"{numeric:.6g}"

    def _sync_breakpoint_sliders_from_state(self):
        n_boundaries = self._current_segment_boundary_count()
        control = (
            self.breakpoint_controls
            if isinstance(self.breakpoint_controls, dict)
            else {}
        )
        slider = control.get("slider")

        if n_boundaries <= 0:
            self.current_boundary_ratios = np.asarray([], dtype=float)
            if slider is not None:
                slider.blockSignals(True)
                slider.set_values([])
                slider.blockSignals(False)
                slider.setEnabled(False)
            return

        ratios = np.asarray(
            getattr(
                self, "current_boundary_ratios", default_boundary_ratios(n_boundaries)
            ),
            dtype=float,
        ).reshape(-1)
        if ratios.size != n_boundaries:
            ratios = default_boundary_ratios(n_boundaries)
        ratios = np.clip(ratios, 0.0, 1.0)
        self.current_boundary_ratios = ratios
        positions = self._boundary_ratios_to_positions(ratios, n_boundaries)
        x_min, x_max = self._x_axis_range_for_boundary_controls()
        axis_label = self._channel_axis_label(self.x_channel)
        if slider is not None:
            slider.blockSignals(True)
            slider.set_values(positions.tolist())
            slider.set_labels(
                [_format_boundary_display_name(idx) for idx in range(n_boundaries)]
            )
            slider.blockSignals(False)
            slider.setEnabled(True)
            slider.setToolTip(f"Boundary positions X₀, X₁, ... on {axis_label}.")
        if hasattr(self, "formula_label") and not getattr(
            self, "_expression_edit_mode", False
        ):
            self._set_formula_label()

    def _on_breakpoint_values_changed(self, positions):
        n_boundaries = self._current_segment_boundary_count()
        if n_boundaries <= 0:
            return

        pos_arr = np.asarray(positions, dtype=float).reshape(-1)
        if pos_arr.size != n_boundaries:
            pos_arr = self._boundary_ratios_to_positions(
                getattr(
                    self,
                    "current_boundary_ratios",
                    default_boundary_ratios(n_boundaries),
                ),
                n_boundaries,
            )
        pos_arr = np.clip(pos_arr, 0.0, 1.0)
        pos_arr = np.maximum.accumulate(pos_arr)
        self.current_boundary_ratios = self._boundary_positions_to_ratios(
            pos_arr, n_boundaries
        )
        self._sync_breakpoint_sliders_from_state()
        self.update_plot(fast=True)

    def _on_breakpoint_slider_pressed(self):
        self.slider_active = True

    def _on_breakpoint_slider_released(self):
        self.slider_active = False
        self.do_full_update()
        self._autosave_fit_details()

    def _create_param_label(self, spec, width):
        """Create a one-line parameter label."""
        symbol_text = self._display_symbol_for_param(spec.key, spec.symbol)
        symbol_html = self._display_symbol_for_param_html(spec.key, spec.symbol)
        tooltip = str(spec.description)
        if symbol_text != spec.key:
            tooltip = f"{tooltip} ({spec.key})"
        label = self._new_label(
            f"{symbol_html}:",
            object_name="paramInline",
            tooltip=tooltip,
            width=width,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    def _ordered_param_keys(self):
        return [spec.key for spec in self.param_specs]

    def _channel_display_name(self, channel_name):
        key = str(channel_name)
        alias = str(self.channels.get(key, "")).strip()
        return alias if alias else key

    def _channel_unit(self, channel_name):
        key = str(channel_name)
        unit = str(getattr(self, "channel_units", {}).get(key, "")).strip()
        return unit

    def _channel_axis_label(self, channel_name):
        base = self._channel_display_name(channel_name)
        unit = self._channel_unit(channel_name)
        return f"{base} [{unit}]" if unit else base

    def _channel_legend_label(self, channel_name):
        key = str(channel_name)
        base = self._channel_display_name(key)
        if base and base != key:
            text = f"{key} ({base})"
        else:
            text = key
        unit = self._channel_unit(key)
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
        config = self._resolve_batch_capture_config(show_errors=False)
        if config is not None and config.regex is not None:
            for key in config.regex.groupindex.keys():
                text = str(key).strip()
                if text and text not in keys:
                    keys.append(text)
        return keys

    def _capture_preview_values(self):
        config = self._resolve_batch_capture_config(show_errors=False)
        if config is None or config.regex is None:
            return {}

        file_path = self._current_loaded_file_path()
        if not file_path:
            candidates = list(getattr(self, "batch_files", []) or [])
            if not candidates:
                candidates = list(getattr(self, "data_files", []) or [])
            file_path = candidates[0] if candidates else None
        if not file_path:
            return {}

        extracted = extract_captures(
            stem_for_file_ref(file_path),
            config.regex,
            config.defaults,
        )
        if extracted is None:
            return {}
        return {str(key): str(value) for key, value in dict(extracted).items()}

    def _on_param_capture_mapping_changed(self, capture_key, _index):
        combo = self.param_capture_combos.get(capture_key)
        if combo is None:
            return
        selected = combo.currentData()
        self.param_capture_map[str(capture_key)] = (
            str(selected) if selected not in (None, "") else None
        )
        self._sync_param_slider_lock_state()
        self._autosave_fit_details()

    def _locked_param_keys_from_capture_mapping(self):
        mapping = self._current_param_capture_map()
        locked = set()
        for key, capture_key in mapping.items():
            if capture_key not in (None, ""):
                locked.add(str(key))
        return locked

    def _parsed_numeric_param_values_from_mapping(self):
        preview_values = self._capture_preview_values()
        mapping = self._current_param_capture_map()
        out: Dict[str, float] = {}
        for param_key, capture_key in mapping.items():
            if capture_key in (None, ""):
                continue
            raw_value = preview_values.get(str(capture_key))
            text = str(raw_value).strip() if raw_value is not None else ""
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

    def _sync_param_slider_lock_state(self):
        param_capture_map = self._current_param_capture_map()
        mapped_values = self._parsed_numeric_param_values_from_mapping()
        any_value_changed = False
        for key, slider in self.param_sliders.items():
            if slider is None:
                continue
            capture_key = param_capture_map.get(str(key))
            is_locked = capture_key not in (None, "")
            min_box = self.param_min_spinboxes.get(key)
            max_box = self.param_max_spinboxes.get(key)
            value_box = self.param_spinboxes.get(key)
            lock_status_label = self.param_lock_status_labels.get(key)
            tail_spacer = self.param_tail_spacers_by_key.get(key)

            for widget in (slider, min_box, max_box, value_box, tail_spacer):
                if widget is None:
                    continue
                widget.setVisible(not is_locked)
                widget.setEnabled(not is_locked)

            if is_locked and str(key) in mapped_values and value_box is not None:
                locked_value = float(mapped_values[str(key)])
                low = float(value_box.minimum())
                high = float(value_box.maximum())
                if not np.isclose(float(value_box.value()), locked_value):
                    value_box.blockSignals(True)
                    value_box.setValue(float(np.clip(locked_value, low, high)))
                    value_box.blockSignals(False)
                    any_value_changed = True
            if is_locked:
                if lock_status_label is not None:
                    lock_status_label.setText(f'Bound to field "{capture_key}"')
                    lock_status_label.setToolTip(
                        f'Parameter is fixed from filename field "{capture_key}".'
                    )
                    lock_status_label.show()
            else:
                slider.setToolTip("Sweep value across active bounds")
                if lock_status_label is not None:
                    lock_status_label.hide()
                if value_box is not None:
                    value_box.setToolTip("Current value")
                if min_box is not None:
                    min_box.setToolTip("Lower bound")
                if max_box is not None:
                    max_box.setToolTip("Upper bound")
        if any_value_changed:
            self.update_plot(fast=False)

    def _parameter_display_items(self):
        items = []
        seen = {}
        for param_key in self._ordered_param_keys():
            symbol_token = str(self._display_name_for_param_key(param_key)).strip()
            if not symbol_token:
                symbol_token = str(param_key)
            count = int(seen.get(symbol_token, 0)) + 1
            seen[symbol_token] = count
            plain_label = symbol_token if count == 1 else f"{symbol_token} {count}"
            rich_base = parameter_symbol_to_html(symbol_token) or html.escape(
                symbol_token
            )
            rich_label = (
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

    def _refresh_param_capture_mapping_controls(self):
        if not hasattr(self, "capture_mapping_layout"):
            return
        clear_layout(self.capture_mapping_layout)
        self.param_capture_combos = {}

        capture_keys = self._available_capture_keys()
        param_keys = self._ordered_param_keys()
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
                2,
            )
            self.param_capture_map = {}
            self._sync_param_slider_lock_state()
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
            mapped = self.param_capture_map.get(capture_key)
            if mapped not in param_keys:
                mapped = None
            next_map[capture_key] = mapped

            label = self._new_label(
                str(capture_key),
                object_name="paramInline",
                tooltip=f"Filename field '{capture_key}'",
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            combo = self._new_combobox(minimum_width=150, rich_text=True)
            if isinstance(combo, RichTextComboBox):
                combo.add_rich_item("Unbound", None, "Unbound")
                for item in parameter_items:
                    combo.add_rich_item(item["plain"], item["key"], item["html"])
            else:
                combo.addItem("Unbound", None)
                for item in parameter_items:
                    combo.addItem(str(item["plain"]), item["key"])
            target_idx = combo.findData(mapped)
            if target_idx < 0:
                target_idx = 0
            combo.setCurrentIndex(target_idx)
            combo.currentIndexChanged.connect(
                lambda index, key=capture_key: self._on_param_capture_mapping_changed(
                    key, index
                )
            )
            value_text = str(preview_values.get(str(capture_key), "")).strip()
            value_label = self._new_label(
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
        self._sync_param_slider_lock_state()

    def _current_param_capture_map(self):
        mapping = {key: None for key in self._ordered_param_keys()}
        for capture_key, param_key in self.param_capture_map.items():
            target = str(param_key) if param_key not in (None, "") else None
            if target in mapping:
                mapping[target] = str(capture_key)
        return mapping

    def _current_file_fixed_params_from_mapping(
        self,
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        parameter_capture_map = self._current_param_capture_map()
        if not any(value not in (None, "") for value in parameter_capture_map.values()):
            return {}, None

        capture_config = self._resolve_batch_capture_config(show_errors=True)
        if capture_config is None:
            return None, "Capture pattern is invalid."
        if capture_config.regex is None:
            return (
                None,
                "Pattern is required when field-to-parameter mappings are used.",
            )

        file_path = self._current_loaded_file_path()
        if not file_path:
            return None, "No current file loaded for field-to-parameter mapping."

        extracted = extract_captures(
            stem_for_file_ref(file_path),
            capture_config.regex,
            capture_config.defaults,
        )
        if extracted is None:
            return None, _BATCH_PATTERN_MISMATCH_ERROR
        return resolve_fixed_params_from_captures(parameter_capture_map, extracted)

    def _available_channel_names(self):
        names = []
        if self.current_data is not None:
            for col in self.current_data.columns:
                text = str(col).strip()
                if text and text not in names:
                    names.append(text)
        else:
            for key in sorted(self.channels.keys()):
                if key not in names:
                    names.append(key)
        return names

    def _expression_reserved_names(self):
        reserved = set(_EXPRESSION_HELPER_NAMES) | {"np"}
        for name in self._available_channel_names():
            if _PARAMETER_NAME_RE.fullmatch(name):
                reserved.add(name)
        return reserved

    def _expression_channel_data(self):
        if self.current_data is None:
            self._expression_channel_data_cache = None
            return {}
        if self._expression_channel_data_cache is not None:
            return self._expression_channel_data_cache

        channels = {}
        for col in self.current_data.columns:
            key = str(col).strip()
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

    def get_fit_parameter_keys(self):
        return [spec.key for spec in self.param_specs]

    def _value_to_slider_position(self, key, value):
        min_box = self.param_min_spinboxes.get(key)
        max_box = self.param_max_spinboxes.get(key)
        if min_box is None or max_box is None:
            return 0
        low = float(min_box.value())
        high = float(max_box.value())
        if np.isclose(low, high):
            return 0
        ratio = (float(value) - low) / (high - low)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return int(round(ratio * self._param_slider_steps))

    def _slider_position_to_value(self, key, slider_position):
        min_box = self.param_min_spinboxes.get(key)
        max_box = self.param_max_spinboxes.get(key)
        if min_box is None or max_box is None:
            return float(slider_position)
        low = float(min_box.value())
        high = float(max_box.value())
        if np.isclose(low, high):
            return low
        ratio = float(slider_position) / float(self._param_slider_steps)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return low + (high - low) * ratio

    def _sync_slider_from_spinbox(self, key):
        spinbox = self.param_spinboxes.get(key)
        slider = self.param_sliders.get(key)
        if spinbox is None or slider is None:
            return
        slider.blockSignals(True)
        slider.setValue(self._value_to_slider_position(key, spinbox.value()))
        slider.blockSignals(False)

    def _on_param_bounds_changed(self, key, source):
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
                high = low
            else:
                min_box.blockSignals(True)
                min_box.setValue(high)
                min_box.blockSignals(False)
                low = high

        updated_specs = []
        for spec in self.param_specs:
            if spec.key != key:
                updated_specs.append(spec)
                continue
            clipped_default = float(np.clip(float(spec.default), low, high))
            updated_specs.append(
                ParameterSpec(
                    key=spec.key,
                    symbol=spec.symbol,
                    description=spec.description,
                    default=clipped_default,
                    min_value=float(low),
                    max_value=float(high),
                    decimals=int(spec.decimals),
                )
            )
        self.param_specs = updated_specs
        self.defaults = self._default_param_values(self.param_specs)

        value_box.blockSignals(True)
        value_box.setMinimum(low)
        value_box.setMaximum(high)
        value_box.setValue(float(np.clip(value_box.value(), low, high)))
        value_box.blockSignals(False)
        self._sync_slider_from_spinbox(key)
        self.update_plot(fast=False)
        self._autosave_fit_details()

    def _build_fit_context(
        self,
        seed_overrides=None,
        fixed_params=None,
    ):
        model_def = self._piecewise_model
        if model_def is None:
            raise ValueError("No compiled piecewise model is available.")
        ordered_keys = list(model_def.global_param_names)
        if not ordered_keys:
            raise ValueError("No parameters are available for fitting.")
        fixed_map = {
            str(key): float(value)
            for key, value in dict(fixed_params or {}).items()
            if str(key).strip()
        }

        current_values = self.get_current_param_map()
        spec_by_key = {spec.key: spec for spec in self.param_specs}
        bounds_by_key = {}
        seed_map = {}
        missing_keys = []
        for key in ordered_keys:
            spec = spec_by_key.get(key)
            if spec is None:
                missing_keys.append(key)
                continue
            low = float(spec.min_value)
            high = float(spec.max_value)
            if low > high:
                low, high = high, low
            bounds_by_key[key] = (low, high)

            if key in current_values:
                seed = float(current_values[key])
            else:
                spec = spec_by_key.get(key)
                if spec is None:
                    missing_keys.append(key)
                    continue
                seed = float(spec.default)
            seed_map[key] = float(np.clip(seed, low, high))
        if missing_keys:
            missing_text = ", ".join(dict.fromkeys(missing_keys))
            raise ValueError(
                f"Model/UI parameter mismatch. Missing controls for: {missing_text}"
            )
        for key, value in fixed_map.items():
            if key in seed_map:
                seed_map[key] = float(value)
        if seed_overrides:
            for key, value in seed_overrides.items():
                if key not in seed_map or key in fixed_map:
                    continue
                low, high = bounds_by_key[key]
                seed_map[key] = float(
                    np.clip(float(value), min(low, high), max(low, high))
                )

        for key in ordered_keys:
            if key in fixed_map:
                continue
            low, high = bounds_by_key[key]
            if np.isclose(low, high):
                raise ValueError(
                    f"Bounds for '{key}' are equal; expand them before fitting."
                )

        n_boundaries = max(0, len(model_def.segment_exprs) - 1)
        boundary_seed = np.asarray(
            getattr(
                self,
                "current_boundary_ratios",
                default_boundary_ratios(n_boundaries),
            ),
            dtype=float,
        )
        if boundary_seed.size != n_boundaries:
            boundary_seed = default_boundary_ratios(n_boundaries)

        return {
            "ordered_keys": ordered_keys,
            "seed_map": seed_map,
            "bounds_map": bounds_by_key,
            "model_def": model_def,
            "boundary_seed": boundary_seed,
            "fixed_params": fixed_map,
        }

    def evaluate_model_map(
        self,
        x_data,
        param_values,
        channel_data=None,
        boundary_ratios=None,
    ):
        _ = channel_data
        model_def = self._piecewise_model
        if model_def is None:
            raise ValueError("No compiled piecewise model is available.")
        spec_by_key = {spec.key: spec for spec in self.param_specs}
        bounds_map = {
            key: (
                float(min(spec.min_value, spec.max_value)),
                float(max(spec.min_value, spec.max_value)),
            )
            for key, spec in spec_by_key.items()
        }
        missing_bounds = [
            key for key in model_def.global_param_names if key not in bounds_map
        ]
        if missing_bounds:
            missing_text = ", ".join(missing_bounds)
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
            missing_text = ", ".join(missing_keys)
            raise ValueError(
                f"Model/UI parameter mismatch. Missing parameter values: {missing_text}"
            )
        segments = _make_segment_specs(model_def, seed_map, bounds_map)
        shared = np.asarray(
            [seed_map[key] for key in model_def.global_param_names], dtype=float
        )
        n_boundaries = max(0, len(segments) - 1)
        if boundary_ratios is None:
            b = np.asarray(
                getattr(
                    self,
                    "current_boundary_ratios",
                    default_boundary_ratios(n_boundaries),
                ),
                dtype=float,
            )
        else:
            b = np.asarray(boundary_ratios, dtype=float)
        if b.size != n_boundaries:
            b = default_boundary_ratios(n_boundaries)
        local_flat = _shared_to_local_flat(model_def, shared, np.clip(b, 0.0, 1.0))
        pred = predict_ordered_piecewise(
            np.asarray(x_data, dtype=float).reshape(-1),
            segments,
            local_flat,
            prefer_jit=True,
        )
        return np.asarray(pred["y_hat"], dtype=float)

    def evaluate_model(self, x_data, params, channel_data=None, boundary_ratios=None):
        """Evaluate active piecewise model from ordered list or key-value map."""
        ordered_keys = self._ordered_param_keys()
        if isinstance(params, dict):
            values = {key: float(params[key]) for key in ordered_keys if key in params}
            if len(values) != len(ordered_keys):
                missing = [key for key in ordered_keys if key not in values]
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
        values = {key: float(params[idx]) for idx, key in enumerate(ordered_keys)}
        return self.evaluate_model_map(
            x_data,
            values,
            channel_data=channel_data,
            boundary_ratios=boundary_ratios,
        )

    def _snapshot_full_model_function(self):
        ordered_keys = list(self._ordered_param_keys())

        def model_func(x_data, *params, column_data=None, boundary_ratios=None):
            if len(params) != len(ordered_keys):
                raise ValueError(
                    f"Expected {len(ordered_keys)} parameters, got {len(params)}."
                )
            values = {key: float(params[idx]) for idx, key in enumerate(ordered_keys)}
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
        ratios = np.asarray(
            getattr(self, "current_boundary_ratios", default_boundary_ratios(n)),
            dtype=float,
        ).reshape(-1)
        if ratios.size != n:
            ratios = default_boundary_ratios(n)
        ratios = np.clip(ratios, 0.0, 1.0)
        positions = self._boundary_ratios_to_positions(ratios, n)
        x_min, x_max = self._x_axis_range_for_boundary_controls()
        span = float(x_max - x_min)
        values = x_min + span * positions
        return {f"break{idx + 1}": float(val) for idx, val in enumerate(values)}

    def _piecewise_boundary_conditions(self, segment_count, include_break_values=False):
        n_segments = int(max(0, segment_count))
        if n_segments <= 0:
            return []
        if n_segments == 1:
            return ["all x"]
        n_boundaries = max(0, n_segments - 1)
        value_map = (
            self._breakpoint_value_map(n_boundaries) if include_break_values else {}
        )

        def break_display_name(name):
            text = str(name)
            match = re.fullmatch(r"break(\d+)", text)
            if match is None:
                return text
            try:
                index = int(match.group(1)) - 1
            except Exception:
                index = 0
            return _format_boundary_display_name(max(0, index))

        def break_token(name):
            display_name = break_display_name(name)
            if name not in value_map:
                return display_name
            return f"{display_name} ({self._format_compact_number(value_map[name])})"

        conditions = []
        for seg_idx in range(1, n_segments + 1):
            if seg_idx == 1:
                conditions.append(f"x < {break_token('break1')}")
            elif seg_idx == n_segments:
                last_break = f"break{n_boundaries}"
                conditions.append(f"x >= {break_token(last_break)}")
            else:
                left_break = f"break{seg_idx - 1}"
                right_break = f"break{seg_idx}"
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
        mid_idx = rows // 2
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

    def _build_piecewise_formula_html(self, target_col, segment_exprs):
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
            pretty_expr = format_expression_pretty(expr_text, name_map=symbol_map)
            colored_expr = self._colorize_formula_text_html(
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
        rows_html = "".join(rows)
        brace_html = "".join(brace_cells)
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

    def _set_formula_label(self):
        """Populate the formula label from the active expression."""
        target_col = None
        rhs_expression = None
        segment_exprs = None
        try:
            target_col, seg_exprs = self._parse_equation_text(
                self.current_expression,
                strict=False,
            )
            segment_exprs = list(seg_exprs)
            rhs_expression = " ; ".join(seg_exprs)
        except Exception:
            target_col = None
            rhs_expression = None

        self.formula_label.setTextFormat(Qt.TextFormat.RichText)
        if target_col is not None and segment_exprs is not None:
            self.formula_label.setText(
                self._build_piecewise_formula_html(target_col, segment_exprs)
            )
            boundary_help = "\n".join(
                f"Segment {idx}: {cond}"
                for idx, cond in enumerate(
                    self._piecewise_boundary_conditions(len(segment_exprs)),
                    start=1,
                )
            )
            display_text = f"{target_col} = {' ; '.join(segment_exprs)}"
            self.formula_label.setToolTip(
                f"Python: {self.current_expression}\nDisplay: {display_text}\n\n"
                f"{boundary_help}\n\n"
                "Click equation to edit."
            )
            return

        pretty_equation = format_equation_pretty(
            self.current_expression,
            name_map=self._parameter_symbol_map(),
        )
        display_text = pretty_equation if pretty_equation else self.current_expression
        colored_text = self._colorize_formula_text_html(
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
    ):
        text = str(display_text).strip()
        if not text:
            return ""

        symbol_map = self._parameter_symbol_map()
        param_tokens = set()
        if rhs_expression:
            try:
                _target, seg_exprs = self._parse_equation_text(
                    f"{target_col or self.y_channel} = {rhs_expression}",
                    strict=True,
                )
                for seg_expr in seg_exprs:
                    for name in extract_segment_parameter_names(seg_expr):
                        param_tokens.add(str(symbol_map.get(name, name)))
            except Exception:
                param_tokens = set()

        column_tokens = {str(name) for name in self._available_channel_names()}
        if target_col:
            column_tokens.add(str(target_col))
        column_tokens_upper = {token.upper() for token in column_tokens}
        constant_tokens = {"e", "pi"}

        token_set = set(column_tokens) | set(param_tokens) | set(constant_tokens)
        special_tokens = sorted(
            [
                token
                for token in token_set
                if token and not _PARAMETER_NAME_RE.fullmatch(token)
            ],
            key=len,
            reverse=True,
        )

        number_token_re = r"(?<![A-Za-z_])(?:\d+\.\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?"
        token_parts = [number_token_re]
        token_parts.extend(re.escape(token) for token in special_tokens)
        token_parts.append(r"[A-Za-z_][A-Za-z0-9_]*")
        token_re = re.compile("|".join(token_parts))
        number_re = re.compile(rf"^{number_token_re}$")

        parts = []
        cursor = 0
        for match in token_re.finditer(text):
            start = match.start()
            end = match.end()
            if start > cursor:
                parts.append(html.escape(text[cursor:start]))
            token = match.group(0)
            rendered = html.escape(token)
            style = None
            if token.upper() in column_tokens_upper:
                style = f"color:{_EXPRESSION_COLUMN_COLOR}; font-weight:600;"
            elif token in param_tokens:
                style = f"color:{_EXPRESSION_PARAM_COLOR}; font-weight:600;"
                rendered = parameter_symbol_to_html(token)
            elif token in constant_tokens or number_re.fullmatch(token):
                style = f"color:{_EXPRESSION_CONSTANT_COLOR};"
            if style:
                parts.append(f'<span style="{style}">{rendered}</span>')
            else:
                parts.append(rendered)
            cursor = end

        if cursor < len(text):
            parts.append(html.escape(text[cursor:]))
        rendered = "".join(parts)
        rendered = re.sub(
            r"\^(-?\d+)",
            lambda m: f"<sup>{html.escape(m.group(1))}</sup>",
            rendered,
        )
        return rendered

    def _is_expression_editor_child(self, widget):
        if widget is None or not hasattr(self, "expression_editor_widget"):
            return False
        current = widget
        while current is not None:
            if current is self.expression_editor_widget:
                return True
            current = current.parentWidget()
        return False

    def _set_expression_edit_mode(self, enabled):
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

    def _enter_expression_edit_mode(self):
        if self._expression_edit_mode:
            return
        self._set_expression_editor_text(self.current_expression)
        self._refresh_expression_highlighting()
        self._set_function_status("", is_error=False)
        self._set_expression_edit_mode(True)
        if hasattr(self, "function_input"):
            self.function_input.setFocus()
            self.function_input.selectAll()

    def _on_expression_input_focus_left(self):
        if not self._expression_edit_mode:
            return
        QTimer.singleShot(0, self._apply_expression_on_focus_leave)

    def _on_expression_input_apply_requested(self):
        if not self._expression_edit_mode:
            return
        self._apply_expression_on_focus_leave(force=True)

    def _apply_expression_on_focus_leave(self, force=False):
        if not self._expression_edit_mode:
            return
        if QApplication.activePopupWidget() is not None:
            return
        active_modal = QApplication.activeModalWidget()
        if active_modal is not None and active_modal is not self:
            return
        if not force:
            focus_widget = QApplication.focusWidget()
            if self._is_expression_editor_child(focus_widget):
                return
        if self.apply_expression_from_input():
            self._set_expression_edit_mode(False)

    def create_plot_frame(self, parent_layout):
        """Create plot section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax_residual = None
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.16)

        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(14, 14))
        self.toolbar.setMaximumHeight(28)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_parameters_frame(self, parent_layout):
        """Create full-width controls + parameters section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        top_controls_layout = QVBoxLayout()
        top_controls_layout.setSpacing(4)

        equation_host = QWidget()
        equation_host.setMaximumWidth(760)
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
        self.expression_editor_widget.setMaximumWidth(760)
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
        self.function_input.setMaximumWidth(760)
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

        self.channel_names_btn = self._new_button(
            "Ch Names",
            handler=self._edit_channel_names,
            tooltip="Edit channel display names used in legends and labels.",
            style_sheet="padding: 0px 6px;",
        )
        self.channel_names_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.channel_token_menu = QMenu(self)
        self.insert_token_btn = self._new_button(
            "Insert",
            tooltip="Insert CSV column tokens into the expression.",
            style_sheet="padding: 0px 6px;",
        )
        self.insert_token_btn.setMenu(self.channel_token_menu)
        self.insert_token_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        editor_actions_widget = QWidget()

        editor_actions_widget.setFixedWidth(176)
        editor_actions_layout = QVBoxLayout(editor_actions_widget)
        editor_actions_layout.setContentsMargins(0, 0, 0, 0)
        editor_actions_layout.setSpacing(1)
        editor_actions_layout.addWidget(self.insert_token_btn, 1)
        editor_actions_layout.addWidget(self.channel_names_btn, 1)
        expr_layout.addWidget(editor_actions_widget, 0)
        editor_layout.addLayout(expr_layout)

        self.function_status_label = self._new_label("", object_name="statusLabel")
        self.function_status_label.hide()
        editor_layout.addWidget(self.function_status_label)
        equation_slot_layout.addWidget(self.expression_editor_widget, 0, 0)
        equation_host_layout.addLayout(equation_slot_layout)
        top_equation_row = QHBoxLayout()
        top_equation_row.setContentsMargins(0, 0, 0, 0)
        top_equation_row.setSpacing(10)
        top_equation_row.addWidget(equation_host, 1)
        self.breakpoint_top_widget = self._build_top_breakpoint_controls_widget()
        top_equation_row.addWidget(
            self.breakpoint_top_widget,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        top_controls_layout.addLayout(top_equation_row)

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
        self.file_combo = self._new_combobox(current_index_changed=self.on_file_changed)
        file_layout.addWidget(self.file_combo, 1)
        self.prev_file_btn = self._make_compact_tool_button(
            "◀", "Previous File", self.prev_file
        )
        file_layout.addWidget(self.prev_file_btn)
        self.next_file_btn = self._make_compact_tool_button(
            "▶", "Next File", self.next_file
        )
        file_layout.addWidget(self.next_file_btn)
        self._sync_file_navigation_buttons()
        source_file_layout.addLayout(file_layout)

        channel_layout = QHBoxLayout()
        channel_layout.setSpacing(4)
        channel_layout.addWidget(self._make_param_header_label("X", width=20))
        self.x_channel_combo = self._new_combobox(
            current_index_changed=self._on_x_channel_changed
        )
        channel_layout.addWidget(self.x_channel_combo, 1)
        channel_layout.addWidget(self._make_param_header_label("Y", width=20))
        self.y_channel_combo = self._new_combobox(
            current_index_changed=self._on_y_channel_changed
        )
        channel_layout.addWidget(self.y_channel_combo, 1)
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
        fit_widget_layout.addWidget(file_group)

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

        self.show_residuals_cb = self._new_button(
            "Residuals",
            checkable=True,
            checked=False,
            toggled_handler=lambda: self.update_plot(fast=False),
        )

        self.smoothing_toggle_btn = self._new_button(
            "Smooth",
            checkable=True,
            checked=self.smoothing_enabled,
            toggled_handler=self._on_smoothing_controls_changed,
            tooltip="Apply moving-average smoothing to channels before fitting/analysis.",
        )
        self.smoothing_enable_cb = self.smoothing_toggle_btn

        fit_actions_row = QHBoxLayout()
        fit_actions_row.setSpacing(4)
        self.auto_fit_btn = self._new_button(
            self.auto_fit_btn_default_text,
            handler=self.auto_fit,
            primary=True,
        )
        fit_actions_row.addWidget(self.auto_fit_btn)

        self.reset_from_batch_btn = self._new_button(
            "Reset",
            handler=self.reset_params_from_batch,
            tooltip="Load parameters for the current file from the batch table row.",
        )
        fit_actions_row.addWidget(self.reset_from_batch_btn)
        fit_actions_row.addStretch(1)
        fit_group_layout.addLayout(fit_actions_row)

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

        self.smoothing_window_spin = self._new_compact_int_spinbox(
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

        fit_widget_layout.addWidget(fit_group)
        self._sync_smoothing_window_enabled()
        self.create_batch_controls_frame(fit_widget_layout)
        layout.addLayout(top_controls_layout)

        self.expression_highlighter = ExpressionSyntaxHighlighter(
            self.function_input.document()
        )

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
        param_header_layout.addWidget(
            self._make_param_header_label("Lower", width=self._param_bound_width)
        )
        slider_header = self._make_param_header_label("Range")
        slider_header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        param_header_layout.addWidget(slider_header, 1)
        param_header_layout.addWidget(
            self._make_param_header_label("Upper", width=self._param_bound_width)
        )
        param_header_layout.addWidget(
            self._make_param_header_label("Value", width=self._param_value_width)
        )
        params_and_fit_layout = QHBoxLayout()
        params_and_fit_layout.setSpacing(8)

        params_left_widget = QWidget()
        params_left_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        params_left_layout = QVBoxLayout(params_left_widget)
        params_left_layout.setContentsMargins(0, 0, 0, 0)
        params_left_layout.setSpacing(6)
        self._param_header_to_rows_gap = params_left_layout.spacing()
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
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        self.param_controls_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.param_controls_scroll.setWidget(self.param_controls_widget)
        params_left_layout.addWidget(
            self.param_controls_scroll, 0, Qt.AlignmentFlag.AlignTop
        )
        params_and_fit_layout.addWidget(
            params_left_widget, 1, Qt.AlignmentFlag.AlignTop
        )

        right_panel_width = 420
        fit_widget.setFixedWidth(right_panel_width)
        fit_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        self.fit_panel_widget = fit_widget
        fit_right_widget = QWidget()
        fit_right_widget.setFixedWidth(right_panel_width)
        fit_right_widget.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum
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
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        params_and_fit_layout.addWidget(
            fit_right_widget,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        params_status_layout = QVBoxLayout()
        params_status_layout.setContentsMargins(0, 0, 0, 0)
        params_status_layout.setSpacing(0)
        params_status_layout.addLayout(params_and_fit_layout)
        layout.addLayout(params_status_layout)
        self.rebuild_manual_param_controls()
        self._rebuild_channel_token_buttons()
        self._set_formula_label()
        self._set_expression_edit_mode(False)
        self._sync_fit_panel_top_spacing()
        QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
        self._sync_param_row_tail_spacers()
        QTimer.singleShot(0, self._sync_param_row_tail_spacers)

        self.stats_text = SingleLineStatusLabel("")
        self.stats_text.setObjectName("statsLine")
        self.stats_text.setStyleSheet("padding: 0px 2px; margin: 0px;")

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _set_function_status(self, message, is_error=False):
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

    def _expression_editor_text(self):
        if not hasattr(self, "function_input"):
            return ""
        raw = self.function_input.toPlainText()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return " ".join(lines).strip()

    def _set_expression_editor_text(self, text):
        if not hasattr(self, "function_input"):
            return
        self.function_input.blockSignals(True)
        self.function_input.setPlainText(str(text))
        self.function_input.blockSignals(False)

    def _resolve_column_name(self, name):
        target = str(name).strip()
        if not target:
            return None
        available = self._available_channel_names()
        if target in available:
            return target
        lookup = {col.upper(): col for col in available}
        return lookup.get(target.upper())

    def _parse_equation_text(self, text, strict=False):
        equation = str(text).strip()
        if not equation:
            raise ValueError("Expression is empty.")

        if "=" in equation:
            left, right = equation.split("=", 1)
            lhs_text = left.strip()
            rhs_text = right.strip()
        else:
            if strict:
                raise ValueError("Use equation form: TARGET = seg1 ; seg2 ; ... ; segN")
            lhs_text = self.y_channel
            rhs_text = equation

        if not _PARAMETER_NAME_RE.fullmatch(lhs_text):
            raise ValueError("Invalid left-hand column. Use a CSV column name.")
        if not rhs_text:
            raise ValueError("Right-hand expression is empty.")

        target_col = self._resolve_column_name(lhs_text)
        if target_col is None:
            available = ", ".join(self._available_channel_names()) or "none"
            raise ValueError(
                f"Target column '{lhs_text}' is not in CSV columns ({available})."
            )
        segments = [part.strip() for part in rhs_text.split(";")]
        if len(segments) < 1:
            raise ValueError(
                "Use one or more segment expressions: TARGET = seg1 ; seg2 ; ... ; segN"
            )
        if any(not seg for seg in segments):
            raise ValueError("Each segment expression must be non-empty.")
        return target_col, segments

    def _on_expression_text_changed(self):
        self._refresh_expression_highlighting()

    def _insert_expression_token(self, token_text):
        if not hasattr(self, "function_input"):
            return
        self.function_input.insertPlainText(str(token_text))
        self.function_input.setFocus()

    def _add_channel_token_button(self, label, token):
        button = self._new_button(
            label,
            handler=lambda _checked=False, t=token: self._insert_expression_token(t),
            min_height=20,
            tooltip="Insert column token into expression.",
            style_sheet=(
                f"""
            QPushButton {{
                color: {_EXPRESSION_COLUMN_COLOR};
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

    def _refresh_expression_highlighting(self):
        if self._highlight_refresh_in_progress:
            return
        if not hasattr(self, "expression_highlighter"):
            return
        self._highlight_refresh_in_progress = True
        try:
            expression_text = self._expression_editor_text()
            columns = self._available_channel_names()
            params = []
            if expression_text:
                try:
                    _target, seg_exprs = self._parse_equation_text(
                        expression_text,
                        strict=False,
                    )
                    seen = set()
                    for seg_expr in seg_exprs:
                        seg_params = extract_segment_parameter_names(seg_expr)
                        for name in seg_params:
                            if name not in seen:
                                seen.add(name)
                                params.append(name)
                except Exception:
                    params = []
            self.expression_highlighter.set_context(columns, params)
        finally:
            self._highlight_refresh_in_progress = False

    def _rebuild_channel_token_buttons(self):
        tokens = ["x"]

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

    def apply_expression_from_input(self):
        if self._apply_expression_in_progress:
            return False
        self._apply_expression_in_progress = True
        try:
            expression_text = self._expression_editor_text()
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
            existing_specs = {spec.key: spec for spec in self.param_specs}
            current_values = self.get_current_param_map()

            try:
                target_col, segment_exprs = self._parse_equation_text(
                    expression_text, strict=True
                )
                self.y_channel = target_col
                if self.x_channel == self.y_channel:
                    available = [
                        col
                        for col in self._available_channel_names()
                        if col != self.y_channel
                    ]
                    if available:
                        self.x_channel = available[0]

                model_def = build_piecewise_model_definition(
                    target_col=target_col,
                    segment_exprs=segment_exprs,
                    channel_names=self._available_channel_names(),
                )
                param_names = list(model_def.global_param_names)
                new_specs = []
                new_defaults = []
                for key in param_names:
                    existing = existing_specs.get(key)
                    if existing is not None:
                        symbol_hint = existing.symbol
                        description = existing.description
                        decimals = existing.decimals
                        min_val, max_val = bounds_map.get(
                            key, (existing.min_value, existing.max_value)
                        )
                    else:
                        symbol_hint = key
                        description = f"Parameter {key}"
                        decimals = 6
                        min_val, max_val = -10.0, 10.0

                    min_val = float(min_val)
                    max_val = float(max_val)
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    default_val = float(current_values.get(key, 0.0))
                    default_val = float(np.clip(default_val, min_val, max_val))
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

            self.param_specs = new_specs
            self.defaults = new_defaults
            self.current_expression = f"{target_col} = {' ; '.join(segment_exprs)}"
            self._set_expression_editor_text(self.current_expression)
            self._piecewise_model = model_def
            self.current_boundary_ratios = default_boundary_ratios(
                max(0, len(model_def.segment_exprs) - 1)
            )
            self.last_popt = None
            self.last_pcov = None
            self.last_r2 = None
            self._last_r2 = None
            self._last_fit_active_keys = []
            self.rebuild_manual_param_controls()
            self._refresh_channel_combos()
            self._set_formula_label()
            self._set_function_status("", is_error=False)
            self._refresh_expression_highlighting()
            self.update_plot(fast=False, preserve_view=False)
            self._reset_plot_home_view()
            if self.batch_results:
                batch_action = self._prompt_batch_results_on_equation_change()
                if batch_action == "cancel":
                    return False
                if batch_action == "wipe":
                    for row in self.batch_results:
                        row["params"] = None
                        row["r2"] = None
                        row["error"] = None
                        row["boundary_ratios"] = None
                        row["boundary_values"] = None
                        row["plot_full"] = None
                        row["plot"] = None
                        row["plot_render_size"] = None
                        row["plot_has_fit"] = None
                        row["_equation_stale"] = False
                else:
                    for row in self.batch_results:
                        row["_equation_stale"] = True
                    self.stats_text.append(
                        "Equation updated; existing batch results were kept and marked stale."
                    )
                self.update_batch_table()
                self._refresh_batch_analysis_if_run()
                self.queue_visible_thumbnail_render()
            self._autosave_fit_details()
            return True
        finally:
            self._apply_expression_in_progress = False

    def create_param_control(self, spec, default_val):
        """Create a compact row for lower/upper bounds and current value."""
        key = spec.key
        layout = QHBoxLayout()
        layout.setSpacing(6)

        name_label = self._create_param_label(spec, width=self._param_name_width)
        layout.addWidget(name_label)

        lock_status_label = self._new_label(
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

        min_box = self._new_compact_param_spinbox(
            spec,
            spec.min_value,
            width=self._param_bound_width,
            object_name="paramBoundBox",
            tooltip="Lower bound",
        )
        min_box.valueChanged.connect(
            lambda _value, name=key: self._on_param_bounds_changed(name, "min")
        )
        layout.addWidget(min_box)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(self._param_slider_steps)
        slider.setFixedHeight(18)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.setToolTip("Sweep value across active bounds")
        layout.addWidget(slider, 1)

        max_box = self._new_compact_param_spinbox(
            spec,
            spec.max_value,
            width=self._param_bound_width,
            object_name="paramBoundBox",
            tooltip="Upper bound",
        )
        max_box.valueChanged.connect(
            lambda _value, name=key: self._on_param_bounds_changed(name, "max")
        )
        layout.addWidget(max_box)

        low = float(min_box.value())
        high = float(max_box.value())
        value_box = self._new_compact_param_spinbox(
            spec,
            np.clip(default_val, low, high),
            minimum=low,
            maximum=high,
            width=self._param_value_width,
            object_name="paramValueBox",
            tooltip="Current value",
        )
        value_box.valueChanged.connect(lambda: self.update_plot(fast=False))
        value_box.valueChanged.connect(
            lambda _value, name=key: self._sync_slider_from_spinbox(name)
        )
        value_box.valueChanged.connect(lambda: self._autosave_fit_details())
        layout.addWidget(value_box)

        tail_spacer = QWidget()
        tail_spacer.setFixedWidth(max(0, int(self._param_tail_placeholder_width)))
        tail_spacer.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(tail_spacer)

        container_layout = QVBoxLayout()
        container_layout.setSpacing(1)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addLayout(layout)

        def slider_to_spinbox(position):
            value = self._slider_position_to_value(key, position)
            value_box.blockSignals(True)
            value_box.setValue(value)
            value_box.blockSignals(False)
            self.update_plot(fast=True)

        def slider_pressed():
            self.slider_active = True

        def slider_released():
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
        )

    def _sync_param_row_tail_spacers(self):
        if not hasattr(self, "param_row_tail_spacers"):
            return
        actions_widget = getattr(self, "param_header_actions_widget", None)
        width = int(self._param_tail_placeholder_width)
        if actions_widget is not None:
            try:
                width = max(
                    width,
                    int(actions_widget.minimumSizeHint().width()),
                    int(actions_widget.sizeHint().width()),
                )
            except Exception:
                pass
        width = max(0, int(width))
        self._param_tail_placeholder_width = width
        for spacer in self.param_row_tail_spacers:
            if spacer is None:
                continue
            spacer.setFixedWidth(width)

    def rebuild_manual_param_controls(self):
        if not hasattr(self, "param_controls_layout"):
            return
        clear_layout(self.param_controls_layout)
        self.param_spinboxes.clear()
        self.param_sliders.clear()
        self.param_min_spinboxes.clear()
        self.param_max_spinboxes.clear()
        self.param_lock_status_labels.clear()
        self.param_tail_spacers_by_key.clear()
        self.param_row_tail_spacers.clear()

        spec_by_key = {spec.key: spec for spec in self.param_specs}
        default_by_key = {
            spec.key: (self.defaults[idx] if idx < len(self.defaults) else 0.0)
            for idx, spec in enumerate(self.param_specs)
        }

        for section in self._ordered_parameter_sections():
            kind = str(section.get("kind"))
            keys = [str(key) for key in (section.get("keys") or [])]

            if kind == "segment":
                seg_idx = int(section.get("index", 0))
                tooltip = ""
                model_def = self._piecewise_model
                if model_def is not None and 1 <= seg_idx <= len(
                    model_def.segment_exprs
                ):
                    tooltip = str(model_def.segment_exprs[seg_idx - 1])
                self.param_controls_layout.addWidget(
                    self._build_param_section_header(
                        f"Segment {seg_idx}",
                        tooltip=tooltip,
                    )
                )
                if not keys:
                    self.param_controls_layout.addWidget(
                        self._new_label(
                            "Shared parameters are listed in earlier segments.",
                            object_name="statusLabel",
                            style_sheet="color: #64748b; font-style: italic;",
                        )
                    )
            elif kind == "boundary":
                self.param_controls_layout.addWidget(
                    self._build_param_boundary_marker(section.get("index", 0))
                )
            elif kind == "shared":
                self.param_controls_layout.addWidget(
                    self._build_param_section_header("Shared Parameters")
                )

            for key in keys:
                spec = spec_by_key.get(key)
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
                ) = self.create_param_control(spec, default_val)
                self.param_spinboxes[spec.key] = spinbox
                self.param_sliders[spec.key] = slider
                self.param_min_spinboxes[spec.key] = min_box
                self.param_max_spinboxes[spec.key] = max_box
                self.param_lock_status_labels[spec.key] = lock_status_label
                self.param_tail_spacers_by_key[spec.key] = tail_spacer
                self.param_row_tail_spacers.append(tail_spacer)
                self.param_controls_layout.addLayout(control_layout)
                self._sync_slider_from_spinbox(spec.key)

        self.param_controls_layout.addStretch(1)
        self._sync_breakpoint_sliders_from_state()
        self._refresh_param_capture_mapping_controls()
        self._sync_param_row_tail_spacers()
        QTimer.singleShot(0, self._sync_param_row_tail_spacers)
        QTimer.singleShot(0, self._sync_fit_panel_top_spacing)

    def _edit_channel_names(self):
        channel_names = self._available_channel_names()
        if not channel_names:
            self.stats_text.append("No channels available to rename.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Channel Names")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(10, 10, 10, 10)
        dialog_layout.setSpacing(8)

        help_label = self._new_label(
            "Set display names and units used in legends and axes.",
            style_sheet="color: #4b5563;",
        )
        dialog_layout.addWidget(help_label)

        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(6)
        grid_layout.setVerticalSpacing(4)
        grid_layout.addWidget(
            self._new_label(
                "Channel",
                object_name="statusLabel",
                style_sheet="font-weight: 700; color: #334155;",
            ),
            0,
            0,
        )
        grid_layout.addWidget(
            self._new_label(
                "Name",
                object_name="statusLabel",
                style_sheet="font-weight: 700; color: #334155;",
            ),
            0,
            1,
        )
        grid_layout.addWidget(
            self._new_label(
                "Unit",
                object_name="statusLabel",
                style_sheet="font-weight: 700; color: #334155;",
            ),
            0,
            2,
        )
        editors = {}
        first_editor = None
        for row_idx, channel_name in enumerate(channel_names, start=1):
            channel_label = self._new_label(
                str(channel_name),
                object_name="paramInline",
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            name_editor = self._new_line_edit(
                str(self.channels.get(channel_name, channel_name))
            )
            name_editor.setPlaceholderText(str(channel_name))
            name_editor.setToolTip(
                f"Display label for {channel_name}. Leave blank to use {channel_name}."
            )
            unit_editor = self._new_line_edit(
                str(getattr(self, "channel_units", {}).get(channel_name, ""))
            )
            unit_editor.setPlaceholderText("unit")
            unit_editor.setToolTip(
                f"Unit for {channel_name} (for example V, mV, s, ms)."
            )
            grid_layout.addWidget(channel_label, row_idx, 0)
            grid_layout.addWidget(name_editor, row_idx, 1)
            grid_layout.addWidget(unit_editor, row_idx, 2)
            editors[channel_name] = (name_editor, unit_editor)
            if first_editor is None:
                first_editor = name_editor
        dialog_layout.addLayout(grid_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)

        if first_editor is not None:

            def _activate_dialog():
                dialog.raise_()
                dialog.activateWindow()
                first_editor.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                first_editor.selectAll()

            QTimer.singleShot(0, _activate_dialog)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        for channel_name, (name_editor, unit_editor) in editors.items():
            value = name_editor.text().strip()
            unit = unit_editor.text().strip()
            self.channels[channel_name] = value or channel_name
            self.channel_units[channel_name] = unit

        self.update_plot(fast=False, preserve_view=False)

    def _refresh_channel_combos(self):
        if self._channel_sync_in_progress:
            return
        if self.current_data is None:
            return
        self._channel_sync_in_progress = True
        try:
            channel_columns = []
            seen_columns = set()
            for col in self.current_data.columns:
                key = str(col).strip()
                if not key or key in seen_columns:
                    continue
                try:
                    self.current_data[col].to_numpy(dtype=float, copy=False)
                except Exception:
                    continue
                seen_columns.add(key)
                channel_columns.append(key)
            if not channel_columns:
                return
            for key in channel_columns:
                existing_label = str(self.channels.get(key, "")).strip()
                if not existing_label:
                    self.channels[key] = key
                if key not in self.channel_units:
                    self.channel_units[key] = ""

            x_fallback = "TIME" if "TIME" in channel_columns else None
            if x_fallback is None:
                for col in channel_columns:
                    if col != self.y_channel:
                        x_fallback = col
                        break
                if x_fallback is None:
                    x_fallback = channel_columns[0]
            x_choice = (
                self.x_channel if self.x_channel in channel_columns else x_fallback
            )
            y_choice = (
                self.y_channel
                if self.y_channel in channel_columns
                else ("CH2" if "CH2" in channel_columns else channel_columns[0])
            )
            if x_choice == y_choice and len(channel_columns) > 1:
                for col in channel_columns:
                    if col != y_choice:
                        x_choice = col
                        break
            self.x_channel = x_choice
            self.y_channel = y_choice
            if hasattr(self, "function_input"):
                expr_text = self._expression_editor_text()
                if expr_text:
                    try:
                        _old_target, seg_exprs = self._parse_equation_text(
                            expr_text, strict=False
                        )
                        normalized = f"{self.y_channel} = {' ; '.join(seg_exprs)}"
                        if normalized != expr_text:
                            self.current_expression = normalized
                            self._set_expression_editor_text(normalized)
                    except Exception:
                        pass
            if hasattr(self, "x_channel_combo"):
                self.x_channel_combo.blockSignals(True)
                self.x_channel_combo.clear()
                for col in channel_columns:
                    self.x_channel_combo.addItem(col, col)
                x_idx = self.x_channel_combo.findData(self.x_channel)
                if x_idx >= 0:
                    self.x_channel_combo.setCurrentIndex(x_idx)
                self.x_channel_combo.blockSignals(False)
            if hasattr(self, "y_channel_combo"):
                self.y_channel_combo.blockSignals(True)
                self.y_channel_combo.clear()
                for col in channel_columns:
                    self.y_channel_combo.addItem(col, col)
                y_idx = self.y_channel_combo.findData(self.y_channel)
                if y_idx >= 0:
                    self.y_channel_combo.setCurrentIndex(y_idx)
                self.y_channel_combo.blockSignals(False)
            self._sync_breakpoint_sliders_from_state()
            self._rebuild_channel_token_buttons()
        finally:
            self._channel_sync_in_progress = False

    def _on_x_channel_changed(self, _index):
        if self._channel_sync_in_progress:
            return
        if not hasattr(self, "x_channel_combo"):
            return
        data = self.x_channel_combo.currentData()
        if not data:
            return
        self.x_channel = str(data)
        if self.x_channel == self.y_channel and self.current_data is not None:
            for col in self._available_channel_names():
                if col != self.x_channel:
                    self.y_channel = col
                    break
        self._sync_breakpoint_sliders_from_state()
        self.update_plot(fast=False, preserve_view=False)

    def _on_y_channel_changed(self, _index):
        if self._channel_sync_in_progress:
            return
        if not hasattr(self, "y_channel_combo"):
            return
        data = self.y_channel_combo.currentData()
        if not data:
            return
        self.y_channel = str(data)
        if self.x_channel == self.y_channel and self.current_data is not None:
            for col in self._available_channel_names():
                if col != self.y_channel:
                    self.x_channel = col
                    break
        try:
            _old_target, seg_exprs = self._parse_equation_text(
                self._expression_editor_text(),
                strict=False,
            )
            normalized = f"{self.y_channel} = {' ; '.join(seg_exprs)}"
            self.current_expression = normalized
            self._set_expression_editor_text(normalized)
            self._set_formula_label()
        except Exception:
            pass
        self._sync_breakpoint_sliders_from_state()
        self.update_plot(fast=False, preserve_view=False)

    def _set_toolbar_home_limits(
        self,
        home_main_xlim,
        home_main_ylim,
        home_residual_ylim=None,
        *,
        keep_current_view=False,
    ):
        toolbar = getattr(self, "toolbar", None)
        if toolbar is None or not hasattr(self, "ax") or self.ax is None:
            return
        current_main_xlim = None
        current_main_ylim = None
        current_residual_ylim = None
        if keep_current_view:
            try:
                current_main_xlim = tuple(self.ax.get_xlim())
            except Exception:
                current_main_xlim = None
            try:
                current_main_ylim = tuple(self.ax.get_ylim())
            except Exception:
                current_main_ylim = None
            if hasattr(self, "ax_residual") and self.ax_residual is not None:
                try:
                    current_residual_ylim = tuple(self.ax_residual.get_ylim())
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

            nav_stack = getattr(toolbar, "_nav_stack", None)
            if nav_stack is not None:
                nav_stack.clear()
            push_current = getattr(toolbar, "push_current", None)
            if callable(push_current):
                push_current()
            set_history_buttons = getattr(toolbar, "set_history_buttons", None)
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

    def _reset_plot_home_view(self):
        """Reset toolbar Home target to the current plotted extents."""
        if not hasattr(self, "ax") or self.ax is None:
            return
        main_xlim = None
        main_ylim = None
        residual_ylim = None
        try:
            main_xlim = tuple(self.ax.get_xlim())
        except Exception:
            pass
        try:
            main_ylim = tuple(self.ax.get_ylim())
        except Exception:
            pass
        if hasattr(self, "ax_residual") and self.ax_residual is not None:
            try:
                residual_ylim = tuple(self.ax_residual.get_ylim())
            except Exception:
                residual_ylim = None
        self._set_toolbar_home_limits(main_xlim, main_ylim, residual_ylim)

    def create_batch_controls_frame(self, parent_layout):
        """Create batch-only controls (shared params/settings are above tabs)."""
        group = QGroupBox("")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        batch_label = self._new_label(
            "Batch Actions",
            style_sheet="font-weight: 600; color: #374151; padding: 1px 2px;",
        )
        layout.addWidget(batch_label)

        self.run_batch_btn_default_text = "Run Batch"
        self.run_batch_btn = self._new_button(
            self.run_batch_btn_default_text,
            handler=self.run_batch_fit,
        )

        export_fit_btn = self._new_button(
            "Export Fit",
            handler=self.export_fit_details,
            tooltip="Export equation, parameter specs, and fit results to JSON.",
        )
        import_fit_btn = self._new_button(
            "Import Fit",
            handler=self.import_fit_details,
            tooltip="Import equation, parameter specs, and fit results from JSON.",
        )

        regex_layout = QHBoxLayout()
        regex_layout.setSpacing(4)
        self.regex_input = self._new_line_edit(
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

        self.batch_parse_feedback_label = self._new_label(
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
        self.cancel_batch_btn = self._new_button(
            "Cancel",
            handler=self.cancel_batch_fit,
            enabled=False,
        )
        actions_row.addWidget(self.cancel_batch_btn)
        actions_row.addWidget(export_fit_btn)
        actions_row.addWidget(import_fit_btn)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self.batch_status_label = self._new_label("", object_name="statusLabel")
        self.batch_status_label.hide()
        layout.addWidget(self.batch_status_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_results_frame(self, parent_layout):
        """Create batch results table."""
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(0)
        self.batch_table.setRowCount(0)
        self.batch_table.cellClicked.connect(self._on_batch_table_cell_clicked)
        self.batch_table_header = RichTextHeaderView(
            Qt.Orientation.Horizontal, self.batch_table
        )
        self.batch_table.setHorizontalHeader(self.batch_table_header)
        batch_header = self.batch_table.horizontalHeader()
        batch_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        batch_header.setStretchLastSection(True)
        batch_header.setMinimumSectionSize(60)
        v_header = self.batch_table.verticalHeader()
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

    def create_batch_analysis_frame(self, parent_layout):
        """Create interactive batch analysis plot controls."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        source_row = QHBoxLayout()
        source_row.setSpacing(4)
        source_row.addWidget(self._new_label("Analysis Source: Completed Batch Run"))
        self.analysis_status_label = self._new_label(
            "Using completed batch results (0 rows).",
            object_name="statusLabel",
        )
        source_row.addWidget(self.analysis_status_label, 1)
        layout.addLayout(source_row)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(4)
        controls_row.addWidget(self._new_label("Field (X):"))
        self.analysis_x_combo = self._new_combobox(
            current_index_changed=self.update_batch_analysis_plot,
            rich_text=True,
        )
        controls_row.addWidget(self.analysis_x_combo, 2)
        self.analysis_clear_x_btn = self._new_button(
            "Clear X",
            handler=self._clear_analysis_x_field,
        )
        controls_row.addWidget(self.analysis_clear_x_btn)
        controls_row.addWidget(self._new_label("Plot:"))
        self.analysis_mode_combo = self._new_combobox(
            items=[
                ("Combined", "combined"),
                ("One per parameter", "separate"),
            ],
            current_index_changed=self.update_batch_analysis_plot,
        )
        controls_row.addWidget(self.analysis_mode_combo)

        self.analysis_show_points_btn = self._new_button(
            "Points",
            checkable=True,
            checked=True,
            toggled_handler=self.update_batch_analysis_plot,
        )
        controls_row.addWidget(self.analysis_show_points_btn)

        self.analysis_show_series_line_btn = self._new_button(
            "Series Line",
            checkable=True,
            checked=False,
            toggled_handler=self.update_batch_analysis_plot,
        )
        controls_row.addWidget(self.analysis_show_series_line_btn)

        self.analysis_fit_line_btn = self._new_button(
            "Best-Fit Lines",
            checkable=True,
            checked=True,
            toggled_handler=self.update_batch_analysis_plot,
        )
        controls_row.addWidget(self.analysis_fit_line_btn)

        self.analysis_legend_btn = self._new_button(
            "Legend",
            checkable=True,
            checked=True,
            toggled_handler=self.update_batch_analysis_plot,
        )
        controls_row.addWidget(self.analysis_legend_btn)
        layout.addLayout(controls_row)

        params_row = QHBoxLayout()
        params_row.setSpacing(4)
        params_row.addWidget(self._new_label("Parameters (Y):"))
        self.analysis_param_buttons = {}
        self.analysis_params_button_layout = QHBoxLayout()
        self.analysis_params_button_layout.setSpacing(4)
        params_row.addLayout(self.analysis_params_button_layout, 1)
        layout.addLayout(params_row)

        self.analysis_fig = Figure(figsize=(10, 3.2), dpi=100)
        self.analysis_fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.2)
        self.analysis_canvas = FigureCanvas(self.analysis_fig)
        if self._analysis_point_pick_cid is None:
            self._analysis_point_pick_cid = self.analysis_canvas.mpl_connect(
                "pick_event",
                self._on_analysis_point_picked,
            )
        layout.addWidget(self.analysis_canvas)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _batch_row_error_text(self, row):
        pattern_error = str(row.get("pattern_error") or "").strip()
        fit_error = str(row.get("error") or "").strip()
        is_stale = bool(row.get("_equation_stale"))

        normalized_fit_error = fit_error.lower().replace(".", "").strip()
        if normalized_fit_error in {"cancelled", "canceled"}:
            fit_error = ""

        parts = []
        if is_stale:
            parts.append("Stale fit (equation changed)")
        if pattern_error:
            parts.append(pattern_error)
        if fit_error and fit_error not in parts:
            parts.append(fit_error)
        return " | ".join(parts)

    def _fit_param_range_violations(self, params):
        values = self._as_float_array(params)
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
            tolerance = 1e-12 * max(1.0, abs(low), abs(high))
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

    def _fit_param_range_error_text(self, violations):
        rows = list(violations or [])
        if not rows:
            return None
        samples = []
        for item in rows[:3]:
            key = str(item.get("key") or "")
            label = self._display_name_for_param_key(key) if key else key
            value = float(item.get("value"))
            low = float(item.get("low"))
            high = float(item.get("high"))
            samples.append(f"{label}={value:.6g} not in [{low:.6g}, {high:.6g}]")
        remainder = len(rows) - len(samples)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        return f"{_FIT_PARAM_RANGE_ERROR_PREFIX} {'; '.join(samples)}{suffix}"

    def _apply_param_range_validation_to_row(self, row):
        normalized = dict(row or {})
        violations = self._fit_param_range_violations(normalized.get("params"))
        violation_keys = [
            str(item.get("key")) for item in violations if item.get("key")
        ]
        normalized["_param_range_violation_keys"] = violation_keys

        range_error = self._fit_param_range_error_text(violations)
        existing_parts = [
            part.strip()
            for part in str(normalized.get("error") or "").split("|")
            if str(part).strip()
        ]
        existing_parts = [
            part
            for part in existing_parts
            if not str(part).startswith(_FIT_PARAM_RANGE_ERROR_PREFIX)
        ]
        if range_error:
            existing_parts.append(range_error)
        normalized["error"] = (
            " | ".join(dict.fromkeys(existing_parts)) if existing_parts else None
        )
        return normalized

    def _current_file_param_violation_keys(self):
        file_path = self._current_loaded_file_path()
        if not file_path:
            return set()
        row_index = self._find_batch_result_index_by_file(file_path)
        if row_index is None or row_index < 0 or row_index >= len(self.batch_results):
            return set()
        row = self.batch_results[row_index]
        violations = self._fit_param_range_violations(row.get("params"))
        return {str(item.get("key")) for item in violations if item.get("key")}

    def _refresh_param_value_error_highlighting(self):
        invalid_by_row = self._current_file_param_violation_keys()
        for spec in self.param_specs:
            key = str(spec.key)
            box = self.param_spinboxes.get(key)
            if box is None:
                continue
            numeric = _finite_float_or_none(box.value())
            low = float(min(spec.min_value, spec.max_value))
            high = float(max(spec.min_value, spec.max_value))
            tolerance = 1e-12 * max(1.0, abs(low), abs(high))
            out_of_spec = numeric is not None and (
                numeric < (low - tolerance) or numeric > (high + tolerance)
            )
            invalid = bool(out_of_spec or key in invalid_by_row)
            box.setStyleSheet("color: #b91c1c;" if invalid else "")

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
            params = self._as_float_array(row.get("params"))
            boundary_values = self._as_float_array(row.get("boundary_values"))
            for item in param_columns:
                idx = int(item["index"])
                if item["kind"] == "param":
                    value = float(params[idx]) if params.size > idx else None
                else:
                    value = (
                        float(boundary_values[idx])
                        if boundary_values.size > idx
                        else None
                    )
                record[str(item["key"])] = value
            record["R2"] = row.get("r2")
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

    def _coerce_numeric_array(self, values):
        numeric = []
        for value in values:
            if value is None:
                numeric.append(np.nan)
                continue
            text = str(value).strip()
            if text == "":
                numeric.append(np.nan)
                continue
            try:
                numeric.append(float(text))
            except Exception:
                numeric.append(np.nan)
        return np.asarray(numeric, dtype=float)

    def _analysis_field_display_text(self, field_key):
        key_text = str(field_key).strip()
        if not key_text:
            return ("", "")
        label_text = key_text
        for item in self._batch_parameter_column_items():
            item_key = str(item.get("key", "")).strip()
            if item_key != key_text:
                continue
            token = str(item.get("token", "")).strip()
            if token:
                label_text = token
            break
        label_html = parameter_symbol_to_html(label_text) or html.escape(label_text)
        return (label_text, label_html)

    def _default_analysis_x_field(self, numeric_columns):
        for key in self.batch_capture_keys:
            if key in numeric_columns:
                return key
        for key in numeric_columns:
            if key not in self.analysis_param_columns and key != "R2":
                return key
        return numeric_columns[0] if numeric_columns else None

    def _refresh_batch_analysis_data(self, preserve_selection):
        raw_records = self._extract_analysis_records_from_batch()
        base_status = "Using completed batch results"

        records = list(raw_records)
        self.analysis_status_label.setText(f"{base_status} ({len(records)} rows).")

        self.analysis_records = records
        self.analysis_columns = self._extract_analysis_columns(records)
        self.analysis_numeric_data = {}
        for column in self.analysis_columns:
            values = [row.get(column, "") for row in records]
            as_numeric = self._coerce_numeric_array(values)
            if np.isfinite(as_numeric).sum() > 0:
                self.analysis_numeric_data[column] = as_numeric

        numeric_columns = list(self.analysis_numeric_data.keys())
        self.analysis_param_columns = [
            str(item["key"])
            for item in self._batch_parameter_column_items()
            if str(item["key"]) in self.analysis_numeric_data
        ]
        if not self.analysis_param_columns:
            self.analysis_param_columns = [
                key for key in numeric_columns if key not in ("R2",)
            ]

        previous_x = self.analysis_x_combo.currentData() if preserve_selection else None
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

        chosen_x = (
            previous_x
            if (preserve_selection and previous_x in numeric_columns)
            else None
        )
        if chosen_x is None:
            chosen_x = self._default_analysis_x_field(numeric_columns)
        x_idx = self.analysis_x_combo.findData(chosen_x)
        if x_idx < 0:
            x_idx = self.analysis_x_combo.findData(None)
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

    def _toggle_analysis_param(self, key):
        control = self.analysis_param_buttons.get(str(key))
        if control is None:
            return
        control.setChecked(not control.isChecked())

    def _rebuild_analysis_param_buttons(self, previous_params):
        while self.analysis_params_button_layout.count():
            item = self.analysis_params_button_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.analysis_param_buttons = {}
        for key in self.analysis_param_columns:
            control = self._new_checkbox(
                "",
                checked=(key in previous_params),
                toggled_handler=self.update_batch_analysis_plot,
                tooltip=f"Parameter '{key}'",
            )
            display_label = str(self._display_name_for_param_key(key)).strip() or str(
                key
            )
            display_html = parameter_symbol_to_html(display_label) or html.escape(
                display_label
            )
            if display_label != str(key):
                display_html = f"{display_html} <span style='color:#64748b;'>({html.escape(str(key))})</span>"
            label = ClickableLabel(display_html)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setToolTip(f"Parameter '{key}'")
            label.clicked.connect(
                lambda _checked=False, param_key=str(key): self._toggle_analysis_param(
                    param_key
                )
            )

            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 2, 0)
            item_layout.setSpacing(2)
            item_layout.addWidget(control)
            item_layout.addWidget(label)
            self.analysis_params_button_layout.addWidget(item_widget)
            self.analysis_param_buttons[str(key)] = control
        self.analysis_params_button_layout.addStretch()

    def _select_all_analysis_params(self):
        for button in self.analysis_param_buttons.values():
            button.setChecked(True)
        self.update_batch_analysis_plot()

    def _clear_analysis_params(self):
        for button in self.analysis_param_buttons.values():
            button.setChecked(False)
        self.update_batch_analysis_plot()

    def _clear_analysis_x_field(self):
        idx = self.analysis_x_combo.findData(None)
        if idx >= 0:
            self.analysis_x_combo.setCurrentIndex(idx)
        self.update_batch_analysis_plot()

    def _show_analysis_message(self, message):
        self._analysis_scatter_files = {}
        self.analysis_fig.clear()
        ax = self.analysis_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        self.analysis_canvas.draw_idle()

    def _analysis_record_file_ref(self, record):
        file_ref = str(record.get("__file_ref") or "").strip()
        if file_ref:
            return file_ref

        display_name = str(record.get("File") or "").strip()
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

    def _on_analysis_point_picked(self, event):
        artist = getattr(event, "artist", None)
        if artist is None:
            return
        file_refs = self._analysis_scatter_files.get(artist)
        if not file_refs:
            return

        picked = getattr(event, "ind", None)
        if picked is None or len(picked) == 0:
            return
        point_idx = int(picked[0])
        if point_idx < 0 or point_idx >= len(file_refs):
            return

        file_ref = file_refs[point_idx]
        if not file_ref:
            self.stats_text.append(
                "Unable to resolve clicked point to a unique source file."
            )
            return
        self._open_file_in_plot_tab(file_ref)

    def _linear_fit(self, x_data, y_data):
        if x_data.size < 2:
            return None
        if np.isclose(float(np.ptp(x_data)), 0.0):
            return None
        try:
            slope, intercept = np.polyfit(x_data, y_data, 1)
            return float(slope), float(intercept)
        except Exception:
            return None

    def update_batch_analysis_plot(self):
        """Plot parameter variation against selected field."""
        if not hasattr(self, "analysis_fig"):
            return
        self._analysis_scatter_files = {}
        if not self.analysis_numeric_data:
            self._show_analysis_message("No numeric data available for analysis.")
            return

        x_field = self.analysis_x_combo.currentData()
        selected_params = self._selected_analysis_params()
        if x_field not in self.analysis_numeric_data:
            self._show_analysis_message("Select an X field to plot.")
            return
        if not selected_params:
            self._show_analysis_message("Select at least one parameter to plot.")
            return

        x_values = self.analysis_numeric_data[x_field]
        mode = self.analysis_mode_combo.currentData()
        show_points = self.analysis_show_points_btn.isChecked()
        show_series_line = self.analysis_show_series_line_btn.isChecked()
        show_fit_lines = self.analysis_fit_line_btn.isChecked()
        show_legend = self.analysis_legend_btn.isChecked()

        if not (show_points or show_series_line or show_fit_lines):
            self._show_analysis_message(
                "Enable at least one plot layer (Points/Line/Fit)."
            )
            return

        self.analysis_fig.clear()
        if mode == "separate" and len(selected_params) > 1:
            axes = self.analysis_fig.subplots(len(selected_params), 1, sharex=True)
            axes = list(np.atleast_1d(axes))
        else:
            axes = [self.analysis_fig.add_subplot(111)]

        plotted_any = False
        for idx, param_name in enumerate(selected_params):
            y_values = self.analysis_numeric_data.get(param_name)
            if y_values is None:
                continue
            mask = np.isfinite(x_values) & np.isfinite(y_values)
            if np.count_nonzero(mask) == 0:
                continue

            plotted_any = True
            x_plot = x_values[mask]
            y_plot = y_values[mask]
            file_refs = [
                self._analysis_record_file_ref(self.analysis_records[row_idx])
                for row_idx in np.flatnonzero(mask)
            ]
            order = np.argsort(x_plot)
            x_sorted = x_plot[order]
            y_sorted = y_plot[order]
            file_refs_sorted = [file_refs[int(order_idx)] for order_idx in order]
            color = palette_color(idx)
            target_ax = axes[idx] if len(axes) > 1 else axes[0]
            param_plot_label = self._display_name_for_param_key_mathtext(param_name)

            if show_points:
                scatter_label = (
                    param_plot_label if not show_series_line else "_nolegend_"
                )
                scatter = target_ax.scatter(
                    x_sorted, y_sorted, s=26, color=color, label=scatter_label
                )
                scatter.set_picker(5)
                self._analysis_scatter_files[scatter] = file_refs_sorted
            if show_series_line:
                target_ax.plot(
                    x_sorted,
                    y_sorted,
                    linewidth=1.4,
                    alpha=0.85,
                    color=color,
                    label=param_plot_label,
                )

            if show_fit_lines:
                fit = self._linear_fit(x_sorted, y_sorted)
                if fit is not None:
                    slope, intercept = fit
                    x_line = np.linspace(
                        float(np.min(x_sorted)), float(np.max(x_sorted)), 200
                    )
                    y_line = slope * x_line + intercept
                    fit_label = (
                        f"{param_plot_label} fit" if len(axes) == 1 else "Best fit"
                    )
                    target_ax.plot(
                        x_line,
                        y_line,
                        linestyle="--",
                        linewidth=1.6,
                        color=color,
                        label=fit_label,
                    )

            if len(axes) > 1:
                target_ax.set_ylabel(param_plot_label)
                target_ax.grid(True, alpha=0.25)
                if show_legend:
                    target_ax.legend(loc="best", fontsize=8)

        if not plotted_any:
            self._show_analysis_message(
                "No finite X/Y pairs available for the selected fields."
            )
            return

        if len(axes) == 1:
            axes[0].set_ylabel("Parameter Value")
            if show_legend:
                axes[0].legend(loc="best", fontsize=8)
            axes[0].grid(True, alpha=0.3)

        if x_field in self.analysis_param_columns:
            x_axis_label = self._display_name_for_param_key_mathtext(x_field)
        else:
            x_axis_label = x_field
        axes[-1].set_xlabel(x_axis_label)
        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw_idle()

    def _current_batch_row_height(self):
        return max(
            self.batch_row_height_min,
            min(self.batch_row_height_max, int(self.batch_row_height)),
        )

    def _current_batch_thumbnail_size(self):
        row_height = self._current_batch_row_height()
        thumb_height = max(24, row_height - 8)
        thumb_width = max(36, int(round(thumb_height * self.batch_thumbnail_aspect)))
        return (thumb_width, thumb_height)

    def _full_batch_thumbnail_size(self):
        full_height = max(
            48,
            int(
                round(
                    (self._current_batch_row_height() - 8)
                    * self.batch_thumbnail_supersample
                )
            ),
        )
        full_width = max(36, int(round(full_height * self.batch_thumbnail_aspect)))
        return (full_width, full_height)

    def _apply_batch_row_heights(self):
        if not hasattr(self, "batch_table"):
            return
        if self._batch_row_height_sync:
            return

        row_height = self._current_batch_row_height()
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

    def _find_table_row_by_file(self, file_path):
        """Find table row index by file path stored in item user data."""
        if self.batch_table.columnCount() == 0:
            return None
        for row_idx in range(self.batch_table.rowCount()):
            item = self.batch_table.item(row_idx, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == file_path:
                return row_idx
        return None

    def _find_batch_result_index_by_file(self, file_path):
        if not file_path:
            return None
        for idx, row in enumerate(self.batch_results):
            if row.get("file") == file_path:
                return idx
        return None

    def _rebuild_batch_capture_keys_from_rows(self):
        keys = []
        for row in self.batch_results:
            captures = dict(row.get("captures") or {})
            for key in captures.keys():
                text = str(key).strip()
                if text and text not in keys:
                    keys.append(text)
        self.batch_capture_keys = keys

    def _batch_parameter_column_items(self):
        items = []
        for idx, spec in enumerate(self.param_specs):
            token = (
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
        n_boundaries = max(
            0,
            len(self._piecewise_model.segment_exprs) - 1
            if self._piecewise_model is not None
            else 0,
        )
        for idx in range(n_boundaries):
            items.append(
                {
                    "kind": "boundary",
                    "index": int(idx),
                    "key": f"X_{idx}",
                    "token": _format_boundary_display_name(idx),
                }
            )
        return items

    @staticmethod
    def _as_float_array(values):
        if values is None:
            return np.asarray([], dtype=float)
        try:
            return np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)

    @staticmethod
    def _json_float_or_none(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    @staticmethod
    def _float_list_or_none(values):
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size != arr.size:
            return None
        return [float(v) for v in arr]

    def _fit_details_sidecar_path(self):
        base_dir = None
        source_text = str(getattr(self, "current_dir", "")).strip()
        if source_text:
            source_path = Path(source_text).expanduser()
            if source_path.is_dir():
                base_dir = source_path
            elif source_path.is_file():
                base_dir = source_path.parent
        if base_dir is None:
            selected = list(getattr(self, "_source_selected_paths", []) or [])
            if selected:
                selected_parent = Path(selected[0]).expanduser().parent
                if selected_parent.exists():
                    base_dir = selected_parent
        if base_dir is None:
            return None
        return Path(base_dir) / FIT_DETAILS_FILENAME

    def _serialize_fit_parameter_specs(self):
        serialized = []
        for spec in self.param_specs:
            min_box = self.param_min_spinboxes.get(spec.key)
            max_box = self.param_max_spinboxes.get(spec.key)
            value_box = self.param_spinboxes.get(spec.key)
            low = float(min_box.value()) if min_box is not None else float(spec.min_value)
            high = (
                float(max_box.value()) if max_box is not None else float(spec.max_value)
            )
            if low > high:
                low, high = high, low
            value = float(value_box.value()) if value_box is not None else float(spec.default)
            serialized.append(
                {
                    "key": str(spec.key),
                    "symbol": str(spec.symbol),
                    "description": str(spec.description),
                    "default": float(np.clip(float(spec.default), low, high)),
                    "min_value": float(low),
                    "max_value": float(high),
                    "decimals": int(spec.decimals),
                    "value": float(np.clip(value, low, high)),
                }
            )
        return serialized

    def _serialize_fit_batch_rows(self):
        rows = []
        for row in list(getattr(self, "batch_results", []) or []):
            file_ref = str(row.get("file") or "").strip()
            if not file_ref:
                continue
            rows.append(
                {
                    "file": file_ref,
                    "file_stem": stem_for_file_ref(file_ref),
                    "captures": dict(row.get("captures") or {}),
                    "params": self._float_list_or_none(row.get("params")),
                    "r2": self._json_float_or_none(row.get("r2")),
                    "error": (
                        str(row.get("error"))
                        if row.get("error") not in (None, "")
                        else None
                    ),
                    "x_channel": str(row.get("x_channel") or ""),
                    "y_channel": str(row.get("y_channel") or ""),
                    "boundary_ratios": self._float_list_or_none(
                        row.get("boundary_ratios")
                    ),
                    "boundary_values": self._float_list_or_none(
                        row.get("boundary_values")
                    ),
                    "pattern_error": (
                        str(row.get("pattern_error"))
                        if row.get("pattern_error") not in (None, "")
                        else None
                    ),
                }
            )
        return rows

    def _batch_table_export_matrix(self):
        """Return the same columns/rows used by Export CSV."""
        param_columns = self._batch_parameter_column_items()
        columns = (
            ["File"]
            + self.batch_capture_keys
            + ["R2"]
            + [str(item["key"]) for item in param_columns]
            + ["Error"]
        )
        rows = []
        for row in list(getattr(self, "batch_results", []) or []):
            file_name = stem_for_file_ref(row.get("file", ""))
            captures = dict(row.get("captures") or {})
            params = self._as_float_array(row.get("params"))
            boundary_values = self._as_float_array(row.get("boundary_values"))
            r2_val = row.get("r2")
            error_text = self._batch_row_error_text(row)
            param_values = []
            for item in param_columns:
                idx = int(item["index"])
                if item["kind"] == "param":
                    value = float(params[idx]) if params.size > idx else ""
                else:
                    value = (
                        float(boundary_values[idx])
                        if boundary_values.size > idx
                        else ""
                    )
                if isinstance(value, (float, np.floating)):
                    param_values.append(f"{float(value):.6f}")
                else:
                    param_values.append(value)
            rows.append(
                [file_name]
                + [captures.get(key, "") for key in self.batch_capture_keys]
                + [f"{r2_val:.6f}" if r2_val is not None else ""]
                + param_values
                + [error_text]
            )
        return columns, rows

    def _table_export_rows_to_batch_results(self, payload):
        """Convert exported CSV-style table payload back into batch row objects."""
        table_payload = dict(payload.get("batch_table_export") or {})
        columns = [str(col) for col in list(table_payload.get("columns") or [])]
        rows = list(table_payload.get("rows") or [])
        if not columns or not rows:
            return []
        if "File" not in columns or "R2" not in columns or "Error" not in columns:
            return []
        try:
            file_idx = columns.index("File")
            r2_idx = columns.index("R2")
            error_idx = columns.index("Error")
        except Exception:
            return []

        capture_keys = columns[file_idx + 1 : r2_idx]
        param_key_to_item = {
            str(item["key"]): dict(item)
            for item in self._batch_parameter_column_items()
        }
        param_cols = columns[r2_idx + 1 : error_idx]
        current_params = list(self.get_current_params())
        imported = []
        for row_values in rows:
            values = list(row_values) if isinstance(row_values, (list, tuple)) else []
            if len(values) < len(columns):
                values += [""] * (len(columns) - len(values))
            file_stem = str(values[file_idx]).strip()
            if not file_stem:
                continue
            captures = {
                str(capture_key): str(values[file_idx + 1 + idx]).strip()
                for idx, capture_key in enumerate(capture_keys)
            }
            params_map = {}
            boundaries_map = {}
            for idx, col_name in enumerate(param_cols, start=r2_idx + 1):
                item = param_key_to_item.get(str(col_name))
                if item is None:
                    continue
                numeric = self._json_float_or_none(values[idx])
                if numeric is None:
                    continue
                if item.get("kind") == "param":
                    params_map[int(item["index"])] = float(numeric)
                else:
                    boundaries_map[int(item["index"])] = float(numeric)

            param_values = []
            for i in range(len(self.param_specs)):
                if i in params_map:
                    param_values.append(float(params_map[i]))
                else:
                    fallback = current_params[i] if i < len(current_params) else 0.0
                    param_values.append(float(fallback))

            expected_boundaries = max(
                0,
                len(self._piecewise_model.segment_exprs) - 1
                if self._piecewise_model is not None
                else 0,
            )
            boundary_values = []
            for i in range(expected_boundaries):
                if i in boundaries_map:
                    boundary_values.append(float(boundaries_map[i]))

            error_text = (
                str(values[error_idx]).strip() if error_idx < len(values) else ""
            )
            imported.append(
                {
                    "file_stem": file_stem,
                    "captures": captures,
                    "params": param_values,
                    "r2": self._json_float_or_none(values[r2_idx]),
                    "error": error_text if error_text else None,
                    "boundary_values": (
                        boundary_values
                        if len(boundary_values) == expected_boundaries
                        else None
                    ),
                }
            )
        return imported

    def _collect_fit_details_payload(self):
        ratios = self._float_list_or_none(getattr(self, "current_boundary_ratios", []))
        table_columns, table_rows = self._batch_table_export_matrix()
        payload = {
            "format": "manual_fit_gui_details",
            "version": 1,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "gui": {
                "equation": str(getattr(self, "current_expression", "")).strip(),
                "x_channel": str(getattr(self, "x_channel", "")).strip(),
                "y_channel": str(getattr(self, "y_channel", "")).strip(),
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
                "boundary_ratios": ratios if ratios is not None else [],
            },
            "parameters": self._serialize_fit_parameter_specs(),
            "batch_results": self._serialize_fit_batch_rows(),
            "batch_table_export": {
                "columns": table_columns,
                "rows": table_rows,
            },
        }
        return payload

    def _write_fit_details_file(self, file_path, *, quiet):
        path = Path(file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._collect_fit_details_payload()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
        if not quiet:
            self.stats_text.append(f"✓ Saved fit details to {path}")
        return True

    def _autosave_fit_details(self):
        if bool(getattr(self, "_fit_details_restore_in_progress", False)):
            return False
        sidecar = self._fit_details_sidecar_path()
        if sidecar is None:
            return False
        try:
            return self._write_fit_details_file(sidecar, quiet=True)
        except Exception as exc:
            self.stats_text.append(f"✗ Auto-save fit details failed: {exc}")
            return False

    def export_fit_details(self):
        sidecar = self._fit_details_sidecar_path()
        start_path = (
            sidecar if sidecar is not None else Path.cwd() / FIT_DETAILS_FILENAME
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Details",
            str(start_path),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            self._write_fit_details_file(file_path, quiet=False)
        except Exception as exc:
            self.stats_text.append(f"✗ Export fit details failed: {exc}")

    def _resolve_import_file_ref(self, row_data):
        file_ref = str(row_data.get("file") or "").strip()
        if file_ref and file_ref in self.data_files:
            return file_ref
        stem = str(row_data.get("file_stem") or "").strip()
        if not stem and file_ref:
            stem = stem_for_file_ref(file_ref)
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
        if not imported_rows and isinstance(payload.get("batch"), Mapping):
            imported_rows = list(payload.get("batch", {}).get("results") or [])
        if not imported_rows:
            imported_rows = self._table_export_rows_to_batch_results(payload)
        if not imported_rows:
            return (0, 0)
        if not self.data_files:
            return (0, len(imported_rows))

        existing_by_file = {
            str(row.get("file")): dict(row)
            for row in list(getattr(self, "batch_results", []) or [])
            if str(row.get("file") or "").strip()
        }
        for source_index, file_ref in enumerate(self.data_files):
            if file_ref not in existing_by_file:
                existing_by_file[file_ref] = make_batch_result_row(
                    source_index=source_index,
                    file_path=file_ref,
                    x_channel=self.x_channel,
                    y_channel=self.y_channel,
                    captures={},
                )

        applied = 0
        skipped = 0
        expected_boundaries = max(
            0,
            len(self._piecewise_model.segment_exprs) - 1
            if self._piecewise_model is not None
            else 0,
        )
        expected_params = len(self.param_specs)

        for raw_row in imported_rows:
            if not isinstance(raw_row, Mapping):
                skipped += 1
                continue
            file_ref = self._resolve_import_file_ref(raw_row)
            if not file_ref:
                skipped += 1
                continue
            row = dict(existing_by_file.get(file_ref) or {})
            row["file"] = file_ref
            row["captures"] = dict(raw_row.get("captures") or row.get("captures") or {})
            row["x_channel"] = self.x_channel
            row["y_channel"] = self.y_channel
            row["error"] = (
                str(raw_row.get("error"))
                if raw_row.get("error") not in (None, "")
                else None
            )
            row["pattern_error"] = (
                str(raw_row.get("pattern_error"))
                if raw_row.get("pattern_error") not in (None, "")
                else row.get("pattern_error")
            )
            row["r2"] = self._json_float_or_none(raw_row.get("r2"))

            params = self._as_float_array(raw_row.get("params"))
            if params.size > 0:
                if params.size >= expected_params:
                    row["params"] = [
                        float(params[idx]) for idx in range(expected_params)
                    ]
                else:
                    padded = list(self.get_current_params())
                    for idx, value in enumerate(params.tolist()):
                        if idx < len(padded):
                            padded[idx] = float(value)
                    row["params"] = padded

            boundary_ratios = self._as_float_array(raw_row.get("boundary_ratios"))
            if boundary_ratios.size == expected_boundaries:
                row["boundary_ratios"] = np.clip(boundary_ratios, 0.0, 1.0)

            boundary_values = self._as_float_array(raw_row.get("boundary_values"))
            if boundary_values.size == expected_boundaries:
                row["boundary_values"] = boundary_values

            row["plot_full"] = None
            row["plot"] = None
            row["plot_render_size"] = None
            row["plot_has_fit"] = has_nonempty_values(row.get("params"))
            if row.get("plot_has_fit"):
                row["_equation_stale"] = False
            row = self._apply_param_range_validation_to_row(row)
            existing_by_file[file_ref] = row
            applied += 1

        ordered_rows = []
        for source_index, file_ref in enumerate(self.data_files):
            row = dict(existing_by_file.get(file_ref) or {})
            row["_source_index"] = int(source_index)
            row["file"] = file_ref
            row["x_channel"] = self.x_channel
            row["y_channel"] = self.y_channel
            ordered_rows.append(row)

        self.batch_results = ordered_rows
        self._rebuild_batch_capture_keys_from_rows()
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        self.queue_visible_thumbnail_render()
        current_file = self._current_loaded_file_path()
        if current_file and self._apply_batch_params_for_file(current_file):
            self.update_plot(fast=False)
        return applied, skipped

    def _apply_fit_details_payload(self, payload, *, source_path=None, auto_load=False):
        if not isinstance(payload, Mapping):
            raise ValueError("Fit details file must contain a JSON object.")

        gui = dict(payload.get("gui") or {})
        if not gui and isinstance(payload.get("settings"), Mapping):
            gui = dict(payload.get("settings") or {})
        expression_text = str(
            gui.get("equation")
            or gui.get("expression")
            or payload.get("equation")
            or payload.get("expression")
            or ""
        ).strip()

        self._fit_details_restore_in_progress = True
        try:
            expression_applied = False
            if expression_text:
                self.current_expression = expression_text
                self._set_expression_editor_text(expression_text)
                if not self.apply_expression_from_input():
                    raise ValueError("Failed to apply stored equation.")
                expression_applied = True

            imported_params = list(payload.get("parameters") or [])
            if imported_params:
                spec_by_key = {}
                value_by_key = {}
                for entry in imported_params:
                    if not isinstance(entry, Mapping):
                        continue
                    key = str(entry.get("key") or "").strip()
                    if not key:
                        continue
                    fallback = next(
                        (spec for spec in self.param_specs if spec.key == key), None
                    )
                    if fallback is None:
                        continue
                    min_value = self._json_float_or_none(entry.get("min_value"))
                    max_value = self._json_float_or_none(entry.get("max_value"))
                    if min_value is None:
                        min_value = float(fallback.min_value)
                    if max_value is None:
                        max_value = float(fallback.max_value)
                    if min_value > max_value:
                        min_value, max_value = max_value, min_value
                    default_value = self._json_float_or_none(entry.get("default"))
                    if default_value is None:
                        default_value = float(fallback.default)
                    default_value = float(np.clip(default_value, min_value, max_value))
                    decimals = entry.get("decimals")
                    try:
                        decimals = int(decimals)
                    except Exception:
                        decimals = int(fallback.decimals)
                    decimals = max(0, min(12, decimals))
                    spec_by_key[key] = ParameterSpec(
                        key=key,
                        symbol=str(entry.get("symbol") or fallback.symbol),
                        description=str(
                            entry.get("description") or fallback.description
                        ),
                        default=default_value,
                        min_value=min_value,
                        max_value=max_value,
                        decimals=decimals,
                    )
                    value_by_key[key] = {
                        "value": entry.get("value"),
                    }

                if spec_by_key:
                    merged_specs = []
                    for spec in self.param_specs:
                        merged_specs.append(spec_by_key.get(spec.key, spec))
                    self.param_specs = merged_specs
                    self.defaults = self._default_param_values(self.param_specs)
                    self.rebuild_manual_param_controls()

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
                        value = self._json_float_or_none(state.get("value"))
                        if value is None:
                            value = float(spec.default)
                        spinbox.setValue(float(np.clip(value, low, high)))

            x_channel = str(gui.get("x_channel") or "").strip()
            y_channel = str(gui.get("y_channel") or "").strip()
            channel_names = list(self._available_channel_names())
            if x_channel in channel_names and hasattr(self, "x_channel_combo"):
                idx = self.x_channel_combo.findData(x_channel)
                if idx >= 0:
                    self.x_channel_combo.setCurrentIndex(idx)
            if (
                (not expression_applied)
                and y_channel in channel_names
                and hasattr(self, "y_channel_combo")
            ):
                idx = self.y_channel_combo.findData(y_channel)
                if idx >= 0:
                    self.y_channel_combo.setCurrentIndex(idx)

            if hasattr(self, "smoothing_toggle_btn"):
                self.smoothing_toggle_btn.setChecked(
                    bool(gui.get("smoothing_enabled", self.smoothing_enabled))
                )
            if hasattr(self, "smoothing_window_spin"):
                window = int(gui.get("smoothing_window", self.smoothing_window) or 1)
                self.smoothing_window_spin.setValue(max(1, window))
            self._on_smoothing_controls_changed()

            if hasattr(self, "regex_input"):
                pattern_text = str(gui.get("capture_pattern") or "").strip()
                self.regex_input.blockSignals(True)
                self.regex_input.setText(pattern_text)
                self.regex_input.blockSignals(False)

            mapping = dict(gui.get("capture_to_param") or {})
            self.param_capture_map = {
                str(key): (str(value) if value not in (None, "") else None)
                for key, value in mapping.items()
            }
            self._refresh_param_capture_mapping_controls()

            ratio_values = gui.get("boundary_ratios")
            if ratio_values is not None:
                ratios = self._as_float_array(ratio_values)
                expected = max(
                    0,
                    len(self._piecewise_model.segment_exprs) - 1
                    if self._piecewise_model is not None
                    else 0,
                )
                if ratios.size == expected:
                    self.current_boundary_ratios = np.clip(ratios, 0.0, 1.0)
                    self._sync_breakpoint_sliders_from_state()

            applied_rows, skipped_rows = self._apply_imported_batch_rows(payload)

            if source_path is not None:
                load_mode = "Auto-loaded" if auto_load else "Imported"
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
            self.update_plot(fast=False)
        finally:
            self._fit_details_restore_in_progress = False

    def _load_fit_details_file(self, file_path, *, auto_load):
        path = Path(file_path).expanduser()
        if not path.exists():
            if not auto_load:
                self.stats_text.append(f"Fit details file not found: {path}")
            return False
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            self.stats_text.append(f"✗ Failed to read fit details: {exc}")
            return False
        try:
            self._apply_fit_details_payload(
                payload, source_path=path, auto_load=auto_load
            )
            return True
        except Exception as exc:
            self.stats_text.append(f"✗ Failed to apply fit details: {exc}")
            return False

    def import_fit_details(self):
        sidecar = self._fit_details_sidecar_path()
        start_dir = (
            sidecar.parent if sidecar is not None else self._source_dialog_start_dir()
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Fit Details",
            str(start_dir / FIT_DETAILS_FILENAME),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not file_path:
            return
        self._load_fit_details_file(file_path, auto_load=False)

    def _autoload_fit_details_from_source(self):
        if bool(getattr(self, "_fit_details_restore_in_progress", False)):
            return False
        sidecar = self._fit_details_sidecar_path()
        if sidecar is None or not sidecar.exists():
            return False
        return self._load_fit_details_file(sidecar, auto_load=True)

    def _upsert_batch_row_from_fit(
        self,
        file_path,
        ordered_keys,
        fit_result,
    ):
        if not file_path:
            return None
        params_by_key = dict((fit_result or {}).get("params_by_key") or {})
        row_params = [float(params_by_key.get(key, 0.0)) for key in ordered_keys]
        row_r2 = (
            float(fit_result["r2"])
            if fit_result is not None and fit_result.get("r2") is not None
            else None
        )
        boundary_vals = (
            fit_result.get("boundary_ratios") if isinstance(fit_result, dict) else None
        )
        if boundary_vals is not None:
            try:
                boundary_vals = np.asarray(boundary_vals, dtype=float).reshape(-1)
            except Exception:
                boundary_vals = None
        boundary_x_vals = None
        if boundary_vals is not None:
            try:
                x_values = self._get_channel_data(self.x_channel)
                n_boundaries = (
                    max(0, len(self._piecewise_model.segment_exprs) - 1)
                    if self._piecewise_model is not None
                    else int(np.asarray(boundary_vals, dtype=float).size)
                )
                boundary_x_vals = boundary_ratios_to_x_values(
                    boundary_vals,
                    x_values,
                    n_boundaries,
                )
            except Exception:
                boundary_x_vals = None

        row_idx = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            captures = {}
            pattern_error = None
            capture_config = self._resolve_batch_capture_config(show_errors=False)
            if capture_config is not None:
                extracted = extract_captures(
                    stem_for_file_ref(file_path),
                    capture_config.regex,
                    capture_config.defaults,
                )
                if extracted is None:
                    pattern_error = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures = dict(extracted)
            row = make_batch_result_row(
                source_index=len(self.batch_results),
                file_path=file_path,
                x_channel=self.x_channel,
                y_channel=self.y_channel,
                captures=captures,
                pattern_error=pattern_error,
            )
            self.batch_results.append(row)
            row_idx = len(self.batch_results) - 1
        else:
            row = self.batch_results[row_idx]

        row["params"] = list(row_params)
        row["r2"] = row_r2
        row["boundary_ratios"] = boundary_vals
        row["boundary_values"] = boundary_x_vals
        row["x_channel"] = self.x_channel
        row["y_channel"] = self.y_channel
        row["plot_full"] = None
        row["plot"] = None
        row["plot_render_size"] = None
        row["plot_has_fit"] = True
        row["_equation_stale"] = False
        row = self._apply_param_range_validation_to_row(row)
        self.batch_results[row_idx] = row
        self._rebuild_batch_capture_keys_from_rows()

        if self.batch_table.rowCount() != len(self.batch_results):
            self.update_batch_table()
        else:
            table_row_idx = self._find_table_row_by_file(file_path)
            if table_row_idx is None:
                self.update_batch_table()
            else:
                self.update_batch_table_row(table_row_idx, row)
        self._start_thumbnail_render(row_indices=[row_idx])
        return row_idx

    def _on_batch_row_resized_by_user(self, _logical_index, _old_size, new_size):
        if self._batch_row_height_sync:
            return
        self.batch_row_height = max(
            self.batch_row_height_min,
            min(self.batch_row_height_max, int(new_size)),
        )
        self._apply_batch_row_heights()
        for row in self.batch_results:
            row_idx = self._find_table_row_by_file(row["file"])
            if row_idx is not None:
                self._update_batch_plot_cell(row_idx, row)
        self.queue_visible_thumbnail_render()

    def _sync_batch_files_from_shared(self, sync_pattern=True):
        """Mirror batch files from shared file list (default: all files in folder)."""
        self.batch_files = list(self.data_files)

        if sync_pattern and hasattr(self, "regex_input") and self.batch_files:
            first_name = stem_for_file_ref(self.batch_files[0])
            if self.regex_input.text() != first_name:
                self.regex_input.blockSignals(True)
                self.regex_input.setText(first_name)
                self.regex_input.blockSignals(False)

        preview_needed_now = bool(self._batch_preview_ready)
        if hasattr(self, "tabs"):
            preview_needed_now = preview_needed_now or (
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
        existing_row = existing or {}
        existing_plot_full = None
        existing_plot = None
        existing_plot_render_size = None
        existing_plot_has_fit = existing_row.get("plot_has_fit")
        if existing_plot_has_fit is None:
            existing_plot_has_fit = has_nonempty_values(existing_row.get("params"))
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
            y_channel=self.y_channel,
            captures=captures,
            params=existing_row.get("params") if preserve_fit_result else None,
            r2=existing_row.get("r2") if preserve_fit_result else None,
            error=existing_row.get("error") if preserve_fit_result else None,
            plot_full=existing_plot_full,
            plot=existing_plot,
            plot_has_fit=(
                bool(existing_plot_has_fit) if existing_plot_full is not None else None
            ),
            plot_render_size=(
                existing_plot_render_size if existing_plot_full is not None else None
            ),
            boundary_ratios=(
                existing_row.get("boundary_ratios") if preserve_fit_result else None
            ),
            boundary_values=(
                existing_row.get("boundary_values") if preserve_fit_result else None
            ),
            pattern_error=pattern_error,
            equation_stale=bool(existing_row.get("_equation_stale")),
        )
        return self._apply_param_range_validation_to_row(row)

    def _source_dialog_start_dir(self):
        current_path = Path(self.current_dir).expanduser()
        start_dir = current_path.parent if current_path.is_file() else current_path
        if not start_dir.exists():
            selected = list(getattr(self, "_source_selected_paths", []) or [])
            if selected:
                selected_parent = Path(selected[0]).expanduser().parent
                if selected_parent.exists():
                    start_dir = selected_parent
        if not start_dir.exists():
            start_dir = Path.cwd()
        return start_dir

    def _confirm_clear_batch_results(self, action_label):
        rows = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return True

        action_text = str(action_label or "this action").strip() or "this action"
        row_count = len(rows)
        plural = "s" if row_count != 1 else ""
        reply = QMessageBox.question(
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

    def _prompt_batch_results_on_rerun(self):
        rows = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return "keep"

        row_count = len(rows)
        plural = "s" if row_count != 1 else ""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Run Batch")
        dialog.setText(f"Batch rerun found {row_count} existing result{plural}.")
        dialog.setInformativeText(
            "Choose whether to keep existing results for comparison/seeding, or clear them first."
        )

        keep_btn = dialog.addButton(
            "Proceed and Keep", QMessageBox.ButtonRole.AcceptRole
        )
        clear_btn = dialog.addButton(
            "Proceed and Clear", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(keep_btn)

        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == cancel_btn:
            return "cancel"
        if clicked == clear_btn:
            return "clear"
        return "keep"

    def _prompt_batch_results_on_equation_change(self):
        rows = list(getattr(self, "batch_results", []) or [])
        if not rows:
            return "wipe"

        row_count = len(rows)
        plural = "s" if row_count != 1 else ""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Equation Changed")
        dialog.setText(
            f"Changing the equation affects {row_count} existing batch result{plural}."
        )
        dialog.setInformativeText(
            "Choose whether to keep old batch results or clear them before continuing."
        )

        keep_btn = dialog.addButton(
            "Proceed and Keep", QMessageBox.ButtonRole.AcceptRole
        )
        wipe_btn = dialog.addButton(
            "Proceed and Clear", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(keep_btn)

        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == cancel_btn:
            return "cancel"
        if clicked == wipe_btn:
            return "wipe"
        return "keep"

    def _apply_data_file_list(self, files, *, empty_message):
        deduped_files = []
        seen = set()
        for file_ref in files:
            text = str(file_ref).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped_files.append(text)

        if not self._confirm_clear_batch_results("Loading a new data source"):
            self.stats_text.append("Load cancelled; existing batch results kept.")
            return False

        self.data_files = deduped_files
        self.file_combo.clear()
        self.current_file_idx = 0

        if not self.data_files:
            self.current_data = None
            self.cached_time_data = None
            self.raw_channel_cache = {}
            self.channel_cache = {}
            self._expression_channel_data_cache = None
            self._last_file_load_error = ""
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
        for idx in range(len(self.data_files)):
            if self.load_file(idx, report_errors=False):
                loaded_ok = True
                loaded_idx = idx
                break

        if loaded_ok:
            if loaded_idx > 0:
                loaded_name = stem_for_file_ref(self.data_files[loaded_idx])
                self.stats_text.setText(
                    f"Loaded '{loaded_name}' after skipping {loaded_idx} unreadable source file(s)."
                )
            elif self.stats_text.text().strip() == "Loading data sources...":
                self.stats_text.clear()
            self._autoload_fit_details_from_source()
            self._sync_file_navigation_buttons()
            return True

        detail = self._last_file_load_error or "No readable data found."
        self.stats_text.setText(f"Failed to load any selected data source. {detail}")
        self._clear_main_plot("No readable data loaded.")
        self._sync_file_navigation_buttons()
        return False

    def _load_selected_csv_files(self, csv_paths):
        selected_csv = []
        for path_text in csv_paths:
            path_obj = Path(path_text).expanduser()
            if path_obj.is_file() and path_obj.suffix.lower() == ".csv":
                selected_csv.append(str(path_obj))

        if not selected_csv:
            self.stats_text.append("No valid CSV files were selected.")
            return False

        self.current_dir = str(Path(selected_csv[0]).parent)
        count = len(selected_csv)
        plural = "s" if count != 1 else ""
        self._source_display_override = f"{count} selected CSV file{plural}"
        self._source_selected_paths = list(selected_csv)
        self._refresh_source_path_label()
        loaded_ok = self._apply_data_file_list(
            selected_csv,
            empty_message="No readable CSV files found in selected set.",
        )
        if loaded_ok:
            self.stats_text.append(f"Loaded {count} selected CSV file{plural}.")
        return loaded_ok

    def load_files(self):
        """Load CSV sources from a directory root, ZIP archive, or single CSV file."""
        source_path = Path(self.current_dir).expanduser()
        files = []
        empty_message = "No CSV files found in selected source."

        self._source_display_override = None
        self._source_selected_paths = []

        if source_path.is_dir():
            csv_files = []
            zip_paths = []
            for candidate in sorted(source_path.rglob("*")):
                if not candidate.is_file():
                    continue
                suffix = candidate.suffix.lower()
                if suffix == ".csv":
                    csv_files.append(str(candidate))
                elif suffix == ".zip":
                    zip_paths.append(candidate)

            zip_members = []
            for zip_path in zip_paths:
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        for member in sorted(zf.namelist()):
                            if member.lower().endswith(".csv") and not member.endswith(
                                "/"
                            ):
                                zip_members.append(f"{zip_path}::{member}")
                except Exception:
                    continue
            files = csv_files + zip_members
        elif source_path.is_file() and source_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(source_path) as zf:
                    files = [
                        f"{source_path}::{member}"
                        for member in sorted(zf.namelist())
                        if member.lower().endswith(".csv") and not member.endswith("/")
                    ]
            except Exception as exc:
                empty_message = f"Failed to read ZIP root: {exc}"
                files = []
        elif source_path.is_file() and source_path.suffix.lower() == ".csv":
            files = [str(source_path)]
        elif not source_path.exists():
            empty_message = f"Selected source does not exist: {source_path}"

        self._refresh_source_path_label()
        self._apply_data_file_list(files, empty_message=empty_message)

    def _clear_main_plot(self, message="No data loaded."):
        if not hasattr(self, "fig") or not hasattr(self, "canvas"):
            return
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax_residual = None
        self._plot_has_residual_axis = False
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel(
            self._channel_axis_label(self.x_channel)
            if hasattr(self, "x_channel")
            else "X"
        )
        self.ax.set_ylabel(
            self._channel_axis_label(self.y_channel)
            if hasattr(self, "y_channel")
            else "Signal"
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

    def load_file(self, idx, *, report_errors=True):
        """Load a specific file."""
        if self._file_load_in_progress:
            return False
        if idx < 0 or idx >= len(self.data_files):
            return False

        self._file_load_in_progress = True
        loaded_ok = False
        self._last_file_load_error = ""
        try:
            self.current_file_idx = idx
            if self.file_combo.currentIndex() != idx:
                self.file_combo.blockSignals(True)
                self.file_combo.setCurrentIndex(idx)
                self.file_combo.blockSignals(False)
            self.last_popt = None
            self.last_pcov = None
            self.last_r2 = None
            self._last_r2 = None

            try:
                file_path = self.data_files[idx]
                self.current_data = read_measurement_csv(file_path)
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
                self._refresh_param_capture_mapping_controls()
                self._apply_batch_params_for_file(file_path)
                self.update_plot(fast=False, preserve_view=False)
                self._reset_plot_home_view()
                loaded_ok = True
            except Exception as e:
                self.current_data = None
                self.cached_time_data = None
                self.raw_channel_cache = {}
                self.channel_cache = {}
                self._expression_channel_data_cache = None
                file_path = self.data_files[idx]
                file_name = stem_for_file_ref(file_path)
                self._last_file_load_error = f"Error loading '{file_name}': {e}"
                if report_errors:
                    self.stats_text.setText(self._last_file_load_error)
                    self._clear_main_plot(f"Failed to load: {file_name}")
        finally:
            self._file_load_in_progress = False
            self._sync_file_navigation_buttons()
            self._refresh_fit_action_buttons()
        return loaded_ok

    def on_file_changed(self, idx):
        """Handle file selection change."""
        if self._file_load_in_progress:
            return
        if idx >= 0:
            self.load_file(idx)

    def prev_file(self):
        """Load previous file."""
        if self.current_file_idx > 0:
            self.load_file(self.current_file_idx - 1)

    def next_file(self):
        """Load next file."""
        if self.current_file_idx < len(self.data_files) - 1:
            self.load_file(self.current_file_idx + 1)

    def get_current_params(self):
        """Get current parameter values."""
        return [
            float(self.param_spinboxes[spec.key].value()) for spec in self.param_specs
        ]

    def get_batch_params(self):
        """Get batch fit initial parameters from shared controls."""
        return self.get_current_params()

    def _apply_batch_params_for_file(self, file_path):
        """Apply batch-fitted parameters for this file if available."""
        if not file_path or not getattr(self, "batch_results", None):
            return False

        matched_row = None
        for row in self.batch_results:
            if row.get("file") != file_path:
                continue
            params = row.get("params")
            if params is None:
                continue
            try:
                if len(params) <= 0:
                    continue
            except Exception:
                continue
            matched_row = row
            break
        if matched_row is None:
            return False

        params = matched_row.get("params")
        if params is None:
            return False
        try:
            params = list(np.asarray(params, dtype=float).reshape(-1))
        except Exception:
            return False
        if len(params) < len(self.param_specs):
            return False

        changed = False
        boundary_changed = False
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
            spinbox.setValue(value)
        boundary_ratios = matched_row.get("boundary_ratios")
        if boundary_ratios is not None:
            try:
                b = np.asarray(boundary_ratios, dtype=float).reshape(-1)
            except Exception:
                b = np.asarray([], dtype=float)
            expected = max(
                0,
                len(self._piecewise_model.segment_exprs) - 1
                if self._piecewise_model is not None
                else 0,
            )
            if b.size == expected:
                new_boundary = np.clip(b, 0.0, 1.0)
                old_boundary = np.asarray(
                    getattr(
                        self,
                        "current_boundary_ratios",
                        default_boundary_ratios(expected),
                    ),
                    dtype=float,
                ).reshape(-1)
                if old_boundary.size != expected or not np.allclose(
                    old_boundary,
                    new_boundary,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    boundary_changed = True
                self.current_boundary_ratios = new_boundary
        self._sync_breakpoint_sliders_from_state()
        self._refresh_param_value_error_highlighting()
        return bool(changed or boundary_changed)

    def reset_params(self):
        """Reset all parameter values to midpoint of their current bounds."""
        for spec in self.param_specs:
            spinbox = self.param_spinboxes.get(spec.key)
            if spinbox is None:
                continue
            value = float((float(spinbox.minimum()) + float(spinbox.maximum())) * 0.5)
            spinbox.setValue(value)

    def reset_params_from_batch(self):
        """Load parameter values for the current file from batch results."""
        current_file = self._current_loaded_file_path()
        if not current_file:
            self.stats_text.append("No current file loaded.")
            return
        if self._is_file_fit_active(current_file):
            self.stats_text.append("Cannot reset from batch while a fit is running.")
            return
        row_idx = self._find_batch_result_index_by_file(current_file)
        if row_idx is None or not has_nonempty_values(
            self.batch_results[row_idx].get("params")
        ):
            self.stats_text.append("No batch-fit parameters found for this file.")
            return
        changed = self._apply_batch_params_for_file(current_file)
        if changed:
            self.update_plot(fast=False)
            self.stats_text.append(
                "Loaded parameters from batch table for current file."
            )
        else:
            self.stats_text.append(
                "Current parameters already match batch table values."
            )

    def do_full_update(self):
        """Perform a complete update including stats."""
        self.update_plot(fast=False)

    def browse_directory(self):
        """Choose source mode and load from folder, ZIP, or selected CSV files."""
        start_dir = self._source_dialog_start_dir()

        source_menu = QMenu(self)
        folder_action = source_menu.addAction("Load Folder...")
        csv_action = source_menu.addAction("Load CSV File(s)...")
        zip_action = source_menu.addAction("Load ZIP Archive...")

        if hasattr(self, "source_path_label"):
            anchor = self.source_path_label.mapToGlobal(
                self.source_path_label.rect().bottomLeft()
            )
        else:
            anchor = self.mapToGlobal(self.rect().center())
        selected_action = source_menu.exec(anchor)
        if selected_action is None:
            return
        if selected_action not in {folder_action, csv_action, zip_action}:
            return

        if selected_action == folder_action:
            selected_dir = QFileDialog.getExistingDirectory(
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

        if selected_action == zip_action:
            selected_zip, _ = QFileDialog.getOpenFileName(
                self,
                "Select ZIP Archive",
                str(start_dir),
                "ZIP Archives (*.zip);;All Files (*.*)",
            )
            if not selected_zip:
                return
            chosen_zip = Path(selected_zip).expanduser()
            if not chosen_zip.exists():
                self.stats_text.append(f"Selected path does not exist: {chosen_zip}")
                return
            self.current_dir = str(chosen_zip)
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

    def auto_fit(self):
        """Start the default piecewise auto-fit in a worker thread."""
        self._start_auto_fit()

    def _start_auto_fit(self):
        """Start auto-fit in a worker thread to keep GUI responsive."""
        if self.current_data is None:
            self.stats_text.append("No data loaded!")
            return

        current_file = self._current_loaded_file_path()
        if not current_file:
            self.stats_text.append("No current file loaded!")
            return

        active_tasks = self._active_fit_tasks_for_file(current_file)
        active_manual_tasks = [
            meta for meta in active_tasks if str(meta.get("kind")) == "manual"
        ]
        if active_manual_tasks:
            self.cancel_auto_fit()
            return
        if active_tasks:
            self.stats_text.append("Current file already has a fit in progress.")
            return

        fixed_params, mapping_error = self._current_file_fixed_params_from_mapping()
        if mapping_error:
            self.stats_text.append(f"Fit setup error: {mapping_error}")
            return

        try:
            # Seed from current UI controls so refits follow the slider state.
            fit_context = self._build_fit_context(
                fixed_params=fixed_params,
            )
        except Exception as exc:
            self.stats_text.append(f"Fit setup error: {exc}")
            return

        capture_config = self._resolve_batch_capture_config(show_errors=False)
        if capture_config is None:
            capture_config = parse_capture_pattern("")

        source_index = int(getattr(self, "current_file_idx", 0))
        self._start_file_fit_task(
            kind="manual",
            file_path=current_file,
            source_index=source_index,
            fit_context=fit_context,
            capture_regex_pattern=capture_config.regex_pattern,
            capture_defaults=capture_config.defaults,
            parameter_capture_map=self._current_param_capture_map(),
        )
        self.stats_text.append(
            f"Auto-fit started for {stem_for_file_ref(current_file)} (full trace)."
        )
        self._refresh_fit_action_buttons()

    def on_fit_finished(self, fit_result):
        """Handle successful fit completion."""
        model_def = self._piecewise_model
        if model_def is None:
            self.on_fit_failed("No compiled piecewise model.")
            return
        ordered_keys = list(model_def.global_param_names)
        params_by_key = dict((fit_result or {}).get("params_by_key") or {})
        best_params = np.asarray(
            [float(params_by_key.get(key, 0.0)) for key in ordered_keys], dtype=float
        )
        self.last_popt = best_params
        self._last_fit_active_keys = list(ordered_keys)
        self.last_pcov = None
        self.last_r2 = (
            float(fit_result["r2"])
            if fit_result is not None and fit_result.get("r2") is not None
            else None
        )
        self._last_r2 = self.last_r2
        if (
            isinstance(fit_result, dict)
            and fit_result.get("boundary_ratios") is not None
        ):
            self.current_boundary_ratios = np.asarray(
                fit_result.get("boundary_ratios"),
                dtype=float,
            )
        self._sync_breakpoint_sliders_from_state()

        for idx, key in enumerate(ordered_keys):
            if key in self.param_spinboxes and idx < len(self.last_popt):
                self.param_spinboxes[key].setValue(self.last_popt[idx])
        self.defaults = list(self.last_popt)

        r2_text = f"{self.last_r2:.6f}" if self.last_r2 is not None else "N/A"
        violations = self._fit_param_range_violations(best_params)
        range_error = self._fit_param_range_error_text(violations)
        if range_error:
            self.stats_text.append(f"✗ Auto-fit failed: {range_error}")
        else:
            self.stats_text.append(
                f"✓ Auto-fit successful! R² (full trace) = {r2_text}"
            )
        summary = ", ".join(
            f"{self._display_name_for_param_key(key)}={self.last_popt[idx]:.4f}"
            for idx, key in enumerate(ordered_keys)
            if idx < len(self.last_popt)
        )
        self.stats_text.append(summary)

        current_file = self._current_loaded_file_path()
        updated_batch_row = (
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
        self.cleanup_fit_thread()

    def on_fit_failed(self, error_text):
        """Handle fit failures."""
        self.stats_text.append(f"✗ Auto-fit failed: {error_text}")
        self.cleanup_fit_thread()

    def on_fit_cancelled(self):
        """Handle fit cancellation."""
        self.stats_text.append("Auto-fit cancelled.")
        self.cleanup_fit_thread()

    def cancel_auto_fit(self):
        """Request cancellation of an in-flight auto-fit."""
        current_file = self._current_loaded_file_path()
        if not current_file:
            return
        current_key = self._fit_task_file_key(current_file)
        cancelled = False
        for task_id, task in list(self.fit_tasks.items()):
            if str(task.get("kind")) != "manual":
                continue
            task_key = str(
                task.get("file_key") or self._fit_task_file_key(task.get("file_path"))
            )
            if task_key != current_key:
                continue
            if str(task.get("status")) == "pending":
                self._finish_fit_task(int(task_id))
            else:
                self._request_worker_cancel(task.get("worker"))
            cancelled = True
        if cancelled:
            self.stats_text.append("Auto-fit cancellation requested...")

    @staticmethod
    def _request_worker_cancel(worker):
        if worker is None:
            return
        request_cancel = getattr(worker, "request_cancel", None)
        if callable(request_cancel):
            try:
                request_cancel()
            except Exception:
                pass

    @staticmethod
    def _shutdown_thread(thread, wait_ms=None, force_terminate=False):
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

    def cleanup_fit_thread(self):
        """Tear down worker/thread and restore button state."""
        self._shutdown_thread(self.fit_thread)
        if self.fit_worker is not None:
            self.fit_worker.deleteLater()
        self.fit_thread = None
        self.fit_worker = None
        self.auto_fit_btn.setEnabled(True)
        self.auto_fit_btn.setText(self.auto_fit_btn_default_text)
        if hasattr(self, "reset_from_batch_btn"):
            self.reset_from_batch_btn.setEnabled(True)

    def cleanup_batch_thread(self, *, force=False):
        self._shutdown_thread(
            self.batch_thread,
            wait_ms=250,
            force_terminate=True,
        )
        if self.batch_worker is not None:
            self.batch_worker.deleteLater()
        self.batch_thread = None
        self.batch_worker = None
        self.batch_fit_in_progress = False
        self._batch_cancel_pending = False
        self._batch_progress_done = 0
        self.run_batch_btn.setEnabled(True)
        self.run_batch_btn.setText(self.run_batch_btn_default_text)
        self.cancel_batch_btn.setEnabled(False)
        self.cancel_batch_btn.setText("Cancel")
        self.batch_status_label.hide()

    def _get_channel_data(self, channel_name):
        if channel_name in self.channel_cache:
            return self.channel_cache[channel_name]
        if self.current_data is None:
            raise ValueError("No data loaded.")
        if channel_name not in self.current_data.columns:
            raise KeyError(f"Channel '{channel_name}' not found in data.")
        values = self.current_data[channel_name].to_numpy(dtype=float, copy=True)
        self.raw_channel_cache[channel_name] = values
        smoothed = self._smooth_channel_values(values)
        self.channel_cache[channel_name] = smoothed
        return smoothed

    def _display_indices(self, n_points):
        if n_points <= 0:
            return np.asarray([], dtype=int)
        target = max(1000, int(self._display_target_points))
        stride = max(1, int(np.ceil(n_points / float(target))))
        return np.arange(0, n_points, stride, dtype=int)

    def _ensure_plot_axes(self, show_residuals):
        if show_residuals == self._plot_has_residual_axis and hasattr(self, "ax"):
            return
        self.fig.clear()
        if show_residuals:
            grid = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            self.ax = self.fig.add_subplot(grid[0])
            self.ax_residual = self.fig.add_subplot(grid[1], sharex=self.ax)
            self.ax.tick_params(labelbottom=False)
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax_residual = None
        self._plot_has_residual_axis = bool(show_residuals)

    def _finite_min_max(self, *arrays):
        y_min = None
        y_max = None
        for arr in arrays:
            if arr is None:
                continue
            values = np.asarray(arr, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            cur_min = float(np.min(finite))
            cur_max = float(np.max(finite))
            y_min = cur_min if y_min is None else min(y_min, cur_min)
            y_max = cur_max if y_max is None else max(y_max, cur_max)

        if y_min is None or y_max is None:
            return (-1.0, 1.0)
        if np.isclose(y_min, y_max):
            pad = 1.0 if np.isclose(y_min, 0.0) else max(1e-6, abs(y_min) * 0.05)
            return (y_min - pad, y_max + pad)

        pad = (y_max - y_min) * 0.05
        if pad <= 0.0:
            pad = 1.0
        return (y_min - pad, y_max + pad)

    def _apply_unique_legend(self, axis, loc="lower right"):
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

    def _prepare_plot_context(self, params):
        x_data = self._get_channel_data(self.x_channel)
        y_data = self._get_channel_data(self.y_channel)
        n_points = len(x_data)
        if n_points == 0:
            return None

        channel_data_full = self._expression_channel_data()
        display_idx = self._display_indices(n_points)
        if display_idx.size == 0:
            return None

        x_display = x_data[display_idx]
        y_display = y_data[display_idx]
        channel_data_display = self._slice_channel_data(channel_data_full, display_idx)

        plot_channel_names = []
        for name in [self.y_channel]:
            key = str(name).strip()
            if not key or key in plot_channel_names:
                continue
            if key in channel_data_display:
                plot_channel_names.append(key)

        if self.current_data is not None:
            for col in self.current_data.columns:
                key = str(col).strip()
                if not key or key == self.x_channel or key in plot_channel_names:
                    continue
                if key in channel_data_display:
                    plot_channel_names.append(key)
        else:
            for key in channel_data_display.keys():
                key_text = str(key).strip()
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
        }

    def _compute_display_series(self, context):
        fitted_display = self.evaluate_model(
            context["x_display"],
            context["params"],
            channel_data=context["channel_data_display"],
        )
        residuals_display = context["y_display"] - fitted_display

        return {
            "fitted_display": fitted_display,
            "residuals_display": residuals_display,
        }

    def _try_fast_plot_update(self, context, series, show_residuals):
        if not hasattr(self, "_plot_lines"):
            return False
        if show_residuals != self._plot_has_residual_axis:
            return False

        if "fitted" in self._plot_lines:
            self._plot_lines["fitted"].set_ydata(series["fitted_display"])
        if "residuals" in self._plot_lines and show_residuals:
            self._plot_lines["residuals"].set_ydata(series["residuals_display"])

        channel_arrays = list(context.get("plot_channel_displays", {}).values())
        y_min, y_max = self._finite_min_max(*channel_arrays, series["fitted_display"])
        self.ax.set_ylim(y_min, y_max)
        x_vals = np.asarray(context["x_display"], dtype=float)
        x_finite = x_vals[np.isfinite(x_vals)]
        if x_finite.size > 0:
            x_min = float(np.min(x_finite))
            x_max = float(np.max(x_finite))
            if np.isclose(x_min, x_max):
                pad = 1.0 if np.isclose(x_min, 0.0) else max(1e-6, abs(x_min) * 0.05)
                x_min -= pad
                x_max += pad
            self.ax.set_xlim(x_min, x_max)
        if show_residuals and self.ax_residual is not None:
            r_min, r_max = self._finite_min_max(series["residuals_display"])
            self.ax_residual.set_ylim(r_min, r_max)
        self.canvas.draw_idle()
        return True

    def _update_stats_panel(self, r2_value):
        text = f"{r2_value:.6f}" if r2_value is not None else "N/A"
        if hasattr(self, "tab_r2_label"):
            self.tab_r2_label.setText(f"R²: {text}")
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

            show_residuals = self.show_residuals_cb.isChecked()
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
                    main_xlim = tuple(self.ax.get_xlim())
                except Exception:
                    main_xlim = None
                try:
                    main_ylim = tuple(self.ax.get_ylim())
                except Exception:
                    main_ylim = None
                if show_residuals and self.ax_residual is not None:
                    try:
                        residual_ylim = tuple(self.ax_residual.get_ylim())
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

            self._plot_lines = {}
            for idx, (channel_name, values) in enumerate(
                context.get("plot_channel_displays", {}).items()
            ):
                color = palette_color(idx)
                channel_label = self._channel_legend_label(channel_name)
                self.ax.plot(
                    context["x_display"],
                    values,
                    label=channel_label,
                    color=color,
                    linewidth=2.2 if channel_name == self.y_channel else 1.6,
                    alpha=1.0 if channel_name == self.y_channel else 0.9,
                )

            (fitted_line,) = self.ax.plot(
                context["x_display"],
                series["fitted_display"],
                label="Fitted",
                color=FIT_CURVE_COLOR,
                linewidth=2,
            )
            self._plot_lines["fitted"] = fitted_line

            if show_residuals and self.ax_residual is not None:
                (residuals_line,) = self.ax_residual.plot(
                    context["x_display"],
                    series["residuals_display"],
                    label="Residuals",
                    color="black",
                    linestyle=":",
                    linewidth=1.4,
                )
                self._plot_lines["residuals"] = residuals_line
                self.ax_residual.axhline(
                    0.0, color="#6b7280", linewidth=1.0, alpha=0.6, linestyle="--"
                )

            if not fast:
                fitted_full = self.evaluate_model(
                    context["x_data"],
                    params,
                    channel_data=context["channel_data_full"],
                )
                r2_value = compute_r2(context["y_data"], fitted_full)
                self._last_r2 = r2_value
            else:
                r2_value = self._last_r2

            channel_arrays = list(context.get("plot_channel_displays", {}).values())
            y_min, y_max = self._finite_min_max(
                *channel_arrays, series["fitted_display"]
            )
            self.ax.set_ylim(y_min, y_max)
            self._apply_unique_legend(self.ax, loc="lower right")
            self.ax.set_xlabel(
                "" if show_residuals else self._channel_axis_label(self.x_channel)
            )
            self.ax.set_ylabel(self._channel_axis_label(self.y_channel))
            x_vals = np.asarray(context["x_display"], dtype=float)
            x_finite = x_vals[np.isfinite(x_vals)]
            if x_finite.size > 0:
                x_min = float(np.min(x_finite))
                x_max = float(np.max(x_finite))
                if np.isclose(x_min, x_max):
                    pad = (
                        1.0 if np.isclose(x_min, 0.0) else max(1e-6, abs(x_min) * 0.05)
                    )
                    x_min -= pad
                    x_max += pad
                self.ax.set_xlim(x_min, x_max)
            self.ax.grid(True, alpha=0.3)
            if show_residuals and self.ax_residual is not None:
                r_min, r_max = self._finite_min_max(series["residuals_display"])
                self.ax_residual.set_ylim(r_min, r_max)
                self.ax_residual.set_ylabel("Residual")
                self.ax_residual.set_xlabel(self._channel_axis_label(self.x_channel))
                self.ax_residual.grid(True, alpha=0.25)
                self._apply_unique_legend(self.ax_residual, loc="upper right")

            home_main_xlim = None
            home_main_ylim = None
            home_residual_ylim = None
            try:
                home_main_xlim = tuple(self.ax.get_xlim())
            except Exception:
                home_main_xlim = None
            try:
                home_main_ylim = tuple(self.ax.get_ylim())
            except Exception:
                home_main_ylim = None
            if show_residuals and self.ax_residual is not None:
                try:
                    home_residual_ylim = tuple(self.ax_residual.get_ylim())
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
            self._update_stats_panel(r2_value)
        except Exception as e:
            self.stats_text.setText(f"Error updating stats: {e}")
            print(f"Error updating stats: {type(e).__name__}: {e}", file=sys.stderr)

    def _apply_fit_row_update(self, row):
        if has_nonempty_values(row.get("params")):
            row["_equation_stale"] = False
        row = self._apply_param_range_validation_to_row(row)
        row_index = row.get("_source_index")
        if row_index is None:
            for idx, existing_row in enumerate(self.batch_results):
                if existing_row.get("file") == row.get("file"):
                    row_index = idx
                    break
        if row_index is None or row_index < 0 or row_index >= len(self.batch_results):
            self.batch_results.append(dict(row))
            self._rebuild_batch_capture_keys_from_rows()
            self.update_batch_table()
            row_index = len(self.batch_results) - 1
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
            row_has_fit = has_nonempty_values(row.get("params"))
            if (not row_has_fit) and ("_equation_stale" not in row):
                row["_equation_stale"] = bool(existing.get("_equation_stale"))
            existing_plot_has_fit = existing.get("plot_has_fit")
            if existing_plot_has_fit is None:
                existing_plot_has_fit = has_nonempty_values(existing.get("params"))
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
            self.batch_results[row_index] = dict(row)
            table_row_idx = self._find_table_row_by_file(row.get("file"))
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)
            else:
                self.update_batch_table()

        row_has_fit = has_nonempty_values(row.get("params"))
        if row_has_fit and row.get("plot_full") is None:
            self._start_thumbnail_render(row_indices=[int(row_index)])

        current_file = self._current_loaded_file_path()
        if current_file and row.get("file") == current_file and row_has_fit:
            self._apply_batch_params_for_file(current_file)
            self.update_plot(fast=False)
        self._refresh_batch_analysis_if_run()

    def _set_batch_row_runtime_fields(self, file_path, **updates):
        row_idx = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            return False
        row = dict(self.batch_results[row_idx])
        changed = False
        for key, value in updates.items():
            if row.get(key) == value:
                continue
            row[key] = value
            changed = True
        if not changed:
            return False
        self.batch_results[row_idx] = row
        table_row_idx = self._find_table_row_by_file(file_path)
        if table_row_idx is None:
            self.update_batch_table()
        else:
            self.update_batch_table_row(table_row_idx, row)
        return True

    def _upsert_fit_error_row(self, file_path, source_index, error_text):
        row_idx = self._find_batch_result_index_by_file(file_path)
        if row_idx is None:
            captures = {}
            pattern_error = None
            capture_config = self._resolve_batch_capture_config(show_errors=False)
            if capture_config is not None:
                extracted = extract_captures(
                    stem_for_file_ref(file_path),
                    capture_config.regex,
                    capture_config.defaults,
                )
                if extracted is None:
                    pattern_error = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures = dict(extracted)
            row = make_batch_result_row(
                source_index=source_index,
                file_path=file_path,
                x_channel=self.x_channel,
                y_channel=self.y_channel,
                captures=captures,
                pattern_error=pattern_error,
            )
            row_idx = len(self.batch_results)
            self.batch_results.append(row)
        row = dict(self.batch_results[row_idx])
        if not has_nonempty_values(row.get("params")):
            row["error"] = str(error_text)
        self.batch_results[row_idx] = row
        table_row_idx = self._find_table_row_by_file(file_path)
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
    ):
        task_id = self._next_fit_task_id()
        existing_idx = self._find_batch_result_index_by_file(file_path)
        existing_row = (
            self.batch_results[existing_idx] if existing_idx is not None else {}
        )
        if str(kind) == "batch":
            current_file = self._current_loaded_file_path()
            if current_file and str(file_path) == str(current_file):
                overridden = dict(existing_row or {})
                ordered_keys = list(fit_context.get("ordered_keys") or ())
                seed_map = dict(fit_context.get("seed_map") or {})
                if ordered_keys and seed_map:
                    try:
                        overridden["params"] = [
                            float(seed_map[key])
                            for key in ordered_keys
                            if key in seed_map
                        ]
                    except Exception:
                        pass
                boundary_seed = np.asarray(
                    fit_context.get("boundary_seed", []), dtype=float
                ).reshape(-1)
                if boundary_seed.size > 0:
                    overridden["boundary_ratios"] = np.clip(boundary_seed, 0.0, 1.0)
                existing_row = overridden
        if str(kind) == "batch":
            existing_rows_by_file = {
                str(row.get("file")): dict(row)
                for row in list(self.batch_results or [])
                if row.get("file")
            }
            if existing_row:
                existing_rows_by_file[str(file_path)] = dict(existing_row)
        else:
            existing_rows_by_file = {file_path: existing_row} if existing_row else {}

        thread = QThread(self)
        worker = BatchFitWorker(
            [file_path],
            [int(source_index)],
            existing_rows_by_file,
            capture_regex_pattern,
            capture_defaults,
            parameter_capture_map,
            fit_context["model_def"],
            fit_context["ordered_keys"],
            fit_context["seed_map"],
            fit_context["bounds_map"],
            fit_context["boundary_seed"],
            self.x_channel,
            self.y_channel,
            use_existing_fit_seed=(str(kind) == "batch"),
            random_restarts=(
                int(getattr(self, "_batch_refit_random_restarts", 0))
                if str(kind) == "batch"
                else 0
            ),
            smoothing_enabled=self.smoothing_enabled,
            smoothing_window=self._effective_smoothing_window(),
        )
        worker.moveToThread(thread)
        self.fit_tasks[task_id] = {
            "id": int(task_id),
            "kind": str(kind),
            "file_path": str(file_path),
            "file_key": self._fit_task_file_key(file_path),
            "source_index": int(source_index),
            "queue_position": (
                int(queue_position) if queue_position not in (None, "") else None
            ),
            "status": "pending",
            "thread": thread,
            "worker": worker,
        }
        if str(kind) == "batch":
            self._batch_active_task_ids.add(int(task_id))
            existing_r2 = None
            row_idx = self._find_batch_result_index_by_file(file_path)
            if row_idx is not None and 0 <= row_idx < len(self.batch_results):
                existing_r2 = self.batch_results[row_idx].get("r2")
            self._set_batch_row_runtime_fields(
                file_path,
                _fit_status="Queued",
                _fit_task_id=int(task_id),
                _queue_position=(
                    int(queue_position) if queue_position not in (None, "") else None
                ),
                _r2_old=existing_r2,
            )
        else:
            self._manual_active_task_ids.add(int(task_id))

        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda _completed, _total, row, tid=task_id: self._on_fit_task_progress(
                tid, row
            )
        )
        worker.finished.connect(
            lambda results, tid=task_id: self._on_fit_task_finished(tid, results)
        )
        worker.failed.connect(
            lambda error_text, tid=task_id: self._on_fit_task_failed(tid, error_text)
        )
        worker.cancelled.connect(lambda tid=task_id: self._on_fit_task_cancelled(tid))
        self._pending_fit_task_ids.append(int(task_id))
        self._schedule_fit_tasks()
        self._refresh_fit_action_buttons()
        self._refresh_batch_controls()
        return int(task_id)

    def _on_fit_task_progress(self, task_id, row):
        if int(task_id) not in self.fit_tasks:
            return
        self._apply_fit_row_update(row)

    def _on_fit_task_finished(self, task_id, results):
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return

        row = None
        if isinstance(results, (list, tuple)) and results:
            row = results[0]
            if row is not None:
                row = self._apply_param_range_validation_to_row(row)
                self._apply_fit_row_update(row)

        if task.get("kind") == "manual":
            file_path = str(task.get("file_path"))
            file_name = stem_for_file_ref(file_path)
            if row is None:
                self.stats_text.append(f"Auto-fit finished with no result: {file_name}")
            elif has_nonempty_values(row.get("params")):
                r2_val = _finite_float_or_none(row.get("r2"))
                r2_text = f"{float(r2_val):.6f}" if r2_val is not None else "N/A"
                row_error = self._batch_row_error_text(row)
                if row_error:
                    self.stats_text.append(
                        f"✗ Auto-fit failed [{file_name}]: {row_error}"
                    )
                else:
                    self.stats_text.append(
                        f"✓ Auto-fit successful [{file_name}] R²={r2_text}"
                    )
                current_file = self._current_loaded_file_path()
                if self._fit_task_file_key(current_file) == self._fit_task_file_key(
                    file_path
                ):
                    params = self._as_float_array(row.get("params"))
                    self.last_popt = params
                    self._last_fit_active_keys = self._ordered_param_keys()
                    self.last_pcov = None
                    self.last_r2 = r2_val
                    self._last_r2 = r2_val
                    self._apply_batch_params_for_file(file_path)
                    self.update_plot(fast=False)
            else:
                error_text = str(row.get("error") or "No fit result.")
                self.stats_text.append(f"✗ Auto-fit failed [{file_name}]: {error_text}")
        elif task.get("kind") == "batch" and row is not None:
            status_text = "Done"
            if bool(row.get("_fit_no_change")):
                status_text = "No Change"
            elif self._batch_row_error_text(row):
                status_text = "Failed"
            elif not has_nonempty_values(row.get("params")):
                status_text = "No Result"
            self._set_batch_row_runtime_fields(
                task.get("file_path"),
                _fit_status=status_text,
                _fit_task_id=None,
            )
            if has_nonempty_values(row.get("params")):
                seed_source = str(row.get("_seed_source") or "").strip().lower()
                if seed_source in {"matching-captures", "closest-captures"}:
                    file_path = str(task.get("file_path"))
                    file_name = stem_for_file_ref(file_path)
                    source_file = str(row.get("_seed_source_file") or "").strip()
                    source_name = (
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

        self._finish_fit_task(int(task_id))

    def _on_fit_task_failed(self, task_id, error_text):
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
        if task.get("kind") == "manual":
            file_name = stem_for_file_ref(file_path)
            self.stats_text.append(f"✗ Auto-fit failed [{file_name}]: {error_text}")
        self._finish_fit_task(int(task_id))

    def _on_fit_task_cancelled(self, task_id):
        task = self.fit_tasks.get(int(task_id))
        if task is None:
            return
        if task.get("kind") == "batch":
            self._set_batch_row_runtime_fields(
                task.get("file_path"),
                _fit_status="Cancelled",
                _fit_task_id=None,
            )
        if task.get("kind") == "manual":
            file_name = stem_for_file_ref(task.get("file_path"))
            self.stats_text.append(f"Auto-fit cancelled: {file_name}")
        self._finish_fit_task(int(task_id))

    def _finish_fit_task(self, task_id, *, force_terminate=False):
        task = self.fit_tasks.pop(int(task_id), None)
        if task is None:
            return

        status = str(task.get("status") or "pending")
        if status == "pending":
            try:
                self._pending_fit_task_ids.remove(int(task_id))
            except ValueError:
                pass

        kind = str(task.get("kind"))
        if kind == "batch":
            self._batch_active_task_ids.discard(int(task_id))
            self._batch_progress_done = min(
                int(self._batch_total_tasks),
                int(self._batch_progress_done) + 1,
            )
            self._set_batch_row_runtime_fields(
                task.get("file_path"),
                _fit_task_id=None,
            )
        else:
            self._manual_active_task_ids.discard(int(task_id))

        worker = task.get("worker")
        thread = task.get("thread")
        if worker is not None:
            try:
                worker.deleteLater()
            except Exception:
                pass
        if status == "running":
            self._shutdown_thread(
                thread,
                wait_ms=250,
                force_terminate=bool(force_terminate),
            )
        elif thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass

        if (
            kind == "batch"
            and self.batch_fit_in_progress
            and int(self._batch_progress_done) >= int(self._batch_total_tasks)
        ):
            self._complete_batch_fit_run()

        self._schedule_fit_tasks()
        self._refresh_fit_action_buttons()
        self._refresh_batch_controls()

    def _complete_batch_fit_run(self):
        cancelled = bool(getattr(self, "_batch_cancel_requested", False))
        self._refresh_batch_analysis_if_run()
        if cancelled:
            self.stats_text.append("Batch fit cancelled.")
        else:
            self.stats_text.append("✓ Batch fit completed.")

        self.batch_fit_in_progress = False
        self._batch_active_task_ids = set()
        self._batch_cancel_pending = False
        self._batch_cancel_requested = False
        self._batch_progress_done = 0
        self._batch_total_tasks = 0
        self.queue_visible_thumbnail_render()
        self._autosave_fit_details()
        self._refresh_batch_controls()

    def run_batch_fit(self):
        """Run batch fitting using the shared file list."""
        if self.batch_fit_in_progress:
            self.stats_text.append("Batch fit is already running.")
            return
        rerun_choice = self._prompt_batch_results_on_rerun()
        if rerun_choice == "cancel":
            self.stats_text.append("Batch fit cancelled; existing batch results kept.")
            return
        clear_existing_results = rerun_choice == "clear"
        previous_rows = list(getattr(self, "batch_results", []) or [])
        self._sync_batch_files_from_shared(sync_pattern=False)
        if not self.batch_files:
            self.stats_text.append("No files available from the shared folder list.")
            return

        capture_config = self._resolve_batch_capture_config(show_errors=True)
        if capture_config is None:
            return

        try:
            fit_context = self._build_fit_context()
        except Exception as exc:
            self.stats_text.append(f"Batch fit setup error: {exc}")
            return

        self._stop_thumbnail_render()
        self._batch_progress_done = 0
        self.batch_fit_in_progress = True
        self._batch_cancel_pending = False
        self._batch_cancel_requested = False

        existing_by_file = {}
        if not clear_existing_results:
            for row in previous_rows:
                file_ref = str(row.get("file") or "").strip()
                if not file_ref:
                    continue
                existing_by_file[self._fit_task_file_key(file_ref)] = row
            for row in list(getattr(self, "batch_results", []) or []):
                file_ref = str(row.get("file") or "").strip()
                if not file_ref:
                    continue
                existing_by_file[self._fit_task_file_key(file_ref)] = row
        self.batch_results = []
        work_items = []

        def _refit_priority(existing_row, source_index_value, has_existing_fit):
            if not has_existing_fit:
                return (0, 0.0, int(source_index_value))
            r2_val = _finite_float_or_none(existing_row.get("r2"))
            if r2_val is None:
                distance = float("inf")
            else:
                distance = float(abs(1.0 - float(r2_val)))
            return (1, -distance, int(source_index_value))

        for source_index, file_path in enumerate(self.batch_files):
            existing = existing_by_file.get(self._fit_task_file_key(file_path), {})
            extracted = extract_captures(
                stem_for_file_ref(file_path),
                capture_config.regex,
                capture_config.defaults,
            )
            captures = extracted if extracted is not None else {}
            pattern_error = _BATCH_PATTERN_MISMATCH_ERROR if extracted is None else None
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
            has_existing_fit = (
                has_nonempty_values(existing.get("params"))
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
        self.update_batch_table()

        prioritized_items = sorted(work_items, key=lambda item: item[3])
        self._batch_total_tasks = len(prioritized_items)
        self._batch_active_task_ids = set()
        self._refresh_batch_controls()
        self.stats_text.append(
            f"Batch fit started (queued, max parallel={self._fit_max_concurrent})."
        )

        parameter_capture_map = self._current_param_capture_map()
        for queue_position, (
            source_index,
            file_path,
            _already_fitted,
            _priority,
        ) in enumerate(prioritized_items, start=1):
            self._start_file_fit_task(
                kind="batch",
                file_path=file_path,
                source_index=source_index,
                fit_context=fit_context,
                capture_regex_pattern=capture_config.regex_pattern,
                capture_defaults=capture_config.defaults,
                parameter_capture_map=parameter_capture_map,
                queue_position=queue_position,
            )

    def on_batch_progress(self, completed, total, row):
        """Update progress label while batch is running."""
        self._batch_progress_done = int(completed)
        self.run_batch_btn.setText(f"Run Batch ({self._batch_progress_done}/{total})")

        row_index = row.get("_source_index")
        if row_index is None:
            for idx, existing_row in enumerate(self.batch_results):
                if existing_row.get("file") == row.get("file"):
                    row_index = idx
                    break
        if row_index is not None and 0 <= row_index < len(self.batch_results):
            existing = self.batch_results[row_index]
            row_has_fit = has_nonempty_values(row.get("params"))
            existing_plot_has_fit = existing.get("plot_has_fit")
            if existing_plot_has_fit is None:
                existing_plot_has_fit = has_nonempty_values(existing.get("params"))
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
            self.batch_results[row_index] = row
            table_row_idx = self._find_table_row_by_file(row["file"])
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)
            if row_has_fit and row.get("plot_full") is None:
                self._start_thumbnail_render(row_indices=[row_index])
            current_file = self._current_loaded_file_path()
            if current_file and row.get("file") == current_file and row_has_fit:
                if self._apply_batch_params_for_file(current_file):
                    self.update_plot(fast=False)

    def on_batch_finished(self, results):
        """Populate table and thumbnails after batch fit finishes."""
        previous_by_file = {row["file"]: row for row in self.batch_results}
        ordered_results = sorted(
            list(results), key=lambda row: int(row.get("_source_index", 0))
        )
        self.batch_results = ordered_results
        for row in self.batch_results:
            existing = previous_by_file.get(row["file"])
            if not existing:
                continue
            row_has_fit = has_nonempty_values(row.get("params"))
            existing_plot_has_fit = existing.get("plot_has_fit")
            if existing_plot_has_fit is None:
                existing_plot_has_fit = has_nonempty_values(existing.get("params"))
            if row_has_fit or existing_plot_has_fit:
                continue
            if existing.get("plot_full") is not None and row.get("plot_full") is None:
                row["plot_full"] = existing["plot_full"]
                row["plot_has_fit"] = False
                row["plot_render_size"] = existing.get("plot_render_size")
            elif existing.get("plot") is not None and row.get("plot") is None:
                row["plot"] = existing["plot"]
                row["plot_has_fit"] = False
                row["plot_render_size"] = existing.get("plot_render_size")
        self.update_batch_table()
        current_file = self._current_loaded_file_path()
        if current_file and self._apply_batch_params_for_file(current_file):
            self.update_plot(fast=False)
        self._refresh_batch_analysis_if_run()
        self.stats_text.append("✓ Batch fit completed.")
        self.cleanup_batch_thread()
        self.queue_visible_thumbnail_render()
        self._autosave_fit_details()

    def on_batch_failed(self, error_text):
        self.stats_text.append(f"✗ Batch fit failed: {error_text}")
        self.cleanup_batch_thread()

    def on_batch_cancelled(self):
        self.stats_text.append("Batch fit cancelled.")
        self.cleanup_batch_thread()

    def _force_stop_batch_fit(self, reason_text):
        if not self.batch_fit_in_progress:
            return
        self._batch_cancel_requested = True
        self.stats_text.append(str(reason_text))
        for task_id in list(self._batch_active_task_ids):
            task = self.fit_tasks.get(int(task_id))
            if task is None:
                continue
            self._request_worker_cancel(task.get("worker"))
            self._set_batch_row_runtime_fields(
                task.get("file_path"),
                _fit_status="Cancelled",
                _fit_task_id=None,
            )
            self._finish_fit_task(int(task_id), force_terminate=True)

    def cancel_batch_fit(self):
        """Request cancellation of an in-flight batch fit."""
        if not self.batch_fit_in_progress:
            return
        if not self._batch_cancel_pending:
            self._batch_cancel_pending = True
            self._batch_cancel_requested = True
            for task_id in list(self._batch_active_task_ids):
                task = self.fit_tasks.get(int(task_id))
                if task is None:
                    continue
                if str(task.get("status")) == "pending":
                    self._set_batch_row_runtime_fields(
                        task.get("file_path"),
                        _fit_status="Cancelled",
                        _fit_task_id=None,
                    )
                    self._finish_fit_task(int(task_id))
                else:
                    self._request_worker_cancel(task.get("worker"))
            self.stats_text.append(
                "Batch cancellation requested... click Cancel again to force stop."
            )
            self._refresh_batch_controls()
            if not self._batch_active_task_ids:
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

    def update_batch_table(self):
        """Refresh batch results table with captures and fit params."""
        if not self.batch_results:
            self.batch_table.setRowCount(0)
            self.batch_table.setColumnCount(0)
            rich_header = getattr(self, "batch_table_header", None)
            if isinstance(rich_header, RichTextHeaderView):
                rich_header.set_section_html_map({})
            return

        sorting_enabled = self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            param_columns = self._batch_parameter_column_items()
            param_column_tokens = [str(item["token"]) for item in param_columns]
            columns = (
                ["Plot"]
                + ["File"]
                + self.batch_capture_keys
                + ["Queue", "Status", "R² Old", "R² New"]
                + param_column_tokens
                + ["Error"]
            )
            self.batch_table.setColumnCount(len(columns))
            self.batch_table.setHorizontalHeaderLabels(columns)
            rich_header = getattr(self, "batch_table_header", None)
            if isinstance(rich_header, RichTextHeaderView):
                param_start = 6 + len(self.batch_capture_keys)
                rich_map = {}
                for offset, token in enumerate(param_column_tokens):
                    rich_html = parameter_symbol_to_html(token)
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

    def update_batch_table_row(self, row_idx, row, suspend_sorting=True):
        """Update a single batch row in the results table."""
        sorting_enabled = suspend_sorting and self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            # Plot column (index 0)
            self._update_batch_plot_cell(row_idx, row)

            # File name column (index 1)
            file_name = stem_for_file_ref(row["file"])
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
            queue_col = 2 + len(self.batch_capture_keys)
            status_col = queue_col + 1
            r2_old_col = status_col + 1
            r2_new_col = r2_old_col + 1
            # Parameter columns come after runtime columns.
            param_start = r2_new_col + 1
            param_columns = self._batch_parameter_column_items()
            error_col = param_start + len(param_columns)

            queue_value = row.get("_queue_position")
            queue_text = (
                str(int(queue_value))
                if isinstance(queue_value, (int, np.integer))
                else ""
            )
            self.batch_table.setItem(
                row_idx,
                queue_col,
                NumericSortTableWidgetItem(queue_text),
            )
            status_text = str(row.get("_fit_status") or "").strip()
            status_item = NumericSortTableWidgetItem(status_text)
            status_lower = status_text.lower()
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
            self.batch_table.setItem(row_idx, status_col, status_item)

            r2_old = _finite_float_or_none(row.get("_r2_old"))
            self.batch_table.setItem(
                row_idx,
                r2_old_col,
                NumericSortTableWidgetItem(f"{float(r2_old):.6f}" if r2_old is not None else ""),
            )
            r2_val = _finite_float_or_none(row.get("r2"))
            self.batch_table.setItem(
                row_idx,
                r2_new_col,
                NumericSortTableWidgetItem(f"{float(r2_val):.6f}" if r2_val is not None else ""),
            )

            params = self._as_float_array(row.get("params"))
            boundary_values = self._as_float_array(row.get("boundary_values"))
            violation_indices = {
                int(item.get("index"))
                for item in self._fit_param_range_violations(row.get("params"))
            }
            for offset, item in enumerate(param_columns):
                idx = int(item["index"])
                if item["kind"] == "param":
                    value = float(params[idx]) if params.size > idx else None
                else:
                    value = (
                        float(boundary_values[idx])
                        if boundary_values.size > idx
                        else None
                    )
                cell_text = f"{value:.6f}" if value is not None else ""
                cell_item = NumericSortTableWidgetItem(cell_text)
                if item["kind"] == "param" and idx in violation_indices:
                    cell_item.setForeground(QBrush(QColor("#b91c1c")))
                self.batch_table.setItem(row_idx, param_start + offset, cell_item)
            error_text = self._batch_row_error_text(row)
            self.batch_table.setItem(
                row_idx,
                error_col,
                NumericSortTableWidgetItem(error_text),
            )
            self._apply_batch_row_error_background(row_idx, bool(error_text))
        finally:
            if sorting_enabled:
                self.batch_table.setSortingEnabled(True)

    def _apply_batch_row_error_background(self, row_idx, is_error):
        """Tint errored rows pale red; force white for non-error rows."""
        if row_idx < 0 or row_idx >= self.batch_table.rowCount():
            return
        color = QColor("#fee2e2") if is_error else QColor("#ffffff")
        for col_idx in range(self.batch_table.columnCount()):
            item = self.batch_table.item(row_idx, col_idx)
            if item is not None:
                item.setBackground(color)

    def _update_batch_plot_cell(self, row_idx, row):
        """Update only the plot thumbnail cell for a batch row."""
        thumb_item = QTableWidgetItem()
        pixmap = self._scaled_batch_plot(row)
        if pixmap is not None:
            thumb_item.setData(Qt.ItemDataRole.DecorationRole, pixmap)
        else:
            thumb_item.setData(Qt.ItemDataRole.DecorationRole, None)
        thumb_item.setData(Qt.ItemDataRole.UserRole, row["file"])  # Store file path
        self.batch_table.setItem(row_idx, 0, thumb_item)

    def _on_batch_table_cell_clicked(self, row_idx, col_idx):
        """Load selected batch row into the shared Plot tab."""
        if row_idx < 0:
            return

        clicked_item = self.batch_table.item(row_idx, col_idx)
        file_path = (
            clicked_item.data(Qt.ItemDataRole.UserRole) if clicked_item else None
        )
        if not file_path:
            for fallback_col in (1, 0):
                file_item = self.batch_table.item(row_idx, fallback_col)
                file_path = (
                    file_item.data(Qt.ItemDataRole.UserRole)
                    if file_item is not None
                    else None
                )
                if file_path:
                    break
        if not file_path:
            return

        self._open_file_in_plot_tab(file_path)

    def _open_file_in_plot_tab(self, file_path):
        if not file_path:
            return False

        if file_path not in self.data_files:
            self.data_files.append(file_path)
            self.file_combo.addItem(stem_for_file_ref(file_path), file_path)
            self._sync_batch_files_from_shared(sync_pattern=False)

        file_idx = self.data_files.index(file_path)
        self.load_file(file_idx)
        self.tabs.setCurrentWidget(self.manual_tab)
        return True

    def _expand_file_column_for_selected_files(self):
        """Expand file column width to show the longest selected file name."""
        if not self.batch_files or self.batch_table.columnCount() < 2:
            return

        font_metrics = self.batch_table.fontMetrics()
        longest_width = 0
        for file_path in self.batch_files:
            file_name = stem_for_file_ref(file_path)
            longest_width = max(
                longest_width, font_metrics.horizontalAdvance(file_name)
            )

        # Account for text padding and small header/sort margin.
        target_width = longest_width + 36
        current_width = self.batch_table.columnWidth(1)
        if target_width > current_width:
            self.batch_table.setColumnWidth(1, target_width)

    def _visible_batch_row_indices(self):
        if not hasattr(self, "batch_table") or self.batch_table.rowCount() == 0:
            return []
        viewport = self.batch_table.viewport().rect()
        model = self.batch_table.model()
        visible = []
        for row_idx in range(self.batch_table.rowCount()):
            rect = self.batch_table.visualRect(model.index(row_idx, 0))
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
                item = self.batch_table.item(table_row, col_idx)
                if item is not None:
                    file_path = item.data(Qt.ItemDataRole.UserRole)
                if file_path:
                    break
            if not file_path:
                continue
            result_idx = result_index_by_file.get(file_path)
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

    def _row_thumbnail_render_size(self, row):
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

    def _batch_row_thumbnail_needs_render(self, row, expected_size):
        source = row.get("plot_full") or row.get("plot")
        if source is None:
            return True

        rendered_size = self._row_thumbnail_render_size(row)
        if rendered_size is None:
            return True
        return tuple(rendered_size) != tuple(expected_size)

    def queue_visible_thumbnail_render(self, *_args):
        if not self.batch_results:
            return
        row_indices = self._visible_batch_result_indices()
        if not row_indices:
            row_indices = list(range(min(len(self.batch_results), 10)))
        row_indices = self._prioritize_thumbnail_rows(row_indices)
        self._start_thumbnail_render(row_indices=row_indices)

    def _start_thumbnail_render(self, row_indices=None):
        """Start background thread to render missing thumbnails."""
        if not self.batch_results:
            return

        expected_size = self._full_batch_thumbnail_size()

        if row_indices is None:
            candidate_rows = list(range(len(self.batch_results)))
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
        self.thumb_worker = ThumbnailRenderWorker(
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

    def _stop_thumbnail_render(self):
        """Stop thumbnail worker/thread if active."""
        self._request_worker_cancel(self.thumb_worker)
        self._shutdown_thread(self.thumb_thread, wait_ms=2000)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        self._pending_thumbnail_rows.clear()

    def _on_thumbnail_rendered(self, idx, total, row_idx):
        """Update table cell when thumbnail is rendered."""
        if row_idx < len(self.batch_results):
            row = self.batch_results[row_idx]
            table_row_idx = self._find_table_row_by_file(row["file"])
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)

    def _on_thumbnails_finished(self):
        """Clean up thumbnail worker when finished."""
        self._shutdown_thread(self.thumb_thread)
        self.thumb_thread = None
        self.thumb_worker = None
        self.thumb_render_in_progress = False
        if self._pending_thumbnail_rows:
            queued = self._prioritize_thumbnail_rows(self._pending_thumbnail_rows)
            self._pending_thumbnail_rows.clear()
            self._start_thumbnail_render(row_indices=queued)

    def _set_batch_parse_feedback(self, message, is_error=False, tooltip=""):
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

    def _resolve_batch_capture_config(self, show_errors):
        pattern_text = (
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

    def _update_batch_capture_feedback(self, config):
        _ = config
        self._set_batch_parse_feedback("", is_error=False)

    def _on_regex_changed(self):
        """Debounce filename pattern changes to avoid excessive updates."""
        self._refresh_param_capture_mapping_controls()
        self.regex_timer.stop()
        self.regex_timer.start(300)  # 300ms debounce

    def prepare_batch_preview(self):
        """Populate preview results before running batch fit."""
        self.regex_timer.stop()
        self._do_prepare_batch_preview()

    def _do_prepare_batch_preview(self):
        """Actually perform the batch preview update."""
        if not self.batch_files:
            self._stop_thumbnail_render()
            self.batch_match_count = 0
            self.batch_unmatched_files = []
            self.batch_capture_keys = []
            self.batch_results = []
            config = self._resolve_batch_capture_config(show_errors=True)
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

        capture_config = self._resolve_batch_capture_config(show_errors=True)
        if capture_config is None:
            return

        self.batch_status_label.hide()

        existing_file_order = [row["file"] for row in self.batch_results]
        files_unchanged = existing_file_order == self.batch_files and bool(
            self.batch_results
        )

        self.batch_capture_keys = []
        self.batch_match_count = 0
        self.batch_unmatched_files = []

        if files_unchanged:
            for source_index, row in enumerate(self.batch_results):
                row["_source_index"] = source_index
                row["x_channel"] = self.x_channel
                row["y_channel"] = self.y_channel
                extracted = extract_captures(
                    stem_for_file_ref(row["file"]),
                    capture_config.regex,
                    capture_config.defaults,
                )
                captures = {}
                if extracted is None:
                    self.batch_unmatched_files.append(stem_for_file_ref(row["file"]))
                    row["pattern_error"] = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures = extracted
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
                extracted = extract_captures(
                    stem_for_file_ref(file_path),
                    capture_config.regex,
                    capture_config.defaults,
                )
                pattern_error = None
                if extracted is None:
                    self.batch_unmatched_files.append(stem_for_file_ref(file_path))
                    pattern_error = _BATCH_PATTERN_MISMATCH_ERROR
                else:
                    captures = extracted
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

    def _refresh_batch_analysis_if_run(self):
        if not hasattr(self, "analysis_status_label"):
            return
        self._refresh_batch_analysis_data(preserve_selection=True)

    def closeEvent(self, event):
        """Ensure worker thread is stopped before closing."""
        self._autosave_fit_details()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        for task in list(self.fit_tasks.values()):
            self._request_worker_cancel(task.get("worker"))
        for task in list(self.fit_tasks.values()):
            thread = task.get("thread")
            worker = task.get("worker")
            if worker is not None:
                try:
                    worker.deleteLater()
                except Exception:
                    pass
            self._shutdown_thread(thread, wait_ms=2000, force_terminate=True)
        self.fit_tasks = {}
        self._pending_fit_task_ids = deque()
        self._batch_active_task_ids = set()
        self._manual_active_task_ids = set()
        self._request_worker_cancel(self.thumb_worker)
        self._shutdown_thread(self.fit_thread, wait_ms=2000, force_terminate=True)
        self._shutdown_thread(self.batch_thread, wait_ms=2000, force_terminate=True)
        self._shutdown_thread(self.thumb_thread, wait_ms=2000, force_terminate=True)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = ManualFitGUI()
    window.show()
    sys.exit(app.exec())
