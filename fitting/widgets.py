"""Custom Qt widgets for fit_gui."""

import math
import re

import numpy as np

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QDoubleSpinBox,
    QLabel,
    QTextEdit,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QStyleOptionHeader,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QRectF
from PyQt6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QTextDocument,
    QFont,
)
from PyQt6.QtCore import pyqtSignal


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
            left_sort = self.data(TABLE_SORT_ROLE)
            right_sort = other.data(TABLE_SORT_ROLE)
            if left_sort is not None and right_sort is not None:
                left_rank = self._to_number(left_sort)
                right_rank = self._to_number(right_sort)
                if left_rank is not None and right_rank is not None:
                    return left_rank < right_rank
                return str(left_sort) < str(right_sort)
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


def format_boundary_display_name(index: int) -> str:
    idx = max(0, int(index))
    return f"X{str(idx).translate(_UNICODE_SUBSCRIPT_TRANS)}"


_RICH_TEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 11
TABLE_SORT_ROLE = int(Qt.ItemDataRole.UserRole) + 12


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
        self._linked_indices = set()  # indices that are part of a link group
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

    def set_linked_indices(self, indices):
        """Mark handle indices that are part of a boundary link group."""
        if indices is None:
            self._linked_indices = set()
        else:
            self._linked_indices = {int(i) for i in indices}
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
            is_linked = idx in self._linked_indices
            if is_active:
                pen_color = QColor("#1d4ed8")
                brush_color = QColor("#2563eb")
            elif is_linked:
                pen_color = QColor("#7c3aed")
                brush_color = QColor("#8b5cf6")
            else:
                pen_color = QColor("#4b5563")
                brush_color = QColor("#6b7280")
            painter.setPen(QPen(pen_color, 1.2))
            painter.setBrush(QBrush(brush_color))
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
                else format_boundary_display_name(idx)
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
