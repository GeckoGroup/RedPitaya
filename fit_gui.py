#!/usr/bin/env python3
"""
Manual Curve Fitting GUI for MI Model
Allows manual adjustment of parameters for failed automatic fits.
"""

import ast
import html
import math
import sys
import re
import csv
import zipfile
from io import BytesIO
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Pattern, Sequence, Tuple
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
from pandas import read_csv
from scipy.optimize import curve_fit, differential_evolution

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
    QFormLayout,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QEvent
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QIcon,
    QPixmap,
    QPalette,
    QColor,
    QTextCharFormat,
    QSyntaxHighlighter,
    QFont,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

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


def display_name_for_file_ref(file_ref: str) -> str:
    """Display only the data-file stem (no zip archive name, no .csv suffix)."""
    return stem_for_file_ref(file_ref)


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
DEFAULT_TARGET_CHANNEL = "CH2"
DEFAULT_EXPRESSION = "abs(a * sin(b * CH3 + pi * phi) + d) ** 2"
DEFAULT_PARAM_SPECS = (
    ParameterSpec(
        key="a",
        symbol="A",
        description="MI Amplitude",
        default=0.74545,
        min_value=0.0,
        max_value=10.0,
    ),
    ParameterSpec(
        key="b",
        symbol="B",
        description="Voltage to Phase",
        default=-0.2175,
        min_value=-2.0,
        max_value=2.0,
    ),
    ParameterSpec(
        key="phi",
        symbol="φ",
        description="MI Phase",
        default=0.0,
        min_value=-2.0,
        max_value=2.0,
    ),
    ParameterSpec(
        key="d",
        symbol="D",
        description="MI Offset",
        default=1.7019,
        min_value=-10.0,
        max_value=10.0,
    ),
)
FIT_CURVE_COLOR = "#16a34a"

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


@dataclass(frozen=True)
class FitOptimizationOptions:
    enabled: bool = False
    n_starts: int = 10
    per_start_maxfev: int = 1000
    seed: int = 1
    use_global_init: bool = True
    de_maxiter: int = 20
    de_popsize: int = 10
    early_stop_r2: float = 0.999
    early_stop_patience: int = 3


class FitCancelledError(RuntimeError):
    """Internal exception used to abort fitting when cancellation is requested."""


def _sum_squared_error(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if np.count_nonzero(valid) == 0:
        return None
    residual = y_true_arr[valid] - y_pred_arr[valid]
    return float(np.dot(residual, residual))


def fit_mode_label(options: Optional[FitOptimizationOptions]) -> str:
    opts = (
        options
        if isinstance(options, FitOptimizationOptions)
        else FitOptimizationOptions()
    )
    if not opts.enabled:
        return "single-start"
    if opts.use_global_init:
        return "multi-start+de"
    return "multi-start"


def build_fit_diagnostics_parts(
    *,
    mode=None,
    attempts=None,
    requested_starts=None,
    seed=None,
    de_used=None,
    best_sse=None,
    include_starts_suffix=False,
):
    parts = []
    if mode:
        parts.append(str(mode))
    if attempts is not None and requested_starts is not None:
        suffix = " starts" if include_starts_suffix else ""
        parts.append(f"best of {attempts}/{requested_starts}{suffix}")
    if seed is not None:
        parts.append(f"seed={seed}")
    if de_used is not None:
        parts.append(f"DE={'on' if bool(de_used) else 'off'}")
    if best_sse is not None:
        try:
            best_sse_value = float(best_sse)
        except Exception:
            best_sse_value = None
        if best_sse_value is not None and np.isfinite(best_sse_value):
            parts.append(f"SSE={best_sse_value:.6g}")
    return parts


def run_multistart_fit(
    model_func,
    x,
    y,
    p0,
    bounds,
    options,
    cancel_check: Optional[Callable[[], bool]] = None,
    seed_offset: int = 0,
):
    opts = (
        options
        if isinstance(options, FitOptimizationOptions)
        else FitOptimizationOptions()
    )
    x_data = np.asarray(x, dtype=float).reshape(-1)
    y_data = np.asarray(y, dtype=float).reshape(-1)
    p0_arr = np.asarray(p0, dtype=float).reshape(-1)
    lower = np.asarray(bounds[0], dtype=float).reshape(-1)
    upper = np.asarray(bounds[1], dtype=float).reshape(-1)
    if x_data.size != y_data.size:
        raise ValueError("x and y must have the same length.")
    if p0_arr.size == 0:
        raise ValueError("Initial parameter vector is empty.")
    if lower.size != p0_arr.size or upper.size != p0_arr.size:
        raise ValueError("Bounds and initial parameters must have matching dimensions.")

    low = np.minimum(lower, upper)
    high = np.maximum(lower, upper)
    bounds_tuple = (low, high)

    def is_cancelled() -> bool:
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except Exception:
            return False

    if is_cancelled():
        raise FitCancelledError("cancelled")

    n_starts = 1
    if opts.enabled:
        n_starts = max(2, int(opts.n_starts))
    per_start_maxfev = max(20, int(opts.per_start_maxfev))
    if not opts.enabled:
        per_start_maxfev = max(2000, per_start_maxfev)
    patience = opts.early_stop_patience
    early_stop_r2 = float(opts.early_stop_r2)
    base_seed = int(opts.seed) + int(seed_offset)

    starts = []
    seen = set()

    def add_start(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != p0_arr.size:
            return
        clipped = np.clip(arr, low, high)
        key = tuple(np.round(clipped, 12))
        if key in seen:
            return
        seen.add(key)
        starts.append(clipped)

    add_start(p0_arr)

    de_attempted = bool(opts.enabled and opts.use_global_init)
    de_success = False
    if de_attempted:

        def de_objective(values):
            if is_cancelled():
                return np.inf
            try:
                predicted = model_func(x_data, *values)
            except Exception:
                return np.inf
            sse = _sum_squared_error(y_data, predicted)
            if sse is None or not np.isfinite(sse):
                return np.inf
            return float(sse)

        def de_callback(_xk, _convergence):
            return is_cancelled()

        de_result = None
        try:
            de_result = differential_evolution(
                de_objective,
                list(zip(low, high)),
                seed=base_seed,
                maxiter=max(1, int(opts.de_maxiter)),
                popsize=max(2, int(opts.de_popsize)),
                polish=False,
                callback=de_callback,
                updating="deferred",
                workers=1,
            )
        except Exception:
            de_result = None
        if is_cancelled():
            raise FitCancelledError("cancelled")
        if de_result is not None and np.all(np.isfinite(de_result.x)):
            add_start(de_result.x)
            de_success = bool(np.isfinite(float(de_result.fun)))

    if opts.enabled:
        add_start((low + high) * 0.5)

    if opts.enabled and len(starts) < n_starts:
        rng = np.random.default_rng(base_seed)
        while len(starts) < n_starts:
            if is_cancelled():
                raise FitCancelledError("cancelled")
            add_start(rng.uniform(low, high))
    if opts.enabled and len(starts) > n_starts:
        starts = starts[:n_starts]

    if not starts:
        add_start(p0_arr)

    best_popt = None
    best_pcov = None
    best_r2 = float("nan")
    best_sse = None
    attempts = 0
    successes = 0
    last_error = None
    no_improvement_count = 0
    rescue_used = False

    def attempt_start(start):
        nonlocal attempts
        nonlocal successes
        nonlocal last_error
        nonlocal no_improvement_count
        nonlocal best_popt
        nonlocal best_pcov
        nonlocal best_r2
        nonlocal best_sse

        if is_cancelled():
            raise FitCancelledError("cancelled")
        attempts += 1
        try:
            popt, pcov = curve_fit(
                model_func,
                x_data,
                y_data,
                p0=start,
                bounds=bounds_tuple,
                method="trf",
                maxfev=per_start_maxfev,
            )
            predicted = model_func(x_data, *popt)
            sse = _sum_squared_error(y_data, predicted)
            if sse is None or not np.isfinite(sse):
                raise RuntimeError("Fit produced non-finite error metric.")
            r2 = compute_r2(y_data, predicted)
            r2 = float(r2) if r2 is not None else float("nan")
            successes += 1

            improvement_tol = 1e-12
            if best_sse is not None:
                improvement_tol = max(improvement_tol, abs(best_sse) * 1e-9)
            improved = best_sse is None or (sse < (best_sse - improvement_tol))
            if improved:
                best_popt = np.asarray(popt, dtype=float)
                best_pcov = np.asarray(pcov, dtype=float)
                best_r2 = r2
                best_sse = float(sse)
                no_improvement_count = 0
            elif best_sse is not None:
                no_improvement_count += 1

            if np.isfinite(best_r2) and best_r2 >= early_stop_r2:
                return True
            if (
                best_sse is not None
                and patience > 0
                and no_improvement_count >= patience
            ):
                return True
        except Exception as exc:
            last_error = exc
            if best_sse is not None:
                no_improvement_count += 1
                if patience > 0 and no_improvement_count >= patience:
                    return True
        return False

    for start in starts:
        if attempt_start(start):
            break

    # In basic mode, rescue from obviously bad local minima before reporting a result.
    if (not opts.enabled) and (not np.isfinite(best_r2) or best_r2 < 0.0):
        rescue_used = True
        no_improvement_count = 0
        starts_before_rescue = len(starts)
        add_start((low + high) * 0.5)
        rng = np.random.default_rng(base_seed + 104729)
        draws = 0
        target_total = max(starts_before_rescue + 5, 6)
        while len(starts) < target_total and draws < 64:
            if is_cancelled():
                raise FitCancelledError("cancelled")
            add_start(rng.uniform(low, high))
            draws += 1

        for start in starts[starts_before_rescue:]:
            if attempt_start(start):
                break

    if best_popt is None or best_pcov is None:
        if last_error is not None:
            raise RuntimeError(
                f"All fit attempts failed ({attempts}/{len(starts)}): {last_error}"
            ) from last_error
        raise RuntimeError(f"All fit attempts failed ({attempts}/{len(starts)}).")

    if not np.isfinite(best_r2):
        raise RuntimeError("Fit produced a non-finite R² score.")
    if best_r2 < -1.0:
        raise RuntimeError(
            f"Fit converged to a poor minimum (R²={best_r2:.6f}). "
            "Try enabling Robust mode or adjusting bounds."
        )

    mode_text = fit_mode_label(opts)
    if rescue_used and not opts.enabled:
        mode_text = "single-start+rescue"

    diagnostics = {
        "mode": mode_text,
        "seed": base_seed,
        "requested_starts": int(len(starts)),
        "attempts": int(attempts),
        "successes": int(successes),
        "best_sse": float(best_sse) if best_sse is not None else None,
        "de_used": bool(de_attempted),
        "de_success": bool(de_success),
        "rescue_used": bool(rescue_used),
    }
    return best_popt, best_pcov, float(best_r2), diagnostics


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
_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "-": "⁻",
        "+": "⁺",
    }
)
_LATEX_PARAMETER_SYMBOLS = {
    # Lowercase greek
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
    # Uppercase greek
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
    "arcsin": "sin⁻¹",
    "arccos": "cos⁻¹",
    "arctan": "tan⁻¹",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "log10": "log₁₀",
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
    while token.startswith("\\"):
        token = token[1:]
    token = token.strip()
    if token.startswith("{") and token.endswith("}") and len(token) >= 3:
        token = token[1:-1].strip()
    return token


def latex_symbol_to_unicode(symbol_text: str) -> Optional[str]:
    token = _normalize_latex_symbol_token(symbol_text)
    if not token:
        return None
    return _LATEX_PARAMETER_SYMBOLS.get(token)


def resolve_parameter_symbol(param_key: str, symbol_hint: Optional[str] = None) -> str:
    key_text = str(param_key).strip()
    if symbol_hint is not None:
        raw_symbol = str(symbol_hint).strip()
        if raw_symbol:
            mapped_symbol = latex_symbol_to_unicode(raw_symbol)
            return mapped_symbol if mapped_symbol else raw_symbol
    mapped_from_key = latex_symbol_to_unicode(key_text)
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
                exponent = str(int(node.right.value)).translate(
                    _SUPERSCRIPT_TRANSLATION
                )
                return f"{left_render}{exponent}"
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
            return "π"
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
        fallback = re.sub(r"\b(?:np|math)\.pi\b", "π", fallback)
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
    fit_attempts=None,
    fit_best_sse=None,
    fit_mode=None,
    fit_seed=None,
    fit_requested_starts=None,
    fit_de_used=None,
    pattern_error=None,
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
        "fit_attempts": fit_attempts,
        "fit_best_sse": fit_best_sse,
        "fit_mode": fit_mode,
        "fit_seed": fit_seed,
        "fit_requested_starts": fit_requested_starts,
        "fit_de_used": fit_de_used,
        "pattern_error": pattern_error,
    }


def render_batch_thumbnail(row, model_func, full_thumbnail_size=(468, 312)):
    """Render a row thumbnail pixmap, including fitted curve when available."""
    try:
        data = read_measurement_csv(row["file"])
        time_col = "TIME" if "TIME" in data.columns else data.columns[0]
        x_col = row.get("x_channel") or (
            "CH3" if "CH3" in data.columns else data.columns[0]
        )
        y_col = row.get("y_channel") or (
            "CH2" if "CH2" in data.columns else data.columns[0]
        )
        time_data = data[time_col].to_numpy(dtype=float, copy=True) * 1e3
        y_data = data[y_col].to_numpy(dtype=float, copy=True)
        x_data = data[x_col].to_numpy(dtype=float, copy=True)
        column_data = {}
        for column in data.columns:
            key = str(column).strip()
            if not key:
                continue
            try:
                column_data[key] = data[column].to_numpy(dtype=float, copy=True)
            except Exception:
                continue

        target_width = max(24, int(full_thumbnail_size[0]))
        target_height = max(24, int(full_thumbnail_size[1]))
        render_dpi = 180
        fig = Figure(
            figsize=(target_width / render_dpi, target_height / render_dpi),
            dpi=render_dpi,
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
        ax = fig.add_subplot(111)
        ax.plot(time_data, y_data, linewidth=1.25, color="C0")
        ax.plot(time_data, x_data, linewidth=1.25, alpha=0.45, color="C1")

        params = row.get("params")
        if params:
            fitted_y = model_func(
                x_data,
                *params,
                column_data=column_data,
            )
            ax.plot(time_data, fitted_y, linewidth=1.25, color=FIT_CURVE_COLOR)

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
    finished = pyqtSignal(object, object, float, object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        x_data,
        y_data,
        p0,
        bounds,
        model_func,
        fit_options=None,
    ):
        super().__init__()
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
        self.model_func = model_func
        self.fit_options = (
            fit_options
            if isinstance(fit_options, FitOptimizationOptions)
            else FitOptimizationOptions()
        )
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            if self.cancel_requested:
                self.cancelled.emit()
                return

            popt, pcov, r2, diagnostics = run_multistart_fit(
                self.model_func,
                self.x_data,
                self.y_data,
                self.p0,
                self.bounds,
                self.fit_options,
                cancel_check=lambda: self.cancel_requested,
            )
            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit(popt, pcov, float(r2), diagnostics)
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
        p0,
        bounds,
        all_param_keys,
        active_param_keys,
        base_params,
        regex_pattern,
        capture_defaults,
        expression_evaluator,
        x_channel,
        y_channel,
        fit_start_pct,
        fit_end_pct,
        fit_options=None,
    ):
        super().__init__()
        self.file_paths = list(file_paths)
        self.p0 = np.asarray(p0, dtype=float)
        self.bounds = bounds
        self.all_param_keys = list(all_param_keys)
        self.active_param_keys = list(active_param_keys)
        self.base_params = [float(val) for val in base_params]
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.capture_defaults = dict(capture_defaults or {})
        self.expression_evaluator = expression_evaluator
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.fit_start_pct = fit_start_pct
        self.fit_end_pct = fit_end_pct
        self.fit_options = (
            fit_options
            if isinstance(fit_options, FitOptimizationOptions)
            else FitOptimizationOptions()
        )
        self.cancel_requested = False
        self._executor = None
        self._futures = []

    def request_cancel(self):
        self.cancel_requested = True
        futures = list(getattr(self, "_futures", []))
        for future in futures:
            try:
                future.cancel()
            except Exception:
                pass
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _fit_single_file(self, source_index, file_path):
        if self.cancel_requested:
            return None
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

        data = read_measurement_csv(file_path)
        x_data = data[self.x_channel].to_numpy(dtype=float, copy=True)
        y_data = data[self.y_channel].to_numpy(dtype=float, copy=True)
        n = len(x_data)
        start = int(np.floor((self.fit_start_pct / 100.0) * max(0, n - 1)))
        end = int(np.ceil((self.fit_end_pct / 100.0) * max(0, n - 1))) + 1
        start = max(0, min(n - 1, start)) if n else 0
        end = max(start + 1, min(n, end)) if n else 0
        fit_slice = slice(start, end)
        x_fit = x_data[fit_slice]
        y_fit = y_data[fit_slice]
        column_data_fit = {}
        for column in data.columns:
            key = str(column).strip()
            if not key:
                continue
            try:
                column_data_fit[key] = data[column].to_numpy(dtype=float, copy=True)[
                    fit_slice
                ]
            except Exception:
                continue

        base_param_map = {
            key: self.base_params[idx] for idx, key in enumerate(self.all_param_keys)
        }

        def fit_model(x_local, *fit_params):
            params = dict(base_param_map)
            for idx, key in enumerate(self.active_param_keys):
                if idx < len(fit_params):
                    params[key] = float(fit_params[idx])
            return self.expression_evaluator(
                x_local,
                params,
                column_data=column_data_fit,
            )

        if self.cancel_requested:
            return None

        popt, _pcov, r2_val, diagnostics = run_multistart_fit(
            fit_model,
            x_fit,
            y_fit,
            self.p0,
            self.bounds,
            self.fit_options,
            cancel_check=lambda: self.cancel_requested,
            seed_offset=source_index,
        )

        if self.cancel_requested:
            return None

        merged_params = {
            key: self.base_params[idx] for idx, key in enumerate(self.all_param_keys)
        }
        for idx, key in enumerate(self.active_param_keys):
            if idx < len(popt):
                merged_params[key] = float(popt[idx])
        row["params"] = [float(merged_params[key]) for key in self.all_param_keys]
        row["r2"] = float(r2_val) if r2_val is not None else None
        row["fit_attempts"] = diagnostics.get("attempts")
        row["fit_best_sse"] = diagnostics.get("best_sse")
        row["fit_mode"] = diagnostics.get("mode")
        row["fit_seed"] = diagnostics.get("seed")
        row["fit_requested_starts"] = diagnostics.get("requested_starts")
        row["fit_de_used"] = diagnostics.get("de_used")
        return row

    @pyqtSlot()
    def run(self):
        results = [None] * len(self.file_paths)
        executor = None
        try:
            total = len(self.file_paths)
            if total == 0:
                self.finished.emit([])
                return

            ideal = int(QThread.idealThreadCount())
            if ideal <= 0:
                ideal = 4
            max_workers = max(1, min(8, ideal))

            completed = 0
            executor = ThreadPoolExecutor(max_workers=max_workers)
            self._executor = executor
            future_to_idx = {
                executor.submit(self._fit_single_file, idx, file_path): idx
                for idx, file_path in enumerate(self.file_paths)
            }
            self._futures = list(future_to_idx.keys())

            while future_to_idx:
                if self.cancel_requested:
                    for pending in list(future_to_idx.keys()):
                        try:
                            pending.cancel()
                        except Exception:
                            pass
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    self.cancelled.emit()
                    return

                done, _pending = wait(
                    list(future_to_idx.keys()),
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue

                for future in done:
                    idx = future_to_idx.pop(future, None)
                    if idx is None:
                        continue
                    try:
                        self._futures.remove(future)
                    except ValueError:
                        pass
                    except Exception:
                        self._futures = [
                            item for item in self._futures if item is not future
                        ]

                    if future.cancelled():
                        if self.cancel_requested:
                            self.cancelled.emit()
                            return
                        row = make_batch_result_row(
                            source_index=idx,
                            file_path=self.file_paths[idx],
                            x_channel=self.x_channel,
                            y_channel=self.y_channel,
                            captures={},
                            error="Cancelled",
                            fit_mode=fit_mode_label(self.fit_options),
                            fit_seed=int(self.fit_options.seed) + int(idx),
                            fit_requested_starts=(
                                max(2, int(self.fit_options.n_starts))
                                if self.fit_options.enabled
                                else 1
                            ),
                            fit_de_used=bool(
                                self.fit_options.enabled
                                and self.fit_options.use_global_init
                            ),
                            pattern_error=None,
                        )
                    else:
                        try:
                            row = future.result()
                        except Exception as exc:
                            extracted = extract_captures(
                                stem_for_file_ref(self.file_paths[idx]),
                                self.regex,
                                self.capture_defaults,
                            )
                            captures = extracted if extracted is not None else {}
                            pattern_error = (
                                _BATCH_PATTERN_MISMATCH_ERROR
                                if extracted is None
                                else None
                            )
                            row = make_batch_result_row(
                                source_index=idx,
                                file_path=self.file_paths[idx],
                                x_channel=self.x_channel,
                                y_channel=self.y_channel,
                                captures=captures,
                                error=str(exc),
                                fit_mode=fit_mode_label(self.fit_options),
                                fit_seed=int(self.fit_options.seed) + int(idx),
                                fit_requested_starts=(
                                    max(2, int(self.fit_options.n_starts))
                                    if self.fit_options.enabled
                                    else 1
                                ),
                                fit_de_used=bool(
                                    self.fit_options.enabled
                                    and self.fit_options.use_global_init
                                ),
                                pattern_error=pattern_error,
                            )
                    if row is None:
                        self.cancelled.emit()
                        return

                    results[idx] = row
                    completed += 1
                    self.progress.emit(completed, total, row)

            if self.cancel_requested:
                self.cancelled.emit()
                return

            self.finished.emit([row for row in results if row is not None])
        except Exception as exc:
            if self.cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        finally:
            futures = list(getattr(self, "_futures", []))
            for future in futures:
                try:
                    future.cancel()
                except Exception:
                    pass
            self._futures = []
            self._executor = None
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass


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
    ):
        super().__init__()
        self.batch_results = batch_results
        self.model_func = model_func
        self.full_thumbnail_size = full_thumbnail_size
        if row_indices is None:
            self.row_indices = list(range(len(batch_results)))
        else:
            self.row_indices = sorted(
                {int(idx) for idx in row_indices if 0 <= int(idx) < len(batch_results)}
            )
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            total = len(self.row_indices)
            for done_idx, row_idx in enumerate(self.row_indices):
                if self.cancel_requested:
                    self.cancelled.emit()
                    return

                row = self.batch_results[row_idx]
                if row.get("plot_full") is not None:
                    self.progress.emit(done_idx + 1, total, row_idx)
                    continue

                pixmap = self.render_thumbnail(row)
                row["plot_full"] = pixmap
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
        )


class ManualFitGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.param_specs = list(DEFAULT_PARAM_SPECS)
        self.param_spinboxes = {}
        self.param_sliders = {}
        self.param_min_spinboxes = {}
        self.param_max_spinboxes = {}
        self.param_error_labels = {}
        self.param_row_tail_spacers = []
        self._param_slider_steps = 2000
        self._param_name_width = 88
        self._param_bound_width = 72
        self._param_value_width = 78
        self._param_error_width = 86
        self._fit_option_label_width = 64
        self._param_tail_placeholder_width = 0
        self._active_fit_keys = []
        self._last_fit_active_keys = []
        self._fit_ordered_keys = []
        self._fit_base_values = {}
        self._apply_expression_in_progress = False
        self._channel_sync_in_progress = False
        self._highlight_refresh_in_progress = False
        self._file_load_in_progress = False
        self._expression_edit_mode = False
        self.current_expression = f"{DEFAULT_TARGET_CHANNEL} = {DEFAULT_EXPRESSION}"
        try:
            self._compiled_expression = compile_expression_function(
                DEFAULT_EXPRESSION,
                [spec.key for spec in self.param_specs],
            )
        except Exception:
            self._compiled_expression = None

        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.setGeometry(100, 100, 900, 900)

        self.data_files = []
        self.current_file_idx = 0
        self.current_data = None
        self.channels = {
            "CH2": "MI output voltage",
            "CH3": "Sig Gen / TTL",
            "CH4": "TTL / trigger",
            "TIME": "Time",
        }
        self.x_channel = "CH3"
        self.y_channel = "CH2"
        self.fit_region_start_pct = 0.0
        self.fit_region_end_pct = 100.0
        self._fit_window_bounds_ms = (None, None)
        self._fit_boundary_positions_ms = ()
        self._fit_boundary_pick_px = 8.0
        self.fit_region_selector = None
        self._suppress_fit_region_selector = False
        self._fit_region_refresh_pending = False
        self._plot_mouse_cid = None
        self.last_popt = None
        self.last_pcov = None
        self.last_fit_r2 = None
        self.last_full_r2 = None
        self.last_fit_diagnostics = None
        self._last_r2_fit = None
        self._last_r2_full = None
        self.fit_options = FitOptimizationOptions()
        self._batch_fit_options = self.fit_options
        self.fit_thread = None
        self.fit_worker = None
        self.batch_thread = None
        self.batch_worker = None
        self.batch_fit_in_progress = False
        self._batch_cancel_pending = False
        self.batch_results = []
        self.batch_files = []
        self.batch_capture_keys = []
        self.batch_match_count = 0
        self.batch_unmatched_files = []
        self.analysis_csv_records = []
        self.analysis_csv_path = None
        self.analysis_records = []
        self.analysis_columns = []
        self.analysis_numeric_data = {}
        self.analysis_param_columns = []
        self.max_thumbnails = 8
        self.thumb_cols = 1
        self.batch_row_height = 64
        self.batch_row_height_min = 40
        self.batch_row_height_max = 320
        self.batch_thumbnail_aspect = 1.5
        self.batch_thumbnail_supersample = 5.0
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

        # Parameter initial values default to midpoint of bounds.
        self.defaults = self._default_param_midpoints(self.param_specs)

        # Optimization: timer for debouncing updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.do_full_update)
        self.slider_active = False

        # Cache for plot data
        self.cached_time_data = None
        self.channel_cache = {}
        self._expression_channel_data_cache = None
        self._display_target_points = 3000
        self._plot_has_residual_axis = False
        self._last_file_load_error = ""

        # Current directory
        self.current_dir = "./AFG_measurements/"
        self._source_display_override = None
        self._source_selected_paths = []

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

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.manual_tab = QWidget()
        self.batch_tab = QWidget()
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.manual_tab, "Plot")
        self.tabs.addTab(self.batch_tab, "Batch Processing")
        self.tabs.addTab(self.analysis_tab, "Batch Analysis")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        manual_layout = QVBoxLayout(self.manual_tab)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(4)

        # Manual mode: interactive plot only (controls are shared above tabs).
        self.create_plot_frame(manual_layout)

        batch_layout = QVBoxLayout(self.batch_tab)
        batch_layout.setContentsMargins(6, 6, 6, 6)
        batch_layout.setSpacing(6)
        self.create_batch_controls_frame(batch_layout)
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
        if self.tabs.currentWidget() not in (self.batch_tab, self.analysis_tab):
            return
        if not self._batch_preview_ready:
            self._batch_preview_ready = True
            self.prepare_batch_preview()
            self._expand_file_column_for_selected_files()
        if self.tabs.currentWidget() == self.analysis_tab:
            self._refresh_batch_analysis_if_run()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_source_path_label()

    def eventFilter(self, watched, event):
        if (
            watched is getattr(self, "param_header_widget", None)
            and event is not None
            and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)
        ):
            QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
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
                background: #f8fafc;
                color: #334155;
                border-color: #d7e0ea;
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
    ):
        combo = QComboBox()
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
        max_width = max(220, int(self.width() * 0.5))
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
            preview = "\n".join(
                display_name_for_file_ref(path) for path in selected_paths[:12]
            )
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

    def _default_param_midpoints(self, specs):
        midpoints = []
        for spec in specs:
            low = float(spec.min_value)
            high = float(spec.max_value)
            if low > high:
                low, high = high, low
            midpoints.append((low + high) * 0.5)
        return midpoints

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
        minimum=-1e12,
        maximum=1e12,
        width=72,
        object_name=None,
        tooltip=None,
    ):
        spinbox = CompactDoubleSpinBox()
        if object_name:
            spinbox.setObjectName(str(object_name))
        spinbox.setDecimals(int(spec.decimals))
        spinbox.setRange(float(minimum), float(maximum))
        spinbox.setSingleStep(float(spec.inferred_step))
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setKeyboardTracking(False)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        spinbox.setFixedWidth(int(width))
        if tooltip:
            spinbox.setToolTip(str(tooltip))
        spinbox.setValue(float(value))
        return spinbox

    def _sync_fit_optimization_controls(self):
        if not hasattr(self, "fit_enabled_cb"):
            return
        enabled = bool(self.fit_enabled_cb.isChecked())
        for widget in getattr(self, "_fit_robust_widgets", []):
            widget.setEnabled(enabled)

        de_enabled = enabled and bool(self.fit_use_de_cb.isChecked())
        for widget in getattr(self, "_fit_de_widgets", []):
            widget.setEnabled(de_enabled)

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

    def _current_fit_optimization_options(self):
        base = (
            self.fit_options
            if isinstance(self.fit_options, FitOptimizationOptions)
            else FitOptimizationOptions()
        )
        if not hasattr(self, "fit_enabled_cb"):
            return base
        opts = FitOptimizationOptions(
            enabled=bool(self.fit_enabled_cb.isChecked()),
            n_starts=max(2, int(self.fit_n_starts_spin.value())),
            per_start_maxfev=max(20, int(self.fit_per_start_fev_spin.value())),
            seed=base.seed,
            use_global_init=bool(self.fit_use_de_cb.isChecked()),
            de_maxiter=max(1, int(self.fit_de_maxiter_spin.value())),
            de_popsize=max(2, int(self.fit_de_popsize_spin.value())),
            early_stop_r2=float(self.fit_early_stop_r2_spin.value()),
            early_stop_patience=max(0, int(self.fit_patience_spin.value())),
        )
        self.fit_options = opts
        return opts

    def _make_param_header_label(self, text, width=None):
        return self._new_label(
            text,
            object_name="paramHeader",
            width=width,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    def _display_symbol_for_param(self, key, symbol_hint=None):
        return resolve_parameter_symbol(key, symbol_hint)

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

    def _create_param_label(self, spec, width):
        """Create a one-line parameter label."""
        symbol_text = self._display_symbol_for_param(spec.key, spec.symbol)
        tooltip = str(spec.description)
        if symbol_text != spec.key:
            tooltip = f"{tooltip} ({spec.key})"
        return self._new_label(
            f"{symbol_text}:",
            object_name="paramInline",
            tooltip=tooltip,
            width=width,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

    def _ordered_param_keys(self):
        return [spec.key for spec in self.param_specs]

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
            if (
                spec.key in self.param_min_spinboxes
                and spec.key in self.param_max_spinboxes
            ):
                low = float(self.param_min_spinboxes[spec.key].value())
                high = float(self.param_max_spinboxes[spec.key].value())
            else:
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

        value_box.blockSignals(True)
        value_box.setMinimum(low)
        value_box.setMaximum(high)
        value_box.setValue(float(np.clip(value_box.value(), low, high)))
        value_box.blockSignals(False)
        self._sync_slider_from_spinbox(key)
        self.update_plot(fast=False)

    def _build_fit_context(self, channel_data=None):
        ordered_keys = self._ordered_param_keys()
        if not ordered_keys:
            raise ValueError("No parameters are available for fitting.")

        current_values = self.get_current_param_map()
        lower, upper = self._parameter_bounds()
        bounds_by_key = {
            key: (float(lower[idx]), float(upper[idx]))
            for idx, key in enumerate(ordered_keys)
        }

        active_keys = self.get_fit_parameter_keys()

        p0 = []
        lower_active = []
        upper_active = []
        for key in active_keys:
            low, high = bounds_by_key[key]
            if np.isclose(low, high):
                raise ValueError(
                    f"Bounds for '{key}' are equal; expand them before fitting."
                )
            start = float(np.clip(current_values[key], low, high))
            p0.append(start)
            lower_active.append(low)
            upper_active.append(high)

        evaluator = self._compiled_expression
        if evaluator is None:
            raise ValueError("No compiled model expression is available.")
        baseline = dict(current_values)
        channels = channel_data or {}

        def fit_model(x_data, *fit_params):
            values = dict(baseline)
            for idx, key in enumerate(active_keys):
                values[key] = float(fit_params[idx])
            return evaluator(
                x_data,
                values,
                column_data=channels,
            )

        return {
            "ordered_keys": ordered_keys,
            "active_keys": active_keys,
            "base_values": baseline,
            "p0": np.asarray(p0, dtype=float),
            "bounds": (lower_active, upper_active),
            "fit_model": fit_model,
        }

    def _merge_active_fit_result(
        self, ordered_keys, active_keys, base_values, fitted_active_values
    ):
        merged = dict(base_values)
        for idx, key in enumerate(active_keys):
            if idx < len(fitted_active_values):
                merged[key] = float(fitted_active_values[idx])
        return [float(merged[key]) for key in ordered_keys]

    def evaluate_model_map(self, x_data, param_values, channel_data=None):
        if self._compiled_expression is None:
            raise ValueError("No compiled function expression is available.")
        return self._compiled_expression(
            x_data,
            param_values,
            column_data=channel_data,
        )

    def evaluate_model(self, x_data, params, channel_data=None):
        """Evaluate active fit model from either ordered list or key-value map."""
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
        )

    def _snapshot_full_model_function(self):
        ordered_keys = list(self._ordered_param_keys())
        evaluator = self._compiled_expression
        if evaluator is None:
            raise ValueError("No compiled function expression is available.")

        def model_func(x_data, *params, column_data=None):
            if len(params) != len(ordered_keys):
                raise ValueError(
                    f"Expected {len(ordered_keys)} parameters, got {len(params)}."
                )
            values = {key: float(params[idx]) for idx, key in enumerate(ordered_keys)}
            return evaluator(
                x_data,
                values,
                column_data=column_data,
            )

        return model_func

    def _set_formula_label(self):
        """Populate the formula label from the active expression."""
        target_col = None
        rhs_expression = None
        try:
            target_col, rhs_expression, _rhs_columns = self._parse_equation_text(
                self.current_expression,
                strict=False,
            )
        except Exception:
            target_col = None
            rhs_expression = None

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
        self.formula_label.setTextFormat(Qt.TextFormat.RichText)
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
                param_names = extract_expression_parameter_names(
                    rhs_expression,
                    reserved_names=self._expression_reserved_names(),
                )
                for name in param_names:
                    param_tokens.add(str(symbol_map.get(name, name)))
            except Exception:
                param_tokens = set()

        column_tokens = {str(name) for name in self._available_channel_names()}
        if target_col:
            column_tokens.add(str(target_col))
        column_tokens_upper = {token.upper() for token in column_tokens}
        constant_tokens = {"π", "e", "pi"}

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
        token_parts = [number_token_re, r"[A-Za-z_][A-Za-z0-9_]*"]
        token_parts.extend(re.escape(token) for token in special_tokens)
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
            escaped = html.escape(token)
            style = None
            if token.upper() in column_tokens_upper:
                style = f"color:{_EXPRESSION_COLUMN_COLOR}; font-weight:600;"
            elif token in param_tokens:
                style = f"color:{_EXPRESSION_PARAM_COLOR}; font-weight:600;"
            elif token in constant_tokens or number_re.fullmatch(token):
                style = f"color:{_EXPRESSION_CONSTANT_COLOR};"
            if style:
                parts.append(f'<span style="{style}">{escaped}</span>')
            else:
                parts.append(escaped)
            cursor = end

        if cursor < len(text):
            parts.append(html.escape(text[cursor:]))
        return "".join(parts)

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
        if self._plot_mouse_cid is None:
            self._plot_mouse_cid = self.canvas.mpl_connect(
                "button_press_event",
                self.on_plot_mouse_press,
            )

        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QSize(14, 14))
        self.toolbar.setMaximumHeight(28)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self._recreate_fit_region_selector()

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_parameters_frame(self, parent_layout):
        """Create full-width controls + parameters section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        top_controls_layout = QHBoxLayout()
        top_controls_layout.setSpacing(6)

        equation_host = QWidget()
        equation_host_layout = QVBoxLayout(equation_host)
        equation_host_layout.setContentsMargins(0, 0, 0, 0)
        equation_host_layout.setSpacing(0)
        equation_slot_layout = QGridLayout()
        equation_slot_layout.setContentsMargins(0, 0, 0, 0)
        equation_slot_layout.setSpacing(0)
        self.formula_label = ClickableLabel(self.current_expression)
        self.formula_label.setMinimumHeight(24)
        self.formula_label.setMaximumHeight(56)
        self.formula_label.setWordWrap(True)
        self.formula_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        self.formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_label.clicked.connect(self._enter_expression_edit_mode)
        equation_slot_layout.addWidget(self.formula_label, 0, 0)

        self.expression_editor_widget = QWidget()
        self.expression_editor_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
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
            "Example: CH2 = offset + amp*sin(2*pi*freq*CH3 + phase)"
        )
        self.function_input.setPlainText(self.current_expression)
        self.function_input.setMinimumHeight(self.formula_label.minimumHeight())
        self.function_input.setMaximumHeight(self.formula_label.maximumHeight())
        self.function_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.function_input.setStyleSheet("padding: 1px 4px;")
        self.function_input.document().setDocumentMargin(1)
        self.function_input.setToolTip("Equation format: target_column = expression")
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
        top_controls_layout.addWidget(equation_host, 5)

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
        top_controls_layout.addWidget(source_file_widget, 3)

        fit_opts = (
            self.fit_options
            if isinstance(self.fit_options, FitOptimizationOptions)
            else FitOptimizationOptions()
        )
        fit_widget = QWidget()
        fit_widget_layout = QVBoxLayout(fit_widget)
        fit_widget_layout.setContentsMargins(0, 0, 0, 0)
        fit_widget_layout.setSpacing(2)
        fit_options_layout = QGridLayout()
        fit_options_layout.setHorizontalSpacing(6)
        fit_options_layout.setVerticalSpacing(2)

        fit_enabled_tooltip = "Enable deterministic multi-start fitting."
        self.fit_enabled_label = self._new_label("Robust")
        self.fit_enabled_label.setToolTip(fit_enabled_tooltip)
        self.fit_enabled_cb = self._new_checkbox(
            "",
            checked=fit_opts.enabled,
            tooltip=fit_enabled_tooltip,
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(self.fit_enabled_label, self.fit_enabled_cb),
            0,
            0,
        )
        fit_use_de_tooltip = "Run differential evolution before local solves."
        self.fit_use_de_label = self._new_label("DE Init")
        self.fit_use_de_label.setToolTip(fit_use_de_tooltip)
        self.fit_use_de_cb = self._new_checkbox(
            "",
            checked=fit_opts.use_global_init,
            tooltip=fit_use_de_tooltip,
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(self.fit_use_de_label, self.fit_use_de_cb),
            0,
            1,
        )
        self.fit_n_starts_spin = self._new_compact_int_spinbox(
            2,
            128,
            fit_opts.n_starts,
            tooltip="Number of bounded local starts.",
        )
        self.fit_de_maxiter_spin = self._new_compact_int_spinbox(
            1, 500, fit_opts.de_maxiter
        )
        self.fit_de_popsize_spin = self._new_compact_int_spinbox(
            2, 200, fit_opts.de_popsize
        )
        self.fit_starts_label = self._new_label("#")
        fit_options_layout.addWidget(
            self._build_fit_option_cell(self.fit_starts_label, self.fit_n_starts_spin),
            1,
            0,
        )
        self.fit_fev_label = self._new_label("Calls")
        self.fit_per_start_fev_spin = self._new_compact_int_spinbox(
            20,
            200000,
            fit_opts.per_start_maxfev,
            single_step=50,
            tooltip="Maximum evaluations per local solve.",
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_fev_label,
                self.fit_per_start_fev_spin,
            ),
            1,
            1,
        )
        self.fit_stop_r2_label = self._new_label("R2")
        self.fit_early_stop_r2_spin = self._new_compact_float_spinbox(
            0.0,
            1.0,
            fit_opts.early_stop_r2,
            decimals=3,
            single_step=0.001,
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_stop_r2_label,
                self.fit_early_stop_r2_spin,
            ),
            3,
            0,
        )
        self.fit_de_iter_label = self._new_label("DE iter")
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_de_iter_label,
                self.fit_de_maxiter_spin,
            ),
            2,
            0,
        )
        self.fit_de_pop_label = self._new_label("DE prop")
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_de_pop_label,
                self.fit_de_popsize_spin,
            ),
            2,
            1,
        )
        self.fit_patience_label = self._new_label("Patience")
        self.fit_patience_spin = self._new_compact_int_spinbox(
            0,
            64,
            fit_opts.early_stop_patience,
            tooltip="Stop after this many non-improving starts.",
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_patience_label,
                self.fit_patience_spin,
            ),
            3,
            1,
        )
        fit_options_layout.setColumnStretch(0, 1)
        fit_options_layout.setColumnStretch(1, 1)

        self._fit_robust_widgets = [
            self.fit_use_de_label,
            self.fit_use_de_cb,
            self.fit_starts_label,
            self.fit_n_starts_spin,
            self.fit_fev_label,
            self.fit_per_start_fev_spin,
            self.fit_patience_label,
            self.fit_patience_spin,
            self.fit_stop_r2_label,
            self.fit_early_stop_r2_spin,
        ]
        self._fit_de_widgets = [
            self.fit_de_iter_label,
            self.fit_de_maxiter_spin,
            self.fit_de_pop_label,
            self.fit_de_popsize_spin,
        ]

        self.fit_enabled_cb.toggled.connect(self._sync_fit_optimization_controls)
        self.fit_use_de_cb.toggled.connect(self._sync_fit_optimization_controls)

        fit_actions_row = QHBoxLayout()
        fit_actions_row.setSpacing(4)
        self.auto_fit_btn = self._new_button(
            "Auto Fit",
            handler=self.auto_fit,
            primary=True,
        )
        fit_actions_row.addWidget(self.auto_fit_btn)

        self.cancel_fit_btn = self._new_button(
            "Cancel",
            handler=self.cancel_auto_fit,
            enabled=False,
        )
        fit_actions_row.addWidget(self.cancel_fit_btn)

        self.show_residuals_cb = self._new_button(
            "Residuals",
            checkable=True,
            checked=False,
            toggled_handler=lambda: self.update_plot(fast=False),
        )
        fit_actions_row.addWidget(self.show_residuals_cb)
        fit_actions_row.addStretch(1)
        fit_widget_layout.addLayout(fit_actions_row)

        fit_widget_layout.addLayout(fit_options_layout)
        self.fit_result_summary_label = self._new_label(
            "R² (fit/full): N/A / N/A",
            object_name="statusLabel",
        )
        fit_widget_layout.addWidget(self.fit_result_summary_label)
        self._sync_fit_optimization_controls()
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
        param_header_layout.addWidget(
            self._make_param_header_label("StdErr", width=self._param_error_width)
        )
        params_and_fit_layout = QHBoxLayout()
        params_and_fit_layout.setSpacing(8)

        params_left_widget = QWidget()
        params_left_layout = QVBoxLayout(params_left_widget)
        params_left_layout.setContentsMargins(0, 0, 0, 0)
        params_left_layout.setSpacing(6)
        self._param_header_to_rows_gap = params_left_layout.spacing()
        params_left_layout.addWidget(self.param_header_widget)

        self.param_controls_layout = QVBoxLayout()
        self.param_controls_layout.setSpacing(6)
        params_left_layout.addLayout(self.param_controls_layout)
        params_and_fit_layout.addWidget(params_left_widget, 1)

        fit_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        fit_right_widget = QWidget()
        fit_right_layout = QVBoxLayout(fit_right_widget)
        fit_right_layout.setContentsMargins(0, 0, 0, 0)
        fit_right_layout.setSpacing(0)
        self.fit_panel_top_spacer = QWidget()
        self.fit_panel_top_spacer.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.fit_panel_top_spacer.setFixedHeight(0)
        fit_right_layout.addWidget(self.fit_panel_top_spacer)
        fit_right_layout.addWidget(fit_widget, 0, Qt.AlignmentFlag.AlignTop)
        params_and_fit_layout.addWidget(fit_right_widget, 0, Qt.AlignmentFlag.AlignTop)
        params_status_layout = QVBoxLayout()
        params_status_layout.setContentsMargins(0, 0, 0, 0)
        params_status_layout.setSpacing(0)
        params_status_layout.addLayout(params_and_fit_layout)
        layout.addLayout(params_status_layout)
        self.rebuild_manual_param_controls()
        self._rebuild_channel_token_buttons()
        self._set_formula_label()
        self._set_expression_edit_mode(False)
        self._update_param_error_labels()
        self._sync_fit_panel_top_spacing()
        QTimer.singleShot(0, self._sync_fit_panel_top_spacing)
        self._sync_param_row_tail_spacers()
        QTimer.singleShot(0, self._sync_param_row_tail_spacers)

        self.stats_text = SingleLineStatusLabel("")
        self.stats_text.setObjectName("statsLine")
        self.stats_text.setStyleSheet("padding: 0px 2px; margin: 0px;")
        params_status_layout.addWidget(self.stats_text)

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
                raise ValueError("Use equation form: Col = expression")
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

        rhs_columns = []
        seen = set()
        for token_name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs_text):
            resolved = self._resolve_column_name(token_name)
            if resolved is None or resolved in seen:
                continue
            seen.add(resolved)
            rhs_columns.append(resolved)

        return target_col, rhs_text, rhs_columns

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
                    _target, rhs_expr, _rhs_columns = self._parse_equation_text(
                        expression_text, strict=False
                    )
                    params = extract_expression_parameter_names(
                        rhs_expr, reserved_names=self._expression_reserved_names()
                    )
                except Exception:
                    params = []
            self.expression_highlighter.set_context(columns, params)
        finally:
            self._highlight_refresh_in_progress = False

    def _rebuild_channel_token_buttons(self):
        seen = []
        seen_set = set()
        for name in self._available_channel_names():
            if name in seen_set:
                continue
            seen_set.add(name)
            seen.append(name)

        if hasattr(self, "channel_token_menu"):
            self.channel_token_menu.clear()
            for token_name in seen:
                action = self.channel_token_menu.addAction(token_name)
                action.triggered.connect(
                    lambda _checked=False, t=token_name: self._insert_expression_token(
                        t
                    )
                )
            if hasattr(self, "insert_token_btn"):
                self.insert_token_btn.setEnabled(bool(seen))

        if hasattr(self, "channel_tokens_layout"):
            clear_layout(self.channel_tokens_layout)
            for token_name in seen:
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

            try:
                target_col, rhs_expression, rhs_columns = self._parse_equation_text(
                    expression_text, strict=True
                )
                self.y_channel = target_col
                if rhs_columns:
                    if self.x_channel not in rhs_columns:
                        self.x_channel = rhs_columns[0]
                elif self.x_channel == self.y_channel:
                    available = [
                        col
                        for col in self._available_channel_names()
                        if col != self.y_channel
                    ]
                    if available:
                        self.x_channel = available[0]

                reserved_names = self._expression_reserved_names()
                param_names = extract_expression_parameter_names(
                    rhs_expression, reserved_names=reserved_names
                )
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
                        min_val, max_val = -20.0, 20.0

                    min_val = float(min_val)
                    max_val = float(max_val)
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    default_val = float((min_val + max_val) * 0.5)
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

                compiled = compile_expression_function(
                    rhs_expression,
                    param_names,
                )
            except Exception as exc:
                self._set_function_status(f"Function error: {exc}", is_error=True)
                self._refresh_expression_highlighting()
                return False

            self.param_specs = new_specs
            self.defaults = new_defaults
            self.current_expression = f"{target_col} = {rhs_expression}"
            self._set_expression_editor_text(self.current_expression)
            self._compiled_expression = compiled
            self.last_popt = None
            self.last_pcov = None
            self.last_fit_r2 = None
            self.last_full_r2 = None
            self._last_fit_active_keys = []
            self.rebuild_manual_param_controls()
            self._refresh_channel_combos()
            self._set_formula_label()
            self._set_function_status("", is_error=False)
            self._refresh_expression_highlighting()
            self.update_plot(fast=False)
            if self.batch_results:
                for row in self.batch_results:
                    row["params"] = None
                    row["r2"] = None
                    row["error"] = None
                    row["plot_full"] = None
                    row["plot"] = None
                self.update_batch_table()
                self._refresh_batch_analysis_if_run()
                self.queue_visible_thumbnail_render()
            return True
        finally:
            self._apply_expression_in_progress = False

    def create_param_control(self, spec, default_val):
        """Create a compact row for lower/upper bounds, value, and fit stderr."""
        key = spec.key
        layout = QHBoxLayout()
        layout.setSpacing(6)

        layout.addWidget(self._create_param_label(spec, width=self._param_name_width))

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
        layout.addWidget(value_box)

        std_err_label = self._new_label(
            "",
            width=self._param_error_width,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            object_name="paramInline",
            tooltip="Auto-fit standard error for this parameter.",
        )
        layout.addWidget(std_err_label)
        tail_spacer = QWidget()
        tail_spacer.setFixedWidth(max(0, int(self._param_tail_placeholder_width)))
        tail_spacer.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(tail_spacer)

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

        slider.valueChanged.connect(slider_to_spinbox)
        slider.sliderPressed.connect(slider_pressed)
        slider.sliderReleased.connect(slider_released)

        return (layout, value_box, slider, min_box, max_box, std_err_label, tail_spacer)

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
        self.param_error_labels.clear()
        self.param_row_tail_spacers.clear()

        for idx, spec in enumerate(self.param_specs):
            default_val = self.defaults[idx] if idx < len(self.defaults) else 0.0

            (
                control_layout,
                spinbox,
                slider,
                min_box,
                max_box,
                std_err_label,
                tail_spacer,
            ) = self.create_param_control(spec, default_val)
            self.param_spinboxes[spec.key] = spinbox
            self.param_sliders[spec.key] = slider
            self.param_min_spinboxes[spec.key] = min_box
            self.param_max_spinboxes[spec.key] = max_box
            self.param_error_labels[spec.key] = std_err_label
            self.param_row_tail_spacers.append(tail_spacer)
            self.param_controls_layout.addLayout(control_layout)
            self._sync_slider_from_spinbox(spec.key)
        self._sync_param_row_tail_spacers()
        QTimer.singleShot(0, self._sync_param_row_tail_spacers)

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
            "Set display names used in legends and labels.",
            style_sheet="color: #4b5563;",
        )
        dialog_layout.addWidget(help_label)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        editors = {}
        first_editor = None
        for channel_name in channel_names:
            editor = self._new_line_edit(str(self.channels.get(channel_name, channel_name)))
            editor.setPlaceholderText(str(channel_name))
            editor.setToolTip(
                f"Display label for {channel_name}. Leave blank to use {channel_name}."
            )
            form_layout.addRow(f"{channel_name}:", editor)
            editors[channel_name] = editor
            if first_editor is None:
                first_editor = editor
        dialog_layout.addLayout(form_layout)

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

        for channel_name, editor in editors.items():
            value = editor.text().strip()
            self.channels[channel_name] = value or channel_name

        self.update_plot(fast=False)

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

            x_choice = (
                self.x_channel
                if self.x_channel in channel_columns
                else ("CH3" if "CH3" in channel_columns else channel_columns[0])
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
                        _old_target, rhs_expr, _rhs_cols = self._parse_equation_text(
                            expr_text, strict=False
                        )
                        normalized = f"{self.y_channel} = {rhs_expr}"
                        if normalized != expr_text:
                            self.current_expression = normalized
                            self._set_expression_editor_text(normalized)
                    except Exception:
                        pass
            self._rebuild_channel_token_buttons()
        finally:
            self._channel_sync_in_progress = False

    def _set_fit_region(self, start_pct, end_pct, refresh=True):
        start = float(np.clip(start_pct, 0.0, 100.0))
        end = float(np.clip(end_pct, 0.0, 100.0))
        if start > end:
            start, end = end, start

        self.fit_region_start_pct = start
        self.fit_region_end_pct = end

        if refresh:
            self.update_plot(fast=False)

    def _schedule_fit_region_refresh(self):
        """Schedule redraw after SpanSelector event completes to avoid UI lag."""
        if self._fit_region_refresh_pending:
            return
        self._fit_region_refresh_pending = True

        def _refresh():
            self._fit_region_refresh_pending = False
            self.update_plot(fast=False)

        QTimer.singleShot(0, _refresh)

    def reset_fit_region(self):
        self._set_fit_region(0.0, 100.0, refresh=True)

    def _fit_region_is_full_area(self):
        return bool(
            np.isclose(self.fit_region_start_pct, 0.0, atol=1e-9)
            and np.isclose(self.fit_region_end_pct, 100.0, atol=1e-9)
        )

    def get_fit_slice(self, n_points):
        start = int(
            np.floor((self.fit_region_start_pct / 100.0) * max(0, n_points - 1))
        )
        end = int(np.ceil((self.fit_region_end_pct / 100.0) * max(0, n_points - 1))) + 1
        start = max(0, min(n_points - 1, start)) if n_points else 0
        end = max(start + 1, min(n_points, end)) if n_points else 0
        return slice(start, end)

    def _fit_window_times(self, time_data, fit_slice):
        n_points = len(time_data)
        if n_points == 0:
            return None, None, 0, 0
        start_idx = int(fit_slice.start if fit_slice.start is not None else 0)
        end_idx = int(fit_slice.stop if fit_slice.stop is not None else n_points)
        start_idx = max(0, min(n_points - 1, start_idx))
        end_idx = max(start_idx + 1, min(n_points, end_idx))
        return (
            float(time_data[start_idx]),
            float(time_data[end_idx - 1]),
            start_idx,
            end_idx,
        )

    def _draw_fit_window_overlay(self, time_data, fit_slice):
        fit_start_t, fit_end_t, start_idx, end_idx = self._fit_window_times(
            time_data, fit_slice
        )
        self._fit_window_bounds_ms = (fit_start_t, fit_end_t)
        boundary_positions = []
        if fit_start_t is None or fit_end_t is None:
            self._fit_boundary_positions_ms = ()
            return

        if start_idx > 0:
            self.ax.axvline(
                fit_start_t,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
                label="Fit boundary",
            )
            boundary_positions.append(float(fit_start_t))
        if end_idx < len(time_data):
            self.ax.axvline(
                fit_end_t,
                color="#dc2626",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
                label="_nolegend_",
            )
            boundary_positions.append(float(fit_end_t))
        self._fit_boundary_positions_ms = tuple(boundary_positions)

    def _split_inside_outside_fit(self, values, fit_slice):
        values = np.asarray(values, dtype=float)
        inside = np.full_like(values, np.nan, dtype=float)
        outside = values.copy()
        inside[fit_slice] = values[fit_slice]
        outside[fit_slice] = np.nan
        return inside, outside

    def _toolbar_mode_active(self):
        return bool(getattr(self.toolbar, "mode", ""))

    def _fit_boundary_click_distance_px(self, event):
        if event is None or event.x is None:
            return None
        if not self._fit_boundary_positions_ms:
            return None
        distances = []
        for x_data in self._fit_boundary_positions_ms:
            try:
                x_px = float(self.ax.transData.transform((float(x_data), 0.0))[0])
            except Exception:
                continue
            distances.append(abs(float(event.x) - x_px))
        if not distances:
            return None
        return float(min(distances))

    def on_plot_mouse_press(self, event):
        if not bool(getattr(event, "dblclick", False)):
            return
        if self.current_data is None or self.cached_time_data is None:
            return
        if event.inaxes != self.ax:
            return
        if self._toolbar_mode_active():
            return
        if self._fit_region_is_full_area():
            return

        distance_px = self._fit_boundary_click_distance_px(event)
        if distance_px is None:
            return
        if distance_px <= float(self._fit_boundary_pick_px):
            self.reset_fit_region()

    def _recreate_fit_region_selector(self):
        if not hasattr(self, "ax"):
            return

        if self.fit_region_selector is not None:
            try:
                self.fit_region_selector.set_active(False)
                self.fit_region_selector.disconnect_events()
            except Exception:
                pass
            self.fit_region_selector = None

        selector_kwargs = dict(
            useblit=True,
            interactive=True,
            props=dict(
                facecolor="none",
                edgecolor="#dc2626",
                linewidth=1.1,
                alpha=0.9,
            ),
        )
        try:
            self.fit_region_selector = SpanSelector(
                self.ax,
                self.on_fit_span_selected,
                "horizontal",
                drag_from_anywhere=True,
                **selector_kwargs,
            )
        except TypeError:
            self.fit_region_selector = SpanSelector(
                self.ax,
                self.on_fit_span_selected,
                "horizontal",
                **selector_kwargs,
            )

    def _sync_fit_region_selector(self):
        if self.fit_region_selector is None:
            return
        fit_start_t, fit_end_t = self._fit_window_bounds_ms
        if fit_start_t is None or fit_end_t is None:
            return
        self._suppress_fit_region_selector = True
        try:
            self.fit_region_selector.extents = (float(fit_start_t), float(fit_end_t))
        finally:
            self._suppress_fit_region_selector = False

    def on_fit_span_selected(self, x_min, x_max):
        if self._suppress_fit_region_selector:
            return
        if self.current_data is None or self.cached_time_data is None:
            return
        if self._toolbar_mode_active():
            return
        if x_min is None or x_max is None:
            return

        lo = float(min(x_min, x_max))
        hi = float(max(x_min, x_max))
        if np.isclose(lo, hi):
            return

        time_data = self.cached_time_data
        if len(time_data) < 2:
            return

        t_min = float(min(time_data[0], time_data[-1]))
        t_max = float(max(time_data[0], time_data[-1]))
        if np.isclose(t_min, t_max):
            return

        lo = float(np.clip(lo, t_min, t_max))
        hi = float(np.clip(hi, t_min, t_max))
        if hi <= lo:
            return

        start_pct = ((lo - t_min) / (t_max - t_min)) * 100.0
        end_pct = ((hi - t_min) / (t_max - t_min)) * 100.0
        min_gap = 100.0 / max(1, len(time_data) - 1)
        if (end_pct - start_pct) < min_gap:
            center = 0.5 * (start_pct + end_pct)
            start_pct = max(0.0, center - (0.5 * min_gap))
            end_pct = min(100.0, start_pct + min_gap)
            if (end_pct - start_pct) < min_gap:
                start_pct = max(0.0, end_pct - min_gap)

        self._set_fit_region(start_pct, end_pct, refresh=False)
        self._schedule_fit_region_refresh()

    def create_stats_frame(self, parent_layout):
        """Create file selection + auto-fit controls + stats display section."""
        group = QGroupBox("")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

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
        layout.addLayout(source_layout)

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
        layout.addLayout(file_layout)

        # Auto-fit control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.auto_fit_btn = self._new_button(
            "Auto Fit",
            handler=self.auto_fit,
            primary=True,
        )
        btn_layout.addWidget(self.auto_fit_btn)

        self.cancel_fit_btn = self._new_button(
            "Cancel",
            handler=self.cancel_auto_fit,
            enabled=False,
        )
        btn_layout.addWidget(self.cancel_fit_btn)

        self.show_residuals_cb = self._new_button(
            "Residuals",
            checkable=True,
            checked=False,
            toggled_handler=lambda: self.update_plot(fast=False),
        )
        btn_layout.addWidget(self.show_residuals_cb)

        layout.addLayout(btn_layout)

        fit_opts = (
            self.fit_options
            if isinstance(self.fit_options, FitOptimizationOptions)
            else FitOptimizationOptions()
        )
        fit_options_layout = QGridLayout()
        fit_options_layout.setHorizontalSpacing(6)
        fit_options_layout.setVerticalSpacing(2)

        fit_enabled_tooltip = "Enable deterministic multi-start fitting."
        self.fit_enabled_label = self._new_label("Robust")
        self.fit_enabled_label.setToolTip(fit_enabled_tooltip)
        self.fit_enabled_cb = self._new_checkbox(
            "",
            checked=fit_opts.enabled,
            tooltip=fit_enabled_tooltip,
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(self.fit_enabled_label, self.fit_enabled_cb),
            0,
            0,
        )
        fit_use_de_tooltip = "Run differential evolution before local solves."
        self.fit_use_de_label = self._new_label("DE Init")
        self.fit_use_de_label.setToolTip(fit_use_de_tooltip)
        self.fit_use_de_cb = self._new_checkbox(
            "",
            checked=fit_opts.use_global_init,
            tooltip=fit_use_de_tooltip,
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(self.fit_use_de_label, self.fit_use_de_cb),
            0,
            1,
        )
        self.fit_n_starts_spin = self._new_compact_int_spinbox(
            2,
            128,
            fit_opts.n_starts,
            tooltip="Number of bounded local starts.",
        )
        self.fit_de_maxiter_spin = self._new_compact_int_spinbox(
            1, 500, fit_opts.de_maxiter
        )
        self.fit_de_popsize_spin = self._new_compact_int_spinbox(
            2, 200, fit_opts.de_popsize
        )
        self.fit_starts_label = self._new_label("#")
        fit_options_layout.addWidget(
            self._build_fit_option_cell(self.fit_starts_label, self.fit_n_starts_spin),
            1,
            0,
        )
        self.fit_fev_label = self._new_label("Calls")
        self.fit_per_start_fev_spin = self._new_compact_int_spinbox(
            20,
            200000,
            fit_opts.per_start_maxfev,
            single_step=50,
            tooltip="Maximum evaluations per local solve.",
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_fev_label,
                self.fit_per_start_fev_spin,
            ),
            1,
            1,
        )
        self.fit_stop_r2_label = self._new_label("R2")
        self.fit_early_stop_r2_spin = self._new_compact_float_spinbox(
            0.0,
            1.0,
            fit_opts.early_stop_r2,
            decimals=3,
            single_step=0.001,
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_stop_r2_label,
                self.fit_early_stop_r2_spin,
            ),
            1,
            2,
        )

        self.fit_de_iter_label = self._new_label("DE iter")
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_de_iter_label,
                self.fit_de_maxiter_spin,
            ),
            2,
            0,
        )
        self.fit_de_pop_label = self._new_label("DE prop")
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_de_pop_label,
                self.fit_de_popsize_spin,
            ),
            2,
            1,
        )
        self.fit_patience_label = self._new_label("Patience")
        self.fit_patience_spin = self._new_compact_int_spinbox(
            0,
            64,
            fit_opts.early_stop_patience,
            tooltip="Stop after this many non-improving starts.",
        )
        fit_options_layout.addWidget(
            self._build_fit_option_cell(
                self.fit_patience_label,
                self.fit_patience_spin,
            ),
            2,
            2,
        )
        fit_options_layout.setColumnStretch(0, 1)
        fit_options_layout.setColumnStretch(1, 1)
        fit_options_layout.setColumnStretch(2, 1)

        self._fit_robust_widgets = [
            self.fit_use_de_label,
            self.fit_use_de_cb,
            self.fit_starts_label,
            self.fit_n_starts_spin,
            self.fit_fev_label,
            self.fit_per_start_fev_spin,
            self.fit_stop_r2_label,
            self.fit_early_stop_r2_spin,
            self.fit_patience_label,
            self.fit_patience_spin,
        ]
        self._fit_de_widgets = [
            self.fit_de_iter_label,
            self.fit_de_maxiter_spin,
            self.fit_de_pop_label,
            self.fit_de_popsize_spin,
        ]

        self.fit_enabled_cb.toggled.connect(self._sync_fit_optimization_controls)
        self.fit_use_de_cb.toggled.connect(self._sync_fit_optimization_controls)
        self._sync_fit_optimization_controls()
        layout.addLayout(fit_options_layout)

        # Single-line shared status display
        self.stats_text = SingleLineStatusLabel("")
        layout.addWidget(self.stats_text)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_batch_controls_frame(self, parent_layout):
        """Create batch-only controls (shared params/settings are above tabs)."""
        group = QGroupBox("")
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        batch_label = self._new_label(
            "Batch actions (uses shared model/params/fit window above)",
            style_sheet="font-weight: 600; color: #374151; padding: 1px 2px;",
        )
        layout.addWidget(batch_label)

        self.run_batch_btn_default_text = "Run Batch"
        self.run_batch_btn = self._new_button(
            self.run_batch_btn_default_text,
            handler=self.run_batch_fit,
        )

        export_table_btn = self._new_button(
            "Export CSV", handler=self.export_batch_table
        )

        regex_layout = QHBoxLayout()
        regex_layout.setSpacing(4)
        regex_layout.addWidget(self._new_label("Pattern:"))
        self.regex_input = self._new_line_edit(
            "data_{freq}_{idx}_ALL",
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
            "Use {field} placeholders to extract columns.",
            object_name="statusLabel",
        )
        layout.addWidget(self.batch_parse_feedback_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)
        actions_row.addWidget(self.run_batch_btn)
        self.cancel_batch_btn = self._new_button(
            "Cancel",
            handler=self.cancel_batch_fit,
            enabled=False,
        )
        actions_row.addWidget(self.cancel_batch_btn)
        actions_row.addWidget(export_table_btn)
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
        source_row.addWidget(self._new_label("Analysis Source:"))
        self.analysis_source_combo = self._new_combobox(
            items=[
                ("Completed Batch Run", "run"),
                ("Loaded Batch CSV", "csv"),
            ],
            current_index_changed=self._on_analysis_source_changed,
        )
        source_row.addWidget(self.analysis_source_combo)
        self.analysis_load_csv_btn = self._new_button(
            "Load Batch CSV",
            handler=self.load_batch_analysis_csv,
            enabled=False,
        )
        source_row.addWidget(self.analysis_load_csv_btn)
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

        self.analysis_refresh_btn = self._new_button(
            "Refresh",
            handler=lambda: self._refresh_batch_analysis_data(preserve_selection=True),
        )
        controls_row.addWidget(self.analysis_refresh_btn)
        layout.addLayout(controls_row)

        params_row = QHBoxLayout()
        params_row.setSpacing(4)
        params_row.addWidget(self._new_label("Parameters (Y):"))
        self.analysis_param_buttons = {}
        self.analysis_params_button_layout = QHBoxLayout()
        self.analysis_params_button_layout.setSpacing(4)
        params_row.addLayout(self.analysis_params_button_layout, 1)
        param_btn_col = QVBoxLayout()
        param_btn_col.setSpacing(4)
        select_all_btn = self._new_button(
            "Select All",
            handler=self._select_all_analysis_params,
        )
        param_btn_col.addWidget(select_all_btn)
        clear_btn = self._new_button("Clear", handler=self._clear_analysis_params)
        param_btn_col.addWidget(clear_btn)
        param_btn_col.addStretch()
        params_row.addLayout(param_btn_col)
        layout.addLayout(params_row)

        self.analysis_fig = Figure(figsize=(10, 3.2), dpi=100)
        self.analysis_fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.2)
        self.analysis_canvas = FigureCanvas(self.analysis_fig)
        layout.addWidget(self.analysis_canvas)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _on_analysis_source_changed(self):
        source = self.analysis_source_combo.currentData()
        self.analysis_load_csv_btn.setEnabled(source == "csv")
        self._refresh_batch_analysis_data(preserve_selection=True)

    def _batch_row_error_text(self, row):
        pattern_error = str(row.get("pattern_error") or "").strip()
        fit_error = str(row.get("error") or "").strip()
        parts = []
        if pattern_error:
            parts.append(pattern_error)
        if fit_error and fit_error not in parts:
            parts.append(fit_error)
        return " | ".join(parts)

    def load_batch_analysis_csv(self):
        """Load a previously exported batch CSV for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Batch CSV",
            str(Path.cwd()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            frame = read_csv(file_path, header=0)
            self.analysis_csv_records = frame.to_dict("records")
            self.analysis_csv_path = file_path
            csv_idx = self.analysis_source_combo.findData("csv")
            if csv_idx >= 0:
                self.analysis_source_combo.setCurrentIndex(csv_idx)
            self._refresh_batch_analysis_data(preserve_selection=False)
            self.stats_text.append(
                f"✓ Loaded analysis CSV: {Path(file_path).name} ({len(self.analysis_csv_records)} rows)"
            )
        except Exception as exc:
            self.stats_text.append(f"✗ Failed to load analysis CSV: {exc}")

    def _extract_analysis_records_from_batch(self):
        records = []
        for row in self.batch_results:
            record = {"File": display_name_for_file_ref(row["file"])}
            captures = row.get("captures") or {}
            for key, value in captures.items():
                record[key] = value
            params = row.get("params") or []
            for idx, spec in enumerate(self.param_specs):
                record[spec.column_name] = params[idx] if idx < len(params) else None
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

    def _default_analysis_x_field(self, numeric_columns):
        for key in self.batch_capture_keys:
            if key in numeric_columns:
                return key
        for key in numeric_columns:
            if key not in self.analysis_param_columns and key != "R2":
                return key
        return numeric_columns[0] if numeric_columns else None

    def _refresh_batch_analysis_data(self, preserve_selection):
        source = self.analysis_source_combo.currentData()
        if source == "csv":
            raw_records = list(self.analysis_csv_records)
            if raw_records:
                file_name = (
                    Path(self.analysis_csv_path).name
                    if self.analysis_csv_path
                    else "CSV"
                )
                base_status = f"Loaded CSV: {file_name}"
            else:
                base_status = "Loaded CSV"
        else:
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
            spec.column_name
            for spec in self.param_specs
            if spec.column_name in self.analysis_numeric_data
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
        self.analysis_x_combo.addItem("Select X Axis...", None)
        for key in numeric_columns:
            self.analysis_x_combo.addItem(key, key)
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

    def _rebuild_analysis_param_buttons(self, previous_params):
        while self.analysis_params_button_layout.count():
            item = self.analysis_params_button_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.analysis_param_buttons = {}
        for key in self.analysis_param_columns:
            button = self._new_button(
                key,
                checkable=True,
                checked=(not previous_params or key in previous_params),
                toggled_handler=self.update_batch_analysis_plot,
            )
            self.analysis_params_button_layout.addWidget(button)
            self.analysis_param_buttons[key] = button
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
        self.analysis_fig.clear()
        ax = self.analysis_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        self.analysis_canvas.draw_idle()

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
            order = np.argsort(x_plot)
            x_sorted = x_plot[order]
            y_sorted = y_plot[order]
            color = f"C{idx % 10}"
            target_ax = axes[idx] if len(axes) > 1 else axes[0]

            if show_points:
                scatter_label = param_name if not show_series_line else "_nolegend_"
                target_ax.scatter(
                    x_sorted, y_sorted, s=26, color=color, label=scatter_label
                )
            if show_series_line:
                target_ax.plot(
                    x_sorted,
                    y_sorted,
                    linewidth=1.4,
                    alpha=0.85,
                    color=color,
                    label=param_name,
                )

            if show_fit_lines:
                fit = self._linear_fit(x_sorted, y_sorted)
                if fit is not None:
                    slope, intercept = fit
                    x_line = np.linspace(
                        float(np.min(x_sorted)), float(np.max(x_sorted)), 200
                    )
                    y_line = slope * x_line + intercept
                    fit_label = f"{param_name} fit" if len(axes) == 1 else "Best fit"
                    target_ax.plot(
                        x_line,
                        y_line,
                        linestyle="--",
                        linewidth=1.6,
                        color=color,
                        label=fit_label,
                    )

            if len(axes) > 1:
                target_ax.set_ylabel(param_name)
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

        axes[-1].set_xlabel(x_field)
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
            24,
            int(
                round(
                    (self.batch_row_height_max - 8) * self.batch_thumbnail_supersample
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
            first_name = display_name_for_file_ref(self.batch_files[0])
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
        preserve_fit_result=False,
    ):
        existing_row = existing or {}
        existing_plot_full = existing_row.get("plot_full")
        if existing_plot_full is None:
            existing_plot_full = existing_row.get("plot")
        if existing_plot_full is None:
            existing_plot_full = existing_row.get("thumbnail")
        return make_batch_result_row(
            source_index=source_index,
            file_path=file_path,
            x_channel=self.x_channel,
            y_channel=self.y_channel,
            captures=captures,
            params=existing_row.get("params") if preserve_fit_result else None,
            r2=existing_row.get("r2") if preserve_fit_result else None,
            error=existing_row.get("error") if preserve_fit_result else None,
            plot_full=existing_plot_full,
            plot=existing_row.get("plot"),
            fit_attempts=existing_row.get("fit_attempts"),
            fit_best_sse=existing_row.get("fit_best_sse"),
            fit_mode=existing_row.get("fit_mode"),
            fit_seed=existing_row.get("fit_seed"),
            fit_requested_starts=existing_row.get("fit_requested_starts"),
            fit_de_used=existing_row.get("fit_de_used"),
            pattern_error=pattern_error,
        )

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

    def _apply_data_file_list(self, files, *, empty_message):
        deduped_files = []
        seen = set()
        for file_ref in files:
            text = str(file_ref).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped_files.append(text)

        self.data_files = deduped_files
        self.file_combo.clear()
        self.current_file_idx = 0

        if not self.data_files:
            self.current_data = None
            self.cached_time_data = None
            self.channel_cache = {}
            self._expression_channel_data_cache = None
            self._last_file_load_error = ""
            self._sync_batch_files_from_shared(sync_pattern=False)
            self.stats_text.setText(str(empty_message))
            self._clear_main_plot("No data loaded.")
            self._sync_file_navigation_buttons()
            return False

        for file_ref in self.data_files:
            self.file_combo.addItem(display_name_for_file_ref(file_ref), file_ref)
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
                loaded_name = display_name_for_file_ref(self.data_files[loaded_idx])
                self.stats_text.setText(
                    f"Loaded '{loaded_name}' after skipping {loaded_idx} unreadable source file(s)."
                )
            elif self.stats_text.text().strip() == "Loading data sources...":
                self.stats_text.clear()
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
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel(self.y_channel if hasattr(self, "y_channel") else "Signal")
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
        self._fit_window_bounds_ms = (None, None)
        self._fit_boundary_positions_ms = ()
        self._recreate_fit_region_selector()
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
            self.last_fit_r2 = None
            self.last_full_r2 = None
            self._last_r2_fit = None
            self._last_r2_full = None

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
                self.channel_cache = {}
                for col in self.current_data.columns:
                    try:
                        self.channel_cache[col] = self.current_data[col].to_numpy(
                            dtype=float, copy=True
                        )
                    except Exception:
                        continue
                self._expression_channel_data_cache = dict(self.channel_cache)
                self._refresh_channel_combos()
                self.update_plot(fast=False)
                loaded_ok = True
            except Exception as e:
                self.current_data = None
                self.cached_time_data = None
                self.channel_cache = {}
                self._expression_channel_data_cache = None
                file_path = self.data_files[idx]
                file_name = display_name_for_file_ref(file_path)
                self._last_file_load_error = f"Error loading '{file_name}': {e}"
                if report_errors:
                    self.stats_text.setText(self._last_file_load_error)
                    self._clear_main_plot(f"Failed to load: {file_name}")
        finally:
            self._file_load_in_progress = False
            self._sync_file_navigation_buttons()
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

    def reset_params(self):
        """Reset all parameter values to midpoint of their current bounds."""
        for spec in self.param_specs:
            spinbox = self.param_spinboxes.get(spec.key)
            if spinbox is None:
                continue
            value = float((float(spinbox.minimum()) + float(spinbox.maximum())) * 0.5)
            spinbox.setValue(value)

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
        """Start auto-fit in a worker thread to keep GUI responsive."""
        if self.current_data is None:
            self.stats_text.append("No data loaded!")
            return

        if self.fit_thread is not None:
            self.stats_text.append("Auto-fit is already running.")
            return

        x_all = self._get_channel_data(self.x_channel)
        y_all = self._get_channel_data(self.y_channel)
        fit_slice = self.get_fit_slice(len(x_all))
        x_data = x_all[fit_slice]
        y_data = y_all[fit_slice]
        fit_channels = self._slice_channel_data(
            self._expression_channel_data(), fit_slice
        )

        try:
            fit_context = self._build_fit_context(channel_data=fit_channels)
        except Exception as exc:
            self.stats_text.append(f"Fit setup error: {exc}")
            return

        self._active_fit_keys = list(fit_context["active_keys"])
        self._fit_ordered_keys = list(fit_context["ordered_keys"])
        self._fit_base_values = dict(fit_context["base_values"])
        fit_options = self._current_fit_optimization_options()
        self.last_fit_diagnostics = None

        self.fit_thread = QThread(self)
        self.fit_worker = FitWorker(
            x_data,
            y_data,
            fit_context["p0"],
            fit_context["bounds"],
            fit_context["fit_model"],
            fit_options,
        )
        self.fit_worker.moveToThread(self.fit_thread)

        self.fit_thread.started.connect(self.fit_worker.run)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.failed.connect(self.on_fit_failed)
        self.fit_worker.cancelled.connect(self.on_fit_cancelled)

        self.auto_fit_btn.setEnabled(False)
        self.cancel_fit_btn.setEnabled(True)
        self.auto_fit_btn.setText("Fitting...")
        self.stats_text.append(
            "\nAuto-fit started "
            f"({fit_mode_label(fit_options)}, starts={max(2, int(fit_options.n_starts)) if fit_options.enabled else 1}, "
            f"seed={fit_options.seed}, DE={'on' if fit_options.enabled and fit_options.use_global_init else 'off'})..."
        )
        self.fit_thread.start()

    def on_fit_finished(self, popt, pcov, r2, diagnostics):
        """Handle successful fit completion."""
        active_params = np.asarray(popt, dtype=float)
        ordered_keys = (
            list(self._fit_ordered_keys)
            if self._fit_ordered_keys
            else self._ordered_param_keys()
        )
        active_keys = list(self._active_fit_keys)
        base_values = (
            dict(self._fit_base_values)
            if self._fit_base_values
            else self.get_current_param_map()
        )
        merged = self._merge_active_fit_result(
            ordered_keys, active_keys, base_values, active_params
        )
        self.last_popt = np.asarray(merged, dtype=float)
        self._last_fit_active_keys = list(active_keys)
        self.last_pcov = np.asarray(pcov, dtype=float)
        self.last_fit_r2 = float(r2)
        self.last_fit_diagnostics = (
            dict(diagnostics) if isinstance(diagnostics, dict) else {}
        )
        self.last_full_r2 = None
        if self.current_data is not None:
            try:
                x_all = self._get_channel_data(self.x_channel)
                y_all = self._get_channel_data(self.y_channel)
                fitted_all = self.evaluate_model(
                    x_all,
                    self.last_popt,
                    channel_data=self._expression_channel_data(),
                )
                self.last_full_r2 = compute_r2(y_all, fitted_all)
            except Exception:
                self.last_full_r2 = None

        for idx, key in enumerate(ordered_keys):
            if key in self.param_spinboxes and idx < len(self.last_popt):
                self.param_spinboxes[key].setValue(self.last_popt[idx])
        self.defaults = list(self.last_popt)

        full_r2_text = (
            f"{self.last_full_r2:.6f}" if self.last_full_r2 is not None else "N/A"
        )
        self.stats_text.append(
            f"✓ Auto-fit successful! R² (full trace) = {full_r2_text}"
        )
        diagnostics_text = build_fit_diagnostics_parts(
            mode=self.last_fit_diagnostics.get("mode"),
            attempts=self.last_fit_diagnostics.get("attempts"),
            requested_starts=self.last_fit_diagnostics.get("requested_starts"),
            seed=self.last_fit_diagnostics.get("seed"),
            de_used=(
                self.last_fit_diagnostics.get("de_used")
                if "de_used" in self.last_fit_diagnostics
                else None
            ),
            best_sse=self.last_fit_diagnostics.get("best_sse"),
            include_starts_suffix=True,
        )
        if diagnostics_text:
            self.stats_text.append("Fit diagnostics: " + ", ".join(diagnostics_text))
        summary = ", ".join(
            f"{self._display_name_for_param_key(key)}={self.last_popt[idx]:.4f}"
            for idx, key in enumerate(ordered_keys)
            if idx < len(self.last_popt)
        )
        self.stats_text.append(summary)
        self.update_plot()
        self.cleanup_fit_thread()

    def on_fit_failed(self, error_text):
        """Handle fit failures."""
        self.stats_text.append(f"✗ Auto-fit failed: {error_text}")
        self.stats_text.append(
            "Try increasing Starts / Fev per start, or enable DE Init, then retry."
        )
        self.cleanup_fit_thread()

    def on_fit_cancelled(self):
        """Handle fit cancellation."""
        self.stats_text.append("Auto-fit cancelled.")
        self.cleanup_fit_thread()

    def cancel_auto_fit(self):
        """Request cancellation of an in-flight auto-fit."""
        if self.fit_worker is not None:
            self._request_worker_cancel(self.fit_worker)
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
        self._active_fit_keys = []
        self._fit_ordered_keys = []
        self._fit_base_values = {}
        self.auto_fit_btn.setEnabled(True)
        self.auto_fit_btn.setText("Auto Fit")
        self.cancel_fit_btn.setEnabled(False)

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
        self.channel_cache[channel_name] = values
        return values

    def _display_indices(self, n_points):
        if n_points <= 0:
            return np.asarray([], dtype=int)
        target = max(1000, int(self._display_target_points))
        stride = max(1, int(np.ceil(n_points / float(target))))
        return np.arange(0, n_points, stride, dtype=int)

    def _fit_mask(self, n_points, fit_slice):
        mask = np.zeros(int(n_points), dtype=bool)
        if n_points <= 0:
            return mask
        start = int(fit_slice.start) if fit_slice.start is not None else 0
        stop = int(fit_slice.stop) if fit_slice.stop is not None else n_points
        start = max(0, min(n_points, start))
        stop = max(start, min(n_points, stop))
        mask[start:stop] = True
        return mask

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
        self._recreate_fit_region_selector()

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
        time_data = self.cached_time_data
        if time_data is None or len(time_data) != n_points:
            time_data = np.arange(n_points, dtype=float)

        fit_slice = self.get_fit_slice(n_points)
        fit_mask_full = self._fit_mask(n_points, fit_slice)
        display_idx = self._display_indices(n_points)
        if display_idx.size == 0:
            return None

        time_display = time_data[display_idx]
        x_display = x_data[display_idx]
        y_display = y_data[display_idx]
        fit_mask_display = fit_mask_full[display_idx]
        channel_data_display = self._slice_channel_data(channel_data_full, display_idx)

        time_axis_key = None
        if self.current_data is not None and len(self.current_data.columns) > 0:
            if "TIME" in self.current_data.columns:
                time_axis_key = "TIME"
            else:
                time_axis_key = str(self.current_data.columns[0]).strip()

        plot_channel_names = []
        for name in [self.y_channel, self.x_channel]:
            key = str(name).strip()
            if not key or key == time_axis_key or key in plot_channel_names:
                continue
            if key in channel_data_display:
                plot_channel_names.append(key)

        if self.current_data is not None:
            for col in self.current_data.columns:
                key = str(col).strip()
                if not key or key == time_axis_key or key in plot_channel_names:
                    continue
                if key in channel_data_display:
                    plot_channel_names.append(key)
        else:
            for key in channel_data_display.keys():
                key_text = str(key).strip()
                if (
                    not key_text
                    or key_text == time_axis_key
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
            "time_data": time_data,
            "fit_slice": fit_slice,
            "fit_mask_full": fit_mask_full,
            "display_idx": display_idx,
            "time_display": time_display,
            "x_display": x_display,
            "y_display": y_display,
            "fit_mask_display": fit_mask_display,
            "channel_data_display": channel_data_display,
            "plot_channel_displays": plot_channel_displays,
        }

    def _compute_display_series(self, context):
        fitted_display_full = self.evaluate_model(
            context["x_display"],
            context["params"],
            channel_data=context["channel_data_display"],
        )
        fitted_display = fitted_display_full

        residuals_display_full = context["y_display"] - fitted_display_full
        residuals_display = np.where(
            context["fit_mask_display"], residuals_display_full, np.nan
        )

        return {
            "fitted_display_full": fitted_display_full,
            "fitted_display": fitted_display,
            "residuals_display_full": residuals_display_full,
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
        self.ax.set_xlim(context["time_data"][0], context["time_data"][-1])
        if show_residuals and self.ax_residual is not None:
            r_min, r_max = self._finite_min_max(series["residuals_display"])
            self.ax_residual.set_ylim(r_min, r_max)
        self.canvas.draw_idle()
        return True

    def _update_param_error_labels(self):
        sigma_by_key = {}
        if self.last_pcov is not None:
            try:
                sigma = np.sqrt(np.maximum(np.diag(self.last_pcov), 0.0)).reshape(-1)
            except Exception:
                sigma = np.asarray([], dtype=float)
            for idx, key in enumerate(self._last_fit_active_keys):
                if idx < sigma.size and np.isfinite(sigma[idx]):
                    sigma_by_key[key] = float(sigma[idx])

        for spec in self.param_specs:
            label = self.param_error_labels.get(spec.key)
            if label is None:
                continue
            sigma_value = sigma_by_key.get(spec.key)
            if sigma_value is None:
                label.setText("")
            else:
                label.setText(f"{sigma_value:.6f}")

    def _update_stats_panel(self, fit_r2_value, full_r2_value):
        self._update_param_error_labels()
        if self.last_popt is not None and self.last_pcov is not None:
            fit_r2_text = f"{fit_r2_value:.6f}" if fit_r2_value is not None else "N/A"
            full_r2_text = (
                f"{full_r2_value:.6f}" if full_r2_value is not None else "N/A"
            )
            if hasattr(self, "fit_result_summary_label"):
                self.fit_result_summary_label.setText(
                    f"R² (fit/full): {fit_r2_text} / {full_r2_text}"
                )
        else:
            if hasattr(self, "fit_result_summary_label"):
                self.fit_result_summary_label.setText("R² (fit/full): N/A / N/A")
        self._sync_param_row_tail_spacers()

    def update_plot(self, fast=False):
        """Update plot with current parameters.

        Args:
            fast: If True, skip expensive operations for smooth slider interaction
        """
        if self.current_data is None:
            return

        # Debounce full updates during slider movement
        if fast and not self.slider_active:
            # Use timer to batch rapid updates
            self.update_timer.stop()
            self.update_timer.start(50)  # 50ms debounce
            return

        try:
            params = self.get_current_params()
            context = self._prepare_plot_context(params)
            if context is None:
                return
            series = self._compute_display_series(context)

            fit_start_t, fit_end_t, _, _ = self._fit_window_times(
                context["time_data"], context["fit_slice"]
            )
            self._fit_window_bounds_ms = (fit_start_t, fit_end_t)
            show_residuals = self.show_residuals_cb.isChecked()

            if fast and self._try_fast_plot_update(context, series, show_residuals):
                return

            self._ensure_plot_axes(show_residuals)
            self.ax.clear()
            if self.ax_residual is not None:
                self.ax_residual.clear()

            self._plot_lines = {}
            self._draw_fit_window_overlay(context["time_data"], context["fit_slice"])
            outside_alpha = 0.30
            for idx, (channel_name, values) in enumerate(
                context.get("plot_channel_displays", {}).items()
            ):
                color = f"C{idx % 10}"
                channel_inside = np.where(context["fit_mask_display"], values, np.nan)
                channel_outside = np.where(
                    context["fit_mask_display"], np.nan, values
                )
                self.ax.plot(
                    context["time_display"],
                    channel_outside,
                    color=color,
                    linewidth=1.5,
                    alpha=outside_alpha,
                    label="_nolegend_",
                )
                channel_label = self.channels.get(channel_name, "")
                if channel_label:
                    channel_label = f"{channel_name} ({channel_label})"
                else:
                    channel_label = channel_name
                self.ax.plot(
                    context["time_display"],
                    channel_inside,
                    label=channel_label,
                    color=color,
                    linewidth=2.2 if channel_name == self.y_channel else 1.6,
                    alpha=1.0 if channel_name == self.y_channel else 0.9,
                )

            (fitted_line,) = self.ax.plot(
                context["time_display"],
                series["fitted_display"],
                label="Fitted",
                color=FIT_CURVE_COLOR,
                linewidth=2,
            )
            self._plot_lines["fitted"] = fitted_line

            if show_residuals and self.ax_residual is not None:
                (residuals_line,) = self.ax_residual.plot(
                    context["time_display"],
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

            # Calculate R² scores (skip during fast updates for smoothness)
            if not fast:
                x_fit = context["x_data"][context["fit_slice"]]
                y_fit = context["y_data"][context["fit_slice"]]
                channel_data_fit = self._slice_channel_data(
                    context["channel_data_full"], context["fit_slice"]
                )
                fitted_fit = self.evaluate_model(
                    x_fit,
                    params,
                    channel_data=channel_data_fit,
                )
                fitted_full = self.evaluate_model(
                    context["x_data"],
                    params,
                    channel_data=context["channel_data_full"],
                )
                fit_r2_value = compute_r2(y_fit, fitted_fit)
                full_r2_value = compute_r2(context["y_data"], fitted_full)
                self._last_r2_fit = fit_r2_value
                self._last_r2_full = full_r2_value
            else:
                fit_r2_value = self._last_r2_fit
                full_r2_value = self._last_r2_full

            channel_arrays = list(context.get("plot_channel_displays", {}).values())
            y_min, y_max = self._finite_min_max(*channel_arrays, series["fitted_display"])
            self.ax.set_ylim(y_min, y_max)
            self._apply_unique_legend(self.ax, loc="lower right")
            self.ax.set_xlabel("" if show_residuals else "Time (ms)")
            self.ax.set_ylabel("Voltage (V)")
            self.ax.set_xlim(context["time_data"][0], context["time_data"][-1])
            self.ax.grid(True, alpha=0.3)
            if show_residuals and self.ax_residual is not None:
                r_min, r_max = self._finite_min_max(series["residuals_display"])
                self.ax_residual.set_ylim(r_min, r_max)
                self.ax_residual.set_ylabel("Residual")
                self.ax_residual.set_xlabel("Time (ms)")
                self.ax_residual.grid(True, alpha=0.25)
                self._apply_unique_legend(self.ax_residual, loc="upper right")
            self._sync_fit_region_selector()
            self.ax.text(
                0.01,
                0.98,
                "Drag to set fit window; double-click a boundary to reset",
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#7f1d1d",
            )

            self.canvas.draw_idle()
            self._update_stats_panel(fit_r2_value, full_r2_value)
        except Exception as e:
            self.stats_text.setText(f"Error updating stats: {e}")

    def run_batch_fit(self):
        """Run batch fitting using the shared file list."""
        if self.batch_thread is not None:
            self.stats_text.append("Batch fit is already running.")
            return
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

        base_param_values = [
            fit_context["base_values"][key] for key in fit_context["ordered_keys"]
        ]
        fit_options = self._current_fit_optimization_options()
        self._batch_fit_options = fit_options
        self._stop_thumbnail_render()
        self._batch_progress_done = 0
        self.batch_fit_in_progress = True
        self._batch_cancel_pending = False

        existing_by_file = {row["file"]: row for row in self.batch_results}
        self.batch_results = []
        for source_index, file_path in enumerate(self.batch_files):
            existing = existing_by_file.get(file_path, {})
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
                    preserve_fit_result=False,
                )
            )
        self.update_batch_table()

        self.batch_thread = QThread(self)
        self.batch_worker = BatchFitWorker(
            self.batch_files,
            fit_context["p0"],
            fit_context["bounds"],
            fit_context["ordered_keys"],
            fit_context["active_keys"],
            base_param_values,
            capture_config.regex_pattern,
            capture_config.defaults,
            self._compiled_expression,
            self.x_channel,
            self.y_channel,
            self.fit_region_start_pct,
            self.fit_region_end_pct,
            fit_options,
        )
        self.batch_worker.moveToThread(self.batch_thread)

        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.failed.connect(self.on_batch_failed)
        self.batch_worker.cancelled.connect(self.on_batch_cancelled)

        self.run_batch_btn.setEnabled(False)
        self.cancel_batch_btn.setEnabled(True)
        self.cancel_batch_btn.setText("Cancel")
        total = len(self.batch_files)
        self.run_batch_btn.setText(f"Run Batch (0/{total})")
        requested_starts = (
            max(2, int(fit_options.n_starts)) if fit_options.enabled else 1
        )
        self.stats_text.append(
            "Batch fit started: "
            f"{fit_mode_label(fit_options)}, starts={requested_starts}, seed={fit_options.seed}, "
            f"DE={'on' if fit_options.enabled and fit_options.use_global_init else 'off'}."
        )
        self.batch_thread.start()

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
            if existing.get("plot_full") is not None and row.get("plot_full") is None:
                row["plot_full"] = existing["plot_full"]
            elif existing.get("plot") is not None and row.get("plot") is None:
                row["plot"] = existing["plot"]
            self.batch_results[row_index] = row
            table_row_idx = self._find_table_row_by_file(row["file"])
            if table_row_idx is not None:
                self.update_batch_table_row(table_row_idx, row)

    def on_batch_finished(self, results):
        """Populate table and thumbnails after batch fit finishes."""
        previous_by_file = {row["file"]: row for row in self.batch_results}
        ordered_results = sorted(
            list(results), key=lambda row: int(row.get("_source_index", 0))
        )
        self.batch_results = ordered_results
        for row in self.batch_results:
            existing = previous_by_file.get(row["file"])
            if existing and existing.get("plot_full") is not None:
                row["plot_full"] = existing["plot_full"]
            elif existing and existing.get("plot") is not None:
                row["plot"] = existing["plot"]
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        self.stats_text.append("✓ Batch fit completed.")
        diagnostic_rows = [
            row for row in self.batch_results if row.get("fit_attempts") is not None
        ]
        if diagnostic_rows:
            attempts = np.asarray(
                [float(row.get("fit_attempts", 0)) for row in diagnostic_rows],
                dtype=float,
            )
            requested = max(
                [
                    int(row.get("fit_requested_starts", 0))
                    for row in diagnostic_rows
                    if row.get("fit_requested_starts") is not None
                ]
                or [0]
            )
            mode_text = diagnostic_rows[0].get(
                "fit_mode", fit_mode_label(self._batch_fit_options)
            )
            seed = diagnostic_rows[0].get("fit_seed")
            de_used = any(bool(row.get("fit_de_used")) for row in diagnostic_rows)
            self.stats_text.append(
                "Batch diagnostics: "
                f"{mode_text}, best of ~{int(np.round(np.mean(attempts)))}"
                f"/{requested if requested > 0 else '?'} starts, "
                f"seed={seed if seed is not None else 'n/a'}, "
                f"DE={'on' if de_used else 'off'}."
            )
        self.cleanup_batch_thread()
        self.queue_visible_thumbnail_render()

    def on_batch_failed(self, error_text):
        self.stats_text.append(f"✗ Batch fit failed: {error_text}")
        self.cleanup_batch_thread()

    def on_batch_cancelled(self):
        self.stats_text.append("Batch fit cancelled.")
        self.cleanup_batch_thread()

    def _force_stop_batch_fit(self, reason_text):
        if self.batch_thread is None and self.batch_worker is None:
            return
        self._request_worker_cancel(self.batch_worker)
        self.stats_text.append(str(reason_text))
        self.cleanup_batch_thread(force=True)

    def cancel_batch_fit(self):
        """Request cancellation of an in-flight batch fit."""
        if self.batch_worker is None and self.batch_thread is None:
            return
        if not self._batch_cancel_pending:
            self._batch_cancel_pending = True
            self._request_worker_cancel(self.batch_worker)
            self.cancel_batch_btn.setText("Force Stop")
            self.stats_text.append(
                "Batch cancellation requested... click Cancel again to force stop."
            )
            QTimer.singleShot(
                1500,
                lambda: (
                    self._force_stop_batch_fit(
                        "⚠ Batch fit did not stop promptly; force-stopped."
                    )
                    if self.batch_thread is not None and self._batch_cancel_pending
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
            return

        sorting_enabled = self.batch_table.isSortingEnabled()
        if sorting_enabled:
            self.batch_table.setSortingEnabled(False)
        try:
            columns = (
                ["Plot"]
                + ["File"]
                + self.batch_capture_keys
                + [spec.column_name for spec in self.param_specs]
                + ["R2", "Error"]
            )
            self.batch_table.setColumnCount(len(columns))
            self.batch_table.setHorizontalHeaderLabels(columns)
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
            file_name = display_name_for_file_ref(row["file"])
            file_item = NumericSortTableWidgetItem(file_name)
            file_item.setData(Qt.ItemDataRole.UserRole, row["file"])
            self.batch_table.setItem(row_idx, 1, file_item)

            # Capture columns (start at index 2)
            for col_idx, key in enumerate(self.batch_capture_keys, start=2):
                value = row.get("captures", {}).get(key, "")
                self.batch_table.setItem(
                    row_idx, col_idx, NumericSortTableWidgetItem(str(value))
                )

            # Parameter columns (start at 2 + len(batch_capture_keys))
            param_start = 2 + len(self.batch_capture_keys)
            params = row.get("params")
            for offset in range(len(self.param_specs)):
                if params and offset < len(params):
                    cell_text = f"{params[offset]:.6f}"
                else:
                    cell_text = ""
                self.batch_table.setItem(
                    row_idx,
                    param_start + offset,
                    NumericSortTableWidgetItem(cell_text),
                )
            r2_val = row.get("r2")
            if r2_val is not None:
                self.batch_table.setItem(
                    row_idx,
                    param_start + len(self.param_specs),
                    NumericSortTableWidgetItem(f"{r2_val:.6f}"),
                )
            else:
                self.batch_table.setItem(
                    row_idx,
                    param_start + len(self.param_specs),
                    NumericSortTableWidgetItem(""),
                )
            error_text = self._batch_row_error_text(row)
            self.batch_table.setItem(
                row_idx,
                param_start + len(self.param_specs) + 1,
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

        if file_path not in self.data_files:
            self.data_files.append(file_path)
            self.file_combo.addItem(display_name_for_file_ref(file_path), file_path)
            self._sync_batch_files_from_shared(sync_pattern=False)

        file_idx = self.data_files.index(file_path)
        self.load_file(file_idx)
        self.tabs.setCurrentWidget(self.manual_tab)

    def _expand_file_column_for_selected_files(self):
        """Expand file column width to show the longest selected file name."""
        if not self.batch_files or self.batch_table.columnCount() < 2:
            return

        font_metrics = self.batch_table.fontMetrics()
        longest_width = 0
        for file_path in self.batch_files:
            file_name = display_name_for_file_ref(file_path)
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

    def queue_visible_thumbnail_render(self, *_args):
        if not self.batch_results or self.batch_fit_in_progress:
            return
        row_indices = self._visible_batch_row_indices()
        if not row_indices:
            row_indices = list(range(min(len(self.batch_results), 10)))
        self._start_thumbnail_render(row_indices=row_indices)

    def _start_thumbnail_render(self, row_indices=None):
        """Start background thread to render missing thumbnails."""
        if not self.batch_results:
            return

        if row_indices is None:
            candidate_rows = list(range(len(self.batch_results)))
        else:
            candidate_rows = sorted(
                {
                    int(idx)
                    for idx in row_indices
                    if 0 <= int(idx) < len(self.batch_results)
                }
            )
        candidate_rows = [
            idx
            for idx in candidate_rows
            if self.batch_results[idx].get("plot_full") is None
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
            full_thumbnail_size=self._full_batch_thumbnail_size(),
            row_indices=candidate_rows,
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
            queued = sorted(self._pending_thumbnail_rows)
            self._pending_thumbnail_rows.clear()
            self._start_thumbnail_render(row_indices=queued)

    def _batch_export_default_filename(self):
        pattern_text = (
            self.regex_input.text().strip() if hasattr(self, "regex_input") else ""
        )
        base = pattern_text or "batch_fit_results"
        base = base.replace("*", "any")
        base = re.sub(r"\{([^{}]+)\}", r"\1", base)
        base = re.sub(r"\s+", "_", base)
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        base = re.sub(r"_+", "_", base).strip("._-")
        if not base:
            base = "batch_fit_results"
        if len(base) > 100:
            base = base[:100].rstrip("._-")
        return f"{base}.csv"

    def export_batch_table(self):
        """Export batch table to CSV."""
        if not self.batch_results:
            self.stats_text.append("No batch results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Batch Table",
            str(Path.cwd() / self._batch_export_default_filename()),
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not file_path:
            return

        columns = (
            ["File"]
            + self.batch_capture_keys
            + [spec.column_name for spec in self.param_specs]
            + ["R2", "Error", "FitMode", "FitAttempts", "FitBestSSE"]
        )

        try:
            with open(file_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                for row in self.batch_results:
                    file_name = display_name_for_file_ref(row["file"])
                    captures = row.get("captures", {})
                    params = row.get("params") or [""] * len(self.param_specs)
                    r2_val = row.get("r2")
                    error_text = self._batch_row_error_text(row)
                    fit_mode = row.get("fit_mode") or ""
                    fit_attempts = row.get("fit_attempts")
                    fit_best_sse = row.get("fit_best_sse")
                    writer.writerow(
                        [file_name]
                        + [captures.get(key, "") for key in self.batch_capture_keys]
                        + [
                            f"{val:.6f}" if isinstance(val, float) else val
                            for val in params
                        ]
                        + [f"{r2_val:.6f}" if r2_val is not None else ""]
                        + [error_text]
                        + [fit_mode]
                        + [str(int(fit_attempts)) if fit_attempts is not None else ""]
                        + [
                            f"{float(fit_best_sse):.6g}"
                            if fit_best_sse is not None
                            else ""
                        ]
                    )
            self.stats_text.append(f"✓ Exported batch table to {file_path}")
        except Exception as exc:
            self.stats_text.append(f"✗ Export failed: {exc}")

    def _set_batch_parse_feedback(self, message, is_error=False, tooltip=""):
        self.batch_parse_feedback_label.setText(message)
        self.batch_parse_feedback_label.setToolTip(tooltip)
        if is_error:
            self.batch_parse_feedback_label.setStyleSheet(
                "color: #b91c1c; font-weight: 600; padding: 1px 2px;"
            )
        else:
            self.batch_parse_feedback_label.setStyleSheet("")

    def _resolve_batch_capture_config(self, show_errors):
        pattern_text = self.regex_input.text().strip()
        try:
            return parse_capture_pattern(pattern_text)
        except Exception as exc:
            if show_errors:
                self._set_batch_parse_feedback(f"Error: {exc}", is_error=True)
                self.batch_status_label.setText(f"Error: {exc}")
                self.batch_status_label.show()
            return None

    def _update_batch_capture_feedback(self, config):
        if config.mode == "off":
            self._set_batch_parse_feedback(
                "Add {field} placeholders to extract filename columns."
            )
            return

        field_text = (
            ", ".join(self.batch_capture_keys) if self.batch_capture_keys else "none"
        )
        self._set_batch_parse_feedback(f"Fields: {field_text}")

    def _on_regex_changed(self):
        """Debounce filename pattern changes to avoid excessive updates."""
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
                self.update_batch_table()
                self._refresh_batch_analysis_if_run()
                return
            self.batch_status_label.hide()
            self._update_batch_capture_feedback(config)
            self.update_batch_table()
            self._refresh_batch_analysis_if_run()
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
                    self.batch_unmatched_files.append(
                        display_name_for_file_ref(row["file"])
                    )
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
                    self.batch_unmatched_files.append(
                        display_name_for_file_ref(file_path)
                    )
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
        self.update_batch_table()
        self._refresh_batch_analysis_if_run()
        if any(
            row.get("plot_full") is None and row.get("plot") is None
            for row in self.batch_results
        ):
            self.queue_visible_thumbnail_render()

    def _refresh_batch_analysis_if_run(self):
        if not hasattr(self, "analysis_source_combo"):
            return
        if self.analysis_source_combo.currentData() == "run":
            self._refresh_batch_analysis_data(preserve_selection=True)

    def closeEvent(self, event):
        """Ensure worker thread is stopped before closing."""
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._request_worker_cancel(self.fit_worker)
        self._request_worker_cancel(self.batch_worker)
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
