"""Batch processing workers and utilities for fit_gui."""

import re
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
)

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import (
    QMutex,
    QObject,
    QThread,
    QWaitCondition,
    pyqtSignal,
    pyqtSlot,
    Qt,
)
from PyQt6.QtGui import QPixmap

from expression import _PARAMETER_NAME_RE
from model import (
    FitCancelledError,
    boundary_ratios_to_x_values,
    has_nonempty_values,
    smooth_channel_array,
    finite_float_or_none,
    _row_has_error,
    FIT_CURVE_COLOR,
    palette_color,
)
from procedure import (
    run_procedure_pipeline,
    _capture_seed_signature,
    _capture_distance,
)
import fit_log as _fit_log
from data_io import read_measurement_csv, stem_for_file_ref
from fit_results import fit_get, fit_set


@dataclass(frozen=True)
class CapturePatternConfig:
    mode: str
    regex_pattern: str
    regex: Optional[Pattern[str]]
    defaults: Dict[str, str]


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

            if not _PARAMETER_NAME_RE.fullmatch(field_name):
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
    row = {
        "_source_index": int(source_index),
        "file": file_path,
        "captures": dict(captures or {}),
        "x_channel": x_channel,
        "y_channel": y_channel,
        "plot_full": plot_full,
        "plot": plot,
        "plot_has_fit": plot_has_fit,
        "plot_render_size": plot_render_size,
        "pattern_error": pattern_error,
        "_equation_stale": bool(equation_stale),
        "_fit_status": (str(fit_status) if fit_status not in (None, "") else None),
        "_queue_position": (
            int(queue_position) if queue_position not in (None, "") else None
        ),
        "_r2_old": r2_old,
        "_fit_task_id": (int(fit_task_id) if fit_task_id not in (None, "") else None),
    }
    fit_set(row, "params", params)
    fit_set(row, "r2", r2)
    fit_set(row, "error", error)
    fit_set(row, "boundary_ratios", boundary_ratios)
    fit_set(row, "boundary_values", boundary_values)
    fit_set(row, "channel_results", None)
    return row


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
        x_col = row.get("x_channel") or data.columns[0]
        y_col = row.get("y_channel") or data.columns[1]

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

        params = fit_get(row, "params")
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
                boundary_ratios=fit_get(row, "boundary_ratios"),
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


# ---------------------------------------------------------------------------
# Cross-file sibling seeding utilities
# ---------------------------------------------------------------------------


def _row_to_sibling_result(
    row: Mapping[str, Any],
    ordered_param_keys: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Convert a canonical fit row to the normalised sibling-result dict.

    Returns ``None`` if the row has no usable fit result.  The returned dict
    has keys ``captures``, ``params_by_key``, ``boundary_ratios_by_channel``,
    ``r2`` — the format expected by ``ProcedureContext.sibling_results``.
    """
    if _row_has_error(row):
        return None
    params_raw = fit_get(row, "params")
    if not has_nonempty_values(params_raw):
        return None
    try:
        params_arr = np.asarray(params_raw, dtype=float).reshape(-1)
    except Exception:
        return None

    params_by_key: Dict[str, float] = {}
    for idx, key in enumerate(ordered_param_keys):
        if idx >= params_arr.size:
            break
        val = float(params_arr[idx])
        if np.isfinite(val):
            params_by_key[str(key)] = val

    r2: Optional[float] = None
    r2_raw = fit_get(row, "r2")
    if r2_raw is not None:
        try:
            r2 = float(r2_raw)
            if not np.isfinite(r2):
                r2 = None
        except Exception:
            pass

    captures = dict(row.get("captures") or {})

    boundary_ratios_by_channel: Dict[str, np.ndarray] = {}
    ch_results = fit_get(row, "channel_results")
    if isinstance(ch_results, dict):
        for ch, cr in ch_results.items():
            if isinstance(cr, dict):
                br = cr.get("boundary_ratios")
                if br is not None:
                    try:
                        boundary_ratios_by_channel[str(ch)] = np.asarray(
                            br, dtype=float
                        ).reshape(-1)
                    except Exception:
                        pass
    if not boundary_ratios_by_channel:
        br = fit_get(row, "boundary_ratios")
        if br is not None:
            y_ch = str(row.get("y_channel") or "").strip()
            if y_ch:
                try:
                    boundary_ratios_by_channel[y_ch] = np.asarray(
                        br, dtype=float
                    ).reshape(-1)
                except Exception:
                    pass

    return {
        "captures": captures,
        "params_by_key": params_by_key,
        "boundary_ratios_by_channel": boundary_ratios_by_channel,
        "r2": r2,
    }


def _result_to_sibling(
    captures: Mapping[str, Any],
    result: Mapping[str, Any],
    multi_channel_model: Any,
) -> Dict[str, Any]:
    """Convert a procedure pipeline result to the normalised sibling-result dict."""
    params_by_key = dict(result.get("params_by_key") or {})
    r2 = finite_float_or_none(result.get("r2"))

    boundary_ratios_by_channel: Dict[str, np.ndarray] = {}
    ch_results = result.get("channel_results") or {}
    if isinstance(ch_results, Mapping):
        for ch, cr in ch_results.items():
            if not isinstance(cr, Mapping):
                continue
            br = cr.get("boundary_ratios")
            if br is not None:
                try:
                    boundary_ratios_by_channel[str(ch)] = np.asarray(
                        br, dtype=float
                    ).reshape(-1)
                except Exception:
                    pass

    return {
        "captures": dict(captures or {}),
        "params_by_key": params_by_key,
        "boundary_ratios_by_channel": boundary_ratios_by_channel,
        "r2": float(r2) if r2 is not None else None,
    }


# ---------------------------------------------------------------------------
# Single persistent worker thread for all fit tasks
# ---------------------------------------------------------------------------


class FitWorkerThread(QThread):
    """Persistent single worker thread that processes fit jobs sequentially.

    Jobs are submitted via ``submit()`` / ``preempt()``
    and results are delivered through Qt signals back to the GUI thread.
    """

    task_progress = pyqtSignal(int, int, int, object)  # task_id, done, total, row
    task_step_completed = pyqtSignal(int, int, object)  # task_id, step_idx, result
    task_attempt_completed = pyqtSignal(
        int, int, int, object
    )  # task_id, step_idx, attempt, info
    task_finished = pyqtSignal(int, list)  # task_id, results
    task_failed = pyqtSignal(int, str)  # task_id, error_text
    task_cancelled = pyqtSignal(int)  # task_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._queue: deque[Tuple[int, dict]] = deque()
        self._cancel_ids: set[int] = set()
        self._current_task_id: Optional[int] = None
        self._current_worker: Any = None
        self._shutdown = False

    # -- Public API (called from GUI thread) --------------------------------

    def submit(self, task_id: int, descriptor: dict) -> None:
        """Append a job to the back of the queue."""
        self._mutex.lock()
        self._queue.append((int(task_id), dict(descriptor)))
        self._mutex.unlock()
        self._condition.wakeOne()

    def preempt(self, task_id: int, descriptor: dict) -> None:
        """Cancel current task, push new job to front, re-queue interrupted task."""
        self._mutex.lock()
        if self._current_task_id is not None:
            self._cancel_ids.add(self._current_task_id)
            worker = self._current_worker
            if worker is not None:
                try:
                    worker.cancel_requested = True
                except Exception:
                    pass
        self._queue.appendleft((int(task_id), dict(descriptor)))
        self._mutex.unlock()
        self._condition.wakeOne()

    def cancel_tasks(self, task_ids: set[int]) -> None:
        """Cancel pending and/or running tasks by id."""
        self._mutex.lock()
        self._cancel_ids.update(int(tid) for tid in task_ids)
        if (
            self._current_task_id is not None
            and self._current_task_id in self._cancel_ids
        ):
            worker = self._current_worker
            if worker is not None:
                try:
                    worker.cancel_requested = True
                except Exception:
                    pass
        self._mutex.unlock()

    def cancel_all(self) -> None:
        """Cancel everything — current task and all pending."""
        self._mutex.lock()
        for tid, _ in self._queue:
            self._cancel_ids.add(tid)
        if self._current_task_id is not None:
            self._cancel_ids.add(self._current_task_id)
            worker = self._current_worker
            if worker is not None:
                try:
                    worker.cancel_requested = True
                except Exception:
                    pass
        self._mutex.unlock()

    def shutdown(self) -> None:
        """Signal the thread to exit after current job completes."""
        self._mutex.lock()
        self._shutdown = True
        self._mutex.unlock()
        self._condition.wakeOne()

    # -- Thread loop --------------------------------------------------------

    def run(self) -> None:
        while True:
            self._mutex.lock()
            while not self._shutdown and len(self._queue) == 0:
                self._condition.wait(self._mutex)
            if self._shutdown and len(self._queue) == 0:
                self._mutex.unlock()
                return
            task_id, descriptor = self._queue.popleft()
            if task_id in self._cancel_ids:
                self._cancel_ids.discard(task_id)
                self._mutex.unlock()
                self.task_cancelled.emit(task_id)
                continue
            self._current_task_id = task_id
            self._current_worker = None
            self._mutex.unlock()

            try:
                self._execute_job(task_id, descriptor)
            except Exception as exc:
                self.task_failed.emit(task_id, str(exc))
            finally:
                self._mutex.lock()
                self._current_task_id = None
                self._current_worker = None
                self._cancel_ids.discard(task_id)
                self._mutex.unlock()

    def _execute_job(self, task_id: int, descriptor: dict) -> None:
        """Build the appropriate worker and run it synchronously."""
        kind = str(descriptor.get("worker_kind", "procedure_batch"))

        if kind == "procedure_single":
            worker = ProcedureFitWorker(**descriptor["worker_args"])
        else:
            worker = BatchProcedureFitWorker(**descriptor["worker_args"])

        self._mutex.lock()
        self._current_worker = worker
        if task_id in self._cancel_ids:
            worker.cancel_requested = True
        self._mutex.unlock()

        results_box: List[Any] = []
        error_box: List[str] = []
        cancelled_box: List[bool] = []

        def _on_progress(*args):
            self.task_progress.emit(task_id, *args)

        if isinstance(worker, ProcedureFitWorker):
            worker.step_completed.connect(
                lambda step_idx, step_result: self.task_progress.emit(
                    task_id, step_idx, 0, step_result
                )
            )
        else:
            worker.progress.connect(_on_progress)

        # Forward step/attempt signals for both worker types.
        worker.step_completed.connect(
            lambda step_idx, step_result: self.task_step_completed.emit(
                task_id, step_idx, step_result
            )
        )
        worker.attempt_completed.connect(
            lambda step_idx, attempt, info: self.task_attempt_completed.emit(
                task_id, step_idx, attempt, info
            )
        )

        worker.finished.connect(results_box.append)
        worker.failed.connect(error_box.append)
        worker.cancelled.connect(lambda: cancelled_box.append(True))

        worker.run()

        if cancelled_box:
            self.task_cancelled.emit(task_id)
        elif error_box:
            self.task_failed.emit(task_id, error_box[0])
        elif results_box:
            result = results_box[0]
            if isinstance(result, list):
                self.task_finished.emit(task_id, result)
            elif isinstance(result, dict):
                self.task_finished.emit(task_id, [result])
            else:
                self.task_finished.emit(task_id, [])
        else:
            self.task_finished.emit(task_id, [])


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
                row["plot_has_fit"] = has_nonempty_values(fit_get(row, "params"))
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


# ── New polymorphic procedure workers ────────────────────────────────


class ProcedureFitWorker(QObject):
    """Run a polymorphic multi-step procedure on a single file.

    Uses ``run_procedure_pipeline`` which supports all step types
    (fit, set_parameter, set_boundaries, randomize_seeds)
    plus retry logic.
    """

    step_completed = pyqtSignal(int, object)
    attempt_completed = pyqtSignal(int, int, object)  # (step_idx, attempt, info)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        x_data,
        y_data_by_channel,
        multi_model,
        procedure,
        seed_map,
        bounds_map,
        boundary_seeds=None,
        bound_values=None,
        boundary_name_groups=None,
        use_jax=False,
    ):
        super().__init__()
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data_by_channel = {
            str(k): np.asarray(v, dtype=float)
            for k, v in dict(y_data_by_channel).items()
        }
        self.multi_model = multi_model
        self.procedure = procedure
        self.seed_map = dict(seed_map)
        self.bounds_map = dict(bounds_map)
        self.boundary_seeds = dict(boundary_seeds or {})
        self.bound_values = dict(bound_values or {})
        self.boundary_name_groups = dict(boundary_name_groups or {})
        self.cancel_requested = False
        self.use_jax = bool(use_jax)

    def request_cancel(self):
        self.cancel_requested = True

    @pyqtSlot()
    def run(self):
        try:
            if self.cancel_requested:
                self.cancelled.emit()
                return

            def _step_cb(step_idx, step_result):
                self.step_completed.emit(step_idx, step_result)

            def _attempt_cb(step_idx, attempt, info):
                self.attempt_completed.emit(step_idx, attempt, info)

            result = run_procedure_pipeline(
                self.x_data,
                self.y_data_by_channel,
                self.multi_model,
                self.procedure,
                self.seed_map,
                self.bounds_map,
                boundary_seeds=self.boundary_seeds,
                cancel_check=lambda: self.cancel_requested,
                step_callback=_step_cb,
                attempt_callback=_attempt_cb,
                bound_values=self.bound_values,
                boundary_name_groups=self.boundary_name_groups,
                use_jax=self.use_jax,
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


class BatchProcedureFitWorker(QObject):
    """Run a procedure across multiple batch files.

    Iterates over files, loads data, resolves captures, and runs the
    procedure pipeline on each.  Emits the same ``progress`` signal
    interface as ``BatchFitWorker`` so the GUI can reuse the same
    result-handling logic.
    """

    progress = pyqtSignal(int, int, object)  # (completed, total, row_result)
    step_completed = pyqtSignal(int, object)  # (step_index, step_result_dict)
    attempt_completed = pyqtSignal(int, int, object)  # (step_index, attempt, info)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        file_paths,
        source_indices,
        regex_pattern,
        capture_defaults,
        parameter_capture_map,
        multi_channel_model,
        ordered_param_keys,
        seed_map,
        bounds_map,
        boundary_seeds_per_channel,
        x_channel,
        procedure,
        smoothing_enabled=False,
        smoothing_window=1,
        boundary_name_groups=None,
        use_jax=False,
        # -- Cross-file sibling seeding --
        existing_rows_by_file=None,
        use_existing_fit_seed=True,
    ):
        super().__init__()
        self.file_paths = list(file_paths)
        self.source_indices = [
            int(i) for i in (source_indices or range(len(self.file_paths)))
        ]
        if len(self.source_indices) != len(self.file_paths):
            self.source_indices = list(range(len(self.file_paths)))
        self.regex = re.compile(regex_pattern) if regex_pattern else None
        self.capture_defaults = dict(capture_defaults or {})
        self.parameter_capture_map = {
            str(k): (str(v) if v not in (None, "") else None)
            for k, v in dict(parameter_capture_map or {}).items()
        }
        self.multi_channel_model = multi_channel_model
        self.ordered_param_keys = list(ordered_param_keys or ())
        self.seed_map = dict(seed_map or {})
        self.bounds_map = dict(bounds_map or {})
        self.boundary_seeds_per_channel = dict(boundary_seeds_per_channel or {})
        self.x_channel = str(x_channel)
        self.procedure = procedure
        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoothing_window = int(smoothing_window)
        self.boundary_name_groups = dict(boundary_name_groups or {})
        self.boundary_name_groups = dict(boundary_name_groups or {})
        self.cancel_requested = False
        self.use_jax = bool(use_jax)
        self.use_existing_fit_seed = bool(use_existing_fit_seed)

        # Derive capture_seed_keys from parameter_capture_map (same logic as
        # BatchFitWorker) so sibling matching uses the same dimension keys.
        self._capture_seed_keys: Tuple[str, ...] = tuple(
            sorted(
                {
                    str(field)
                    for field in self.parameter_capture_map.values()
                    if field not in (None, "")
                }
            )
        )

        # Build the normalised sibling-results dict from any pre-existing fit
        # rows the GUI already knows about.
        self._sibling_results: Dict[str, Dict[str, Any]] = {}
        _sibling_skipped = 0
        if existing_rows_by_file:
            for file_key, row in dict(existing_rows_by_file).items():
                sibling = _row_to_sibling_result(row, self.ordered_param_keys)
                if sibling is not None:
                    self._sibling_results[str(file_key)] = sibling
                else:
                    _sibling_skipped += 1
        _fit_log.detail(
            "BatchProcedureFitWorker init: "
            f"files={len(self.file_paths)} "
            f"existing_rows={len(existing_rows_by_file or {})} "
            f"siblings_built={len(self._sibling_results)} "
            f"siblings_skipped={_sibling_skipped} "
            f"capture_seed_keys={self._capture_seed_keys} "
            f"param_capture_map={dict(self.parameter_capture_map)} "
            f"seed_from_siblings={getattr(self.procedure, 'seed_from_siblings', '?')}"
        )

    def request_cancel(self):
        self.cancel_requested = True

    def _load_file_data(self, file_path):
        """Load and smooth data for a single file."""
        df = read_measurement_csv(file_path)
        if df is None or df.empty:
            return None, None
        x_data = df[self.x_channel].to_numpy(dtype=float, copy=True)
        if self.smoothing_enabled:
            x_data = smooth_channel_array(x_data, self.smoothing_window)
        y_data_by_channel = {}
        for ch_model in self.multi_channel_model.channel_models:
            col = ch_model.target_col
            if col in df.columns:
                y = df[col].to_numpy(dtype=float, copy=True)
                if self.smoothing_enabled:
                    y = smooth_channel_array(y, self.smoothing_window)
                y_data_by_channel[col] = y
        return x_data, y_data_by_channel

    @pyqtSlot()
    def run(self):
        results = []
        try:
            total = len(self.file_paths)
            if total == 0:
                self.finished.emit([])
                return

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
                self.progress.emit(idx + 1, total, row)

            if self.cancel_requested:
                self.cancelled.emit()
                return
            self.finished.emit(results)
        except Exception as exc:
            if self.cancel_requested:
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))

    def _fit_single_file(self, source_index, file_path):
        """Load data, resolve captures, run procedure, return result row."""
        # Extract captures.
        stem = stem_for_file_ref(file_path)
        captures = extract_captures(stem, self.regex, self.capture_defaults)
        if captures is None:
            return make_batch_result_row(
                source_index,
                file_path,
                self.x_channel,
                "",
                pattern_error=_BATCH_PATTERN_MISMATCH_ERROR,
            )

        # Resolve bound params from captures.
        bound_values, error = resolve_fixed_params_from_captures(
            self.parameter_capture_map,
            captures,
        )
        if error:
            return make_batch_result_row(
                source_index,
                file_path,
                self.x_channel,
                "",
                captures=captures,
                error=error,
            )

        # Load data.
        try:
            x_data, y_data_by_channel = self._load_file_data(file_path)
        except Exception as exc:
            return make_batch_result_row(
                source_index,
                file_path,
                self.x_channel,
                "",
                captures=captures,
                error=str(exc),
            )
        if x_data is None or not y_data_by_channel:
            return make_batch_result_row(
                source_index,
                file_path,
                self.x_channel,
                "",
                captures=captures,
                error="Failed to load data.",
            )

        # Determine primary target channel.
        primary_target = self.multi_channel_model.primary.target_col

        # --- Cross-file seeding: override seed_map from best sibling ---
        file_seed_map = dict(self.seed_map)
        file_boundary_seeds = dict(self.boundary_seeds_per_channel)
        file_key = str(file_path)

        if self.use_existing_fit_seed and self._sibling_results:
            # Find best sibling by capture proximity.
            if not self._capture_seed_keys:
                _fit_log.detail(
                    "pre-procedure sibling search skipped: "
                    "no capture_seed_keys (need bound_params in procedure steps)"
                )
            else:
                _fit_log.detail(
                    "pre-procedure sibling search: "
                    f"file={stem_for_file_ref(file_path)} "
                    f"captures={dict(captures)} "
                    f"seed_keys={self._capture_seed_keys} "
                    f"n_siblings={len(self._sibling_results)}"
                )
                best_sibling: Optional[Dict[str, Any]] = None
                closest_sibling: Optional[Dict[str, Any]] = None
                closest_distance: Optional[float] = None
                current_sig = _capture_seed_signature(captures, self._capture_seed_keys)
                _fit_log.detail(f"current signature: {current_sig}")
                for sib_key, sib in self._sibling_results.items():
                    if sib_key == file_key:
                        continue
                    sib_caps = sib.get("captures")
                    if not isinstance(sib_caps, dict):
                        continue
                    if not sib.get("params_by_key"):
                        continue
                    sib_sig = _capture_seed_signature(sib_caps, self._capture_seed_keys)
                    if (
                        current_sig is not None
                        and sib_sig is not None
                        and current_sig == sib_sig
                    ):
                        if best_sibling is None or (sib.get("r2") or 0) > (
                            best_sibling.get("r2") or 0
                        ):
                            best_sibling = sib
                        continue
                    dist = _capture_distance(
                        captures, sib_caps, self._capture_seed_keys
                    )
                    if dist is None:
                        continue
                    if closest_sibling is None or float(dist) < float(closest_distance):
                        closest_sibling = sib
                        closest_distance = float(dist)
                    elif np.isclose(float(dist), float(closest_distance)) and (
                        (sib.get("r2") or 0) > (closest_sibling.get("r2") or 0)
                    ):
                        closest_sibling = sib
                        closest_distance = float(dist)
                chosen = best_sibling or closest_sibling
                if chosen is not None:
                    applied_params = 0
                    for key, val in (chosen.get("params_by_key") or {}).items():
                        if key in file_seed_map:
                            try:
                                v = float(val)
                                if np.isfinite(v):
                                    file_seed_map[str(key)] = v
                                    applied_params += 1
                            except (TypeError, ValueError):
                                pass
                    applied_boundaries = 0
                    for ch, ratios in (
                        chosen.get("boundary_ratios_by_channel") or {}
                    ).items():
                        if ch in file_boundary_seeds and ratios is not None:
                            try:
                                file_boundary_seeds[str(ch)] = np.asarray(
                                    ratios, dtype=float
                                ).reshape(-1)
                                applied_boundaries += 1
                            except Exception:
                                pass
                    sib_kind = (
                        "matching-captures"
                        if best_sibling is not None
                        else "closest-captures"
                    )
                    _fit_log.detail(
                        "pre-procedure sibling seed applied: "
                        f"kind={sib_kind} "
                        f"r2={chosen.get('r2')} "
                        f"params={applied_params} boundaries={applied_boundaries}"
                    )
                else:
                    _fit_log.detail(
                        "pre-procedure sibling seed: "
                        f"{len(self._sibling_results)} sibling(s) available "
                        "but no match found"
                    )
        elif self.use_existing_fit_seed:
            _fit_log.detail(
                "pre-procedure sibling search: no existing sibling results "
                "(first batch run or all prior rows lack fit params)"
            )

        # Run the procedure pipeline.
        try:

            def _step_cb(step_idx, step_result):
                self.step_completed.emit(step_idx, step_result)

            def _attempt_cb(step_idx, attempt, info):
                self.attempt_completed.emit(step_idx, attempt, info)

            result = run_procedure_pipeline(
                x_data,
                y_data_by_channel,
                self.multi_channel_model,
                self.procedure,
                file_seed_map,
                dict(self.bounds_map),
                boundary_seeds=file_boundary_seeds,
                cancel_check=lambda: self.cancel_requested,
                step_callback=_step_cb,
                attempt_callback=_attempt_cb,
                bound_values=dict(bound_values or {}),
                boundary_name_groups=self.boundary_name_groups,
                use_jax=self.use_jax,
                # Cross-file sibling seeding context for procedure retries.
                captures=captures,
                sibling_results=dict(self._sibling_results),
                capture_seed_keys=self._capture_seed_keys,
            )
        except FitCancelledError:
            return None
        except Exception as exc:
            return make_batch_result_row(
                source_index,
                file_path,
                self.x_channel,
                primary_target,
                captures=captures,
                error=str(exc),
            )

        # Extract final param values as ordered array.
        params_by_key = result.get("params_by_key") or {}
        param_array = np.array(
            [float(params_by_key.get(k, 0.0)) for k in self.ordered_param_keys],
            dtype=float,
        )

        r2 = result.get("r2")
        boundary_ratios = None
        boundary_values = None
        ch_results_raw = result.get("channel_results") or {}
        ch_results: Dict[str, Dict[str, Any]] = {}
        if isinstance(ch_results_raw, Mapping):
            for raw_target, raw_entry in ch_results_raw.items():
                if not isinstance(raw_entry, Mapping):
                    continue
                entry: Dict[str, Any] = {}
                raw_ratios = raw_entry.get("boundary_ratios")
                if raw_ratios is not None:
                    try:
                        ratios_arr = np.asarray(raw_ratios, dtype=float).reshape(-1)
                    except Exception:
                        ratios_arr = np.asarray([], dtype=float)
                    entry["boundary_ratios"] = np.asarray(
                        ratios_arr, dtype=float
                    ).copy()
                raw_r2 = finite_float_or_none(raw_entry.get("r2"))
                if raw_r2 is not None:
                    entry["r2"] = float(raw_r2)
                if entry:
                    ch_results[str(raw_target)] = entry
        if primary_target in ch_results:
            ch_r = ch_results[primary_target]
            br = ch_r.get("boundary_ratios")
            if br is not None:
                boundary_ratios = np.asarray(br, dtype=float).reshape(-1)
                n_boundaries = max(
                    0,
                    len(
                        getattr(
                            getattr(self.multi_channel_model, "primary", None),
                            "segment_exprs",
                            (),
                        )
                        or ()
                    )
                    - 1,
                )
                if n_boundaries <= 0:
                    n_boundaries = int(boundary_ratios.size)
                try:
                    boundary_values = boundary_ratios_to_x_values(
                        boundary_ratios,
                        x_data,
                        n_boundaries,
                    )
                except Exception:
                    pass

        row = make_batch_result_row(
            source_index,
            file_path,
            self.x_channel,
            primary_target,
            captures=captures,
            params=param_array,
            r2=r2,
            boundary_ratios=boundary_ratios,
            boundary_values=boundary_values,
        )
        if ch_results:
            fit_set(row, "channel_results", ch_results)
        row["_procedure_result"] = {
            "step_results": list(result.get("step_results") or []),
            "r2": finite_float_or_none(result.get("r2")),
            "stopped_at_step": result.get("stopped_at_step"),
        }

        # --- Write-back: make this result available as a sibling for subsequent
        # files in the same batch run. ---
        if not _row_has_error(row):
            sibling = _result_to_sibling(captures, result, self.multi_channel_model)
            self._sibling_results[file_key] = sibling

        return row
