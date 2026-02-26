"""Batch processing workers and utilities for fit_gui."""

import re
from dataclasses import dataclass
from io import BytesIO
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Pattern,
    Set,
    Tuple,
)

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QPixmap

from expression import _PARAMETER_NAME_RE
from model import (
    FitCancelledError,
    boundary_ratios_to_x_values,
    has_nonempty_values,
    is_fit_row_improved,
    run_piecewise_fit_pipeline,
    smooth_channel_array,
    finite_float_or_none,
    _row_has_error,
    FIT_CURVE_COLOR,
    palette_color,
)
from data_io import read_measurement_csv, stem_for_file_ref


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
        left_num = finite_float_or_none(left)
        right_num = finite_float_or_none(right)
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
