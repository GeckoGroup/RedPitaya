"""Data loading/parsing utilities for fit_gui."""

import re
import zipfile
from io import BytesIO
from pathlib import Path

from pandas import read_csv


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
