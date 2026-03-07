"""Data loading/parsing utilities for fit_gui."""

import re
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

from pandas import read_csv

SUPPORTED_ARCHIVE_EXTENSIONS = (".zip", ".tar.xz")


def _archive_suffix(path_text: str):
    lower = str(path_text).strip().lower()
    for suffix in SUPPORTED_ARCHIVE_EXTENSIONS:
        if lower.endswith(suffix):
            return suffix
    return None


def is_supported_archive_path(path) -> bool:
    return _archive_suffix(str(path)) is not None


def split_archive_file_ref(file_ref: str):
    if "::" not in file_ref:
        return None
    archive_path, member = file_ref.split("::", 1)
    if not is_supported_archive_path(archive_path):
        return None
    member_name = str(member).strip()
    if not member_name:
        return None
    return archive_path, member_name


def list_archive_csv_members(archive_path) -> list[str]:
    archive_text = str(archive_path)
    archive_suffix = _archive_suffix(archive_text)
    if archive_suffix == ".zip":
        with zipfile.ZipFile(archive_text) as zf:
            return sorted(
                member
                for member in zf.namelist()
                if member.lower().endswith(".csv") and not member.endswith("/")
            )
    if archive_suffix == ".tar.xz":
        with tarfile.open(archive_text, mode="r:xz") as tf:
            return sorted(
                member.name
                for member in tf.getmembers()
                if member.isfile() and member.name.lower().endswith(".csv")
            )
    raise ValueError(f"Unsupported archive format: {archive_text}")


def read_archive_member_bytes(archive_path: str, member: str) -> bytes:
    archive_suffix = _archive_suffix(archive_path)
    if archive_suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open(member) as handle:
                return handle.read()
    if archive_suffix == ".tar.xz":
        with tarfile.open(archive_path, mode="r:xz") as tf:
            extracted = tf.extractfile(member)
            if extracted is None:
                raise KeyError(f"Archive member is not a regular file: {member}")
            with extracted:
                return extracted.read()
    raise ValueError(f"Unsupported archive format: {archive_path}")


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
    """Read CSV data from plain files or archive members and normalize names."""

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

    archive_ref = split_archive_file_ref(file_ref)
    if archive_ref is not None:
        archive_path, member = archive_ref
        raw = read_archive_member_bytes(archive_path, member)
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
