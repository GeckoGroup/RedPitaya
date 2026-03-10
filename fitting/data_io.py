"""Data loading/parsing utilities for fit_gui."""

import re
import tarfile
import threading
import time
import zipfile
from collections import OrderedDict
import hashlib
import shutil
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from pandas import read_csv

SUPPORTED_ARCHIVE_EXTENSIONS = (".zip", ".tar.xz")
_MEMBER_CACHE_MAX_ENTRIES = 128
_MEMBER_CACHE_MAX_BYTES = 256 * 1024 * 1024  # 256 MiB
_TAR_PREFETCH_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB
_TAR_PREFETCH_ARCHIVES_MAX = 2
_TAR_EXTRACT_ARCHIVES_MAX = 2
_MEMBER_LIST_CACHE_MAX = 16

_cache_lock = threading.RLock()
_archive_member_cache: OrderedDict[tuple, bytes] = OrderedDict()
_archive_member_cache_bytes: int = 0
_archive_member_list_cache: OrderedDict[tuple, tuple[str, ...]] = OrderedDict()
_tar_prefetch_cache: OrderedDict[tuple, Optional[Dict[str, bytes]]] = OrderedDict()
_tar_extracted_cache: OrderedDict[tuple, Optional[Tuple[Dict[str, str], str]]] = (
    OrderedDict()
)
_CACHE_MISS = object()
ProgressCallback = Optional[Callable[[str, Optional[float]], None]]


def _emit_progress(
    progress_cb: ProgressCallback,
    message: str,
    fraction: Optional[float] = None,
) -> None:
    if progress_cb is None:
        return
    normalized: Optional[float] = None
    if fraction is not None:
        try:
            normalized = float(fraction)
        except Exception:
            normalized = None
        if normalized is not None:
            normalized = min(1.0, max(0.0, normalized))
    try:
        progress_cb(str(message or ""), normalized)
    except Exception:
        pass


def _normalize_archive_path(path_text: str) -> str:
    path_obj = Path(str(path_text)).expanduser()
    try:
        path_obj = path_obj.resolve(strict=False)
    except Exception:
        pass
    return str(path_obj)


def _archive_signature(path_text: str):
    normalized = _normalize_archive_path(path_text)
    try:
        stat_result = Path(normalized).stat()
        return normalized, int(stat_result.st_mtime_ns), int(stat_result.st_size)
    except Exception:
        return normalized, None, None


def _cache_get_member(cache_key: tuple) -> Optional[bytes]:
    with _cache_lock:
        payload: Optional[bytes] = _archive_member_cache.pop(cache_key, None)
        if payload is None:
            return None
        _archive_member_cache[cache_key] = payload
        return payload


def _cache_put_member(cache_key: tuple, payload: bytes) -> None:
    global _archive_member_cache_bytes
    with _cache_lock:
        existing = _archive_member_cache.pop(cache_key, None)
        if existing is not None:
            _archive_member_cache_bytes -= len(existing)
        _archive_member_cache[cache_key] = payload
        _archive_member_cache_bytes += len(payload)
        while (
            len(_archive_member_cache) > _MEMBER_CACHE_MAX_ENTRIES
            or _archive_member_cache_bytes > _MEMBER_CACHE_MAX_BYTES
        ):
            _evicted_key, evicted_payload = _archive_member_cache.popitem(last=False)
            _archive_member_cache_bytes -= len(evicted_payload)


def _cache_get_member_list(cache_key: tuple):
    with _cache_lock:
        cached = _archive_member_list_cache.pop(cache_key, None)
        if cached is None:
            return None
        _archive_member_list_cache[cache_key] = cached
        return list(cached)


def _cache_put_member_list(cache_key: tuple, members: list[str]) -> None:
    with _cache_lock:
        _archive_member_list_cache.pop(cache_key, None)
        _archive_member_list_cache[cache_key] = tuple(members)
        while len(_archive_member_list_cache) > _MEMBER_LIST_CACHE_MAX:
            _archive_member_list_cache.popitem(last=False)


def _cache_get_prefetched_tar_members(cache_key: tuple):
    with _cache_lock:
        cached = _tar_prefetch_cache.pop(cache_key, _CACHE_MISS)
        if cached is _CACHE_MISS:
            return False, None
        _tar_prefetch_cache[cache_key] = cached
        return True, cached


def _cache_put_prefetched_tar_members(
    cache_key: tuple, members_by_name: Optional[Dict[str, bytes]]
) -> None:
    with _cache_lock:
        _tar_prefetch_cache.pop(cache_key, None)
        _tar_prefetch_cache[cache_key] = members_by_name
        while len(_tar_prefetch_cache) > _TAR_PREFETCH_ARCHIVES_MAX:
            _tar_prefetch_cache.popitem(last=False)


def _cache_get_extracted_tar_members(cache_key: tuple):
    with _cache_lock:
        cached = _tar_extracted_cache.pop(cache_key, _CACHE_MISS)
        if cached is _CACHE_MISS:
            return False, None
        _tar_extracted_cache[cache_key] = cached
        if cached is None:
            return True, None
        member_map, _cache_dir = cached
        return True, member_map


def _cache_put_extracted_tar_members(
    cache_key: tuple, cached_payload: Optional[Tuple[Dict[str, str], str]]
) -> None:
    with _cache_lock:
        existing_payload = _tar_extracted_cache.pop(cache_key, None)
        if existing_payload is not None:
            _existing_members, existing_cache_dir = existing_payload
            if (
                cached_payload is None
                or existing_cache_dir
                != (cached_payload[1] if cached_payload is not None else None)
            ):
                shutil.rmtree(existing_cache_dir, ignore_errors=True)
        _tar_extracted_cache[cache_key] = cached_payload
        while len(_tar_extracted_cache) > _TAR_EXTRACT_ARCHIVES_MAX:
            _evicted_key, evicted_payload = _tar_extracted_cache.popitem(last=False)
            if evicted_payload is None:
                continue
            _evicted_members, evicted_cache_dir = evicted_payload
            shutil.rmtree(evicted_cache_dir, ignore_errors=True)


def _tar_extract_cache_dir(archive_sig: tuple) -> Path:
    token = hashlib.sha1(repr(archive_sig).encode("utf-8")).hexdigest()
    return Path("/tmp") / "redpitaya_tar_cache" / token


def _maybe_extract_tar_xz_members_to_disk(
    archive_path: str,
    archive_sig: tuple,
    *,
    progress_cb: ProgressCallback = None,
) -> Optional[Dict[str, str]]:
    found, cached = _cache_get_extracted_tar_members(archive_sig)
    if found:
        return cached

    extracted_members: Optional[Dict[str, str]] = None
    cache_dir = _tar_extract_cache_dir(archive_sig)
    _emit_progress(progress_cb, "Decompressing archive to local cache...", 0.0)
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        members_map: Dict[str, str] = {}
        bytes_done: int = 0
        total_bytes: int = 0
        last_emit: float = 0.0
        with tarfile.open(archive_path, mode="r:xz") as tf:
            members = [
                member
                for member in tf.getmembers()
                if member.isfile() and member.name.lower().endswith(".csv")
            ]
            total_bytes = sum(max(0, int(member.size)) for member in members)
            member_count: int = len(members)
            for member_idx, member in enumerate(members, start=1):
                if not member.isfile() or not member.name.lower().endswith(".csv"):
                    continue
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                source = tf.extractfile(member)
                if source is None:
                    continue
                target_path = cache_dir.joinpath(member_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with source, open(target_path, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        bytes_done += len(chunk)
                        now = time.monotonic()
                        if (now - last_emit) >= 0.08:
                            last_emit = now
                            if total_bytes > 0:
                                fraction = float(bytes_done) / float(total_bytes)
                            else:
                                fraction = float(member_idx) / float(
                                    max(1, member_count)
                                )
                            _emit_progress(
                                progress_cb,
                                f"Decompressing archive ({member_idx}/{max(1, member_count)})...",
                                fraction,
                            )
                members_map[member.name] = str(target_path)
        extracted_members = members_map
        _emit_progress(progress_cb, "Archive cache ready.", 1.0)
    except Exception:
        extracted_members = None
        shutil.rmtree(cache_dir, ignore_errors=True)
        _emit_progress(progress_cb, "Archive cache build failed; using fallback read.")

    cached_payload: Optional[Tuple[Dict[str, str], str]] = None
    if extracted_members is not None:
        cached_payload = (extracted_members, str(cache_dir))
    _cache_put_extracted_tar_members(archive_sig, cached_payload)
    return extracted_members


def _maybe_prefetch_tar_xz_members(
    archive_path: str,
    archive_sig: tuple,
    *,
    progress_cb: ProgressCallback = None,
) -> Optional[Dict[str, bytes]]:
    found, cached = _cache_get_prefetched_tar_members(archive_sig)
    if found:
        return cached

    members_by_name: Optional[Dict[str, bytes]] = None
    _emit_progress(progress_cb, "Reading compressed archive...", 0.0)
    try:
        with tarfile.open(archive_path, mode="r:xz") as tf:
            members = [
                member
                for member in tf.getmembers()
                if member.isfile() and member.name.lower().endswith(".csv")
            ]
            total_bytes: int = sum(int(member.size) for member in members)
            if total_bytes <= _TAR_PREFETCH_MAX_BYTES:
                extracted: dict[str, bytes] = {}
                bytes_done: int = 0
                member_count: int = len(members)
                for member_idx, member in enumerate(members, start=1):
                    handle = tf.extractfile(member)
                    if handle is None:
                        continue
                    with handle:
                        payload = handle.read()
                    extracted[member.name] = payload
                    bytes_done += len(payload)
                    if total_bytes > 0:
                        fraction = float(bytes_done) / float(total_bytes)
                    else:
                        fraction = float(member_idx) / float(max(1, member_count))
                    _emit_progress(
                        progress_cb,
                        f"Reading compressed archive ({member_idx}/{max(1, member_count)})...",
                        fraction,
                    )
                members_by_name = extracted
            else:
                _emit_progress(
                    progress_cb,
                    "Archive is large; switching to streamed extraction cache...",
                )
    except Exception:
        members_by_name = None
        _emit_progress(progress_cb, "Compressed archive read failed; using fallback.")

    _cache_put_prefetched_tar_members(archive_sig, members_by_name)
    return members_by_name


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


class ArchiveCsvMemberStream:
    """Incremental archive CSV-member iterator for non-blocking UI scans."""

    def __init__(self, archive_path: str):
        archive_text = _normalize_archive_path(str(archive_path))
        archive_suffix = _archive_suffix(archive_text)
        if archive_suffix not in SUPPORTED_ARCHIVE_EXTENSIONS:
            raise ValueError(f"Unsupported archive format: {archive_text}")
        self.archive_path: str = archive_text
        self.archive_suffix: str = str(archive_suffix)
        self.done: bool = False
        self._zip = None
        self._tar = None
        self._iterator = None

        if self.archive_suffix == ".zip":
            self._zip = zipfile.ZipFile(self.archive_path)
            self._iterator = iter(self._zip.namelist())
            return

        self._tar = tarfile.open(self.archive_path, mode="r:xz")
        self._iterator = iter(self._tar)

    def next_batch(self, max_items: int = 128) -> list[str]:
        if self.done:
            return []
        max_items = max(1, int(max_items))
        items: list[str] = []
        while len(items) < max_items:
            try:
                entry = next(self._iterator)
            except StopIteration:
                self.done = True
                self.close()
                break

            if self.archive_suffix == ".zip":
                member_name: str = str(entry)
                if member_name.lower().endswith(".csv") and not member_name.endswith(
                    "/"
                ):
                    items.append(member_name)
                continue

            # TAR stream: entry is TarInfo.
            member = entry
            if member.isfile() and str(member.name).lower().endswith(".csv"):
                items.append(str(member.name))
        return items

    def close(self) -> None:
        if self._zip is not None:
            try:
                self._zip.close()
            except Exception:
                pass
            self._zip = None
        if self._tar is not None:
            try:
                self._tar.close()
            except Exception:
                pass
            self._tar = None
        self._iterator = iter(())
        self.done = True

    def __del__(self):
        self.close()


def open_archive_csv_member_stream(archive_path: str) -> ArchiveCsvMemberStream:
    return ArchiveCsvMemberStream(archive_path)


def list_archive_csv_members(
    archive_path, progress_cb: ProgressCallback = None
) -> list[str]:
    archive_text = _normalize_archive_path(str(archive_path))
    archive_suffix = _archive_suffix(archive_text)
    archive_sig = _archive_signature(archive_text)
    list_cache_key = ("members", archive_sig)
    cached_members = _cache_get_member_list(list_cache_key)
    if cached_members is not None:
        _emit_progress(progress_cb, "Archive member list loaded from cache.", 1.0)
        return cached_members

    if archive_suffix == ".zip":
        _emit_progress(progress_cb, "Reading ZIP archive index...", 0.0)
        with zipfile.ZipFile(archive_text) as zf:
            members = sorted(
                member
                for member in zf.namelist()
                if member.lower().endswith(".csv") and not member.endswith("/")
            )
        _cache_put_member_list(list_cache_key, members)
        _emit_progress(progress_cb, "Archive index ready.", 1.0)
        return members

    if archive_suffix == ".tar.xz":
        _emit_progress(progress_cb, "Reading TAR.XZ archive index...", 0.0)
        prefetched_found, prefetched_members = _cache_get_prefetched_tar_members(
            archive_sig
        )
        if prefetched_found and prefetched_members is not None:
            members = sorted(prefetched_members.keys())
            _cache_put_member_list(list_cache_key, members)
            _emit_progress(progress_cb, "Archive index ready.", 1.0)
            return members

        extracted_found, extracted_members = _cache_get_extracted_tar_members(
            archive_sig
        )
        if extracted_found and extracted_members is not None:
            members = sorted(extracted_members.keys())
            _cache_put_member_list(list_cache_key, members)
            _emit_progress(progress_cb, "Archive index ready.", 1.0)
            return members

        with tarfile.open(archive_text, mode="r:xz") as tf:
            members = sorted(
                member.name
                for member in tf.getmembers()
                if member.isfile() and member.name.lower().endswith(".csv")
            )
        _cache_put_member_list(list_cache_key, members)
        _emit_progress(progress_cb, "Archive index ready.", 1.0)
        return members

    raise ValueError(f"Unsupported archive format: {archive_text}")


def read_archive_member_bytes(
    archive_path: str,
    member: str,
    *,
    progress_cb: ProgressCallback = None,
) -> bytes:
    archive_text = _normalize_archive_path(str(archive_path))
    member_name = str(member).strip()
    archive_suffix = _archive_suffix(archive_text)
    archive_sig = _archive_signature(archive_text)
    member_cache_key = ("member", archive_sig, member_name)
    cached_payload = _cache_get_member(member_cache_key)
    if cached_payload is not None:
        _emit_progress(progress_cb, "Loaded archive member from cache.", 1.0)
        return cached_payload

    if archive_suffix == ".zip":
        _emit_progress(progress_cb, "Reading ZIP member...", 0.0)
        with zipfile.ZipFile(archive_text) as zf:
            with zf.open(member_name) as handle:
                payload = handle.read()
        _cache_put_member(member_cache_key, payload)
        _emit_progress(progress_cb, "Archive member ready.", 1.0)
        return payload

    if archive_suffix == ".tar.xz":
        prefetched_members = _maybe_prefetch_tar_xz_members(
            archive_text,
            archive_sig,
            progress_cb=progress_cb,
        )
        if prefetched_members is not None:
            if member_name not in prefetched_members:
                raise KeyError(f"Archive member is not a regular file: {member_name}")
            payload = prefetched_members[member_name]
            _cache_put_member(member_cache_key, payload)
            _emit_progress(progress_cb, "Archive member ready.", 1.0)
            return payload

        extracted_members = _maybe_extract_tar_xz_members_to_disk(
            archive_text,
            archive_sig,
            progress_cb=progress_cb,
        )
        if extracted_members is not None:
            extracted_path = extracted_members.get(member_name)
            if extracted_path is None:
                raise KeyError(f"Archive member is not a regular file: {member_name}")
            _emit_progress(progress_cb, "Reading extracted archive member...")
            with open(extracted_path, "rb") as handle:
                payload = handle.read()
            _cache_put_member(member_cache_key, payload)
            _emit_progress(progress_cb, "Archive member ready.", 1.0)
            return payload

        _emit_progress(progress_cb, "Reading TAR.XZ member...", 0.0)
        with tarfile.open(archive_text, mode="r:xz") as tf:
            extracted = tf.extractfile(member_name)
            if extracted is None:
                raise KeyError(f"Archive member is not a regular file: {member_name}")
            with extracted:
                payload = extracted.read()
        _cache_put_member(member_cache_key, payload)
        _emit_progress(progress_cb, "Archive member ready.", 1.0)
        return payload

    raise ValueError(f"Unsupported archive format: {archive_text}")


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


def read_measurement_csv(
    file_ref: str,
    *,
    progress_cb: ProgressCallback = None,
):
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
        _emit_progress(progress_cb, "Loading compressed CSV member...", 0.0)

        def _archive_progress(message: str, fraction: Optional[float]) -> None:
            mapped = None
            if fraction is not None:
                mapped = 0.85 * float(max(0.0, min(1.0, fraction)))
            _emit_progress(progress_cb, message, mapped)

        raw = read_archive_member_bytes(
            archive_path,
            member,
            progress_cb=_archive_progress,
        )
        _emit_progress(progress_cb, "Parsing CSV data...", 0.9)
        preview_lines = raw.decode("utf-8", errors="ignore").splitlines()
        header_row = detect_header_row(preview_lines)
        read_kwargs = {"header": 0, "low_memory": False}
        if header_row > 0:
            read_kwargs["skiprows"] = header_row
        frame = read_csv(BytesIO(raw), **read_kwargs)
        if frame.shape[1] < 2 and header_row == 0:
            frame = read_csv(BytesIO(raw), skiprows=13, header=0, low_memory=False)
    else:
        _emit_progress(progress_cb, "Reading CSV file...", 0.2)
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
        _emit_progress(progress_cb, "Parsing CSV data...", 0.9)
        frame = read_csv(file_ref, **read_kwargs)
        if frame.shape[1] < 2 and header_row == 0:
            frame = read_csv(file_ref, skiprows=13, header=0, low_memory=False)

    frame = frame.rename(
        columns={col: normalize_column_name(col) for col in frame.columns}
    )
    if "TIME" not in frame.columns and "TIME(S)" in frame.columns:
        frame = frame.rename(columns={"TIME(S)": "TIME"})
    _emit_progress(progress_cb, "CSV load complete.", 1.0)
    return frame


def stem_for_file_ref(file_ref: str) -> str:
    if "::" in file_ref:
        _zip_path, member = file_ref.split("::", 1)
        return Path(member).stem
    return Path(file_ref).stem
