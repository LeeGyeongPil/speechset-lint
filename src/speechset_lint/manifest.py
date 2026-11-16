"""Read NeMo JSONL, HF audiofolder (CSV/JSONL) and LJSpeech manifests as-is.

Every row keeps its original bytes (``raw``) so a cleaned manifest can be
written back without re-serializing anything.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from speechset_lint.errors import ManifestError, UnknownFormatError, os_reason

FORMATS = ("nemo", "audiofolder", "ljspeech")
TEXT_COLUMNS = ("transcription", "text", "sentence", "normalized_text")


@dataclass
class Row:
    line_no: int
    raw: bytes
    audio_path: str  # as written in the manifest (LJSpeech: derived wavs/<id>.<ext>)
    text: str
    duration: float | None = None  # NeMo `duration`, when present
    path: Path | None = None  # resolved filesystem path


@dataclass
class Manifest:
    path: Path
    fmt: str
    rows: list[Row]
    header: bytes = b""  # audiofolder CSV header line(s), kept when writing clean files
    trailing_newline: bool = True


def _split_lines(data: bytes) -> list[bytes]:
    """Split on ``\\n`` only (keeping terminators) so other separators stay intact."""
    parts = data.split(b"\n")
    lines = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _decode(data: bytes, path: Path, line: int) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("e.utf8_column", path=path, line=line, column=exc.start + 1) from exc


def _reject_binary(data: bytes, path: Path) -> None:
    """UTF-16/32 text and binary files carry NUL bytes; say so instead of failing inside a parser."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        raise ManifestError("e.utf16", path=path)
    nul = data.find(b"\x00")
    if nul >= 0:
        raise ManifestError("e.nul", path=path, line=data[:nul].count(b"\n") + 1)


def detect_format(path: Path, lines: list[bytes]) -> str:
    """Guess the format from the first non-blank line; raise UnknownFormatError otherwise."""
    first_no, first = next(
        ((no, ln.decode("utf-8", "replace")) for no, ln in enumerate(lines, start=1) if ln.strip()), (0, None)
    )
    if first is None:
        raise UnknownFormatError("e.empty", path=path)
    stripped = first.strip().lstrip("﻿")
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            # 깨진 JSON 은 "키가 없다"가 아니라 깨진 위치를 알려 준다
            raise ManifestError(
                "e.invalid_json", path=path, line=first_no, column=exc.colno, reason=exc.msg
            ) from None
        if isinstance(obj, dict):
            if "audio_filepath" in obj:
                return "nemo"
            if "file_name" in obj:
                return "audiofolder"
        raise UnknownFormatError("e.json_keys", path=path)
    columns = [c.strip().lstrip("﻿") for c in next(csv.reader([stripped]), [])]
    if "file_name" in columns:
        return "audiofolder"
    if "|" in stripped:
        return "ljspeech"
    raise UnknownFormatError("e.detect_fail", path=path)


def _pick_text(obj: dict, text_field: str | None) -> str:
    if text_field is not None:
        value = obj.get(text_field, "")
    else:
        value = ""
        for key in TEXT_COLUMNS:
            if key in obj:
                value = obj[key]
                break
    return "" if value is None else str(value)


def _jsonl_rows(lines: list[bytes], path: Path, fmt: str, text_field: str | None) -> list[Row]:
    key = "audio_filepath" if fmt == "nemo" else "file_name"
    rows: list[Row] = []
    for idx, raw in enumerate(lines, start=1):
        try:
            body = raw.decode("utf-8").strip().lstrip("﻿")
        except UnicodeDecodeError:
            raise ManifestError("e.utf8_line", path=path, line=idx) from None
        if not body:
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            # 한국어 화면에는 파서의 영어 사유 대신 위치(열)만 보여 준다
            raise ManifestError("e.invalid_json", path=path, line=idx, column=exc.colno, reason=exc.msg) from None
        if not isinstance(obj, dict) or not isinstance(obj.get(key), str) or not obj[key]:
            raise ManifestError("e.missing_field", path=path, line=idx, key=key, fmt=fmt)
        if text_field is not None and not rows and text_field not in obj:
            # 없는 필드를 지정하면 모든 행이 empty_text 로 보이므로 첫 행에서 바로 알린다
            raise ManifestError(
                "e.text_field_not_key", path=path, line=idx, field=text_field, found=", ".join(map(str, obj))
            )
        if fmt == "audiofolder" and not any(k in obj for k in TEXT_COLUMNS) and text_field is None:
            raise ManifestError("e.no_transcript_field", path=path, line=idx, cols=", ".join(TEXT_COLUMNS))
        duration = None
        if fmt == "nemo" and obj.get("duration") is not None:
            try:
                duration = float(obj["duration"])
            except (TypeError, ValueError):
                raise ManifestError("e.bad_duration", path=path, line=idx, value=repr(obj["duration"])) from None
        rows.append(Row(idx, raw, obj[key], _pick_text(obj, text_field), duration))
    return rows


def _csv_records(lines: list[bytes], path: Path) -> list[tuple[int, bytes, list[str]]]:
    texts = [_decode(ln, path, idx) for idx, ln in enumerate(lines, start=1)]
    reader = csv.reader(iter(texts))
    out: list[tuple[int, bytes, list[str]]] = []
    prev = 0
    try:
        for record in reader:
            end = reader.line_num
            if record:
                out.append((prev + 1, b"".join(lines[prev:end]), record))
            prev = end
    except csv.Error as exc:
        raise ManifestError("e.invalid_csv", path=path, line=reader.line_num, reason=exc) from exc
    return out


def _audiofolder_csv_rows(
    lines: list[bytes], path: Path, text_field: str | None
) -> tuple[list[Row], bytes]:
    records = _csv_records(lines, path)
    if not records:
        raise ManifestError("e.csv_empty", path=path)
    header_line, header_raw, header = records[0]
    header = [h.strip().lstrip("﻿") for h in header]
    if "file_name" not in header:
        raise ManifestError("e.csv_no_file_name", path=path, line=header_line, found=", ".join(header))
    name_idx = header.index("file_name")
    if text_field is not None and text_field.isdigit():
        text_idx: int | None = int(text_field)
        if text_idx >= len(header):
            raise ManifestError(
                "e.text_column_range", path=path, line=header_line, index=text_idx,
                count=len(header), last=len(header) - 1,
            )
    elif text_field is not None:
        if text_field not in header:
            raise ManifestError("e.text_field_missing", path=path, field=text_field, found=", ".join(header))
        text_idx = header.index(text_field)
    else:
        text_idx = next((header.index(c) for c in TEXT_COLUMNS if c in header), None)
        if text_idx is None:
            raise ManifestError("e.csv_no_transcript", path=path, cols=", ".join(TEXT_COLUMNS), found=", ".join(header))
    rows: list[Row] = []
    for line_no, raw, record in records[1:]:
        if name_idx >= len(record) or not record[name_idx].strip():
            raise ManifestError("e.empty_file_name", path=path, line=line_no)
        text = record[text_idx] if text_idx < len(record) else ""
        rows.append(Row(line_no, raw, record[name_idx], text))
    return rows, header_raw


def _ljspeech_rows(lines: list[bytes], path: Path, text_field: str | None, audio_ext: str) -> list[Row]:
    ext = audio_ext.lstrip(".")
    column: int | None = None
    if text_field is not None:
        if not text_field.isdigit():
            raise ManifestError("e.lj_text_field")
        column = int(text_field)
    rows: list[Row] = []
    for idx, raw in enumerate(lines, start=1):
        try:
            body = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            raise ManifestError("e.utf8_line", path=path, line=idx) from None
        if not body.strip():
            continue
        parts = body.split("|")
        if len(parts) < 2 or not parts[0].strip():
            raise ManifestError("e.lj_bad_row", path=path, line=idx)
        if column is not None:
            if not rows and column >= len(parts):
                # 첫 행에 그 열이 없으면 모든 전사가 빈 값이 되므로 입력 오류로 본다
                raise ManifestError(
                    "e.text_column_range", path=path, line=idx, index=column, count=len(parts), last=len(parts) - 1
                )
            text = parts[column] if column < len(parts) else ""
        else:
            text = parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]
        rows.append(Row(idx, raw, f"wavs/{parts[0].strip()}.{ext}", text))
    return rows


def resolve_audio_path(manifest_dir: Path, audio_path: str) -> Path:
    """Absolute paths are kept; relative ones are tried against the manifest folder, then the cwd."""
    candidate = Path(audio_path)
    if candidate.is_absolute():
        return candidate
    primary = manifest_dir / candidate
    if primary.exists():
        return primary
    fallback = Path.cwd() / candidate
    return fallback if fallback.exists() else primary


def load_manifest_full(
    path: str | os.PathLike[str],
    fmt: str = "auto",
    *,
    text_field: str | None = None,
    audio_ext: str = "wav",
) -> Manifest:
    source = Path(path)
    if fmt != "auto" and fmt not in FORMATS:
        raise ManifestError("e.unknown_format", fmt=fmt, choices=", ".join(FORMATS))
    try:
        data = source.read_bytes()
    except FileNotFoundError:
        raise ManifestError("e.not_found", path=source) from None
    except OSError as exc:  # a folder, no permission ...
        raise ManifestError("e.cannot_read", path=source, reason=os_reason(exc)) from exc
    _reject_binary(data, source)
    lines = _split_lines(data)
    if not any(ln.strip().lstrip(b"\xef\xbb\xbf") for ln in lines):
        # 빈 파일은 --format 을 줘도 고칠 수 없으니 형식과 상관없이 같은 오류를 낸다
        raise ManifestError("e.empty", path=source)
    detected = detect_format(source, lines) if fmt == "auto" else fmt
    header = b""
    jsonl = _looks_like_jsonl(lines)
    if detected == "nemo":
        if not jsonl:
            raise ManifestError("e.nemo_needs_json", path=source)
        rows = _jsonl_rows(lines, source, "nemo", text_field)
    elif detected == "audiofolder":
        if jsonl:
            rows = _jsonl_rows(lines, source, "audiofolder", text_field)
        else:
            rows, header = _audiofolder_csv_rows(lines, source, text_field)
    else:
        rows = _ljspeech_rows(lines, source, text_field, audio_ext)
    for row in rows:
        row.path = resolve_audio_path(source.parent, row.audio_path)
    return Manifest(source, detected, rows, header, bool(data.endswith(b"\n")))


def _looks_like_jsonl(lines: list[bytes]) -> bool:
    first = next((ln.decode("utf-8", "replace").strip().lstrip("﻿") for ln in lines if ln.strip()), "")
    return first.startswith("{")


def load_manifest(
    path: str | os.PathLike[str],
    fmt: str = "auto",
    *,
    text_field: str | None = None,
    audio_ext: str = "wav",
) -> list[Row]:
    """Parse a manifest and return its rows (format auto-detected by default)."""
    return load_manifest_full(path, fmt, text_field=text_field, audio_ext=audio_ext).rows
