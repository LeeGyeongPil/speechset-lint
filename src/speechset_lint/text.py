"""Transcript helpers: normalization, character sets, speaking rate."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from speechset_lint.errors import SpeechsetLintError, os_reason


def is_nfc(text: str) -> bool:
    return unicodedata.is_normalized("NFC", text)


def control_chars(text: str) -> list[str]:
    """Control characters (category Cc) present in the text, sorted, unique."""
    return sorted({c for c in text if unicodedata.category(c) == "Cc"})


def chars_per_second(text: str, seconds: float) -> float | None:
    """Non-whitespace characters per second, or None when length is unknown."""
    if seconds <= 0:
        return None
    return sum(1 for c in text if not c.isspace()) / seconds


def load_alphabet(path: str | Path) -> frozenset[str]:
    """Load allowed characters from a HF ``vocab.json`` or a one-char-per-line file.

    Multi-character entries (``<pad>``, ``[UNK]`` ...) are ignored and ``|`` is
    the word delimiter, so it stands for a space.
    """
    source = Path(path)
    # 모든 오류는 메시지 키 + 파라미터로 올린다 (CLI 가 고른 언어로 출력, 종료 코드 2)
    try:
        raw = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SpeechsetLintError("e.alphabet_not_found", path=source) from None
    except OSError as exc:
        raise SpeechsetLintError("e.alphabet_read", path=source, reason=os_reason(exc)) from exc
    except UnicodeDecodeError as exc:  # e.g. a CP949 or UTF-16 file
        raise SpeechsetLintError("e.alphabet_utf8", path=source, column=exc.start + 1) from None
    # 메모장이 붙이는 BOM 과 Windows 줄바꿈(\r\n)은 글자로 세지 않는다
    raw = raw.removeprefix("﻿").replace("\r\n", "\n")
    tokens: list[str]
    if source.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SpeechsetLintError(
                "e.alphabet_json", path=source, line=exc.lineno, column=exc.colno, reason=exc.msg
            ) from None
        if isinstance(data, (dict, list)):
            tokens = [str(k) for k in data]  # vocab.json keys, or list items
        else:
            raise SpeechsetLintError("e.alphabet_shape", path=source)
    else:
        tokens = raw.split("\n")
    chars = {t for t in tokens if len(t) == 1}
    if "|" in chars:
        chars.discard("|")
        chars.add(" ")
    if not chars:
        raise SpeechsetLintError("e.alphabet_empty", path=source)
    return frozenset(chars)
