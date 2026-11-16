"""Run every check over a manifest and assemble a Report."""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from speechset_lint import audio as audio_mod
from speechset_lint import text as text_mod
from speechset_lint import tokens as tokens_mod
from speechset_lint.errors import SpeechsetLintError
from speechset_lint.i18n import Msg
from speechset_lint.manifest import Manifest, Row, load_manifest_full
from speechset_lint.presets import (
    ERROR,
    WARNING,
    WAV2VEC2_RATE,
    Preset,
    ctc_labels_needed,
    get_preset,
    wav2vec2_frames,
)

SCHEMA_VERSION = 1
DURATION_TOLERANCE = 0.1  # seconds, NeMo `duration` versus the audio header
EPS = 1e-9


@dataclass(frozen=True)
class Issue:
    line_no: int
    audio_path: str
    code: str
    severity: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    rows: list[Row]
    issues: list[Issue]
    summary: dict[str, Any]
    preset: Preset
    fmt: str = ""
    manifest: Path | None = None
    header: bytes = b""
    trailing_newline: bool = True
    signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest": str(self.manifest) if self.manifest else None,
            "format": self.fmt,
            "preset": self.preset.name,
            "signal": self.signal,
            "summary": self.summary,
            "issues": [
                {
                    "line": i.line_no,
                    "audio": i.audio_path,
                    "code": i.code,
                    "severity": i.severity,
                    "detail": i.detail,
                }
                for i in self.issues
            ],
        }

    def has_errors(self, fail_on: str = ERROR) -> bool:
        """True when the run should fail: errors, or warnings too with ``fail_on='warning'``."""
        if fail_on not in (ERROR, WARNING):
            raise SpeechsetLintError("e.bad_choice", name="fail_on", value=fail_on, choices="error, warning")
        if self.summary["errors"]:
            return True
        return fail_on == WARNING and bool(self.summary["warnings"])

    def clean_rows(self, drop: str = ERROR) -> list[Row]:
        """Rows left after removing those with an error (or warning with ``drop='warning'``)."""
        if drop not in (ERROR, WARNING):
            raise SpeechsetLintError("e.bad_choice", name="drop", value=drop, choices="error, warning")
        bad = {ERROR} if drop == ERROR else {ERROR, WARNING}
        dropped = {i.line_no for i in self.issues if i.severity in bad}
        return [r for r in self.rows if r.line_no not in dropped]


def _validate_numbers(extra_tokens: int, sample_rate: int | None, limits: dict[str, float | None]) -> None:
    """Reject values that would silently flag every row (negative limits, zero sample rate)."""
    if isinstance(extra_tokens, bool) or not isinstance(extra_tokens, int) or extra_tokens < 0:
        raise SpeechsetLintError("e.not_negative", name="extra_tokens", value=extra_tokens)
    if sample_rate is not None and (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
        raise SpeechsetLintError("e.not_positive", name="sample_rate", value=sample_rate)
    for name, value in limits.items():
        if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            raise SpeechsetLintError("e.not_negative", name=name, value=value)


def _check_ranges(cfg: Preset) -> None:
    """Minimum above maximum (after preset defaults) means every clip fails; say so up front."""
    pairs = (("name.seconds", cfg.min_seconds, cfg.max_seconds), ("name.cps", cfg.min_cps, cfg.max_cps))
    for name_key, low, high in pairs:
        if low is not None and high is not None and low > high:
            raise SpeechsetLintError("e.min_gt_max", name=Msg(name_key), low=low, high=high)


def _load_alphabet(alphabet: Any) -> frozenset[str] | None:
    if alphabet is None:
        return None
    if isinstance(alphabet, (str, os.PathLike)):
        return text_mod.load_alphabet(alphabet)  # a path; pass a set/list for literal characters
    return frozenset(alphabet)


def check(
    manifest: str | os.PathLike[str],
    *,
    fmt: str = "auto",
    preset: str = "whisper",
    tokenizer: str | os.PathLike[str] | None = None,
    alphabet: Any = None,
    signal: bool | None = None,
    dedupe: bool = True,
    extra_tokens: int = 2,
    audio_ext: str = "wav",
    text_field: str | None = None,
    sample_rate: int | None = None,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
    min_cps: float | None = None,
    max_cps: float | None = None,
) -> Report:
    """Lint ``manifest`` for ``preset`` and return a :class:`Report`.

    ``signal=None`` uses the preset default (on for ``tts``). ``alphabet`` may be a
    path to ``vocab.json``/a character list file, or an iterable of characters.
    """
    started = time.perf_counter()
    # 입력 숫자부터 검사한다: 음수 한도는 모든 행을 오류로 만들어 원인을 숨긴다
    _validate_numbers(
        extra_tokens,
        sample_rate,
        {"min_seconds": min_seconds, "max_seconds": max_seconds, "min_cps": min_cps, "max_cps": max_cps},
    )
    cfg = get_preset(
        preset,
        sample_rate=sample_rate,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        min_cps=min_cps,
        max_cps=max_cps,
    )
    _check_ranges(cfg)
    data: Manifest = load_manifest_full(manifest, fmt, text_field=text_field, audio_ext=audio_ext)
    allowed = _load_alphabet(alphabet)
    use_signal = cfg.signal if signal is None else signal
    tokenizer_obj = tokens_mod.load_tokenizer(tokenizer) if tokenizer is not None else None
    exact_counts: list[int] | None = None
    if tokenizer_obj is not None and cfg.max_label_tokens is not None:
        exact_counts = tokens_mod.count_tokens(tokenizer_obj, [r.text.strip() for r in data.rows])

    issues: list[Issue] = []
    headers: dict[Path, audio_mod.AudioInfo | str] = {}
    first_by_path: dict[Path, Row] = {}
    first_by_hash: dict[str, Row] = {}

    def add(row: Row, code: str, **detail: Any) -> None:
        severity = cfg.severity.get(code)
        if severity is not None:
            issues.append(Issue(row.line_no, row.audio_path, code, severity, detail))

    for index, row in enumerate(data.rows):
        path = row.path or Path(row.audio_path)
        text = row.text
        stripped = text.strip()
        info: audio_mod.AudioInfo | None = None

        if not stripped:
            add(row, "empty_text")

        previous = first_by_path.get(path)
        repeated = previous is not None
        if repeated:
            add(row, "repeated_path", first_line=previous.line_no)
        else:
            first_by_path[path] = row

        if not path.is_file():
            found = _same_stem(path)
            detail: dict[str, Any] = {"path": str(path)}
            if found:
                detail["found_ext"] = found
            add(row, "missing_file", **detail)
        else:
            cached = headers.get(path)
            if cached is None:
                try:
                    cached = audio_mod.read_header(path)
                except audio_mod.UnreadableAudio as exc:
                    cached = str(exc)
                headers[path] = cached
            if isinstance(cached, str):
                if not repeated:
                    add(row, "unreadable_audio", error=cached)
            else:
                info = cached
                if not repeated:
                    _check_audio(row, info, cfg, add)
                    if use_signal:
                        _check_signal(row, path, cfg, add)
                    if dedupe:
                        digest = audio_mod.file_sha1(path)
                        first = first_by_hash.setdefault(digest, row)
                        if first is not row:
                            add(row, "duplicate_audio", first_line=first.line_no, first_audio=first.audio_path)

        _check_text(row, stripped, text, info.duration if info else None, cfg, allowed, add)

        if exact_counts is not None and cfg.max_label_tokens is not None:
            if exact_counts[index] + extra_tokens > cfg.max_label_tokens:
                add(row, "label_tokens_over", tokens=exact_counts[index], extra=extra_tokens, limit=cfg.max_label_tokens)
        elif cfg.max_label_tokens is not None and stripped:
            bound = tokens_mod.utf8_upper_bound(stripped)
            if bound + extra_tokens > cfg.max_label_tokens:
                add(row, "label_tokens_maybe_over", tokens=bound, extra=extra_tokens, limit=cfg.max_label_tokens)

        if "ctc_unalignable" in cfg.severity and info is not None and stripped:
            frames = wav2vec2_frames(round(info.duration * WAV2VEC2_RATE))
            needed = ctc_labels_needed(stripped)
            if frames < needed:
                add(row, "ctc_unalignable", frames=frames, labels=needed)

        if row.duration is not None and info is not None and abs(row.duration - info.duration) > DURATION_TOLERANCE + 1e-6:
            add(
                row,
                "duration_mismatch",
                manifest=round(row.duration, 3),
                header=round(info.duration, 3),
                diff=round(abs(row.duration - info.duration), 3),
            )

    issues.sort(key=lambda i: (i.line_no, i.code))
    by_code = dict(sorted(Counter(i.code for i in issues).items()))
    summary = {
        "rows": len(data.rows),
        "errors": sum(1 for i in issues if i.severity == ERROR),
        "warnings": sum(1 for i in issues if i.severity == WARNING),
        "by_code": by_code,
        "seconds": round(time.perf_counter() - started, 3),
    }
    return Report(
        rows=data.rows,
        issues=issues,
        summary=summary,
        preset=cfg,
        fmt=data.fmt,
        manifest=data.path,
        header=data.header,
        trailing_newline=data.trailing_newline,
        signal=use_signal,
    )


def _same_stem(path: Path) -> list[str]:
    """Extensions of sibling files with the same name stem (for the wrong-extension hint)."""
    try:
        return sorted(p.suffix for p in path.parent.iterdir() if p.stem == path.stem and p != path and p.is_file())
    except OSError:
        return []


def _check_audio(row: Row, info: audio_mod.AudioInfo, cfg: Preset, add) -> None:
    if cfg.sample_rate is not None and info.sample_rate != cfg.sample_rate:
        add(row, "sample_rate_mismatch", actual=info.sample_rate, expected=cfg.sample_rate)
    if info.channels != 1:
        add(row, "not_mono", channels=info.channels)
    seconds = info.duration
    if cfg.max_seconds is not None and seconds > cfg.max_seconds + EPS:
        add(row, "too_long", seconds=round(seconds, 3), limit=cfg.max_seconds)
    if cfg.min_seconds is not None and seconds < cfg.min_seconds - EPS:
        add(row, "too_short", seconds=round(seconds, 3), limit=cfg.min_seconds)


def _check_signal(row: Row, path: Path, cfg: Preset, add) -> None:
    try:
        metrics = audio_mod.signal_metrics(path, clip_level=cfg.clip_level, silence_db=cfg.silence_db)
    except audio_mod.UnreadableAudio as exc:
        add(row, "unreadable_audio", error=str(exc))
        return
    if metrics.clip_ratio >= cfg.clip_ratio:
        add(row, "clipping", ratio=round(metrics.clip_ratio, 6), limit=cfg.clip_ratio)
    if metrics.silence_ratio >= cfg.silence_ratio:
        add(row, "mostly_silent", ratio=round(metrics.silence_ratio, 6), db=cfg.silence_db, limit=cfg.silence_ratio)


def _check_text(
    row: Row,
    stripped: str,
    text: str,
    seconds: float | None,
    cfg: Preset,
    allowed: frozenset[str] | None,
    add,
) -> None:
    if not text:
        return
    if not text_mod.is_nfc(text):
        add(row, "not_nfc")
    controls = text_mod.control_chars(text)
    if controls:
        add(row, "control_chars", chars=[f"U+{ord(c):04X}" for c in controls])
    if allowed is not None:
        unknown = sorted({c for c in stripped if not c.isspace() and c not in allowed})
        if unknown:
            add(row, "oov_chars", chars=unknown)
    if seconds:
        cps = text_mod.chars_per_second(stripped, seconds)
        if cps is not None:
            if cfg.max_cps is not None and cps > cfg.max_cps:
                add(row, "char_rate_high", cps=round(cps, 2), limit=cfg.max_cps)
            elif cfg.min_cps is not None and stripped and cps < cfg.min_cps:
                add(row, "char_rate_low", cps=round(cps, 2), limit=cfg.min_cps)
