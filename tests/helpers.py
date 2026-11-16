"""Shared test helpers: tiny audio files, manifests and a tokenizer fixture."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def tone(seconds: float, rate: int = 16000, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, data: np.ndarray, rate: int = 16000, subtype: str = "PCM_16") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, rate, subtype=subtype)
    return path


def write_nemo(path: Path, rows: list[tuple[str, str, float | None]]) -> Path:
    """rows: (audio_filepath, text, duration or None)."""
    lines = []
    for audio, text, duration in rows:
        obj: dict = {"audio_filepath": audio, "text": text}
        if duration is not None:
            obj["duration"] = duration
        lines.append(json.dumps(obj, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def codes_by_audio(report) -> dict[str, set[str]]:
    """Map audio file name -> set of issue codes (format independent)."""
    out: dict[str, set[str]] = {Path(r.audio_path).name: set() for r in report.rows}
    for issue in report.issues:
        out[Path(issue.audio_path).name].add(issue.code)
    return out


def codes_of(report, name: str) -> set[str]:
    return codes_by_audio(report)[name]


# Names that stay in Latin letters inside Korean messages: formats, encodings, packages and
# manifest field names. Any other Latin word of 4+ letters is an untranslated English fragment.
KO_ALLOWED_WORDS = (
    "jsonl|json|csv|utf|nul|bom|nemo|audiofolder|ljspeech|whisper|speechset|lint|tokenizers|"
    "vocab|transcription|sentence|normalized|text|duration|audio|filepath|file_name"
)
ENGLISH_FRAGMENT = re.compile(rf"(?<![\w-])(?!(?:{KO_ALLOWED_WORDS})\b)[A-Za-z]{{4,}}\b", re.I)


def english_fragments(text: str) -> list[str]:
    """Latin words in ``text`` that are not paths, flags, quoted values or allowed names."""
    cleaned = re.sub(r"https?://\S+|--[\w-]+|'[^']*'|`[^`]*`|\S*[/\\.]\S*", " ", text)
    return ENGLISH_FRAGMENT.findall(cleaned)


def make_tokenizer(path: Path) -> Path:
    """Offline WordLevel tokenizer that adds 3 special tokens around every text."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    tok = Tokenizer(WordLevel({"[UNK]": 0, "w": 1, "<a>": 2, "<b>": 3, "<c>": 4}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="<a> <b> $A <c>",
        special_tokens=[("<a>", 2), ("<b>", 3), ("<c>", 4)],
    )
    tok.save(str(path))
    return path
