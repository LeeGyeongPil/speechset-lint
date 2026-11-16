"""Synthetic practice dataset: sine tones and silence, never real speech.

Each clip is built to trigger one issue code so the demo (and the tests) can show
every kind of problem without downloading anything.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from speechset_lint.errors import SpeechsetLintError, os_reason

import numpy as np
import soundfile as sf

RATE = 16000


@dataclass(frozen=True)
class DemoClip:
    name: str  # file name inside the audio folder
    text: str
    expect: tuple[str, ...]  # issue codes the whisper preset + signal check should raise


def _tone(freq: float, seconds: float, rate: int = RATE, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# The text of the 700-byte clip: long enough that its UTF-8 byte upper bound passes 448.
_LONG_TEXT = ("lorem ipsum dolor sit amet " * 26).strip()

CLIPS: tuple[DemoClip, ...] = (
    DemoClip("ok.wav", "hello world this is a test", ()),
    DemoClip("stereo44k.wav", "hello world this is a test", ("sample_rate_mismatch", "not_mono")),
    DemoClip("long35.wav", "a clip that runs far too long", ("too_long", "char_rate_low")),
    DemoClip("clipped.wav", "this one is too loud", ("clipping",)),
    DemoClip("silent.wav", "this clip is just silence", ("mostly_silent",)),
    DemoClip("short.wav", "a very short clip", ("char_rate_high",)),
    DemoClip("float32.wav", "stored as float samples", ()),
    DemoClip("ok.flac", "same tone in a flac file", ()),
    DemoClip("dup.wav", "hello world this is a test", ("duplicate_audio",)),
    DemoClip("missing.wav", "this audio file was never written", ("missing_file",)),
    DemoClip("tokens.wav", _LONG_TEXT, ("char_rate_high", "label_tokens_maybe_over")),
)

DEMO_CODES = (
    "too_long",
    "sample_rate_mismatch",
    "not_mono",
    "char_rate_high",
    "char_rate_low",
    "label_tokens_maybe_over",
    "missing_file",
    "duplicate_audio",
    "mostly_silent",
    "clipping",
)


def write_audio(folder: Path) -> dict[str, float]:
    """Write every clip (except the missing one) into ``folder``; return durations in seconds."""
    folder.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}

    def put(name: str, data: np.ndarray, rate: int = RATE, subtype: str = "PCM_16") -> None:
        sf.write(str(folder / name), data, rate, subtype=subtype)
        durations[name] = len(data) / rate

    put("ok.wav", _tone(440, 2.0))
    stereo = np.stack([_tone(300, 2.0, 44100), _tone(310, 2.0, 44100)], axis=1)
    put("stereo44k.wav", stereo, 44100)
    put("long35.wav", _tone(500, 35.0))
    put("clipped.wav", np.clip(_tone(600, 2.0, amp=1.5), -1.0, 1.0))
    put("silent.wav", np.zeros(2 * RATE, dtype=np.float32))
    put("short.wav", _tone(700, 0.2))
    put("float32.wav", _tone(800, 2.0), subtype="FLOAT")
    put("ok.flac", _tone(900, 2.0), subtype="PCM_16")
    shutil.copyfile(folder / "ok.wav", folder / "dup.wav")
    durations["dup.wav"] = durations["ok.wav"]
    put("tokens.wav", _tone(1000, 2.0))
    durations["missing.wav"] = 2.0  # listed in the manifest but never written
    return durations


MARKER = ".speechset-demo"


def write_manifests(root: Path, clips: tuple[DemoClip, ...] = CLIPS, *, force: bool = False) -> dict[str, Path]:
    """Create ``nemo/``, ``audiofolder/`` and ``ljspeech/`` datasets under ``root``.

    Returns the manifest path of each format.
    """
    if root.exists() and not root.is_dir():
        raise SpeechsetLintError("e.demo_not_dir", path=root)
    if root.is_dir() and not force and not (root / MARKER).exists() and any(root.iterdir()):
        raise SpeechsetLintError("e.demo_nonempty", path=root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / MARKER).write_text("created by speechset-lint demo\n", encoding="utf-8")
        scratch = Path(tempfile.mkdtemp(prefix="speechset-demo-"))  # outside root: never touches user files
        try:
            return _write_all(root, scratch, clips)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except OSError as exc:  # 권한 없음·공간 부족 등도 키 + 번역된 사유로 올린다
        raise SpeechsetLintError("e.demo_write", path=root, reason=os_reason(exc)) from exc


def _write_all(root: Path, scratch: Path, clips: tuple[DemoClip, ...]) -> dict[str, Path]:
    durations = write_audio(scratch)
    paths: dict[str, Path] = {}

    nemo = root / "nemo"
    shutil.copytree(scratch, nemo / "audio", dirs_exist_ok=True)
    lines = [
        json.dumps({"audio_filepath": f"audio/{c.name}", "text": c.text, "duration": round(durations[c.name], 3)})
        for c in clips
    ]
    (nemo / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["nemo"] = nemo / "manifest.jsonl"

    folder = root / "audiofolder"
    shutil.copytree(scratch, folder / "audio", dirs_exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["file_name", "transcription"])
    for c in clips:
        writer.writerow([f"audio/{c.name}", c.text])
    (folder / "metadata.csv").write_text(buffer.getvalue(), encoding="utf-8")
    paths["audiofolder"] = folder / "metadata.csv"

    lj = root / "ljspeech"
    shutil.copytree(scratch, lj / "wavs", dirs_exist_ok=True)
    # LJSpeech ids carry no extension; non-wav clips get a suffix so ids stay unique
    # (ok.flac -> id ok_flac, file wavs/ok_flac.flac) and the extension hint can fire.
    rows = []
    for c in clips:
        stem, ext = Path(c.name).stem, Path(c.name).suffix
        if ext != ".wav" and (lj / "wavs" / c.name).exists():
            (lj / "wavs" / c.name).rename(lj / "wavs" / f"{stem}_{ext[1:]}{ext}")
        ident = stem if ext == ".wav" else f"{stem}_{ext[1:]}"
        rows.append(f"{ident}|{c.text}|{c.text}")
    (lj / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    paths["ljspeech"] = lj / "metadata.csv"

    return paths
