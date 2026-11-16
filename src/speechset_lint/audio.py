"""Audio header, hash and signal helpers.

Headers are read with the standard ``wave`` module first (fast, no decode) and
fall back to ``soundfile`` for formats ``wave`` cannot parse (float WAV, FLAC,
OGG ...). Signal metrics decode in blocks so memory stays bounded.
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class UnreadableAudio(Exception):
    """Neither ``wave`` nor ``soundfile`` could read the file header."""


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    via: str  # "wave" or "soundfile"

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


@dataclass(frozen=True)
class SignalMetrics:
    clip_ratio: float
    silence_ratio: float
    peak_dbfs: float


def read_header(path: Path) -> AudioInfo:
    """Return sample rate, channels and length from the file header only."""
    wave_error: Exception | None = None
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate > 0:
                return AudioInfo(rate, handle.getnchannels(), handle.getnframes(), "wave")
    except (wave.Error, EOFError, ValueError, OSError) as exc:
        wave_error = exc
    try:
        info = sf.info(str(path))
    except Exception as exc:  # libsndfile raises RuntimeError subclasses
        reason = str(exc).strip() or type(exc).__name__
        if wave_error is not None:
            reason = f"{reason} (wave: {wave_error})"
        raise UnreadableAudio(reason) from exc
    return AudioInfo(int(info.samplerate), int(info.channels), int(info.frames), "soundfile")


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-1 of the file bytes (used only to spot exact duplicates)."""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def signal_metrics(
    path: Path,
    *,
    clip_level: float = 0.999,
    silence_db: float = -50.0,
) -> SignalMetrics:
    """Decode the file in blocks and measure clipping, silence and peak level.

    Channels are folded with the per-frame maximum so one loud channel is not
    hidden by a quiet one.
    """
    silence_level = 10 ** (silence_db / 20)
    total = clipped = silent = 0
    peak = 0.0
    try:
        for block in sf.blocks(str(path), blocksize=1 << 16, dtype="float32", always_2d=True):
            level = np.abs(block).max(axis=1)
            total += level.size
            clipped += int(np.count_nonzero(level >= clip_level))
            silent += int(np.count_nonzero(level < silence_level))
            if level.size:
                peak = max(peak, float(level.max()))
    except Exception as exc:
        raise UnreadableAudio(str(exc).strip() or type(exc).__name__) from exc
    if total == 0:
        return SignalMetrics(0.0, 1.0, float("-inf"))
    peak_db = 20 * float(np.log10(peak)) if peak > 0 else float("-inf")
    return SignalMetrics(clipped / total, silent / total, peak_db)
