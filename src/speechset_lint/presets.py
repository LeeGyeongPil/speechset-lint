"""Presets: thresholds and severities per target model family.

Every issue code a preset does not list is switched off for that preset. The
table below is the single source of truth (the README copies it).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from speechset_lint.errors import SpeechsetLintError

ERROR = "error"
WARNING = "warning"
INFO = "info"
SEVERITIES = (ERROR, WARNING, INFO)

# wav2vec2 feature encoder: 7 conv layers (kernel, stride)
_CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
_CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)
WAV2VEC2_RATE = 16000


def wav2vec2_frames(num_samples: int) -> int:
    """Frames the wav2vec2 convolutional encoder emits for ``num_samples`` samples."""
    length = num_samples
    for kernel, stride in zip(_CONV_KERNELS, _CONV_STRIDES):
        if length < kernel:
            return 0
        length = (length - kernel) // stride + 1
    return length


def ctc_labels_needed(text: str) -> int:
    """Minimum CTC alignment length: one frame per character plus a blank between repeats."""
    chars = text.strip()
    repeats = sum(1 for a, b in zip(chars, chars[1:]) if a == b)
    return len(chars) + repeats


@dataclass(frozen=True)
class Preset:
    name: str
    sample_rate: int | None = None
    min_seconds: float | None = None
    max_seconds: float | None = None
    max_label_tokens: int | None = None
    min_cps: float | None = 1.0
    max_cps: float | None = 30.0
    signal: bool = False
    clip_level: float = 0.999
    clip_ratio: float = 0.001
    silence_db: float = -50.0
    silence_ratio: float = 0.95
    severity: Mapping[str, str] = field(default_factory=dict)


_COMMON = {
    "missing_file": ERROR,
    "unreadable_audio": ERROR,
    "empty_text": ERROR,
    "duration_mismatch": WARNING,
    "char_rate_high": WARNING,
    "char_rate_low": WARNING,
    "not_nfc": WARNING,
    "control_chars": WARNING,
    "duplicate_audio": WARNING,
    "repeated_path": WARNING,
    "oov_chars": WARNING,
    "clipping": WARNING,
    "mostly_silent": ERROR,
}

PRESETS: dict[str, Preset] = {
    "whisper": Preset(
        name="whisper",
        sample_rate=16000,
        max_seconds=30.0,
        max_label_tokens=448,
        severity={
            **_COMMON,
            "sample_rate_mismatch": WARNING,
            "not_mono": WARNING,
            "too_long": ERROR,
            "label_tokens_over": ERROR,
            "label_tokens_maybe_over": WARNING,
        },
    ),
    "ctc": Preset(
        name="ctc",
        sample_rate=16000,
        severity={
            **_COMMON,
            "oov_chars": ERROR,
            "sample_rate_mismatch": WARNING,
            "not_mono": WARNING,
            "ctc_unalignable": ERROR,
        },
    ),
    "tts": Preset(
        name="tts",
        sample_rate=22050,
        min_seconds=1.0,
        max_seconds=20.0,
        signal=True,
        severity={
            **_COMMON,
            "sample_rate_mismatch": ERROR,
            "not_mono": ERROR,
            "too_long": WARNING,
            "too_short": WARNING,
        },
    ),
}


def get_preset(name: str, **overrides: object) -> Preset:
    """Look up a preset by name and apply non-None keyword overrides."""
    try:
        base = PRESETS[name]
    except KeyError:
        raise SpeechsetLintError("e.unknown_preset", name=name, choices=", ".join(sorted(PRESETS))) from None
    changes = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **changes) if changes else base
