"""speechset-lint: pre-flight lint for speech fine-tuning manifests."""

from speechset_lint.checks import Issue, Report, check
from speechset_lint.clean import write_clean
from speechset_lint.errors import (
    CleanError,
    ManifestError,
    SpeechsetLintError,
    TokenizerUnavailableError,
    UnknownFormatError,
)
from speechset_lint.manifest import Row, load_manifest
from speechset_lint.presets import PRESETS, Preset

__version__ = "0.2.0"

__all__ = [
    "PRESETS",
    "CleanError",
    "Issue",
    "ManifestError",
    "Preset",
    "Report",
    "Row",
    "SpeechsetLintError",
    "TokenizerUnavailableError",
    "UnknownFormatError",
    "__version__",
    "check",
    "load_manifest",
    "write_clean",
]
