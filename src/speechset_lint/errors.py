"""Exception types raised by speechset-lint."""

from __future__ import annotations

import errno

from speechset_lint.i18n import Msg, t


class SpeechsetLintError(Exception):
    """Base class for all errors raised on purpose by this package.

    ``key`` and ``params`` name a message in ``speechset_lint.i18n.MESSAGES``; ``str(exc)``
    is always the English text, and the CLI renders the same key in the chosen language.
    The first argument must be a message key, never a finished sentence
    (tests/test_message_keys.py checks every raise site).
    """

    def __init__(self, key: str, /, **params: object) -> None:
        self.key = key
        self.params = params
        super().__init__(t("en", key, **params))


class ManifestError(SpeechsetLintError):
    """The manifest cannot be parsed (the message includes the line number)."""


class UnknownFormatError(ManifestError):
    """The manifest format could not be detected; pass ``fmt=`` explicitly."""


class TokenizerUnavailableError(SpeechsetLintError):
    """A tokenizer was requested but cannot be used (missing package or file)."""


class CleanError(SpeechsetLintError):
    """The cleaned manifest cannot be written where it was requested."""


# 자주 나오는 OS 오류 번호만 번역 키로 바꾼다. 나머지는 번호를 보여 준다.
_OS_REASONS = {
    errno.ENOENT: "os.enoent",
    errno.EACCES: "os.eacces",
    errno.EPERM: "os.eacces",
    errno.EISDIR: "os.eisdir",
    errno.ENOTDIR: "os.enotdir",
    errno.ENOSPC: "os.enospc",
    errno.EROFS: "os.erofs",
}


def os_reason(exc: OSError) -> Msg:
    """The reason of an ``OSError`` as a message the CLI can show in English or Korean."""
    key = _OS_REASONS.get(exc.errno or 0)
    if key is None:
        return Msg("os.other", errno=errno.errorcode.get(exc.errno or 0, exc.errno))
    return Msg(key)
