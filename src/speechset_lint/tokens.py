"""Label token counting: exact via tokenizer.json, otherwise a byte upper bound."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from speechset_lint.errors import TokenizerUnavailableError


def utf8_upper_bound(text: str) -> int:
    """Upper bound on byte-level BPE tokens: every token covers at least one byte."""
    return len(text.encode("utf-8"))


def load_tokenizer(path: str | Path) -> Any:
    """Load a Hugging Face ``tokenizer.json`` (needs the optional ``tokenizers``)."""
    # 경로 오류를 먼저 본다: 패키지가 없어도 경로가 틀렸다는 사실이 더 먼저 고칠 것이다
    source = Path(path)
    if not source.is_file():
        raise TokenizerUnavailableError("e.tok_not_found", path=source)
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise TokenizerUnavailableError("e.tok_missing") from exc
    try:
        return Tokenizer.from_file(str(source))
    except Exception as exc:
        # 영어 화면에만 tokenizers 의 원래 사유를 붙인다 (한국어 문구는 사유 없이 원인을 설명)
        raise TokenizerUnavailableError("e.tok_load", path=source, reason=str(exc).strip() or type(exc).__name__) from exc


def count_tokens(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    """Token counts including the special tokens the tokenizer's post-processor adds."""
    if not texts:
        return []
    return [len(enc.ids) for enc in tokenizer.encode_batch(list(texts))]
