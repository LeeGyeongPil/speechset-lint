"""Pytest fixtures. Language env vars are cleared so output is deterministic."""

from __future__ import annotations

import pytest

from helpers import make_tokenizer
from speechset_lint.demo import write_manifests


@pytest.fixture(autouse=True)
def _clean_lang_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SPEECHSET_LINT_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def demo_manifests(tmp_path):
    """The synthetic dataset written in all three formats (fresh per test)."""
    return write_manifests(tmp_path / "demo")


@pytest.fixture()
def tokenizer_json(tmp_path):
    return make_tokenizer(tmp_path / "tokenizer.json")
