from __future__ import annotations

import json
import re

import pytest

from speechset_lint.cli import main
from speechset_lint.i18n import MESSAGES, resolve_lang, t
from speechset_lint.presets import PRESETS


def test_lang_resolution_order():
    assert resolve_lang(None, {}) == "en"
    assert resolve_lang(None, {"LANG": "ko_KR.UTF-8"}) == "ko"
    assert resolve_lang(None, {"LANG": "en_US.UTF-8"}) == "en"
    assert resolve_lang(None, {"LANG": "de_DE.UTF-8"}) == "en"  # not Korean -> English
    # LC_ALL > LC_MESSAGES > LANG
    assert resolve_lang(None, {"LC_ALL": "ko_KR.UTF-8", "LANG": "en_US"}) == "ko"
    assert resolve_lang(None, {"LC_ALL": "en_US", "LC_MESSAGES": "ko_KR", "LANG": "ko_KR"}) == "en"
    assert resolve_lang(None, {"LC_MESSAGES": "ko_KR", "LANG": "en_US"}) == "ko"
    # SPEECHSET_LINT_LANG beats the locale variables
    assert resolve_lang(None, {"SPEECHSET_LINT_LANG": "en", "LC_ALL": "ko_KR"}) == "en"
    assert resolve_lang(None, {"SPEECHSET_LINT_LANG": "ko", "LANG": "en_US"}) == "ko"
    # --lang beats everything
    assert resolve_lang("en", {"SPEECHSET_LINT_LANG": "ko", "LC_ALL": "ko_KR"}) == "en"
    assert resolve_lang("ko", {"SPEECHSET_LINT_LANG": "en"}) == "ko"
    # an invalid env value is ignored
    assert resolve_lang(None, {"SPEECHSET_LINT_LANG": "fr", "LANG": "ko_KR"}) == "ko"


# Keys whose `{reason}` is an English parser message (json, csv, tokenizers). The Korean
# text leaves it out and explains the cause in Korean instead (QA Q-9c).
ENGLISH_ONLY_REASON = {"e.invalid_json", "e.invalid_csv", "e.alphabet_json", "e.tok_load"}


def test_message_key_parity():
    assert set(MESSAGES["en"]) == set(MESSAGES["ko"])
    for key in MESSAGES["en"]:
        assert MESSAGES["en"][key].strip() and MESSAGES["ko"][key].strip(), key
        fields_en = sorted(re.findall(r"\{(\w+)", MESSAGES["en"][key]))
        fields_ko = sorted(re.findall(r"\{(\w+)", MESSAGES["ko"][key]))
        if key in ENGLISH_ONLY_REASON:
            fields_en.remove("reason")
        assert fields_en == fields_ko, f"placeholder mismatch in {key}"


def test_every_issue_code_has_text_in_both_languages():
    codes = {code for preset in PRESETS.values() for code in preset.severity}
    for lang in ("en", "ko"):
        for code in codes:
            assert f"detail.{code}" in MESSAGES[lang], code
            assert f"why.{code}" in MESSAGES[lang], code
    assert t("xx", "col.line") == "line"  # unknown language falls back to English


def test_json_codes_language_independent(capsys, demo_manifests):
    results = {}
    for lang in ("en", "ko"):
        main(["--lang", lang, "check", str(demo_manifests["nemo"]), "--signal", "--json"])
        data = json.loads(capsys.readouterr().out)
        data["summary"].pop("seconds")
        results[lang] = data
    assert results["en"] == results["ko"]


def test_language_switch_changes_screen(capsys, demo_manifests, monkeypatch):
    manifest = str(demo_manifests["nemo"])
    main(["--lang", "en", "check", manifest])
    en = capsys.readouterr().out
    main(["--lang", "ko", "check", manifest])
    ko = capsys.readouterr().out
    assert "severity" in en and "rows checked" in en
    assert "심각도" in ko and "행 점검" in ko
    assert "심각도" not in en and "severity" not in ko
    # environment variable picks Korean when no --lang is given; --lang still wins
    monkeypatch.setenv("SPEECHSET_LINT_LANG", "ko")
    main(["check", manifest])
    assert "심각도" in capsys.readouterr().out
    main(["--lang", "en", "check", manifest])
    assert "severity" in capsys.readouterr().out


@pytest.mark.parametrize(("lang", "needle", "other"), [("ko", "사용법", "usage:"), ("en", "usage:", "사용법")])
def test_help_is_localized(capsys, lang, needle, other):
    with pytest.raises(SystemExit) as exc:
        main(["--lang", lang, "check", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert needle in out
    assert other not in out
    assert ("옵션" in out) == (lang == "ko")
