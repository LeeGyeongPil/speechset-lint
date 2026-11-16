"""Every user input error: exit 2, no traceback, one message in the chosen language (QA Q-7, Q-8, Q-9).

Each case builds a bad input in a temp folder, runs the CLI in English and in Korean, and checks
which message key was used, so a new raise site without a key (Q-7) or with an English-only
reason (Q-9c) fails here.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import english_fragments, write_nemo
from speechset_lint import cli
from speechset_lint.checks import check
from speechset_lint.clean import write_clean
from speechset_lint.demo import write_manifests
from speechset_lint.errors import SpeechsetLintError
from speechset_lint.text import load_alphabet

HANGUL = re.compile(r"[가-힣]")


def _file(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _nemo(d: Path) -> str:
    """A valid one-row NeMo manifest (the audio file does not need to exist)."""
    return str(write_nemo(d / "m.jsonl", [("a.wav", "hello", None)]))


def _demo(d: Path, fmt: str = "nemo") -> Path:
    paths = write_manifests(d / "demo")
    return paths[fmt]


# (id, expected message key, function that returns the CLI arguments after --lang)
CASES = [
    # --- manifest file
    ("manifest-missing", "e.not_found", lambda d: ["check", str(d / "nope.jsonl")]),
    ("manifest-is-folder", "e.cannot_read", lambda d: ["check", str(d)]),
    ("empty-auto", "e.empty", lambda d: ["check", _file(d / "e.jsonl", b"")]),
    ("empty-format-nemo", "e.empty", lambda d: ["check", _file(d / "e.jsonl", b"\n \n"), "--format", "nemo"]),
    ("empty-format-ljspeech", "e.empty", lambda d: ["check", _file(d / "e.csv", b""), "--format", "ljspeech"]),
    ("utf16", "e.utf16", lambda d: ["check", _file(d / "u.jsonl", b"\xff\xfe{\x00}\x00")]),
    ("nul", "e.nul", lambda d: ["check", _file(d / "n.csv", b"a|b\x00c\n")]),
    ("utf8-jsonl", "e.utf8_line",
     lambda d: ["check", _file(d / "u.jsonl", b'{"audio_filepath": "a.wav", "text": "\xff"}\n'), "--format", "nemo"]),
    ("utf8-csv", "e.utf8_column",
     lambda d: ["check", _file(d / "metadata.csv", "file_name,text\na.wav,".encode() + "안녕".encode("cp949") + b"\n")]),
    ("invalid-json-line1", "e.invalid_json", lambda d: ["check", _file(d / "j.jsonl", b'{"audio_filepath": "a.wav"\n')]),
    ("invalid-json-line2", "e.invalid_json",
     lambda d: ["check", _file(d / "j.jsonl", b'{"audio_filepath": "a.wav", "text": "x"}\n{"audio_filepath": \n')]),
    ("json-unknown-keys", "e.json_keys", lambda d: ["check", _file(d / "x.jsonl", b'{"nope": 1}\n')]),
    ("undetectable", "e.detect_fail", lambda d: ["check", _file(d / "x.txt", b"hello world\n")]),
    ("nemo-missing-field", "e.missing_field",
     lambda d: ["check", _file(d / "m.jsonl", b'{"audio_filepath": "a.wav", "text": "x"}\n{"text": "y"}\n')]),
    ("jsonl-no-transcript", "e.no_transcript_field", lambda d: ["check", _file(d / "m.jsonl", b'{"file_name": "a.wav"}\n')]),
    ("bad-duration", "e.bad_duration",
     lambda d: ["check", _file(d / "m.jsonl", b'{"audio_filepath": "a.wav", "text": "x", "duration": "long"}\n')]),
    ("nemo-needs-json", "e.nemo_needs_json", lambda d: ["check", _file(d / "m.csv", b"a|x\n"), "--format", "nemo"]),
    # --- columns and --text-field
    ("csv-no-file-name", "e.csv_no_file_name",
     lambda d: ["check", _file(d / "m.csv", b"fn,text\na.wav,x\n"), "--format", "audiofolder"]),
    ("csv-no-transcript", "e.csv_no_transcript", lambda d: ["check", _file(d / "m.csv", b"file_name,spk\na.wav,x\n")]),
    ("csv-empty-file-name", "e.empty_file_name", lambda d: ["check", _file(d / "m.csv", b"file_name,text\n,x\n")]),
    ("csv-field-too-large", "e.invalid_csv",
     lambda d: ["check", _file(d / "m.csv", b'file_name,text\na.wav,"' + b"x" * 140_000 + b'"\n')]),
    ("csv-text-field-missing", "e.text_field_missing",
     lambda d: ["check", _file(d / "m.csv", b"file_name,text\na.wav,x\n"), "--text-field", "zzz"]),
    ("csv-text-column-range", "e.text_column_range",
     lambda d: ["check", _file(d / "m.csv", b"file_name,text\na.wav,x\n"), "--text-field", "7"]),
    ("nemo-text-field-missing", "e.text_field_not_key", lambda d: ["check", _nemo(d), "--text-field", "zzz"]),
    ("lj-text-field-name", "e.lj_text_field", lambda d: ["check", _file(d / "m.csv", b"a|x|x\n"), "--text-field", "text"]),
    ("lj-text-column-range", "e.text_column_range", lambda d: ["check", _file(d / "m.csv", b"a|x|x\n"), "--text-field", "5"]),
    ("lj-bad-row", "e.lj_bad_row", lambda d: ["check", _file(d / "m.csv", b"a|x\nnope\n"), "--format", "ljspeech"]),
    # --- tokenizer and alphabet files
    ("tokenizer-missing", "e.tok_not_found", lambda d: ["check", _nemo(d), "--tokenizer", str(d / "nope.json")]),
    ("tokenizer-invalid", "e.tok_load", lambda d: ["check", _nemo(d), "--tokenizer", _file(d / "tok.json", b"{}")]),
    ("alphabet-missing", "e.alphabet_not_found", lambda d: ["check", _nemo(d), "--alphabet", str(d / "nope.json")]),
    ("alphabet-folder", "e.alphabet_read", lambda d: ["check", _nemo(d), "--alphabet", str(d)]),
    ("alphabet-not-utf8", "e.alphabet_utf8",
     lambda d: ["check", _nemo(d), "--alphabet", _file(d / "chars.txt", "a\n".encode() + "가".encode("cp949"))]),
    ("alphabet-bad-json", "e.alphabet_json", lambda d: ["check", _nemo(d), "--alphabet", _file(d / "vocab.json", b"{nope")]),
    ("alphabet-json-shape", "e.alphabet_shape", lambda d: ["check", _nemo(d), "--alphabet", _file(d / "vocab.json", b"42")]),
    ("alphabet-empty", "e.alphabet_empty",
     lambda d: ["check", _nemo(d), "--alphabet", _file(d / "chars.txt", b"<pad>\n[UNK]\n")]),
    # --- thresholds that would flag every clip
    ("min-gt-max-seconds", "e.min_gt_max", lambda d: ["check", _nemo(d), "--min-seconds", "10", "--max-seconds", "5"]),
    ("min-gt-max-cps", "e.min_gt_max", lambda d: ["check", _nemo(d), "--min-cps", "50"]),
    # --- --write-clean
    ("clean-overwrite", "e.clean_overwrite", lambda d: ["check", str(_demo(d)), "--write-clean", str(_demo(d))]),
    ("clean-other-folder", "e.clean_same_folder",
     lambda d: ["check", str(_demo(d, "ljspeech")), "--write-clean", str(d / "x.csv")]),
    ("clean-target-is-folder", "e.clean_write",
     lambda d: ["check", str(_demo(d)), "--write-clean", str(_demo(d).parent / "audio")]),
    # --- demo
    ("demo-nonempty", "e.demo_nonempty", lambda d: ["demo", "--out", str(Path(_file(d / "mine.txt", b"x")).parent)]),
    ("demo-out-is-file", "e.demo_not_dir", lambda d: ["demo", "--out", _file(d / "out", b"x")]),
    ("demo-out-under-file", "e.demo_write", lambda d: ["demo", "--out", str(Path(_file(d / "f", b"x")) / "sub")]),
]


@pytest.fixture()
def keys_seen(monkeypatch):
    """Record the message key of every error the CLI prints (and still print it)."""
    seen: list[str] = []
    original = cli._fail

    def spy(lang, exc):
        seen.append(getattr(exc, "key", type(exc).__name__))
        return original(lang, exc)

    monkeypatch.setattr(cli, "_fail", spy)
    return seen


@pytest.mark.parametrize("lang", ["en", "ko"])
@pytest.mark.parametrize("case_id,key,build", CASES, ids=[c[0] for c in CASES])
def test_input_error_exits_2_in_chosen_language(tmp_path, capsys, keys_seen, lang, case_id, key, build):
    argv = build(tmp_path)
    capsys.readouterr()  # drop anything the setup printed
    status = cli.main(["--lang", lang, *argv])  # an uncaught exception would fail the test here
    out, err = capsys.readouterr()
    assert status == 2, (case_id, err)
    assert keys_seen == [key], err
    assert "Traceback" not in err
    lines = err.strip().splitlines()
    expected_lines = 2 if key in cli.TIPS else 1  # a tip only when it adds a concrete command
    assert len(lines) == expected_lines, err
    if lang == "ko":
        assert lines[0].startswith("오류: ")
        assert HANGUL.search(err)
        assert not english_fragments(err.replace(str(tmp_path), "<tmp>")), err
    else:
        assert lines[0].startswith("error: ")
        assert not HANGUL.search(err.replace("안녕", "")), err
    if argv[0] == "demo":
        assert out == "", "the demo banner is printed only after the data was written"


@pytest.mark.parametrize("lang,expected", [("en", "alphabet file not found"), ("ko", "알파벳 파일을 찾을 수 없습니다")])
def test_qa_repro_wrong_alphabet_path_in_a_real_process(tmp_path, lang, expected):
    """QA Q-7 repro: a wrong --alphabet path used to print a KeyError traceback and exit 1."""
    manifest = _nemo(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "speechset_lint", "--lang", lang, "check", manifest, "--alphabet", "nope.json"],
        capture_output=True, text=True, encoding="utf-8", cwd=tmp_path, timeout=120,
    )
    assert result.returncode == 2, result.stderr
    assert "Traceback" not in result.stderr and "KeyError" not in result.stderr
    assert expected in result.stderr


NUMBER_ARGS = [
    ("--extra-tokens", "-5"),
    ("--extra-tokens", "1.5"),
    ("--max-seconds", "-1"),
    ("--min-seconds", "nan"),
    ("--max-cps", "inf"),
    ("--min-cps", "abc"),
    ("--sample-rate", "0"),
    ("--sample-rate", "abc"),
]


@pytest.mark.parametrize("lang", ["en", "ko"])
@pytest.mark.parametrize("flag,value", NUMBER_ARGS)
def test_bad_numbers_are_rejected_before_checking(tmp_path, capsys, lang, flag, value):
    """QA Q-9b: negative limits used to be accepted and flagged every clip."""
    with pytest.raises(SystemExit) as info:
        cli.main(["--lang", lang, "check", _nemo(tmp_path), flag, value])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert flag in err and value in err and "Traceback" not in err
    if lang == "ko":
        assert "인자" in err and "잘못된 값" in err
        message = err.splitlines()[-1].split("오류: ", 1)[1]  # after "speechset-lint check: 오류: "
        assert not english_fragments(message), err
    else:
        assert "invalid value" in err


def test_valid_numbers_still_pass(tmp_path, capsys):
    status = cli.main(["check", _nemo(tmp_path), "--extra-tokens", "0", "--max-seconds", "25", "--sample-rate", "16000"])
    assert status == 1  # the audio file is missing, so the report has an error; no input error
    assert "error:" not in capsys.readouterr().err


def test_empty_manifest_does_not_suggest_format(tmp_path, capsys):
    """QA Q-8: an empty file cannot be fixed with --format, so the message does not suggest it."""
    path = _file(tmp_path / "e.jsonl", b"")
    for extra in ([], ["--format", "nemo"], ["--format", "audiofolder"]):
        assert cli.main(["check", path, *extra]) == 2
        err = capsys.readouterr().err
        assert "empty" in err and "--format" not in err and "JSON object" not in err


def test_tokenizer_path_error_has_no_install_tip(tmp_path, capsys):
    """QA Q-8: a wrong --tokenizer path must not suggest pip install."""
    assert cli.main(["--lang", "ko", "check", _nemo(tmp_path), "--tokenizer", "nope.json"]) == 2
    err = capsys.readouterr().err
    assert "tokenizer.json" in err and "pip install" not in err


def test_nemo_text_field_lists_found_keys(tmp_path, capsys):
    """QA Q-9a: a missing NeMo --text-field is an input error, not 11 empty_text issues."""
    assert cli.main(["check", _nemo(tmp_path), "--text-field", "zzz"]) == 2
    err = capsys.readouterr().err
    assert "'zzz'" in err and "audio_filepath, text" in err


def test_korean_invalid_json_shows_column_not_parser_text(tmp_path, capsys):
    """QA Q-9c: Korean output gives the column instead of Python's English reason."""
    path = _file(tmp_path / "j.jsonl", b'{"audio_filepath": "a.wav", "text": "x"}\n{"audio_filepath" 1}\n')
    assert cli.main(["--lang", "ko", "check", path]) == 2
    err = capsys.readouterr().err
    assert "2번 줄 19번째 글자" in err and "Expecting" not in err
    assert cli.main(["--lang", "en", "check", path]) == 2
    assert "line 2, column 19: invalid JSON (Expecting ':' delimiter)" in capsys.readouterr().err


def test_clean_tips_give_a_command_to_copy(tmp_path, capsys):
    manifest = _demo(tmp_path)
    assert cli.main(["check", str(manifest), "--write-clean", str(manifest)]) == 2
    err = capsys.readouterr().err
    assert f"--write-clean {manifest.parent / 'clean.jsonl'}" in err


# --- the same errors through the library API -------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,key",
    [
        ({"alphabet": "nope.json"}, "e.alphabet_not_found"),
        ({"extra_tokens": -1}, "e.not_negative"),
        ({"max_seconds": float("nan")}, "e.not_negative"),
        ({"min_cps": -0.5}, "e.not_negative"),
        ({"sample_rate": 0}, "e.not_positive"),
        ({"min_seconds": 40.0}, "e.min_gt_max"),
        ({"preset": "nope"}, "e.unknown_preset"),
        ({"fmt": "nope"}, "e.unknown_format"),
        ({"text_field": "zzz"}, "e.text_field_not_key"),
    ],
)
def test_library_raises_speechset_errors_with_keys(tmp_path, kwargs, key):
    with pytest.raises(SpeechsetLintError) as info:
        check(_nemo(tmp_path), **kwargs)
    assert info.value.key == key
    assert str(info.value)  # the English text renders (no KeyError)


def test_library_choice_errors_have_keys(tmp_path):
    report = check(_nemo(tmp_path))
    with pytest.raises(SpeechsetLintError) as info:
        report.has_errors("fatal")
    assert info.value.key == "e.bad_choice"
    with pytest.raises(SpeechsetLintError) as info:
        write_clean(report, tmp_path / "clean.jsonl", drop="all")
    assert info.value.key == "e.bad_choice"


def test_unknown_message_key_fails_loudly():
    """A raise site with a sentence instead of a key must fail in tests, not hide (see test_message_keys)."""
    with pytest.raises(KeyError):
        SpeechsetLintError("cannot read alphabet file x: No such file or directory")


def test_alphabet_with_bom_and_windows_line_endings(tmp_path):
    path = tmp_path / "chars.txt"
    path.write_bytes("﻿a\r\nb\r\n|\r\n".encode())
    assert load_alphabet(path) == frozenset({"a", "b", " "})
    vocab = tmp_path / "vocab.json"
    vocab.write_bytes(b"\xef\xbb\xbf" + json.dumps({"a": 0, "|": 1, "<pad>": 2}).encode())
    assert load_alphabet(vocab) == frozenset({"a", " "})
