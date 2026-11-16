"""Input errors: translated messages (Q-1), NUL/UTF-16 files (Q-2), line numbers (Q-3), demo safety (Q-4), tips (Q-5)."""

from __future__ import annotations

import re

import pytest

from speechset_lint.cli import main
from speechset_lint.demo import write_manifests
from speechset_lint.errors import ManifestError, SpeechsetLintError
from speechset_lint.i18n import MESSAGES
from speechset_lint.manifest import load_manifest

# a Latin-letter word of 4+ letters is an English sentence fragment (Korean text has none here)
ENGLISH_WORD = re.compile(r"(?<![\w-])(?!(?:jsonl|nemo|audiofolder|ljspeech|whisper|speechset|lint|check|demo|json|csv|utf|nul|file_name|text|tokenizers|tokenizer|path|force|format|preset|manifest|clean|wav|pip|install|error|warning|auto|ctc|null|bom)\b)[A-Za-z]{4,}\b", re.I)


def _run(capsys, *args):
    status = main(list(args))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def _korean_only(text: str) -> None:
    cleaned = re.sub(r"https?://\S+|--[\w-]+|'[^']*'|`[^`]*`|\S*[/\\.]\S*", " ", text)
    assert not ENGLISH_WORD.findall(cleaned), text


def test_every_error_key_has_both_languages():
    for key in (k for k in MESSAGES["en"] if k.startswith("e.")):
        assert key in MESSAGES["ko"]


def test_exception_str_is_english_and_carries_key():
    with pytest.raises(ManifestError) as info:
        load_manifest("does-not-exist.jsonl")
    assert info.value.key == "e.not_found"
    assert "manifest not found" in str(info.value)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda d: (d / "nope.jsonl", ()), id="missing"),
        pytest.param(lambda d: (_write(d / "e.jsonl", b""), ()), id="empty"),
        pytest.param(lambda d: (_write(d / "j.jsonl", b'{"audio_filepath": "a.wav"\n'), ()), id="bad-json"),
        pytest.param(lambda d: (_write(d / "u.jsonl", b'{"audio_filepath": "a.wav", "text": "\xff"}\n'), ("--format", "nemo")), id="utf8"),
        pytest.param(lambda d: (_write(d / "x.jsonl", b'{"nope": 1}\n'), ()), id="unknown-keys"),
    ],
)
def test_korean_input_errors_have_no_english_sentence(tmp_path, capsys, build):
    manifest, extra = build(tmp_path)
    status, _, err = _run(capsys, "--lang", "ko", "check", str(manifest), *extra)
    assert status == 2
    assert err.startswith("오류")
    _korean_only(err.splitlines()[0].split(str(tmp_path))[-1])


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


def test_korean_message_matches_english_key(tmp_path, capsys):
    status, _, err_en = _run(capsys, "--lang", "en", "check", str(tmp_path / "zzz.jsonl"))
    _, _, err_ko = _run(capsys, "--lang", "ko", "check", str(tmp_path / "zzz.jsonl"))
    assert status == 2
    assert "manifest not found" in err_en
    assert "찾을 수 없습니다" in err_ko and "manifest not found" not in err_ko


def test_korean_argparse_errors_are_translated(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--lang", "ko", "check"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "사용법" in err and "필요합니다" in err and "required" not in err
    with pytest.raises(SystemExit):
        main(["--lang", "ko", "check", "m.jsonl", "--preset", "nope"])
    err = capsys.readouterr().err
    assert "잘못된 선택지" in err and "invalid choice" not in err
    with pytest.raises(SystemExit):
        main(["--lang", "ko", "check", "m.jsonl", "--min-seconds", "abc"])
    err = capsys.readouterr().err
    assert "잘못된 값" in err and "invalid" not in err


def test_english_argparse_errors_stay_english(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--lang", "en", "check"])
    assert info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_nul_byte_first_line_exits_2_without_traceback(tmp_path, capsys):
    path = _write(tmp_path / "lj.csv", b"ok|a\x00b\n")
    status, _, err = _run(capsys, "--lang", "en", "check", str(path))
    assert status == 2
    assert "NUL" in err and "UTF-8" in err and "Traceback" not in err
    status, _, err = _run(capsys, "--lang", "ko", "check", str(path))
    assert status == 2 and "NUL" in err and "다시 저장" in err


def test_nul_byte_on_later_line_names_that_line(tmp_path):
    path = _write(tmp_path / "m.csv", b"file_name,text\na.wav,ok\nb.wav,b\x00d\n")
    with pytest.raises(ManifestError, match="line 3 contains NUL"):
        load_manifest(path)


@pytest.mark.parametrize("bom", [b"\xff\xfe", b"\xfe\xff"])
def test_utf16_file_exits_2_with_hint(tmp_path, capsys, bom):
    path = _write(tmp_path / "bad.jsonl", bom + '{"audio_filepath": "a.wav"}\n'.encode("utf-16-le"))
    status, _, err = _run(capsys, "check", str(path))
    assert status == 2
    assert "UTF-16" in err and "Traceback" not in err


def test_utf8_error_reports_the_real_line(tmp_path):
    path = _write(tmp_path / "metadata.csv", "file_name,text\nA.wav,안녕\nb.wav,".encode() + "안녕".encode("cp949") + b"\n")
    with pytest.raises(ManifestError, match=r"line 3: not valid UTF-8 \(column 7\)"):
        load_manifest(path)


def test_demo_refuses_non_empty_folder_and_keeps_files(tmp_path, capsys):
    (tmp_path / "_audio").mkdir()
    keep = tmp_path / "_audio" / "mine.wav"
    keep.write_bytes(b"precious")
    (tmp_path / "nemo").mkdir()
    (tmp_path / "nemo" / "manifest.jsonl").write_text("mine\n", encoding="utf-8")
    status, _, err = _run(capsys, "--lang", "en", "demo", "--out", str(tmp_path))
    assert status == 2 and "refusing" in err and "--force" in err
    assert keep.read_bytes() == b"precious"
    assert (tmp_path / "nemo" / "manifest.jsonl").read_text(encoding="utf-8") == "mine\n"
    with pytest.raises(SpeechsetLintError):
        write_manifests(tmp_path)


def test_demo_force_and_rerun_in_demo_folder(tmp_path, capsys):
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")
    assert _run(capsys, "demo", "--out", str(tmp_path), "--force")[0] == 0
    assert _run(capsys, "demo", "--out", str(tmp_path))[0] == 0  # marker file: safe to rerun
    assert (tmp_path / "other.txt").exists()
    assert not (tmp_path / "_audio").exists()  # scratch lives outside the output folder


def test_overwrite_tip_is_specific(tmp_path, capsys):
    out = tmp_path / "demo"
    write_manifests(out)
    manifest = out / "nemo" / "manifest.jsonl"
    status, _, err = _run(capsys, "--lang", "en", "check", str(manifest), "--write-clean", str(manifest))
    assert status == 2
    assert "different file name" in err and "next to the manifest" not in err
    status, _, err = _run(capsys, "--lang", "en", "check", str(out / "ljspeech" / "metadata.csv"),
                          "--write-clean", str(tmp_path / "x.csv"))
    assert status == 2 and "next to the manifest" in err
