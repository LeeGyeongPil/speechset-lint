from __future__ import annotations

import json
import shutil
import unicodedata

from helpers import codes_of, tone, write_nemo, write_wav
from speechset_lint import check


def test_alphabet_oov_and_nfc(tmp_path):
    write_wav(tmp_path / "a.wav", tone(2.0))
    write_wav(tmp_path / "b.wav", tone(2.0, freq=330))
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"<pad>": 0, "|": 1, "a": 2, "b": 3, "á": 4}), encoding="utf-8")
    decomposed = unicodedata.normalize("NFD", "áá")  # a + combining acute, twice
    manifest = write_nemo(
        tmp_path / "m.jsonl",
        [("a.wav", "ab! ba?", None), ("b.wav", decomposed, None)],
    )
    report = check(manifest, preset="ctc", alphabet=vocab)
    oov = next(i for i in report.issues if i.code == "oov_chars" and i.line_no == 1)
    assert oov.detail["chars"] == ["!", "?"]
    assert oov.severity == "error"  # ctc preset: an unknown character breaks the vocabulary
    assert "not_nfc" in codes_of(report, "b.wav")
    # the same text in NFC form is clean
    clean = write_nemo(tmp_path / "c.jsonl", [("a.wav", "áá ab", None)])
    assert check(clean, preset="ctc", alphabet=vocab).issues == []


def test_alphabet_text_file_and_literal_set(tmp_path):
    write_wav(tmp_path / "a.wav", tone(2.0))
    chars = tmp_path / "chars.txt"
    chars.write_text("a\nb\n|\n", encoding="utf-8")
    manifest = write_nemo(tmp_path / "m.jsonl", [("a.wav", "ab ba z", None)])
    for alphabet in (chars, set("ab ")):
        report = check(manifest, preset="whisper", alphabet=alphabet)
        assert [i.detail["chars"] for i in report.issues if i.code == "oov_chars"] == [["z"]]


def test_duplicates_repeated_path_duration_mismatch_empty_text(tmp_path):
    a = write_wav(tmp_path / "a.wav", tone(2.0))
    shutil.copyfile(a, tmp_path / "b.wav")  # same bytes, other path
    write_wav(tmp_path / "c.wav", tone(2.0, freq=700))
    manifest = write_nemo(
        tmp_path / "m.jsonl",
        [
            ("a.wav", "first sentence here", 2.0),
            ("b.wav", "copy of the first", 2.0),
            ("a.wav", "same path again", 2.0),
            ("c.wav", "duration is wrong", 2.11),
            ("c.wav", "", 2.0),
        ],
    )
    report = check(manifest)
    by_line = {}
    for issue in report.issues:
        by_line.setdefault(issue.line_no, {})[issue.code] = issue
    assert by_line[2]["duplicate_audio"].detail["first_line"] == 1
    assert by_line[3]["repeated_path"].detail["first_line"] == 1
    assert "duplicate_audio" not in by_line[3]  # a repeated path is not reported twice
    assert by_line[4]["duration_mismatch"].detail["diff"] == 0.11
    assert by_line[5]["empty_text"].severity == "error"
    assert 1 not in by_line
    # exactly 0.1 s apart is tolerated
    ok = write_nemo(tmp_path / "ok.jsonl", [("a.wav", "first sentence here", 2.1)])
    assert check(ok).issues == []


def test_no_dedupe_skips_hash_check(tmp_path):
    a = write_wav(tmp_path / "a.wav", tone(2.0))
    shutil.copyfile(a, tmp_path / "b.wav")
    manifest = write_nemo(tmp_path / "m.jsonl", [("a.wav", "first sentence here", None), ("b.wav", "copy of the first", None)])
    assert any(i.code == "duplicate_audio" for i in check(manifest).issues)
    assert not any(i.code == "duplicate_audio" for i in check(manifest, dedupe=False).issues)
