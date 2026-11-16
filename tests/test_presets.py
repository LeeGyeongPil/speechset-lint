from __future__ import annotations

import numpy as np

from helpers import codes_of, tone, write_nemo, write_wav
from speechset_lint import check
from speechset_lint.presets import ctc_labels_needed, wav2vec2_frames


def _issue(report, code):
    return next((i for i in report.issues if i.code == code), None)


def test_whisper_30s_boundary(tmp_path):
    write_wav(tmp_path / "exact.wav", np.zeros(480000, dtype=np.float32))
    write_wav(tmp_path / "over.wav", np.zeros(480160, dtype=np.float32))  # 30.01 s
    manifest = write_nemo(
        tmp_path / "m.jsonl",
        [("exact.wav", "a normal sentence spoken slowly " * 3, None), ("over.wav", "a normal sentence spoken slowly " * 3, None)],
    )
    report = check(manifest, preset="whisper")
    assert "too_long" not in codes_of(report, "exact.wav")
    issue = _issue(report, "too_long")
    assert issue is not None and issue.audio_path == "over.wav"
    assert issue.severity == "error"
    assert issue.detail["limit"] == 30.0


def test_whisper_label_448_vs_449(tmp_path, tokenizer_json):
    write_wav(tmp_path / "a.wav", tone(20.0))
    # the fixture tokenizer adds 3 special tokens, --extra-tokens defaults to 2: n words -> n + 5
    at_limit = " ".join(["w"] * 443)  # 446 + 2 = 448
    over = " ".join(["w"] * 444)  # 447 + 2 = 449
    manifest = write_nemo(tmp_path / "m.jsonl", [("a.wav", at_limit, None), ("a.wav", over, None)])
    report = check(manifest, tokenizer=tokenizer_json, dedupe=False)
    over_issues = [i for i in report.issues if i.code == "label_tokens_over"]
    assert [i.line_no for i in over_issues] == [2]
    assert over_issues[0].severity == "error"
    assert over_issues[0].detail == {"tokens": 447, "extra": 2, "limit": 448}
    assert not any(i.code == "label_tokens_maybe_over" for i in report.issues)
    # a different extra-tokens setting moves the boundary
    relaxed = check(manifest, tokenizer=tokenizer_json, extra_tokens=1, dedupe=False)
    assert not any(i.code == "label_tokens_over" for i in relaxed.issues)


def test_byte_upper_bound_without_tokenizer(tmp_path):
    write_wav(tmp_path / "a.wav", tone(20.0))
    manifest = write_nemo(
        tmp_path / "m.jsonl",
        [("a.wav", "a" * 446, None), ("a.wav", "a" * 447, None), ("a.wav", "가" * 150, None)],  # 446+2, 447+2, 450+2
    )
    report = check(manifest, dedupe=False)
    flagged = {i.line_no: i for i in report.issues if i.code == "label_tokens_maybe_over"}
    assert sorted(flagged) == [2, 3]
    assert flagged[2].severity == "warning"
    assert flagged[3].detail["tokens"] == 450  # 150 Hangul syllables are 3 bytes each
    assert not any(i.code == "label_tokens_over" for i in report.issues)


def test_ctc_frame_count_and_alignability(tmp_path):
    assert wav2vec2_frames(16000) == 49
    assert wav2vec2_frames(3200) == 9
    assert wav2vec2_frames(5) == 0
    assert ctc_labels_needed("hello") == 6  # "ll" needs a blank between

    write_wav(tmp_path / "one.wav", tone(1.0))
    write_wav(tmp_path / "short.wav", tone(0.2, freq=300))
    fits = "ab" * 24 + "c"  # 49 labels, no repeats
    too_many = "ab" * 25  # 50 labels
    repeats = "aabb" + "cd" * 22  # 48 characters + 2 repeats = 50 labels
    manifest = write_nemo(
        tmp_path / "m.jsonl",
        [
            ("one.wav", fits, None),
            ("one.wav", too_many, None),
            ("one.wav", repeats, None),
            ("short.wav", "ab" * 29, None),
        ],
    )
    report = check(manifest, preset="ctc", dedupe=False)
    bad = {i.line_no: i for i in report.issues if i.code == "ctc_unalignable"}
    assert sorted(bad) == [2, 3, 4]
    assert bad[2].detail == {"frames": 49, "labels": 50}
    assert bad[3].detail == {"frames": 49, "labels": 50}
    assert bad[4].detail == {"frames": 9, "labels": 58}
    assert bad[4].severity == "error"
    # ctc_unalignable is a ctc-only check
    assert not any(i.code == "ctc_unalignable" for i in check(manifest, preset="whisper", dedupe=False).issues)
