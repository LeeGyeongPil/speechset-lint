from __future__ import annotations

import wave

import pytest

from helpers import codes_of, tone, write_nemo, write_wav
from speechset_lint import check
from speechset_lint.audio import UnreadableAudio, read_header


def test_header_wave_then_soundfile_fallback(tmp_path, monkeypatch):
    pcm = write_wav(tmp_path / "pcm.wav", tone(2.0))
    flt = write_wav(tmp_path / "float.wav", tone(1.5, 22050), 22050, subtype="FLOAT")
    flac = write_wav(tmp_path / "a.flac", tone(0.5), 16000)

    info = read_header(pcm)
    assert (info.via, info.sample_rate, info.channels, info.frames) == ("wave", 16000, 1, 32000)
    assert info.duration == 2.0

    # float WAV: Python < 3.15 refuses it in `wave`, newer versions read it; same numbers either way
    info = read_header(flt)
    assert (info.sample_rate, info.channels, info.frames) == (22050, 1, 33075)
    assert info.duration == pytest.approx(1.5)

    # force the fallback: when `wave` fails, soundfile must give identical numbers
    def refuse(*args, **kwargs):
        raise wave.Error("unknown format: 3")

    monkeypatch.setattr(wave, "open", refuse)
    for path, rate, frames in ((flt, 22050, 33075), (pcm, 16000, 32000)):
        info = read_header(path)
        assert (info.via, info.sample_rate, info.channels, info.frames) == ("soundfile", rate, 1, frames)
    monkeypatch.undo()

    info = read_header(flac)
    assert (info.via, info.sample_rate, info.channels, info.frames) == ("soundfile", 16000, 1, 8000)


def test_corrupt_file_is_issue_not_crash(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"this is definitely not audio" * 10)
    (tmp_path / "empty.wav").write_bytes(b"")
    write_wav(tmp_path / "good.wav", tone(2.0))
    manifest = write_nemo(
        tmp_path / "m.jsonl",
        [
            ("bad.wav", "some text here", None),
            ("empty.wav", "other text here", None),
            ("good.wav", "hello world again", None),
            ("gone.wav", "never written", None),
        ],
    )
    with pytest.raises(UnreadableAudio):
        read_header(tmp_path / "bad.wav")
    report = check(manifest)
    assert report.summary["rows"] == 4
    assert codes_of(report, "bad.wav") == {"unreadable_audio"}
    assert codes_of(report, "empty.wav") == {"unreadable_audio"}
    assert codes_of(report, "good.wav") == set()
    assert codes_of(report, "gone.wav") == {"missing_file"}  # later rows were still checked
