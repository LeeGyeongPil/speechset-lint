from __future__ import annotations

import numpy as np

from helpers import codes_of, tone, write_nemo, write_wav
from speechset_lint import check


def test_tts_preset_signal_checks(tmp_path):
    rate = 22050
    write_wav(tmp_path / "ok.wav", tone(2.0, rate), rate)
    write_wav(tmp_path / "rate16k.wav", tone(2.0, 16000, freq=500), 16000)
    stereo = np.stack([tone(2.0, rate, 300), tone(2.0, rate, 310)], axis=1)
    write_wav(tmp_path / "stereo.wav", stereo, rate)
    write_wav(tmp_path / "silent.wav", np.zeros(2 * rate, dtype=np.float32), rate)
    write_wav(tmp_path / "clipped.wav", np.clip(tone(2.0, rate, 600, amp=1.5), -1, 1), rate)
    text = "this sentence is fine here"  # 22 characters over 2 s: a normal speaking rate
    manifest = write_nemo(
        tmp_path / "m.jsonl",
        [(n, text, None) for n in ("ok.wav", "rate16k.wav", "stereo.wav", "silent.wav", "clipped.wav")],
    )
    report = check(manifest, preset="tts")  # signal defaults to on for tts
    assert report.signal is True
    severity = {(i.audio_path, i.code): i.severity for i in report.issues}
    assert codes_of(report, "ok.wav") == set()
    assert severity[("rate16k.wav", "sample_rate_mismatch")] == "error"
    assert severity[("stereo.wav", "not_mono")] == "error"
    assert severity[("silent.wav", "mostly_silent")] == "error"
    assert severity[("clipped.wav", "clipping")] == "warning"
    # silence is invisible to the characters-per-second check: that is why the signal check exists
    assert codes_of(report, "silent.wav") == {"mostly_silent"}
    # the same dataset passes the signal checks when they are switched off
    off = check(manifest, preset="tts", signal=False)
    assert off.signal is False
    assert codes_of(off, "silent.wav") == set()
    assert codes_of(off, "clipped.wav") == set()


def test_signal_thresholds_are_inclusive(tmp_path):
    rate = 16000
    data = tone(1.0, rate)
    data[:15] = 1.0  # 15 of 16000 samples = 0.094% < 0.1%
    write_wav(tmp_path / "few.wav", data, rate)
    data = tone(1.0, rate)
    data[:17] = 1.0  # 0.106% >= 0.1%
    write_wav(tmp_path / "many.wav", data, rate)
    manifest = write_nemo(tmp_path / "m.jsonl", [("few.wav", "short test text", None), ("many.wav", "short test text", None)])
    report = check(manifest, signal=True, dedupe=False)
    assert codes_of(report, "few.wav") == set()
    assert codes_of(report, "many.wav") == {"clipping"}
