from __future__ import annotations

import json

from helpers import tone, write_nemo, write_wav
from speechset_lint.cli import main


def _run(capsys, *argv):
    status = main(["--lang", "en", *argv])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_json_shape_and_exit_codes(tmp_path, capsys, demo_manifests):
    # errors present -> exit 1, JSON shape as documented
    status, out, _ = _run(capsys, "check", str(demo_manifests["nemo"]), "--json", "--signal")
    assert status == 1
    data = json.loads(out)
    assert data["schema_version"] == 1
    assert set(data["summary"]) >= {"rows", "errors", "warnings", "by_code"}
    assert data["summary"]["rows"] == 11
    assert data["summary"]["errors"] == 3
    assert data["summary"]["by_code"]["too_long"] == 1
    assert set(data["issues"][0]) == {"line", "audio", "code", "severity", "detail"}
    assert data["issues"][0]["line"] >= 1

    # warnings only: exit 0 by default, 1 with --fail-on warning
    write_wav(tmp_path / "ok.wav", tone(2.0))
    write_wav(tmp_path / "hi.wav", tone(0.2, freq=900))
    warn_only = write_nemo(tmp_path / "w.jsonl", [("ok.wav", "hello world this is a test", None), ("hi.wav", "a very short clip", None)])
    status, out, _ = _run(capsys, "check", str(warn_only), "--json")
    assert status == 0
    assert json.loads(out)["summary"] == {**json.loads(out)["summary"], "errors": 0, "warnings": 1}
    assert _run(capsys, "check", str(warn_only), "--fail-on", "warning")[0] == 1

    # fully clean -> 0 even with --fail-on warning
    clean = write_nemo(tmp_path / "c.jsonl", [("ok.wav", "hello world this is a test", None)])
    assert _run(capsys, "check", str(clean), "--fail-on", "warning")[0] == 0

    # missing manifest -> 2 with a fix hint
    status, _, err = _run(capsys, "check", str(tmp_path / "nope.jsonl"))
    assert status == 2
    assert "manifest not found" in err


def test_table_output_names_line_and_code(capsys, demo_manifests):
    status, out, _ = _run(capsys, "check", str(demo_manifests["nemo"]), "--signal", "--explain")
    assert status == 1
    assert "long35.wav" in out and "too_long" in out
    assert "result: FAIL (exit 1)" in out
    assert "-> Whisper silently cuts audio after 30 s" in out


def test_option_overrides_change_thresholds(tmp_path, capsys):
    write_wav(tmp_path / "a.wav", tone(2.0))
    manifest = write_nemo(tmp_path / "m.jsonl", [("a.wav", "hello world this is a test", None)])
    assert _run(capsys, "check", str(manifest), "--max-seconds", "1.5", "--json")[0] == 1
    assert _run(capsys, "check", str(manifest), "--sample-rate", "8000", "--fail-on", "warning")[0] == 1
    assert _run(capsys, "check", str(manifest), "--max-cps", "5", "--fail-on", "warning")[0] == 1


def test_tokenizer_without_file_exits_2(tmp_path, capsys):
    write_wav(tmp_path / "a.wav", tone(2.0))
    manifest = write_nemo(tmp_path / "m.jsonl", [("a.wav", "hello", None)])
    status, _, err = _run(capsys, "check", str(manifest), "--tokenizer", str(tmp_path / "missing.json"))
    assert status == 2
    assert "tokenizer file not found" in err
