from __future__ import annotations

import pytest

from helpers import tone, write_nemo, write_wav
from speechset_lint import check, write_clean
from speechset_lint.cli import main
from speechset_lint.errors import CleanError


def _lines(path):
    return path.read_bytes().splitlines(keepends=True)


@pytest.mark.parametrize("fmt", ["nemo", "audiofolder", "ljspeech"])
def test_write_clean_roundtrip_all_formats(demo_manifests, capsys, fmt):
    source = demo_manifests[fmt]
    out = source.parent / f"clean{source.suffix}"
    base = ["--lang", "en", "check", str(source), "--signal"]
    assert main(base + ["--write-clean", str(out)]) == 1  # the original has errors
    capsys.readouterr()

    original = _lines(source)
    kept = _lines(out)
    assert 0 < len(kept) < len(original)
    for line in kept:
        assert line in original  # byte-identical survivors
    if fmt == "audiofolder":
        assert kept[0] == original[0]  # header kept
    # the three error rows are gone: too_long, mostly_silent, missing file (plus flac for ljspeech)
    assert not any(b"long35" in ln or b"silent" in ln or b"missing" in ln for ln in kept)
    # the cleaned file passes
    assert main(["--lang", "en", "check", str(out), "--signal"]) == 0


@pytest.mark.parametrize("fmt", ["nemo", "audiofolder", "ljspeech"])
def test_drop_warning_removes_warning_rows(demo_manifests, fmt):
    report = check(demo_manifests[fmt], signal=True)
    out = demo_manifests[fmt].parent / "strict.txt"
    kept = write_clean(report, out, drop="warning")
    assert kept == len(report.clean_rows("warning")) < len(report.clean_rows("error"))
    assert main(["--lang", "en", "check", str(out), "--signal", "--fail-on", "warning", "--format", fmt]) == 0


def test_write_clean_other_dir_rejected(demo_manifests, tmp_path, capsys):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    status = main(["--lang", "en", "check", str(demo_manifests["audiofolder"]), "--write-clean", str(elsewhere / "m.csv")])
    assert status == 2
    err = capsys.readouterr().err
    assert "same folder" in err
    assert not (elsewhere / "m.csv").exists()
    with pytest.raises(CleanError):
        write_clean(check(demo_manifests["ljspeech"]), elsewhere / "m.csv")


def test_write_clean_refuses_to_overwrite_input(demo_manifests):
    with pytest.raises(CleanError, match="overwrite"):
        write_clean(check(demo_manifests["nemo"]), demo_manifests["nemo"])


def test_nemo_with_absolute_paths_may_go_anywhere(tmp_path):
    good = write_wav(tmp_path / "good.wav", tone(2.0))
    manifest = write_nemo(tmp_path / "m.jsonl", [(str(good), "hello world this is a test", None), (str(tmp_path / "gone.wav"), "x y z", None)])
    other = tmp_path / "sub"
    other.mkdir()
    assert write_clean(check(manifest), other / "clean.jsonl") == 1
