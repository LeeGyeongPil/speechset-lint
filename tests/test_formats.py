from __future__ import annotations

import csv
import json

from helpers import codes_by_audio
from speechset_lint import check, load_manifest
from speechset_lint.cli import main
from speechset_lint.demo import CLIPS, write_manifests


def test_three_formats_same_issue_sets(tmp_path):
    clips = tuple(c for c in CLIPS if not c.name.endswith(".flac"))  # LJSpeech needs one extension
    paths = write_manifests(tmp_path / "d", clips)
    # a fourth variant: audiofolder metadata.jsonl next to the CSV's audio folder
    jsonl = paths["audiofolder"].parent / "metadata.jsonl"
    with open(paths["audiofolder"], newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    reports = {
        "nemo": check(paths["nemo"], signal=True),
        "audiofolder-csv": check(paths["audiofolder"], signal=True),
        "audiofolder-jsonl": check(jsonl, signal=True),
        "ljspeech": check(paths["ljspeech"], signal=True),
    }
    assert reports["nemo"].fmt == "nemo"
    assert reports["audiofolder-csv"].fmt == reports["audiofolder-jsonl"].fmt == "audiofolder"
    assert reports["ljspeech"].fmt == "ljspeech"
    expected = codes_by_audio(reports["nemo"])
    assert expected["long35.wav"] == {"too_long", "char_rate_low"}
    for name, report in reports.items():
        assert codes_by_audio(report) == expected, name


def test_audiofolder_file_name_is_relative_to_metadata_folder(tmp_path):
    paths = write_manifests(tmp_path / "d")
    rows = load_manifest(paths["audiofolder"])
    assert rows[0].path == paths["audiofolder"].parent / "audio" / "ok.wav"
    assert rows[0].path.is_file()


def test_ljspeech_audio_ext_hint(tmp_path):
    paths = write_manifests(tmp_path / "d")
    wav = check(paths["ljspeech"])  # default --audio-ext wav
    flac_issue = next(i for i in wav.issues if i.audio_path.endswith("ok_flac.wav"))
    assert flac_issue.code == "missing_file"
    assert flac_issue.detail["found_ext"] == [".flac"]

    flac = check(paths["ljspeech"], audio_ext="flac")
    by_name = codes_by_audio(flac)
    assert "missing_file" not in by_name["ok_flac.flac"]
    wav_issue = next(i for i in flac.issues if i.audio_path == "wavs/ok.flac" and i.code == "missing_file")
    assert wav_issue.detail["found_ext"] == [".wav"]


def test_unknown_format_exit_2(tmp_path, capsys):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello world, this is not a manifest\n", encoding="utf-8")
    assert main(["--lang", "en", "check", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "--format" in err


def test_bad_json_line_reports_line_number(tmp_path, capsys):
    bad = tmp_path / "m.jsonl"
    bad.write_text('{"audio_filepath": "a.wav", "text": "x"}\n{broken\n', encoding="utf-8")
    assert main(["--lang", "en", "check", str(bad)]) == 2
    assert "line 2" in capsys.readouterr().err
