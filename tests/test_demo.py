from __future__ import annotations

from speechset_lint import check
from speechset_lint.cli import main
from speechset_lint.demo import CLIPS, DEMO_CODES


def test_demo_reproduces_every_issue_kind(tmp_path, capsys):
    status = main(["--lang", "en", "demo", "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert status == 0
    assert out.splitlines()[0].startswith("synthetic practice data")
    for code in DEMO_CODES:
        assert code in out, code


def test_demo_clips_raise_exactly_the_expected_codes(demo_manifests):
    report = check(demo_manifests["nemo"], preset="whisper", signal=True)
    seen: dict[str, set[str]] = {c.name: set() for c in CLIPS}
    for issue in report.issues:
        seen[issue.audio_path.split("/")[-1]].add(issue.code)
    for clip in CLIPS:
        assert seen[clip.name] == set(clip.expect), clip.name


def test_demo_korean_banner_still_says_synthetic(tmp_path, capsys):
    assert main(["--lang", "ko", "demo", "--out", str(tmp_path / "out")]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert "synthetic" in first and "합성" in first
