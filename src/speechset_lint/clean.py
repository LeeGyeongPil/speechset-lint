"""Write the filtered manifest in the same format, keeping surviving lines byte-exact."""

from __future__ import annotations

import os
from pathlib import Path

from speechset_lint.checks import Report
from speechset_lint.errors import CleanError, os_reason


def _needs_same_folder(report: Report, kept_audio: list[str]) -> bool:
    """Relative audio paths only stay valid next to the original manifest."""
    if report.fmt in ("audiofolder", "ljspeech"):
        return True
    return any(not os.path.isabs(p) for p in kept_audio)


def write_clean(report: Report, out_path: str | os.PathLike[str], drop: str = "error") -> int:
    """Write the manifest without failing rows to ``out_path``; return rows written."""
    target = Path(out_path)
    if report.manifest is None:
        raise CleanError("e.clean_no_manifest")
    rows = report.clean_rows(drop)
    # 팁에 그대로 붙여 쓸 수 있는 예시 경로: 원본 옆의 clean.<확장자>
    suggest = report.manifest.parent / f"clean{report.manifest.suffix}"
    if suggest.name == report.manifest.name:  # the input is already called clean.*
        suggest = report.manifest.parent / f"{report.manifest.stem}.clean{report.manifest.suffix}"
    if target.resolve() == report.manifest.resolve():
        raise CleanError("e.clean_overwrite", path=target, suggest=suggest)
    if _needs_same_folder(report, [r.audio_path for r in rows]):
        if target.resolve().parent != report.manifest.resolve().parent:
            raise CleanError(
                "e.clean_same_folder", path=target, fmt=report.fmt,
                folder=report.manifest.resolve().parent, suggest=suggest,
            )
    chunks: list[bytes] = [report.header] if report.header else []
    for index, row in enumerate(rows):
        raw = row.raw
        if not raw.endswith(b"\n") and index < len(rows) - 1:
            raw += b"\n"  # only the last line of a file can lack a terminator
        chunks.append(raw)
    try:
        target.write_bytes(b"".join(chunks))
    except OSError as exc:
        raise CleanError("e.clean_write", path=target, reason=os_reason(exc)) from exc
    return len(rows)
