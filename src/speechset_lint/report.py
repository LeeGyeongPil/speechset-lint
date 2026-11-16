"""Human-readable table and JSON rendering of a Report."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from speechset_lint.checks import Issue, Report
from speechset_lint.i18n import t

AUDIO_WIDTH = 34


def render_detail(lang: str, issue: Issue) -> str:
    """Localized one-line description of an issue (the JSON detail stays raw)."""
    values: dict[str, Any] = dict(issue.detail)
    if issue.code == "missing_file":
        exts = values.get("found_ext")
        values["hint"] = t(lang, "hint.found_ext", exts=", ".join(exts)) if exts else ""
    if "ratio" in values:
        values["percent"] = values["ratio"] * 100
    if issue.code in ("clipping", "mostly_silent"):
        values["limit_percent"] = values["limit"] * 100
    for key, value in list(values.items()):
        if isinstance(value, list):
            values[key] = " ".join(str(v) for v in value)
    return t(lang, f"detail.{issue.code}", **values)


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3):]


def render_table(report: Report, lang: str, *, explain: bool = False) -> str:
    """Table of rows with issues, a summary and the result line."""
    out: list[str] = []
    out.append(
        t(
            lang,
            "out.header",
            manifest=report.manifest,
            fmt=report.fmt,
            preset=report.preset.name,
            signal=t(lang, "out.signal_on" if report.signal else "out.signal_off"),
        )
    )
    if report.issues:
        headers = [t(lang, f"col.{k}") for k in ("line", "audio", "severity", "code", "detail")]
        body = [
            [
                str(i.line_no),
                _shorten(i.audio_path, AUDIO_WIDTH),
                t(lang, f"sev.{i.severity}"),
                i.code,
                render_detail(lang, i),
            ]
            for i in report.issues
        ]
        widths = [max(_width(h), *(_width(r[c]) for r in body)) for c, h in enumerate(headers[:-1])]
        out.append("  ".join(_pad(h, w) for h, w in zip(headers[:-1], widths)) + "  " + headers[-1])
        out.append("  ".join("-" * w for w in widths) + "  " + "-" * 6)
        for issue, cells in zip(report.issues, body):
            out.append("  ".join(_pad(c, w) for c, w in zip(cells[:-1], widths)) + "  " + cells[-1])
            if explain:
                out.append(" " * (widths[0] + 2) + t(lang, "out.why", why=t(lang, f"why.{issue.code}")))
    else:
        out.append(t(lang, "out.no_issues"))
    s = report.summary
    out.append("")
    out.append(t(lang, "out.summary", rows=s["rows"], errors=s["errors"], warnings=s["warnings"], seconds=s["seconds"]))
    if s["by_code"]:
        out.append(t(lang, "out.by_code", codes=", ".join(f"{k}={v}" for k, v in s["by_code"].items())))
    return "\n".join(out)


def render_json(report: Report) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
