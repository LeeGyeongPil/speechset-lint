"""Command line interface: ``speechset-lint check|demo``.

Exit codes: 0 nothing at the failing level, 1 issues found, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Sequence

from speechset_lint import __version__
from speechset_lint.checks import check
from speechset_lint.clean import write_clean
from speechset_lint.demo import write_manifests
from speechset_lint.errors import SpeechsetLintError, os_reason
from speechset_lint.i18n import LANGS, resolve_lang, t
from speechset_lint.manifest import FORMATS
from speechset_lint.presets import PRESETS
from speechset_lint.report import render_json, render_table

EXIT_OK, EXIT_ISSUES, EXIT_USAGE = 0, 1, 2


def _prescan_lang(argv: Sequence[str]) -> str | None:
    """Find ``--lang`` before the real parser exists, so its help can be translated."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def _translate_argparse(lang: str, message: str) -> str:
    """Map argparse's English error text to a message key; unknown shapes keep the original detail."""
    if m := re.match(r"the following arguments are required: (.*)$", message):
        return t(lang, "e.arg_required", names=m.group(1))
    if m := re.match(r"argument (.+?): invalid choice: (.+?) \(choose from (.*)\)$", message):
        return t(lang, "e.arg_choice", name=m.group(1), value=m.group(2), choices=m.group(3))
    if m := re.match(r"argument (.+?): invalid \w+ value: (.*)$", message):
        return t(lang, "e.arg_value", name=m.group(1), value=m.group(2))
    if m := re.match(r"argument (.+?): expected one argument$", message):
        return t(lang, "e.arg_expected", name=m.group(1))
    if m := re.match(r"unrecognized arguments: (.*)$", message):
        return t(lang, "e.arg_unrecognized", names=m.group(1))
    if m := re.match(r"argument (.+?): (.*)$", message, re.S):
        # our own type checks (_number_type) already wrote the detail in `lang`
        return t(lang, "e.arg_detail", name=m.group(1), detail=m.group(2))
    return t(lang, "e.arg_other", detail=message)


def build_parser(lang: str) -> argparse.ArgumentParser:
    usage_prefix = t(lang, "help.usage") + ": "

    class Formatter(argparse.HelpFormatter):
        def add_usage(self, usage, actions, groups, prefix=None):  # type: ignore[override]
            return super().add_usage(usage, actions, groups, usage_prefix if prefix is None else prefix)

    class Parser(argparse.ArgumentParser):
        """Prints argparse's own errors (missing/invalid arguments) in the chosen language."""

        def error(self, message: str):  # type: ignore[override]
            self.print_usage(sys.stderr)
            self.exit(EXIT_USAGE, f"{self.prog}: {t(lang, 'err.prefix')}: {_translate_argparse(lang, message)}\n")

    def parser(**kw) -> argparse.ArgumentParser:
        p = Parser(formatter_class=Formatter, add_help=False, **kw)
        p._positionals.title = t(lang, "help.positionals")
        p._optionals.title = t(lang, "help.options")
        p.add_argument("-h", "--help", action="help", help=t(lang, "help.help"))
        return p

    root = parser(prog="speechset-lint", description=t(lang, "app.desc"), epilog=t(lang, "app.epilog"))
    root.add_argument("--lang", choices=LANGS, help=t(lang, "arg.lang"))
    root.add_argument("--version", action="version", version=f"speechset-lint {__version__}", help=t(lang, "arg.version"))
    sub = root.add_subparsers(dest="command", title=t(lang, "help.commands"), metavar="{check,demo}")
    sub.required = True

    c = sub.add_parser("check", help=t(lang, "cmd.check"), description=t(lang, "cmd.check"),
                       formatter_class=Formatter, add_help=False)
    c._positionals.title = t(lang, "help.positionals")
    c._optionals.title = t(lang, "help.options")
    c.add_argument("-h", "--help", action="help", help=t(lang, "help.help"))
    c.add_argument("manifest", metavar="MANIFEST", help=t(lang, "arg.manifest"))
    c.add_argument("--format", dest="fmt", choices=("auto", *FORMATS), default="auto", help=t(lang, "arg.format"))
    c.add_argument("--preset", choices=sorted(PRESETS), default="whisper", help=t(lang, "arg.preset"))
    c.add_argument("--tokenizer", metavar="PATH", help=t(lang, "arg.tokenizer"))
    whole, positive, number = _number_type(lang, int), _number_type(lang, int, positive=True), _number_type(lang, float)
    c.add_argument("--extra-tokens", type=whole, default=2, metavar="N", help=t(lang, "arg.extra_tokens"))
    c.add_argument("--alphabet", metavar="PATH", help=t(lang, "arg.alphabet"))
    c.add_argument("--sample-rate", type=positive, metavar="HZ", help=t(lang, "arg.sample_rate"))
    c.add_argument("--min-seconds", type=number, metavar="S", help=t(lang, "arg.min_seconds"))
    c.add_argument("--max-seconds", type=number, metavar="S", help=t(lang, "arg.max_seconds"))
    c.add_argument("--min-cps", type=number, metavar="X", help=t(lang, "arg.min_cps"))
    c.add_argument("--max-cps", type=number, metavar="X", help=t(lang, "arg.max_cps"))
    c.add_argument("--audio-ext", default="wav", metavar="EXT", help=t(lang, "arg.audio_ext"))
    c.add_argument("--text-field", metavar="NAME|INDEX", help=t(lang, "arg.text_field"))
    c.add_argument("--signal", dest="signal", action="store_const", const=True, default=None, help=t(lang, "arg.signal"))
    c.add_argument("--no-signal", dest="signal", action="store_const", const=False, help=t(lang, "arg.no_signal"))
    c.add_argument("--no-dedupe", action="store_true", help=t(lang, "arg.no_dedupe"))
    c.add_argument("--json", dest="as_json", action="store_true", help=t(lang, "arg.json"))
    c.add_argument("--write-clean", metavar="PATH", help=t(lang, "arg.write_clean"))
    c.add_argument("--drop", choices=("error", "warning"), default="error", help=t(lang, "arg.drop"))
    c.add_argument("--fail-on", choices=("error", "warning"), default="error", help=t(lang, "arg.fail_on"))
    c.add_argument("--explain", action="store_true", help=t(lang, "arg.explain"))

    d = sub.add_parser("demo", help=t(lang, "cmd.demo"), description=t(lang, "cmd.demo"),
                       formatter_class=Formatter, add_help=False)
    d._positionals.title = t(lang, "help.positionals")
    d._optionals.title = t(lang, "help.options")
    d.add_argument("-h", "--help", action="help", help=t(lang, "help.help"))
    d.add_argument("--out", default="speechset-demo", metavar="DIR", help=t(lang, "arg.demo_out"))
    d.add_argument("--format", dest="fmt", choices=FORMATS, default="nemo", help=t(lang, "arg.demo_format"))
    d.add_argument("--force", action="store_true", help=t(lang, "arg.demo_force"))
    return root


# 팁은 메시지에 없는 정보(바로 쓸 수 있는 명령)를 더할 때만 붙인다.
# 메시지가 이미 고칠 방법을 말하면(--format, pip install ...) 팁을 붙이지 않는다.
TIPS = {
    "e.clean_overwrite": "err.tip.clean_name",
    "e.clean_same_folder": "err.tip.clean",
    "e.demo_nonempty": "err.tip.demo",
}


def _fail(lang: str, exc: SpeechsetLintError | OSError) -> int:
    """Print an input error in the chosen language and return the usage exit code (2)."""
    if isinstance(exc, SpeechsetLintError):
        text = t(lang, exc.key, **exc.params)  # same key as str(exc), rendered in `lang`
        tip = TIPS.get(exc.key)
    else:
        text = t(lang, "e.os", reason=os_reason(exc))
        tip = None
    print(f"{t(lang, 'err.prefix')}: {text}", file=sys.stderr)
    if tip:
        print(t(lang, tip, **exc.params), file=sys.stderr)
    return EXIT_USAGE


def _number_type(lang: str, cast: type, *, positive: bool = False):
    """argparse ``type=`` that rejects negative (or, for ``positive``, zero) values in the chosen language."""

    def parse(text: str):
        try:
            value = cast(text)
        except ValueError:
            raise argparse.ArgumentTypeError(t(lang, "e.arg_whole" if cast is int else "e.arg_number", value=text)) from None
        if positive and value <= 0:
            raise argparse.ArgumentTypeError(t(lang, "e.arg_positive", value=text))
        if not math.isfinite(value) or value < 0:  # nan, inf, -1 would flag every clip
            raise argparse.ArgumentTypeError(t(lang, "e.arg_not_negative", value=text))
        return value

    parse.__name__ = cast.__name__
    return parse


def _run_check(args: argparse.Namespace, lang: str, *, explain: bool, show_result: bool = True) -> int:
    try:
        report = check(
            args.manifest,
            fmt=args.fmt,
            preset=args.preset,
            tokenizer=args.tokenizer,
            alphabet=args.alphabet,
            signal=args.signal,
            dedupe=not args.no_dedupe,
            extra_tokens=args.extra_tokens,
            audio_ext=args.audio_ext,
            text_field=args.text_field,
            sample_rate=args.sample_rate,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            min_cps=args.min_cps,
            max_cps=args.max_cps,
        )
    except SpeechsetLintError as exc:
        return _fail(lang, exc)
    print(render_json(report) if args.as_json else render_table(report, lang, explain=explain))
    status = EXIT_ISSUES if report.has_errors(args.fail_on) else EXIT_OK
    if args.write_clean:
        try:
            kept = write_clean(report, args.write_clean, args.drop)
        except SpeechsetLintError as exc:
            return _fail(lang, exc)
        message = t(lang, "out.clean", kept=kept, total=report.summary["rows"], path=args.write_clean)
        print(message, file=sys.stderr if args.as_json else sys.stdout)
    if not args.as_json and show_result:
        print(t(lang, "out.fail" if status else "out.pass"))
    return status


def _run_demo(args: argparse.Namespace, lang: str) -> int:
    out = Path(args.out)
    try:
        manifests = write_manifests(out, force=args.force)
    except (OSError, SpeechsetLintError) as exc:
        return _fail(lang, exc)
    print(t(lang, "demo.banner"))  # 데이터를 실제로 쓴 뒤에만 "합성 연습 데이터" 안내를 낸다
    manifest = manifests[args.fmt]
    rows = len(manifest.read_text(encoding="utf-8").strip().splitlines()) - (1 if args.fmt == "audiofolder" else 0)
    print(t(lang, "demo.wrote", rows=rows, out=out))
    command = f"speechset-lint check {manifest} --preset whisper --signal --explain"
    print(t(lang, "demo.cmd", cmd=command))
    print()
    check_args = argparse.Namespace(
        manifest=str(manifest), fmt=args.fmt, preset="whisper", tokenizer=None, alphabet=None, signal=True,
        no_dedupe=False, extra_tokens=2, audio_ext="wav", text_field=None, sample_rate=None, min_seconds=None,
        max_seconds=None, min_cps=None, max_cps=None, as_json=False, write_clean=None, drop="error",
        fail_on="error",
    )
    status = _run_check(check_args, lang, explain=True, show_result=False)
    print()
    print(t(lang, "demo.next"))
    return EXIT_OK if status in (EXIT_OK, EXIT_ISSUES) else status


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    lang = resolve_lang(_prescan_lang(args_list))
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    args = build_parser(lang).parse_args(args_list)
    lang = resolve_lang(args.lang)
    if args.command == "demo":
        return _run_demo(args, lang)
    return _run_check(args, lang, explain=args.explain)


if __name__ == "__main__":
    raise SystemExit(main())
