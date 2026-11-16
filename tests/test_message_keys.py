"""Static check of every message key used in src/ (QA Q-7 root cause).

Q-7 happened because a raise site passed a finished English sentence where a message key
belongs. This test reads the source with ``ast`` and fails on:
- an exception of this package (or ``Msg``/``t``) whose key is not a string literal,
- a key that is missing from the English or Korean table,
- a placeholder in either language that the call site does not pass,
- a built-in exception raised with a message (user input errors must use keys),
- an error/tip key in the tables that no code uses.
"""

from __future__ import annotations

import ast
import string
from pathlib import Path

import pytest

from speechset_lint.i18n import MESSAGES

SRC = Path(__file__).resolve().parents[1] / "src" / "speechset_lint"
KEYED = {"SpeechsetLintError", "ManifestError", "UnknownFormatError", "TokenizerUnavailableError", "CleanError", "Msg"}
# raises that are not user-facing messages: an internal signal caught in checks.py, and
# argparse type errors whose text is already rendered with t()
ALLOWED_RAISES = {"UnreadableAudio", "argparse.ArgumentTypeError", "SystemExit"}


def _name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return f"{_name(func.value)}.{func.attr}"
    return ""


def _placeholders(template: str) -> set[str]:
    return {field.split(".")[0].split("[")[0] for _, field, _, _ in string.Formatter().parse(template) if field}


def _calls():
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield path.name, node


def _keyed_calls(with_kind: bool = False):
    """(file, line, key node, keyword names[, kind]) for every keyed exception, Msg() and t() call."""
    for file, node in _calls():
        name = _name(node.func).split(".")[-1]
        kwargs = {k.arg for k in node.keywords}
        if name in KEYED and node.args:
            item = (file, node.lineno, node.args[0], kwargs, "msg" if name == "Msg" else "raise")
        elif name == "t" and len(node.args) >= 2:
            item = (file, node.lineno, node.args[1], kwargs, "t")
        else:
            continue
        yield item if with_kind else item[:4]


def test_every_keyed_call_uses_a_literal_known_key_with_its_placeholders():
    problems = []
    for file, line, key_node, kwargs, kind in _keyed_calls(with_kind=True):
        where = f"{file}:{line}"
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            if kind != "raise":
                continue  # t()/Msg() with keys taken from literal tables (TIPS, _OS_REASONS, f"detail.{code}")
            problems.append(f"{where}: exception key is not a string literal")
            continue
        key = key_node.value
        for lang in ("en", "ko"):
            if key not in MESSAGES[lang]:
                problems.append(f"{where}: {key!r} missing in {lang}")
                continue
            if None in kwargs:  # **params: checked at run time by the CLI tests
                continue
            missing = _placeholders(MESSAGES[lang][key]) - kwargs
            if missing:
                problems.append(f"{where}: {key!r} ({lang}) needs {sorted(missing)}")
    assert not problems, "\n".join(problems)


def test_found_keyed_calls():
    """Guard the guard: the walker must actually see the raise sites (text.py had 4 in Q-7)."""
    files = [file for file, _, _, _, kind in _keyed_calls(with_kind=True) if kind == "raise"]
    assert files.count("text.py") >= 6
    assert files.count("manifest.py") >= 20


def test_no_builtin_exception_with_a_message():
    problems = []
    for path in sorted(SRC.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                name = _name(node.exc.func)
                if name.split(".")[-1] not in KEYED and name not in ALLOWED_RAISES:
                    problems.append(f"{path.name}:{node.lineno}: raise {name}(...)")
    assert not problems, "\n".join(problems)


def test_cli_dynamic_keys_come_from_literal_tables():
    from speechset_lint import cli
    from speechset_lint.errors import _OS_REASONS

    for error_key, tip_key in cli.TIPS.items():
        assert error_key in MESSAGES["en"] and tip_key in MESSAGES["en"]
        # the tip may only use parameters its error always passes
        assert _placeholders(MESSAGES["en"][tip_key]) <= _raise_params(error_key), tip_key
    for key in _OS_REASONS.values():
        assert key in MESSAGES["en"] and key in MESSAGES["ko"]


def _raise_params(key: str) -> set[str]:
    sites = [kwargs for _, _, node, kwargs in _keyed_calls() if isinstance(node, ast.Constant) and node.value == key]
    assert sites, key
    return set.intersection(*sites)


@pytest.mark.parametrize("prefix", ["e.", "err.", "os.", "name."])
def test_no_unused_error_keys(prefix):
    used: set[str] = set()
    for path in sorted(SRC.glob("*.py")):
        if path.name == "i18n.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                used.add(node.value)
    unused = sorted(k for k in MESSAGES["en"] if k.startswith(prefix) and k not in used)
    assert not unused
