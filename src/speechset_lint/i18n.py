"""Screen messages in English and Korean, plus language resolution.

All user-visible text lives in ``MESSAGES``. Issue ``code`` values in JSON output are
never translated. Library exceptions carry a message key; ``str(exc)`` is English and the CLI renders the key in the chosen language.
"""

from __future__ import annotations

from typing import Mapping

LANGS = ("en", "ko")
ENV_VAR = "SPEECHSET_LINT_LANG"


def _from_value(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered.startswith("ko"):
        return "ko"
    if lowered.startswith("en"):
        return "en"
    return None


def resolve_lang(cli_value: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """``--lang`` > SPEECHSET_LINT_LANG > LC_ALL > LC_MESSAGES > LANG; anything but Korean is English."""
    import os

    environ = os.environ if env is None else env
    if cli_value in LANGS:
        return cli_value  # type: ignore[return-value]
    explicit = _from_value(environ.get(ENV_VAR))
    if explicit:
        return explicit
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = environ.get(name)
        if value:  # the first non-empty locale variable decides, as in POSIX
            return "ko" if value.strip().lower().startswith("ko") else "en"
    return "en"


class Msg:
    """A message key used as a parameter of another message (for example an OS error reason).

    ``t()`` renders it in the same language as the outer message, so a Korean error
    never embeds an English reason such as "No such file or directory".
    """

    def __init__(self, key: str, /, **params: object) -> None:
        self.key = key
        self.params = params

    def __repr__(self) -> str:
        return f"Msg({self.key!r})"


def t(lang: str, key: str, /, **kw: object) -> str:
    """Look up ``key`` for ``lang`` (falls back to English) and format it.

    An unknown key raises ``KeyError`` on purpose: tests/test_message_keys.py checks
    every key used in the source, so a typo fails the test run instead of a user's run.
    """
    table = MESSAGES.get(lang, MESSAGES["en"])
    template = table.get(key) or MESSAGES["en"][key]
    if not kw:
        return template
    # Msg 파라미터는 바깥 문장과 같은 언어로 먼저 렌더링한다
    rendered = {name: t(lang, v.key, **v.params) if isinstance(v, Msg) else v for name, v in kw.items()}
    return template.format(**rendered)


MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # --- CLI help
        "help.usage": "usage",
        "help.options": "options",
        "help.positionals": "arguments",
        "help.commands": "commands",
        "help.help": "show this help message and exit",
        "app.desc": "Pre-flight lint for speech fine-tuning manifests (Whisper, wav2vec2 CTC, TTS).",
        "app.epilog": "Exit codes: 0 no issues at the failing level, 1 issues found, 2 usage or input error.",
        "cmd.check": "check a manifest and report issues",
        "cmd.demo": "create a synthetic practice dataset and check it",
        "arg.lang": "screen language (default: SPEECHSET_LINT_LANG, then LC_ALL/LC_MESSAGES/LANG)",
        "arg.version": "show the version and exit",
        "arg.manifest": "NeMo JSONL, audiofolder metadata.csv/.jsonl, or LJSpeech metadata.csv",
        "arg.format": "manifest format (default: auto-detect)",
        "arg.preset": "target model family; sets thresholds and severities",
        "arg.tokenizer": "path to a tokenizer.json for exact label token counts (needs the [tokens] extra)",
        "arg.extra_tokens": "tokens the training code adds around each label (default: 2)",
        "arg.alphabet": "vocab.json or a one-character-per-line file; other characters are reported",
        "arg.sample_rate": "expected sample rate in Hz (default: from the preset)",
        "arg.min_seconds": "minimum clip length in seconds",
        "arg.max_seconds": "maximum clip length in seconds",
        "arg.min_cps": "minimum non-space characters per second",
        "arg.max_cps": "maximum non-space characters per second",
        "arg.audio_ext": "audio extension for LJSpeech ids (default: wav)",
        "arg.text_field": "transcript column or field name (LJSpeech: 0-based column index)",
        "arg.signal": "decode audio to detect clipping and silence (default for the tts preset)",
        "arg.no_signal": "do not decode audio (headers only)",
        "arg.no_dedupe": "skip the duplicate-audio hash check",
        "arg.json": "print a JSON report instead of the table",
        "arg.write_clean": "write a same-format manifest without the failing rows to this path",
        "arg.drop": "which rows --write-clean removes (default: error)",
        "arg.fail_on": "lowest severity that makes the exit code 1 (default: error)",
        "arg.explain": "add a line saying what each issue causes in training",
        "arg.demo_out": "folder for the synthetic dataset (default: ./speechset-demo)",
        "arg.demo_format": "which manifest format to check (all three are written)",
        # --- table and summary
        "col.line": "line",
        "col.audio": "audio",
        "col.severity": "severity",
        "col.code": "code",
        "col.detail": "detail",
        "sev.error": "error",
        "sev.warning": "warning",
        "sev.info": "info",
        "out.header": "manifest: {manifest} | format: {fmt} | preset: {preset} | signal check: {signal}",
        "out.signal_on": "on",
        "out.signal_off": "off (headers only)",
        "out.no_issues": "no issues found",
        "out.summary": "{rows} rows checked: {errors} errors, {warnings} warnings ({seconds:.2f}s)",
        "out.by_code": "by code: {codes}",
        "out.why": "-> {why}",
        "out.clean": "wrote {kept} of {total} rows to {path}",
        "out.fail": "result: FAIL (exit 1)",
        "out.pass": "result: OK (exit 0)",
        # --- CLI errors
        "err.prefix": "error",
        # tips only add what the message does not already say (a concrete command to run)
        "err.tip.clean": "tip: write the cleaned file next to the manifest, e.g. --write-clean {suggest}",
        "err.tip.clean_name": "tip: choose a different file name, e.g. --write-clean {suggest}",
        "err.tip.demo": "tip: pass --out with an empty or new folder, or add --force to write into this one",
        "arg.demo_force": "write into a non-empty folder anyway (existing demo files are overwritten)",
        # --- OS error reasons (rendered inside other messages)
        "os.enoent": "no such file or folder",
        "os.eacces": "permission denied",
        "os.eisdir": "it is a folder, not a file",
        "os.enotdir": "a part of the path is not a folder",
        "os.enospc": "no space left on the device",
        "os.erofs": "the file system is read-only",
        "os.other": "operating system error {errno}",
        "e.not_found": "manifest not found: {path}; check the path",
        "e.cannot_read": "cannot read manifest {path}: {reason}",
        "e.unknown_format": "unknown format '{fmt}'; choose one of: auto, {choices}",
        "e.empty": "{path}: the manifest is empty (no non-blank lines)",
        "e.text_field_not_key": "{path}: line {line}: --text-field '{field}' is not a key of the first JSON object (found: {found})",
        "e.text_column_range": "{path}: line {line}: --text-field column {index} does not exist; the row has {count} columns (0 to {last})",
        "e.alphabet_not_found": "alphabet file not found: {path}; check the --alphabet path",
        "e.alphabet_read": "cannot read alphabet file {path}: {reason}",
        "e.alphabet_utf8": "alphabet file {path} is not valid UTF-8 (byte {column}); re-save it as UTF-8",
        "e.alphabet_json": "alphabet file {path} is not valid JSON (line {line}, column {column}: {reason})",
        "e.alphabet_shape": "alphabet file {path} must be a JSON object (vocab.json) or a list of characters",
        "e.alphabet_empty": "alphabet file {path} has no single-character entries; use a vocab.json or a text file with one character per line",
        "e.not_negative": "{name} must be a finite number of zero or more, got {value}",
        "e.not_positive": "{name} must be a whole number above zero, got {value}",
        "e.min_gt_max": "the minimum {name} ({low:g}) is larger than the maximum ({high:g}); every clip would be flagged",
        "e.bad_choice": "{name} must be one of {choices}, got {value!r}",
        "name.seconds": "clip length in seconds (--min-seconds/--max-seconds)",
        "name.cps": "characters per second (--min-cps/--max-cps)",
        # argparse type checks: argparse adds "argument NAME: " in front (see e.arg_detail)
        "e.arg_number": "invalid value: {value} (expected a number)",
        "e.arg_whole": "invalid value: {value} (expected a whole number)",
        "e.arg_not_negative": "invalid value: {value} (expected a number of zero or more)",
        "e.arg_positive": "invalid value: {value} (expected a whole number above zero)",
        "e.arg_detail": "argument {name}: {detail}",
        "e.json_keys": "{path}: JSON lines need an 'audio_filepath' (NeMo) or 'file_name' (audiofolder) key; pass --format nemo|audiofolder|ljspeech to choose explicitly",
        "e.detect_fail": "{path}: cannot tell the format (no 'file_name' header, no '|' separator, not JSON); pass --format nemo|audiofolder|ljspeech to choose explicitly",
        "e.nul": "{path}: line {line} contains NUL bytes; the file may be UTF-16 or binary, re-save it as UTF-8",
        "e.utf16": "{path}: the file starts with a UTF-16/32 byte order mark; re-save it as UTF-8",
        "e.utf8_column": "{path}: line {line}: not valid UTF-8 (column {column}); re-save the manifest as UTF-8",
        "e.utf8_line": "{path}: line {line}: not valid UTF-8; re-save the manifest as UTF-8",
        "e.invalid_json": "{path}: line {line}, column {column}: invalid JSON ({reason}); each line must be one JSON object",
        "e.missing_field": "{path}: line {line}: missing required string field '{key}' for the {fmt} format",
        "e.no_transcript_field": "{path}: line {line}: no transcript field (looked for {cols}); pass --text-field",
        "e.bad_duration": "{path}: line {line}: 'duration' must be a number, got {value}",
        "e.invalid_csv": "{path}: line {line}: invalid CSV ({reason})",
        "e.csv_empty": "{path}: the CSV has no header row with a 'file_name' column",
        "e.csv_no_file_name": "{path}: line {line}: header has no 'file_name' column (found: {found})",
        "e.text_field_missing": "{path}: --text-field '{field}' is not a column (found: {found})",
        "e.csv_no_transcript": "{path}: no transcript column (looked for {cols}; found: {found}); pass --text-field NAME",
        "e.empty_file_name": "{path}: line {line}: empty or missing 'file_name' value",
        "e.lj_text_field": "LJSpeech manifests have no column names; use --text-field with a 0-based column index",
        "e.lj_bad_row": "{path}: line {line}: expected 'id|text|normalized text' separated by '|'",
        "e.nemo_needs_json": "{path}: the nemo format needs JSON lines but the first line is not a JSON object",
        "e.unknown_preset": "unknown preset '{name}'; choose one of: {choices}",
        "e.tok_missing": "the 'tokenizers' package is not installed; install it with `pip install 'speechset-lint[tokens]'` or omit --tokenizer to use the UTF-8 byte upper bound",
        "e.tok_not_found": "tokenizer file not found: {path}; download tokenizer.json from the model repo (e.g. https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json)",
        "e.tok_load": "cannot load tokenizer {path}: {reason}",
        "e.clean_no_manifest": "the report has no manifest path; run check() on a file first",
        "e.clean_overwrite": "refusing to overwrite the input manifest {path}",
        "e.clean_same_folder": "cannot write {path}: audio paths in this {fmt} manifest are relative to {folder}, so the cleaned file must go in the same folder (rewriting paths for another folder is not supported yet)",
        "e.clean_write": "cannot write {path}: {reason}",
        "e.demo_not_dir": "{path} exists and is not a folder",
        "e.demo_write": "cannot write the demo dataset under {path}: {reason}",
        "e.demo_nonempty": "{path} is not empty and was not created by this demo; refusing to write into it",
        "e.os": "{reason}",
        "e.arg_required": "the following arguments are required: {names}",
        "e.arg_choice": "argument {name}: invalid choice: {value} (choose from {choices})",
        "e.arg_value": "argument {name}: invalid value: {value}",
        "e.arg_expected": "argument {name}: expected one argument",
        "e.arg_unrecognized": "unrecognized arguments: {names}",
        "e.arg_other": "invalid arguments ({detail})",
        # --- demo
        "demo.banner": "synthetic practice data: sine tones and silence, not real speech",
        "demo.wrote": "wrote {rows} rows and the audio under {out} (nemo/, audiofolder/, ljspeech/)",
        "demo.cmd": "checking it like this: {cmd}",
        "demo.next": "try it on your data: speechset-lint check <manifest> --preset whisper",
        # --- issue details
        "detail.missing_file": "file not found: {path}{hint}",
        "hint.found_ext": " (a file with the same name exists as {exts}; check --audio-ext or the path)",
        "detail.unreadable_audio": "cannot read the audio: {error}",
        "detail.empty_text": "transcript is empty",
        "detail.sample_rate_mismatch": "sample rate {actual} Hz, expected {expected} Hz",
        "detail.not_mono": "{channels} channels, expected mono",
        "detail.too_long": "{seconds:.2f} s is longer than the {limit:g} s limit",
        "detail.too_short": "{seconds:.2f} s is shorter than the {limit:g} s minimum",
        "detail.duration_mismatch": "manifest duration {manifest:.2f} s, audio header {header:.2f} s (off by {diff:.2f} s)",
        "detail.char_rate_high": "{cps:.1f} characters/s is above {limit:g}",
        "detail.char_rate_low": "{cps:.1f} characters/s is below {limit:g}",
        "detail.ctc_unalignable": "{frames} frames < {labels} labels needed",
        "detail.label_tokens_over": "{tokens}+{extra} > {limit} label tokens",
        "detail.label_tokens_maybe_over": "up to {tokens}+{extra} > {limit} tokens (UTF-8 byte upper bound); pass --tokenizer for an exact count",
        "detail.oov_chars": "characters not in the alphabet: {chars}",
        "detail.not_nfc": "transcript is not NFC-normalized (decomposed characters)",
        "detail.control_chars": "control characters in the transcript: {chars}",
        "detail.duplicate_audio": "same bytes as line {first_line} ({first_audio})",
        "detail.repeated_path": "same path as line {first_line}",
        "detail.clipping": "{percent:.2f}% of samples at full scale (limit {limit_percent:g}%)",
        "detail.mostly_silent": "{percent:.1f}% of samples below {db:g} dBFS (limit {limit_percent:g}%)",
        # --- what each issue causes
        "why.missing_file": "the data loader raises FileNotFoundError partway through an epoch",
        "why.unreadable_audio": "loading this clip fails and can crash the whole run",
        "why.empty_text": "an empty label gives a zero-length target: inf/NaN CTC loss or noise training",
        "why.sample_rate_mismatch": "audio is resampled silently or fed at the wrong speed and pitch",
        "why.not_mono": "extra channels break feature extraction or are mixed down unpredictably",
        "why.too_long": "Whisper silently cuts audio after 30 s, so the label no longer matches the audio",
        "why.too_short": "too little audio for a stable training example",
        "why.duration_mismatch": "the manifest is out of sync with the file; length filters and batching use wrong values",
        "why.char_rate_high": "far more text than audio: wrong or misaligned transcript",
        "why.char_rate_low": "far more audio than text: missing transcript, or silence",
        "why.ctc_unalignable": "CTC needs one frame per label (plus blanks between repeats); otherwise the loss is inf",
        "why.label_tokens_over": "training stops with 'size of tensor a (449) must match the size of tensor b (448)'",
        "why.label_tokens_maybe_over": "may hit the same 448-token error; check with --tokenizer",
        "why.oov_chars": "unknown characters become <unk> or break the CTC vocabulary",
        "why.not_nfc": "decomposed characters tokenize differently from composed ones",
        "why.control_chars": "stray control characters end up inside labels",
        "why.duplicate_audio": "duplicates are over-weighted and can leak into validation",
        "why.repeated_path": "the same clip is listed twice and is over-weighted",
        "why.clipping": "clipped audio teaches the model distortion",
        "why.mostly_silent": "almost no speech: the label has nothing to align to",
    },
    "ko": {
        "help.usage": "사용법",
        "help.options": "옵션",
        "help.positionals": "인자",
        "help.commands": "명령",
        "help.help": "이 도움말을 보여 주고 끝냅니다",
        "app.desc": "음성 파인튜닝 매니페스트(Whisper, wav2vec2 CTC, TTS)를 학습 전에 점검합니다.",
        "app.epilog": "종료 코드: 0 기준 이상 이슈 없음, 1 이슈 발견, 2 사용법·입력 오류.",
        "cmd.check": "매니페스트를 점검하고 이슈를 보고합니다",
        "cmd.demo": "합성 연습 데이터셋을 만들어 점검합니다",
        "arg.lang": "화면 언어 (기본: SPEECHSET_LINT_LANG, 다음 LC_ALL/LC_MESSAGES/LANG)",
        "arg.version": "버전을 보여 주고 끝냅니다",
        "arg.manifest": "NeMo JSONL, audiofolder metadata.csv/.jsonl, 또는 LJSpeech metadata.csv",
        "arg.format": "매니페스트 형식 (기본: 자동 감지)",
        "arg.preset": "대상 모델 계열. 기준값과 심각도를 정합니다",
        "arg.tokenizer": "정확한 라벨 토큰 수를 위한 tokenizer.json 경로 ([tokens] extra 필요)",
        "arg.extra_tokens": "학습 코드가 라벨 앞뒤에 더하는 토큰 수 (기본: 2)",
        "arg.alphabet": "vocab.json 또는 한 줄에 한 글자인 파일. 그 밖의 글자를 보고합니다",
        "arg.sample_rate": "기대하는 샘플레이트(Hz) (기본: 프리셋 값)",
        "arg.min_seconds": "클립 최소 길이(초)",
        "arg.max_seconds": "클립 최대 길이(초)",
        "arg.min_cps": "초당 최소 글자 수(공백 제외)",
        "arg.max_cps": "초당 최대 글자 수(공백 제외)",
        "arg.audio_ext": "LJSpeech id에 붙일 오디오 확장자 (기본: wav)",
        "arg.text_field": "전사 열·필드 이름 (LJSpeech: 0부터 시작하는 열 번호)",
        "arg.signal": "오디오를 디코드해 클리핑·무음을 검사합니다 (tts 프리셋은 기본 켬)",
        "arg.no_signal": "오디오를 디코드하지 않습니다 (헤더만)",
        "arg.no_dedupe": "중복 오디오 해시 검사를 건너뜁니다",
        "arg.json": "표 대신 JSON 보고서를 출력합니다",
        "arg.write_clean": "실패한 행을 뺀 같은 형식의 매니페스트를 이 경로에 씁니다",
        "arg.drop": "--write-clean 이 뺄 행 (기본: error)",
        "arg.fail_on": "종료 코드를 1로 만드는 최저 심각도 (기본: error)",
        "arg.explain": "각 이슈가 학습에서 일으키는 문제를 한 줄씩 덧붙입니다",
        "arg.demo_out": "합성 데이터셋을 만들 폴더 (기본: ./speechset-demo)",
        "arg.demo_format": "점검할 매니페스트 형식 (세 형식 모두 만들어 둡니다)",
        "col.line": "줄",
        "col.audio": "오디오",
        "col.severity": "심각도",
        "col.code": "코드",
        "col.detail": "내용",
        "sev.error": "오류",
        "sev.warning": "경고",
        "sev.info": "정보",
        "out.header": "매니페스트: {manifest} | 형식: {fmt} | 프리셋: {preset} | 신호 검사: {signal}",
        "out.signal_on": "켬",
        "out.signal_off": "끔 (헤더만)",
        "out.no_issues": "이슈가 없습니다",
        "out.summary": "{rows}행 점검: 오류 {errors}개, 경고 {warnings}개 ({seconds:.2f}초)",
        "out.by_code": "코드별: {codes}",
        "out.why": "-> {why}",
        "out.clean": "{total}행 중 {kept}행을 {path} 에 썼습니다",
        "out.fail": "결과: 실패 (종료 코드 1)",
        "out.pass": "결과: 통과 (종료 코드 0)",
        "err.prefix": "오류",
        "err.tip.clean": "팁: 걸러낸 파일은 매니페스트와 같은 폴더에 쓰세요 (예: --write-clean {suggest})",
        "err.tip.clean_name": "팁: 다른 파일 이름을 고르세요 (예: --write-clean {suggest})",
        "err.tip.demo": "팁: --out 에 비어 있거나 새 폴더를 주거나, 이 폴더에 쓰려면 --force 를 붙이세요",
        "arg.demo_force": "비어 있지 않은 폴더에도 씁니다 (기존 데모 파일을 덮어씀)",
        "os.enoent": "파일이나 폴더가 없습니다",
        "os.eacces": "권한이 없습니다",
        "os.eisdir": "파일이 아니라 폴더입니다",
        "os.enotdir": "경로 중간에 폴더가 아닌 것이 있습니다",
        "os.enospc": "저장 공간이 부족합니다",
        "os.erofs": "읽기 전용 파일 시스템입니다",
        "os.other": "운영체제 오류 {errno}",
        "e.not_found": "매니페스트를 찾을 수 없습니다: {path}; 경로를 확인하세요",
        "e.cannot_read": "매니페스트를 읽을 수 없습니다 {path}: {reason}",
        "e.unknown_format": "알 수 없는 형식 '{fmt}'; 다음 중에서 고르세요: auto, {choices}",
        "e.empty": "{path}: 매니페스트가 비어 있습니다 (내용이 있는 줄이 없음)",
        "e.text_field_not_key": "{path}: {line}번 줄: --text-field '{field}' 는 첫 JSON 객체의 키가 아닙니다 (있는 키: {found})",
        "e.text_column_range": "{path}: {line}번 줄: --text-field 열 번호 {index} 가 없습니다; 이 행의 열은 {count}개입니다 (0 ~ {last})",
        "e.alphabet_not_found": "알파벳 파일을 찾을 수 없습니다: {path}; --alphabet 경로를 확인하세요",
        "e.alphabet_read": "알파벳 파일을 읽을 수 없습니다 {path}: {reason}",
        "e.alphabet_utf8": "알파벳 파일 {path} 이 UTF-8 이 아닙니다 ({column}번째 바이트); UTF-8 로 다시 저장하세요",
        "e.alphabet_json": "알파벳 파일 {path} 이 올바른 JSON 이 아닙니다 ({line}번 줄 {column}번째 글자)",
        "e.alphabet_shape": "알파벳 파일 {path} 은 JSON 객체(vocab.json)나 문자 목록이어야 합니다",
        "e.alphabet_empty": "알파벳 파일 {path} 에 한 글자짜리 항목이 없습니다; vocab.json 이나 한 줄에 한 글자씩 쓴 텍스트 파일을 쓰세요",
        "e.not_negative": "{name} 은(는) 0 이상의 유한한 숫자여야 합니다 (받은 값: {value})",
        "e.not_positive": "{name} 은(는) 0보다 큰 정수여야 합니다 (받은 값: {value})",
        "e.min_gt_max": "{name} 최솟값({low:g})이 최댓값({high:g})보다 큽니다; 모든 클립이 걸리게 됩니다",
        "e.bad_choice": "{name} 은(는) {choices} 중 하나여야 합니다 (받은 값: {value!r})",
        "name.seconds": "클립 길이(초, --min-seconds/--max-seconds)",
        "name.cps": "초당 글자 수(--min-cps/--max-cps)",
        "e.arg_number": "잘못된 값: {value} (숫자여야 합니다)",
        "e.arg_whole": "잘못된 값: {value} (정수여야 합니다)",
        "e.arg_not_negative": "잘못된 값: {value} (0 이상의 숫자여야 합니다)",
        "e.arg_positive": "잘못된 값: {value} (0보다 큰 정수여야 합니다)",
        "e.arg_detail": "인자 {name}: {detail}",
        "e.json_keys": "{path}: JSON 줄에는 'audio_filepath'(NeMo) 또는 'file_name'(audiofolder) 키가 필요합니다; --format nemo|audiofolder|ljspeech 로 직접 지정하세요",
        "e.detect_fail": "{path}: 형식을 알 수 없습니다 ('file_name' 헤더도, '|' 구분자도, JSON 도 아님); --format nemo|audiofolder|ljspeech 로 직접 지정하세요",
        "e.nul": "{path}: {line}번 줄에 NUL 바이트가 있습니다; UTF-16 이거나 바이너리 파일일 수 있으니 UTF-8 로 다시 저장하세요",
        "e.utf16": "{path}: 파일이 UTF-16/32 바이트 순서 표시(BOM)로 시작합니다; UTF-8 로 다시 저장하세요",
        "e.utf8_column": "{path}: {line}번 줄: UTF-8 이 아닙니다 ({column}번째 바이트); 매니페스트를 UTF-8 로 다시 저장하세요",
        "e.utf8_line": "{path}: {line}번 줄: UTF-8 이 아닙니다; 매니페스트를 UTF-8 로 다시 저장하세요",
        "e.invalid_json": "{path}: {line}번 줄 {column}번째 글자: 잘못된 JSON 입니다; 각 줄은 JSON 객체 하나여야 합니다",
        "e.missing_field": "{path}: {line}번 줄: {fmt} 형식에 필요한 문자열 필드 '{key}' 가 없습니다",
        "e.no_transcript_field": "{path}: {line}번 줄: 전사 필드가 없습니다 (찾은 후보: {cols}); --text-field 를 지정하세요",
        "e.bad_duration": "{path}: {line}번 줄: 'duration' 은 숫자여야 합니다 (받은 값: {value})",
        "e.invalid_csv": "{path}: {line}번 줄: CSV 를 읽을 수 없습니다; 닫히지 않은 따옴표나 너무 긴 필드가 있는지 확인하세요",
        "e.csv_empty": "{path}: CSV 에 'file_name' 열이 있는 헤더 행이 없습니다",
        "e.csv_no_file_name": "{path}: {line}번 줄: 헤더에 'file_name' 열이 없습니다 (있는 열: {found})",
        "e.text_field_missing": "{path}: --text-field '{field}' 는 열 이름이 아닙니다 (있는 열: {found})",
        "e.csv_no_transcript": "{path}: 전사 열이 없습니다 (찾은 후보: {cols}; 있는 열: {found}); --text-field 로 전사 열 이름을 지정하세요",
        "e.empty_file_name": "{path}: {line}번 줄: 'file_name' 값이 비었거나 없습니다",
        "e.lj_text_field": "LJSpeech 매니페스트에는 열 이름이 없습니다; --text-field 에 0부터 시작하는 열 번호를 주세요",
        "e.lj_bad_row": "{path}: {line}번 줄: '|' 로 구분된 'id|text|normalized text' 형태여야 합니다",
        "e.nemo_needs_json": "{path}: nemo 형식은 JSON 줄이 필요한데 첫 줄이 JSON 객체가 아닙니다",
        "e.unknown_preset": "알 수 없는 프리셋 '{name}'; 다음 중에서 고르세요: {choices}",
        "e.tok_missing": "'tokenizers' 패키지가 설치되어 있지 않습니다; `pip install 'speechset-lint[tokens]'` 로 설치하거나 --tokenizer 를 빼고 UTF-8 바이트 상한을 쓰세요",
        "e.tok_not_found": "토크나이저 파일을 찾을 수 없습니다: {path}; 모델 저장소에서 tokenizer.json 을 받으세요 (예: https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json)",
        "e.tok_load": "토크나이저를 불러올 수 없습니다 {path}: 설치된 tokenizers 패키지가 읽을 수 있는 tokenizer.json 이 아닙니다",
        "e.clean_no_manifest": "보고서에 매니페스트 경로가 없습니다; 먼저 파일에 check() 를 실행하세요",
        "e.clean_overwrite": "입력 매니페스트 {path} 를 덮어쓰지 않습니다",
        "e.clean_same_folder": "{path} 에 쓸 수 없습니다: 이 {fmt} 매니페스트의 오디오 경로는 {folder} 기준 상대 경로라서 걸러낸 파일도 같은 폴더에 있어야 합니다 (다른 폴더용 경로 다시 쓰기는 아직 지원하지 않습니다)",
        "e.clean_write": "{path} 에 쓸 수 없습니다: {reason}",
        "e.demo_not_dir": "{path} 가 이미 있고 폴더가 아닙니다",
        "e.demo_write": "{path} 아래에 데모 데이터를 쓸 수 없습니다: {reason}",
        "e.demo_nonempty": "{path} 는 비어 있지 않고 이 데모가 만든 폴더도 아니라서 쓰지 않습니다",
        "e.os": "{reason}",
        "e.arg_required": "다음 인자가 필요합니다: {names}",
        "e.arg_choice": "인자 {name}: 잘못된 선택지: {value} (선택 가능: {choices})",
        "e.arg_value": "인자 {name}: 잘못된 값: {value}",
        "e.arg_expected": "인자 {name}: 값이 하나 필요합니다",
        "e.arg_unrecognized": "알 수 없는 인자: {names}",
        "e.arg_other": "잘못된 인자입니다 ({detail})",
        "demo.banner": "합성(synthetic) 연습 데이터: 사인파와 무음이며 실제 음성이 아닙니다",
        "demo.wrote": "{rows}행과 오디오를 {out} 아래에 만들었습니다 (nemo/, audiofolder/, ljspeech/)",
        "demo.cmd": "다음처럼 점검합니다: {cmd}",
        "demo.next": "내 데이터로 해 보기: speechset-lint check <매니페스트> --preset whisper",
        "detail.missing_file": "파일이 없습니다: {path}{hint}",
        "hint.found_ext": " (같은 이름의 {exts} 파일이 있습니다. --audio-ext 나 경로를 확인하세요)",
        "detail.unreadable_audio": "오디오를 읽을 수 없습니다: {error}",
        "detail.empty_text": "전사가 비어 있습니다",
        "detail.sample_rate_mismatch": "샘플레이트 {actual} Hz, 기대값 {expected} Hz",
        "detail.not_mono": "{channels}채널, 모노여야 합니다",
        "detail.too_long": "{seconds:.2f}초가 한도 {limit:g}초보다 깁니다",
        "detail.too_short": "{seconds:.2f}초가 최소 {limit:g}초보다 짧습니다",
        "detail.duration_mismatch": "매니페스트 길이 {manifest:.2f}초, 오디오 헤더 {header:.2f}초 ({diff:.2f}초 차이)",
        "detail.char_rate_high": "초당 {cps:.1f}자가 {limit:g}자보다 많습니다",
        "detail.char_rate_low": "초당 {cps:.1f}자가 {limit:g}자보다 적습니다",
        "detail.ctc_unalignable": "프레임 {frames}개 < 필요한 라벨 {labels}개",
        "detail.label_tokens_over": "라벨 토큰 {tokens}+{extra} > {limit}",
        "detail.label_tokens_maybe_over": "최대 {tokens}+{extra} > {limit} 토큰 (UTF-8 바이트 상한). 정확한 개수는 --tokenizer 로 확인하세요",
        "detail.oov_chars": "어휘에 없는 글자: {chars}",
        "detail.not_nfc": "전사가 NFC 정규화되어 있지 않습니다 (분해된 글자)",
        "detail.control_chars": "전사에 제어 문자가 있습니다: {chars}",
        "detail.duplicate_audio": "{first_line}번 줄({first_audio})과 바이트가 같습니다",
        "detail.repeated_path": "{first_line}번 줄과 경로가 같습니다",
        "detail.clipping": "샘플의 {percent:.2f}%가 최대 진폭입니다 (한도 {limit_percent:g}%)",
        "detail.mostly_silent": "샘플의 {percent:.1f}%가 {db:g} dBFS 미만입니다 (한도 {limit_percent:g}%)",
        "why.missing_file": "데이터 로더가 학습 도중 FileNotFoundError 를 냅니다",
        "why.unreadable_audio": "이 클립을 읽지 못해 학습 전체가 멈출 수 있습니다",
        "why.empty_text": "빈 라벨은 길이 0 타깃이 되어 CTC 손실이 inf/NaN 이 되거나 잡음 학습이 됩니다",
        "why.sample_rate_mismatch": "오디오가 조용히 리샘플링되거나 잘못된 속도·음높이로 입력됩니다",
        "why.not_mono": "여분 채널이 특징 추출을 깨뜨리거나 예측할 수 없게 섞입니다",
        "why.too_long": "Whisper 는 30초 뒤를 조용히 잘라내서 라벨이 오디오와 맞지 않게 됩니다",
        "why.too_short": "안정적인 학습 샘플이 되기에는 오디오가 너무 짧습니다",
        "why.duration_mismatch": "매니페스트가 파일과 어긋나 길이 필터·배치가 잘못된 값을 씁니다",
        "why.char_rate_high": "오디오에 비해 글자가 너무 많음: 전사가 틀렸거나 어긋났습니다",
        "why.char_rate_low": "글자에 비해 오디오가 너무 김: 전사가 빠졌거나 무음입니다",
        "why.ctc_unalignable": "CTC 는 라벨마다 프레임이 하나 필요합니다(반복 글자 사이에 공백 포함). 모자라면 손실이 inf 가 됩니다",
        "why.label_tokens_over": "'size of tensor a (449) must match the size of tensor b (448)' 오류로 학습이 멈춥니다",
        "why.label_tokens_maybe_over": "같은 448 토큰 오류가 날 수 있습니다. --tokenizer 로 확인하세요",
        "why.oov_chars": "모르는 글자가 <unk> 가 되거나 CTC 어휘를 깨뜨립니다",
        "why.not_nfc": "분해된 글자는 조합형과 다르게 토큰화됩니다",
        "why.control_chars": "의도하지 않은 제어 문자가 라벨 안에 들어갑니다",
        "why.duplicate_audio": "중복 샘플이 과대 반영되고 검증 데이터로 새어 들어갈 수 있습니다",
        "why.repeated_path": "같은 클립이 두 번 올라 과대 반영됩니다",
        "why.clipping": "클리핑된 오디오는 모델에 왜곡을 가르칩니다",
        "why.mostly_silent": "음성이 거의 없어 라벨이 맞출 대상이 없습니다",
    },
}
