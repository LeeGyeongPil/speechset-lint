# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-11-17

### Changed
- Release preparation only; no behavior change.

## [0.2.0] - 2026-11-17

### Changed
- Require Python 3.11+ (3.10 is EOL); bump filelock to >=4.2.1 (security).

## [0.1.0] - 2026-11-17

### Added
- `speechset-lint check` for NeMo JSONL, Hugging Face audiofolder (`metadata.csv` / `metadata.jsonl`) and LJSpeech manifests, with format auto-detection.
- Header-only audio checks (standard `wave` first, `soundfile` fallback): missing or unreadable files, sample rate, channels, length, duration mismatch.
- Transcript checks: empty text, characters per second, NFC, control characters, characters outside a `vocab.json` alphabet, exact label token counts from a `tokenizer.json` or a UTF-8 byte upper bound, CTC alignability.
- Optional signal checks (`--signal`): clipping and mostly silent clips; duplicate audio by SHA-1.
- `whisper`, `ctc` and `tts` presets.
- Table and `--json` reports, exit codes 0/1/2, `--fail-on`, `--write-clean` for a filtered manifest in the same format.
- `speechset-lint demo` with a synthetic practice dataset.
- English and Korean screens (`--lang`, `SPEECHSET_LINT_LANG`, locale variables).
- Python API: `check()`, `load_manifest()`, `write_clean()`, `PRESETS`.

[Unreleased]: https://github.com/LeeGyeongPil/speechset-lint/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/LeeGyeongPil/speechset-lint/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/LeeGyeongPil/speechset-lint/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/LeeGyeongPil/speechset-lint/releases/tag/v0.1.0
