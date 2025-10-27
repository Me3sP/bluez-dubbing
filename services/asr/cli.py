from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from common_schemas.models import ASRRequest, ASRResultWrapper
from app import runner_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr-cli",
        description="Run the ASR worker locally and dump the transcription result.",
    )
    parser.add_argument(
        "audio",
        help="Path or URL to the audio file to transcribe.",
    )
    parser.add_argument(
        "--model-key",
        default="whisperx",
        help="Model key configured in the ASR registry (default: whisperx).",
    )
    parser.add_argument(
        "--runner-index",
        type=int,
        default=0,
        help="Runner index to use (default: 0).",
    )
    parser.add_argument(
        "--language-hint",
        default=None,
        help="Optional language hint for WhisperX (e.g., en).",
    )
    parser.add_argument(
        "--allow-short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow diarization for short clips (default: enabled).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to store the JSON result. Prints to stdout if omitted.",
    )
    return parser


def run(args: argparse.Namespace) -> ASRResultWrapper:
    audio_path = args.audio
    if not str(audio_path).startswith(("http://", "https://")):
        path_obj = Path(audio_path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")
        audio_path = str(path_obj)

    request = ASRRequest(
        audio_url=audio_path,
        language_hint=args.language_hint,
        allow_short=args.allow_short,
    )
    return runner_api.call_worker(args.model_key, request, ASRResultWrapper, args.runner_index)


def _emit(result: ASRResultWrapper, output: Optional[Path]) -> None:
    payload = json.dumps(result.model_dump(), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"✅ ASR result saved to {output}")
    else:
        print(payload)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run(args)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"❌ ASR failed: {exc}", file=sys.stderr)
        return 1

    _emit(result, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
