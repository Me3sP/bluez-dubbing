from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from common_schemas.models import ASRResponse
from app import runner_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr-cli-1",
        description="Run WhisperX runner 1 (alignment + optional diarization) locally.",
    )
    parser.add_argument(
        "input",
        help="Path to a JSON file containing an ASRResponse payload (typically the output of cli_0).",
    )
    parser.add_argument(
        "--model-key",
        default="whisperx",
        help="Model key configured in the ASR registry (default: whisperx).",
    )
    parser.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable diarization during alignment (default: enabled).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to store the aligned JSON result. Prints to stdout if omitted.",
    )
    return parser


def run(args: argparse.Namespace) -> ASRResponse:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    request = ASRResponse(**payload)
    request.extra = dict(request.extra or {})
    request.extra["enable_diarization"] = args.diarize
    return runner_api.call_worker(args.model_key, request, ASRResponse, runner_index=1)


def _emit(result: ASRResponse, output: Optional[Path]) -> None:
    payload = json.dumps(result.model_dump(), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"✅ Aligned ASR result saved to {output}")
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
        print(f"❌ Runner 1 failed: {exc}", file=sys.stderr)
        return 1

    _emit(result, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
