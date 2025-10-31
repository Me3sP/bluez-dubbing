from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from common_schemas.models import ASRRequest, ASRResponse
from app import runner_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr-cli",
        description="Run WhisperX runner 0 followed by runner 1 (alignment + optional diarization).",
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
        "--language-hint",
        default=None,
        help="Optional language hint for WhisperX (e.g., en).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers for diarization.",
    )
    parser.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable diarization during runner 1 (default: enabled).",
    )
    parser.add_argument(
        "--raw-output-json",
        type=Path,
        help="Optional path to store the raw (runner 0) JSON result.",
    )
    parser.add_argument(
        "--aligned-output-json",
        type=Path,
        help="Optional path to store the aligned (runner 1) JSON result.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to store a combined JSON payload containing both raw and aligned results. Prints to stdout if omitted.",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[ASRResponse, ASRResponse]:
    audio_path = args.audio
    if not str(audio_path).startswith(("http://", "https://")):
        path_obj = Path(audio_path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")
        audio_path = str(path_obj)

    request = ASRRequest(
        audio_url=audio_path,
        language_hint=args.language_hint,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    raw_result = runner_api.call_worker(args.model_key, request, ASRResponse, runner_index=0)

    raw_result.extra = dict(raw_result.extra or {})
    if args.min_speakers is not None:
        raw_result.extra["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        raw_result.extra["max_speakers"] = args.max_speakers

    align_payload = ASRResponse(**raw_result.model_dump())
    align_payload.extra = dict(align_payload.extra or {})
    align_payload.extra["enable_diarization"] = args.diarize

    aligned_result = runner_api.call_worker(args.model_key, align_payload, ASRResponse, runner_index=1)
    return raw_result, aligned_result


def _emit(raw: ASRResponse, aligned: ASRResponse, args: argparse.Namespace) -> None:
    if args.raw_output_json:
        args.raw_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output_json.write_text(json.dumps(raw.model_dump(), indent=2), encoding="utf-8")
        print(f"✅ Raw ASR result saved to {args.raw_output_json}")

    if args.aligned_output_json:
        args.aligned_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.aligned_output_json.write_text(json.dumps(aligned.model_dump(), indent=2), encoding="utf-8")
        print(f"✅ Aligned ASR result saved to {args.aligned_output_json}")

    combined = {"raw": raw.model_dump(), "aligned": aligned.model_dump()}
    payload = json.dumps(combined, indent=2)

    output_path: Optional[Path] = args.output_json
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"✅ Combined ASR result saved to {output_path}")
    else:
        print(payload)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        raw, aligned = run(args)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"❌ ASR failed: {exc}", file=sys.stderr)
        return 1

    _emit(raw, aligned, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
