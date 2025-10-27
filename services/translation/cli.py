from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from common_schemas.models import ASRResponse, TranslateRequest
from app import runner_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translation-cli",
        description="Run the translation worker locally on serialized segments.",
    )
    parser.add_argument(
        "input",
        help="Path to a JSON file containing segments (ASR response or TranslateRequest-like payload).",
    )
    parser.add_argument(
        "--target-lang",
        required=True,
        help="Target language code (e.g., fr).",
    )
    parser.add_argument(
        "--source-lang",
        default=None,
        help="Optional source language code (e.g., en).",
    )
    parser.add_argument(
        "--model-key",
        default="facebook_m2m100",
        help="Model key configured in the translation registry (default: facebook_m2m100).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to store the translated JSON result. Prints to stdout if omitted.",
    )
    return parser


def _load_segments(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "segments" in payload:
        return payload
    if "aligned" in payload and isinstance(payload["aligned"], dict):
        return payload["aligned"]
    if "raw" in payload and isinstance(payload["raw"], dict):
        return payload["raw"]
    raise ValueError("Input JSON must contain either 'segments', 'raw', or 'aligned'.")


def run(args: argparse.Namespace) -> ASRResponse:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    segments_payload = _load_segments(payload)
    request = TranslateRequest(
        segments=[dict(seg) for seg in segments_payload.get("segments", [])],
        source_lang=args.source_lang or segments_payload.get("language"),
        target_lang=args.target_lang,
    )
    return runner_api.call_worker(args.model_key, request, ASRResponse)


def _emit(result: ASRResponse, output: Optional[Path]) -> None:
    payload = json.dumps(result.model_dump(), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"✅ Translation saved to {output}")
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
    except ValueError as exc:
        print(f"❌ Invalid input: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Translation failed: {exc}", file=sys.stderr)
        return 1

    _emit(result, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
