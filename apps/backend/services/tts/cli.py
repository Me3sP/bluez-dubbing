from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common_schemas.models import SegmentAudioIn, TTSRequest, TTSResponse
from app import runner_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-cli",
        description="Run the TTS worker locally to synthesize speech segments.",
    )
    parser.add_argument(
        "input",
        help="Path to a JSON file containing translated segments (ASRResponse-like).",
    )
    parser.add_argument(
        "--model-key",
        default="chatterbox",
        help="Model key configured in the TTS registry (default: chatterbox).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("./tts_workspace"),
        help="Workspace directory for generated audio (default: ./tts_workspace).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code override for TTS.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to store the TTS JSON result. Prints to stdout if omitted.",
    )
    return parser


def _load_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "segments" in payload:
        return payload["segments"]
    if "aligned" in payload and isinstance(payload["aligned"], dict):
        return payload["aligned"].get("segments", [])
    raise ValueError("Input JSON must contain 'segments' or an 'aligned' section with segments.")


def run(args: argparse.Namespace) -> TTSResponse:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    segments_data = _load_segments(payload)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    segments = [SegmentAudioIn(**seg) for seg in segments_data]
    request = TTSRequest(segments=segments, workspace=str(workspace), language=args.language)
    return runner_api.call_worker(args.model_key, request, TTSResponse)


def _emit(result: TTSResponse, output: Optional[Path]) -> None:
    payload = json.dumps(result.model_dump(), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"✅ TTS result saved to {output}")
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
        print(f"❌ TTS failed: {exc}", file=sys.stderr)
        return 1

    _emit(result, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
