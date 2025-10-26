from __future__ import annotations
import contextlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from common_schemas.models import SegmentAudioOut, TTSRequest, TTSResponse

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, int]] = {}
_MODEL_LOCK = threading.Lock()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, device: str):
    key = (model_name, device)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached:
            return cached

        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        sample_rate = model.sr

        _MODEL_CACHE[key] = (model, sample_rate)
        return _MODEL_CACHE[key]


def _synthesize(req: TTSRequest) -> TTSResponse:
    workspace_path = Path(req.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    model_name = (req.extra or {}).get("model_name", "chatterbox_multilingual")
    device = _device()
    model, sample_rate = _load_model(model_name, device)

    out = TTSResponse()
    generation_kwargs = (req.extra or {}).get("generate", {})

    for i, segment in enumerate(req.segments):
        audio_prompt = segment.audio_prompt_url or None
        if audio_prompt and not Path(audio_prompt).exists():
            raise FileNotFoundError(f"Audio prompt file not found: {audio_prompt}")

        wav = model.generate(
            segment.text,
            language_id=segment.lang,
            audio_prompt_path=audio_prompt,
            **generation_kwargs,
        )

        output_file = workspace_path / "tts" / f"seg-{i}.wav"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_file), wav, sample_rate)

        out.segments.append(
            SegmentAudioOut(
                start=segment.start,
                end=segment.end,
                audio_url=str(output_file),
                speaker_id=segment.speaker_id,
                lang=segment.lang,
                sample_rate=sample_rate,
            )
        )

    return out


def _run_once():
    req = TTSRequest(**json.loads(sys.stdin.read()))
    with contextlib.redirect_stdout(sys.stderr):
        out = _synthesize(req)
    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()


def _loop():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        req = TTSRequest(**json.loads(line))
        with contextlib.redirect_stdout(sys.stderr):
            out = _synthesize(req)
        sys.stdout.write(out.model_dump_json() + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if os.getenv("PERSISTENT_WORKER") == "1":
        _loop()
    else:
        _run_once()
