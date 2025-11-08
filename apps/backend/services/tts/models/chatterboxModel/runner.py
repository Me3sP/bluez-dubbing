from __future__ import annotations
import contextlib
import json
import sys
import threading
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from common_schemas.models import SegmentAudioOut, TTSRequest, TTSResponse

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, int]] = {}
_MODEL_LOCK = threading.Lock()
_LOGGER = None


def _get_logger(log_level) -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("tts.chatterbox")
        if not _LOGGER.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            _LOGGER.addHandler(handler)
        _LOGGER.setLevel(log_level)
        _LOGGER.propagate = False
    return _LOGGER


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, device: str, log_level: int):
    logger = _get_logger(log_level)
    key = (model_name, device)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached:
            logger.debug("Using cached chatterbox model=%s device=%s.", model_name, device)
            return cached

        load_start = time.perf_counter()
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        sample_rate = model.sr

        _MODEL_CACHE[key] = (model, sample_rate)
        logger.info(
            "Loaded chatterbox model=%s device=%s in %.2fs.",
            model_name,
            device,
            time.perf_counter() - load_start,
        )
        return _MODEL_CACHE[key]


def _synthesize(req: TTSRequest) -> TTSResponse:
    log_level = req.extra.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level, logging.INFO)
    logger = _get_logger(log_level)
    run_start = time.perf_counter()
    workspace_path = Path(req.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    model_name = (req.extra or {}).get("model_name", "chatterbox_multilingual")
    device = _device()
    model, sample_rate = _load_model(model_name, device, log_level)

    out = TTSResponse()
    generation_kwargs = (req.extra or {}).get("generate", {})
    logger.info(
        "Starting chatterbox synthesis segments=%d workspace=%s model=%s device=%s",
        len(req.segments or []),
        req.workspace,
        model_name,
        device,
    )

    for i, segment in enumerate(req.segments):
        seg_start = time.perf_counter()
        audio_prompt = segment.audio_prompt_url or None
        if audio_prompt and not Path(audio_prompt).exists():
            raise FileNotFoundError(f"Audio prompt file not found: {audio_prompt}")

        if segment.legacy_audio_path:
            output_file = Path(segment.legacy_audio_path)
            if not output_file.is_absolute():
                output_file = (workspace_path / output_file).resolve()
            else:
                output_file = output_file.resolve()
            try:
                output_file.relative_to(workspace_path.resolve())
            except ValueError as exc:  # ensure overwrite stays inside workspace
                raise RuntimeError(f"legacy_audio_path must reside inside workspace: {segment.legacy_audio_path}") from exc
        else:
            identifier = segment.segment_id or f"seg-{i}"
            output_file = workspace_path / f"{identifier}.wav"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        wav = model.generate(
            segment.text,
            language_id=segment.lang,
            audio_prompt_path=audio_prompt,
            **generation_kwargs,
        )

        ta.save(str(output_file), wav, sample_rate)

        out.segments.append(
            SegmentAudioOut(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                audio_url=str(output_file),
                speaker_id=segment.speaker_id,
                lang=segment.lang,
                sample_rate=sample_rate,
                segment_id=segment.segment_id,
            )
        )
        logger.info(
            "Generated segment %d lang=%s prompt=%s duration=%.2fs",
            i,
            segment.lang,
            bool(audio_prompt),
            time.perf_counter() - seg_start,
        )

    logger.info(
        "Completed chatterbox synthesis in %.2fs (segments=%d).",
        time.perf_counter() - run_start,
        len(out.segments),
    )
    return out


def _run_once():
    req = TTSRequest(**json.loads(sys.stdin.read()))
    with contextlib.redirect_stdout(sys.stderr):
        out = _synthesize(req)
    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _run_once()
