import asyncio
import random
import os
import sys, json, contextlib
import subprocess
import logging
import time
from pathlib import Path

import edge_tts
from edge_tts import VoicesManager

from common_schemas.models import TTSRequest, TTSResponse, SegmentAudioOut
from common_schemas.service_utils import get_service_logger


def convert_mp3_to_wav(src_mp3: Path, dst_wav: Path, sample_rate: int = 24000, channels: int = 1, logger: logging.Logger | None = None) -> None:
    # Requires ffmpeg in PATH
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_mp3), "-ar", str(sample_rate), "-ac", str(channels), str(dst_wav)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if logger:
        logger.info(
            "Converted MP3->WAV in %.2fs (%s -> %s).",
            time.perf_counter() - start,
            src_mp3.name,
            dst_wav.name,
        )

def pick_voice_name(voices: VoicesManager, lang: str, gender: str | None, seed_key: str, default_voice: str) -> str:

    if not lang:
        logger.warning("No language specified, falling back to default voice.")
        return default_voice
    
    # stable selection per seed_key
    rng = random.Random(seed_key)

    def choose(cands):
        return rng.choice(cands)["Name"] if cands else None

    # Try exact matches first
    cands = voices.find(Language=lang, Gender=gender) if gender else voices.find(Language=lang)
    name = choose(cands)
    if name:
        return name

    # Try any voice in lang
    cands = voices.find(Language=lang)
    name = choose(cands)
    if name:
        return name

    # If lang is "es", "fr" etc., try locales starting with lang-
    if len(lang) == 2:
        cands = [v for v in voices.voices if v.get("Locale", "").startswith(f"{lang}-")]
        name = choose(cands)
        if name:
            return name

    # Fallback to any gender in English
    cands = voices.find(Name=default_voice) or voices.find(Language="en")
    name = choose(cands) or default_voice
    return name

async def synthesize_all(req: TTSRequest, out_format: str, default_voice: str, gender: str | None, logger: logging.Logger, **kwargs) -> TTSResponse:
    out = TTSResponse()
    workspace_path = Path(req.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    voices = await VoicesManager.create()

    # keep voice consistent per speaker+lang
    speaker_voice_map: dict[tuple[str | None, str], str] = {}

    workspace_root = workspace_path.resolve()

    for i, segment in enumerate(req.segments):
        # Decide output file paths
        if segment.legacy_audio_path:
            final_path = Path(segment.legacy_audio_path)
            if not final_path.is_absolute():
                final_path = (workspace_path / final_path).resolve()
            else:
                final_path = final_path.resolve()
            try:
                final_path.relative_to(workspace_root)
            except ValueError as exc:
                raise RuntimeError(f"legacy_audio_path must reside inside workspace: {segment.legacy_audio_path}") from exc
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if out_format == "wav":
                wav_path = final_path
                mp3_path = final_path.with_suffix(".mp3")
            else:
                mp3_path = final_path
                wav_path = final_path.with_suffix(".wav")
        else:
            identifier = segment.segment_id or f"seg-{i}"
            mp3_path = workspace_path / f"{identifier}.mp3"
            wav_path = workspace_path / f"{identifier}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)

        # Choose voice
        key = (segment.speaker_id, segment.lang)
        voice_name = speaker_voice_map.get(key)
        if not voice_name:
            segment.lang = segment.lang or req.language
            # Default gender (Male or Female), set gender=None for any
            voice_name = pick_voice_name(voices, segment.lang, gender, seed_key=f"{segment.speaker_id}-{segment.lang}", default_voice=default_voice)
            speaker_voice_map[key] = voice_name

        # Synthesize to MP3
        seg_start = time.perf_counter()
        communicate = edge_tts.Communicate(segment.text, voice_name, **kwargs)
        await communicate.save(str(mp3_path))

        # Post-process format
        if out_format == "wav":
            convert_mp3_to_wav(mp3_path, wav_path, logger=logger)
            audio_url = str(wav_path)
            # remove mp3
            with contextlib.suppress(Exception):
                mp3_path.unlink()
        else:
            audio_url = str(mp3_path)

        logger.info(
            "Synthesized segment %d voice=%s lang=%s duration=%.2fs",
            i,
            voice_name,
            segment.lang,
            time.perf_counter() - seg_start,
        )
        out.segments.append(SegmentAudioOut(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            audio_prompt_url=segment.audio_prompt_url,
            audio_url=audio_url,
            speaker_id=segment.speaker_id,
            lang=segment.lang,
            segment_id=segment.segment_id,
        ))
    return out

if __name__ == "__main__":
    req = TTSRequest(**json.loads(sys.stdin.read()))
    params = (req.extra or {}).get("params", {})
    general_cfg = params.get("general", {})
    communicate_cfg = params.get("communicate", {})
    out_format = general_cfg.get("out_format", "wav")
    default_voice = general_cfg.get("default_voice", "en-US-AriaNeural")
    gender = general_cfg.get("gender", None)
    log_level = params.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level, logging.INFO)
    logger = get_service_logger("tts.edge_tts", log_level)

    with contextlib.redirect_stdout(sys.stderr):
        run_start = time.perf_counter()
        logger.info(
            "Starting Edge TTS run segments=%d workspace=%s format=%s default_voice=%s gender=%s",
            len(req.segments or []),
            req.workspace,
            out_format,
            default_voice,
            gender,
        )
        try:
            result = asyncio.run(
                synthesize_all(
                    req,
                    out_format=out_format,
                    default_voice=default_voice,
                    gender=gender,
                    logger=logger,
                    **communicate_cfg,
                )
            )
        except Exception as e:
            # Surface a clean error to orchestrator
            sys.stderr.write(f"[edge-tts] Error: {e}\n")
            raise
        else:
            logger.info(
                "Completed Edge TTS run in %.2fs (segments=%d).",
                time.perf_counter() - run_start,
                len(result.segments),
            )
    sys.stdout.write(result.model_dump_json() + "\n")
    sys.stdout.flush()
