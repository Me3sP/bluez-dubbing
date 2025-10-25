import asyncio
import random
import os
import sys, json, contextlib
import subprocess
from pathlib import Path

import edge_tts
from edge_tts import VoicesManager

from common_schemas.models import TTSRequest, TTSResponse, SegmentAudioOut


def convert_mp3_to_wav(src_mp3: Path, dst_wav: Path, sample_rate: int = 24000, channels: int = 1) -> None:
    # Requires ffmpeg in PATH
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_mp3), "-ar", str(sample_rate), "-ac", str(channels), str(dst_wav)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def pick_voice_name(voices: VoicesManager, lang: str, gender: str | None, seed_key: str, default_voice: str) -> str:
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

async def synthesize_all(req: TTSRequest, out_format: str, default_voice: str, gender: str | None, **kwargs) -> TTSResponse:
    out = TTSResponse()
    workspace_path = Path(req.workspace)
    voices = await VoicesManager.create()

    # keep voice consistent per speaker+lang
    speaker_voice_map: dict[tuple[str | None, str], str] = {}

    for i, segment in enumerate(req.segments):
        # Decide output file paths
        base_dir = workspace_path / "tts"
        base_dir.mkdir(parents=True, exist_ok=True)
        mp3_path = base_dir / f"seg-{i}.mp3"
        wav_path = base_dir / f"seg-{i}.wav"

        # Choose voice
        key = (segment.speaker_id, segment.lang)
        voice_name = speaker_voice_map.get(key)
        if not voice_name:
            # Default gender (Male or Female), set gender=None for any
            voice_name = pick_voice_name(voices, segment.lang, gender, seed_key=f"{segment.speaker_id}-{segment.lang}", default_voice=default_voice)
            speaker_voice_map[key] = voice_name

        # Synthesize to MP3
        communicate = edge_tts.Communicate(segment.text, voice_name, **kwargs)
        await communicate.save(str(mp3_path))

        # Post-process format
        if out_format == "wav":
            convert_mp3_to_wav(mp3_path, wav_path)
            audio_url = str(wav_path)
            # remove mp3
            with contextlib.suppress(Exception):
                mp3_path.unlink()
        else:
            audio_url = str(mp3_path)

        out.segments.append(SegmentAudioOut(
            start=segment.start,
            end=segment.end,
            audio_url=audio_url,
            speaker_id=segment.speaker_id,
            lang=segment.lang
        ))
    return out

if __name__ == "__main__":
    req = TTSRequest(**json.loads(sys.stdin.read()))
    extra = req.extra.get("params", {})
    out_format = extra.get("general", {}).get("out_format", "wav")
    default_voice = extra.get("general", {}).get("default_voice", "en-US-AriaNeural")
    gender = extra.get("general", {}).get("gender", None)

    with contextlib.redirect_stdout(sys.stderr):
        try:
            result = asyncio.run(synthesize_all(req, out_format=out_format, default_voice=default_voice, gender=gender, **extra.get("communicate", {})))
        except Exception as e:
            # Surface a clean error to orchestrator
            sys.stderr.write(f"[edge-tts] Error: {e}\n")
            raise
    sys.stdout.write(result.model_dump_json() + "\n")
    sys.stdout.flush()