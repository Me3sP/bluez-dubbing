import asyncio
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query

from common_schemas.models import (
    ASRRequest,
    ASRResponse,
    SegmentAudioIn,
    TTSRequest,
    TTSResponse,
    TranslateRequest,
)
from common_schemas.utils import (
    alignerWrapper,
    attach_segment_audio_clips,
    map_by_text_overlap,
)
from media_processing.audio_processing import (
    concatenate_audio,
    get_audio_duration,
    overlay_on_background,
    trim_audio_with_vad,
)
from media_processing.final_pass import final
from media_processing.subtitles_handling import STYLE_PRESETS, build_subtitles_from_asr_result
from preprocessing.media_separation import (
    filter_supported_models_grouped,
    get_non_vocals_stem,
    separation,
)

logger = logging.getLogger("bluez.orchestrator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(title="orchestrator")

ASR_URL = "http://localhost:8001/v1/transcribe"
TR_URL = "http://localhost:8002/v1/translate"
TTS_URL = "http://localhost:8003/v1/synthesize"

BASE = Path(__file__).resolve().parents[3]  # bluez-dubbing root
OUTS = BASE / "outs"
SEPARATION_CACHE = BASE / "cache" / "audio_separation"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")


@app.on_event("startup")
async def startup_event() -> None:
    timeout = httpx.Timeout(connect=10.0, read=1200.0, write=10.0, pool=None)
    app.state.http_client = httpx.AsyncClient(timeout=timeout)
    OUTS.mkdir(parents=True, exist_ok=True)
    SEPARATION_CACHE.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    client = getattr(app.state, "http_client", None)
    if client:
        await client.aclose()


def get_http_client() -> httpx.AsyncClient:
    client = getattr(app.state, "http_client", None)
    if client is None:
        raise RuntimeError("HTTP client not initialized; startup event did not run.")
    return client


class StepTimer:
    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def time(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.timings[label] = duration
            logger.info("%s completed in %.2fs", label, duration)


@dataclass
class WorkspaceManager:
    workspace: Path
    workspace_id: str
    persist_intermediate: bool

    @classmethod
    def create(cls, base: Path, persist_intermediate: bool) -> "WorkspaceManager":
        workspace_id = str(uuid.uuid4())
        workspace = base / workspace_id
        workspace.mkdir(parents=True, exist_ok=True)
        return cls(workspace=workspace, workspace_id=workspace_id, persist_intermediate=persist_intermediate)

    def ensure_dir(self, relative: str | Path) -> Path:
        path = self.workspace / Path(relative)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file_path(self, relative: str | Path) -> Path:
        path = self.workspace / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def maybe_dump_json(self, relative: str | Path, payload: Any, *, force: bool = False) -> str:
        path = self.file_path(relative)
        if self.persist_intermediate or force:
            path.write_text(json.dumps(payload, indent=2))
            return str(path)
        return ""


async def run_in_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def run_subprocess(cmd: List[str], *, description: str) -> subprocess.CompletedProcess:
    def _run():
        return subprocess.run(cmd, check=True, capture_output=True, text=True)

    try:
        return await asyncio.to_thread(_run)
    except subprocess.CalledProcessError as exc:
        logger.error("%s\nSTDOUT: %s\nSTDERR: %s", description, exc.stdout, exc.stderr)
        raise HTTPException(500, f"{description}: {exc.stderr}") from exc


def resolve_media_path(video_url: str) -> str:
    if video_url.startswith(("http://", "https://")):
        return video_url

    video_path = Path(video_url)
    if not video_path.is_absolute():
        video_path = BASE / video_url
    if not video_path.exists():
        raise HTTPException(404, f"Media file not found: {video_url}")
    return str(video_path)


async def extract_audio_to_workspace(source_url: str, raw_audio_path: Path) -> None:
    if source_url.endswith(VIDEO_EXTENSIONS):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_url,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(raw_audio_path),
        ]
        await run_subprocess(cmd, description="FFmpeg audio extraction failed")
    elif source_url.endswith(AUDIO_EXTENSIONS):
        if source_url.endswith(".wav"):
            await run_in_thread(shutil.copy, source_url, raw_audio_path)
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                source_url,
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(raw_audio_path),
            ]
            await run_subprocess(cmd, description="FFmpeg audio conversion failed")
    else:
        raise HTTPException(400, f"Unsupported file format: {source_url}")

    if not raw_audio_path.exists() or raw_audio_path.stat().st_size == 0:
        raise HTTPException(500, "Audio extraction produced an empty file")


def separation_cache_key(raw_audio_path: Path, sep_model: str) -> str:
    stat = raw_audio_path.stat()
    key = f"{raw_audio_path.stat().st_size}-{stat.st_mtime_ns}-{sep_model}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


async def load_cached_separation(cache_key: str, vocals_target: Path, background_target: Path) -> bool:
    cache_dir = SEPARATION_CACHE / cache_key
    vocals_cache = cache_dir / "vocals.wav"
    background_cache = cache_dir / "background.wav"
    if not vocals_cache.exists() or not background_cache.exists():
        return False

    await run_in_thread(shutil.copy, vocals_cache, vocals_target)
    await run_in_thread(shutil.copy, background_cache, background_target)
    return True


async def store_separation_cache(cache_key: str, vocals_source: Path, background_source: Path) -> None:
    cache_dir = SEPARATION_CACHE / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    await run_in_thread(shutil.copy, vocals_source, cache_dir / "vocals.wav")
    await run_in_thread(shutil.copy, background_source, cache_dir / "background.wav")


async def maybe_run_audio_separation(
    workspace: WorkspaceManager,
    raw_audio_path: Path,
    sep_model: str,
    audio_sep: bool,
    dubbing_strategy: str,
) -> Tuple[Optional[Path], Optional[Path], str]:
    if not audio_sep and dubbing_strategy != "full_replacement":
        return None, None, dubbing_strategy

    vocals_path = workspace.file_path("preprocessing/vocals.wav")
    background_path = workspace.file_path("preprocessing/background.wav")
    cache_key = separation_cache_key(raw_audio_path, sep_model)

    if await load_cached_separation(cache_key, vocals_path, background_path):
        logger.info("Loaded separated stems from cache")
        return vocals_path, background_path, dubbing_strategy

    model_file_dir = BASE / "models_cache" / "audio-separator-models" / Path(sep_model).stem
    logger.info("Starting audio separation with model %s", sep_model)
    try:
        await run_in_thread(
            separation,
            input_file=str(raw_audio_path),
            output_dir=str(vocals_path.parent),
            model_filename=sep_model,
            output_format="WAV",
            custom_output_names={"vocals": "vocals", get_non_vocals_stem(sep_model): "background"},
            model_file_dir=str(model_file_dir),
        )
    except ValueError as exc:
        logger.error("Audio separation failed with %s", exc)
        logger.error("Supported models:\n%s", json.dumps(filter_supported_models_grouped(), indent=2))
        return None, None, "default"

    if not vocals_path.exists() or not background_path.exists():
        logger.warning("Separation completed but expected stems are missing; falling back to default dubbing strategy")
        return None, None, "default"

    await store_separation_cache(cache_key, vocals_path, background_path)
    return vocals_path, background_path, dubbing_strategy


async def run_asr_step(
    client: httpx.AsyncClient,
    raw_audio_path: Path,
    asr_model: str,
    source_lang: Optional[str],
    allow_short: bool,
) -> Tuple[ASRResponse, ASRResponse]:
    asr_req = ASRRequest(
        audio_url=str(raw_audio_path),
        language_hint=source_lang if source_lang else None,
        allow_short=allow_short,
    )
    response = await client.post(ASR_URL, params={"model_key": asr_model}, json=asr_req.model_dump())
    if response.status_code != 200:
        raise HTTPException(500, f"ASR failed: {response.text}")

    response_data = response.json()
    if not isinstance(response_data, dict) or "raw" not in response_data or "aligned" not in response_data:
        raise HTTPException(500, f"ASR returned unexpected payload: {list(response_data.keys())}")

    return ASRResponse(**response_data["raw"]), ASRResponse(**response_data["aligned"])


async def run_translation_step(
    client: httpx.AsyncClient,
    tr_model: str,
    segments: List[Dict[str, Any]],
    source_lang: Optional[str],
    target_lang: Optional[str],
) -> ASRResponse:
    tr_req = TranslateRequest(
        segments=segments,
        source_lang=source_lang if source_lang else None,
        target_lang=target_lang if target_lang else None,
    )
    response = await client.post(TR_URL, params={"model_key": tr_model}, json=tr_req.model_dump())
    if response.status_code != 200:
        raise HTTPException(500, f"Translation failed: {response.text}")
    return ASRResponse(**response.json())


async def align_translation_segments(
    tr_result: ASRResponse,
    raw_asr_result: ASRResponse,
    aligned_asr_result: ASRResponse,
    translation_strategy: str,
    target_lang: str,
) -> ASRResponse:
    mappings = map_by_text_overlap(raw_asr_result.model_dump()["segments"], aligned_asr_result.model_dump()["segments"])
    for idx, tr in zip(mappings.keys(), tr_result.segments):
        mappings[idx]["full_text"] = tr.text

    return alignerWrapper(mappings, translation_strategy, target_lang, max_look_distance=3, verbose=True)


async def synthesize_tts(
    client: httpx.AsyncClient,
    tts_model: str,
    tr_result: ASRResponse,
    target_lang: str,
    workspace_path: Path,
) -> TTSResponse:
    tts_segments = [
        SegmentAudioIn(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker_id=seg.speaker_id,
            lang=tr_result.language or target_lang,
            audio_prompt_url=seg.audio_url if seg.speaker_id else None,
        )
        for seg in tr_result.segments
    ]

    tts_req = TTSRequest(segments=tts_segments, workspace=str(workspace_path), language=target_lang)
    response = await client.post(TTS_URL, params={"model_key": tts_model}, json=tts_req.model_dump())
    if response.status_code != 200:
        raise HTTPException(500, f"TTS failed: {response.text}")
    return TTSResponse(**response.json())


async def trim_tts_segments(tts_result: TTSResponse, vad_dir: Path) -> TTSResponse:
    vad_dir.mkdir(parents=True, exist_ok=True)
    trimmed_segments = []
    for idx, seg in enumerate(tts_result.segments):
        original_audio = Path(seg.audio_url)
        trimmed_audio = vad_dir / f"trimmed_{idx}_{original_audio.stem}.wav"
        try:
            _, output_path = await run_in_thread(
                trim_audio_with_vad,
                audio_path=seg.audio_url,
                output_path=trimmed_audio,
                several_seg=False,
            )
            seg.audio_url = str(output_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VAD trimming failed for segment %s: %s", idx, exc)
        trimmed_segments.append(seg)
    tts_result.segments = trimmed_segments
    return tts_result


async def concatenate_segments(
    tts_segments: List[Dict[str, Any]],
    output_file: Path,
    target_duration: float,
    translation_segments: List[Dict[str, Any]],
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    return await run_in_thread(
        concatenate_audio,
        segments=tts_segments,
        output_file=str(output_file),
        target_duration=target_duration,
        alpha=0.25,
        min_dur=0.4,
        translation_segments=translation_segments,
    )


async def overlay_segments_on_background(
    segments: List[Dict[str, Any]],
    background_path: Path,
    output_path: Path,
    sophisticated: bool,
    speech_track: Path,
) -> None:
    await run_in_thread(
        overlay_on_background,
        segments,
        background_path=str(background_path),
        output_path=str(output_path),
        ducking_db=0.0,
        sophisticated=sophisticated,
        speech_track=str(speech_track),
    )


async def align_dubbed_audio(
    client: httpx.AsyncClient,
    asr_model: str,
    tr_result: ASRResponse,
    translation_segments: Optional[List[Dict[str, Any]]],
    final_audio_path: Path,
) -> ASRResponse:
    tr_result.audio_url = str(final_audio_path)
    tr_dict = tr_result.model_dump()
    if translation_segments is not None:
        tr_dict["segments"] = translation_segments

    response = await client.post(
        ASR_URL,
        params={"model_key": asr_model, "runner_index": 1},
        json=tr_dict,
    )
    if response.status_code != 200:
        raise HTTPException(500, f"Second alignment failed: {response.text}")
    return ASRResponse(**response.json()["aligned"])


async def finalize_media(
    video_path: str,
    audio_path: Optional[Path],
    dubbed_path: Path,
    output_path: Optional[Path],
    subtitle_path: Optional[Path],
    style: Optional[Any],
    mobile_optimized: bool,
    dubbing_strategy: str,
) -> None:
    await run_in_thread(
        final,
        video_path=video_path,
        audio_path=str(audio_path) if audio_path else "",
        dubbed_path=str(dubbed_path),
        output_path=str(output_path) if output_path else "",
        subtitle_path=str(subtitle_path) if subtitle_path else None,
        sub_style=style,
        mobile_optimized=mobile_optimized,
        dubbing_strategy=dubbing_strategy,
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/v1/dub")
async def dub(
    video_url: str,
    target_work: str = Query(
        "dub",
        description="Target work type, e.g., 'dub': for full dubbing or 'sub': for subtitles only",
    ),
    target_lang: Optional[str] = None,
    source_lang: Optional[str] = "en",
    sep_model: str = Query("melband_roformer_big_beta5e.ckpt"),
    asr_model: str = Query("whisperx"),
    tr_model: str = Query("facebook_m2m100"),
    tts_model: str = Query("chatterbox"),
    audio_sep: bool = Query(True, description="Whether to perform audio source separation"),
    perform_vad_trimming: bool = Query(True, description="Whether to perform VAD-based silence trimming after TTS"),
    translation_strategy: str = Query(
        "default",
        description="Translation strategy to use: either translate directly over the short ASR aligned segments or translate the full text and then align the translated result after",
    ),
    dubbing_strategy: str = Query(
        "default",
        description="Dubbing strategy to use, either translation over (original audio ducked) or full replacement",
    ),
    sophisticated_dub_timing: bool = Query(
        True,
        description="Whether to use sophisticated timing for full replacement dubbing strategy",
    ),
    subtitle_style: Optional[str] = Query(
        None,
        description="Subtitle style preset: default, minimal, bold, netflix",
    ),
    persist_intermediate: bool = Query(
        True,
        description="Persist intermediate artifacts (disable for lower latency and disk usage)",
    ),
):
    """
    Complete dubbing pipeline orchestrator.
    """
    if target_work == "sub" and subtitle_style is None:
        subtitle_style = "default_mobile"

    workspace = WorkspaceManager.create(OUTS, persist_intermediate)
    step_timer = StepTimer()
    client = get_http_client()

    resolved_video_url = resolve_media_path(video_url)
    raw_audio_path = workspace.file_path("preprocessing/raw_audio.wav")

    try:
        with step_timer.time("extract_audio"):
            await extract_audio_to_workspace(resolved_video_url, raw_audio_path)

        vocals_path: Optional[Path] = None
        background_path: Optional[Path] = None

        if target_work != "sub":
            with step_timer.time("audio_separation"):
                vocals_path, background_path, dubbing_strategy = await maybe_run_audio_separation(
                    workspace,
                    raw_audio_path,
                    sep_model,
                    audio_sep,
                    dubbing_strategy,
                )

        allow_short = translation_strategy.split("_")[0] != "long"
        with step_timer.time("asr"):
            raw_asr_result, aligned_asr_result = await run_asr_step(
                client, raw_audio_path, asr_model, source_lang, allow_short
            )

        source_lang = source_lang or raw_asr_result.language

        asr_raw_path = workspace.maybe_dump_json("asr/asr_0_result.json", raw_asr_result.model_dump())
        asr_aligned_path = workspace.maybe_dump_json("asr/asr_0_aligned_result.json", aligned_asr_result.model_dump())

        srt_path_0 = ""
        vtt_path_0 = ""
        subtitles_dir: Optional[Path] = None

        if subtitle_style is not None:
            with step_timer.time("subtitles_original"):
                subtitles_dir = workspace.ensure_dir("subtitles")
                srt_path_0, vtt_path_0 = build_subtitles_from_asr_result(
                    data=aligned_asr_result.model_dump(),
                    output_dir=subtitles_dir,
                    custom_name="original",
                    formats=["srt", "vtt"],
                    mobile_mode=subtitle_style.split("_")[-1] == "mobile",
                )

        tr_result: Optional[ASRResponse] = None
        tr_out_path = ""
        if target_lang is not None:
            segments_for_translation = (
                raw_asr_result.model_dump()["segments"]
                if translation_strategy.split("_")[0] == "long"
                else aligned_asr_result.model_dump()["segments"]
            )
            with step_timer.time("translation"):
                tr_result = await run_translation_step(
                    client,
                    tr_model,
                    segments_for_translation,
                    source_lang,
                    target_lang,
                )
            tr_out_path = workspace.maybe_dump_json("translation/translation_result.json", tr_result.model_dump())

        tr_aligned_origin_path = ""
        if (
            tr_result
            and translation_strategy.split("_")[0] == "long"
            and len(aligned_asr_result.segments) > 1
            and target_lang is not None
        ):
            with step_timer.time("translation_alignment"):
                tr_result = await align_translation_segments(
                    tr_result,
                    raw_asr_result,
                    aligned_asr_result,
                    translation_strategy,
                    target_lang,
                )
            tr_aligned_origin_path = workspace.maybe_dump_json(
                "translation/translation_aligned_W_origin_result.json", tr_result.model_dump()
            )

        tr_aligned_tts_path = ""
        tts_out_path = ""
        speech_track: Optional[Path] = None
        final_audio_path: Optional[Path] = None
        srt_path_1 = srt_path_0
        vtt_path_1 = vtt_path_0

        if target_work == "sub":
            if target_lang is not None and tr_result and subtitles_dir:
                with step_timer.time("subtitles_translation"):
                    srt_path_1, vtt_path_1 = build_subtitles_from_asr_result(
                        data=tr_result.model_dump(),
                        output_dir=subtitles_dir,
                        custom_name=f"dubbed_{target_lang}",
                        formats=["srt", "vtt"],
                        mobile_mode=subtitle_style.split("_")[-1] == "mobile",
                    )
        else:
            if target_lang is None:
                raise HTTPException(400, "Target language must be specified for dubbing")

            tr_result = tr_result or aligned_asr_result
            tr_result.audio_url = str(vocals_path) if vocals_path else str(raw_audio_path)

            prompt_audio_dir = workspace.ensure_dir("prompts")
            with step_timer.time("prompt_attachment"):
                updated = await run_in_thread(
                    attach_segment_audio_clips,
                    asr_dump=tr_result.model_dump(),
                    output_dir=prompt_audio_dir,
                    min_duration=9.0,
                    max_duration=40.0,
                    one_per_speaker=True,
                )
            tr_result = ASRResponse(**updated)

            with step_timer.time("tts"):
                tts_result = await synthesize_tts(client, tts_model, tr_result, target_lang, workspace.workspace)
            tts_out_path = workspace.maybe_dump_json("tts/tts_result.json", tts_result.model_dump())

            if perform_vad_trimming:
                with step_timer.time("tts_vad_trim"):
                    vad_dir = workspace.ensure_dir("vad_trimmed")
                    tts_result = await trim_tts_segments(tts_result, vad_dir)
                workspace.maybe_dump_json("tts/tts_result.json", tts_result.model_dump())
            else:
                logger.info("Skipping VAD-based trimming after TTS")

            with step_timer.time("audio_concatenate"):
                audio_processing_dir = workspace.ensure_dir("audio_processing")
                speech_track = audio_processing_dir / "dubbed_speech_track.wav"
                concatenated_path, translation_segments = await concatenate_segments(
                    tts_result.model_dump()["segments"],
                    speech_track,
                    target_duration=get_audio_duration(raw_audio_path),
                    translation_segments=tr_result.model_dump()["segments"],
                )

            final_audio_path = Path(concatenated_path)
            if dubbing_strategy == "full_replacement" and background_path:
                with step_timer.time("audio_overlay"):
                    audio_processing_dir = speech_track.parent
                    final_audio_path = audio_processing_dir / "final_dubbed_audio.wav"
                    await overlay_segments_on_background(
                        tts_result.model_dump()["segments"],
                        background_path=background_path,
                        output_path=final_audio_path,
                        sophisticated=sophisticated_dub_timing,
                        speech_track=speech_track,
                    )
            else:
                logger.info("Using translation-over dubbing strategy")

            if subtitle_style is not None:
                with step_timer.time("dubbed_alignment"):
                    aligned_tts = await align_dubbed_audio(
                        client,
                        asr_model,
                        tr_result,
                        translation_segments,
                        final_audio_path,
                    )
                tr_aligned_tts_path = workspace.maybe_dump_json(
                    "translation/translation_aligned_W_dubbedvoice_result.json",
                    aligned_tts.model_dump(),
                )
                if subtitles_dir:
                    srt_path_1, vtt_path_1 = build_subtitles_from_asr_result(
                        data=aligned_tts.model_dump(),
                        output_dir=subtitles_dir,
                        custom_name=f"dubbed_{target_lang}",
                        formats=["srt", "vtt"],
                        mobile_mode=subtitle_style.split("_")[-1] == "mobile",
                    )

        style = (
            STYLE_PRESETS.get(subtitle_style.split("_")[0], STYLE_PRESETS["default"]) if subtitle_style is not None else None
        )

        if target_work != "sub":
            dubbed_path = workspace.file_path(f"dubbed_video_{target_lang}.mp4")
            final_output = (
                workspace.file_path(f"dubbed_video_{target_lang}_with_{subtitle_style.split('_')[0]}_subs.mp4")
                if subtitle_style is not None
                else None
            )
        else:
            dubbed_path = Path(resolved_video_url)
            final_output = (
                workspace.file_path(f"subtitled_video_{target_lang or source_lang}_with_{subtitle_style.split('_')[0]}_subs.mp4")
                if subtitle_style is not None
                else None
            )

        with step_timer.time("final_pass"):
            await finalize_media(
                resolved_video_url,
                final_audio_path,
                dubbed_path,
                final_output,
                Path(vtt_path_1) if vtt_path_1 else None,
                style,
                subtitle_style.split("_")[-1] == "mobile" if subtitle_style is not None else False,
                dubbing_strategy,
            )

        final_result: Dict[str, Any] = {
            "workspace_id": workspace.workspace_id,
            "final_video_path": str(final_output) if final_output else str(dubbed_path),
            "final_audio_path": str(final_audio_path) if final_audio_path else "",
            "speech_track": str(speech_track) if speech_track else "",
            "subtitles": {
                "original": {"srt": str(srt_path_0) if srt_path_0 else "", "vtt": str(vtt_path_0) if vtt_path_0 else ""},
                "aligned": {"srt": str(srt_path_1) if srt_path_1 else "", "vtt": str(vtt_path_1) if vtt_path_1 else ""},
            },
            "intermediate_files": {
                "asr_original": asr_raw_path,
                "asr_aligned": asr_aligned_path,
                "translation": tr_out_path,
                "translation_aligned_W_origin": tr_aligned_origin_path,
                "translation_aligned_W_dubbedvoice": tr_aligned_tts_path,
                "tts": tts_out_path,
                "vocals": str(vocals_path) if vocals_path else "",
                "background": str(background_path) if background_path else "",
            },
            "timings": step_timer.timings,
        }

        workspace.maybe_dump_json("final_result.json", final_result, force=True)
        return final_result

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc
