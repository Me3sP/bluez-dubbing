import asyncio
import contextvars
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

BASE = Path(__file__).resolve().parents[3]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from fastapi import File, FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

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
    TRANSLATION_STRATEGIES,
    DUBBING_STRATEGIES,
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
from services.asr.app.registry import WORKERS as ASR_WORKERS
from services.translation.app.registry import WORKERS as TR_WORKERS
from services.tts.app.registry import WORKERS as TTS_WORKERS

ENABLE_UI = os.getenv("ORCHESTRATOR_ENABLE_UI", "1").lower() not in {"0", "false", "no", "off"}

logger = logging.getLogger("bluez.orchestrator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(title="orchestrator")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates")) if ENABLE_UI else None

ASR_URL = os.getenv("ASR_URL", "http://localhost:8001/v1/transcribe")
TR_URL = os.getenv("TR_URL", "http://localhost:8002/v1/translate")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8003/v1/synthesize")

OUTS = BASE / "outs"
SEPARATION_CACHE = BASE / "cache" / "audio_separation"
UPLOADS_DIR = BASE / "ui_uploads"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

PROGRESS_REPORTER: contextvars.ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = contextvars.ContextVar(
    "progress_reporter", default=None
)

app.mount("/outs", StaticFiles(directory=str(OUTS)), name="outs")
if ENABLE_UI:
    assets_dir = BASE / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.on_event("startup")
async def startup_event() -> None:
    timeout = httpx.Timeout(connect=10.0, read=1200.0, write=10.0, pool=None)
    app.state.http_client = httpx.AsyncClient(timeout=timeout)
    OUTS.mkdir(parents=True, exist_ok=True)
    SEPARATION_CACHE.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


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


def list_worker_models(workers: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for key, worker in workers.items():
        languages = getattr(worker, "languages", None)
        items.append(
            {
                "key": key,
                "languages": sorted(set(languages)) if languages else [],
            }
        )
    return items


def list_audio_separation_models() -> List[Dict[str, Any]]:
    grouped = filter_supported_models_grouped()
    response: List[Dict[str, Any]] = []
    for arch, configs in grouped.items():
        response.append(
            {
                "architecture": arch,
                "models": [{"filename": cfg.get("filename"), "stems": cfg.get("stems", [])} for cfg in configs],
            }
        )
    return response


def resolve_model_choice(
    requested: Optional[str],
    workers: Dict[str, Any],
    language: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    if not workers:
        raise HTTPException(500, "No models registered for requested service")

    if requested:
        normalized = requested.strip()
        if normalized and normalized.lower() != "auto":
            return normalized

    if language:
        for key, worker in workers.items():
            supported = getattr(worker, "languages", None)
            if not supported or language in supported:
                return key

    if fallback and fallback in workers:
        return fallback

    return next(iter(workers))


class StepTimer:
    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def time(self, label: str):
        reporter = PROGRESS_REPORTER.get()
        if reporter:
            try:
                reporter({"type": "step", "event": "start", "step": label})
            except Exception:  # noqa: BLE001
                logger.debug("Progress reporter failed for start of %s", label, exc_info=True)
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.timings[label] = duration
            logger.info("%s completed in %.2fs", label, duration)
            if reporter:
                try:
                    reporter({"type": "step", "event": "end", "step": label, "duration": duration})
                except Exception:  # noqa: BLE001
                    logger.debug("Progress reporter failed for end of %s", label, exc_info=True)


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
        if self.persist_intermediate or force:
            path = self.file_path(relative)
            path.write_text(json.dumps(payload, indent=2))
            return str(path)
        return ""
    
    def temp_file(self, suffix=""):
        """Return a temporary file path for ephemeral use."""
        return tempfile.NamedTemporaryFile(delete=False, suffix=suffix)


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


async def download_media_to_workspace(source_url: str, download_dir: Path) -> Path:
    logger.info("Downloading remote media: %s", source_url)

    def _download() -> Path:
        try:
            import yt_dlp  # noqa: WPS433
        except ImportError as exc:
            raise HTTPException(
                500,
                "yt-dlp is required to download remote media sources. "
                "Install it in the orchestrator environment (e.g., `uv add yt-dlp`).",
            ) from exc

        ydl_opts = {
            "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "format": "bv*+ba/b",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=True)
            filename = ydl.prepare_filename(info)
            requested = info.get("requested_downloads") or []
            for entry in requested:
                filepath = entry.get("filepath")
                if filepath:
                    filename = filepath
                    break
            return Path(filename)

    download_dir.mkdir(parents=True, exist_ok=True)
    local_path = await run_in_thread(_download)
    if not local_path.exists():
        raise HTTPException(500, f"Failed to download media from {source_url}")
    logger.info("Remote media downloaded to %s", local_path)
    return local_path


async def prepare_media_source(
    source: str,
    workspace: WorkspaceManager,
    temp_dir_factory: Callable[[str], Path],
) -> str:
    if not source:
        raise HTTPException(400, "video_url must be provided")

    source = source.strip()
    if source.startswith(("http://", "https://")):
        download_dir = workspace.ensure_dir("downloads") if workspace.persist_intermediate else temp_dir_factory("downloads")
        local_path = await download_media_to_workspace(source, download_dir)
        return str(local_path)

    return resolve_media_path(source)


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
    preprocessing_dir: Path,
    raw_audio_path: Path,
    sep_model: str,
    audio_sep: bool,
    dubbing_strategy: str,
) -> Tuple[Optional[Path], Optional[Path], str]:
    if not audio_sep and dubbing_strategy != "full_replacement":
        return None, None, dubbing_strategy
    
    vocals_path = preprocessing_dir / "vocals.wav"
    background_path = preprocessing_dir / "background.wav"

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
        error_log = (
            f"Translation service call failed (model={tr_model}, "
            f"segments={len(segments)}, source_lang={source_lang}, target_lang={target_lang}): {response.text}"
        )
        logger.error(error_log)
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


def parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "on", "yes"}


def safe_filename(name: str) -> str:
    stem = Path(name or "upload").name
    return stem.replace(" ", "_") or "upload"


def build_file_payload(path_str: str | None) -> Optional[Dict[str, str]]:
    if not path_str:
        return None
    resolved = Path(path_str).resolve()
    if not resolved.exists():
        return None
    try:
        resolved.relative_to(OUTS)
    except ValueError:
        return {"path": str(resolved)}
    query = urllib.parse.quote(str(resolved), safe="")
    return {"path": str(resolved), "url": f"/ui/file?path={query}"}


if ENABLE_UI:

    @app.get("/ui", response_class=HTMLResponse)
    async def ui_home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("ui.html", {"request": request})


    @app.get("/ui/options")
    async def ui_options() -> JSONResponse:
        return JSONResponse(
            {
                "asr_models": list_worker_models(ASR_WORKERS),
                "translation_models": list_worker_models(TR_WORKERS),
                "tts_models": list_worker_models(TTS_WORKERS),
                "audio_separation_models": list_audio_separation_models(),
                "translation_strategies": TRANSLATION_STRATEGIES,
                "dubbing_strategies": DUBBING_STRATEGIES,
                "subtitle_styles": sorted(STYLE_PRESETS.keys()),
            }
        )


    @app.get("/ui/file")
    async def ui_file(path: str) -> FileResponse:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(OUTS)
        except ValueError as exc:
            raise HTTPException(403, "Invalid path") from exc

        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(resolved)


    @app.post("/ui/run")
    async def ui_run(
        request: Request,
        file: UploadFile | None = File(None),
        video_url: Optional[str] = Form(None),
        target_work: str = Form("dub"),
        target_lang: Optional[str] = Form(None),
        source_lang: Optional[str] = Form("en"),
        asr_model: Optional[str] = Form("auto"),
        tr_model: Optional[str] = Form("auto"),
        tts_model: Optional[str] = Form("auto"),
        sep_model: Optional[str] = Form("auto"),
        audio_sep: str = Form("true"),
        perform_vad_trimming: str = Form("true"),
        translation_strategy: str = Form("default"),
        dubbing_strategy: str = Form("default"),
        sophisticated_dub_timing: str = Form("true"),
        subtitle_style: Optional[str] = Form(None),
        persist_intermediate: str = Form("false"),
    ) -> StreamingResponse:
        uploads_dir = UPLOADS_DIR
        uploads_dir.mkdir(parents=True, exist_ok=True)

        target_work = (target_work or "dub").strip().lower() or "dub"
        if target_work not in {"dub", "sub"}:
            target_work = "dub"
        target_lang = (target_lang or "").strip() or None
        source_lang = (source_lang or "").strip() or None
        asr_model = (asr_model or "").strip()
        tr_model = (tr_model or "").strip()
        tts_model = (tts_model or "").strip()
        sep_model = (sep_model or "").strip()
        translation_strategy = (translation_strategy or "default").strip() or "default"
        dubbing_strategy = (dubbing_strategy or "default").strip() or "default"
        subtitle_style = (subtitle_style or "").strip() or None
        source_media: Optional[str] = None
        upload_path: Optional[Path] = None

        if file and file.filename:
            filename = f"{uuid.uuid4()}_{safe_filename(file.filename)}"
            upload_path = uploads_dir / filename
            with upload_path.open("wb") as dest:
                shutil.copyfileobj(file.file, dest)
            source_media = str(upload_path)
        elif video_url and video_url.strip():
            source_media = video_url.strip()
        else:
            raise HTTPException(400, "Provide either a media file or a video link.")

        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        def report(event: Dict[str, Any]) -> None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Progress queue full; dropping event: %s", event)

        async def run_pipeline() -> None:
            token = PROGRESS_REPORTER.set(report)
            try:
                result = await dub(
                    video_url=str(source_media),
                    target_work=target_work,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    sep_model=sep_model or "",
                    asr_model=asr_model or "",
                    tr_model=tr_model or "",
                    tts_model=tts_model or "",
                    audio_sep=parse_bool(audio_sep),
                    perform_vad_trimming=parse_bool(perform_vad_trimming),
                    translation_strategy=translation_strategy,
                    dubbing_strategy=dubbing_strategy,
                    sophisticated_dub_timing=parse_bool(sophisticated_dub_timing),
                    subtitle_style=subtitle_style,
                    persist_intermediate=parse_bool(persist_intermediate),
                )
                payload = {
                    "type": "result",
                    "result": {
                        "workspace_id": result.get("workspace_id"),
                        "source_media": result.get("source_media"),
                        "final_video": build_file_payload(result.get("final_video_path")),
                        "final_audio": build_file_payload(result.get("final_audio_path")),
                        "speech_track": build_file_payload(result.get("speech_track")),
                        "subtitles": {
                            "original": {
                                "srt": build_file_payload(result.get("subtitles", {}).get("original", {}).get("srt")),
                                "vtt": build_file_payload(result.get("subtitles", {}).get("original", {}).get("vtt")),
                            },
                            "aligned": {
                                "srt": build_file_payload(result.get("subtitles", {}).get("aligned", {}).get("srt")),
                                "vtt": build_file_payload(result.get("subtitles", {}).get("aligned", {}).get("vtt")),
                            },
                        },
                        "models": result.get("models", {}),
                        "timings": result.get("timings", {}),
                    },
                }
                await queue.put(payload)
            except HTTPException as exc:
                await queue.put({"type": "error", "status": exc.status_code, "message": exc.detail})
            except Exception as exc:  # noqa: BLE001
                logger.exception("UI pipeline run failed: %s", exc)
                await queue.put({"type": "error", "message": str(exc)})
            finally:
                PROGRESS_REPORTER.reset(token)
                if upload_path and upload_path.exists():
                    upload_path.unlink(missing_ok=True)
                await queue.put({"type": "complete"})

        async def event_stream():
            task = asyncio.create_task(run_pipeline())
            try:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "complete":
                        break
            finally:
                if not task.done():
                    task.cancel()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

else:

    @app.get("/ui", response_class=HTMLResponse)
    async def ui_home_disabled(request: Request) -> HTMLResponse:  # type: ignore[override]
        raise HTTPException(404, "UI mode disabled for this orchestrator instance")


    @app.get("/ui/options")
    async def ui_options_disabled() -> JSONResponse:
        raise HTTPException(404, "UI mode disabled for this orchestrator instance")


    @app.get("/ui/file")
    async def ui_file_disabled(path: str) -> FileResponse:  # type: ignore[override]
        raise HTTPException(404, "UI mode disabled for this orchestrator instance")


    @app.post("/ui/run")
    async def ui_run_disabled(*_: Any, **__: Any) -> StreamingResponse:
        raise HTTPException(404, "UI mode disabled for this orchestrator instance")


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
    original_source = video_url
    video_url = video_url.strip()
    if target_work == "sub" and subtitle_style is None:
        subtitle_style = "default_mobile"

    source_lang = (source_lang or "").strip() or None
    target_lang = (target_lang or "").strip() or None

    sep_model = (sep_model or "").strip()
    if not sep_model or sep_model.lower() == "auto":
        # Try the default value for dub first
        default_sep = "melband_roformer_big_beta5e.ckpt"
        separation_options = list_audio_separation_models()
        found = False
        # Check if default_sep exists in the available models
        for group in separation_options:
            for option in group.get("models", []):
                if option.get("filename") == default_sep:
                    sep_model = default_sep
                    found = True
                    break
            if found:
                break
        # If not found, pick the first available model
        if not found:
            for group in separation_options:
                for option in group.get("models", []):
                    candidate = option.get("filename")
                    if candidate:
                        sep_model = candidate
                        found = True
                        break
                if found:
                    break
        if not found:
            raise HTTPException(500, "No audio separation models available")

    asr_model = resolve_model_choice(asr_model, ASR_WORKERS, source_lang, fallback="whisperx")
    tr_model = resolve_model_choice(tr_model, TR_WORKERS, target_lang or source_lang, fallback="facebook_m2m100")
    tts_model = resolve_model_choice(tts_model, TTS_WORKERS, target_lang, fallback="chatterbox")
    selected_models = {
        "asr": asr_model,
        "translation": tr_model,
        "tts": tts_model,
        "separation": sep_model,
    }

    if translation_strategy not in TRANSLATION_STRATEGIES:
        translation_strategy = TRANSLATION_STRATEGIES[0]
    if dubbing_strategy not in DUBBING_STRATEGIES:
        dubbing_strategy = DUBBING_STRATEGIES[0]

    workspace = WorkspaceManager.create(OUTS, persist_intermediate)
    step_timer = StepTimer()
    client = get_http_client()

    temp_dirs: List[Path] = []
    transient_dubbed_path = False
    subtitles_dir: Optional[Path] = None
    prompt_audio_dir: Optional[Path] = None
    tts_output_dir: Optional[Path] = None
    vad_dir: Optional[Path] = None
    audio_processing_dir: Optional[Path] = None
    final_output: Optional[Path] = None
    dubbed_path: Optional[Path] = None

    def make_temp_dir(label: str) -> Path:
        path = Path(tempfile.mkdtemp(prefix=f"bluez_{label}_"))
        temp_dirs.append(path)
        return path

    resolved_video_url = await prepare_media_source(video_url, workspace, make_temp_dir)

    if workspace.persist_intermediate:
        preprocessing_dir = workspace.ensure_dir("preprocessing")
    else:
        preprocessing_dir = make_temp_dir("preprocessing")

    raw_audio_path = preprocessing_dir / "raw_audio.wav"

    try:
        with step_timer.time("extract_audio"):
            await extract_audio_to_workspace(resolved_video_url, raw_audio_path)

        vocals_path: Optional[Path] = None
        background_path: Optional[Path] = None

        if target_work != "sub":
            with step_timer.time("audio_separation"):

                vocals_path, background_path, dubbing_strategy = await maybe_run_audio_separation(
                    preprocessing_dir,
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

                if workspace.persist_intermediate:
                    subtitles_dir = workspace.ensure_dir("subtitles")
                else:
                    subtitles_dir = make_temp_dir("subtitles")

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
        translation_segments: Optional[List[Dict[str, Any]]] = None
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

            if workspace.persist_intermediate:
                prompt_audio_dir = workspace.ensure_dir("prompts")
            else:
                prompt_audio_dir = make_temp_dir("prompts")

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

                if workspace.persist_intermediate:
                    tts_output_dir = workspace.ensure_dir("tts")
                else:
                    tts_output_dir = make_temp_dir("tts")

                tts_result = await synthesize_tts(client, tts_model, tr_result, target_lang, tts_output_dir)
            tts_out_path = workspace.maybe_dump_json("tts/tts_result.json", tts_result.model_dump())

            if perform_vad_trimming:
                with step_timer.time("tts_vad_trim"):
                    if workspace.persist_intermediate:
                        vad_dir = workspace.ensure_dir("vad_trimmed")
                    else:
                        vad_dir = make_temp_dir("vad_trimmed")
                    tts_result = await trim_tts_segments(tts_result, vad_dir)
                workspace.maybe_dump_json("tts/tts_result.json", tts_result.model_dump())
            else:
                logger.info("Skipping VAD-based trimming after TTS")

            with step_timer.time("audio_concatenate"):
                if workspace.persist_intermediate:
                    audio_processing_dir = workspace.ensure_dir("audio_processing")
                else:
                    audio_processing_dir = make_temp_dir("audio_processing")

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
            if workspace.persist_intermediate or subtitle_style is None:
                dubbed_path = workspace.file_path(f"dubbed_video_{target_lang}.mp4")
            else:
                with workspace.temp_file(suffix=".mp4") as tmp:
                    dubbed_path = Path(tmp.name)
                transient_dubbed_path = True
            
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

        def keep_if_persistent(path: Optional[str | Path]) -> str:
            if not path:
                return ""
            return str(path) if workspace.persist_intermediate else ""

        final_result: Dict[str, Any] = {
            "workspace_id": workspace.workspace_id,
            "final_video_path": str(final_output) if final_output else str(dubbed_path),
            "final_audio_path": keep_if_persistent(final_audio_path),
            "speech_track": keep_if_persistent(speech_track),
            "source_media": original_source,
            "models": selected_models,
            "subtitles": {
                "original": {"srt": keep_if_persistent(srt_path_0), "vtt": keep_if_persistent(vtt_path_0)},
                "aligned": {"srt": keep_if_persistent(srt_path_1), "vtt": keep_if_persistent(vtt_path_1)},
            },
            "intermediate_files": {
                "asr_original": keep_if_persistent(asr_raw_path),
                "asr_aligned": keep_if_persistent(asr_aligned_path),
                "translation": keep_if_persistent(tr_out_path),
                "translation_aligned_W_origin": keep_if_persistent(tr_aligned_origin_path),
                "translation_aligned_W_dubbedvoice": keep_if_persistent(tr_aligned_tts_path),
                "tts": keep_if_persistent(tts_out_path),
                "vocals": keep_if_persistent(vocals_path),
                "background": keep_if_persistent(background_path),
            },
            "timings": step_timer.timings,
        }

        workspace.maybe_dump_json("final_result.json", final_result)

        return final_result

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        raise HTTPException(500, f"Pipeline failed: {exc}") from exc
    finally:
        if not workspace.persist_intermediate:
            for path in temp_dirs:
                shutil.rmtree(path, ignore_errors=True)
            if transient_dubbed_path and dubbed_path:
                dubbed_path.unlink(missing_ok=True)
