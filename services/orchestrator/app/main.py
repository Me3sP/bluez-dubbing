import json
import uuid
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import *
from preprocessing.media_separation import separation
from media_processing.subtitles_handling import write_srt, write_vtt
from media_processing.audio_processing import rubberband_to_duration, concatenate_audio_simple, get_audio_duration
from media_processing.final_pass import apply_audio_to_video
import shutil
import subprocess

app = FastAPI(title="orchestrator")

ASR_URL = "http://localhost:8001/v1/transcribe"
TR_URL  = "http://localhost:8002/v1/translate"
TTS_URL = "http://localhost:8003/v1/synthesize"

BASE = Path(__file__).resolve().parents[3]  # bluez-dubbing root
OUTS = BASE / "outs"

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/dub")
async def dub(
    video_url: str,
    target_lang: str,
    source_lang: str | None = None,
    sep_model: str = Query("UVR-MDX-NET-Inst_HQ_3.onnx"),
    asr_model: str = Query("whisperx"),
    tr_model: str = Query("facebook_m2m100"),
    tts_model: str = Query("chatterbox"),
):
    """
    Complete dubbing pipeline:
    1. Extract & separate audio (vocals + background)
    2. ASR on vocals
    3. Translate transcription
    4. Generate subtitles (SRT/VTT)
    5. TTS synthesis
    6. Resize/align TTS segments
    7. Overlay dubbed audio on background
    8. Replace video audio stream & burn subtitles
    """
    workspace_id = str(uuid.uuid4())
    workspace = OUTS / workspace_id
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        # ========== STEP 1: Preprocessing - Audio separation ==========
        preprocessing_out = workspace / "preprocessing"
        preprocessing_out.mkdir(exist_ok=True)
        
        raw_audio_path = preprocessing_out / "raw_audio.wav"
        vocals_path = preprocessing_out / "vocals.wav"
        background_path = preprocessing_out / "background.wav"
        
        # Extract audio from video if needed
        if video_url.endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv')):
            try:
                # Check if video file exists (for local files)
                if not video_url.startswith(('http://', 'https://')):
                    video_path = Path(video_url)

                    if not video_path.is_absolute():
                        # Resolve relative to project BASE directory
                        video_path = BASE / video_url


                    if not video_path.exists():
                        raise HTTPException(404, f"Video file not found: {video_url}")
                    
                    # Update video_url to absolute path string
                    video_url = str(video_path)
                
                # Extract audio with ffmpeg
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', video_url,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ar', '44100',  # 44.1kHz sample rate
                    '-ac', '2',  # Stereo
                    str(raw_audio_path)
                ], 
                check=True, 
                capture_output=True,  # Capture both stdout and stderr
                text=True
                )
                
                # Verify output file was created
                if not raw_audio_path.exists() or raw_audio_path.stat().st_size == 0:
                    raise HTTPException(500, "Audio extraction failed: output file is empty or missing")
                    
            except subprocess.CalledProcessError as e:
                raise HTTPException(500, f"FFmpeg audio extraction failed: {e.stderr}")
            except Exception as e:
                raise HTTPException(500, f"Audio extraction error: {str(e)}")
                
        elif video_url.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')):
            # Already audio - just copy or convert to WAV
            try:
                if not video_url.startswith(('http://', 'https://')):
                    source_path = Path(video_url)

                    if not source_path.is_absolute():
                        source_path = BASE / video_url

                    if not source_path.exists():
                        raise HTTPException(404, f"Audio file not found: {video_url}")
                    
                    video_url = str(source_path)
                
                # If already WAV with correct format, just copy
                if video_url.endswith('.wav'):
                    shutil.copy(video_url, raw_audio_path)
                else:
                    # Convert to WAV format
                    subprocess.run([
                        'ffmpeg', '-y', '-i', video_url,
                        '-acodec', 'pcm_s16le',
                        '-ar', '44100',
                        '-ac', '2',
                        str(raw_audio_path)
                    ], check=True, capture_output=True, text=True)
                
                if not raw_audio_path.exists() or raw_audio_path.stat().st_size == 0:
                    raise HTTPException(500, "Audio file copy/conversion failed")
                    
            except subprocess.CalledProcessError as e:
                raise HTTPException(500, f"Audio conversion failed: {e.stderr}")
            except Exception as e:
                raise HTTPException(500, f"Audio processing error: {str(e)}")
        else:
            raise HTTPException(400, f"Unsupported file format: {video_url}")
        
        # Separate vocals and background
        separation(
            input_file=str(raw_audio_path),
            output_dir=str(preprocessing_out),
            model_filename=sep_model,
            output_format="WAV",
            custom_output_names={"vocals": "vocals", "instrumental": "background"}
        )

        # ========== STEP 2: ASR - Transcribe vocals ==========
        async with httpx.AsyncClient(timeout=300) as client:
            asr_req = ASRRequest(
                audio_url=str(vocals_path),
                language_hint=source_lang
            )
            r = await client.post(
                ASR_URL,
                params={"model_key": asr_model},
                json=asr_req.model_dump()
            )
            if r.status_code != 200:
                raise HTTPException(500, f"ASR failed: {r.text}")
            asr_result = ASRResponse(**r.json())
        
        # Save ASR output
        asr_out = workspace / "asr" / "asr_result.json"
        asr_out.parent.mkdir(exist_ok=True, parents=True)
        with open(asr_out, "w") as f:
            f.write(asr_result.model_dump_json(indent=2))

        # ========== STEP 3: Translation ==========
        async with httpx.AsyncClient(timeout=300) as client:
            # Build translation request from ASR segments

            tr_req = TranslateRequest(
                segments=asr_result.segments,
                source_lang=asr_result.language or source_lang,
                target_lang=target_lang
            )
            
            r = await client.post(
                TR_URL,
                params={"model_key": tr_model},
                json=tr_req.model_dump()
            )
            if r.status_code != 200:
                raise HTTPException(500, f"Translation failed: {r.text}")
            tr_result = TranslateResponse(**r.json())
        
        # Save translation output
        tr_out = workspace / "translation" / "translation_result.json"
        tr_out.parent.mkdir(exist_ok=True, parents=True)
        with open(tr_out, "w") as f:
            f.write(tr_result.model_dump_json(indent=2))

        # ========== STEP 4: Generate Subtitles ==========
        subtitles_dir = workspace / "subtitles"
        subtitles_dir.mkdir(exist_ok=True)
        
        # Prepare subtitle data (use translated segments)
        subtitle_chunks = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            }
            for seg in tr_result.segments
        ]
        
        srt_path = subtitles_dir / "subtitles.srt"
        vtt_path = subtitles_dir / "subtitles.vtt"
        write_srt(subtitle_chunks, srt_path)
        write_vtt(subtitle_chunks, vtt_path)

        # ========== STEP 5: TTS - Synthesize dubbed audio ==========
        async with httpx.AsyncClient(timeout=600) as client:
            # Map translated segments to TTS input
            tts_segments = [
                SegmentAudioIn(
                    start=tr_seg.start,
                    end=tr_seg.end,
                    text=tr_seg.text,
                    speaker_id=tr_seg.speaker_id,
                    lang=tr_result.language or target_lang,
                    audio_prompt_url=str(raw_audio_path) if tr_seg.speaker_id else None
                )
                for tr_seg in tr_result.segments
            ]
            
            tts_req = TTSRequest(segments=tts_segments , workspace=str(workspace))
            
            r = await client.post(
                TTS_URL,
                params={"model_key": tts_model},
                json=tts_req.model_dump()
            )
            if r.status_code != 200:
                raise HTTPException(500, f"TTS failed: {r.text}")
            tts_result = TTSResponse(**r.json())
        
        # Save TTS output
        tts_out = workspace / "tts" / "tts_result.json"
        tts_out.parent.mkdir(exist_ok=True, parents=True)
        with open(tts_out, "w") as f:
            f.write(tts_result.model_dump_json(indent=2))

        # ========== STEP 6: Audio processing - Resize & align segments ==========
        audio_processing_dir = workspace / "audio_processing"
        audio_processing_dir.mkdir(exist_ok=True)
        
        # Resize TTS segments to match original timing
        # resized_segments = adjust_audio_speed(
        #     input_files=tts_result.segments.model_dump_json(),
        #     output_dir=audio_processing_dir
        # )
        
        # Overlay dubbed segments on background
        final_audio_path = audio_processing_dir / "final_dubbed_audio.wav"
        # overlay_on_background(
        #     dubbed_segments=resized_segments,
        #     background_path=background_path,
        #     output_path=final_audio_path,
        #     original_duration=get_audio_duration(raw_audio_path)
        # )
        
        concatenate_audio_simple(
            audio_files=[seg.audio_url for seg in tts_result.segments],
            output_file=final_audio_path,
        )

        rubberband_to_duration(
            in_wav=str(final_audio_path),
            target_ms=int(get_audio_duration(raw_audio_path)*1000),
            out_wav=str(final_audio_path)
        )
        # ========== STEP 7: Final pass - Replace video audio & burn subtitles ==========
        final_output = workspace / f"dubbed_video_{target_lang}.mp4"
        apply_audio_to_video(
            video_path=video_url,
            audio_path=final_audio_path,
            subtitle_path=vtt_path,
            output_path=final_output
        )

        final_result = {
            "workspace_id": workspace_id,
            "video_path": str(final_output),
            "audio_path": str(final_audio_path),
            "subtitles": {
                "srt": str(srt_path),
                "vtt": str(vtt_path)
            },
            "intermediate_files": {
                "asr": str(asr_out),
                "translation": str(tr_out),
                "tts": str(tts_out),
                "vocals": str(vocals_path),
                "background": str(background_path)
            }
        }

        with open(workspace / "final_result.json", "w") as f:
            f.write(json.dumps(final_result, indent=2))

        return final_result

    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {str(e)}")