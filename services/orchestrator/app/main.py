import json, sys, os
from typing import Tuple, List, Optional
import uuid
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import *
from preprocessing.media_separation import separation
from media_processing.subtitles_handling import build_subtitles_from_asr_result, STYLE_PRESETS
from media_processing.audio_processing import concatenate_audio, get_audio_duration, trim_audio_with_vad, overlay_on_background
from media_processing.final_pass import final
import shutil
import subprocess
from common_schemas.utils import map_by_text_overlap, alignerWrapper, LANGUAGE_MATCHING, attach_segment_audio_clips

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
    target_lang: str = "fr",
    source_lang: str | None = "en",
    sep_model: str = Query("melband_roformer_big_beta5e.ckpt"),
    asr_model: str = Query("whisperx"),
    tr_model: str = Query("facebook_m2m100"),
    tts_model: str = Query("chatterbox"),
    audio_sep: bool = Query(True, description="Whether to perform audio source separation"),
    perform_vad_trimming: bool = Query(True, description="Whether to perform VAD-based silence trimming after TTS"),
    dubbing_strategy: str = Query("default", description="Dubbing strategy to use, either translation over (original audio ducked) or full replacement"),
    sophisticated_dub_timing: bool = Query(False, description="Whether to use sophisticated timing for full replacement dubbing strategy"),
    mobile_optimized: bool = Query(True, description="Optimize for mobile viewing"),
    allow_short_translations: bool = Query(True, description="Allow short translations for alignment"),
    segments_aligner_model: str = Query("default", description="Model for translated segment alignment with the source segments"),
    allow_merging: bool = Query(False, description="Allow merging segments during alignment"),
    subtitle_style: str | None = Query(None, description="Subtitle style preset: default, minimal, bold, netflix"),
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
        if audio_sep or dubbing_strategy == "full_replacement":
            model_file_dir = BASE / "models_cache" / "audio-separator-models" / Path(sep_model).stem
            print("Performing audio source separation...", file=sys.stderr)
            separation(
                input_file=str(raw_audio_path),
                output_dir=str(preprocessing_out),
                model_filename=sep_model,
                output_format="WAV",
                custom_output_names={"vocals": "vocals", "other": "background"},
                model_file_dir=str(model_file_dir)
            )

        # ========== STEP 2: ASR - Transcribe vocals ==========
        async with httpx.AsyncClient(timeout=1200) as client:
            asr_req = ASRRequest(
                audio_url=str(raw_audio_path),
                language_hint=LANGUAGE_MATCHING[source_lang]["asr"] if source_lang else None,
                allow_short=allow_short_translations
            )

            # Call ASR service for the transcription step without alignement
            r = await client.post(
                ASR_URL,
                params={"model_key": asr_model},
                json=asr_req.model_dump()
            )
            if r.status_code != 200:
                raise HTTPException(500, f"ASR failed: {r.text}")

            # Get the response JSON
            response_data = r.json()

            # 🔍 Debug: Print the actual response structure
            print(f"🔍 ASR Response keys: {response_data.keys() if isinstance(response_data, dict) else type(response_data)}", file=sys.stderr)
            print(f"🔍 ASR Response: {json.dumps(response_data, indent=2)[:500]}...", file=sys.stderr)

            # Check if response has expected structure
            if not isinstance(response_data, dict):
                raise HTTPException(500, f"ASR returned invalid response type: {type(response_data)}")

            if "raw" not in response_data:
                raise HTTPException(500, f"ASR response missing 'raw' key. Available keys: {list(response_data.keys())}")

            if "aligned" not in response_data:
                raise HTTPException(500, f"ASR response missing 'aligned' key. Available keys: {list(response_data.keys())}")

            raw_asr_result = ASRResponse(**response_data["raw"])
            aligned_asr_result = ASRResponse(**response_data["aligned"])
                    
        # Save ASR output
        asr_out_0_0 = workspace / "asr" / "asr_0_result.json"
        asr_out_0_0.parent.mkdir(exist_ok=True, parents=True)
        with open(asr_out_0_0, "w") as f:
            f.write(raw_asr_result.model_dump_json(indent=2))

        asr_out_0_1 = workspace / "asr" / "asr_0_aligned_result.json"
        asr_out_0_1.parent.mkdir(exist_ok=True, parents=True)
        with open(asr_out_0_1, "w") as f:
            f.write(aligned_asr_result.model_dump_json(indent=2))


        # ========== STEP 3: Generate Subtitles ==========
        srt_path_0, vtt_path_0 = None, None

        if subtitle_style is not None:
            subtitles_dir = workspace / "subtitles"
            subtitles_dir.mkdir(exist_ok=True)

            srt_path_0, vtt_path_0 = build_subtitles_from_asr_result(
                data=aligned_asr_result.model_dump(),
                output_dir=subtitles_dir,
                custom_name="original",
                formats=["srt", "vtt"],
                mobile_mode=mobile_optimized
            )

        # ========== STEP 4: Translation ==========
        async with httpx.AsyncClient(timeout=1200) as client:
            # Build translation request from ASR segments

            tr_req = TranslateRequest(
                segments=aligned_asr_result.segments if allow_short_translations else raw_asr_result.segments,
                source_lang=LANGUAGE_MATCHING[source_lang]["translation"] if source_lang else None,
                target_lang=LANGUAGE_MATCHING[target_lang]["translation"] if target_lang else None
            )
            
            r = await client.post(
                TR_URL,
                params={"model_key": tr_model},
                json=tr_req.model_dump()
            )
            if r.status_code != 200:
                raise HTTPException(500, f"Translation failed: {r.text}")
            tr_result = ASRResponse(**r.json())
        
        # Save translation output
        tr_out = workspace / "translation" / "translation_result.json"
        tr_out.parent.mkdir(exist_ok=True, parents=True)
        with open(tr_out, "w") as f:
            f.write(tr_result.model_dump_json(indent=2))

        # ========== STEP 4.5: Align Translation to Original Segments ==========
        if (not allow_short_translations) and (len(aligned_asr_result.segments) > 1) and dubbing_strategy != "full_replacement":  # Only needed if multiple segments and for translation over
            print("🔄 Aligning translated segments with original ASR segments...", file=sys.stderr)

            mappings = map_by_text_overlap( raw_asr_result.model_dump()["segments"], aligned_asr_result.model_dump()["segments"])

            for idx, tr in zip(mappings.keys(), tr_result.segments):
                mappings[idx]["full_text"] = tr.text

            print(f"Alignment mappings: {mappings[0]['segments']}")

            tr_result = alignerWrapper(mappings, segments_aligner_model, target_lang, allow_merging=allow_merging, max_look_distance=3, verbose=True)

            # Save aligned translation output
            tr_aligned_origin = workspace / "translation" / "translation_aligned_W_origin_result.json"
            tr_aligned_origin.parent.mkdir(exist_ok=True, parents=True)
            with open(tr_aligned_origin, "w") as f:
                f.write(tr_result.model_dump_json(indent=2))
        else:
            # Skip alignment, use translation as-is
            print(f"⏭️  Skipping alignment (allow_short_translations={allow_short_translations}, segments={len(aligned_asr_result.segments)})")
            tr_aligned_origin = None
        

        # ========== STEP 5: TTS - Synthesize dubbed audio ==========
        tr_result.audio_url = str(vocals_path)  # Provide original audio for context
        prompt_audio_dir = workspace / "prompts"
        updated = attach_segment_audio_clips(
            asr_dump=tr_result.model_dump(),
            output_dir=prompt_audio_dir,
            min_duration=9.0,
            max_duration=40.0,
            one_per_speaker=True
        )
        tr_result = ASRResponse(**updated)

        async with httpx.AsyncClient(timeout=1200) as client:
            # Map translated segments to TTS input
            tts_segments = [
                SegmentAudioIn(
                    start=tr_seg.start,
                    end=tr_seg.end,
                    text=tr_seg.text,
                    speaker_id=tr_seg.speaker_id,
                    lang=tr_result.language or target_lang,
                    audio_prompt_url=tr_seg.audio_url if tr_seg.speaker_id else None
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

        # ========== STEP 5.5: VAD Trimming - Remove silence after speech ==========
        if not perform_vad_trimming:
            print("⏭️  Skipping VAD-based silence trimming after TTS")
        else:
            print("🔊 Performing VAD-based silence trimming after TTS...", file=sys.stderr)
            vad_dir = workspace / "vad_trimmed"
            vad_dir.mkdir(exist_ok=True)
            
            trimmed_segments = []
            several_seg = False  # Only one segment per audio file from TTS
            for i, seg in enumerate(tts_result.segments):
                original_audio = Path(seg.audio_url)

                if not several_seg:
                    trimmed_audio = vad_dir / f"trimmed_{i}_{original_audio.stem}.wav"
                else:
                    trimmed_audio = vad_dir

                try:
                    # Trim audio to last voice activity
                    actual_durations, output_files = trim_audio_with_vad(
                        audio_path=seg.audio_url,
                        output_path=trimmed_audio,
                        several_seg=several_seg
                    )
                    
                    # Update segment with trimmed audio path
                    if not several_seg:
                        seg.audio_url = str(trimmed_audio)
                    else:
                        seg.audio_url = [ str(f) for f in output_files ]
                    trimmed_segments.append(seg)
                    
                except Exception as e:
                    # If VAD fails, keep original segment
                    print(f"VAD trimming failed for segment {i}: {e}", file=sys.stderr)
                    trimmed_segments.append(seg)
            
            # Update tts_result with trimmed segments
            tts_result.segments = trimmed_segments
        
            # Save updated TTS result with trimmed paths
            with open(tts_out, "w") as f:
                f.write(tts_result.model_dump_json(indent=2))

        # ========== STEP 6: Audio processing - Resize & concatenate segments ==========
        audio_processing_dir = workspace / "audio_processing"
        audio_processing_dir.mkdir(exist_ok=True)
        
        
        # merge dubbed segments
        speech_track = audio_processing_dir / "dubbed_speech_track.wav"
        concatenate_audio(
                segments=tts_result.model_dump()["segments"],
                output_file=speech_track,
                target_duration=get_audio_duration(raw_audio_path)
        )

        final_audio_path = speech_track 
        if dubbing_strategy == "full_replacement":
            final_audio_path = audio_processing_dir / "final_dubbed_audio.wav"
            print(f"Using full replacement dubbing strategy with {sophisticated_dub_timing} for the voice over time manipulation", file=sys.stderr)
            overlay_on_background(tts_result.model_dump()["segments"], background_path=background_path, output_path=final_audio_path, sophisticated=sophisticated_dub_timing)

        else:
            print("Using translation Over dubbing strategy", file=sys.stderr)


        # ============= STEP 7:WORD ALIGNMENT with ASR on final audio ============
        
        # Update tr_result audio_url to point to final dubbed audio
        tr_result.audio_url = str(speech_track)

        tr_aligned_tts, srt_path_1, vtt_path_1 = None, None, None

        if subtitle_style is not None:
            # aligned the translated segments with the speech track for better dubbing sync
            import time
            
            alignment_start = time.time()
            print(f"Starting alignment at {alignment_start}", file=sys.stderr)
            
            async with httpx.AsyncClient(timeout=1200) as client:
                r = await client.post(
                    ASR_URL,
                    params={"model_key": asr_model, "runner_index": 1},
                    json=tr_result.model_dump()
                )
                if r.status_code != 200:
                    raise HTTPException(500, f"Second Alignment failed: {r.text}")
                asr_result = ASRResponse(**r.json()["aligned"])
            
            alignment_duration = time.time() - alignment_start
            print(f"Alignment completed in {alignment_duration:.2f} seconds", file=sys.stderr)


            # Save ASR output
            tr_aligned_tts = workspace / "translation" / "translation_aligned_W_dubbedvoice_result.json"
            tr_aligned_tts.parent.mkdir(exist_ok=True, parents=True)
            with open(tr_aligned_tts, "w") as f:
                f.write(asr_result.model_dump_json(indent=2))


            srt_path_1, vtt_path_1 = build_subtitles_from_asr_result(
                data=asr_result.model_dump(),
                output_dir=subtitles_dir,
                custom_name=f"dubbed_{target_lang}",
                formats=["srt", "vtt"],
                mobile_mode=mobile_optimized
            )

        # ========== STEP 8: Final pass - Replace video audio & burn subtitles ==========

        style = STYLE_PRESETS.get(subtitle_style, STYLE_PRESETS["default"]) if subtitle_style is not None else None
        dubbed_path = workspace / f"dubbed_video_{target_lang}.mp4"
        final_output = workspace / f"dubbed_video_{target_lang}_with_{subtitle_style}_subs.mp4" if subtitle_style is not None else None


        print("begining final pass...", file=sys.stderr)

        final(
            video_path=video_url,
            audio_path=final_audio_path,
            dubbed_path=dubbed_path,
            output_path=final_output,
            subtitle_path=vtt_path_1,
            sub_style=style,
            mobile_optimized=mobile_optimized,
            dubbing_strategy=dubbing_strategy
        )

        
        final_result = {
            "workspace_id": workspace_id,
            "final_video_path": str(final_output),
            "final_audio_path": str(final_audio_path),
            "speech_track": str(speech_track),
            "subtitles": {
            "original": {
                "srt": str(srt_path_0) if srt_path_0 else "",
                "vtt": str(vtt_path_0) if vtt_path_0 else ""
            },
            "aligned": {
                "srt": str(srt_path_1) if srt_path_1 else "",
                "vtt": str(vtt_path_1) if vtt_path_1 else ""
            }
            },
            "intermediate_files": {
            "asr_original": str(asr_out_0_0),
            "asr_aligned": str(asr_out_0_1),
            "translation": str(tr_out),
            "translation_aligned_W_origin": str(tr_aligned_origin) if tr_aligned_origin else "",
            "translation_aligned_W_dubbedvoice": str(tr_aligned_tts) if tr_aligned_tts else "",
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