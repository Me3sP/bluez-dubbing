import json, sys, os
from typing import Tuple, List, Optional
import uuid
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from common_schemas.models import *
from preprocessing.media_separation import separation
from media_processing.subtitles_handling import build_subtitles_from_asr_result, STYLE_PRESETS
from media_processing.audio_processing import concatenate_audio, get_audio_duration, trim_audio_with_vad
from media_processing.final_pass import final
import shutil
import subprocess
from common_schemas.utils import SophisticatedAligner, ProportionalAligner

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
    sep_model: str = Query("UVR-MDX-NET-Inst_HQ_3.onnx"),
    asr_model: str = Query("whisperx"),
    tr_model: str = Query("facebook_m2m100"),
    tts_model: str = Query("chatterbox"),
    mobile_optimized: bool = Query(True, description="Optimize for mobile viewing"),
    allow_short_translations: bool = Query(True, description="Allow short translations for alignment"),
    segments_aligner_model: str = Query("default", description="Model for translated segment alignment with the source segments"),
    allow_merging: bool = Query(False, description="Allow merging segments during alignment"),
    subtitle_style: str = Query("default", description="Subtitle style preset: default, minimal, bold, netflix"),
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
        # separation(
        #     input_file=str(raw_audio_path),
        #     output_dir=str(preprocessing_out),
        #     model_filename=sep_model,
        #     output_format="WAV",
        #     custom_output_names={"vocals": "vocals", "instrumental": "background"}
        # )

        # ========== STEP 2: ASR - Transcribe vocals ==========
        async with httpx.AsyncClient(timeout=300) as client:
            asr_req = ASRRequest(
                audio_url=str(raw_audio_path),
                language_hint=source_lang
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
        async with httpx.AsyncClient(timeout=600) as client:
            # Build translation request from ASR segments

            tr_req = TranslateRequest(
                segments=aligned_asr_result.segments if allow_short_translations else raw_asr_result.segments,
                source_lang=raw_asr_result.language or source_lang,
                target_lang=target_lang
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
        if (not allow_short_translations) and (len(aligned_asr_result.segments) > 1):  # Only needed if multiple segments
            
            # Get full translated text
            full_translation = " ".join([seg.text for seg in tr_result.segments])
            
            # Align translation back to original segments
            if segments_aligner_model in ["proportional", "default"]:
                aligner = ProportionalAligner()
                aligned_translations =aligner.align_segments(
                    source_segments=None, 
                    translated_text=full_translation,
                    verbose=True,
                    max_look_distance=3,
                    source_metadata=aligned_asr_result.model_dump()["segments"]
                    
                )
            elif segments_aligner_model == "sophisticated":
                aligner = SophisticatedAligner(matching_method="i", allow_merging=allow_merging)
                aligned_translations = aligner.align_segments(
                    source_segments=None, 
                    translated_text=full_translation,
                    verbose=True,
                    max_look_distance=3,
                    source_metadata=aligned_asr_result.model_dump()["segments"]
                )

            else:
                # Fallback to proportional aligner
                aligner = ProportionalAligner()
                aligned_translations = aligner.align_segments(
                    source_segments=None, 
                    translated_text=full_translation,
                    verbose=True,
                    max_look_distance=3,
                    source_metadata=aligned_asr_result.model_dump()["segments"]
                    
                )
            # Each segment ready for TTS synthesis
            Tresponse_segments = ASRResponse()

            # Update translation segments with aligned text
            for i, seg in enumerate(aligned_translations):

                print("="*40)
                print("="*20 + " Debug: Sophisticated Aligner Output " + "*"*20)
                print("="*40)

                print(f"Segment {i}: '{seg.original_text}' → '{seg.translated_text}'")
                T_segment = Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.translated_text,
                    lang=target_lang
                )
                Tresponse_segments.segments.append(T_segment)

            Tresponse_segments.language = target_lang
            tr_result = Tresponse_segments  # Replace with aligned segments

            print(f"✅ Aligned {len(aligned_translations)} translated segments")

            # Save aligned translation output
            tr_aligned_origin = workspace / "translation" / "translation_aligned_W_origin_result.json"
            tr_aligned_origin.parent.mkdir(exist_ok=True, parents=True)
            with open(tr_aligned_origin, "w") as f:
                f.write(tr_result.model_dump_json(indent=2))
        else:
            # Skip alignment, use translation as-is
            print(f"⏭️  Skipping alignment (allow_short_translations={allow_short_translations}, segments={len(aligned_asr_result.segments)})")
            
            # Still save the translation result
            tr_aligned_origin = workspace / "translation" / "translation_aligned_W_origin_result.json"
            tr_aligned_origin.parent.mkdir(exist_ok=True, parents=True)
            with open(tr_aligned_origin, "w") as f:
                f.write(tr_result.model_dump_json(indent=2))
        

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

        # ========== STEP 5.5: VAD Trimming - Remove silence after speech ==========
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
        final_audio_path = audio_processing_dir / "final_dubbed_audio.wav"
        
        concatenate_audio(
            segments=tts_result.segments,
            output_file=final_audio_path,
            target_duration=get_audio_duration(raw_audio_path)
        )

        # ============= STEP 7:WORD ALIGNMENT with ASR on final audio ============
        # Check if audio is too long
        final_audio_duration = get_audio_duration(final_audio_path)
        print(f"Final audio duration: {final_audio_duration:.2f}s", file=sys.stderr)
        
        if final_audio_duration > 600:  # 10 minutes
            print("Warning: Long audio may cause alignment timeout", file=sys.stderr)
        
        # Update tr_result audio_url to point to final dubbed audio
        tr_result.audio_url = str(final_audio_path)
       
        # aligned the translated segments with the final audio for better dubbing sync
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

        style = STYLE_PRESETS.get(subtitle_style, STYLE_PRESETS["default"])
        dubbed_path = workspace / f"dubbed_video_{target_lang}.mp4"
        final_output = workspace / f"dubbed_video_{target_lang}_with_{subtitle_style}_subs.mp4"


        print("begining final pass...", file=sys.stderr)

        final(
            video_path=video_url,
            audio_path=final_audio_path,
            dubbed_path=dubbed_path,
            subtitle_path=vtt_path_1,
            output_path=final_output,
            style=style,
            mobile_optimized=mobile_optimized
        )

        
        final_result = {
            "workspace_id": workspace_id,
            "final_video_path": str(final_output),
            "audio_path": str(final_audio_path),
            "subtitles": {
            "original": {
                "srt": str(srt_path_0),
                "vtt": str(vtt_path_0)
            },
            "aligned": {
                "srt": str(srt_path_1),
                "vtt": str(vtt_path_1)
            }
            },
            "intermediate_files": {
            "asr_original": str(asr_out_0_0),
            "asr_aligned": str(asr_out_0_1),
            "translation": str(tr_out),
            "translation_aligned_W_origin": str(tr_aligned_origin),
            "translation_aligned_W_dubbedvoice": str(tr_aligned_tts),
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