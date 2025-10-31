import sys, json, logging, contextlib, time
from common_schemas.models import ASRRequest, ASRResponse, Segment, Word
from common_schemas.utils import convert_whisperx_result_to_Segment, create_word_segments
import whisperx
from whisperx.diarize import DiarizationPipeline
import os
from dotenv import load_dotenv
# from pathlib import Path
import gc
import torch
from pathlib import Path 


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":

    try:
        BASE = Path(__file__).resolve().parents[4]
        req = ASRRequest(**json.loads(sys.stdin.read()))

        logger = logging.getLogger("whisperx.runner")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        run_start = time.perf_counter()

        with contextlib.redirect_stdout(sys.stderr):
            extra = req.extra or {}
            whisper_model = extra.get("model_name", "large")
            batch_size = extra.get("batch_size", 16) # reduce if low on GPU mem
            compute_type = extra.get("compute_type", "float16") # change to "int8" if low on GPU mem (may reduce accuracy)

            load_dotenv()
            YOUR_HF_TOKEN = os.getenv("HF_TOKEN")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and compute_type not in {"int8", "float32"}:
                logger.info("Switching compute_type from %s to float32 for CPU inference.", compute_type)
                compute_type = "float32"

            logger.info(
                "Starting ASR run for audio=%s model=%s device=%s batch_size=%s compute_type=%s",
                req.audio_url,
                whisper_model,
                device,
                batch_size,
                compute_type,
            )

            model_dir = BASE / "models_cache" / "asr"  # save model to local path (optional)
            # model_dir = "./model_cache/asr"

            load_start = time.perf_counter()
            model = whisperx.load_model(whisper_model, device, compute_type=compute_type, download_root=str(model_dir))
            logger.info("Loaded WhisperX model in %.2fs.", time.perf_counter() - load_start)

            audio_load_start = time.perf_counter()
            audio = whisperx.load_audio(req.audio_url)
            logger.info("Loaded audio in %.2fs.", time.perf_counter() - audio_load_start)

            # 1. Transcribe with original whisper (batched)
            transcribe_start = time.perf_counter()
            if req.language_hint:
                logger.info("Transcribing with language hint=%s.", req.language_hint)
                result_0 = model.transcribe(audio, batch_size=batch_size, language=req.language_hint)
            else:
                logger.info("Transcribing with automatic language detection.")
                result_0 = model.transcribe(audio, batch_size=batch_size)
            logger.info(
                "Transcription finished in %.2fs (segments=%d).",
                time.perf_counter() - transcribe_start,
                len(result_0.get("segments", [])),
            )

            # Preserve language before align overwrites result
            language = result_0.get("language")
            logger.info("Detected language=%s.", language)

            # delete model if low on GPU resources
            gc.collect()
            _clear_cuda_cache()
            del model

            # 2. Assign speaker labels
            diarization_model_name = extra.get("diarization_model")
            diarize_segments = None
            if diarization_model_name:
                if not YOUR_HF_TOKEN:
                    logger.warning(
                        "HF_TOKEN not found in environment. HuggingFace authentication is required for model %s.",
                        diarization_model_name,
                    )
                logger.info(
                    "Starting diarization with model=%s (min_speakers=%s, max_speakers=%s).",
                    diarization_model_name,
                    req.min_speakers,
                    req.max_speakers,
                )
                diarize_start = time.perf_counter()
                diarize_model = DiarizationPipeline(
                    model_name=diarization_model_name,
                    use_auth_token=YOUR_HF_TOKEN,
                    device=device,
                )
                diarize_segments = diarize_model(audio, min_speakers=req.min_speakers, max_speakers=req.max_speakers)
                logger.info(
                    "Diarization produced %d segments in %.2fs.",
                    len(diarize_segments),
                    time.perf_counter() - diarize_start,
                )
                del diarize_model
                _clear_cuda_cache()
            else:
                logger.info("Skipping diarization because no model was configured.")

            # 3. Align whisper output
            align_start = time.perf_counter()
            model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
            result = whisperx.align(result_0["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            if diarize_segments is not None:
                result = whisperx.assign_word_speakers(diarize_segments, result)
            else:
                logger.info("Skipping speaker attribution because diarization output is missing.")
            if isinstance(result, dict):
                aligned_segment_count = len(result.get("segments", []))
            else:
                segments_attr = getattr(result, "segments", None)
                aligned_segment_count = len(segments_attr) if segments_attr is not None else len(result)

            logger.info(
                "Alignment finished in %.2fs (segments=%d).",
                time.perf_counter() - align_start,
                aligned_segment_count,
            )

            # delete model if low on GPU resources
            del audio
            gc.collect()
            _clear_cuda_cache()
            del model_a

            # Convert raw result to schema format
            raw_segments_out: list[Segment] = convert_whisperx_result_to_Segment(result_0)
            raw_word_segments_out: list[Word] = create_word_segments(result_0, raw_segments_out)

            raw_output = ASRResponse(
                segments=raw_segments_out,
                WordSegments=raw_word_segments_out or None,
                language=language,
            )

            # Convert aligned result to schema format
            aligned_segments_out: list[Segment] = convert_whisperx_result_to_Segment(result)
            aligned_word_segments_out: list[Word] = create_word_segments(result, aligned_segments_out)

            aligned_output = ASRResponse(
                segments=aligned_segments_out,
                WordSegments=aligned_word_segments_out or None,
                language=language,
            )
            logger.info(
                "Completed ASR pipeline. raw_segments=%d aligned_segments=%d language=%s",
                len(raw_segments_out),
                len(aligned_segments_out),
                language,
            )
            logger.info("Total ASR runtime %.2fs.", time.perf_counter() - run_start)

        # Write to stdout as JSON with both raw and aligned results
        output_data = {
            "raw": raw_output.model_dump(),
            "aligned": aligned_output.model_dump()
        }
        sys.stdout.write(json.dumps(output_data, indent=2) + "\n")
        sys.stdout.flush()

    except Exception as e:
        # Write error to stderr and exit with error code
        error_data = {"error": str(e), "type": type(e).__name__}
        sys.stderr.write(f"❌ ASR Runner Error: {json.dumps(error_data, indent=2)}\n")
        sys.exit(1)
