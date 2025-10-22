import sys, json, logging, warnings, contextlib
from common_schemas.models import ASRRequest, ASRResponse, Segment, Word
from common_schemas.utils import convert_whisperx_result_to_Segment, create_word_segments
import whisperx
from whisperx.diarize import DiarizationPipeline
import os
from dotenv import load_dotenv
# from pathlib import Path
import gc
import torch 

if __name__ == "__main__":

    try:

        req = ASRRequest(**json.loads(sys.stdin.read()))

        with contextlib.redirect_stdout(sys.stderr):

            load_dotenv()
            YOUR_HF_TOKEN = os.getenv("HF_TOKEN")

            device = "cuda" # if torch.cuda.is_available() else "cpu"
            batch_size = 16 # reduce if low on GPU mem
            compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
            model_dir = "./model_cache/asr/" # save model to local path (optional)

            model = whisperx.load_model("large", device, compute_type=compute_type, download_root=model_dir)
            audio = whisperx.load_audio(req.audio_url)

            # 1. Transcribe with original whisper (batched)
            if req.language_hint:
                result_0 = model.transcribe(audio, batch_size=batch_size, language=req.language_hint)
            else:
                result_0 = model.transcribe(audio, batch_size=batch_size)

            # Preserve language before align overwrites result
            language = result_0.get("language")

            # delete model if low on GPU resources
            gc.collect()
            torch.cuda.empty_cache()
            del model

            # 2. Assign speaker labels
            diarize_model = DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio)
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            if not req.allow_short:
                result_0 = whisperx.assign_word_speakers(diarize_segments, result_0)

            # 3. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
            result = whisperx.align(result_0["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            if req.allow_short:
                result = whisperx.assign_word_speakers(diarize_segments, result)

            # delete model if low on GPU resources
            del audio
            gc.collect()
            torch.cuda.empty_cache()
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