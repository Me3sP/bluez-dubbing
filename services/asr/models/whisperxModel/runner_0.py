import sys, json, logging, warnings, contextlib
from common_schemas.models import ASRRequest, ASRResponse, Segment, Word
from common_schemas.utils import convert_whisperx_result_to_Segment, create_word_segments
import whisperx
import os
from dotenv import load_dotenv
# from pathlib import Path
import gc
import torch 


if __name__ == "__main__":

    req = ASRRequest(**json.loads(sys.stdin.read()))

    with contextlib.redirect_stdout(sys.stderr):

        device = "cuda" # if torch.cuda.is_available() else "cpu"
        batch_size = 16 # reduce if low on GPU mem
        compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        model_dir = "./model_cache/asr/" # save model to local path (optional)

        model = whisperx.load_model("large", device, compute_type=compute_type)
        audio = whisperx.load_audio(req.audio_url)

        # 1. Transcribe with original whisper (batched)
        if req.language_hint:
            result = model.transcribe(audio, batch_size=batch_size, language=req.language_hint)
        else:
            result = model.transcribe(audio, batch_size=batch_size)

        # Preserve language before align overwrites result
        language = result.get("language")

        # delete model if low on GPU resources
        gc.collect()
        torch.cuda.empty_cache()
        del model

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=req.language_hint, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # delete model if low on GPU resources
        del audio
        gc.collect()
        torch.cuda.empty_cache()
        del model_a

        segments_out: list[Segment] = convert_whisperx_result_to_Segment(result)
        word_segments_out: list[Word] = create_word_segments(result, segments_out)

        out = ASRResponse(
            segments=segments_out,
            WordSegments=word_segments_out or None,
            language= language,
        )

    # Write to stdout as before
    sys.stdout.write(out.model_dump_json())
    sys.stdout.flush()
