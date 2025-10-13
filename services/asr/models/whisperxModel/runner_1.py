import sys, json, logging, warnings, contextlib
from common_schemas.models import ASRResponse, Word, Segment
from common_schemas.utils import convert_whisperx_result_to_Segment, create_word_segments
import whisperx
import os
import gc
import torch

if __name__ == "__main__":
    req = ASRResponse(**json.loads(sys.stdin.read()))
    req = req.model_dump()

    with contextlib.redirect_stdout(sys.stderr):
        device = "cuda"
        batch_size = 16
        compute_type = "float16"
        model_dir = "./model_cache/asr/"
        
        # Clear GPU cache before loading
        gc.collect()
        torch.cuda.empty_cache()
        
        audio = whisperx.load_audio(req["audio_url"])

        # Use cached model
        model_a, metadata = whisperx.load_align_model(
            language_code=req["language"], 
                device=device
        )
        
        result = whisperx.align(
            req["segments"], 
            model_a, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )

        # Clean up audio from memory
        del audio
        gc.collect()
        torch.cuda.empty_cache()
        del model_a

        segments_out: list[Segment] = convert_whisperx_result_to_Segment(result)
        word_segments_out: list[Word] = create_word_segments(result, segments_out)

        out = ASRResponse(
            segments=segments_out,
            WordSegments=word_segments_out or None,
            language=req["language"],
        )

    sys.stdout.write(out.model_dump_json())
    sys.stdout.flush()