import sys, json, logging, warnings, contextlib
from common_schemas.models import ASRRequest, ASRResponse, Segment
import whisperx
import os
from dotenv import load_dotenv
# import uuid
# from pathlib import Path
# import gc
# import torch 


if __name__ == "__main__":
    req = ASRRequest(**json.loads(sys.stdin.read()))

    # # Silence libraries and progress bars; redirect prints to stderr
    # os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # os.environ.setdefault("TQDM_DISABLE", "1")
    # logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
    # warnings.simplefilter("ignore")

    with contextlib.redirect_stdout(sys.stderr):

        device = "cuda" # if torch.cuda.is_available() else "cpu"
        batch_size = 16 # reduce if low on GPU mem
        compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        model_dir = "./model_cache/asr/" # save model to local path (optional)

        model = whisperx.load_model("large", device, compute_type=compute_type, download_root=model_dir)
        audio = whisperx.load_audio(req.audio_url)

        # 1. Transcribe with original whisper (batched)
        if req.language_hint:
            result = model.transcribe(audio, batch_size=batch_size, language=req.language_hint)
        else:
            result = model.transcribe(audio, batch_size=batch_size)

        # Preserve language before align overwrites result
        language = result.get("language")

        #delete model if low on GPU resources
        import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

        load_dotenv()  # Loads environment variables from a .env file
        HF_TOKEN = os.environ.get("HF_TOKEN")

        # 2. Assign speaker labels (only if token available)
        diarize_segments = None
        if HF_TOKEN:
            diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
            # add min/max number of speakers if known
            min_speakers = 1
            max_speakers = 1
            diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        segments_out: list[Segment] = []
        for seg in result.get("segments", []):
            seg_speaker = seg.get("speaker")
            segments_out.append(
                Segment(
                    start=float(seg["start"]) if seg.get("start") is not None else None,
                    end=float(seg["end"]) if seg.get("end") is not None else None,
                    text=(seg.get("text") or "").strip(),
                    words= None,
                    speaker_id=seg_speaker,
                )
            )



        out = ASRResponse(
            segments=segments_out,
            WordSegments= None,
            language=language,
        )


        # Save output to dedicated workspace
    
        # Create unique workspace
        # workspace_id = str(uuid.uuid4())
        # BASE = Path(__file__).resolve().parents[4]
        # output_dir = BASE / "outs" / "asr_outputs" / workspace_id  # matches your repo structure
        # output_dir.mkdir(parents=True, exist_ok=True)

        # # Save result to file
        # output_file = output_dir / "asr_result.json"
        # with open(output_file, 'w') as f:
        #     f.write(out.model_dump_json())

    
    # Also write to stdout as before
    sys.stdout.write(out.model_dump_json())
    sys.stdout.flush()
