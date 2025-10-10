import sys, json, logging, warnings, contextlib
from common_schemas.models import ASRResponse, Word, Segment
import whisperx
import os
# import uuid
# from pathlib import Path
# import gc
# import torch 


if __name__ == "__main__":
    req = ASRResponse(**json.loads(sys.stdin.read()))

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

        # Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=req.language_hint, device=device)
        result = whisperx.align(req.segments, model_a, metadata, audio, device, return_char_alignments=False)

        # delete model if low on GPU resources
        # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

        
        # ---- Convert WhisperX result to ASRResponse ----
        def to_word(w: dict, seg_speaker: str | None) -> Word:
            # words can use "word" or "text" as the token key
            text = (w.get("word") or w.get("text") or "").strip()
            start = w.get("start")
            end = w.get("end", start)
            speaker = w.get("speaker") or seg_speaker
            score = w.get("score")
            return Word(
                start=float(start) if start is not None else 0.0,
                end=float(end) if end is not None else 0.0,
                text=text,
                score=score,
                speaker_id=speaker,
            )

        segments_out: list[Segment] = []
        for seg in result.get("segments", []):
            seg_speaker = seg.get("speaker")
            words_list = [to_word(w, seg_speaker) for w in seg.get("words", [])]
            segments_out.append(
                Segment(
                    start=float(seg["start"]) if seg.get("start") is not None else None,
                    end=float(seg["end"]) if seg.get("end") is not None else None,
                    text=(seg.get("text") or "").strip(),
                    words=words_list or None,
                    speaker_id=seg_speaker,
                )
            )

        # Prefer explicit word_segments from WhisperX; otherwise, flatten from segments
        word_segments_out: list[Word] = []
        raw_word_segments = result.get("word_segments")
        if isinstance(raw_word_segments, list) and raw_word_segments:
            for w in raw_word_segments:
                word_segments_out.append(to_word(w, seg_speaker=None))
        else:
            for s in segments_out:
                if s.words:
                    word_segments_out.extend(s.words)

        out = ASRResponse(
            segments=segments_out,
            WordSegments=word_segments_out or None,
            language= req.language,
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
