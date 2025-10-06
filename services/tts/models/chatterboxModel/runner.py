
import sys, json, os, contextlib
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from common_schemas.models import TTSRequest, TTSResponse, SegmentAudioOut
from pathlib import Path
import uuid

if __name__ == "__main__":

    req = TTSRequest(**json.loads(sys.stdin.read()))

    out = TTSResponse()

    

    with contextlib.redirect_stdout(sys.stderr):

        workspace_id = str(uuid.uuid4())
        BASE = Path(__file__).resolve().parents[4]
        output_base = BASE / "outs" / "tts_outputs" / workspace_id  # matches your repo structure
        output_base.mkdir(parents=True, exist_ok=True)

        for i, segment in enumerate(req.segments):

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Multilingual examples
            multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

            wav = multilingual_model.generate(segment.text, language_id=segment.lang, audio_prompt_path=segment.audio_prompt_url)
            ta.save(str(output_base / f"seg-{i}.wav"), wav, multilingual_model.sr)

            out.segments.append(SegmentAudioOut(
                start=segment.start,
                end=segment.end,
                audio_url=str(output_base / f"seg-{i}.wav"),
                speaker_id=segment.speaker_id,
                lang=segment.lang,
                sample_rate=multilingual_model.sr
            ))

            # Save result to file
            output_file = output_base / "tts_result.json"
            with open(output_file, 'w') as f:
                f.write(out.model_dump_json())

    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()