import sys, json, os, contextlib
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from common_schemas.models import TTSRequest, TTSResponse, SegmentAudioOut
from pathlib import Path

if __name__ == "__main__":

    req = TTSRequest(**json.loads(sys.stdin.read()))

    out = TTSResponse()

    with contextlib.redirect_stdout(sys.stderr):
        
        # Convert workspace string to Path object
        workspace_path = Path(req.workspace)
        workspace_path.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model once outside the loop
        multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

        for i, segment in enumerate(req.segments):
            
            # Handle audio prompt path
            audio_prompt = None
            if segment.audio_prompt_url:
                audio_prompt = segment.audio_prompt_url
                # Verify file exists
                if not Path(audio_prompt).exists():
                    raise FileNotFoundError(f"Audio prompt file not found: {audio_prompt}")
            
            # Generate audio
            wav = multilingual_model.generate(
                segment.text, 
                language_id=segment.lang, 
                audio_prompt_path=audio_prompt
            )
            
            # Save audio file
            output_file = workspace_path / "tts" / f"seg-{i}.wav"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            ta.save(str(output_file), wav, multilingual_model.sr)

            out.segments.append(SegmentAudioOut(
                start=segment.start,
                end=segment.end,
                audio_url=str(output_file),
                speaker_id=segment.speaker_id,
                lang=segment.lang,
                sample_rate=multilingual_model.sr
            ))

    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()