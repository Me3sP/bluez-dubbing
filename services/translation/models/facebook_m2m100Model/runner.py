from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from common_schemas.models import TranslateResponse, TranslateRequest, Segment
import json, uuid, sys, os, contextlib
from pathlib import Path

if __name__ == "__main__":

    req = TranslateRequest(**json.loads(sys.stdin.read()))

    out = TranslateResponse()

    

    with contextlib.redirect_stdout(sys.stderr):

        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

        for i, segment in enumerate(req.segments):

            tokenizer.src_lang = req.source_lang

            encoded_en = tokenizer(segment.text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(req.target_lang))

            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            out.segments.append(Segment(start=segment.start, end=segment.end, text=translated_text, speaker_id=segment.speaker_id, lang=req.target_lang))
            out.language = req.target_lang

            workspace_id = str(uuid.uuid4())
            BASE = Path(__file__).resolve().parents[4]
            output_base = BASE / "outs" / "translation_outputs" / workspace_id  # matches your repo structure
            output_base.mkdir(parents=True, exist_ok=True)

            # Save result to file
            output_file = output_base / "translated_segments_result.json"
            with open(output_file, 'w') as f:
                f.write(out.model_dump_json())


    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()

    


    