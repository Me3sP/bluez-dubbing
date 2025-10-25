from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from common_schemas.models import ASRResponse, TranslateRequest, Segment
import json, uuid, sys, os, contextlib
import torch
from pathlib import Path

if __name__ == "__main__":

    req = TranslateRequest(**json.loads(sys.stdin.read()))

    out = ASRResponse()

    

    with contextlib.redirect_stdout(sys.stderr):
        model_name = req.extra.get("model_name", "facebook/m2m100_418M")

        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)

        # Move model to GPU
        model = model.to(device)

        for i, segment in enumerate(req.segments):

            tokenizer.src_lang = req.source_lang

            encoded_en = tokenizer(segment.text, return_tensors="pt")

            # Move input tensors to GPU
            encoded_en = {k: v.to(device) for k, v in encoded_en.items()}
            
            generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(req.target_lang))

            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=req.extra.get("skip_special_tokens", True))[0]

            out.segments.append(Segment(start=segment.start, end=segment.end, text=translated_text, speaker_id=segment.speaker_id, lang=req.target_lang))
            out.language = req.target_lang


    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()

    


    