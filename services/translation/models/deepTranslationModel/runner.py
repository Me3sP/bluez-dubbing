from deep_translator import GoogleTranslator
from common_schemas.models import ASRResponse, TranslateRequest, Segment
import json, uuid, sys, os, contextlib
import torch
from pathlib import Path

if __name__ == "__main__":

    req = TranslateRequest(**json.loads(sys.stdin.read()))

    out = ASRResponse()


    with contextlib.redirect_stdout(sys.stderr):

        for i, segment in enumerate(req.segments):

            # Use any translator you like, in this example GoogleTranslator
            translated_text = GoogleTranslator(source=req.source_lang, target=req.target_lang).translate(segment.text)

            out.segments.append(Segment(start=segment.start, end=segment.end, text=translated_text, speaker_id=segment.speaker_id, lang=req.target_lang))
            out.language = req.target_lang


    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()