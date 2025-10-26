from __future__ import annotations
import contextlib
import json
import os
import sys
import threading
from typing import Dict, Tuple

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from common_schemas.models import ASRResponse, Segment, TranslateRequest

_MODEL_CACHE: Dict[str, Tuple[M2M100ForConditionalGeneration, M2M100Tokenizer, str]] = {}
_MODEL_LOCK = threading.Lock()


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str) -> Tuple[M2M100ForConditionalGeneration, M2M100Tokenizer, str]:
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached:
            return cached

        device = _get_device()
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = model.to(device)
        _MODEL_CACHE[model_name] = (model, tokenizer, device)
        return _MODEL_CACHE[model_name]


def _translate(req: TranslateRequest) -> ASRResponse:
    model_name = (req.extra or {}).get("model_name", "facebook/m2m100_418M")
    model, tokenizer, device = _load_model(model_name)

    out = ASRResponse()
    skip_special = (req.extra or {}).get("skip_special_tokens", True)

    for segment in req.segments:
        tokenizer.src_lang = req.source_lang
        encoded = tokenizer(segment.text, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(req.target_lang),
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=skip_special)[0]
        out.segments.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=translated_text,
                speaker_id=segment.speaker_id,
                lang=req.target_lang,
            )
        )
        out.language = req.target_lang

    return out


def _run_once():
    req = TranslateRequest(**json.loads(sys.stdin.read()))
    with contextlib.redirect_stdout(sys.stderr):
        out = _translate(req)
    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()


def _loop():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        req = TranslateRequest(**json.loads(line))
        with contextlib.redirect_stdout(sys.stderr):
            out = _translate(req)
        sys.stdout.write(out.model_dump_json() + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if os.getenv("PERSISTENT_WORKER") == "1":
        _loop()
    else:
        _run_once()
