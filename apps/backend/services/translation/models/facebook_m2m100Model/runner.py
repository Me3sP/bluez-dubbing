from __future__ import annotations
import contextlib
import json
import sys
import threading
import logging
import time
from typing import Dict, Tuple

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from common_schemas.models import ASRResponse, Segment, TranslateRequest
from common_schemas.service_utils import get_service_logger

_MODEL_CACHE: Dict[str, Tuple[M2M100ForConditionalGeneration, M2M100Tokenizer, str]] = {}
_MODEL_LOCK = threading.Lock()


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, log_level: int) -> Tuple[M2M100ForConditionalGeneration, M2M100Tokenizer, str]:
    logger = get_service_logger("translation.facebook_m2m100", log_level)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached:
            logger.debug("Using cached model=%s on device=%s.", model_name, cached[2])
            return cached

        device = _get_device()
        load_start = time.perf_counter()
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = model.to(device)
        _MODEL_CACHE[model_name] = (model, tokenizer, device)
        logger.info(
            "Loaded model=%s on device=%s in %.2fs.",
            model_name,
            device,
            time.perf_counter() - load_start,
        )
        return _MODEL_CACHE[model_name]


def _translate(req: TranslateRequest) -> ASRResponse:
    log_level = req.extra.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level, logging.INFO)
    logger = get_service_logger("translation.facebook_m2m100", log_level)
    run_start = time.perf_counter()
    model_name = (req.extra or {}).get("model_name", "facebook/m2m100_418M")
    model, tokenizer, device = _load_model(model_name, log_level)

    out = ASRResponse()
    skip_special = (req.extra or {}).get("skip_special_tokens", True)
    logger.info(
        "Starting M2M translation segments=%d source=%s target=%s model=%s device=%s",
        len(req.segments or []),
        req.source_lang,
        req.target_lang,
        model_name,
        device,
    )

    for segment in req.segments:
        seg_start = time.perf_counter()
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
        logger.info(
            "Translated segment text_len=%d duration=%.2fs",
            len(segment.text or ""),
            time.perf_counter() - seg_start,
        )

    logger.info(
        "Completed M2M translation in %.2fs (segments=%d).",
        time.perf_counter() - run_start,
        len(out.segments),
    )
    return out


def _run_once():
    req = TranslateRequest(**json.loads(sys.stdin.read()))
    with contextlib.redirect_stdout(sys.stderr):
        out = _translate(req)
    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _run_once()
