from deep_translator import (
    GoogleTranslator,
    ChatGptTranslator,
    MicrosoftTranslator,
    PonsTranslator,
    LingueeTranslator,
    MyMemoryTranslator,
    YandexTranslator,
    PapagoTranslator,
    DeeplTranslator,
    QcriTranslator,
    single_detection,
    batch_detection,
)  # any of them is usable but be aware that some require API keys
from common_schemas.models import ASRResponse, TranslateRequest, Segment
from common_schemas.service_utils import get_service_logger
import json, sys, os, contextlib, logging, time


def build_translator(req: "TranslateRequest", logger: logging.Logger):
    # Map provider names to translator classes
    translators = {
        "google": GoogleTranslator,
        "deepl": DeeplTranslator,
        "microsoft": MicrosoftTranslator,
        "chatgpt": ChatGptTranslator,
        "pons": PonsTranslator,
        "linguee": LingueeTranslator,
        "mymemory": MyMemoryTranslator,
        "yandex": YandexTranslator,
        "papago": PapagoTranslator,
        "qcri": QcriTranslator,
    }
    
    extra_0 = req.extra or {}
    provider = extra_0.get("model_name", "google").lower()

    TranslatorCls = translators.get(provider, GoogleTranslator)
    if TranslatorCls is GoogleTranslator and provider not in translators:
        logger.warning("Unknown translator provider '%s'. Falling back to Google Translator.", provider)
    logger.info("Using provider=%s for translation.", TranslatorCls.__name__)

    # Ensure source/target are present unless explicitly provided
    kwargs = {}
    kwargs.setdefault("source", getattr(req, "source_lang", None) or "auto")
    kwargs.setdefault("target", getattr(req, "target_lang", None))

    # Optionally source API keys from env if not provided
    if provider == "deepl":
        kwargs.setdefault("api_key", os.getenv("DEEPL_API_KEY"))
    elif provider == "microsoft":
        kwargs.setdefault("api_key", os.getenv("AZURE_TRANSLATOR_KEY"))
        region = os.getenv("AZURE_TRANSLATOR_REGION")
        if region is not None:
            kwargs.setdefault("region", region)
    elif provider == "chatgpt":
        kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        # Model can be passed via provider_kwargs or env
        if "model" not in kwargs and os.getenv("OPENAI_MODEL"):
            kwargs["model"] = os.getenv("OPENAI_MODEL")

    logger.debug("Translator kwargs=%s", {k: ("***" if "key" in k else v) for k, v in kwargs.items()})
    return TranslatorCls(**kwargs)

if __name__ == "__main__":

    req = TranslateRequest(**json.loads(sys.stdin.read()))
    # Read from extra config (default to INFO if not set)
    log_level = req.extra.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level, logging.INFO)
    # Configure logging
    logger = get_service_logger("translation.deep_translator", log_level)

    out = ASRResponse()

    if req.source_lang == "zh":
        req.source_lang = "zh-CN" # or "zh-TW" based on your needs
    if req.target_lang == "zh":
        req.target_lang = "zh-CN" # or "zh-TW" based on your needs

    with contextlib.redirect_stdout(sys.stderr):
        start = time.perf_counter()
        logger.info(
            "Starting translation run segments=%d source=%s target=%s",
            len(req.segments or []),
            req.source_lang,
            req.target_lang,
        )

        translator = build_translator(req, logger)

        for i, segment in enumerate(req.segments):
            seg_start = time.perf_counter()
            try:
                translated_text = translator.translate(segment.text)
            except Exception as exc:
                # Fallback to Google if selected provider fails
                logger.warning(
                    "Provider translation failed for segment=%d (%s). Falling back to Google. error=%s",
                    i,
                    type(exc).__name__,
                    exc,
                )
                fallback = GoogleTranslator(source=getattr(req, "source_lang", None) or "auto",
                                            target=req.target_lang)
                translated_text = fallback.translate(segment.text)

            out.segments.append(Segment(
                start=segment.start,
                end=segment.end,
                text=translated_text,
                speaker_id=segment.speaker_id,
                lang=req.target_lang
            ))
            out.language = req.target_lang
            logger.info(
                "Translated segment %d duration=%.2fs chars_in=%d",
                i,
                time.perf_counter() - seg_start,
                len(segment.text or ""),
            )
        logger.info(
            "Completed translation run in %.2fs (segments=%d).",
            time.perf_counter() - start,
            len(out.segments),
        )

    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()
