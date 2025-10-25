from deep_translator import (GoogleTranslator,
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
                             batch_detection) # any of them is usable but be aware that some require API keys
from common_schemas.models import ASRResponse, TranslateRequest, Segment
import json, sys, os, contextlib

def build_translator(req: "TranslateRequest"):
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

    return TranslatorCls(**kwargs)

if __name__ == "__main__":

    req = TranslateRequest(**json.loads(sys.stdin.read()))

    out = ASRResponse()

    if req.source_lang == "zh":
        req.source_lang = "zh-CN" # or "zh-TW" based on your needs
    if req.target_lang == "zh":
        req.target_lang = "zh-CN" # or "zh-TW" based on your needs

    with contextlib.redirect_stdout(sys.stderr):

        translator = build_translator(req)

        for i, segment in enumerate(req.segments):
            try:
                translated_text = translator.translate(segment.text)
            except Exception:
                # Fallback to Google if selected provider fails
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

    sys.stdout.write(out.model_dump_json() + "\n")
    sys.stdout.flush()